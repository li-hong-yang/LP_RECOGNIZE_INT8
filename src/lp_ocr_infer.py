from os import pread
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import yaml
from easydict import EasyDict as edict



class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:

            if decode_flag:
                item = item.decode('utf-8','strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

   
    def numpy_decode(self,t,preds,len_dict=None):
        # t = preds.argmax(2) shape[40]
        preds = preds.reshape(40,84)
        score = []
        char_list = []

        for i in range(len(t)):
            if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                char_list.append(self.alphabet[t[i] - 1])
                score.append({self.alphabet[t[i] - 1]: round(preds[i][t[i]].item(),4)})

        lp = ''.join(char_list)
        return lp,score


class ocr_trt(object):
    
    
    def __init__(self, engine_file_path):
        # Create a Context on this device,

        with open('src/lp_config.yaml', 'r') as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
            self.config = edict(config) 
        self.converter = strLabelConverter(alphabet = """京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZ港学使警澳挂军北南广沈兰成济海民航空""")
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def model_infer(self, input_image):
        
        # Make self the active context, pushing it on top of the context stack.

        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        
        
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        # Here we use the first row of output in that batch_size = 1
        # output = host_outputs[0]
        # loc, conf, landms = host_outputs
        
        # print(landms.shape)
        return host_outputs

    
    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()


    def preprocess(self,img):
        img = cv2.imread(img)
        img = cv2.copyMakeBorder(img, 2, 2, 6, 6, cv2.BORDER_REPLICATE)
        img = cv2.resize(img, (self.config.MODEL.IMAGE_SIZE.W, self.config.MODEL.IMAGE_SIZE.H),  interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.config.MODEL.IMAGE_SIZE.H, self.config.MODEL.IMAGE_SIZE.W, 3))
        img = img.astype(np.float32)
        img = (img / 255. - self.config.DATASET.MEAN) / self.config.DATASET.STD
        img = img.transpose([2, 0, 1])
        input = img.reshape(1, *img.shape)
        return input
    def postprocess(self,preds):
        preds = np.array(preds)
        preds = preds.reshape(40, 1, 84)       
        preds_ = preds.argmax(2)  
        preds_ = preds_.transpose(1, 0).reshape(-1)
        sim_pred,scores = self.converter.numpy_decode((preds_.data),preds,len_dict=None)     
        return sim_pred

    def infer(self,img):
        img = self.preprocess(img)
        preds = self.model_infer(img)
        output = self.postprocess(preds)
        return output

    
   

    
    
    
     
    
   
