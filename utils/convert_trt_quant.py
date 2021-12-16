import numpy as np
import torch
import torch.nn as nn
import util_trt
import glob,os,cv2
import argparse

BATCH_SIZE = 1
BATCH = 1
height = 32
width = 160
std = 0.193
mean = 0.588


def parse_arg():
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument('--calib',help='int8 calib path',type=str,default='/home/deyang_test')                    
    parser.add_argument('--onnx_model_path',help='onnx path',type=str,default='models_save/ATT_LP_RECGNIZE.onnx')
    parser.add_argument('--int8_mode',help='quantization mode',type=bool,default=True)
    parser.add_argument('--fp16_mode',help='quantization mode',type=bool,default=False)
    parser.add_argument('--engine_model_path',help='save engine path',type=str,default="models_save/ATT_LP_RECGNIZE.engine")
    parser.add_argument('--calibration_table',help='save calib path',type=str,default="models_save/ATT_LP_RECGNIZE.cache")
    args = parser.parse_args()

    return args

def preprocess_v1(image_raw):
    img = cv2.copyMakeBorder(image_raw, 2, 2, 6, 6, cv2.BORDER_REPLICATE)

    img_h, img_w,_ = img.shape
    img = cv2.resize(img, (width,height), interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (height, width, 3))
    img = img.astype(np.float32)
    img = (img/255. - mean) / std 
    img = img.transpose([2, 0, 1])
  
    return img


def preprocess(img):
    img = cv2.copyMakeBorder(img, 2, 2, 6, 6, cv2.BORDER_REPLICATE)

    img_h, img_w,_ = img.shape
    img = cv2.resize(img, (width,height), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img = (img/255. - mean) / std 
    img = img.transpose([2, 0, 1])
  
    return img

class DataLoader:
    def __init__(self,args):
        self.index = 0
        self.length = BATCH
        self.batch_size = BATCH_SIZE
        self.img_list = glob.glob(os.path.join(args.calib, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(args.calib) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size,3,height,width), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess_v1(img)
                self.calibration_data[i] = img

            self.index += 1

            # example only
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length

def main():
    args = parse_arg()
    # onnx2trt
    
    print('*** onnx to tensorrt begin ***')
    # calibration
    calibration_stream = DataLoader(args)
    # fixed_engine,校准产生校准表
    engine_fixed = util_trt.get_engine(BATCH_SIZE, args.onnx_model_path, args.engine_model_path, fp16_mode=args.fp16_mode, 
        int8_mode=args.int8_mode, calibration_stream=calibration_stream, calibration_table_path=args.calibration_table, save_engine=True)
    assert engine_fixed, 'Broken engine_fixed'
    print('*** onnx to tensorrt completed ***\n')
    
if __name__ == '__main__':
    main()
    
