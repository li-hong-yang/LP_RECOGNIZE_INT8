import numpy as np
import torch
import torch.nn as nn
import util_trt
import glob,os,cv2

BATCH_SIZE = 1
BATCH = 1
height = 32
width = 160
CALIB_IMG_DIR = '/home/deyang_test'
# onnx_model_path = "./CORNER-NEW-MERGE.onnx"
# onnx_model_path = "./ATT-ADD-CORNER.onnx"
onnx_model_path = "./ATT_LP_RECGNIZE.onnx"
std = 0.193
mean = 0.588

def preprocess_v1(image_raw):
    img = cv2.copyMakeBorder(image_raw, 2, 2, 6, 6, cv2.BORDER_REPLICATE)

    img_h, img_w,_ = img.shape
    img = cv2.resize(img, (width,height), interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (height, width, 3))
    img = img.astype(np.float32)
    img = (img/255. - mean) / std 
    img = img.transpose([2, 0, 1])
    # print(img.shape)
  
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
    def __init__(self):
        self.index = 0
        self.length = BATCH
        self.batch_size = BATCH_SIZE
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(CALIB_IMG_DIR, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(CALIB_IMG_DIR) + str(self.batch_size * self.length) + ' images to calib'
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
    # onnx2trt
    fp16_mode = False
    int8_mode = True 
    print('*** onnx to tensorrt begin ***')
    # calibration
    calibration_stream = DataLoader()
    engine_model_path = "models_save/ATT_LP_RECGNIZE.engine"
    calibration_table = 'models_save/ATT_LP_RECGNIZE.cache'
    # fixed_engine,校准产生校准表
    engine_fixed = util_trt.get_engine(BATCH_SIZE, onnx_model_path, engine_model_path, fp16_mode=fp16_mode, 
        int8_mode=int8_mode, calibration_stream=calibration_stream, calibration_table_path=calibration_table, save_engine=True)
    assert engine_fixed, 'Broken engine_fixed'
    print('*** onnx to tensorrt completed ***\n')
    
if __name__ == '__main__':
    main()
    
