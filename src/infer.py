from lp_ocr_infer import ocr_trt
import cv2
import argparse


def parse_arg():
    parser = argparse.ArgumentParser(description="int8 model infer")
    parser.add_argument('--engine_model_path',help='save engine path',type=str,default="models_save/ATT_LP_RECGNIZE.engine")
    parser.add_argument('--img_path',help='save calib path',type=str,default="data/0.jpg")
    args = parser.parse_args()

    return args

if __name__ == '__main__':  
    args = parse_arg()
    model = ocr_trt(args.engine_model_path)
    lp = model.infer(args.img_path)
    print(lp)
    model.destroy()
