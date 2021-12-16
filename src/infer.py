from lp_ocr_infer import ocr_trt
import cv2
model = ocr_trt("/root/project/int8test/LP_RECOGNIZE_INT8/models_save/lp_int8.trt")
lp = model.infer("/root/project/int8test/LP_RECOGNIZE_INT8/data/0.jpg")
print(lp)
model.destroy()
