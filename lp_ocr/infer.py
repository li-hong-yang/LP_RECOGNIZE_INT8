from lp_ocr_infer import lp_ocr
import cv2
car_img=cv2.imread('/root/project/int8test/LP_RECOGNIZE_INT8/lp_ocr/0.jpg')
ocr_input = lp_ocr(car_img)
print(ocr_input)