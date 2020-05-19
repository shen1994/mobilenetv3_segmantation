import cv2
import time
from examination import Examination

if __name__ == '__main__':

    evaluator = Examination(6, 
        "/home/westwell/Desktop/mobilenetv3_segmantation/scripts/save/mobilenetv3_large.pth")

    time_start = time.time()
    for i in range(100):
    	image = cv2.imread("../dataset/rgb/train/2.png", 1)
    	pred = evaluator.eval(image)
    print("using time when predicting one: ", (time.time() - time_start) / 100.)

    cv2.imwrite("./a.png", pred * 20)