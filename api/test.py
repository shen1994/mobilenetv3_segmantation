import cv2
from examination import Examination

if __name__ == '__main__':

    evaluator = Examination(6, 
        "/home/westwell/Desktop/mobilenetv3_segmantation/scripts/save/mobilenetv3_large.pth")

    image = cv2.imread("../dataset/rgb/train/1.png", 1)
    pred = evaluator.eval(image)

    cv2.imwrite("./a.png", pred * 20)