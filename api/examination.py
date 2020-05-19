import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from model import get_segmentation_model

class Examintion(object):
    def __init__(self):
        self.model = get_segmentation_model("mobilenetv3_large", 
        									classnum=6, 
        									root="save/mobilenetv3_large.pth")
        self.model.to(torch.device("cuda"))
        self.model.eval()

    def eval(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32) / 255.
        image = image[np.newaxis, ...]
        image = torch.tensor(image).cuda()

        with torch.no_grad():
            outputs = self.model(image)

        pred = torch.argmax(outputs[0], 1)
        pred = pred.cpu().data.numpy()
        predict = np.array(pred.squeeze(0), dtype=np.uint8)

        return predict


if __name__ == '__main__':
    cudnn.benchmark = True

    evaluator = Examintion()
    image = cv2.imread("../dataset/rgb/train/1.png", 1)
    pred = evaluator.eval(image)
    cv2.imwrite("./a.png", pred * 20)
