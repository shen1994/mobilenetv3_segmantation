import cv2
import torch
import numpy as np

from model import get_segmentation_model

class Examination(object):
    def __init__(self, classnum, model_path):
        torch.backends.cudnn.benchmark = True
        self.model = get_segmentation_model("mobilenetv3_large", classnum=classnum, root=model_path)
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
