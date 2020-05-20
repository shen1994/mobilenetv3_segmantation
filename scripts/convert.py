import os
import cv2
import sys
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torchvision
import torch
from torch.autograd import Variable
import onnx
print(torch.__version__)
from core.model import get_segmentation_model

input_name = ['input']
output_name = ['output']
input = Variable(torch.randn(1, 3, None, None)).cuda()
model = get_segmentation_model(model="mobilenetv3_large", classnum=5,
                               aux=False, pretrained=True, pretrained_base=False)
torch.onnx.export(model, input, 
	'save/mobilenetv3_large.onnx', 
	input_names=input_name, 
	output_names=output_name, verbose=True)