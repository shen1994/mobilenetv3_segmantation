import torchvision
import torch
from torch.autograd import Variable
import onnx
print(torch.__version__)

input_name = ['input']
output_name = ['output']
input = Variable(torch.randn(1, 3, None, None)).cuda()
model = get_segmentation_model(model=args.model, classnum=args.classnum,
                                            aux=args.aux, pretrained=True, pretrained_base=False)
torch.onnx.export(model, input, 'resnet50.onnx', input_names=input_name, output_names=output_name, verbose=True)