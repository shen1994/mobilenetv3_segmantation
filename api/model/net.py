import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import SegBaseModel

class MobileNetV3Seg(SegBaseModel):
    def __init__(self, nclass, aux=False, backbone='mobilenetv3_large', pretrained_base=False, **kwargs):
        super(MobileNetV3Seg, self).__init__(nclass, aux, backbone, pretrained_base, **kwargs)
        self.head = _SegHead(nclass, self.mode, **kwargs)
        if aux:
            inter_channels = 40 if self.mode == 'large' else 24
            self.auxlayer = nn.Conv2d(inter_channels, nclass, 1)

    def forward(self, x):
        size = x.size()[2:]
        _, c2, _, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c2)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _SegHead(nn.Module):
    def __init__(self, nclass, mode='large', norm_layer=nn.BatchNorm2d, **kwargs):
        super(_SegHead, self).__init__()
        in_channels = 960 if mode == 'large' else 576
        self.lr_aspp = _LRASPP(in_channels, norm_layer, **kwargs)
        self.project = nn.Conv2d(128, nclass, 1)

    def forward(self, x):
        x = self.lr_aspp(x)
        return self.project(x)


# TODO: check Lite R-ASPP
class _LRASPP(nn.Module):
    """Lite R-ASPP"""

    def __init__(self, in_channels, norm_layer, **kwargs):
        super(_LRASPP, self).__init__()
        out_channels = 128
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )
        self.b1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(48, 48), stride=(16, 20)),  # check it
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        # feat2 = self.b1(x)
        # feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        # x = feat1 * feat2  # check it
        x = feat1
        return x

def get_mobilenetv3_large_seg(classnum=5, root='save/mobilenetv3_large.pth', **kwargs):

    model = MobileNetV3Seg(classnum, aux=False, backbone='mobilenetv3_large',
                           pretrained_base=False, **kwargs)

    model.load_state_dict(torch.load('save/mobilenetv3_large.pth'))

    return model


def get_mobilenetv3_small_seg(classnum=5, root='save/mobilenetv3_small.pth', **kwargs):

    model = MobileNetV3Seg(classnum, aux=False, backbone='mobilenetv3_small',
                           pretrained_base=False, **kwargs)

    model.load_state_dict(torch.load(root))
    
    return model