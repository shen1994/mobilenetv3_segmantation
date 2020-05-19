import os
from .net import *

def get_segmentation_model(model, **kwargs):
    models = {
        'mobilenetv3_large': get_mobilenetv3_large_seg,
        'mobilenetv3_small': get_mobilenetv3_small_seg,
    }
    return models[model](**kwargs)
