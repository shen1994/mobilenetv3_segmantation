"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .cityscapes import CitySegmentation

def get_segmentation_dataset(**kwargs):
    """Segmentation Datasets"""
    return CitySegmentation(**kwargs)
