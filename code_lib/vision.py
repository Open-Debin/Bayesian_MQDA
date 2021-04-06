import os
import sys
import pdb
import glob
import time
import pickle
import pprint
import random
import numpy as np
import scipy as sp
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from pathlib import Path
from natsort import natsorted
from PIL import ImageEnhance

transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out

# For Resnet10
class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


transform05=transforms.Compose(
            [transforms.Resize(84),
             transforms.CenterCrop(84),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# RelationNet2: Xueting Zhang
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                ])
# RelationNet2: Xueting Zhang
train_transform = transforms.Compose([
            transforms.RandomResizedCrop(84, scale=(0.75,1.0)),
            transforms.RandomHorizontalFlip(),
            transform
        ])

# RelationNet2: Xueting Zhang
test_transform = transforms.Compose([
            transforms.Resize(96), # 256->224
            transforms.CenterCrop(84),
            transform
])
# Data augumentation for meta_train+val
meta_trainval_transform = transforms.Compose([
            transforms.Resize(96), # 256->224
            transforms.FiveCrop(84),
            transforms.Lambda(lambda crops: torch.stack([transform(crop) for crop in crops]))
            
])


aug_mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

aug_std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

wrn28_transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(aug_mean['cifar10'], aug_std['cifar10']),
    ])