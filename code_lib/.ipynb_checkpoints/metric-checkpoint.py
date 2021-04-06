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
from code_lib import util

def auroc(distance, open_closed_indicate, open_size, close_size, descending=False):
    index = [ e for e in range(len(distance))]
    random.shuffle(index)
    distance=distance[index]
    open_closed_indicate=open_closed_indicate[index]
    _, position = distance.sort(descending=descending)
    ranked_indicate = open_closed_indicate[position]
    sum_open_position_rank = 0
    for rank_id, set_indicate in enumerate(ranked_indicate):
        if set_indicate == 1:
            sum_open_position_rank+=(rank_id+1)
    # reference: https://www.cnblogs.com/hit-joseph/p/11448792.html
    auroc_numerator = sum_open_position_rank - 0.5*open_size*(open_size+1)
    auroc_denominator = open_size*close_size
        
    return auroc_numerator/auroc_denominator

def accuracy(predicts, targets):
    """Computes the precision@k for the specified values of k"""
    if len(predicts) != len(targets):
        raise ValueError(f'len(output_pred) == len(target)')
    rewards = [1 if item_predict == item_target else 0 for item_predict, item_target in zip(predicts, targets)]
    return round(np.sum(rewards)/float(len(rewards)) * 100,2)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val_list = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.val_list.append(val)
        
    def mean_confidence_interval(self):
        return mean_confidence_interval(self.val_list)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return round(m,2),round(h,2)


class CountdownTimer:
    def __init__(self, total_steps):
        self.iter_time = AverageMeter()
        self.timestamp = time.time()
        self.total_steps = total_steps
        self.now_step = 0
        
    def step(self):
        self.iter_time.update(time.time() - self.timestamp)
        self.timestamp = time.time()
        self.now_step += 1
        return util.convert_secs2time(self.iter_time.avg * (self.total_steps - self.now_step), True)

class BestAcc:
    def __init__(self):
        self.value = 0
    def update(self, acc, h_, index):
        if acc > self.value:
            self.value = acc
            self.h_ = h_
            self.index = index