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

class NameIdMapping:
    def __init__(self, manyshot_path):
        classes = [d.name for d in os.scandir(manyshot_path) if d.is_dir()]
        classes.sort()
        self.class2idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.way = len(classes)
        
    def name2id(self, support_names):
        index_list = [self.class2idx[self._name2class(name)] for name in support_names ]
        return index_list
        
    def _name2class(self,name):
        cate = os.path.basename(os.path.dirname(name)).split('.')[0]
        return cate

class FoldersIdMapping:
    def __init__(self, folders):
        folders.sort()
        self.class2index = {os.path.basename(folder_path):index for index, folder_path in enumerate(folders)}
        self.index2class = {index:cate for cate, index in self.class2index.items()}
        
    def name2id(self, support_names):
        index_list = [self.class2index[self._name2class(name)] for name in support_names ]
        return index_list
    
    def _name2class(self,name):
        cate = os.path.basename(os.path.dirname(name)).split('.')[0]
        return cate
    
    def id2category(self, ids):
        cate_list = [self.index2class[id] for id in ids ]
        return cate_list
    
    def pure_name2id(self, support_names):
        index_list = [self.class2index[name] for name in support_names ]
        return index_list
        
#=======           
        
def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

def time_now():
  ISOTIMEFORMAT='%d-%h-%Y-%H-%M-%S'
  string = '{:}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def convert_secs2time(epoch_time, string=True):
    hours = int(epoch_time / 3600)
    minutes = int((epoch_time - 3600*hours) / 60)
    seconds = int(epoch_time - 3600*hours - 60*minutes)
    if string:
        time_format = '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
        return time_format
    else:
        return hours, minutes, seconds

class Logger(object):
    def __init__(self, log_dir, title, args=False):
        """Create a summary writer logging to log_dir."""
        self.log_dir = Path("{:}".format(str(log_dir)))
        if not self.log_dir.exists(): os.makedirs(str(self.log_dir))
        self.title = title
        self.log_file = '{:}/{:}_rename_date-{:}.txt'.format(self.log_dir,title, time_now())
        self.file_writer = open(self.log_file, 'a')
        
        if args:
            for key, value in vars(args).items():
                self.print('  [{:18s}] : {:}'.format(key, value))
        self.print('{:} --- args ---'.format(time_now()))
        
    def checkpoint(self, name):
        return self.log_dir / name
    
    def print(self, string, fprint=True, is_pp=False):
        if is_pp: pp.pprint (string)
        else:     print(string)
        if fprint:
          self.file_writer.write('{:}\n'.format(string))
          self.file_writer.flush()
            
    def write(self, string):
        self.file_writer.write('{:}\n'.format(string))
        self.file_writer.flush()  
        
    def rename(self, name):
        new_name = self.log_file.replace('rename',name)
        self.file_writer.close()
        os.rename(self.log_file, new_name)

def search(target, target_space = './', reverse=True):
    ckps = glob.glob("{:}/*.pth".format(target_space, target))
    ckps = natsorted(ckps)
#     ckps.sort()
    if reverse:
        ckps.reverse()
#     else:
        
    return ckps

