import os, sys, torch
import numpy as np
import torch.utils.data as data
import pdb

class TestEpisodeMetadata(data.Dataset):

  def __init__(self, root, name, total):
    self.name = name 
    if name is None:
      self.root_dir = root
    else:
      self.root_dir = os.path.join(root, name)
    self.total = total
    self.files = []
    num=0
    for index in range(total):
        xfile = os.path.join(self.root_dir, '{:06d}.pth'.format(index)) # '/data/QDA_FSL/Code_by_Others/urt/src/train-10000/000000.pth'
#         pdb.set_trace()
#         if not os.path.exists(xfile):
#             num+=1
#             print(xfile.replace('/data/QDA_FSL/Code_by_Others/urt/src/',''))
        assert os.path.exists(xfile), '{:}'.format(xfile)
        self.files.append(xfile)
  def __getitem__(self, index):
    xfile = self.files[index]
    xdata = torch.load(xfile, map_location='cpu')
    context_features = xdata['context_features']
    context_labels   = xdata['context_labels']
    target_features  = xdata['target_features']
    target_labels    = xdata['target_labels']
    return torch.IntTensor([index]), context_features, context_labels, target_features, target_labels

  def __len__(self):
    return len(self.files)


class TrainEpisodeMetadata(data.Dataset):

  def __init__(self, root, name, total):
    self.name = name 
    if name is None:
      self.root_dir = root
    else:
      self.root_dir = os.path.join(root, name)
    self.total = total
    self.files = []
    num=0
    for index in range(total):
        xfile = os.path.join(self.root_dir, '{:06d}.pth'.format(index)) # '/data/QDA_FSL/Code_by_Others/urt/src/train-10000/000000.pth'
#         pdb.set_trace()
#         if not os.path.exists(xfile):
#             num+=1
#             print(xfile.replace('/data/QDA_FSL/Code_by_Others/urt/src/',''))
        assert os.path.exists(xfile), '{:}'.format(xfile)
        self.files.append(xfile)
  def __getitem__(self, index):
    xfile = self.files[index]
    xdata = torch.load(xfile, map_location='cpu')
    context_features = xdata['context_features']
    context_labels   = xdata['context_labels']
    target_features  = xdata['target_features']
    target_labels    = xdata['target_labels']
#     return torch.IntTensor([index]), context_features, context_labels, target_features, target_labels
#     pdb.set_trace()
#     print('{:} {:}'.format(xdata['ep_domain'], xdata['domain_idx']))
    return xdata['domain_idx'], context_features, context_labels, target_features, target_labels

  def __len__(self):
    return len(self.files)
    
