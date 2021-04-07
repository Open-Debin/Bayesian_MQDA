import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim.optimizer import Optimizer
import math

def obtain_accuracy(output, target, topk=(1,)):
  with torch.no_grad():
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # bs*k
    pred = pred.t()  # t: transpose, k*bs
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 1*bs --> k*bs

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


def obtain_per_class_accuracy(predictions, xtargets):
  top1 = torch.argmax(predictions, dim=1)
  cls2accs = []
  for cls in sorted(list(set(xtargets.tolist()))):
    selects  = xtargets == cls
    accuracy = (top1[selects] == xtargets[selects]).float().mean() * 100
    cls2accs.append( accuracy.item() )
  return sum(cls2accs) / len(cls2accs)

def spm_to_tensor(sparse_mx):
  sparse_mx = sparse_mx.tocoo().astype(np.float32)
  indices = torch.from_numpy(np.vstack(
          (sparse_mx.row, sparse_mx.col))).long()
  values = torch.from_numpy(sparse_mx.data)
  shape = torch.Size(sparse_mx.shape)
  return torch.sparse.FloatTensor(indices, values, shape)
