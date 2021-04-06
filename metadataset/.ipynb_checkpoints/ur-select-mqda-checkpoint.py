#!/usr/bin/env python3
import torch
import pickle
import torch.nn as nn
import torchvision
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))

import os, sys, time, argparse
import collections
import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tabulate import tabulate
import random, json
from pathlib import Path
urt_lib_dir = (Path(__file__).parent / 'lib').resolve()
mqda_lib_dir = (Path(__file__).parent.parent ).resolve()
if str(urt_lib_dir) not in sys.path: sys.path.insert(0, str(urt_lib_dir))
if str(mqda_lib_dir) not in sys.path: sys.path.insert(0, str(mqda_lib_dir))

from datasets import get_eval_datasets, get_train_dataset
from data.meta_dataset_reader import TRAIN_METADATASET_NAMES, ALL_METADATASET_NAMES
from models.model_utils import cosine_sim
from models.new_model_helpers import extract_features
from models.losses import prototype_loss
from models.models_dict import DATASET_MODELS_DICT
from models.new_prop_prototype import MultiHeadURT, MultiHeadURT_value, get_lambda_urt_avg, apply_urt_avg_selection 
from utils import convert_secs2time, time_string, AverageMeter, show_results, pre_load_results
from paths import META_RECORDS_ROOT

from config_utils import Logger

from basic_code import util
from basic_code.classifiers_define import MetaQDA   
from models.ur_select import SurModel
import math
import collections
import inspect
# from tool.gpu_mem_track import  MemTracker
# from tool.gpu_memory_log import gpu_memory_log
# frame = inspect.currentframe()          # define a frame to track
# gpu_tracker = MemTracker(frame)  

def split_data(full_cont_fea, full_cont_label, full_targ_fea, full_targ_label, n_split):
    n_class = max(full_targ_label)
    c_f_dict = collections.defaultdict(list)
    c_l_dict = collections.defaultdict(list)
    t_f_dict = collections.defaultdict(list)
    t_l_dict = collections.defaultdict(list)
    # split
    for id_class in range(n_class+1):
        cont_fea = full_cont_fea[full_cont_label == id_class]
        cont_label = full_cont_label[full_cont_label == id_class]
        targ_fea = full_targ_fea[full_targ_label == id_class]
        targ_label = full_targ_label[full_targ_label == id_class]
        
        len_cont = len(cont_label)
        len_targ = len(targ_label)
        stride_cont = math.floor(len_cont/float(n_split))
        stride_targ = math.floor(len_targ/float(n_split))

        list_cont = [stride_cont] * (n_split-1) + [len_cont -  stride_cont *  (n_split-1)]
        list_targ = [stride_targ] * (n_split-1) + [len_targ -  stride_targ *  (n_split-1)]
        try:
            cont_fea_split = torch.split(cont_fea, list_cont, dim=0)
            cont_label_split = torch.split(cont_label, list_cont, dim=0)
            targ_fea_split = torch.split(targ_fea, list_targ, dim=0)
            targ_label_split = torch.split(targ_label, list_targ, dim=0)
        except:
            pdb.set_trace()
        for index, (c_f, c_l, t_f, t_l) in enumerate(zip(cont_fea_split, cont_label_split, targ_fea_split, targ_label_split)):
            c_f_dict[index].append(c_f)
            c_l_dict[index].append(c_l)
            t_f_dict[index].append(t_f)
            t_l_dict[index].append(t_l)
    # fuse       
    for index in range(n_split):
        c_f_dict[index] = torch.cat(c_f_dict[index])
        c_l_dict[index] = torch.cat(c_l_dict[index])
        t_f_dict[index] = torch.cat(t_f_dict[index])
        t_l_dict[index] = torch.cat(t_l_dict[index])
    # output 
    for index in range(n_split):
        yield c_f_dict[index], c_l_dict[index], t_f_dict[index], t_l_dict[index]
      
def load_config():

    parser = argparse.ArgumentParser(description='Train URT networks')
    parser.add_argument('--save_dir', type=str, help="The saved path in dir.")
    parser.add_argument('--cache_dir', type=str, help="The saved path in dir.")
    parser.add_argument('--seed', default=-1, type=int, help="The random seed.")
    parser.add_argument('--interval.train', type=int, default=100, help='The number to log training information')
    parser.add_argument('--interval.test', type=int, default=2000, help='The number to log training information')
    parser.add_argument('--interval.train.reset', type=int, default=500, help='The number to log training information')

    # model args
    parser.add_argument('--model.backbone', default='resnet18', help="Use ResNet18 for experiments (default: False)")
    parser.add_argument('--model.classifier', type=str, default='cosine', choices=['none', 'linear', 'cosine'], help="Do classification using cosine similatity between activations and weights")

    # urt model 
    parser.add_argument('--urt.variant', type=str)
    parser.add_argument('--urt.temp', type=str)
    parser.add_argument('--urt.head', type=int, default=2)
    parser.add_argument('--urt.penalty_coef', type=float)
    
    # train args
    parser.add_argument('--train.max_iter', type=int, help='number of epochs to train (default: 10000)')
    parser.add_argument('--train.weight_decay', type=float, help="weight decay coef")
    parser.add_argument('--train.optimizer', type=str, help='optimization method (default: momentum)')

    parser.add_argument('--train.scheduler', type=str, help='optimization method (default: momentum)')
    parser.add_argument('--train.learning_rate', type=float, help='learning rate (default: 0.0001)')
    parser.add_argument('--train.lr_decay_step_gamma', type=float, metavar='DECAY_GAMMA')
    parser.add_argument('--train.lr_step', type=int, help='the value to divide learning rate by when decayin lr')
    
    # mqda related
    parser.add_argument('--fea_way', default='cat', type=str, help='cat or mean the features for the input of mqda')
    parser.add_argument('--fea_dim', default=4096, type=int, help='dimensional of the features for the input of mqda')
    parser.add_argument('--fea_norm', default='L2N', type=str, help='')
    parser.add_argument('--n_domains', default=8, type=int, help='')
    parser.add_argument('--accumulation_steps', default=2, type=int, help="parameters is too large！to avoid over memory")
    xargs = vars(parser.parse_args())
    return xargs


def get_cosine_logits(selected_target, proto, temp):
    n_query, feat_dim   = selected_target.shape
    n_classes, feat_dim = proto.shape 
    logits = temp * F.cosine_similarity(selected_target.view(n_query, 1, feat_dim), proto.view(1, n_classes, feat_dim), dim=-1)
    return logits

def test_all_dataset_mqda(xargs, test_loaders, ur_select_model, logger, writter, mode, training_iter, cosine_temp):
  our_name   = 'ur-select-mqda'
  accs_names = [our_name]
  alg2data2accuracy = collections.OrderedDict()
  alg2data2accuracy['sur-paper'], alg2data2accuracy['urt'], alg2data2accuracy['mqda_base'] = pre_load_results()
  alg2data2accuracy[our_name] = {name: [] for name in test_loaders.keys()}

  logger.print('\n{:} starting evaluate the {:} set at the {:}-th iteration.'.format(time_string(), mode, training_iter))
  for idata, (test_dataset, loader) in enumerate(test_loaders.items()):
    logger.print('===>>> {:} --->>> {:02d}/{:02d} --->>> {:}'.format(time_string(), idata, len(test_loaders), test_dataset))
    our_losses = AverageMeter()
    for idx, (_, context_features, context_labels, target_features, target_labels) in enumerate(loader):
      context_features, context_labels = context_features.squeeze(0).cuda(), context_labels.squeeze(0).cuda()
      target_features, target_labels = target_features.squeeze(0).cuda(), target_labels.squeeze(0).cuda()
      # ur-select-mqda
      logits = ur_select_model(context_features, context_labels, target_features)
      '''=====>MQDA'''
      with torch.no_grad():
        loss   = F.cross_entropy(logits, target_labels)
        our_losses.update(loss.item())
        predicts = torch.argmax(logits, dim=-1)
        final_acc = torch.eq(target_labels, predicts).float().mean().item()
        alg2data2accuracy[our_name][test_dataset].append(final_acc)
    base_name = '{:}-{:}'.format(test_dataset, mode)
    writter.add_scalar("{:}-our-loss".format(base_name), our_losses.avg, training_iter)
    writter.add_scalar("{:}-our-acc".format(base_name) , np.mean(alg2data2accuracy[our_name][test_dataset]), training_iter)


  dataset_names = list(test_loaders.keys())
  torch.save(alg2data2accuracy, '{:}/perform-{:}.tar'.format(xargs['save_dir'], xargs['seed'], mode))
  show_results(dataset_names, alg2data2accuracy, ('sur-paper', our_name), logger.print)
  logger.print("\n")

def train_mqda(xargs, train_loader, ur_select_model, logger, writter, mode, start_iter, cosine_temp, optimizer, lr_scheduler):
  max_iter = xargs['train.max_iter']
  if start_iter == max_iter:
        return max_iter - 1
  our_losses, our_accuracies = AverageMeter(), AverageMeter()
  iter_time, timestamp = AverageMeter(), time.time()
  for index, (_, context_features, context_labels, target_features, target_labels) in enumerate(train_loader):
    if index < start_iter:
        continue
#     pdb.set_trace()
    context_features, context_labels = context_features.squeeze(0).cuda(), context_labels.squeeze(0).cuda()
    target_features, target_labels = target_features.squeeze(0).cuda(), target_labels.squeeze(0).cuda()
    # split data
#     for context_features, context_labels, target_features, target_labels in split_data(context_features, context_labels, target_features, target_labels, xargs['accumulation_steps']):
        # ur-select-mqda
    logits = ur_select_model(context_features, context_labels, target_features)
    loss = F.cross_entropy(logits, target_labels)
    # accumulation loss
#     loss = loss / float(xargs['accumulation_steps'])
    loss.backward()
    torch.cuda.empty_cache()
    optimizer.step()   
    optimizer.zero_grad()
    lr_scheduler.step()

    with torch.no_grad():
        predicts  = torch.argmax(logits, dim=-1)
        final_acc = torch.eq(target_labels, predicts).float().mean().item()
        our_losses.update(loss.item())
        our_accuracies.update(final_acc * 100)

    if index % xargs['interval.train'] == 0 or index+1 == max_iter:
      logger.print("{:} [{:5d}/{:5d}] [OUR] lr: {:}, loss: {:.5f}, accuracy: {:.4f}".format(time_string(), index, max_iter, lr_scheduler.get_last_lr(), our_losses.avg, our_accuracies.avg))
      writter.add_scalar("lr", lr_scheduler.get_last_lr()[0], index)
      writter.add_scalar("train_loss", our_losses.avg, index)
      writter.add_scalar("train_acc", our_accuracies.avg, index)
      with torch.no_grad():
          info = {'args'      : xargs,
                'train_iter': index,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : lr_scheduler.state_dict(),
                'sur_mqda' : ur_select_model}
#           torch.save(info, "{:}/ckp-seed-{:}-iter-{:}.pth".format(xargs['save_dir'], xargs['seed'], index))
          torch.save(info, "{:}/ckp-seed-{:}.pth".format(xargs['save_dir'], xargs['seed'], index))
      if index+1 == max_iter:
          last_ckp_path = log_dir / 'last-ckp-seed-{:}.pth'.format(seed)
          torch.save(info, last_ckp_path)

      # Reset the count
      if index % xargs['interval.train.reset'] == 0:
            our_losses.reset()
            our_accuracies.reset()
      time_str  = convert_secs2time(iter_time.avg * (max_iter - index), True)
      logger.print("iteration [{:5d}/{:5d}], still need {:}".format(index, max_iter, time_str))
    
    # measure time
    iter_time.update(time.time() - timestamp)
    timestamp = time.time()
  return index

def main(xargs):
  # set up logger
  xargs['save_dir'] = xargs['save_dir'] +'/seed_'+str(xargs['seed'])
  log_dir = Path(xargs['save_dir']).resolve()
  log_dir.mkdir(parents=True, exist_ok=True)

  if xargs['seed'] is None or xargs['seed'] < 0:
    seed = len(list(Path(log_dir).glob("*.txt")))
  else:
    seed = xargs['seed']
  xargs['seed'] = seed
  random.seed(seed)
  torch.manual_seed(seed)
  logger = Logger(str(log_dir), seed)
  logger.print('{:} --- args ---'.format(time_string()))
  for key, value in xargs.items():
    logger.print('  [{:10s}] : {:}'.format(key, value))
  logger.print('{:} --- args ---'.format(time_string()))
  writter = SummaryWriter(log_dir)
  # Setting up datasets
  extractor_domains = TRAIN_METADATASET_NAMES
  train_dataset     = get_train_dataset(xargs['cache_dir'], xargs['train.max_iter'])
  train_loader      = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True) 
  # The validation loaders.
  val_datasets = get_eval_datasets(os.path.join(xargs['cache_dir'], 'val-600'), TRAIN_METADATASET_NAMES)
  val_loaders = collections.OrderedDict()
  for name, dataset in val_datasets.items():
    val_loaders[name] = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
  # The test loaders
  test_datasets     = get_eval_datasets(os.path.join(xargs['cache_dir'], 'test-600'), ALL_METADATASET_NAMES)
  test_loaders = collections.OrderedDict()
  for name, dataset in test_datasets.items():
    test_loaders[name] = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    
  # init prop model
  cosine_temp = nn.Parameter(torch.tensor(10.0).cuda())
  classifier = MetaQDA(reg_param=0.3, fea_dim = xargs['fea_dim'], input_process=xargs['fea_norm'])
  ur_select_model = SurModel(classifier, n_domains=xargs['n_domains']).to(torch.device("cuda"))
  optimizer  = torch.optim.Adam(ur_select_model.parameters(), lr=xargs['train.learning_rate'], weight_decay=xargs['train.weight_decay'])
  logger.print(optimizer)
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=xargs['train.max_iter'])
  logger.print(lr_scheduler)

  # load checkpoint optional
#   last_ckp_path = log_dir / 'last-ckp-seed-{:}.pth'.format(seed)
  last_ckp_path = log_dir /'ckp-seed-0.pth'
  if last_ckp_path.exists():
    checkpoint  = torch.load(last_ckp_path)
    start_iter  = checkpoint['train_iter'] + 1
    ur_select_model = checkpoint['sur_mqda']
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    logger.print ('load checkpoint from {:}'.format(last_ckp_path))
  else:
    logger.print ('randomly initialiization')
    start_iter = 0

  "MQDA===>"
  index = train_mqda(xargs, train_loader, ur_select_model, logger, writter, 'meta_train', start_iter, cosine_temp, optimizer, lr_scheduler)
#     if (index+1) % xargs['interval.test'] == 0 or index+1 == max_iter:
  test_all_dataset_mqda(xargs, val_loaders, ur_select_model, logger, writter, "meta_eval", index, cosine_temp)
  test_all_dataset_mqda(xargs, test_loaders, ur_select_model, logger, writter, "meta_test", index, cosine_temp)


if __name__ == '__main__':
    xargs = load_config()
    main(xargs)
