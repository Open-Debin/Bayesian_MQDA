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
from basic_code.classifiers_define import NearestCentroid_SimpleShot as NCC
from sklearn.neighbors import NearestCentroid

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
    parser.add_argument('--fea_way', default='mean', type=str, help='cat or mean the features for the input of mqda')
    parser.add_argument('--fea_dim', default=512, type=int, help='dimensional of the features for the input of mqda')
    parser.add_argument('--fea_norm', default='L2N', type=str, help='')
    parser.add_argument('--device', default='cpu', type=str, help='')
    xargs = vars(parser.parse_args())
    return xargs


def get_cosine_logits(selected_target, proto, temp):
    n_query, feat_dim   = selected_target.shape
    n_classes, feat_dim = proto.shape 
    logits = temp * F.cosine_similarity(selected_target.view(n_query, 1, feat_dim), proto.view(1, n_classes, feat_dim), dim=-1)
    return logits

def test_all_dataset_mqda(xargs, test_loaders, logger, writter, mode):
  our_name   = 'ncc'
  accs_names = [our_name]
  alg2data2accuracy = collections.OrderedDict()
  alg2data2accuracy['sur-paper'], alg2data2accuracy['urt'], alg2data2accuracy['mqda_base'] = pre_load_results()
  alg2data2accuracy[our_name] = {name: [] for name in test_loaders.keys()}

  logger.print('\n{:} starting evaluate the {:}.'.format(time_string(), mode))
  for idata, (test_dataset, loader) in enumerate(test_loaders.items()):
    logger.print('===>>> {:} --->>> {:02d}/{:02d} --->>> {:}'.format(time_string(), idata, len(test_loaders), test_dataset))
    our_losses = AverageMeter()
    for idx, (_, context_features, context_labels, target_features, target_labels) in enumerate(loader):
      context_features, context_labels = context_features.squeeze(0).to(torch.device(xargs['device'])), context_labels.squeeze(0).to(torch.device(xargs['device']))
      target_features, target_labels = target_features.squeeze(0).to(torch.device(xargs['device'])), target_labels.squeeze(0).to(torch.device(xargs['device']))
      '''MQDA===>'''
      if xargs['fea_norm'] == "L2N":
        context_features = F.normalize(context_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)
      if xargs['fea_way'] == "mean":
        context_features = context_features.mean(dim=1)
        target_features = target_features.mean(dim=1)
      if xargs['fea_way'] == "cat":
        n_samples, _, _ = context_features.shape
        context_features = context_features.view(n_samples, -1)
        n_samples, _, _ = target_features.shape
        target_features = target_features.view(n_samples, -1)
        # PCA
        context_features = torch.matmul(context_features, xargs['PCA_V_matrix'][:, :xargs['fea_dim']])
        target_features = torch.matmul(target_features, xargs['PCA_V_matrix'][:, :xargs['fea_dim']])
        
        context_features = F.normalize(context_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)        
      clf = NearestCentroid()
      clf.fit(context_features.numpy(), context_labels.numpy())
#       logits = clf.predict(target_features)
      '''=====>MQDA'''
      with torch.no_grad():
        final_acc = clf.score(target_features.numpy(), target_labels.numpy())
        alg2data2accuracy[our_name][test_dataset].append(final_acc)
    base_name = '{:}-{:}'.format(test_dataset, mode)
    writter.add_scalar("{:}-our-loss".format(base_name), our_losses.avg)
    writter.add_scalar("{:}-our-acc".format(base_name) , np.mean(alg2data2accuracy[our_name][test_dataset]))


  dataset_names = list(test_loaders.keys())
  torch.save(alg2data2accuracy, '{:}/perform-seed-{:}-{:}.pth'.format(xargs['save_dir'], xargs['seed'], mode))
  show_results(dataset_names, alg2data2accuracy, ('sur-paper', our_name), logger.print)
  logger.print("\n")

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
  classifier = NCC()
  # load checkpoint optional
  "MQDA===>"
#   pca_fiton_metatrain(xargs, train_loader)
  xargs['PCA_V_matrix'] = torch.load('PCA_S_V_10.pth')[1]
  test_all_dataset_mqda(xargs, val_loaders, logger, writter, "meta_eval")
  test_all_dataset_mqda(xargs, test_loaders, logger, writter, "meta_test")


if __name__ == '__main__':
    xargs = load_config()
    main(xargs)
