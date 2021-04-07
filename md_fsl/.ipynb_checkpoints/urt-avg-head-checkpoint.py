#!/usr/bin/env python3
import os, sys, time, argparse
import collections
import pdb
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tabulate import tabulate
import random, json
from pathlib import Path
urt_lib_dir = (Path(__file__).parent / 'lib').resolve()
mqda_lib_dir = Path(__file__).parent.resolve().parent
if str(urt_lib_dir) not in sys.path: sys.path.insert(0, str(urt_lib_dir))
if str(mqda_lib_dir) not in sys.path: sys.path.insert(0, str(mqda_lib_dir))

from datasets import get_eval_datasets, get_train_dataset
from data.meta_dataset_reader import TRAIN_METADATASET_NAMES, ALL_METADATASET_NAMES
from models.model_utils import cosine_sim
from models.new_model_helpers import extract_features
from models.losses import prototype_loss
from models.models_dict import DATASET_MODELS_DICT
from models.new_prop_prototype import MultiHeadURT, MultiHeadURT_value, get_lambda_urt_avg, apply_urt_avg_selection 
from utils import show_results, pre_load_results
from code_lib.metric import AverageMeter
from code_lib.util import convert_secs2time
from paths import META_RECORDS_ROOT

from config_utils import Logger

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def load_config():

  parser = argparse.ArgumentParser(description='Train URT networks')
  parser.add_argument('--save_dir', default='../active_log/urt-avg_exp', type=str, help="The saved path in dir.")
  parser.add_argument('--cache_dir', default='../../code_others/data_urt/src', type=str, help="The saved path in dir.")
  parser.add_argument('--seed', default=0, type=int, help="The random seed.")
  parser.add_argument('--interval.train', type=int, default=100, help='The number to log training information')
  parser.add_argument('--interval.test', type=int, default=10000, help='The number to log training information')
  parser.add_argument('--interval.train.reset', type=int, default=500, help='The number to log training information')

  # model args
  parser.add_argument('--model.backbone', default='resnet18', help="Use ResNet18 for experiments (default: False)")
  parser.add_argument('--model.classifier', type=str, default='cosine', choices=['none', 'linear', 'cosine'], help="Do classification using cosine similatity between activations and weights")

  # urt model 
  parser.add_argument('--urt.variant', type=str)
  parser.add_argument('--urt.temp', default=1, type=str)
  parser.add_argument('--urt.head', default=2, type=int)
  parser.add_argument('--urt.penalty_coef', default=0.1, type=float)
  # train args
  parser.add_argument('--train.max_iter', type=int, default=10000, help='number of epochs to train (default: 10000)')
  parser.add_argument('--train.weight_decay', type=float, default=1e-5, help="weight decay coef")
  parser.add_argument('--train.optimizer', type=str, help='optimization method (default: momentum)')

  parser.add_argument('--train.scheduler', type=str, help='optimization method (default: momentum)')
  parser.add_argument('--train.learning_rate', type=float, default=1e-2,help='learning rate (default: 0.0001)')
  parser.add_argument('--train.lr_decay_step_gamma', type=float, default=0.9, metavar='DECAY_GAMMA')
  parser.add_argument('--train.lr_step', type=int, help='the value to divide learning rate by when decayin lr')

  xargs = vars(parser.parse_args())
  return xargs

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d-%X'
  string = '[{:}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def get_cosine_logits(selected_target, proto, temp):
  n_query, feat_dim   = selected_target.shape
  n_classes, feat_dim = proto.shape 
  logits = temp * F.cosine_similarity(selected_target.view(n_query, 1, feat_dim), proto.view(1, n_classes, feat_dim), dim=-1)
  return logits


def test_all_dataset(xargs, test_loaders, URT_model, logger, writter, mode, training_iter, cosine_temp):
  URT_model.eval()
  our_name   = 'urt_exp'
  accs_names = [our_name]
  alg2data2accuracy = collections.OrderedDict()
  alg2data2accuracy['sur-paper'], alg2data2accuracy['urt-paper'], alg2data2accuracy['urt-exp_record'] = pre_load_results()
  alg2data2accuracy[our_name] = {name: [] for name in test_loaders.keys()}

  logger.print('\n{:} starting evaluate the {:} set at the {:}-th iteration.'.format(time_string(), mode, training_iter))
  with torch.no_grad():
      for idata, (test_dataset, loader) in enumerate(test_loaders.items()):
        logger.print('===>>> {:} --->>> {:02d}/{:02d} --->>> {:}'.format(time_string(), idata, len(test_loaders), test_dataset))
        our_losses = AverageMeter()
        for idx, (_, context_features, context_labels, target_features, target_labels) in enumerate(loader):
          context_features, context_labels = context_features.squeeze(0).cuda(), context_labels.squeeze(0).cuda()
          target_features, target_labels = target_features.squeeze(0).cuda(), target_labels.squeeze(0).cuda()
          n_classes = len(np.unique(context_labels.cpu().numpy()))
          # optimize selection parameters and perform feature selection
          avg_urt_params = get_lambda_urt_avg(context_features, context_labels, n_classes, URT_model, normalize=True)

          urt_context_features = apply_urt_avg_selection(context_features, avg_urt_params, normalize=True)
          urt_target_features  = apply_urt_avg_selection(target_features, avg_urt_params, normalize=True) 
          proto_list  = []
          for label in range(n_classes):
            proto = urt_context_features[context_labels == label].mean(dim=0)
            proto_list.append(proto)
          urt_proto = torch.stack(proto_list)

          #if random.random() > 0.99:
          #  print("urt avg score {}".format(avg_urt_params))
          #  print("-"*20)
          logits = get_cosine_logits(urt_target_features, urt_proto, cosine_temp)
          loss   = F.cross_entropy(logits, target_labels)
          our_losses.update(loss.item())
          predicts = torch.argmax(logits, dim=-1)
          final_acc = torch.eq(target_labels, predicts).float().mean().item()
          alg2data2accuracy[our_name][test_dataset].append(final_acc)
        base_name = '{:}-{:}'.format(test_dataset, mode)
        writter.add_scalar("{:}-our-loss".format(base_name), our_losses.avg, training_iter)
        writter.add_scalar("{:}-our-acc".format(base_name) , np.mean(alg2data2accuracy[our_name][test_dataset]), training_iter)

  dataset_names = list(test_loaders.keys())
  show_results(dataset_names, alg2data2accuracy, ('urt-paper', 'urt-exp_record', our_name), logger.print)
  logger.print("\n")

def main(xargs):

  # set up logger
  log_dir = Path(xargs['save_dir']).resolve()
  log_dir.mkdir(parents=True, exist_ok=True)

  if xargs['seed'] is None or xargs['seed'] < 0:
    seed = len(list(Path(log_dir).glob("*.txt")))
  else:
    seed = xargs['seed']
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

  class_name_dict   = collections.OrderedDict()
  for d in extractor_domains:
    with open("{:}/{:}/dataset_spec.json".format(META_RECORDS_ROOT, d)) as f:
      data = json.load(f) 
      class_name_dict[d] = data['class_names']

  # init prop model
  URT_model  = MultiHeadURT(key_dim=512, query_dim=8*512, hid_dim=1024, temp=1, att="dotproduct", n_head=xargs['urt.head'])
  URT_model  = torch.nn.DataParallel(URT_model)
  URT_model  = URT_model.cuda()
  cosine_temp = nn.Parameter(torch.tensor(10.0).cuda())
  params = [p for p in URT_model.parameters()] + [cosine_temp]

  optimizer  = torch.optim.Adam(params, lr=xargs['train.learning_rate'], weight_decay=xargs['train.weight_decay'])
  logger.print(optimizer)
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=xargs['train.max_iter'])
  logger.print(lr_scheduler)

  # load checkpoint optional
  last_ckp_path = log_dir / 'last-ckp-seed-{:}.pth'.format(seed)
  if last_ckp_path.exists():
    checkpoint  = torch.load(last_ckp_path)
    start_iter  = checkpoint['train_iter'] + 1
    URT_model.load_state_dict(checkpoint['URT_model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    logger.print ('load checkpoint from {:}'.format(last_ckp_path))
  else:
    logger.print ('randomly initialiization')
    start_iter = 0
  max_iter = xargs['train.max_iter']

  our_losses, our_accuracies = AverageMeter(), AverageMeter()
  iter_time, timestamp = AverageMeter(), time.time()

  for index, (_, context_features, context_labels, target_features, target_labels) in enumerate(train_loader):
    if index < start_iter:
        continue
    context_features, context_labels = context_features.squeeze(0).cuda(), context_labels.squeeze(0).cuda()
    target_features, target_labels = target_features.squeeze(0).cuda(), target_labels.squeeze(0).cuda()
    URT_model.train()
    n_classes = len(np.unique(context_labels.cpu().numpy()))
    # optimize selection parameters and perform feature selection
    avg_urt_params = get_lambda_urt_avg(context_features, context_labels, n_classes, URT_model, normalize=True)
    # identity matrix panelize to be sparse, only focus on one aspect
    penalty = torch.pow( torch.norm( torch.transpose(avg_urt_params, 0, 1) @ avg_urt_params - torch.eye(xargs['urt.head']).cuda() ), 2) # scalar:-> torch.mm(a_u_p.t(), a_u_p)
    # [num_sample, encoders, dim=512]=>weight attention => n_samples * (n_head * 512)
    urt_context_features = apply_urt_avg_selection(context_features, avg_urt_params, normalize=True)
    urt_target_features  = apply_urt_avg_selection(target_features, avg_urt_params, normalize=True) 
    proto_list  = []
    for label in range(n_classes):
        proto = urt_context_features[context_labels == label].mean(dim=0)
        proto_list.append(proto)
    urt_proto = torch.stack(proto_list)
    logits = get_cosine_logits(urt_target_features, urt_proto, cosine_temp) 
    loss = F.cross_entropy(logits, target_labels) + xargs['urt.penalty_coef']*penalty
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    with torch.no_grad():
        predicts  = torch.argmax(logits, dim=-1)
        final_acc = torch.eq(target_labels, predicts).float().mean().item()
        our_losses.update(loss.item())
        our_accuracies.update(final_acc * 100)

    if index % xargs['interval.train'] == 0 or index+1 == max_iter:
        logger.print("{:} [{:5d}/{:5d}] [OUR] lr: {:}, loss: {:.5f}, accuracy: {:.4f}, still need {:}".format(time_string(), index, max_iter, lr_scheduler.get_last_lr(), our_losses.avg, our_accuracies.avg, convert_secs2time(iter_time.avg * (max_iter - index), True)))
        writter.add_scalar("lr", lr_scheduler.get_last_lr()[0], index)
        writter.add_scalar("train_loss", our_losses.avg, index)
        writter.add_scalar("train_acc", our_accuracies.avg, index)
        
    if index % xargs['interval.train.reset'] == 0 or index+1 == max_iter:
        # Reset the count
        our_losses.reset()
        our_accuracies.reset()
        
        # Save model
        with torch.no_grad():
            info = {'args'      : xargs,
                'train_iter': index,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : lr_scheduler.state_dict(),
                'URT_model' : URT_model.state_dict()}
            torch.save(info, "{:}/ckp-seed-{:}-iter-{:}.pth".format(log_dir, seed, index))
            torch.save(info, last_ckp_path)

    # measure time
    iter_time.update(time.time() - timestamp)
    timestamp = time.time()

    if (index+1) % xargs['interval.test'] == 0 or index+1 == max_iter:
        test_all_dataset(xargs, val_loaders,  URT_model, logger, writter, "eval", index, cosine_temp)
  test_all_dataset(xargs, test_loaders, URT_model, logger, writter, "test", index, cosine_temp)


if __name__ == '__main__':
    xargs = load_config()
    main(xargs)
