#!/usr/bin/env python3
import pickle
import collections
import os, sys, time, argparse
import pdb
import torch
print(torch.__version__)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
urt_lib_dir = (Path(__file__).parent / 'lib').resolve()
mqda_lib_dir = Path(__file__).parent.resolve().parent
if str(urt_lib_dir) not in sys.path: sys.path.insert(0, str(urt_lib_dir))
if str(mqda_lib_dir) not in sys.path: sys.path.insert(0, str(mqda_lib_dir))

from datasets import get_eval_datasets, get_train_dataset
from data.meta_dataset_reader import TRAIN_METADATASET_NAMES, ALL_METADATASET_NAMES
from utils import show_results, pre_load_results
from code_lib import util, save, load, metric
from code_models import config, losses
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# CUDA_VISIBLE_DEVICES=0 python ur-mean-cat-mqda.py -title map_mean_norm_1e-3_sgd_reg0.0_ --lr 1e-3 --x_norm false
def main():
    args = load_config()
    util.set_random(args.seed)
    args.save_dir = Path(args.log_dir) / args.title
    logger = util.Logger(args.save_dir, args.seed, args)
    train_dataset = get_train_dataset(args.x_cache_dir, args.n_epoch * args.n_episode_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True) 
    val_datasets = get_eval_datasets(os.path.join(args.x_cache_dir, 'val-600'), TRAIN_METADATASET_NAMES)
    val_loaders = collections.OrderedDict()
    for name, dataset in val_datasets.items():
        val_loaders[name] = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    test_datasets = get_eval_datasets(os.path.join(args.x_cache_dir, 'test-600'), ALL_METADATASET_NAMES)
    test_loaders = collections.OrderedDict()
    for name, dataset in test_datasets.items():
        test_loaders[name] = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    
    classifier = config.meta_qda(args).to(DEVICE)  
    optimizer = config.optimizer(classifier.parameters(), args)    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_episode_train*args.n_epoch)
    logger.print(optimizer)
    logger.print(lr_scheduler)
    
    ckps = util.search('*pth', target_space = args.save_dir)
    if ckps:
        classifier, args, optimizer_state, lr_schedule_state = load.checkpoint(ckps[0])
        start_iter = args.index_episode + 1
        optimizer.load_state_dict(optimizer_state)
        lr_scheduler.load_state_dict(lr_schedule_state)
        logger.print ('load checkpoint from {:}'.format(ckps[0]))
    else:
        logger.print ('randomly initialiization')
        start_iter = 0
        
    args.pca_v_matrix = torch.load('PCA_S_V_10.pth')[1].cuda()
    index = train_mqda(args, train_loader, classifier, logger, 'meta_train', start_iter, optimizer, lr_scheduler)
#     test_all_dataset_mqda(args, val_loaders,  classifier, logger, "meta_eval", index)
    test_all_dataset_mqda(args, test_loaders, classifier, logger, "meta_test", index)

def test_all_dataset_mqda(args, test_loaders, classifier, logger, mode, training_iter):
    our_name = 'mqda'
    accs_names = [our_name]
    alg2data2accuracy = collections.OrderedDict()
    alg2data2accuracy['sur-paper'], alg2data2accuracy['urt'], alg2data2accuracy['mqda_base'] = pre_load_results()
    alg2data2accuracy[our_name] = {name: [] for name in test_loaders.keys()}
    
    logger.print('\n{:} starting evaluate the {:} set at the {:}-th iteration.'.format(util.time_now(), mode, training_iter))
    with torch.no_grad():
        for dataset_id, (test_dataset, loader) in enumerate(test_loaders.items()):
            logger.print('===>>> {:} --->>> {:02d}/{:02d} --->>> {:}'.format(util.time_now(), dataset_id, len(test_loaders), test_dataset))
            our_losses = metric.AverageMeter()
            for episode_index, (_, context_features, context_labels, target_features, target_labels) in enumerate(loader):
                context_features, context_labels = context_features.squeeze(0).to(DEVICE), context_labels.squeeze(0).to(DEVICE)
                target_features, target_labels = target_features.squeeze(0).to(DEVICE), target_labels.squeeze(0).to(DEVICE)
                if args.x_norm == "L2N":
                    context_features = F.normalize(context_features, dim=-1)
                    target_features = F.normalize(target_features, dim=-1)
                if args.x_fusion == "mean":
                    context_features = context_features.mean(dim=1)
                    target_features = target_features.mean(dim=1)
                    #context_features = F.normalize(context_features, dim=-1)
                    #target_features = F.normalize(target_features, dim=-1)
                if args.x_fusion == "cat":
                    n_samples, _, _ = context_features.shape
                    context_features = context_features.view(n_samples, -1)
                    n_samples, _, _ = target_features.shape
                    target_features = target_features.view(n_samples, -1)

                    context_features = torch.matmul(context_features, args.pca_v_matrix[:, :args.feature.dim])
                    target_features = torch.matmul(target_features, args.pca_v_matrix[:, :args.feature.dim])
                    context_features = F.normalize(context_features, dim=-1)
                    target_features = F.normalize(target_features, dim=-1)        

                classifier.fit_image_label(context_features, context_labels) # compute \mu and \Sigma on support set
                logits = classifier.predict(target_features)
                loss = F.cross_entropy(logits, target_labels)
                our_losses.update(loss.item())
                final_acc = torch.eq(target_labels, torch.argmax(logits, dim=-1)).float().mean().item()
                alg2data2accuracy[our_name][test_dataset].append(final_acc)
#                 break
            base_name = '{:}-{:}'.format(test_dataset, mode)

    dataset_names = list(test_loaders.keys())
    torch.save(alg2data2accuracy, '{:}/perform-{:}.tar'.format(args.save_dir, args.seed, mode))
    show_results(dataset_names, alg2data2accuracy, ('sur-paper', our_name), logger.print)
    logger.print("\n")

def train_mqda(args, train_loader, classifier, logger, mode, start_iter, optimizer, lr_scheduler):
    if start_iter == args.n_epoch * args.n_episode_train:
        return args.n_episode - 1
    our_losses, our_accuracies = metric.AverageMeter(), metric.AverageMeter()
    timer = metric.CountdownTimer(total_steps=args.n_epoch * args.n_episode_train )
    for episode_index, (_, context_features, context_labels, target_features, target_labels) in enumerate(train_loader):
#         break
        if episode_index < start_iter:
            continue
        context_features, context_labels = context_features.squeeze(0).to(DEVICE), context_labels.squeeze(0).to(DEVICE)
        target_features, target_labels = target_features.squeeze(0).to(DEVICE), target_labels.squeeze(0).to(DEVICE)

        if args.x_norm == "L2N":
            context_features = F.normalize(context_features, dim=-1)
            target_features = F.normalize(target_features, dim=-1)
        if args.x_fusion == "mean":
            context_features = context_features.mean(dim=1)
            target_features = target_features.mean(dim=1)
#             context_features = F.normalize(context_features, dim=-1)
#             target_features = F.normalize(target_features, dim=-1)
        if args.x_fusion == "cat":
            n_samples, _, _ = context_features.shape
            context_features = context_features.view(n_samples, -1)
            n_samples, _, _ = target_features.shape
            target_features = target_features.view(n_samples, -1)
            # PCA
            context_features_pca = torch.matmul(context_features, args.pca_v_matrix[:, :args.feature_dim])
            target_features_pca = torch.matmul(target_features, args.pca_v_matrix[:args.feature_dim])

            context_features_pca = F.normalize(context_features_pca, dim=-1)
            target_features_pca = F.normalize(target_features_pca, dim=-1)
        
        classifier.fit_image_label(context_features, context_labels) # compute \mu and \Sigma on support set
        logits = classifier.predict(target_features)
        ce_loss = F.cross_entropy(logits, target_labels)
        "===>MQDA"
        chlsk_loss = args.cholesky_alpha * losses.cholesky_loss(classifier.lower_triu)
        loss = ce_loss + chlsk_loss
        losses.backward_propagation(loss, optimizer)
        lr_scheduler.step()

        final_acc = torch.eq(target_labels, torch.argmax(logits, dim=-1)).float().mean().item()
        our_losses.update(loss.item())
        our_accuracies.update(final_acc * 100)

        if episode_index % args.n_episode_train == 0 or episode_index+1 == args.n_epoch * args.n_episode_train:
            logger.print("[{:5d}/{:5d}] lr: {:}, loss: {:.5f}, accuracy: {:.4f}, still need {:} {:}".format(episode_index, args.n_epoch * args.n_episode_train, lr_scheduler.get_last_lr(), our_losses.avg, our_accuracies.avg, timer.step(), util.time_now()))
            save.checkpoint_episode(args, episode_index, optimizer.state_dict(), lr_scheduler.state_dict(), classifier.state_dict(), args.save_dir)
            our_losses.reset()
            our_accuracies.reset()
    return episode_index

def load_config():
    parser = argparse.ArgumentParser(description='Train URT networks')
    parser.add_argument('-l_d','--log_dir', default='../active_log',type=str, help="The saved path in dir.")
    parser.add_argument('-title','--title', default='dev', type=str, help="The saved path in dir.")
    parser.add_argument('--x_cache_dir', default='../../code_others/data_urt/src', type=str, help="The saved path in dir.")
    parser.add_argument('--seed', default=0, type=int, help="The random seed.")
    parser.add_argument('--n_epoch', type=int, default=100, help='The number to log training information')
    parser.add_argument('--n_episode_train', default=100, type=int, help='number of episode to train (default: 10000)')
    parser.add_argument('--index_episode', default=0, type=int, help='')
    
    # train args
    parser.add_argument('--optimizer', default='sgd',type=str, help='optimization method (default: momentum)')
    parser.add_argument('-lr','--lr', default=1e-4, type=float, help='learning rate (default: 0.0001)')
    
    parser.add_argument('--x_fusion', default='mean', type=str, help='cat or mean the features for the input of mqda')
    parser.add_argument('--x_dim', default=512, type=int, help='dimensional of the features for the input of mqda')
    parser.add_argument('--x_norm', default='L2N', type=str, help='')
    
    # mqda related
    parser.add_argument('--cholesky_alpha',default=0, type=float,help="hope cholesky_alpha > feature dim + 1 ")
    parser.add_argument('-strategy','--strategy', default='map', type=str, help='')
    parser.add_argument('--reg_param', default=0.0, type=float, help='')
    parser.add_argument('--Nj', default=1, type=int, help='')
    
    return parser.parse_args()

if __name__ == '__main__':
    main()
