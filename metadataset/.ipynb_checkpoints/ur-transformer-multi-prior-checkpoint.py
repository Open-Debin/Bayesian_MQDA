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
from itertools import chain
from pathlib import Path
urt_lib_dir = (Path(__file__).parent / 'lib').resolve()
mqda_lib_dir = Path(__file__).parent.resolve().parent
if str(urt_lib_dir) not in sys.path: sys.path.insert(0, str(urt_lib_dir))
if str(mqda_lib_dir) not in sys.path: sys.path.insert(0, str(mqda_lib_dir))
from models.model_utils import cosine_sim
from models.new_prop_prototype import MultiHeadURT, MultiHeadURT_value, get_lambda_urt_avg, apply_urt_avg_selection 
from datasets import get_eval_datasets, get_train_dataset
from data.meta_dataset_reader import TRAIN_METADATASET_NAMES, ALL_METADATASET_NAMES
from utils import show_results, pre_load_results
from code_lib import util, save, load, metric
from code_models import config, losses
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# CUDA_VISIBLE_DEVICES=5 python ur-transformer-multi-prior.py -title multiprior_select_map_3e-4_adam_reg0.95_oracle --lr 1e-4 --x_norm false
def main():
    args = load_config()
    util.set_random(args.seed)
    args.save_dir = Path(args.log_dir) / args.title
    logger = util.Logger(args.save_dir, args.seed, args)
    train_dataset = get_train_dataset(args.x_cache_dir, args.n_epoch * args.n_episode_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True) 
    val_datasets = get_eval_datasets(os.path.join(args.x_cache_dir, 'val-600'), TRAIN_METADATASET_NAMES)
    val_loaders = collections.OrderedDict()
    for name, dataset in val_datasets.items():
        val_loaders[name] = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True).__iter__()
    test_datasets = get_eval_datasets(os.path.join(args.x_cache_dir, 'test-600'), ALL_METADATASET_NAMES)
    test_loaders = collections.OrderedDict()
    for name, dataset in test_datasets.items():
        test_loaders[name] = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True).__iter__()    
   
    # init prop model
    URT_model  = MultiHeadURT(key_dim=512, query_dim=8*512, hid_dim=1024, temp=1, att="dotproduct", n_head=args.urt_head)
    URT_model  = torch.nn.DataParallel(URT_model)
    URT_model  = URT_model.cuda()
    cosine_temp = nn.Parameter(torch.tensor(10.0).cuda())
    params = [p for p in URT_model.parameters()] + [cosine_temp]    
    classifier = config.multi_prior(args).to(DEVICE)
    optimizer = config.optimizer(chain(params, classifier.parameters()), args)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_episode_train*args.n_epoch+5000)
    logger.print(optimizer)
    logger.print(lr_scheduler)
    ckps = util.search('*pth', target_space = args.save_dir)
    if ckps:
#         pdb.set_trace()
        urt_state, classifier, args, optimizer_state, lr_schedule_state = load.urt_multiprior(ckps[0])
        URT_model.load_state_dict(urt_state)
        start_iter = args.index_episode + 1
#         pdb.set_trace()
        optimizer.load_state_dict(optimizer_state)
        lr_scheduler.load_state_dict(lr_schedule_state)
        logger.print('load checkpoint from {:}'.format(ckps[0]))
    else:
        logger.print('randomly initialiization')
        start_iter = 0
        
    args.pca_v_matrix = torch.load('PCA_S_V_10.pth')[1].cuda()
    train_mqda(args, train_loader, URT_model, classifier, logger, start_iter, optimizer, lr_scheduler)
    optimizer = config.optimizer(classifier.prior_weight.parameters(), args)    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_episode_train*args.n_epoch+5000)
    val_mqda(args, val_loaders, URT_model, classifier, logger, start_iter, optimizer, lr_scheduler)
    test_all_dataset_mqda(args, test_loaders, URT_model, classifier, logger, "meta_test")

def train_mqda(args, train_loader, URT_model, classifier, logger, start_iter, optimizer, lr_scheduler):
    print('begin train')
    URT_model.train()
    if start_iter == args.n_epoch * args.n_episode_train:
        return start_iter - 1
    our_losses, our_accuracies = metric.AverageMeter(), metric.AverageMeter()
    timer = metric.CountdownTimer(total_steps=args.n_epoch * args.n_episode_train )
    for episode_index, (domain_id, x_spt, y_spt, x_qry, y_qry) in enumerate(train_loader):
        if episode_index < start_iter:
            continue
        x_spt, y_spt = x_spt.squeeze(0).to(DEVICE), y_spt.squeeze(0).to(DEVICE)
        x_qry, y_qry = x_qry.squeeze(0).to(DEVICE), y_qry.squeeze(0).to(DEVICE)

        n_classes = len(np.unique(y_spt.cpu().numpy()))
        # optimize selection parameters and perform feature selection
        avg_urt_params = get_lambda_urt_avg(x_spt, y_spt, n_classes, URT_model, normalize=True)
        # identity matrix panelize to be sparse, only focus on one aspect
        penalty = torch.pow( torch.norm( torch.transpose(avg_urt_params, 0, 1) @ avg_urt_params - torch.eye(args.urt_head).cuda() ), 2) # scalar:-> torch.mm(a_u_p.t(), a_u_p)
        # [num_sample, encoders, dim=512]=>weight attention => n_samples * (n_head * 512)
        urt_x_spt = apply_urt_avg_selection(x_spt, avg_urt_params, normalize=True)
        urt_x_qry  = apply_urt_avg_selection(x_qry, avg_urt_params, normalize=True) 
        # mqda
        if args.x_norm == "L2N":
            urt_x_spt = F.normalize(urt_x_spt, dim=-1)
            urt_x_qry  = F.normalize(urt_x_qry, dim=-1)
        classifier.fit_image_label(urt_x_spt, y_spt, domain_id.item()) # compute \mu and \Sigma on support set
        logits = classifier.predict(urt_x_qry)
        ce_loss = F.cross_entropy(logits, y_qry)
        "===>MQDA"
        chlsk_loss = args.cholesky_alpha * losses.cholesky_loss(classifier.lower_triu)
        loss = ce_loss + chlsk_loss
        losses.backward_propagation(loss, classifier.parameters(), optimizer)
        lr_scheduler.step()

        final_acc = torch.eq(y_qry, torch.argmax(logits, dim=-1)).float().mean().item()
        our_losses.update(loss.item())
        our_accuracies.update(final_acc * 100)
        
        if episode_index % args.n_episode_train == 0 or episode_index+1 == args.n_epoch * args.n_episode_train:
            logger.print("[{:5d}/{:5d}] lr: {:}, loss: {:.5f}, accuracy: {:.4f}, still need {:} {:}".format(episode_index, args.n_epoch * args.n_episode_train, lr_scheduler.get_last_lr(), our_losses.avg, our_accuracies.avg, timer.step(), util.time_now()))
            save.urt_mqda(args, episode_index, optimizer.state_dict(), lr_scheduler.state_dict(), URT_model.state_dict(), classifier.state_dict(), args.save_dir, 'train')
            our_losses.reset()
            our_accuracies.reset()
        else:
            timer.step()
            
def val_mqda(args, val_loaders, URT_model, classifier, logger, start_iter, optimizer, lr_scheduler):
    print('begin val')
    URT_model.train()
    if start_iter == args.n_epoch * args.n_episode_train:
        return args.n_episode - 1
    our_losses, our_accuracies = metric.AverageMeter(), metric.AverageMeter()
    timer = metric.CountdownTimer(total_steps=args.n_epoch * args.n_episode_train )
    for episode_index in range(600):
        for name_dataset, loader in val_loaders.items():
            dataset_id_tensor, x_spt, y_spt, x_qry, y_qry = loader.next()
            if episode_index < start_iter:
                continue
            x_spt, y_spt = x_spt.squeeze(0).to(DEVICE), y_spt.squeeze(0).to(DEVICE)
            x_qry, y_qry = x_qry.squeeze(0).to(DEVICE), y_qry.squeeze(0).to(DEVICE)

            n_classes = len(np.unique(y_spt.cpu().numpy()))
            # optimize selection parameters and perform feature selection
            avg_urt_params = get_lambda_urt_avg(x_spt, y_spt, n_classes, URT_model, normalize=True)
            # identity matrix panelize to be sparse, only focus on one aspect
            penalty = torch.pow( torch.norm( torch.transpose(avg_urt_params, 0, 1) @ avg_urt_params - torch.eye(args.urt_head).cuda() ), 2) # scalar:-> torch.mm(a_u_p.t(), a_u_p)
            # [num_sample, encoders, dim=512]=>weight attention => n_samples * (n_head * 512)
            urt_x_spt = apply_urt_avg_selection(x_spt, avg_urt_params, normalize=True)
            urt_x_qry  = apply_urt_avg_selection(x_qry, avg_urt_params, normalize=True) 
            # mqda
            if args.x_norm == "L2N":
                urt_x_spt = F.normalize(urt_x_spt, dim=-1)
                urt_x_qry = F.normalize(urt_x_qry, dim=-1)
            if args.strategy == 'map_select_prior':
                classifier.fit_image_label(urt_x_spt, y_spt, dataset_id_tensor.item()) # compute \mu and \Sigma on support set
#                 classifier.fit_image_label(urt_x_spt, y_spt) # compute \mu and \Sigma on support set
            if args.strategy == 'map_weighted_prior':
                classifier.fit_val_image_label(urt_x_spt, y_spt) # compute \mu and \Sigma on support set
            if args.strategy == 'map_select_prior':
                logits = classifier.predict(urt_x_qry)
            if args.strategy == 'map_weighted_prior':
                logits = classifier.predict_val(urt_x_qry)
            ce_loss = F.cross_entropy(logits, y_qry)
            "===>MQDA"
            # Due to the function design, the cholesky loss works on train part, but do not work on val part
#             chlsk_loss = args.cholesky_alpha * losses.cholesky_loss(classifier.lower_triu)
            loss = ce_loss # + chlsk_loss
            losses.backward_propagation(loss, classifier.parameters(), optimizer)
            lr_scheduler.step()

            final_acc = torch.eq(y_qry, torch.argmax(logits, dim=-1)).float().mean().item()
            our_losses.update(loss.item())
            our_accuracies.update(final_acc * 100)

            if episode_index % (args.n_episode_train/10.) == 0 or episode_index+1 == 600:
                logger.print("[val {:5d}/{:5d}] lr: {:}, loss: {:.5f}, accuracy: {:.4f}, still need {:} {:}".format(episode_index, args.n_epoch * args.n_episode_train, lr_scheduler.get_last_lr(), our_losses.avg, our_accuracies.avg, timer.step(), util.time_now()))
                save.urt_mqda(args, episode_index, optimizer.state_dict(), lr_scheduler.state_dict(), URT_model.state_dict(), classifier.state_dict(), args.save_dir, 'val')
                our_losses.reset()
                our_accuracies.reset()

def test_all_dataset_mqda(args, test_loaders, URT_model, classifier, logger, mode):
    print('begin test')
    URT_model.eval() 
    our_name = 'mqda'
    accs_names = [our_name]
    alg2data2accuracy = collections.OrderedDict()
    alg2data2accuracy['sur'], alg2data2accuracy['urt'], _ = pre_load_results()
    alg2data2accuracy[our_name] = {name: [] for name in test_loaders.keys()}
    
    logger.print('\n{:} starting evaluate the {:} set at the {:}-th iteration.'.format(util.time_now(), mode, args.index_episode))
    with torch.no_grad():
        for dataset_id, (test_dataset, loader) in enumerate(test_loaders.items()):
            if dataset_id >=8:
                break
            logger.print('===>>> {:} --->>> {:02d}/{:02d} --->>> {:}'.format(util.time_now(), dataset_id, len(test_loaders), test_dataset))
            our_losses = metric.AverageMeter()
            for episode_index, (dataset_id_tensor, x_spt, y_spt, x_qry, y_qry) in enumerate(loader):
                x_spt, y_spt = x_spt.squeeze(0).to(DEVICE), y_spt.squeeze(0).to(DEVICE)
                x_qry, y_qry = x_qry.squeeze(0).to(DEVICE), y_qry.squeeze(0).to(DEVICE)

                n_classes = len(np.unique(y_spt.cpu().numpy()))
                # optimize selection parameters and perform feature selection
                avg_urt_params = get_lambda_urt_avg(x_spt, y_spt, n_classes, URT_model, normalize=True)
                # identity matrix panelize to be sparse, only focus on one aspect
                penalty = torch.pow( torch.norm( torch.transpose(avg_urt_params, 0, 1) @ avg_urt_params - torch.eye(args.urt_head).cuda() ), 2) # scalar:-> torch.mm(a_u_p.t(), a_u_p)
                # [num_sample, encoders, dim=512]=>weight attention => n_samples * (n_head * 512)
                urt_x_spt = apply_urt_avg_selection(x_spt, avg_urt_params, normalize=True)
                urt_x_qry  = apply_urt_avg_selection(x_qry, avg_urt_params, normalize=True) 
                # mqda
                if args.x_norm == "L2N":
                    urt_x_spt = F.normalize(urt_x_spt, dim=-1)
                    urt_x_qry  = F.normalize(urt_x_qry, dim=-1)
                
                if args.strategy == 'map_select_prior':
                    classifier.fit_image_label(urt_x_spt, y_spt, dataset_id_tensor.item()) # compute \mu and \Sigma on support set
#                     classifier.fit_image_label(urt_x_spt, y_spt) # compute \mu and \Sigma on support set
                if args.strategy == 'map_weighted_prior':
                    classifier.fit_val_image_label(urt_x_spt, y_spt) # compute \mu and \Sigma on support set
                if args.strategy == 'map_select_prior':
                    logits = classifier.predict(urt_x_qry)
                if args.strategy == 'map_weighted_prior':
                    logits = classifier.predict_val(urt_x_qry)
                    
                final_acc = torch.eq(y_qry, torch.argmax(logits, dim=-1)).float().mean().item()
                alg2data2accuracy[our_name][test_dataset].append(final_acc)
    
            base_name = '{:}-{:}'.format(test_dataset, mode)

    dataset_names = list(test_loaders.keys())
    torch.save(alg2data2accuracy, '{:}/perform-{:}.tar'.format(args.save_dir, args.seed, mode))
    show_results(dataset_names, alg2data2accuracy, ('sur', 'urt', our_name), logger.print)
    logger.print("\n")

                
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
    parser.add_argument('--optimizer', default='adam',type=str, help='optimization method (default: momentum)')
    parser.add_argument('-lr','--lr', default=3e-4, type=float, help='learning rate (default: 0.0001)')
    
    parser.add_argument('--x_dim', default=1024, type=int, help='dimensional of the features for the input of mqda')
    parser.add_argument('--x_norm', default='L2N', type=str, help='')
    
    # urt related
    parser.add_argument('--urt_head', default=2, type=int, help='')
#     parser.add_argument('--Nj', default=1, type=int, help='')
    
    # mqda related
    parser.add_argument('--cholesky_alpha',default=0, type=float,help="hope cholesky_alpha > feature dim + 1 ")
    parser.add_argument('-strategy','--strategy', default='map_select_prior', type=str, help='map_select_prior, map_weighted_prior')
    parser.add_argument('--reg_param', default=0.95, type=float, help='')
    parser.add_argument('--Nj', default=1, type=int, help='')
    
    return parser.parse_args()

if __name__ == '__main__':
    main()
