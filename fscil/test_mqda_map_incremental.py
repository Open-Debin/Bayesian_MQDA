import os
import sys
from pathlib import Path
mqda_lib_dir = Path(__file__).parent.resolve().parent
sys.path.insert(0, str(mqda_lib_dir))
import pdb
import time
import random
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from code_lib import generate, util, load, save, metric
from code_models import config, losses, classifiers, layers
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# CUDA_VISIBLE_DEVICES=2 python efficient_test_mqda_incremental_session.py -l_n mini_resnet18_incremental49_baseSize30_sgd_map_0.01_5_features
# torch.cuda.set_device(3)
def main():
    util.set_random(4603)
    xargs = config_local_parameters()
    path2features = '../../data_src/fea_mqda/mini-resnet18-incremental-fea-49.pkl'
    feature_or_logits = 1 if 'logits' in xargs.log_name else 0
    x_np, x_centre_np = load.numpy_features(xargs, path2features, feature_or_logits)  
    log_dir = '../log/incremental/{:}'.format(xargs.log_name)
    ckps = util.search('*pth', target_space = log_dir)
    logger = util.Logger(log_dir, 'mini')
    timer = metric.CountdownTimer(total_steps=len(ckps))
    for index_epoch, checkpoint_item in enumerate(ckps):
        learner, args, _,_ = load.checkpoint(checkpoint_item)
        args.n_episode_test = xargs.n_episode_test
        acc_base, acc = manyshot_fewclass_incremental_training(x_np, learner, args, xargs, timer, logger)  
    logger.rename('mqda')
    
def manyshot_fewclass_incremental_training(x_np, learner, args, xargs, timer, logger):
    folders_base_train, folders_base_test = generate.folders(args.base_path), generate.folders(args.base_path.replace('train','test'))
    mapping = util.NameIdMapping(args.base_path)
    xargs.n_base = len(folders_base_train)
    learner.train()
    task_base_train = generate.Task(folders_base_train, xargs.n_base, 500, 0)
    ms_names_spt, _ = generate.name_label(task_base_train, split="support").__iter__().next()
    base_x_spt = load.name2tensor_feature(ms_names_spt, args.path2image, x_np)
    base_y_spt = torch.tensor(mapping.name2id(ms_names_spt)).to(DEVICE)
    
    task_base_test = generate.Task(folders_base_test, xargs.n_base, 0, xargs.k_qry)
    base_names_qry, _ = generate.name_label(task_base_test, split="query").__iter__().next()
    base_x_qry = load.name2tensor_feature(base_names_qry, args.path2image, x_np)
    base_y_qry = torch.tensor(mapping.name2id(base_names_qry)).to(DEVICE)
    
    ms_norm_x_spt = layers.centre_l2norm(base_x_spt, args.x_centre)
    base_x_qry = layers.centre_l2norm(base_x_qry, args.x_centre) 
    
    learner.fit_image_label(ms_norm_x_spt, base_y_spt)
    outputs_ms = learner.predict(base_x_qry)
    acc_base = metric.accuracy(outputs_ms.argmax(dim=1), base_y_qry)
    ce_loss_ms = F.cross_entropy(outputs_ms, base_y_qry) 

    acc_s1, acc_s2, acc_s3, acc_s4, acc_s5, acc_s6, acc_s7, acc_s8 = metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter() , metric.AverageMeter()

    for index_episode in range(args.n_episode_test):
        util.set_random(index_episode*index_episode*2)
        acc_s = incremental_process(base_x_qry, base_y_qry, x_np, learner, args, xargs)
        logger.print(args.title_in_screen.format(args.index_epoch+1, args.n_epoch, index_episode+1, args.n_episode_test, -10, timer.step(), util.time_now()))
        
        acc_s1.update(acc_s[0])
        acc_s2.update(acc_s[1]) 
        acc_s3.update(acc_s[2]) 
        acc_s4.update(acc_s[3]) 
        acc_s5.update(acc_s[4]) 
        acc_s6.update(acc_s[5]) 
        acc_s7.update(acc_s[6])   
        acc_s8.update(acc_s[7]) 
        logger.print(xargs.acc_in_screen.format(acc_base, acc_s[0], acc_s[1], acc_s[2], acc_s[3], acc_s[4], acc_s[5], acc_s[6], acc_s[7]))
    acc1, h1 = acc_s1.mean_confidence_interval()
    acc2, h2 = acc_s2.mean_confidence_interval()
    acc3, h3 = acc_s3.mean_confidence_interval()
    acc4, h4 = acc_s4.mean_confidence_interval()
    acc5, h5 = acc_s5.mean_confidence_interval()
    acc6, h6 = acc_s6.mean_confidence_interval()
    acc7, h7 = acc_s7.mean_confidence_interval()
    acc8, h8 = acc_s8.mean_confidence_interval()
    logger.print(xargs.acc_h_in_screen.format(args.index_epoch + 1, args.n_epoch, acc_base,acc1, h1, acc2, h2, acc3, h3, acc4, h4, acc5, h5, acc6, h6, acc7, h7, acc8, h8))
    return acc_base, acc_s

def incremental_process(base_x_qry, base_y_qry, x_np, learner, args, xargs):
    
    temp_many_few_mu = learner.mu[:xargs.n_base]
    temp_many_few_sigma = learner.sigma_inv[:xargs.n_base]
    candidate_session = list(range(2, xargs.n_session_test + 2))
    random.shuffle(candidate_session)
    for index_session in range(2, xargs.n_session_test + 2):
        torch.cuda.empty_cache()
        session_path_train = args.base_path.replace('session1','session'+str(candidate_session[index_session-2]))
        folder_session_train, folder_session_test = generate.folders(session_path_train), generate.folders(session_path_train.replace('train', 'test'))
        mapping = util.NameIdMapping(session_path_train)
        learner.train()

        for i in range(random.randint(1,10)):
            task_base_train = generate.Task(folder_session_train, args.n_way, args.k_spt, 0)
            names_spt, _ = generate.name_label(task_base_train, split="support").__iter__().next()
        x_spt = load.name2tensor_feature(names_spt, args.path2image, x_np)
        novel_y_spt = torch.tensor(mapping.name2id(names_spt)).to(DEVICE)
        
        task_base_test = generate.Task(folder_session_test, args.n_way, 0, xargs.k_qry)
        names_qry, _ = generate.name_label(task_base_test, split="query").__iter__().next()
        x_qry = load.name2tensor_feature(names_qry, args.path2image, x_np)
        novel_y_qry = torch.tensor(mapping.name2id(names_qry)).to(DEVICE)

        novel_x_spt = layers.centre_l2norm(x_spt, args.x_centre)
        novel_x_qry = layers.centre_l2norm(x_qry, args.x_centre) 
        if index_session ==2:
            joint_x_qry = torch.cat([base_x_qry, novel_x_qry])
            joint_y_qry = torch.cat([base_y_qry, novel_y_qry+xargs.n_base+args.n_way*(index_session-2)]) 
        else:
            joint_x_qry = torch.cat([joint_x_qry, novel_x_qry])
            joint_y_qry = torch.cat([joint_y_qry, novel_y_qry+xargs.n_base+args.n_way*(index_session-2)])
            
        learner.fit_image_label(novel_x_spt, novel_y_spt)
        temp_many_few_mu.extend(learner.mu)
        temp_many_few_sigma.extend(learner.sigma_inv)

        learner.mu = temp_many_few_mu 
        learner.sigma_inv = temp_many_few_sigma
        
    acc_session = [None for i in range(xargs.n_session_test)]
    outputs_joint = learner.predict(joint_x_qry)

    for index_session in range(xargs.n_session_test): 
        torch.cuda.empty_cache()
        logits = outputs_joint[:10000-args.k_spt*xargs.k_qry*index_session,:100-args.n_way*index_session]
        labels = joint_y_qry[:10000-args.k_spt*xargs.k_qry*index_session]
        acc_session[xargs.n_session_test-1-index_session] = metric.accuracy(logits.argmax(dim=1), labels )
    
    return acc_session

def config_local_parameters():
    parser = argparse.ArgumentParser(description= 'GFSL Map testing script' )
    parser.add_argument('--k_qry', default=100, type=int,  help='number of data in each class of query set')
    parser.add_argument('--n_base', default=20, type=int, help='number of base class')
    parser.add_argument('-l_n', '--log_name', default='mini_conv4_sgd_Fix1_map_0.001_5_logits64',  help='path to log') 
    parser.add_argument('-n_l', '--net_label', default='simpleshot', help='the encoder')
    parser.add_argument('-t_e', '--n_episode_test', default=10, type=int,  help='')
    parser.add_argument('--n_session_test', default=8, type=int, help='number of incremental step')
    args = parser.parse_args()
    args.acc_in_screen = 'acc_base:{:}, acc1:{:}; acc2:{:}; acc3:{:}; acc4:{:}; acc5:{:}; acc6:{:}; acc7:{:}; acc8:{:}'
    args.acc_h_in_screen = 'epoch [{:3d}/{:3d}], acc_base:{:}; acc1:{:} h1:{:}; acc2:{:} h2:{:}; acc3:{:} h3:{:}; acc4:{:} h4:{:}; acc5:{:} h5:{:}; acc6:{:} h6:{:}; acc7:{:} h7:{:}; acc8:{:} h8:{:}'
    return args

if __name__=='__main__':
    main()
