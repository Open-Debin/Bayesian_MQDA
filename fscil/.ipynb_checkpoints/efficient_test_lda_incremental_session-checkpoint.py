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
# 


# torch.cuda.set_device(3)
def main():
    util.set_random(4603)
    xargs = config_local_parameters()
    domain_net, net_arch, x_domain = xargs.path2log.split('_')[0], xargs.path2log.split('_')[1], xargs.domain_feature
#     path2features = '../../data_src/fea_mqda/mini-conv4-incremental-fea-179.pkl'
    path2features = '../../data_src/fea_mqda/mini-resnet18-incremental-fea-49.pkl'
    feature_or_logits = [1 if 'logits' in xargs.path2log else 0][0]
    x_np, x_centre_np = load.numpy_features(xargs, path2features, feature_or_logits)  
    log_dir = '../active_log/incremental/{:}'.format(xargs.path2log)
    ckps = util.search('*pth', target_space = log_dir)
    logger = util.Logger(log_dir, x_domain)
    best_acc = metric.BestAcc()
    timer = metric.CountdownTimer(total_steps=len(ckps))
#     mu, sigma_inv = compute_manyshot_mu_sigma_mle(xargs, x_np, folders_base, mapping)
    for index_epoch, checkpoint_item in enumerate(ckps):
#         if index_epoch <=2 :
#             print('skip',index_epoch)
#             continue
        classifier, args, _,_ = load.checkpoint(checkpoint_item)
#         pdb.set_trace()
        args.n_episode_test = xargs.n_episode_test
        acc, h_ = manyshot_fewclass_incremental_training(x_np, classifier, args, xargs, timer, logger)
        best_acc.update(acc, h_, args.index_epoch)    
        logger.print(xargs.string_in_screen.format(args.index_epoch + 1, args.n_epoch, args.n_episode_test, timer.step(), util.time_now()))
    logger.rename('epoch_{:}_acc_{:}_h_{:}'.format(best_acc.index, best_acc.value, best_acc.h_))
 
    
def manyshot_fewclass_incremental_training(x_np, classifier, args, xargs, timer, logger):
    folders_base_train, folders_base_test = generate.folders(args.base_path), generate.folders(args.base_path.replace('train','test'))
    mapping = util.NameIdMapping(args.base_path)
    xargs.n_base = len(folders_base_train)
    classifier.train()
    task_base_train = generate.Task(folders_base_train, xargs.n_base, 200, 1)
    ms_names_spt, _ = generate.name_label(task_base_train, split="support").__iter__().next()
    ms_x_spt = load.name2tensor_feature(ms_names_spt, args.path2image, x_np)
    ms_y_spt = torch.tensor(mapping.name2id(ms_names_spt)).to(DEVICE)
    
    task_base_test = generate.Task(folders_base_test, xargs.n_base, 0, xargs.k_qry)
    ms_names_qry, _ = generate.name_label(task_base_test, split="query").__iter__().next()
    ms_x_qry = load.name2tensor_feature(ms_names_qry, args.path2image, x_np)
    ms_y_qry = torch.tensor(mapping.name2id(ms_names_qry)).to(DEVICE)
    
    ms_norm_x_spt = layers.centre_l2norm(ms_x_spt, args.x_centre)
    ms_norm_x_qry = layers.centre_l2norm(ms_x_qry, args.x_centre) 
    
    classifier.fit_image_label(ms_norm_x_spt, ms_y_spt)
    outputs_ms = classifier.predict(ms_norm_x_qry)
    acc_base = metric.accuracy(outputs_ms.argmax(dim=1), ms_y_qry)
    ce_loss_ms = F.cross_entropy(outputs_ms, ms_y_qry) 
    
    acc_s1, acc_s2, acc_s3, acc_s4, acc_s5, acc_s6, acc_s7, acc_s8 = metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter() , metric.AverageMeter()
    
    for index_episode in range(args.n_episode_test):
        acc_s, loss = incremental_process(ms_norm_x_qry, ms_y_qry, x_np, classifier, args, xargs)
        logger.print(args.title_in_screen.format(args.index_epoch+1, args.n_epoch, index_episode+1, args.n_episode_test, loss, timer.step(), util.time_now()))
        acc_s1.update(acc_s[0])
        acc_s2.update(acc_s[1]) 
        acc_s3.update(acc_s[2]) 
        acc_s4.update(acc_s[3]) 
        acc_s5.update(acc_s[4]) 
        acc_s6.update(acc_s[5]) 
        acc_s7.update(acc_s[6])   
        acc_s8.update(acc_s[7]) 
#         metric.mean_confidence_interval(acc_s[0])
        logger.print(xargs.acc_in_screen.format(acc_base, acc_s[0], acc_s[1], acc_s[2], acc_s[3], acc_s[4], acc_s[5], acc_s[6], acc_s[7]))
    acc1, h1 = acc_s1.mean_confidence_interval()
    acc2, h2 = acc_s2.mean_confidence_interval()
    acc3, h3 = acc_s3.mean_confidence_interval()
    acc4, h4 = acc_s4.mean_confidence_interval()
    acc5, h5 = acc_s5.mean_confidence_interval()
    acc6, h6 = acc_s6.mean_confidence_interval()
    acc7, h7 = acc_s7.mean_confidence_interval()
    acc8, h8 = acc_s8.mean_confidence_interval()
    logger.print(xargs.acc_h_in_screen.format(args.index_epoch + 1, args.n_epoch, acc_base, acc1, h1, acc2, h2, acc3, h3, acc4, h4, acc5, h5, acc6, h6, acc7, h7, acc8, h8))
    return (acc1+acc2+acc3+acc4+acc5+acc6+acc7+acc8)/7, (h1+h2+h3+h4+h5+h6+h7+h8)/7   

def incremental_process(ms_norm_x_qry, ms_y_qry, np_features, classifier, args, xargs):
    
    temp_many_few_mu = classifier.mu[:xargs.n_base]
    temp_many_few_sigma = classifier.sigma_inv[:xargs.n_base]
#     temp_many_few_biases = classifier.biases[:xargs.n_base]
#     temp_many_few_common_part = classifier.common_part[:xargs.n_base]
    
    for index_session in range(2, xargs.n_session_test + 2):
        session_path_train = args.base_path.replace('session1','session'+str(index_session))
        folder_session_train, folder_session_test = generate.folders(session_path_train), generate.folders(session_path_train.replace('train', 'test'))
        mapping = util.NameIdMapping(session_path_train)
        classifier.train()
        task_base_train = generate.Task(folder_session_train, args.n_way, args.k_spt, 0)
        names_spt, _ = generate.name_label(task_base_train, split="support").__iter__().next()
        x_spt = load.name2tensor_feature(names_spt, args.path2image, np_features)
        fs_y_spt = torch.tensor(mapping.name2id(names_spt)).to(DEVICE)

        task_base_test = generate.Task(folder_session_test, args.n_way, 0, xargs.k_qry)
        names_qry, _ = generate.name_label(task_base_test, split="query").__iter__().next()
        x_qry = load.name2tensor_feature(names_qry, args.path2image, np_features)
        fs_y_qry = torch.tensor(mapping.name2id(names_qry)).to(DEVICE)

        fs_norm_x_spt = layers.centre_l2norm(x_spt, args.x_centre)
        fs_norm_x_qry = layers.centre_l2norm(x_qry, args.x_centre) 
        if index_session ==2:
#             joint_x_spt, joint_y_spt = joint_many_few_x_y(ms_norm_x_spt, ms_y_spt, fs_norm_x_spt, fs_y_spt+xargs.n_base+args.n_way*index_session)
            joint_x_qry, joint_y_qry = joint_many_few_x_y(ms_norm_x_qry, ms_y_qry, fs_norm_x_qry, fs_y_qry+xargs.n_base+args.n_way*(index_session-2))
        else:
#             joint_x_spt, joint_y_spt = joint_many_few_x_y(joint_x_spt, joint_y_spt, fs_norm_x_spt, fs_y_spt+args.n_base+args.n_way*index_session)
            joint_x_qry, joint_y_qry = joint_many_few_x_y(joint_x_qry, joint_y_qry, fs_norm_x_qry, fs_y_qry+xargs.n_base+args.n_way*(index_session-2))
            
        classifier.fit_image_label(fs_norm_x_spt, fs_y_spt)
        temp_many_few_mu.extend(classifier.mu)
        temp_many_few_sigma.extend(classifier.sigma_inv)
#         temp_many_few_biases.extend(classifier.biases)
#         temp_many_few_common_part.extend(classifier.common_part)
        classifier.mu = temp_many_few_mu 
        classifier.sigma_inv = temp_many_few_sigma
#         classifier.biases = temp_many_few_biases
#         classifier.common_part = temp_many_few_common_part
        
    acc_session = [None for i in range(xargs.n_session_test)]
    outputs_joint = classifier.predict(joint_x_qry)
#     pdb.set_trace()

    for index_session in range(xargs.n_session_test): 
        logits = outputs_joint[:10000-args.k_spt*xargs.k_qry*index_session,:100-args.n_way*index_session]
        labels = joint_y_qry[:10000-args.k_spt*xargs.k_qry*index_session]
        acc_session[xargs.n_session_test-1-index_session] = metric.accuracy(logits.argmax(dim=1), labels )
        ce_loss = F.cross_entropy(logits, labels)
    
    return acc_session, ce_loss

def joint_many_few_x_y(ms_x, ms_y, fs_x, fs_y):
    joint_x = torch.cat([ms_x, fs_x])
    joint_y = torch.cat([ms_y, fs_y])
    
    return joint_x, joint_y

def config_local_parameters():
    parser = argparse.ArgumentParser(description= 'GFSL Map testing script' )
    parser.add_argument('--k_qry', default=100, type=int,  help='number of data in each class of query set')
    parser.add_argument('--n_base', default=20, type=int, help='number of base class')
    parser.add_argument('-p2l', '--path2log', default='mini_conv4_sgd_Fix1_map_0.001_5_logits64',  help='path to log') 
    parser.add_argument('-n_l', '--net_label', default='simpleshot', help='the encoder')
    parser.add_argument('-d_f', '--domain_feature', default='mini',  help='domain of the feature') 
    parser.add_argument('-t_e', '--n_episode_test', default=1, type=int,  help='')
    parser.add_argument('--n_session_test', default=8, type=int, help='number of incremental step')
    parser.add_argument('-manyshot_strategy', '--manyshot_strategy', default='mqda', type=str,  help='mle, mqda')
#     parser.add_argument('--path2image', default='../../data_src/images/', help='path to all datasets: CUB, Mini, etc')
    args = parser.parse_args()
    args.string_in_screen = "epoch [{:3d}/{:3d}], test episode:{:}, needed [{:}], [{:}]" 
    args.acc_in_screen = 'acc_base:{:}, acc1:{:}; acc2:{:}; acc3:{:}; acc4:{:}; acc5:{:}; acc6:{:}; acc7:{:}; acc8:{:}'
    args.acc_h_in_screen = 'epoch [{:3d}/{:3d}], acc_base:{:}, acc1:{:} h1:{:}; acc2:{:} h2:{:}; acc3:{:} h3:{:}; acc4:{:} h4:{:}; acc5:{:} h5:{:}; acc6:{:} h6:{:}; acc7:{:} h7:{:}'
    return args

if __name__=='__main__':
    main()
