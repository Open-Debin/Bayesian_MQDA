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
# CUDA_VISIBLE_DEVICES=3 python test_temp.py -p2l mini_conv4_incremental_sgd_map_0.01_5_logits -d_f mini
# torch.cuda.set_device(3)
def main():
    util.set_random(4603)
    xargs = config_local_parameters()
    domain_net, net_arch, x_domain = xargs.path2log.split('_')[0], xargs.path2log.split('_')[1], xargs.domain_feature
    path2features = '../../data_src/fea_mqda/{:}-{:}-{:}-fea.pkl'.format(domain_net, net_arch, x_domain)
    path2aux_features = '../../data_src/fea_mqda/{:}-{:}-{:}-aux-fea.pkl'.format(domain_net, net_arch, x_domain)
    feature_or_logits = [1 if 'logits' in xargs.path2log else 0][0]
    x_np, x_centre_np = load.numpy_features(xargs, path2features, feature_or_logits)  
    x_np_aux, _ = load.numpy_features(xargs, path2aux_features, feature_or_logits)

    seenclass_basedata_path='{:}{:}/train'.format(xargs.path2image, domain_net)
    seenclass_auxdata_path='{:}{:}/train_aux'.format(xargs.path2image, domain_net)
    val_class ='{:}{:}/val'.format(xargs.path2image, x_domain)
    test_class ='{:}{:}/test'.format(xargs.path2image, x_domain)
    mapping = util.NameIdMapping(seenclass_basedata_path)
    folders_base, folders_aux, folders_novel_val_test = generate.folders(seenclass_basedata_path), generate.folders(seenclass_auxdata_path), generate.folders(val_class, test_class)
    xargs.n_base = len(folders_base)
    log_dir = '../active_log/{:}'.format(xargs.path2log)
    ckps = util.search('*pth', target_space = log_dir)
    logger = util.Logger(log_dir, x_domain+'_'+xargs.manyshot_strategy)
    best_acc = metric.BestAcc()
    timer = metric.CountdownTimer(total_steps=len(ckps))
#     mu, sigma_inv = compute_manyshot_mu_sigma_mle(xargs, x_np, folders_base, mapping)
    for index_epoch, checkpoint_item in enumerate(ckps):
        classifier, args, _,_ = load.checkpoint(checkpoint_item)
        args.n_episode_test = xargs.n_episode_test
        args.path2image = xargs.path2image
        if xargs.manyshot_strategy == 'mqda':
            print('manyshot mu_sigma compute by mqda')
        acc, h_ = manyshot_fewclass_incremental_training(folders_base, folders_aux, folders_novel_val_test, x_np, x_np_aux, classifier, mapping, args, xargs, timer, logger)
        best_acc.update(acc, h_, args.index_epoch)    
        logger.print(xargs.string_in_screen.format(args.index_epoch + 1, args.n_epoch, args.n_episode_test, timer.step(), util.time_now()))
    logger.rename('epoch_{:}_acc_{:}_h_{:}'.format(best_acc.index, best_acc.value, best_acc.h_))
 
    
def manyshot_fewclass_incremental_training(folders_base, folders_aux, folders_novel_val_test, x_np, x_np_aux, classifier, mapping, args, xargs, timer, logger):
    classifier.train()
    task_base_train = generate.Task(folders_base, 64, 200, args.k_qry)
    ms_names_spt, _ = generate.name_label(task_base_train, split="support").__iter__().next()
    ms_x_spt = load.name2tensor_feature(ms_names_spt, args.path2image, x_np)
    ms_y_spt = torch.tensor(mapping.name2id(ms_names_spt)).to(DEVICE)
    
    task_base_test = generate.Task(folders_aux, 64, 200, args.k_qry)
    ms_names_qry, _ = generate.name_label(task_base_test, split="query").__iter__().next()
    ms_x_qry = load.name2tensor_feature(ms_names_qry, args.path2image, x_np_aux)
    ms_y_qry = torch.tensor(mapping.name2id(ms_names_qry)).to(DEVICE)
    
    ms_norm_x_spt = layers.centre_l2norm(ms_x_spt, args.x_centre)
    ms_norm_x_qry = layers.centre_l2norm(ms_x_qry, args.x_centre)
    
    
    classifier.fit_image_label(ms_norm_x_spt, ms_y_spt)
    outputs_ms = classifier.predict(ms_norm_x_qry)
    acc_base = metric.accuracy(outputs_ms.argmax(dim=1), ms_y_qry)
    ce_loss_ms = F.cross_entropy(outputs_ms, ms_y_qry) 
    
    acc_s1, acc_s2, acc_s3, acc_s4, acc_s5, acc_s6, acc_s7 = metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter(), metric.AverageMeter()
    
    for index_episode in range(args.n_episode_train):
        acc_s, loss = incremental_process(ms_norm_x_qry, ms_y_qry, folders_novel_val_test, x_np, classifier, args, xargs)
        logger.print(args.title_in_screen.format(args.index_epoch+1, args.n_epoch, index_episode+1, args.n_episode_train, loss, timer.step(), util.time_now()))
        acc_s1.update(acc_s[0])
        acc_s2.update(acc_s[1]) 
        acc_s3.update(acc_s[2]) 
        acc_s4.update(acc_s[3]) 
        acc_s5.update(acc_s[4]) 
        acc_s6.update(acc_s[5]) 
        acc_s7.update(acc_s[6])      
#         metric.mean_confidence_interval(acc_s[0])
        logger.print(xargs.acc_in_screen.format(acc_base, acc_s[0], acc_s[1], acc_s[2], acc_s[3], acc_s[4], acc_s[5], acc_s[6]))
    acc1, h1 = acc_s1.mean_confidence_interval()
    acc2, h2 = acc_s2.mean_confidence_interval()
    acc3, h3 = acc_s3.mean_confidence_interval()
    acc4, h4 = acc_s4.mean_confidence_interval()
    acc5, h5 = acc_s5.mean_confidence_interval()
    acc6, h6 = acc_s6.mean_confidence_interval()
    acc7, h7 = acc_s7.mean_confidence_interval()
    logger.print(xargs.acc_h_in_screen.format(args.index_epoch + 1, args.n_epoch, acc_base, acc1, h1, acc2, h2, acc3, h3, acc4, h4, acc5, h5, acc6, h6, acc7, h7))
    return (acc1+acc2+acc3+acc4+acc5+acc6+acc7)/7, (h1+h2+h3+h4+h5+h6+h7)/7   

def incremental_process(ms_norm_x_qry, ms_y_qry, folders_fewshot, np_features, classifier, args, xargs):
    acc_session = [None for i in range(args.n_session)]
    temp_many_few_mu = classifier.mu[:xargs.n_base]
    temp_many_few_sigma = classifier.sigma_inv[:xargs.n_base]
    for index_session in range(args.n_session):
        folder_session = folders_fewshot[index_session * args.n_way: (index_session + 1) * args.n_way]
        task_few_session = generate.Task(folder_session, args.n_way, args.k_spt, args.k_qry)
        (fs_x_spt, fs_y_spt), (fs_x_qry, fs_y_qry) = load.tensor_features_from(task_few_session, np_features, args)
        fs_norm_x_spt = layers.centre_l2norm(fs_x_spt, args.x_centre)
        fs_norm_x_qry = layers.centre_l2norm(fs_x_qry, args.x_centre)
        
        if index_session ==0:
#             joint_x_spt, joint_y_spt = joint_many_few_x_y(ms_norm_x_spt, ms_y_spt, fs_norm_x_spt, fs_y_spt+xargs.n_base+args.n_way*index_session)
            joint_x_qry, joint_y_qry = joint_many_few_x_y(ms_norm_x_qry, ms_y_qry, fs_norm_x_qry, fs_y_qry+xargs.n_base+args.n_way*index_session)
        else:
#             joint_x_spt, joint_y_spt = joint_many_few_x_y(joint_x_spt, joint_y_spt, fs_norm_x_spt, fs_y_spt+args.n_base+args.n_way*index_session)
            joint_x_qry, joint_y_qry = joint_many_few_x_y(joint_x_qry, joint_y_qry, fs_norm_x_qry, fs_y_qry+xargs.n_base+args.n_way*index_session)
            
        classifier.fit_image_label(fs_norm_x_spt, fs_y_spt)
        temp_many_few_mu.extend(classifier.mu)
        temp_many_few_sigma.extend(classifier.sigma_inv)
        classifier.mu = temp_many_few_mu 
        classifier.sigma_inv = temp_many_few_sigma

        outputs_joint = classifier.predict(joint_x_qry)
        acc_session[index_session] = metric.accuracy(outputs_joint.argmax(dim=1), joint_y_qry)
        ce_loss = F.cross_entropy(outputs_joint, joint_y_qry)
    
    return acc_session, ce_loss

def joint_many_few_x_y(ms_x, ms_y, fs_x, fs_y):
    joint_x = torch.cat([ms_x, fs_x])
    joint_y = torch.cat([ms_y, fs_y])
    
    return joint_x, joint_y

def config_local_parameters():
    parser = argparse.ArgumentParser(description= 'GFSL Map testing script' )
    parser.add_argument('-p2l', '--path2log', default='mini_conv4_sgd_Fix1_map_0.001_5_logits64',  help='path to log') 
    parser.add_argument('-n_l', '--net_label', default='simpleshot', help='the encoder')
    parser.add_argument('-d_f', '--domain_feature', default='mini',  help='domain of the feature') 
    parser.add_argument('-t_e', '--n_episode_test', default=10, type=int,  help='')
    parser.add_argument('-manyshot_strategy', '--manyshot_strategy', default='mqda', type=str,  help='mle, mqda')
    parser.add_argument('--path2image', default='../../data_src/images/', help='path to all datasets: CUB, Mini, etc')
    args = parser.parse_args()
    args.string_in_screen = "epoch [{:3d}/{:3d}], episode:[{:3d}/{:3d}], needed [{:}], [{:}]" 
    args.acc_in_screen = 'acc_base:{:}, acc1:{:}; acc2:{:}; acc3:{:}; acc4:{:}; acc5:{:}; acc6:{:}; acc7:{:}'
    args.acc_h_in_screen = 'epoch [{:3d}/{:3d}], acc_base:{:} h_base:{:}, acc1:{:} h1:{:}; acc2:{:} h2:{:}; acc3:{:} h3:{:}; acc4:{:} h4:{:}; acc5:{:} h5:{:}; acc6:{:} h6:{:}; acc7:{:} h7:{:}'
    return args

if __name__=='__main__':
    main()
