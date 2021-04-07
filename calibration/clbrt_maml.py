import os
import pdb
import time
import pickle
import random
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_code import util
from basic_code import generate, load
from basic_code.classifiers_define import MaximumLikelihood, MAMLQDA, QDA_
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# CUDA_VISIBLE_DEVICES=4 python clbrt_maml.py -p2l daisy1mamlnorm_mini_resnet18_2_0.001_1_fea512
# CUDA_VISIBLE_DEVICES=2 python test_mqda.py -p2l mini_conv4_sgd_Fix1_map_0.0001_5_logits64 -d_s train
# torch.cuda.set_device(3)
def main():
    util.set_random(4603)
    xargs = config_local_parameters()
    net_domain, net_arch, data_domain = xargs.path2log.split('_')[1], xargs.path2log.split('_')[2], xargs.domain_feature
    
    path2features = '../data_src/fea_mqda/{:}-{:}-{:}-fea.pkl'.format(net_domain, net_arch, data_domain)
    feature_or_logits = [1 if 'logits' in xargs.path2log else 0][0]
    np_features, centre_base_val, _ = load.numpy_features(path2features, feature_or_logits)
    centre_base_val = torch.tensor(centre_base_val).to(DEVICE)
        
    novelclass ='../data_src/images/{:}/{:}'.format(data_domain, xargs.data_split)
    folders_novel = generate.folders(novelclass)
    checkpoint_item = './active_log/{:}/epoch-{:}.pth'.format(xargs.path2log, xargs.chekpoint_id)
    qda, args = load.maml_checkpoint(checkpoint_item)
    args.update_step_test = xargs.update_step_test
    maml_qda = MAMLQDA(args, qda, np_features) 
    args.episode_test_num = xargs.test_episode
    args.path2image = str(Path(xargs.path2image).resolve())+'/'
    val_data(maml_qda, args)
    test_data(folders_novel, maml_qda, args)    
    
def val_data( maml_qda, args):
    
    maml_qda.eval()
    with open('./results/clbrt/outputs/reused_val-{:}S.pkl'.format(args.k_spt),'rb') as tf:
        reused_val = pickle.load(tf)
    save_outputs = {}
    for unused_i in range(args.episode_test_num ):#args.metatest_episode):
        if unused_i % 1 == 0:
            print(unused_i)
        name_spt, y_spt = reused_val[unused_i]['spt_names'], reused_val[unused_i]['spt_labels']
        name_qry, y_qry = reused_val[unused_i]['qry_names'], reused_val[unused_i]['qry_labels']  
        
        for index, item in enumerate(name_spt):
            name_spt[index] = str(Path(item).resolve())

        for index, item in enumerate(name_qry):
            name_qry[index] = str(Path(item).resolve())
            
        x_spt = load.name2tensor_feature(name_spt, args.path2image, maml_qda.np_features)
        x_qry = load.name2tensor_feature(name_qry, args.path2image, maml_qda.np_features)

        y_spt = y_spt.to(DEVICE)
        y_qry = y_qry.to(DEVICE)
        
        x_spt = util.centre_l2norm(x_spt, args.x_centre)
        x_qry = util.centre_l2norm(x_qry, args.x_centre)    
    
        logits = maml_qda.calibration(x_spt, y_spt, x_qry, y_qry)
        
        save_outputs[unused_i] = {}
        save_outputs[unused_i]['logits'] = logits
        save_outputs[unused_i]['label'] = y_qry.cpu().numpy()
        save_outputs[unused_i]['names'] = [ item.split('/')[-2] for item in name_qry]

    with open('./results/clbrt/outputs/loglab-maml-{:}-{:}S-{:}.pkl'.format(args.net_arch, args.k_spt,'val'),'wb') as f:
        pickle.dump(save_outputs, f)
            
def test_data(folders, maml_qda, args):
    
    maml_qda.eval()
    save_outputs = {}
    for unused_i in range(args.episode_test_num ):#args.metatest_episode):
        if unused_i % 1 == 0:
            print(unused_i)
            
        novel_task = generate.Task(folders, args.n_way, args.k_spt, args.k_qry)

        name_spt, y_spt = generate.name_label(novel_task, "support").__iter__().next()
        name_qry, y_qry = generate.name_label(novel_task, "query").__iter__().next()
        x_spt = load.name2tensor_feature(name_spt, args.path2image, maml_qda.np_features)
        x_qry = load.name2tensor_feature(name_qry, args.path2image, maml_qda.np_features)

        y_spt = y_spt.to(DEVICE)
        y_qry = y_qry.to(DEVICE)
        
        x_spt_norm = util.centre_l2norm(x_spt, args.x_centre)
        x_qry_norm = util.centre_l2norm(x_qry, args.x_centre)    
    
        logits = maml_qda.calibration(x_spt, y_spt, x_qry, y_qry)
        
        save_outputs[unused_i] = {}
        save_outputs[unused_i]['logits'] = logits
        save_outputs[unused_i]['label'] = y_qry.cpu().numpy()
        save_outputs[unused_i]['names'] = [ item.split('/')[-2] for item in name_qry]
        
    with open('./results/clbrt/outputs/loglab-maml-{:}-{:}S-{:}.pkl'.format(args.net_arch, args.k_spt,'test'),'wb') as f:
        pickle.dump(save_outputs, f)
        
def config_local_parameters():
    parser = argparse.ArgumentParser(description= 'FSL&CFSL testing script' )
    parser.add_argument('-p2l', '--path2log', default='mini_conv4_sgd_Fix1_map_0.001_5_logits64',  help='path to log') 
    parser.add_argument('-ckp_id', '--chekpoint_id', default=25, help='the encoder')
    parser.add_argument('-d_f', '--domain_feature', default='mini',  help='domain of the feature') 
    parser.add_argument('-t_e', '--test_episode', default=500, type=int,  help='')
    parser.add_argument('-d_s', '--data_split', default='test', type=str,  help='')
    parser.add_argument('-ust','--update_step_test', type=int, help='update steps for finetunning', default=1)
    parser.add_argument('--path2image', default='../data_src/images/', help='path to all datasets: CUB, Mini, etc')    
    return parser.parse_args()

if __name__=='__main__':
    main()
6