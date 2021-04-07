import os
import sys
import pdb
import time
import pickle
import random
import argparse
import numpy as np
from pathlib import Path
mqda_lib_dir = Path(__file__).parent.resolve().parent
sys.path.insert(0, str(mqda_lib_dir))
import torch
import torch.nn as nn
import torch.nn.functional as F
from code_lib import generate, load, util, metric
from code_models import classifiers, layers
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# CUDA_VISIBLE_DEVICES=4 python clbrt_mqda.py -p2l mini_conv4_reg0.0_sgd_map_0.0001_5_logits -reg 0.0 -ckp_id 0

# CUDA_VISIBLE_DEVICES=2 python clbrt_mqda_multiepoch.py -p2l mini_conv4_0.5_sgd_fb_0.01_5N5K_logits -d_s train
# torch.cuda.set_device(3)
def main(log, index):
    util.set_random(4603)
    xargs = config_local_parameters()
    xargs.path2log = log
    xargs.chekpoint_id = index
    net_domain, net_arch, data_domain = xargs.path2log.split('_')[0], xargs.path2log.split('_')[1], xargs.domain_feature
    
    xargs.path2features = '../../data_src/fea_mqda/{:}-{:}-{:}-fea.pkl'.format(net_domain, net_arch, data_domain)
    xargs.feature_or_logits = [1 if 'logits' in xargs.path2log else 0][0]
    x_np, _ = load.numpy_features(xargs, xargs.path2features, xargs.feature_or_logits)
        
    novelclass ='../../data_src/images/{:}/{:}'.format(data_domain, xargs.data_split)
    folders_novel = generate.folders(novelclass)
    checkpoint_item = '../active_log/basic/{:}/epoch-{:}.pth'.format(xargs.path2log, xargs.chekpoint_id)
    classifier, args,_,_ = load.checkpoint(checkpoint_item)
    args.episode_test_num = xargs.test_episode
#     args.path2image = str(Path(xargs.path2image).resolve())+'/'
    args.path2image = xargs.path2image
    args.np_features = np_features
    try:
        args.n_way = args.way
        args.k_spt = args.support_shot
        args.k_qry = args.query_shot
    except:
        pass
    
#     val_data(classifier, args)
    test_data(folders_novel, classifier, args)    
    
def val_data(classifier, args):
    
    classifier.eval()
    with open('../results/clbrt/outputs/reused_val-{:}S.pkl'.format(args.k_spt),'rb') as tf:
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
            
        x_spt = load.name2tensor_feature(name_spt, args.path2image, args.np_features)
        x_qry = load.name2tensor_feature(name_qry, args.path2image, args.np_features)

        y_spt = y_spt.to(DEVICE)
        y_qry = y_qry.to(DEVICE)
        
        x_spt = layers.centre_l2norm(x_spt, args.x_centre)
        x_qry = layers.centre_l2norm(x_qry, args.x_centre)    
    
        classifier.fit_image_label(x_spt, y_spt)
        logits = classifier.predict(x_qry)
        
        save_outputs[unused_i] = {}
        save_outputs[unused_i]['logits'] = logits
        save_outputs[unused_i]['label'] = y_qry.cpu().numpy()
        save_outputs[unused_i]['names'] = [ item.split('/')[-2] for item in name_qry]

#     with open('./results/clbrt/outputs/loglab-m-{:}-{:}S-{:}.pkl'.format(args.net_arch, args.k_spt,'val'),'wb') as f:
#         pickle.dump(save_outputs, f)

    with open('../results/clbrt/outputs/henry-{:}-{:}S-{:}.pkl'.format(args.net_arch, args.k_spt,'test'),'wb') as f:
        pickle.dump(save_outputs, f)
    
            
def test_data(folders, classifier, args):
    accuracies = metric.AverageMeter()   
    classifier.eval()
    save_outputs = {}
    for unused_i in range(args.episode_test_num ):#args.metatest_episode):
        if unused_i % 1 == 0:
            print(unused_i)
            
        novel_task = generate.Task(folders, args.n_way, args.k_spt, args.k_qry)

        name_spt, y_spt = generate.name_label(novel_task, "support").__iter__().next()
        name_qry, y_qry = generate.name_label(novel_task, "query").__iter__().next()
        x_spt = load.name2tensor_feature(name_spt, args.path2image, args.np_features)
        x_qry = load.name2tensor_feature(name_qry, args.path2image, args.np_features)

        y_spt = y_spt.to(DEVICE)
        y_qry = y_qry.to(DEVICE)
        
        x_spt_norm = layers.centre_l2norm(x_spt, args.x_centre)
        x_qry_norm = layers.centre_l2norm(x_qry, args.x_centre)    
    
        classifier.fit_image_label(x_spt_norm, y_spt)
        logits = classifier.predict(x_qry_norm)
        accuracies.update(metric.accuracy(logits.argmax(dim=1), y_qry))
            
    
        save_outputs[unused_i] = {}
        save_outputs[unused_i]['logits'] = logits
        save_outputs[unused_i]['label'] = y_qry.cpu().numpy()
        save_outputs[unused_i]['names'] = [ item.split('/')[-2] for item in name_qry]
        
    accuracy, h_ = accuracies.mean_confidence_interval() 
    print('acc {:} h_{:}'.format(accuracy, h_))
    with open('../results/clbrt/outputs/loglab-m-{:}-{:}S-{:}.pkl'.format(args.net_arch, args.k_spt,'test'),'wb') as f:
#         pickle.dump(save_outputs, f)
#     with open('./results/clbrt/outputs/InvReg{:}-alpha{:}-{:}-{:}S-{:}.pkl'.format(args.choleky_alpha, args.index_epoch, args.net_arch, args.k_spt,'test'),'wb') as f:
        pickle.dump(save_outputs, f)    
        
def config_local_parameters():
    parser = argparse.ArgumentParser(description= 'FSL&CFSL testing script' )
    parser.add_argument('-p2l', '--path2log', default='mini_conv4_sgd_Fix1_map_0.001_5_logits64',  help='path to log') 
    parser.add_argument('-ckp_id', '--chekpoint_id', default=0, help='the encoder') # 95
    parser.add_argument('-d_f', '--domain_feature', default='mini',  help='domain of the feature') 
    parser.add_argument('-t_e', '--test_episode', default=500, type=int,  help='')
    parser.add_argument('-d_s', '--data_split', default='test', type=str,  help='')
    parser.add_argument('-ust','--update_step_test', type=int, help='update steps for finetunning', default=1)
    parser.add_argument('-reg', '--reg_param', default=0.0, type=float,  help='')
    parser.add_argument('--path2image', default='../data_src/images/', help='path to all datasets: CUB, Mini, etc')    
    return parser.parse_args()

if __name__=='__main__':
    log_list = ['mini_conv4_cholesky65.0_sgd_map_0.0001_5_logits',
                'mini_conv4_cholesky200.0_sgd_map_0.0001_5_logits',
                'mini_conv4_cholesky500.0_sgd_map_0.0001_5_logits',
                'mini_conv4_cholesky2000.0_sgd_map_0.0001_5_logits']
    for log in log_list:
        for index in [0, 30, 50, 80, 99]:
            main(log, index)
