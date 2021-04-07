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
# CUDA_VISIBLE_DEVICES=7 python clbrt_mqda_Old.py -p2l mini_s2m2r_0.8_adam_fb_0.0001_5_fea_ece093
# CUDA_VISIBLE_DEVICES=7 python clbrt_mqda_Old.py -p2l mini_s2m2r_0.8_sgd_fb_0.0001_1_fea_ece3.02
# CUDA_VISIBLE_DEVICES=6 python clbrt_mqda_Old.py -p2l mini_conv4_0.5_sgd_fb_0.01_5N1K_logits -type False
# torch.cuda.set_device(3)
def main():
    util.set_random(4603)
    xargs = config_local_parameters()
    vars_item = vars(xargs)
    for item in vars_item:
        print(item)
        print(vars_item[item])
#         print("vars_item:{:}, {:}", format(item, vars_item[item]))
    net_domain, net_arch = xargs.path2log.split('_')[0], xargs.path2log.split('_')[1]
    
    path2features = '../../data_src/fea_mqda/{:}-{:}-{:}-fea.pkl'.format(net_domain, net_arch, xargs.domain_feature)
    feature_or_logits = [1 if 'logits' in xargs.path2log else 0][0]
    np_features, centre_base_val = load.numpy_features(xargs, path2features, feature_or_logits)
        
    novelclass ='../../data_src/images/{:}/{:}'.format(xargs.domain_feature, xargs.data_split)
    folders_novel = generate.folders(novelclass)
    checkpoint_item = '../active_log/basic/{:}/epoch-{:}.pth'.format(xargs.path2log, xargs.chekpoint_id)
    classifier, args,_,_ = load.checkpoint(checkpoint_item)
    args.episode_test_num = xargs.test_episode
#     args.path2image = str(Path(xargs.path2image).resolve())+'/'
    args.path2image = xargs.path2image
    args.x_centre = torch.tensor(centre_base_val).to(DEVICE)
    args.np_features = np_features
    try:
        args.n_way = args.way
        args.k_spt = args.support_shot
        args.k_qry = args.query_shot
    except:
        pass
    with torch.no_grad():
        val_data(classifier, xargs, args)
        test_data(folders_novel, classifier, xargs, args) 
        
        # bellow setting for cross-domain
        if xargs.cross_domain:
            path2features = '../../data_src/fea_mqda/{:}-{:}-cub-fea.pkl'.format(net_domain, net_arch)
            np_features, centre_base_val = load.numpy_features(xargs, path2features, feature_or_logits)
            args.x_centre = torch.tensor(centre_base_val).to(DEVICE)
            args.np_features = np_features
            folders_cub = generate.folders('../../data_src/images/cub/test')

            test_data(folders_cub, classifier, xargs, args, 'cub')  
    
def val_data(classifier, xargs, args):
    
    classifier.eval()
    if xargs.classifier_type == 'ncc':
        classifier = classifiers.NearestCentroid_SimpleShot()
    elif xargs.classifier_type == 'qda_map':
        classifier = classifiers.MetaQDA_MAP(args).to(DEVICE)
        
    with open('../results/clbrt/outputs/reused_val-{:}S.pkl'.format(args.k_spt),'rb') as tf:
        reused_val = pickle.load(tf)
    save_outputs = {}
    for unused_i in range(args.episode_test_num ):#args.metatest_episode):
        if unused_i % 1 == 0:
            print(unused_i)
        name_spt, y_spt = reused_val[unused_i]['spt_names'], reused_val[unused_i]['spt_labels']
        name_qry, y_qry = reused_val[unused_i]['qry_names'], reused_val[unused_i]['qry_labels']  
        
        for index, item in enumerate(name_spt):
#             name_spt[index] = str(Path(item).resolve())
            name_spt[index] = '../'+str(Path(item))

        for index, item in enumerate(name_qry):
#             name_qry[index] = str(Path(item).resolve())
            name_qry[index] = '../'+str(Path(item))
            
        x_spt = load.name2tensor_feature(name_spt, args.path2image, args.np_features)
        x_qry = load.name2tensor_feature(name_qry, args.path2image, args.np_features)

        y_spt = y_spt.to(DEVICE)
        y_qry = y_qry.to(DEVICE)
        
        x_spt_norm = layers.centre_l2norm(x_spt, args.x_centre)
        x_qry_norm = layers.centre_l2norm(x_qry, args.x_centre)    
    
        if args.strategy == 'fastfb_openset':
            predictor = classifier.fit_image_label(x_spt_norm, y_spt)
            logits = predictor.predict(x_qry_norm) 
        elif xargs.classifier_type == 'lin_clf':
            linear_clf = classifiers.mixup_lc_fsl(args.n_way, x_spt_norm, y_spt)
            logits = linear_clf(x_qry_norm)
        else:            
            classifier.fit_image_label(x_spt_norm, y_spt)
            logits = classifier.predict(x_qry_norm)   
        
        save_outputs[unused_i] = {}
        save_outputs[unused_i]['logits'] = logits
        save_outputs[unused_i]['label'] = y_qry.cpu().numpy()
        save_outputs[unused_i]['names'] = [ item.split('/')[-2] for item in name_qry]

    with open('./henry-{:}-{:}S-{:}.pkl'.format(args.net_arch, args.k_spt,'val'),'wb') as f:
        pickle.dump(save_outputs, f)
            
def test_data(folders, classifier, xargs, args, title=None):
    
    classifier.eval()
    if xargs.classifier_type == 'ncc':
        classifier = classifiers.NearestCentroid_SimpleShot()
    elif xargs.classifier_type == 'qda_map':
        classifier = classifiers.MetaQDA_MAP(args).to(DEVICE) 
        
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
    

        if args.strategy == 'fastfb_openset':
            predictor = classifier.fit_image_label(x_spt_norm, y_spt)
            logits = predictor.predict(x_qry_norm)
        elif xargs.classifier_type == 'lin_clf':
            linear_clf = classifiers.mixup_lc_fsl(args.n_way, x_spt_norm, y_spt)
            logits = linear_clf(x_qry_norm)
        else:            
            classifier.fit_image_label(x_spt_norm, y_spt)
            logits = classifier.predict(x_qry_norm)               
        
        save_outputs[unused_i] = {}
        save_outputs[unused_i]['logits'] = logits
        save_outputs[unused_i]['label'] = y_qry.cpu().numpy()
        save_outputs[unused_i]['names'] = [ item.split('/')[-2] for item in name_qry]
        
    if not title:
        with open('./henry-{:}-{:}S-{:}.pkl'.format(args.net_arch, args.k_spt,'test'),'wb') as f:
            pickle.dump(save_outputs, f)
    else:
        with open('./henry-{:}-{:}S-{:}-{:}.pkl'.format(args.net_arch, args.k_spt,'test', title),'wb') as f:
            pickle.dump(save_outputs, f)
            
def config_local_parameters():
    parser = argparse.ArgumentParser(description= 'FSL&CFSL testing script' )
    parser.add_argument('-p2l', '--path2log', default='mini_conv4_sgd_Fix1_map_0.001_5_logits64',  help='path to log') 
    parser.add_argument('-ckp_id', '--chekpoint_id', default=99, type=int, help='the encoder') # 95
    parser.add_argument('-d_f', '--domain_feature', default='mini',  help='domain of the feature') 
    parser.add_argument('-t_e', '--test_episode', default=500, type=int,  help='')
    parser.add_argument('-d_s', '--data_split', default='test', type=str,  help='')
    parser.add_argument('-ust','--update_step_test', type=int, help='update steps for finetunning', default=1)
    parser.add_argument('-type', '--classifier_type', default='lin_clf', type=str,  help='ncc, qda_map, lin_clf')
    parser.add_argument('-c_d', '--cross_domain', default=True)
    parser.add_argument('--path2image', default='../../data_src/images/', help='path to all datasets: CUB, Mini, etc')    
    return parser.parse_args()

if __name__=='__main__':
    main()
6