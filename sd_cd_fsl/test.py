import os
import sys
import pdb
import time
import pickle
import random
import argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve().parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from code_lib import generate, load, metric, util
from code_models.classifiers import MaximumLikelihood
from code_models import layers, losses
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
def main():
    xargs = config_local_parameters()
    net_domain, net_arch = xargs.log_name.split('_')[0], xargs.log_name.split('_')[1]
    data_domain = xargs.x_domain if xargs.x_domain else net_domain
    xargs.feature_or_logits = 1 if 'logits' in xargs.log_name else 0
    xargs.path2features = '../../data_src/fea_mqda/{:}-{:}-{:}-fea.pkl'.format(net_domain, net_arch, data_domain)
    x_np, _ = load.numpy_features(xargs, xargs.path2features, xargs.feature_or_logits)
        
    novelclass ='../../data_src/images/{:}/{:}'.format(data_domain, xargs.data_split)
    folders_novel = generate.folders(novelclass)
    
    log_dir = '../log/SD_CD_FSL/{:}'.format(xargs.log_name)
    ckps = util.search('*pth', target_space = log_dir)
    logger = util.Logger(log_dir, data_domain)
    best_acc = metric.BestAcc()
    timer = metric.CountdownTimer(total_steps=len(ckps))
    for epoch_index, checkpoint_item in enumerate(ckps):
        classifier, args,_,_ = load.checkpoint(checkpoint_item)
        acc, h_ = metatest_epoch(folders_novel, x_np, classifier, logger, args, xargs)
        best_acc.update(acc, h_, args.index_epoch)
        logger.print(xargs.string_in_screen.format(args.index_epoch + 1, args.n_epoch, xargs.n_episode_test, acc, h_, timer.step(), util.time_now())) 
    logger.rename('epoch_{:}_acc_{:}_h_{:}'.format(best_acc.index, best_acc.value, best_acc.h_))
    
def metatest_epoch(folders, x_np, classifier, logger, args, xargs):
    args.path2image = xargs.path2image
    accuracies = metric.AverageMeter()
    classifier.eval()
    with torch.no_grad():
        for _ in range(xargs.n_episode_test):
            novel_task = generate.Task(folders, args.n_way, args.k_spt, args.k_qry)
            (x_spt, y_spt), (x_qry, y_qry) = load.tensor_features_from(novel_task, x_np, args)
        
            x_spt_norm = layers.centre_l2norm(x_spt, args.x_centre)
            x_qry_norm = layers.centre_l2norm(x_qry, args.x_centre)
            
            classifier.fit_image_label(x_spt_norm, y_spt)
            outputs = classifier.predict(x_qry_norm)
            accuracies.update(metric.accuracy(outputs.argmax(dim=1), y_qry))                
            
    accuracy, h_ = accuracies.mean_confidence_interval() 
    return accuracy, h_

def config_local_parameters():
    parser = argparse.ArgumentParser(description= 'SD&CDFSL testing script' )
    parser.add_argument('-l_n', '--log_name', required=True) 
    parser.add_argument('-x_d', '--x_domain', default=None,  help='domain of features for test set') 
    parser.add_argument('-e_t', '--n_episode_test', default=600, type=int)
    parser.add_argument('-d_s', '--data_split', default='test', type=str)
    parser.add_argument('--path2image', default='../../data_src/images/', help='path to all datasets: CUB, Mini, etc')
    args = parser.parse_args()
    args.string_in_screen = "epoch [{:3d}/{:3d}], episode:{:}, acc:{:} h:{:}, needed [{:}], [{:}]" 
    return args

if __name__=='__main__':
    main()
