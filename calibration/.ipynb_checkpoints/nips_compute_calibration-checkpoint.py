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
from code_lib import generate, load, metric, util
from code_models.classifiers import MaximumLikelihood
from code_models import layers, losses, classifiers
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# CUDA_VISIBLE_DEVICES=5 python test_mqda.py -p2l mini_conv4_delete_0.8_sgd_map_0.01_5_logits -d_f mini -d_f cub -d_s train
# CUDA_VISIBLE_DEVICES=6 python nips_compute_calibration.py -p2l mini_conv4_0.5_sgd_fb_0.01_5N1K_logits --domain_feature cars
# torch.cuda.set_device(3)
def main():
    util.set_random(4603)
    xargs = config_local_parameters()
    net_domain, net_arch, data_domain = xargs.path2log.split('_')[0], xargs.path2log.split('_')[1], xargs.domain_feature
    
    xargs.path2features = '../../data_src/fea_mqda/{:}-{:}-{:}-fea.pkl'.format(net_domain, net_arch, data_domain)
#     xargs.path2features = '../data_src/fea_mqda/{:}-{:}-{:}-aux-fea.pkl'.format(net_domain, net_arch, data_domain)
    xargs.feature_or_logits = [1 if 'logits' in xargs.path2log else 0][0]
    x_np, _ = load.numpy_features(xargs, xargs.path2features, xargs.feature_or_logits)
    
    
    valclass ='../../data_src/images/{:}/{:}'.format(data_domain, 'val')    
    novelclass ='../../data_src/images/{:}/{:}'.format(data_domain, xargs.data_split)
    folders_val = generate.folders(valclass)
    folders_novel = generate.folders(novelclass)
    
    log_dir = '../active_log/basic/{:}'.format(xargs.path2log)
    ckps = util.search('*pth', target_space = log_dir, reverse=True)
    print(ckps[0])
#     pdb.set_trace()
    logger = util.Logger(log_dir, data_domain)
    best_acc = metric.BestAcc()
    timer = metric.CountdownTimer(total_steps=len(ckps))
    for epoch_index, checkpoint_item in enumerate(ckps):
#         if epoch_index ==0:
#             continue
        classifier, args,_,_ = load.checkpoint(checkpoint_item)
        args.n_episode_test = xargs.n_episode_test
        args.path2image = xargs.path2image
        ece_mean = metatest_epoch(folders_val, folders_novel, x_np, classifier, logger, args, xargs)
        best_acc.update(ece_mean, 0, args.index_epoch)
        logger.print(xargs.string_in_screen.format(args.index_epoch + 1, args.n_epoch, args.n_episode_test, ece_mean, timer.step(), util.time_now())) 
    logger.rename('uncertainty_{:}_ece_{:}'.format(best_acc.index, best_acc.value, best_acc.h_))
    
def metatest_epoch(folders_val, folders, x_np, classifier, logger, args, xargs):

    # 1. Find the value of temperature (calibration)    
    print("Calibration: finding temperature hyperparameter...")
    ece_module = losses.ECELoss_NIPs()
    temperature = 1.0
#     # A__
#     temperature_list = list()
#     for _ in range(xargs.repeat):#repeat):
# #         _set_seed(0) # random seed
#         logits, targets = episode_loop(folders_val, x_np, classifier, logger, args)
#         temperature = ece_module.calibrate(logits, targets, iterations=300, lr=0.01).item()
#         if(temperature>0): temperature_list.append(temperature)
#         print("Calibration: temperature", temperature, "; mean temperature", np.mean(temperature_list))
#     # Filtering invalid temperatures (e.g. temp<0)
#     if(len(temperature_list)>0):temperature = np.mean(temperature_list) 
#     else: temperature = 1.0
#     # __A

#     2. Use the temperature to record the ECE
#     repeat the test N times changing the seed in range [seed, seed+repeat]
    ece_list = list()
    for i in range(args.seed, args.seed+xargs.repeat):
        if(args.seed!=0): _set_seed(i)
        else: _set_seed(0)
        logits, targets = episode_loop(folders, x_np, classifier, logger, args)
        #ece = ece_module.forward(logits, targets, temperature, onevsrest=params.method=='DKT').item()
        ece = ece_module.forward(logits, targets, temperature, onevsrest=False).item()
        ece_list.append(ece)
        print("ECE:", np.mean(ece_list), "+-", np.std(ece_list))

    # 3. Print the final ECE (averaged over all seeds)
    print("-----------------------------")
    print('Seeds = %d | Overall ECE = %4.4f +- %4.4f' %(xargs.repeat, np.mean(ece_list), np.std(ece_list)))
    print("-----------------------------")     
    
    return np.mean(ece_list)

def episode_loop(folders, x_np, classifier, logger, args):
    accuracies = metric.AverageMeter()
    classifier.eval()
#     classifier = classifiers.NearestCentroid_SimpleShot()
    with torch.no_grad():
        logits_list = list()
        targets_list = list()
        for _ in range(args.n_episode_test):
            novel_task = generate.Task(folders, args.n_way, args.k_spt, args.k_qry)
            (x_spt, y_spt), (x_qry, y_qry) = load.tensor_features_from(novel_task, x_np, args)
            x_spt_norm = layers.centre_l2norm(x_spt, args.x_centre)
            x_qry_norm = layers.centre_l2norm(x_qry, args.x_centre)
            classifier.fit_image_label(x_spt_norm, y_spt)
            logits = classifier.predict(x_qry_norm)
            accuracies.update(metric.accuracy(logits.argmax(dim=1), y_qry))
            logits_list.append(logits)
            targets_list.append(y_qry)
    accuracy, h_ = accuracies.mean_confidence_interval() 
    print('acc:{:}, h:{:}'.format(accuracy, h_))
    return torch.cat(logits_list, 0), torch.cat(targets_list, 0)

def _set_seed(seed, verbose=True):
    if(seed!=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
        if(verbose): print("[INFO] Setting SEED: " + str(seed))   
    else:
        if(verbose): print("[INFO] Setting SEED: None")

def config_local_parameters():
    parser = argparse.ArgumentParser(description= 'FSL&CFSL testing script' )
    parser.add_argument('-p2l', '--path2log', default='mini_conv4_sgd_Fix1_map_0.001_5_logits64',  help='path to log') 
    parser.add_argument('-n_l', '--net_label', default='simpleshot', help='the encoder')
    parser.add_argument('-d_f', '--domain_feature', default='mini',  help='domain of the feature') 
    parser.add_argument('-e_t', '--n_episode_test', default=500, type=int,  help='')
    parser.add_argument('-d_s', '--data_split', default='test', type=str,  help='')
    parser.add_argument('-repeat', '--repeat', default=1, type=int,  help='')
    parser.add_argument('--path2image', default='../../data_src/images/', help='path to all datasets: CUB, Mini, etc')
    args = parser.parse_args()
    args.string_in_screen = "epoch [{:3d}/{:3d}], episode:{:}, ece:{:}, needed [{:}], [{:}]" 
    return args

if __name__=='__main__':
    main()
