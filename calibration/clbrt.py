import os
import pdb
import glob
import math
import time
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from natsort import natsorted
from basic_code import util
from basic_code.classifiers_define import MetaQDA
from basic_code import task_test_generator as dataloder
from urt_code.lib.config_utils import Logger
from urt_code.lib.utils import convert_secs2time, AverageMeter
from basic_code.temperature_scaling import ModelWithTemperature, LogitsWithTemperature
from results.clbrt.calibration_sce import run_calibration
from collections import defaultdict
# python clbrt.py
SEED = 1
print('random seed: '.format(SEED))
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
cudnn.deterministic = True
# CUDA_VISIBLE_DEVICES=5 python clbrt.py
# 79 mini_conv4_sgd_trC_0.01_Gau_5S_logits64-T/
# 58 mini_conv4_sgd_trC_0.01_Gau_1S_logits64-T/
# 82 mini_s2m2r_sgd_trC_L2N_0.0001_1_fea640-T/
# 78 mini_s2m2r_adam_trC_L2N_0.0001_5S_fea640-T/
# 10 mini_conv4_sgd_trC_L2N_0.01_5_logits64_tdis/
def main():
    temperate_calibrt=False
    train_val = 'test'
    index_iter = 10
    parser = argparse.ArgumentParser(description= 'few-shot testing script' )
    parser.add_argument('-p2m', '--path2model', default='mini_conv4_sgd_trC_L2N_0.01_5_logits64_tdis/iter-{:}.pth'.format(index_iter), help='path to model') 
#     parser.add_argument('-p2m', '--path2model', default='mini_s2m2r_adam_trC_L2N_0.0001_5S_fea640/iter-{:}.pth'.format(index_iter), help='path to model') 
    parser.add_argument('-d_f', '--domain_feature', default='mini',  help='domain of the feature') 
    parser.add_argument('-t_e', '--test_episode', default=600, type=int,  help='')
    parser.add_argument('--n_bins',default=20)
    xargs = parser.parse_args()
    # Print Key Hyperparameters by Indicator
#     pdb.set_trace()
    domain_net=xargs.path2model.split('_')[0]
    net=xargs.path2model.split('_')[1]
    domain_data=xargs.domain_feature
    model_dir = './log_mqda/{:}'.format(xargs.path2model)
    with open('../data_src/fea_mqda/{:}-{:}-{:}-fea.pkl'.format(domain_net,net,domain_data),'rb') as f_read:
        presaved_features = pickle.load(f_read)            
    cudnn.benchmark = True
    mtr_path='../data_src/images/{:}/train'.format(domain_data)
    mva_path='../data_src/images/{:}/val'.format(domain_data)
    mte_path='../data_src/images/{:}/test'.format(domain_data)
    ''' ================================= Meta Test ================================= '''
#     _, _, mte_folders = dataloder.get_data_folders(mte_path, mte_path, mte_path)
    mte_folders,_, _ = dataloder.get_data_folders(mtr_path, mva_path, mte_path)
    item = torch.load(model_dir)
    args = item['args']
    classifier = MetaQDA(fea_dim=args.feature_dim, input_process=args.fea_trans, bool_autoft=args.bool_autoft, cholesky_alpha=args.choleky_alpha, prior=args.distri).to(torch.device("cuda"))
    classifier.load_state_dict(item['mqda'])
#         classifier= item['mqda']
    epoch = item['epoch']
    args.metatest_episode = xargs.test_episode
    clbrt_cate, clbrt_prob, clbrt_bool = metatest_epoch(mte_folders, presaved_features, classifier, args, clbrt=True,temperate_calibrt=temperate_calibrt, train_val=train_val)
    if temperate_calibrt:
        with open('./results/clbrt/{:}/mqda{:}-temp-{:}-{:}-{:}.pkl'.format(domain_net,index_iter,net,domain_data,args.support_shot),'wb') as f:
            pickle.dump([clbrt_cate, clbrt_prob, clbrt_bool], f)
    else:
        with open('./results/clbrt/{:}/mqda{:}-{:}-{:}-{:}.pkl'.format(domain_net,index_iter,net,domain_data,args.support_shot),'wb') as f:
            pickle.dump([clbrt_cate, clbrt_prob, clbrt_bool], f)
    print('Eva&Debin')
#     ece, sce, ace, ece_fract_true, ece_mean_prob = run_calibration([clbrt_cate,clbrt_prob,clbrt_bool], n_bins=xargs.n_bins)
    
#     clbrt_error =  'clbrt_ece_{:}_sce_{:}_ace_{:}'.format(ece,sce,ace)
#     ece_sce_ace=defaultdict(list)
#     ece_sce_ace['ece_fract']=ece_fract_true
#     ece_sce_ace['ece_prob']=ece_mean_prob
#     with open('{:}/{:}'.format(log_dir,clbrt_error),'wb') as f:
#         pickle.dump(ece_sce_ace, f)
#     print('Eva&Debin')
    
def metatest_epoch(mte_folders, presaved_features, classifier, args, clbrt=False, temperate_calibrt=True, train_val='train'):
    accuracies = []
    clbrt_cate=[]
    clbrt_prob=[]
    clbrt_bool=[]
    save_outputs={}
#     reused_val={}
    print(train_val)
    with open('./results/clbrt/outputs/reused_val-{:}S.pkl'.format(args.support_shot),'rb') as tf:
        reused_val = pickle.load(tf)
    if temperate_calibrt:
        args.support_shot = math.ceil(args.support_shot*1.2) # ceil(1shot*1.3) = 2 ceil(5shot*1.3)=7
    
    max_num = 500     
    for unused_i in range(max_num):#args.metatest_episode):
        if unused_i % 1 == 0:
            print(unused_i)
        if train_val =='val':
            support_image_names, support_labels = reused_val[unused_i]['spt_names'], reused_val[unused_i]['spt_labels']
            query_image_names, query_labels = reused_val[unused_i]['qry_names'], reused_val[unused_i]['qry_labels']  
        else:
#             print('itme')
            task_test = dataloder.DataLoadTask(mte_folders, args.support_way, args.support_shot, args.query_shot)
            # Load Features
            support_image_names, support_labels = dataloder.get_name_label(task_test, num_per_class=args.support_shot, split="train").__iter__().next()
            query_image_names, query_labels = dataloder.get_name_label(task_test, num_per_class=args.query_shot, split="test").__iter__().next()
            batch_size = query_labels.shape[0]
#         pdb.set_trace()
        support_features_cpu=util.name2fea(presaved_features, support_image_names, args.datapath, args.bool_logits)
        query_features_cpu=util.name2fea(presaved_features, query_image_names, args.datapath, args.bool_logits)
        if args.fea_trans == 'trC_L2N':
            centre_ = presaved_features['trCentre'][args.bool_logits]
        elif args.fea_trans == 'sptC_L2N':
            centre_ = support_features_cpu.mean(0)
        else:
            raise ValueError('args.fea_trans should be tr/spt_centre, but your input is', args.fea_trans )
        # Inference
        if temperate_calibrt:
            total = len(support_labels)
            indices = support_labels.sort()[1]
            index_naive = np.linspace(0,total,num=5,endpoint=False, dtype=int)
            index_base = np.ones(total)
            index_base[indices[index_naive]] = 0
            index_val = indices[index_naive]
            spt_base_f = support_features_cpu[index_base==1]
            spt_base_l = support_labels[index_base==1]
            spt_val_f = support_features_cpu[index_val]
            spt_val_l = support_labels[index_val]
            
            classifier.fit(torch.tensor(spt_base_f).cuda(), spt_base_l, torch.tensor(centre_).cuda())

            outputs_spt_val = classifier.predict(torch.tensor(spt_val_f).cuda(), torch.tensor(centre_).cuda())[0]
            outputs_qry, cholesky_loss_logitem, cholesky_loss_traceitem = classifier.predict(torch.tensor(query_features_cpu).cuda(),torch.tensor(centre_).cuda())[0:3]
            scaled_model = LogitsWithTemperature()
            outputs_spt_val = outputs_spt_val.detach().cpu()
            outputs_qry = outputs_qry.detach().cpu()
#             pdb.set_trace()
            scaled_model.set_temperature(outputs_spt_val, spt_val_l.cuda())
            outputs = scaled_model.forward(outputs_qry)
        else:
            classifier.fit(torch.tensor(support_features_cpu).cuda(), support_labels, torch.tensor(centre_).cuda())
            outputs, cholesky_loss_logitem, cholesky_loss_traceitem = classifier.predict(torch.tensor(query_features_cpu).cuda(),torch.tensor(centre_).cuda())[0:3]
        outputs = outputs.detach().cpu()
        
#         reused_val[unused_i]={}
#         reused_val[unused_i]['spt_names'] = support_image_names
#         reused_val[unused_i]['spt_labels'] = support_labels
#         reused_val[unused_i]['qry_names'] = query_image_names
#         reused_val[unused_i]['qry_labels'] = query_labels
        
        save_outputs[unused_i]={}
        save_outputs[unused_i]['logits']=outputs
        save_outputs[unused_i]['label']=query_labels.cpu().numpy()
#         pdb.set_trace()
        save_outputs[unused_i]['names']=[ item.split('/')[-2] for item in query_image_names]
        cholesky_loss_logitem = cholesky_loss_logitem.detach().cpu()
        cholesky_loss_traceitem = cholesky_loss_traceitem.detach().cpu()
        # Record Accuracy
#         pdb.set_trace()
        ce_loss = F.cross_entropy(outputs, query_labels)
        if args.choleky_alpha:
            choleske_loss_scale = ((args.choleky_alpha - args.feature_dim - 1) * (args.feature_dim/2) + 1e-1) * args.choleky_weight_scale
            cholesky_loss_logitem, cholesky_loss_traceitem = cholesky_loss_logitem/choleske_loss_scale, cholesky_loss_traceitem/choleske_loss_scale
            loss = ce_loss + (cholesky_loss_logitem + cholesky_loss_traceitem)
        else:
            loss=ce_loss
        predicts = outputs.max(dim=1)[1]
        metatest_acc, bool_index =util.accuracy(predicts, query_labels)
        accuracies.append(metatest_acc)
        if clbrt:
            clbrt_cate.extend([ item.split('/')[-2] for item in query_image_names])
            clbrt_prob.extend(list(F.softmax(outputs, dim=1).max(dim=1)[0]))
            clbrt_bool.extend(bool_index)
    if clbrt:
        # confidence_interval for accuray record
        accuracy, h_ = util.mean_confidence_interval(accuracies)
        accuracy = round(accuracy * 100, 2)
        h_ = round(h_ * 100, 2)
        loss = round(ce_loss.item(),2)
        
    with open('./results/clbrt/outputs/loglab-m-{:}-{:}S-{:}-tdis.pkl'.format('conv4', args.support_shot,train_val),'wb') as f:
        pickle.dump(save_outputs, f)
    pdb.set_trace()
    
    print('acc:{:}, h:{:}, loss:{:}'.format(accuracy, h_, loss))
    return np.array(clbrt_cate), np.array(clbrt_prob), np.array(clbrt_bool)
if __name__=='__main__':
    main()
