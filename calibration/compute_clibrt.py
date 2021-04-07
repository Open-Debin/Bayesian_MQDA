import os
import sys
import pdb
import pickle
import argparse
sys.path.append('..')
import numpy as np
import torch
import torchvision as tv
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from code_lib import util
from code_models.losses import LogitsWithTemperature
from results.clbrt.calibration_sce import run_calibration
parser = argparse.ArgumentParser(description= 'FSL&CFSL testing script' )
parser.add_argument('-ckp_id', '--chekpoint_id', default=0, type=int,help='the encoder') # 95
parser.add_argument('-alpha', '--cholesky_alpha', default=0.0, type=float,  help='')
parser.add_argument('--head', default=0.1, type=float,  help='')
parser.add_argument('--end', default=30, type=float,  help='')
parser.add_argument('--interval', default=60, type=int,  help='')
parser.add_argument('--val_file', type=str,  help='')
parser.add_argument('--test_file',default=None, type=str,  help='')
parser.add_argument('--temp', default=None, type=float,  help='')

args = parser.parse_args()
# python compute_clibrt.py --val_file henry-s2m2r-1S-val.pkl --head 0.1 --end 3.1 --interval 30
# python compute_clibrt.py --test_file henry-s2m2r-1S-test.pkl --temp 0.32
# python compute_clibrt.py --test_file henry-s2m2r-1S-test-cub.pkl --temp 1
net='resnet18'
method='m' # ncc lc m
shot=5
# train_f = 'loglab-{:}-{:}-{:}S-train.pkl'.format(method,net,shot)
# val_f = 'loglab-{:}-{:}-{:}S-val.pkl'.format(method,net,shot)
# test_f = 'loglab-{:}-{:}-{:}S-test.pkl'.format(method,net,shot)
# val_f = 'henry-s2m2r-5S-val.pkl'
# # val_f = 'loglab-m-resnet18-1S-test.pkl'
# # test_f = 'loglab-maml-resnet18-5S-test.pkl'
# val_f = 'loglab-m-conv4-5S-test-maml.pkl'
# test_f = 'loglab-m-conv4-5S-test-maml.pkl'

# train_f = 'loglab-{:}-{:}-{:}S-train-maml.pkl'.format(method,net,shot)
# val_f = 'loglab-{:}-{:}-{:}S-val-maml.pkl'.format(method,net,shot)
# test_f = 'loglab-{:}-{:}-{:}S-test-maml.pkl'.format(method,net,shot)
# with open(train_f,'rb') as tf:
#     train_loader = pickle.load(tf)
# val_f = 'henry-s2m2r-1S-val.pkl'
val_f = args.val_file
if args.test_file:
    val_f = args.test_file
#val_f = 'henry-s2m2r-1S-test.pkl'
with open(val_f,'rb') as vf:
    val_loader = pickle.load(vf)
# with open(test_f,'rb') as tf:
#     test_loader = pickle.load(tf)
# val_loader = test_loader
scaled_model = LogitsWithTemperature()
# for index_task in train_loader:
#     print(scaled_model.temperature)
#     logits = train_loader[index_task]['logits']
#     label = torch.tensor(train_loader[index_task]['label'])
#     scaled_model.set_temperature(logits, label)

# clbrt_cate=[]
# clbrt_prob=[]
# clbrt_prob_temp=[]
# clbrt_bool=[]

# scaled_model.temperature=torch.nn.Parameter(torch.ones(1).cuda() * 1.5)
# for index_task in val_loader:

#     logits = val_loader[index_task]['logits']
#     labels = torch.tensor(val_loader[index_task]['label'])
#     names = val_loader[index_task]['names']
#     temp_logits = scaled_model.forward(logits)

#     _, bool_index =util.accuracy(logits.max(dim=1)[1], labels)
#     clbrt_cate.extend(names)
#     clbrt_prob.extend(list(F.softmax(logits, dim=1).max(dim=1)[0]))
#     clbrt_prob_temp.extend(list(F.softmax(temp_logits.detach().cpu(), dim=1).max(dim=1)[0]))
#     clbrt_bool.extend(bool_index)
        
    
# scaled_model.temperature=torch.nn.Parameter(torch.ones(1).cuda() * 1.35)
# for index_task in test_loader:
#     logits = test_loader[index_task]['logits']
#     labels = torch.tensor(test_loader[index_task]['label'])
#     names = test_loader[index_task]['names']
# #     pdb.set_trace()
#     temp_logits = scaled_model.forward(logits)
    
#     _, bool_index =util.accuracy(logits.max(dim=1)[1], labels)
#     clbrt_cate.extend(names)
#     clbrt_prob.extend(list(F.softmax(logits, dim=1).detach().cpu().max(dim=1)[0]))
#     clbrt_prob_temp.extend(list(F.softmax(temp_logits.detach().cpu(), dim=1).max(dim=1)[0]))
#     clbrt_bool.extend(bool_index)

# with open('logits-{:}-t-{:}-{:}.pkl'.format(method, net,shot),'wb') as f:
#     print('logits-{:}-t-{:}-{:}.pkl'.format(method, net,shot))
#     pickle.dump([np.array(clbrt_cate), np.array(clbrt_prob), np.array(clbrt_prob_temp), np.array(clbrt_bool)], f)
# print('Eva&Debin')   
# pdb.set_trace()
# for index in [0, 30, 50, 80, 99]:
#     val_f = 'InvReg{:}-alpha{:}-conv4-5S-test.pkl'.format(args.cholesky_alpha, index)
#     print(val_f)
with open(val_f,'rb') as vf:
    val_loader = pickle.load(vf)
scaled_model = LogitsWithTemperature()
points_estimate = np.linspace(args.head,args.end,args.interval,False)
if args.temp:
    points_estimate = np.linspace(1,1.1,1,False)
for i in points_estimate:
    if args.temp:
        i = args.temp
    clbrt_cate=[]
    clbrt_prob=[]
    clbrt_prob_temp=[]
    clbrt_bool=[]
    scaled_model.temperature=torch.nn.Parameter(torch.ones(1).cuda() * i)
    for index_task in val_loader:

        logits = val_loader[index_task]['logits']
#         pdb.set_trace()
        labels = torch.tensor(val_loader[index_task]['label'])
        names = val_loader[index_task]['names']
        temp_logits = scaled_model.forward(logits)
#         pdb.set_trace()
        bool_index = torch.eq(logits.max(dim=1)[1].cpu(), labels).int().numpy()
#         _, bool_index =util.accuracy(logits.max(dim=1)[1], labels)
        clbrt_cate.extend(names)
        clbrt_prob.extend(list(F.softmax(logits, dim=1).max(dim=1)[0]))
        clbrt_prob_temp.extend(list(F.softmax(temp_logits.detach().cpu(), dim=1).max(dim=1)[0]))
        clbrt_bool.extend(bool_index)
#     pdb.set_trace()
    ece = run_calibration([np.array(clbrt_cate), np.array(clbrt_prob_temp), np.array(clbrt_bool)], 20, False)[0]
#         print('value:{:} ece:{:}'.format(round(i,3), ece))
    print('i:{:}, ece:{:}'.format(i,ece))
