import os
import pdb
import argparse
import torch
import torch.optim as optim    
from code_models import ConvNets
from code_models import classifiers

def meta_qda(args):
    
    if args.strategy == 'map':
        print("use map version")
        classifier = classifiers.MetaQDA_MAP(args)
    elif args.strategy == 'fb':
        print("use fb_slow version")
        classifier = classifiers.MetaQDA_FB(args)   
    else:
        raise ValueError('strategy should be map or fb, but your strategy is: ', strategy)
    
    return classifier

def optimizer(params, args, wd=1e-4):

    if args.optimizer == 'adam':
        optimizer  = optim.Adam(params, lr=args.lr, weight_decay=wd)
    elif args.optimizer == 'adamw':
        optimizer  = optim.AdamW(params, lr=args.lr, weight_decay=wd)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=wd, nesterov=False)
    else:
        raise ValueError('args.optimier should be adamW, adam, sgd')
        
    return optimizer

def lr_scheduler(optimizer, args):
    if args.lr_scheduler == 'multsteplr':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.n_episode_train* args.n_epoch * 0.5), int(args.n_episode_train*args.n_epoch * 0.83)], gamma=0.1)
    elif args.lr_scheduler == 'consinelr':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.n_episode_train*args.n_epoch)
    else:
        raise ValueError('args.lr_scheduler should be MultiStepLR or CosineLR')
    return lr_scheduler

def encoder(arch, num_classes, parameter_dir=False):
    if arch == "conv4":
        _structure = ConvNets.Conv4(num_classes=num_classes)
        if parameter_dir:
            parameter = torch.load(parameter_dir)
    elif arch == 'resnet18':
        _structure = ConvNets.resnet18(num_classes=num_classes, remove_linear=False)
#         _structure.lc = None
        if parameter_dir:
            parameter = torch.load(parameter_dir)['state_dict']
            parameter.pop('module.fc.weight')
            parameter.pop('module.fc.bias')
    else:
        raise ValueError('encoder should be conv4_64, conv4_128, resnet18, but the input is', arch )
        
    if parameter_dir:
        model_state_dict = _structure.state_dict()
        for key in parameter:
            model_state_dict[key.replace('module.', '')] = parameter[key]
        _structure.load_state_dict(model_state_dict)
    return _structure #