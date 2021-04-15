from __future__ import print_function
import os
import pdb
import pickle
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from code_lib import generate, util
from code_models import config
from code_models import classifiers
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data(data_list, batch=64, _transform=''):
    train_dataset = generate.Dataset_for_SD_CD(
        image_roots=data_list,
        transform=_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False)
    
    return train_loader

def batch_name(data_list, batch=64):
    train_dataset = generate.ImageNameDataset(
        image_roots=data_list)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False)
    
    return train_loader


def LoadParameter(_structure, _parameterDir=None):
    if _parameterDir:
        _parameterDir = torch.load(_parameterDir)
        model_state_dict = _structure.state_dict()

        for key in _parameterDir:
            model_state_dict[key.replace('module.', '')] = _parameterDir[key]

        _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure).cuda()

    return model

def numpy_features(args, path2features, logits):
    with open(path2features,'rb') as np_features:
        np_features = pickle.load(np_features)
        for key in np_features:
            np_features[key] = np_features[key][logits]
        global_centre = np_features['trCentre']
        args.x_dim = global_centre.size
    return dict(np_features), global_centre

def np_features_from(task, np_features, args):
    support_names, support_labels = generate.name_label(task, "support").__iter__().next()
    query_names, query_labels = generate.name_label(task, "query").__iter__().next()
    support_features = util.name2np_feature(support_names, args.path2image, np_features)
    query_features = util.name2np_feature(query_names, args.path2image, np_features)

    support_labels = support_labels.to(DEVICE)
    query_labels = query_labels.to(DEVICE)
    
    return (support_features, support_labels), (query_features, query_labels)

def tensor_features_from(task, np_features, args, mapping=False):
    support_names, support_labels = generate.name_label(task, "support").__iter__().next()
    query_names, query_labels = generate.name_label(task, "query").__iter__().next()
    support_features = name2tensor_feature(support_names, args.path2image, np_features)
    query_features = name2tensor_feature(query_names, args.path2image, np_features)
    
    if mapping:
        support_labels = torch.tensor(mapping.name2id(support_names)).to(DEVICE)
        query_labels = torch.tensor(mapping.name2id(query_names)).to(DEVICE)
    else:
        support_labels = support_labels.to(DEVICE)
        query_labels = query_labels.to(DEVICE)
    
    return (support_features, support_labels), (query_features, query_labels)

def name2np_feature(names, path2image_dir, np_feature_dict):
    names=[e.replace(path2image_dir,'') for e in names]
    np_feature_list=[]
    for name in names:
        np_feature_list.append(np_feature_dict[name])   
        
    np_features = np.stack(np_feature_list, axis=0)
    return np_features

def name2tensor_feature(names, path2image_dir, np_feature_dict):
    names=[e.replace(path2image_dir,'') for e in names]
    np_feature_list=[]
    for name in names:
        np_feature_list.append(np_feature_dict[name])   
        
    tensor_features = torch.tensor(np.stack(np_feature_list)).to(DEVICE)
    return tensor_features

def checkpoint(checkpoint):
    checkpoint = torch.load(os.path.join(checkpoint))
    args = checkpoint['args']
    classifier = config.meta_qda(args).to(DEVICE)  
    classifier.load_state_dict(checkpoint['mqda'])
        
    return classifier, args, checkpoint['optim'], checkpoint['scheduler']

def urt_mqda(checkpoint):
    checkpoint = torch.load(os.path.join(checkpoint))
    args = checkpoint['args']
    classifier = config.meta_qda(args).to(DEVICE)  
    classifier.load_state_dict(checkpoint['mqda'])
    
    return checkpoint['urt'], classifier, args, checkpoint['optim'], checkpoint['scheduler']