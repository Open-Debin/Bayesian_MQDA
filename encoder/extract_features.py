import os
import sys
import pdb
import pickle
import argparse
import collections
sys.path.append('..')
import numpy as np
from PIL import Image
import torch
from code_models import ConvNets
from code_lib import util, vision, generate, load, save

def main():
    xargs = config_parameters()
    if xargs.encoder == "resnet18":
        model = torch.nn.DataParallel(ConvNets.resnet18(num_classes=64)).cuda()
        path = './model_parameters/{:}/resnet18.pth.tar'.format(xargs.dataset_train, xargs.encoder)
        model.load_state_dict(torch.load(path)['state_dict'])
    else:
        raise ValueError('This Code Only support for ResNet18, but your input is', xargs.encoder)
        
    save_dir = '../data/features/{:}-{:}-{:}-fea.pkl'.format(xargs.dataset_train, xargs.encoder, xargs.dataset_inference)
    print(path)
    print(save_dir)
   
    path2train = xargs.path2image+'{:}/train'.format(xargs.dataset_inference)
    train_val_list = generate.data_list(path2train)+generate.data_list(path2train.replace('train','val'))
    test_list = generate.data_list(path2train.replace('train','test'))
    
    trva_loader = load.data(train_val_list, batch=256, _transform=vision.test_transform )
    te_loader = load.data(test_list, batch=256, _transform=vision.test_transform )
    
    dict_maker = generate.FeatureDictMaker(model, xargs.path2image)
    dict_maker._extract_features(trva_loader)
    fea_mean_trva = dict_maker.mean_features()
    logits_mean_trva = dict_maker.mean_logits()
    dict_maker._extract_features(te_loader)
    fea_dict, logits_dict = dict_maker.make_dict()

    save.features_in_dict(fea_dict, fea_mean_trva, logits_dict, logits_mean_trva, save_dir)
    
def config_parameters():
    parser = argparse.ArgumentParser(description= 'few-shot testing script' )
    parser.add_argument('--dataset_train', help='which domain of dataset used for training network') 
    parser.add_argument('--encoder', help='net architecture')
    parser.add_argument('--dataset_inference', help='which domain of dataset used for feature extraction')
    parser.add_argument('--path2image', default='../data/')    
    return parser.parse_args()

if __name__=='__main__': main()
