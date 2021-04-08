import pdb
import os
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.utils.weight_norm import WeightNorm
from code_models import ConvNets, layers
from code_lib import util, load
torch.set_default_tensor_type(torch.FloatTensor)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MetaQDA_FB_FixShot(nn.Module):
    def __init__(self, fea_dim):
        super(MetaQDA_FB_FixShot, self).__init__()
        self.feature_dim = fea_dim
#         self.fix_Nj = fix_Nj
        self.plogdet = PLogDet.apply
        
        self.m = torch.nn.Parameter(torch.zeros(1, self.feature_dim))
        self.kappa = torch.nn.Parameter(torch.tensor(0.1))
        self.nu = torch.nn.Parameter(torch.tensor(self.feature_dim, dtype=torch.float32))
        self.triu_S_diag = torch.nn.Parameter(torch.ones(self.feature_dim))
        self.triu_S_lower = torch.nn.Parameter(torch.eye(self.feature_dim))
        self.triu_S_lower_mask = torch.triu(torch.ones(self.feature_dim, self.feature_dim), diagonal=1).t().to(DEVICE)

    def fit_image_label(self, X, y):
        self.classes = np.unique(y.cpu())
        self.mu = [None for i in self.classes]
        self.sigma_inv = [None for i in self.classes]
        self.biases = [None for i in self.classes]
        self.fix_Nj = 5 #X.shape[0]/(y.max().item()+1)
        
        kappa_n, m_with_weight, xmean_weight, scatter_matrix, self.sharedpart, bias_sharedpart = self.compute_shared_part()
        for j in self.classes:
            X_j = X[y==j]
            x_mean = torch.mean(X_j, dim=0, keepdim=True)
            self.mu[j] = m_with_weight+x_mean.mul(xmean_weight)
            sigma_j_no_scale = scatter_matrix+mean_outer(X_j)+self_outer(x_mean-self.m).mul(torch.abs(self.kappa)+1e-6).div(kappa_n) 
            sigma_j = torch.div(sigma_j_no_scale, (kappa_n+1).div(kappa_n.mul(self.sharedpart)))
            self.sigma_inv[j] = torch.inverse(sigma_j)
            self.biases[j] = bias_sharedpart-0.5*torch.logdet(sigma_j)
            
    def predict(self, X):
        predicts_matrix=[]      
        for i in range(X.shape[0]):
            neg_distrances=[]
            for mu, sigma_inv, bias in zip(self.mu, self.sigma_inv, self.biases):
                neg_distrances.append(bias-0.5*(self.sharedpart+self.feature_dim)*torch.log(1.0+(1.0/self.sharedpart)*self.compute_distance(X[i, :],mu,sigma_inv)))
            predicts_matrix.append(torch.cat(neg_distrances, dim=1))
        predicts_matrix = torch.cat(predicts_matrix, dim=0)
        return predicts_matrix
    
    def compute_shared_part(self):
        self.lower_triu = torch.diag(torch.abs(self.triu_S_diag))+self.triu_S_lower*self.triu_S_lower_mask
        kappa_n = torch.abs(self.kappa) + 1e-6 + self.fix_Nj
        m_with_weight = torch.abs(self.kappa + 1e-6).div(kappa_n)*self.m
#         pdb.set_trace()
        xmean_weight = torch.div(self.fix_Nj, kappa_n)
        scatter_matrix = self_outer(self.lower_triu)
        sharedpart = torch.clamp(self.nu, min=self.feature_dim-1+1e-6)+self.fix_Nj-self.feature_dim+2
        bias_sharedpart = torch.lgamma(0.5*(sharedpart+self.feature_dim))-torch.lgamma(0.5*sharedpart)-0.5*self.feature_dim*torch.log(sharedpart)
        
        return kappa_n, m_with_weight, xmean_weight, scatter_matrix, sharedpart, bias_sharedpart
        
    def compute_distance(self, x, mu, sigma_inv):
        diff = x - mu
        gaussian_dist = torch.mm(torch.mm(diff, sigma_inv), diff.t())
        return gaussian_dist
    
class MetaQDA_MAP(nn.Module):
    def __init__(self, args):
        super(MetaQDA_MAP, self).__init__()
        self.x_dim = args.x_dim
        self.m = torch.nn.Parameter(torch.zeros(1, self.x_dim))
        self.kappa = torch.nn.Parameter(torch.tensor(0.1))
        self.nu = torch.nn.Parameter(torch.tensor(self.x_dim, dtype=torch.float32))
        self.triu_diag = torch.nn.Parameter(torch.ones(self.x_dim))
        self.triu_lower = torch.nn.Parameter(torch.eye(self.x_dim))
        self.triu_mask = torch.triu(torch.ones(self.x_dim, self.x_dim), diagonal=1).t().to(DEVICE)
    
    def fit_image_label(self, X, y):
        self.classes = np.unique(y.cpu())
        self.mu = [None for i in self.classes]
        self.sigma_inv = [None for i in self.classes]
        self.lower_triu = torch.diag(torch.abs(self.triu_diag)) + self.triu_lower * self.triu_mask
        
        kappa_ = torch.abs(self.kappa) + 1e-6
        nu_ = torch.clamp(self.nu, min=self.x_dim-1+1e-6)
        for j in self.classes:
            X_j = X[y==j]
            N_j = X_j.shape[0]
            self.mu[j] = kappa_.div(kappa_ + N_j)*self.m+torch.div(N_j, kappa_ + N_j)*torch.mean(X_j, dim=0, keepdim=True)
            sigma_part = self_outer(self.lower_triu) + sum_outer(X_j) + kappa_ * self_outer(self.m)-(kappa_ + N_j) * self_outer(self.mu[j])
            sigma_j = torch.div(sigma_part, nu_ + N_j + self.x_dim + 2)
            self.sigma_inv[j] = self.regularize(torch.inverse(sigma_j))          

    def predict(self, X):
        predicts_matrix=[]      
        for i in range(X.shape[0]):
            neg_distrances=[]
            for mu, sigma_inv in zip(self.mu, self.sigma_inv):
                diff = X[i, :] - mu
                gaussian_dist = torch.mm(torch.mm(diff, sigma_inv), diff.t())
                neg_distrances.append(-1*gaussian_dist)

            predicts_matrix.append(torch.cat(neg_distrances, dim=1))
        predicts_matrix = torch.cat(predicts_matrix, dim=0)
        return predicts_matrix
    
    def regularize(self, sigma):
        return (sigma + torch.eye(self.x_dim).to(DEVICE)).div(2)

    
class MetaQDA_FB(nn.Module):
    def __init__(self, args):
        super(MetaQDA_FB, self).__init__()
        self.reg_param = args.reg_param
        self.feature_dim = args.x_dim
        self.m = torch.nn.Parameter(torch.zeros(1, self.feature_dim))
        self.kappa = torch.nn.Parameter(torch.tensor(0.1))
        self.nu = torch.nn.Parameter(torch.tensor(self.feature_dim, dtype=torch.float32))
        self.triu_diag = torch.nn.Parameter(torch.ones(self.feature_dim))
        self.triu_lower = torch.nn.Parameter(torch.eye(self.feature_dim))
        self.triu_lower_mask = torch.triu(torch.ones(self.feature_dim, self.feature_dim), diagonal=1).t().to(DEVICE)
        
    def fit_image_label(self, X, y):
        self.classes = np.unique(y.cpu())
        self.mu = [None for i in self.classes]
        self.sigma_inv = [None for i in self.classes]
        self.biases = [None for i in self.classes]
        self.common_part = [None for i in self.classes]
        self.lower_triu = torch.diag(torch.abs(self.triu_diag)) + self.triu_lower * self.triu_lower_mask
        
        kappa_ = torch.abs(self.kappa) + 1e-6
        nu_ = torch.clamp(self.nu, min=self.feature_dim-1+1e-6)
        for j in self.classes:
            X_j = X[y==j]
            N_j = X_j.shape[0]
            self.mu[j] = kappa_.div(kappa_ + N_j)*self.m+torch.div(N_j, kappa_ + N_j)*torch.mean(X_j, dim=0, keepdim=True)
            sigma_j_no_scale = self_outer(self.lower_triu) + sum_outer(X_j) + kappa_ * self_outer(self.m)-(kappa_ + N_j) * self_outer(self.mu[j])
            sigma_j = torch.div(sigma_j_no_scale, (nu_ + N_j - self.feature_dim + 1)*(kappa_+N_j))*(kappa_+N_j+1)
            self.sigma_inv[j] = self.regularize(torch.inverse(sigma_j))          
            self.common_part[j] = nu_ + N_j + 1 - self.feature_dim
            self.biases[j] = torch.lgamma(0.5*(self.common_part[j]+self.feature_dim))-torch.lgamma(0.5*self.common_part[j])-0.5* self.feature_dim *torch.log(self.common_part[j])-0.5*torch.logdet(sigma_j)
           
    def predict(self, X):
        predicts_matrix=[]      
        for i in range(X.shape[0]):
            neg_distrances=[]
            for mu, sigma_inv, bias, common_part in zip(self.mu, self.sigma_inv, self.biases, self.common_part):
                neg_distrances.append(bias-0.5*(common_part+self.feature_dim)*torch.log(1.0+(1.0/common_part)*self.compute_distance(X[i, :],mu,sigma_inv)))
            predicts_matrix.append(torch.cat(neg_distrances, dim=1))
        predicts_matrix = torch.cat(predicts_matrix, dim=0)
        return predicts_matrix
    
    def compute_distance(self, x, mu, sigma_inv):
        diff = x - mu
        gaussian_dist = torch.mm(torch.mm(diff, sigma_inv), diff.t())
        return gaussian_dist
    
    def regularize(self, sigma):
        return (1-self.reg_param) * sigma + self.reg_param * torch.eye(self.feature_dim).to(DEVICE)   
    
### common basic function
def self_outer(x):
    dim1, dim2 = x.shape
    if dim1 >= dim2:
        return torch.mm(x, x.t())
    elif dim1 < dim2:
        return torch.mm(x.t(), x)
    
def mean_outer(x):
    feature_dim = x.shape[1]
    S_ = torch.zeros((feature_dim,feature_dim), dtype=torch.float32).cuda()
    for index, a_v in enumerate(x):
        S_ += torch.mm(a_v.unsqueeze(1), a_v.unsqueeze(0))
    
    return S_.div(index+1)

def sum_outer(x):
    feature_dim = x.shape[1]
    S_ = torch.zeros((feature_dim,feature_dim), dtype=torch.float32).cuda()
    for a_v in x:
        S_ += torch.mm(a_v.unsqueeze(1), a_v.unsqueeze(0))
    
    return S_

