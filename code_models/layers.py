import pdb
import numpy as np
import torch
import torch.nn.functional as F
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def l2norm(x):
    x = F.normalize(x, p=2, dim=1)
    return x

def centre_l2norm(x, centre):
    x = F.normalize(x-centre, p=2, dim=1)
    return x

def inverted_dropout(X, keep_prob):
    X=X.float()
    assert 0 <= keep_prob  <= 1
    if keep_prob == 0:
        return torch.zeros_like(X)
    
    mask = (torch.rand(X.shape) <= keep_prob).float().cuda()
    return mask * X / keep_prob

def x_dict2mean_outer(x_dict):
    x_centre = x_dict['trCentre']
    feature_dim = x_centre.size
    S_ = np.zeros((feature_dim,feature_dim))
    num = 0
    for key_str, x in x_dict.items():
        if 'test' in key_str:
            continue
        num+=1
        S_ += np.outer(x, x)

    return torch.tensor(S_/num).to(DEVICE)


                
                
            
            
        