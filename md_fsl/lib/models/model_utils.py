import os
import torch
import shutil
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import (MultiStepLR, ExponentialLR,
                                      CosineAnnealingWarmRestarts,
                                      CosineAnnealingLR)
from utils import check_dir, device
from paths import PROJECT_ROOT



def cosine_sim(embeds, prots):
    prots = prots.unsqueeze(0)
    embeds = embeds.unsqueeze(1)
    return F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30)



class CosineClassifier(nn.Module):
    def __init__(self, n_feat, num_classes):
        super(CosineClassifier, self).__init__()
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        weight = torch.FloatTensor(n_feat, num_classes).normal_(
                    0.0, np.sqrt(2.0 / num_classes))
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-12)
        weight = torch.nn.functional.normalize(self.weight, p=2, dim=0, eps=1e-12)
        cos_dist = x_norm @ weight
        scores = self.scale * cos_dist
        return scores

    def extra_repr(self):
        s = 'CosineClassifier: input_channels={}, num_classes={}; learned_scale: {}'.format(
            self.weight.shape[0], self.weight.shape[1], self.scale.item())
        return s


class CosineConv(nn.Module):
    def __init__(self, n_feat, num_classes, kernel_size=1):
        super(CosineConv, self).__init__()
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        weight = torch.FloatTensor(num_classes, n_feat, 1, 1).normal_(
                    0.0, np.sqrt(2.0/num_classes))
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        x_normalized = torch.nn.functional.normalize(
            x, p=2, dim=1, eps=1e-12)
        weight = torch.nn.functional.normalize(
            self.weight, p=2, dim=1, eps=1e-12)

        cos_dist = torch.nn.functional.conv2d(x_normalized, weight)
        scores = self.scale * cos_dist
        return scores

    def extra_repr(self):
        s = 'CosineConv: num_inputs={}, num_classes={}, kernel_size=1; scale_value: {}'.format(
            self.weight.shape[0], self.weight.shape[1], self.scale.item())
        return s
