import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random, math
 

# TODO: integrate the two functions into the following codes
def get_dotproduct_score(proto, cache, model):
  proto_emb   = model['linear_q'](proto)
  s_cache_emb = model['linear_k'](cache)
  raw_score   = F.cosine_similarity(proto_emb.unsqueeze(1), s_cache_emb.unsqueeze(0), dim=-1)
  return raw_score  


def get_mlp_score(proto, cache, model):
  n_proto, fea_dim = proto.shape
  n_cache, fea_dim = cache.shape
  raw_score = model['w']( model['nonlinear'](model['w1'](proto).view(n_proto, 1, fea_dim) + model['w2'](cache).view(1, n_cache, fea_dim) ) )
  return raw_score.squeeze(-1)


# this model does not need query, only key and value
class MultiHeadURT_value(nn.Module):
  def __init__(self, fea_dim, hid_dim, temp=1, n_head=1):
    super(MultiHeadURT_value, self).__init__()
    self.w1 = nn.Linear(fea_dim, hid_dim)
    self.w2 = nn.Linear(hid_dim, n_head)
    self.temp     = temp

  def forward(self, cat_proto):
    # cat_proto n_class*8*512
    n_class, n_extractors, fea_dim = cat_proto.shape
    raw_score = self.w2(self.w1(cat_proto)) # n_class*8*n_head 
    score   = F.softmax(self.temp * raw_score, dim=1)
    return score


class URTPropagation(nn.Module):

  def __init__(self, key_dim, query_dim, hid_dim, temp=1, att="cosine"):
    super(URTPropagation, self).__init__()
    self.linear_q = nn.Linear(query_dim, hid_dim, bias=True)
    self.linear_k = nn.Linear(key_dim, hid_dim, bias=True)
    #self.linear_v_w = nn.Parameter(torch.rand(8, key_dim, key_dim))
    self.linear_v_w = nn.Parameter( torch.eye(key_dim).unsqueeze(0).repeat(8,1,1)) 
    self.temp     = temp
    self.att      = att
    # how different the init is
    for m in self.modules():
      if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)

  def forward_transform(self, samples):
    bs, n_extractors, fea_dim = samples.shape
    '''
    if self.training:
      w_trans = torch.nn.functional.gumbel_softmax(self.linear_v_w, tau=10, hard=True)
    else:
      # y_soft = torch.softmax(self.linear_v_w, -1)
      # index = y_soft.max(-1, keepdim=True)[1]
      index = self.linear_v_w.max(-1, keepdim=True)[1]
      y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
      w_trans = y_hard
      # w_trans = y_hard - y_soft.detach() + y_soft
    '''
    w_trans = self.linear_v_w 
    # compute regularization
    regularization = w_trans @ torch.transpose(w_trans, 1, 2)
    samples = samples.view(bs, n_extractors, fea_dim, 1)
    w_trans = w_trans.view(1, 8, fea_dim, fea_dim)
    return torch.matmul(w_trans, samples).view(bs, n_extractors, fea_dim), (regularization**2).sum()

  def forward(self, cat_proto):
    # cat_proto n_class*8*512 
    # return: n_class*8
    n_class, n_extractors, fea_dim = cat_proto.shape
    q       = cat_proto.view(n_class, -1) # n_class * 8_512
    k       = cat_proto                   # n_class * 8 * 512
    q_emb   = self.linear_q(q)            # n_class * hid_dim
    k_emb   = self.linear_k(k)            # n_class * 8 * hid_dim  | 8 * hid_dim
    if self.att == "cosine":
      raw_score   = F.cosine_similarity(q_emb.view(n_class, 1, -1), k_emb.view(n_class, n_extractors, -1), dim=-1)
    elif self.att == "dotproduct":
      raw_score   = torch.sum( q_emb.view(n_class, 1, -1) * k_emb.view(n_class, n_extractors, -1), dim=-1 ) / (math.sqrt(fea_dim)) 
    else:
      raise ValueError('invalid att type : {:}'.format(self.att))
    score   = F.softmax(self.temp * raw_score, dim=1)
    return score


class MultiHeadURT(nn.Module):
  def __init__(self, key_dim, query_dim, hid_dim, temp=1, att="cosine", n_head=1):
    super(MultiHeadURT, self).__init__()
    layers = []
    for _ in range(n_head):
      layer = URTPropagation(key_dim, query_dim, hid_dim, temp, att)
      layers.append(layer)
    self.layers = nn.ModuleList(layers)

  def forward(self, cat_proto):
    score_lst = []
    for i, layer in enumerate(self.layers):
      score = layer(cat_proto)
      score_lst.append(score)
    # n_class * n_extractor * n_head
    return torch.stack(score_lst, dim=-1)

def get_lambda_urt_sample(context_features, context_labels, target_features, num_labels, model, normalize=True):
  if normalize:
    context_features = F.normalize(context_features, dim=-1)
    target_features  = F.normalize(target_features, dim=-1)
  score_context, urt_context = model(context_features)
  score_target, urt_target = model(target_features)
  proto_list  = []
  for label in range(num_labels):
    proto = urt_context[context_labels == label].mean(dim=0)
    proto_list.append(proto)
  urt_proto = torch.stack(proto_list)
  # n_samples*8*512
  return score_context, urt_proto, score_target, urt_target

def get_lambda_urt_avg(context_features, context_labels, num_labels, model, normalize=True):
  if normalize:
    context_features = F.normalize(context_features, dim=-1)
  proto_list  = []
  for label in range(num_labels):
    proto = context_features[context_labels == label].mean(dim=0) # [496, 8, 512] => [32, 8, 512] => [8,512]
    proto_list.append(proto)
  proto = torch.stack(proto_list) # [8, 512] => [num_labels, 8, 512]
  # n_class*8*512
  # @q: why different class mean
  score_proto  = model(proto) # [num_labels, 8, feature_dim] => [num_labels, 8, n_head] 
  # n_extractors * n_head
  return torch.mean(score_proto, dim=0) # [num_labels, 8, n_head] => [8, n_head]

def apply_urt_avg_selection(context_features, selection_params, normalize, value="sum", transform=None):
  selection_params = torch.transpose(selection_params, 0, 1) # n_head * 8
  n_samples, n_extractors, fea_dim = context_features.shape
  urt_fea_lst = []
  if normalize:
    context_features = F.normalize(context_features, dim=-1)
  regularization_losses = []
  for i, params in enumerate(selection_params):
    # class-wise lambda 
    if transform:
      trans_features, reg_loss = transform.module.layers[i].forward_transform(context_features)
      regularization_losses.append(reg_loss)
    else:
      trans_features = context_features
    if value == "sum":
      urt_features = torch.sum(params.view(1,n_extractors,1) * trans_features, dim=1) # n_sample * 512 : attention=wetigh mean of feature encoder 
    elif value == "cat":
      urt_features = params.view(1,n_extractors,1) * trans_features  # n_sample * 8 * 512
    urt_fea_lst.append(urt_features)
  if len(regularization_losses) == 0:
    return torch.stack( urt_fea_lst, dim=1 ).view(n_samples, -1) # n_sample * (n_head * 512) or n_sample * (n_head * 8 * 512)
  else:
    return torch.stack( urt_fea_lst, dim=1 ).view(n_samples, -1), sum(regularization_losses)


def apply_urt_selection(context_features, context_labels, selection_params, normalize):
  # class-wise lambda 
  if normalize:
    context_features = F.normalize(context_features, dim=-1)
  lambda_lst = []
  for lab in context_labels:
    lambda_lst.append(selection_params[lab])
  lambda_tensor = torch.stack(lambda_lst, dim=0)
  n_sample, n_extractors = lambda_tensor.shape
  urt_features = torch.sum(lambda_tensor.view(n_sample, n_extractors, 1) * context_features, dim=1) 
  return urt_features

class PropagationLayer(nn.Module):

  def __init__(self, input_dim=512, hid_dim=128, temp=1, transform=False):
    super(PropagationLayer, self).__init__()
    self.linear_q = nn.Linear(input_dim, hid_dim, bias=False)
    self.linear_k = nn.Linear(input_dim, hid_dim, bias=False)
    self.temp     = temp
    if transform:
      self.transform = nn.Linear(input_dim, input_dim)

    for m in self.modules():
      if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)

  def forward(self, proto, s_cache, data2nclss, use_topk):
    if 'transform' in self.__dict__:
      proto   = self.transform(proto)
      s_cache = self.transform(s_cache)
    proto_emb   = self.linear_q(proto)
    s_cache_emb = self.linear_k(s_cache)
    raw_score   = F.cosine_similarity(proto_emb.unsqueeze(1), s_cache_emb.unsqueeze(0), dim=-1)
    score       = F.softmax(self.temp * raw_score, dim=1)
    prop_proto  = torch.matmul( score, s_cache )  # n_class * n_cache  @ n_cache * n_dim
    if random.random() > 0.99:
      print("top_1_idx: {} in {} cache".format(torch.topk(raw_score, 1)[1], len(s_cache)))
      print("score: {}".format(score))
      print("mean:{}, var:{}, min:{}, max:{}".format(torch.mean(score, dim=1).data, torch.var(score, dim=1).data, torch.min(score, dim=1)[0].data, torch.max(score, dim=1)[0].data))
    return raw_score, prop_proto


class MultiHeadPropagationLayer(nn.Module):

  def __init__(self, input_dim, hid_dim, temp, transform, n_head):
    super(MultiHeadPropagationLayer, self).__init__()
    layers = []
    for _ in range(n_head):
      layer = PropagationLayer(input_dim, hid_dim, temp, transform)
      layers.append(layer)
    self.layers = nn.ModuleList(layers)

  def forward(self, proto, s_cache, data2nclss, use_topk):
    raw_score_lst, prop_proto_lst = [], []
    for i, layer in enumerate(self.layers):
      raw_score, prop_proto = layer(proto, s_cache, data2nclss, use_topk)
      if torch.isnan(raw_score).any() or torch.isnan(prop_proto).any(): import pdb; pdb.set_trace()
      raw_score_lst.append(raw_score)
      prop_proto_lst.append(prop_proto)
    return torch.stack(raw_score_lst, dim=0).mean(0), torch.stack(prop_proto_lst, dim=0).mean(0)

def get_prototypes(features, labels, num_labels, model, cache):
  proto_list  = []
  for label in range(num_labels):
    proto = features[labels == label].mean(dim=0)
    proto_list.append(proto)
  proto = torch.stack(proto_list)
  num_devices = torch.cuda.device_count()
  num_slots, feature_dim = cache.shape
  cache_for_parallel = cache.view(1, num_slots, feature_dim).expand(num_devices, num_slots, feature_dim)
  raw_score, prop_proto = model(proto, cache_for_parallel)
  return raw_score, proto, prop_proto
