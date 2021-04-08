import os, sys, copy, torch
import collections

from typing import List, Dict, Text
from .resnet18 import resnet18
from .resnet18_film import resnet18 as resnet18_film

MODEL_DICT = {'resnet18': resnet18,
              'resnet18_pnf': resnet18_film}


class CheckPointer(object):
  def __init__(self, model_name, model=None, optimizer=None):
    self.model = model
    self.optimizer = optimizer
    self.model_name = model_name
    TORCH_HOME = 'TORCH_HOME'
    if TORCH_HOME in os.environ:
      TORCH_HOME = os.environ[TORCH_HOME]
    else:
      TORCH_HOME = os.path.join(os.environ['HOME'], '.torch')
    self.model_path = os.path.join(TORCH_HOME, 'sur-weights', model_name)
    self.last_ckpt = os.path.join(self.model_path, 'checkpoint.pth.tar')
    self.best_ckpt = os.path.join(self.model_path, 'model_best.pth.tar')

  def restore_model(self, ckpt='last', model=True, optimizer=True, strict=True):
    if not os.path.exists(self.model_path):
      assert False, "Model is not found at {}".format(self.model_path)
    self.last_ckpt = os.path.join(self.model_path, 'checkpoint.pth.tar')
    self.best_ckpt = os.path.join(self.model_path, 'model_best.pth.tar')
    ckpt_path = self.last_ckpt if ckpt == 'last' else self.best_ckpt

    if os.path.isfile(ckpt_path):
      print("=> loading {} checkpoint '{}'".format(ckpt, ckpt_path))
      ch = torch.load(ckpt_path, map_location='cpu')
      if self.model is not None and model:
        self.model.load_state_dict(ch['state_dict'], strict=strict)
      if self.optimizer is not None and optimizer:
        self.optimizer.load_state_dict(ch['optimizer'])
    else:
      assert False, "No checkpoint! %s" % ckpt_path

    return ch.get('epoch', None), ch.get('best_val_loss', None), ch.get('best_val_acc', None)


def get_extractors(trainsets: List[Text],
                   dataset_models: Dict[Text, Text],
                   backbone: Text, classifier: Text, bn_train_mode: bool, use_cuda: bool = True):
  extractors = collections.OrderedDict()
  for dataset_name in trainsets:
    if dataset_name not in dataset_models:
      continue
    if dataset_name == 'ilsvrc_2012':
      extractor = MODEL_DICT['resnet18'](classifier=classifier, num_classes=None, global_pool=False, dropout=0.0)
    else:
      extractor = MODEL_DICT[backbone](classifier=classifier, num_classes=None, global_pool=False, dropout=0.0)
    extractor.train(bn_train_mode)
    print('Create {:}\'s network with BN={:}'.format(dataset_models[dataset_name], bn_train_mode))
    if backbone == 'resnet18_pnf' and dataset_name != 'ilsvrc_2012':
      weights = copy.deepcopy(extractors['ilsvrc_2012'].module.state_dict())
      extractor.load_state_dict(weights, strict=False)
    checkpointer = CheckPointer(dataset_models[dataset_name], extractor, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)
    if use_cuda:
      extractor = extractor.cuda()
    extractors[dataset_name] = torch.nn.DataParallel(extractor)
  return extractors


def extract_features(extractors, images):
  all_features = []
  for name, extractor in extractors.items():
    features = extractor(images)
    all_features.append(features)
  return torch.stack(all_features, dim=1) # batch x #extractors x #features
