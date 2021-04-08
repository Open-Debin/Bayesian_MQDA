from collections import OrderedDict
from .EpisodeMetadata import TrainEpisodeMetadata, TestEpisodeMetadata
import pdb


def get_eval_datasets(root, dataset_names, num=600):
  #eval_dataset_names = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']
  datasets = OrderedDict()
  for name in dataset_names:
    dataset = TestEpisodeMetadata(root, name, num)
    datasets[name] = dataset
  return datasets


def get_train_dataset(root, num=10000):
  # pdb.set_trace()
  # root: /data/QDA_FSL/Code_by_Others/urt/src
  # train-10000
  # 10000
  return TrainEpisodeMetadata(root, 'train-{:}'.format(num), num)
