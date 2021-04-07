# Bayesian_MQDA
《Shallow Bayesian Meta Learning for Real World Few-shot Recognition》
Arxiv Version: https://arxiv.org/abs/2101.02833

## 0. Outline
#### Preparation
  - download data
  - download models
  - extract features
#### Experiments
  - Single Domain Few Shot Learning
  - Cross Domain Few Shot Testing
  - Multi Domain Few Shot Learning: Meta-dataset
  - Few-shot Class Incremental Learning

## 1. Preparation
#### 1.1 download data
dropbox link
#### 1.2 download models
dropbox link
#### 1.3 extract features

## 2. Experiments
#### 2.1 Single Domain Few Shot Learning
'''e.g.: {5Way} {5Shot}, using encoder {conv4} on {mini(ImageNet)} dataset, using MetaQDA {MAP} version'''

python train.py --n_way 5 --k_spt 5 --net_domain mini --net_arch conv4 --strategy map

python test.py -l_n log_name

#### 2.2 Cross Domain Few Shot Learning
''' e.g.: testing trained models {log_name} on {cub} dataset''' 

python test.py -l_n log_name -x_d cub

#### 2.3 Multi Domain Few Shot Learning: Meta-dataset
python train_urt_mqda.py

python test_urt_mqda.py -l_n log_name

#### 2.4 Few Shot Class Incremental Learning

