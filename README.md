# Bayesian_MQDA
《Shallow Bayesian Meta Learning for Real World Few-shot Recognition》

Arxiv Version: https://arxiv.org/abs/2101.02833

### 0. Outline
#### Preparation
  - download data
  - download encoders
  - extract features
#### Experiments
  - Single-Domain Few Shot Learning
  - Cross-Domain Few Shot Testing
  - Multi-Domain Few Shot Learning: Meta-dataset
  - Few Shot Class Incremental Learning

## 1. Preparation
#### 1.1 download data
E.g., as for mini-Imagenet, please download [mini-Imagenet](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE) and put it in ./data/mini and run proc_image.py to preprocess generate train/val/test datasets. (This process method is based on [maml](https://github.com/cbfinn/maml)).

#### 1.2 download models
name format: ```$net_domain-$net_arch.pkl```. for example: ```mini-conv.pkl```.
We provide networks like: conv4, resnet18, wrn_28_20

[download link.]

please put the models at '''./project_dir/encoder/pretrained_models```

#### 1.3 extract features

```
python extract_features.py --encoder 'encoder name' --dataset 'dataset name'
```


## 2. Experiments
* ``` --lr ```: initial learning rate
* ``` --feature_or_logits ```: using features or logits from the encoder. 0 is features; 1 is logits

the following log_name is created after the training process
#### 2.1 Single-Domain Few Shot Learning
e.g.: 5Way-5Shot, using encoder ```conv4``` on ```miniImagenet``` dataset, using ```MetaQDA_MAP``` version

```
python train.py --n_way 5 --k_spt 5 --net_domain mini --net_arch conv4 --strategy map
python test.py -l_n log_name
```

#### 2.2 Cross-Domain Few Shot Testing
e.g.: testing trained models ```$log_name``` on ```cub``` dataset
```
python test.py -l_n log_name -x_d cub
```
#### 2.3 Multi-Domain Few Shot Learning: Meta-dataset
```
python train_urt_mqda.py
python test_urt_mqda.py -l_n log_name
```
#### 2.4 Few-shot Class Incremental Learning
```
python train_mqda_incremental.py
python test_mqda_map_incremental.py -l_n log_name
```

