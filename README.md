
### Shallow Bayesian Meta Learning for Real-World Few-Shot Recognition
Arxiv Version: https://arxiv.org/abs/2101.02833

### 0. Outline
#### Preparation
  - download data
  - download models
  - extract features
#### Experiments
  - Instruction
  - Single-Domain Few Shot Learning
  - Cross-Domain Few Shot Testing
  - Multi-Domain Few Shot Learning: Meta-dataset
  - Few Shot Class Incremental Learning

## 1. Preparation
#### 1.1 download data
dropbox link
#### 1.2 download models
dropbox link
#### 1.3 extract features

## 2. Experiments
#### 2.1 Instructions
* ``` --lr ```: initial learning rate
* ``` --feature_or_logits ```: using features or logits from the encoder. 0 is features; 1 is logits
* etc.
the following log_name is created after the training process
#### 2.2 Single-Domain Few Shot Learning
'''e.g.: {5Way} {5Shot}, using encoder {conv4} on {mini(ImageNet)} dataset, using MetaQDA {MAP} version'''

```
python train.py --n_way 5 --k_spt 5 --net_domain mini --net_arch conv4 --strategy map
python test.py -l_n log_name
```

#### 2.3 Cross-Domain Few Shot Testing
''' e.g.: testing trained models {log_name} on {cub} dataset''' 
```
python test.py -l_n log_name -x_d cub
```
#### 2.4 Multi-Domain Few Shot Learning: Meta-dataset
```
python train_urt_mqda.py
python test_urt_mqda.py -l_n log_name
```
#### 2.5 Few-shot Class Incremental Learning
```
python train_mqda_incremental.py
python test_mqda_map_incremental.py -l_n log_name
```
