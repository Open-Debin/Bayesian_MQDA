# Bayesian_MQDA
《Shallow Bayesian Meta Learning for Real World Few-shot Recognition》

Arxiv Version: https://arxiv.org/abs/2101.02833

### 0. Outline
#### Preparation
  - download data
  - download encoders
  - extract features
#### Experiments
  - (sd fsl) Single-Domain Few Shot Learning
  - (cd fsl) Cross-Domain Few Shot Testing
  - (md fsl) Multi-Domain Few Shot Learning: Meta-dataset
  - (fscil) Few Shot Class Incremental Learning
  - Compute Calibration Error
#### References
  - code references

## 1. Preparation
#### 1.1 download datasets
e.g., as for mini-Imagenet, please download [mini-Imagenet](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE) and put it at ```./data/mini``` and run ```proc_image.py``` to preprocess generate train/val/test datasets. (This process method is based on [maml](https://github.com/cbfinn/maml)).

For CUB_200_2011, please reference the instruction at ${Project_dir}/data/cub/source/readme.md

For Cars, please reference the instruction at ${Project_dir}/data/cars/source/readme.md

#### 1.2 download models
name format: ```$net_domain-$net_arch.pkl```. e.g. ```mini-conv4.pkl```.

network backbones: conv4, resnet18, wrn_28_20

[download link.](https://pan.baidu.com/s/1Jcoyo-xk70axNr4ETZ0s1A) passport: 5kxg

save models at ```$project_dir/encoder/model_parameters```

#### 1.3 extract features

```
python extract_features.py --encoder 'conv4/resnet18/wrn' --dataset_train 'mini/tiered/cifarfs' --dataset_inference 'mini/tiered/cifarfs/cub/cars'
```


## 2. Experiments
* ``` --lr ```: initial learning rate
* ``` --feature_or_logits ```: using features or logits from the encoder. 0 is features; 1 is logits

the following ``` "log name" ``` is created after the training process
#### 2.1 Single-Domain Few Shot Learning
e.g.: 5Way-5Shot, using encoder ```conv4``` on ```miniImagenet``` dataset, using ```MetaQDA_MAP``` version

```
python train.py --n_way 5 --k_spt 5 --net_arch conv4 --net_domain mini --strategy map
python test.py -l_n "log name"
```

#### 2.2 Cross-Domain Few Shot Testing
e.g.: testing trained models ``` "log name" ``` on ```cub``` dataset
```
python test.py -l_n "log name" -x_d cub
```
#### 2.3 Multi-Domain Few Shot Learning: Meta-dataset
1. Installation Meta-Dataset
Follow the the "User instructions" in the [Meta-Dataset repository](https://github.com/google-research/meta-dataset#user-instructions) for "Installation" and "Downloading and converting datasets".

2. Download features
download all the extracted features from [HERE](https://drive.google.com/drive/folders/1Z3gsa4TSSiH2wTZj1Jp5bD7UEKPOVzx5?usp=sharing) and put it in the ${Project_dir}/data/meta-dataset/

```
python train_urt_mqda.py
python test_urt_mqda.py -l_n "log name"
```
#### 2.4 Few-shot Class Incremental Learning
```
python train_mqda_incremental.py
python test_mqda_map_incremental.py -l_n "log name"
```
#### 2.5 Compute Calibration Error
```
python compute_ece.py -l_n "log name"
```

## 3. References
- dataloader reference: [RelationNet](https://github.com/floodsung/LearningToCompare_FSL/blob/master/miniimagenet/task_generator_test.py)
- cub, cars dataset reference: [CrossDomainFewShot](https://github.com/hytseng0509/CrossDomainFewShot/blob/master/filelists/process.py)
- metadataset train.py reference: [urt](https://github.com/liulu112601/URT/blob/master/fast-exps/urt-avg-head.py)
