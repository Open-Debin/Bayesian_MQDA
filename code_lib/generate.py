# code is based on https://github.com/katerakelly/pytorch-maml
import os
import pdb
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torchvision
import torchvision.transforms as transforms
from code_lib import vision, util

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def folders(*path2images):
    folders = []
    for path_item in path2images:
        folders.extend([os.path.join(path_item, label) \
                    for label in os.listdir(path_item) \
                    if os.path.isdir(os.path.join(path_item, label)) \
                    ])
     # CUB    
    random.seed(1)
    random.shuffle(folders)

    return folders

class Task(object):
    def __init__(self, character_folders, num_classes, support_num, query_num):

        self.character_folders = character_folders
        self.num_classes = num_classes
        self.support_num = support_num
        self.query_num = query_num

        class_folders = random.sample(self.character_folders, self.num_classes) # select Way of file from folders
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))   # {'./data/test/n01930112': 7, ..., }
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:

            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.train_roots += samples[c][:support_num]                  # Support Set
            self.test_roots += samples[c][support_num:support_num+query_num] # Query Set
            
        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        # pdb.set_trace()
        return os.path.dirname(sample)
#         return os.path.join(*sample.split('/')[:-1])

class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, transform2 = None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.transform2 = transform2
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'support' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'support' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class FewShotFeatureset(Dataset):

    def __init__(self, task, split='support'):
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'support' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'support' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")        

class MiniImagenet(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(MiniImagenet, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        if self.transform is not None:
            image1 = self.transform(image)
        if self.transform2 is not None:
            image2 = self.transform2(image)
            image1 = torch.cat([image1.unsqueeze(0), image2], dim=0)
            
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image1, label#, image_root

class ImageLabelPair(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(ImageLabelPair, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        label = self.labels[idx]
        image = Image.open(image_root).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
    
    
class NameLabelPair(FewShotFeatureset):

    def __init__(self, *args, **kwargs):
        super(NameLabelPair, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        label = self.labels[idx]
        return image_root, label


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_cl, num_inst,shuffle=True):

        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batches = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
        else:
            batches = [[i+j*self.num_inst for i in range(self.num_inst)] for j in range(self.num_cl)]
        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]

        if self.shuffle:
            random.shuffle(batches)
            for sublist in batches:
                   random.shuffle(sublist)
        batches = [item for sublist in batches for item in sublist]
        batches.sort()  ### the sort is the key for [0 1 2 0 1 2 0 1 2] => [0 0 0 1 1 1 2 2 2]
#         pdb.set_trace()
        return iter(batches)

    def __len__(self):
        return 1

def get_mini_imagenet_data_loader(task, num_per_class=1, split='train',shuffle = False):
#     normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])

#     dataset = MiniImagenet(task,split=split,transform=transforms.Compose([transforms.ToTensor(),normalize]))

    if split == 'train':
        dataset = MiniImagenet(task,split=split,
#                                transform2 = util.meta_trainval_transform,
                               transform = vision.test_transform #util.test_transform
#                                transform=util.TransformLoader(224).get_composed_transform()
#                                transform= util.transform05
                          )
        sampler = ClassBalancedSamplerOld(num_per_class,task.num_classes, task.support_num,shuffle=True)

    else:
        dataset = MiniImagenet(task,split=split,
                               transform = vision.test_transform #meta_trainval_transform
#                                transform=util.TransformLoader(224).get_composed_transform()
#                                transform= util.transform05
                          )
        sampler = ClassBalancedSampler(task.num_classes, task.query_num, shuffle=shuffle)

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler, pin_memory=True)
    # loader = DataLoader(dataset, batch_size=15, sampler=sampler, pin_memory=True)
    return loader

def image_label(task, split='', transform=''):
    if split == 'support':
        dataset = ImageLabelPair(task,split=split, transform = transform)
        sampler = ClassBalancedSampler(task.num_classes, task.support_num, shuffle=True)
        loader = DataLoader(dataset, batch_size=task.support_num*task.num_classes, sampler=sampler, pin_memory=True)
    elif split == 'query':
        dataset = ImageLabelPair(task,split=split, transform = transform)
        sampler = ClassBalancedSampler(task.num_classes, task.query_num, shuffle=False)
        loader = DataLoader(dataset, batch_size=task.query_num*task.num_classes, sampler=sampler, pin_memory=True)
    return loader

def name_label(task, split=''):
    dataset = NameLabelPair(task,split=split)
    if split == 'support':
        sampler = ClassBalancedSampler(task.num_classes, task.support_num, shuffle=False)
#         sampler = ClassBalancedSamplerOld(task.support_num, task.num_classes, task.support_num, shuffle=False)
        loader = DataLoader(dataset, batch_size=task.support_num*task.num_classes, sampler=sampler, pin_memory=True)
    elif split == 'query':
        sampler = ClassBalancedSampler(task.num_classes, task.query_num, shuffle=False)
        loader = DataLoader(dataset, batch_size=task.query_num*task.num_classes, sampler=sampler, pin_memory=True)
    return loader



class ImageNameDataset(Dataset):

    def __init__(self, image_roots):
        self.image_roots = [os.path.join(image_roots, x) for x in os.listdir(image_roots)]

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image_name = self.image_roots[idx]
        
        return image_name


class ImageDataset(Dataset):

    def __init__(self, image_roots, transform=None):
        self.transform = transform # Torch operations on the input image
        image_names = os.listdir(image_roots)
        self.image_roots = [os.path.join(image_roots,name) for name in image_names]

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]   
        
        image = Image.open(image_root)
        image = image.convert('RGB')
        if self.transform is not None:
            image1 = self.transform(image)
            
        return image1, image_root

    
class ImageFileDataset(Dataset):

    def __init__(self, image_roots, file_path, transform=None):
        self.transform = transform # Torch operations on the input image
        with open(file_path,'r') as rfile:
            self.image_roots = [os.path.join(image_roots,name.strip().replace('MINI-ImageNet','mini')) for item in rfile]

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]   
        
        image = Image.open(image_root)
        image = image.convert('RGB')
        if self.transform is not None:
            image1 = self.transform(image)
            
        return image1, image_root
    
    
def data_list(data_root):
    data_list = []
    for root, dirs, files in os.walk(data_root):
        for name in files:
            full_name = os.path.join(root,name)
            data_list.append(full_name)
    return data_list

class Dataset_for_SD_CD(Dataset):

    def __init__(self, image_roots, transform=None):
        self.transform = transform # Torch operations on the input image
        self.image_roots = image_roots

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        if self.transform is not None:
            image1 = self.transform(image)
            
        return image1, image_root
    
class FeatureDictMaker:
    def __init__(self, model, head_root):
        self.feature_list = []
        self.logits_list=[]
        self.name_list = [] 
        self.model = model
        self.model.eval()
        self.head_root = head_root
        
    def _extract_features(self, dataloader):
        with torch.no_grad():
            for img, name in dataloader:
                fea, logits = self.model(img.to('cuda'), True)
                self.fea_list.append(fea.cpu())
                self.logits_list.append(logits.cpu())
                self.name_list=self.name_list+name 
                
    def mean_features(self,):
        return torch.cat(self.fea_list, dim = 0).mean(dim=0)
    
    def mean_logits(self,):
        return torch.cat(self.logits_list, dim = 0).mean(dim=0) 
    
    def make_dict(self):    
        fea_tensor = torch.cat(self.fea_list, dim = 0)
        logits_tensor = torch.cat(self.logits_list, dim = 0)
        fea_dict = collections.defaultdict(list)
        logits_dict = collections.defaultdict(list)
        for item_f, item_l, label in zip(fea_tensor,logits_tensor, self.name_list):
            fea_dict[label.replace(self.head_root,'')].append(item_f)
            logits_dict[label.replace(self.head_root,'')].append(item_l)

        return fea_dict, logits_dict
    


def whole_fit_fake(folder_fake_close_set, nj=256):
    if len(folder_fake_close_set) != 4:
        raise ValueError("Big Error")
    
    data_path0 = load_data(folder_fake_close_set[0], nj, vision.wrn28_transform_cifar10)
    data_path1 = load_data(folder_fake_close_set[1], nj, vision.wrn28_transform_cifar10)
    data_path2 = load_data(folder_fake_close_set[2], nj, vision.wrn28_transform_cifar10)
    data_path3 = load_data(folder_fake_close_set[3], nj, vision.wrn28_transform_cifar10)
    
    return data_path0, data_path1, data_path2, data_path3
    
def whole_fit_true(folder_close_set, nj):
    if len(folder_close_set) != 6:
        raise ValueError("Big Error")
    
    data_path0 = load_data(folder_close_set[0], nj, vision.wrn28_transform_cifar10)
    data_path1 = load_data(folder_close_set[1], nj, vision.wrn28_transform_cifar10)
    data_path2 = load_data(folder_close_set[2], nj, vision.wrn28_transform_cifar10)
    data_path3 = load_data(folder_close_set[3], nj, vision.wrn28_transform_cifar10)   
    data_path4 = load_data(folder_close_set[4], nj, vision.wrn28_transform_cifar10)
    data_path5 = load_data(folder_close_set[5], nj, vision.wrn28_transform_cifar10)
    
    return data_path0, data_path1, data_path2, data_path3, data_path4, data_path5