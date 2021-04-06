# code is based on https://github.com/katerakelly/pytorch-maml
import pdb
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from basic_code import util


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

def folders(train_folder=''):
    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
     # CUB    
    random.seed(1)
    random.shuffle(metatrain_folders)

    return metatrain_folders

class DataLoadTask(object):

    def __init__(self, character_folders, num_classes, support_num, query_num):

        self.character_folders = character_folders
        self.num_classes = num_classes
        self.support_num = support_num
        self.query_num = query_num

        class_folders = random.sample(self.character_folders,self.num_classes) # select Way of file from folders
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
        return os.path.join(*sample.split('/')[:-1])

class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, transform2 = None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.transform2 = transform2
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

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

class MiniImagenetName(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(MiniImagenetName, self).__init__(*args, **kwargs)

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
        return iter(batches)

    def __len__(self):
        return 1

class ClassBalancedSamplerOld(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]

        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_img_loader(task, num_per_class=1, split='train',shuffle = False):
#     normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])

#     dataset = MiniImagenet(task,split=split,transform=transforms.Compose([transforms.ToTensor(),normalize]))

    if split == 'train':
        dataset = MiniImagenet(task,split=split,
#                                transform2 = util.meta_trainval_transform,
                               transform = util.test_transform #util.test_transform
#                                transform=util.TransformLoader(224).get_composed_transform()
#                                transform= util.transform05
                          )
        sampler = ClassBalancedSamplerOld(num_per_class,task.num_classes, task.support_num,shuffle=True)

    else:
        dataset = MiniImagenet(task,split=split,
                               transform = util.test_transform #meta_trainval_transform
#                                transform=util.TransformLoader(224).get_composed_transform()
#                                transform= util.transform05
                          )
        sampler = ClassBalancedSampler(task.num_classes, task.query_num, shuffle=shuffle)

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler, pin_memory=True)
    # loader = DataLoader(dataset, batch_size=15, sampler=sampler, pin_memory=True)
    return loader

def get_name_label(task, num_per_class=1, split='', shuffle=False):

    if split == 'train':
        dataset = MiniImagenetName(task,split=split)
        sampler = ClassBalancedSamplerOld(num_per_class,task.num_classes, task.support_num,shuffle=True)

    elif split == 'test':
        dataset = MiniImagenetName(task,split=split)
        sampler = ClassBalancedSampler(task.num_classes, task.query_num, shuffle=shuffle)

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler, pin_memory=True)
    return loader



class RBE_dataset(Dataset):
    def __init__(self, data_dir, transform=None,head_root=''):
        self.data_dir=data_dir
        self.imgs_list=os.listdir(data_dir)
        self.transform=transform
        self.head_root=head_root

    def __getitem__(self, index):

        image_path = os.path.join(self.data_dir,self.imgs_list[index])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image_t = self.transform(image)

        return image_t, image_path.replace(self.head_root,'')

    def __len__(self):
        return len(self.imgs_list)

#