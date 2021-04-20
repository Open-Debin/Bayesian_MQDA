import os
import pdb
import json
import shutil
import random
import numpy as np
from scipy.io import loadmat
from os.path import join#isfile, isdir, join

data_path = './source/cars_train'
savedir = './'
dataset_list = ['base','val','novel']

data_list = np.array(loadmat('source/devkit/cars_train_annos.mat')['annotations'][0])
class_list = np.array(loadmat('source/devkit/cars_meta.mat')['class_names'][0])
classfile_list_all = [[] for i in range(len(class_list))]

for i in range(len(data_list)):
  folder_path = join(data_path, data_list[i][-1][0])
  classfile_list_all[data_list[i][-2][0][0] - 1].append(folder_path)

for i in range(len(classfile_list_all)):
  random.shuffle(classfile_list_all[i])

'''folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])'''

for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'base' in dataset:
            if (i%2 == 0):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset:
            if (i%4 == 1):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'novel' in dataset:
            if (i%4 == 3):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item[0]  for item in class_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)

    
filename = 'base.json'
target_root = './train'
with open(filename,'r') as f_obj:
    file_json = json.load(f_obj)

image_names = file_json['image_names']
image_labels = file_json['image_labels']

for img_old_dir, label in zip(image_names,image_labels):
    img_new_dir = os.path.join(target_root, str(label),os.path.basename(img_old_dir))
    img_new_root = os.path.dirname(img_new_dir)
    if not os.path.exists(img_new_root):
        os.makedirs(img_new_root)
#     pdb.set_trace()
    shutil.copy(img_old_dir,img_new_dir)


filename = 'val.json'
target_root = './val'
with open(filename,'r') as f_obj:
    file_json = json.load(f_obj)

image_names = file_json['image_names']
image_labels = file_json['image_labels']

for img_old_dir, label in zip(image_names,image_labels):
    img_new_dir = os.path.join(target_root, str(label),os.path.basename(img_old_dir))
    img_new_root = os.path.dirname(img_new_dir)
    if not os.path.exists(img_new_root):
        os.makedirs(img_new_root)
#     pdb.set_trace()
    shutil.copy(img_old_dir,img_new_dir)    
    
    
filename = 'novel.json'
target_root = './test'
with open(filename,'r') as f_obj:
    file_json = json.load(f_obj)

image_names = file_json['image_names']
image_labels = file_json['image_labels']

for img_old_dir, label in zip(image_names,image_labels):
    img_new_dir = os.path.join(target_root, str(label),os.path.basename(img_old_dir))
    img_new_root = os.path.dirname(img_new_dir)
    if not os.path.exists(img_new_root):
        os.makedirs(img_new_root)
#     pdb.set_trace()
    shutil.copy(img_old_dir,img_new_dir)