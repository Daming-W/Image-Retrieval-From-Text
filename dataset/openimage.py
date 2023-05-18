import csv
import os
import os.path
import tarfile
#from urlparse.request import urlparse
from urllib import parse
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import dataset.util
from dataset.util import *

class openimage_dataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform=None, target_transform=None,class_num:int = None):
        self.root = root
        with open(annFile, 'r') as f:
            names = f.readlines()
        #name = names.strip('\n').split(' ')
        self.name = names
        #self.label = name[:,1]
        self.transform = transform
        self.class_num = class_num
        self.target_transform = target_transform
        print('load class_nums = ',self.class_num)

    def __getitem__(self, index):
        name = self.name[index]
        path = name.strip('\n').split(',')[0]
        num = name.strip('\n').split(',')[1]
        num = num.strip(' ').split(' ')
        num = np.array([int(i) for i in num])
    #    print('load class_nums = ',self.class_num)
        label = np.zeros([self.class_num])
        label[num] = 1
        label = torch.tensor(label, dtype=torch.float)
        if os.path.exists(os.path.join(self.root, path))==False:
            label = np.zeros([self.class_num])
            label = torch.tensor(label, dtype=torch.long)
            img = np.zeros((448,448,3))
            img = Image.fromarray(np.uint8(img))
        else:
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.name)