import os
import pickle
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def load_data(dataset_path, file_name , preprocess):
    """Load CIFAR-10 data from given dataset_path"""
    file_path = os.path.join(dataset_path, file_name)
    with open(file_path, 'rb') as f:
        # load data with pickle
        entry = pickle.load(f, encoding='latin1')
        images_array = entry['data']
        labels = entry['labels']
        # reshape the array to 32x32 color image
        images_array = images_array.reshape(-1, 3, 32, 32)  # '-1' means the value will be deduced from the shape of array
        images =[]
        # loop array by [0] which is the data idx
        for i in range(images_array.shape[0]):
            # array->img and do preprocess
            image_array = images_array[i].transpose(1, 2, 0)
            image = Image.fromarray(image_array)
            image = preprocess(image.convert('RGB'))
            images.append(image)
        # images,labels : list of data
    return images, labels

class cifar10_dataset(Dataset):
    def __init__(self, dataset_path,preprocess, kind='test'):
        super().__init__()
        # split train-valid-test if need
        if kind == 'train':
            self.file_list = ['data_batch_1', 'data_batch_2','data_batch_3','data_batch_4']
        elif kind == 'valid':
            self.file_list = ['data_batch_5']
        else:
            self.file_list = ['test_batch']
        # create list for store data
        self.images,self.labels = [],[]
        # save list from load_data
        for file_name in self.file_list:
            file_path = os.path.join(dataset_path, file_name)
            images, labels = load_data(dataset_path, file_name, preprocess)
            self.images = images
            self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
       image = self.images[idx]
       label = self.labels[idx]
       return image, label
