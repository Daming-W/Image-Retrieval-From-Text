import clip_modified
import torch
import torchvision
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import patches as mtp_ptch
from torchvision import transforms
from tqdm.notebook import tqdm
import argparse
import cv2
import json
import heapq
from sklearn.preprocessing import normalize
from sklearn.metrics import average_precision_score
import utils.aslloss 
from utils.model import getCLIP, getCAM
from utils.preprocess import getImageTranform
from utils.dataset import ImageNetDataset
from utils.imagenet_utils import *
from utils.evaluation_tools import *
from tqdm import tqdm
from dataset.get_dataset import get_datasets

parser = argparse.ArgumentParser()
# data
parser.add_argument("--dataset_dir", type=str, 
                    default='/home/notebook/data/group/projects/tagging/caption/datasets/public/coco2014',
                    help="directory of dataset")
parser.add_argument("--dataset", type=str, default='coco', help="dataset name used for evaluation" )
parser.add_argument("--num_class", type=int, default=1, help="total number of classes")      
parser.add_argument("--save_dir", type=str, default='eval_result', help="directory to save the result")
parser.add_argument("--resize", type=int, default=1, help="Resize image or not")

# hyper parameters
parser.add_argument("--gpu_id", type=int, default=0, help="GPU to work on")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--workers", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=20, help="the number of epochs")

# model and cls
parser.add_argument("--clip_model_name", type=str, default='RN101', help="Model name of CLIP")
parser.add_argument("--sentence_prefix", type=str, default='word', help="input text type: \"sentence\", \"word\"")
parser.add_argument("--sim2log", type=str, default='softmax', help="method to compute logits from similarity")
args = parser.parse_args()

# Data loading code
if args.dataset=='coco':
    args.num_class = 80
    args.dataset_dir = '/home/notebook/data/group/projects/tagging/caption/datasets/public/coco2014'
elif args.dataset=='voc':
    args.num_class = 20
    args.dataset_dir = '/home/notebook/data/group/projects/tagging/caption/datasets/public/voc2007'
elif args.dataset=='openimage':
    args.num_class = 567
    args.dataset_dir = '/home/notebook/data/group/projects/tagging/caption/datasets/public/open-images-v6'
else:
    NotImplementedError

label_list=[]
if args.dataset == 'coco':
    json_path = os.path.join(args.dataset_dir,'annotations/instances_val2014.json')
    json_labels = json.load(open(json_path,"r"))
    for i in json_labels["categories"]:
        label_list.append(i['name'])

elif args.dataset == 'voc':
    label_list  = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
                    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

elif args.dataset == 'openimage':
    label_list = []

else:
    raise NotImplementedError()

train_dataset, val_dataset = get_datasets(args)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

sample = next(iter(val_loader))
print(sample[0])
print(sample[1])

'''
for i, img in enumerate(sample[0]):
    if i==1:
        img = img.detach().numpy()
        print(img.shape)
        print(type(img))
        img = img.reshape(224,224,3)
        plt.imshow(img)
print(f"Label: {sample[1]}")
'''

# get target model
class Resnet50(torch.nn.Module):
    def __init__(self, num_class):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.fc = torch.nn.Sequential(torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=resnet.fc.in_features, out_features=num_class))
        self.base_model = resnet
        self.Sigmoid = torch.nn.Sigmoid()
    def forward(self, input):
        logits = self.Sigmoid(self.base_model(input))
        return logits

RN50_model = Resnet50(args.num_class).cuda()

target_layer = RN50_model.base_model.layer4[-1]
RN50_model_cam = getCAM(model_name="GradCAM_original",model=RN50_model, 
    target_layer=target_layer, gpu_id=args.gpu_id,reshape_transform=None)

input = sample[0].cuda()
heatmap = RN50_model_cam(input)
print(heatmap.shape)

heatmap_group=[]
for index in range(80):
    heatmap = RN50_model_cam(input,target_category=index)
    heatmap_group.append(heatmap)
print("heatmap group shape: ", np.asarray(heatmap_group).shape)

# testing CLIP cam batch output
tea_model, target_layer, reshape_transform, preprocess = getCLIP(
        model_name="RN50", gpu_id=args.gpu_id)

CLIP_model_cam = getCAM(model_name="GradCAM", model=tea_model, 
                        target_layer=tea_model.visual.layer4[-1],
                        gpu_id=args.gpu_id, reshape_transform=reshape_transform)
                       
input = sample[0].cuda()
heatmap_group=[]
text_features =[]
for label in label_list:
    text_tokens = clip_modified.tokenize("this is a photo of a "+label).cuda()
    text_features.append(tea_model.encode_text(text_tokens))

for text_feature in text_features:
    heatmap = CLIP_model_cam(input,text_feature)
    heatmap_group.append(heatmap)
print("heatmap group shape: ", np.asarray(heatmap_group).shape)

'''
# testing CLIP cam batch output: Method2
text_labels_p = [f"this is a photo of a {label}" for label in label_list]
text_tokens = clip_modified.tokenize(text_labels_p).cuda()
text_features = tea_model.encode_text(text_tokens)
print(text_features.shape)
print(text_features[0].shape)
text_features[0] = text_features[0].expand(1,-1)
print(text_features[0].shape)

heatmap = CLIP_model_cam(input,text_features[0])
'''