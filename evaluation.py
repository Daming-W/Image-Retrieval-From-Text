import os
import pickle
import torch
import random
import torchvision
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os,argparse,cv2,glob,scipy,heapq
from tqdm import tqdm
from matplotlib import patches as mtp_ptch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import clip_modified
from utils.model import getCLIP, getCAM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

######################
### ArgumentParser ###
######################
def parse_arguments():
    parser = argparse.ArgumentParser()
    # data preparation
    parser.add_argument("--dataset_path", type=str, default="/root/autodl-tmp/clip_ir/cifar10/cifar10-batches")
    parser.add_argument("--batch_size", type=int, default=32, help='batch size which is the candidates size')
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU to work on")
    parser.add_argument("--num_workers", type=int, default=2, help="num_workers")
    # deep learning CLIP method
    parser.add_argument("--clip_model_name", type=str, default='RN50', help="visual Model name of CLIP")
    # old-fashioned method
    parser.add_argument("--num_clusters", type=int, default=3, help="number of pre-defined classes")
    parser.add_argument("--num_pca_components", type=int, default=100, help="pca target dimensions")

    return parser.parse_args()

#########################
### Data Loading Code ###
#########################
def load_data(dataset_path, file_name):
    """Load CIFAR-10 data from given dataset_path"""
    file_path = os.path.join(dataset_path, file_name)
    with open(file_path, 'rb') as f:
        # load data with pickle
        entry = pickle.load(f, encoding='latin1')
        images = entry['data']
        labels = entry['labels']
        # reshape the array to 32x32 color image
        images = images.reshape(-1, 3, 32, 32)  # '-1' means the value will be deduced from the shape of array
    return images, labels

class cifar10_dataset(Dataset):
    def __init__(self, dataset_path, kind='train'):
        super().__init__()
        train_list = ['data_batch_1', 'data_batch_2','data_batch_3','data_batch_4']
        valid_list = ['data_batch_5']
        test_list = ['test_batch']

        if kind == 'train':
            self.file_list = train_list
        elif kind == 'valid':
            self.file_list = valid_list
        else:
            self.file_list = test_list
        self.images = []
        self.labels = []
        for file_name in self.file_list:
            file_path = os.path.join(dataset_path, file_name)
            images, labels = load_data(dataset_path, file_name)
            self.images.append(images)
            self.labels.append(labels)

        # concatenate the data together
        self.images = np.concatenate(self.images, axis=0)   # NCHW
        self.images = self.images.transpose((0, 2, 3, 1))   # convert to HWC
        self.labels = np.concatenate(self.labels, axis=0)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
       image = self.images[idx]
       label = self.labels[idx]
       return image, label

################################
### CLIP Retrieval Pred Func ###
################################
def retrieval_clip(image_query, text_tokens, clip_model):
    # encode text
    text_emb = clip_model.encode_text(text_tokens)
    # set score list
    sim_score = []
    # encode images
    for i,image in enumerate(image_query):
        image = image.cuda(non_blocking=True)
        with torch.no_grad():
            image_query_emb = clip_model.encode_image(image)

        logits_per_image,logits_per_text = clip_model(image, text_tokens)
        sim_score.append(float((logits_per_image+logits_per_text)/2))
        print(f'logits_per_image:{logits_per_image.item()} & logits_per_text:{logits_per_text.item()}')
    # compute the softmax
    sim_score_softmax = scipy.special.softmax(sim_score)
    print(f'sim_score         : {sim_score}')
    print(f'sim_score_softmax : {sim_score_softmax}')
    print(f'the retrieved image index is : {np.argmax(sim_score_softmax)+1}')
    
    #return 


###########################
### evaluation function ###
###########################
def evaluate_clip(args, labels_dict, dataloader, clip_model):
    # loop validation dataloader
    for i, (images,labels) in enumerate(dataloader):
        # convert labels to set to get all retrivable label
        labels_set={label.item() for label in labels}
        # get one label from set to test
        label_id=random.randint(0,len(labels_set))
        text = labels_dict[label_id]
        # load prompt and tokenize
        print(f'the random text for IR is {text}')
        prompt = "this is a photo of a "
        text_tokens = clip_modified.tokenize(prompt+text).cuda()
        # set ground truth 
        ground_truth = [1 if label==label_id else 0 for label in labels]

        # do clip-ir
        #retrieval_clip(image_query, text_tokens, clip_model)

#################
### Main Func ###
#################
if __name__ == '__main__':
    # get args
    args = parse_arguments()

    # create labels dict for cifar
    labels_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 
    4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

    # define image transform
    transform = torchvision.transforms.ToTensor()
    # get CLIP and preprocess method
    clip_model,preprocess=clip_modified.load(args.clip_model_name, 
                                            device = args.gpu_id, 
                                            jit = False)
    clip_model = clip_model.cuda()
    print(f'finish loading model (with visual backbone: {args.clip_model_name})')

    # get dataset and dataloader
    cifar10 = cifar10_dataset(dataset_path = args.dataset_path)
    dataloader = DataLoader(cifar10,batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=True)
    print(f'total batches num : {len(dataloader)}')
    print(f'total img-cls num : {len(dataloader)*args.batch_size}')

    # test or visualize
    img,label = iter(dataloader).next()
    print(label)

    evaluate_clip(args, labels_dict, dataloader, clip_model)