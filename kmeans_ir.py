from collections import Counter
import os
import argparse
import random
import cv2
import numpy as np
import pickle
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torchvision.transforms.functional as TF

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataset import voc, cifar

######################
### ArgumentParser ###
######################

def parse_arguments():
    parser = argparse.ArgumentParser()
    # flag
    parser.add_argument("--eval_mode", type=bool, default=True, help="retrival on public data or not")
    # data preparation --path
    parser.add_argument("--dataset_path", type=str, default="/root/autodl-tmp/clip_ir/cifar10/cifar10-batches", help="directory of dataset")
    parser.add_argument("--dataset", type=str, default='voc', help="dataset name used for evaluation" )
    # data preparation --parameters
    parser.add_argument("--BoF_size", type=int, default=10, help='number of each class images for building BoF')
    parser.add_argument("--batch_size", type=int, default=200, help='batch size which is the candidates size')
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU to work on")
    parser.add_argument("--workers", type=int, default=2, help="num_workers")
    # old-fashioned method
    parser.add_argument("--num_clusters", type=int, default=10, help="number of pre-defined classes")
    parser.add_argument("--num_pca_components", type=int, default=80, help="pca target dimensions")
    parser.add_argument("--feature_extraction", type=str, default="pca", choices=["hog", "pca"])
    return parser.parse_args()

######################################
### Build BOF with prepared folder ###
######################################

def get_BoF_data_cifar(args,cifar10):
    image_list = list([] for _ in range(10))
    bof_dataloader = DataLoader(cifar10,num_workers=args.workers,shuffle=True)
    for image, label in bof_dataloader:
        # get inspected label
        label_index = label.detach()[0].item()
        # check and append
        if len(image_list[label_index])<args.BoF_size:
            image_list[label_index].append(image.squeeze(0))
        if all(len(sublist) == args.BoF_size for sublist in image_list):
            # image_list: a list with number of classes sublists in it
            return image_list

def get_BoF_data_voc(args,voc2007):
    image_list = list([] for _ in range(20))
    bof_dataloader = DataLoader(voc2007,num_workers=args.workers,shuffle=True)
    for image, label in bof_dataloader:
        # get inspected label
        label_list = label.squeeze().tolist()
        label_index_list =[]
        for index in range(len(label_list)):
            if label_list[index]==1:label_index_list.append(index) 
        # get one of labels as index
        label_index = random.choice(label_index_list)
        # check and append
        if len(image_list[label_index])<args.BoF_size:
            image_list[label_index].append(image.squeeze(0))
        if all(len(sublist) == args.BoF_size for sublist in image_list):
            # image_list: a list with number of classes sublists in it
            return image_list

##########################
### Feature Extraction ###
##########################

def process_img(img):
    # convert image type and shape for opencv
    img = np.array(img) * 255
    img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.uint8)
    return img

def hog_f(args, img):
    # create HOG module for different image size
    if args.dataset == "cifar":
        win_size = (8, 8)  
        block_size = (4, 4)  
        block_stride = (2, 2)  
        cell_size = (2, 2) 
        num_bins = 8
    elif args.dataset == "voc":
        win_size = (32, 32)  
        block_size = (16, 16)  
        block_stride = (8, 8)  
        cell_size = (8, 8) 
        num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    # hog computation
    img_f = hog.compute(img)
    return np.array(img_f)

def img_feature_bag(images):
    # compute BoF 
    all_images_f,classes = [],[]
    class_num = 0
    for imgs in images:
        # get each image in the same label sublist
        for img in imgs:
            if img is not None:
                # process the images
                img = process_img(img)
                # get features of each image
                all_images_f.append(hog_f(args, img))
                classes.append(f'{class_num}')
        class_num += 1
    all_images_f = np.array(all_images_f)
    return classes, all_images_f

import numpy as np
from numpy import linalg as LA

def pca(images):
    def preprocess(A_pp):
        # Preprocess the data by normalizing
        mu = np.mean(A_pp, axis=1)
        row, column = A_pp.shape
        Q = []
        for i in range(row):
            # Subtract the mean from each row
            Q.append(A_pp[i] - mu[i])  
        Q_norms = LA.norm(Q, np.inf, axis=1)
        A = np.zeros((row, column))
        # Normalize each row
        for i in range(row):
            if Q_norms[i] == 0:
                A[i, :] = Q[i] / 1
            else:
                A[i, :] = Q[i] / Q_norms[i] 

        return A, Q_norms, mu

    def eigen_ped(A):
        # get eigenvalues and eigenvectors of A^T * A
        d, v = LA.eig(np.dot(A.T, A))
        row, column = A_pp.shape
        # padding the eigenvalues with zeros
        F = np.dot(A, v)
        D = np.pad(d, (0, row - column), mode='constant')  
        _, column = F.shape
        # norm each column of F
        for i in range(column):
            F[:, i] = F[:, i] / LA.norm(F[:, i]) 

        return F, D

    def reduce_dimensionality(image_vector, k, F, D, A_means, Q_norms):
        # reduce dim and calculate compression ratio
        row, column = F.shape
        index = np.argsort(D[:column])
        idx = index[::-1]
        nominator = 0
        for i in range(k):
            nominator += D[idx[i]]
        compressed_image = np.zeros(column)
        for i in range(k):
            # compressed image vector
            compressed_image[i] = np.dot(F[:, i], (image_vector - A_means) / Q_norms)  
        # return compressed image and compression ratio
        return compressed_image, nominator / np.sum(D)  

    images_list, classes = [], []  
    class_num = 0

    if len(images) != args.num_clusters:
        # Process individual images
        for img in images:
            if img is not None:
                img = process_img(img)  
                images_list.append(img.flatten())  
    else:
        # Process images in clusters
        for imgs in images:
            for img in imgs:
                if img is not None:
                    img = process_img(img)
                    images_list.append(img.flatten()) 
                    classes.append(f'{class_num}')
            class_num += 1

    A_pp = np.stack(images_list).T 
    # Preprocess the image matrix
    A, Q_norms, A_means = preprocess(A_pp) 
    F, D = eigen_ped(A)  
    F_real = np.real(F)

    img_f = []
    for i in range(len(images_list)):
        # reduce dim of each image
        fea, _ = reduce_dimensionality(A_pp[:, i], args.num_pca_components, F, D, A_means, Q_norms)
        img_f.append(fea)  

    img_f = np.array(img_f)
    return classes, img_f

###########################
### Retrieval function  ###
###########################

def retrieval_kmeans_hog(args,image_query, label_index, kmeans_model):
    preds = []
    for img in image_query:
        # compute feature vector for each image
        img = process_img(img)
        img_f = hog_f(args, img)
        img_f = np.expand_dims(img_f, axis=0)
        preds.append(kmeans_model.predict(img_f))
    # convert to binary prediction type
    results = [1 if pred==label_index  else 0 for pred in preds]
    return results

def retrieval_kmeans_pca(args,image_query, label_index, kmeans_model):
    images_list,preds = [],[]
    # apply pca to compute feature vectors for image query
    for img in image_query:
        images_list.append(img)
    _, images_f = pca(images_list)
    # loop feature vec to make prediction by kmeans
    for f in images_f:
        f = np.expand_dims(f, axis=0)
        preds.append(kmeans_model.predict(f))
    # convert to binary prediction type
    results = [1 if pred==label_index  else 0 for pred in preds]
    return results

def kmeans_process(args, image_feature, labels_dict):
    # Initialise Kmeans
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0)
    # fit data
    kmeans.fit(image_feature)
    # get centers of each cluster
    centers = kmeans.cluster_centers_
    # create a dict to store the center of each cluster
    cluster_centers_dict = {}
    for i in range(len(labels_dict)):
        cluster_centers_dict[labels_dict[i]] = centers[i]
    # print(cluster_centers_dict)
    # show the lables of each image
    number_counts = Counter(kmeans.labels_)
    # for number, count in number_counts.items():
    #     print(f"Number {number}: {count}")
    return kmeans
        

###########################
### evaluation function ###
###########################

def evaluate(args, labels_dict, dataloader, kmeans_model):
    # loop validation dataloader
    precision_list, recall_list, fscore_list = [],[],[]

    for i, (images,labels) in enumerate(dataloader):
        # to delete the last batch
        if len(images) != args.batch_size and args.feature_extraction == "pca":
            break
        # cifar label come with label-index
        if args.dataset == 'cifar' or args.dataset == 'cifar10':

            # convert labels to set to get all retrivable labels
            labels_set={label.item() for label in labels}
            # get one label from set to test
            label_rand=random.choice(list(labels_set))
            # set ground truth 
            ground_truth = [1 if label==label_rand else 0 for label in labels]

        # voc labels come with one-hot type
        elif args.dataset == 'voc' or args.dataset == 'voc2007':

            # get union list for retrivable labels
            union_list = [int(max(values).item()) for values in zip(*list(labels))]
            union_label = [i for i, value in enumerate(union_list) if value == 1]
            # get one label from set to test
            label_rand = random.choice(union_label)
            # set ground truth
            ground_truth = [1 if label[label_rand]==1 else 0 for label in labels]

        # do retrieval
        if args.feature_extraction == "hog":
            bin_score = retrieval_kmeans_hog(args, images, label_rand, kmeans_model)
        else: # pca
            bin_score = retrieval_kmeans_pca(args, images, label_rand, kmeans_model)
        
        # evalute
        precision, recall, fscore, support = precision_recall_fscore_support(ground_truth,bin_score)
        print(precision, ' ',recall,' ',fscore)
        precision_list.append(np.mean(precision))
        recall_list.append(np.mean(recall))
        fscore_list.append(np.mean(fscore))
    print(f'evaluation result: precision:{np.mean(precision_list)} recall:{np.mean(recall_list)} fscore:{np.mean(fscore_list)}')


#################
### Main Func ###
#################
if __name__ == '__main__':

    # Setup ignoring warnings
    import warnings
    warnings.filterwarnings("ignore")
    np.seterr(divide='ignore',invalid='ignore')

    # get all args
    args = parse_arguments()
    print('parse arguments all set')

    # do evaluation
    if args.eval_mode==True:
        print('do evaluation')

        # get dataset and dataloader
        if args.dataset == 'voc' or args.dataset == 'voc2007':
            
            args.num_clusters = 20
            args.dataset_path = '/root/autodl-tmp/clip_ir/VOC2007'
            voc2017 = voc.Voc2007Classification(root=args.dataset_path, set='test', 
                                                transform=None)
            dataloader = DataLoader(voc2017,
                                    batch_size=args.batch_size,
                                    num_workers=args.workers,
                                    shuffle=True)
            # create labels dict for voc
            labels_dict = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 
            4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 
            10: 'dining table', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 
            15: 'potted plant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tv monitor'}
            
            # get labeled images data with user input BoF size
            image_list = get_BoF_data_voc(args,voc2017)

        # get dataset and dataloader
        elif args.dataset == 'cifar' or args.dataset == 'cifar10':

            args.num_clusters = 10
            args.dataset_path = '/root/autodl-tmp/clip_ir/cifar10/cifar10-batches'
            cifar10 = cifar.cifar10_dataset(args.dataset_path,None)
            dataloader = DataLoader(cifar10,
                                    batch_size=args.batch_size,
                                    num_workers=args.workers,
                                    shuffle=True)
            # create labels dict for cifar
            labels_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 
            4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
            
            # get labeled images data with user input BoF size
            image_list = get_BoF_data_cifar(args,cifar10)
            
        else: print('Dataset asked not found')

        print(f'total batches num : {len(dataloader)}')
        print(f'total img-cls num : {len(dataloader)*args.batch_size}')

        # build BoF -> return all BoF used features [10*100, fea_dim]
        if args.feature_extraction == "hog":
            classes, image_feature = img_feature_bag(image_list)
            print(f'feature extration method : {args.feature_extraction}')

        elif args.feature_extraction == "pca": 
            # pca requires retrieval batch must cooperate with BoF
            args.batch_size = args.num_clusters * args.BoF_size
            classes, image_feature = pca(image_list)
            print(f'feature extration method : {args.feature_extraction}')

        # BoF to kMeans -> return labelled Kmeans Model
        kmeans = kmeans_process(args, image_feature, labels_dict)
        # Kmeans object do retrieval and evaluation
        evaluate(args, labels_dict, dataloader, kmeans)

    else:
        # set image candidates list
        print('retrieve customized data')
