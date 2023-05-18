import os
import argparse
import cv2
import numpy as np
import time
import glob
import scipy
import pickle
from PIL import Image, ImageDraw

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.nn.functional import cosine_similarity

from torch.utils.data import Dataset, DataLoader
import clip_modified

######################
### ArgumentParser ###
######################
def parse_arguments():
    parser = argparse.ArgumentParser()
    # flag
    parser.add_argument("--eval_mode", type=bool, default=True, help="retrival on public data or not")
    # data preparation --path
    parser.add_argument("--dataset_path", type=str, default="/root/autodl-tmp/clip_ir/cifar10/cifar10-batches")
    parser.add_argument("--BoF_image_path", type=str, default="/root/autodl-tmp/clip_ir/bof_images/")
    parser.add_argument("--test_image_path", type=str, default="/root/autodl-tmp/clip_ir/bof_images/test/")
    # data preparation --parameters
    parser.add_argument("--BoF_size", type=int, default=10, help='number of each class images for building BoF')
    parser.add_argument("--batch_size", type=int, default=32, help='batch size which is the candidates size')
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU to work on")
    parser.add_argument("--workers", type=int, default=2, help="num_workers")
    # deep learning CLIP method
    parser.add_argument("--clip_model_name", type=str, default='RN50', help="visual Model name of CLIP")
    # old-fashioned method
    parser.add_argument("--num_clusters", type=int, default=10, help="number of pre-defined classes")
    parser.add_argument("--num_pca_components", type=int, default=100, help="pca target dimensions")

    return parser.parse_args()

#########################
### Data Loading Code ###
#########################
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
            if preprocess!=None:
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

######################################
### Build BOF with prepared folder ###
######################################

def get_BoF_data(args,cifar10):
    image_list = list([] for _ in range(args.num_clusters))
    bof_dataloader = DataLoader(cifar10,num_workers=args.workers,shuffle=True)
    for image, label in bof_dataloader:
        label_index = label.detach()[0].item()

        if len(image_list[label_index])<args.BoF_size:
            image_list[label_index].append(image)
        
        if all(len(sublist) == args.BoF_size for sublist in image_list):
            # image_list: a list with number of classes sublists in it
            return image_list

# old oversion BOF
def build_BoF(args,kmeans,pca,sift):
    all_descriptors=[]
    label_list=[]
    for folder in os.listdir(args.BoF_image_path):
        folder_path = os.path.join(args.BoF_image_path, folder)
        # Check if it's a folder
        if os.path.isdir(folder_path) and 'test' not in folder_path:
            # Traverse all images in the subfolder
            label_list.append(folder_path.split('/')[-1])
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # Check if it's a file
                if os.path.isfile(file_path):
                    # Read the image
                    img = cv2.imread(file_path)
                    # Check if it's a valid image
                    if img is not None:
                        # Convert the image to grayscale
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # Extract SIFT keypoints and descriptors
                        _, descriptors = sift.detectAndCompute(gray, None)
                        # Append descriptors to the list
                        if descriptors is not None:
                            all_descriptors.extend(descriptors)

    # Stack all descriptors vertically
    all_descriptors = np.vstack(all_descriptors)
    # Apply PCA to reduce dimensionality
    pca.fit(all_descriptors)
    reduced_descriptors = pca.transform(all_descriptors)
    # Fit KMeans to create a bag of features
    kmeans.fit(reduced_descriptors)
    BoF_dict = {key: value for key, value in zip(label_list, kmeans.cluster_centers_)}
    return BoF_dict
def retrieval_single_img(args,BoF_dict,pca,sift):
    score_list,all_descriptors = [],[]
    # ask user for input text and check valid
    while True:
        input_text = str(input(f'input keyword for retrieval from {BoF_dict.keys()} : '))
        if input_text not in BoF_dict : 
            print('check the retrivable text and retry...')
            time.sleep(3)
        else:
            break
    # get label descriptor by user input
    label_descriptor = BoF_dict[input_text]
    print(f'get input label descriptor and ready to retrieve : {input_text}')
    # get image
    for img_path in glob.glob(os.path.join(args.test_image_path,'*.JPEG')):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # feature extraction
        _, descriptors = sift.detectAndCompute(gray, None)
         # Append descriptors to the list
        if descriptors is not None:
            all_descriptors.extend(descriptors)
    # Stack all descriptors vertically
    all_descriptors = np.vstack(all_descriptors)
    # Apply PCA to reduce dimensionality
    pca.fit(all_descriptors)
    reduced_descriptors = pca.transform(all_descriptors)
    print(f'finish computing image candidates with {len(reduced_descriptors)} images totally')
    # loop each clusters and get the simlarity
    print(len(reduced_descriptors))
    for query_descriptor in reduced_descriptors:
        score_list.append(np.mean(scipy.spatial.distance.cosine(query_descriptor,label_descriptor)))
    print('finish computing the cosine similarity..')
    return score_list

###########################
### Retrieval function  ###
###########################
def retrieval_kmeans(args,image_query, label_index, kmeans_model):
    # do something
    return sim_scaore

###########################
### evaluation function ###
###########################

def evaluate(args, labels_dict, dataloader, kmeans_model):
    # loop validation dataloader
    precision_list, recall_list, fscore_list = [],[],[]
    for i, (images,labels) in enumerate(dataloader):   
        # convert labels to set to get all retrivable label
        labels_set={label.item() for label in labels}
        # get one label from set to test
        label_id=random.randint(0,len(labels_set))
        text = labels_dict[label_id]


#################
### Main Func ###
#################
if __name__ == '__main__':
    # get all args
    args = parse_arguments()
    print('parse arguments all set')
    
    # get preprocess method
    _,preprocess = clip_modified.load(args.clip_model_name, device = args.gpu_id, jit = False)
    
    # create labels dict for cifar
    labels_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 
    4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

    # do evaluation
    if args.eval_mode==True:
        print('do evaluation')
        # get dataset and dataloader
        cifar10 = cifar10_dataset(args.dataset_path,preprocess)
        dataloader = DataLoader(cifar10,batch_size=args.batch_size,
                                num_workers=args.workers,shuffle=True)
        print(f'total batches num : {len(dataloader)}')
        print(f'total img-cls num : {len(dataloader)*args.batch_size}')

        # get labeled images data with user input BoF size
        image_list = get_BoF_data(args,cifar10)
        # build BoF -> return all BoF used features [10*100, fea_dim]

        # BoF to kMeans -> return labelled Kmeans Model

        # Kmeans object do retrieval and evaluation

    else:
        # set image candidates list
        print('retrieve customized images')
