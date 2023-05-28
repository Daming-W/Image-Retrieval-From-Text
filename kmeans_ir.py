import argparse
import random
import cv2
import numpy as np
from numpy import linalg as LA
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
    parser.add_argument("--dataset", type=str, default='cifar', help="dataset name used for evaluation" )
    # data preparation --parameters
    parser.add_argument("--BoF_size", type=int, default=10, help='number of each class images for building BoF')
    parser.add_argument("--batch_size", type=int, default=100, help='batch size which is the candidates size')
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU to work on")
    parser.add_argument("--workers", type=int, default=2, help="num_workers")
    # old-fashioned method
    parser.add_argument("--num_clusters", type=int, default=10, help="number of pre-defined classes")
    # parser.add_argument("--num_pca_components", type=int, default=30, help="pca target dimensions")
    parser.add_argument("--feature_extraction", type=str, default="hog", choices=["hog", "lbp"])
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
    # print(img.shape)
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
        win_size = (128, 128)  
        block_size = (64, 64)  
        block_stride = (32, 32)  
        cell_size = (32, 32) 
        num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    # hog computation
    img_f = hog.compute(img)
    return np.array(img_f)

def LBP(I):

    B = np.zeros(np.shape(I))

    code = np.array([[1, 2, 4], [8, 0, 16], [32, 64, 128]])

    # loop over all pixels except border pixels
    for i in np.arange(1, I.shape[0]-2):
        for j in np.arange(1, I.shape[1]-2):
            w = I[i-1:i+2, j-1:j+2]
            w = w >= I[i, j]
            w = w * code
            B[i, j] = np.sum(w)

    h, edges = np.histogram(B[1:-1, 1:-1], density=True, bins=256)
    return np.array(h), edges

def img_feature_bag(args, images):
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
                if args.feature_extraction == "hog":
                    img_f = hog_f(args, img)
                else:
                    img_f, edges = LBP(img)
                # print(len(img_f))
                all_images_f.append(img_f)
                classes.append(f'{class_num}')
        class_num += 1
    all_images_f = np.array(all_images_f)
    return classes, all_images_f


###########################
### Retrieval function  ###
###########################

def retrieval_kmeans(args,image_query, label_index, kmeans_model):
    preds = []
    for img in image_query:
        # compute feature vector for each image
        img = process_img(img)
        if args.feature_extraction == "hog":
            img_f = hog_f(args, img)
        else:
            img_f, edges = LBP(img)
        img_f = np.expand_dims(img_f, axis=0)
        
        preds.append(kmeans_model.predict(img_f))
    # convert to binary prediction type
    results = [1 if pred==label_index  else 0 for pred in preds]
    return results

def kmeans_process(args, image_feature, labels_dict):

    models = []

    for i in range(0, args.num_clusters):
        model_i = KMeans(n_clusters=1)
        models.append(model_i)

    sub = []
    num = 0
    c = 0
    centers = []

    for i in range(len(image_feature)):
        img = image_feature[i].astype(np.float32)
        sub.append(img)
        num += 1
        if num == args.BoF_size:
            num = 0
            sub = np.array(sub, dtype=np.float32)
            models[c].fit(sub)
            centers.append(models[c].cluster_centers_)
            sub = []
            c += 1
    all_centroids = np.concatenate([centroid for centroid in centers], axis=0)
    model = KMeans(n_clusters=args.num_clusters, init=all_centroids)
    model.fit(image_feature)

    return model
        

###########################
### evaluation function ###
###########################

def evaluate(args, labels_dict, dataloader, kmeans_model, if_cluster):
    # loop validation dataloader
    precision_list, recall_list, fscore_list = [],[],[]

    for i, (images,labels) in enumerate(dataloader):
        # to delete the last batch
        if len(images) != args.batch_size and args.feature_extraction == "sift":
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
        bin_score = retrieval_kmeans(args, images, label_rand, kmeans_model)
        
        # evalute
        precision, recall, fscore, support = precision_recall_fscore_support(ground_truth,bin_score)
        # print(precision, ' ',recall,' ',fscore)
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
            args.dataset_path = '/Users/hehongyuan/Desktop/Computer_Study/CV/Image-Retrieval-From-Text/VOC'
            voc2007_train = voc.Voc2007Classification(root=args.dataset_path, set='train', 
                                                transform=None)
            voc2007_val = voc.Voc2007Classification(root=args.dataset_path, set='val', 
                                                transform=None)
            voc2007_test = voc.Voc2007Classification(root=args.dataset_path, set='test', 
                                                transform=None)
            voc2007 = ConcatDataset([voc2007_train,voc2007_val,voc2007_test])
            
            dataloader = DataLoader(voc2007,
                                    batch_size=args.batch_size,
                                    num_workers=args.workers,
                                    shuffle=True)
            # create labels dict for voc
            labels_dict = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 
            4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 
            10: 'dining table', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 
            15: 'potted plant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tv monitor'}
            
            # get labeled images data with user input BoF size
            image_list = get_BoF_data_voc(args,voc2007)

        # get dataset and dataloader
        elif args.dataset == 'cifar' or args.dataset == 'cifar10':

            args.num_clusters = 10
            args.dataset_path = '/Users/hehongyuan/Desktop/Computer_Study/CV/clip_ir/cifar10/cifar10-batches'
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
        print(f'BoF: {args.BoF_size}')
        print(f'Batch_szie: {args.batch_size}')

        if_cluster = False

        # build BoF -> return all BoF used features [10*100, fea_dim]
        classes, image_feature = img_feature_bag(args, image_list)
        print(f'feature extration method : {args.feature_extraction}')

        # BoF to kMeans -> return labelled Kmeans Model
        kmeans = kmeans_process(args, image_feature, labels_dict)

        start = datetime.now()
        # Kmeans object do retrieval and evaluation
        evaluate(args, labels_dict, dataloader, kmeans, if_cluster)
        end = datetime.now()

        print(f'Retrieval uses {end - start}')

    else:
        # set image candidates list
        print('retrieve customized data')


#######################
### Unused function ###
#######################

### 1 ###
# def retrieval_kmeans_sift(args,image_query, label_index, kmeans_model, if_cluster):
#     images_list,preds = [],[]
#     # apply pca to compute feature vectors for image query
#     for img in image_query:
#         images_list.append(img)
#     if_cluster = True
#     _, images_f = orb(images_list, if_cluster)
#     # loop feature vec to make prediction by kmeans
#     for f in images_f:
#         f = np.expand_dims(f, axis=0)
#         preds.append(kmeans_model.predict(f))
#     # convert to binary prediction type
#     results = [1 if pred==label_index  else 0 for pred in preds]
#     return results


### 2 ###

# def orb(images, if_cluster):

#     orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.5, edgeThreshold = 31, WTA_K = 4, patchSize = 40)
#     images_descriptors=[]
#     classes = []
#     class_num = 0

#     if if_cluster:
#         for img in images:
#             if img is not None:
#                 img = process_img(img)
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 kp, des = orb.detectAndCompute(img, None)
#                 images_descriptors.append(des)
#     else:
#         for image in images:
#             for img in image:
#                 if img is not None:
#                     img = process_img(img)
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                     kp, des = orb.detectAndCompute(img, None)
#                     images_descriptors.append(des)
#                     classes.append(f'{class_num}')
#             class_num += 1


#     features_list = []
#     # print(len(images_descriptors))
#     for descriptors in images_descriptors:
#         if descriptors is None:
#             features_array = np.array([]).reshape(0, 128)
#         else:
#             for descriptor in descriptors:
#                 # print(descriptor)
#                 features_list.append(descriptor)
#                 features_array = np.array(features_list)
                
#     kmeans_mini = MiniBatchKMeans(n_clusters=50, random_state=13, init='random')

#     nr_intervals = 10
#     array_len = np.shape(features_array)[0]
#     interval_len = array_len // nr_intervals

#     for i in range(nr_intervals):
#         # print(f'Computing fitting batch: {i}')
#         # we train the Mini Batch KMeans in 50 intervals of our total number of descriptors
#         kmeans_mini.partial_fit(features_array[i * interval_len: min((i + 1) * interval_len, array_len),:])

#     predicted_clusters_for_images = []

#     for descriptors in images_descriptors:

#         image_descriptors_clusters = kmeans_mini.predict(descriptors) # places each descriptor of the image in a cluster
#         predicted_clusters_for_images.append(image_descriptors_clusters)

#     images_clusters = []
#     for clusters in predicted_clusters_for_images:
#         list_to_string = [str(cluster) for cluster in clusters]
#         images_clusters.append(' '.join(list_to_string))

#     vectorizer = TfidfVectorizer(norm='l2')
#     images_vectors = vectorizer.fit_transform(images_clusters).toarray()

#     return classes, images_vectors


### 3 ###

# def pca(images, if_cluster):
#     def preprocess(A_pp):
#         # Preprocess the data by normalizing
#         mu = np.mean(A_pp, axis=1)
#         row, column = A_pp.shape
#         Q = []
#         for i in range(row):
#             # Subtract the mean from each row
#             Q.append(A_pp[i] - mu[i])  
#         Q_norms = LA.norm(Q, np.inf, axis=1)
#         A = np.zeros((row, column))
#         # Normalize each row
#         for i in range(row):
#             if Q_norms[i] == 0:
#                 A[i, :] = Q[i] / 1
#             else:
#                 A[i, :] = Q[i] / Q_norms[i] 

#         return A, Q_norms, mu

#     def eigen_ped(A):
#         # get eigenvalues and eigenvectors of A^T * A
#         d, v = LA.eig(np.dot(A.T, A))
#         row, column = A_pp.shape
#         # padding the eigenvalues with zeros
#         F = np.dot(A, v)
#         D = np.pad(d, (0, row - column), mode='constant')  
#         _, column = F.shape
#         # norm each column of F
#         for i in range(column):
#             F[:, i] = F[:, i] / LA.norm(F[:, i]) 

#         return F, D

#     def reduce_dimensionality(image_vector, k, F, D, A_means, Q_norms):
#         # reduce dim and calculate compression ratio
#         row, column = F.shape
#         index = np.argsort(D[:column])
#         idx = index[::-1]
#         nominator = 0
#         for i in range(k):
#             nominator += D[idx[i]]
#         compressed_image = np.zeros(column)
#         for i in range(k):
#             # compressed image vector
#             compressed_image[i] = np.dot(F[:, i], (image_vector - A_means) / Q_norms)  
#         # return compressed image and compression ratio
#         return compressed_image, nominator / np.sum(D)  

#     images_list, classes = [], []  
#     class_num = 0

#     # if len(images) != args.num_clusters:
#     if if_cluster is True:
#         # Process individual images
#         for img in images:
#             if img is not None:
#                 img = process_img(img)  
#                 images_list.append(img.flatten())  
#     else:
#         # Process images in clusters
#         for imgs in images:
#             for img in imgs:
#                 if img is not None:
#                     img = process_img(img)
#                     images_list.append(img.flatten()) 
#                     classes.append(f'{class_num}')
#             class_num += 1

#     A_pp = np.stack(images_list).T 
#     # Preprocess the image matrix
#     A, Q_norms, A_means = preprocess(A_pp) 
#     F, D = eigen_ped(A)  
#     F_real = np.real(F)

#     img_f = []
#     for i in range(len(images_list)):
#         # reduce dim of each image
#         fea, _ = reduce_dimensionality(A_pp[:, i], 40, F, D, A_means, Q_norms)
#         img_f.append(fea)  

#     img_f = np.array(img_f)
#     return classes, img_f
