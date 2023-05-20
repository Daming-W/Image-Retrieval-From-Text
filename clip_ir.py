import torch
import torchvision
import pickle
import random
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os,argparse,cv2,glob,scipy,heapq
from tqdm import tqdm
from matplotlib import patches as mtp_ptch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support

import clip_modified
from utils.model import getCLIP, getCAM
from utils.preprocess import getImageTranform
from utils.dataset import ImageNetDataset
from utils.imagenet_utils import *
from utils.evaluation_tools import *
from dataset import voc, cifar

######################
### ArgumentParser ###
######################
def parse_arguments():
    parser = argparse.ArgumentParser()
    # flag
    parser.add_argument("--eval_mode", type=bool, default=True, help="retrival on public data or not")
    # data
    parser.add_argument("--dataset_path", type=str, default="/root/autodl-tmp/clip_ir/cifar10/cifar10-batches", help="directory of dataset")
    parser.add_argument("--dataset", type=str, default='cifar', help="dataset name used for evaluation" )
    parser.add_argument("--num_class", type=int, default=1, help="total number of classes")      
    parser.add_argument("--save_dir", type=str, default='eval_result', help="directory to save the result")
    parser.add_argument("--resize", type=int, default=1, help="Resize image or not")
    parser.add_argument("--image_candidatas_dir", type=str, default='/root/autodl-tmp/clip_img_retrieval/images_candidates/', help="retrival candidates images dir")
    # hyper parameters
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU to work on")
    parser.add_argument("--batch_size", type=int, default=10, help='batch size which is the candidates size')
    parser.add_argument("--workers", type=int, default=0, help="workers")
    parser.add_argument("--sim_threshold", type=float, default=5.5, help="threshold to determine retrieval")
    # model and cls
    parser.add_argument("--clip_model_name", type=str, default='RN50', help="visual Model name of CLIP")
    parser.add_argument("--sentence_prefix", type=str, default='word', help="input text type: \"sentence\", \"word\"")
    parser.add_argument("--sim2log", type=str, default='softmax', help="method to compute logits from similarity")
    # kmeans
    parser.add_argument("--num_clusters", type=int, default=3, help="number of pre-defined classes")
    parser.add_argument("--num_pca_components", type=int, default=100, help="pca target dimensions")
    return parser.parse_args()


###########################
### Retrieval Pred Func ###
###########################
def retrieval_clip(args, image_query, text_tokens, clip_model):
    print(f'retrievalling from {len(image_query)} images...')

    # encode texts
    text_tokens = text_tokens.cuda(non_blocking=True)
    text_emb = clip_model.encode_text(text_tokens)

    # run CLIP with evaluation model
    if args.eval_mode==True:
        # encode images
        image_query = image_query.cuda(non_blocking=True)
        img_query_emb = clip_model.encode_image(image_query)
        # get cosine score
        sim_score = np.dot(img_query_emb.detach().cpu().numpy(), text_emb.detach().cpu().numpy().T)
        sim_score = sim_score.ravel().tolist()
        print(f'sim_score: {sim_score}')
        # when retrieve own images,return the most likely image
        # thus compute the softmax
        bin_score = [1 if sim>=args.sim_threshold else 0 for sim in sim_score]
        return bin_score,sim_score

    # run CLIP for user testing
    if args.eval_mode==False:
        # set score list 
        sim_score = []
        # loop query and get sim score
        for i,image in enumerate(image_query):
            image = image.cuda(non_blocking=True)
            optimizer.zero_grad()
            logits_per_image,logits_per_text = clip_model(image, text_tokens)
            sim_score.append(float((logits_per_image+logits_per_text)/2))
        # compute the softmax
        sim_score_softmax = scipy.special.softmax(sim_score)
        print(f'sim_score         : {sim_score}')
        print(f'sim_score_softmax : {sim_score_softmax}')
        print(f'the retrieved image index is : {np.argmax(sim_score_softmax)+1}')

########################
### evalute function ###
########################
def evaluate_clip(args, labels_dict, dataloader, clip_model):
    # loop validation dataloader
    precision_list, recall_list, fscore_list = [],[],[]

    for i, (images,labels) in enumerate(dataloader):
        
        # cifar label come with label-index
        if args.dataset == 'cifar' or args.dataset == 'cifar10':
            # convert labels to set to get all retrivable labels
            labels_set={label.item() for label in labels}
            # get one label from set to test
            label_rand=random.choice(list(labels_set))
            text = labels_dict[label_rand]
            # load prompt and tokenize
            print(f'the random text for IR is {text}')
            prompt = "this is a photo of a "
            text_tokens = clip_modified.tokenize(prompt+text)
            # set ground truth 
            ground_truth = [1 if label==label_rand else 0 for label in labels]

        # voc labels come with one-hot type
        elif args.dataset == 'voc' or args.dataset == 'voc2007':
            # get union list for retrivable labels
            union_list = [int(max(values).item()) for values in zip(*list(labels))]
            union_label = [i for i, value in enumerate(union_list) if value == 1]
            # get one label from set to test
            label_rand = random.choice(union_label)
            print(label_rand)
            text = labels_dict[label_rand]
            # load prompt and tokenize
            print(f'the random text for IR is {text}')
            prompt = "this is a photo of a "
            text_tokens = clip_modified.tokenize(prompt+text)
            # set ground truth
            ground_truth = [1 if label[label_rand]==1 else 0 for label in labels]
            print(f'labels{labels}') 
            print(f'label_rand{label_rand}')
            print(f'text{text}')
            print(f'ground_truth{ground_truth}')

            break

        # do clip-ir
        bin_score,sim_score = retrieval_clip(args, images, text_tokens, clip_model)
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

    # get model ==> Clip model
    clip_model,preprocess = clip_modified.load(args.clip_model_name, device = args.gpu_id, jit = False)
    clip_model = clip_model.cuda()
    print(f'finish loading model (with visual backbone: {args.clip_model_name})')

    # do evaluation
    if args.eval_mode==True:
        print('do evaluation')

        # get dataset and dataloader
        if args.dataset == 'voc' or args.dataset == 'voc2007':

            args.dataset_path = '/root/autodl-tmp/clip_ir/VOC2007'
            voc2017 = voc.Voc2007Classification(root=args.dataset_path, set='test', 
                                                transform=preprocess)
            dataloader = DataLoader(voc2017,
                                    batch_size=args.batch_size,
                                    num_workers=args.workers,
                                    shuffle=True)
            # create labels dict for cifar
            labels_dict = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 
            4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 
            10: 'dining table', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 
            15: 'potted plant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tv monitor'}

        elif args.dataset == 'cifar' or args.dataset == 'cifar10':

            args.dataset_path = '/root/autodl-tmp/clip_ir/cifar10/cifar10-batches'
            cifar10 = cifar.cifar10_dataset(args.dataset_path,preprocess)
            dataloader = DataLoader(cifar10,
                                    batch_size=args.batch_size,
                                    num_workers=args.workers,
                                    shuffle=True)
            # create labels dict for cifar
            labels_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 
            4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
        
        elif args.dataset == 'caltech' or args.dataset == 'caltech101':
            
            args.dataset_path = '/root/autodl-nas/caltech-101/'
            caltech = torchvision.datasets.Caltech101(root = args.dataset_path,
                                                      transform = preprocess
                                                      )
            dataloader = DataLoader(caltech,
                                    batch_size=args.batch_size,
                                    num_workers=args.workers,
                                    shuffle=True)

        else: print('Dataset asked not found')

        print(f'total batches num : {len(dataloader)}')
        print(f'total img-cls num : {len(dataloader)*args.batch_size}')
        
        # main evaluation func
        evaluate_clip(args, labels_dict, dataloader, clip_model)

    # retrieve for own image query
    else:
        # set image candidates list
        print('retrieve customized images')
        img_query=[]
        for img_path in glob.glob(os.path.join('/root/autodl-tmp/clip_ir/image_candidates/','*.jpeg')):
            img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0)
            print(img.shape)
            img_query.append(img)
        # set text content 
        input_text = str(input('input keyword for retrieval: '))
        prompt = "this is a photo of a "
        text_tokens = clip_modified.tokenize(prompt+input_text)
        retrieval_clip(args,img_query, text_tokens, clip_model)
