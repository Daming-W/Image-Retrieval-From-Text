import clip_modified
import torch
from PIL import Image
import numpy as np
import argparse
import os
import time
from utils.evaluation_tools import getHeatMap, getHeatMapNoBBox, getHeatMapOneBBox, MaskToBBox
from utils.model import getCLIP, getCAM
from utils.preprocess import getImageTranform
from clip_modified import build_model
import cv2

RESIZE = 1

ImageTransform = getImageTranform(resize=RESIZE)
originalTransform = getImageTranform(resize=RESIZE, normalized=False)


def clipcam(CLIP_MODEL_NAME, CAM_MODEL_NAME, images, sentence, DISTILL_NUM = 0, ATTACK = None, GPU_ID = 'cpu'):
    
    model, target_layer, reshape_transform = getCLIP(model_name=CLIP_MODEL_NAME, gpu_id=GPU_ID)
    
    cam = getCAM(model_name=CAM_MODEL_NAME, model=model, target_layer=target_layer, gpu_id=GPU_ID, reshape_transform=reshape_transform)
    MASK_THRESHOLD = get_mask_threshold(CLIP_MODEL_NAME)
    
    final_img = get_clipcam_single(cam, model, MASK_THRESHOLD, images[0], sentence, DISTILL_NUM = DISTILL_NUM, ATTACK = ATTACK, GPU_ID = GPU_ID)

    del cam
    del model
    return final_img

# get mask for generating bbox if need
def get_mask_threshold(CLIP_MODEL_NAME):
    if CLIP_MODEL_NAME == 'RN50':
        MASK_THRESHOLD = 0.2
    if CLIP_MODEL_NAME == 'RN101':
        MASK_THRESHOLD = 0.2
    if CLIP_MODEL_NAME == 'ViT-B/16':
        MASK_THRESHOLD = 0.3
    if CLIP_MODEL_NAME == 'ViT-B/32':
        MASK_THRESHOLD = 0.3
    return MASK_THRESHOLD

# compute heatmap over the original image
def get_clipcam(clipcam, model, MASK_THRESHOLD, image, sentence = None, DISTILL_NUM = 0, ATTACK = None, GPU_ID = 'cpu'):
    image = image
    orig_image = image
    image = ImageTransform(image)
    orig_image = originalTransform(orig_image)
    image = image.unsqueeze(0)
    image = image.to(GPU_ID)
    orig_image = orig_image.to(GPU_ID)

    if sentence == None:
        sentence = input(f"Please enter the query sentence: ")
    text  = clip_modified.tokenize(sentence)
    text = text.to(GPU_ID)
    text_features = model.encode_text(text)
    
    logits_per_image,logits_per_text = model(image, text)
    sim_score = float((logits_per_image+logits_per_text)/2)
    print(f'sim score : {sim_score}')

    grayscale_cam = clipcam(input_tensor=image, text_tensor=text_features)[0, :]
    grayscale_cam_total = grayscale_cam[np.newaxis, :]

    grayscale_cam_mask = np.where(grayscale_cam_total < MASK_THRESHOLD, 0, 1)
    pred_bbox, pred_mask = MaskToBBox(grayscale_cam_mask, 1)
    final_img = getHeatMapOneBBox(grayscale_cam, orig_image.permute(1, 2, 0).cpu().numpy(), pred_bbox, sentence)
    #final_img = getHeatMapNoBBox(grayscale_cam,orig_image.permute(1, 2, 0).cpu().numpy(),'test')
    return final_img

# setup args
parser = argparse.ArgumentParser()
parser.add_argument("--do_compare", type=bool, default =False,
                    help="if true, show both clip orig and ft to compare")
parser.add_argument("--image_folder_path", type=str, default='/root/autodl-tmp/clip_ir/image_candidates/',
                    help="single image path or 4 images directory (grid)")                
parser.add_argument("--image_name", type=str,
                    help="single image path or 4 images directory (grid)")
parser.add_argument("--gpu_id", type=str, default='cpu',
                    help="GPU id to work on, \'cpu\'.")
parser.add_argument("--clip_model_name", type=str,
                    default='RN50', help="Model name of CLIP")
parser.add_argument("--cam_model_name", type=str,
                    default='GradCAM', help="Model name of GradCAM")
parser.add_argument("--resize", type=int,
                    default=1, help="Resize image or not")
parser.add_argument("--distill_num", type=int, default=0,
                    help="Number of iterative masking")
parser.add_argument("--attack_type", type=str, default=None,
                    help="attack type: \"snow\", \"fog\"")
parser.add_argument("--sentence", type=str, default='',
                    help="input text")
parser.add_argument("--vis_res_folder_path", type=str, default='/root/autodl-tmp/clip_ir/cam_vis/',
                    help="path of vis results")                  
args = parser.parse_args()

id = input(f'image id: ')
args.image_name = str(id) + '.jpeg'
args.sentence = str(input(f'sentence: '))

# setup device
if args.gpu_id != 'cpu':
    args.gpu_id = int(args.gpu_id)
# Image read target image
image_path=args.image_folder_path + args.image_name
if os.path.isfile(image_path):
    img = Image.open(image_path)
    images = [img]
else:
    images = []
    for f in os.listdir(image_path):
        images.append(Image.open(os.path.join(image_path, f)))

        
# get model and load weights 
itc_model, target_layer, reshape_transform = getCLIP(model_name=args.clip_model_name, gpu_id=args.gpu_id)
print('model loaded')

# get CAM model based on itc model and cam method
cam = getCAM(model_name=args.cam_model_name, 
            model=itc_model, target_layer=target_layer, 
            gpu_id=args.gpu_id, reshape_transform=reshape_transform)

# compute cam
final_img_1 = get_clipcam(cam, itc_model, get_mask_threshold(args.clip_model_name), 
                            image=img, sentence=args.sentence, 
                            DISTILL_NUM = args.distill_num, ATTACK = args.attack_type, GPU_ID = args.gpu_id)

final_img_1.save(args.vis_res_folder_path + f'{id}_{args.sentence}.png')
print('visualization done')
