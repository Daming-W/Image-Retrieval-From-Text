#####
Cross-modality Image Retrieval from Text: old-fashioned or deep-learning
-------------------------------------------------------------------------

Retrieval tasks are typical applications in computer science, and the advancement of multimedia technology has brought new requirements for retrieval across different modalities. 
This research project has constructed a two-stage clustering-based retrieval framework, as well as a deep learning-based retrieval algorithm using the CLIP model, 
which demonstrates zero-shot ability in the multimodal domain. By employing two different styles of methods on classification datasets under various testing scenarios, 
this work analyzes and demonstrates the differences and similarities in terms of performance, computational requirements, and application environments of the retrieval algorithms.

Deep Learning method aims to apply multimodal method--CLIP to image retrieval task.
By adopting the zero-shot ablity demonstrated by CLIP, the proposed novel pipeline could
cope variant text input with different fine-grains. Besides, the Grad-CAM is introduced
here to present CLIP's explainability, even further rough text-guided object localization.

The statistic-based machine learning IR (image-retrieval) system
is combining SIFT/HOG/shape Histogram/PCA/Kmeans which is expected to present a good 
retrieval performance when facing serveral natural image candidates. And the overall framework is conbined two stages.

And in this study, two styled method would be compared and anlyzed based on experiments taken
on few well-known classfication dataset.
#####




Preparations:
--------------

        pip install -r requirements.txt

if tesing on cifar or voc2007, please read/download(can be done by torchvsion to make life easy) to dir: ./ciar10 or ./VOC2007




image retrieval
-----------------


1: refer to clip_ir.py and kmeans_ir.py

2: edit the correspoding arguments:

for CLIP-IR

    #i:      batch-size which stand for candidates images number
    #ii:     clip visual backbone type
    #iii:    similarity threshold to determine true/false
    #of course, and the datasets would be used to evalute or customized data path
    
    python clip_ir.py
    
for Kmeans-IR

    #i:      BoF-size which indicate when building the Kmeans model, the number of samples feed in each cluster
    #ii:     Batch-size which is same as the above 
    #iii:    num-cluster which is the known number of retrievable labels
    #of course, and the datasets would be used to evalute or customized data path
    
    python kmeans_ir.py
    
Then all set and ready to go.




CAM Visualization for CLIP
(zero-shot text-guided object localization tech)
-------------------------------------------------

1: edit cam_vis.py arguments：image_folder_path;               vis_res_folder_path

2: python cam_vis.py

3: follow the instruction and input image file name and the text want to retrieval

4： the cam result image will save in cam_vis folder



acknowledgement
---------------------------------
great thanks to the following works
https://github.com/openai/CLIP.git
https://github.com/aiiu-lab/CLIPCAM.git
