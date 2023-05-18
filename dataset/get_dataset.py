import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset
from dataset.voc import Voc2007Classification, Voc2007ClassificationTest
from dataset.openimage import openimage_dataset
from randaugment import RandAugment
import os

def get_datasets(args):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_size = 224
    train_data_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                            RandAugment(),
                                            transforms.ToTensor(),
                                            normalize])

    test_data_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    
    if args.dataset == 'coco' or args.dataset == 'coco14':
        dataset_dir = args.dataset_dir
        train_dataset = CoCoDataset(
            image_dir=os.path.join(dataset_dir,'train2014'),
            anno_path=os.path.join(dataset_dir, 'annotations/instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path='data/coco/train_label_vectors_coco14.npy',
            #missing_label=args.missing_label
        )

        val_dataset = CoCoDataset(
            image_dir=os.path.join(dataset_dir,'val2014'),
            anno_path=os.path.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )    

    elif args.dataset == 'voc' or args.dataset == 'voc2007':
        dataset_dir = args.dataset_dir
        train_dataset = Voc2007Classification(
            root=dataset_dir, set='trainval', transform=train_data_transform
        )
        val_dataset = Voc2007Classification(
            root=dataset_dir, set='test', transform=test_data_transform
        )

    elif args.dataset == 'openimage':
        dataset_dir = args.dataset_dir
        train_dataset = openimage_dataset(root=dataset_dir, annFile='/home/notebook/code/personal/S9051045/q2l_distill/lib/dataset/openimage_train_full.txt',transform=train_data_transform,class_num=args.num_class)
        val_dataset = openimage_dataset(root=dataset_dir, annFile='/home/notebook/code/personal/S9051045/q2l_distill/lib/dataset/openimage_test_full.txt',transform=test_data_transform,class_num=args.num_class)
    
    else:
        raise NotImplementedError("Unknown dataset %s" % args.dataset)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))

    return train_dataset, val_dataset
