import os
import pandas as pd
import torch
import json

from transforms import transforms
# from utils.autoaugment import ImageNetPolicy

# pretrained model checkpoints
pretrained_model = {'resnet50': './models/pretrained/resnet50-19c8e357.pth',}

CLASS_15 = {
    '0': [24, 27],
    '1': [5],
    '2': [13],
    '3': [11],
    '4': [3, 7, 8, 9, 19],
    '5': [4, 6, 20, 21],
    '6': [14],
    '7': [15, 16],
    '8': [2, 22, 23],
    '9': [0],
    '10': [17, 25],
    '11': [10],
    '12': [1],
    '13': [12],
    '14': [18, 26],
}

CLASS_9 = {
    '0': [0, 1],
    '1': [2, 3],
    '2': [4, 5],
    '3': [6],
    '4': [7, 8, 9],
    "5": [10],
    '6': [11],
    '7': [12],
    '8': [13, 14]
}

# transforms dict
def load_data_transformers(resize_reso=512, crop_reso=224):
    Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
        'common_aug': transforms.Compose([
            # transforms.Resize((resize_reso, resize_reso)),
            # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
        ]),
        'train_totensor': transforms.Compose([
            # transforms.Resize((crop_reso, crop_reso)),
            # ImageNetPolicy(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val_totensor': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test_totensor': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            # transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'None': None,
    }
    return data_transforms


class LoadConfig(object):
    def __init__(self, args, version):
        if version == 'train':
            get_list = ['train', 'val', 'test']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test', 'val']
        else:
            raise Exception("train/val/test ???\n")

        ###############################
        #### add dataset info here ####
        ###############################

        # put image data in $PATH/data
        # put annotation txt file in $PATH/anno

        if args.dataset == 'intent':
            self.dataset = args.dataset
            self.rawdata_root = '/data/zzy_data/intent_resize'
            self.anno_root = '/home/zhangziyi/intentonomy/annotations'
            self.numcls = 28
        else:
            raise Exception('dataset not defined ???')


        if 'train' in get_list:
            with open(os.path.join(self.anno_root, "intent_train.json")) as f:
                 self.train_anno = json.load(f)

        if 'val' in get_list:
            with open(os.path.join(self.anno_root, "intentonomy_val2020.json")) as f:
                 self.val_anno = json.load(f)

        if 'test' in get_list:
            with open(os.path.join(self.anno_root, "intentonomy_test2020.json")) as f:
                 self.test_anno = json.load(f)

        self.save_dir = '/data/zzy_data/zzy_model/new_model'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.backbone = args.backbone
        self.use_backbone = True




