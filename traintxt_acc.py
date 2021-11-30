#coding=utf-8
import os
import json
import csv
import argparse
import pandas as pd
import numpy as np
from math import ceil
from tqdm import tqdm
import pickle
import shutil
from sklearn.metrics import f1_score
from typing import List, Tuple, Dict
from collections import OrderedDict
import json
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torchvision import datasets, models
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from transforms import transforms
from models.LoadModel import MainModel
from utils.dataset_DCL import collate_fn4train, collate_fn4test, collate_fn4val, dataset_train
from config import LoadConfig, load_data_transformers
from pytorch_transformers import BertModel, BertConfig, BertTokenizer
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 进行3D图像绘制



import math


import pdb

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

num_class = 28
text_class = {
    '0': 'Being good looking, attractive',
    '1': 'Beat people in a competition',
    '2': 'To communicate or express myself',
    '3': 'Being creative (e.g., artistically, scientifically, intellectually). Being unique or different',
    '4': 'Exploration - Being curious and adventurous. Having an exciting, stimulating life',
    '5': 'Having an easy and comfortable life',
    '6': 'Enjoying life',
    '7': 'Appreciating fine design (man-made wonders like architectures)',
    '8': 'Appreciating fine design (artwork)',
    '9': 'Appreciating other cultures',
    '10': 'Being a good parent (teaching, transmitting values). Being emotionally close to my children',
    '11': 'Being happy and content. Feeling satisfied with one’s life. Feeling good about myself',
    '12': 'Being ambitious, hard-working',
    '13': 'Achieving harmony and oneness (with self and the universe)',
    '14': 'Being physically active, fit, healthy, e.g. maintaining a healthy weight, eating nutritious foods. To be physically able to do my daily/routine activities. Having athletic ability',
    '15': 'Being in love',
    '16': 'Being in love with animal',
    '17': 'Inspiring others, Influencing, persuading others',
    '18': 'To keep things manageable. To make plans',
    '19': 'Experiencing natural beauty',
    '20': 'Being really passionate about something',
    '21': 'Being playful, carefree, lighthearted',
    '22': 'Sharing my feelings with others',
    '23': 'Being part of a social group. Having people to do things with. Having close friends, others to rely on. Making friends, drawing others near',
    '24': 'Being successful in my occupation. Having a good job',
    '25': 'Teaching others',
    '26': 'Keeping things in order (my desk, office, house, etc.)',
    '27': 'Having work I really like',
}
cos = nn.CosineSimilarity(dim=-1, eps=1e-10)


def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='intent', type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--b', dest='batch_size',
                        default=100, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        default=16, type=int)
    parser.add_argument('--ver', dest='version',
                        default='test', type=str)
    parser.add_argument('--save', dest='resume',
                        default=None, type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=448, type=int)
    parser.add_argument('--ss', dest='save_suffix',
                        default=None, type=str)
    parser.add_argument('--acc_report', dest='acc_report',
                        action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    # if args.submit:
    args.version = 'train'
    if args.save_suffix == '':
        raise Exception('**** miss --ss save suffix is needed. ')

    Config = LoadConfig(args, args.version)
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution)
    dataloader = {}
    train_set = dataset_train(Config=Config, \
                        anno=Config.train_anno, \
                        # common_aug=transformers["common_aug"], \
                        # swap = transformers["swap"],\
                        totensor=transformers["test_totensor"], \
                        train=True)

    dataloader['train'] = torch.utils.data.DataLoader(train_set, \
                                                      batch_size=args.batch_size, \
                                                      shuffle=True, \
                                                      num_workers=args.num_workers, \
                                                      collate_fn=collate_fn4test)

    setattr(dataloader['train'], 'total_item_len', len(train_set))

    cudnn.benchmark = True

    model = MainModel(Config)
    model_dict = model.state_dict()
    pretrained_dict = torch.load("/data/sqhy_model/new_model/hierarchy_82511_intent/weights_ce+loc_90_98.pth")
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    model = nn.DataParallel(model)

    model.train(False)
    with torch.no_grad():
        # val set

        predict = []
        val_labels = []
        all_num = 0
        sum_batch = []
        for batch_cnt_val, data_val in enumerate(dataloader["train"]):
            inputs, labels, img_name = data_val
            inputs = Variable(inputs.cuda())

            outputs = model(inputs)
            val_labels.extend(labels)
            predict.extend(-F.sigmoid(outputs[0]).cpu().numpy())   # 3 直线的结果, 取负的排列从大到小

        loc = np.argsort(predict, axis=1)     # 从大到小排列
        for k in range(len(val_labels)):
            mid_label = [i for i, j in enumerate(val_labels[k]) if j > 0]
            mid_loc = loc[k][:len(mid_label)]
            mid = 0
            for w in mid_loc:
                if w in mid_label:
                    mid = mid + 1
            mid = mid / len(mid_label)
            sum_batch.append(mid)
        sum_all = sum(sum_batch)/len(val_labels)
        print(sum_all)













