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
    pretrained_dict = torch.load("/data/sqhy_model/new_model/hierarchy_82511_intent/weights_ce+loc_98_98.pth")
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    model = nn.DataParallel(model)

    with torch.no_grad():
        val_output = []
        val_labels = []
        val_feature = []
        new_feature = []
        new_labels = []

        for batch_cnt_val, data_val in enumerate(dataloader["train"]):
            inputs, labels, img_name= data_val
            inputs = Variable(inputs.cuda())

            outputs = model(inputs)
            val_labels.extend(labels)
            val_output.extend(outputs[1])

        for k in range(len(val_labels)):
            mid_label = [i for i, j in enumerate(val_labels[k]) if j > 0]
            for w in range(len(mid_label)):
                new_feature.append(val_output[k])
                new_labels.append(mid_label[w])

            # val_labels.append(mid_label)


        writer = SummaryWriter()        # tensorboard   TSEN
        new_feature = torch.stack(new_feature,dim=0)
        # val_feature = torch.cat(val_feature)
        writer.add_embedding(new_feature, metadata=new_labels)
        writer.close()


        # tsne = TSNE(n_components=3, init='pca', random_state=0)   # sklearn TSEN
        # new_feature = torch.stack(new_feature, dim=0)
        # result = tsne.fit_transform(new_feature.cpu())
        # # x_min, x_max = new_feature.min(0), new_feature.max(0)
        # # X_norm = (new_feature - x_min) / (x_max - x_min)  # 归一化
        # # X_norm = F.normalize(new_feature, dim=1)
        # # plt.figure(figsize=(4, 4))
        # # for i in range(X_norm.shape[0]):
        # #     plt.text(X_norm[i, 0], X_norm[i, 1], str(new_labels[i]), color=plt.cm.Set1(new_labels[i]),
        # #              fontdict={'weight': 'bold', 'size': 9})
        # # plt.xticks([])
        # # plt.yticks([])
        # # plt.show()
        #
        # x_min, x_max = np.min(result, 0), np.max(result, 0)
        # result = (result - x_min) / (x_max - x_min)
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(result[:, 0], result[:, 1], result[:, 2],
        #            c=plt.cm.Set1(new_labels[:23828:]))
        # # 关闭了plot的坐标显示
        # plt.axis('off')
        # plt.show()











