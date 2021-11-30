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
from utils.dataset_DCL import collate_fn4train, collate_fn4test, collate_fn4val, dataset_test
from config import LoadConfig, load_data_transformers

import pdb

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

def compute_f1(
        multihot_targets: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> Tuple[float, float, float]:
    # change scores to predict_labels
    predict_labels = scores > threshold
    predict_labels = predict_labels.astype(np.int)

    # get f1 scores
    f1 = {}
    f1["micro"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="micro"
    )
    f1["samples"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="samples"
    )
    f1["macro"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="macro"
    )
    f1["none"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average=None
    )
    return f1["micro"], f1["samples"], f1["macro"], f1["none"]


def multihot(x: List[List[int]], nb_classes: int) -> np.ndarray:
    """transform to multihot encoding

    Arguments:
        x: list of multi-class integer labels, in the range
            [0, nb_classes-1]
        nb_classes: number of classes for the multi-hot vector

    Returns:
        multihot: multihot vector of type int, (num_samples, nb_classes)
    """
    num_samples = len(x)

    multihot = np.zeros((num_samples, nb_classes), dtype=np.int32)
    for idx, labs in enumerate(x):
        for lab in labs:
            multihot[idx, lab] = 1

    return multihot.astype(np.int)

def get_best_f1_scores(
    multihot_targets: np.ndarray,
    scores: np.ndarray,
    threshold_end: float = 0.05
) -> Dict[str, float]:
    """
    get the optimal macro f1 score by tuning threshold
    """
    thrs = np.linspace(
        threshold_end, 0.95, int(np.round((0.95 - threshold_end) / 0.05)) + 1,
        endpoint=True
    )
    f1_micros = []
    f1_macros = []
    f1_samples = []
    f1_none = []
    for thr in thrs:
        _micros, _samples, _macros, _none = compute_f1(multihot_targets, scores, thr)
        f1_micros.append(_micros)
        f1_samples.append(_samples)
        f1_macros.append(_macros)
        f1_none.append(_none)

    f1_macros_m = max(f1_macros)
    b_thr = np.argmax(f1_macros)

    f1_micros_m = f1_micros[b_thr]
    f1_samples_m = f1_samples[b_thr]
    f1_none_m = f1_none[b_thr]
    f1 = {}
    f1["micro"] = f1_micros_m
    f1["macro"] = f1_macros_m
    f1["samples"] = f1_samples_m
    f1["threshold"] = thrs[b_thr]
    f1["none"] = f1_none_m
    return f1

if __name__ == '__main__':
    args = parse_args()
    print(args)
    # if args.submit:
    args.version = 'test'
    if args.save_suffix == '':
        raise Exception('**** miss --ss save suffix is needed. ')

    Config = LoadConfig(args, args.version)
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution)
    dataloader = {}
    test_set = dataset_test(Config,\
                       anno=Config.test_anno ,\
                       # swap=transformers["None"],\
                       # swap=transformers["None"],\
                       totensor=transformers['test_totensor'],\
                       test=True)

    dataloader["test"] = torch.utils.data.DataLoader(test_set,\
                                             batch_size=args.batch_size,\
                                             shuffle=False,\
                                             num_workers=args.num_workers,\
                                             collate_fn=collate_fn4test)

    setattr(dataloader["test"], 'total_item_len', len(test_set))

    val_set = dataset_test(Config, \
                            anno=Config.val_anno, \
                            # unswap=transformers["None"], \
                            # swap=transformers["None"], \
                            totensor=transformers['val_totensor'], \
                            test=True)

    dataloader["val"] = torch.utils.data.DataLoader(val_set, \
                                             batch_size=args.batch_size, \
                                             shuffle=False, \
                                             num_workers=args.num_workers, \
                                             collate_fn=collate_fn4test)

    setattr(dataloader["val"], 'total_item_len', len(val_set))

    cudnn.benchmark = True

    model = MainModel(Config)
    model_dict=model.state_dict()
    pretrained_dict=torch.load("./net_model/cross_entropy_71517_intent/weights_lr0.001_34_98.pth")
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    model = nn.DataParallel(model)

    model.train(False)
    with torch.no_grad():
        count_bar = tqdm(total=dataloader["test"].__len__())
        count = 0
        test_annotation = defaultdict()
        test_annotation['annotations'] = []
        for batch_cnt_test, data_test in enumerate(dataloader["test"]):
            count_bar.update(1)
            inputs_test, labels_test, img_name_test = data_test
            inputs_test = Variable(inputs_test.cuda())

            outputs_test = model(inputs_test)
            predict_test = F.softmax(outputs_test[0])

            for m in range(len(inputs_test)):
                test = dict()
                test['id'] = m + count * dataloader['test'].batch_size
                test['image_id'] = img_name_test[m]
                test['test_score'] = np.asarray(predict_test[m].cpu()).tolist()
                test['test_target'] = labels_test[m]

                test_annotation['annotations'].append(dict(test))
            count = count + 1

        test_json = json.dumps(test_annotation, indent=1)
        with open('test_result_ce_0.001_34.json', 'a+') as json_file:
            json_file.write(test_json)

    # val set
        # val_size = ceil(len(val_set) / dataloader["val"].batch_size)  # ceil向上取整
        # result_gather = {}
        count_bar1 = tqdm(total=dataloader["val"].__len__())
        count1 = 0
        val_annotation = defaultdict()
        val_annotation['annotations'] = []
        for batch_cnt_val, data_val in enumerate(dataloader["val"]):
            count_bar1.update(1)
            inputs, labels, img_name = data_val
            inputs = Variable(inputs.cuda())

            outputs = model(inputs)
            predict = F.softmax(outputs[0])

            for m in range(len(inputs)):
                val = dict()
                val['id'] = m + count1 * dataloader['val'].batch_size
                val['image_id'] = img_name[m]
                val['val_score'] = np.asarray(predict[m].cpu()).tolist()
                val['val_target'] = labels[m]

                val_annotation['annotations'].append(dict(val))
            count1 = count1 + 1

        val_json = json.dumps(val_annotation, indent=1)
        with open('val_result_ce_0.001_34.json', 'a+') as json_file:
            json_file.write(val_json)





