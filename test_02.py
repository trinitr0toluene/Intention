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
                        default=10, type=int)
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
    val_targets = []
    val_scores = []
    test_targets = []
    test_scores = []

    # val
    with open("./val_result_ce_0.001_34.json") as f:
        val_result = json.load(f)
    val_mid = val_result['annotations']
    for i in range(len(val_mid)):
        val_targets.append(val_mid[i]['val_target'])
        val_scores.append(val_mid[i]['val_score'])
    val_scores = np.array(val_scores)

    # test
    with open("./test_result_ce_0.001_34.json") as f:
        test_result = json.load(f)
    test_mid = test_result['annotations']
    for i in range(len(test_mid)):
        test_targets.append(test_mid[i]['test_target'])
        test_scores.append(test_mid[i]['test_score'])
    test_scores = np.array(test_scores)

    # get optimal threshold using val set
    multihot_targets = multihot(val_targets, 28)
    f1_dict = get_best_f1_scores(multihot_targets, val_scores)

    multihot_targets = multihot(test_targets, 28)
    test_micro, test_samples, test_macro, test_none = compute_f1(multihot_targets, test_scores,f1_dict["threshold"])

    print(f1_dict)
    print(test_micro)
    print(test_samples)
    print(test_macro)
    print(test_none)










