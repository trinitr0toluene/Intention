#coding=utf-8
import os
import datetime
import argparse
import logging
import pandas as pd

import torch
import torch.nn as nn
from  torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from transforms import transforms
from utils.train_model import train
from models.LoadModel import MainModel
from config import LoadConfig, load_data_transformers
from utils.dataset_DCL import collate_fn4train, collate_fn4val, collate_fn4test, collate_fn4backbone, dataset,dataset_test
# import pytorch_warmup as warmup
from warmup_scheduler import GradualWarmupScheduler
import warnings
import numpy as np
import random
import pdb

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机数种子
# setup_seed(38)


# parameters setting
def parse_args():
    parser = argparse.ArgumentParser(description='intention parameters')
    parser.add_argument('--data', dest='dataset',
                        default='intent', type=str)
    # parser.add_argument('--save', dest='resume',
    #                     default='/data/sqhy_model/new_model/hierarchy_82415_intent/weights_ce+loc_26_98.pth',type=str)
    parser.add_argument('--save', dest='resume',
                        default=None, type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--auto_resume', dest='auto_resume',
                        action='store_true')
    parser.add_argument('--epoch', dest='epoch',
                        default=200, type=int)
    parser.add_argument('--tb', dest='train_batch',
                        default=128, type=int)
    parser.add_argument('--vb', dest='val_batch',
                        default=128, type=int)
    parser.add_argument('--sp', dest='save_point',
                        default=5000, type=int)
    parser.add_argument('--cp', dest='check_point',
                        default=5000, type=int)
    parser.add_argument('--lr', dest='base_lr',
                        default=1e-3, type=float)
    parser.add_argument('--lr_step', dest='decay_step',
                        default=60, type=int)
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio',
                        default=10.0, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=1,  type=int)
    parser.add_argument('--tnw', dest='train_num_workers',
                        default=6, type=int)
    parser.add_argument('--vnw', dest='val_num_workers',
                        default=6, type=int)
    parser.add_argument('--detail', dest='discribe',
                        default='hierarchy', type=str)      # 模型文件夹名称
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=224, type=int)
    args = parser.parse_args()
    return args

def auto_load_resume(load_dir):
    folders = os.listdir(load_dir)
    date_list = [int(x.split('_')[1].replace(' ',0)) for x in folders]
    choosed = folders[date_list.index(max(date_list))]
    weight_list = os.listdir(os.path.join(load_dir, choosed))
    acc_list = [x[:-4].split('_')[-1] if x[:7]=='weights' else 0 for x in weight_list]
    acc_list = [float(x) for x in acc_list]
    choosed_w = weight_list[acc_list.index(max(acc_list))]
    return os.path.join(load_dir, choosed, choosed_w)


if __name__ == '__main__':
    args = parse_args()
    print(args, flush=True)
    Config = LoadConfig(args, 'train')
    setup_seed(38)
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution)

    # inital dataloader
    train_set = dataset(Config=Config,\
                        anno=Config.train_anno,\
                        common_aug=transformers["common_aug"],\
                        totensor=transformers["train_totensor"],\
                        train=True)

    val_set = dataset_test(Config=Config,\
                           anno=Config.val_anno,\
                           totensor=transformers["test_totensor"],\
                           test=True)

    test_set = dataset_test(Config, \
                            anno=Config.test_anno, \
                            totensor=transformers['test_totensor'], \
                            test=True)

    dataloader = {}
    dataloader["test"] = torch.utils.data.DataLoader(test_set, \
                                                     batch_size=args.val_batch, \
                                                     shuffle=False, \
                                                     num_workers=args.train_num_workers, \
                                                     collate_fn=collate_fn4test)
    setattr(dataloader["test"], 'total_item_len', len(test_set))

    dataloader['train'] = torch.utils.data.DataLoader(train_set,\
                                                      batch_size=args.train_batch,\
                                                      shuffle=True,\
                                                      num_workers=args.train_num_workers,\
                                                      collate_fn=collate_fn4test,
                                                      drop_last=True if Config.use_backbone else False,
                                                      pin_memory=True)

    setattr(dataloader['train'], 'total_item_len', len(train_set))

    dataloader['val'] = torch.utils.data.DataLoader(val_set,\
                                                    batch_size=args.val_batch,\
                                                    shuffle=False,\
                                                    num_workers=args.val_num_workers,\
                                                    collate_fn=collate_fn4test)

    setattr(dataloader['val'], 'total_item_len', len(val_set))
    setattr(dataloader['val'], 'num_cls', Config.numcls)


    cudnn.benchmark = True

    print('Choose model and train set', flush=True)
    model = MainModel(Config)

    # load model
    if (args.resume is None) and (not args.auto_resume):
        print('train from imagenet pretrained models ...', flush=True)
    else:
        if not args.resume is None:
            resume = args.resume
            print('load from pretrained checkpoint %s ...'% resume, flush=True)
        elif args.auto_resume:
            resume = auto_load_resume(Config.save_dir)
            print('load from %s ...'%resume, flush=True)
        else:
            raise Exception("no checkpoints to load")

        model_dict = model.state_dict()
        pretrained_dict = torch.load(resume)
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    print('Set cache dir', flush=True)
    time = datetime.datetime.now()
    filename = '%s_%d%d%d_%s'%(args.discribe, time.month, time.day, time.hour, Config.dataset)
    save_dir = os.path.join(Config.save_dir, filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.cuda()
    model = nn.DataParallel(model)

    ignored_params = list(map(id, model.module.classifier.parameters()))
    print('the num of new layers:', len(ignored_params), flush=True)
    base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())

    lr_ratio = args.cls_lr_ratio
    base_lr = args.base_lr
    optimizer = optim.SGD([{'params': base_params},
                           {'params': model.module.classifier.parameters(), 'lr': base_lr}], lr=base_lr, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=1)
    # warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=5)
    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=exp_lr_scheduler)
    # https://github.com/ildoonet/pytorch-gradual-warmup-lr

    # train entry
    train(Config,
          model,
          epoch_num=args.epoch,
          start_epoch=args.start_epoch,
          optimizer=optimizer,
          base_lr = base_lr,
          exp_lr_scheduler=exp_lr_scheduler,
          warmup_scheduler=warmup_scheduler,
          data_loader=dataloader,
          save_dir=save_dir,
          data_size=args.crop_resolution,
          savepoint=args.save_point,
          checkpoint=args.check_point)


