# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import pandas
import random
import PIL.Image as Image
from PIL import ImageStat
import numpy as np

import pdb

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(0)

def random_sample(img_names, labels):
    anno_dict = {}
    img_list = []
    anno_list = []
    for img, anno in zip(img_names, labels):
        if not anno['category_ids_softprob'] in anno_dict:
            anno_dict[anno['category_ids_softprob']] = [img['filename']]
        else:
            anno_dict[anno['category_ids_softprob']].append(img['filename'])

        # if not anno in anno_dict:
        #     anno_dict[anno] = [img]
        # else:
        #     anno_dict[anno].append(img)

    for anno['category_ids_softprob'] in anno_dict.keys():
        anno_len = len(anno_dict[anno])
        fetch_keys = random.sample(list(range(anno_len)), anno_len//10)
        img_list.extend([anno_dict[anno][x] for x in fetch_keys])
        anno_list.extend([anno for x in fetch_keys])
    # for anno in anno_dict.keys():
    #     anno_len = len(anno_dict[anno])
    #     fetch_keys = random.sample(list(range(anno_len)), anno_len//10)
    #     img_list.extend([anno_dict[anno][x] for x in fetch_keys])
    #     anno_list.extend([anno for x in fetch_keys])

    return img_list, anno_list



class dataset(data.Dataset):
    def __init__(self, Config, anno, common_aug=None, totensor=None, train=False, train_val=False, test=False):
        self.root_path = Config.rawdata_root
        self.numcls = Config.numcls
        self.dataset = Config.dataset
        if isinstance(anno, pandas.core.frame.DataFrame):
            self.paths = anno['ImageName'].tolist()
            self.labels = anno['label'].tolist()
        elif isinstance(anno, dict):
            self.paths = anno['annotations']
            self.labels = anno['annotations']
        # train_val 用的train数据集，用在后面val
        # if train_val:
        #     self.paths, self.labels = random_sample(self.paths, self.labels)
        self.common_aug = common_aug
        self.totensor = totensor
        self.cfg = Config
        self.train = train
        self.test = test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        ww = 'low/' + self.paths[item]['image_id'] + '.jpg'
        img_path = os.path.join(self.root_path, ww)
        img = self.pil_loader(img_path)
        img = self.common_aug(img)
        img = self.totensor(img)
        label = self.labels[item]['category_ids_softprob']
        return img, label, self.paths[item]['image_id']

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    # def crop_image(self, image, cropnum):
    #     width, high = image.size
    #     crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
    #     crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
    #     im_list = []
    #     for j in range(len(crop_y) - 1):
    #         for i in range(len(crop_x) - 1):
    #             im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
    #     return im_list


    def get_weighted_sampler(self):
        img_nums = len(self.labels)
        weights = [self.labels.count(x) for x in range(self.numcls)]
        return torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=img_nums)


def collate_fn4train(batch):
    imgs = []
    label = []
    label_swap = []
    law_swap = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[1])
        label.append(sample[2])
        label.append(sample[2])
        if sample[3] == -1:
            label_swap.append(1)
            label_swap.append(0)
        else:
            label_swap.append(sample[2])
            label_swap.append(sample[3])
        law_swap.append(sample[4])
        law_swap.append(sample[5])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, label_swap, law_swap, img_name

def collate_fn4val(batch):
    imgs = []
    label = []
    label_swap = []
    law_swap = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        if sample[3] == -1:
            label_swap.append(1)
        else:
            label_swap.append(sample[2])
        law_swap.append(sample[3])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, label_swap, law_swap, img_name

def collate_fn4backbone(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        if len(sample) == 8:
            label.append(sample[3])
        else:
            label.append(sample[1])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name


def collate_fn4test(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        img_name.append(sample[2])
    return torch.stack(imgs, 0), label, img_name



class dataset_test(data.Dataset):
    def __init__(self, Config, anno, common_aug=None, totensor=None, train=False, train_val=False, test=False):
        self.root_path = Config.rawdata_root
        self.numcls = Config.numcls
        self.dataset = Config.dataset
        if isinstance(anno, pandas.core.frame.DataFrame):
            self.paths = anno['ImageName'].tolist()
            self.labels = anno['label'].tolist()
        elif isinstance(anno, dict):
            self.paths = anno['annotations']
            self.labels = anno['annotations']
        # train_val 用的train数据集，用在后面val
        # if train_val:
        #     self.paths, self.labels = random_sample(self.paths, self.labels)
        # self.common_aug = common_aug
        self.totensor = totensor
        self.cfg = Config
        self.train = train
        self.test = test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = "low/" + self.paths[item]['image_id'] + ".jpg"
        img_path = os.path.join(self.root_path, img_path)
        img = self.pil_loader(img_path)
        if self.test:
            img = self.totensor(img)
            label = self.labels[item]['category_ids']
            return img, label, self.paths[item]['image_id']


    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def crop_image(self, image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list


    def get_weighted_sampler(self):
        img_nums = len(self.labels)
        weights = [self.labels.count(x) for x in range(self.numcls)]
        return torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=img_nums)


class dataset_train(data.Dataset):
    def __init__(self, Config, anno, common_aug=None, totensor=None, train=False, train_val=False, test=False):
        self.root_path = Config.rawdata_root
        self.numcls = Config.numcls
        self.dataset = Config.dataset
        if isinstance(anno, pandas.core.frame.DataFrame):
            self.paths = anno['ImageName'].tolist()
            self.labels = anno['label'].tolist()
        elif isinstance(anno, dict):
            self.paths = anno['annotations']
            self.labels = anno['annotations']
        # train_val 用的train数据集，用在后面val
        # if train_val:
        #     self.paths, self.labels = random_sample(self.paths, self.labels)
        # self.common_aug = common_aug
        self.totensor = totensor
        self.cfg = Config
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = "low/" + self.paths[item]['image_id'] + ".jpg"
        img_path = os.path.join(self.root_path, img_path)
        img = self.pil_loader(img_path)
        img = self.totensor(img)
        label = self.labels[item]['category_ids_softprob']
        text = self.labels[item]['caption'][0]
        return img, label, self.paths[item]['image_id'], text

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')