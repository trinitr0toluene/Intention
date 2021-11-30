# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
from models.LoadModel import MainModel
from config import LoadConfig, load_data_transformers
import argparse
import torch
import torch.nn as nn
from utils.dataset_DCL import collate_fn4train, collate_fn4test, collate_fn4val, dataset_test,dataset_train
from math import ceil
import os
import gc

def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='intent', type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--b', dest='batch_size',
                        default=1, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        default=6, type=int)
    parser.add_argument('--ver', dest='version',
                        default='val', type=str)
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

# input image
# LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
# IMG_URL = '/media/alice/datafile/hongya/data/hmdb_rgb+flow/Bartender_of_America_Bartending_School_pour_u_cm_np1_fr_med_2/img_00021.jpg'
args = parse_args()
args.version = 'train'
Config = LoadConfig(args, args.version)
transformers = load_data_transformers(args.resize_resolution, args.crop_resolution)
dataloader = {}
test_set = dataset_train(Config,\
                       anno=Config.train_anno ,\
                       totensor=transformers['train_totensor'],\
                       test=True)

dataloader["train"] = torch.utils.data.DataLoader(test_set,\
                                             batch_size=args.batch_size,\
                                             shuffle=False,\
                                             num_workers=args.num_workers,\
                                             collate_fn=collate_fn4test)

setattr(dataloader["train"], 'total_item_len', len(test_set))

# cudnn.benchmark = True

net = MainModel(Config)
model_dict=net.state_dict()
pretrained_dict=torch.load("/data/sqhy_model/new_model/hierarchy_103116_intent/weights_ce+loc_51_98.pth")
pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)
# print(net)
# net.cuda()
# net = nn.DataParallel(net)

net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    # print(output)
    features_blobs.append(output.data.cpu().numpy())
    # print(features_blobs)
    # print("11111111")

net.model._modules["7"]._modules["2"].register_forward_hook(hook_feature)  # 额外分支的分类层前一个conv4
# features_blobs = list(net.model._modules["7"]._modules["2"].conv3.parameters())

# # get the softmax weight

weight_softmax = list(net.classifier.parameters())
weight_softmax = np.squeeze(weight_softmax[0].cpu().data.numpy())
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (448, 448)
    bz, nc,h,w = feature_conv.shape
    # print(bz)
    # print(nc)
    # print(h)
    # ww = np.squeeze(feature_conv)
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc,h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)   #这两行是做归一化，减去最小值，除以最大值，
        cam_img = np.uint8(255 * cam_img)   #恢复到输入的图片尺寸
        output_cam.append(cv2.resize(cam_img, size_upsample))
        cam_img = np.uint8(255 * cam)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

with torch.no_grad():

    # val_size = ceil(len(test_set) / dataloader.batch_size) # ceil向上取整
    result_gather = {}
    n = 0

    for batch_cnt_val, data_val in enumerate(dataloader["train"]):
        inputs, labels, img_name = data_val
        n = n + 1
        if n >= 10000:
            outputs = net(inputs)
            # for i in range(len(outputs[0])):
            h_x = F.softmax(outputs[-1]).data.squeeze()
            probs, idx = h_x.sort(0, True)
            # probs = probs.numpy()
            idx = idx.numpy()

            # net.model._modules["7"]._modules["2"].conv3.register_forward_hook(hook_feature)  # 额外分支的分类层前一个conv4
            # weight_softmax = list(net.classifier.parameters())
            # weight_softmax = np.squeeze(weight_softmax[0].cpu().data.numpy())
            # print(len(features_blobs))
            CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
            image_path = os.path.join("/data/sqhy_data/intent_resize/low", img_name[0]+".jpg")
            print(image_path)
            img = cv2.imread(image_path)
            height, width, _ = img.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img * 0.5
            ww = image_path.split("/")
            new_path = '/data/sqhy_data/intent_cam/ce+train/' + ww[-1]
            # print(new_path)
            cv2.imwrite(new_path, result)
                # cv2.imshow('CAM.jpg')
            print(n)
        else:
            print("end")






# def returnCAM(feature_conv, weight_softmax, class_idx):
#     # generate the class activation maps upsample to 256x256
#     size_upsample = (256, 256)
#     bz, nc, h, w = feature_conv.shape
#     output_cam = []
#     for idx in class_idx:
#         cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
#         # cam = cam.reshape(h, w)
#         # cam = cam - np.min(cam)
#         # cam_img = cam / np.max(cam)   这两行是做归一化，减去最小值，除以最大值，
#         # cam_img = np.uint8(255 * cam_img)   恢复到输入的图片尺寸
#         # output_cam.append(cv2.resize(cam_img, size_upsample))
#         cam_img = np.uint8(255 * cam)
#         output_cam.append(cv2.resize(cam_img, size_upsample))
#     return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

# response = requests.get(IMG_URL)
# img_pil = Image.open(io.BytesIO(response.content))
# img_pil = Image.open(IMG_URL)
# img_pil.save('test.jpg')
#
# img_tensor = preprocess(img_pil)
# img_variable = Variable(img_tensor.unsqueeze(0).unsqueeze(0))
# logit = net(img_variable)
#
# # download the imagenet category list
# # classes = {int(key):value for (key, value)
# #           in requests.get(LABELS_URL).json().items()}
#
# h_x = F.softmax(logit, dim=1).data.squeeze()
# probs, idx = h_x.sort(0, True)
# # probs = probs.numpy()
# idx = idx.numpy()
#
# # output the prediction
# # for i in range(0, 5):
# #     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
#
# # generate class activation mapping for the top1 prediction
# CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
#
# # render the CAM and output
# # print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
# img = cv2.imread('test.jpg')
# height, width, _ = img.shape
# heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
# result = heatmap * 0.3 + img * 0.5
# cv2.imwrite('CAM.jpg', result)

