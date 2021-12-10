import numpy as np
from torch import nn
from torch.nn import init
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels
from models import resnet

from config import pretrained_model
from pytorch_transformers import BertModel, BertConfig, BertTokenizer

import pdb

num_classes = 28
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

def weights_init_text(m):
    m.weight.data.uniform_(-0.1, 0.1)

def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features + fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)

class MainModel(nn.Module):
    def __init__(self, config):
        super(MainModel, self).__init__()
        self.num_classes = config.numcls
        self.backbone_arch = config.backbone
        # self.use_Asoftmax = config.use_Asoftmax
        print(self.backbone_arch)


        if self.backbone_arch in dir(models):
            self.model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], resnet.get_inplanes()) # [3,4,6,4]是resnet50网络结构中每一层的block数量, get_inplanes是特征通道的数量，也就是卷积核的数量
            model_dict = self.model.state_dict()
            # print(self.model.name_parameters())
            # self.model = getattr(models, self.backbone_arch)()
            if self.backbone_arch in pretrained_model:
                pretrain = torch.load(pretrained_model[self.backbone_arch])
                # pretrain = {k: v for k, v in pretrain['state_dict'].items() if k in self.model.state_dict() and k[:6] == 'layer4' or k[:2] == 'fc'}
                model_dict.update(pretrain)
                self.model.load_state_dict(model_dict)
                # pretrain = torch.load(pretrained_model[self.backbone_arch])
                # self.model.load_state_dict(pretrain['state_dict'])
                # self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
        else:
            if self.backbone_arch in pretrained_model:
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained=None)
            else:
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000)

        if self.backbone_arch == 'resnet50' or self.backbone_arch == 'se_resnet50':
            # print(self.model)
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            # print(self.model)
        if self.backbone_arch == 'senet154':
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if self.backbone_arch == 'se_resnext101_32x4d':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'se_resnet101':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        # # original classification
        pai = 0.01
        self.classifier = nn.Linear(2048, self.num_classes, bias=True)
        nn.init.constant_(self.classifier.bias, -np.log((1 - pai)/pai))


    def forward(self, x):
        out = []
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        out.append(self.classifier(x))    # 正常的28分类
        # out.append(x)   # 画特征图时用
        return out

