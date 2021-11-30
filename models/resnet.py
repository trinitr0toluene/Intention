import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 2

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1000):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv2d(n_input_channels,
                               self.in_planes,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=(3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.conv13 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        # self.conv23 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        # self.conv33 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)
        # self.maxpool333 = nn.AdaptiveMaxPool2d(output_size=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)

        # self.conv13 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        # self.y1 = self.conv13(layers[1])

        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        # self.conv23 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)

        # self.y1 += layers[2]
        # self.y2 = self.conv23(layers[2])
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        # self.conv33 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)

        # self.y2 += layers[3]
        # self.y3 = self.conv33(self.y3)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        # self.y3 += layers[4]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    # def upsample(self, in_plane,out_plane, stride):
    #     out = nn.Sequential(
    #                 conv1x1(self.in_planes, out_plane, stride),
    #                 nn.BatchNorm2d(out_plane))
    #     return out

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool2d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion))
                # downsample = conv1x1(self.in_planes, planes * block.expansion, stride)
                # downsample = self.bn3(downsample)
        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)
        # return layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        # layer1_result = x

        x = self.layer2(x)
        # layer2_result = x

        # y1 = self.upsample(layer1_result[0],layer2_result[0],stride=1)
        # y1 = self.maxpool(y1)
        # y1 += layer2_result

        x = self.layer3(x)
        # layer3_result = x

        # y2 = self.upsample(y1[0],layer3_result[0],stride = 1)
        # y2 = self.maxpool(y2)
        # y2 += layer3_result

        x = self.layer4(x)
        # layer4_result = x

        # y3 = self.upsample(y2[0],layer4_result[0],stride=1)
        # y3 = self.maxpool(y3)
        # y3 +=layer4_result

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model
