"""preactresnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from quantizers.lsq import Conv2dLSQ, InputConv2dLSQ, LinearLSQ


class PreActBasic(nn.Module):

    expansion = 1
    def __init__(self, in_channels, out_channels, stride, bit=4):
        super().__init__()
        conv_layer = partial(Conv2dLSQ, bit=bit)

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            conv_layer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv_layer(out_channels, out_channels * PreActBasic.expansion, kernel_size=3, padding=1)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
            self.shortcut = conv_layer(in_channels, out_channels * PreActBasic.expansion, 1, stride=stride)

    def forward(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return res + shortcut


class PreActBottleNeck(nn.Module):

    expansion = 4
    def __init__(self, in_channels, out_channels, stride, bit=4):
        super().__init__()
        conv_layer = partial(Conv2dLSQ, bit=bit)

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            conv_layer(in_channels, out_channels, 1, stride=stride),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv_layer(out_channels, out_channels, 3, padding=1),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv_layer(out_channels, out_channels * PreActBottleNeck.expansion, 1)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * PreActBottleNeck.expansion:
            self.shortcut = conv_layer(in_channels, out_channels * PreActBottleNeck.expansion, 1, stride=stride)

    def forward(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return res + shortcut

class PreActResNet(nn.Module):

    def __init__(self, block, num_block, class_num=100, bit=4):
        super().__init__()
        self.input_channels = 64
        _OutputLinearLSQ = partial(LinearLSQ, bit=8)
        _InputConv2dLSQ = partial(InputConv2dLSQ, bit=8)

        self.pre = nn.Sequential(
            _InputConv2dLSQ(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_layers(block, num_block[0], 64,  1, bit=bit)
        self.stage2 = self._make_layers(block, num_block[1], 128, 2, bit=bit)
        self.stage3 = self._make_layers(block, num_block[2], 256, 2, bit=bit)
        self.stage4 = self._make_layers(block, num_block[3], 512, 2, bit=bit)

        self.linear = _OutputLinearLSQ(self.input_channels, class_num)

    def _make_layers(self, block, block_num, out_channels, stride, bit=4):
        layers = []

        layers.append(block(self.input_channels, out_channels, stride, bit=bit))
        self.input_channels = out_channels * block.expansion

        while block_num - 1:
            layers.append(block(self.input_channels, out_channels, 1, bit=bit))
            self.input_channels = out_channels * block.expansion
            block_num -= 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

def preactresnet18(bit=4):
    return PreActResNet(PreActBasic, [2, 2, 2, 2], bit=bit)

def preactresnet34(bit=4):
    return PreActResNet(PreActBasic, [3, 4, 6, 3], bit=bit)

def preactresnet50(bit=4):
    return PreActResNet(PreActBottleNeck, [3, 4, 6, 3], bit=bit)

def preactresnet101(bit=4):
    return PreActResNet(PreActBottleNeck, [3, 4, 23, 3], bit=bit)

def preactresnet152(bit=4):
    return PreActResNet(PreActBottleNeck, [3, 8, 36, 3], bit=bit)

