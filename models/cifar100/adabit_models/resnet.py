"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from quantizers.adabit import SwitchBN2d,  QConv2d, QLinear
from utils.config import FLAGS


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels , kernel_size=1, stride=1, bias=False):
        super().__init__()
        self.conv1 = QConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.bn1 = SwitchBN2d(out_channels, affine=not getattr(FLAGS, 'stats_sharing', False))

    def forward(self, x):
        return self.bn1(self.conv1(x))


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.conv1 = QConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = SwitchBN2d(out_channels, affine=not getattr(FLAGS, 'stats_sharing', False))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = QConv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = SwitchBN2d(out_channels * BasicBlock.expansion, affine=not getattr(FLAGS, 'stats_sharing', False))

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = Shortcut(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return nn.ReLU(inplace=True)(out + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = QConv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = SwitchBN2d(out_channels, affine=not getattr(FLAGS, 'stats_sharing', False))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = QConv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn2 = SwitchBN2d(out_channels, affine=not getattr(FLAGS, 'stats_sharing', False))
        self.conv3 = QConv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False)
        self.bn3 = SwitchBN2d(out_channels * BottleNeck.expansion, affine=not getattr(FLAGS, 'stats_sharing', False))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = Shortcut(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        return nn.ReLU(inplace=True)(out + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64
            
        self.conv1 = QConv2d(3, 64, kernel_size=3, padding=1, bias=False, bitw_min=max(FLAGS.bits_list), bita_min=8, weight_only=True)
        self.bn1 = SwitchBN2d(64, affine=not getattr(FLAGS, 'stats_sharing', False))
        self.relu = nn.ReLU(inplace=True)

        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = QLinear(512 * block.expansion, num_classes, bitw_min=max(FLAGS.bits_list))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(out)
        output = self.relu(out)
        
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

    def switch_precision(self, bit):
        self.current_bit = bit
        for n, m in self.named_modules():
            if type(m) in (QConv2d, SwitchBN2d): # no change for the first and last layer
                m.set_quantizer_runtime_bitwidth(bit)

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])
