import torch as t
import torch.nn.functional as F
import torch.nn as nn
import torch
import math


__all__ = ['SWConv2dLSQ', 'InputSWConv2dLSQ', 'SWLinearLSQ']

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LSQQuantizer(t.nn.Module):
    def __init__(self, is_activation=False):
        super(LSQQuantizer,self).__init__()

        self.alpha = nn.Parameter(torch.Tensor(1))
        self.is_activation = is_activation
        self.register_buffer('init_state', torch.zeros(1))        
        
    def set_quantizer_runtime_bitwidth(self, bit):
        self.bit = bit

    def forward(self, x):
        if is_activation:
            Qn = 0
            Qp = 2 ** self.bit - 1
        else:
            Qn = -2 ** (self.bit - 1)
            Qp = 2 ** (self.bit - 1) - 1

        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.detach().abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        
        g = 1.0 / math.sqrt(x.numel() * Qp)
        _alpha = grad_scale(self.alpha, g)
        x_q = round_pass((x / _alpha).clamp(Qn, Qp)) * _alpha
        return x_q

    def __repr__(self):
        return "LSQQuantizer (bit=%s, is_activation=%s)" % (self.bit, self.is_activation)


class SWConv2dLSQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):


        super(SWConv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.quan_w = LSQQuantizer(is_activation=False)
        self.quan_a = LSQQuantizer(is_activation=True)

    def set_quantizer_runtime_bitwidth(self, bit):
        self.quan_w.set_quantizer_runtime_bitwidth(bit)
        self.quan_a.set_quantizer_runtime_bitwidth(bit)


    def forward(self, x):
        if self.bit == 32:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(self.quan_a(x), self.quan_w(self.weight), self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

class InputSWConv2dLSQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):


        super(InputSWConv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.quan_w = LSQQuantizer(is_activation=False)
        self.quan_a = LSQQuantizer(is_activation=False)

    def set_quantizer_runtime_bitwidth(self, bit):
        self.quan_w.set_quantizer_runtime_bitwidth(bit)
        self.quan_a.set_quantizer_runtime_bitwidth(bit)

    def forward(self, x):
        if self.bit == 32:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(self.quan_a(x), self.quan_w(self.weight), self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

bits_list = [8,6,5,4]
switchbn = True
class SWBatchNorm2d(nn.Module):
    def __init__(self, num_features,
                 eps=1e-05, momentum=0.1,
                 affine=True):
        super(SWBatchNorm2d, self).__init__()
        self.num_features = num_features
        bns = []
        for i in range(len(bits_list)):
            bns.append(nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine))
        self.bn = nn.ModuleList(bns)
        self.bit = bits_list[-1]
        self.ignore_model_profiling = True
        if affine:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        else:
            self.weight = nn.Parameter(torch.Tensor(self.num_features))
            self.bias = nn.Parameter(torch.Tensor(self.num_features))
        self.affine = affine

    def set_quantizer_runtime_bitwidth(self, bit):
        self.bit = bit

    def forward(self, input):
        if switchbn:
            if self.bit in bits_list:
                idx = bits_list.index(self.bit)
            else:
                idx = 0
            y = self.bn[idx](input)
            if not self.affine:
                y = self.weight[None, :, None, None] * y + self.bias[None, :, None, None]
        else:
            y = self.bn[0](input)
        return y

class SWLinearLSQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SWLinearLSQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.quan_w = LSQQuantizer(is_activation=False)
        self.quan_a = LSQQuantizer(is_activation=True)

    def set_quantizer_runtime_bitwidth(self, bit):
        self.quan_w.set_quantizer_runtime_bitwidth(bit)
        self.quan_a.set_quantizer_runtime_bitwidth(bit)

    def forward(self, x):
        if self.bit == 32:
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(self.quan_a(x), self.quan_w(self.weight), self.bias)

