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

        self.quan_w = LSQQuantizer(bit=8, is_activation=False)
        self.quan_a = LSQQuantizer(bit=8, is_activation=True)

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

        self.quan_w = LSQQuantizer(bit=8, is_activation=False)
        self.quan_a = LSQQuantizer(bit=8, is_activation=False)

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

class SWLinearLSQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SWLinearLSQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.quan_w = LSQQuantizer(bit=8, is_activation=False)
        self.quan_a = LSQQuantizer(bit=8, is_activation=True)

    def set_quantizer_runtime_bitwidth(self, bit):
        self.quan_w.set_quantizer_runtime_bitwidth(bit)
        self.quan_a.set_quantizer_runtime_bitwidth(bit)

    def forward(self, x):
        if self.bit == 32:
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(self.quan_a(x), self.quan_w(self.weight), self.bias)