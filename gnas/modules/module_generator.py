from torch import nn
import torch
from modules.identity import Identity


def generate_dw_conv(in_channels, out_channels, kernel):
    padding = int((kernel - 1) / 2)
    conv1 = nn.Sequential(nn.ReLU(),
                          nn.Conv2d(in_channels, in_channels, kernel, padding=padding, groups=in_channels, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(),
                          nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
                          nn.BatchNorm2d(out_channels))
    return conv1

def dw_conv3x3(in_channels, out_channels):
    return generate_dw_conv(in_channels, out_channels, 3)


def dw_conv5x5(in_channels, out_channels):
    return generate_dw_conv(in_channels, out_channels, 5)

def dw_conv7x7(in_channels, out_channels):
    return generate_dw_conv(in_channels, out_channels, 7)

def max_pool3x3(in_channels, out_channels):
    return nn.Sequential(nn.ReLU(),
                         nn.MaxPool2d(3, stride=1, padding=1))

def avg_pool3x3(in_channels, out_channels):
    return nn.Sequential(nn.ReLU(),
                         nn.AvgPool2d(3, stride=1, padding=1))

def identity(in_channels, out_channels):
    return Identity()


class DilConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
                        nn.ReLU(inplace=False),
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=in_channels, bias=False),
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                        nn.BatchNorm2d(out_channels, affine=affine))

    def forward(self, x):
        return self.op(x)


def dilated_conv3x3(in_channels, out_channels):

    padding = 0
    Dil3_layer = DilConv(in_channels, out_channels, kernel_size=1, padding=2 * padding, dilation=2)
    return Dil3_layer


def dilated_conv5x5(in_channels, out_channels):
    padding = 2
    Dil5_layer = DilConv(in_channels, out_channels, kernel_size=3, padding=padding, dilation=2)

    return Dil5_layer


def dilated_conv7x7(in_channels, out_channels):
    padding = 4
    Dil7_layer = DilConv(in_channels, out_channels, kernel_size=5, padding=padding, dilation=2)
    return Dil7_layer


__op_dict__ = {
               'Dw3x3': dw_conv3x3,
               'Dw5x5': dw_conv5x5,
               'Dw7x7': dw_conv7x7,
               'Identity': identity,
               'Max3x3': max_pool3x3,
               'Avg3x3': avg_pool3x3,
               'Dc3x3': dilated_conv3x3,
               'Dc5x5': dilated_conv5x5,
               'Dc7x7': dilated_conv7x7,
               }


def generate_op(op_list, in_channels, out_channels):
    return [__op_dict__.get(nl)(in_channels, out_channels) for nl in op_list]
