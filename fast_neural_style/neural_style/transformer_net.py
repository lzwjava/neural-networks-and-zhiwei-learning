import torch
import torch.nn as nn


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()

    def forward(self, x):
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

    def forward(self, x):
        return x


class UpsampleConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()

    def forward(self, x):
        return x
