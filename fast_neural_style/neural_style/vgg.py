import torch.nn as nn


class Vgg16(nn.Module):

    def __init__(self):
        super(Vgg16, self).__init__()

    def forward(self, x):
        return x
