import torch
import torch.nn as nn


class Polynomial(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self(Polynomial, self).__init__()
        print(torch.rand())
        self.a = nn.Parameter(torch.rand())

    def forward(self):
        pass


model = Polynomial()
