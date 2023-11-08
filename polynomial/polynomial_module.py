import torch
import torch.nn as nn


class Polynomial(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return x


def main():
    model = Polynomial()
    loss_fn = nn.MSELoss()


if __name__ == '__main__':
    main()
