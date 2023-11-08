import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


class Polynomial(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return x


def plot(x, y):
    print(x)
    print(y)

    x = x.numpy()
    y = y.numpy()

    plt.plot(x, y)
    plt.show()


def main():
    model = Polynomial()
    loss_fn = nn.MSELoss()

    step = 0
    lr = 1e-5

    optimizer = optim.SGD(model.parameters(), lr)

    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    # plot(x, y)

    while True:
        pass


if __name__ == '__main__':
    main()
