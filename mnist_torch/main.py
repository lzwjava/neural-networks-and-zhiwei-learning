import torch
import torch.nn as nn
from torchvision import datasets

def main():
    dataset1 = datasets.MNIST('../data', train = True, download=True)
    print(dataset1)
    pass

if __name__ == '__main__':
    main()
        