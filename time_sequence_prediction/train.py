import numpy as np
import torch


def main():
    steps = 15
    np.random.seed(0)
    torch.manual_seed(0)
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    print(input)
    print(target)


if __name__ == '__main__':
    main()
