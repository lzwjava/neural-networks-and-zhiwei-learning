import torch


def main():
    POLY_DEGREE = 4
    W_target = torch.randn(POLY_DEGREE, 1) * 5
    b_target = torch.randn(1) * 5
    print(W_target)
    print(b_target)


if __name__ == '__main__':
    main()
