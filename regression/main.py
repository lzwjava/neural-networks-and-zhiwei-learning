import torch
from itertools import count

POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5


def poly_desc(W, b):
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f}x^{} '.format(w, i + 1)
    result += '{:+.2f}'.format(b[0])
    return result


def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE + 1)], 1)


def f(x):
    return x.mm(W_target) + b_target.item()


def get_batch(batch_size=32):
    random = torch.randn(32)
    x = make_features(random)
    y = f(x)
    return x, y


def train():
    fc = torch.nn.Linear(W_target.size(0), 1)
    for batch_idx in count(1):
        batch_x, batch_y = get_batch()


def main():
    print(W_target)
    print(b_target)

    desc = poly_desc(W_target.view(-1), b_target)
    print(desc)

    batch = get_batch()
    print(batch)


if __name__ == '__main__':
    main()
