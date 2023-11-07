import torch


def poly_desc(W, b):
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f}x^{} '.format(w, i + 1)
    result += '{:+.2f}'.format(b[0])
    return result


def main():
    POLY_DEGREE = 4
    W_target = torch.randn(POLY_DEGREE, 1) * 5
    b_target = torch.randn(1) * 5
    print(W_target)
    print(b_target)

    desc = poly_desc(W_target.view(-1), b_target)
    print(desc)


if __name__ == '__main__':
    main()
