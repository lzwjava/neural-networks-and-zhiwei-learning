import argparse

import torchvision.models as models
from dataclasses import dataclass

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))


# print(model_names)


@dataclass
class Args:
    data: str = 'imagenet'
    arch: str = 'resnet18'
    workers: int = 4
    epochs: int = 90
    start_epoch: int = 0
    batch_size: int = 256
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    print_freq: int = 10
    resume: str = ''
    evaluate: bool = False
    pretrained: bool = False
    seed: int = None


def main():
    parser = argparse.ArgumentParser('imagenet training')
    parser.add_argument('data', default=Args.data)
    parser.add_argument('--arch', default=Args.arch, choices=model_names, help=' | '.join(model_names))
    parser.add_argument('--workers', type=int, default=Args.workers)
    parser.add_argument('--epochs', type=int, default=Args.epochs)
    parser.add_argument('--start-epoch', type=int, default=Args.start_epoch)
    parser.add_argument('--batch-size', type=int, default=Args.batch_size)
    parser.add_argument('--lr', type=float, default=Args.lr)
    parser.add_argument('--momentum', type=float, default=Args.momentum)
    parser.add_argument('--weight-decay', type=float, default=Args.weight_decay)
    parser.add_argument('--print-freq', type=int, default=Args.print_freq)
    parser.add_argument('--resume', type=str, default=Args.resume)
    parser.add_argument('--evaluate', action='store_true', default=Args.evaluate)
    parser.add_argument('--pretrained', action='store_true', default=Args.pretrained)

    args = parser.parse_args()

    args = Args(**vars(args))
    print(args)


if __name__ == '__main__':
    main()
