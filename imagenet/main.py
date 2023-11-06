import argparse
import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from enum import Enum
import torch.distributed as dist

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

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.pretrained:
        model = models.__dict__[args.arch](pretrained=True)
    else:
        model = models.__dict__[args.arch]()

    device = get_device()

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    best_accl = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f'loading checking point {args.resume}')
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_accl = checkpoint['best_accl']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f'load checkpoint {args.resume} epoch: {args.start_epoch}')
        else:
            print(f'no checkpoint found at {args.resume}')

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None
    val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler
    )


def train(train_loader, model, loss_fn, optimizer, epoch, args):
    batch_time = AverageMeter('time', ':6.3f')
    data_time = AverageMeter('data', ':6.3f')
    losses = AverageMeter('loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        

class Summary(Enum):
    NONE = 0,
    AVERAGE = 1,
    SUM = 2,
    COUNT = 3


class AverageMeter(object):

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = get_device()
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type')

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=''):
        self.meters = meters
        self.prefix = prefix
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [' *']
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


if __name__ == '__main__':
    main()
