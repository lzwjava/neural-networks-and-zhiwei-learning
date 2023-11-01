import argparse

import torch

from word_language_model import data

from dataclasses import dataclass
import argparse


@dataclass
class Args:
    data: str = './data/wikitext-2'
    model: str = 'LSTM'
    emsize: int = 200
    nhid: int = 200
    nlayers: int = 2
    lr: float = 20
    clip: float = 0.25
    epochs: int = 40
    batch_size: int = 20
    bptt: int = 35
    dropout: float = 0.2
    tied: bool = True
    seed: int = 1111
    cuda: bool = True
    log_interval: int = 200
    save: str = 'model.pt'
    onnx_export: str = ''
    nhead: int = 2
    dry_run: bool = True


def batchify(data: torch.Tensor, bsz):
    nbatch = data.size(0) // bsz


def main():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument('--data', type=str, default=Args.data)
    parser.add_argument('--model', type=str, default=Args.model)
    parser.add_argument('--emsize', type=int, default=Args.emsize)
    parser.add_argument('--nhid', type=int, default=Args.nhid)
    parser.add_argument('--nlayers', type=int, default=Args.nlayers)
    parser.add_argument('--lr', type=float, default=Args.lr)
    parser.add_argument('--clip', type=float, default=Args.clip)
    parser.add_argument('--epochs', type=int, default=Args.epochs)
    parser.add_argument('--batch-size', type=int, default=Args.batch_size)
    parser.add_argument('--bptt', type=int, default=Args.bptt)
    parser.add_argument('--dropout', type=float, default=Args.dropout)
    parser.add_argument('--tied', default=Args.tied)
    parser.add_argument('--seed', type=int, default=Args.seed)
    parser.add_argument('--cuda', type=bool, default=Args.cuda)
    parser.add_argument('--log-interval', type=int, default=Args.log_interval)
    parser.add_argument('--save', type=str, default=Args.save)
    parser.add_argument('--onnx-export', type=str, default=Args.onnx_export)
    parser.add_argument('--nhead', type=int, default=Args.nhead)
    parser.add_argument('--dry-run', type=bool, default=Args.dry_run)

    args = parser.parse_args()

    args = Args(**vars(args))

    print(args)
    print(args.nhid)

    torch.manual_seed(args.seed)

    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    corpus = data.Corpus(args.data)


if __name__ == '__main__':
    main()
