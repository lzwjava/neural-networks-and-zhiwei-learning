import argparse
import torch

import data


def main():
    parser = argparse.ArgumentParser('PyTorch Word Language Model')
    parser.add_argument('--data', type=str, default='./data/wikitext-2', help='location of data')
    parser.add_argument('--checkpoint', type=str, default='./model.pt', help='checkpoint')
    parser.add_argument('--out-f', type=str, default='generated.txt', help='output file')
    parser.add_argument('--words', type=int, default=1000, help='words')
    parser.add_argument('--seed', type=int, default=1111, help='seed')
    parser.add_argument('--cuda', type=bool, default=True, help='cuda')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--log-interval', type=int, default=100, help='log interval')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.temperature < 1e-3:
        parser.error('temperature has to be greater or equal 1e-3')

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)

    model.eval()

    corpus = data.Corpus(args.data)
    ntokens = len(corpus.dictionary)

    is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
    if not is_transformer_model:
        hidden = model.init_hidden(1)

    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    print(ntokens)
    print(input)


if __name__ == '__main__':
    main()
