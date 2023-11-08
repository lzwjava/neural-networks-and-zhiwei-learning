import argparse
from dataclasses import dataclass
import os
import time
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    block_size: int = None
    vocab_size: int = None
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4


class CausalBow(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.block_size = config.block_size
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, config.block_size, config.block_size))

        # torch.tril(torch.ones(3, 3)).view(1, 3, 3)

    def forward(self, x):
        B, T, C = x.size()

        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ x

        return y


class BoWBlock(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()

        self.cbow = CausalBow(config)

        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, config.n_embd2),
            c_proj=nn.Linear(config.n_embd2, config.n_embd)
        ))

        m = self.mlp

        self.mlpf = lambda x: m.c_proj(F.tanh(m.c_fc(x)))

    def forward(self, x):
        x = x + self.cbow(x)
        x = x + self.mlpf(x)
        return x


class Bow(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        self.context_block = BoWBlock(config)

        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device

        b, t = idx.size()

        # print(f"{idx.size()=}")

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)

        x = tok_emb + pos_emb

        x = self.context_block(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


class CharDataset(Dataset):

    def __init__(self, words, chars, max_len):
        self.words = words
        self.chars = chars
        self.max_len = max_len
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        # print(self.stoi)
        self.itos = {i: ch for ch, i in self.stoi.items()}
        # print(self.itos)

    def __len__(self) -> int:
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1

    def get_output_length(self):
        return self.max_len + 1

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_len + 1, dtype=torch.long)
        y = torch.zeros(self.max_len + 1, dtype=torch.long)
        x[1:1 + len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix) + 1:] = -1
        return x, y


def create_datasets(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines]

    word_count = len(lines)

    test_size = int(word_count * 0.1)

    rp = torch.randperm(word_count).tolist()

    # print(rp)    
    train_words = [lines[i] for i in rp[:-test_size]]

    test_words = [lines[i] for i in rp[-test_size:]]

    print(word_count)
    print(test_size)
    print(len(train_words))
    print(len(test_words))

    # print(test_words)

    chars = sorted(set(''.join(lines)))

    max_len = len(max(lines, key=len))

    print(f'chars:{chars}')
    print(f'max_len={max_len}')

    train_dataset = CharDataset(train_words, chars, max_len)
    test_dataset = CharDataset(test_words, chars, max_len)

    return train_dataset, test_dataset


class InfiniteDataLoader:

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)

        return batch


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    block_size = model.get_block_size()

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        print(idx_cond)
        print(f"{idx_cond.size()=}")

        logits, _ = model(idx_cond)

        print('logits')
        print(logits)
        print(f"{logits.size()=}")

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            print(f"{v=}")
            logits[logits < v[:, [-1]]] = -float('inf')

        probs = F.softmax(logits, dim=-1)

        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


if __name__ == '__main__':
    print('main')
    parser = argparse.ArgumentParser(description="make more")
    parser.add_argument('--input-file', type=str, default='names.txt', help="input file")
    parser.add_argument('--seed', type=int, default=3047, help="seed")
    parser.add_argument('--work-dir', type=str, default='out', help="output working directory")
    args = parser.parse_args()
    print(vars(args))

    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.work_dir, exist_ok=True)

    train_dataset, test_dataset = create_datasets(args.input_file)

    contains = train_dataset.contains('jack')
    print(contains)
    test_contains = test_dataset.contains('jack')
    print(test_contains)

    ix = train_dataset.encode('jack')
    print(ix)
    word = train_dataset.decode(ix.tolist())
    print(word)

    x0, y0 = train_dataset[0]
    print(f'x0={x0}')
    print(f'y0={y0}')

    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()

    print(f"{vocab_size=}, {block_size=}")

    n_layer = 4
    n_head = 4
    n_embd = 64
    n_embd2 = 64
    device = 'cuda'
    learning_rate = 5e-4
    weight_decay = 0.01
    batch_size = 32
    num_workers = 4
    max_steps = 100

    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                         n_layer=n_layer, n_head=n_head,
                         n_embd=n_embd, n_embd2=n_embd2)

    bow = CausalBow(config)
    # print(bow)

    model = Bow(config)

    model.to(device)

    print(f"model #params: {sum(p.numel() for p in model.parameters())}")

    for p in model.parameters():
        # print(p)
        print(p.size())

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.99),
                                  eps=1e-8)

    batch_loader = InfiniteDataLoader(train_dataset, batch_size=batch_size, pin_memory=False, num_workers=4)

    best_loss = None
    step = 0

    while True:

        t0 = time.time()

        batch = batch_loader.next()
        batch = [t.to(device) for t in batch]
        X, Y = batch

        logits, loss = model(X, Y)

        # print(logits)
        # print(loss)

        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step % 10 == 0):
            print(f"step {step} | loss {loss.item():.4f}")

        step += 1
        if step >= max_steps:
            break

    X_init = torch.zeros(10, 1, dtype=torch.long).to(device)

    output_len = train_dataset.get_output_length() - 1

    result = generate(model, X_init, output_len, top_k=2, do_sample=True)

    print(result)
