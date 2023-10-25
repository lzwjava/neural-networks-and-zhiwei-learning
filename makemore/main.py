import argparse
import os
import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    
    def __init__(self, words, chars, max_len):
        self.words = words
        self.chars = chars
        self.max_len = max_len
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        # print(self.stoi)
        self.itos = {i:ch for ch, i in self.stoi.items()}
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
        ix = torch.tensor([self.stoi[w]  for w in word], dtype=torch.long)
        return ix
    
    def decode(self, ix):
        word = ''.join(self.itos[i]  for i in ix)
        return word
    
    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_len + 1, dtype=torch.long)
        y = torch.zeros(self.max_len + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 
        return x, y    

def create_datasets(input_file):
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    lines = [l.strip() for l in lines]
    
    word_count = len(lines)
    
    test_size = int(word_count * 0.1)
    
    rp = torch.randperm(word_count).tolist()
    
    # print(rp)    
    train_words = [lines[i]   for i in rp[:-test_size]]
                        
    test_words = [lines[i]   for i in rp[-test_size:]]
    
    print(word_count)
    print(test_size)
    print(len(train_words))
    print(len(test_words))
    
    # print(test_words)
    
    chars = sorted(set(''.join(lines)))
    
    max_len = len(max(lines, key = len))
    
    print(f'chars:{chars}')
    print(f'max_len={max_len}')
    
    train_dataset = CharDataset(train_words, chars, max_len)
    test_dataset = CharDataset(test_words, chars, max_len)
    
    return train_dataset, test_dataset

if __name__ == '__main__':
    print('main')
    parser = argparse.ArgumentParser(description= "make more")
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
    
    