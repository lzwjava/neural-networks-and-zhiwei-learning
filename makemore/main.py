import argparse
import os
import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    
    def __init__(self, words):
        self.words = words
    
    def __len__(self) -> int:
        return len(self.words)
    
    def __getitem__(self, index) -> str:
        return self.words[index]

def create_datasets(input_file):
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    lines = [l.strip() for l in lines]
    
    word_count = len(lines)
    
    test_size = int(word_count * 0.1)
    
    rp = torch.randperm(word_count).to_list()
    
    print(rp)
    
    train_words = lines[0:word_count-test_size]
    test_words = lines[word_count-test_size:]
    
    print(word_count)
    print(test_size)
    print(len(train_words))
    print(len(test_words))
    
    return CharDataset(train_words), CharDataset(test_words)

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