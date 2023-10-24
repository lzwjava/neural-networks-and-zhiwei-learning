import argparse
import torch
import os

def create_datasets(input_file):
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    lines = [l.strip() for l in lines]
    
    word_count = len(lines)
    
    print(word_count)
    
    print(lines)
    
    return [], []

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