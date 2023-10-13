import os
import requests

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)
        
with open(input_file_path, 'r') as f:
    data = f.read()

print(f"length: {len(data)}")

chars = sorted(list(set(data)))
vocab_size = len(chars)
print('all the unique characters:', ''.join(chars))
print(f'vocab_size: {vocab_size}')

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

print(encode('abc'))
print(decode([2, 3, 4]))
