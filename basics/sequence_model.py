import torch
import torch.nn as nn

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)
print(lstm)

inputs = [torch.randn(1, 3) for _ in range(5)]
print(inputs)

print(torch.randn(1, 3))

hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))

print(f'{inputs[0]=}')
print(f'{inputs[0].view(1, 1, -1)=}')

for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)

print(i.view(1, 1, -1))


def prepare_sequence(seq, to_ix):
    pass


training_data = [
    ("The dog ate the apple".split(), ['DET', 'NN', 'V', 'DET', 'NN']),
    (('Everyday read that book').split(), ['NN', 'V', 'DET', 'NN'])
]

print(training_data)

word_to_idx = {}

for sent, tags in training_data:
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

print(word_to_idx)

tag_to_idx = {'DET': 0, 'NN': 1, 'V': 2}

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

