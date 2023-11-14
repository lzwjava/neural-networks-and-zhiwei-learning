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

print(f'{inputs[0]}')
print(f'{inputs[0].view(1, 1, -1)=}')

for i in inputs:
    print(f'{i.size()}')
    print(f'{i.view(1, 1, -1)=}')
    out, hidden = lstm(i.view(1, 1, -1), hidden)


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


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        return embeds


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_idx), len(tag_to_idx))
loss_fn = nn.NLLLoss()
