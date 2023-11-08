import torch.nn as nn

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)
print(embeds.weight)