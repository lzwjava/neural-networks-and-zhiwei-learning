import torch.nn as nn
import torch

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)
print(embeds.weight)

lookup_tensor = torch.tensor([word_to_ix['hello']], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)

print(embeds(torch.tensor(1)))
