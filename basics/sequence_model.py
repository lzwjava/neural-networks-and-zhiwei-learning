import torch
import torch.nn as nn

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)
print(lstm)

inputs = [torch.randn(1, 3) for _ in range(5)]
print(inputs)

print(torch.randn(1, 3))
