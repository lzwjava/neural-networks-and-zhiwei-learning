import torch
import torch.nn as nn

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)
print(embeds.weight)

lookup_tensor = torch.tensor([word_to_ix['hello']], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)

print(embeds(torch.tensor(1)))

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

print(test_sentence)

ngrams = [
    ([
         test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)], test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]

print(ngrams)
