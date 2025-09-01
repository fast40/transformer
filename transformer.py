import time

from typing_extensions import NoExtraItems
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


torch.manual_seed(1337)


with open('input.txt', 'r') as file:
    tokens = file.read()
    vocab = sorted(list(set(tokens)))

VOCAB_SIZE = len(vocab)


print(''.join(vocab))

stoi = { token: i for i, token in enumerate(vocab) }
itos = { i: token for i, token in enumerate(vocab) }

tokens = torch.tensor([stoi[token] for token in tokens])  # 200k words


CONTEXT_LENGTH = 8
N_EMBED = 32
HEAD_SIZE = 32


class AttentionHead(nn.Module):
    def __init__(self, head_size):  # head_size is the dimension of the key/query/value vectors
        super().__init__()

        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)

        self.register_buffer('mask', ~torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH)).bool())
        self.scale_factor = head_size ** -0.5

    def forward(self, x):
        # yeah im a beast what are you going to do about it
        return (self.key(x) @ self.query(x).transpose(-2, -1) * self.scale_factor).masked_fill(self.mask, -torch.inf).softmax(dim=-1) @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList(AttentionHead(head_size) for _ in range(n_heads))  # nn.ModuleList is basically a normal list but it registers the modules you put in it

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.tok_embedding = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.pos_embedding = nn.Embedding(CONTEXT_LENGTH, N_EMBED)

        self.head = MultiHeadAttention(4, N_EMBED // 4)  # TODO: investigate the performance slowdown when the 4 is a 1

        self.final_linear = nn.Linear(HEAD_SIZE, VOCAB_SIZE)  # TODO: I guess we do have bias here. figure out why/if it matters.

        self.register_buffer('attention_mask', ~torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH)).bool())
        self.register_buffer('positions', torch.arange(CONTEXT_LENGTH))  # TODO: check if making this a buffer rather than inline actually affects performance, and if so how

    def forward(self, x, targets=None):
        B, T = x.shape

        embeddings = self.tok_embedding(x) + self.pos_embedding(self.positions[:T]) # B, T, N_EMBED
        
        head_output = self.head(embeddings)

        logits = self.final_linear(head_output) # now the logits is BxTxVOCAB_SIZE. We have logits for the prediction for the next word after EVERY word in the list of T words.

        if targets is not None:
            loss = F.cross_entropy(logits.transpose(-2, -1), targets)

            return logits, loss

        return logits

    def generate(self, x, num_tokens):
        content = torch.zeros((num_tokens+len(x),), dtype=int, device=next(self.parameters()).device)
        content[0:len(x)] = x

        for i in range(num_tokens):
            logits = self(content[i:i+CONTEXT_LENGTH].view(1, -1))
            probs = F.softmax(logits[0][-1], dim=0)
            content[i+CONTEXT_LENGTH] = torch.multinomial(probs, 1)

        return ''.join(itos[token.item()] for token in content)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

print('len', len(tokens))

BATCH_SIZE = 1000

t1 = time.perf_counter()

for i in range(100_000):

    indices = torch.randint(0, len(tokens) - CONTEXT_LENGTH, (BATCH_SIZE, 1)).repeat(1, CONTEXT_LENGTH) + torch.arange(CONTEXT_LENGTH)

    train_x = tokens[indices].to(device)
    train_y = tokens[indices+1].to(device)

    optimizer.zero_grad()

    logits, loss = model(train_x, train_y)

    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        # print(loss.item())
        print(model.generate(tokens[:CONTEXT_LENGTH], 200))
