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


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.tok_embedding = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.pos_embedding = nn.Embedding(VOCAB_SIZE, N_EMBED)

        self.key = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.query = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.value = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)

        self.final_linear = nn.Linear(HEAD_SIZE, VOCAB_SIZE)  # TODO: I guess we do have bias here. figure out why/if it matters.

        self.register_buffer('attention_mask', ~torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH)).bool())
        self.register_buffer('positions', torch.arange(CONTEXT_LENGTH))  # TODO: check if making this a buffer rather than inline actually affects performance, and if so how

    def forward(self, x, targets=None):
        B, T = x.shape

        embeddings = self.tok_embedding(x) + self.pos_embedding(self.positions[:T]) # B, T, N_EMBED
        
        key = self.key(embeddings) # B, T, HEAD_SIZE
        query = self.key(embeddings) # B, T, HEAD_SIZE
        value = self.key(embeddings) # B, T, HEAD_SIZE

        affinity = query @ key.transpose(-2, -1)  # B, T, H @ B, T, H. What do we want here? We want a TxT. Because we want a table that maps each token to each other token. That's where the transpose comes in.
        affinity = affinity.masked_fill(self.attention_mask[:T, :T], float('-inf'))
        affinity = F.softmax(affinity, dim=2)

        # affinity is B TxT matrices. Value is TxH. so they are good to go. Now we have a list of B TxH matrices.

        head_output = affinity @ value

        logits = self.final_linear(head_output) # now the logits is BxTxVOCAB_SIZE. We have logits for the prediction for the next word after EVERY word in the list of T words.

        if targets is not None:
            # loss = F.cross_entropy(logits.transpose(-2, -1), targets)
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))

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
