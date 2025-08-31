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


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)

    def forward(self, x, targets=None):
        logits = self.embedding(x)

        if targets is not None:
            loss = F.cross_entropy(logits, targets)

            return F.softmax(logits, dim=1), loss

        return F.softmax(logits, dim=1)

    def generate(self, num_tokens):
        content = torch.zeros((num_tokens,), dtype=int, device=next(self.parameters()).device)
        content[0] = stoi['T']

        for i in range(num_tokens - 1):
            content[i + 1 ] = torch.multinomial(self(content[i].view(-1))[0], 1)

        return ''.join(itos[token.item()] for token in content)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

print('len', len(tokens))

BATCH_SIZE = 10000

for i in range(100_000):
    indices = torch.randint(0, len(tokens) - 1, (BATCH_SIZE,))
    train_x = tokens[indices].to(device)
    train_y = tokens[indices + 1].to(device)

    optimizer.zero_grad()

    logits, loss = model(train_x, train_y)

    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        # print(loss.item())
        print(model.generate(200))
