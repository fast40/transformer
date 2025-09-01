import time

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


torch.manual_seed(1337)


with open('input.txt', 'r') as file:
    train_tokens = file.read()
    vocab = sorted(list(set(train_tokens)))

VOCAB_SIZE = len(vocab)

print(''.join(vocab))

stoi = { token: i for i, token in enumerate(vocab) }
itos = { i: token for i, token in enumerate(vocab) }

tokens = torch.tensor([stoi[token] for token in train_tokens])  # 200k words
train_tokens = tokens[:int(len(tokens) * 0.9)]
test_tokens = tokens[int(len(tokens) * 0.9):]


BATCH_SIZE = 100
CONTEXT_LENGTH = 512
N_EMBED = 384
DROPOUT = 0.2
N_HEADS = 8


class AttentionHead(nn.Module):
    def __init__(self, head_size):  # head_size is the dimension of the key/query/value vectors
        super().__init__()

        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)

        self.register_buffer('mask', ~torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH)).bool())
        self.scale_factor = head_size ** -0.5

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        affinities = self.dropout((self.key(x) @ self.query(x).transpose(-2, -1) * self.scale_factor).masked_fill(self.mask, -torch.inf).softmax(dim=-1)) # yeah im a beast what are you going to do about it

        return affinities @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList(AttentionHead(head_size) for _ in range(n_heads))  # nn.ModuleList is basically a normal list but it registers the modules you put in it
        self.projection_layer = nn.Linear(N_EMBED, N_EMBED)  # head_size is N_EMBED in this case; so this layer is also designed to nicely integrate all the cat'd data. Honestly I want to read more about this I don't really get it. # N_EMBED output dim is critical, input could be something else
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.projection_layer(torch.cat([head(x) for head in self.heads], dim=-1)))


class Block(nn.Module):
    def __init__(self, n_heads):
        super().__init__()

        self.head = MultiHeadAttention(n_heads, N_EMBED // n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(N_EMBED, N_EMBED * 4),
            nn.ReLU(),
            nn.Linear(N_EMBED * 4, N_EMBED),  # projection layer back into residual pathway
            nn.Dropout()
        )

        self.layer_norm_1 = nn.LayerNorm(N_EMBED)
        self.layer_norm_2 = nn.LayerNorm(N_EMBED)

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = x + self.head(self.layer_norm_1(x))  # TODO: figure out why F.layer_norm exists. I think it has something to do with elementwise_affine that karpathy didn't know why it exists.
        x = x + self.feed_forward(self.layer_norm_2(x))

        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.tok_embedding = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.pos_embedding = nn.Embedding(CONTEXT_LENGTH, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_HEADS) for _ in range(6)], nn.LayerNorm(N_EMBED))  # TODO: why do multiple heads perform better than a single head with dimension equal to the sum of the dimensions of all the smaller heads?

        # self.head = MultiHeadAttention(4, N_EMBED // 4)  # TODO: investigate the performance slowdown when the 4 is a 1

        self.final_linear = nn.Linear(N_EMBED, VOCAB_SIZE)  # TODO: I guess we do have bias here. figure out why/if it matters.

        self.register_buffer('attention_mask', ~torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH)).bool())
        self.register_buffer('positions', torch.arange(CONTEXT_LENGTH))  # TODO: check if making this a buffer rather than inline actually affects performance, and if so how

        self.ffwd = nn.Sequential(
            nn.Linear(N_EMBED, N_EMBED),
            nn.ReLU()
        )

    def forward(self, x, targets=None):
        B, T = x.shape

        embeddings = self.tok_embedding(x) + self.pos_embedding(self.positions[:T]) # B, T, N_EMBED
        
        head_output = self.blocks(embeddings)

        logits = self.final_linear(self.ffwd(head_output)) # now the logits is BxTxVOCAB_SIZE. We have logits for the prediction for the next word after EVERY word in the list of T words.


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

        return ''.join(itos[token.item()] for token in content[len(x):])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

print('len', len(train_tokens))

t1 = time.perf_counter()

@torch.no_grad
def get_test_loss():
    t = time.perf_counter()
    n_loss = 3
    loss_values = torch.zeros((n_loss,))

    for i in range(n_loss):
        indices = torch.randint(0, len(test_tokens) - CONTEXT_LENGTH, (BATCH_SIZE, 1)).repeat(1, CONTEXT_LENGTH) + torch.arange(CONTEXT_LENGTH)

        test_x = test_tokens[indices].to(device)
        test_y = test_tokens[indices+1].to(device)

        optimizer.zero_grad()

        _, loss = model(test_x, test_y)

        loss_values[i] = loss

    print(loss_values)
    print(time.perf_counter() - t)

    return loss_values.mean()


while True:
    try:
        for i in range(1_000_000):

            indices = torch.randint(0, len(train_tokens) - CONTEXT_LENGTH, (BATCH_SIZE, 1)).repeat(1, CONTEXT_LENGTH) + torch.arange(CONTEXT_LENGTH)

            train_x = train_tokens[indices].to(device)
            train_y = train_tokens[indices+1].to(device)

            optimizer.zero_grad()

            logits, loss = model(train_x, train_y)

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(loss.item())
                # print(model.generate(tokens[:CONTEXT_LENGTH], 200))
    except KeyboardInterrupt:
        torch.save(model.state_dict(), f'model_checkpoint_{round(time.time() * 1000)}')
        model.eval()
        print(model.generate(train_tokens[:CONTEXT_LENGTH], 2000))
        print(f'test_loss: {get_test_loss()}')
        model.train()
        time.sleep(0.5)
