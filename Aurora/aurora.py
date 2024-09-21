import torch
from torch import nn
import torch.functional as F
from dataclasses import dataclass
import torch.optim as optim

@dataclass
class AURConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layers: int = 18
    n_head: int = 12
    embed_size: int = 786

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.embed_size, 3 * config.embed_size)
        self.c_proj = nn.Linear(config.embed_size, config.embed_size)

        self.n_head = config.n_head
        self.n_embd = config.embed_size

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.l_1 = nn.Linear(config.embed_size, 4 * config.embed_size)
        self.gelu = nn.GELU(approximate="none")
        self.l_2 = nn.Linear(4 * config.embed_size, config.embed_size)

    def forward(self, x):
        x = self.l_1(x)
        x = self.gelu(x)
        x = self.l_2(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.embed_size)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_size)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.ln_1(x)
        x = self.attn(x)
        x = self.ln_2(x)
        x = self.mlp(x)
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            embed=nn.Embedding(config.vocab_size, config.embed_size),
            pos_embed=nn.Embedding(config.block_size, config.embed_size),
            hidden=nn.ModuleList(Block(config) for _ in range(config.n_layers)),
            norm=nn.LayerNorm(config.embed_size)
        ))
        self.head_classifier = nn.Linear(config.embed_size, config.vocab_size)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)

        idx = self.transformer.embed(x)
        pos = self.transformer.pos_embed(pos)
        x = idx + pos

        for block in self.transformer.hidden:
            x = block(x)

        x = self.transformer.norm(x)
        logits = self.head_classifier(x)
        return logits

class Model():
    def __init__(self, config):
        self.config = config
        self.gpt = GPT(config)
        self.optimizer = optim.Adam(self.gpt.parameters(), lr=0.001)

    def train(self, epochs, X_train, y_train, save_path='model_weights.pth'):
        for epoch in range(epochs):
            self.gpt.train()
            self.optimizer.zero_grad()
            
            output = self.gpt(X_train)
            loss = nn.MSELoss()(output.view(-1, self.config.vocab_size), y_train.view(-1, self.config.vocab_size))  # Calculate loss
            
            loss.backward()
            self.optimizer.step()
            
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        torch.save(self.gpt.state_dict(), save_path)
        print(f'Model weights saved to {save_path}')

import torch
from torch.utils.data import DataLoader, TensorDataset

config = AURConfig()


def load_shakespeare_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = sorted(set(text))
    char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

    text_as_int = [char_to_idx[char] for char in text]
    return text_as_int, len(vocab), char_to_idx, idx_to_char

file_path = "The Complete Works of William Shakespeare copy.txt"
text_as_int, vocab_size, char_to_idx, idx_to_char = load_shakespeare_text(file_path)

seq_length = config.block_size
X_train = [text_as_int[i:i + seq_length] for i in range(0, len(text_as_int) - seq_length)]
y_train = [text_as_int[i + 1:i + 1 + seq_length] for i in range(0, len(text_as_int) - seq_length)]

X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

config.vocab_size = vocab_size
model = Model(config)

for epoch in range(10):
    for X_batch, y_batch in train_loader:
        model.train(1, X_batch, y_batch)