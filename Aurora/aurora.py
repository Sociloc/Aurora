import torch
from torch import nn
import torch.functional as F
from dataclasses import dataclass

@dataclass
class AURConfig:
    block_size = 1024
    vocab_size = 50257
    n_layers = 18
    n_head = 12
    embed_size = 786

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

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
            embed = nn.Embedding(config.vocab_size, config.embed_size),
            pos_embed = nn.Embedding(config.block_size, config.embed_size),
            hidden = nn.ModuleList(Block(config) for _ in range(config.n_layers)),
            norm = nn.LayerNorm(config.embed_size)
        ))
        self.head_classifier = nn.Linear(config.embed_size, config.vocab_size)
    
    def forward(self, x):
        B, T = x.size()

        pos = torch.arange(0, T, dtype=torch.long, device=x.device)

        idx = self.transformer.embed(x)
        pos = self.transformer.embed(pos)
        x = idx + pos

        for block in self.transformer.hidden:
            x = block(x)

        x = self.transformer.norm(x)
        logits = self.head_classifier(x)

        return logits


        

