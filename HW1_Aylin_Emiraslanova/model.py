# hw1-build: caspian-7

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    use_positional_embeddings: bool = True
    norm_style: str = "pre"


class CausalSelfAttentionK(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=False
        )

        self.proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            is_causal = False
        else:
            is_causal = True

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.resid_dropout(y)

        if use_cache:
            return y, (k, v)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(
            config.n_embd,
            4 * config.n_embd,
            bias=False
        )

        self.fc2 = nn.Linear(
            4 * config.n_embd,
            config.n_embd,
            bias=False
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.attn = CausalSelfAttentionK(config)
        self.mlp = MLP(config)

        self.norm_style = config.norm_style

    def forward(self, x, past_kv=None, use_cache=False):

        if self.norm_style == "post":

            if use_cache:
                attn_out, present_kv = self.attn(
                    x, past_kv=past_kv, use_cache=True
                )
            else:
                attn_out = self.attn(x)

            x = self.ln1(x + attn_out)
            x = self.ln2(x + self.mlp(x))

        else:

            if use_cache:
                attn_out, present_kv = self.attn(
                    self.ln1(x), past_kv=past_kv, use_cache=True
                )
            else:
                attn_out = self.attn(self.ln1(x))

            x = x + attn_out
            x = x + self.mlp(self.ln2(x))

        if use_cache:
            return x, present_kv

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        ) if config.use_positional_embeddings else None

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            Block(config)
            for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        B, T = idx.shape

        past_length = 0 if past_kv is None else past_kv[0][0].size(2)

        if past_length + T > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        positions = torch.arange(
            past_length,
            past_length + T,
            device=idx.device
        )

        tok_emb = self.token_embedding(idx)

        if self.position_embedding is not None:
            pos_emb = self.position_embedding(positions)
            x = self.dropout(tok_emb + pos_emb)
        else:
            x = self.dropout(tok_emb)

        presents = []

        for layer_index, block in enumerate(self.blocks):
            layer_past = None if past_kv is None else past_kv[layer_index]

            if use_cache:
                x, present = block(
                    x,
                    past_kv=layer_past,
                    use_cache=True
                )
                presents.append(present)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        if use_cache:
            return logits, loss, presents

        return logits, loss

    def num_params(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
