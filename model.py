import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
from dataclasses import dataclass
from typing import Optional


class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x):
        return t.einsum('dbp -> bpd', self.W_E[:, x])


class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def forward(self, x):
        return x @ self.W_U


class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(t.randn(max_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]


@dataclass(frozen=True)
class Config:
    p: int = 113
    frac_train: float = 0.3
    seed: int = 0

    d_model: int = 128
    d_mlp: Optional[int] = None
    num_heads: int = 4
    num_layers: int = 1
    n_ctx: int = 3
    d_vocab: Optional[int] = None
    act_type: str = 'ReLU'
    use_ln: bool = False

    num_epochs: int = 40_000

    # Once test_acc reaches `adaptive_logging_thresh`, multiply eval_every and fourier_every
    # by `adaptive_logging_factor` to reduce storage in post-grok regime.
    adaptive_logging:         bool  = False
    adaptive_logging_thresh:  float = 0.99
    adaptive_logging_factor:  int   = 10

    device: t.device = (
        t.device("cuda") if t.cuda.is_available()
        else t.device("mps") if t.backends.mps.is_available()
        else t.device("cpu")
    )

    def __post_init__(self):
        if self.d_vocab is None:
            object.__setattr__(self, 'd_vocab', self.p + 1)
        if self.d_mlp is None:
            object.__setattr__(self, 'd_mlp', 4 * self.d_model)

    @property
    def d_head(self) -> int:
        return self.d_model // self.num_heads

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx):
        super().__init__()
        # Per-head matrices: W_K[i] has shape (d_head, d_model)
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads) / np.sqrt(d_model))

        self.register_buffer('mask', t.tril(t.ones((n_ctx, n_ctx))))
        self.d_head = d_head

    def forward(self, x):
        k = t.einsum('ihd,bpd->biph', self.W_K, x)
        q = t.einsum('ihd,bpd->biph', self.W_Q, x)
        v = t.einsum('ihd,bpd->biph', self.W_V, x)
        attn_scores_pre = t.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = t.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = F.softmax(attn_scores_masked / np.sqrt(self.d_head), dim=-1)
        z = t.einsum('biph,biqp->biqh', v, attn_matrix)

        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = t.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out
    
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type):
        super().__init__()
        self.W_in  = nn.Parameter(t.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in  = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))
        self.act_type = act_type
        assert act_type in ['ReLU', 'GeLU']

    def forward(self, x):
        x = t.einsum('md,bpd->bpm', self.W_in, x) + self.b_in
        if self.act_type == 'ReLU':
            x = F.relu(x)
        elif self.act_type == 'GeLU':
            x = F.gelu(x)
        x = t.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type):
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_head, n_ctx)
        self.mlp  = MLP(d_model, d_mlp, act_type)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embed     = Embed(d_vocab=config.d_vocab, d_model=config.d_model)
        self.pos_embed = PosEmbed(max_ctx=config.n_ctx, d_model=config.d_model)
        self.blocks    = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                d_mlp=config.d_mlp,
                d_head=config.d_head,
                num_heads=config.num_heads,
                n_ctx=config.n_ctx,
                act_type=config.act_type,
            )
            for _ in range(config.num_layers)
        ])
        self.unembed = Unembed(d_vocab=config.d_vocab, d_model=config.d_model)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.unembed(x)
        return x
