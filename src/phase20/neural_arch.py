"""Phase 20 -- neural architectures for the P2 neural screen (fold-0 discovery).

Two canonical tabular neural architectures, implemented from scratch in PyTorch
(the certified .venv has torch 2.11.0+cu128 with CUDA). Both are genuinely
different inductive biases from the tree-heavy champion and its existing
RealMLP/TabM (feedforward) members:

  FTTransformer  attention over feature tokens (Gorishniy et al. 2021).
  DCNv2          explicit feature crosses x_{l+1}=x_0*(W_l x_l+b_l)+x_l plus a
                 deep MLP (Wang et al. 2021).

All architectures consume the SAME 22-feature champion route representation
(numeric, standardized on FIT-train only — fold-safe). A single interface:
``build(name, n_features, **cfg) -> nn.Module`` with ``forward(x) -> logit``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer encoder. forward(x)->logit, x:(B,n_features)."""

    def __init__(self, n_features: int, d_token: int = 192, n_layers: int = 3,
                 n_heads: int = 8, dropout: float = 0.1, head_hidden: int = 192):
        super().__init__()
        # Feature tokenizer: t_j = x_j * w_j + b_j, w_j/b_j in R^{d_token}.
        self.W = nn.Parameter(torch.empty(n_features, d_token))
        self.b = nn.Parameter(torch.empty(n_features, d_token))
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=4 * d_token,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, head_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 1))
        self._init()

    def _init(self) -> None:
        nn.init.normal_(self.W, std=0.2)
        nn.init.zeros_(self.b)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        tok = x.unsqueeze(-1) * self.W + self.b          # (B, n_feat, d_token)
        cls = self.cls.expand(bsz, -1, -1)               # (B, 1, d_token)
        h = self.encoder(torch.cat([cls, tok], dim=1))   # (B, 1+n_feat, d_token)
        h = self.norm(h[:, 0])                           # CLS token
        return self.head(h).squeeze(-1)                  # (B,)


class DCNv2(nn.Module):
    """Deep & Cross Network v2: cross network (explicit crosses) + deep MLP."""

    def __init__(self, n_features: int, d_hidden: int = 256, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.cross_w = nn.ModuleList([nn.Linear(n_features, 1) for _ in range(n_layers)])
        self.cross_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(n_features)) for _ in range(n_layers)])
        deep = []
        d = n_features
        for _ in range(n_layers):
            deep += [nn.Linear(d, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(),
                     nn.Dropout(dropout)]
            d = d_hidden
        self.deep = nn.Sequential(*deep)
        self.out = nn.Linear(n_features + d_hidden, 1)
        # Init cross weights small so crosses start near identity.
        for w in self.cross_w:
            nn.init.xavier_uniform_(w.weight, gain=0.5)
            nn.init.zeros_(w.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xc = x
        for w, b in zip(self.cross_w, self.cross_b):
            xl = w(xc) + b                               # (B, n_feat)
            xc = x0 * xl + xc                            # residual cross
        xd = self.deep(x)                                # (B, d_hidden)
        return self.out(torch.cat([xc, xd], dim=1)).squeeze(-1)


def build(name: str, n_features: int, **cfg) -> nn.Module:
    """Construct an architecture by name. Unrecognized names raise."""
    name = name.lower()
    if name in ("fttransformer", "ft-transformer", "ftt"):
        return FTTransformer(n_features, **cfg)
    if name in ("dcnv2", "dcn-v2", "dcn"):
        return DCNv2(n_features, **cfg)
    raise ValueError(f"unknown neural architecture: {name!r}")
