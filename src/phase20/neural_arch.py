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
import torch.nn.functional as F


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


class DCNv2Full(nn.Module):
    """Canonical DCNv2 with a FULL cross matrix (S7 fidelity repair, item 17).

    x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l  with  W_l ∈ R^{n_feat × n_feat}.
    The earlier DCNv2 proxy used Linear(n_features, 1) — a rank-1 cross with the
    output direction fixed to the all-ones vector (a SCALAR gate per layer). This
    class is identical in every other respect (deep MLP branch, concat head, init)
    so the ONLY delta is the cross matrix rank. n_feat ≤ 512 makes the dense W cheap.
    """

    def __init__(self, n_features: int, d_hidden: int = 256, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.cross_w = nn.ModuleList(
            [nn.Linear(n_features, n_features, bias=False) for _ in range(n_layers)])
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
        for w in self.cross_w:
            nn.init.xavier_uniform_(w.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xc = x
        for w, b in zip(self.cross_w, self.cross_b):
            xl = w(xc) + b                                # (B, n_feat), full W
            xc = x0 * xl + xc                             # residual cross
        xd = self.deep(x)
        return self.out(torch.cat([xc, xd], dim=1)).squeeze(-1)


class FactLog(nn.Module):
    """Factorized (bilinear) logistic: logit = w·x + Σ_{r=1..R} (u_r·x)(v_r·x).

    S7 repair for item 65: the row had been closed by an ADDITIVE logistic
    negative (a different model). Here the interaction term is a rank-R
    factorization of the quadratic form — a genuine interaction model, trained
    end-to-end with the same AdamW + inner-val ES loop as the other neural archs.
    U/V init 0.5/sqrt(F) keeps the initial bilinear sum at ~unit std for
    standardized inputs.
    """

    def __init__(self, n_features: int, rank: int = 16):
        super().__init__()
        self.w = nn.Linear(n_features, 1)
        scale = 0.5 / n_features ** 0.5
        self.U = nn.Parameter(scale * torch.randn(rank, n_features))
        self.V = nn.Parameter(scale * torch.randn(rank, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xu = x @ self.U.T                                 # (B, R)
        xv = x @ self.V.T                                 # (B, R)
        return self.w(x).squeeze(-1) + (xu * xv).sum(-1)  # (B,)


class PLRMLP(nn.Module):
    """Periodic (PLR) numerical embeddings + MLP (Gorishniy et al. 2024).

    Each numeric feature is embedded via learnable periodic functions
    cos/sin(2pi * (omega*x + phi)) across several frequencies, concatenated, then
    passed through an MLP. Captures smooth non-linear per-feature structure.
    """

    def __init__(self, n_features: int, n_freq: int = 16, d_hidden: int = 256,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(n_features, n_freq) * 0.5)
        self.phi = nn.Parameter(torch.rand(n_features, n_freq))
        d_in = n_features * 2 * n_freq + n_features  # periodic + raw skip
        layers = []
        d = d_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, d_hidden), nn.BatchNorm1d(d_hidden), nn.GELU(),
                       nn.Dropout(dropout)]
            d = d_hidden
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ang = 2.0 * torch.pi * (x.unsqueeze(-1) * self.omega + self.phi)  # (B,F,Freq)
        emb = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1).flatten(1)  # (B,F*2*Freq)
        return self.mlp(torch.cat([emb, x], dim=1)).squeeze(-1)


class TabResNet(nn.Module):
    """Residual MLP (Tabular ResNet): projection + pre-activation residual blocks."""

    class _Block(nn.Module):
        def __init__(self, d: int, dropout: float):
            super().__init__()
            self.fc1 = nn.Linear(d, d)
            self.bn1 = nn.BatchNorm1d(d)
            self.fc2 = nn.Linear(d, d)
            self.bn2 = nn.BatchNorm1d(d)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = F.relu(self.bn1(self.fc1(x)))
            h = self.drop(F.relu(self.bn2(self.fc2(h))))
            return x + h

    def __init__(self, n_features: int, d_hidden: int = 256, n_blocks: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_features, d_hidden),
                                  nn.BatchNorm1d(d_hidden), nn.ReLU())
        self.blocks = nn.ModuleList(
            [TabResNet._Block(d_hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for b in self.blocks:
            h = b(h)
        return self.head(F.relu(h)).squeeze(-1)


class MoE(nn.Module):
    """Supervised Mixture-of-Experts: gated weighted sum of expert MLP logits."""

    def __init__(self, n_features: int, d_hidden: int = 128, n_experts: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        def expert():
            return nn.Sequential(nn.Linear(n_features, d_hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_hidden, 1))

        self.experts = nn.ModuleList([expert() for _ in range(n_experts)])
        self.gate = nn.Sequential(nn.Linear(n_features, d_hidden), nn.ReLU(),
                                  nn.Linear(d_hidden, n_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = F.softmax(self.gate(x), dim=-1)                          # (B,E)
        logits = torch.stack([e(x).squeeze(-1) for e in self.experts], -1)  # (B,E)
        return (g * logits).sum(-1)                                 # (B,)


class InteractionNet(nn.Module):
    """Explicit pairwise feature-interaction network (service-interaction style).

    Appends all pairwise products x_i*x_j (a low-rank quadratic feature map) to the
    raw features, then feeds [x, pairwise] through an MLP — so interactions between
    service/survey features are modeled directly rather than left to the learner.
    """

    def __init__(self, n_features: int, d_hidden: int = 256, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        n_pairs = n_features * (n_features - 1) // 2
        # Fixed upper-triangular index pairs for the outer product.
        ii, jj = torch.triu_indices(n_features, n_features, offset=1)
        self.register_buffer("ii", ii)
        self.register_buffer("jj", jj)
        d_in = n_features + n_pairs
        layers = []
        d = d_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, d_hidden), nn.BatchNorm1d(d_hidden), nn.GELU(),
                       nn.Dropout(dropout)]
            d = d_hidden
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pairs = x[:, self.ii] * x[:, self.jj]      # (B, n_pairs)
        return self.mlp(torch.cat([x, pairs], dim=1)).squeeze(-1)


def build(name: str, n_features: int, **cfg) -> nn.Module:
    """Construct an architecture by name. Unrecognized names raise."""
    name = name.lower().replace("-", "").replace("_", "")
    if name in ("fttransformer", "ftt"):
        return FTTransformer(n_features, **cfg)
    if name in ("dcnv2", "dcn"):
        return DCNv2(n_features, **cfg)
    if name in ("dcnv2full", "dcnfull", "dcnv2canonical"):
        return DCNv2Full(n_features, **cfg)
    if name in ("factlog", "factorizedlogistic", "bilinearlogistic"):
        return FactLog(n_features, **cfg)
    if name in ("plrmlp", "plr"):
        return PLRMLP(n_features, **cfg)
    if name in ("tabresnet", "resnet"):
        return TabResNet(n_features, **cfg)
    if name in ("moe", "supervisedmoe"):
        return MoE(n_features, **cfg)
    if name in ("interactionnet", "servicenet", "serviceinteraction"):
        return InteractionNet(n_features, **cfg)
    raise ValueError(f"unknown neural architecture: {name!r}")
