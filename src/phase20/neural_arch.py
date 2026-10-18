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
    # --- Phase-20 batch 2 (catalogue items 16, 18-26) ---
    if name in ("saint",):
        return SAINT(n_features, **cfg)
    if name in ("autoint",):
        return AutoInt(n_features, **cfg)
    if name in ("deepfm",):
        return DeepFM(n_features, **cfg)
    if name in ("xdeepfm", "cin"):
        return xDeepFM(n_features, **cfg)
    if name in ("node", "softtrees"):
        return NODE(n_features, **cfg)
    if name in ("grande",):
        return GRANDE(n_features, **cfg)
    if name in ("tabnet",):
        return TabNet(n_features, **cfg)
    if name in ("excelformer", "spa"):
        return ExcelFormer(n_features, **cfg)
    if name in ("amformer", "localityattention"):
        return AMFormer(n_features, **cfg)
    if name in ("t2gformer", "t2g"):
        return T2GFormer(n_features, **cfg)
    if name in ("trompt",):
        return Trompt(n_features, **cfg)
    raise ValueError(f"unknown neural architecture: {name!r}")

# ---------------------------------------------------------------------------
# Phase-20 batch 2: catalogue items 16, 18-26 (SAINT, AutoInt, DeepFM/xDeepFM,
# NODE, GRANDE, TabNet, ExcelFormer, AMFormer, T2G-Former, Trompt).
# All consume standardized numeric x:(B, n_features) and return logit (B,).
# Canonical inductive biases, compact sizes (route view = 22 features).
# ---------------------------------------------------------------------------


def _sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation (Martins & Astudillo, 2016) - sparse simplex proj."""
    dim = dim if dim >= 0 else dim + z.dim()
    zz, _ = torch.sort(z, dim=dim, descending=True)
    cum = torch.cumsum(zz, dim=dim)
    k = torch.arange(1, zz.shape[dim] + 1, device=z.device, dtype=z.dtype)
    k = k.view([1 if i != dim else -1 for i in range(z.dim())])
    support = zz - (cum - 1.0) / k > 0
    k_z = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (cum.gather(dim, k_z - 1) - 1.0) / k_z
    return torch.clamp(z - tau, min=0.0)


class _TriangularScalarAttention(nn.Module):
    """Additive (content-based) attention of one query over F feature tokens."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:  # tok:(B,F,d)->(B,d)
        h = torch.tanh(self.proj(tok))                     # (B,F,d)
        a = torch.softmax(h.mean(-1), dim=1)               # (B,F)
        return (a.unsqueeze(-1) * tok).sum(1)              # (B,d)


class SAINT(nn.Module):
    """SAINT: interleaved column attention (per row over features) and row
    attention (per feature over in-batch rows, chunk-bounded) + FFN blocks."""

    def __init__(self, n_features: int, d_token: int = 64, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1, row_chunk: int = 1024):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d_token))
        self.b = nn.Parameter(torch.zeros(n_features, d_token))
        self.row_chunk = row_chunk
        col_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=2 * d_token,
            dropout=dropout, batch_first=True, norm_first=True)
        self.col = nn.ModuleList(
            [nn.TransformerEncoder(col_layer, num_layers=1) for _ in range(n_layers)])
        row_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=1, dim_feedforward=2 * d_token,
            dropout=dropout, batch_first=True, norm_first=True)
        self.row = nn.ModuleList(
            [nn.TransformerEncoder(row_layer, num_layers=1) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(nn.Linear(d_token, d_token), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_token, 1))
        nn.init.normal_(self.W, std=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(-1) * self.W + self.b                 # (B,F,d)
        for c, r in zip(self.col, self.row):
            h = c(h)                                           # col attention
            b, f, d = h.shape
            hc = h.reshape(f, b, d)                            # (F,B,d): per feature over rows
            n = hc.shape[1]
            outs = []
            for s in range(0, n, self.row_chunk):              # bounded row attention
                outs.append(r(hc[:, s:s + self.row_chunk, :]))
            h = torch.cat(outs, dim=1).reshape(b, f, d)        # row attention
        return self.head(self.norm(h.mean(1))).squeeze(-1)


class AutoInt(nn.Module):
    """AutoInt (Song et al. 2019): multi-head self-attention over feature tokens
    with residual linear layers, then head over the final token set."""

    def __init__(self, n_features: int, d_token: int = 64, n_layers: int = 3,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d_token))
        self.b = nn.Parameter(torch.zeros(n_features, d_token))
        self.attn = nn.ModuleList()
        self.res = nn.ModuleList()
        for _ in range(n_layers):
            self.attn.append(nn.MultiheadAttention(
                d_token, n_heads, dropout=dropout, batch_first=True))
            self.res.append(nn.Linear(d_token, d_token))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token * n_features, 1)
        nn.init.normal_(self.W, std=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(-1) * self.W + self.b                 # (B,F,d)
        for a, rl in zip(self.attn, self.res):
            o, _ = a(h, h, h, need_weights=False)
            h = self.norm(rl(h) + self.drop(o))
        return self.head(h.flatten(1)).squeeze(-1)


class DeepFM(nn.Module):
    """DeepFM (Guo et al. 2017): linear + 2nd-order FM on feature embeddings
    (e_j = x_j * w_j) + deep MLP. FM factorizes pairwise interactions."""

    def __init__(self, n_features: int, d_emb: int = 16, d_hidden: int = 128,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.E = nn.Parameter(torch.randn(n_features, d_emb) / n_features ** 0.5)
        self.linear = nn.Linear(n_features, 1)
        deep = []
        d = n_features
        for _ in range(n_layers):
            deep += [nn.Linear(d, d_hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = d_hidden
        deep += [nn.Linear(d, 1)]
        self.deep = nn.Sequential(*deep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = x.unsqueeze(-1) * self.E                          # (B,F,d)
        s = e.sum(1)                                          # (B,d)
        fm = 0.5 * (s * s - (e * e).sum(1)).sum(-1)            # (B,)
        return self.linear(x).squeeze(-1) + fm + self.deep(x).squeeze(-1)


class xDeepFM(nn.Module):
    """xDeepFM (Lian et al. 2018): CIN (compressed interaction network) +
    deep MLP + linear. CIN builds explicit bounded-degree feature interactions
    with vector-wise compression, distinct from FM's factorization."""

    def __init__(self, n_features: int, d_emb: int = 16, cin_dims: tuple = (32, 32),
                 d_hidden: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.E = nn.Parameter(torch.randn(n_features, d_emb) / n_features ** 0.5)
        self.linear = nn.Linear(n_features, 1)
        # CIN layer k: W_k (d_out, H_prev * n_features) compresses the field-pair
        # outer product H_prev x F -> d_out maps (applied over the map axis).
        self.cin_W = nn.ParameterList()
        prev = n_features
        for d_out in cin_dims:
            self.cin_W.append(
                nn.Parameter(torch.randn(d_out, prev * n_features) * 0.05))
            prev = d_out
        self.cin_out = nn.Linear(d_emb * len(cin_dims), 1)
        deep = []
        d = n_features
        for _ in range(n_layers):
            deep += [nn.Linear(d, d_hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = d_hidden
        deep += [nn.Linear(d, 1)]
        self.deep = nn.Sequential(*deep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x.unsqueeze(-1) * self.E                         # (B,F,d)
        xl, parts = x0, []
        for w in self.cin_W:                                  # w: (d_out, H_prev*F)
            b, h, d = xl.shape
            outer = (xl.unsqueeze(2) * x0.unsqueeze(1)).reshape(b, h * x.shape[1], d)
            xl = torch.einsum("kf,bfd->bkd", w, outer)        # (B,D_out,d)
            parts.append(xl.sum(1))                           # (B,d)
        cin = self.cin_out(torch.cat(parts, dim=1)).squeeze(-1)
        return self.linear(x).squeeze(-1) + cin + self.deep(x).squeeze(-1)


class SoftTree(nn.Module):
    """Differentiable oblivious soft tree (NODE-style): per-depth shared gates
    sigmoid((x.w_d+b_d)/tau), path-probability products, learnable leaf values."""

    def __init__(self, n_features: int, depth: int = 6, n_trees: int = 8,
                 temperature: float = 1.0):
        super().__init__()
        self.depth = depth
        self.n_trees = n_trees
        self.tau = temperature
        self.gates = nn.ParameterList()
        nodes = 1
        for _ in range(depth):
            self.gates.append(nn.Parameter(torch.randn(n_trees, nodes, n_features) * 0.1))
            nodes *= 2
        self.leaf = nn.Parameter(torch.zeros(n_trees, 2 ** depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = torch.ones(b, self.n_trees, 1, device=x.device, dtype=x.dtype)
        for w in self.gates:
            g = torch.sigmoid((torch.einsum('bf,tnf->btn', x, w)) / self.tau)  # (B,T,N)
            # children interleaved: index 2n = left (1-g), 2n+1 = right (g)
            h = torch.stack([h * (1 - g), h * g], dim=-1).reshape(b, self.n_trees, -1)
        return (h * torch.sigmoid(self.leaf)).sum(-1).mean(-1)  # (B,)


class NODE(nn.Module):
    """Neural Oblivious Decision Ensembles (Popovidchik et al. 2020): ensemble
    of differentiable soft oblivious trees, summed with a linear skip."""

    def __init__(self, n_features: int, depth: int = 6, n_trees: int = 8,
                 temperature: float = 1.0):
        super().__init__()
        self.trees = SoftTree(n_features, depth=depth, n_trees=n_trees,
                              temperature=temperature)
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trees(x) + self.linear(x).squeeze(-1)


class GRANDE(nn.Module):
    """GRANDE (Arik et al. 2024, distilled): genetically-informed random deep
    ensemble - K subnetworks, each seeing a fixed random feature mask and a
    learnable gate; sum of gated subnetwork logits."""

    def __init__(self, n_features: int, n_subnets: int = 8, d_hidden: int = 64,
                 mask_keep: float = 0.5, seed: int = 1, dropout: float = 0.1):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        masks = torch.rand(n_subnets, n_features, generator=g) < mask_keep
        masks |= ~masks.any(1, keepdim=True)                  # never all-zero
        self.register_buffer("masks", masks.float())
        self.sub = nn.ModuleList([
            nn.Sequential(nn.Linear(n_features, d_hidden), nn.ReLU(),
                          nn.Dropout(dropout), nn.Linear(d_hidden, 1))
            for _ in range(n_subnets)])
        self.gates = nn.Parameter(torch.zeros(n_subnets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xm = x.unsqueeze(1) * self.masks.unsqueeze(0)         # (B,K,F)
        logits = torch.cat([s(xm[:, i]) for i, s in enumerate(self.sub)], dim=1)
        return (torch.sigmoid(self.gates) * logits.squeeze(-1)).sum(-1)  # (B,)


class TabNet(nn.Module):
    """TabNet (Arik & Pfister 2019), compact adaptation: steps of attentive-transformor
    masking (sparsemax over hidden units with prior/D-matrix inhibition) + FC
    layers; step logits summed. Masks live in hidden space (d), not raw-feature
    space, so the mask can gate the recurrent hidden state h (B,d) directly."""

    def __init__(self, n_features: int, d_hidden: int = 64, n_steps: int = 5,
                 dropout: float = 0.1, relaxation: float = 1.5):
        super().__init__()
        self.n_steps = n_steps
        self.gamma = relaxation
        self.bn = nn.BatchNorm1d(n_features)
        self.init = nn.Linear(n_features, d_hidden)
        self.att = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.arrow = nn.ModuleList()
        for _ in range(n_steps):
            self.att.append(nn.Linear(d_hidden, d_hidden))
            self.fc.append(nn.Sequential(
                nn.Linear(d_hidden, d_hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(d_hidden, d_hidden)))
            self.arrow.append(nn.Linear(d_hidden, d_hidden))
        self.heads = nn.ModuleList([nn.Linear(d_hidden, 1) for _ in range(n_steps)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.init(self.bn(x)))                     # (B,d)
        prior = torch.zeros_like(h)
        out = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i in range(self.n_steps):
            m = _sparsemax(self.att[i](h) - prior)            # (B,d) sparse mask
            m = m * F.relu(self.arrow[i](h))                  # direction-aware masking
            h = self.fc[i](h * m)                             # masked transform
            out = out + self.heads[i](h).squeeze(-1)          # step logit
            prior = self.gamma * m                            # D-matrix prior
        return out


class ExcelFormer(nn.Module):
    """ExcelFormer (Yan et al. 2024, distilled): semi-permeable attention -
    attention weights gated elementwise by a sigmoid of their own logits
    (membrane that blocks low-affinity pairs), + FFN."""

    def __init__(self, n_features: int, d_token: int = 64, n_layers: int = 3,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d_token))
        self.b = nn.Parameter(torch.zeros(n_features, d_token))
        self.n_heads = n_heads
        self.dh = d_token // n_heads
        self.qkv = nn.ModuleList([nn.Linear(d_token, 3 * d_token) for _ in range(n_layers)])
        self.alpha = nn.ParameterList(
            [nn.Parameter(torch.ones(n_heads)) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.Linear(d_token, 2 * d_token), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_token, d_token)) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(nn.Linear(d_token, d_token), nn.GELU(),
                                  nn.Linear(d_token, 1))
        nn.init.normal_(self.W, std=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(-1) * self.W + self.b
        for qa, ffn, a in zip(self.qkv, self.ffn, self.alpha):
            b, f, d = h.shape
            qkv = qa(h).reshape(b, f, 3, self.n_heads, self.dh)
            q, k, v = (qkv[:, :, i].permute(0, 2, 1, 3) for i in range(3))
            att = (q @ k.transpose(-2, -1)) / self.dh ** 0.5    # (B,H,F,F)
            soft = F.softmax(att, dim=-1)
            gate = torch.sigmoid(a.view(1, -1, 1, 1) * att)     # semi-permeable
            w = soft * gate
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
            o = (w @ v).transpose(1, 2).reshape(b, f, d)
            h = self.norm(h + self.drop(o) + ffn(h))
        return self.head(h.mean(1)).squeeze(-1)


class AMFormer(nn.Module):
    """AMFormer (Zhao et al. 2024, distilled): attention with multi-head
    locality - each head attends over its own contiguous slice of feature
    tokens (local receptive field per head) + shared FFN."""

    def __init__(self, n_features: int, d_token: int = 64, n_layers: int = 3,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d_token))
        self.b = nn.Parameter(torch.zeros(n_features, d_token))
        self.n_heads = n_heads
        self.dh = d_token // n_heads
        # contiguous per-head feature slices
        slices = [s.tolist() for s in torch.tensor_split(torch.arange(n_features), n_heads)]
        idx = torch.cat([torch.tensor(s) for s in slices])
        self.register_buffer("slice_idx", idx)
        self.slice_sizes = [len(s) for s in slices]
        self.qkv = nn.ModuleList([nn.Linear(d_token, 3 * d_token) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.Linear(d_token, 2 * d_token), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_token, d_token)) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(nn.Linear(d_token, d_token), nn.GELU(),
                                  nn.Linear(d_token, 1))
        nn.init.normal_(self.W, std=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(-1) * self.W + self.b
        for qa, ffn in zip(self.qkv, self.ffn):
            b, f, d = h.shape
            qkv = qa(h).reshape(b, f, 3, self.n_heads, self.dh)
            q, k, v = (qkv[:, :, i].permute(0, 2, 1, 3) for i in range(3))
            hs = torch.split(self.slice_idx.to(q.device), self.slice_sizes)
            outs = []
            for hi, sl in enumerate(hs):        # head hi: keys/values live in its
                ki, vi = k[:, hi][:, sl], v[:, hi][:, sl]   # local feature slice;
                att = F.softmax((q[:, hi] @ ki.transpose(-2, -1)) / self.dh ** 0.5,
                                dim=-1)                       # all F queries attend
                outs.append(att @ vi)                         # (B,F,dh)
            o = torch.cat(outs, dim=-1).reshape(b, f, d)      # (B,F,d)
            h = self.norm(h + self.drop(o) + ffn(h))
        return self.head(h.mean(1)).squeeze(-1)


class T2GFormer(nn.Module):
    """T2G-Former (Zhao et al. 2023, distilled): tokens-to-gates transformer -
    feature tokens pass through self-attention; attention-pooled per-expert
    gates then modulate a mixture of MLP experts on the raw features."""

    def __init__(self, n_features: int, d_token: int = 64, n_layers: int = 2,
                 n_heads: int = 4, n_experts: int = 4, d_hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d_token))
        self.b = nn.Parameter(torch.zeros(n_features, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=2 * d_token,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.queries = nn.Parameter(torch.randn(n_experts, d_token) * 0.1)
        self.gate_proj = nn.Linear(d_token, 1)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(n_features, d_hidden), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(d_hidden, 1))
            for _ in range(n_experts)])
        nn.init.normal_(self.W, std=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.encoder(x.unsqueeze(-1) * self.W + self.b)    # (B,F,d)
        scores = torch.einsum('bfd,gd->bfg', tok, self.queries)  # (B,F,G)
        a = F.softmax(scores, dim=1)
        pooled = torch.einsum('bfg,bfd->bgd', a, tok)            # (B,G,d)
        gates = torch.sigmoid(self.gate_proj(pooled)).squeeze(-1)  # (B,G)
        logits = torch.cat([e(x) for e in self.experts], dim=1).squeeze(-1)
        return (gates * logits).sum(-1)


class Trompt(nn.Module):
    """Trompt (Wu et al. 2023, distilled): parallel prompts - P learnable prompt
    vectors each attend over feature tokens; per-prompt heads; averaged logit."""

    def __init__(self, n_features: int, d_token: int = 64, n_prompts: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d_token))
        self.b = nn.Parameter(torch.zeros(n_features, d_token))
        self.prompts = nn.Parameter(torch.randn(n_prompts, d_token) * 0.1)
        self.attn = _TriangularScalarAttention(d_token)
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(d_token, d_token), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_token, 1)) for _ in range(n_prompts)])
        nn.init.normal_(self.W, std=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = x.unsqueeze(-1) * self.W + self.b                # (B,F,d)
        hs = []
        for p, hd in zip(self.prompts, self.heads):
            ctx = self.attn(tok + p.view(1, 1, -1))             # (B,d)
            hs.append(hd(ctx).squeeze(-1))
        return torch.stack(hs, -1).mean(-1)                    # (B,)
