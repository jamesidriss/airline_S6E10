"""Small satisfaction-label-free masked covariate encoder.

Age and Flight Distance have numeric and categorical twins; masking either field
hides both representations. Rating 0 stays an explicit category, never missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.features.view import RAW21


def prepare_covariates(frame):
    if list(frame.columns) != RAW21:
        raise ValueError("encoder accepts exactly the 21 covariates, no id/label/TE")
    category_cols = RAW21[:19]  # 13 ratings, META4, Age, Flight Distance
    maps, codes = {}, []
    for col in category_cols:
        v = frame[col]
        levels = sorted(v.dropna().unique().tolist())
        maps[col] = levels
        codes.append(pd.Categorical(v, categories=levels).codes.astype("int64"))
    cat = np.column_stack(codes)
    if (cat < 0).any():
        raise ValueError("unexpected missing categorical covariate")
    numeric = frame[RAW21[17:21]].to_numpy(dtype="float32")
    missing = ~np.isfinite(numeric)
    median = np.nanmedian(numeric, axis=0)
    numeric = np.where(missing, median, numeric)
    # Delays are long tailed; keep zeros explicit and compress the positive tail.
    numeric[:, 2:] = np.log1p(np.maximum(numeric[:, 2:], 0))
    mean, scale = numeric.mean(axis=0), numeric.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    numeric = (numeric - mean) / scale
    return cat, numeric, missing.astype("float32"), {"maps": maps, "median": median.tolist(),
                "mean": mean.tolist(), "scale": scale.tolist(), "category_cols": category_cols,
                "cardinalities": [len(maps[c]) for c in category_cols]}


class MaskedEncoder(nn.Module):
    def __init__(self, cardinalities, width=256, latent=32, embedding=8):
        super().__init__()
        self.cardinalities = cardinalities
        self.embeddings = nn.ModuleList([nn.Embedding(n + 1, embedding) for n in cardinalities])
        self.encoder = nn.Sequential(nn.Linear(len(cardinalities) * embedding + 4 + 4 + 21, width),
                                     nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 128), nn.SiLU())
        self.heads = nn.ModuleList([nn.Linear(128, n) for n in cardinalities[:17]])
        self.numeric_head = nn.Linear(128, 4)

    def encode(self, cat, numeric, missing, mask):
        emb = [e(torch.where(mask[:, j], self.cardinalities[j], cat[:, j]))
               for j, e in enumerate(self.embeddings)]
        values = torch.where(mask[:, 17:21], 0., numeric)
        return self.encoder(torch.cat(emb + [values, missing, mask.to(numeric.dtype)], dim=1))

    def forward(self, cat, numeric, missing, mask):
        z = self.encode(cat, numeric, missing, mask)
        h = self.decoder(z)
        return z, [head(h) for head in self.heads], self.numeric_head(h)


def masked_loss(model, cat, numeric, missing, mask):
    _, heads, nhead = model(cat, numeric, missing, mask)
    total, denom = torch.zeros((), device=cat.device), torch.zeros((), device=cat.device)
    for j, logits in enumerate(heads):
        weight = 1.5 if j in (15, 16) else 1.0  # travel type and Class
        losses = F.cross_entropy(logits, cat[:, j], reduction="none")
        total = total + weight * (losses * mask[:, j]).sum()
        denom = denom + weight * mask[:, j].sum()
    losses = F.smooth_l1_loss(nhead, numeric, reduction="none")
    weights = torch.tensor([.25, .25, 1., 1.], device=cat.device)
    observed = (~missing.bool()) & mask[:, 17:21]
    total = total + (losses * observed * weights).sum()
    denom = denom + (observed * weights).sum()
    return total / denom.clamp_min(1)
