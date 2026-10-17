"""Phase 20 -- multi-task neural with ordinal auxiliary loss (catalogue items 35+36).

Mechanism (vs the CLOSED SSL32 masked-reconstruction pretraining):
  * SUPERVISED end-to-end: the satisfaction loss and the auxiliary rating loss
    shape the SAME shared representation simultaneously (SSL32 was unsupervised
    pretraining followed by supervised fine-tuning -- a different, closed branch).
  * BERT-style INPUT MASKING: the 13 rating columns are dropped (set to the
    standardized mean) for a per-sample random subset during training. Without
    masking, an auxiliary head predicting ratings from a shared rep can be
    minimized by copying the rating inputs through idle dimensions -- no
    representation pressure at all. Masking forces the aux head to infer each
    masked rating from the OTHER features (incl. satisfaction-correlated
    structure), which is exactly the rating<->satisfaction coupling hypothesis.
  * ORDINAL aux loss (CORAL-style cumulative BCE over the 5 thresholds of a
    0..5 rating), item 36: respects the ordering that a generic MSE/CE would
    discard.

Fold-safety: training uses FIT only; early stopping is judged on satisfaction
AUC of an inner FIT split under INFERENCE conditions (no masking); the eval
fold is never touched by the loss or the scaler.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _RatingCORALHead(nn.Module):
    """Shared h -> 13 ratings x 5 cumulative thresholds logits."""

    def __init__(self, d_in: int, n_ratings: int = 13, n_thresholds: int = 5,
                 d_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, n_ratings * n_thresholds))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).view(h.shape[0], -1)  # (B, 13*5)


class MultiTaskPLR(nn.Module):
    """PLR trunk -> shared representation -> (main satisfaction logit, ordinal
    rating logits). forward(x, mask_ratings=None) -> (main_logit, aux_logits).

    x: (B, n_features) standardized. mask_ratings: optional (B, 13) bool tensor;
    True = the rating input column was zeroed for this sample (aux loss applies
    only at masked entries)."""

    RATING_SLICE = slice(0, 13)  # the 13 service ratings are view columns 0..12

    def __init__(self, n_features: int, n_ratings: int = 13, n_freq: int = 16,
                 d_trunk: int = 256, trunk_layers: int = 2, d_shared: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.n_ratings = n_ratings
        d_trunk_in = n_features * 2 * n_freq + n_features
        layers = []
        d = d_trunk_in
        for _ in range(trunk_layers):
            layers += [nn.Linear(d, d_trunk), nn.BatchNorm1d(d_trunk), nn.GELU(),
                       nn.Dropout(dropout)]
            d = d_trunk
        self.encoder = nn.Sequential(*layers)
        self.shared = nn.Sequential(nn.Linear(d_trunk, d_shared), nn.GELU())
        self.main_head = nn.Linear(d_shared, 1)
        self.aux_head = _RatingCORALHead(d_shared, n_ratings)
        self.omega = nn.Parameter(torch.randn(n_features, n_freq) * 0.5)
        self.phi = nn.Parameter(torch.rand(n_features, n_freq))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        ang = 2.0 * torch.pi * (x.unsqueeze(-1) * self.omega + self.phi)
        emb = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1).flatten(1)
        return self.shared(self.encoder(torch.cat([emb, x], dim=1)))

    def forward(self, x: torch.Tensor, mask_ratings: torch.Tensor | None = None):
        if mask_ratings is not None:
            x = x.clone()
            x[:, self.RATING_SLICE] = x[:, self.RATING_SLICE].masked_fill(
                mask_ratings, 0.0)
        h = self.embed(x)
        return self.main_head(h).squeeze(-1), self.aux_head(h)


def coral_targets(ratings: torch.Tensor, n_thresholds: int = 5) -> torch.Tensor:
    """ratings: (B, 13) int64 in 0..5 -> (B, 13*5) float cumulative targets 1[r>k]."""
    th = torch.arange(n_thresholds, device=ratings.device).view(1, 1, -1)
    return (ratings.unsqueeze(-1) > th).float().view(ratings.shape[0], -1)


def aux_loss(aux_logits: torch.Tensor, ratings: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
    """CORAL BCE restricted to masked entries. Returns 0.0 (differentiable) if
    the batch has no masked rating."""
    if not mask.any():
        return aux_logits.sum() * 0.0
    tgt = coral_targets(ratings)                      # (B, 13*5)
    m = mask.unsqueeze(-1).expand(-1, -1, 5).reshape_as(tgt)
    return nn.functional.binary_cross_entropy_with_logits(
        aux_logits[m], tgt[m])
