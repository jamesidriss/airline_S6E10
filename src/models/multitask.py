"""Matched satisfaction head and covariate reconstruction regularization."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from src.models.masked import MaskedEncoder, masked_loss


class SatisfactionMultitask(nn.Module):
    def __init__(self, cardinalities, feature_count):
        super().__init__()
        self.covariate = MaskedEncoder(cardinalities)
        self.satisfaction = nn.Sequential(nn.Linear(32 + feature_count, 256), nn.SiLU(),
            nn.Dropout(.05), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, features, cat, numeric, missing):
        mask = torch.zeros((len(cat), 21), dtype=torch.bool, device=cat.device)
        latent = self.covariate.encode(cat, numeric, missing, mask)
        return self.satisfaction(torch.cat([features, latent], dim=1)).flatten()

    def auxiliary_loss(self, cat, numeric, missing, mask):
        return masked_loss(self.covariate, cat, numeric, missing, mask)


def normalization_fit(matrix, train_rows, chunk=8192):
    """Training-only moments with bounded temporary memory."""
    total = np.zeros(matrix.shape[1], dtype='float64')
    squares = total.copy()
    for begin in range(0, len(train_rows), chunk):
        block = matrix[train_rows[begin:begin+chunk]].astype('float64')
        total += block.sum(axis=0)
        squares += np.einsum('ij,ij->j', block, block)
    mean = total / len(train_rows)
    scale = np.sqrt(np.maximum(squares / len(train_rows) - mean**2, 0))
    return mean.astype('float32'), np.maximum(scale, 1e-6).astype('float32')


def normalized_gpu(matrix, mean, scale, chunk=8192):
    result = torch.empty(matrix.shape, device='cuda', dtype=torch.float32)
    for begin in range(0, len(matrix), chunk):
        block = np.clip((matrix[begin:begin+chunk] - mean) / scale, -10, 10)
        result[begin:begin+len(block)] = torch.from_numpy(block).to('cuda')
    return result
