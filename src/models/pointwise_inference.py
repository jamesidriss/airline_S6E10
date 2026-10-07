"""Batch the row-independent TabPFN decoder projection without reducing context."""
from __future__ import annotations
from types import MethodType
import torch


def install_decoder_projection_chunks(model, rows=32768):
    if rows < 1:
        raise ValueError('Projection chunk size must be positive')
    heads = model.heads
    if hasattr(heads, '_sol_original_project_decoder_keys'):
        raise ValueError('Projection batching is already installed')
    if type(heads).__name__ != 'MultiTaskHeads':
        raise TypeError('Only the audited TabPFN 3.5 MultiTaskHeads is supported')
    original = heads.project_decoder_keys
    heads._sol_original_project_decoder_keys = original

    def project(self, train_emb):
        if torch.is_grad_enabled() or train_emb.shape[1] <= rows:
            return self._sol_original_project_decoder_keys(train_emb)
        # Norm, residual MLP and key projection operate only on each row's
        # feature dimension. Every context row still contributes its keys.
        result = None
        for start in range(0, train_emb.shape[1], rows):
            stop = min(start+rows, train_emb.shape[1])
            part = self._sol_original_project_decoder_keys(train_emb[:, start:stop])
            if result is None:
                result = torch.empty((part.shape[0], train_emb.shape[1], *part.shape[2:]),
                                     dtype=part.dtype, device=part.device)
            result[:, start:stop] = part
        return result

    heads.project_decoder_keys = MethodType(project, heads)
    return model


def install_decoder_gelu_reuse(model, rows=32768):
    """Reuse the fresh linear-output buffer at the decoder's GELU activation.

    Unlike chunking linear projections, this preserves their complete matrix
    dimensions and algorithms. The activation has no cross-row dependency.
    """
    if rows < 1:
        raise ValueError('Activation chunk size must be positive')
    activations = [m for m in model.heads.mlp_classification.mlp.modules()
                   if isinstance(m, torch.nn.GELU)]
    if len(activations) != 1:
        raise TypeError('Expected the audited decoder Linear/GELU/Linear MLP')
    activation = activations[0]
    if hasattr(activation, '_sol_original_forward'):
        raise ValueError('Activation reuse is already installed')
    activation._sol_original_forward = activation.forward

    def forward(self, x):
        if torch.is_grad_enabled():
            return self._sol_original_forward(x)
        axis = x.ndim-2
        for start in range(0, x.shape[axis], rows):
            part = x.narrow(axis, start, min(rows, x.shape[axis]-start))
            part.copy_(self._sol_original_forward(part))
        return x

    activation.forward = MethodType(forward, activation)
    return model
