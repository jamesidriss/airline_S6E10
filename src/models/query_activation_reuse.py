"""Reuse fresh query-scaling GELU buffers; preserve complete linear projections."""
from __future__ import annotations
from types import MethodType
import torch


def install_query_activation_reuse(model,rows=32768):
    from tabpfn.architectures.tabpfn_v3_5 import SoftmaxScalingMLP
    if rows<1:raise ValueError('Activation chunk rows must be positive')
    activations=[a for m in model.modules() if isinstance(m,SoftmaxScalingMLP)
        for a in m.query_mlp.modules() if isinstance(a,torch.nn.GELU)]
    if not activations:raise TypeError('Expected TabPFN3.5 query scaling activations')
    for activation in activations:
        if hasattr(activation,'_sol_original_query_gelu'):raise ValueError('Query activation reuse already installed')
        activation._sol_original_query_gelu=activation.forward
        def forward(self,x):
            if torch.is_grad_enabled() or x.device.type!='cuda' or x.ndim!=4 or x.shape[1]<=rows:
                return self._sol_original_query_gelu(x)
            # Query scaling is (batch, sequence, head, hidden). Chunk only the
            # elementwise activation along sequence; never either Linear call.
            for start in range(0,x.shape[1],rows):
                part=x[:,start:min(start+rows,x.shape[1])]
                part.copy_(self._sol_original_query_gelu(part))
            return x
        activation.forward=MethodType(forward,activation)
    return len(activations)
