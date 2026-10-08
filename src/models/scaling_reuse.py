"""Reuse fresh SoftmaxScalingMLP outputs; keep queries and GEMM shapes intact."""
from __future__ import annotations
from types import MethodType
import torch


def install_scaling_reuse(model,min_rows=10000):
    from tabpfn.architectures.tabpfn_v3_5 import SoftmaxScalingMLP,_safe_log_seqlen
    if min_rows<1:
        raise ValueError('min_rows must be positive')
    modules=[m for m in model.modules() if isinstance(m,SoftmaxScalingMLP)]
    if not modules:
        raise TypeError('Expected audited TabPFN3.5 query scaling modules')
    for module in modules:
        if hasattr(module,'_sol_original_scaling_forward'):
            raise ValueError('Query scaling reuse is already installed')
        module._sol_original_scaling_forward=module.forward
        def forward(self,q,n):
            if torch.is_grad_enabled() or q.device.type!='cuda' or q.shape[1]<min_rows:
                return self._sol_original_scaling_forward(q,n)
            logn=_safe_log_seqlen(n,q.device,q.dtype).reshape(1,1)
            base=self.base_mlp(logn).view(1,1,self.num_heads,self.head_dim)
            # Both complete linear projections retain their original dimensions.
            # Their last output is fresh; no query/KV/input storage is overwritten.
            modulation=self.query_mlp(q)
            if (torch.result_type(base,modulation)!=modulation.dtype
                    or torch.result_type(q,modulation)!=modulation.dtype):
                del modulation,base
                return self._sol_original_scaling_forward(q,n)
            torch.tanh(modulation,out=modulation)
            modulation.add_(1)
            torch.mul(base,modulation,out=modulation)
            torch.mul(q,modulation,out=modulation)
            return modulation
        module.forward=MethodType(forward,module)
    return len(modules)
