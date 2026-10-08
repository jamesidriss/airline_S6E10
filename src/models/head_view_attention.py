"""Opt-in MQA head views; preserve kernel dimensions and every context row."""
from __future__ import annotations
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from src.models.windows_attention import WindowsMQABackend


class HeadViewBackend(WindowsMQABackend):
    name = 'sol_windows_mqa_head_views'

    def run(self,q,k,v,**kwargs):
        if k is None or v is None or k.shape[2] not in (1,q.shape[2]) or torch.is_grad_enabled():
            return super().run(q,k,v,**kwargs)
        if k.shape!=v.shape:
            raise ValueError('Invalid grouped attention geometry')
        if torch.is_autocast_enabled('cuda'):
            dtype=torch.get_autocast_dtype('cuda')
            q,k,v=q.to(dtype),k.to(dtype),v.to(dtype)
        # MHA needs only a permutation view; MQA uses a zero head stride.
        # Logical SDPA dimensions match the materialized original path.
        key=k.permute(0,2,1,3).expand(-1,q.shape[2],-1,-1)
        value=v.permute(0,2,1,3).expand(-1,q.shape[2],-1,-1)
        independent=q.untyped_storage().data_ptr() not in {
            k.untyped_storage().data_ptr(),v.untyped_storage().data_ptr()}
        output=q if self.reuse_query_output and independent else torch.empty_like(q)
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            for start in range(0,q.shape[1],self.query_chunk_size):
                stop=min(start+self.query_chunk_size,q.shape[1])
                out=torch.nn.functional.scaled_dot_product_attention(
                    q[:,start:stop].permute(0,2,1,3).contiguous(),key,value,dropout_p=0.0)
                output[:,start:stop]=out.permute(0,2,1,3)
        return output


def register_head_views():
    from tabpfn.architectures.shared.attention_backends import register_attention_backend
    backend=HeadViewBackend(reuse_query_output=True)
    register_attention_backend(backend)
    return backend
