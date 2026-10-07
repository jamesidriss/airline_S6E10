"""Memory-efficient MQA attention for Windows builds without FlashAttention.

PyTorch's efficient attention kernel does not accept enable_gqa=True. Repeating
the KV heads explicitly implements the same attention and makes it eligible.
TabPFN's supported backend registry is used; installed packages stay intact.
"""
from __future__ import annotations
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def expand_heads(q, k, v):
    if q.shape[2] % k.shape[2] or k.shape != v.shape:
        raise ValueError('Invalid grouped attention geometry')
    repeat = q.shape[2] // k.shape[2]
    return k.repeat_interleave(repeat, dim=2), v.repeat_interleave(repeat, dim=2)


class WindowsMQABackend:
    name = 'sol_windows_efficient_mqa'

    def is_preferred(self, spec):
        return (spec.device.type == 'cuda' and not spec.is_grad_enabled
                and spec.num_heads > spec.num_kv_heads
                and spec.seq_len_kv is not None and spec.seq_len_kv >= 10000)

    def run(self, q, k, v, **kwargs):
        assert k is not None and v is not None
        # Match the dtype casting of the ordinary autocast SDPA path.
        if torch.is_autocast_enabled('cuda'):
            dtype = torch.get_autocast_dtype('cuda')
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        k, v = expand_heads(q, k, v)
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3).contiguous(),
                k.permute(0, 2, 1, 3).contiguous(),
                v.permute(0, 2, 1, 3).contiguous(), dropout_p=0.0)
        return out.permute(0, 2, 1, 3)


BACKEND = WindowsMQABackend()


def register():
    from tabpfn.architectures.shared.attention_backends import register_attention_backend
    register_attention_backend(BACKEND)


def verify_gpu_equivalence():
    """Numerical gate before any expensive run, with a small independent reference."""
    generator = torch.Generator(device='cuda').manual_seed(2701)
    q = torch.randn(2, 31, 8, 64, generator=generator, device='cuda', dtype=torch.float16)
    k = torch.randn(2, 43, 1, 64, generator=generator, device='cuda', dtype=torch.float16)
    v = torch.randn(2, 43, 1, 64, generator=generator, device='cuda', dtype=torch.float16)
    with torch.no_grad():
        actual = BACKEND.run(q, k, v)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            expected = torch.nn.functional.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), enable_gqa=True).permute(0, 2, 1, 3)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    return float((actual - expected).abs().max())
