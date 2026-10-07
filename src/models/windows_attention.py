"""Memory-efficient MQA attention for Windows builds without FlashAttention.

PyTorch's efficient attention kernel does not accept enable_gqa=True. Repeating
the KV heads explicitly implements the same attention and makes it eligible.
TabPFN's supported backend registry is used; installed packages stay intact.
Query chunks retain every key: each row's softmax has the same full context.
"""
from __future__ import annotations
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def expand_heads(q, k, v):
    if q.shape[2] % k.shape[2] or k.shape != v.shape:
        raise ValueError('Invalid grouped attention geometry')
    repeat = q.shape[2] // k.shape[2]
    if repeat == 1:
        return k, v
    return k.repeat_interleave(repeat, dim=2), v.repeat_interleave(repeat, dim=2)


class WindowsMQABackend:
    name = 'sol_windows_efficient_mqa'

    def __init__(self, query_chunk_size=2048):
        if query_chunk_size < 1:
            raise ValueError('query_chunk_size must be positive')
        self.query_chunk_size = query_chunk_size

    def is_preferred(self, spec):
        return (spec.device.type == 'cuda' and not spec.is_grad_enabled
                and spec.seq_len_kv is not None and spec.seq_len_kv >= 10000
                and (spec.num_heads > spec.num_kv_heads
                     or (spec.seq_len_q is not None and spec.seq_len_q > self.query_chunk_size)))

    def run(self, q, k, v, **kwargs):
        assert k is not None and v is not None
        # Match the dtype casting of the ordinary autocast SDPA path.
        if torch.is_autocast_enabled('cuda'):
            dtype = torch.get_autocast_dtype('cuda')
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        if q.shape[2] % k.shape[2] or k.shape != v.shape:
            raise ValueError('Invalid grouped attention geometry')
        # Expand in kernel layout to avoid a second full-context contiguous copy.
        repeat = q.shape[2] // k.shape[2]
        key = k.permute(0, 2, 1, 3).repeat_interleave(repeat, dim=1).contiguous()
        value = v.permute(0, 2, 1, 3).repeat_interleave(repeat, dim=1).contiguous()
        output = torch.empty_like(q)
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            for start in range(0, q.shape[1], self.query_chunk_size):
                stop = min(start + self.query_chunk_size, q.shape[1])
                out = torch.nn.functional.scaled_dot_product_attention(
                    q[:, start:stop].permute(0, 2, 1, 3).contiguous(),
                    key, value, dropout_p=0.0)
                output[:, start:stop] = out.permute(0, 2, 1, 3)
        return output


BACKEND = WindowsMQABackend()


def register():
    from tabpfn.architectures.shared.attention_backends import register_attention_backend
    register_attention_backend(BACKEND)


def verify_gpu_equivalence(dtype=torch.float16):
    """Numerical gate before any expensive run, with a small independent reference."""
    generator = torch.Generator(device='cuda').manual_seed(2701)
    q = torch.randn(2, 31, 8, 64, generator=generator, device='cuda', dtype=dtype)
    gaps = []
    with torch.no_grad():
        for heads in (1, 8):
            k = torch.randn(2, 43, heads, 64, generator=generator, device='cuda', dtype=dtype)
            v = torch.randn(2, 43, heads, 64, generator=generator, device='cuda', dtype=dtype)
            # Force multiple calls, including a partial final query chunk.
            actual = WindowsMQABackend(query_chunk_size=7).run(q, k, v)
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), enable_gqa=heads == 1).permute(0, 2, 1, 3)
            tolerance = 1.6e-2 if dtype == torch.bfloat16 else 2e-3
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            gaps.append(float((actual - expected).abs().max()))
    return max(gaps)
