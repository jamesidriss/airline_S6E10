"""Target-independent foundation-model frames and exact MQA head expansion."""
import numpy as np
import pandas as pd
import torch
from scripts.run_sol_tabpfn import frames
from src.features.view import RAW21
from src.models.windows_attention import expand_heads


def example():
    data = {c: np.arange(12) % 6 for c in RAW21}
    for c in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
        data[c] = ['a', 'b'] * 6
    data['Flight Distance'] = np.arange(12) * 117 + 33
    data['satisfaction'] = np.arange(12) % 2
    data['id'] = np.arange(12) + 5000
    return pd.DataFrame(data)


def test_foundation_frame_ignores_targets_ids_and_poisoned_te():
    original = example()
    poisoned = original.copy()
    poisoned['satisfaction'] = 1 - original['satisfaction']
    poisoned['id'] = original['id'] + 100000
    poisoned['te_poison'] = poisoned['satisfaction'] * 1e9
    a = frames(original.iloc[:9], original.iloc[9:], True)
    b = frames(poisoned.iloc[:9], poisoned.iloc[9:], True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[2:] == b[2:]
    assert a[2] == RAW21 + ['Flight Distance coarse10']
    assert a[3][-2:] == [RAW21.index('Flight Distance'), len(RAW21)]
    assert np.array_equal(a[0][:, -1], original['Flight Distance'].to_numpy()[:9] // 10)


def test_explicit_head_expansion_matches_grouped_attention():
    generator = torch.Generator().manual_seed(701)
    q = torch.randn(2, 7, 4, 8, generator=generator)
    k = torch.randn(2, 13, 1, 8, generator=generator)
    v = torch.randn(2, 13, 1, 8, generator=generator)
    expanded_k, expanded_v = expand_heads(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), enable_gqa=True)
    actual = torch.nn.functional.scaled_dot_product_attention(q.permute(0, 2, 1, 3), expanded_k.permute(0, 2, 1, 3), expanded_v.permute(0, 2, 1, 3))
    torch.testing.assert_close(actual, expected)


def test_invalid_grouped_head_geometry_is_rejected():
    try:
        expand_heads(torch.zeros(1, 2, 3, 8), torch.zeros(1, 4, 2, 8), torch.zeros(1, 4, 2, 8))
    except ValueError:
        return
    raise AssertionError('Non-divisible query/KV heads were accepted')


def test_resource_guard_rejects_insufficient_disk_before_fit():
    from collections import namedtuple
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    from src.models.resource_guard import inference_guard
    Usage = namedtuple('Usage', 'total used free')
    with TemporaryDirectory() as directory, patch('src.models.resource_guard.shutil.disk_usage', return_value=Usage(10**12, 10**12-1024, 1024)):
        try:
            with inference_guard(directory, {}):
                raise AssertionError('Fit was allowed with only 1 KiB free disk')
        except RuntimeError as error:
            assert 'disk reserve' in str(error)


def test_query_chunks_preserve_full_context_for_mha_and_mqa():
    from contextlib import nullcontext
    from unittest.mock import patch
    from src.models.windows_attention import WindowsMQABackend
    generator = torch.Generator().manual_seed(703)
    q = torch.randn(2, 19, 4, 8, generator=generator)
    for heads in (1, 4):
        k = torch.randn(2, 23, heads, 8, generator=generator)
        v = torch.randn(2, 23, heads, 8, generator=generator)
        with patch('src.models.windows_attention.sdpa_kernel', return_value=nullcontext()):
            actual = WindowsMQABackend(query_chunk_size=7).run(q, k, v)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3), enable_gqa=heads == 1).permute(0, 2, 1, 3)
        torch.testing.assert_close(actual, expected)


def test_opt_in_query_reuse_preserves_keys_values_and_refuses_aliases():
    from contextlib import nullcontext
    from unittest.mock import patch
    from src.models.windows_attention import WindowsMQABackend
    generator = torch.Generator().manual_seed(705)
    for alias in (False, True):
        q = torch.randn(2, 19, 4, 8, generator=generator)
        k = q if alias else torch.randn(2, 23, 4, 8, generator=generator)
        v = torch.randn_like(k)
        original_k, original_v = k.clone(), v.clone()
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        with torch.no_grad(), patch('src.models.windows_attention.sdpa_kernel', return_value=nullcontext()):
            actual = WindowsMQABackend(query_chunk_size=7, reuse_query_output=True).run(q, k, v)
        torch.testing.assert_close(actual, expected)
        assert torch.equal(k, original_k) and torch.equal(v, original_v)
        assert (actual.data_ptr() == q.data_ptr()) != alias
def test_decoder_projection_chunks_keep_every_context_row():
    from types import SimpleNamespace
    from src.models.pointwise_inference import install_decoder_projection_chunks
    from tabpfn.architectures.tabpfn_v3_5 import MultiTaskHeads
    torch.manual_seed(707)
    heads = MultiTaskHeads(input_size=8, max_num_classes=2, num_buckets=4,
        decoder_head_dim=4, decoder_num_heads=2, mlp_dim_feedforward=16,
        norm_factory=torch.nn.LayerNorm)
    heads.eval()
    # The shipped residual MLP initializes its final layer at zero. Nonzero
    # weights make this check exercise normalization, both projections and GELU.
    for parameter in heads.mlp_classification.parameters():
        torch.nn.init.uniform_(parameter, -.3, .3)
    x = torch.randn(2, 19, 8)
    with torch.no_grad():
        expected = heads.project_decoder_keys(x)
        install_decoder_projection_chunks(SimpleNamespace(heads=heads), rows=7)
        actual = heads.project_decoder_keys(x)
    assert actual.shape == expected.shape and actual.shape[1] == 19
    torch.testing.assert_close(actual, expected)
def test_decoder_activation_reuse_preserves_matrix_shapes_and_gradients():
    from copy import deepcopy
    from types import SimpleNamespace
    from src.models.pointwise_inference import install_decoder_gelu_reuse
    torch.manual_seed(709)
    mlp = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8))
    reference = deepcopy(mlp)
    model = SimpleNamespace(heads=SimpleNamespace(mlp_classification=SimpleNamespace(mlp=mlp)))
    install_decoder_gelu_reuse(model, rows=7)
    x = torch.randn(2, 19, 8)
    with torch.no_grad():
        actual, expected = mlp(x), reference(x)
    # CPU strided GELU kernels can differ by one float32 rounding unit.
    # The complete-model GPU probability gate remains independently stricter.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)
    with torch.enable_grad():
        a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
        mlp(a).sum().backward(); reference(b).sum().backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
def test_scaling_reuse_preserves_cpu_and_gradient_paths():
    import torch
    from tabpfn.architectures.tabpfn_v3_5 import SoftmaxScalingMLP
    from src.models.scaling_reuse import install_scaling_reuse
    torch.manual_seed(51)
    module=SoftmaxScalingMLP(4,16)
    with torch.no_grad():
        module.query_mlp[-1].weight.normal_(0,.03)
    original=module.forward
    q=torch.randn(2,17,4,16,requires_grad=True)
    expected=original(q,699635)
    expected_grad=torch.autograd.grad(expected.sum(),q)[0]
    install_scaling_reuse(torch.nn.Sequential(module),min_rows=1)
    actual=module(q,699635)
    actual_grad=torch.autograd.grad(actual.sum(),q)[0]
    assert torch.equal(actual,expected) and torch.equal(actual_grad,expected_grad)


def test_head_views_preserve_all_keys_and_alias_guard():
    from contextlib import nullcontext
    from unittest.mock import patch
    from src.models.head_view_attention import HeadViewBackend
    generator=torch.Generator().manual_seed(73)
    for reuse in (False,True):
        q=torch.randn(2,19,4,8,generator=generator)
        k=torch.randn(2,23,1,8,generator=generator)
        v=torch.randn(2,23,1,8,generator=generator)
        old_k,old_v=k.clone(),v.clone()
        expected=torch.nn.functional.scaled_dot_product_attention(
            q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3),
            enable_gqa=True).permute(0,2,1,3)
        with torch.no_grad(),patch('src.models.head_view_attention.sdpa_kernel',return_value=nullcontext()):
            actual=HeadViewBackend(query_chunk_size=7,reuse_query_output=reuse).run(q,k,v)
        torch.testing.assert_close(actual,expected)
        assert torch.equal(k,old_k) and torch.equal(v,old_v)
        assert (actual.data_ptr()==q.data_ptr())==reuse
    # A query alias into the single-head key storage must never be overwritten.
    k=torch.randn(2,19,1,8,generator=generator)
    q=k.expand(-1,-1,4,-1)
    v=torch.randn_like(k); old=k.clone()
    with torch.no_grad(),patch('src.models.head_view_attention.sdpa_kernel',return_value=nullcontext()):
        actual=HeadViewBackend(query_chunk_size=7,reuse_query_output=True).run(q,k,v)
    assert actual.untyped_storage().data_ptr()!=k.untyped_storage().data_ptr()
    assert torch.equal(k,old)

