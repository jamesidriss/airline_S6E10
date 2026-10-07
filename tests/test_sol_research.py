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
