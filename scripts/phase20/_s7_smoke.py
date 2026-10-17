"""Smoke test for the S7 repair additions (run manually, not in the suite).

Verifies on CPU (fast, no GPU needed) that:
  1. DCNv2Full builds and forward()s with full cross matrices (n_feat x n_feat).
  2. FactLog builds and forward()s with the bilinear rank-16 interaction term.
  3. neural_screen_fold0.train_and_predict honors infer_chunk (bounded forward).
  4. classical build_model('rffsvm') returns a fitted pipeline with predict_proba.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def test_dcnv2full_full_cross():
    from src.phase20.neural_arch import DCNv2Full
    n_feat = 285
    m = DCNv2Full(n_feat, d_hidden=256, n_layers=3)
    # cross_w layers must be (n_feat, n_feat) — full matrix, not rank-1.
    for w in m.cross_w:
        assert w.weight.shape == (n_feat, n_feat), f"bad cross shape {w.weight.shape}"
        assert w.in_features == n_feat and w.out_features == n_feat
    x = torch.randn(8, n_feat)
    out = m(x)
    assert out.shape == (8,), f"bad output shape {out.shape}"
    assert torch.isfinite(out).all()
    # params: 3 * (285*285) cross weights + 3 * 285 cross biases + deep + out.
    n_params = sum(p.numel() for p in m.parameters())
    print(f"DCNv2Full OK: n_params={n_params:,}")


def test_factlog_bilinear():
    from src.phase20.neural_arch import FactLog
    n_feat, rank = 285, 16
    m = FactLog(n_feat, rank=rank)
    assert m.U.shape == (rank, n_feat) and m.V.shape == (rank, n_feat)
    x = torch.randn(8, n_feat)
    out = m(x)
    assert out.shape == (8,)
    assert torch.isfinite(out).all()
    # Bilinear term must actually vary with input (not constant).
    out2 = m(torch.randn(8, n_feat))
    assert not torch.allclose(out, out2)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"FactLog OK: n_params={n_params:,}")


def test_infer_chunk_bounded():
    """train_and_predict with infer_chunk must not crash and must match full-batch
    logits for a deterministic model (no dropout, eval-mode identity)."""
    from scripts.phase20.neural_screen_fold0 import train_and_predict
    torch.manual_seed(0)
    n, d = 512, 8
    Xtr = np.random.randn(n, d).astype(np.float32)
    ytr = (np.random.rand(n) > 0.5).astype(np.float32)
    Xval = np.random.randn(100, d).astype(np.float32)
    yval = (np.random.rand(100) > 0.5).astype(np.float32)
    Xeval = np.random.randn(200, d).astype(np.float32)

    class _Const(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(d, 1)

        def forward(self, x):
            return self.lin(x).squeeze(-1)

    cfg = {"lr": 0.001, "batch_size": 64, "max_epochs": 2, "es_patience": 2}
    logits_a, bv_a = train_and_predict(_Const(), Xtr, ytr, Xval, yval, Xeval, cfg,
                                       "cpu", 0, infer_chunk=65536)
    torch.manual_seed(0)
    logits_b, bv_b = train_and_predict(_Const(), Xtr, ytr, Xval, yval, Xeval, cfg,
                                       "cpu", 0, infer_chunk=37)
    assert logits_a.shape == (200,) and logits_b.shape == (200,)
    assert np.allclose(logits_a, logits_b, atol=1e-5), "infer_chunk changed eval logits!"
    print(f"infer_chunk OK: chunked==full (max|diff|={np.abs(logits_a - logits_b).max():.2e})")


def test_rffsvm_classical():
    from scripts.phase20.classical_screen_fold0 import build_model
    m = build_model("rffsvm", 20)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 20)).astype(np.float32)
    y = (rng.random(2000) > 0.5).astype(np.int8)
    m.fit(X, y)
    p = m.predict_proba(X[:50])
    assert p.shape == (50,), f"bad proba shape {p.shape}"
    assert (p > 0).all() and (p < 1).all()
    print(f"rffsvm OK: proba range [{p.min():.4f}, {p.max():.4f}]")


if __name__ == "__main__":
    test_dcnv2full_full_cross()
    test_factlog_bilinear()
    test_infer_chunk_bounded()
    test_rffsvm_classical()
    print("ALL S7 SMOKE TESTS PASSED")
