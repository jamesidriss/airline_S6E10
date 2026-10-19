"""Local CPU dry-run of the raw8 kernel's port/patch equivalence logic.

Runs the exact `create_model` / `prepare` / one-member engine / lean-patch code
paths imported from research/phase23/kaggle/raw8/main.py on CPU (CUDA hidden),
on a small slice of the real raw21 geometry, and re-checks the three-way
equivalence the Kaggle probe will certify on T4:

    stock predict_proba  vs  ported engine  vs  ported+lean-patched engine

This is a code-equivalence test only. It computes no AUC and produces no
predictive result; it cannot and does not touch the frozen prereg.

Environment notes for this off-GPU run:
  * CUDA_VISIBLE_DEVICES="" keeps the local RTX 5070 Ti (hardware stop) untouched.
  * tabpfn 9.1.0's attention picker calls torch.cuda.get_device_capability when
    torch.cuda.is_available() is True; with devices hidden that query raises
    "Invalid device id".  The dry-run therefore pins `gqa_is_supported` to False,
    which is exactly what the stock check returns on Kaggle's T4 (compute
    capability 7.5 < 8): the T4-equivalent non-GQA attention branch.

    .\.venv\Scripts\python.exe scripts\phase23\dryrun_port_equivalence.py
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hardware stop: never engage the local GPU
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "4")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import file_sha256  # noqa: E402

CKPT = ROOT / "artifacts/tabpfn35/tabpfn-v3.5-20260909.safetensors"
CKPT_SHA = "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"
OUT = ROOT / "reports/phase23/port_equivalence_local_cpu.json"
N_CTX = 4096
N_QRY = 256


def load_kernel_module():
    path = ROOT / "research/phase23/kaggle/raw8/main.py"
    spec = importlib.util.spec_from_file_location("raw8_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_slice():
    import pandas as pd
    from src.common import load_cached_parquet

    mod = load_kernel_module()
    tr, te = load_cached_parquet()
    cov = pd.concat([tr[mod.RAW21], te[mod.RAW21]], ignore_index=True)
    for c in mod.META4:
        levels = sorted(cov[c].unique().tolist())
        cov[c] = pd.Categorical(cov[c], categories=levels).codes
    cov = cov.fillna(-1)
    X = cov.iloc[: len(tr)].to_numpy(dtype="float32")
    y = tr[mod.TARGET].astype(str).str.lower().map({"true": 1, "false": 0}).to_numpy(np.int8)
    return mod, X, y


def pin_t4_attention_branch():
    """T4-equivalent: compute capability 7.5 < 8 so stock gqa_is_supported() is False."""
    from tabpfn.architectures.shared import scaled_dot_product_attention as sdpa_mod

    sdpa_mod.gqa_is_supported = lambda: False


def safe_release(clf):
    import gc

    clf.executor_ = None
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def main():
    started = time.monotonic()
    assert file_sha256(CKPT) == CKPT_SHA, "local checkpoint drift"
    pin_t4_attention_branch()
    mod, X, y = build_slice()
    Xc, yc = X[:N_CTX], y[:N_CTX]
    Xq = X[N_CTX : N_CTX + N_QRY]

    report = {
        "purpose": "local CPU dry-run of the raw8 kernel port/patch equivalence logic",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gqa_is_supported_pinned_false": True,
        "gqa_note": "T4 compute capability 7.5 < 8 -> the stock check returns False on Kaggle too",
        "n_context": N_CTX,
        "n_query": N_QRY,
        "ckpt_sha256": CKPT_SHA,
        "n_estimators": 1,
    }

    base_params = mod.make_params(CKPT, clf_device="cpu")
    base_params["n_estimators"] = 1

    def variants():
        yield "exact", {}
        yield "no_int8_cache", {"kv_cache_precision": "auto"}
        yield "relaxed", {"kv_cache_precision": "auto", "keep_cache_on_device": False,
                          "memory_saving_mode": "auto"}

    chosen = None
    for tag, override in variants():
        try:
            params = dict(base_params, **override)
            clf = mod.create_model(params, icl_bf16=False,
                                   chunk_cells=mod.CHUNK_CELLS, col_chunk=mod.COL_CHUNK)
            clf.fit(Xc, yc)
            p = clf.predict_proba(Xq)[:, 1]
            chosen = (tag, override, p)
            safe_release(clf)
            break
        except Exception as e:  # noqa: BLE001
            print(f"variant {tag} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
    assert chosen is not None, "no stock variant ran on CPU"
    tag, override, p_stock = chosen
    report["variant"] = tag
    report["variant_override"] = override
    print("stock variant:", tag, flush=True)

    params = dict(base_params, **override)

    mod.restore_stock_forward()
    clf_a = mod.create_model(params, icl_bf16=False, chunk_cells=mod.CHUNK_CELLS,
                             col_chunk=mod.COL_CHUNK)
    prep_a = mod.prepare(clf_a, Xc, yc)
    mod.build_member_engine(clf_a, prep_a, 0)
    lg_a = mod.predict_raw(clf_a, Xq, 256, label="dry-a")
    p_a = mod.probabilities(clf_a, lg_a)[:, 1]
    safe_release(clf_a)

    mod.apply_lean_patch()
    clf_b = mod.create_model(params, icl_bf16=False, chunk_cells=mod.CHUNK_CELLS,
                             col_chunk=mod.COL_CHUNK)
    prep_b = mod.prepare(clf_b, Xc, yc)
    mod.build_member_engine(clf_b, prep_b, 0)
    lg_b = mod.predict_raw(clf_b, Xq, 256, label="dry-b")
    p_b = mod.probabilities(clf_b, lg_b)[:, 1]
    safe_release(clf_b)

    d_stock_port = float(np.max(np.abs(p_stock - p_a)))
    d_port_lean = float(np.max(np.abs(p_a - p_b)))
    report.update(
        stock_vs_port=d_stock_port,
        port_vs_lean=d_port_lean,
        gate=2e-6,
        pass_=bool(d_stock_port <= 2e-6 and d_port_lean <= 2e-6),
        members=len(prep_b[2]),
        member0_manifest=prep_b[3]["members"][0],
        cpu_wall_seconds=round(time.monotonic() - started, 1),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("stock_vs_port", "port_vs_lean", "pass_",
                                             "cpu_wall_seconds")}, indent=1))
    return 0 if report["pass_"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
