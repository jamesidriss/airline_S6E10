"""Phase 20 -- Kumo Tabular (NVIDIA, OpenMDW-1.1) fold-0 foundation screen.

Screens Kumo as a candidate 11th ensemble member under the SAME phase-19 protocol: it sees the
champion's raw21+coarse10 route representation (built by frames()), a fold-0-safe stratified
context drawn only from FIT (folds 1-4), and is scored on fold-0 rows against the hash-asserted
champion portfolio/route. Kumo is a pure in-context learner (forward(context, query)); no
fine-tuning, no target encoding. Runs in the ISOLATED .venv_ag (autogluon + sdm), never the
certified stack.

This is a SCREEN, not a promotion: it cannot pass the OOF admission gate. It decides whether Kumo
merits a full 5-fold OOF + admission-gate run. Anti-leakage: context is a subset of FIT; the query
rows' target column is dropped before the model sees it; champion reference is hash-asserted.

Run (in .venv_ag, GPU):
  .\\.venv_ag\\Scripts\\python.exe scripts/phase20/kumo_screen_fold0.py --context 4096 --eval-rows 20000 --size medium
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, REPORTS, arr_sha256, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.phase19.foundation_screen import CONTEXT_SEED, screen_metrics, stratified_context  # noqa: E402
from scripts.phase19.foundation_screen_fold0 import load_champion_fold0  # noqa: E402

OUT = REPORTS / "phase20" / "kumo_screen"
ART = ARTIFACTS / "phase20" / "kumo_screen"

# Kumo checkpoint pinned via structured-data-models (sdm 0.2.1); weights are auto-resolved by sdm.
# Recorded for reproducibility. OpenMDW-1.1 = ELIGIBLE (outputs unrestricted).
KUMO_REPO = "sdm.models.KumoTabular (structured-data-models==0.2.1)"


def log(msg: str) -> None:
    print(f"[kumo] {msg}", flush=True)


def _to_category_codes(col_float: np.ndarray) -> "pd.Categorical":
    """Convert a float32 code column (from frames()) to a pandas Categorical (label-free)."""
    import pandas as pd
    codes = np.rint(col_float).astype(np.int64)
    return pd.Categorical(codes)


def build_table(Xtr: np.ndarray, y: np.ndarray, names, cats, device: str):
    """Build a sdm TableTensor over ALL train rows (features + target) with categorical stypes.

    The categorical columns (indices from frames()) are marked as Categorical so Kumo processes
    them natively (its intended input path) instead of as ordinal numbers. This uses the SAME
    underlying codes the other phase-19 models received; the stype metadata is label-free schema.
    """
    import pandas as pd
    import sdm
    import torch
    df = pd.DataFrame(Xtr, columns=[str(n) for n in names])
    cat_set = set(int(c) for c in cats)
    for j, nm in enumerate(df.columns):
        if j in cat_set:
            df[nm] = _to_category_codes(Xtr[:, j])
    df["target"] = pd.Series(pd.Categorical(y.astype(np.int64)))
    stypes = sdm.infer_stypes(df, overrides={"target": "categorical"})
    tt = sdm.TableTensor.from_pandas(df=df, stypes=stypes, device=torch.device(device))
    return tt, stypes


def kumo_predict(tt, ctx: np.ndarray, eval_idx: np.ndarray, size: str, device: str,
                 num_estimators: int, dtype: str, seed: int) -> np.ndarray:
    """In-context predict: forward(context, query) -> positive-class probabilities for eval rows."""
    import sdm
    import torch
    torch.manual_seed(seed)
    dev = torch.device(device)
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    model = sdm.models.KumoTabular(task="classification", size=size, device=dev).eval()
    x_ctx = tt[ctx].drop_columns("target")
    y_ctx = tt[ctx, "target"]
    x_q = tt[eval_idx].drop_columns("target")
    autocast_on = dev.type == "cuda" and dt != torch.float32
    with torch.amp.autocast(dev.type, dtype=dt, enabled=autocast_on):
        probs_tt = model(x_context=x_ctx, y_context=y_ctx, x_query=x_q, num_estimators=num_estimators)
    pdf = probs_tt.to_pandas()  # columns ['1','0'], row sums == 1
    pos_col = "1" if "1" in pdf.columns else pdf.columns[-1]
    p1 = pdf[pos_col].to_numpy(dtype=np.float64)
    assert p1.shape == (len(eval_idx),) and np.isfinite(p1).all(), "bad Kumo probabilities"
    assert ((p1 >= 0) & (p1 <= 1)).all(), "Kumo probs out of [0,1]"
    del model
    gc.collect()
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    return p1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--eval-rows", type=int, default=20000)
    ap.add_argument("--size", choices=["small", "medium", "large"], default="medium")
    ap.add_argument("--num-estimators", type=int, default=8)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="kumo_fold0_ctx4k_e20k_med")
    ap.add_argument("--seed", type=int, default=1201)
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    from scripts.run_sol_tabpfn import frames  # champion feature builder (bit-exact)

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds

    Xtr, _Xte, names, cats, maps = frames(tr, te, route=True)
    assert Xtr.shape[0] == len(y), "feature/row misalignment"

    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)
    assert not np.intersect1d(FIT, EVAL).size

    route_all, port_all = load_champion_fold0(folds)
    eval_idx = EVAL if (not args.eval_rows or args.eval_rows >= len(EVAL)) else EVAL[:args.eval_rows]
    route0, port0 = route_all[eval_idx], port_all[eval_idx]
    y_eval = y[eval_idx]

    ctx = stratified_context(FIT, y, args.context, seed=CONTEXT_SEED)
    log(f"rows={len(y)} FIT={len(FIT)} EVAL={len(eval_idx)} context={len(ctx)} "
        f"features={Xtr.shape[1]} device={args.device} size={args.size} dtype={args.dtype} "
        f"num_est={args.num_estimators}")

    tt, stypes = build_table(Xtr, y, names, cats, args.device)
    n_cat = sum(1 for v in stypes.values() if "categorical" in str(v).lower())
    log(f"table built; categorical cols marked: {n_cat}")

    pred_path = ART / f"kumo_{args.tag}.npy"
    meta_path = OUT / f"kumo_{args.tag}.json"
    if pred_path.exists() and meta_path.exists():
        saved = np.load(pred_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") == "DONE" and arr_sha256(saved) == meta.get("prediction_sha256") \
                and meta.get("eval_rows") == int(len(eval_idx)):
            log("verified prediction exists; reusing")
            m = meta["metrics"]
            log(f"AUC={m['standalone_auc']:.6f} port_delta={m['portfolio_delta']:+.3e}")
            return 0

    log(f"fit context {len(ctx)} -> predict {len(eval_idx)} (in-context, no fine-tune)")
    t1 = time.time()
    try:
        pred = kumo_predict(tt, ctx, eval_idx, args.size, args.device, args.num_estimators,
                            args.dtype, args.seed)
        np.save(pred_path, pred.astype("float32"))
        m = screen_metrics(y_eval, pred, port0, route0)
        status = "DONE"
        err = None
    except Exception as exc:  # resource/impl limits are recorded, not fatal
        pred = None
        m = None
        status = "FAILED"
        err = repr(exc)
    secs = time.time() - t1

    meta = {
        "model": "kumo", "repo": KUMO_REPO, "status": status, "error": err,
        "utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "context_rows": int(len(ctx)), "context_seed": CONTEXT_SEED,
        "eval_rows": int(len(eval_idx)), "eval_fold": 0, "fit_folds": [1, 2, 3, 4],
        "device": args.device, "size": args.size, "dtype": args.dtype,
        "num_estimators": args.num_estimators, "seed": args.seed, "tag": args.tag,
        "features": list(names), "categorical_indices": list(cats),
        "stypes_categorical": [k for k, v in stypes.items() if "categorical" in str(v).lower()],
        "protocol": "in-context only (forward(context,query)); query target dropped; champion hash-asserted",
        "prediction_sha256": arr_sha256(np.load(pred_path)) if pred is not None else None,
        "metrics": m, "fit_predict_seconds": secs,
        "prereg_note": "phase-20 Kumo screen; reuses phase-19 foundation-screen protocol + thresholds",
    }
    save_json(meta, meta_path)
    if m is not None:
        log(f"AUC={m['standalone_auc']:.6f} port_delta={m['portfolio_delta']:+.3e} "
            f"corr_port={m['logit_pearson_vs_portfolio']:.4f} ({secs:.1f}s)")
    else:
        log(f"FAILED: {err}")
    return 0 if (status == "DONE" and m is not None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
