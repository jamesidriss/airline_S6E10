"""Phase 20 -- supervised fine-tuning of TabICL-v2, fold-0 screen (items 31+13).

The ONLY reopening mechanism named by the FLM-rich and neural closures: the
zero-shot TabICL had the best decorrelation of any candidate (corr 0.9828 vs
champion route arm) but standalone lagged at 0.9563. Fine-tuning the pretrained
checkpoint on FIT (supervised meta-batches, package-native
FinetunedTabICLClassifier) tests whether strength can be added without losing
decorrelation.

Fold-safe: X_train/X_val are both subsets of FIT (inner stratified 10% split,
seed 20261010-derived); the eval fold is never seen by training or epoch
selection. The route view is label-free. Champion reference hash-asserted.

Run (certified .venv, GPU):
  .\\.venv\\Scripts\\python.exe scripts/phase20/tabicl_ft_fold0.py
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, REPORTS, arr_sha256, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.phase19.foundation_screen import screen_metrics  # noqa: E402
from scripts.phase19.foundation_screen_fold0 import load_champion_fold0  # noqa: E402
from scripts.phase20.neural_screen_fold0 import inner_es_split  # noqa: E402

OUT = REPORTS / "phase20" / "tabicl_ft"
ART = ARTIFACTS / "phase20" / "tabicl_ft"
PREREG = ROOT / "configs" / "phase20" / "tabicl_ft_prereg.json"


def log(msg: str) -> None:
    print(f"[tabicl-ft] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="tabicl_ft_route_fold0")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    frozen = prereg["protocol"]["model"]["frozen_args"]

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    # tabicl._finetune imports four LR-scheduler functions from `transformers`,
    # which is absent from the certified .venv. Installing it in place would
    # alter the certified stack (forbidden); the auditable shim provides the
    # exact LambdaLR semantics the fine-tuner's cosine_warmup scheduler needs.
    shim_dir = str(ROOT / "scripts" / "phase20" / "_shims")
    if shim_dir not in sys.path:
        sys.path.insert(0, shim_dir)
    import transformers as _tf_shim
    assert all(hasattr(_tf_shim, n) for n in (
        "get_constant_schedule", "get_linear_schedule_with_warmup",
        "get_cosine_schedule_with_warmup", "get_polynomial_decay_schedule_with_warmup"))

    from tabicl import FinetunedTabICLClassifier

    from scripts.run_sol_tabpfn import frames

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds

    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)
    assert not np.intersect1d(FIT, EVAL).size

    route_all, port_all = load_champion_fold0(folds)
    route0, port0 = route_all[EVAL], port_all[EVAL]
    y_eval = y[EVAL]

    Xall, _Xte, names, cats, _maps = frames(tr, te, route=True)
    assert Xall.shape[0] == len(y) and Xall.shape[1] == 22

    # inner 10% stratified of FIT as the fine-tuner's validation (fold-safe)
    fit_pos = np.arange(len(FIT))
    train_pos, val_pos = inner_es_split(fit_pos, y[FIT], frac=0.1, seed=20261010)
    Xtr = Xall[FIT][train_pos].astype(np.float32)
    ytr = y[FIT][train_pos].astype(np.int64)
    Xva = Xall[FIT][val_pos].astype(np.float32)
    yva = y[FIT][val_pos].astype(np.int64)
    Xeval = Xall[EVAL].astype(np.float32)

    log(f"rows={len(y)} FIT={len(FIT)} train={len(train_pos)} val={len(val_pos)} "
        f"EVAL={len(EVAL)} features={Xall.shape[1]} frozen={frozen}")

    pred_path = ART / f"{args.tag}.npy"
    meta_path = OUT / f"{args.tag}.json"
    if pred_path.exists() and meta_path.exists():
        saved = np.load(pred_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") == "DONE" and arr_sha256(saved) == meta.get("prediction_sha256") \
                and meta.get("eval_rows") == int(len(EVAL)):
            m = meta["metrics"]
            log(f"verified prediction exists; reusing AUC={m['standalone_auc']:.6f} "
                f"port_delta={m['portfolio_delta']:+.3e}")
            return 0

    clf = FinetunedTabICLClassifier(
        epochs=int(frozen["epochs"]), learning_rate=float(frozen["learning_rate"]),
        weight_decay=float(frozen["weight_decay"]),
        n_estimators_finetune=int(frozen["n_estimators_finetune"]),
        n_estimators_inference=int(frozen["n_estimators_inference"]),
        max_data_size=int(frozen["max_data_size"]),
        device=frozen["device"], verbose=True, random_state=1201)

    log("fine-tuning on FIT-train with FIT-val early stopping...")
    t1 = time.time()
    try:
        clf.fit(Xtr, ytr, X_val=Xva, y_val=yva)
        ft_secs = time.time() - t1
        log(f"fine-tune done in {ft_secs:.1f}s; predicting {len(EVAL)} fold-0 rows")
        chunks = []
        bs = 8192
        for b in range(0, len(EVAL), bs):
            pr = clf.predict_proba(Xeval[b:b + bs])
            if getattr(pr, "ndim", 0) == 2:
                pr = pr[:, 1]
            chunks.append(np.asarray(pr, dtype=np.float64))
        pred = np.concatenate(chunks)
        assert pred.shape == (len(EVAL),) and np.isfinite(pred).all()
        np.save(pred_path, pred.astype("float32"))
        m = screen_metrics(y_eval, pred, port0, route0)
        status = "DONE"
        err = None
    except Exception as exc:
        m = None
        ft_secs = time.time() - t1
        status = "FAILED"
        err = repr(exc)
    secs = time.time() - t1

    meta = {
        "model": "tabicl_finetuned", "status": status, "error": err,
        "frozen_args": frozen, "train_rows": int(len(train_pos)),
        "val_rows": int(len(val_pos)), "inner_split_seed": 20261010,
        "utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "eval_rows": int(len(EVAL)), "eval_fold": 0, "fit_folds": [1, 2, 3, 4],
        "view": "route (champion raw22, label-free)",
        "device": args.device, "seed": args.seed, "tag": args.tag,
        "features": list(names), "categorical_indices": list(cats),
        "protocol": "supervised fine-tune on FIT-train, FIT-val early stop, predict fold-0; "
                    "champion hash-asserted",
        "prediction_sha256": arr_sha256(np.load(pred_path)) if status == "DONE" else None,
        "metrics": m, "fit_predict_seconds": secs, "finetune_seconds": ft_secs,
        "prereg": {"path": PREREG.relative_to(ROOT).as_posix()},
    }
    save_json(meta, meta_path)
    if m is not None:
        log(f"AUC={m['standalone_auc']:.6f} port_delta={m['portfolio_delta']:+.3e} "
            f"corr_port={m['logit_pearson_vs_portfolio']:.4f} "
            f"corr_route={m['logit_pearson_vs_route']:.4f} ({secs:.1f}s)")
    else:
        log(f"FAILED: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
