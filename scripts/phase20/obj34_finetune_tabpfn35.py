"""Phase 20 -- OBJ34 fine-tune TabPFN v3.5 fold-0 screen (F2 discovery).

The only official weight-update path for the champion architecture
(tabpfn 9.1.0 FinetunedTabPFNClassifier; audit closed row 33: no LoRA/adapter
surface exists in any screened FLM). Fine-tunes the champion's OWN v3.5
checkpoint (read-only, sha-asserted) on FIT via the Prior-Labs finetuning
module, then predicts fold-0 and scores against the hash-asserted champion
portfolio/route and the frozen-rule route_ref.

Anti-leakage: fit on FIT only; the API's inner validation split is carved from
FIT only (library contract -- same role as our inner-10% ES split); context
subsample drawn from FIT only; no target encoding (route view is label-free);
champion checkpoint read-only; output_dir=None (no intermediate checkpoints
written anywhere).

Frozen rule: PROMISING iff port_delta >= +5e-5 AND
Spearman(candidate, route_ref) < 0.995.

Run:
  .\\.venv\\Scripts\\python.exe scripts/phase20/obj34_finetune_tabpfn35.py
"""
from __future__ import annotations

import argparse
import hashlib
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
from src.phase19.foundation_screen import _cl, logit, screen_metrics  # noqa: E402
from scripts.phase19.foundation_screen_fold0 import load_champion_fold0  # noqa: E402

OUT = REPORTS / "phase20" / "obj34_finetune_tabpfn35"
ART = ARTIFACTS / "phase20" / "obj34_finetune_tabpfn35"
PREREG = "configs/phase20/obj34_finetune_tabpfn35_prereg_v2.json"
# v2 = crash-amendment of the v1 prereg (00542b47..., commit 77c6ad0); run v1 died
# before any predictive result (OOM in the library's inner-val eval, fp32 attention
# materialization). See "amendment_reason"/"delta_from_v1" inside the prereg file.
PREREG_SHA = "e4cd5276c95aae66c9ff1c0de9da6304e575da263e1e0c135cd1ed83c8c09f1a"  # pinned before the rerun (preregister-before-result)
ROUTE_REF = "sol_route_aux10_primary_final"
CKPT = ARTIFACTS / "tabpfn35" / "tabpfn-v3.5-20260909.safetensors"
CKPT_SHA256 = "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"
BATCH = 8192

# frozen estimator config (prereg)
FT_PARAMS = dict(
    device="cuda", random_state=1201,
    time_limit=2400,
    epochs=30, early_stopping=True, early_stopping_patience=8,
    validation_split_ratio=0.1, validation_frequency=1, min_delta=0.0001,
    learning_rate=1e-05, weight_decay=0.01, grad_clip_value=1.0,
    use_lr_scheduler=True,
    n_finetune_ctx_plus_query_samples=50000,
    n_estimators_finetune=2,
    n_estimators_validation=2,
    n_estimators_final_inference=2,
    use_activation_checkpointing=True,
)


def log(msg: str) -> None:
    print(f"[obj34] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()

    sha = hashlib.sha256((ROOT / PREREG).read_bytes()).hexdigest()
    assert sha == PREREG_SHA, f"prereg sha mismatch {sha}"
    log(f"prereg sha256 {sha}")

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    # ---- champion checkpoint integrity (READ-ONLY) ----
    assert CKPT.exists(), f"missing champion checkpoint {CKPT}"
    got = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    assert got == CKPT_SHA256, f"champion checkpoint hash drift: {got}"
    log(f"champion v3.5 ckpt sha256 OK ({got[:16]}...) read-only")

    import torch
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNClassifier

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds

    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)
    assert not np.intersect1d(FIT, EVAL).size

    route_all, port_all = load_champion_fold0(folds)

    from scripts.run_sol_tabpfn import frames

    Xall, _Xte, names, cats, maps = frames(tr, te, route=True)
    assert Xall.shape[0] == len(y), "feature/row misalignment"

    from src.submission.store import load_oof

    ref = load_oof(ROUTE_REF).astype(np.float64)
    assert np.isfinite(ref).all() and ref.min() >= 0.0 and ref.max() <= 1.0
    ref0 = ref[EVAL]
    route0, port0 = route_all[EVAL], port_all[EVAL]
    y_eval = y[EVAL]

    log(f"FIT={len(FIT)} EVAL={len(EVAL)} features={Xall.shape[1]} "
        f"cats={[int(c) for c in cats]}")

    torch.cuda.reset_peak_memory_stats()
    clf = FinetunedTabPFNClassifier(
        model_version=ModelVersion.V3_5,
        extra_classifier_kwargs={"model_path": str(CKPT.resolve()),
                                 "categorical_features_indices": cats,
                                 # champion fold-0 inference params (prereg v2):
                                 # 'autocast' keeps the inner-val eval forward in
                                 # bf16 (no fp32 attention materialization); the
                                 # other two pin the frozen recipe.
                                 "inference_precision": "autocast",
                                 "kv_cache_precision": "int8",
                                 "auto_scale_n_estimators": False},
        **FT_PARAMS,
    )
    t1 = time.time()
    clf.fit(Xall[FIT], y[FIT])  # inner validation split carved from FIT only
    fit_s = time.time() - t1
    log(f"fine-tune done in {fit_s:.1f}s")

    t2 = time.time()
    chunks = []
    for begin in range(0, len(EVAL), BATCH):
        stop = min(begin + BATCH, len(EVAL))
        chunks.append(clf.predict_proba(Xall[EVAL[begin:stop]])[:, 1])
        if (stop // BATCH) % 10 == 0:
            log(f"predicted {stop}/{len(EVAL)}")
    pred = np.concatenate(chunks).astype(np.float64)
    predict_s = time.time() - t2
    assert pred.shape == (len(EVAL),)
    assert np.isfinite(pred).all() and (pred >= 0).all() and (pred <= 1).all()
    log(f"predict done in {predict_s:.1f}s")

    m = screen_metrics(y_eval, pred, port0, route0)

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    auc_ref = float(roc_auc_score(y_eval, ref0))
    blend = 1.0 / (1.0 + np.exp(-(logit(_cl(pred)) + logit(_cl(ref0))) / 2.0))
    auc_blend_ref = float(roc_auc_score(y_eval, blend))
    port_delta = auc_blend_ref - auc_ref
    spearman = float(spearmanr(pred, ref0).statistic)
    prom = bool(port_delta >= 5e-5 and spearman < 0.995)

    pred_path = ART / "finetuned_v35_fold0.npy"
    np.save(pred_path, pred.astype("float32"))

    meta = {
        "model": "tabpfn-v3.5-finetuned", "status": "DONE",
        "utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "prereg": PREREG, "prereg_sha256": sha,
        "checkpoint": str(CKPT.relative_to(ROOT).as_posix()),
        "checkpoint_sha256": got, "checkpoint_use": "read-only, output_dir=None",
        "recipe": "official FinetunedTabPFNClassifier full FT on FIT (inner "
                  "validation split from FIT only, early stopping); "
                  "n_estimators_final_inference=2; route view",
        "ft_params": FT_PARAMS,
        "fit_rows": int(len(FIT)), "eval_fold": 0, "eval_rows": int(len(EVAL)),
        "fit_folds": [1, 2, 3, 4],
        "features": list(names), "categorical_indices": [int(c) for c in cats],
        "label_free_category_maps": maps,
        "route_ref": ROUTE_REF, "route_ref_auc_fold0": auc_ref,
        "prediction_sha256": arr_sha256(np.load(pred_path)),
        "metrics": m,
        "frozen_rule": {
            "port_delta": port_delta, "route_ref_auc": auc_ref,
            "blend_auc": auc_blend_ref, "spearman_vs_route_ref": spearman,
            "promising": prom,
        },
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "fit_seconds": fit_s, "predict_seconds": predict_s,
        "total_seconds": time.time() - t0,
    }
    save_json(meta, OUT / "finetuned_v35_fold0.json")

    fr = meta["frozen_rule"]
    log(f"standalone={m['standalone_auc']:.6f} (ref {auc_ref:.6f}) "
        f"blend={fr['blend_auc']:.6f} port_delta={port_delta:+.3e} "
        f"spearman={spearman:.4f} -> PROMISING={prom}")
    log(f"screen table: port_delta={m['portfolio_delta']:+.3e} "
        f"pearson_port={m['logit_pearson_vs_portfolio']:.4f}")
    log(f"peak vram {meta['peak_vram_gb']}GB; total {meta['total_seconds']:.1f}s")
    log(f"report -> {(OUT / 'finetuned_v35_fold0.json').relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
