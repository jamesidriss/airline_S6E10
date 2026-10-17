"""Phase 20 -- RECOVERY of H20-TABICL-FT fb100k after the GPU driver crash.

Preregistered in configs/phase20/tabicl_ft_recovery_prereg.json (sha256
a9383ab9...) BEFORE any fold-0 prediction for this arm existed.

Established cause (Windows device logs, not inference): nvlddmkm Event ID 153
at 2026-10-17T15:07:04 and T17:07:32 killed both fine-tune processes at OS
level; LiveKernelEvent 141/144/19e at 14:42:45 during load. No Python
exception, no system OOM event, RAM free at inspection. This CORRECTS the
earlier "allocator fragmentation" hypothesis (withdrawn, append-only ledger
correction recorded separately).

Recovered state: artifacts/phase20/tabicl_ft/ckpt/best.ckpt, epoch=8/10,
best_metric=0.9579433793073343 (pretrained baseline was 0.9578), all tensors
finite, optimizer/scheduler state present.

Action (frozen): evaluate the epoch-8 best checkpoint -- a literal prefix of
the frozen trajectory, selected by the library's own best-metric rule -- via
the library's documented reload path ``TabICLClassifier(model_path=...)``,
fit the final inference estimator (n_estimators=8) on the SAME 100000-row
fine-tune training set, predict fold-0 in 8192-row chunks, compute the frozen
screen metrics, apply the unchanged decision rule.

DEVIATION (disclosed in the result JSON): training completed 8 of 10 epochs.
No foreign weight update occurred. If this arm were PROMISING, the completed
10-epoch arm would still be owed before any OOF admission.

Run (certified .venv, GPU):
  .\\.venv\\Scripts\\python.exe scripts/phase20/tabicl_ft_recover.py
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

# Allocator-only robustness (cannot alter computed values; must precede torch).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, REPORTS, arr_sha256, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.phase19.foundation_screen import screen_metrics  # noqa: E402
from scripts.phase19.foundation_screen_fold0 import load_champion_fold0  # noqa: E402
from scripts.phase20.neural_screen_fold0 import inner_es_split  # noqa: E402

OUT = REPORTS / "phase20" / "tabicl_ft"
ART = ARTIFACTS / "phase20" / "tabicl_ft"
CKPT = ART / "ckpt" / "best.ckpt"
RECOVERY_PREREG = ROOT / "configs" / "phase20" / "tabicl_ft_recovery_prereg.json"
PREREG = ROOT / "configs" / "phase20" / "tabicl_ft_prereg.json"

# Frozen reference: zero-shot TabICL route fold-0 standalone (phase 19).
ZERO_SHOT_STANDALONE = 0.956343
EXPECTED_EPOCH = 8
EXPECTED_BEST_METRIC = 0.9579433793073343
ZERO_SHOT_NEGATIVE_BAR = ZERO_SHOT_STANDALONE + 2e-3  # 0.958343


def log(msg: str) -> None:
    print(f"[tabicl-recover] {msg}", flush=True)


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def apply_decision_rule(m: dict) -> dict:
    """Unchanged frozen rule from tabicl_ft_prereg.json / fallback prereg."""
    sa = m["standalone_auc"]
    pd_ = m["portfolio_delta"]
    corr_p = m["logit_pearson_vs_portfolio"]
    corr_r = m["logit_pearson_vs_route"]
    if sa < ZERO_SHOT_NEGATIVE_BAR:
        verdict = "NEGATIVE"
        reason = (f"standalone {sa:.6f} < zero-shot 0.956343 + 2e-3; fine-tuning added "
                  f"no material strength (applies to the reduced-dose arm)")
    elif pd_ >= 5e-5 and corr_p < 0.995:
        verdict = "PROMISING"
        reason = (f"port_delta {pd_:+.3e} >= +5e-5 AND corr_vs_portfolio {corr_p:.4f} < 0.995; "
                  f"full 5-fold OOF owed (full-dose arm still owed under extended plan)")
    elif sa >= ZERO_SHOT_NEGATIVE_BAR and corr_r >= 0.99:
        verdict = "DECORRELATION_COLLAPSE"
        reason = (f"standalone improved but corr_vs_route {corr_r:.4f} >= 0.99; "
                  f"fine-tuning destroyed the decorrelation that motivated it")
    else:
        verdict = "NEGATIVE"
        reason = (f"standalone {sa:.6f} >= bar but port_delta {pd_:+.3e} < +5e-5 or "
                  f"corr_vs_portfolio {corr_p:.4f} >= 0.995; no admissible portfolio value")
    return {"verdict": verdict, "reason": reason,
            "rules": {"promising": "port_delta >= +5e-5 AND corr_vs_portfolio < 0.995",
                      "negative": f"standalone < {ZERO_SHOT_NEGATIVE_BAR:.6f} (zero-shot + 2e-3) "
                                  f"or no admissible portfolio value",
                      "decorrelation_collapse": "standalone improves but corr_vs_route >= 0.99"}}


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    assert CKPT.exists(), f"recovery checkpoint missing: {CKPT}"
    ckpt_sha = file_sha256(CKPT)
    log(f"checkpoint {CKPT.name} sha256={ckpt_sha[:16]}...")

    import torch
    from tabicl import TabICLClassifier
    from tabicl._sklearn.preprocessing import TransformToNumerical

    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    assert int(ck["epoch"]) == EXPECTED_EPOCH, f"checkpoint epoch {ck['epoch']} != {EXPECTED_EPOCH}"
    assert abs(float(ck["best_metric"]) - EXPECTED_BEST_METRIC) < 1e-9, \
        f"checkpoint best_metric {ck['best_metric']} != {EXPECTED_BEST_METRIC}"
    n_finite = sum(1 for v in ck["state_dict"].values()
                   if v.is_floating_point() and torch.isfinite(v).all().item())
    n_float = sum(1 for v in ck["state_dict"].values() if v.is_floating_point())
    assert n_finite == n_float, f"non-finite tensors in checkpoint: {n_float - n_finite}"
    del ck
    log(f"checkpoint verified: epoch={EXPECTED_EPOCH} best_metric={EXPECTED_BEST_METRIC} "
        f"all {n_float} float tensors finite")

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

    # Rebuild the EXACT fine-tune training set of the crashed run: inner 10%
    # stratified of FIT (seed 20261010) then stratified 100k subsample
    # (rng 20261011). The final estimator must fit on the same rows the
    # checkpoint was trained on.
    fit_pos = np.arange(len(FIT))
    train_pos, _val_pos = inner_es_split(fit_pos, y[FIT], frac=0.1, seed=20261010)
    n_train_full = int(len(train_pos))
    rng = np.random.default_rng(20261011)
    ytr_full = y[FIT][train_pos]
    keep_parts = []
    for cls in np.unique(ytr_full):
        cls_idx = np.flatnonzero(ytr_full == cls)
        n_cls = int(round(100000 * len(cls_idx) / n_train_full))
        keep_parts.append(rng.choice(cls_idx, size=min(n_cls, len(cls_idx)), replace=False))
    keep = np.sort(np.concatenate(keep_parts))
    train_pos = train_pos[keep]
    assert len(train_pos) == 100000, f"subsample rebuilt {len(train_pos)} != 100000"
    Xtr = Xall[FIT][train_pos].astype(np.float32)
    ytr = y[FIT][train_pos].astype(np.int64)
    Xeval = Xall[EVAL].astype(np.float32)
    log(f"rebuilt fine-tune train set: {len(train_pos)} rows (stratified, rng 20261011); "
        f"EVAL={len(EVAL)}")

    pred_path = ART / "tabicl_ft_route_fold0_fb100k.npy"
    meta_path = OUT / "tabicl_ft_route_fold0_fb100k.json"
    if pred_path.exists() and meta_path.exists():
        saved = np.load(pred_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") == "DONE" and arr_sha256(saved) == meta.get("prediction_sha256") \
                and meta.get("eval_rows") == int(len(EVAL)):
            m = meta["metrics"]
            log(f"verified prediction exists; reusing AUC={m['standalone_auc']:.6f} "
                f"port_delta={m['portfolio_delta']:+.3e}")
            return 0

    # The fine-tuner's authoritative outer encoder, rebuilt deterministically
    # on the same 100k rows (TransformToNumerical = SimpleImputer/ordinal, no
    # randomness; fit_transform == fit + transform for this pipeline).
    enc = TransformToNumerical()
    Xtr_enc = np.asarray(enc.fit_transform(Xtr), dtype=np.float64)

    # Final inference estimator = the library's documented reload path for
    # fine-tuned checkpoints (pretraining schema: config + state_dict). All
    # classifier kwargs mirror the fine-tuner's _create_inner_estimator.
    inner = TabICLClassifier(
        n_estimators=8, norm_methods=None, feat_shuffle_method="latin",
        class_shuffle_method="shift", outlier_threshold=4.0, softmax_temperature=0.9,
        average_logits=True, support_many_classes=True, batch_size=8, kv_cache=False,
        model_path=str(CKPT), allow_auto_download=False,
        checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
        device="cuda", random_state=1201, verbose=False)

    log("fitting final inference estimator (8 members) on the 100k fine-tune train set...")
    t1 = time.time()
    batch_size_fallback = False
    try:
        inner.fit(Xtr_enc, ytr)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        batch_size_fallback = True
        inner.batch_size = 4
        log("OOM during estimator fit; retrying with batch_size 8 -> 4 (pure batching)")
        inner.fit(Xtr_enc, ytr)
    fit_secs = time.time() - t1
    log(f"estimator fit in {fit_secs:.1f}s; predicting {len(EVAL)} fold-0 rows in 8192-row chunks")

    chunks = []
    bs = 8192
    t2 = time.time()
    try:
        for b in range(0, len(EVAL), bs):
            xe = np.asarray(enc.transform(Xeval[b:b + bs]), dtype=np.float64)
            try:
                pr = inner.predict_proba(xe)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_size_fallback = True
                inner.batch_size = 4
                log(f"OOM at chunk {b // bs}; continuing with batch_size=4")
                pr = inner.predict_proba(xe)
            if getattr(pr, "ndim", 0) == 2:
                pr = pr[:, 1]
            chunks.append(np.asarray(pr, dtype=np.float64))
        pred = np.concatenate(chunks)
        assert pred.shape == (len(EVAL),) and np.isfinite(pred).all()
        np.save(pred_path, pred.astype("float32"))
        m = screen_metrics(y_eval, pred, port0, route0)
        m["delta_vs_zeroshot"] = m["standalone_auc"] - ZERO_SHOT_STANDALONE
        decision = apply_decision_rule(m)
        status = "DONE"
        err = None
    except Exception as exc:
        m = None
        decision = None
        status = "FAILED"
        err = repr(exc)
    pred_secs = time.time() - t2
    total_secs = time.time() - t0

    prereg_sha = file_sha256(RECOVERY_PREREG)
    meta = {
        "model": "tabicl_finetuned_recovered", "status": status, "error": err,
        "arm": "fb100k", "recovered": True,
        "recovery_prereg": {"path": RECOVERY_PREREG.relative_to(ROOT).as_posix(),
                            "sha256": prereg_sha},
        "recovery_cause": "GPU driver crash (nvlddmkm Event 153 at 2026-10-17T15:07:04 and "
                          "T17:07:32; LiveKernelEvent 141/144/19e at 14:42:45) terminated the "
                          "fine-tune process at OS level at the start of epoch 9/10; earlier "
                          "allocator-fragmentation hypothesis WITHDRAWN (not established)",
        "checkpoint": {"path": CKPT.relative_to(ROOT).as_posix(), "sha256": ckpt_sha,
                       "epoch": EXPECTED_EPOCH, "best_metric": EXPECTED_BEST_METRIC,
                       "pretrained_baseline_val": 0.9578,
                       "val_trajectory": [0.9578, 0.9578, 0.9578, 0.9579, 0.9579,
                                          0.9579, 0.9579, EXPECTED_BEST_METRIC]},
        "deviation": "training completed 8 of 10 epochs (GPU driver crash); evaluated weights "
                     "are the frozen trajectory's own best-metric checkpoint -- no foreign "
                     "weight update occurred; PROMISING verdict still owes the completed "
                     "10-epoch arm before any OOF admission",
        "resource_adaptation": "inner batch_size 8 -> 4 (pure batching, identical math)"
                               if batch_size_fallback else None,
        "train_rows_full_dose": n_train_full, "train_rows": int(len(train_pos)),
        "inner_split_seed": 20261010, "subsample_seed": 20261011,
        "allocator_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "frozen_args_source": PREREG.relative_to(ROOT).as_posix(),
        "utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "eval_rows": int(len(EVAL)), "eval_fold": 0, "fit_folds": [1, 2, 3, 4],
        "view": "route (champion raw22, label-free)",
        "device": "cuda", "tag": "tabicl_ft_route_fold0_fb100k",
        "features": list(names), "categorical_indices": list(cats),
        "protocol": "reload epoch-8 fine-tuned checkpoint via TabICLClassifier(model_path=...), "
                    "fit final estimator on the same 100k fine-tune train rows, predict fold-0; "
                    "champion hash-asserted",
        "prediction_sha256": arr_sha256(np.load(pred_path)) if status == "DONE" else None,
        "metrics": m, "decision": decision,
        "fit_seconds": fit_secs, "predict_seconds": pred_secs, "total_seconds": total_secs,
        "prereg": {"path": PREREG.relative_to(ROOT).as_posix()},
    }
    save_json(meta, meta_path)
    if m is not None:
        log(f"AUC={m['standalone_auc']:.6f} (delta vs zero-shot {m['delta_vs_zeroshot']:+.3e}) "
            f"port_delta={m['portfolio_delta']:+.3e} corr_port={m['logit_pearson_vs_portfolio']:.4f} "
            f"corr_route={m['logit_pearson_vs_route']:.4f}")
        log(f"DECISION: {decision['verdict']} -- {decision['reason']}")
        log(f"times: fit {fit_secs:.1f}s predict {pred_secs:.1f}s total {total_secs:.1f}s")
        return 0
    log(f"FAILED: {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
