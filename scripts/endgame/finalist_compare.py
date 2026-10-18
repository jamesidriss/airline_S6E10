"""Endgame private-first finalist comparator (CPU only; saved OOF arrays).

Computes, for a candidate OOF vector against the true v7 champion baseline:
  - pooled and per-fold AUCs, paired deltas, mean/SE/t, positive-fold count,
    and the AGENTS.md admission gate arithmetic (mean >= 1.5e-5, mean >= 2.5*SE,
    >= 4/5 positive) recomputed from arrays (never copied from a report)
  - stratified within-fold row bootstrap CI of the pooled paired delta
  - generator-batch-aware block bootstrap (contiguous id ranges, matching the
    blocked_id_scheme used for fold-safety tests)
  - logit Pearson + Spearman correlation and top-1% rank-set overlap
  - per-segment AUC deltas (reuses phase-18 segment definitions)
  - train-vs-test covariate drift (standardized mean differences for numeric
    columns, total-variation distance for categorical columns)

With no candidate, it emits the baseline self-report (v7 bootstrap CI, segment
AUCs, drift) used by the finalist-selection record.

Usage:
  ./.venv/Scripts/python.exe scripts/endgame/finalist_compare.py --candidate <store_id>
  ./.venv/Scripts/python.exe scripts/endgame/finalist_compare.py   # baseline only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import arr_sha256, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.submission.store import load_oof  # noqa: E402
from scripts.phase18.build_error_atlas import segment_masks  # noqa: E402

BASELINE = "v7_sol_tabpfn_internal2_aux10"
FOLDS = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"
OUT = ROOT / "reports" / "endgame" / "candidate_comparison.json"
GATE_GAIN = 1.5e-5
GATE_T = 2.5
GATE_POSITIVE = 4


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def stratified_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, folds: np.ndarray,
                         n_boot: int = 2000, seed: int = 7) -> dict:
    """Resample rows within each fold (keeps fold composition), recompute pooled AUC delta."""
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    idx_by_fold = [np.where(folds == k)[0] for k in range(5)]
    for b_i in range(n_boot):
        picks = [rng.choice(ix, size=len(ix), replace=True) for ix in idx_by_fold]
        idx = np.concatenate(picks)
        deltas[b_i] = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
    return {
        "n_boot": n_boot, "seed": seed,
        "mean": float(deltas.mean()), "se": float(deltas.std(ddof=1)),
        "ci_2.5": float(np.quantile(deltas, 0.025)), "ci_50": float(np.quantile(deltas, 0.5)),
        "ci_97.5": float(np.quantile(deltas, 0.975)),
        "p_delta_gt_0": float((deltas > 0).mean()),
    }


def block_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, ids: np.ndarray,
                    n_blocks: int = 50, n_boot: int = 2000, seed: int = 11) -> dict:
    """Generator-batch-aware: resample contiguous id-range blocks (blocked_id_scheme analogue)."""
    order = np.argsort(ids, kind="stable")
    blocks = np.array_split(order, n_blocks)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[c] for c in chosen])
        yy, aa, bb = y[idx], a[idx], b[idx]
        if yy.min() == yy.max():
            deltas[i] = 0.0
            continue
        deltas[i] = roc_auc_score(yy, aa) - roc_auc_score(yy, bb)
    return {
        "n_blocks": n_blocks, "n_boot": n_boot, "seed": seed,
        "mean": float(deltas.mean()), "se": float(deltas.std(ddof=1)),
        "ci_2.5": float(np.quantile(deltas, 0.025)), "ci_97.5": float(np.quantile(deltas, 0.975)),
        "p_delta_gt_0": float((deltas > 0).mean()),
    }


def drift(train, test) -> dict:
    num_cols = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    cat_cols = ["Gender", "Customer Type", "Type of Travel", "Class"] + \
               ["Inflight wifi service", "Seat comfort", "Cleanliness", "Food and drink",
                "Inflight entertainment", "On-board service", "Baggage handling", "Leg room service",
                "Online boarding", "Checkin service", "Gate location", "Ease of Online booking",
                "Departure/Arrival time convenient"]
    out = {"numeric_smd": {}, "categorical_tv": {}}
    for c in num_cols:
        a = pd_numeric(train, c)
        b = pd_numeric(test, c)
        fa, fb = np.isfinite(a), np.isfinite(b)
        if fa.sum() < 10 or fb.sum() < 10:
            out["numeric_smd"][c] = 0.0
            continue
        va, vb = float(a[fa].var(ddof=1)), float(b[fb].var(ddof=1))
        pooled = np.sqrt(0.5 * (va + vb))
        out["numeric_smd"][c] = float((b[fb].mean() - a[fa].mean()) / pooled) if pooled > 0 else 0.0
    for c in cat_cols:
        pa = train[c].astype("category").cat.codes
        pb = test[c].astype("category").cat.codes
        cats = set(pa.unique()) | set(pb.unique())
        ta = np.array([(pa == k).mean() for k in cats])
        tb = np.array([(pb == k).mean() for k in cats])
        out["categorical_tv"][c] = float(0.5 * np.abs(ta - tb).sum())
    out["max_abs_smd"] = float(max(abs(v) for v in out["numeric_smd"].values()))
    out["max_tv"] = float(max(out["categorical_tv"].values()))
    return out


def pd_numeric(df, col):
    s = pd_series(df, col)
    return s.astype("float64").to_numpy()


def pd_series(df, col):
    return df[col]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=None, help="store exp_id of the candidate OOF")
    ap.add_argument("--baseline", default=BASELINE)
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = np.load(FOLDS)

    base = load_oof(args.baseline)
    base_auc = float(roc_auc_score(y, base))
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git": git_commit(),
        "baseline": {"exp_id": args.baseline, "oof_auc": base_auc, "oof_sha256": arr_sha256(base)},
        "candidate": None,
        "train_test_drift": drift(tr, te),
        "reference_noise_scale": {
            "paired_public_lb_noise": 2e-4,
            "subset_variance_h91": "6.0e-7..6.3e-7 per fold (data composition), seed variance 1.8e-9..7.2e-9",
        },
    }

    # baseline self-report (bootstrap CI of the champion's own OOF AUC via fold-blocked resampling)
    rng = np.random.default_rng(3)
    aucs = []
    idx_by_fold = [np.where(folds == k)[0] for k in range(5)]
    for _ in range(1000):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by_fold])
        aucs.append(roc_auc_score(y[idx], base[idx]))
    report["baseline"]["bootstrap_oof_auc"] = {
        "n_boot": 1000, "mean": float(np.mean(aucs)), "se": float(np.std(aucs, ddof=1)),
        "ci_2.5": float(np.quantile(aucs, 0.025)), "ci_97.5": float(np.quantile(aucs, 0.975)),
    }
    report["baseline"]["per_fold_auc"] = [float(roc_auc_score(y[folds == k], base[folds == k]))
                                          for k in range(5)]

    if args.candidate:
        cand = load_oof(args.candidate)
        cand_auc = float(roc_auc_score(y, cand))
        per_fold = []
        deltas = []
        for k in range(5):
            m = folds == k
            a_k = float(roc_auc_score(y[m], cand[m]))
            b_k = float(roc_auc_score(y[m], base[m]))
            per_fold.append({"fold": k, "candidate_auc": a_k, "baseline_auc": b_k, "delta": a_k - b_k})
            deltas.append(a_k - b_k)
        d = np.array(deltas)
        mean_delta = float(d.mean())
        se_delta = float(d.std(ddof=1) / np.sqrt(5))
        t_stat = mean_delta / se_delta if se_delta > 0 else 0.0
        pos = int((d > 0).sum())
        gate_pass = bool(mean_delta >= GATE_GAIN and mean_delta >= GATE_T * se_delta and pos >= GATE_POSITIVE)
        lp = logit
        seg = segment_masks(tr)
        seg_rows = {}
        for name, mask in seg.items():
            idx = np.where(mask.to_numpy())[0]
            if len(idx) < 2000 or len(np.unique(y[idx])) < 2:
                continue
            a_s = float(roc_auc_score(y[idx], cand[idx]))
            b_s = float(roc_auc_score(y[idx], base[idx]))
            seg_rows[name] = {"rows": int(len(idx)), "candidate_auc": a_s, "baseline_auc": b_s,
                              "delta": a_s - b_s}
        k = max(1, int(0.01 * len(y)))
        top_c = set(np.argsort(-cand)[:k].tolist())
        top_b = set(np.argsort(-base)[:k].tolist())
        report["candidate"] = {
            "exp_id": args.candidate, "oof_auc": cand_auc, "oof_sha256": arr_sha256(cand),
            "pooled_delta": cand_auc - base_auc,
            "per_fold": per_fold,
            "paired": {"mean": mean_delta, "se": se_delta, "t": t_stat,
                       "positive_folds": pos, "folds_positive": f"{pos}/5"},
            "admission_gate_recomputed": {
                "required_mean_gain": GATE_GAIN, "required_t": GATE_T, "required_positive": GATE_POSITIVE,
                "mean_ok": mean_delta >= GATE_GAIN, "t_ok": mean_delta >= GATE_T * se_delta,
                "positives_ok": pos >= GATE_POSITIVE, "passes": gate_pass,
            },
            "correlation": {
                "logit_pearson": float(np.corrcoef(lp(cand), lp(base))[0, 1]),
                "spearman": float(spearmanr(cand, base).statistic),
                "top1pct_rank_set_overlap": len(top_c & top_b) / k,
            },
            "bootstrap_stratified": stratified_bootstrap(y, cand, base, folds),
            "bootstrap_block50": block_bootstrap(y, cand, base, ids),
            "segments": seg_rows,
            "segment_worst": sorted(seg_rows.items(), key=lambda kv: kv[1]["delta"])[:5],
        }

    save_json(report, OUT)
    print(f"baseline {args.baseline} OOF {base_auc:.12f} sha {report['baseline']['oof_sha256'][:16]}")
    bb = report["baseline"]["bootstrap_oof_auc"]
    print(f"  baseline bootstrap AUC {bb['mean']:.6f} +/- {bb['se']:.6f} "
          f"[{bb['ci_2.5']:.6f}, {bb['ci_97.5']:.6f}]")
    print(f"  per-fold {[round(v, 6) for v in report['baseline']['per_fold_auc']]}")
    print(f"  drift: max|smd|={report['train_test_drift']['max_abs_smd']:.4f} "
          f"max tv={report['train_test_drift']['max_tv']:.4f}")
    if report["candidate"]:
        c = report["candidate"]
        print(f"candidate {c['exp_id']} OOF {c['oof_auc']:.12f} pooled delta {c['pooled_delta']:+.3e}")
        print(f"  paired mean {c['paired']['mean']:+.3e} se {c['paired']['se']:.3e} "
              f"t {c['paired']['t']:.2f} pos {c['paired']['folds_positive']} "
              f"gate={c['admission_gate_recomputed']['passes']}")
        print(f"  corr logit {c['correlation']['logit_pearson']:.5f} spearman {c['correlation']['spearman']:.5f}")
        print(f"  bootstrap strat CI [{c['bootstrap_stratified']['ci_2.5']:+.3e}, "
              f"{c['bootstrap_stratified']['ci_97.5']:+.3e}] P(>0)={c['bootstrap_stratified']['p_delta_gt_0']:.3f}")
        print(f"  bootstrap block50 CI [{c['bootstrap_block50']['ci_2.5']:+.3e}, "
              f"{c['bootstrap_block50']['ci_97.5']:+.3e}] P(>0)={c['bootstrap_block50']['p_delta_gt_0']:.3f}")
    print(f"-> {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())