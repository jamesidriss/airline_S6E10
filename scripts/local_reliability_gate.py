"""Is region-local reliability real? An honest test of the community's "local-reliability blend".

A public notebook ("S6E10 | Local-Reliability Residual Blend", public LB 0.96152) post-processes a
shared external OOF library: it measures each model's *local* ROC-AUC inside equal-frequency
score regions and gates the correction by that reliability. Its reported number is not adoptable
-- the correction is fitted and then evaluated on the same OOF vector, and the inputs are another
team's predictions rather than our own -- so we ignore the notebook and test the underlying
question ourselves, under the discipline we have applied everywhere else.

Question: does a member being more reliable in some *region of the score range* carry exploitable
signal, once the gating is cross-fitted so no region weight is ever fitted on the rows it scores?

Implementation (nested, honest):
  * the score range is split into N equal-frequency bins of the *blend's own* OOF rank,
  * every member's local AUC inside each bin is measured on the **fold-train rows only**,
  * bin weights are turned into a multiplicative correction,
  * the held-out fold's rows are scored with weights derived only from the other folds,
  * so the reported AUC is out-of-sample with respect to the gating itself.

If the honest number does not beat equal weighting, region gating is a fitting artefact and we
reject it with evidence rather than by assertion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.ensemble import lab  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import strat_bootstrap_auc_delta  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402


def rank01(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(np.asarray(x, dtype="float64")) / len(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-auc", type=float, default=0.9605)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=8,
                    help="number of members the local-reliability search is allowed to use")
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=150)
    args = ap.parse_args()

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y, tr[ID_COL]).folds
    idx = store._load_index()

    P, meta = {}, {}
    for k, v in idx.items():
        m = v.get("meta", {})
        if not m.get("auc") or m.get("family") in ("blend", "stack"):
            continue
        if m["auc"] < args.min_auc or "test" not in v:
            continue
        o = store.load_oof(k)
        if len(o) == len(y):
            P[k] = o.astype("float64")
            meta[k] = m
    names = list(P)
    # work in rank space: the gate is about ranking quality, and rank space makes members
    # commensurable without any calibration assumption
    R = np.column_stack([rank01(P[n]) for n in names])
    anchor = R.mean(axis=1)
    bins = np.minimum((rank01(anchor) * args.bins).astype(int), args.bins - 1)

    print(f"pool={len(names)} members, bins={args.bins}, top_k={args.top_k}")

    def gated_oof() -> np.ndarray:
        """Cross-fitted region gating: weights for fold k come only from rows with folds != k."""
        out = np.zeros(len(y))
        for k in sorted(set(folds.tolist())):
            trn = np.where(folds != k)[0]
            tst = np.where(folds == k)[0]
            w = np.ones(len(names))
            base_local = np.zeros(args.bins)
            for b in range(args.bins):
                m = trn[bins[trn] == b]
                if m.sum() < 200 or y[m].min() == y[m].max():
                    continue
                base_local[b] = roc_auc_score(y[m], anchor[m])
            best_gain, best_w = 0.0, w.copy()
            for j in range(len(names)):
                gain = np.zeros(args.bins)
                for b in range(args.bins):
                    m = trn[bins[trn] == b]
                    if m.sum() < 200 or y[m].min() == y[m].max():
                        continue
                    a = roc_auc_score(y[m], R[m, j])
                    gain[b] = a - base_local[b]
                if gain.mean() <= 0:
                    continue
                cand = w.copy()
                for b in range(args.bins):
                    if base_local[b] <= 0:
                        continue
                    cand[j] *= 1.0 + args.power * gain[b]
                cand = cand / cand.sum()
                if gain.mean() > best_gain:
                    best_gain, best_w = gain.mean(), cand
            out[tst] = R[tst] @ best_w
        return out

    gated = gated_oof()
    equal = anchor
    a_eq = float(roc_auc_score(y, equal))
    a_gt = float(roc_auc_score(y, gated))
    print(f"  equal-weight rank blend : {a_eq:.6f}")
    print(f"  cross-fitted region gate: {a_gt:.6f}   delta = {a_gt - a_eq:+.6f}")
    bm, lo, hi = strat_bootstrap_auc_delta(y, gated, equal, n_boot=args.n_boot)
    print(f"  bootstrap delta {bm:+.6f}  95% CI [{lo:+.6f}, {hi:+.6f}]  "
          f"{'region gating is REAL' if lo > 0 else 'inside noise -> fitting artefact'}")

    # per-fold, so we can see whether any fold carries the whole effect
    from src.ensemble.lab import fold_aucs

    fe, fg = fold_aucs(y, equal, folds), fold_aucs(y, gated, folds)
    print("  per-fold delta:", [round(a - b, 6) for a, b in zip(fg, fe)])
    print("  folds improved:", sum(1 for a, b in zip(fg, fe) if a > b), "/", len(fe))

    out = {"pool": len(names), "bins": args.bins, "power": args.power,
           "equal_auc": a_eq, "gated_auc": a_gt, "delta": a_gt - a_eq,
           "boot": [bm, lo, hi], "per_fold_delta": [a - b for a, b in zip(fg, fe)]}
    for nb in (5, 20, 40):
        args.bins = nb
        g = gated_oof()
        a = float(roc_auc_score(y, g))
        print(f"  sensitivity bins={nb:<3} gated={a:.6f}  delta_vs_equal={a - a_eq:+.6f}")
        out[f"bins{nb}_auc"] = a
    save_json(out, REPORTS / "local_reliability_gate.json")
    print("\nwrote", REPORTS / "local_reliability_gate.json")


if __name__ == "__main__":
    main()