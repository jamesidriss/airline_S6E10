"""Caruana-style greedy ensemble selection with replacement over the whole OOF pool.

The logistic stack is a parametric combiner; greedy selection with replacement is the
non-parametric one the public blends actually use, and it cannot assign negative weights. If
the 0.9622 frontier is a geometry artefact rather than an information limit, this is the
combiner that would show it.

Fold safety: the greedy path is chosen on the four training folds (on a 200k-row subsample of
them, for speed) and applied unchanged to the held-out fold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import TRAIN_CSV  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

POOL = Path(r"C:\Users\James\AppData\Local\Temp\opencode\s6e10_oof_pool\_pool\pool.npz")
OUT = Path(r"C:\Users\James\AppData\Local\Temp\opencode\s6e10_oof_pool\_pool")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--candidates", type=int, default=150)
    ap.add_argument("--subsample", type=int, default=200_000)
    ap.add_argument("--own", action="store_true", default=True)
    ap.add_argument("--tag", default="greedy")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    train = pd.read_csv(TRAIN_CSV, usecols=["id", "satisfaction"])
    y = train["satisfaction"].to_numpy(dtype="int8")
    folds = get_scheme(args.folds, y, train["id"]).folds
    n = len(y)

    P = np.load(POOL)
    names = sorted(k[5:] for k in P.files if k.startswith("oof__"))
    cols = [logit(np.clip(P["oof__" + nm].astype("float64"), 1e-6, 1 - 1e-6)) for nm in names]
    if args.own:
        for row in store.list_all():
            if not (row.get("auc") and row.get("has_test")):
                continue
            try:
                v = np.asarray(store.load_oof(row["exp_id"]), dtype="float64")
            except Exception:
                continue
            if v.shape == (n,) and np.isfinite(v).all():
                names.append("own__" + row["exp_id"])
                cols.append(logit(np.clip(v, 1e-6, 1 - 1e-6)))
    X = np.column_stack(cols).astype("float32")
    print(f"{len(names)} candidate vectors")

    rng = np.random.default_rng(11)
    oof = np.zeros(n)
    paths = []
    for k in sorted(set(folds.tolist())):
        tr = np.where(folds != k)[0]
        te = np.where(folds == k)[0]
        trs = rng.choice(tr, min(args.subsample, len(tr)), replace=False)
        auc_tr = np.array([roc_auc_score(y[trs], X[trs, c]) for c in range(X.shape[1])])
        cand = list(np.argsort(-auc_tr)[: args.candidates])
        counts = np.zeros(len(cand))
        running = np.zeros(len(trs))
        running_te = np.zeros(len(te))
        for r in range(args.rounds):
            best, best_auc = None, -1.0
            for j, c in enumerate(cand):
                trial = (running * r + X[trs, c]) / (r + 1)
                a = roc_auc_score(y[trs], trial)
                if a > best_auc:
                    best_auc, best = a, j
            counts[best] += 1
            running = (running * r + X[trs, cand[best]]) / (r + 1)
            running_te = (running_te * r + X[te, cand[best]]) / (r + 1)
        oof[te] = running_te
        w = counts / counts.sum()
        paths.append([(names[cand[j]], round(float(w[j]), 3)) for j in np.argsort(-w)[:6]])
        print(f"  fold{k}: {paths[-1]}", flush=True)

    def fa(v):
        return [float(roc_auc_score(y[folds == j], v[folds == j])) for j in range(int(folds.max()) + 1)]

    v7 = store.load_oof("v7_sol_tabpfn_internal2_aux10")
    xv = np.load(ROOT / "artifacts/phase24_candidateX/candidate_x_oof.npy")
    fg, fx, fv = fa(oof), fa(xv), fa(v7)
    # Each fold's greedy path has its own weights, so the fold predictions are not on a common
    # scale: pooling them across folds mixes different offsets and understates the combiner.
    # The honest headline is the mean of the per-fold AUCs, and the paired deltas use the same
    # per-fold convention.
    d = np.asarray(fx) - np.asarray(fg)
    out = {"tag": args.tag, "scheme": args.folds, "rounds": args.rounds,
           "n_candidates": len(names), "headline": "mean_of_fold_aucs",
           "mean_fold_auc": float(np.mean(fg)), "fold_aucs": fg,
           "pooled_auc_not_comparable": float(roc_auc_score(y, oof)),
           "v7_mean_fold_auc": float(np.mean(fv)), "v7_pooled": float(roc_auc_score(y, v7)),
           "x_mean_fold_auc": float(np.mean(fx)), "x_pooled": float(roc_auc_score(y, xv)),
           "delta_vs_v7_mean_fold": float(np.mean(fg) - np.mean(fv)),
           "delta_vs_x_mean_fold": float(np.mean(fg) - np.mean(fx)),
           "paired_vs_x": {"per_fold": d.tolist(), "mean": float(d.mean()),
                           "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))),
                           "pos": int((d > 0).sum())},
           "paths": paths}
    print(json.dumps({k: out[k] for k in ("mean_fold_auc", "x_mean_fold_auc", "v7_mean_fold_auc",
                                          "delta_vs_v7_mean_fold", "delta_vs_x_mean_fold",
                                          "paired_vs_x")}, indent=1))
    if args.save:
        np.save(OUT / f"oof_{args.tag}.npy", oof)
        (ROOT / "reports" / "phase25").mkdir(parents=True, exist_ok=True)
        (ROOT / "reports" / "phase25" / f"greedy_{args.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
