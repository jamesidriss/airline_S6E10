"""Nested stack over the whole own-model store, which the Phase-24 census never did.

Phase 24 screened each stored vector *pairwise* against v7; it never fitted a joint model over
all of them. The public pool reaches 0.9622 from ~100 merely-decent members, so the same
question has to be asked of our own 100+ vectors: does a fitted joint combination beat the
best single champion?

Fold safety: member ranking and coefficients are learned on the four training folds of each
outer split and applied to the held-out fold, so the pooled AUC is out-of-sample for the
stack as well as for the members.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import REPORTS, TRAIN_CSV  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

OUT = Path(r"C:\Users\James\AppData\Local\Temp\opencode\s6e10_oof_pool\_pool")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--kind", default="logit", choices=["logit", "ridge"])
    ap.add_argument("--tag", default="store60")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    train = pd.read_csv(TRAIN_CSV, usecols=["id", "satisfaction"])
    y = train["satisfaction"].to_numpy(dtype="int8")
    folds = get_scheme(args.folds, y, train["id"]).folds
    n = len(y)

    rows = [r for r in store.list_all() if r.get("auc")]
    names, cols, aucs = [], [], []
    for r in rows:
        if not r.get("has_test"):
            continue
        try:
            v = np.asarray(store.load_oof(r["exp_id"]), dtype="float64")
        except Exception:
            continue
        if v.shape != (n,) or not np.isfinite(v).all():
            continue
        names.append(r["exp_id"])
        cols.append(logit(np.clip(v, 1e-6, 1 - 1e-6)))
        aucs.append(float(r["auc"]))
    X = np.column_stack(cols).astype("float32")
    print(f"{len(names)} stored vectors with test predictions")
    X -= X.mean(0)
    X /= X.std(0) + 1e-9

    def member_auc(c: int, rows_: np.ndarray) -> float:
        return roc_auc_score(y[rows_], X[rows_, c])

    oof = np.zeros(n)
    picks = []
    for k in sorted(set(folds.tolist())):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        order = sorted(range(len(names)), key=lambda c: -member_auc(c, tr))[: args.topk]
        picks.append([names[c] for c in order[:5]])
        Z = X[:, order]
        if args.kind == "logit":
            m = LogisticRegression(C=args.C, max_iter=300).fit(Z[tr], y[tr])
            oof[te] = m.decision_function(Z[te])
        else:
            m = Ridge(alpha=1.0 / max(args.C, 1e-9)).fit(Z[tr], y[tr])
            oof[te] = m.predict(Z[te])

    def fa(v):
        return [float(roc_auc_score(y[folds == k], v[folds == k])) for k in range(int(folds.max()) + 1)]

    v7 = store.load_oof("v7_sol_tabpfn_internal2_aux10")
    xvec = np.load(ROOT / "artifacts/phase24_candidateX/candidate_x_oof.npy")
    out = {"tag": args.tag, "scheme": args.folds, "topk": args.topk, "C": args.C, "kind": args.kind,
           "n_members": len(names), "pooled_auc": float(roc_auc_score(y, oof)), "fold_aucs": fa(oof),
           "v7_pooled": float(roc_auc_score(y, v7)), "x_pooled": float(roc_auc_score(y, xvec)),
           "delta_vs_v7": float(roc_auc_score(y, oof)) - float(roc_auc_score(y, v7)),
           "delta_vs_x": float(roc_auc_score(y, oof)) - float(roc_auc_score(y, xvec)),
           "top5_per_fold": picks}
    print(json.dumps({k: out[k] for k in ("tag", "n_members", "pooled_auc", "v7_pooled",
                                          "delta_vs_v7", "delta_vs_x")}, indent=1))
    if args.save:
        np.save(OUT / f"oof_{args.tag}.npy", oof)
        (ROOT / "reports" / "phase25").mkdir(parents=True, exist_ok=True)
        (ROOT / "reports" / "phase25" / f"store_stack_{args.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
