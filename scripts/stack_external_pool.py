"""Fold-safe stack over the public OOF pool.

For each outer fold the member ranking and the meta coefficients are learned on the OTHER
folds only, then applied to the held-out fold. The pooled AUC is therefore out-of-sample
with respect to the stack as well as the base models. Member ranking inside the training
rows keeps the top-k selection honest.
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
from src.validation.folds import get_scheme  # noqa: E402

POOL = Path(r"C:\Users\James\AppData\Local\Temp\opencode\s6e10_oof_pool\_pool\pool.npz")
OUT = Path(r"C:\Users\James\AppData\Local\Temp\opencode\s6e10_oof_pool\_pool")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--topk", type=int, default=120)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--kind", default="logit", choices=["logit", "ridge"])
    ap.add_argument("--tag", default="stack120")
    ap.add_argument("--own-top", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    train = pd.read_csv(TRAIN_CSV, usecols=["id", "satisfaction"])
    y = train["satisfaction"].to_numpy(dtype="int8")
    folds = get_scheme(args.folds, y, train["id"]).folds
    P = np.load(POOL)
    names = sorted(k[5:] for k in P.files if k.startswith("oof__"))
    cols = [logit(np.clip(P["oof__" + n].astype("float64"), 1e-6, 1 - 1e-6)) for n in names]
    if args.own_top:
        from src.submission import store as store_mod
        rows = [r for r in store_mod.list_all() if r.get("auc")]
        rows.sort(key=lambda r: -float(r["auc"]))
        for r in rows[: args.own_top]:
            try:
                v = np.asarray(store_mod.load_oof(r["exp_id"]), dtype="float64")
            except Exception:
                continue
            if v.shape == (len(y),) and np.isfinite(v).all():
                names.append("own__" + r["exp_id"])
                cols.append(logit(np.clip(v, 1e-6, 1 - 1e-6)))
    X = np.column_stack(cols).astype("float32")
    mus, sds = X.mean(0), X.std(0) + 1e-9
    X -= mus
    X /= sds

    def member_auc(col: np.ndarray, rows: np.ndarray) -> float:
        return roc_auc_score(y[rows], X[rows, col])

    ref = names.index("najiama_05_blend_satisfaction") if "najiama_05_blend_satisfaction" in names else 0
    oof = np.zeros(len(y))
    picks = []
    for k in sorted(set(folds.tolist())):
        tr = np.where(folds != k)[0]
        te = np.where(folds == k)[0]
        order = sorted(range(len(names)), key=lambda c: -member_auc(c, tr))
        sel = order[: args.topk]
        picks.append([names[c] for c in sel[:8]])
        Z = X[:, sel]
        if args.kind == "logit":
            m = LogisticRegression(C=args.C, max_iter=300, solver="lbfgs")
            m.fit(Z[tr], y[tr])
            oof[te] = m.decision_function(Z[te])
        else:
            m = Ridge(alpha=1.0 / max(args.C, 1e-9))
            m.fit(Z[tr], y[tr])
            oof[te] = m.predict(Z[te])

    def fa(v):
        return [float(roc_auc_score(y[folds == k], v[folds == k])) for k in range(int(folds.max()) + 1)]

    pooled = float(roc_auc_score(y, oof))
    ref_f = fa(X[:, ref])
    base_f = [float(roc_auc_score(y[folds == k], X[folds == k, ref])) for k in range(int(folds.max()) + 1)]
    out = {"tag": args.tag, "scheme": args.folds, "topk": args.topk, "C": args.C,
           "kind": args.kind, "n_members": len(names),
           "pooled_auc": pooled, "fold_aucs": fa(oof),
           "ref_member": names[ref], "ref_pooled": float(roc_auc_score(y, X[:, ref])),
           "ref_fold_aucs": base_f,
           "delta_vs_ref": pooled - float(roc_auc_score(y, X[:, ref])),
           "top8_per_fold": picks}
    print(json.dumps({k: out[k] for k in ("tag", "pooled_auc", "delta_vs_ref", "fold_aucs",
                                          "ref_fold_aucs")}, indent=1))
    if args.save:
        np.save(OUT / f"oof_{args.tag}.npy", oof)
        (OUT / f"meta_{args.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
