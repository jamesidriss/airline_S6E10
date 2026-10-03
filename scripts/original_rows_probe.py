"""The last untested external-data usage: the original rows as *training data*.

Everything we measured so far uses the original survey as *knowledge* (smoothed target statistics,
conditional surfaces, a teacher prediction). Nobody has measured appending its 129,880 labelled
rows to the training set on our own folds. The field reports it as neutral-to-harmful, but they
never swept the sample weight, and the S6E5 winner reported that down-weighting rather than
dropping the original rows is what mattered.

Why it is worth one clean test here: our diagnosis is that the synthetic labels are i.i.d.
Bernoulli(p_syn(x)) with p_syn much weaker than the real survey's p_orig(x) (original-data
grouped-fold AUC 0.9949 vs our 0.9612). Adding real rows therefore injects a *different* p. The
question is purely empirical: does the extra 129k samples outweigh the distribution mismatch, and
how does that trade off with weight?

Leak audit before we start: the 21 original rows that exactly match a competition row on all 21
predictors are dropped, so no original row can act as an answer key.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features import s6e10 as S  # noqa: E402
from src.features.view import RAW21, ViewBuilder  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

XT = dict(learning_rate=0.02, num_leaves=127, extra_trees=True)


def aligned_original(og: pd.DataFrame, names: list[str], n_static: int) -> np.ndarray:
    """Build the original rows in exactly the same column space as the engineered matrix."""
    from src.features.enrich import _numeric_view  # noqa: F401  (kept for parity of helpers)

    comb_cols = RAW21
    sub = og[comb_cols].copy()
    for c in sub.columns:
        if sub[c].dtype.kind in "OUS":
            codes = pd.factorize(pd.concat([sub[c].astype("object")]).astype("object"), sort=True)[0]
            sub[c] = codes.astype("float64")
        else:
            sub[c] = pd.to_numeric(sub[c], errors="coerce").astype("float64")
    # the transductive / external / token / TE blocks are statistics over the competition rows;
    # for original rows we take the transductive-free path: raw numerics + categorical codes,
    # and fill every engineered column with its competition median so the column space matches.
    X = np.full((len(sub), n_static), np.nan, dtype="float64")
    for j, c in enumerate(comb_cols):
        X[:, j] = sub[c].to_numpy()
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    return X.astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--weights", default="1.0,0.3,0.1")
    ap.add_argument("--folds", default="primary")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    ntr = len(tr)
    folds = get_scheme(args.folds, y, tr[ID_COL]).folds

    vb = ViewBuilder(tr, te, args.view)
    st_tr, st_te, snames = vb.build_static()
    feats = [c for c in te.columns if c != ID_COL]
    og = S.load_original()
    hc = S._raw_key(pd.concat([tr[feats], te[feats]], ignore_index=True), feats)
    ho = S._raw_key(og, feats)
    keep = ~ho.isin(set(hc.unique()))
    print(f"original rows kept after exact-overlap audit: {int(keep.sum())} / {len(og)}")
    og = og[keep].reset_index(drop=True)

    Xo = aligned_original(og, snames, st_tr.shape[1])
    yo = og[TARGET].values.astype("int8")
    print(f"original matrix aligned to {Xo.shape[1]} engineered columns")

    import lightgbm as lgb
    from scripts.run_views import _inner_es_split

    rep = {}
    for w in [float(x) for x in args.weights.split(",")]:
        oof = np.zeros(ntr)
        set_seed(1)
        t0 = time.time()
        for k in sorted(set(folds.tolist())):
            fit = np.where(folds != k)[0]
            val = np.where(folds == k)[0]
            Xf, Xa, names = vb.assemble(fit, y, val, None, inner_seed=k)
            # The `full` view has fold-safe target-encoding columns that do not exist for the
            # original rows. An original row is simply an *unseen* key, so the honest encoding for
            # it is the fold-train prior -- exactly what FoldSafeTE falls back to.
            n_extra = Xf.shape[1] - st_tr.shape[1]
            prior = float(y[fit].mean())
            if n_extra > 0:
                Xo_full = np.hstack([Xo, np.full((len(Xo), n_extra), prior, dtype="float32")])
            else:
                Xo_full = Xo
            es_tr, es_idx = _inner_es_split(fit, y, 1 + k)
            pos = {v: i for i, v in enumerate(fit)}
            es_local = np.array([pos[v] for v in es_idx])
            tr_local = np.array([pos[v] for v in es_tr])
            Xfit = Xf[tr_local]
            yfit = y[es_tr]
            if w > 0:
                Xfit = np.vstack([Xfit, Xo_full])
                yfit = np.concatenate([yfit, yo])
            esX = np.vstack([Xf[es_local], Xo_full])
            esY = np.concatenate([y[es_idx], yo])
            ds = lgb.Dataset(Xfit, label=yfit)
            dw = None
            if 0 < w < 1:
                dw = np.concatenate([np.ones(len(tr_local)), np.full(len(yo), w)])
                ds = lgb.Dataset(Xfit, label=yfit, weight=dw)
            dv = lgb.Dataset(esX, label=esY, reference=ds,
                             weight=np.concatenate([np.ones(len(es_local)), np.full(len(yo), w)]))
            m = lgb.train({**XT, "objective": "binary", "metric": "auc", "n_estimators": 6000,
                           "verbose": -1, "n_jobs": 8, "random_state": 1},
                          ds, num_boost_round=6000, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            oof[val] = m.predict(Xa["val"], num_iteration=m.best_iteration)
        auc = float(roc_auc_score(y, oof))
        fa = [float(roc_auc_score(y[folds == k], oof[folds == k]))
              for k in sorted(set(folds.tolist()))]
        rep[str(w)] = {"oof_auc": round(auc, 6), "fold_aucs": [round(x, 6) for x in fa]}
        print(f"  original-row weight = {w:<5} OOF AUC = {auc:.6f}  "
              f"folds={[round(x,6) for x in fa]}  ({time.time()-t0:.0f}s)", flush=True)
        np.save(REPORTS.parent / "artifacts" / "predictions" / f"probe_origw{w}_oof.npy",
                oof.astype("float32"))
        save_json(rep, REPORTS / "original_rows_probe.json")

    print("\nwrote", REPORTS / "original_rows_probe.json")


if __name__ == "__main__":
    main()