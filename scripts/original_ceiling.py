"""Is there any headroom left, or are we at the noise ceiling?

Labels in S6E10 are i.i.d. Bernoulli(p(x)). The Bayes AUC is therefore the AUC of the *true* p,
and the binding constraint is how well p can be estimated from ~700k noisy labels.

The original 129,880-row survey is a sample from the same population with real labels, so it gives
an independent read on p. Its naive 5-fold AUC (0.9948) is duplicate-inflated: 49% of rows share
a (13 ratings + 4 categoricals) key, so validation rows have near-twins in the training fold.
Re-measuring with **grouped folds** -- all rows sharing a key land in the same fold -- removes that
inflation and gives an honest estimate of how learnable p is.

If the honest number is near 0.961-0.963 we are at the ceiling and the only remaining lever is
variance reduction (more averaging). If it is much higher, the synthetic label noise is the
binding constraint and the same conclusion holds -- but for a different, provable reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, save_json  # noqa: E402
from src.features.s6e10 import META4, SURVEY13, load_original  # noqa: E402

KEY_SETS = {
    "S13+M4+age": SURVEY13 + META4 + ["Age"],
    "S13+M4": SURVEY13 + META4,
    "S13": SURVEY13,
}


def main() -> None:
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold

    og = load_original()
    yo = og["satisfaction"].values.astype("int8")
    feats = [c for c in og.columns if c != "satisfaction"]

    X = og[feats].copy()
    for c in X.columns:
        if X[c].dtype.kind in "OUS":
            X[c] = pd.factorize(X[c].astype("object"), sort=True)[0].astype("float32")
        else:
            X[c] = X[c].astype("float32")
    X = X.fillna(-1.0)

    rep = {}
    for name, cols in KEY_SETS.items():
        g = pd.util.hash_pandas_object(og[cols].astype(object), index=False)
        for mode in ("grouped", "stratified"):
            aucs = []
            for seed in (0, 1, 2):
                if mode == "grouped":
                    sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                    it = sp.split(X, yo, groups=g.values)
                else:
                    from sklearn.model_selection import StratifiedKFold
                    it = StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yo)
                oof = np.zeros(len(yo))
                for a, b in it:
                    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=63,
                                           colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                                           n_jobs=8, verbose=-1, random_state=seed)
                    m.fit(X.iloc[a], yo[a])
                    oof[b] = m.predict_proba(X.iloc[b])[:, 1]
                aucs.append(float(roc_auc_score(yo, oof)))
            rep[f"{name}|{mode}"] = {"auc_mean": round(float(np.mean(aucs)), 6),
                                     "auc_std": round(float(np.std(aucs)), 6),
                                     "per_seed": [round(a, 6) for a in aucs]}
            print(f"{name:<12} {mode:<11} AUC = {np.mean(aucs):.6f} "
                  f"(sd {np.std(aucs):.6f})  {[round(a,6) for a in aucs]}")

    # how many original rows share a key at all
    for name, cols in KEY_SETS.items():
        g = pd.util.hash_pandas_object(og[cols].astype(object), index=False)
        vc = g.value_counts()
        print(f"{name:<12} n_groups={len(vc):>7}  rows in a group of size>1: "
              f"{float(vc[vc>1].sum()/len(g)):.4f}")

    save_json(rep, REPORTS / "original_ceiling.json")
    print("\nwrote", REPORTS / "original_ceiling.json")


if __name__ == "__main__":
    main()