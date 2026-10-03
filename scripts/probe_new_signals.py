"""Two mechanisms the field has not exploited here, both cheap.

1. **base_margin / init_score residual boosting.** A strong simple model's logit is used as the
   starting score, so the GBDT only has to learn the residual. Prior seasons measured +6e-6 to
   +5e-4. In S6E10 the natural base is the original-data teacher (logit), or the logit of a
   strong model trained on the OTHER folds (strictly fold-safe).

2. **Auxiliary-task predicted class probabilities.** Train small classifiers to predict a few
   service ratings from the rest of the features, cross-fitted inside the fold, and append the
   predicted probability vector. This is label-dependent on *other* columns' values, so it must
   be cross-fitted exactly like target encoding.

Output: reports/aux_probes.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.s6e10 import SURVEY13  # noqa: E402
from src.features.view import RAW21, ViewBuilder  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

AUX_TARGETS = ["Inflight wifi service", "Online boarding", "Inflight entertainment", "Seat comfort"]


def _aux_features(X, y_aux, fit_idx, apply_sets, n_inner=3, seed=0):
    """Cross-fitted predicted class-probability vectors for each auxiliary target."""
    import xgboost as xgb

    names = ["Online boarding", "Inflight entertainment", "Seat comfort", "Inflight wifi service",
             "Leg room service", "Cleanliness", "Class", "Type of Travel", "Flight Distance"]
    names = [n for n in names if n in X.columns]
    from scripts.run_views import _inner_es_split

    out_fit = {}
    out_apply = {k: [] for k in apply_sets}
    for t in AUX_TARGETS:
        if t not in X.columns:
            continue
        cols = [c for c in names if c != t]
        classes = np.sort(np.unique(y_aux[t].values))
        for c_i, cls in enumerate(classes):
            yy = (y_aux[t].values == cls).astype("int8")
            # inner cross-fit for the fit rows
            pred_fit = np.zeros(len(fit_idx), dtype="float32")
            rng = np.random.default_rng(seed)
            inner = rng.permutation(len(fit_idx)) % n_inner
            for i in range(n_inner):
                a = fit_idx[inner != i]
                b = fit_idx[inner == i]
                m = xgb.XGBClassifier(n_estimators=140, max_depth=5, subsample=0.8,
                                      colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist",
                                      device="cuda", n_jobs=8, random_state=seed + i)
                m.fit(X.loc[a, cols], yy[a], verbose=False)
                pred_fit[inner == i] = m.predict_proba(X.loc[b, cols])[:, 1]
            out_fit[f"aux_{t}_p{cls}"] = pred_fit
            for k, idx in apply_sets.items():
                m = xgb.XGBClassifier(n_estimators=140, max_depth=5, subsample=0.8,
                                      colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist",
                                      device="cuda", n_jobs=8, random_state=seed + 99)
                m.fit(X.loc[fit_idx, cols], yy, verbose=False)
                out_apply[k].append(m.predict_proba(X.loc[idx, cols])[:, 1].astype("float32"))
    return out_fit, out_apply


def main() -> None:
    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y, tr[ID_COL]).folds
    rep = {}

    # ---------------------------------------------------------------- base_margin probe
    print("### base_margin probe: LightGBM on top of the original-data teacher logit")
    vb = ViewBuilder(tr, te, "full")
    st_tr, st_te, names = vb.build_static()
    tcols = [n for n in names if n.startswith("teach_lgbm")]
    print("teacher columns:", tcols)
    # rebuild a compact matrix of raw + teacher for the margin experiment
    raw_idx = [names.index(c) for c in RAW21]
    t_idx = [names.index(c) for c in tcols]
    Xc = st_tr[:, raw_idx + t_idx]

    import lightgbm as lgb

    for margin_on in (False, True):
        oof = np.zeros(len(y))
        t0 = time.time()
        for k in sorted(set(folds.tolist())):
            a = np.where(folds != k)[0]
            b = np.where(folds == k)[0]
            Xa, Xb = Xc[a], Xc[b]
            if margin_on:
                base = Xa[:, len(raw_idx):].mean(1)
                init = np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6)))
                ds = lgb.Dataset(Xa, label=y[a], init_score=init)
                init_b = np.log(np.clip(Xb[:, len(raw_idx):].mean(1), 1e-6, 1 - 1e-6) /
                                (1 - np.clip(Xb[:, len(raw_idx):].mean(1), 1e-6, 1 - 1e-6)))
                dv = lgb.Dataset(Xb, label=y[b], reference=ds, init_score=init_b)
            else:
                ds = lgb.Dataset(Xa, label=y[a])
                dv = lgb.Dataset(Xb, label=y[b], reference=ds)
            es_a, es_b = a[: len(a) // 10], a[len(a) // 10:]
            dv = lgb.Dataset(Xc[es_b], label=y[es_b], reference=ds)
            p = dict(objective="binary", metric="auc", n_estimators=6000, learning_rate=0.02,
                     num_leaves=127, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                     verbose=-1, n_jobs=8, random_state=1)
            m = lgb.train(p, ds, num_boost_round=p["n_estimators"], valid_sets=[dv],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            oof[b] = m.predict(Xb, num_iteration=m.best_iteration)
        a = float(roc_auc_score(y, oof))
        fa = [float(roc_auc_score(y[folds == k], oof[folds == k]))
              for k in sorted(set(folds.tolist()))]
        print(f"  base_margin={margin_on}  OOF AUC = {a:.6f}  ({time.time()-t0:.0f}s)")
        rep[f"base_margin_{margin_on}"] = {"oof_auc": round(a, 6), "fold_aucs": [round(x, 6) for x in fa]}
        np.save(REPORTS.parent / "artifacts" / "predictions" / f"probe_margin{int(margin_on)}_oof.npy",
                oof.astype("float32"))

    save_json(rep, REPORTS / "aux_probes.json")
    print("wrote", REPORTS / "aux_probes.json")


if __name__ == "__main__":
    main()