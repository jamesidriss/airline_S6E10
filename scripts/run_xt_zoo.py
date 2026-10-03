"""Zoo pass 2: focus on what pass 1 showed works.

Pass 1's surprise was `extra_trees=True` in LightGBM (extremely randomised splits) beating every
other single GBDT by ~2.3e-4. That is consistent with the diagnosis that labels are i.i.d.
Bernoulli(p(x)): random splits decorrelate the trees more aggressively and average the noise out
better than a fully grown deterministic tree.

This pass therefore explores the random-split family (LightGBM `extra_trees`, XGBoost
`colsample_bynode`/`grow_policy=lossguide`, ExtraTreesClassifier) plus its interaction with
learning rate, leaf count and row/column sampling.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
import experiments_ledger as led  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.run_zoo import LGBM, XGB, ETR, predict_test, run_gbdt  # noqa: E402

XT: list[tuple[str, str, str, str, dict, int]] = [
    # --- LightGBM extra_trees family -------------------------------------------------
    ("xt_d127_s1", LGBM, "full", "primary", dict(learning_rate=0.02, num_leaves=127,
                                                 extra_trees=True), 1),
    ("xt_d127_s2", LGBM, "full", "primary", dict(learning_rate=0.02, num_leaves=127,
                                                 extra_trees=True), 2),
    ("xt_d63", LGBM, "full", "primary", dict(learning_rate=0.03, num_leaves=63,
                                             extra_trees=True), 3),
    ("xt_d255", LGBM, "full", "primary", dict(learning_rate=0.02, num_leaves=255,
                                               min_child_samples=80, extra_trees=True), 4),
    ("xt_d127_cs05", LGBM, "full", "primary", dict(learning_rate=0.02, num_leaves=127,
                                                   colsample_bytree=0.5, extra_trees=True), 5),
    ("xt_d127_mcs100", LGBM, "full", "primary", dict(learning_rate=0.02, num_leaves=127,
                                                     min_child_samples=100, extra_trees=True), 6),
    ("xt_d127_ss06", LGBM, "full", "primary", dict(learning_rate=0.02, num_leaves=127,
                                                   subsample=0.6, extra_trees=True), 7),
    ("xt_d127_bin63", LGBM, "full", "primary", dict(learning_rate=0.02, num_leaves=127,
                                                    max_bin=63, extra_trees=True), 8),
    ("xt_d63_lr015", LGBM, "full", "primary", dict(learning_rate=0.015, num_leaves=63,
                                                   extra_trees=True), 9),
    ("xt_ogs", LGBM, "full_ogsurf", "primary", dict(learning_rate=0.02, num_leaves=127,
                                                    extra_trees=True), 10),
    ("xt_core3", LGBM, "core3", "primary", dict(learning_rate=0.02, num_leaves=127,
                                                extra_trees=True), 11),
    ("xt_f10", LGBM, "full", "block10", dict(learning_rate=0.02, num_leaves=127,
                                              extra_trees=True), 1),
    # --- XGBoost randomised-grow family ---------------------------------------------
    ("xgb_lossguide", XGB, "full", "primary", dict(learning_rate=0.03, max_depth=0, grow_policy="lossguide",
                                                    max_leaves=127, min_child_weight=8,
                                                    colsample_bytree=0.8), 12),
    ("xgb_bynode", XGB, "full", "primary", dict(learning_rate=0.03, max_depth=8,
                                                colsample_bynode=0.7, colsample_bytree=0.8), 13),
    # --- sklearn ExtraTreesClassifier ---------------------------------------------------
    ("et_full", ETR, "full", "primary", dict(n_estimators=1000, max_features=0.5,
                                             min_samples_leaf=4), 14),
    ("et_full_mf03", ETR, "full", "primary", dict(n_estimators=1000, max_features=0.3,
                                                  min_samples_leaf=8), 15),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="xt")
    ap.add_argument("--save-test", action="store_true")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    ntr, nte = len(tr), len(te)
    idx = store._load_index()
    rows = []
    views = sorted({v for n, f, v, s, p, sd in XT if not args.only or n in args.only.split(",")})
    vbs, schemes = {}, {}
    for v in views:
        vb = ViewBuilder(tr, te, v)
        vb.build_static()
        vbs[v] = vb
    for s in sorted({sc for n, f, v, sc, p, sd in XT if not args.only or n in args.only.split(",")}):
        schemes[s] = get_scheme(s, y, tr[ID_COL]).folds

    for name, family, view, scheme, params, seed in XT:
        if args.only and name not in args.only.split(","):
            continue
        eid = f"{args.tag}_{name}"
        if eid in idx and "test" in idx[eid]:
            print(f"[xt] skip {eid}")
            continue
        folds = schemes[scheme]
        vb = vbs[view]
        oof, tst, iters = np.zeros(ntr), np.zeros(nte), []
        nks = len(set(folds.tolist()))
        set_seed(seed)
        t0 = time.time()
        for k in sorted(set(folds.tolist())):
            fit = np.where(folds != k)[0]
            val = np.where(folds == k)[0]
            Xf, Xa, _ = vb.assemble(fit, y, val, None, inner_seed=k)
            o, it = run_gbdt(family, Xf, Xa["val"], None, y, fit, val, params, seed)
            iters.append(it)
            oof[val] = o
            del Xf, Xa
        n_it = int(np.median(iters)) if any(iters) else 0
        set_seed(seed)
        for k in sorted(set(folds.tolist())):
            fit = np.where(folds != k)[0]
            val = np.where(folds == k)[0]
            Xf, Xa, _ = vb.assemble(fit, y, val, np.arange(ntr, ntr + nte), inner_seed=k)
            tst += predict_test(family, Xf, Xa["test"], y, fit, params, seed, n_it) / nks
            del Xf, Xa
        auc = float(roc_auc_score(y, oof))
        fa = [float(roc_auc_score(y[folds == k], oof[folds == k])) for k in sorted(set(folds.tolist()))]
        store.save(eid, oof, tst if args.save_test else None, fold_scheme=scheme,
                   meta={"family": family, "featureset": view, "auc": round(auc, 6),
                         "params": params, "seed": seed, "n_iter": n_it})
        led.log_experiment(family=family, featureset=view, params=params, seed=seed,
                           fold_scheme=scheme, oof_auc=auc, fold_aucs=fa, oof=oof,
                           duration_s=time.time() - t0, data_hash=f"tr{ntr}-te{nte}",
                           test_pred=tst if args.save_test else None, exp_id=eid,
                           notes="xt-zoo", verdict="zoo-member",
                           extra={"fold_ids": folds.tolist()})
        print(f"[xt] {name:<16} {family:<5} {view:<15} {scheme:<9} OOF={auc:.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        rows.append({"name": name, "exp_id": eid, "family": family, "view": view,
                     "scheme": scheme, "seed": seed, "oof_auc": round(auc, 6)})
        save_json(rows, REPORTS / f"{args.tag}_results.json")

    print("\n=== XT SUMMARY ===")
    print(pd.DataFrame(rows).sort_values("oof_auc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()