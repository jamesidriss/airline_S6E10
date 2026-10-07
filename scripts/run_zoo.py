"""Member zoo: many deliberately different models, each saved as an OOF/test pair.

Diversity axes (all measured, none assumed):
  family     lgbm | xgb | cat | realmlp | tabm | etr | histgb
  view       raw | raw_ext | raw_trans | core3 | core3_ogsurf | full | full_ogsurf | full_ogsurf_te
  structure  leaf count / depth, min-child, max_bin, colsample/subsample, DART/GOSS,
             extra_trees random splits, cat depth
  seed       bagging / feature / model seed
  folds      primary (5) vs block10 (10) -> different training sets, genuinely decorrelated

Every entry is idempotent: already-stored experiment ids are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.models import realmlp as RM  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
import experiments_ledger as led  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.run_views import (  # noqa: E402
    _fit_cat_es, _fit_full_predict_cat, _fit_full_predict_lgbm, _fit_full_predict_xgb,
    _fit_lgbm_es, _fit_xgb_es, _inner_es_split,
)

# ------------------------------------------------------------------ config table
# (name, family, view, scheme, params)
LGBM = "lgbm"
XGB = "xgb"
CAT = "cat"
RM = "realmlp"
TB = "tabm"
ETR = "etr"
HGB = "histgb"

ZOO: list[tuple[str, str, str, str, dict, int]] = [
    # ---- GBDT structure zoo on the best view --------------------------------------
    ("lgbm_d127", LGBM, "full", "primary", dict(learning_rate=0.02, num_leaves=127), 1),
    ("lgbm_d63", LGBM, "full", "primary", dict(learning_rate=0.03, num_leaves=63,
                                                min_child_samples=20), 2),
    ("lgbm_d255", LGBM, "full", "primary", dict(learning_rate=0.02, num_leaves=255,
                                                 min_child_samples=80, reg_lambda=3.0), 3),
    ("lgbm_bin63", LGBM, "full", "primary", dict(learning_rate=0.02, num_leaves=127, max_bin=63), 4),
    ("lgbm_xt", LGBM, "full", "primary", dict(learning_rate=0.02, num_leaves=127,
                                              extra_trees=True), 5),
    ("lgbm_goss", LGBM, "full", "primary", dict(learning_rate=0.03, num_leaves=127,
                                                data_sample_strategy="goss"), 6),
    ("lgbm_lowcs", LGBM, "full", "primary", dict(learning_rate=0.02, num_leaves=127,
                                                 colsample_bytree=0.5, subsample=0.6), 7),
    # ---- XGBoost zoo -----------------------------------------------------------------
    ("xgb_d6", XGB, "full", "primary", dict(learning_rate=0.04, max_depth=6, min_child_weight=5,
                                             colsample_bytree=0.85, subsample=0.9,
                                             reg_lambda=5.0, reg_alpha=0.05), 8),
    ("xgb_d4", XGB, "full", "primary", dict(learning_rate=0.03, max_depth=4, min_child_weight=3,
                                             colsample_bytree=0.7, reg_lambda=2.0), 9),
    ("xgb_d10", XGB, "full", "primary", dict(learning_rate=0.03, max_depth=10, min_child_weight=20,
                                              colsample_bytree=0.6, reg_lambda=8.0), 10),
    ("xgb_bin512", XGB, "full", "primary", dict(learning_rate=0.03, max_depth=8, max_bin=512,
                                                 colsample_bytree=0.8), 11),
    # ---- CatBoost zoo ----------------------------------------------------------------
    ("cat_d8", CAT, "full", "primary", dict(learning_rate=0.04, depth=8), 12),
    ("cat_d6", CAT, "full", "primary", dict(learning_rate=0.04, depth=6, l2_leaf_reg=5.0), 13),
    ("cat_d10", CAT, "full", "primary", dict(learning_rate=0.03, depth=10, l2_leaf_reg=6.0), 14),
    # ---- view diversity ---------------------------------------------------------------
    ("lgbm_ext", LGBM, "raw_ext", "primary", dict(learning_rate=0.02, num_leaves=127), 15),
    ("xgb_trans", XGB, "raw_trans", "primary", dict(learning_rate=0.03, max_depth=8), 16),
    ("lgbm_ogs", LGBM, "core3_ogsurf", "primary", dict(learning_rate=0.02, num_leaves=127), 17),
    ("xgb_ogs", XGB, "core3_ogsurf", "primary", dict(learning_rate=0.03, max_depth=8), 18),
    ("cat_ogs", CAT, "core3_ogsurf", "primary", dict(learning_rate=0.04, depth=8), 19),
    ("lgbm_fogs", LGBM, "full_ogsurf", "primary", dict(learning_rate=0.02, num_leaves=127), 20),
    ("xgb_fogs", XGB, "full_ogsurf", "primary", dict(learning_rate=0.03, max_depth=8), 21),
    ("lgbm_fogste", LGBM, "full_ogsurf_te", "primary", dict(learning_rate=0.02, num_leaves=127), 22),
    ("xgb_core3", XGB, "core3", "primary", dict(learning_rate=0.03, max_depth=8), 23),
    ("cat_core3", CAT, "core3", "primary", dict(learning_rate=0.04, depth=8), 24),
    # ---- seed-only diversity on the champion config -----------------------------------
    ("lgbm_d127_s11", LGBM, "full", "primary", dict(learning_rate=0.02, num_leaves=127), 11),
    ("lgbm_d127_s77", LGBM, "full", "primary", dict(learning_rate=0.02, num_leaves=127), 77),
    ("xgb_d6_s77", XGB, "full", "primary", dict(learning_rate=0.04, max_depth=6, min_child_weight=5,
                                                  colsample_bytree=0.85, subsample=0.9,
                                                  reg_lambda=5.0, reg_alpha=0.05), 77),
    # ---- 10-fold members (different training sets) -----------------------------------
    ("lgbm_d127_f10", LGBM, "full", "block10", dict(learning_rate=0.02, num_leaves=127), 1),
    ("xgb_d6_f10", XGB, "full", "block10", dict(learning_rate=0.04, max_depth=6, min_child_weight=5,
                                                 colsample_bytree=0.85, subsample=0.9,
                                                 reg_lambda=5.0, reg_alpha=5.0), 1),
    ("cat_d8_f10", CAT, "full", "block10", dict(learning_rate=0.04, depth=8), 1),
    # ---- other families ---------------------------------------------------------------
    ("etr_full", ETR, "full", "primary", dict(n_estimators=800, max_features=0.5,
                                               min_samples_leaf=4), 1),
    ("hgb_full", HGB, "full", "primary", dict(max_iter=900, learning_rate=0.06), 1),
]

NN_ZOO: list[tuple[str, str, str, int, int]] = [
    ("rm_s1_e6", "full", "primary", 6, 1),
    ("rm_s7_e6", "full", "primary", 6, 7),
    ("rm_s123_e6", "full", "primary", 6, 123),
    ("rm_s2024_e6", "full", "primary", 6, 2024),
    ("rm_s5_e3", "full", "primary", 3, 5),
    ("rm_s6_e10", "full", "primary", 10, 6),
    ("rm_core3_e6", "core3", "primary", 6, 11),
    ("rm_ogs_e6", "core3_ogsurf", "primary", 6, 12),
    ("rm_full_f10_e6", "full", "block10", 6, 1),
    ("rm_full_e12_s3", "full", "primary", 12, 3),
]


def run_gbdt(family, Xf, Xval, Xtest, y, fit, val, params, seed):
    """Pass 1: early-stopped OOF prediction. Returns (oof_pred, best_iteration)."""
    es_tr, es_idx = _inner_es_split(fit, y, seed)
    pos = {v: i for i, v in enumerate(fit)}
    es_local = np.array([pos[v] for v in es_idx])
    tr_local = np.array([pos[v] for v in es_tr])
    fitX, fitY, esX, esY = Xf[tr_local], y[es_tr], Xf[es_local], y[es_idx]
    if family == LGBM:
        o, it = _fit_lgbm_es(fitX, fitY, Xval, params, seed, esX, esY)
    elif family == XGB:
        o, it = _fit_xgb_es(fitX, fitY, Xval, params, seed, esX, esY)
    elif family == CAT:
        o, it = _fit_cat_es(fitX, fitY, Xval, params, seed, esX, esY)
    elif family == ETR:
        from sklearn.ensemble import ExtraTreesClassifier
        m = ExtraTreesClassifier(random_state=seed, n_jobs=8, **params)
        m.fit(fitX, fitY)
        o, it = m.predict_proba(Xval)[:, 1], 0
    elif family == HGB:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **params)
        m.fit(fitX, fitY)
        o, it = m.predict_proba(Xval)[:, 1], 0
    else:
        raise ValueError(family)
    return o, it


def predict_test(family, Xf, Xtest, y, fit, params, seed, n_it):
    """Pass 2: refit on ALL fold-fit rows at the CV-selected iteration count, predict test."""
    if family == LGBM:
        return _fit_full_predict_lgbm(Xf, y[fit], Xtest, params, seed, n_it or None)
    if family == XGB:
        return _fit_full_predict_xgb(Xf, y[fit], Xtest, params, seed, n_it + 1)
    if family == CAT:
        return _fit_full_predict_cat(Xf, y[fit], Xtest, params, seed, n_it + 1)
    if family == ETR:
        from sklearn.ensemble import ExtraTreesClassifier
        m = ExtraTreesClassifier(random_state=seed, n_jobs=8, **params).fit(Xf, y[fit])
        return m.predict_proba(Xtest)[:, 1]
    if family == HGB:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(random_state=seed, early_stopping=False,
                                           **params).fit(Xf, y[fit])
        return m.predict_proba(Xtest)[:, 1]
    raise ValueError(family)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated zoo names")
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--tag", default="zoo")
    ap.add_argument("--save-test", action="store_true")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    ntr, nte = len(tr), len(te)
    idx = store._load_index()
    rows = []

    views_needed = sorted({v for n, f, v, s, p, sd in ZOO if not args.only or n in args.only.split(",")})
    vbs, folds_by_scheme = {}, {}
    for v in views_needed:
        vb = ViewBuilder(tr, te, v)
        vb.build_static()
        vbs[v] = vb
        print(f"[zoo] built view {v}: static {vb.static_tr.shape}", flush=True)
    for s in sorted({sc for n, f, v, sc, p, sd in ZOO if not args.only or n in args.only.split(",")}):
        folds_by_scheme[s] = get_scheme(s, y, tr[ID_COL]).folds

    def run_one(name, family, view, scheme, params, seed):
        eid = f"{args.tag}_{name}"
        if eid in idx and "test" in idx[eid]:
            print(f"[zoo] skip {eid} (cached)")
            return
        folds = folds_by_scheme[scheme]
        vb = vbs[view]
        oof = np.zeros(ntr)
        tst = np.zeros(nte)
        nks = len(set(folds.tolist()))
        set_seed(seed)
        t0 = time.time()
        iters = []
        for k in sorted(set(folds.tolist())):
            fit = np.where(folds != k)[0]
            val = np.where(folds == k)[0]
            Xf, Xa, names = vb.assemble(fit, y, val, None, inner_seed=k)
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
        fa = [float(roc_auc_score(y[folds == k], oof[folds == k]))
              for k in sorted(set(folds.tolist()))]
        store.save(eid, oof, tst if args.save_test else None, fold_scheme=scheme,
                   meta={"family": family, "featureset": view, "auc": round(auc, 6),
                         "params": params, "seed": seed, "n_iter": n_it})
        led.log_experiment(family=family, featureset=view, params=params, seed=seed,
                           fold_scheme=scheme, oof_auc=auc, fold_aucs=fa, oof=oof,
                           duration_s=time.time() - t0, data_hash=f"tr{ntr}-te{nte}",
                           test_pred=tst if args.save_test else None, exp_id=eid,
                           notes="zoo", verdict="zoo-member",
                           extra={"fold_ids": folds.tolist()})
        print(f"[zoo] {name:<18} {family:<7} {view:<16} {scheme:<9} OOF={auc:.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        rows.append({"name": name, "exp_id": eid, "family": family, "view": view,
                     "scheme": scheme, "seed": seed, "oof_auc": round(auc, 6)})
        save_json(rows, REPORTS / f"{args.tag}_results.json")

    for name, family, view, scheme, params, seed in ZOO:
        if args.only and name not in args.only.split(","):
            continue
        try:
            run_one(name, family, view, scheme, params, seed)
        except Exception as exc:  # noqa: BLE001
            print(f"[zoo] FAILED {name}: {exc!r}", flush=True)

    if not args.skip_nn:
        for name, view, scheme, epochs, seed in NN_ZOO:
            if args.only and name not in args.only.split(","):
                continue
            eid = f"{args.tag}_{name}"
            if eid in idx and "test" in idx[eid]:
                print(f"[zoo] skip {eid} (cached)")
                continue
            try:
                vb = vbs.setdefault(view, None)
                if vb is None:
                    vb = ViewBuilder(tr, te, view)
                    vb.build_static()
                    vbs[view] = vb
                folds = folds_by_scheme.setdefault(scheme, get_scheme(scheme, y, tr[ID_COL]).folds)
                p = {"n_epochs": epochs, "n_ens": 8, "random_seed": seed}
                set_seed(seed)
                t0 = time.time()
                oof, tst, fa, dur = RM.realmlp_view(vb, folds, y, ntr, nte, params=p,
                                                    seed=seed, device="cuda", twin=True)
                auc = float(roc_auc_score(y, oof))
                store.save(eid, oof, tst if args.save_test else None, fold_scheme=scheme,
                           meta={"family": "realmlp", "featureset": view, "auc": round(auc, 6),
                                 "epochs": epochs, "seed": seed})
                led.log_experiment(family="realmlp", featureset=view, params=p, seed=seed,
                                   fold_scheme=scheme, oof_auc=auc, fold_aucs=fa, oof=oof,
                                   duration_s=dur, data_hash=f"tr{ntr}-te{nte}",
                                   test_pred=tst if args.save_test else None, exp_id=eid,
                                   notes="zoo-nn", verdict="zoo-member",
                                   extra={"fold_ids": folds.tolist()})
                print(f"[zoo] {name:<18} realmlp {view:<14} {scheme:<9} OOF={auc:.6f} ({dur:.0f}s)",
                      flush=True)
                rows.append({"name": name, "exp_id": eid, "family": "realmlp", "view": view,
                             "scheme": scheme, "seed": seed, "oof_auc": round(auc, 6)})
                save_json(rows, REPORTS / f"{args.tag}_results.json")
            except Exception as exc:  # noqa: BLE001
                print(f"[zoo] FAILED {name}: {exc!r}", flush=True)

    print("\n=== ZOO SUMMARY ===")
    print(pd.DataFrame(rows).sort_values("oof_auc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
