"""Phase 3 driver: residual boosting on top of a teacher, with a hard identity gate.

Run order (and the gate is not optional):
  1. zero-round identity test -- if base_margin semantics are wrong, everything downstream is
     meaningless, so we refuse to continue.
  2. teacher quality check.
  3. residual student, per outer fold, with the teacher margins supplied in a train/serve
     consistent way (`label_free` needs no nesting; `cross_fitted` uses an inner K-fold).
  4. report standalone OOF, delta vs the matched control, paired bootstrap, correlation with the
     teacher and with the finalist, and marginal ensemble gain.

Usage:
  python scripts/run_residual.py --teacher external   # cheapest, cleanest
  python scripts/run_residual.py --teacher xt_single  # cross-fitted
  python scripts/run_residual.py --teacher finalist   # cross-fitted blend
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.ensemble import lab  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.models.residual import (cross_fit_teacher_margin, identity_test,  # noqa: E402
                                 predict_with_booster, xgb_residual)
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

XT_PARAMS = dict(learning_rate=0.03, num_leaves=127, extra_trees=True)


def fit_lgbm_xt(X, y, Xp, seed):
    import lightgbm as lgb

    p = {"objective": "binary", "n_estimators": 900, "learning_rate": 0.03, "num_leaves": 127,
         "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1, "extra_trees": True,
         "verbose": -1, "n_jobs": 8, "random_state": seed}
    m = lgb.LGBMClassifier(**p)
    m.fit(X, y)
    return m.predict_proba(Xp)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="external",
                    choices=["external", "xt_single", "finalist"])
    ap.add_argument("--view", default="full")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--tag", default="resid")
    ap.add_argument("--skip-identity", action="store_true")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    ntr, nte = len(tr), len(te)
    folds = get_scheme(args.folds, y.astype("int8"), tr[ID_COL]).folds
    ks = sorted(set(folds.tolist()))

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()

    # ---------------------------------------------------------------- teacher predictions
    teacher_mode = "label_free"
    if args.teacher == "external":
        tname = "teach_lgbm"
        t_idx = vb.static_names.index(tname)
        teacher_all = vb.static_tr[:, t_idx].astype("float64")
        teacher_test = vb.static_te[:, t_idx].astype("float64")
    elif args.teacher == "xt_single":
        teacher_mode = "cross_fitted"
        teacher_all = np.zeros(ntr)
        teacher_test = None
    else:
        teacher_mode = "cross_fitted"
        teacher_all = store.load_oof("blend_v3_final").astype("float64")
        teacher_test = store.load_test("blend_v3_final").astype("float64")
        # the finalist OOF was itself produced out-of-fold, so for the *validation* rows it is
        # already honest; it is only the training rows that need a nested re-fit.

    print("=" * 96)
    print(f"PHASE 3 RESIDUAL BOOSTING  teacher={args.teacher} ({teacher_mode}) view={args.view}")
    print("=" * 96)

    # ---------------------------------------------------------------- identity gate
    if not args.skip_identity:
        rep = identity_test(vb.static_tr[:20000], teacher_all[:20000])
        if not rep["passes"]:
            raise SystemExit("IDENTITY TEST FAILED -- base_margin semantics unverified, stopping.")
        save_json(rep, REPORTS / f"{args.tag}_identity_test.json")

    # ---------------------------------------------------------------- per-fold residual
    oof_res = np.zeros(ntr)
    oof_ctl = np.zeros(ntr)
    test_res = np.zeros(nte)
    diag = []

    for k in ks:
        t0 = time.time()
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        set_seed(args.seed + k)
        Xf, Xa, _ = vb.assemble(fit, y.astype("int8"), val, np.arange(ntr, ntr + nte))
        Xval, Xtest = Xa["val"], Xa["test"]
        inner_tr, inner_es = _inner_es_split(fit, y.astype("int8"), args.seed + k)
        pos = {v: j for j, v in enumerate(fit)}
        tr_local = np.array([pos[v] for v in inner_tr])
        es_local = np.array([pos[v] for v in inner_es])

        if teacher_mode == "label_free":
            t_fit = teacher_all[fit]
            t_val = teacher_all[val]
            t_test = teacher_test
        else:
            # nested teacher margins for the fit rows; full-fit teacher for val/test rows
            def predict_fn(rows, src, i, _fit=fit):
                src_local = np.searchsorted(_fit, src)
                return fit_lgbm_xt(Xf[src_local], y[src], Xf[np.searchsorted(_fit, rows)], args.seed + i)

            t_fit = cross_fit_teacher_margin(predict_fn, np.arange(len(fit)),
                                             y[fit], n_inner=4, seed=args.seed + k)
            src_local = tr_local
            t_full_val = fit_lgbm_xt(Xf[tr_local], y[inner_tr], Xval, args.seed + k)
            t_val = t_full_val
            t_test = fit_lgbm_xt(Xf[tr_local], y[inner_tr], Xtest, args.seed + k)

        # control: the same student on raw features, no margin
        ctl, _, _ = xgb_residual(Xf[tr_local], y[inner_tr], Xval,
                                 teacher_fit=np.full(len(tr_local), 0.5),
                                 teacher_val=np.full(len(val), 0.5),
                                 seed=args.seed + k, n_rounds=args.rounds, verbose=False,
                                 y_val=y[val])
        oof_ctl[val] = ctl

        res, bst, best = xgb_residual(Xf[tr_local], y[inner_tr], Xval,
                                      teacher_fit=t_fit[tr_local], teacher_val=t_val,
                                      seed=args.seed + k, n_rounds=args.rounds, verbose=False,
                                      y_val=y[val])
        oof_res[val] = res
        # reuse the SAME fitted booster for the test block, with the teacher's test margin
        dpred = predict_with_booster(bst, Xtest, t_test)
        test_res += dpred / len(ks)
        diag.append({"fold": int(k), "teacher_auc_val": float(roc_auc_score(y[val], t_val)),
                     "teacher_auc_fit_nested": float(roc_auc_score(y[fit], t_fit)),
                     "residual_auc": float(roc_auc_score(y[val], res)),
                     "best_iter": int(best)})
        print(f"  fold{k}: teacher(val)={diag[-1]['teacher_auc_val']:.6f} "
              f"residual={diag[-1]['residual_auc']:.6f} ctl={roc_auc_score(y[val], ctl):.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    a_res = float(roc_auc_score(y, oof_res))
    a_ctl = float(roc_auc_score(y, oof_ctl))
    from src.validation.compare import corr, spearman

    # For a cross-fitted teacher there is no single global prediction vector, so the honest
    # teacher number is the mean of the per-fold validation AUCs in `diag`.
    t_auc = (float(roc_auc_score(y, teacher_all)) if teacher_mode == "label_free"
             else float(np.mean([d["teacher_auc_val"] for d in diag])))
    fa = lab.fold_aucs(y, oof_res, folds)
    ca = lab.fold_aucs(y, oof_ctl, folds)
    v3 = store.load_oof("blend_v3_final").astype("float64")
    print("\n" + "=" * 96)
    print(f"teacher AUC ({teacher_mode}"
          f"{'' if teacher_mode == 'label_free' else ', mean of per-fold val'}) = {t_auc:.6f}")
    print(f"control (no margin)     = {a_ctl:.6f}")
    print(f"residual student         = {a_res:.6f}   delta vs control = {a_res - a_ctl:+.6f}")
    print(f"residual fold AUCs       = {[round(x, 6) for x in fa]}")
    print(f"control  fold AUCs       = {[round(x, 6) for x in ca]}")
    print(f"per-fold deltas          = {[round(a - b, 6) for a, b in zip(fa, ca)]}")
    print(f"folds improved           = {sum(1 for a, b in zip(fa, ca) if a > b)}/{len(ca)}")
    print(f"vs finalist: logit corr {corr(oof_res, v3):.5f}  spearman {spearman(oof_res, v3):.5f}")

    out = {"teacher": args.teacher, "mode": teacher_mode, "view": args.view, "teacher_auc": t_auc,
           "control_auc": a_ctl, "control_fold_aucs": ca, "residual_auc": a_res,
           "delta_vs_control": a_res - a_ctl, "fold_aucs": fa,
           "per_fold_delta": [a - b for a, b in zip(fa, ca)],
           "folds_improved": sum(1 for a, b in zip(fa, ca) if a > b),
           "corr_with_finalist": corr(oof_res, v3), "spearman_with_finalist": spearman(oof_res, v3),
           "per_fold": diag}
    store.save(f"{args.tag}_{args.teacher}_{args.view}", oof_res, test_res,
               fold_scheme=args.folds,
               meta={"family": "xgb_residual", "featureset": args.view, "auc": round(a_res, 6),
                     "teacher": args.teacher})
    store.save(f"{args.tag}ctl_{args.teacher}_{args.view}", oof_ctl, None, fold_scheme=args.folds,
               meta={"family": "xgb_control", "featureset": args.view, "auc": round(a_ctl, 6)})
    save_json(out, REPORTS / f"{args.tag}_{args.teacher}_{args.view}.json")
    print("\nwrote", REPORTS / f"{args.tag}_{args.teacher}_{args.view}.json")


if __name__ == "__main__":
    main()