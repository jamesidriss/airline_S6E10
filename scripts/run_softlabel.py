"""Phase 2: soft-target / denoising campaign.

Setup
-----
Teacher = the model trained ONLY on the original 129,880-row survey (`teach_lgbm` in the view).
It depends on no competition label, so it cannot leak. The *calibration* of that teacher is fitted
per outer fold on that fold's training rows only.

For each outer fold k, with fit rows F and validation rows V:
  1. fit calibration of the teacher on F  (logit slope/intercept, or isotonic)
  2. apply it to both F and V
  3. build soft targets for the FIT rows only:  q = (1 - lambda) * y + lambda * p_calibrated
  4. train a model that genuinely accepts fractional labels, predict V, score

Controls
--------
  * lambda = 0 with the *same* objective -> isolates the effect of soft labels from the effect of
    the objective choice
  * the standard hard-label binary model -> the real champion baseline

Objectives are only used where fractional labels were *verified* to be accepted
(see reports/soft_label_support.json): LightGBM `cross_entropy`, CatBoost `CrossEntropy`,
torch BCE. XGBoost's binary objective rejects fractional labels and is deliberately not used
rather than silently coercing them.

Usage:
  python scripts/run_softlabel.py --model lgbm --cal logit --lams 0,0.05,0.1,0.2,0.35,0.5
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
from src.features.softlabel import (ExternalTeacherCalibrator, optimal_lambda_shrinkage,  # noqa: E402
                                    soft_targets)
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
import experiments_ledger as led  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

TEACHER_COL = "teach_lgbm"


def soft_bce_grad(preds, dataset):
    """BCE gradients for fractional targets (used only as a fallback; the native
    cross_entropy objective already accepts fractional labels in this LightGBM version)."""
    q = dataset.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    return p - q, np.maximum(p * (1.0 - p), 1e-6)


def run_lgbm(Xf, q, Xv, params, seed, es_X, es_y_hard):
    """Train on fractional targets `q`, but early-stop on HARD labels.

    This distinction is essential and was the cause of a false negative. ROC-AUC is rank
    invariant, so evaluating it against *fractional* labels is immediately satisfied by merely
    reproducing the prior's ranking -- early stopping then fires after one boosting round and the
    model never learns anything (observed: median best_iteration = 1 for every lambda > 0). The
    early-stopping holdout is carved from inside the fit rows, so its hard labels are
    legitimately available.
    """
    import lightgbm as lgb

    p = {"objective": "cross_entropy", "metric": "auc", "n_estimators": 6000,
         "learning_rate": 0.02, "num_leaves": 127, "min_child_samples": 40,
         "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1, "reg_lambda": 1.0,
         "max_bin": 255, "verbose": -1, "n_jobs": 8, "random_state": seed,
         "bagging_seed": seed + 1, "feature_fraction_seed": seed + 2}
    p.update(params)
    ds = lgb.Dataset(Xf, label=q)
    dv = lgb.Dataset(es_X, label=es_y_hard, reference=ds)
    m = lgb.train(p, ds, num_boost_round=p["n_estimators"], valid_sets=[dv],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict(Xv, num_iteration=m.best_iteration), int(m.best_iteration)


def run_cat(Xf, q, Xv, params, seed, es_X, es_y_hard):
    """Same contract as run_lgbm: soft training targets, hard-label early stopping."""
    from catboost import CatBoostClassifier

    p = dict(iterations=6000, learning_rate=0.04, depth=8, l2_leaf_reg=3.0,
             loss_function="CrossEntropy", eval_metric="AUC", random_seed=seed,
             thread_count=8, verbose=0, allow_writing_files=False)
    p.update(params)
    m = CatBoostClassifier(**p)
    m.fit(Xf, q, eval_set=(es_X, es_y_hard), early_stopping_rounds=300, verbose=0)
    return m.predict_proba(Xv)[:, 1], int(m.get_best_iteration())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--model", default="lgbm", choices=["lgbm", "cat"])
    ap.add_argument("--cal", default="logit", choices=["logit", "isotonic"])
    ap.add_argument("--lams", default="0,0.02,0.05,0.10,0.20,0.35,0.50")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--tag", default="soft")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--extra-trees", action="store_true")
    ap.add_argument("--save-test", action="store_true")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    ntr, nte = len(tr), len(te)
    folds = get_scheme(args.folds, y.astype("int8"), tr[ID_COL]).folds

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()
    if TEACHER_COL not in vb.static_names:
        raise SystemExit(f"teacher column {TEACHER_COL} not in view {args.view}")
    t_idx = vb.static_names.index(TEACHER_COL)
    teacher_prob_all = vb.static_tr[:, t_idx].astype("float64")

    params = {}
    if args.extra_trees:
        params["extra_trees"] = True
    if args.model == "cat":
        params = {}

    lams = [float(x) for x in args.lams.split(",")]
    oofs = {lam: np.zeros(ntr) for lam in lams}
    testp = {lam: np.zeros(nte) for lam in lams}
    iters = {lam: [] for lam in lams}
    diag = []

    ks = sorted(set(folds.tolist()))
    for k in ks:
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        set_seed(args.seed + k)

        cal = ExternalTeacherCalibrator(args.cal).fit(teacher_prob_all[fit], y[fit])
        p_cal_all = cal.apply(teacher_prob_all)     # calibrated for every row, params from fit
        diag.append({"fold": int(k), "cal": args.cal, "n_fit": int(cal.n_fit),
                     "support_ok": bool(cal.support_ok),
                     "teacher_auc_on_fit": round(cal.fit_auc_teacher, 6),
                     "calibrated_auc_on_fit": round(cal.fit_auc_calibrated, 6),
                     "coef": cal.params.get("coef"), "intercept": cal.params.get("intercept")})
        print(f"  fold{k}: teacher AUC(fit)={cal.fit_auc_teacher:.6f} -> "
              f"calibrated={cal.fit_auc_calibrated:.6f}  "
              f"{'support_ok' if cal.support_ok else 'UNSUPPORTED ISOTONIC'}", flush=True)

        Xf, Xa, names = vb.assemble(fit, y.astype("int8"), val, np.arange(ntr, ntr + nte))
        Xtest = Xa["test"]
        # NOTE: _inner_es_split returns (train_subset, es_subset) -- the first value is what we
        # train on and the second is the early-stopping holdout carved out of the SAME fit rows.
        inner_tr, inner_es = _inner_es_split(fit, y.astype("int8"), args.seed + k)
        pos = {v: i for i, v in enumerate(fit)}
        tr_local = np.array([pos[v] for v in inner_tr])
        es_local = np.array([pos[v] for v in inner_es])

        for lam in lams:
            q_all = soft_targets(y[fit], p_cal_all[fit], lam)
            Xtr_f, ytr_f = Xf[tr_local], q_all[tr_local]
            # early stopping uses HARD labels from inside the fit rows (see run_lgbm docstring)
            esX, esY_hard = Xf[es_local], y[inner_es].astype("float64")
            t0 = time.time()
            if args.model == "lgbm":
                o, it = run_lgbm(Xtr_f, ytr_f, Xa["val"], params, args.seed + k, esX, esY_hard)
            else:
                o, it = run_cat(Xtr_f, ytr_f, Xa["val"], params, args.seed + k, esX, esY_hard)
            oofs[lam][val] = o
            iters[lam].append(it)

        print(f"  fold{k} done in {time.time()-t0:.0f}s", flush=True)

    print("\n" + "=" * 96)
    print(f"SOFT-TARGET RESULTS  model={args.model} view={args.view} cal={args.cal} "
          f"extra_trees={args.extra_trees} scheme={args.folds}")
    print("=" * 96)
    rows = []
    ctrl = None
    for lam in lams:
        oof = oofs[lam]
        auc = float(roc_auc_score(y, oof))
        fa = [float(roc_auc_score(y[folds == kk], oof[folds == kk])) for kk in ks]
        if lam == 0.0 and ctrl is None:
            ctrl = oof
        delta = None if ctrl is None or lam == 0.0 else auc - float(roc_auc_score(y, ctrl))
        rows.append({"lam": lam, "oof_auc": round(auc, 6),
                     "delta_vs_lambda0": None if delta is None else round(delta, 6),
                     "fold_aucs": [round(x, 6) for x in fa],
                     "median_iter": int(np.median(iters[lam])) if iters[lam] else None})
        print(f"  lambda={lam:<5} OOF AUC={auc:.6f}  "
              f"delta={'' if delta is None else f'{delta:+.6f}'}  "
              f"folds={[round(x, 6) for x in fa]}")

    # store every lambda so the best can be blended later
    best = max(rows, key=lambda r: r["oof_auc"])
    out = {"model": args.model, "view": args.view, "cal": args.cal, "scheme": args.folds,
           "extra_trees": args.extra_trees, "rows": rows, "calibration_diagnostics": diag,
           "best_lambda": best["lam"], "best_auc": best["oof_auc"]}
    for lam in lams:
        store.save(f"{args.tag}_{args.model}_{args.cal}_{args.view}_l{lam}_{args.folds}",
                   oofs[lam], None, fold_scheme=args.folds,
                   meta={"family": f"{args.model}_soft", "featureset": args.view,
                         "auc": round(float(roc_auc_score(y, oofs[lam])), 6), "lam": lam})
    save_json(out, REPORTS / f"{args.tag}_{args.model}_{args.cal}_{args.view}.json")
    print(f"\nbest lambda={best['lam']} auc={best['oof_auc']:.6f}")
    print("wrote", REPORTS / f"{args.tag}_{args.model}_{args.cal}_{args.view}.json")


if __name__ == "__main__":
    main()