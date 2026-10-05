"""Phase 7: recover the inner-ES holdout with a leakage-safe fixed-iteration full-fit policy.

The measured problem
--------------------
scripts/audit_train_fractions.py shows every CV-stage OOF model trains on only 72% of the labelled
rows (503,739 of 699,635) for the 5-fold primary scheme, not the 80% the outer-fit block would
suggest: the inner early-stopping holdout removes a further 8 percentage points. The 10-fold scheme
recovers part of it (81%).

Projected through our own measured learning curve (AUC ~ a + b*n^(-1/5), +62.9e-5 per doubling):

    72% -> 80%   0.152 doublings   ~ +9.5e-5
    72% -> 90%   0.322 doublings   ~ +2.0e-4
    72% -> 100%  0.474 doublings   ~ +3.0e-4

That is far larger than anything else measured in this campaign, and it is grounded in our own data.

Why the fix is not trivial
--------------------------
The iteration count is currently chosen by early stopping on the inner holdout, so simply training
on all outer-fit rows would leave nothing to stop on. And the iteration count must NOT be derived
from the target fold's own validation labels -- that is exactly the leak the immutable fold protocol
exists to prevent.

Protocol (per outer fold k, fully leakage-free)
-----------------------------------------------
  1. Split the OUTER-FIT rows into `--inner-folds` inner folds.
  2. For each inner fold: train on the rest, early-stop on a 10% inner holdout carved from THAT
     training portion, record best_iteration. These models never see outer fold k at all.
  3. Take the MEDIAN best_iteration across inner folds (predeclared aggregator; not tuned).
  4. Retrain from scratch on 100% of the outer-fit rows with that FIXED iteration count. No early
     stopping, no held-out data of any kind.
  5. Score the outer fold exactly once.

The alternative of taking the median best_iteration from the OTHER outer folds is cheaper but not
clean: fold j's model was trained on outer-fit rows that include fold k's labels, so fold k's labels
would leak into the iteration scalar. The inner-CV route avoids that entirely at the cost of a few
extra fits.

The control is the current protocol, run in the same process on the same rows, so the comparison
isolates the training fraction.

Usage:
  python scripts/run_fullfit.py --folds 0,1 --inner-folds 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

# The exact champion single-model configuration: `z4_xt_f10_s4` in the prediction store is
# view=full / lgbm / learning_rate 0.02 / num_leaves 127 / extra_trees true. The remaining keys are
# the defaults in scripts/run_views.py::_fit_lgbm_es, so the control here is bit-for-bit the
# existing protocol rather than a lookalike.
CHAMPION = {"objective": "binary", "metric": "auc", "learning_rate": 0.02,
            "num_leaves": 127, "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
            "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "extra_trees": True,
            "verbose": -1, "n_jobs": 8}


def _logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_es(X, y, esX, esY, seed, rounds):
    """Train with early stopping on the inner holdout. Returns (model, best_iter)."""
    import lightgbm as lgb

    p = dict(CHAMPION)
    p.update(n_estimators=rounds, random_state=seed, bagging_seed=seed + 1,
             feature_fraction_seed=seed + 2)
    ds = lgb.Dataset(X, label=y)
    dv = lgb.Dataset(esX, label=esY, reference=ds)
    m = lgb.train(p, ds, num_boost_round=rounds, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    return m, int(m.best_iteration or rounds)

def fit_fixed(X, y, seed, n_iter):
    """Train on ALL given rows with a FIXED iteration count. No held-out data exists here."""
    import lightgbm as lgb

    p = dict(CHAMPION)
    p.update(n_estimators=int(n_iter), random_state=seed, bagging_seed=seed + 1,
             feature_fraction_seed=seed + 2)
    ds = lgb.Dataset(X, label=y)
    # no valid_sets and no early stopping: the iteration count is already fixed
    return lgb.train(p, ds, num_boost_round=int(n_iter))


def pick_iteration_leakage_free(Xfit, yfit, seed, inner_folds, rounds, frac=0.10):
    """Median best_iteration from an inner CV that never touches the outer validation fold.

    Also returns the training-set size each inner model actually saw, because the resulting
    iteration count is optimal for THAT size, not for the full outer-fit block.
    """
    from sklearn.model_selection import StratifiedKFold

    iters, sizes = [], []
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + 4242)
    for i, (a_rel, b_rel) in enumerate(skf.split(np.zeros(len(yfit)), yfit.astype("int8"))):
        a = np.sort(a_rel)                          # inner-train positions inside yfit
        # carve the early-stopping holdout OUT of the inner-train portion only. Both returned
        # index sets are positions in the SAME space as Xfit/yfit (0..len(yfit)-1), so they index
        # Xfit directly -- remapping them into `a` would silently read the wrong rows.
        tr_sub, es_sub = _inner_es_split(a, yfit, seed + 700 + i, frac=frac)
        assert len(tr_sub) + len(es_sub) == len(a) and not (set(tr_sub) & set(es_sub))
        m, it = fit_es(Xfit[tr_sub], yfit[tr_sub], Xfit[es_sub], yfit[es_sub], seed + i, rounds)
        iters.append(it)
        sizes.append(len(tr_sub))
    return int(np.median(iters)), iters, int(np.median(sizes))


# Predeclared size-correction exponent, from scripts/iteration_scaling.py: OLS of log(best_iter) on
# log(n) over the THREE largest learning-curve points (n = 251,869 / 377,803 / 503,739),
# exponent 0.7527, R^2(loglog) = 0.98371, and it reproduces the observed 504k count to within 2.3%.
SIZE_EXPONENT = 0.7527


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--scheme", default="primary")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--tag", default="fullfit")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    y = y_int.astype("float64")
    n_total = len(y)
    folds = get_scheme(args.scheme, y_int, tr[ID_COL]).folds

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()

    fin_all = store.load_oof("blend_v3_final").astype("float64")
    results = []

    for k in [int(x) for x in args.folds.split(",")]:
        print(f"\n{'='*96}\nfold {k}   scheme={args.scheme}\n{'='*96}", flush=True)
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        Xf, Xout, names = vb.assemble(fit, y_int, val, None, inner_seed=k)
        Xv = Xout["val"]
        set_seed(args.seed + k)

        # ---------------- CONTROL: current protocol (inner ES, 72% of all labels) -------------
        inner_tr, inner_es = _inner_es_split(fit, y_int, args.seed + k)
        pos = {int(v): i for i, v in enumerate(fit)}
        tr_local = np.array([pos[int(v)] for v in inner_tr])
        es_local = np.array([pos[int(v)] for v in inner_es])
        t0 = time.time()
        m_ctl, it_ctl = fit_es(Xf[tr_local], y[inner_tr], Xf[es_local], y[inner_es],
                               args.seed + k, args.rounds)
        p_ctl = 1.0 / (1.0 + np.exp(-np.clip(m_ctl.predict(Xv, num_iteration=it_ctl), -35, 35)))
        auc_ctl = float(roc_auc_score(y[val], p_ctl))
        t_ctl = time.time() - t0
        print(f"  CONTROL  (inner-ES, {len(tr_local):,}/{n_total:,} = {len(tr_local)/n_total:.2%} "
              f"of labels)  AUC={auc_ctl:.6f} iter={it_ctl} ({t_ctl:.0f}s)", flush=True)

        # ---------------- FULL-FIT: leakage-safe fixed iteration on 100% of outer-fit ---------
        t0 = time.time()
        n_inner, inner_iters, n_inner_arm = pick_iteration_leakage_free(
            Xf, y[fit], args.seed + k, args.inner_folds, args.rounds)
        t_pick = time.time() - t0
        n_final = len(fit)

        # GUARD. A silently wrong index space in the iteration picker produced a median of 5
        # rounds and a fake -483e-5 "rejection" before this check existed. An inner-CV median that
        # differs from the control's own early-stopping count by more than 3x is a harness fault,
        # not a finding, so it aborts instead of being reported.
        ratio = n_inner / max(it_ctl, 1)
        if not (1 / 3.0 <= ratio <= 3.0):
            raise RuntimeError(
                f"fold {k}: inner-CV median best_iter={n_inner} vs control ES best_iter={it_ctl} "
                f"(ratio {ratio:.3f}). That is outside [1/3, 3], which cannot happen for a correct "
                f"iteration picker on this data -- treat it as a harness bug, not a result. "
                f"inner iters={inner_iters}")
        print(f"  iteration sources (neither ever sees fold {k}):")
        print(f"    inner-CV median = {n_inner} from {inner_iters}, each measured at "
              f"~{n_inner_arm:,} rows ({n_inner_arm/n_final:.0%} of the final fit)  ({t_pick:.0f}s)")
        print(f"    control ES best_iter = {it_ctl}, measured at {len(tr_local):,} rows "
              f"({len(tr_local)/n_final:.0%} of the final fit)")

        arms = [
            # name,            iteration count,                  provenance
            ("innercv_raw",    n_inner,
             "median inner-CV best_iteration, used WITHOUT a size correction"),
            ("innercv_scaled", int(round(n_inner * (n_final / n_inner_arm) ** SIZE_EXPONENT)),
             f"median inner-CV best_iteration x (n_final/n_inner_arm)^{SIZE_EXPONENT}"),
            ("ctl_scaled",     int(round(it_ctl * (n_final / len(tr_local)) ** SIZE_EXPONENT)),
             f"the control's own ES best_iteration x (n_final/n_ctrl)^{SIZE_EXPONENT}"),
        ]

        arm_res = []
        for name, n_iter, prov in arms:
            t0 = time.time()
            m_full = fit_fixed(Xf, y[fit], args.seed + k, n_iter)
            p_full = 1.0 / (1.0 + np.exp(-np.clip(m_full.predict(Xv), -35, 35)))
            auc_full = float(roc_auc_score(y[val], p_full))
            t_full = time.time() - t0
            delta = auc_full - auc_ctl
            bg_w = {}
            base_auc_v = float(roc_auc_score(y[val], fin_all[val]))
            for w in (0.05, 0.10, 0.20, 0.30, 0.50, 1.0):
                mix = w * _logit(p_full) + (1 - w) * _logit(fin_all[val])
                bg_w[str(w)] = float(roc_auc_score(y[val], mix)) - base_auc_v
            print(f"    {name:<14} iter={n_iter:>5}  AUC={auc_full:.6f}  "
                  f"delta={delta*1e5:+7.1f}e-5  ({t_full:.0f}s)")
            print(f"      {prov}")
            print(f"      corr(full,control) logit={corr(_logit(p_full), _logit(p_ctl)):.5f} "
                  f"spearman={spearman(p_full, p_ctl):.5f} | "
                  f"corr(full,finalist) logit={corr(_logit(p_full), _logit(fin_all[val])):.5f} "
                  f"spearman={spearman(p_full, fin_all[val]):.5f}")
            print(f"      finalist blend gains: "
                  f"{', '.join(f'w={w}:{g*1e5:+.1f}' for w, g in bg_w.items())}")
            arm_res.append({
                "arm": name, "iteration": int(n_iter), "provenance": prov,
                "auc": auc_full, "delta": delta, "delta_e5": delta * 1e5,
                "iter_ratio_vs_control": n_iter / max(it_ctl, 1), "seconds": round(t_full, 1),
                "corr_with_control_logit": corr(_logit(p_full), _logit(p_ctl)),
                "corr_with_finalist_logit": corr(_logit(p_full), _logit(fin_all[val])),
                "spearman_with_finalist": spearman(p_full, fin_all[val]),
                "finalist_blend_gains": bg_w,
            })
            np.save(REPORTS / f"{args.tag}_{name}_fold{k}.npy", p_full.astype("float32"))

        results.append({
            "fold": k, "scheme": args.scheme, "view": args.view, "n_features": len(names),
            "n_total_labelled": n_total,
            "control": {"n_rows": int(len(tr_local)), "frac_of_all_labels": len(tr_local) / n_total,
                        "auc": auc_ctl, "iter": it_ctl, "seconds": round(t_ctl, 1)},
            "full_fit_common": {"n_rows": int(n_final),
                                "frac_of_all_labels": n_final / n_total,
                                "iteration_pick_seconds": round(t_pick, 1),
                                "inner_cv_iters": inner_iters,
                                "inner_cv_median": int(n_inner),
                                "inner_cv_rows_per_arm": int(n_inner_arm)},
            "size_exponent": SIZE_EXPONENT,
            "arms": arm_res,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip()[:12],
        })
        np.save(REPORTS / f"{args.tag}_control_fold{k}.npy", p_ctl.astype("float32"))

    print("\n" + "=" * 96)
    hdr = results[0]["control"]
    print(f"  control rows={hdr['n_rows']:,} ({hdr['frac_of_all_labels']:.1%} of labels)   "
          f"full-fit rows={results[0]['full_fit_common']['n_rows']:,} "
          f"({results[0]['full_fit_common']['frac_of_all_labels']:.1%} of labels)")
    print(f"{'arm':<16}{'fold':>6}{'iter':>7}{'AUC':>12}{'delta_e5':>11}{'sec':>7}")
    for r in results:
        for a in r["arms"]:
            print(f"{a['arm']:<16}{r['fold']:>6}{a['iteration']:>7}{a['auc']:>12.6f}"
                  f"{a['delta_e5']:>+11.1f}{a['seconds']:>7.0f}")

    print("\n=== per-arm summary ===")
    summary = {}
    for a0 in results[0]["arms"]:
        name = a0["arm"]
        ds = [r["arms"][[a["arm"] for a in r["arms"]].index(name)]["delta"] for r in results]
        m_d = float(np.mean(ds))
        pos = sum(1 for d in ds if d > 0)
        summary[name] = {"mean_delta": m_d, "mean_delta_e5": m_d * 1e5,
                         "folds_positive": pos, "n_folds": len(ds),
                         "per_fold_delta_e5": [d * 1e5 for d in ds]}
        print(f"  {name:<16} mean {m_d*1e5:+7.1f}e-5   positive {pos}/{len(ds)}   "
              f"per-fold {' '.join(f'{d*1e5:+.1f}' for d in ds)}")

    print("\n" + "=" * 96)
    n = len(results)
    any_promote = False
    for name, s in summary.items():
        if s["folds_positive"] == n and s["mean_delta"] >= 5e-5:
            v = "PROMOTE to all folds"
            any_promote = True
        elif s["folds_positive"] == 0:
            v = "REJECT (no fold improves)"
        elif s["folds_positive"] != n:
            v = "SIGNS FLIP -- diagnose before escalating"
        else:
            v = "all positive but mean < +5e-5 -- weak, hold"
        print(f"  {name:<16} {v}")
    print(f"\n  headline: mean delta over {n} fold(s), control AUC "
          f"{np.mean([r['control']['auc'] for r in results]):.6f}")

    save_json({"results": results, "arm_summary": summary,
               "any_arm_promotes": any_promote, "n_folds": n,
               "size_exponent": SIZE_EXPONENT,
               "protocol": ("per outer fold: the iteration count comes ONLY from information "
                            "inside the outer-FIT rows -- (a) a "
                            f"{args.inner_folds}-fold inner CV, each inner model early-stopping on "
                            "a 10% holdout carved from its own inner-train portion, or (b) the "
                            "control's own inner-ES run, which also lies entirely inside outer-fit. "
                            "Either way the outer fold's labels never influence it. Then a "
                            "from-scratch fit on 100% of the outer-fit rows at that fixed count "
                            "with no early stopping, and the outer fold is scored exactly once."),
               "arms_differ_only_in": ("the ESTIMATOR of the iteration count. All arms train on "
                                       "the same 100% of outer-fit rows, so agreement between them "
                                       "is evidence about the estimator, not a search over it."),
               "size_correction_source": "scripts/iteration_scaling.py, exponent 0.7527",
               "control": "the existing inner-ES protocol, same process, rows, params and seed"},
              REPORTS / f"{args.tag}_{args.scheme}.json")
    print("wrote", REPORTS / f"{args.tag}_{args.scheme}.json")


if __name__ == "__main__":
    main()