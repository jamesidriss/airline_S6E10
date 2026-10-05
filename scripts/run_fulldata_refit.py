"""Full-data (100% of labelled rows) test-time refit -- the biggest unexploited lever found.

The gap this closes
-------------------
`scripts/run_views.py` builds every test prediction as the average over K fold models, each fitted
on that fold's OUTER-FIT subset: 559,708 rows (80.00% of the labels) at K=5, 629,672 (90.00%) at
K=10. **No path in this repository trains a test model on 100% of the labels.** So 20% (or 10%) of
the real labels are never shown to any member that votes on the test set, although we already hold
them. The fold protocol exists to make OOF *meaningful*; test-time inference needs no OOF, so once
every model-selection decision is frozen there is no reason a test model must exclude any row.

Evidence that this is worth doing, and by how much
--------------------------------------------------
1. `scripts/iteration_scaling.py` fits best_iter vs training size on the fold-0 subsample curve.
2. `scripts/validate_curve_on_folds.py` shows that law predicts the ALREADY-MEASURED 5->10 fold
   gain a priori, from training fraction alone: predicted +10.7e-5, observed +9.9e-5, a 7.4%
   relative error whose residual (-0.8e-5) is smaller than the seed-to-seed spread within either
   scheme (4.4e-5 and 6.4e-5). That is out-of-sample predictive power on an axis the law was not
   fitted to.
3. Extrapolating it, 80%->100% predicts **+20.2e-5** for a single learner.

BUT the same law OVER-states gains at the top of the range: `scripts/run_fullfit.py` measured
72%->80% at **+4.0e-5**, not the +9.6e-5 the law predicts -- about 40% of the predicted value. So
+20.2e-5 is treated as an optimistic bound, not a point estimate. Recording this honestly is the
whole point; a full-data member that gains nothing is still worth having, but it must not be sold
as a +2e-4 win.

Why the iteration count is an argument, not a detail
---------------------------------------------------
Fold 0 showed AUC is nearly flat in iteration count over 811..863 (0.961338/0.961339) and clearly
degraded by aggressive correction at 1191 (0.961240, i.e. -5.9e-5 against the control). So the
danger of a fixed iteration count is OVER-iteration, not under-iteration, and the remedy is to
shorten the extrapolation rather than to pick a cleverer exponent.

PREDECLARED ITERATION POLICY (fixed before any full-data output existed)
-------------------------------------------------------------------------
Measure the count where it is cheap to measure honestly and as close to the target size as possible,
then extrapolate the short remaining distance:

  1. carve a stratified 5% holdout H from all 699,635 labelled rows
  2. fit on the other 95% (664,653 rows) with early stopping on H  -> best_iter
  3. correct for the short remaining gap:  best_iter x (1.00/0.95)^0.7527 = best_iter x 1.0387
  4. fit the FINAL model on 100% of rows at that fixed count, no early stopping
  5. average over seeds in logit space

Step 3 is a factor of 1.039. The alternative -- taking the 5-fold OOF control's count (measured at
72% of labels) and correcting by (1.00/0.72)^0.7527 = 1.281 -- is rejected in advance because it
extrapolates over a 1.39x range instead of a 1.05x range, and the fold-0 evidence is that
distant-range corrections over-iterate. The exponent itself is measured
(`scripts/iteration_scaling.py`, R^2 = 0.98371, reproduces the observed 504k count to 2.3%), not
invented; at 1.039x the policy is almost insensitive to which exponent is used.

No OOF, by construction
-----------------------
A model trained on every labelled row cannot be scored on labels it has seen. These members are
therefore written to `artifacts/fulldata/` with `"oof": null` and an explicit note, NOT through
`src.submission.store.save` (which would require inventing an OOF array). They must never be mixed
into blend-weight fitting or admitted on a fabricated OOF. Their evidence is the cross-validated
training-policy gain, quoted as such.

Usage:
  python scripts/run_fulldata_refit.py --configs xt_l127,det_l127 --seeds 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402

OUT = Path("artifacts") / "fulldata"

# Predeclared compact config set. Chosen for genuine structural diversity, not to maximise member
# count: the campaign's own lesson is that near-clone members buy ~1e-6..5e-6 against a +1.5e-5
# admission gate, whereas extra_trees only pays on a rich view.
LGBM_BASE = {"objective": "binary", "metric": "auc", "learning_rate": 0.02, "num_leaves": 127,
             "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
             "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "verbose": -1, "n_jobs": 8}
CONFIGS = {
    "xt_l127":  {"family": "lgbm", "view": "full", "params": dict(LGBM_BASE, extra_trees=True)},
    "xt_l63":   {"family": "lgbm", "view": "full",
                 "params": dict(LGBM_BASE, extra_trees=True, num_leaves=63, learning_rate=0.03)},
    "det_l127": {"family": "lgbm", "view": "full", "params": dict(LGBM_BASE)},
}
# Which existing store member each config should be compared against, as a diagnostic only.
REFERENCE = {"xt_l127": "z4_xt_f10_s4", "xt_l63": "xt_xt_d63_lr015", "det_l127": "xt_xt_d127_s2"}

SIZE_EXPONENT = 0.7527      # scripts/iteration_scaling.py
ES_FRACTION = 0.05          # holdout used ONLY to measure the iteration count
MAX_ROUNDS = 4000           # cap for the measurement fit


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(p):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(p, dtype="float64"), -35, 35)))


def es_holdout(y, frac, seed):
    """Stratified holdout carved from ALL rows, used ONLY to measure the iteration count."""
    rng = np.random.default_rng(seed + 7717)
    idx = np.arange(len(y))
    pos, neg = idx[y == 1], idx[y == 0]
    n = int(len(y) * frac)
    h = np.unique(np.concatenate([rng.choice(pos, int(n * len(pos) / len(y)), replace=False),
                                  rng.choice(neg, int(n * len(neg) / len(y)), replace=False)]))
    return np.setdiff1d(idx, h), h


def measure_iterations(Xf, y, params, seed, esX, esY, fit_rows, max_rounds=MAX_ROUNDS):
    """Early-stop on a holdout INSIDE the training data. Honest, and only 5% away from full size."""
    import lightgbm as lgb
    p = dict(params)
    p.update(n_estimators=max_rounds, random_state=seed, bagging_seed=seed + 1,
             feature_fraction_seed=seed + 2)
    ds = lgb.Dataset(Xf[fit_rows], label=y[fit_rows].astype("float64"))
    dv = lgb.Dataset(Xf[esX], label=esY.astype("float64"), reference=ds)
    m = lgb.train(p, ds, num_boost_round=max_rounds, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    return int(m.best_iteration or max_rounds)


def build_full(view, tr, te, y):
    """Feature matrix over ALL labelled rows, plus the test matrix, from one fold-safe assembly.

    `assemble` needs a non-empty val index; a 1000-row slice of train is passed purely to satisfy
    that, and its output is discarded. The fold-safe target-encoding blocks are cross-fitted INSIDE
    the fit index, which here is every labelled row -- legal, because the test set's labels are not
    involved at any point.
    """
    vb = ViewBuilder(tr, te, view)
    vb.build_static()
    all_rows = np.arange(len(tr))
    dummy_val = np.arange(1000)
    test_rows = np.arange(len(tr), len(tr) + len(te))
    Xf, Xa, names = vb.assemble(all_rows, y, dummy_val, test_rows, inner_seed=0)
    return Xf, Xa["test"], names, vb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="xt_l127")
    ap.add_argument("--seeds", default="4")
    ap.add_argument("--es-frac", type=float, default=ES_FRACTION)
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    n_all = len(y)
    OUT.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    names_out = [c for c in args.configs.split(",") if c]

    cache = {}
    manifest = {
        "created": "2026-10-05",
        "n_train_rows": int(n_all),
        "train_fraction_of_all_labels": 1.0,
        "oof": None,
        "why_no_oof": ("a model trained on every labelled row cannot be scored on labels it has "
                       "seen; these members are evidence-qualified as a cross-validated "
                       "training-policy gain and MUST NOT be admitted to a blend on a fabricated "
                       "OOF, nor used to fit blend weights."),
        "iteration_policy": {
            "step_1": f"stratified {args.es_frac:.0%} holdout from all rows, used ONLY to measure "
                      f"the iteration count (its labels ARE competition labels, which is legal: "
                      f"nothing outside the training set is consulted)",
            "step_2": "early-stopped fit on the remaining rows",
            "step_3": f"multiply by (1.00/{1-args.es_frac:.2f})^{SIZE_EXPONENT} to cover the "
                      f"short remaining gap",
            "step_4": "final fit on 100% of rows at that fixed count, no early stopping",
            "size_exponent": SIZE_EXPONENT,
            "correction_factor": (1.0 / (1 - args.es_frac)) ** SIZE_EXPONENT,
            "why_not_the_72pct_count": ("taking the 5-fold control count and correcting by 1.281 "
                                        "extrapolates over a 1.39x range instead of 1.05x; fold 0 "
                                        "showed distant-range corrections over-iterate and cost "
                                        "-5.9e-5"),
        },
        "prediction_geometry": "equal-weight average in LOGIT space (matches blend convention)",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()[:12],
        "members": [],
    }

    for cname in names_out:
        cfg = CONFIGS[cname]
        if cfg["view"] not in cache:
            cache[cfg["view"]] = build_full(cfg["view"], tr, te, y)
        Xf, Xt, fnames, _vb = cache[cfg["view"]]
        print(f"\n  [{cname}] full-data fit matrix {Xf.shape}, test matrix {Xt.shape}, "
              f"{len(fnames)} features", flush=True)

        # ---- predeclared iteration policy: measure at 95%, extrapolate 1.04x, fit at 100% ----
        fit_rows, es_rows = es_holdout(y, args.es_frac, seeds[0])
        t0 = time.time()
        n_meas = measure_iterations(Xf, y, cfg["params"], seeds[0], es_rows, y[es_rows], fit_rows)
        n_iter = int(round(n_meas * (1.0 / (1 - args.es_frac)) ** SIZE_EXPONENT))
        print(f"    iteration: measured {n_meas} at {len(fit_rows):,} rows "
              f"({len(fit_rows)/n_all:.1%}), corrected x"
              f"{(1.0/(1-args.es_frac))**SIZE_EXPONENT:.4f} -> {n_iter} for {n_all:,} rows "
              f"({time.time()-t0:.0f}s)", flush=True)

        per_seed = []
        for sd in seeds:
            t0 = time.time()
            set_seed(sd)
            if cfg["family"] == "lgbm":
                import lightgbm as lgb
                p = dict(cfg["params"])
                p.update(n_estimators=int(n_iter), random_state=sd, bagging_seed=sd + 1,
                         feature_fraction_seed=sd + 2)
                ds = lgb.Dataset(Xf, label=y.astype("float64"))
                # no valid_sets and no early stopping: the iteration count is already fixed
                m = lgb.train(p, ds, num_boost_round=int(n_iter))
                pr = m.predict(Xt)
            else:  # pragma: no cover - only lgbm is wired up in this first pass
                raise NotImplementedError(cfg["family"])
            per_seed.append(sig(pr))
            print(f"    {cname} seed={sd} iters={n_iter} done in {time.time()-t0:.0f}s", flush=True)

        Z = np.mean([logit(p) for p in per_seed], axis=0)
        pred = sig(Z)
        path = OUT / f"{cname}_s{'-'.join(map(str, seeds))}_it{n_iter}_test.npy"
        np.save(path, pred.astype("float32"))

        entry = {
            "config": cname, "family": cfg["family"], "view": cfg["view"],
            "n_train_rows": int(n_all), "train_fraction": 1.0, "n_features": int(Xf.shape[1]),
            "iterations": int(n_iter), "iterations_measured": int(n_meas),
            "iterations_provenance": manifest["iteration_policy"],
            "seeds": seeds, "n_seed_models": len(per_seed),
            "params": cfg["params"], "path": str(path),
            "seed_spread_spearman": {
                str(seeds[i]): spearman(per_seed[i], per_seed[0])
                for i in range(1, len(per_seed))} if len(per_seed) > 1 else {},
            "diagnostic_vs_existing_member": None,
        }
        ref = REFERENCE.get(cname)
        if ref:
            try:
                r = store.load_test(ref)
                entry["diagnostic_vs_existing_member"] = {
                    "member": ref,
                    "spearman": spearman(pred, r),
                    "logit_corr": corr(logit(pred), logit(r)),
                    "note": ("how much the 100% refit actually moves the ranking versus the "
                             "fold-averaged 80%/90% member it replaces. High values mean the "
                             "change is cosmetic; low values mean it is substantive."),
                }
                print(f"    diagnostic vs {ref}: spearman="
                      f"{entry['diagnostic_vs_existing_member']['spearman']:.6f} "
                      f"logit_corr={entry['diagnostic_vs_existing_member']['logit_corr']:.6f}")
            except Exception as exc:  # noqa: BLE001
                entry["diagnostic_vs_existing_member"] = {"member": ref, "error": str(exc)[:120]}
        manifest["members"].append(entry)

    manifest["configs"] = names_out
    save_json(manifest, OUT / "manifest.json")
    print("\nwrote", OUT / "manifest.json")
    for m in manifest["members"]:
        print(f"  {m['config']:<12} rows={m['n_train_rows']:,} "
              f"iters={m['iterations']} (measured {m['iterations_measured']}) "
              f"seeds={m['seeds']} -> {m['path']}")


if __name__ == "__main__":
    main()
