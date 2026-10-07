"""Full-data (100% of labelled rows) refit of the v3_final ensemble members.

Why replace member-by-member rather than build a new ensemble
------------------------------------------------------------
Each v3_final member's stored test prediction is the average over K fold models, each fitted on that
fold's OUTER-FIT subset -- 559,708 rows (80.00% of labels) at K=5, 629,672 (90.00%) at K=10. No path
in this repository trains a test model on 100% of the labels, so 20%/10% of the real labels are
never shown to any member that votes on the test set, although we already hold them.

Substituting member i's prediction with a 100%-data fit of the IDENTICAL config and seed keeps the
blend's structure, weights and member identity completely intact while strictly increasing the
information behind every member. The alternative -- a new ensemble -- would change the weights, and
the weights are the one thing fitted on OOF and therefore the one thing we must not disturb.

The trade-off, decided rather than dodged
-----------------------------------------
Replacing a K-fold average with a single model trades variance reduction for data:

  data      80% -> 100% is 0.322 doublings. The measured 72%->80% step (0.152 doublings) delivered
            +2.2e-5 mean / +4.0e-5 best fold, so scaling gives roughly +4.7e-5 to +8.5e-5.
  averaging a K=5 fold average suppresses seed noise whose OOF spread we measured at 4.4e-5 (shadow)
            and 6.4e-5 (block10). Dropping to one model returns about +2e-5 of that noise.

Net is clearly positive for a single model, and MORE so inside this blend: 59 members averaged in logit
space already absorb per-member seed noise, so the averaging term is largely already banked while the
data term is not.

Iteration policy, predeclared -- and then corrected downward on new evidence
------------------------------------------------------------------------
Over-iteration is the failure mode, not under-iteration, and it is worse than a single bad draw
suggests. On fold 0, 811 and 863 rounds were indistinguishable while 1191 rounds cost -5.9e-5
against the control. The bagging experiment then showed the argmax is genuinely noisy: changing only
`subsample` from 0.8 to 0.9 moved the early-stopped count from 797 to 860 on fold 0 (harmless) but
from 731 to 1082 on fold 1, which cost **-18.8e-5**.

So the count is stabilised by repetition and measured as close to the final size as possible:

  1. three independent stratified 5% holdouts from all rows, used ONLY to measure the count
  2. early-stopped fits on the other 95% -> three best_iter values
  3. take the MEDIAN of the three
  4. final fit on 100% of rows at that fixed count, no early stopping

A CORRECTION WAS DROPPED, deliberately
--------------------------------------
An earlier draft multiplied the median by (1.00/0.95)^0.7527 = 1.0387, on the theory that best_iter
grows with training size. That theory came from `scripts/iteration_scaling.py`, which fitted
best_iter ~ n^0.7527 to the fold-0 SUBSAMPLE curve. The direct measurement at 95% of the labels then
contradicted its sign:

    training rows        ES argmax                     source
    503,739  (72%)       797, 731                      fold-0 / fold-1 controls, 10% holdout
    664,655  (95%)       [613, 846, 641]               full-data measurement, three 5% holdouts
    503,739  (72%)       879 mean over 3 seeds         fold-0 subsample curve, frac = 1.0

The most recent and most direct measurement at the largest size gives a LOWER count, not a higher
one. With `extra_trees` plus `colsample_bytree=0.8` and `subsample=0.8` each tree already sees ~64%
of the columns and rows, so more data makes split statistics less noisy and the validation curve
flatter -- which both lowers the optimum and makes the argmax less well determined. The 38% spread
across three holdouts at the same size is the flatness showing up directly.

So no upward correction is applied. The median of a measurement taken at 95% of the target size is
used as-is: it involves essentially no extrapolation, and the residual under-iteration it risks is the
safe direction. The factor actually matters little -- 666 versus ~700 rounds is a 5% difference, and
fold 0 measured 811 versus 863 (a 6% difference) as worth exactly 0.0e-5.

No OOF, by construction
-----------------------
A model trained on every labelled row cannot be scored on labels it has seen. These predictions go to
`artifacts/fulldata/` with `"oof": null` and must NEVER be blend-weight-fitted or admitted on a
fabricated OOF. Their evidence is the cross-validated training-policy gain, quoted as such.

Usage:
  python scripts/refit_members_fulldata.py --families lgbm --seeds-per-config 1
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
MANIFEST = REPORTS / "finalist_v3_final.json"

# EXACT defaults of scripts/run_views.py::_fit_full_predict_lgbm, so a full-data fit is the same
# configuration as the member it replaces with more rows and nothing else changed.
TEST_REFIT_DEFAULTS = {"objective": "binary", "n_estimators": 1200, "learning_rate": 0.02,
                       "num_leaves": 127, "min_child_samples": 40, "reg_lambda": 1.0,
                       "max_bin": 255, "colsample_bytree": 0.8, "subsample": 0.8,
                       "subsample_freq": 1, "verbose": -1, "n_jobs": 8}

SIZE_EXPONENT = 0.7527
ES_FRACTION = 0.05
MAX_ROUNDS = 4000
N_ITER_SPLITS = 3           # independent 5pct holdouts used to stabilise the iteration count


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(p):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(p, dtype="float64"), -35, 35)))


def es_holdout(y, frac, seed):
    rng = np.random.default_rng(seed + 7717)
    idx = np.arange(len(y))
    pos, neg = idx[y == 1], idx[y == 0]
    n = int(len(y) * frac)
    h = np.unique(np.concatenate([rng.choice(pos, int(n * len(pos) / len(y)), replace=False),
                                  rng.choice(neg, int(n * len(neg) / len(y)), replace=False)]))
    return np.setdiff1d(idx, h), h


def lgbm_params(member_params, seed, metric=None):
    """EXACT defaults of scripts/run_views.py::_fit_full_predict_lgbm, so a full-data fit is the
    same configuration as the member it replaces with more rows and nothing else changed.

    `metric` matters and is easy to get wrong: LightGBM early-stops on the FIRST metric in the
    parameter dict, and its default for a binary objective is binary_logloss. The iteration count
    must therefore be selected on AUC, which is what the whole campaign measures. `metric="auc"` is
    passed ONLY for the measurement fits, where early stopping actually happens; the final
    fixed-count fit has no valid set and so has nothing to stop on.
    """
    p = dict(TEST_REFIT_DEFAULTS)
    p.update({k: v for k, v in (member_params or {}).items() if k != "n_estimators"})
    p.update(random_state=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    p.pop("metric", None)
    if metric:
        p["metric"] = metric
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="lgbm")
    ap.add_argument("--views", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--es-frac", type=float, default=ES_FRACTION)
    args = ap.parse_args()

    import lightgbm as lgb

    tr, te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    y = y_int.astype("float64")
    n_all = len(y)
    OUT.mkdir(parents=True, exist_ok=True)

    final = json.loads(MANIFEST.read_text(encoding="utf-8"))
    members = final["members"]
    fams = {f for f in args.families.split(",") if f}
    if args.views:
        keep = {v for v in args.views.split(",") if v}
        members = [m for m in members if m.get("featureset") in keep]
    members = [m for m in members if m["family"] in fams]
    if args.limit:
        members = members[:args.limit]
    # Process in view order so only ONE assembled design matrix is resident at a time. Four views at
    # ~800 MB each plus LightGBM's internal state does not fit comfortably alongside anything else.
    members = sorted(members, key=lambda m: str(m.get("featureset")))

    print(f"  refitting {len(members)} members from {final['name']} on {n_all:,} rows "
          f"(100% of labels)\n")

    vcache, fit_rows, es_rows = {}, None, None
    iter_cache = {}
    out = {"source_finalist": final["name"], "n_train_rows": int(n_all),
           "train_fraction_of_all_labels": 1.0, "oof": None,
           "why_no_oof": ("trained on every labelled row, so it cannot be scored on labels it has "
                          "seen; reported as a cross-validated training-policy gain, never as OOF"),
           "iteration_policy": {
               "measure": f"early stop on {N_ITER_SPLITS} independent {args.es_frac:.0%} "
                          f"stratified holdouts, fit on the rest",
               "aggregate": f"MEDIAN of the {N_ITER_SPLITS} counts",
               "size_correction": ("NONE. The direct 95%-of-labels measurement returns a LOWER count "
                                   "than the 72% one, so the upward trend fitted to the fold-0 "
                                   "subsample curve does not hold in this regime. The median of a "
                                   "measurement taken at 95% of the target size is used as-is, which "
                                   "involves essentially no extrapolation."),
               "final": "fit on 100% of rows at that fixed count, no early stopping",
               "superseded_size_exponent": SIZE_EXPONENT,
               "why": ("over-iteration is the measured failure mode (1191 rounds cost -5.9e-5, and a "
                       "subsample change alone moved the argmax 731 -> 1082), so a single argmax is "
                       "not trusted and no upward extrapolation is applied. Under-iteration is the "
                       "safe residual."),
           },
           "test_refit_defaults": TEST_REFIT_DEFAULTS,
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()[:12],
           "members": []}

    fin_test = store.load_test("blend_v3_final")

    for mi, m in enumerate(members):
        view = m.get("featureset")
        if view not in vcache:
            vcache.clear()          # one view resident at a time; members arrive in view order
            vb = ViewBuilder(tr, te, view)
            vb.build_static()
            # fit_idx = EVERY labelled row; val_idx is a throwaway slice that only exists because
            # assemble() requires one. The fold-safe TE blocks are cross-fitted inside fit_idx,
            # which is legal: the test set's labels are never involved.
            Xf, Xa, fnames = vb.assemble(np.arange(n_all), y_int, np.arange(500),
                                         np.arange(n_all, n_all + len(te)), inner_seed=0)
            vcache[view] = (Xf, Xa["test"], len(fnames))
            print(f"  [view {view}] fit {Xf.shape}  test {Xa['test'].shape}  "
                  f"{len(fnames)} features", flush=True)
        Xf, Xt, nfeat = vcache[view]

        if fit_rows is None:
            fit_rows, es_rows = es_holdout(y_int, args.es_frac, 0)
        cfg_key = (view, json.dumps(m.get("params"), sort_keys=True))
        if cfg_key not in iter_cache:
            t0 = time.time()
            # THREE independent 5% holdouts. A single one cannot reliably locate a flat optimum:
            # the bagging experiment saw the early-stopped count jump 731 -> 1082 on fold 1 purely
            # from `subsample` 0.8 -> 0.9, and the 1082 fit cost -18.8e-5. The median discards
            # exactly that kind of wild draw.
            meas_list = []
            for sp in range(N_ITER_SPLITS):
                fr_, er_ = es_holdout(y_int, args.es_frac, 1000 * (sp + 1))
                p = lgbm_params(m.get("params"), 0, metric="auc")
                p["n_estimators"] = MAX_ROUNDS
                ds = lgb.Dataset(Xf[fr_], label=y[fr_])
                dv = lgb.Dataset(Xf[er_], label=y[er_], reference=ds)
                mm = lgb.train(p, ds, num_boost_round=MAX_ROUNDS, valid_sets=[dv],
                               callbacks=[lgb.early_stopping(300, verbose=False)])
                meas_list.append(int(mm.best_iteration or MAX_ROUNDS))
            med = int(np.median(meas_list))
            # NO size correction. The direct 95%-of-labels measurement returns a LOWER count than the
            # 72% one (see the module docstring), so the upward trend fitted to the fold-0 subsample
            # curve does not hold in this regime. Over-iteration is the measured failure mode and the
            # residual under-iteration is the safe direction. 666 vs ~700 rounds is a 5% difference
            # and fold 0 measured 811 vs 863 (6% apart) as worth exactly 0.0e-5.
            iter_cache[cfg_key] = (meas_list, med, med)
            print(f"    iteration for {view} {json.dumps(m.get('params'))[:60]}: "
                  f"{meas_list} at {len(fit_rows):,} rows -> median {med} used for {n_all:,} "
                  f"(no upward correction) ({time.time()-t0:.0f}s)", flush=True)
        meas_list, med, n_iter = iter_cache[cfg_key]

        seed = int(m.get("seed", 1) or 1)
        t0 = time.time()
        set_seed(seed)
        p = lgbm_params(m.get("params"), seed)
        p["n_estimators"] = n_iter
        ds = lgb.Dataset(Xf, label=y)
        mdl = lgb.train(p, ds, num_boost_round=n_iter)     # no valid_sets: count already fixed
        pred = sig(mdl.predict(Xt))
        dt = time.time() - t0

        path = OUT / f"fd_{m['exp_id']}_it{n_iter}_test.npy"
        np.save(path, pred.astype("float32"))
        old = store.load_test(m["exp_id"])
        entry = {"exp_id": m["exp_id"], "family": m["family"], "view": view,
                 "seed": seed, "params": m.get("params"), "n_features": int(nfeat),
                 "n_train_rows": int(n_all), "train_fraction": 1.0,
                 "iterations": n_iter, "iterations_measured_at_95pct": med, "iterations_measured_splits": meas_list,
                 "path": str(path), "seconds": round(dt, 1),
                 "vs_member_spearman": spearman(pred, old),
                 "vs_member_logit_corr": corr(logit(pred), logit(old)),
                 "vs_finalist_spearman": spearman(pred, fin_test)}
        out["members"].append(entry)
        print(f"  [{mi+1}/{len(members)}] {m['exp_id']:<26} iter={n_iter:>5} "
              f"spearman(member)={entry['vs_member_spearman']:.5f} ({dt:.0f}s)", flush=True)

    # ---- build the substituted finalist and pre-flight it ----
    #
    # The blend must be over ALL finalist members, not only the ones this run refitted. An earlier
    # version iterated the FILTERED list, so with --families lgbm it averaged just those 32 members
    # and reported a candidate that was really a 32-member blend rather than the v3 geometry with 32
    # substitutions. It under-reported the correlation to v3 (0.998672 instead of 0.999262) and, worse,
    # mislabelled it. scripts/make_submission_v4.py is the authoritative builder; this block is a
    # convenience summary and must use `final["members"]`.
    n_sub = 0
    for m in final["members"]:
        e = next((x for x in out["members"] if x["exp_id"] == m["exp_id"]), None)
        if e is None:
            continue
        e["recomputed_candidate_test_spearman"] = None
        n_sub += 1
    if n_sub:
        mix = np.zeros(len(te), dtype="float64")
        n_all_members = 0
        for m in final["members"]:
            e = next((x for x in out["members"] if x["exp_id"] == m["exp_id"]), None)
            p = (np.load(e["path"]) if e is not None else store.load_test(m["exp_id"]))
            mix += logit(np.asarray(p, dtype="float64"))
            n_all_members += 1
        cand = sig(mix / n_all_members)
        np.save(OUT / "candidate_v4_fulldata_test.npy", cand.astype("float32"))
        out["candidate"] = {"path": str(OUT / "candidate_v4_fulldata_test.npy"),
                            "n_members_total": n_all_members,
                            "n_members_refit_on_100pct": n_sub,
                            "spearman_vs_v3_final": spearman(cand, fin_test),
                            "logit_corr_vs_v3_final": corr(logit(cand), logit(fin_test)),
                            "note": ("built over ALL finalist members, substituting the full-data "
                                     "prediction where one exists. Members not refit keep their "
                                     "existing fold-averaged test prediction, so this is a partial "
                                     "substitution, not a uniform one. The weights are untouched -- "
                                     "they were fitted on OOF and must not be disturbed. The "
                                     "authoritative builder is scripts/make_submission_v4.py.")}
        print(f"\n  candidate: {n_sub}/{n_all_members} members refit on 100% of labels")
        print(f"    spearman vs v3_final = {out['candidate']['spearman_vs_v3_final']:.6f}")
        print(f"    logit corr vs v3_final = {out['candidate']['logit_corr_vs_v3_final']:.6f}")

    save_json(out, OUT / "members_fulldata.json")
    print("\nwrote", OUT / "members_fulldata.json")


if __name__ == "__main__":
    main()
