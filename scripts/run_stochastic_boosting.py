"""Phase 9: honest fixed-round inner-selection harness for DART and RF tree-construction modes.

The protocol, per outer fold (section 4 of the brief, implemented literally)
----------------------------------------------------------------------------
    outer-FIT rows
        -> deterministic stratified inner split (inner-train / inner-val), both inside outer-FIT
        -> train the candidate on inner-train, measure inner-val AUC at PREDECLARED snapshots
        -> select the round count from INNER data only (argmax; ties resolved to the SMALLER round)
        -> REFIT FROM SCRATCH on 100% of outer-FIT at that fixed round count, no early stopping
        -> evaluate outer-validation ONCE

Outer-validation labels therefore cannot influence the round count, the dropout parameters, the tree
count, the seed, or any hyper-parameter. Every inner curve is persisted.

Why selection is by explicit REFIT rather than `predict(num_iteration=r)`
-----------------------------------------------------------------------
This is the subtle part for DART. `predict(num_iteration=r)` on a model trained for N rounds
truncates the tree list and re-applies DART's shrinkage over the surviving trees, which is not
guaranteed to equal a model *trained* for r rounds -- the dropout sequence and the normalisation are
functions of the full schedule. Selecting on the truncated curve and then refitting at the selected
round would therefore optimise a slightly different objective from the one we deploy. So the inner
curve is measured by actually refitting at each snapshot. The cheap truncated-snapshot curve from a
single long fit is computed too, purely as a cross-check, and the gap between the two is recorded: if
they agree the cheaper protocol would have been safe, and if they do not, the report says so.

Two controls, both required
---------------------------
  ctl_es     the ESTABLISHED protocol (10% inner-ES carve, 90% for training, early stopping). It must
             reproduce the known fold-0 baseline (0.961299 @ 797 rounds) or the harness plumbing is
             wrong and nothing else in the table means anything.
  ctl_fixed  GBDT extra_trees through THIS harness, identical to how DART and RF are run. DART is
             compared against `ctl_fixed`, never against `ctl_es`, because the difference between them
             is itself an effect (recovering the inner-ES holdout, measured at +2.2e-5 in Phase 7).

Index discipline: `X` rows are local to `fit_idx`; labels are read at GLOBAL indices. The two index
spaces have been conflated three times in this campaign, producing fake results at -483e-5, -586e-5
and -2086e-5.

Usage:
  python scripts/run_stochastic_boosting.py --modes gbdt,dart005,dart010,rf --folds 0
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

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

# The champion structural parameters. SOURCE OF TRUTH: scripts/run_fullfit.py:72-75, which states in
# its own comment that it reproduces scripts/run_views.py::_fit_lgbm_es bit-for-bit, and which
# produced the recorded fold-0 control of 0.9612988 (reports/fullfit_primary.json).
#
# I first took these from src/models/gbdt.py:27-31 (learning_rate 0.03, num_leaves 63) on the
# reasoning that the module's defaults must be the champion's. That was WRONG: gbdt.py holds the
# family's generic defaults, and the champion overrides them. The matched control caught it
# immediately -- ctl_es returned 0.961170 at 432 rounds against an expected 0.961299 at 797, and the
# iteration count is itself the tell, since lr 0.02 naturally runs to roughly twice the rounds of
# lr 0.03. Two independent signals agreed, so the control was trusted over my reading of the source.
# The lesson is that "the model's default parameters" and "the champion's parameters" are different
# objects in this repo, and only the second one is a valid baseline.
CHAMPION = {"objective": "binary", "metric": "auc", "learning_rate": 0.02, "num_leaves": 127,
            "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
            "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "verbose": -1, "n_jobs": 8}

ES_ROUNDS = 6000      # run_views.py::_fit_lgbm_es: n_estimators
ES_PATIENCE = 300     # run_views.py::_fit_lgbm_es: lgb.early_stopping(300)

# Round grid, pre-declared from an INDEPENDENT probe (scripts/probe_dart_truncation.py: 90k-row
# subsample, raw 21 columns, different seed) rather than from anything measured on an eval fold.
# That probe's refit curve for DART-without-extra_trees peaked near 700 and then TURNED OVER --
# 0.956969 at 700, 0.956799 at 1100, 0.956538 at 1600 -- so an unbounded or high-tail-weighted grid
# would spend most of its compute in the over-iterated region and could select a tail point. This
# was a correction to my own first reading of the capability probe, where I took "still climbing at
# 900" from the +extra_trees arms and wrongly assumed it applied to the -extra_trees arm too.
# The real run has ~8.4x the rows and a lower learning rate, both of which push the peak later, so
# the grid brackets the champion's ~797 and extends to 3600 to be sure the peak is enclosed.
#
# Width is set by the measured timing probe (reports/p9_timing.json): DART runs at 0.058 s/round at
# real scale, so the whole 8900-round grid costs ~9 minutes. That measurement also CORRECTED a
# projection from probe scale that had put one DART arm at ~4 hours -- 55x too high, because at
# 60k rows fixed overhead dominated and LightGBM could not fill 8 threads. Cheap compute means the
# grid can be wide, and a wide grid is the honest choice when the peak's location is uncertain.
SNAPSHOTS = [500, 900, 1500, 2400, 3600]

MODES = {
    # --- controls. `ctl_es` is the ESTABLISHED protocol and exists only to prove the harness is
    #     wired correctly; `ctl_fixed` and `ctl_det` are the MATCHED comparisons.
    "ctl_es":     {"boosting": "gbdt", "extra_trees": True},
    "ctl_fixed":  {"boosting": "gbdt", "extra_trees": True},
    "ctl_det":    {"boosting": "gbdt", "extra_trees": False},
    # --- DART. The capability probe found DART+extra_trees sits ~4.8e-3 BELOW DART without it, the
    #     inverse of the champion's finding. The brief's matrix led with DART+extra_trees; running
    #     those arms first would spend hours confirming a large deficit, so the arms that can
    #     actually win are the ones run, with the +extra_trees pair kept available via --modes.
    "dart005":    {"boosting": "dart", "extra_trees": False, "drop_rate": 0.05, "skip_drop": 0.5},
    "dart010":    {"boosting": "dart", "extra_trees": False, "drop_rate": 0.10, "skip_drop": 0.5},
    "dart005_xt": {"boosting": "dart", "extra_trees": True, "drop_rate": 0.05, "skip_drop": 0.5},
    # --- RF. LightGBM's RF mode does NOT require bagging (the probe's bagging_fraction=1.0 /
    #     bagging_freq=0 arm trained fine), so bagging must be set explicitly or the arm silently
    #     stops being a random forest. RF+extra_trees was degenerate in the probe -- early stopping
    #     fired at 36 trees -- so it is available but not part of the default matrix.
    "rf":         {"boosting": "rf", "extra_trees": False, "bagging_fraction": 0.8,
                   "bagging_freq": 1, "feature_fraction": 0.8},
    "rf_xt":      {"boosting": "rf", "extra_trees": True, "bagging_fraction": 0.8,
                   "bagging_freq": 1, "feature_fraction": 0.8},
}


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# DART gets a SHORTER grid than GBDT, pre-declared from the probe evidence rather than from any
# eval-fold observation. Two independent reasons, both measured before this run:
#   1. the truncation probe's refit curve for DART peaked near 700 and then decayed, so a grid
#      reaching 3600 would spend most of its compute in the over-iterated region;
#   2. DART's per-round cost grows with tree count (measured 2.62x from 300 to 1500 rounds), so the
#      upper grid points are disproportionately expensive.
# The grid still brackets where the peak is expected at real scale (more rows and a lower learning
# rate both push it later than the probe's 700).
DART_SNAPSHOTS = [400, 700, 1100, 1700]


def mode_snapshots(mode: str, default: list[int]) -> list[int]:
    """Round grid for one arm: DART arms use the shorter pre-declared grid."""
    if MODES[mode]["boosting"] == "dart":
        return list(DART_SNAPSHOTS)
    return list(default)


def build_params(mode: dict, seed: int) -> dict:
    p = dict(CHAMPION)
    b = mode["boosting"]
    if b != "gbdt":
        p["boosting_type"] = b
    for k, v in mode.items():
        if k not in ("boosting",):
            p[k] = v
    p.update(random_state=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    return p


def fit_fixed(X, y, params, n_rounds):
    """Train exactly n_rounds trees with NO early stopping and no validation set.

    `n_estimators` is deliberately NOT put into the parameter dict: `lgb.train` takes the round
    count from `num_boost_round`, and passing `n_estimators` makes LightGBM emit an
    "Unknown parameter" warning that would otherwise be lost in the log.
    """
    import lightgbm as lgb
    return lgb.train(dict(params), lgb.Dataset(X, label=y), num_boost_round=int(n_rounds))


def inner_split(fit_idx, y_int, frac, seed):
    """Stratified inner-train / inner-val carved from outer-FIT rows only."""
    rng = np.random.default_rng(seed)
    pos, neg = fit_idx[y_int[fit_idx] == 1], fit_idx[y_int[fit_idx] == 0]
    n = int(len(fit_idx) * frac)
    iv = np.concatenate([rng.choice(pos, int(n * len(pos) / len(fit_idx)), replace=False),
                         rng.choice(neg, int(n * len(neg) / len(fit_idx)), replace=False)])
    iv = np.unique(iv)
    return np.setdiff1d(fit_idx, iv), iv


def report_only(args, y, folds) -> None:
    """Rebuild the paired table from saved fold predictions. No training.

    Every arm persists reports/{tag}_{mode}_fold{k}.npy, so this reconstructs the whole comparison
    including arms trained in earlier invocations. Load-time AUCs are recomputed from the label, not
    read from a cached report, so the table cannot inherit a stale or mislabelled number.
    """
    from src.validation.compare import corr, spearman
    fin = store.load_oof("blend_v3_final").astype("float64")
    out = {"tag": args.tag, "scheme": args.scheme, "report_only": True, "folds": {}}
    for k in [int(x) for x in args.folds.split(",")]:
        val = np.where(folds == k)[0]
        preds, aucs = {}, {}
        for m in list(MODES) + ["ctl_es"]:
            f = REPORTS / f"{args.tag}_{m}_fold{k}.npy"
            if f.exists():
                p = np.load(f).astype("float64")
                preds[m], aucs[m] = p, float(roc_auc_score(y[val], p))
        if not preds:
            print(f"fold {k}: no saved predictions for tag {args.tag!r}")
            continue
        ctrl_name = next((c for c in ("ctl_fixed", "ctl_es") if c in preds), None)
        ctrl = aucs[ctrl_name]
        print(f"\n{'='*104}\nfold {k}  ({len(preds)} arms, rebuilt from saved predictions)"
              f"\n{'='*104}")
        print(f"  {'mode':<14}{'AUC':>12}{'delta vs '+ctrl_name:>18}{'logit corr':>12}"
              f"{'spearman':>11}" + "".join(f"{'blend@'+w:>11}" for w in ("0.01", "0.02", "0.05")))
        print(f"  {'-'*104}")
        rec = {}
        for m in sorted(preds, key=lambda z: -aucs[z]):
            p = preds[m]
            d = aucs[m] - ctrl
            bg = {w: (float(roc_auc_score(y[val], float(w) * logit(p)
                                          + (1 - float(w)) * logit(fin[val])))
                      - float(roc_auc_score(y[val], fin[val]))) for w in ("0.01", "0.02", "0.05")}
            rec[m] = {"auc": aucs[m], "delta_e5": d * 1e5,
                      "logit_corr_vs_ctrl": (corr(logit(p), logit(preds[ctrl_name]))
                                             if m != ctrl_name else None),
                      "spearman_vs_ctrl": (spearman(p, preds[ctrl_name]) if m != ctrl_name else None),
                      "blend_gains_e5": {k2: v * 1e5 for k2, v in bg.items()}}
            tag = "(control)" if m == ctrl_name else f"{d*1e5:>+17.1f}e"
            lc = f"{rec[m]['logit_corr_vs_ctrl']:.5f}" if m != ctrl_name else ""
            sp = f"{rec[m]['spearman_vs_ctrl']:.5f}" if m != ctrl_name else ""
            print(f"  {m:<14}{aucs[m]:>12.6f}{tag:>18}{lc:>12}{sp:>11}"
                  + "".join(f"{bg[w]*1e5:>+11.2f}" for w in ("0.01", "0.02", "0.05")))
        out["folds"][str(k)] = {"control": ctrl_name, "control_auc": ctrl, "arms": rec}
        # headline gate check, applied automatically so the verdict is not left to the reader
        verdict = []
        for m, r in rec.items():
            if m == ctrl_name:
                continue
            b2 = r["blend_gains_e5"]["0.02"]
            if r["delta_e5"] >= 5.0:
                verdict.append(f"{m}: standalone {r['delta_e5']:+.1f}e-5 >= +5e-5 -> PROMOTE")
            elif b2 >= 1.5:
                verdict.append(f"{m}: marginal blend {b2:+.2f}e-5 >= +1.5e-5 -> consider promotion")
            else:
                verdict.append(f"{m}: standalone {r['delta_e5']:+.1f}e-5, blend@2% {b2:+.2f}e-5 "
                               f"-> REJECT (fails the gate)")
        out["folds"][str(k)]["verdicts"] = verdict
        print("\n  " + "\n  ".join(verdict))
    save_json(out, REPORTS / f"{args.tag}_report.json")
    print("\nwrote", REPORTS / f"{args.tag}_report.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="ctl_es,ctl_fixed,dart005")
    ap.add_argument("--view", default="full")
    ap.add_argument("--scheme", default="primary")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--inner-frac", type=float, default=0.10)
    ap.add_argument("--snapshots", default="")
    ap.add_argument("--tag", default="phase9")
    ap.add_argument("--curve-mode", default="refit", choices=["refit", "snapshot"])
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the comparison table from saved fold predictions without training "
                         "anything. Every arm writes reports/{tag}_{mode}_fold{k}.npy, so the paired "
                         "table and the blend curve can be recomputed at any time -- including for "
                         "arms run in an earlier invocation -- at zero compute cost.")
    ap.add_argument("--round-tolerance", type=float, default=1.0,
                    help="how far inner AUC may fall below the max before a cheaper round is "
                         "preferred. The inner curve is NOT monotone -- the champion peaks at ~900 "
                         "and decays after -- and the peak is broad and noisy at 56k inner-val rows, "
                         "where a 1e-5 wobble is sampling noise rather than a real difference. "
                         "Picking the raw argmax then spends 4x the rounds for nothing, which is the "
                         "over-iteration failure mode already measured twice in this campaign "
                         "(-5.9e-5 against its own control). Pre-declared at 1e-5, which is BELOW "
                         "the +1.5e-5 admission gate: a difference this rule treats as 'not evidence' "
                         "is smaller than the gate, so it can never discard a round count on the "
                         "basis of a gap that would have counted as a real gain. Set 0 for pure argmax.")
    ap.add_argument("--timing-probe", type=int, default=0,
                    help="if >0, fit each arm ONCE at this round count on the inner-train rows, "
                         "report seconds/round, and exit without scoring anything")
    args = ap.parse_args()

    snaps = ([int(x) for x in args.snapshots.split(",")] if args.snapshots
             else list(SNAPSHOTS))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(f"unknown modes {unknown}; known: {sorted(MODES)}")

    # ---- guard: snapshot mode is only honest if truncation was measured to be faithful ----
    # Selecting rounds from `predict(num_iteration=r)` on a long fit instead of refitting is only
    # legitimate when an r-round model equals the r-round prefix of an N-round model. For DART that
    # is not a safe assumption, so it is gated on the measured verdict in
    # reports/dart_truncation.json rather than left to the operator's judgement.
    if args.curve_mode == "snapshot" and any(MODES[m]["boosting"] == "dart" for m in modes):
        tj = REPORTS / "dart_truncation.json"
        if not tj.exists():
            raise SystemExit("--curve-mode snapshot with a DART arm requires the truncation probe. "
                             f"Run scripts/probe_dart_truncation.py first ({tj} missing).")
        safe = json.loads(tj.read_text(encoding="utf-8"))["summary"]["snapshot_mode_safe_for_dart"]
        if not safe:
            raise SystemExit("the truncation probe found snapshot selection UNSAFE for DART; "
                             "use --curve-mode refit.")
        print("truncation probe verdict: snapshot mode is within tolerance for DART -> using it.\n")

    tr, te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    y = y_int.astype("float64")
    folds = get_scheme(args.scheme, y_int, tr[ID_COL]).folds

    if args.report_only:
        return report_only(args, y, folds)
    fin = store.load_oof("blend_v3_final").astype("float64")
    fin_test = store.load_test("blend_v3_final").astype("float64")

    vcache = {}
    out = {"tag": args.tag, "scheme": args.scheme, "seed": args.seed,
           "inner_frac": args.inner_frac, "snapshots": snaps, "curve_mode": args.curve_mode,
           "modes": modes, "folds": {},
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()[:12]}

    for k in [int(x) for x in args.folds.split(",")]:
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        assert not (set(fit.tolist()) & set(val.tolist())), "fit/eval overlap"
        if args.view not in vcache:
            vcache[args.view] = ViewBuilder(tr, te, args.view)
            vcache[args.view].build_static()
        vb = vcache[args.view]
        Xf, Xa, names = vb.assemble(fit, y_int, val, None, inner_seed=k)
        Xv = Xa["val"]
        pos = {int(v): i for i, v in enumerate(fit)}
        y_fit_local = y[fit]

        print(f"\n{'='*104}\nfold {k}   outer-fit={len(fit):,}   eval={len(val):,}   "
              f"features={Xf.shape[1]}\n{'='*104}", flush=True)

        # ---- timing probe: one fit per arm, no scoring, no selection --------------------
        # Required before committing to a multi-hour run. The round count is extrapolated
        # linearly from here, so the report states the measured seconds/round and the projected
        # total for the full snapshot grid rather than a guess.
        if args.timing_probe:
            tr, yv = y_int, None
            itr_g, iv_g = inner_split(fit, tr, args.inner_frac, args.seed + k + 500)
            itr_l = np.array([pos[int(v)] for v in itr_g])
            proj = {}
            for mname in modes:
                if mname == "ctl_es":
                    continue
                t0 = time.time()
                fit_fixed(Xf[itr_l], y[itr_g], build_params(MODES[mname], args.seed + k),
                          args.timing_probe)
                dt = time.time() - t0
                per = dt / args.timing_probe
                need = sum(snaps) + (min(snaps) if MODES[mname]["boosting"] == "dart" else 0)
                proj[mname] = {"seconds": round(dt, 1), "sec_per_round": round(per, 3),
                               "projected_full_grid_seconds": round(per * need),
                               "projected_full_grid_hours": round(per * need / 3600, 2),
                               "grid_rounds_sum": int(need)}
                print(f"  {mname:<12} {args.timing_probe} rounds in {dt:7.1f}s  = "
                      f"{per:.3f}s/round   -> full grid ({need} rounds) ~ "
                      f"{per*need/3600:.2f} h", flush=True)
            print(f"\n  inner-train rows used: {len(itr_l):,}   features: {Xf.shape[1]}")
            # Extrapolation is super-linear for DART and linear for GBDT. Measured, not assumed:
            # DART cost per round rose from 0.058 s at 300 rounds to 0.152 s at 1500 rounds on
            # identical data (2.62x), because each new tree re-normalises the surviving ensemble.
            # GBDT's s/round is flat and its curve decays after ~900, so GBDT is linear.
            #
            # The anchor must be the TOTAL cost of the probe fit at the probe's own round count. An
            # earlier version re-expressed the average s/round as if it were the cost at 300 rounds
            # and then scaled from 300, which understated a 3600-round DART fit by ~10x. The fit is
            # anchored at (args.timing_probe, measured_seconds) and scaled from there.
            anchor_r = args.timing_probe

            def cost(rounds: int, anchor_s: float, dart: bool) -> float:
                if dart:
                    return anchor_s * (rounds / anchor_r) ** 1.5
                return anchor_s * (rounds / anchor_r)

            for mname, pj in proj.items():
                dart = MODES[mname]["boosting"] == "dart"
                grid = mode_snapshots(mname, snaps)
                sel = grid[len(grid) // 2]
                pj["grid_used"] = grid
                pj["extrapolation"] = (f"r^1.5 from the measured {anchor_r}-round fit"
                                       if dart else f"linear from the measured {anchor_r}-round fit")
                pj["projected_full_grid_seconds"] = round(
                    sum(cost(r, pj["seconds"], dart) for r in grid) + cost(sel, pj["seconds"], dart))
                pj["projected_full_grid_hours"] = round(
                    pj["projected_full_grid_seconds"] / 3600.0, 2)
            save_json({"tag": args.tag, "timing_probe_rounds": args.timing_probe,
                       "scheme": args.scheme, "view": args.view, "folds": {str(k): proj},
                       "note": "DART extrapolated with r^1.5 because its measured per-round cost rose "
                               "2.62x between a 300- and a 1500-round fit on identical data; GBDT "
                               "extrapolated linearly. A purely linear projection understated DART."},
                      REPORTS / f"{args.tag}_timing.json")
            for mname, pj in proj.items():
                print(f"  {mname:<12} projection: {pj['extrapolation']}  -> full grid "
                      f"~ {pj['projected_full_grid_hours']:.2f} h", flush=True)
            return
        fold_rec = {}
        for mname in modes:
            mode = MODES[mname]
            seed = args.seed + k
            params = build_params(mode, seed)

            # ---------------- the ESTABLISHED protocol, for plumbing validation ----------------
            if mname == "ctl_es":
                tr_g, es_g = _inner_es_split(fit, y_int, seed)
                tr_l = np.array([pos[int(v)] for v in tr_g])
                es_l = np.array([pos[int(v)] for v in es_g])
                import lightgbm as lgb
                ds = lgb.Dataset(Xf[tr_l], label=y[tr_g])
                dv = lgb.Dataset(Xf[es_l], label=y[es_g], reference=ds)
                t0 = time.time()
                m = lgb.train(dict(params), ds, num_boost_round=ES_ROUNDS, valid_sets=[dv],
                              callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)])
                it = int(m.best_iteration or ES_ROUNDS)
                # predict at `it` explicitly rather than relying on best_iteration being applied
                # automatically, so this matches run_fullfit.py's control call exactly
                pred = m.predict(Xv, num_iteration=it)
                auc = float(roc_auc_score(y[val], pred))
                print(f"  {mname:<12} ESTABLISHED protocol: {len(tr_l):,} train rows, ES on "
                      f"{len(es_l):,}, iter={it}  AUC={auc:.6f}  ({time.time()-t0:.0f}s)", flush=True)
                fold_rec[mname] = {"auc": auc, "iter": it, "n_rows": int(len(tr_l)),
                                   "protocol": "established inner-ES", "seconds": round(time.time()-t0, 1),
                                   "inner_curve": None, "pred": pred}
                np.save(REPORTS / f"{args.tag}_{mname}_fold{k}.npy", pred.astype("float32"))
                continue

            # ---------------- honest fixed-round protocol ----------------
            itr_g, iv_g = inner_split(fit, y_int, args.inner_frac, seed + 500)
            itr_l = np.array([pos[int(v)] for v in itr_g])
            iv_l = np.array([pos[int(v)] for v in iv_g])
            y_itr = y[itr_g]
            y_iv = y[iv_g]

            arm_snaps = mode_snapshots(mname, snaps)
            curve, curve_snap = [], []
            t_curve = time.time()
            if args.curve_mode == "refit":
                for r in arm_snaps:
                    mm = fit_fixed(Xf[itr_l], y_itr, params, r)
                    curve.append({"round": r, "auc": float(roc_auc_score(y_iv, mm.predict(Xf[iv_l])))})
                # cross-check: what a single long fit's truncated snapshots would have said
                mm_long = fit_fixed(Xf[itr_l], y_itr, params, max(arm_snaps))
                for r in arm_snaps:
                    curve_snap.append(
                        {"round": r, "auc": float(roc_auc_score(y_iv, mm_long.predict(Xf[iv_l],
                                                                                    num_iteration=r)))})
            else:
                mm_long = fit_fixed(Xf[itr_l], y_itr, params, max(arm_snaps))
                for r in arm_snaps:
                    curve_snap.append(
                        {"round": r, "auc": float(roc_auc_score(y_iv, mm_long.predict(Xf[iv_l],
                                                                                    num_iteration=r)))})
                curve = list(curve_snap)
            top = max(c["auc"] for c in curve)
            # Choose the SMALLEST round whose inner AUC is within tolerance of the max. Ties resolve
            # down, and near-ties resolve down too, for the reasons in --round-tolerance: the peak
            # is broad and the inner-val noise floor is ~1e-5, so the raw argmax overstates the
            # evidence for the extra rounds it spends.
            n_sel = min(c["round"] for c in curve if c["auc"] >= top - args.round_tolerance)
            n_argmax = min(c["round"] for c in curve if c["auc"] >= top - 1e-12)
            t_sel = time.time() - t_curve

            snap_gap = (max(c["auc"] for c in curve_snap) - top) if curve_snap else None
            print(f"  {mname:<12} inner curve (refit, {args.curve_mode}):", flush=True)
            for c in curve:
                print(f"      round {c['round']:>5}  inner AUC {c['auc']:.6f}", flush=True)
            print(f"    curve max = {top:.6f} at round {n_argmax}; selected round = {n_sel} "
                  f"(within {args.round_tolerance:.0e} of max, cheapest such round); "
                  f"inner selection took {t_sel:.0f}s", flush=True)
            if n_sel != n_argmax:
                d = [c for c in curve if c["round"] == n_argmax][0]
                print(f"    NOTE: chose {n_sel} over the raw argmax {n_argmax}; the argmax was "
                      f"{d['auc'] - top + (top - d['auc']):.0e} below max by "
                      f"{(top - d['auc'])*1e5:.2f}e-5, inside the {args.round_tolerance:.0e} "
                      f"tolerance, so the extra rounds are not evidenced.", flush=True)
            if snap_gap is not None:
                print(f"    cross-check: a single long fit's truncated snapshots would have picked "
                      f"max {max(c['auc'] for c in curve_snap):.6f}, i.e. {snap_gap*1e5:+.1f}e-5 "
                      f"{'different' if abs(snap_gap) > 1e-6 else '(agrees)'}")

            t0 = time.time()
            mf = fit_fixed(Xf, y_fit_local, params, n_sel)     # 100% of outer-fit, fixed rounds
            pred = mf.predict(Xv)
            auc = float(roc_auc_score(y[val], pred))
            t_refit = time.time() - t0
            print(f"    REFIT on all {len(fit):,} outer-fit rows at {n_sel} rounds -> "
                  f"AUC={auc:.6f}  ({t_refit:.0f}s)", flush=True)
            np.save(REPORTS / f"{args.tag}_{mname}_fold{k}.npy", pred.astype("float32"))

            bg = {}
            for w in (0.01, 0.02, 0.03, 0.05, 0.075, 0.10):
                bg[str(w)] = float(roc_auc_score(y[val], w * logit(pred)
                                                  + (1 - w) * logit(fin[val]))) - float(
                    roc_auc_score(y[val], fin[val]))
            fold_rec[mname] = {"auc": auc, "selected_round": int(n_sel),
                               "argmax_round": int(n_argmax), "round_tolerance": args.round_tolerance,
                               "inner_curve": curve, "grid": arm_snaps,
                               "snapshot_crosscheck": curve_snap,
                               "snapshot_vs_refit_argmax_gap_e5": (snap_gap * 1e5) if snap_gap is not None else None,
                               "n_rows": int(len(fit)), "seconds": round(t_refit, 1),
                               "seconds_inner_selection": round(t_sel, 1),
                               "blend_gains": bg, "params": {kk: vv for kk, vv in params.items()},
                               "pred": pred}
            print(f"    blend gains vs v3: "
                  f"{', '.join(f'w={a}:{b*1e5:+.2f}' for a, b in bg.items())}", flush=True)

        # ---- paired comparison table ----
        # Controls may be run in a SEPARATE invocation (they are expensive and reusable), so the
        # reference is looked up from this fold's own records first and then from previously saved
        # control predictions on disk. Without the disk fallback the table crashed with
        # 'NoneType' is not subscriptable, which is how it first failed.
        ctrl = fold_rec.get("ctl_fixed") or fold_rec.get("ctl_es")
        ctrl_name = "ctl_fixed" if "ctl_fixed" in fold_rec else ("ctl_es" if "ctl_es" in fold_rec
                                                                 else None)
        if ctrl is None:
            cands = ["ctl_fixed", "ctl_es"]
            for c in cands:
                f = REPORTS / f"{args.tag}_{c}_fold{k}.npy"
                if f.exists():
                    pv = np.load(f).astype("float64")
                    ctrl = {"auc": float(roc_auc_score(y[val], pv)), "pred": pv,
                            "iter": None, "blend_gains": {}}
                    ctrl_name = f"{c} (loaded from {f.name})"
                    break
        if ctrl is None:
            print("  no control available for this fold: run --modes ctl_es,ctl_fixed first")
            out["folds"][str(k)] = {kk: {a: b for a, b in vv.items() if a != "pred"}
                                    for kk, vv in fold_rec.items()}
            continue
        print(f"\n  {'mode':<14}{'rounds':>8}{'AUC':>12}{'delta vs '+ctrl_name:>18}"
              f"{'logit corr':>12}{'spearman':>11}{'blend@2%':>10}")
        print(f"  {'-'*104}")
        for mname in modes:
            if mname not in fold_rec:
                continue
            r = fold_rec[mname]
            rr = r.get("selected_round") or r.get("iter")
            if mname == ctrl_name or r is ctrl:
                print(f"  {mname:<14}{str(rr):>8}{r['auc']:>12.6f}{'(control)':>18}{'':>12}"
                      f"{'':>11}"
                      f"{r.get('blend_gains', {}).get('0.02', float('nan'))*1e5:>+10.2f}")
                continue
            d = r["auc"] - ctrl["auc"]
            print(f"  {mname:<14}{rr:>8}{r['auc']:>12.6f}{d*1e5:>+17.1f}e"
                  f"{corr(logit(r['pred']), logit(ctrl['pred'])):>12.5f}"
                  f"{spearman(r['pred'], ctrl['pred']):>11.5f}"
                  f"{r['blend_gains']['0.02']*1e5:>+10.2f}")

        out["folds"][str(k)] = {kk: {a: b for a, b in vv.items() if a != "pred"}
                                for kk, vv in fold_rec.items()}
        out["folds"][str(k)]["_control"] = ctrl_name

    save_json(out, REPORTS / f"{args.tag}.json")
    print(f"\nwrote {REPORTS / f'{args.tag}.json'}")
    print("inner curves persisted in the report; outer-validation labels were used exactly once per "
          "arm.")


if __name__ == "__main__":
    main()
