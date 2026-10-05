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

# The champion structural parameters, copied from src/models/gbdt.py:27-31 so that the matched
# control is the champion and not an approximation of it. Only the tree-construction mode varies
# between arms; leaves, min_child_samples, colsample, subsample, lambda, max_bin and learning rate
# are held at the champion's values. Getting these wrong would make the control unable to reproduce
# the known baseline, which is the only check that the harness is wired correctly.
CHAMPION = {"objective": "binary", "metric": "auc", "learning_rate": 0.03, "num_leaves": 63,
            "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
            "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "verbose": -1, "n_jobs": 8}

ES_ROUNDS = 4000      # src/models/gbdt.py: n_estimators
ES_PATIENCE = 200     # src/models/gbdt.py: lgb.early_stopping(200)

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
            save_json({"tag": args.tag, "timing_probe_rounds": args.timing_probe,
                       "scheme": args.scheme, "view": args.view, "folds": {str(k): proj},
                       "note": "projections are linear in rounds from a single measured fit and "
                               "will understate cost for DART, whose per-round normalisation "
                               "overhead grows with the tree count"},
                      REPORTS / f"{args.tag}_timing.json")
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

            curve, curve_snap = [], []
            t_curve = time.time()
            if args.curve_mode == "refit":
                for r in snaps:
                    mm = fit_fixed(Xf[itr_l], y_itr, params, r)
                    curve.append({"round": r, "auc": float(roc_auc_score(y_iv, mm.predict(Xf[iv_l])))})
                # cross-check: what a single long fit's truncated snapshots would have said
                mm_long = fit_fixed(Xf[itr_l], y_itr, params, max(snaps))
                for r in snaps:
                    curve_snap.append(
                        {"round": r, "auc": float(roc_auc_score(y_iv, mm_long.predict(Xf[iv_l],
                                                                                    num_iteration=r)))})
            else:
                mm_long = fit_fixed(Xf[itr_l], y_itr, params, max(snaps))
                for r in snaps:
                    curve_snap.append(
                        {"round": r, "auc": float(roc_auc_score(y_iv, mm_long.predict(Xf[iv_l],
                                                                                    num_iteration=r)))})
                curve = list(curve_snap)
            best = max(curve, key=lambda c: c["auc"])
            # ties resolved to the SMALLER round: over-iteration is the one failure mode we have
            # actually measured in this campaign (1191 rounds cost -5.9e-5 against its control)
            top = max(c["auc"] for c in curve)
            n_sel = min(c["round"] for c in curve if c["auc"] >= top - 1e-12)
            t_sel = time.time() - t_curve

            snap_gap = (max(c["auc"] for c in curve_snap) - top) if curve_snap else None
            print(f"  {mname:<12} inner curve (refit, {args.curve_mode}):", flush=True)
            for c in curve:
                print(f"      round {c['round']:>5}  inner AUC {c['auc']:.6f}", flush=True)
            print(f"    selected round = {n_sel} (argmax, ties to the smaller round); "
                  f"inner selection took {t_sel:.0f}s", flush=True)
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
            fold_rec[mname] = {"auc": auc, "selected_round": int(n_sel), "inner_curve": curve,
                               "snapshot_crosscheck": curve_snap,
                               "snapshot_vs_refit_argmax_gap_e5": (snap_gap * 1e5) if snap_gap is not None else None,
                               "n_rows": int(len(fit)), "seconds": round(t_refit, 1),
                               "seconds_inner_selection": round(t_sel, 1),
                               "blend_gains": bg, "params": {kk: vv for kk, vv in params.items()},
                               "pred": pred}
            print(f"    blend gains vs v3: "
                  f"{', '.join(f'w={a}:{b*1e5:+.2f}' for a, b in bg.items())}", flush=True)

        # ---- paired comparison table ----
        ctrl = fold_rec.get("ctl_fixed") or fold_rec.get("ctl_es")
        ctrl_name = "ctl_fixed" if "ctl_fixed" in fold_rec else "ctl_es"
        print(f"\n  {'mode':<14}{'rounds':>8}{'AUC':>12}{'delta vs '+ctrl_name:>18}"
              f"{'logit corr':>12}{'spearman':>11}{'blend@2%':>10}")
        print(f"  {'-'*104}")
        for mname in modes:
            if mname not in fold_rec:
                continue
            r = fold_rec[mname]
            rr = r.get("selected_round") or r.get("iter")
            if mname == ctrl_name:
                print(f"  {mname:<14}{rr:>8}{r['auc']:>12.6f}{'(control)':>18}{'':>12}{'':>11}"
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
