"""Capability probe: what do DART and RF actually DO in this LightGBM build?

Why probe before training
-------------------------
The whole Phase 9 design rests on three claims that are easy to believe and easy to get wrong:

  1. `boosting_type="dart"` trains and predicts,
  2. ordinary early stopping selects rounds for it in the normal way,
  3. `boosting_type="rf"` trains under its own stochasticity constraints.

Claim 2 is the dangerous one. It is well known that DART's prediction is built from the trees that
survived the final dropout pass, so the model's output changes as more trees are added rather than
improving monotonically, and `best_iteration` from a standard `early_stopping` callback is therefore
not obviously a meaningful round selector. The prompt explicitly warns against trusting a standard
callback for DART. So this probe measures it rather than assuming it: it trains DART on real data and
compares the callback's chosen round against the actual inner-validation curve.

Everything here is small and fast (a 20% row subsample, few hundred trees). The point is to fail
cheaply and loudly, before committing a multi-hour run.

What is reported per mode
-------------------------
  trains                 did lgb.train return a usable model
  best_iteration         what the early_stopping callback reported, or None
  callback_usable        whether that number matches the argmax of the measured AUC curve
  auc_at_best            AUC at the reported round
  auc_argmax             best AUC actually observed over the snapshots
  round_of_argmax        where the real optimum sits
  extratrees_compat      whether extra_trees trains at all with this mode
  notes                  anything the probe learned that the harness must respect

Usage: python scripts/probe_dart_rf.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import RAW21  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

SNAPSHOTS = [100, 200, 300, 500, 700, 900, 1200]

BASE = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 40, "colsample_bytree": 0.8, "verbose": -1, "n_jobs": 8}


def prep():
    tr, _te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y, tr[ID_COL]).folds
    fit = np.where(folds != 0)[0]
    val = np.where(folds == 0)[0]
    rng = np.random.default_rng(0)
    sub = rng.choice(fit, size=90_000, replace=False)
    a = fit[:0]
    # split the subsample into inner train / inner val, both from outer-fit only
    pos, neg = sub[y[sub] == 1], sub[y[sub] == 0]
    iv = np.concatenate([rng.choice(pos, 12_000, replace=False),
                         rng.choice(neg, 18_000, replace=False)])
    itr = np.setdiff1d(sub, iv)
    # Encode every non-numeric raw column by sorted category order. The categorical set is NOT
    # assumed -- asking for float32 from a frame containing e.g. 'Female' raises, and hard-coding
    # the column names would silently rot if the schema ever changed.
    cols = list(RAW21)
    X = np.empty((len(tr), len(cols)), dtype="float32")
    ncat = []
    for j, c in enumerate(cols):
        s = tr[c]
        if pd.api.types.is_numeric_dtype(s):
            X[:, j] = s.to_numpy(dtype="float32")
        else:
            codes, _ = pd.factorize(s.astype(str), sort=True)
            X[:, j] = codes.astype("float32")
            ncat.append(c)
    X = np.nan_to_num(X, nan=-999.0)
    print(f"probe matrix: {X.shape[1]} columns, of which {len(ncat)} categoricals "
          f"encoded as ordinal codes: {ncat}")
    return X, y.astype("float64"), itr, iv, val


def probe(name, params, X, y, itr, iv, n_estimators, use_callback):
    import lightgbm as lgb
    print(f"\n{'-'*94}\n{name}\n{'-'*94}")
    print(f"  params: {json.dumps({k: v for k, v in params.items() if k not in BASE})}")
    rec = {"name": name, "extra_params": {k: v for k, v in params.items() if k not in BASE},
           "n_estimators_cap": n_estimators, "used_callback": use_callback}
    t0 = time.time()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p = dict(BASE)
            p.update(params)
            ds = lgb.Dataset(X[itr], label=y[itr])
            dv = lgb.Dataset(X[iv], label=y[iv], reference=ds)
            cbs = [lgb.early_stopping(50, verbose=False)] if use_callback else []
            m = lgb.train(p, ds, num_boost_round=n_estimators, valid_sets=[dv], callbacks=cbs)
            rec["warnings"] = [str(x.message)[:120] for x in w][:4]
        rec["trains"] = True
        rec["seconds"] = round(time.time() - t0, 1)
        bi = getattr(m, "best_iteration", None)
        rec["best_iteration"] = int(bi) if bi else None
        rec["num_trees"] = int(m.num_trees())
    except Exception as exc:  # noqa: BLE001
        rec["trains"] = False
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        rec["seconds"] = round(time.time() - t0, 1)
        print(f"  DOES NOT TRAIN: {rec['error']}")
        print(f"  ({rec['seconds']}s)")
        return rec

    # ---- measure the real inner-validation curve at snapshots ----
    # This is the measurement the prompt demands for DART: does the callback's round match the
    # actual optimum, or is DART's round selection a different problem?
    curve = []
    for r in SNAPSHOTS:
        if r > m.num_trees():
            break
        pr = m.predict(X[iv], num_iteration=r)
        curve.append((int(r), float(roc_auc_score(y[iv], pr))))
    rec["curve"] = [{"round": r, "auc": a} for r, a in curve]
    if curve:
        best_r, best_a = max(curve, key=lambda t: t[1])
        rec["auc_argmax"] = best_a
        rec["round_of_argmax"] = best_r
        rec["auc_at_reported_best"] = (float(roc_auc_score(
            y[iv], m.predict(X[iv], num_iteration=bi))) if bi else None)
        rec["callback_usable"] = bool(bi and abs(bi - best_r) <= 150)
        print(f"  trained in {rec['seconds']}s   trees={rec['num_trees']}   "
              f"best_iteration reported = {bi}")
        print(f"  {'round':>7}{'inner AUC':>12}{'delta vs prev':>15}")
        prev = None
        for r, a in curve:
            d = "" if prev is None else f"{(a-prev)*1e5:+.1f}e-5"
            print(f"  {r:>7}{a:>12.6f}{d:>15}")
            prev = a
        print(f"  argmax at round {best_r} ({best_a:.6f}); callback said "
              f"{bi}; AUC there = {rec['auc_at_reported_best']}")
        gap = (rec["auc_at_reported_best"] or 0) - best_a
        print(f"  callback loses {gap*1e5:+.1f}e-5 against the measured optimum -> "
              f"callback_usable={rec['callback_usable']}")
        if rec["warnings"]:
            print(f"  warnings: {rec['warnings']}")
    return rec


def main() -> None:
    X, y, itr, iv, val = prep()
    print(f"probe data: inner-train={len(itr):,}  inner-val={len(iv):,}  "
          f"features={X.shape[1]}")
    print(f"LightGBM capabilities come from the installed build; nothing here is assumed.")

    out = {"lightgbm_version": __import__("lightgbm").__version__, "snapshots": SNAPSHOTS,
           "probes": []}

    # ---- GBDT control, so the probe has a reference point ----
    out["probes"].append(probe("G0  GBDT extra_trees  (reference)", {"extra_trees": True},
                               X, y, itr, iv, 700, True))

    # ---- DART ----
    for nm, extra in (("D1  DART drop 0.05 skip 0.5 + extra_trees",
                       {"boosting_type": "dart", "extra_trees": True,
                        "drop_rate": 0.05, "skip_drop": 0.5}),
                      ("D2  DART drop 0.10 skip 0.5 + extra_trees",
                       {"boosting_type": "dart", "extra_trees": True,
                        "drop_rate": 0.10, "skip_drop": 0.5}),
                      ("D3  DART drop 0.05 skip 0.5, NO extra_trees (the extra_trees control)",
                       {"boosting_type": "dart", "drop_rate": 0.05, "skip_drop": 0.5})):
        out["probes"].append(probe(nm, extra, X, y, itr, iv, 900, True))

    # ---- RF, and the constraints it actually imposes ----
    for nm, extra in (("R1  RF bagging .8 freq 1, ordinary thresholds",
                       {"boosting_type": "rf", "bagging_fraction": 0.8, "bagging_freq": 1,
                        "feature_fraction": 0.8}),
                      ("R2  RF bagging .8 freq 1 + extra_trees",
                       {"boosting_type": "rf", "extra_trees": True, "bagging_fraction": 0.8,
                        "bagging_freq": 1, "feature_fraction": 0.8}),
                      ("R3  RF bagging 1.0 freq 0 (should be REJECTED by LightGBM)",
                       {"boosting_type": "rf", "bagging_fraction": 1.0, "bagging_freq": 0})):
        out["probes"].append(probe(nm, extra, X, y, itr, iv, 700, True))

    print("\n" + "=" * 94)
    print("SUMMARY")
    print("=" * 94)
    print(f"  {'probe':<52}{'trains':>8}{'trees':>7}{'best_iter':>10}{'argmax':>8}{'usable':>8}")
    for p in out["probes"]:
        if not p.get("trains"):
            print(f"  {p['name'][:50]:<52}{'NO':>8}   {str(p.get('error'))[:40]}")
            continue
        print(f"  {p['name'][:50]:<52}{'yes':>8}{p['num_trees']:>7}"
              f"{str(p['best_iteration']):>10}{p.get('round_of_argmax', '-'):>8}"
              f"{str(p.get('callback_usable')):>8}")

    dart = [p for p in out["probes"] if "dart" in str(p.get("extra_params", {}))]
    rf = [p for p in out["probes"] if p.get("extra_params", {}).get("boosting_type") == "rf"]
    print("\n  DECISIONS THE PROBE FORCES ON THE HARNESS:")
    if any(p.get("trains") for p in dart):
        bad = [p["name"] for p in dart if p.get("trains") and not p.get("callback_usable")]
        if bad:
            print(f"  * DART trains. The callback's round selection DISAGREES with the measured")
            print(f"    argmax for {len(bad)} of {len(dart)} configs, so the fixed-round "
                  f"inner-selection protocol is\n    MANDATORY, not optional: "
                  f"{[b.split()[0] for b in bad]}")
        else:
            print("  * DART trains, and the callback's round happened to land near the measured")
            print("    argmax on this subsample. That is not a guarantee -- DART's curve is not")
            print("    monotone in rounds -- so the inner-selection protocol is still used.")
        print("  * DART prediction at a given num_iteration is the ensemble surviving the final")
        print("    dropout pass, so snapshots of the curve are required, not just a best_iteration.")
    else:
        print("  * DART DOES NOT TRAIN in this build.")
    if any(p.get("trains") for p in rf):
        bad_rf = [p for p in rf if not p.get("trains")]
        print(f"  * RF trains under bagging_fraction<1 with bagging_freq>0 "
              f"({len(bad_rf)} config(s) correctly refused by LightGBM).")
    else:
        print("  * RF DOES NOT TRAIN in this build.")
    save_json(out, REPORTS / "dart_rf_capability.json")
    print("\nwrote", REPORTS / "dart_rf_capability.json")


if __name__ == "__main__":
    main()
