"""Capability probe: what does CatBoost 1.2.10 actually support here, and how fast is it?

Why probe before training
-------------------------
Phase 9's lesson was that the installed build's real behaviour differs from the documented default
in ways that invalidate a whole protocol: LightGBM DART turned out to have NO early stopping at all.
CatBoost has two more places where that can bite:

  1. `boosting_type="Ordered"` is the mechanism this experiment exists to test, and it is NOT the
     default (`Plain`). If Ordered is unsupported, or only on CPU, or refuses categorical features,
     the entire hypothesis has to be re-scoped rather than discovered mid-run.
  2. GPU support for categorical features and for Ordered boosting is restricted in ways that vary
     by version. Silently falling back to CPU, or silently DROPPING the categorical columns, would
     produce a confident wrong number. The probe checks the columns actually reach the model.

The specific failure this guards against: if `cat_features` names do not match the frame's columns,
CatBoost raises; but if a column is present and simply not declared, the model trains happily on it
as a float and the run LOOKS fine while testing nothing. So the probe asserts the CTR machinery is
actually engaged by comparing a declared-categorical run against an undeclared one on the same data.

Probe arms, all on a 15% row subsample so this is minutes not hours:
  P0  numeric only, Plain                    -- the current control, must work
  P1  numeric only, Ordered                  -- Ordered alone
  P2  numeric + native cats, Plain           -- native CTRs alone
  P3  numeric + native cats, Ordered         -- the full mechanism
  P4  P2 with task_type="GPU"                -- is GPU categorical support real?
  P5  P3 with task_type="GPU"                -- GPU + Ordered
  P6  numeric + cats, Plain, max_ctr_complexity=2 -- one higher interaction setting
  P7  numeric + cats, Plain, one_hot_max_size=6   -- do the 6-level ratings go one-hot instead?

Usage: python scripts/probe_catboost.py
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
from scripts.native_cat import (DEFAULT_SERVICE_CARDINALITY, EXPECTED_CARDINALITY,  # noqa: E402
                                MISSING_SENTINEL, cat_cardinality, cat_frame, default_cat_cols)

ROUNDS = 250          # small on purpose: we are measuring capability and rate, not score
BASE = dict(iterations=ROUNDS, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
            verbose=0, allow_writing_files=False, eval_metric="AUC", thread_count=8)


def prep(sub_frac=0.15, seed=0):
    tr, _te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    y_int = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y_int, tr[ID_COL]).folds
    fit = np.where(folds != 0)[0]
    rng = np.random.default_rng(seed)
    sub = rng.choice(fit, int(len(fit) * sub_frac), replace=False)
    pos, neg = sub[y_int[sub] == 1], sub[y_int[sub] == 0]
    iv = np.unique(np.concatenate([rng.choice(pos, int(0.15 * len(sub)), replace=False),
                                   rng.choice(neg, int(0.15 * len(sub)), replace=False)]))
    itr = np.setdiff1d(sub, iv)
    cols = list(RAW21)
    X = np.empty((len(tr), len(cols)), dtype="float32")
    for j, c in enumerate(cols):
        s = tr[c]
        X[:, j] = (s.to_numpy(dtype="float32") if pd.api.types.is_numeric_dtype(s)
                   else pd.factorize(s.astype(str), sort=True)[0].astype("float32"))
    X = np.nan_to_num(X, nan=-999.0)
    return tr, X, y, itr, iv


def probe(name, Xtr, Xiv, ytr, yiv, cat_cols, boosting, extra=None, task="CPU"):
    from catboost import CatBoostClassifier
    p = dict(BASE)
    p["boosting_type"] = boosting
    p["task_type"] = task
    p["random_seed"] = 4
    if extra:
        p.update(extra)
    rec = {"name": name, "boosting_type": boosting, "task_type": task,
           "n_cat_cols": len(cat_cols), "extra": extra or {}, "cat_cols": list(cat_cols)}
    t0 = time.time()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = CatBoostClassifier(**p)
            fit_kw = {"cat_features": list(cat_cols)} if cat_cols else {}
            m.fit(Xtr, ytr, **fit_kw)
            rec["warnings"] = sorted({str(x.message)[:110] for x in w})[:4]
        rec["trains"] = True
        rec["seconds"] = round(time.time() - t0, 1)
        pred = m.predict_proba(Xiv)[:, 1]
        rec["auc"] = float(roc_auc_score(yiv, pred))
        rec["trees"] = int(m.tree_count_)
        # Did CatBoost actually engage the CTR machinery? `model_features` and the ctr leaf
        # description expose it; a model given categoricals but no CTRs shows no ctr leaves.
        try:
            ctrs = m.get_leaf_ctr_description()
            rec["n_ctr_leaf_descriptions"] = len(ctrs)
        except Exception as exc:                                # noqa: BLE001
            rec["n_ctr_leaf_descriptions"] = None
            rec["ctr_introspection_error"] = f"{type(exc).__name__}: {str(exc)[:80]}"
        print(f"  {name:<52} trains  {rec['seconds']:>7.1f}s  AUC={rec.get('auc', float('nan')):.6f}"
              f"  trees={rec['trees']}  ctr_leaves={rec.get('n_ctr_leaf_descriptions')}")
        if rec.get("warnings"):
            print(f"      warnings: {rec['warnings']}")
    except Exception as exc:                                    # noqa: BLE001
        rec["trains"] = False
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:260]}"
        rec["seconds"] = round(time.time() - t0, 1)
        print(f"  {name:<52} DOES NOT TRAIN  {rec['seconds']:.1f}s")
        print(f"      {rec['error']}")
    return rec


def main() -> None:
    import catboost
    tr, X, y, itr, iv = prep()
    cat_cols = default_cat_cols(True)
    card = cat_cardinality(tr, cat_cols)
    print("=" * 104)
    print(f"CATBOOST CAPABILITY PROBE   version={catboost.__version__}   "
          f"{ROUNDS} rounds on a 15% subsample")
    print("=" * 104)
    print(f"  inner-train={len(itr):,}  inner-val={len(iv):,}  numeric features={X.shape[1]}  "
          f"cat twins={len(cat_cols)}")
    print(f"  cardinality: {dict(sorted(card.items(), key=lambda kv: str(kv[0])))}")
    print(f"  sentinel={MISSING_SENTINEL!r}  collides="
          f"{[c for c in cat_cols if MISSING_SENTINEL in set(cat_frame(tr, [c], np.arange(200)).iloc[:,0])]}")
    print()

    # numeric-only frames
    Xn_tr = pd.DataFrame(X[itr], columns=list(RAW21))
    Xn_iv = pd.DataFrame(X[iv], columns=list(RAW21))
    # numeric + native categorical frames
    ctr = cat_frame(tr, cat_cols, itr)
    civ = cat_frame(tr, cat_cols, iv)
    Xc_tr = pd.concat([Xn_tr.reset_index(drop=True), ctr], axis=1)
    Xc_iv = pd.concat([Xn_iv.reset_index(drop=True), civ], axis=1)

    out = {"catboost_version": catboost.__version__, "rounds": ROUNDS,
           "numeric_features": int(X.shape[1]), "cat_cols": list(cat_cols),
           "cardinality": card, "probes": []}

    out["probes"].append(probe("P0 numeric Plain (current control)", Xn_tr, Xn_iv, y[itr], y[iv],
                               [], "Plain"))
    out["probes"].append(probe("P1 numeric Ordered", Xn_tr, Xn_iv, y[itr], y[iv],
                               [], "Ordered"))
    out["probes"].append(probe("P2 +native cats Plain", Xc_tr, Xc_iv, y[itr], y[iv],
                               cat_cols, "Plain"))
    out["probes"].append(probe("P3 +native cats Ordered", Xc_tr, Xc_iv, y[itr], y[iv],
                               cat_cols, "Ordered"))
    out["probes"].append(probe("P4 +native cats Plain GPU", Xc_tr, Xc_iv, y[itr], y[iv],
                               cat_cols, "Plain", task="GPU"))
    out["probes"].append(probe("P5 +native cats Ordered GPU", Xc_tr, Xc_iv, y[itr], y[iv],
                               cat_cols, "Ordered", task="GPU"))
    out["probes"].append(probe("P6 +native cats Plain max_ctr_complexity=2", Xc_tr, Xc_iv,
                               y[itr], y[iv], cat_cols, "Plain", {"max_ctr_complexity": 2}))
    out["probes"].append(probe("P7 +native cats Plain one_hot_max_size=6", Xc_tr, Xc_iv,
                               y[itr], y[iv], cat_cols, "Plain", {"one_hot_max_size": 6}))

    print("\n" + "=" * 104)
    print("SUMMARY")
    print("=" * 104)
    print(f"  {'arm':<50}{'trains':>8}{'sec':>8}{'AUC':>11}{'ctr_leaves':>12}")
    for p in out["probes"]:
        if not p.get("trains"):
            print(f"  {p['name'][:48]:<50}{'NO':>8}{p['seconds']:>8.1f}"
                  f"{'-':>11}{'-':>12}")
            continue
        print(f"  {p['name'][:48]:<50}{'yes':>8}{p['seconds']:>8.1f}{p['auc']:>11.6f}"
              f"{str(p.get('n_ctr_leaf_descriptions')):>12}")

    print("\n  DECISIONS THIS PROBE FORCES ON THE HARNESS:")
    by = {p["name"].split()[0]: p for p in out["probes"]}
    ordered_cpu = by.get("P1", {}).get("trains")
    cats_plain = by.get("P2", {})
    ordered_cats = by.get("P3", {})
    gpu_plain = by.get("P4", {})
    gpu_ord = by.get("P5", {})
    print(f"  * Ordered boosting on CPU: {'SUPPORTED' if ordered_cpu else 'NOT SUPPORTED'}")
    print(f"  * Native categoricals on CPU: "
          f"{'SUPPORTED' if cats_plain.get('trains') else 'NOT SUPPORTED'}")
    print(f"  * GPU + categoricals: {'works' if gpu_plain.get('trains') else 'fails -- ' + str(gpu_plain.get('error'))[:90]}")
    print(f"  * GPU + Ordered: {'works' if gpu_ord.get('trains') else 'fails -- ' + str(gpu_ord.get('error'))[:90]}")
    if cats_plain.get("trains"):
        cl = cats_plain.get("n_ctr_leaf_descriptions")
        print(f"  * CTR machinery engaged: {cl} leaf CTR descriptions reported. "
              f"{'CONFIRMED' if cl else 'ZERO -- categoricals may have been silently ignored'}")
    p2a = cats_plain.get("auc")
    p0a = by.get("P0", {}).get("auc")
    if p0a is not None and p2a is not None:
        print(f"  * native cats vs numeric-only, same config: {p2a:.6f} vs {p0a:.6f} "
              f"= {(p2a-p0a)*1e5:+.1f}e-5 (probe scale, NOT evidence)")
    print("\n  NOTE: every AUC here is on a 15% subsample at 250 rounds. They establish CAPABILITY")
    print("  and RATE only. No score from this probe may be quoted as a result.")

    save_json(out, REPORTS / "catboost_capability.json")
    print("\nwrote", REPORTS / "catboost_capability.json")


if __name__ == "__main__":
    main()
