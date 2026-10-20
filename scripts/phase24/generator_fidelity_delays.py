"""Phase 24 -- generator-fidelity part B: delay blocks and the no-delay source arm.

Completes the mechanism behind the source/synthetic AUC gap:
 1. delay-positive-rate blocks (delay==0 vs ranges) in both datasets, since 91% of
    synthetic delay values are exactly zero and quantile deciles collapse;
 2. a source-only 5-fold arm WITHOUT the two delay columns, to quantify how much of
    the source's 0.98881 is carried by the columns the generator rewrote.

Read-only; no competition labels are used in the source arm.

    .\\.venv\\Scripts\\python.exe scripts/phase24/generator_fidelity_delays.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import load_cached_parquet, save_json  # noqa: E402

ORIG = ROOT / "data/original/arseniyshutko__binary-aviation-satisfaction-129k/data.csv"
OUT = ROOT / "reports/phase24/generator_fidelity_delays.json"
DELAYS = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
BLOCKS = [(0, 0), (1, 5), (6, 15), (16, 30), (31, 60), (61, 120), (121, 10**9)]


def block_rates(x, y):
    x = np.asarray(x, dtype=float)
    out = []
    for lo, hi in BLOCKS:
        m = np.isfinite(x) & (x >= lo) & (x <= hi)
        out.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                    "pos_rate": float(y[m].mean()) if m.sum() >= 50 else None})
    return out


def main() -> None:
    t0 = time.time()
    tr, _ = load_cached_parquet()
    yc = tr.satisfaction.to_numpy(dtype="int8")
    og = pd.read_csv(ORIG)
    ys = og["satisfaction"].to_numpy().astype("int8")

    out = {"status": "COMPLETE", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "delay_blocks": {}, "source_arms": {}}

    for c in DELAYS:
        out["delay_blocks"][c] = {
            "source": block_rates(og[c], ys),
            "synthetic": block_rates(tr[c], yc),
        }

    feats = [c for c in og.columns if c != "satisfaction"]
    nod = [c for c in feats if c not in DELAYS]
    X = og[feats].copy()
    Xn = og[nod].copy()
    for frame in (X, Xn):
        for c in frame.columns:
            if not pd.api.types.is_numeric_dtype(frame[c]):
                frame[c] = frame[c].astype("category").cat.codes
    X = X.astype("float32").to_numpy()
    Xn = Xn.astype("float32").to_numpy()

    import lightgbm as lgb

    params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=255,
                  min_data_in_leaf=40, feature_fraction=0.45, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, extra_trees=True, verbose=-1,
                  num_threads=8, seed=20261010)

    def cv(Xm):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261010)
        aucs = []
        for tr_i, va_i in skf.split(Xm, ys):
            inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
            fit_i, es_i = next(iter(inner.split(Xm[tr_i], ys[tr_i])))
            m = lgb.train(params, lgb.Dataset(Xm[tr_i][fit_i], ys[tr_i][fit_i]),
                          num_boost_round=4000,
                          valid_sets=[lgb.Dataset(Xm[tr_i][es_i], ys[tr_i][es_i])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            aucs.append(float(roc_auc_score(ys[va_i], m.predict(Xm[va_i], num_iteration=m.best_iteration))))
        return float(np.mean(aucs)), aucs

    a_full, f_full = cv(X)
    a_nod, f_nod = cv(Xn)
    out["source_arms"] = {
        "with_delays": {"auc": a_full, "folds": f_full},
        "without_delays": {"auc": a_nod, "folds": f_nod},
        "delay_contribution": a_full - a_nod,
        "n_features_with": len(feats), "n_features_without": len(nod),
    }
    out["cpu_seconds"] = time.time() - t0
    save_json(out, OUT)
    print(json.dumps({"source_arms": out["source_arms"], "cpu_seconds": out["cpu_seconds"]}, indent=1))
    for c in DELAYS:
        print("\n", c)
        print("  source   ", [(b["lo"], b["hi"], b["n"], None if b["pos_rate"] is None else round(b["pos_rate"], 3))
                               for b in out["delay_blocks"][c]["source"]])
        print("  synthetic", [(b["lo"], b["hi"], b["n"], None if b["pos_rate"] is None else round(b["pos_rate"], 3))
                               for b in out["delay_blocks"][c]["synthetic"]])


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"refuse to overwrite {OUT}")
    main()
