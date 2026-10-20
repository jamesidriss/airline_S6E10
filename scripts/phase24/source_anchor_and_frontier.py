"""Phase 24 -- information anchor of the ORIGINAL source distribution + achievable frontier.

Two read-only measurements, no competition labels used in any fit of the source part:

A. SOURCE ANCHOR. 5-fold CV on the original 129,880-row airline survey (the generator's
   training distribution), with the same LightGBM configuration family as the champion's
   auxiliary parent, plus a 50% subsample arm to separate "source ceiling at 130k" from
   "source still data-limited". Also the 13-rating-profile lookup AUC on the source, to be
   compared with the competition value 0.933454 (phase 7 structural probe).

B. FRONTIER. For every stored OOF vector: standalone pooled AUC, logit correlation with
   v7, and the best single-weight blend delta vs v7. Plots the (AUC, rho) -> delta surface
   that any future parent must beat.

Writes reports/phase24/source_anchor.json and reports/phase24/frontier.json.
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

from src.common import arr_sha256, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

ORIG = ROOT / "data/original/arseniyshutko__binary-aviation-satisfaction-129k/data.csv"
OUT_A = ROOT / "reports/phase24/source_anchor.json"
OUT_B = ROOT / "reports/phase24/frontier.json"
PRED = ROOT / "artifacts/predictions"
CHAMP = "v7_sol_tabpfn_internal2_aux10"

RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def source_anchor():
    og = pd.read_csv(ORIG)
    tgt = "satisfaction"
    y = (og[tgt].astype(str).str.lower().str.strip() == "satisfied").to_numpy().astype("int8")
    feats = [c for c in og.columns if c != tgt]
    X = og[feats].copy()
    # ordinal-encode strings deterministically (sorted category order)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    X = X.astype("float32").to_numpy()

    import lightgbm as lgb

    params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=255,
                  min_data_in_leaf=40, feature_fraction=0.45, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, extra_trees=True, verbose=-1, num_threads=8,
                  seed=20261010)

    def cv_auc(Xm, ym, n_splits=5, rounds=4000):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20261010)
        aucs, iters = [], []
        for tr_i, va_i in skf.split(Xm, ym):
            inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
            fit_i, es_i = next(iter(inner.split(Xm[tr_i], ym[tr_i])))
            dtr = lgb.Dataset(Xm[tr_i][fit_i], ym[tr_i][fit_i])
            des = lgb.Dataset(Xm[tr_i][es_i], ym[tr_i][es_i])
            m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[des],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            aucs.append(float(roc_auc_score(ym[va_i], m.predict(Xm[va_i], num_iteration=m.best_iteration))))
            iters.append(int(m.best_iteration))
        return float(np.mean(aucs)), aucs, iters

    t0 = time.time()
    full_auc, full_folds, full_iters = cv_auc(X, y)
    rng = np.random.default_rng(20261010)
    idx = rng.permutation(len(y))[: len(y) // 2]
    half_auc, half_folds, half_iters = cv_auc(X[idx], y[idx])

    # 13-rating profile lookup, leave-one-out group mean (source only)
    key = og[RATINGS].astype("int64").astype(str).agg("|".join, axis=1)
    grp = pd.DataFrame({"k": key, "y": y})
    agg = grp.groupby("k")["y"].agg(["sum", "count"])
    loo = (grp["k"].map(agg["sum"]) - grp["y"]) / (grp["k"].map(agg["count"]) - 1).replace(0, np.nan)
    mask = loo.notna().to_numpy()
    lookup_auc = float(roc_auc_score(y[mask], loo[mask])) if mask.sum() > 1000 else None
    purity = float(((agg["sum"] == 0) | (agg["sum"] == agg["count"])).mul(agg["count"]).sum() / agg["count"].sum())

    out = {
        "status": "COMPLETE",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(ORIG),
        "n_rows": int(len(y)),
        "pos_rate": float(y.mean()),
        "n_features": int(X.shape[1]),
        "cv5_auc_full": full_auc,
        "cv5_folds_full": full_folds,
        "best_iters_full": full_iters,
        "cv5_auc_half": half_auc,
        "cv5_folds_half": half_folds,
        "half_n": int(len(idx)),
        "source_doubling_gain": full_auc - half_auc,
        "rating_profile_lookup_auc": lookup_auc,
        "rating_profile_pure_fraction": purity,
        "rating_profile_n_groups": int(len(agg)),
        "cpu_seconds": time.time() - t0,
        "reading": (
            "This is the AUC the SOURCE distribution supports at 130k rows with a strong GBDT "
            "and at 65k rows to expose its own data limitation. It is a reference anchor for the "
            "synthetic competition distribution, not a competition score and not a Bayes ceiling."
        ),
    }
    save_json(out, OUT_A)
    print(json.dumps({k: v for k, v in out.items() if k not in ("cv5_folds_full", "cv5_folds_half", "best_iters_full")}, indent=1))


def frontier():
    tr, _ = load_cached_parquet()
    y = tr.satisfaction.to_numpy(dtype="int8")
    folds = get_scheme("primary", y, tr["id"].to_numpy()).folds
    v7 = np.load(PRED / f"{CHAMP}_oof.npy").astype(np.float64)
    base = roc_auc_score(y, v7)
    rows = []
    for p in sorted(PRED.glob("*_oof.npy")):
        name = p.name[: -len("_oof.npy")]
        if name == CHAMP:
            continue
        a = np.load(p)
        if a.shape != v7.shape or not np.isfinite(a).all():
            continue
        pv = a.astype(np.float64)
        if pv.min() < -1e-6 or pv.max() > 1 + 1e-6:
            pv = 1 / (1 + np.exp(-np.clip(pv, -60, 60)))
        lg = logit(pv)
        corr = float(np.corrcoef(lg, logit(v7))[0, 1])
        best = (-9.9, None)
        for w in (0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3):
            g = roc_auc_score(y, (1 - w) * logit(v7) + w * lg) - base
            if g > best[0]:
                best = (g, w)
        rows.append({"name": name, "auc": float(roc_auc_score(y, pv)), "corr": corr,
                     "best_w": best[1], "best_delta": best[0]})
    rows.sort(key=lambda r: -r["best_delta"])
    # frontier envelope: best delta among vectors with corr >= c
    env = {}
    for c in (0.990, 0.992, 0.994, 0.995, 0.996, 0.997, 0.998):
        cand = [r for r in rows if r["corr"] >= c]
        env[str(c)] = max(cand, key=lambda r: r["best_delta"])["best_delta"] if cand else None
    out = {
        "status": "COMPLETE",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "champion_auc": base,
        "n_vectors": len(rows),
        "frontier_envelope_best_delta_by_min_corr": env,
        "rows": rows,
        "reading": (
            "Any deployable third parent must sit on the right-hand side of this surface: high "
            "standalone AUC AND low logit correlation with the champion. The envelope reports the "
            "best measured blend delta among stored vectors at or above each correlation floor."
        ),
    }
    save_json(out, OUT_B)
    print(json.dumps({"n": len(rows), "envelope": env,
                      "top": [(r["name"], round(r["auc"], 6), round(r["corr"], 4), round(r["best_delta"], 7)) for r in rows[:10]]},
                     indent=1))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "source"):
        source_anchor()
    if which in ("both", "frontier"):
        frontier()
