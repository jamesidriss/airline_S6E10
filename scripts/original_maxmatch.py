"""Maximise recoverable signal from the original real dataset.

Key discovery: a large fraction of synthetic rows are *copies* of original rows on the
13 discrete survey ratings (+ the 4 categorical columns), sometimes uniquely.
This script searches for the row key that maximises usable coverage x label purity,
then measures how much AUC that adds on top of a raw-feature LightGBM.

All statistics used to build features come from ORIGINAL labels only, so no synthetic
validation label is involved (no leakage). Rules S6E10 allow external data.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, TARGET, ID_COL, load_cached_parquet, save_json, set_seed  # noqa: E402
from scripts.analyze_original import norm_cols  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORIG_PATH = ROOT / "data" / "original" / "arseniyshutko__binary-aviation-satisfaction-129k" / "data.csv"

SURVEY13 = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
            "Gate location", "Food and drink", "Online boarding", "Seat comfort",
            "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
            "Checkin service", "Cleanliness"]
META4 = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUMX = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]


def load_all():
    tr, te = load_cached_parquet()
    feats = [c for c in te.columns if c != ID_COL]
    og = norm_cols(pd.read_csv(ORIG_PATH))
    return tr, te, feats, og


def _dedup(cols: list[str]) -> list[str]:
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def hkey(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    cols = _dedup(cols)
    d = df[cols].copy()
    for c in cols:
        if d[c].dtype.kind == "f":
            d[c] = d[c].astype("float64").round(6)
        elif d[c].dtype.kind in "OU":
            d[c] = d[c].astype(str)
    return pd.util.hash_pandas_object(d.astype(object), index=False)


def main() -> None:
    rep: dict = {}
    tr, te, feats, og = load_all()
    yo = og[TARGET].astype(int).values
    ntr = len(tr)
    y = tr[TARGET].values

    # ---------------- key search
    print("=" * 118)
    print("KEY SEARCH: coverage / uniqueness / label-purity of exact original-row lookups")
    print("=" * 118)
    print(f"{'key':<62} {'cov':>7} {'uniq':>7} {'pure_uniq':>9} {'agree':>7}")
    results = {}
    candidates: list[list[str]] = []
    candidates.append(SURVEY13)
    candidates.append(META4)
    candidates.append(META4 + ["Age"])
    candidates.append(META4 + ["Flight Distance"])
    candidates.append(META4 + ["Age", "Flight Distance"])
    candidates.append(SURVEY13 + META4)
    candidates.append(SURVEY13 + ["Class"])
    candidates.append(SURVEY13 + ["Class", "Online boarding"])
    candidates.append(SURVEY13 + META4 + ["Age"])
    candidates.append(SURVEY13 + META4 + ["Flight Distance"])
    candidates.append(SURVEY13 + META4 + ["Age", "Flight Distance"])
    candidates.append(META4 + ["Departure Delay in Minutes"])
    candidates.append(META4 + ["Age", "Departure Delay in Minutes"])
    candidates.append(SURVEY13 + META4 + ["Departure Delay in Minutes"])
    candidates.append(SURVEY13 + META4 + ["Age", "Departure Delay in Minutes"])
    candidates.append(SURVEY13 + META4 + NUMX)
    candidates.append(META4 + NUMX)

    for cols in candidates:
        cols = _dedup(cols)
        nm = "+".join(["S13" if c in SURVEY13 else c for c in cols])
        nm = nm.replace("+S13+S13", "+S13").replace("S13+S13", "S13")
        if nm in results:
            continue
        ho = hkey(og, cols)
        hs_all = pd.concat([hkey(tr, cols), hkey(te, cols)], ignore_index=True)
        d = pd.DataFrame({"h": ho.values, "y": yo})
        g = d.groupby("h")["y"].agg(["size", "mean"])
        p = hs_all.map(g["mean"]).values
        n = hs_all.map(g["size"]).values
        cov = float((~np.isnan(p)).mean())
        uniq = n == 1
        u = uniq[:ntr]
        agree = float(((p[:ntr][u] > 0.5) == (y[u] == 1)).mean()) if u.sum() > 100 else float("nan")
        purity = float(np.nanmean(np.abs(p - 0.5) * 2)) if cov else 0.0
        results[nm] = {"cols": cols, "coverage": round(cov, 5), "unique_frac": round(float(uniq.mean()), 5),
                       "label_agreement_unique": None if np.isnan(agree) else round(agree, 5),
                       "mean_label_confidence": round(purity, 5),
                       "auc_unique_subset": round(float(roc_auc_score(y[u], p[:ntr][u])), 5) if u.sum() > 100 else None}
        print(f"{nm:<62} {cov:>7.4f} {float(uniq.mean()):>7.4f} {agree:>9.5f} {purity:>7.4f}")

    rep["key_search"] = results

    # ---------------- tuple-frequency fingerprints (copy evidence)
    print("\n" + "=" * 118)
    print("SURVEY13 TUPLE FREQUENCY: synthetic vs original (copy evidence)")
    print("=" * 118)
    syn = pd.concat([tr[SURVEY13], te[SURVEY13]], ignore_index=True)
    co = og[SURVEY13].value_counts()
    cs = syn[SURVEY13].value_counts()
    cmp = pd.DataFrame({"orig": co, "syn": cs}).fillna(0).astype(int)
    cmp["syn_exp"] = cmp["syn"] / cmp["syn"].sum()
    cmp["orig_exp"] = cmp["orig"] / cmp["orig"].sum()
    cmp["ratio"] = (cmp["syn_exp"] / cmp["orig_exp"].replace(0, np.nan))
    cmp = cmp.sort_values("syn", ascending=False)
    print(cmp.head(25).to_string())
    print(f"\n#distinct survey13 tuples: original={len(co)}, synthetic={len(cs)}")
    print(f"top-1 tuple: original {co.iloc[0]} ({co.iloc[0]/co.sum():.5f}), synthetic {cs.iloc[0]} ({cs.iloc[0]/cs.sum():.5f})")
    print(f"fraction of synthetic rows whose tuple is in original: {syn[SURVEY13].apply(tuple, axis=1).isin(set(map(tuple, co.index))).mean() if len(syn)<3000 else float(cs.reindex(co.index).fillna(0).sum()/cs.sum()):.5f}")
    rep["tuple_freq"] = {"n_orig_tuples": int(len(co)), "n_syn_tuples": int(len(cs)),
                         "top1_orig": int(co.iloc[0]), "top1_syn": int(cs.iloc[0]),
                         "head": cmp.head(40).reset_index().astype(str).to_dict("records")}

    # ---------------- how much AUC does adding the lookup give?
    print("\n" + "=" * 118)
    print("AUC IMPACT: raw features vs raw + original-lookup features (primary 5-fold)")
    print("=" * 118)
    from src.validation.folds import get_scheme
    import lightgbm as lgb

    fs = get_scheme("primary", y, tr[ID_COL])
    folds = fs.folds

    def raw_matrix(df):
        X = df[feats].copy()
        for c in X.columns:
            if X[c].dtype.kind in "OUS":
                X[c] = pd.factorize(X[c], sort=True)[0].astype("float32")
            else:
                X[c] = X[c].astype("float32")
        return X.fillna(-1)

    # build lookup features from ORIGINAL labels only
    lookup_cols = [_dedup([SURVEY13 + META4, SURVEY13 + META4 + ["Age"], SURVEY13 + META4 + ["Flight Distance"],
                   SURVEY13 + META4 + NUMX, META4 + ["Age", "Flight Distance"], META4, META4 + ["Age"]]) for _ in range(0)] or [
        _dedup(SURVEY13 + META4), _dedup(SURVEY13 + META4 + ["Age"]),
        _dedup(SURVEY13 + META4 + ["Flight Distance"]), _dedup(SURVEY13 + META4 + NUMX),
        _dedup(META4 + ["Age", "Flight Distance"]), _dedup(META4), _dedup(META4 + ["Age"]),
    ]
    lookup_prior_w = 40.0

    def build_lookup(syn_df: pd.DataFrame):
        hs_all = pd.concat([hkey(tr, cols), hkey(te, cols)], ignore_index=True)
        ho = hkey(og, cols)
        d = pd.DataFrame({"h": ho.values, "y": yo})
        g = d.groupby("h")["y"].agg(["sum", "size"])
        pr = (g["sum"] + lookup_prior_w * yo.mean()) / (g["size"] + lookup_prior_w)
        size = g["size"]
        v = hs_all.map(pr).values.astype("float32")
        cnt = hs_all.map(size).values.astype("float32")
        base = yo.mean()
        out_v = np.where(np.isnan(v), base, v).astype("float32")
        out_c = np.where(np.isnan(cnt), 0.0, np.log1p(np.nan_to_num(cnt))).astype("float32")
        uniq_flag = np.where(np.nan_to_num(cnt, nan=0.0) == 1, 1.0, 0.0).astype("float32")
        return np.column_stack([out_v, out_c, uniq_flag])

    Ltr = build_lookup(None)[:ntr]
    print("lookup feature matrix:", Ltr.shape)
    cov_any = ((Ltr[:, 1] > 0) | (Ltr[:, 2] > 0)).mean()
    print(f"rows with at least one original match: {cov_any:.4f}")

    Xtr = raw_matrix(tr)
    Xte = raw_matrix(te)
    variants = {
        "raw": (Xtr.values.astype("float32"), None),
        "raw+lookup": (np.column_stack([Xtr.values, Ltr]), Ltr),
    }
    for vname, (M, _) in variants.items():
        oof = np.zeros(ntr)
        for k in range(5):
            a = np.where(folds != k)[0]
            b = np.where(folds == k)[0]
            m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.04, num_leaves=63,
                                   colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                                   n_jobs=8, verbose=-1, random_state=1)
            m.fit(M[a], y[a])
            oof[b] = m.predict_proba(M[b])[:, 1]
        auc = roc_auc_score(y, oof)
        print(f"  {vname:<28} OOF AUC = {auc:.6f}")
        rep.setdefault("auc", {})[vname] = round(float(auc), 6)
        np.save(REPORTS.parent / "artifacts" / "predictions" / f"probe_{vname}_oof.npy", oof.astype("float32"))

    save_json(rep, REPORTS / "original_maxmatch.json")
    print("\nwrote", REPORTS / "original_maxmatch.json")


if __name__ == "__main__":
    main()