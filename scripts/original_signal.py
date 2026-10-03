"""Quantify the value of the original real dataset.

Three questions:
  Q1  What is the Bayes-ish ceiling? 5-fold AUC on the ORIGINAL data with the 21 features.
  Q2  How informative is an exact 14-survey-tuple lookup into the original data?
        - how many synthetic rows have a matching original row
        - for unique matches, does the original label agree with the synthetic label?
  Q3  What is the predictive power of smoothed P(satisfaction | key) statistics computed
      from ORIGINAL labels only (no synthetic validation label involved)?

Output: reports/original_signal.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, TARGET, ID_COL, load_cached_parquet, save_json  # noqa: E402
from scripts.analyze_original import norm_cols  # noqa: E402

ORIG_PATH = Path(__file__).resolve().parents[1] / "data" / "original" / \
    "arseniyshutko__binary-aviation-satisfaction-129k" / "data.csv"

SURVEY = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
          "Gate location", "Food and drink", "Online boarding", "Seat comfort",
          "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
          "Checkin service", "Cleanliness", "Cleanliness"][:13] + ["Cleanliness"]
SURVEY = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
          "Gate location", "Food and drink", "Online boarding", "Seat comfort",
          "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
          "Checkin service", "Cleanliness"]
META_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]


def main() -> None:
    rep: dict = {}
    tr, te = load_cached_parquet()
    feats = [c for c in te.columns if c != ID_COL]
    og = norm_cols(pd.read_csv(ORIG_PATH))
    print("original", og.shape, list(og.columns))
    print("label counts:", og[TARGET].value_counts().to_dict())

    # ---------------------------------------------------------------- Q1
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    Xo = og[feats].copy()
    yo = og[TARGET].astype(int).values
    for c in Xo.columns:
        if Xo[c].dtype.kind in "OUS":
            Xo[c] = pd.factorize(Xo[c], sort=True)[0].astype("float32")
        else:
            Xo[c] = Xo[c].astype("float32")
    Xo = Xo.fillna(-1)
    skf = StratifiedKFold(5, shuffle=True, random_state=20261010)
    oof = np.zeros(len(Xo))
    for a, b in skf.split(Xo, yo):
        m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                               colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                               n_jobs=8, verbose=-1)
        m.fit(Xo.iloc[a], yo[a])
        oof[b] = m.predict_proba(Xo.iloc[b])[:, 1]
    rep["original_lgbm_auc"] = float(roc_auc_score(yo, oof))
    print(f"\n[Q1] ORIGINAL-data 5-fold LightGBM AUC (21 features): {rep['original_lgbm_auc']:.6f}")

    # same but survey-only
    Xs = Xo[SURVEY]
    oofs = np.zeros(len(Xs))
    for a, b in skf.split(Xs, yo):
        m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, n_jobs=8, verbose=-1)
        m.fit(Xs.iloc[a], yo[a])
        oofs[b] = m.predict_proba(Xs.iloc[b])[:, 1]
    rep["original_lgbm_auc_survey_only"] = float(roc_auc_score(yo, oofs))
    print(f"[Q1b] ORIGINAL-data 5-fold LightGBM AUC (13 survey cols only): {rep['original_lgbm_auc_survey_only']:.6f}")

    # ---------------------------------------------------------------- Q2
    print("\n[Q2] exact survey-tuple lookup into original data")
    syn = pd.concat([tr[feats], te[feats]], ignore_index=True)
    isyn = np.r_[np.ones(len(tr), np.int8), np.zeros(len(te), np.int8)]

    key = [SURVEY, SURVEY + META_COLS, SURVEY + META_COLS + ["Age", "Flight Distance"]]
    names = ["survey13", "survey13+meta4", "survey13+meta4+age+dist"]
    rep["lookups"] = {}
    for cols, nm in zip(key, names):
        ho = pd.util.hash_pandas_object(og[cols].astype(object).round(6) if og[cols].dtypes.astype(str).str.contains("float").any() else og[cols].astype(object), index=False)
        hs = pd.util.hash_pandas_object(syn[cols].astype(object), index=False)
        df_o = pd.DataFrame({"h": ho.values, "y": yo})
        grp = df_o.groupby("h")["y"].agg(["size", "mean"])
        lut = grp["mean"]
        sizes = grp["size"]
        p = hs.map(lut).values
        n = hs.map(sizes).values
        frac_any = float((~np.isnan(p)).mean())
        uniq = n == 1
        frac_uniq = float(uniq.mean())
        auc_all = roc_auc_score(isyn, np.nan_to_num(p, nan=0.5))
        yy = tr[TARGET].values
        m_uni = uniq[: len(tr)]
        res = {
            "frac_synth_with_match": round(frac_any, 5),
            "frac_synth_unique_match": round(frac_uniq, 5),
            "auc_of_lut_on_synthetic_train_labels": round(float(auc_all), 5),
            "n_unique_keys": int(len(lut)),
        }
        if m_uni.sum() > 100:
            pu = p[: len(tr)][m_uni]
            res["label_agreement_on_unique_matches"] = round(float((( pu > 0.5 ) == (yy[m_uni] == 1)).mean()), 5)
            res["auc_on_unique_matches_only"] = round(float(roc_auc_score(yy[m_uni], pu)), 5)
            res["n_unique_matches_in_train"] = int(m_uni.sum())
        rep["lookups"][nm] = res
        print(f"  {nm:<26} match={frac_any:.4f} unique={frac_uniq:.4f} "
              f"auc={res['auc_of_lut_on_synthetic_train_labels']:.5f} "
              f"agree={res.get('label_agreement_on_unique_matches','-')} "
              f"auc_uniq={res.get('auc_on_unique_matches_only','-')}")

    # ---------------------------------------------------------------- Q3
    # Smoothed P(y|key) from ORIGINAL labels, single-column and survey-block, no synthetic y used.
    print("\n[Q3] smoothed original-derived target statistics")
    def te_smooth(cols, prior_w=50.0):
        ho = pd.util.hash_pandas_object(og[cols].astype(object), index=False)
        hs = pd.util.hash_pandas_object(syn[cols].astype(object), index=False)
        d = pd.DataFrame({"h": ho.values, "y": yo})
        s = d.groupby("h")["y"].agg(["sum", "size"])
        p = (s["sum"] + prior_w * yo.mean()) / (s["size"] + prior_w)
        return hs.map(p).values

    feats_from_orig = {}
    for c in feats:
        v = te_smooth([c])
        feats_from_orig[f"origTE__{c}"] = v
    for cols, nm in [(SURVEY, "survey13"), (META_COLS, "meta4")]:
        v = te_smooth(cols)
        feats_from_orig[f"origTE__{nm}"] = v
    for cols, nm in [(["Class", "Online boarding"], "Class_x_OB"),
                     (["Type of Travel", "Customer Type"], "ToT_x_CT"),
                     (["Class", "Online boarding", "Seat comfort"], "Class_OB_SC")]:
        v = te_smooth(cols)
        feats_from_orig[f"origTE__{nm}"] = v

    rep["orig_te_aucs"] = {}
    yy = tr[TARGET].values
    for nm, v in feats_from_orig.items():
        vv = v[: len(tr)]
        vv = np.nan_to_num(vv, nan=float(np.nanmean(vv)))
        try:
            a = roc_auc_score(yy, vv)
        except Exception:  # noqa: BLE001
            continue
        rep["orig_te_aucs"][nm] = round(float(max(a, 1 - a)), 5)
    for nm, a in sorted(rep["orig_te_aucs"].items(), key=lambda x: -x[1]):
        print(f"   {nm:<40} auc={a:.5f}  (nan frac {np.isnan(feats_from_orig[nm]).mean():.4f})")

    # combined "prior" score
    keys = ["Class_x_OB", "ToT_x_CT", "survey13", "Online boarding", "Type of Travel", "Class"]
    have = [k for k in keys if f"origTE__{k}" in feats_from_orig]
    M = np.column_stack([np.nan_to_num(feats_from_orig[f"origTE__{k}"][: len(tr)], nan=0.5) for k in have])
    import lightgbm as lgb2
    oof2 = np.zeros(len(tr))
    for a, b in skf.split(M, yy):
        m = lgb2.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, n_jobs=8, verbose=-1)
        m.fit(M[a], yy[a])
        oof2[b] = m.predict_proba(M[b])[:, 1]
    rep["origTE_combo_auc"] = round(float(roc_auc_score(yy, oof2)), 6)
    print(f"\n[Q3b] stacked original-derived TE features only: AUC={rep['origTE_combo_auc']:.6f}")

    save_json(rep, REPORTS / "original_signal.json")
    np.savez_compressed(
        REPORTS.parent / "artifacts" / "features" / "orig_te_features.npz",
        **{k: v.astype("float32") for k, v in feats_from_orig.items()},
    )
    print("\nwrote", REPORTS / "original_signal.json")


if __name__ == "__main__":
    main()