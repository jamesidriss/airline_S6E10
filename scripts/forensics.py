"""Data forensics: schema, exact-value structure, digits, drift, adversarial validation.

Everything here is target-free EXCEPT the explicitly-labelled target-rate surface and the
adversarial classifier (whose target is the train/test origin flag, never `satisfaction`).

Output: reports/forensics_report.md + reports/forensics/*.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, TARGET, ID_COL, load_cached_parquet, save_json  # noqa: E402

OUT = REPORTS / "forensics"


def sec(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def col_profile(df: pd.DataFrame, name: str) -> dict:
    s = df[name]
    d = {
        "dtype": str(s.dtype),
        "n_null": int(s.isna().sum()),
        "n_unique": int(s.nunique(dropna=False)),
    }
    if pd.api.types.is_numeric_dtype(s) and s.dtype.kind == "f":
        v = s.dropna()
        d.update(
            min=float(v.min()), max=float(v.max()),
            mean=float(v.mean()), std=float(v.std()),
            all_integer=bool(np.all(np.equal(np.mod(v, 1), 0))),
            n_decimals_max=int(max((len(str(x).split(".")[1]) if "." in str(x) else 0) for x in v.head(50000))),
        )
    return d


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr, te = load_cached_parquet()
    rep: dict = {}

    # ---------------------------------------------------------------- schema
    sec("SCHEMA")
    print("train", tr.shape, "test", te.shape)
    print("train cols:", list(tr.columns))
    print("test  cols:", list(te.columns))
    assert list(te.columns) == [c for c in tr.columns if c != TARGET], "column mismatch"
    print("positive rate:", float(tr[TARGET].mean()))
    print("train id range:", int(tr[ID_COL].min()), int(tr[ID_COL].max()), "unique", tr[ID_COL].nunique())
    print("test  id range:", int(te[ID_COL].min()), int(te[ID_COL].max()), "unique", te[ID_COL].nunique())
    ids_all = pd.concat([tr[ID_COL], te[ID_COL]])
    print("total id min/max/unique:", int(ids_all.min()), int(ids_all.max()), ids_all.nunique(),
          "contiguous:", bool((np.sort(ids_all.values) == np.arange(ids_all.min(), ids_all.max() + 1)).all()))

    feats = [c for c in te.columns if c != ID_COL]
    prof = {c: {"train": col_profile(tr, c), "test": col_profile(te, c)} for c in feats}
    rep["profiles"] = prof

    # ---------------------------------------------------------------- exact value reuse
    sec("EXACT VALUE VOCABULARIES (train vs test)")
    reuse = {}
    for c in feats:
        vtr = set(tr[c].dropna().unique().tolist())
        vte = set(te[c].dropna().unique().tolist())
        reuse[c] = {
            "n_train_vals": len(vtr),
            "n_test_vals": len(vte),
            "test_vals_unseen_in_train": len(vte - vtr),
            "frac_test_rows_unseen_value": float(te[c].isin(vte - vtr).mean()),
        }
    print(pd.DataFrame(reuse).T.to_string())
    rep["value_reuse"] = reuse

    # ---------------------------------------------------------------- value counts / digit structure
    sec("SURVEY-LIKE COLUMNS: value counts (train | test)")
    survey_like = [c for c in feats if prof[c]["train"]["n_unique"] <= 20]
    print("survey-like (n_unique<=20):", survey_like)
    vc = {}
    for c in survey_like:
        a = tr[c].value_counts(dropna=False).sort_index()
        b = te[c].value_counts(dropna=False).sort_index()
        t = pd.DataFrame({"train": a, "test": b}).fillna(0).astype(int)
        t["train_frac"] = (t["train"] / t["train"].sum()).round(5)
        t["test_frac"] = (t["test"] / t["test"].sum()).round(5)
        t["diff"] = (t["test_frac"] - t["train_frac"]).round(5)
        vc[c] = t.reset_index().rename(columns={"index": "value"}).to_dict("records")
        print(f"\n--- {c} ---")
        print(t.to_string())
    rep["survey_value_counts"] = vc

    # ---------------------------------------------------------------- numeric digit structure
    sec("INTEGER STRUCTURE (are values integer / quantized?)")
    ints = {}
    for c in feats:
        p = prof[c]["train"]
        if p.get("all_integer"):
            s = tr[c]
            nv = p["n_unique"]
            ints[c] = {
                "n_unique": nv,
                "min": p["min"], "max": p["max"],
                "top10_share": float(s.value_counts(normalize=True).head(10).sum()),
                "top1_share": float(s.value_counts(normalize=True).iloc[0]),
            }
    print(pd.DataFrame(ints).T.to_string())
    rep["integer_columns"] = ints

    # ---------------------------------------------------------------- target rate surfaces
    sec("TARGET RATE BY CATEGORY (single column)")
    tr_s = {}
    for c in feats:
        if prof[c]["train"]["n_unique"] <= 25:
            g = tr.groupby(c, dropna=False, observed=True)[TARGET].agg(["mean", "size"]).reset_index()
            g.columns = [c, "pos_rate", "n"]
            g = g.sort_values("pos_rate", ascending=False)
            tr_s[c] = g.to_dict("records")
            print(f"\n--- {c} --- (spread {g['pos_rate'].max()-g['pos_rate'].min():.4f})")
            print(g.head(12).to_string(index=False))
    rep["target_rate_surfaces"] = tr_s

    # ---------------------------------------------------------------- marginal drift (KS)
    sec("TRAIN vs TEST MARGINAL DRIFT")
    from scipy import stats

    ks = []
    for c in feats:
        a = pd.to_numeric(tr[c], errors="coerce").dropna().values.astype("float64")
        b = pd.to_numeric(te[c], errors="coerce").dropna().values.astype("float64")
        st, p = stats.ks_2samp(a, b)
        ks.append({"col": c, "ks": float(st), "p": float(p),
                   "mean_train": float(np.mean(a)), "mean_test": float(np.mean(b)),
                   "std_train": float(np.std(a)), "std_test": float(np.std(b))})
    ksd = pd.DataFrame(ks).sort_values("ks", ascending=False)
    print(ksd.to_string(index=False))
    rep["marginal_ks"] = ks

    # ---------------------------------------------------------------- id analysis
    sec("ID ANALYSIS (train)")
    idt = tr[ID_COL].values
    print("corr(id, target) pearson:", float(np.corrcoef(idt, tr[TARGET].values)[0, 1]))
    # does id predict target at all? binned pos-rate
    nb = 20
    b = pd.cut(tr[ID_COL], nb)
    g = tr.groupby(b, observed=True)[TARGET].agg(["mean", "size"])
    print("pos-rate by id bin (20 bins):")
    print(g.round(5).to_string())
    rep["id_bin_posrate"] = g.reset_index().astype(str).to_dict("records")
    rep["id_target_corr"] = float(np.corrcoef(idt, tr[TARGET].values)[0, 1])

    # id modulo / digit predictive power (screening)
    idfeat = {
        "id_mod2": idt % 2, "id_mod3": idt % 3, "id_mod5": idt % 5, "id_mod7": idt % 7,
        "id_mod10": idt % 10, "id_mod100": idt % 100,
        "id_div1e4": idt // 10000, "id_div1e5": idt // 100000, "id_div1e3": idt // 1000,
        "id_units": idt % 10, "id_tens": (idt // 10) % 10, "id_hund": (idt // 100) % 10,
    }
    print("\nAUC of id-derived feature for target (screen; 0.5 = no signal):")
    from sklearn.metrics import roc_auc_score

    for k, v in idfeat.items():
        try:
            a = roc_auc_score(tr[TARGET].values, v)
            print(f"  {k:12s} auc={max(a, 1-a):.5f}")
            rep.setdefault("id_feature_screen", {})[k] = float(max(a, 1 - a))
        except Exception as e:  # noqa: BLE001
            print(" ", k, e)

    # ---------------------------------------------------------------- adversarial validation
    sec("ADVERSARIAL VALIDATION (train vs test)")
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    both = pd.concat([tr.drop(columns=[TARGET]), te], axis=0, ignore_index=True)
    is_te = np.r_[np.zeros(len(tr), dtype=np.int8), np.ones(len(te), dtype=np.int8)]
    # subsample for speed
    rng = np.random.default_rng(0)
    idx = rng.choice(len(both), size=300_000, replace=False)
    Xa = both.iloc[idx].copy()
    ya = is_te[idx]
    Xa = Xa.drop(columns=[ID_COL])
    for c in Xa.columns:
        if Xa[c].dtype.kind in "OUS":
            Xa[c] = pd.factorize(Xa[c], sort=True)[0].astype("float32")
        elif Xa[c].dtype.kind == "f":
            Xa[c] = Xa[c].astype("float32")
    skf = StratifiedKFold(3, shuffle=True, random_state=0)
    oof = np.zeros(len(Xa))
    for k, (a, b) in enumerate(skf.split(Xa, ya)):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63, verbose=-1,
                               n_jobs=8)
        m.fit(Xa.iloc[a], ya[a])
        oof[b] = m.predict_proba(Xa.iloc[b])[:, 1]
    adv = roc_auc_score(ya, oof)
    print(f"adversarial AUC (no id, 300k sample, 3-fold lgbm): {adv:.5f}")
    rep["adversarial_auc_no_id"] = float(adv)

    # with id-derived features
    Xb = Xa.copy()
    Xb["id_div1e5"] = (both.iloc[idx][ID_COL].values // 100000).astype("int32")
    Xb["id_div1e4"] = (both.iloc[idx][ID_COL].values // 10000).astype("int32")
    Xb["id_mod1000"] = (both.iloc[idx][ID_COL].values % 1000).astype("int32")
    oof2 = np.zeros(len(Xb))
    for k, (a, b) in enumerate(skf.split(Xb, ya)):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63, verbose=-1, n_jobs=8)
        m.fit(Xb.iloc[a], ya[a])
        oof2[b] = m.predict_proba(Xb.iloc[b])[:, 1]
    adv2 = roc_auc_score(ya, oof2)
    print(f"adversarial AUC (with id-block feats):                  {adv2:.5f}")
    rep["adversarial_auc_with_id"] = float(adv2)

    save_json(rep, OUT / "forensics.json")
    print("\nwrote", OUT / "forensics.json")


if __name__ == "__main__":
    main()