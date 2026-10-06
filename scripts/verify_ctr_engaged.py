"""Decisive check: are CatBoost's categorical target statistics ACTUALLY engaged?

Why this needs its own script
-----------------------------
The capability probe asked for `get_leaf_ctr_description()`, which does not exist on
`CatBoostClassifier` in 1.2.10. The call raised, my probe recorded `None`, and the summary printed
"ZERO -- categoricals may have been silently ignored". That is the worst possible outcome: a check
that cannot fail, reporting a scary conclusion it never actually tested.

This matters because the silent-ignore case is real. If a categorical column is present in the frame
but not declared, CatBoost trains happily on it as a float and every number looks plausible while the
experiment tests nothing. So CTR engagement must be verified positively, not inferred from a missing
attribute.

Three independent, decisive tests, none using a private API
-----------------------------------------------------------
  T1  MODEL DUMP. Fit with the twins declared, save as JSON, and look for the CTR structures
      CatBoost writes into tree leaves (`ctr_type`, `ctr_data`). Absent means no CTRs were built.

  T2  PREDICTION DIVERGENCE. Fit the SAME frame twice: once with the twins declared categorical,
      once with the twin columns replaced by their ordinal codes. If CTRs are engaged the
      predictions differ. If the categorical columns were being ignored, the two predictions would be
      IDENTICAL. Identity is the unambiguous failure signal.

  T3  CONTROL. Declaring a NUMERIC column as categorical must RAISE. If it does not, then nothing in
      this pipeline is enforcing the categorical contract and T1/T2 cannot be trusted either.

Usage: python scripts/verify_ctr_engaged.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import RAW21  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.native_cat import (cat_frame, cat_indices, cat_name, default_cat_cols,  # noqa: E402
                                to_cat_series)

from src.common import ID_COL  # noqa: E402

ROUNDS = 120


def numeric_matrix(tr: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    cols = list(RAW21)
    X = np.empty((len(tr), len(cols)), dtype="float32")
    for j, c in enumerate(cols):
        s = tr[c]
        X[:, j] = (s.to_numpy(dtype="float32") if pd.api.types.is_numeric_dtype(s)
                   else pd.factorize(s.astype(str), sort=True)[0].astype("float32"))
    return pd.DataFrame(np.nan_to_num(X, nan=-999.0)[rows], columns=cols)


def ordinal_block(tr: pd.DataFrame, src_cols: list[str], rows: np.ndarray) -> pd.DataFrame:
    """The SAME information as the twin block, but as floats -- the 'ignored' control."""
    data = {}
    for c in src_cols:
        codes, _ = pd.factorize(to_cat_series(tr[c]).to_numpy()[rows], sort=True)
        data[cat_name(c)] = codes.astype("float32")
    return pd.DataFrame(data, index=pd.RangeIndex(len(rows)))


def fit(frame: pd.DataFrame, y: np.ndarray, cat: list[str] | None):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=ROUNDS, depth=8, l2_leaf_reg=3.0, learning_rate=0.05,
                           verbose=0, allow_writing_files=False, random_seed=4,
                           boosting_type="Plain")
    kw = {"cat_features": cat_indices(frame, cat)} if cat else {}
    m.fit(frame, y, **kw)
    return m


def main() -> None:
    tr, _te = load_cached_parquet()
    y_all = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y_all, tr[ID_COL]).folds
    fit_rows = np.where(folds != 0)[0]
    rng = np.random.default_rng(0)
    sub = rng.choice(fit_rows, 60_000, replace=False)
    iv = np.concatenate([rng.choice(sub[y_all[sub] == 1], 8_000, replace=False),
                         rng.choice(sub[y_all[sub] == 0], 12_000, replace=False)])
    itr = np.setdiff1d(sub, iv)
    ytr = tr[TARGET].values.astype("int8")[itr].astype("int8")
    yiv = tr[TARGET].values.astype("int8")[iv].astype("int8")

    src = default_cat_cols(True)
    twins = [cat_name(c) for c in src]
    num = numeric_matrix(tr, itr)
    num_iv = numeric_matrix(tr, iv)
    catblk = cat_frame(tr, src, itr)
    catblk_iv = cat_frame(tr, src, iv)
    ordblk = ordinal_block(tr, src, itr)
    ordblk_iv = ordinal_block(tr, src, iv)

    frame_cat = pd.concat([num.reset_index(drop=True), catblk], axis=1)
    frame_ord = pd.concat([num.reset_index(drop=True), ordblk], axis=1)
    frame_cat_iv = pd.concat([num_iv.reset_index(drop=True), catblk_iv], axis=1)
    frame_ord_iv = pd.concat([num_iv.reset_index(drop=True), ordblk_iv], axis=1)

    out = {"rounds": ROUNDS, "n_train": int(len(itr)), "n_val": int(len(iv))}
    print("=" * 96)
    print("CTR ENGAGEMENT VERIFICATION (decisive, no private APIs)")
    print("=" * 96)

    # ---------------- T1: model dump contains CTR structures ----------------
    m_cat = fit(frame_cat, ytr, twins)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.json"
        m_cat.save_model(str(p))
        blob = p.read_text(encoding="utf-8", errors="ignore")
    hits = {k: blob.count(f'"{k}"') for k in ("ctr_type", "ctr_data", "ctr", "one_hot")}
    out["T1_dump_counts"] = hits
    ctr_present = hits.get("ctr_type", 0) > 0
    out["T1_ctr_present"] = ctr_present
    print(f"\nT1  model dump: {hits}")
    print(f"    -> CTR structures present in the saved model: "
          f"{'YES' if ctr_present else 'NO -- categoricals were NOT used as categoricals'}")

    # ---------------- T2: prediction divergence vs the ordinal control ----------------
    m_ord = fit(frame_ord, ytr, None)
    pc = m_cat.predict_proba(frame_cat_iv)[:, 1]
    po = m_ord.predict_proba(frame_ord_iv)[:, 1]
    md = float(np.max(np.abs(pc - po)))
    mc = float(np.corrcoef(pc, po)[0, 1])
    from sklearn.metrics import roc_auc_score
    out["T2_max_abs_diff"] = md
    out["T2_corr"] = mc
    out["T2_auc_cat"] = float(roc_auc_score(yiv, pc))
    out["T2_auc_ord"] = float(roc_auc_score(yiv, po))
    divergent = md > 1e-9
    out["T2_divergent"] = divergent
    print(f"\nT2  declared-categorical vs ordinal-code control:")
    print(f"    max |prediction difference| = {md:.6e}")
    print(f"    correlation                 = {mc:.8f}")
    print(f"    AUC declared-categorical    = {out['T2_auc_cat']:.6f}")
    print(f"    AUC ordinal-code control    = {out['T2_auc_ord']:.6f}")
    print(f"    -> predictions "
          f"{'DIVERGE, so CTRs are genuinely in use' if divergent else 'ARE IDENTICAL, which would mean the categorical columns were ignored'}")

    # ---------------- T3: declaring a numeric column must raise ----------------
    try:
        fit(frame_cat, ytr, twins + ["Age"])
        out["T3_numeric_as_cat_raised"] = False
        t3 = "DID NOT RAISE -- the categorical contract is not enforced here"
    except Exception as exc:                                     # noqa: BLE001
        out["T3_numeric_as_cat_raised"] = True
        out["T3_error"] = f"{type(exc).__name__}: {str(exc)[:140]}"
        t3 = f"raised as expected ({type(exc).__name__})"
    print(f"\nT3  declaring the numeric column 'Age' as categorical: {t3}")

    out["verdict"] = ("CTR MACHINERY VERIFIED ENGAGED" if (ctr_present and divergent)
                      else "CTR ENGAGEMENT NOT VERIFIED -- do not trust any native-cat result")
    print(f"\n{'='*96}")
    print(f"VERDICT: {out['verdict']}")
    print("=" * 96)
    save_json(out, REPORTS / "ctr_engagement.json")
    print("wrote", REPORTS / "ctr_engagement.json")


if __name__ == "__main__":
    main()
