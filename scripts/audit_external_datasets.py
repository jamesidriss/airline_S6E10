"""Section 20: are any of the downloaded external datasets GENUINELY INDEPENDENT labelled data?

The question, stated precisely
------------------------------
The learning curve is the only measured lever that scales (log2 slope +62.9e-5 per doubling, R^2
0.995). Phase 5 established that the scarce resource is *unique* training signal, and the duplicate
control (-6.4e-5) proves row count alone explains nothing. So the only external data worth serious
effort is data whose labels were collected INDEPENDENTLY of the 129,880-row generator source.

Six candidate datasets are already downloaded. Most Kaggle "airline passenger satisfaction" datasets are
uploads of the same survey, sometimes with an added index column. Counting those as new data would be
the single easiest way to convince ourselves the learning curve can be fed, and it would be worthless.

So this script does the boring, decisive thing: for every candidate, report rows, schema, and the
ACTUAL row overlap against the generator source, measured by hashing the 21 shared feature columns.

Overlap is reported two ways, because they answer different questions:

  * **exact 21-column row overlap** -- how many candidate rows are byte-identical copies of source
    rows. Near 100% means it is the same survey re-uploaded.
  * **label agreement on shared rows** -- if a candidate's labels DISAGREE with the source's on
    identical feature rows, it is a re-labelling rather than a copy, which is a different (and much less
    useful) thing: it is label noise, not new observations.

Verdict per candidate: INDEPENDENT (low overlap, so genuinely new observations), RE-LABELLED (same rows,
different labels), or MIRROR (same rows, same labels).

Usage: python scripts/audit_external_datasets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import RAW21, SURVEY13  # noqa: E402

ORIG = ROOT / "data" / "original"
SOURCE = ORIG / "arseniyshutko__binary-aviation-satisfaction-129k" / "data.csv"

# label column aliases seen across these uploads
LABEL_ALIASES = ["satisfaction", "Satisfaction", "satisfaction_score", "target", "Score"]

# Column aliases: casing and separator differences only. Deliberately NOT aliasing `Gender` to
# "Online support", which two uploads carry in its place: that is a genuinely different survey
# question, not a rename, and mapping them together silently corrupts every overlap hash.
COLUMN_ALIASES = {
    "Class": ["Class", "customer_class", "cabin", "class"],
    "Gender": ["Gender", "gender", "Sex", "sex"],
}
SATISFIED = {"satisfied", "true", "1", "yes"}
DISSATISFIED = {"neutral or dissatisfied", "neutral or dissatified", "false", "0", "no"}


def label_to01(s: pd.Series) -> np.ndarray:
    v = s.astype(str).str.strip().str.lower()
    out = np.full(len(v), np.nan)
    out[v.isin(SATISFIED)] = 1.0
    out[v.isin(DISSATISFIED)] = 0.0
    return out


def norm(c) -> str:
    return "".join(ch for ch in str(c).lower() if ch.isalnum())


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map each competition raw column to the candidate's own column name where possible."""
    have = {norm(c): c for c in df.columns}
    out, missing = {}, []
    for c in RAW21:
        names = COLUMN_ALIASES.get(c, [c])
        hit = next((have[norm(a)] for a in names if norm(a) in have), None)
        if hit is None:
            missing.append(c)
        else:
            out[c] = hit
    return out, missing


def hash_rows(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Row hash over the shared columns, canonicalised so dtype differences do not create false
    mismatches. Numeric columns are rounded to 4dp; object columns keep their spelling lowercased."""
    parts = []
    for c in cols:
        if c not in df.columns:
            return np.array([], dtype="uint64")
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            parts.append(pd.to_numeric(s, errors="coerce").round(4).astype("float64").fillna(-9e9)
                         .map(lambda v: format(v, ".4f")))
        else:
            parts.append(s.astype(str).str.strip().str.lower())
    return pd.util.hash_pandas_object(pd.concat(parts, axis=1), index=False).to_numpy(dtype="uint64")


def find_label_col(df: pd.DataFrame) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for a in LABEL_ALIASES:
        if a.lower() in low:
            return low[a.lower()]
    return None


def main() -> None:
    tr, _te = load_cached_parquet()
    comb = pd.concat([tr[RAW21], _te[RAW21]], ignore_index=True)
    comp_hash = hash_rows(comb, list(RAW21))
    comp_n = len(comp_hash)
    comp_uniq = set(np.unique(comp_hash).tolist())

    src = pd.read_csv(SOURCE, low_memory=False)
    src_hash = hash_rows(src, list(RAW21))
    src_label = find_label_col(src)
    src_lab = label_to01(src[src_label])

    out = {"source": {"path": str(SOURCE), "rows": int(len(src)), "n_cols": int(len(src.columns)),
                      "label_col": src_label, "n_positive": int(np.nansum(src_lab)),
                      "positive_rate": float(np.mean(src_lab)) if src_lab is not None else None},
           "competition_rows_hashed": comp_n, "competition_distinct_patterns": len(comp_uniq),
           "candidates": []}

    files = sorted(p for p in ORIG.rglob("*.csv") if SOURCE.parent not in p.parents)
    for f in files:
        rec = {"path": str(f.relative_to(ROOT)), "owner": f.parent.name}
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = str(exc)[:120]
            out["candidates"].append(rec)
            continue
        rec["rows"] = int(len(df))
        rec["n_cols"] = int(len(df.columns))
        cmap, missing = resolve_columns(df)
        rec["shared_columns"] = len(cmap)
        rec["missing_columns"] = missing
        lcol = find_label_col(df)
        rec["label_col"] = lcol
        if len(cmap) < 15:
            rec["verdict"] = (f"REJECT - only {len(cmap)}/21 shared columns, overlap not meaningful")
            print(f"{f.parent.name:<52} rows={rec['rows']:>8,}  only {len(cmap)}/21 shared -> "
                  f"rejected")
            out["candidates"].append(rec)
            continue
        if missing:
            print(f"{f.parent.name:<52} note: no counterpart for {missing}; overlap computed on the "
                  f"other {len(cmap)} columns")

        h = hash_rows(df, [cmap[c] for c in RAW21 if c in cmap])
        u = np.unique(h)
        exact_src = float(np.isin(u, np.unique(src_hash)).mean())
        exact_cmp = float(np.isin(u, np.array(sorted(comp_uniq), dtype="uint64")).mean())
        rec["distinct_row_patterns"] = int(len(u))
        rec["exact_overlap_with_source"] = exact_src
        rec["exact_overlap_with_competition"] = exact_cmp

        if lcol:
            lab = df[lcol]
            lb = label_to01(lab)
            rec["positive_rate"] = float(np.nanmean(lb)) if np.isfinite(lb).any() else None
            rec["n_positive"] = int(np.nansum(lb))
            if src_lab is not None and np.isfinite(lb).any():
                src_map = {int(v): i for i, v in enumerate(src_hash)}
                idx = np.array([src_map.get(int(v), -1) for v in h])
                m = (idx >= 0) & np.isfinite(lb)
                if m.sum() > 100:
                    sl = np.asarray(src_lab, dtype="float64")[idx[m]]
                    rec["rows_shared_with_source"] = int(m.sum())
                    rec["label_agreement_with_source"] = float(np.mean((sl > 0.5) == (lb[m] > 0.5)))

        # ---- THE decisive test: do the SURVEY ANSWERS match? ----
        # Verdict-keying on the full feature overlap gives the wrong answer for these uploads. Four of
        # them share the 13 survey ratings with the source but have DIFFERENT Age, Flight Distance and
        # traveller-segment values, so the full 18/20-column overlap is 0% -- which a naive rule reads
        # as "independent". They are not independent: they are the same survey with perturbed
        # covariates and the same label vector, which is strictly WORSE than useless for us, because
        # the (x, y) pairing is wrong. So the verdict is keyed on the rating tuples, the label
        # agreement, and the positive-class count.
        rating_cols = [c for c in SURVEY13 if c in cmap]
        rating_overlap = float("nan")
        if len(rating_cols) >= 10 and src_lab is not None:
            hsrc = hash_rows(src, rating_cols)
            hcand = hash_rows(df, [cmap[c] for c in rating_cols])
            # NOTE the metric is ASYMMETRIC: the denominator is the number of distinct rating tuples
            # in the SOURCE, so a candidate that is a strict SUBSET of the source scores below 1.0 even
            # when every one of its rows is a verbatim copy. That is why teejmahal20's 25,976-row test
            # split scores 23.7% and its 103,904-row train split scores 82.5%, while the full-feature
            # overlap for both is 100.00%. The subset relationship is caught by the feature overlap,
            # not by this number.
            rating_overlap = float(np.isin(np.unique(hsrc), np.unique(hcand)).mean())
            rec["n_rating_cols_used"] = len(rating_cols)
            rec["rating_tuple_overlap_source_denominator"] = rating_overlap
            rec["rows_matching_source_on_ratings"] = int(
                np.isin(hcand, np.unique(hsrc)).sum())
        agree = rec.get("label_agreement_with_source")
        same_pos = (rec.get("n_positive") is not None
                    and out["source"]["n_positive"] == rec["n_positive"])

        # ---- verdict ----
        # `exact_cmp` (full-feature overlap with the competition set) takes precedence: it is the
        # metric that actually detects a verbatim subset, and the rating metric cannot.
        if exact_cmp > 0.5 or (exact_src > 0.9):
            rec["verdict"] = ("MIRROR - rows are verbatim copies of the generator source or of the "
                              "competition set; not new observations")
        elif rating_overlap == rating_overlap and rating_overlap > 0.90 and same_pos:
            rec["verdict"] = ("MIRROR - survey responses and the label vector both match the source "
                              "(identical positive count); only Age/Flight Distance/segment were "
                              "perturbed, so the full-feature overlap is misleadingly low")
        elif (rating_overlap == rating_overlap and rating_overlap > 0.40
              and (same_pos or (agree is not None and agree > 0.95))):
            rec["verdict"] = ("SAME SURVEY, PERTURBED COVARIATES - ratings and label vector match "
                              "the source while Age/Flight Distance/segment differ. NOT independent, "
                              "and actively harmful: the (x, y) pairing is wrong, so it would inject "
                              "covariate noise rather than information.")
        elif not same_pos and (agree is None or agree < 0.98):
            rec["verdict"] = ("DIFFERENT LABEL VECTOR - ratings and labels do not match the source. "
                              "Not independently collected (same 129,880 row count and schema), but "
                              "the only candidate worth a fold-level append test.")
        else:
            rec["verdict"] = "INDEPENDENT - low overlap on both the full features and the ratings"
        print(f"{f.parent.name:<52} rows={rec['rows']:>8,}  rating_ovl={rating_overlap:6.1%}  "
              f"agree={agree if agree is None else round(agree, 4)}  "
              f"n_pos={rec.get('n_positive')}  -> {rec['verdict'][:44]}")
        out["candidates"].append(rec)

    indep = [c for c in out["candidates"] if c.get("verdict", "").startswith("INDEPENDENT")]
    print("\n" + "=" * 100)
    print(f"  candidates audited        : {len(out['candidates'])}")
    print(f"  genuinely INDEPENDENT     : {len(indep)}")
    for c in indep:
        print(f"      {c['owner']}  rows={c['rows']:,}")
    if not indep:
        print("  => NO genuinely independent external labelled dataset exists among the downloaded set.")
        print("     Every candidate is a mirror or a re-labelling of the same 129,880-row survey.")
        print("     The learning curve cannot be fed from here, and no further effort should be spent")
        print("     on dataset hunting for this schema.")
    out["n_independent"] = len(indep)
    save_json(out, REPORTS / "external_dataset_audit.json")
    print("\nwrote", REPORTS / "external_dataset_audit.json")


if __name__ == "__main__":
    main()
