"""Compare candidate original (real) datasets against the S6E10 synthetic data.

Fingerprints the generator: value support, class vocabulary, per-column value support
overlap, marginal distributions, joint row matches, and label agreement on matched rows.

Output: reports/original_analysis.json + prints a dense summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, TARGET, ID_COL, load_cached_parquet, save_json, file_sha256  # noqa: E402

ORIG = Path(__file__).resolve().parents[1] / "data" / "original"

RENAME = {
    "Type of Travel": "Type of Travel",
}


def load_candidates() -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(ORIG.rglob("*.csv")):
        key = str(p.parent.name) + "/" + p.name
        try:
            df = pd.read_csv(p)
        except Exception as exc:  # noqa: BLE001
            print("skip", key, exc)
            continue
        out[key] = df
    return out


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to the S6E10 schema."""
    ren = {}
    for c in df.columns:
        cc = c.strip()
        low = cc.lower()
        if low == "type of travel":
            ren[c] = "Type of Travel"
        elif low == "customer type":
            ren[c] = "Customer Type"
        elif low == "arrival delay in minutes":
            ren[c] = "Arrival Delay in Minutes"
        elif low == "departure delay in minutes":
            ren[c] = "Departure Delay in Minutes"
        elif low == "gate location":
            ren[c] = "Gate location"
        elif low == "inflight wifi service":
            ren[c] = "Inflight wifi service"
        elif low == "class":
            ren[c] = "Class"
        elif low == "age":
            ren[c] = "Age"
        elif low == "flight distance":
            ren[c] = "Flight Distance"
        elif low == "gender":
            ren[c] = "Gender"
        elif low in ("departure/arrival time convenient", "departure/arrival time convenient "):
            ren[c] = "Departure/Arrival time convenient"
        elif low in ("satisfaction", "satisfaction_level"):
            ren[c] = TARGET
    d = df.rename(columns=ren)
    return d


def main() -> None:
    tr, te = load_cached_parquet()
    feats = [c for c in te.columns if c != ID_COL]
    syn = pd.concat([tr[feats], te[feats]], ignore_index=True)
    syn_n = len(syn)

    cands = load_candidates()
    rep: dict = {"synthetic_rows": int(syn_n), "candidates": {}}

    for key, raw in cands.items():
        d = norm_cols(raw)
        if TARGET not in d.columns:
            d[TARGET] = np.nan
        missing = [c for c in feats if c not in d.columns]
        common = [c for c in feats if c in d.columns]
        entry = {
            "shape": list(raw.shape),
            "raw_cols": list(raw.columns),
            "sha256_16": file_sha256(ORIG / key.split("/")[0] / key.split("/")[1])[:16]
            if False else None,
            "missing_vs_syn": missing,
            "n_common_cols": len(common),
        }
        if len(common) < 10:
            rep["candidates"][key] = entry
            continue

        # --- value support comparison
        supp = {}
        for c in common:
            v = set(pd.to_numeric(d[c], errors="coerce").dropna().unique().tolist()) if d[c].dtype.kind in "if" else set(d[c].dropna().unique().tolist())
            vs = set(syn[c].dropna().unique().tolist())
            inter = v & vs
            supp[c] = {
                "n_orig_vals": len(v),
                "n_syn_vals": len(vs),
                "n_inter": len(inter),
                "orig_vals_not_in_syn": sorted([str(x) for x in (v - vs)])[:15],
                "syn_vals_not_in_orig": sorted([str(x) for x in (vs - v)])[:15],
                "frac_syn_rows_covered_by_orig_support": float(syn[c].isin(inter).mean()),
            }
        entry["support"] = supp

        # --- label vocab
        entry["label_counts"] = {str(k): int(v) for k, v in d[TARGET].value_counts(dropna=False).items()}

        # --- single column marginals vs synthetic (KS on numeric, TVD on categorical)
        from scipy import stats

        marg = {}
        for c in common:
            if pd.api.types.is_numeric_dtype(syn[c]):
                a = pd.to_numeric(d[c], errors="coerce").dropna().astype("float64")
                b = pd.to_numeric(syn[c], errors="coerce").dropna().astype("float64")
                if len(a) > 3 and a.nunique() > 1:
                    st, p = stats.ks_2samp(a, b)
                    marg[c] = {"ks": round(float(st), 5), "mean_orig": round(float(a.mean()), 4),
                               "mean_syn": round(float(b.mean()), 4),
                               "std_orig": round(float(a.std()), 4), "std_syn": round(float(b.std()), 4)}
            else:
                pa = d[c].astype(str).value_counts(normalize=True)
                pb = syn[c].astype(str).value_counts(normalize=True)
                idx = pa.index.union(pb.index)
                tvd = float(0.5 * (pa.reindex(idx).fillna(0) - pb.reindex(idx).fillna(0)).abs().sum())
                marg[c] = {"tvd": round(tvd, 5), "orig_vals": sorted(pa.index.tolist())[:8],
                           "syn_vals": sorted(pb.index.tolist())[:8]}
        entry["marginals"] = marg

        # --- exact multi-column row match
        keycols = [c for c in common]
        if len(keycols) >= 10:
            dk = d[keycols].copy()
            sk = syn[keycols].copy()
            for c in keycols:
                if dk[c].dtype.kind in "if":
                    dk[c] = dk[c].astype("float64").round(4)
                    sk[c] = sk[c].astype("float64").round(4)
                else:
                    dk[c] = dk[c].astype(str)
                    sk[c] = sk[c].astype(str)
            # hash rows
            def rh(df):
                return pd.util.hash_pandas_object(df.astype(object), index=False)

            h_orig = rh(dk)
            h_syn = rh(sk)
            so = set(h_orig.unique())
            matched = h_syn.isin(so)
            entry["exact_full_row_match_frac_synth"] = round(float(matched.mean()), 6)
            entry["n_exact_matches"] = int(matched.sum())
            # which column set gives matches? try the 14 survey columns only
            for sub_name, sub in [("survey14", [c for c in common if c not in ("Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes")]),
                                  ("all_but_delay", [c for c in common if "Delay" not in c]),
                                  ("all_but_dist_age", [c for c in common if c not in ("Flight Distance", "Age")])]:
                if len(sub) >= 5:
                    hh = rh(d[sub].astype(object))
                    hs = rh(syn[sub].astype(object))
                    entry[f"exact_match_frac[{sub_name}]"] = round(float(hs.isin(set(hh.unique())).mean()), 6)

        rep["candidates"][key] = entry

    save_json(rep, REPORTS / "original_analysis.json")

    # ---- dense printout
    print("=" * 110)
    print(f"{'candidate':<70} {'shape':<14} {'common':>6} {'rowmatch':>9}")
    print("=" * 110)
    for k, v in rep["candidates"].items():
        print(f"{k:<70} {str(v.get('shape')):<14} {v.get('n_common_cols', 0):>6} "
              f"{str(v.get('exact_full_row_match_frac_synth', '-')):>9}")


if __name__ == "__main__":
    main()