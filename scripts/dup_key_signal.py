"""DECISIVE TEST: do exact duplicate feature-key groups carry recoverable label signal?

48.96% of train rows share an (13 ratings + 4 categorical) key with another row. If the
generator drew labels i.i.d. per row from p(x), a fold-safe group target encoding adds nothing.
If instead labels are tied to the source row, a fold-safe group encoding is enormously powerful.

Every statistic below is fitted on the fold's FIT rows only and applied to its held-out rows,
so the numbers are leakage-free out-of-fold estimates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.s6e10 import META4, NUMS, SURVEY13  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

LADDER = [
    ("S13", SURVEY13),
    ("S13+M4", SURVEY13 + META4),
    ("S13+M4+age", SURVEY13 + META4 + ["Age"]),
    ("S13+M4+age+fd_bin", SURVEY13 + META4 + ["Age", "_fd_bin250"]),
    ("S13+M4+age+fd", SURVEY13 + META4 + ["Age", "Flight Distance"]),
    ("S13[9 strongest]+M4+age", ["Online boarding", "Inflight entertainment", "Seat comfort",
                                 "Inflight wifi service", "Checkin service", "Cleanliness",
                                 "On-board service", "Leg room service", "Baggage handling"] + META4 + ["Age"]),
]


def key_series(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> pd.Series:
    need_fd_bin = any(c == "_fd_bin250" for c in cols)
    src_tr, src_te = tr.copy(), te.copy()
    if need_fd_bin:
        for d in (src_tr, src_te):
            d["_fd_bin250"] = np.floor(pd.to_numeric(d["Flight Distance"], errors="coerce").fillna(-1) / 250)
    cols = [c for c in cols if not c.startswith("_")]
    d = pd.concat([src_tr[cols], src_te[cols]], axis=0, ignore_index=True).astype(object)
    d = d.round(6) if any(src_tr[c].dtype.kind == "f" for c in cols) else d
    return pd.util.hash_pandas_object(d, index=False)


def main() -> None:
    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    ntr = len(tr)
    folds = get_scheme("primary", tr[TARGET].values.astype("int8"), tr[ID_COL]).folds
    prior = float(y.mean())
    rep = {}

    print(f"{'key':<26} {'cover':>7} {'auc_te_s1':>11} {'auc_te_s10':>11} {'auc_te_s50':>11} {'cov_uniq':>9}")
    for name, cols in LADDER:
        h = key_series(tr, te, cols)
        htr = h.iloc[:ntr].reset_index(drop=True)
        hte = h.iloc[ntr:].reset_index(drop=True)
        cover = float(htr.isin(set(hte.unique())).mean())
        aucs, covs = {}, {}
        for smooth in (1.0, 10.0, 50.0):
            oof = np.full(ntr, prior)
            seen_uniq = np.zeros(ntr, dtype=bool)
            for k in sorted(set(folds.tolist())):
                a = np.where(folds != k)[0]
                b = np.where(folds == k)[0]
                d = pd.DataFrame({"h": htr.iloc[a].values, "y": y[a]})
                s = d.groupby("h")["y"].agg(["sum", "size"])
                tab = ((s["sum"] + smooth * prior) / (s["size"] + smooth))
                cnt = s["size"]
                oof[b] = htr.iloc[b].map(tab).fillna(prior).to_numpy()
                if smooth == 1.0:
                    seen_uniq[b] = htr.iloc[b].map(cnt).fillna(0).to_numpy() == 1
            aucs[smooth] = float(roc_auc_score(y, oof))
            covs[smooth] = float(seen_uniq.mean())
        rep[name] = {"cols": cols, "test_coverage": round(cover, 5),
                     "auc_te": {str(k): round(v, 6) for k, v in aucs.items()},
                     "frac_exactly_one_match_in_fit": round(covs[1.0], 5)}
        print(f"{name:<26} {cover:>7.4f} {aucs[1.0]:>11.6f} {aucs[10.0]:>11.6f} "
              f"{aucs[50.0]:>11.6f} {covs[1.0]:>9.4f}")

    save_json(rep, REPORTS / "duplicate_key_signal.json")
    print("\nwrote", REPORTS / "duplicate_key_signal.json")


if __name__ == "__main__":
    main()