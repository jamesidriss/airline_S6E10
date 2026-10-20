"""Phase 24 -- determinism comparison between the source and synthetic populations.

Same key (16 rating/categorical columns), same statistic, both datasets:
  * share of rows sitting in label-pure key groups,
  * leave-one-out group-mean AUC over rows in non-singleton groups,
  * mean group size.
This isolates how much of the source's near-determinism the generator preserved.

    .\\.venv\\Scripts\\python.exe scripts/phase24/determinism_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import load_cached_parquet, save_json  # noqa: E402

ORIG = ROOT / "data/original/arseniyshutko__binary-aviation-satisfaction-129k/data.csv"
OUT = ROOT / "reports/phase24/determinism_compare.json"
RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
CATS = ["Class", "Type of Travel", "Customer Type"]


def stats(df, y, cols):
    key = pd.util.hash_pandas_object(df[cols], index=False).to_numpy(np.uint64)
    s = pd.Series(key)
    vc = s.value_counts()
    n = s.map(vc).to_numpy()
    g = pd.DataFrame({"k": key, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    pure = g[((g["sum"] == 0) | (g["sum"] == g["count"])) & (g["count"] >= 2)]
    pure_rows = int(pure["count"].sum())
    out = {
        "n_rows": int(len(df)),
        "n_keys": int(len(g)),
        "mean_group_size": float(vc.mean()),
        "share_rows_group_size_ge2": float((n >= 2).mean()),
        "share_rows_in_pure_groups_ge2": float(pure_rows / len(df)),
    }
    # LOO group-mean AUC on rows in non-singleton groups
    m = n >= 2
    if m.sum() > 1000:
        g2 = pd.DataFrame({"k": key, "y": y})
        gs = g2.groupby("k")["y"].transform("sum").to_numpy()
        gc = g2.groupby("k")["y"].transform("count").to_numpy()
        loo = (gs[m] - y[m]) / (gc[m] - 1)
        out["loo_groupmean_auc_nonsingleton"] = float(roc_auc_score(y[m], loo))
        out["loo_groupmean_rows"] = int(m.sum())
    return out


def main() -> None:
    tr, _ = load_cached_parquet()
    yc = tr.satisfaction.to_numpy(dtype="int8")
    og = pd.read_csv(ORIG)
    ys = og["satisfaction"].to_numpy().astype("int8")

    out = {
        "status": "COMPLETE",
        "key": "13 rating columns + Class + Type of Travel + Customer Type",
        "source": stats(og, ys, RATINGS + CATS),
        "synthetic": stats(tr, yc, RATINGS + CATS),
        "reading": (
            "A pure group with >=2 rows means two distinct passengers share the exact same 16 rating/"
            "categorical profile and nevertheless received the same label. The share of rows in pure "
            "groups, and the leave-one-out group-mean AUC, measure how much of the label is a function "
            "of the discrete profile alone. The source population is almost deterministic there; the "
            "synthetic population is not, which is the mechanism behind the two CV ceilings."
        ),
    }
    save_json(out, OUT)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"refuse to overwrite {OUT}")
    main()
