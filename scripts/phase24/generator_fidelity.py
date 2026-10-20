"""Phase 24 -- generator-fidelity diagnostic on the continuous covariates.

The source anchor shows the ORIGINAL 129,880-row distribution supports CV AUC
0.98881 with the same 21 predictors, while the synthetic competition distribution
plateaus at 0.96185 with 5.4x more rows and a much stronger model stack.  This
script tests one mechanism for that gap: whether the generator preserved the
label relation of the continuous covariates that carry most of the source signal
(the two delay columns in particular, whose marginals the phase-4 domain audit
already showed to be rewritten).

Read-only, no competition labels are used in any source-side fit.

    .\\.venv\\Scripts\\python.exe scripts\\phase24/generator_fidelity.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import load_cached_parquet, save_json  # noqa: E402

ORIG = ROOT / "data/original/arseniyshutko__binary-aviation-satisfaction-129k/data.csv"
OUT = ROOT / "reports/phase24/generator_fidelity.json"
NUM = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]


def load_source():
    og = pd.read_csv(ORIG)
    y = og["satisfaction"].to_numpy().astype("int8")
    return og, y


def bin_rate(x, y, q=10):
    x = np.asarray(x, dtype=float)
    edges = np.quantile(x, np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, q - 1)
    out = []
    for b in range(q):
        m = idx == b
        out.append({"bin": b, "n": int(m.sum()), "lo": float(edges[b]), "hi": float(edges[b + 1]),
                    "pos_rate": float(y[m].mean()) if m.any() else None})
    return out


def main() -> None:
    t0 = time.time()
    tr, _ = load_cached_parquet()
    yc = tr.satisfaction.to_numpy(dtype="int8")
    og, ys = load_source()

    out = {
        "status": "COMPLETE",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {"n": int(len(ys)), "pos_rate": float(ys.mean())},
        "synthetic": {"n": int(len(yc)), "pos_rate": float(yc.mean())},
        "single_feature_auc": {},
        "delay_decile_pos_rate": {},
        "missingness": {},
    }

    for c in NUM:
        s = og[c].to_numpy(dtype=float)
        m = np.isfinite(s)
        # direction-free strength: max(auc, 1-auc) so the number is comparable across columns
        a_s = float(roc_auc_score(ys[m], s[m]))
        c_vals = tr[c].to_numpy(dtype=float)
        mc = np.isfinite(c_vals)
        a_c = float(roc_auc_score(yc[mc], c_vals[mc]))
        out["single_feature_auc"][c] = {
            "source_auc": a_s, "source_strength": max(a_s, 1 - a_s), "source_missing_frac": float((~m).mean()),
            "synthetic_auc": a_c, "synthetic_strength": max(a_c, 1 - a_c), "synthetic_missing_frac": float((~mc).mean()),
            "strength_ratio_synth_over_source": (max(a_c, 1 - a_c) - 0.5) / (max(a_s, 1 - a_s) - 0.5)
            if max(a_s, 1 - a_s) > 0.5 else None,
        }

    for c in ("Departure Delay in Minutes", "Arrival Delay in Minutes"):
        s = og[c].to_numpy(dtype=float)
        m = np.isfinite(s)
        c_vals = tr[c].to_numpy(dtype=float)
        mc = np.isfinite(c_vals)
        out["delay_decile_pos_rate"][c] = {
            "source": bin_rate(s[m], ys[m]),
            "synthetic": bin_rate(c_vals[mc], yc[mc]),
        }

    # how much of each delay column is structurally different
    for c in ("Departure Delay in Minutes", "Arrival Delay in Minutes"):
        s = og[c].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        c_vals = tr[c].to_numpy(dtype=float)
        c_vals = c_vals[np.isfinite(c_vals)]
        out["missingness"][c] = {
            "source_q": [float(np.quantile(s, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.999)],
            "synthetic_q": [float(np.quantile(c_vals, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.999)],
            "source_frac_zero": float((s == 0).mean()),
            "synthetic_frac_zero": float((c_vals == 0).mean()),
        }

    out["cpu_seconds"] = time.time() - t0
    save_json(out, OUT)
    print(json.dumps({k: v for k, v in out.items() if k != "delay_decile_pos_rate"}, indent=1))
    for c, d in out["delay_decile_pos_rate"].items():
        print("\n", c)
        print("  source   ", [round(b["pos_rate"], 3) if b["pos_rate"] is not None else None for b in d["source"]])
        print("  synthetic", [round(b["pos_rate"], 3) if b["pos_rate"] is not None else None for b in d["synthetic"]])


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"refuse to overwrite {OUT}")
    main()
