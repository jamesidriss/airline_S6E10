"""How much does the second finalist slot actually buy?

Only the better of the two selected submissions is judged, so the second slot is worth
`E[max(AUC_A, AUC_B)] - AUC_A` on the private split, not `AUC_B - AUC_A`. With top vectors
correlated above 0.998 that expectation is tiny, and it has to be compared with the
*insurance* value of a methodologically independent pipeline.

Nothing here is fitted to the public leaderboard: every number is computed from OOF
predictions on the immutable primary folds, with the private split's row count (239,875).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import REPORTS, TRAIN_CSV, save_json  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

POOL = Path(r"C:\Users\James\AppData\Local\Temp\opencode\s6e10_oof_pool\_pool\pool.npz")
PRIVATE_ROWS = 239_875


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--tag", default="second_slot")
    args = ap.parse_args()

    tr = pd.read_csv(TRAIN_CSV, usecols=["id", "satisfaction"])
    y = tr["satisfaction"].to_numpy(dtype="int8")
    folds = get_scheme("primary", y, tr["id"]).folds
    x = np.load(ROOT / "artifacts/phase24_candidateX/candidate_x_oof.npy").astype("float64")
    P = np.load(POOL)

    cands = {}
    for name, eid in (("v7_internal", "v7_sol_tabpfn_internal2_aux10"),
                      ("v6_internal", "v6_sol_tabpfn_route_aux10")):
        try:
            cands[name] = store.load_oof(eid).astype("float64")
        except Exception as exc:  # noqa: BLE001
            print(f"skip {name}: {exc}")
    for nm in ("najiama_05_blend_satisfaction", "najiama_06_blend_satisfaction",
               "goodpjw200_stack245", "goodpjw200_stack145"):
        key = "oof__" + nm
        if key in P.files:
            cands[nm.split("_blend")[0]] = P[key].astype("float64")

    rng = np.random.default_rng(7)
    out = {"private_rows": PRIVATE_ROWS, "x_oof": float(roc_auc_score(y, x)), "candidates": {}}
    for name, v in cands.items():
        if v.shape != x.shape:
            continue
        rho = float(pd.Series(v).corr(pd.Series(x), method="spearman"))
        d_pooled = float(roc_auc_score(y, v) - roc_auc_score(y, x))
        # paired bootstrap at the private split's size: distribution of the AUC difference
        deltas = []
        for _ in range(args.boot):
            idx = rng.choice(len(y), PRIVATE_ROWS, replace=False)
            yy = y[idx]
            deltas.append(roc_auc_score(yy, v[idx]) - roc_auc_score(yy, x[idx]))
        d = np.asarray(deltas)
        out["candidates"][name] = {
            "oof_auc": float(roc_auc_score(y, v)),
            "delta_vs_x_pooled": d_pooled,
            "spearman_vs_x": rho,
            "priv_delta_mean": float(d.mean()),
            "priv_delta_sd": float(d.std(ddof=1)),
            "p_beats_x_on_private": float((d > 0).mean()),
            "expected_gain_if_max": float(d[d > 0].mean() * (d > 0).mean()) if (d > 0).any() else 0.0,
        }
        c = out["candidates"][name]
        print(f"{name:16s} OOF {c['oof_auc']:.6f}  d={d_pooled*1e4:+.2f}e-4  rho={rho:.4f}  "
              f"P(B>X on private)={c['p_beats_x_on_private']:.2f}  "
              f"E[max gain]={c['expected_gain_if_max']*1e4:.3f}e-4")

    save_json(out, REPORTS / f"{args.tag}.json")


if __name__ == "__main__":
    main()
