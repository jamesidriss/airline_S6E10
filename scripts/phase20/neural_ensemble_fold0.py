"""Phase-20 neural-ensemble fold-0 screen.

Averages the FULL-view (rich 285-feature) neural architectures' saved fold-0
predictions in logit space (equal-logit, matching the champion's `equal_logit`
contract) and screens the blend against the champion portfolio.

Rationale: the 6 archs sit on a similar standalone plateau but have decorrelation
among themselves (attention vs crosses vs residual vs gating vs interaction). A
logit-averaged ensemble is typically BOTH stronger and less correlated with the
champion than any single arch. This is the predeclared next mechanism (ensemble,
not re-tuning a single arch).

Anti-leakage: reads only SAVED fold-0 predictions (each trained on FIT, inner-10%
ES from FIT only, champion hash-asserted). This is a SCREEN (fold 0 only) -- it
cannot pass the admission gate, only decide whether the ensemble merits full OOF.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import ARTIFACTS, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.phase19.foundation_screen import screen_metrics  # noqa: E402
from src.validation.compare import logit  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.phase19.foundation_screen_fold0 import load_champion_fold0  # noqa: E402

ART = ARTIFACTS / "phase20" / "neural_screen"
OUT = REPORTS / "phase20" / "neural_screen"
EPS = 1e-7

ARCHS = ["fttransformer", "dcnv2", "plrmlp", "tabresnet", "moe", "interactionnet"]


def _cl(p):
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full", choices=["full", "route"])
    ap.add_argument("--members", default="",
                    help="comma list of archs to include; default = all that have a saved pred")
    ap.add_argument("--eval-rows", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    members = [m.strip() for m in args.members.split(",") if m.strip()] or ARCHS

    tr, _te = load_cached_parquet()
    y = tr[TARGET].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds
    EVAL = np.flatnonzero(folds == 0)

    tag_suffix = "" if args.view == "full" else "_route"
    preds = []
    used = []
    for m in members:
        p = ART / f"{m}{tag_suffix}_fold0.npy"
        if p.exists():
            arr = np.load(p).astype(np.float64)
            assert arr.shape[0] == len(EVAL), f"{m}: pred/row mismatch"
            preds.append(arr)
            used.append(m)
        else:
            print(f"[ens] skip {m}: no saved pred {p.name}")
    if len(preds) < 2:
        raise SystemExit(f"need >=2 member predictions to ensemble; found {len(preds)}")
    print(f"[ens] view={args.view} members={used}")

    # equal-logit average of member probabilities
    logit_stack = np.stack([logit(_cl(p)) for p in preds], axis=0)
    mean_logit = logit_stack.mean(axis=0)
    blend_pred = 1.0 / (1.0 + np.exp(-mean_logit))

    route_all, port_all = load_champion_fold0(folds)
    eval_n = len(EVAL) if (not args.eval_rows or args.eval_rows >= len(EVAL)) else args.eval_rows
    eval_idx = EVAL[:eval_n]
    y_eval = y[eval_idx]
    pred0 = blend_pred[:eval_n]
    port0, route0 = port_all[eval_idx], route_all[eval_idx]

    m = screen_metrics(y_eval, pred0, port0, route0)
    secs = time.time() - t0
    tag = args.tag or f"neural_ens_{args.view}_fold0"

    # pairwise member logit correlation (decorrelation among themselves)
    corr = np.corrcoef(logit_stack)
    n = len(used)
    off = [corr[i, j] for i in range(n) for j in range(i + 1, n)]
    mean_pair_corr = float(np.mean(off)) if off else 1.0

    meta = {
        "kind": "neural_equal_logit_ensemble", "view": args.view, "members": used,
        "n_members": len(used), "mean_pairwise_member_logit_corr": round(mean_pair_corr, 4),
        "eval_rows": int(eval_n), "eval_fold": 0, "fit_folds": [1, 2, 3, 4],
        "utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "equal-logit average of saved fold-0 member preds (each FIT-trained, "
                    "inner-10% ES, champion hash-asserted); SCREEN only",
        "metrics": m, "seconds": secs,
    }
    save_json(meta, OUT / f"{tag}.json")
    np.save(ART / f"{tag}.npy", pred0.astype("float32"))

    bar = m["portfolio_delta"] >= 5e-5 and m["logit_pearson_vs_portfolio"] < 0.995
    print(f"[ens] AUC={m['standalone_auc']:.6f} port_delta={m['portfolio_delta']:+.3e} "
          f"corr_port={m['logit_pearson_vs_portfolio']:.4f} mean_member_corr={mean_pair_corr:.4f}")
    print(f"[ens] PROMISING_BAR_MET={bar}  ({secs:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
