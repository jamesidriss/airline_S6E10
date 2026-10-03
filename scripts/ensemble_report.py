"""Ensemble report: correlations, blend geometry, nested stacks, admission gates.

Usage: python scripts/ensemble_report.py --exp-ids a,b,c [--test-weights]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.ensemble import lab  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import report as cmpreport  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-ids", required=True)
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--tag", default="ens")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme(args.folds, y, tr[ID_COL]).folds
    ids = args.exp_ids.split(",")
    P = {i: store.load_oof(i).astype("float64") for i in ids}
    idx = store._load_index()

    out: dict = {"exp_ids": ids, "folds": args.folds}
    print("=" * 110)
    print("STANDALONE")
    print("=" * 110)
    rows = []
    for i in ids:
        a = lab.auc(y, P[i])
        f = lab.fold_aucs(y, P[i], folds)
        rows.append({"exp_id": i, "auc": round(a, 6), "fold_aucs": [round(x, 6) for x in f],
                     "family": idx[i].get("meta", {}).get("family"),
                     "view": idx[i].get("meta", {}).get("featureset")})
    print(pd.DataFrame(rows).to_string(index=False))
    out["standalone"] = rows

    print("\n" + "=" * 110)
    print("LOGIT-SPACE PEARSON CORRELATION")
    print("=" * 110)
    C = lab.corr_matrix(P, "logit")
    print(C.round(5).to_string())
    out["corr_logit"] = C.round(6).to_dict()

    print("\n" + "=" * 110)
    print("SPEARMAN CORRELATION")
    print("=" * 110)
    S = lab.spearman_matrix(P)
    print(S.round(5).to_string())
    out["spearman"] = S.round(6).to_dict()

    print("\n" + "=" * 110)
    print("BLENDS")
    print("=" * 110)
    blends = {}
    for kind in ("prob", "logit", "rank"):
        b = lab.equal_blend(P, y, folds, kind=kind)
        blends[f"equal_{kind}"] = b
        print(f"  equal-{kind:<6} auc={b['auc']:.6f}  folds={[round(x,6) for x in b['fold_aucs']]}")
    g = lab.greedy_hill_climb(P, y, folds, kind="logit")
    blends["greedy_logit"] = g
    print(f"  greedy-logit auc={g['auc']:.6f}  weights={ {n: round(w,4) for n,w in zip(ids,g['weights']) if w>0} }")
    for Cc in (0.03, 0.1, 0.3, 1.0, 3.0):
        try:
            s = lab.logistic_stack_nested(P, y, folds, kind="logit", C=Cc)
            blends[f"logit_lr_C{Cc}"] = {k: v for k, v in s.items() if k != "oof"}
            print(f"  nested logit-LR C={Cc:<5} auc={s['auc']:.6f}  "
                  f"folds={[round(x,6) for x in s['fold_aucs']]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  nested logit-LR C={Cc} failed: {exc}")
    out["blends"] = {k: {kk: vv for kk, vv in v.items() if kk != "oof"} for k, v in blends.items()}

    print("\n" + "=" * 110)
    print("ADMISSION GATE vs CHAMPION (mean >= 1.5e-5, >= 2.5 fold-SE, >= 4/5 folds)")
    print("=" * 110)
    base = max(rows, key=lambda r: r["auc"])
    bid = base["exp_id"]
    gates = []
    for cand in ids:
        if cand == bid:
            continue
        g = lab.admission_gate(cand, base["fold_aucs"], lab.fold_aucs(y, P[cand], folds))
        gates.append(g)
        print(f"  {g}")
    out["gates"] = gates

    best_key = max(blends, key=lambda k: blends[k]["auc"])
    print(f"\nbest blend: {best_key} auc={blends[best_key]['auc']:.6f}  "
          f"(champion single {bid} auc={base['auc']:.6f})")
    out["best_blend"] = best_key
    save_json(out, REPORTS / f"{args.tag}_report.json")
    print("wrote", REPORTS / f"{args.tag}_report.json")


if __name__ == "__main__":
    main()