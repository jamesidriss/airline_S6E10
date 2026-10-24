"""Measure the estimator-averaging gain of the raw-context TabPFN 3.5 member, fold by fold.

The phase-23 raw8 hypothesis is that the official 8-estimator recipe (8 randomized prompts over
the same 559,708-row context) beats the 1-estimator member that already sits inside v7 by about
+3e-5 on top of v7 at w=0.10. On a 15.9 GiB card the 8 caches cannot coexist, so the recipe is
realized as 8 *sequential units*, one process each, with distinct `random_state`, aggregated by
the official equal-logit average. That is the same geometry the Kaggle kernel used (its
`units/` directory).

Before paying for all 40 units, this script measures the gain from the units we do have. The
variance-reduction law makes the extrapolation exact under exchangeability of the units:
a mean of n units removes (1 - 1/n) of the between-unit variance, so
    gain_8 = gain_n * (1 - 1/8) / (1 - 1/n) = gain_2 * 1.75  (for n = 2).

Nothing here uses an evaluation label: the units are all fitted on the outer FIT rows of the
immutable primary folds, and every AUC is scored on the held-out fold.

Usage:
  python scripts/raw8_probe.py --fold 0 --units artifacts/sol_tabpfn35_predict_guard/raw_f0.npy \
      artifacts/p25e_raw_s1202/raw_f0.npy --sweep
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

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import fold_auc  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402


def lg(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--units", nargs="+", required=True)
    ap.add_argument("--ids", nargs="*", default=None, help="matching ids_*.npy per unit, optional")
    ap.add_argument("--champion", default="v7_sol_tabpfn_internal2_aux10")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--tag", default="raw8_probe")
    args = ap.parse_args()

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y, tr[ID_COL]).folds
    vf = np.where(folds == args.fold)[0]
    yv = y[vf]

    # align every unit onto the fold's row order through its saved ids
    preds = []
    for i, path in enumerate(args.units):
        p = Path(path)
        v = np.asarray(np.load(p), dtype="float64")
        ids_path = None
        if args.ids and i < len(args.ids) and args.ids[i]:
            ids_path = Path(args.ids[i])
        else:
            cand = p.parent / f"ids_f{args.fold}.npy"
            ids_path = cand if cand.exists() else None
        if ids_path is not None and ids_path.exists():
            ids = np.load(ids_path)
            pos = pd.Series(np.arange(len(ids)), index=ids).reindex(vf)
            assert pos.notna().all(), f"ids in {ids_path} do not cover fold {args.fold}"
            v = v[pos.to_numpy(int)]
        else:
            assert len(v) == len(vf), f"{p} has {len(v)} rows, fold has {len(vf)}"
        preds.append(v)
        print(f"unit {i}: {p.name} from {p.parent.name}  auc={roc_auc_score(yv, v):.9f}")

    L = np.column_stack([lg(v) for v in preds])
    out = {"fold": args.fold, "n_rows": int(len(vf)), "units": [str(u) for u in args.units],
           "unit_auc": [float(roc_auc_score(yv, v)) for v in preds], "curve": []}
    base = out["unit_auc"][0]
    for n in range(2, L.shape[1] + 1):
        mix = L[:, :n].mean(axis=1)
        a = float(roc_auc_score(yv, mix))
        # paired bootstrap over rows for the n-unit vs 1-unit difference
        rng = np.random.default_rng(11)
        d = []
        for _ in range(200):
            idx = rng.choice(len(vf), len(vf), replace=True)
            if yv[idx].min() == yv[idx].max():
                continue
            d.append(roc_auc_score(yv[idx], mix[idx]) - roc_auc_score(yv[idx], L[idx, 0]))
        d = np.asarray(d)
        out["curve"].append({"n": n, "auc": a, "gain_vs_unit1": a - base,
                             "boot_mean": float(d.mean()), "boot_sd": float(d.std(ddof=1)),
                             "boot_pos": float((d > 0).mean())})
        print(f"  mean of {n} units: auc={a:.9f}  gain={a-base:+.3e}  "
              f"boot {d.mean()*1e4:+.3f}e-4 (sd {d.std(ddof=1)*1e4:.2f}, "
              f"P>0 {float((d>0).mean()):.2f})")
    if len(out["curve"]) >= 1:
        n = out["curve"][-1]["n"]
        out["extrapolated_gain_8"] = out["curve"][-1]["gain_vs_unit1"] * (7 / 8) / (1 - 1 / n)
        print(f"\nextrapolated 8-estimator gain over 1 unit: {out['extrapolated_gain_8']*1e4:+.3f}e-4")

    if args.sweep:
        champ = store.load_oof(args.champion).astype("float64")[vf]
        mix = lg(L.mean(axis=1))
        rows = []
        for w in (0.00, 0.05, 0.10, 0.15, 0.25, 0.50):
            blend = (1 - w) * lg(champ) + w * mix
            a = float(roc_auc_score(yv, blend))
            rows.append({"w": w, "auc": a, "delta": a - float(roc_auc_score(yv, champ))})
            print(f"  v7 + {w:.2f} raw_units: auc={a:.9f}  delta={rows[-1]['delta']:+.3e}")
        out["sweep_vs_champion"] = rows

    save_json(out, REPORTS / f"{args.tag}.json")


if __name__ == "__main__":
    main()
