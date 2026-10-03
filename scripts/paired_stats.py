"""Paired comparison of two stored experiments with honest uncertainty.

Reports the overall delta, the per-fold paired deltas, a stratified paired bootstrap CI on the
AUC difference, and both Pearson and Spearman correlation. Used to decide whether an ablation
result is real or inside the noise floor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import (corr, fold_auc, paired_fold_deltas,  # noqa: E402
                                    spearman, strat_bootstrap_auc_delta)
from src.validation.folds import get_scheme  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme(args.folds, y, tr[ID_COL]).folds

    base = store.load_oof(args.base).astype("float64")
    ba = float(np.mean(fold_auc(y, base, folds)))
    bf = fold_auc(y, base, folds)
    print(f"BASE {args.base}  auc={ba:.6f}  folds={[round(x,6) for x in bf]}")

    out = {"base": args.base, "base_auc": ba, "cands": {}}
    for cid in args.cands.split(","):
        c = store.load_oof(cid).astype("float64")
        ca = float(np.mean(fold_auc(y, c, folds)))
        cf = fold_auc(y, c, folds)
        m, per = paired_fold_deltas(y, c, base, folds)
        bm, lo, hi = strat_bootstrap_auc_delta(y, c, base, n_boot=args.n_boot)
        units = np.array(per) / 1e-5
        rec = {
            "auc": ca, "delta": ca - ba,
            "mean_paired_fold_delta": m,
            "per_fold_delta": [float(x) for x in per],
            "pos_folds": int(sum(1 for x in per if x > 0)),
            "fold_se_units": float(units.std(ddof=1) / np.sqrt(len(units))),
            "mean_units": float(units.mean()),
            "bootstrap_mean": bm, "ci95": [lo, hi],
            "pearson": corr(c, base), "spearman": spearman(c, base),
            "significant": bool((lo > 0) or (hi < 0)),
        }
        out["cands"][cid] = rec
        print(f"  {cid:<34} auc={ca:.6f}  delta={ca-ba:+.6f} "
              f"({units.mean():+.2f}u, SE {rec['fold_se_units']:.2f}u)  "
              f"folds+={rec['pos_folds']}/5  "
              f"boot95=[{lo:+.6f},{hi:+.6f}]  "
              f"{'SIGNIFICANT' if rec['significant'] else 'inside noise'}  "
              f"rho={rec['spearman']:.5f}")

    save_json(out, REPORTS / "ablation_paired_stats.json")
    print("\nwrote", REPORTS / "ablation_paired_stats.json")


if __name__ == "__main__":
    main()