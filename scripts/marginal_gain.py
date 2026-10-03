"""Marginal ensemble gain of a candidate member against the current pool.

A member can be individually worse than the champion and still be worth admitting if its errors
are different. This measures the gain from adding it to the equal-logit blend of everything else,
with a paired bootstrap over folds so the decision is not made on a single pooled number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.ensemble import lab  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--min-auc", type=float, default=0.9605)
    ap.add_argument("--require-test", action="store_true")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y, tr[ID_COL]).folds
    idx = store._load_index()

    pool = {}
    for k, v in idx.items():
        m = v.get("meta", {})
        if not m.get("auc") or m.get("family") in ("blend", "stack"):
            continue
        if m["auc"] < args.min_auc:
            continue
        if args.require_test and "test" not in v:
            continue
        o = store.load_oof(k)
        if len(o) == len(y):
            pool[k] = o.astype("float64")
    names = list(pool)
    M = np.column_stack([lab.tform(pool[n], "logit") for n in names])
    base_auc = lab.auc(y, M.mean(axis=1))
    print(f"pool = {len(names)} members, equal-logit AUC = {base_auc:.6f}")

    out = {"pool_n": len(names), "pool_auc": base_auc, "cands": {}}
    for cid in args.cands.split(","):
        if cid not in pool:
            print(f"  {cid:<36} NOT IN POOL")
            continue
        i = names.index(cid)
        keep = [j for j in range(len(names)) if j != i]
        without = lab.auc(y, M[:, keep].mean(axis=1))
        withc = lab.auc(y, M.mean(axis=1))
        # paired bootstrap of (with - without) on the two vectors
        from src.validation.compare import strat_bootstrap_auc_delta

        v_with = M.mean(axis=1)
        v_without = M[:, keep].mean(axis=1)
        bm, lo, hi = strat_bootstrap_auc_delta(y, v_with, v_without, n_boot=args.n_boot)
        out["cands"][cid] = {"with": withc, "without": without, "gain": withc - without,
                             "boot": [bm, lo, hi],
                             "significant": bool(lo > 0 or hi < 0)}
        print(f"  {cid:<36} with={withc:.6f} without={without:.6f} "
              f"gain={withc-without:+.6f}  boot95=[{lo:+.6f},{hi:+.6f}]  "
              f"{'WORTH ADMITTING' if lo > 0 else 'negligible / harmful'}")

    save_json(out, REPORTS / "marginal_gain.json")
    print("\nwrote", REPORTS / "marginal_gain.json")


if __name__ == "__main__":
    main()