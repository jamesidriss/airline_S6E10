"""Learning curve on one immutable fold: is 0.9622 a ceiling or a sample-size artefact?

Every candidate we have measured lands within 2e-5 of 0.9622. Two very different worlds produce
that: (a) the estimator is already at the Bayes limit of the synthetic label function, or (b) the
estimator is data-starved and the limit is higher, in which case more data / lower-variance
fitting is a real lever worth 1e-4 or more.

The discriminating measurement is the slope of AUC against the number of training rows, on a
*fixed* evaluation fold so the comparison is paired: train on nested subsets of fold 0's fit
rows (25 %, 50 %, 100 %) and score the same fold 0. Extrapolating that slope to "infinite data"
gives the headroom still available to a better estimator.

Config is identical to `abl_lgbm_full_primary` so the 100 % point is directly comparable with
the stored number (`fold0 0.961103`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _fit_lgbm_es, _inner_es_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--fracs", default="0.25,0.50,1.00")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--tag", default="learning_curve")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y, tr[ID_COL]).folds
    vf, vy = np.where(folds == args.fold)[0], None
    fit_all = np.where(folds != args.fold)[0]

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()
    Xf, Xa, names = vb.assemble(fit_all, y, vf)
    print(f"view={args.view} fold={args.fold} fit={len(fit_all)} eval={len(vf)} cols={len(names)}", flush=True)

    from sklearn.metrics import roc_auc_score
    params = {"extra_trees": True, "num_leaves": 127, "learning_rate": 0.02}
    out = {"view": args.view, "fold": args.fold, "n_features": len(names), "points": {}}
    for frac in [float(f) for f in args.fracs.split(",")]:
        rng = np.random.default_rng(2026)
        sel = fit_all if frac >= 1.0 else np.sort(rng.choice(fit_all, int(len(fit_all) * frac), replace=False))
        pos_of = {v: i for i, v in enumerate(fit_all)}
        loc = np.array([pos_of[v] for v in sel])
        aucs, iters = [], []
        for s in range(args.seeds):
            es_tr, es_idx = _inner_es_split(sel, y, 1 + s)
            ltr = np.array([pos_of[v] for v in es_tr])
            les = np.array([pos_of[v] for v in es_idx])
            set_seed(1 + s * 17)
            p, it = _fit_lgbm_es(Xf[ltr], y[es_tr], Xa["val"], params, 1 + s * 17,
                                 Xf[les], y[es_idx])
            aucs.append(float(roc_auc_score(y[vf], p)))
            iters.append(int(it))
            print(f"  frac={frac:.2f} seed={s} auc={aucs[-1]:.6f} iter={it} ({time.time():.0f})", flush=True)
        out["points"][f"{frac:.2f}"] = {"n_fit": int(len(sel)), "auc_mean": float(np.mean(aucs)),
                                        "auc_seeds": aucs, "iters": iters}
        save_json(out, REPORTS / f"{args.tag}.json")

    ks = sorted(out["points"], key=lambda k: out["points"][k]["n_fit"])
    ns = [out["points"][k]["n_fit"] for k in ks]
    au = [out["points"][k]["auc_mean"] for k in ks]
    print("\n  n_fit      auc")
    for n, a in zip(ns, au):
        print(f"  {n:7d}  {a:.6f}")
    if len(ks) >= 2:
        slope = (au[-1] - au[-2]) / (np.log(ns[-1]) - np.log(ns[-2]))
        out["slope_per_log2"] = float(slope * np.log(2))
        # headroom to infinite data from a power law fit on the last two points
        out["headroom_two_point_e4"] = float(slope * (np.log(10 ** 9) - np.log(ns[-1])) * 1e4)
        print(f"\nslope = {out['slope_per_log2']*1e4:+.3f} e-4 per doubling of n")
        print(f"extrapolated headroom to 1e9 rows = {out['headroom_two_point_e4']:+.3f} e-4")
    save_json(out, REPORTS / f"{args.tag}.json")


if __name__ == "__main__":
    main()
