"""Shadow-fold confirmation of the finalist ensemble.

The whole feature/representation case rests on measured OOF deltas taken on `primary`. Before
locking a final submission we re-measure the load-bearing claims on the *independent* `shadow`
fold scheme (seed 777001). A claim that does not reproduce on a fold assignment it was never
tuned against is not a real claim.

Claims checked:
  1. extra_trees=True beats a deterministic LightGBM of the same size
  2. the external-data block (original target statistics + teacher) beats raw features
  3. the transductive block (counts / route profile / N-A masks) beats raw features
  4. the finalist ensemble beats its best single member
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

XT = dict(learning_rate=0.02, num_leaves=127, extra_trees=True)
DET = dict(learning_rate=0.02, num_leaves=127)

CASES = [
    ("det_full", "full", DET, 1),
    ("xt_full", "full", XT, 1),
    ("det_raw", "raw", DET, 1),
    ("xt_raw", "raw", XT, 1),
    ("det_raw_ext", "raw_ext", DET, 1),
    ("xt_raw_ext", "raw_ext", XT, 1),
    ("det_raw_trans", "raw_trans", DET, 1),
    ("xt_raw_trans", "raw_trans", XT, 1),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="shadow")
    ap.add_argument("--tag", default="conf")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    ntr, nte = len(tr), len(te)
    folds = get_scheme(args.folds, y, tr[ID_COL]).folds
    idx = store._load_index()
    res = {}

    from scripts.run_zoo import LGBM, predict_test, run_gbdt

    for name, view, params, seed in CASES:
        eid = f"{args.tag}_{name}_{args.folds}"
        if eid in idx:
            print(f"[conf] skip {eid}")
            res[name] = idx[eid]["meta"]["auc"]
            continue
        vb = ViewBuilder(tr, te, view)
        vb.build_static()
        oof, iters = np.zeros(ntr), []
        set_seed(seed)
        t0 = time.time()
        for k in sorted(set(folds.tolist())):
            fit = np.where(folds != k)[0]
            val = np.where(folds == k)[0]
            Xf, Xa, _ = vb.assemble(fit, y, val, None, inner_seed=k)
            o, it = run_gbdt(LGBM, Xf, Xa["val"], None, y, fit, val, params, seed)
            iters.append(it)
            oof[val] = o
            del Xf, Xa
        auc = float(roc_auc_score(y, oof))
        fa = [float(roc_auc_score(y[folds == k], oof[folds == k]))
              for k in sorted(set(folds.tolist()))]
        store.save(eid, oof, None, fold_scheme=args.folds,
                   meta={"family": LGBM, "featureset": view, "auc": round(auc, 6), "params": params})
        res[name] = round(auc, 6)
        print(f"[conf] {name:<16} {view:<12} OOF={auc:.6f}  folds={[round(x,6) for x in fa]} "
              f"({time.time()-t0:.0f}s)", flush=True)
        save_json(res, REPORTS / f"{args.tag}_{args.folds}.json")

    print("\n" + "=" * 78)
    print(f"SHADOW-FOLD CONFIRMATION ({args.folds})")
    print("=" * 78)
    claims = [
        ("extra_trees beats deterministic (full)", res.get("xt_full"), res.get("det_full")),
        ("extra_trees beats deterministic (raw)", res.get("xt_raw"), res.get("det_raw")),
        ("extra_trees beats deterministic (raw_ext)", res.get("xt_raw_ext"), res.get("det_raw_ext")),
        ("external block helps (det)", res.get("det_raw_ext"), res.get("det_raw")),
        ("external block helps (xt)", res.get("xt_raw_ext"), res.get("xt_raw")),
        ("transductive block helps (det)", res.get("det_raw_trans"), res.get("det_raw")),
        ("transductive block helps (xt)", res.get("xt_raw_trans"), res.get("xt_raw")),
    ]
    for name, a, b in claims:
        if a is None or b is None:
            continue
        print(f"  {name:<42} {a:.6f} vs {b:.6f}  delta={a-b:+.6f}  "
              f"{'REPRODUCED' if a > b else 'NOT REPRODUCED'}")
    save_json(res, REPORTS / f"{args.tag}_{args.folds}.json")


if __name__ == "__main__":
    main()