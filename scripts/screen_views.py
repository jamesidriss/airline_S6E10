"""Fold-safe 5-fold screen of a feature view against the stored `full` baseline.

The baseline is `abl_lgbm_full_primary` (extra_trees, 127 leaves, lr 0.02, seed 1), which is the
exact config this screen reruns, so the comparison is paired fold by fold. The verdict uses the
AGENTS.md gate: mean paired fold gain >= 1.5e-5, >= 2.5 x fold SE, positive in >= 4 of 5 folds.

No test predictions are produced: a screen that fails the gate should not pay for them, and a
screen that passes is re-run through `run_views.py --save-test` before anything is deployed.

Usage:
  python scripts/screen_views.py --views full_tec,full_all21te --model lgbm --tag p25a
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.view import VIEWS, ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import fold_auc  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _fit_lgbm_es, _fit_xgb_es, _inner_es_split  # noqa: E402

BASELINE = "abl_lgbm_full_primary"


def gate(deltas: list[float]) -> dict:
    d = np.asarray(deltas) * 1e4
    mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"mean_e4": round(mean, 4), "se_e4": round(se, 4), "t": round(mean / se, 2) if se else None,
            "pos": int((d > 0).sum()), "pass": bool(mean >= 0.15 and mean >= 2.5 * se and (d > 0).sum() >= 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="full_tec")
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="screen")
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    ntr = len(tr)
    params = {"extra_trees": True, "num_leaves": args.num_leaves, "learning_rate": args.learning_rate}
    base_oof = store.load_oof(BASELINE)
    result_path = REPORTS / f"{args.tag}_screen.json"
    out = json.loads(result_path.read_text()) if result_path.exists() else {}

    for view in args.views.split(","):
        if view in out:
            print(f"skip cached {view}")
            continue
        t0 = time.time()
        folds = get_scheme(args.folds, y, tr[ID_COL]).folds
        vb = ViewBuilder(tr, te, view)
        vb.build_static()
        oof = np.zeros(ntr)
        iters = []
        for k in sorted(set(folds.tolist())):
            fit = np.where(folds != k)[0]
            val = np.where(folds == k)[0]
            Xf, Xa, names = vb.assemble(fit, y, val)
            es_tr, es_idx = _inner_es_split(fit, y, args.seed + k)
            pos_of = {v: i for i, v in enumerate(fit)}
            tr_local = np.array([pos_of[v] for v in es_tr])
            es_local = np.array([pos_of[v] for v in es_idx])
            set_seed(args.seed)
            if args.model == "lgbm":
                p, it = _fit_lgbm_es(Xf[tr_local], y[es_tr], Xa["val"], params, args.seed,
                                     Xf[es_local], y[es_idx])
            elif args.model == "xgb":
                p, it = _fit_xgb_es(Xf[tr_local], y[es_tr], Xa["val"], params, args.seed,
                                    Xf[es_local], y[es_idx])
            else:
                raise ValueError(args.model)
            oof[val] = p
            iters.append(it)
            print(f"  [{view}/{args.model}] fold{k} auc={fold_auc(y, oof, folds)[k]:.6f} iter={it} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        fa = fold_auc(y, oof, folds)
        bf = fold_auc(y, base_oof, folds)
        d = [a - b for a, b in zip(fa, bf)]
        eid = f"{args.tag}_{args.model}_{view}_{args.folds}"
        pooled = float(roc_auc_score(y, oof))
        base_pooled = float(roc_auc_score(y, base_oof))
        store.save(eid, oof, None, fold_scheme=args.folds,
                   meta={"family": args.model, "featureset": view, "auc": round(pooled, 6),
                         "n_features": int(len(names)), "baseline": BASELINE})
        out[view] = {"pooled": pooled, "baseline_pooled": base_pooled,
                     "fold_aucs": [round(x, 6) for x in fa], "baseline_fold_aucs": [round(x, 6) for x in bf],
                     "paired_deltas": [round(x, 8) for x in d], "gate": gate(d),
                     "n_features": int(len(names)), "blocks": VIEWS[view], "iters": iters,
                     "seconds": round(time.time() - t0, 1), "exp_id": eid}
        save_json(out, result_path)
        print(f"== {view}: pooled {out[view]['pooled']:.6f} vs {out[view]['baseline_pooled']:.6f} "
              f"gate={out[view]['gate']}", flush=True)


if __name__ == "__main__":
    main()
