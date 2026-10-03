"""Zoo pass 4: fold-assignment diversity.

Observation that motivates this pass: our best *single* model (0.961242, 10-fold extra_trees
LightGBM) beats the strongest single model published anywhere in the S6E10 field (0.961166), yet
our stack (0.961487) sits below their stack (0.961647). A stack of better members losing to a
stack of worse members means their members are *more decorrelated*, not better.

Every member so far shares the same fold assignment (`primary`, seed 20261010), so all of them
train on the same 80 % subsets and their errors share a common component. This pass varies the
fold assignment itself — `shadow` (seed 777001) and `block10` — which is the one diversity axis
we have not used.

This doubles as the robustness check: if a conclusion holds on `primary` and reappears on
`shadow`, it is not an artefact of one fold split.
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
from src.models import realmlp as RM  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
import experiments_ledger as led  # noqa: E402

from scripts.run_zoo import CAT, LGBM, XGB, predict_test, run_gbdt  # noqa: E402

XT = dict(learning_rate=0.02, num_leaves=127, extra_trees=True)

GBDT4: list[tuple[str, str, str, str, dict, int]] = [
    # random-split LightGBM on the independent fold assignment
    ("xt_sh_s1", LGBM, "full", "shadow", dict(XT), 1),
    ("xt_sh_s2", LGBM, "full", "shadow", dict(XT), 2),
    ("xt_sh_s3", LGBM, "full", "shadow", dict(XT), 3),
    ("xt_sh_s4", LGBM, "full", "shadow", dict(XT), 4),
    ("xt_sh_s5", LGBM, "full", "shadow", dict(XT), 5),
    # deterministic LightGBM on the shadow assignment (diversity without the XT bias)
    ("lgb_sh_s1", LGBM, "full", "shadow", dict(learning_rate=0.02, num_leaves=127), 1),
    ("lgb_sh_s2", LGBM, "full", "shadow", dict(learning_rate=0.02, num_leaves=127), 2),
    # xgboost on the shadow assignment
    ("xgb_sh_s1", XGB, "full", "shadow", dict(learning_rate=0.04, max_depth=6, min_child_weight=5,
                                                colsample_bytree=0.85, subsample=0.9,
                                                reg_lambda=5.0, reg_alpha=0.05), 1),
    # more 10-fold random-split LightGBM (different 90 % subsets again)
    ("xt_f10_s3", LGBM, "full", "block10", dict(XT), 3),
    ("xt_f10_s4", LGBM, "full", "block10", dict(XT), 4),
    ("xt_sh_f10", LGBM, "full", "block10", dict(XT), 7),
    # catboost on the shadow assignment (only 2 cat members use primary)
    ("cat_sh_s1", CAT, "full", "shadow", dict(learning_rate=0.03, depth=10, l2_leaf_reg=6.0), 1),
]

NN4: list[tuple[str, str, str, int, int, dict, int]] = [
    ("rm_sh_s1", "full", "shadow", 6, 8, dict(batch_size=256), 1),
    ("rm_sh_s2", "full", "shadow", 6, 8, dict(batch_size=256), 2),
    ("rm_sh_s3", "full", "shadow", 6, 8, dict(batch_size=256), 3),
    ("rm_f10_e6", "full", "block10", 6, 8, dict(batch_size=256), 1),
    ("rm_sh_ogs", "full_ogsurf", "shadow", 6, 8, dict(batch_size=256), 4),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="z4")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-gbdt", action="store_true")
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--save-test", action="store_true")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    ntr, nte = len(tr), len(te)
    idx = store._load_index()
    sel = args.only.split(",") if args.only else None
    rows = []

    def cached(eid):
        return eid in idx and "test" in idx[eid]

    if not args.skip_gbdt:
        for name, family, view, scheme, params, seed in GBDT4:
            if sel and name not in sel:
                continue
            eid = f"{args.tag}_{name}"
            if cached(eid):
                print(f"[z4] skip {eid}")
                continue
            vb = ViewBuilder(tr, te, view)
            vb.build_static()
            folds = get_scheme(scheme, y, tr[ID_COL]).folds
            oof, tst, iters = np.zeros(ntr), np.zeros(nte), []
            nks = len(set(folds.tolist()))
            set_seed(seed)
            t0 = time.time()
            for k in sorted(set(folds.tolist())):
                fit = np.where(folds != k)[0]
                val = np.where(folds == k)[0]
                Xf, Xa, _ = vb.assemble(fit, y, val, None, inner_seed=k)
                o, it = run_gbdt(family, Xf, Xa["val"], None, y, fit, val, params, seed)
                iters.append(it)
                oof[val] = o
                del Xf, Xa
            n_it = int(np.median(iters)) if any(iters) else 0
            set_seed(seed)
            for k in sorted(set(folds.tolist())):
                fit = np.where(folds != k)[0]
                val = np.where(folds == k)[0]
                Xf, Xa, _ = vb.assemble(fit, y, val, np.arange(ntr, ntr + nte), inner_seed=k)
                tst += predict_test(family, Xf, Xa["test"], y, fit, params, seed, n_it) / nks
                del Xf, Xa
            auc = float(roc_auc_score(y, oof))
            fa = [float(roc_auc_score(y[folds == k], oof[folds == k]))
                  for k in sorted(set(folds.tolist()))]
            store.save(eid, oof, tst if args.save_test else None, fold_scheme=scheme,
                       meta={"family": family, "featureset": view, "auc": round(auc, 6),
                             "params": params, "seed": seed})
            led.log_experiment(family=family, featureset=view, params=params, seed=seed,
                               fold_scheme=scheme, oof_auc=auc, fold_aucs=fa, oof=oof,
                               duration_s=time.time() - t0, data_hash=f"tr{ntr}-te{nte}",
                               test_pred=tst if args.save_test else None, exp_id=eid,
                               notes="z4-fold-diversity", verdict="zoo-member",
                               extra={"fold_ids": folds.tolist()})
            print(f"[z4] {name:<14} {family:<5} {scheme:<9} OOF={auc:.6f} ({time.time()-t0:.0f}s)",
                  flush=True)
            rows.append({"name": name, "exp_id": eid, "scheme": scheme, "oof_auc": round(auc, 6)})
            save_json(rows, REPORTS / f"{args.tag}_results.json")

    if not args.skip_nn:
        for name, view, scheme, epochs, ens, extra, seed in NN4:
            if sel and name not in sel:
                continue
            eid = f"{args.tag}_{name}"
            if cached(eid):
                print(f"[z4] skip {eid}")
                continue
            vb = ViewBuilder(tr, te, view)
            vb.build_static()
            folds = get_scheme(scheme, y, tr[ID_COL]).folds
            p = {"n_epochs": epochs, "n_ens": ens, "random_seed": seed}
            p.update(extra)
            set_seed(seed)
            t0 = time.time()
            oof, tst, fa, dur = RM.realmlp_view(vb, folds, y, ntr, nte, params=p, seed=seed,
                                                device="cuda", twin=True)
            auc = float(roc_auc_score(y, oof))
            store.save(eid, oof, tst if args.save_test else None, fold_scheme=scheme,
                       meta={"family": "realmlp", "featureset": view, "auc": round(auc, 6),
                             "params": p})
            led.log_experiment(family="realmlp", featureset=view, params=p, seed=seed,
                               fold_scheme=scheme, oof_auc=auc, fold_aucs=fa, oof=oof,
                               duration_s=dur, data_hash=f"tr{ntr}-te{nte}",
                               test_pred=tst if args.save_test else None, exp_id=eid,
                               notes="z4-nn", verdict="zoo-member",
                               extra={"fold_ids": folds.tolist()})
            print(f"[z4] {name:<14} realmlp {scheme:<9} OOF={auc:.6f} ({dur:.0f}s)", flush=True)
            rows.append({"name": name, "exp_id": eid, "scheme": scheme, "oof_auc": round(auc, 6)})
            save_json(rows, REPORTS / f"{args.tag}_results.json")

    print("\n=== Z4 SUMMARY ===")
    print(pd.DataFrame(rows).sort_values("oof_auc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()