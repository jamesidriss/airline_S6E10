"""Zoo pass 3: close the two measured gaps.

(a) **RealMLP optimisation.** Our RealMLP (0.96082) sits ~3.5e-4 below the strongest RealMLP
    reported in the field. Their recipe uses `batch_size=256`; ours used 4096. At 560k training
    rows that is a 16x difference in gradient steps per epoch, which is the most likely cause.
    This pass sweeps batch size, epochs and width.

(b) **CatBoost breadth.** We have only one CatBoost member, and it is one of only three
    genuinely different algorithm families in the pool. More depth/loss variants are cheap
    diversity.

(c) **10-fold random-split LightGBM**, whose fold models see different 90% subsets and are
    therefore decorrelated from the 5-fold ones.
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

from scripts.run_zoo import CAT, LGBM, predict_test, run_gbdt  # noqa: E402

# name, family, view, scheme, params, seed
GBDT3: list[tuple[str, str, str, str, dict, int]] = [
    ("cat_d6", CAT, "full", "primary", dict(learning_rate=0.04, depth=6, l2_leaf_reg=5.0), 1),
    ("cat_d10", CAT, "full", "primary", dict(learning_rate=0.03, depth=10, l2_leaf_reg=6.0), 2),
    ("cat_d8_s2", CAT, "full", "primary", dict(learning_rate=0.04, depth=8), 3),
    ("cat_core3", CAT, "core3", "primary", dict(learning_rate=0.04, depth=8), 4),
    ("cat_ogs", CAT, "full_ogsurf", "primary", dict(learning_rate=0.04, depth=8), 5),
    ("cat_f10", CAT, "full", "block10", dict(learning_rate=0.04, depth=8), 1),
    ("xt_f10", LGBM, "full", "block10", dict(learning_rate=0.02, num_leaves=127,
                                             extra_trees=True), 1),
    ("xt_f10_s2", LGBM, "full", "block10", dict(learning_rate=0.02, num_leaves=127,
                                                extra_trees=True), 2),
]

# name, view, scheme, epochs, ens, extra params
NN3: list[tuple[str, str, str, int, int, dict]] = [
    ("rm_bs256_e6", "full", "primary", 6, 8, dict(batch_size=256)),
    ("rm_bs1024_e8", "full", "primary", 8, 8, dict(batch_size=1024)),
    ("rm_bs256_e10", "full", "primary", 10, 8, dict(batch_size=256)),
    ("rm_bs512_e6_wide", "full", "primary", 6, 8,
     dict(batch_size=512, hidden_sizes=[1024, 512, 256])),
    ("rm_bs256_e6_ogs", "full_ogsurf", "primary", 6, 8, dict(batch_size=256)),
    ("rm_bs256_e6_core3", "core3", "primary", 6, 8, dict(batch_size=256)),
    ("rm_bs256_f10", "full", "block10", 6, 8, dict(batch_size=256)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="z3")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-gbdt", action="store_true")
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--save-test", action="store_true")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    ntr, nte = len(tr), len(te)
    idx = store._load_index()
    rows = []
    sel = args.only.split(",") if args.only else None

    def cached(eid):
        return eid in idx and "test" in idx[eid]

    if not args.skip_gbdt:
        for name, family, view, scheme, params, seed in GBDT3:
            if sel and name not in sel:
                continue
            eid = f"{args.tag}_{name}"
            if cached(eid):
                print(f"[z3] skip {eid}")
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
                               notes="z3", verdict="zoo-member", extra={"fold_ids": folds.tolist()})
            print(f"[z3] {name:<18} {family:<5} {view:<14} {scheme:<9} OOF={auc:.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            rows.append({"name": name, "exp_id": eid, "family": family, "oof_auc": round(auc, 6)})
            save_json(rows, REPORTS / f"{args.tag}_results.json")

    if not args.skip_nn:
        for name, view, scheme, epochs, ens, extra in NN3:
            if sel and name not in sel:
                continue
            eid = f"{args.tag}_{name}"
            if cached(eid):
                print(f"[z3] skip {eid}")
                continue
            vb = ViewBuilder(tr, te, view)
            vb.build_static()
            folds = get_scheme(scheme, y, tr[ID_COL]).folds
            p = {"n_epochs": epochs, "n_ens": ens, "random_seed": 1}
            p.update(extra)
            set_seed(1)
            t0 = time.time()
            oof, tst, fa, dur = RM.realmlp_view(vb, folds, y, ntr, nte, params=p, seed=1,
                                                device="cuda", twin=True)
            auc = float(roc_auc_score(y, oof))
            store.save(eid, oof, tst if args.save_test else None, fold_scheme=scheme,
                       meta={"family": "realmlp", "featureset": view, "auc": round(auc, 6),
                             "params": p})
            led.log_experiment(family="realmlp", featureset=view, params=p, seed=1,
                               fold_scheme=scheme, oof_auc=auc, fold_aucs=fa, oof=oof,
                               duration_s=dur, data_hash=f"tr{ntr}-te{nte}",
                               test_pred=tst if args.save_test else None, exp_id=eid,
                               notes="z3-nn", verdict="zoo-member",
                               extra={"fold_ids": folds.tolist()})
            print(f"[z3] {name:<18} realmlp {view:<14} {scheme:<9} OOF={auc:.6f} ({dur:.0f}s)",
                  flush=True)
            rows.append({"name": name, "exp_id": eid, "family": "realmlp", "oof_auc": round(auc, 6)})
            save_json(rows, REPORTS / f"{args.tag}_results.json")

    print("\n=== Z3 SUMMARY ===")
    print(pd.DataFrame(rows).sort_values("oof_auc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()