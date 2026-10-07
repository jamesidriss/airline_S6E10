"""Run neural / other model families on a ViewBuilder's per-fold matrices.

Usage:
  python scripts/run_models.py --view full --model realmlp --folds primary
  python scripts/run_models.py --view full --model tabm,realmlp_notwin
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.models import realmlp as RM  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
import experiments_ledger as led  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--model", default="realmlp")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--tag", default="nn")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--ens", type=int, default=8)
    ap.add_argument("--no-twin", action="store_true")
    ap.add_argument("--save-test", action="store_true")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    ntr, nte = len(tr), len(te)
    dh = f"tr{ntr}-te{nte}"
    extra = json.loads(args.params)

    cache = REPORTS / f"{args.tag}_{args.model}_results.json"
    summary = json.loads(cache.read_text()) if cache.exists() else []

    for scheme in args.folds.split(","):
        folds = get_scheme(scheme, y, tr[ID_COL]).folds
        vb = ViewBuilder(tr, te, args.view)
        vb.build_static()
        for mdl in args.model.split(","):
            key = f"{RM.TRAINING_PROTOCOL}|{mdl}|{args.view}|{scheme}|{args.seed}|e{args.epochs}|ns{args.ens}|tw{0 if args.no_twin else 1}"
            if any(s.get("key") == key for s in summary):
                print("skip cached", key)
                continue
            p = {"n_epochs": args.epochs, "n_ens": args.ens, "random_seed": args.seed}
            p.update(extra)
            print(f"\n### {mdl} view={args.view} scheme={scheme} params={p} twin={not args.no_twin}", flush=True)
            set_seed(args.seed)
            t0 = time.time()
            twin = not args.no_twin
            if mdl == "realmlp":
                oof, test, fa, dur = RM.realmlp_view(vb, folds, y, ntr, nte, params=p,
                                                     seed=args.seed, device=args.device, twin=twin)
            elif mdl == "tabm":
                oof, test, fa, dur = RM.tabm_view(vb, folds, y, ntr, nte, params=p,
                                                  seed=args.seed, device=args.device, twin=twin)
            else:
                raise ValueError(mdl)
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y, oof))
            print(f"  ==> {mdl}/{args.view}/{scheme} OOF AUC = {auc:.6f} ({dur:.0f}s) "
                  f"folds={[round(x,6) for x in fa]}", flush=True)
            eid = f"{args.tag}_{mdl}_{args.view}_{scheme}_s{args.seed}_e{args.epochs}_{RM.TRAINING_PROTOCOL}"
            store.save(eid, oof, test if args.save_test else None, fold_scheme=scheme,
                       meta={"family": mdl, "featureset": args.view, "auc": round(auc, 6), "seed": args.seed,
                             "training_protocol": RM.TRAINING_PROTOCOL})
            led.log_experiment(family=mdl, featureset=args.view, params=p, seed=args.seed,
                               fold_scheme=scheme, oof_auc=auc, fold_aucs=fa, oof=oof,
                               duration_s=dur, data_hash=dh,
                               test_pred=test if args.save_test else None, exp_id=eid,
                               notes=f"twin={twin}", verdict="nn-view",
                               extra={"fold_ids": folds.tolist(), "params": p, "twin": twin})
            summary.append({"key": key, "model": mdl, "view": args.view, "scheme": scheme,
                            "seed": args.seed, "epochs": args.epochs, "ens": args.ens,
                            "twin": twin, "oof_auc": round(auc, 6),
                            "fold_aucs": [round(x, 6) for x in fa], "exp_id": eid,
                            "dur_s": round(dur)})
            save_json(summary, cache)
    print("\n=== SUMMARY ===")
    print(pd.DataFrame(summary).sort_values("oof_auc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
