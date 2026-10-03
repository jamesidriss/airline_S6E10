"""Zoo pass 5: the two genuinely untried model families, plus a stronger averaging budget.

Rationale. Every member so far is LightGBM / XGBoost / CatBoost / RealMLP. Logit-space
correlation among them is 0.995-0.999, and adding more of them now buys +3e-6 to +5e-6 each,
i.e. the ensemble is saturated *within these families*. The one lever still open is a model
family we have never run:

  * **TabM** (pytabkit): a deep ensemble of parameter-efficient linear layers. Different
    inductive bias from both GBDTs and MLPs, and it was the single strongest family in
    Playground S6E1.
  * **RealMLP with a large `n_ens`**: `n_ens` is an *internal* averaging mechanism. Since the
    whole extra_trees result says "average away the i.i.d. label noise", raising the internal
    ensemble budget is the direct extension of the one mechanism that has actually worked.

Note on the plateau: 0.9615 is the current system plateau, not a proven ceiling. The
original-data grouped-fold result (0.9949 vs our 0.961) is *evidence* that the synthetic p(x) is
weaker, but it is an inference, not a bound.
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

# name, model, view, scheme, epochs, n_ens / tabm_k, extra params, seed
Z5: list[tuple[str, str, str, str, int, int, dict, int]] = [
    ("tabm_e25", "tabm", "full", "primary", 25, 8, dict(batch_size=2048, d_block=256,
                                                       n_blocks=3, lr=0.002), 1),
    ("tabm_e25_s2", "tabm", "full", "primary", 25, 8, dict(batch_size=2048, d_block=256,
                                                          n_blocks=3, lr=0.002), 2),
    ("tabm_e40_core3", "tabm", "core3", "primary", 40, 8, dict(batch_size=2048, d_block=192,
                                                              n_blocks=3, lr=0.002), 3),
    ("rm_ens32", "realmlp", "full", "primary", 6, 32, dict(batch_size=256), 1),
    ("rm_ens32_s2", "realmlp", "full", "primary", 6, 32, dict(batch_size=256), 2),
    ("rm_ens16_ogs", "realmlp", "full_ogsurf", "primary", 6, 16, dict(batch_size=256), 3),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="z5")
    ap.add_argument("--only", default="")
    ap.add_argument("--save-test", action="store_true")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    ntr, nte = len(tr), len(te)
    idx = store._load_index()
    sel = args.only.split(",") if args.only else None
    rows = []

    for name, model, view, scheme, epochs, ens, extra, seed in Z5:
        if sel and name not in sel:
            continue
        eid = f"{args.tag}_{name}"
        if eid in idx and "test" in idx[eid]:
            print(f"[z5] skip {eid}")
            continue
        vb = ViewBuilder(tr, te, view)
        vb.build_static()
        folds = get_scheme(scheme, y, tr[ID_COL]).folds
        set_seed(seed)
        t0 = time.time()
        try:
            if model == "tabm":
                p = {"n_epochs": epochs, "tabm_k": ens, "random_seed": seed}
                p.update(extra)
                oof, tst, fa, dur = RM.tabm_view(vb, folds, y, ntr, nte, params=p,
                                                  seed=seed, device="cuda", twin=True)
            else:
                p = {"n_epochs": epochs, "n_ens": ens, "random_seed": seed}
                p.update(extra)
                oof, tst, fa, dur = RM.realmlp_view(vb, folds, y, ntr, nte, params=p,
                                                    seed=seed, device="cuda", twin=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[z5] FAILED {name}: {exc!r}", flush=True)
            continue
        auc = float(roc_auc_score(y, oof))
        store.save(eid, oof, tst if args.save_test else None, fold_scheme=scheme,
                   meta={"family": model, "featureset": view, "auc": round(auc, 6), "params": p})
        led.log_experiment(family=model, featureset=view, params=p, seed=seed,
                           fold_scheme=scheme, oof_auc=auc, fold_aucs=fa, oof=oof,
                           duration_s=dur, data_hash=f"tr{ntr}-te{nte}",
                           test_pred=tst if args.save_test else None, exp_id=eid,
                           notes="z5-new-family", verdict="zoo-member",
                           extra={"fold_ids": folds.tolist()})
        print(f"[z5] {name:<16} {model:<8} {view:<13} OOF={auc:.6f} ({dur:.0f}s)", flush=True)
        rows.append({"name": name, "exp_id": eid, "model": model, "view": view,
                     "oof_auc": round(auc, 6)})
        save_json(rows, REPORTS / f"{args.tag}_results.json")

    print("\n=== Z5 SUMMARY ===")
    print(pd.DataFrame(rows).sort_values("oof_auc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()