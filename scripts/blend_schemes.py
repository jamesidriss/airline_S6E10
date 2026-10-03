"""Blend weighting schemes that carry NO fitted selection freedom.

Equal-weighting every member lets a large, highly correlated family (here: LightGBM) dominate by
count. Family-balanced weighting is a *structural* prior, not a fit: each family gets equal total
weight, split evenly inside the family. Both schemes have zero degrees of freedom, so neither can
overfit the OOF vector the way a fitted weight vector can.

Reported alongside the nested logit-LR stack as a sanity check.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.ensemble import lab  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402


def family_of(meta: dict) -> str:
    f = meta.get("family", "?")
    if f == "lgbm" and meta.get("params", {}).get("extra_trees"):
        return "lgbm_xt"
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-auc", type=float, default=0.9605)
    ap.add_argument("--require-test", action="store_true")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--name", default="")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme(args.folds, y, tr[ID_COL]).folds
    idx = store._load_index()

    P, meta = {}, {}
    for k, v in idx.items():
        m = v.get("meta", {})
        if not m.get("auc") or m.get("family") in ("blend", "stack"):
            continue
        if m["auc"] < args.min_auc:
            continue
        if args.require_test and "test" not in v:
            continue
        o = store.load_oof(k)
        if len(o) != len(y):
            continue
        P[k] = o.astype("float64")
        meta[k] = m
    names = list(P)
    M = np.column_stack([lab.tform(P[n], "logit") for n in names])
    T = None
    if all("test" in idx[n] for n in names):
        T = np.column_stack([lab.tform(store.load_test(n).astype("float64"), "logit")
                             for n in names])

    fams = defaultdict(list)
    for i, n in enumerate(names):
        fams[family_of(meta[n])].append(i)
    print("family counts:", {k: len(v) for k, v in fams.items()})

    schemes = {}
    # 1. plain equal weight
    schemes["equal_all"] = np.ones(len(names)) / len(names)
    # 2. family-balanced: equal weight per family, split evenly inside the family
    w = np.zeros(len(names))
    for f, ii in fams.items():
        for i in ii:
            w[i] = 1.0 / (len(fams) * len(ii))
    schemes["family_balanced"] = w
    # 3. quality-weighted: softmax over rank with a fixed temperature (no fitted parameters)
    aucs = np.array([meta[n]["auc"] for n in names])
    z = (aucs - aucs.max()) / 2e-4
    schemes["quality_softmax_t2e-4"] = np.exp(z) / np.exp(z).sum()

    res = {}
    print(f"\n{'scheme':<22} {'OOF':>10}  folds")
    for nm, w in schemes.items():
        p = M @ w
        a = lab.auc(y, p)
        f = lab.fold_aucs(y, p, folds)
        res[nm] = {"oof_auc": a, "fold_aucs": f,
                   "weights": {n: round(float(x), 5) for n, x in zip(names, w) if x > 1e-6}}
        print(f"{nm:<22} {a:>10.6f}  {[round(x,6) for x in f]}")

    # nested logit-LR as an independent reference
    try:
        ns = lab.logistic_stack_nested(P, y, folds, kind="logit", C=1.0)
        res["nested_logit_lr"] = {"oof_auc": ns["auc"], "fold_aucs": ns["fold_aucs"]}
        print(f"{'nested_logit_lr':<22} {ns['auc']:>10.6f}")
    except Exception as exc:  # noqa: BLE001
        print("nested stack failed:", exc)

    best = max(res, key=lambda k: res[k]["oof_auc"])
    print(f"\nbest scheme: {best} = {res[best]['oof_auc']:.6f}")
    save_json(res, REPORTS / f"blend_schemes{('_' + args.name) if args.name else ''}.json")

    if args.build and T is not None and args.name:
        from scipy.special import expit

        w = schemes[best]
        store.save(f"blend_{args.name}", expit(M @ w), expit(T @ w), fold_scheme=args.folds,
                   meta={"family": "blend", "featureset": best, "auc": res[best]["oof_auc"],
                         "members": names})
        print(f"saved blend_{args.name}")


if __name__ == "__main__":
    main()