"""Gated ensemble selection.

Forward selection in logit space with the admission gate applied at every step, so the reported
stack OOF never contains a member that did not earn its place. Two controls against fitting the
single OOF vector:

1. the **nested** stacker (meta coefficients fitted on 4 folds, scored on the 5th);
2. the **admission gate** (mean paired fold gain >= 1.5e-5, >= 2.5 x paired fold SE, >= 4/5 folds).

Output: reports/stack_<name>.json and a saved blended prediction pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.ensemble import lab  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stack")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--min-units", type=float, default=1.5)
    ap.add_argument("--se-mult", type=float, default=2.5)
    ap.add_argument("--min-pos", type=int, default=4)
    ap.add_argument("--max-members", type=int, default=24)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme(args.folds, y, tr[ID_COL]).folds
    idx = store._load_index()
    excl = set(args.exclude.split(",")) if args.exclude else set()

    cand = []
    for k, v in idx.items():
        if k in excl or not v.get("meta", {}).get("auc"):
            continue
        if v.get("fold_scheme", "primary") not in (None, args.folds):
            continue
        o = store.load_oof(k).astype("float64")
        if len(o) != len(y):
            continue
        cand.append((k, float(v["meta"]["auc"]), v.get("meta", {})))
    cand.sort(key=lambda t: -t[1])

    print(f"### {len(cand)} candidate members (scheme={args.folds})")
    for k, a, m in cand:
        print(f"  {a:.6f}  {k:<44} {m.get('family','')}/{m.get('featureset','')}")

    T = {k: lab.tform(store.load_oof(k).astype("float64"), "logit") for k, _, _ in cand}
    have_test = {k: ("test" in idx[k]) for k, _, _ in cand}

    # ---------------------------------------------------------------- forward selection
    selected: list[str] = [cand[0][0]]
    best_auc = cand[0][1]
    best_folds = lab.fold_aucs(y, T[selected[0]], folds)
    history = [{"step": 0, "added": selected[0], "auc": round(best_auc, 6), "n": 1,
                "gate": None}]
    print(f"\nstep 0: seed = {selected[0]}  auc={best_auc:.6f}")

    remaining = [k for k, _, _ in cand if k != selected[0]]
    while remaining and len(selected) < args.max_members:
        trial_best = None
        for k in remaining:
            p = (T[selected[0]] * len(selected) + T[k]) / (len(selected) + 1)
            a = lab.auc(y, p)
            f = lab.fold_aucs(y, p, folds)
            g = lab.admission_gate(k, best_folds, f, args.min_units, args.se_mult, args.min_pos)
            if trial_best is None or a > trial_best[1]:
                trial_best = (k, a, f, g)
        k, a, f, g = trial_best
        ok = g["admit"]
        print(f"  candidate {k:<44} auc={a:.6f}  gate: mean={g['mean_units']:.2f}u "
              f"se={g['fold_se_units']:.2f}u pos={g['pos_folds']}/5 -> "
              f"{'ADMIT' if ok else 'reject'}")
        history.append({"step": len(selected), "added": k, "auc": round(a, 6),
                        "n": len(selected) + 1, "gate": g})
        if not ok:
            remaining.remove(k)
            if not remaining:
                break
            continue
        selected.append(k)
        best_auc, best_folds = a, f
        remaining.remove(k)
        print(f"    -> accepted. stack = {len(selected)} members, auc={best_auc:.6f}")

    P = {k: store.load_oof(k).astype("float64") for k in selected}
    w = np.full(len(selected), 1.0 / len(selected))
    M = np.column_stack([T[k] for k in selected]) @ w
    final_auc = lab.auc(y, M)
    final_folds = lab.fold_aucs(y, M, folds)
    print(f"\nSELECTED {len(selected)} members")
    for k, wi in zip(selected, w):
        print(f"   {wi:.4f}  {k}")
    print(f"stack OOF AUC = {final_auc:.6f}  folds={[round(x,6) for x in final_folds]}")

    # nested logistic stack as an independent check
    try:
        ns = lab.logistic_stack_nested(P, y, folds, kind="logit", C=1.0)
        print(f"nested logit-LR stack  = {ns['auc']:.6f}")
    except Exception as exc:  # noqa: BLE001
        ns = {"auc": None}
        print("nested stack failed:", exc)

    out = {"members": selected, "weights": w.tolist(), "stack_oof_auc": final_auc,
           "fold_aucs": final_folds, "nested_logit_lr_auc": ns["auc"],
           "best_single_auc": cand[0][1], "n_candidates": len(cand), "history": history}
    save_json(out, REPORTS / f"{args.tag}_report.json")

    if not args.dry_run and all(have_test[k] for k in selected):
        from scipy.special import expit

        MT = np.column_stack([lab.tform(store.load_test(k).astype("float64"), "logit")
                              for k in selected]) @ w
        store.save(f"{args.tag}_final", M, expit(MT), fold_scheme=args.folds,
                   meta={"family": "stack", "featureset": "gated", "auc": round(final_auc, 6),
                         "members": selected})
        print("saved stack predictions")
    else:
        missing = [k for k in selected if not have_test[k]]
        print("NO TEST PREDICTIONS for:", missing)


if __name__ == "__main__":
    main()