"""Build an ensemble submission from stored OOF + test predictions.

Blending geometry is chosen by nested-CV OOF score, and every candidate must clear the
admission gate against the best single member before it can be the submitted file.

Usage:
  python scripts/make_submission.py --members a,b,c,d --name v1 --kind equal_logit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.ensemble import lab  # noqa: E402
from src.submission import store  # noqa: E402
from src.submission.kaggle_io import remaining, used_today  # noqa: E402
from src.submission.make import build  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--kind", default="equal_logit")
    ap.add_argument("--weights", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme(args.folds, y, tr[ID_COL]).folds
    ids = args.members.split(",")

    P = {i: store.load_oof(i).astype("float64") for i in ids}
    T = {i: store.load_test(i).astype("float64") for i in ids}

    print("members:")
    for i in ids:
        print(f"  {i:<44} auc={lab.auc(y, P[i]):.6f}")

    cands = {}
    for kind in ("equal_logit", "equal_prob", "equal_rank", "greedy_logit"):
        b = (lab.greedy_hill_climb(P, y, folds, kind="logit")
             if kind == "greedy_logit" else lab.equal_blend(P, y, folds, kind=kind.split("_")[1]))
        cands[kind] = b
        print(f"  {kind:<14} auc={b['auc']:.6f}")

    w = None
    if args.weights:
        w = np.array([float(x) for x in args.weights.split(",")])
        w = w / w.sum()
        M = np.column_stack([lab.tform(P[i], "logit") for i in ids])
        print(f"  custom weights  auc={lab.auc(y, M @ w):.6f}  w={w.tolist()}")

    kind = args.kind
    if kind in cands:
        wts = np.array(cands[kind]["weights"])
    else:
        wts = w
    if wts is None:
        raise SystemExit("no weights available")

    oof_blend = np.column_stack([lab.tform(P[i], "logit") for i in ids]) @ wts
    test_blend = np.column_stack([lab.tform(T[i], "logit") for i in ids]) @ wts
    auc = lab.auc(y, oof_blend)
    fa = lab.fold_aucs(y, oof_blend, folds)

    # rank-space variant (ROC-AUC only cares about the order)
    oof_rank = np.column_stack([lab.tform(P[i], "rank") for i in ids]) @ wts
    test_rank = np.column_stack([lab.tform(T[i], "rank") for i in ids]) @ wts
    print(f"  FINAL blend      auc={auc:.6f}  folds={[round(x,6) for x in fa]}")
    print(f"  rank-space twin  auc={lab.auc(y, oof_rank):.6f}")
    print(f"  test pred range  [{test_blend.min():.6f}, {test_blend.max():.6f}]")

    # A logit-space blend lives on the real line; sigmoid is strictly monotone so the induced
    # ranking -- and therefore ROC-AUC -- is unchanged, while the file becomes a valid
    # probability submission.
    from scipy.special import expit

    test_prob = expit(test_blend)

    store.save(f"blend_{args.name}", expit(oof_blend), test_prob, fold_scheme=args.folds,
               meta={"family": "blend", "featureset": kind, "auc": round(auc, 6),
                     "members": ids, "weights": wts.tolist()})
    if not args.dry_run:
        build(test_prob, name=args.name,
              notes=f"exp=blend_{args.name}; kind={kind}; oof={auc:.6f}; members={','.join(ids)}",
              oof_auc=round(auc, 6), members=list(ids))
    save_json({"members": ids, "kind": kind, "weights": wts.tolist(), "oof_auc": auc,
               "fold_aucs": fa, "rank_twin_auc": lab.auc(y, oof_rank),
               "submissions_used_today": used_today(), "remaining": remaining()},
              REPORTS / f"submission_{args.name}.json")
    print(f"kaggle submissions used today: {used_today()}/10")


if __name__ == "__main__":
    main()