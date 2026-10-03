"""Write the blend's test predictions as a Kaggle submission file (with the pre-flight gate)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet  # noqa: E402
from src.ensemble import lab  # noqa: E402
from src.submission import store  # noqa: E402
from src.submission.kaggle_io import remaining, used_today  # noqa: E402
from src.submission.make import build  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True, help="a stored blend/stack prediction pair")
    ap.add_argument("--name", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y, tr[ID_COL]).folds
    oof = store.load_oof(args.exp_id).astype("float64")
    test = store.load_test(args.exp_id).astype("float64")
    auc = lab.auc(y, oof)
    fa = lab.fold_aucs(y, oof, folds)
    print(f"{args.exp_id}: OOF AUC = {auc:.6f}  folds={[round(x,6) for x in fa]}")
    print(f"test rows={len(test)} range=[{test.min():.6f},{test.max():.6f}]")
    if args.dry_run:
        print("dry run: no file written")
        return
    build(test, name=args.name,
          notes=f"exp={args.exp_id}; OOF(primary5)={auc:.6f}; equal-logit over all admitted members",
          oof_auc=round(auc, 6))
    print(f"kaggle submissions used today: {used_today()}/10 (remaining {remaining()})")


if __name__ == "__main__":
    main()