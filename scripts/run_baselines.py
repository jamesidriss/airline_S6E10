"""Wave A: clean baselines on the immutable `primary` fold scheme.

Usage:  python scripts/run_baselines.py [--models lgbm,xgb,cat] [--fs numeric,numeric+cat]
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
from src.features.sets import META4, NUMCOLS, SURVEY13, exact_cat_copies, numeric_frame  # noqa: E402
from src.models import gbdt  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import report  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
import experiments_ledger as led  # noqa: E402


def _code_cat(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], test_only: bool = False) -> pd.DataFrame:
    """Exact-value categorical duplicates, coded with a joint sorted train+test vocabulary.

    Target-free, hence transductively safe (no label ever enters the vocabulary).
    """
    src = te if test_only else tr
    out = pd.DataFrame(index=src.index)
    for c in cols:
        raw = exact_cat_copies(tr, [c])[c + "__cat"]
        codes = pd.factorize(
            pd.concat([raw, exact_cat_copies(te, [c])[c + "__cat"]], ignore_index=True).astype("object"),
            sort=True,
        )[0]
        out[c + "__cat"] = (codes[len(tr):] if test_only else codes[: len(tr)]).astype("float32")
    return out


def build_features(fs: str, tr: pd.DataFrame, te: pd.DataFrame):
    feats = [c for c in te.columns if c != ID_COL]

    def _num(df):
        out = df[feats].copy()
        for c in out.columns:
            out[c] = pd.factorize(out[c].astype("object"), sort=True)[0].astype("float32") \
                if out[c].dtype.kind in "OUS" else out[c].astype("float32")
        return out

    if fs == "numeric":
        Xtr, Xte = _num(tr), _num(te)
    elif fs == "numeric+cat":
        Xtr = pd.concat([_num(tr), _code_cat(tr, te, feats)], axis=1)
        Xte = pd.concat([_num(te), _code_cat(tr, te, feats, test_only=True)], axis=1)
    elif fs == "numeric+cat_survey":
        catcols = SURVEY13 + META4 + NUMCOLS
        Xtr = pd.concat([_num(tr), _code_cat(tr, te, catcols)], axis=1)
        Xte = pd.concat([_num(te), _code_cat(tr, te, catcols, test_only=True)], axis=1)
    else:
        raise ValueError(fs)
    return Xtr.astype("float32").values, Xte.astype("float32").values, list(Xtr.columns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lgbm,xgb,cat")
    ap.add_argument("--fs", default="numeric,numeric+cat")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--tag", default="waveA")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save-test", action="store_true")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    dh = f"tr{tr.shape}-te{te.shape}"
    summary = []

    for scheme in args.folds.split(","):
        folds = get_scheme(scheme, y, tr[ID_COL]).folds
        for fs in args.fs.split(","):
            set_seed(args.seed)
            Xtr, Xte, names = build_features(fs, tr, te)
            print(f"\n### scheme={scheme} fs={fs} shape={Xtr.shape}", flush=True)
            for mdl in args.models.split(","):
                if mdl == "cat":
                    # catboost on integer-coded categoricals
                    fn = gbdt.catboost
                    kw = {"cat_idx": None}
                elif mdl == "lgbm":
                    fn, kw = gbdt.lgbm, {}
                elif mdl == "xgb":
                    fn, kw = gbdt.xgboost, {"device": "cuda"}
                else:
                    continue
                t0 = time.time()
                oof, test, fauc, dur = fn(Xtr, y, Xte, folds, seed=args.seed, **kw)
                auc = float(gbdt.roc_auc_score(y, oof))
                print(f"  ==> {mdl}/{fs}/{scheme} OOF AUC = {auc:.6f}  ({dur:.0f}s)", flush=True)
                eid = f"{args.tag}_{mdl}_{fs}_{scheme}"
                store.save(eid, oof, test if args.save_test else None, fold_scheme=scheme,
                           meta={"family": mdl, "featureset": fs, "auc": round(auc, 6)})
                led.log_experiment(
                    family=mdl, featureset=fs, params={}, seed=args.seed, fold_scheme=scheme,
                    oof_auc=auc, fold_aucs=fauc, oof=oof, duration_s=dur, data_hash=dh,
                    test_pred=test if args.save_test else None, exp_id=eid,
                    notes=f"{args.tag}", verdict="baseline",
                    extra={"fold_ids": folds.tolist()},
                )
                summary.append({"exp_id": eid, "family": mdl, "fs": fs, "scheme": scheme,
                                "oof_auc": round(auc, 6), "fold_aucs": [round(x, 6) for x in fauc]})
    save_json(summary, REPORTS / f"{args.tag}_summary.json")
    print("\n=== SUMMARY ===")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()