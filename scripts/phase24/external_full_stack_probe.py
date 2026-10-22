"""Phase 24 -- full fold-safe stack of every downloadable external OOF vector vs v7.

Members: 11 Deotte TFM OOFs, 8 goodpjw2008 TabPFN/TabICL OOFs, 7 najiama blend OOFs.
All weights are fitted on FIT rows of our immutable primary folds; eval folds are untouched.
This answers whether the public blend level (0.9622) is reproducible as a *deployable* stack
whose test side we also hold.

    .\\.venv\\Scripts\\python.exe scripts/phase24/external_full_stack_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import load_cached_parquet  # noqa: E402
from src.validation.compare import fold_auc, logit, paired_fold_deltas  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

EXT = ROOT / "data/external/s6e10_public"
OUT = ROOT / "reports/phase24/external_full_stack_probe.json"
V7_OOF = ROOT / "artifacts/predictions/v7_sol_tabpfn_internal2_aux10_oof.npy"


def csv_vec(path: Path, ids: np.ndarray) -> np.ndarray:
    d = pd.read_csv(path)
    idc = "id" if "id" in d.columns else d.columns[0]
    valc = "satisfaction" if "satisfaction" in d.columns else [c for c in d.columns if c != idc][-1]
    s = d.set_index(idc)[valc].reindex(ids)
    if s.isna().any():
        raise ValueError(f"{path.name}: {int(s.isna().sum())} missing ids")
    return s.to_numpy(np.float64)


def main() -> None:
    train, _ = load_cached_parquet()
    y = train["satisfaction"].to_numpy(np.int8)
    ids = train["id"].to_numpy()
    folds = get_scheme("primary", y, train["id"]).folds
    v7 = np.load(V7_OOF).astype(np.float64)

    names: list[str] = ["v7"]
    cols: list[np.ndarray] = [logit(v7)]

    for p in sorted((EXT / "s6e10-tfm-oof-and-test-predictions").glob("oof_*.npy")):
        a = np.load(p).astype(np.float64)
        names.append("tfm_" + p.stem.replace("oof_", ""))
        cols.append(logit(a))
    for p in sorted((EXT / "s6e10-tabpfn-member").glob("oof_*.csv")):
        names.append("gp_" + p.stem.replace("oof_", ""))
        cols.append(logit(csv_vec(p, ids)))
    for p in sorted((EXT / "s6e10-oof").glob("*_oof.csv")):
        names.append("nj_" + p.stem.replace("_oof", ""))
        cols.append(logit(csv_vec(p, ids)))

    X = np.column_stack(cols)
    assert X.shape[1] == len(names)

    # Nested logistic stack: weights fitted on FIT rows of each fold only.
    oof = np.zeros(len(y))
    coefs = []
    for k in sorted(set(folds.tolist())):
        fit = folds != k
        ev = ~fit
        mu, sd = X[fit].mean(0), X[fit].std(0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(C=0.3, max_iter=3000, solver="lbfgs")
        clf.fit((X[fit] - mu) / sd, y[fit])
        oof[ev] = clf.decision_function((X[ev] - mu) / sd)
        coefs.append(clf.coef_.ravel().tolist())

    per_fold = fold_auc(y, oof, folds)
    mean_delta, deltas = paired_fold_deltas(y, oof, v7, folds)
    d = np.asarray(deltas)
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))

    # Simple robust reference: v7 + nj05 only, 0.85/0.15 on the logit scale.
    v7L = logit(v7)
    nj05L = logit(csv_vec(EXT / "s6e10-oof/05_blend_oof.csv", ids))
    ref = 0.15 * v7L + 0.85 * nj05L
    ref_delta, ref_per = paired_fold_deltas(y, ref, v7, folds)
    rd = np.asarray(ref_per)
    ref_t = float(rd.mean() / (rd.std(ddof=1) / np.sqrt(len(rd))))

    out = {
        "members": names,
        "n_members": len(names),
        "champion_auc": float(roc_auc_score(y, v7)),
        "full_stack": {
            "pooled_auc": float(roc_auc_score(y, oof)),
            "fold_auc": per_fold,
            "delta_vs_v7": mean_delta,
            "per_fold_delta": deltas,
            "t": t,
            "folds_positive": int((d > 0).sum()),
            "n_folds_fitting": len(coefs),
        },
        "v7_nj05_085": {
            "pooled_auc": float(roc_auc_score(y, ref)),
            "delta_vs_v7": ref_delta,
            "per_fold_delta": ref_per,
            "t": ref_t,
            "folds_positive": int((rd > 0).sum()),
        },
        "note": "all fitted weights estimated on FIT rows only; eval folds untouched",
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"members: {len(names)}")
    print(f"full stack   AUC {out['full_stack']['pooled_auc']:.6f}  vs v7 "
          f"{out['full_stack']['delta_vs_v7']:+.2e}  t {out['full_stack']['t']:.2f}  "
          f"{out['full_stack']['folds_positive']}/5")
    print(f"v7+nj05 .15  AUC {out['v7_nj05_085']['pooled_auc']:.6f}  vs v7 "
          f"{out['v7_nj05_085']['delta_vs_v7']:+.2e}  t {out['v7_nj05_085']['t']:.2f}  "
          f"{out['v7_nj05_085']['folds_positive']}/5")


if __name__ == "__main__":
    main()
