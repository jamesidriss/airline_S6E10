"""Phase 24 -- fold-safe stack of the external TFM library, measured on our primary folds.

The 11 Deotte TFM members are individually weaker than v7 (see external_oof_probe.json) but
they are architecturally diverse.  The public ecosystem's strongest honest vector (najiama
blend 05, OOF 0.962234) is a blend of *stacks* of exactly this kind, so the question this
script answers is whether the stack gain is reproducible inside our own FIT boundary.

Every stack weight is fitted on FIT rows only (our immutable primary folds); the eval fold is
never touched.  Nothing here consults the public leaderboard.

    .\\.venv\\Scripts\\python.exe scripts/phase24/external_stack_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import load_cached_parquet  # noqa: E402
from src.validation.compare import fold_auc, logit, paired_fold_deltas  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

EXT = ROOT / "data/external/s6e10_public"
TFM_DIR = EXT / "s6e10-tfm-oof-and-test-predictions"
OUT = ROOT / "reports/phase24/external_stack_probe.json"
V7_OOF = ROOT / "artifacts/predictions/v7_sol_tabpfn_internal2_aux10_oof.npy"


def load_members(n: int) -> tuple[list[str], np.ndarray]:
    names, cols = [], []
    for p in sorted(TFM_DIR.glob("oof_*.npy")):
        a = np.load(p).astype(np.float64)
        if a.shape != (n,):
            raise ValueError(f"{p.name}: shape {a.shape}")
        names.append(p.stem.replace("oof_", ""))
        cols.append(logit(a))
    return names, np.column_stack(cols)


def nj_vec(path: Path, ids: np.ndarray) -> np.ndarray:
    import pandas as pd

    d = pd.read_csv(path)
    s = d.set_index("id")["satisfaction"].reindex(ids)
    if s.isna().any():
        raise ValueError(f"{path.name}: missing ids")
    return s.to_numpy(np.float64)


def stats(name: str, oof: np.ndarray, y: np.ndarray, folds: np.ndarray, v7: np.ndarray) -> dict:
    per_fold = fold_auc(y, oof, folds)
    mean_delta, deltas = paired_fold_deltas(y, oof, v7, folds)
    d = np.asarray(deltas)
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std(ddof=1) > 0 else float("nan")
    return {
        "name": name,
        "pooled_auc": float(roc_auc_score(y, oof)),
        "fold_auc": per_fold,
        "delta_vs_v7": mean_delta,
        "per_fold_delta": deltas,
        "t": t,
        "folds_positive": int((d > 0).sum()),
    }


def nested_logistic(X: np.ndarray, y: np.ndarray, folds: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Out-of-fold logistic stack: weights fitted on FIT rows of each fold only."""
    oof = np.zeros(len(y), dtype=np.float64)
    for k in sorted(set(folds.tolist())):
        fit = folds != k
        ev = ~fit
        mu, sd = X[fit].mean(0), X[fit].std(0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit((X[fit] - mu) / sd, y[fit])
        oof[ev] = clf.decision_function((X[ev] - mu) / sd)
    return oof


def nested_weight(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, list]:
    """Out-of-fold single-weight blend on the logit scale, weight on a 0..1 grid."""
    grid = np.linspace(0.0, 1.0, 41)
    oof = np.zeros(len(y), dtype=np.float64)
    weights = []
    for k in sorted(set(folds.tolist())):
        fit = folds != k
        ev = ~fit
        best_w, best = 0.0, -1.0
        for w in grid:
            s = (1 - w) * X[fit, 0] + w * X[fit, 1]
            a = roc_auc_score(y[fit], s)
            if a > best:
                best, best_w = a, float(w)
        weights.append(best_w)
        oof[ev] = (1 - best_w) * X[ev, 0] + best_w * X[ev, 1]
    return oof, weights


def main() -> None:
    train, _ = load_cached_parquet()
    y = train["satisfaction"].to_numpy(np.int8)
    ids = train["id"].to_numpy()
    folds = get_scheme("primary", y, train["id"]).folds
    v7 = np.load(V7_OOF).astype(np.float64)
    names, X = load_members(len(y))
    v7L = logit(v7)

    rows = []
    equal = X.mean(1)
    rows.append(stats("tfm_equal11", equal, y, folds, v7))
    rows.append(stats("tfm_logistic11_nested", nested_logistic(X, y, folds), y, folds, v7))

    # v7 + the 11-member equal stack
    rows.append(stats("v7+tfm_equal11", logit(1 / (1 + np.exp(-(0.5 * v7L + 0.5 * equal)))), y, folds, v7))
    rows.append(stats("v7+tfm_logistic11_nested", nested_logistic(np.column_stack([v7L, X]), y, folds), y, folds, v7))

    # v7 + the strongest public honest vector
    nj05 = nj_vec(EXT / "s6e10-oof/05_blend_oof.csv", ids)
    nj05L = logit(nj05)
    rows.append(stats("nj05_alone", nj05, y, folds, v7))
    oof_w, w = nested_weight(np.column_stack([v7L, nj05L]), y, folds)
    rows.append({**stats("v7+nj05_nested_weight", oof_w, y, folds, v7), "weights": w})
    rows.append(stats("v7+nj05_equal", 0.5 * v7L + 0.5 * nj05L, y, folds, v7))

    # the two public blends together with v7
    nj06 = nj_vec(EXT / "s6e10-oof/06_blend_oof.csv", ids)
    nj06L = logit(nj06)
    X3 = np.column_stack([v7L, nj05L, nj06L])
    rows.append(stats("v7+nj05+nj06_logistic_nested", nested_logistic(X3, y, folds), y, folds, v7))

    OUT.write_text(json.dumps({
        "champion": {"name": "v7", "pooled_auc": float(roc_auc_score(y, v7))},
        "members": names,
        "results": rows,
        "note": "every fitted weight is estimated on FIT rows only; eval folds are untouched",
    }, indent=1), encoding="utf-8")

    for r in rows:
        extra = f" w={r['weights']}" if "weights" in r else ""
        print(f"{r['name']:30} AUC {r['pooled_auc']:.6f}  vs v7 {r['delta_vs_v7']:+.2e}  "
              f"t {r['t']:6.2f}  {r['folds_positive']}/5{extra}")


if __name__ == "__main__":
    main()
