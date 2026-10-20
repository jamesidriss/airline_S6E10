"""Phase 24 -- quick parent decomposition of the champion portfolio (read-only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import load_cached_parquet  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    s = s[order]
    y = y[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    pos = y.sum()
    neg = len(y) - pos
    return float((ranks[y > 0].sum() - pos * (pos + 1) / 2) / (pos * neg))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def main() -> None:
    tr, _ = load_cached_parquet()
    y = tr["satisfaction"].values.astype("int8")
    folds = get_scheme("primary", y, tr["id"]).folds
    base = Path("artifacts/sol_route_aux10_primary_final")
    v7 = np.load("artifacts/predictions/v7_sol_tabpfn_internal2_aux10_oof.npy").astype("float64")
    out = {"v7_pooled": auc(y, v7), "v7_per_fold": [auc(y[folds == k], v7[folds == k]) for k in range(5)]}
    for name in ("route_oof", "aux10_oof", "candidate_oof"):
        a = np.load(base / f"{name}.npy").astype("float64")
        out[name] = {
            "pooled": auc(y, a),
            "per_fold": [auc(y[folds == k], a[folds == k]) for k in range(5)],
            "corr_logit_v7": float(np.corrcoef(logit(a), logit(v7))[0, 1]),
            "equal_blend_with_v7": auc(y, 0.5 * logit(a) + 0.5 * logit(v7)),
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
