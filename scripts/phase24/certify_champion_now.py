"""Phase 24 -- re-verify the certified champion artifacts from disk (read-only).

Checks the OOF vector, the test vector, the uploaded CSV, the id order, and the
recomposed composition identity. Writes nothing; prints a JSON block that the
phase-24 certification document quotes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import load_cached_parquet  # noqa: E402


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def main() -> None:
    v7_oof = ROOT / "artifacts/predictions/v7_sol_tabpfn_internal2_aux10_oof.npy"
    port_oof = ROOT / "artifacts/phase17_B1/primary/portfolio_oof.npy"
    t_route = ROOT / "artifacts/v7_sol_tabpfn_internal2_aux10_test/route.npy"
    t_aux = ROOT / "artifacts/v7_sol_tabpfn_internal2_aux10_test/aux10.npy"
    t_cand = ROOT / "artifacts/v7_sol_tabpfn_internal2_aux10_test/candidate.npy"
    t_ids = ROOT / "artifacts/v7_sol_tabpfn_internal2_aux10_test/test_ids.npy"
    csv = ROOT / "submissions/v7_sol_tabpfn_internal2_aux10.csv"
    sample = ROOT / "data/raw/sample_submission.csv"

    tr, te = load_cached_parquet()
    route = np.load(t_route).astype(np.float64)
    aux = np.load(t_aux).astype(np.float64)
    comp = 1.0 / (1.0 + np.exp(-(logit(route) + logit(aux)) / 2.0))
    cand = np.load(t_cand).astype(np.float64)

    df = pd.read_csv(csv)
    samp = pd.read_csv(sample)
    ids = np.load(t_ids)
    y = tr.satisfaction.to_numpy(dtype="int8")
    oof = np.load(v7_oof).astype(np.float64)
    from sklearn.metrics import roc_auc_score

    out = {
        "utc": pd.Timestamp.utcnow().isoformat(),
        "champion_oof": {
            "path": str(v7_oof.relative_to(ROOT)),
            "sha256": sha(v7_oof),
            "auc": float(roc_auc_score(y, oof)),
            "rows": int(len(oof)),
        },
        "champion_portfolio_oof": {
            "path": str(port_oof.relative_to(ROOT)),
            "sha256": sha(port_oof),
            "bit_equal_to_store_vector": bool(np.array_equal(np.load(port_oof), oof.astype(np.float32))),
        },
        "test_candidate": {
            "path": str(t_cand.relative_to(ROOT)),
            "sha256": sha(t_cand),
            "rows": int(len(cand)),
        },
        "recomposition_max_abs_vs_candidate": float(np.max(np.abs(comp - cand))),
        "csv": {
            "path": str(csv.relative_to(ROOT)),
            "sha256": sha(csv),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "id_matches_test_ids": bool(np.array_equal(df["id"].to_numpy(), ids)),
            "id_matches_sample_submission": bool(
                len(df) == len(samp) and np.array_equal(df["id"].to_numpy(), samp["id"].to_numpy())
            ),
            "prob_min": float(df.iloc[:, 1].min()),
            "prob_max": float(df.iloc[:, 1].max()),
            "prob_mean": float(df.iloc[:, 1].mean()),
            "corr_with_candidate": float(np.corrcoef(df.iloc[:, 1].to_numpy(), cand)[0, 1]),
            "max_abs_vs_candidate": float(np.max(np.abs(df.iloc[:, 1].to_numpy() - cand))),
        },
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
