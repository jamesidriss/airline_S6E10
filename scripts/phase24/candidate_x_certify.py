"""Phase 24 -- build and certify candidate X.

Candidate X = sigmoid(0.15 * logit(v7) + 0.85 * logit(nj05))

- v7  : our certified champion, OOF 0.9618529228157235, test sha 2b744d0d...
- nj05: najiama/s6e10-oof 05_blend, a public blend of honest public OOF members,
        measured on OUR immutable primary folds at +3.747e-4 vs v7 (t 11.36, 5/5).

The fixed 0.85 weight sits on the flat FIT-optimal plateau (0.825-0.850 in all five folds).
Nothing is fitted on eval rows; nothing consults the public leaderboard.  This script writes
the candidate arrays, the submission CSV and a certification JSON with hashes.  It does NOT
upload.

    .\\.venv\\Scripts\\python.exe scripts/phase24/candidate_x_certify.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import arr_sha256, file_sha256, load_cached_parquet  # noqa: E402
from src.validation.compare import fold_auc, logit, paired_fold_deltas  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

V7_OOF = ROOT / "artifacts/predictions/v7_sol_tabpfn_internal2_aux10_oof.npy"
V7_TEST = ROOT / "artifacts/v7_sol_tabpfn_internal2_aux10_test/candidate.npy"
NJ05_OOF = ROOT / "data/external/s6e10_public/s6e10-oof/05_blend_oof.csv"
NJ05_SUB = ROOT / "data/external/s6e10_public/s6e10-oof/05_submission.csv"
OUTDIR = ROOT / "artifacts/phase24_candidateX"
OUT_JSON = ROOT / "reports/phase24/candidate_x_certification.json"
W_NJ05 = 0.85


def blend(a: np.ndarray, b: np.ndarray, w_b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-((1.0 - w_b) * logit(a) + w_b * logit(b))))


def fold_stats(y: np.ndarray, oof: np.ndarray, v7: np.ndarray, folds: np.ndarray) -> dict:
    per = fold_auc(y, oof, folds)
    mean_delta, deltas = paired_fold_deltas(y, oof, v7, folds)
    d = np.asarray(deltas)
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std(ddof=1) > 0 else float("nan")
    return {
        "fold_auc": per,
        "fold_mean_auc": float(np.mean(per)),
        "delta_vs_v7": mean_delta,
        "per_fold_delta": deltas,
        "t": t,
        "folds_positive": int((d > 0).sum()),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    train, test = load_cached_parquet()
    y = train["satisfaction"].to_numpy(np.int8)
    tr_ids = train["id"].to_numpy()
    te_ids = test["id"].to_numpy()

    v7 = np.load(V7_OOF).astype(np.float64)
    nj = pd.read_csv(NJ05_OOF).set_index("id")["satisfaction"].reindex(tr_ids)
    if nj.isna().any():
        raise AssertionError("nj05 OOF does not cover train ids")
    nj = nj.to_numpy(np.float64)

    x_oof = blend(v7, nj, W_NJ05)
    primary = get_scheme("primary", y, train["id"]).folds
    shadow = get_scheme("shadow", y, train["id"]).folds

    v7_test = np.load(V7_TEST).astype(np.float64)
    sub = pd.read_csv(NJ05_SUB)
    if not np.array_equal(sub["id"].to_numpy(), te_ids):
        raise AssertionError("nj05 submission ids are not exactly the test ids in order")
    nj_test = sub["satisfaction"].to_numpy(np.float64)
    x_test = blend(v7_test, nj_test, W_NJ05)

    np.save(OUTDIR / "candidate_x_oof.npy", x_oof.astype(np.float32))
    np.save(OUTDIR / "candidate_x_test.npy", x_test.astype(np.float32))
    pd.DataFrame({"id": te_ids, "satisfaction": x_test}).to_csv(OUTDIR / "candidate_x.csv", index=False)

    out = {
        "candidate": "X = sigmoid(0.15*logit(v7) + 0.85*logit(nj05))",
        "w_nj05": W_NJ05,
        "champion_auc": float(roc_auc_score(y, v7)),
        "candidate_oof_auc": float(roc_auc_score(y, x_oof)),
        "nj05_oof_auc": float(roc_auc_score(y, nj)),
        "nj05_oof_vs_v7": fold_stats(y, nj, v7, primary),
        "primary": fold_stats(y, x_oof, v7, primary),
        "shadow": fold_stats(y, x_oof, v7, shadow),
        "test": {
            "rows": int(len(te_ids)),
            "ids_equal_test": bool(np.array_equal(te_ids, test["id"].to_numpy())),
            "prob_min": float(x_test.min()),
            "prob_max": float(x_test.max()),
            "prob_mean": float(x_test.mean()),
            "nan": int(np.isnan(x_test).sum()),
            "corr_logit_v7_test": float(np.corrcoef(logit(x_test), logit(v7_test))[0, 1]),
            "arr_sha256": arr_sha256(x_test.astype(np.float32)),
            "npy_sha256": file_sha256(OUTDIR / "candidate_x_test.npy"),
            "csv_sha256": file_sha256(OUTDIR / "candidate_x.csv"),
        },
        "inputs": {
            "v7_oof_sha256": file_sha256(V7_OOF),
            "v7_test_sha256": file_sha256(V7_TEST),
            "nj05_oof_sha256": file_sha256(NJ05_OOF),
            "nj05_sub_sha256": file_sha256(NJ05_SUB),
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")

    for arm in ("primary", "shadow"):
        s = out[arm]
        print(f"{arm:8} AUC {s['fold_mean_auc']:.9f}  vs v7 {s['delta_vs_v7']:+.3e}  "
              f"t {s['t']:6.2f}  {s['folds_positive']}/5")
    print(f"pooled OOF AUC {out['candidate_oof_auc']:.9f}")
    print(f"test rows {out['test']['rows']}  corr {out['test']['corr_logit_v7_test']:.6f}  "
          f"sha {out['test']['arr_sha256'][:16]}")


if __name__ == "__main__":
    main()
