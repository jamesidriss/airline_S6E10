"""Submission builder with a hard pre-flight validation gate.

Refuses to write a file unless every check passes:
  * columns and order exactly match ``sample_submission.csv``
  * row count matches ``test.csv``
  * ids match ``test.csv`` exactly, in order
  * no NaN / +-Inf
  * values inside [0, 1] and finite
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import ID_COL, SAMPLE_CSV, SUBMISSIONS, TARGET, TEST_CSV, ROOT, git_commit, arr_sha256

MANIFEST = SUBMISSIONS / "manifest.csv"


def build(pred: np.ndarray, name: str, notes: str = "", oof_auc: float | None = None,
          members: list[str] | None = None, allow_out_of_range: float = 1e-9) -> Path:
    sample = pd.read_csv(SAMPLE_CSV)
    test = pd.read_csv(TEST_CSV, usecols=[ID_COL])
    p = np.asarray(pred, dtype="float64").ravel()

    # ------------------------------------------------------------------ gate
    problems = []
    if len(p) != len(sample):
        problems.append(f"length {len(p)} != sample {len(sample)}")
    if len(p) != len(test):
        problems.append(f"length {len(p)} != test {len(test)}")
    if not np.all(np.isfinite(p)):
        problems.append("non-finite values present")
    if p.min() < -allow_out_of_range or p.max() > 1 + allow_out_of_range:
        problems.append(f"values outside [0,1]: min={p.min()} max={p.max()}")
    if list(sample.columns) != [ID_COL, TARGET]:
        problems.append(f"sample columns are {list(sample.columns)}")
    if not np.array_equal(sample[ID_COL].to_numpy(), test[ID_COL].to_numpy()):
        problems.append("sample ids != test ids")
    if problems:
        raise ValueError("SUBMISSION REJECTED: " + "; ".join(problems))

    out = sample.copy()
    out[TARGET] = np.clip(p, 0.0, 1.0)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    path = SUBMISSIONS / f"{name}.csv"
    out.to_csv(path, index=False)

    row = {
        "name": name,
        "created": pd.Timestamp.utcnow().isoformat(),
        "oof_auc": oof_auc,
        "n_rows": len(out),
        "pred_sha256_16": arr_sha256(out[TARGET].to_numpy())[:16],
        "members": ";".join(members or []),
        "git": git_commit(),
        "notes": notes,
    }
    df = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(MANIFEST, index=False)
    print(f"[submission] wrote {path}  rows={len(out)}  sha16={row['pred_sha256_16']}")
    return path