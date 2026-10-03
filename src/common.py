"""Single source of truth for paths, data loading, hashing and config.

No target information is ever touched here.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repo root (src/ -> repo)
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
ARTIFACTS = ROOT / "artifacts"
PREDICTIONS = ARTIFACTS / "predictions"
FEATURES = ARTIFACTS / "features"
REPORTS = ROOT / "reports"
EXPERIMENTS = ROOT / "experiments"
SUBMISSIONS = ROOT / "submissions"
RESEARCH = ROOT / "research"

TRAIN_CSV = RAW / "train.csv"
TEST_CSV = RAW / "test.csv"
SAMPLE_CSV = RAW / "sample_submission.csv"

TARGET = "satisfaction"
ID_COL = "id"

for _d in (CACHE, ARTIFACTS, PREDICTIONS, FEATURES, REPORTS, EXPERIMENTS, SUBMISSIONS, RESEARCH):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------------------
def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def arr_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def git_commit() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    hashes: dict


def load_raw(dtype: dict | None = None) -> Dataset:
    """Load train/test CSVs preserving exact values.

    Deliberately keeps float64 so exact-value categorical lookups are not corrupted
    by float32 rounding.
    """
    h = {
        "train.csv": file_sha256(TRAIN_CSV)[:16] if TRAIN_CSV.exists() else None,
        "test.csv": file_sha256(TEST_CSV)[:16] if TEST_CSV.exists() else None,
        "sample_submission.csv": file_sha256(SAMPLE_CSV)[:16] if SAMPLE_CSV.exists() else None,
    }
    tr = pd.read_csv(TRAIN_CSV, dtype=dtype)
    te = pd.read_csv(TEST_CSV, dtype=dtype)
    sa = pd.read_csv(SAMPLE_CSV)
    return Dataset(tr, te, sa, h)


def load_cached_parquet() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Faster path: parquet conversion preserves exact float values identically."""
    ftr, fte = CACHE / "train.parquet", CACHE / "test.parquet"
    if not (ftr.exists() and fte.exists()):
        ds = load_raw()
        tr, te = ds.train, ds.test
        ftr.parent.mkdir(parents=True, exist_ok=True)
        tr.to_parquet(ftr, index=False)
        te.to_parquet(fte, index=False)
    return pd.read_parquet(ftr), pd.read_parquet(fte)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass


def save_json(obj, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))