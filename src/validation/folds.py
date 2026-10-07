"""Immutable validation registry.

Fold assignments are computed once, saved to disk, and reused by every experiment.
Three schemes:

  * ``primary``  : StratifiedKFold(n_splits=5, shuffle=True, seed=20261010)
                   -> the shared discovery CV. Never changed.
  * ``shadow``   : StratifiedKFold(n_splits=5, shuffle=True, seed=777001)
                   -> independent confirmation. Use sparingly.
  * ``block10``  : StratifiedKFold(n_splits=10, shuffle=True, seed=20261010)
                   -> finalists / robustness.

A fourth, ``blocked_id``, is *not* registered as a model-selection scheme: it exists only
to test whether ``id``-derived features survive a generator-batch-aware split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.common import CACHE, arr_sha256, set_seed

SEED_PRIMARY = 20261010
SEED_SHADOW = 777001
N_SPLITS = 5

REGISTRY_PATH = CACHE / "folds"


@dataclass(frozen=True)
class FoldScheme:
    name: str
    n_splits: int
    seed: int
    folds: np.ndarray  # int array, len == n_train_rows

    def validate_ids(self, ids: pd.Series) -> None:
        expected = pd.Index(ids.values)
        got = pd.Index(self.folds)
        if len(expected) != len(got):
            raise AssertionError("fold array length != number of rows")
        h = pd.util.hash_pandas_object(pd.Series(np.asarray(self.folds)), index=False).sum()
        if self._id_hash is not None and h != self._id_hash:
            raise AssertionError("fold array does not match registered ids")

    _id_hash: int | None = None


def _build(name: str, y: np.ndarray, n_splits: int, seed: int) -> FoldScheme:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    f = np.full(len(y), -1, dtype=np.int8)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        f[va] = k
    if (f < 0).any():
        raise AssertionError("unassigned rows in fold assignment")
    counts = np.bincount(f, minlength=n_splits)
    if not np.all(counts > 0):
        raise AssertionError("empty fold")
    # overlap check
    assert len(np.unique(f)) == n_splits
    fs = FoldScheme(name=name, n_splits=n_splits, seed=seed, folds=f)
    return fs


def get_scheme(name: str, y: np.ndarray, ids: pd.Series, n_splits: int | None = None, seed: int | None = None, force: bool = False) -> FoldScheme:
    """Load (or create once) a registered fold scheme. Fails loudly on tampering."""
    REGISTRY_PATH.mkdir(parents=True, exist_ok=True)
    meta_path = REGISTRY_PATH / "registry.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    y = np.asarray(y)
    yhash = int(pd.util.hash_pandas_object(pd.Series(y), index=False).sum())

    if name in meta and not force:
        assert meta[name]["y_hash"] == yhash, f"registry y_hash mismatch for scheme {name}"
        arr = np.load(REGISTRY_PATH / meta[name]["file"])
        assert len(arr) == len(y)
        assert set(np.unique(arr)) == set(range(meta[name]["n_splits"])), "invalid fold labels"
        ah, ih = arr_sha256(arr), arr_sha256(np.asarray(ids))
        if "fold_sha256" in meta[name]:
            assert meta[name]["fold_sha256"] == ah, "immutable fold array was altered"
            assert meta[name]["ids_sha256"] == ih, "fold row ids/order changed"
        else:
            # Bind the existing assignment; never regenerate it during migration.
            meta[name].update(fold_sha256=ah, ids_sha256=ih)
            meta_path.write_text(json.dumps(meta, indent=2))
        return FoldScheme(name, meta[name]["n_splits"], meta[name]["seed"], arr)

    if name in meta:
        raise ValueError("registered folds are immutable; force regeneration is forbidden")

    defaults = {
        "primary": (N_SPLITS, SEED_PRIMARY),
        "shadow": (N_SPLITS, SEED_SHADOW),
        "block10": (10, SEED_PRIMARY),
    }
    if name in defaults and n_splits is None and seed is None:
        n_splits, seed = defaults[name]
    if n_splits is None or seed is None:
        raise ValueError(f"unknown scheme {name}; pass n_splits and seed")

    set_seed(seed)
    fs = _build(name, y, n_splits, seed)
    fname = f"{name}_{n_splits}f_seed{seed}.npy"
    np.save(REGISTRY_PATH / fname, fs.folds)
    meta[name] = {"file": fname, "n_splits": n_splits, "seed": seed, "y_hash": yhash,
                  "fold_sha256": arr_sha256(fs.folds), "ids_sha256": arr_sha256(np.asarray(ids)),
                  "fold_sizes": np.bincount(fs.folds, minlength=n_splits).tolist()}
    meta_path.write_text(json.dumps(meta, indent=2))
    return fs


def blocked_id_scheme(y: np.ndarray, n_blocks: int = 10, seed: int = 4242) -> np.ndarray:
    """Blocked (contiguous) split on row order: tests id/order-derived features honestly.

    Row order in train.csv is the generator's output order. Contiguous blocks keep
    adjacent id ranges inside the same fold, which is the hardest case for any
    id-derived shortcut.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    # stratify blocks by the block-level positive rate to keep folds balanced
    edges = np.linspace(0, len(y), n_blocks + 1).astype(int)
    block_rate = np.array([y[edges[i]:edges[i + 1]].mean() for i in range(n_blocks)])
    order = np.argsort(block_rate)
    fold = np.full(len(y), -1, dtype=np.int8)
    for k in range(n_blocks):
        bi = order[k % n_blocks]
        fold[edges[bi]:edges[bi + 1]] = int(k % n_blocks)
    # k % n_blocks == k since n_blocks == n_folds here
    assert (fold >= 0).all()
    return fold


def fold_report(folds: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for k in range(int(folds.max()) + 1):
        m = folds == k
        out[k] = {"n": int(m.sum()), "pos_rate": float(y[m].mean())}
    return out
