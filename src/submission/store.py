"""Prediction store: every useful OOF / test prediction pair, float32, hash-indexed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.common import PREDICTIONS, arr_sha256

INDEX = PREDICTIONS / "index.json"


def _load_index() -> dict:
    if INDEX.exists():
        return json.loads(INDEX.read_text(encoding="utf-8"))
    return {}


def save(exp_id: str, oof: np.ndarray, test: np.ndarray | None, fold_scheme: str = "primary",
         meta: dict | None = None) -> dict:
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    oof = np.asarray(oof, dtype="float32")
    op = PREDICTIONS / f"{exp_id}_oof.npy"
    np.save(op, oof)
    entry = {"exp_id": exp_id, "oof": str(op.relative_to(PREDICTIONS.parent.parent)),
             "oof_sha": arr_sha256(oof), "fold_scheme": fold_scheme, "meta": meta or {}}
    if test is not None:
        test = np.asarray(test, dtype="float32")
        tp = PREDICTIONS / f"{exp_id}_test.npy"
        np.save(tp, test)
        entry["test"] = str(tp.relative_to(PREDICTIONS.parent.parent))
        entry["test_sha"] = arr_sha256(test)
    idx = _load_index()
    if entry["oof_sha"] in {v.get("oof_sha") for v in idx.values()}:
        entry["duplicate_of"] = next(k for k, v in idx.items() if v.get("oof_sha") == entry["oof_sha"])
    idx[exp_id] = entry
    INDEX.write_text(json.dumps(idx, indent=2, default=str), encoding="utf-8")
    return entry


def load_oof(exp_id: str) -> np.ndarray:
    idx = _load_index()
    return np.load(PREDICTIONS.parent.parent / idx[exp_id]["oof"])


def load_test(exp_id: str) -> np.ndarray:
    idx = _load_index()
    return np.load(PREDICTIONS.parent.parent / idx[exp_id]["test"])


def list_all() -> list[dict]:
    idx = _load_index()
    out = []
    for k, v in idx.items():
        out.append({"exp_id": k, "has_test": "test" in v, "oof_sha": v.get("oof_sha", "")[:12],
                    "dup": v.get("duplicate_of", ""), **v.get("meta", {})})
    return sorted(out, key=lambda d: d["exp_id"])