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
    idx = _load_index()
    previous = idx.get(exp_id)
    oof_hash = arr_sha256(oof)
    if previous is not None:
        if previous['oof_sha'] != oof_hash:
            raise FileExistsError(f'Preserve existing OOF: {exp_id}; use a new experiment ID')
        load_oof(exp_id)  # check the banked file before any write
        if 'test' in previous:
            load_test(exp_id)
            if test is not None and arr_sha256(np.asarray(test, dtype='float32')) != previous['test_sha']:
                raise FileExistsError(f'Preserve existing test prediction: {exp_id}; use a new experiment ID')
            return previous
        if test is None:
            return previous
    elif op.exists():
        raise FileExistsError(f'Unindexed prediction already exists: {op}; preserve it and use a new ID')
    tp = PREDICTIONS / f"{exp_id}_test.npy"
    if test is not None and tp.exists():
        raise FileExistsError(f'Unindexed test prediction already exists: {tp}; preserve it and use a new ID')
    np.save(op, oof)
    entry = {"exp_id": exp_id, "oof": str(op.relative_to(PREDICTIONS.parent.parent)),
             "oof_sha": oof_hash, "fold_scheme": fold_scheme, "meta": meta or {}}
    if test is not None:
        test = np.asarray(test, dtype="float32")
        np.save(tp, test)
        entry["test"] = str(tp.relative_to(PREDICTIONS.parent.parent))
        entry["test_sha"] = arr_sha256(test)
    if entry["oof_sha"] in {v.get("oof_sha") for v in idx.values()}:
        entry["duplicate_of"] = next(k for k, v in idx.items() if v.get("oof_sha") == entry["oof_sha"])
    idx[exp_id] = entry
    temporary = INDEX.with_name('index.partial.json')
    temporary.write_text(json.dumps(idx, indent=2, default=str), encoding="utf-8")
    temporary.replace(INDEX)
    return entry


def load_oof(exp_id: str) -> np.ndarray:
    idx = _load_index()
    out = np.load(PREDICTIONS.parent.parent / idx[exp_id]["oof"])
    if arr_sha256(out) != idx[exp_id]["oof_sha"]:
        raise ValueError(f"prediction hash mismatch: {exp_id}/oof")
    return out


def load_test(exp_id: str) -> np.ndarray:
    idx = _load_index()
    out = np.load(PREDICTIONS.parent.parent / idx[exp_id]["test"])
    if arr_sha256(out) != idx[exp_id]["test_sha"]:
        raise ValueError(f"prediction hash mismatch: {exp_id}/test")
    return out


def list_all() -> list[dict]:
    idx = _load_index()
    out = []
    for k, v in idx.items():
        out.append({"exp_id": k, "has_test": "test" in v, "oof_sha": v.get("oof_sha", "")[:12],
                    "dup": v.get("duplicate_of", ""), **v.get("meta", {})})
    return sorted(out, key=lambda d: d["exp_id"])
