"""Experiment ledger: append-only JSONL with hashes, deltas and verdicts."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.common import EXPERIMENTS, git_commit, arr_sha256

LEDGER = EXPERIMENTS / "ledger.jsonl"
CHAMPION_FILE = EXPERIMENTS / "champion.json"


@dataclass
class Record:
    exp_id: str
    ts: str
    git: str
    family: str
    featureset: str
    fold_scheme: str
    params: dict
    seed: int
    oof_auc: float
    fold_aucs: list
    oof_std: float
    delta_vs_champion: float | None
    paired_fold_deltas: list
    corr_with_champion: float | None
    duration_s: float
    oof_hash: str
    test_hash: str | None
    data_hash: str
    submitted: bool = False
    public_lb: float | None = None
    notes: str = ""
    verdict: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def append(rec: Record) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec.to_dict(), default=str) + "\n")


def load_all() -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def get_champion() -> dict | None:
    if not CHAMPION_FILE.exists():
        return None
    return json.loads(CHAMPION_FILE.read_text(encoding="utf-8"))


def set_champion(rec: Record, public_lb: float | None = None) -> None:
    d = rec.to_dict()
    d["public_lb"] = public_lb
    CHAMPION_FILE.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")


def next_id(prefix: str = "exp") -> str:
    n = len(load_all()) + 1
    return f"{prefix}{n:04d}"


def log_experiment(
    *,
    family: str,
    featureset: str,
    params: dict,
    seed: int,
    fold_scheme: str,
    oof_auc: float,
    fold_aucs: list,
    oof: np.ndarray,
    duration_s: float,
    data_hash: str,
    test_pred: np.ndarray | None = None,
    exp_id: str | None = None,
    notes: str = "",
    verdict: str = "",
    extra: dict | None = None,
) -> Record:
    """Compute champion comparison metrics, write the record, return it."""
    ch = get_champion()
    paired, corr, delta = [], None, None
    if ch is not None:
        ch_oof = np.load(Path(ch["oof_path"]))
        if len(ch_oof) == len(oof):
            corr = float(np.corrcoef(ch_oof, oof)[0, 1])
            for k in range(len(fold_aucs)):
                m = None
                paired.append(None)
            # paired fold deltas need fold ids; caller supplies them in extra
            fids = (extra or {}).get("fold_ids")
            if fids is not None:
                paired = [float(fold_aucs[k] - ch["fold_aucs"][k]) for k in range(len(fold_aucs))]
                delta = float(oof_auc - ch["oof_auc"])
    rec = Record(
        exp_id=exp_id or next_id(),
        ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git=git_commit(),
        family=family,
        featureset=featureset,
        fold_scheme=fold_scheme,
        params=params,
        seed=seed,
        oof_auc=float(oof_auc),
        fold_aucs=[float(x) for x in fold_aucs],
        oof_std=float(np.std(fold_aucs)),
        delta_vs_champion=None if delta is None else float(delta),
        paired_fold_deltas=[float(x) for x in paired] if paired else [],
        corr_with_champion=corr,
        duration_s=round(float(duration_s), 1),
        oof_hash=arr_sha256(np.asarray(oof, dtype="float32")),
        test_hash=None if test_pred is None else arr_sha256(np.asarray(test_pred, dtype="float32")),
        data_hash=data_hash,
        notes=notes,
        verdict=verdict,
        extra={k: v for k, v in (extra or {}).items() if k != "fold_ids"},
    )
    append(rec)
    return rec