"""Experiment registry with atomic writes and duplicate protection (Phase18).

Registry file: experiments/phase18_registry.jsonl (append-only records) plus a
pointer directory artifacts/phase18/<fingerprint>/ for run outputs.

Rules enforced here:
  * A completed valid fingerprint may not be retrained under the same ID.
  * A scientifically invalid/rejected fingerprint may not be reused silently.
  * Writes are atomic: write to temp file, fsync, os.replace.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from src.common import EXPERIMENTS
from src.phase18.fingerprint import git_head

REGISTRY = EXPERIMENTS / "phase18_registry.jsonl"

VALID_STATUSES = {
    "PLANNED", "RUNNING", "COMPLETE", "RESOURCE_BLOCKED",
    "INVALID", "NEGATIVE", "NULL", "POSITIVE_UNCONFIRMED",
    "REPLICATED", "PRIMARY_ADMISSION_PASS", "SHADOW_CONFIRMED",
    "TEST_CERTIFIED", "SUBMITTED", "FINALIST",
}


def atomic_write_text(path: Path, text: str) -> None:
    """Temp-file + os.replace so a partial write never looks complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str))


def append_record(rec: dict) -> None:
    if rec.get("status") not in VALID_STATUSES:
        raise ValueError(f"invalid status {rec.get('status')!r}")
    rec.setdefault("utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    rec.setdefault("git", git_head())
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, sort_keys=True, default=str) + "\n"
    with open(REGISTRY, "a", encoding="utf-8", newline="\n") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def load_all() -> list[dict]:
    if not REGISTRY.exists():
        return []
    out = []
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def lookup(fingerprint: str) -> list[dict]:
    return [r for r in load_all() if r.get("fingerprint") == fingerprint]


def check_reuse_allowed(fingerprint: str) -> tuple[bool, str]:
    """Return (allowed_to_train, reason). Duplicate-protection policy."""
    recs = lookup(fingerprint)
    if not recs:
        return True, "no prior record"
    latest = recs[-1]
    status = latest.get("status")
    if status == "COMPLETE":
        return False, f"identical valid job already COMPLETE ({latest.get('exp_id')}); reuse outputs"
    if status == "RESOURCE_BLOCKED":
        return True, "prior attempt was resource-blocked; resume permitted under same contract"
    if status == "INVALID":
        return False, "prior attempt INVALID (scientific bug); predictions must not be reused; fix code -> new fingerprint"
    if status in {"NEGATIVE", "NULL"}:
        return True, "prior identical run rejected scientifically; retraining requires a NEW hypothesis id (documented)"
    return True, f"status {status}; continuing is permitted"


def artifacts_dir(fingerprint: str) -> Path:
    from src.common import ARTIFACTS
    return ARTIFACTS / "phase18" / fingerprint[:16]
