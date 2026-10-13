"""Deterministic experiment fingerprinting (Phase18).

A fingerprint is the sha256 over a canonical JSON of:
  data_hash, fold_hash, feature_config, model_config, seed,
  source_hash (of the producing script + its imported project modules),
  training_policy, inference_policy, dependency_versions.

Before running a job, search the registry for an identical valid fingerprint:
  - valid & complete            -> reuse (never retrain)
  - resource-failed             -> resume only under compatible contract
  - scientifically invalid      -> never reuse predictions
  - rejected (null/negative)    -> retrain only with a NEW hypothesis id
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# Dependency versions that materially affect numerics. Missing ones record null.
_TRACKED = (
    "numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost",
    "torch", "scipy", "pyarrow",
)


def dependency_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _TRACKED:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - absence is a recorded fact, not an error
            out[name] = None
    out["python"] = platform.python_version()
    return out


def source_hash(paths: list[Path]) -> str:
    """sha256 over (relpath, bytes) of the listed source files, sorted by path."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def project_source_hash(entry: Path) -> str:
    """Hash an entry script plus every src/** module it transitively imports.

    Lightweight static scan: follows `from src.x import ...` / `import src.x`
    lines recursively. Deliberately conservative: unknown imports are ignored
    (they are captured by dependency_versions instead).
    """
    seen: set[Path] = set()
    queue = [entry]
    while queue:
        f = queue.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        text = f.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            s = line.strip()
            mod = None
            if s.startswith("from src.") or s.startswith("from scripts."):
                mod = s.split()[1]
            elif s.startswith("import src.") or s.startswith("import scripts."):
                mod = s.split()[1].split(",")[0]
            if mod:
                rel = mod.replace(".", "/") + ".py"
                queue.append(ROOT / rel)
    return source_hash(sorted(seen))


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(
    *,
    data_hash: str,
    fold_hash: str,
    feature_config: dict,
    model_config: dict,
    seed: int,
    source_entry: Path,
    training_policy: dict,
    inference_policy: dict,
) -> dict:
    payload = {
        "data_hash": data_hash,
        "fold_hash": fold_hash,
        "feature_config": feature_config,
        "model_config": model_config,
        "seed": int(seed),
        "source_hash": project_source_hash(Path(source_entry)),
        "training_policy": training_policy,
        "inference_policy": inference_policy,
        "dependency_versions": dependency_versions(),
    }
    fp = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return {"fingerprint": fp, "payload": payload}


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def python_exe() -> str:
    return sys.executable
