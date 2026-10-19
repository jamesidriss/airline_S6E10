"""Ingest the downloaded raw8 Kaggle outputs into artifacts/phase23/<dest>/ with provenance.

Refuses to overwrite an existing ingest manifest (append-only discipline; --force
archives the previous manifest first). Copies the OOF/test vectors, the per-unit
files, and the result/probe JSONs, then records sha256 of every ingested file.

    .\\.venv\\Scripts\\python.exe scripts\\phase23\\ingest_raw8.py <downloaded_dir> \\
        --ref jamesripoll/s6e10-phase23-raw8-tabpfn35:3 --tag ""
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import arr_sha256, file_sha256, git_commit  # noqa: E402

N_TRAIN = 699635
N_TEST = 299844


def sha_or_none(p: Path) -> str | None:
    return file_sha256(p) if p.exists() else None


def check_vector(path: Path, n: int, what: str) -> dict:
    v = np.load(path, allow_pickle=False)
    if v.shape != (n,):
        raise SystemExit(f"{what}: bad shape {v.shape} != {(n,)}")
    n_nan = int(np.isnan(v).sum())
    finite = v[np.isfinite(v)]
    if finite.size and ((finite < 0) | (finite > 1)).any():
        raise SystemExit(f"{what}: values outside [0,1]")
    return {"shape": list(v.shape), "dtype": str(v.dtype), "n_nan": n_nan,
            "coverage": 1.0 - n_nan / n, "arr_sha256": arr_sha256(v)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="directory returned by `kaggle kernels output`")
    ap.add_argument("--ref", required=True, help="kernel ref:version that produced the outputs")
    ap.add_argument("--tag", default="", help="output prefix used by the kernel variant")
    ap.add_argument("--dest", default=None, help="override destination directory")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = args.source.resolve()
    if not src.is_dir():
        raise SystemExit(f"not a directory: {src}")
    dest = (ROOT / args.dest) if args.dest else (
        ROOT / ("artifacts/phase23/raw8" if not args.tag else f"artifacts/phase23/{args.tag.rstrip('_')}")
    )
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "ingest_manifest.json"
    if manifest_path.exists():
        if not args.force:
            raise SystemExit(f"refusing to overwrite {manifest_path} (use --force)")
        shutil.copy2(manifest_path, dest / "ingest_manifest.prev.json")

    tag = args.tag
    oof_path = src / f"{tag}raw8_oof.npy"
    if not oof_path.exists():
        raise SystemExit(f"missing {oof_path}")
    test_path = src / f"{tag}raw8_test.npy"
    result_path = src / f"{tag}raw8_result.json"
    probe_path = src / f"{tag}probe_summary.json"
    units_dir = src / f"{tag}units"

    info = {
        "oof": check_vector(oof_path, N_TRAIN, "raw8_oof"),
        "test": check_vector(test_path, N_TEST, "raw8_test") if test_path.exists() else None,
    }
    unit_files = sorted(units_dir.rglob("*")) if units_dir.is_dir() else []
    unit_files = [p for p in unit_files if p.is_file()]
    n_meta = sum(1 for p in unit_files if p.name.endswith("_meta.json"))

    copied = []
    for p in [oof_path, test_path, result_path, probe_path]:
        if p.exists():
            shutil.copy2(p, dest / p.name)
            copied.append((p.name, file_sha256(dest / p.name)))
    for p in unit_files:
        rel = p.relative_to(src)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        copied.append((str(rel).replace("\\", "/"), file_sha256(out)))

    try:
        dest_str = str(dest.relative_to(ROOT))
    except ValueError:
        dest_str = str(dest)
    manifest = {
        "ingested_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_commit(),
        "kernel_ref": args.ref,
        "tag": tag,
        "source_dir": str(src),
        "destination": dest_str,
        "vectors": info,
        "unit_files": len(unit_files),
        "unit_meta_files": n_meta,
        "copied": [{"file": n, "sha256": h} for n, h in copied],
        "oof_sha256": info["oof"]["arr_sha256"],
        "notes": [
            "unit files are the per-(fold, member) raw logits + metas; they are the durable evidence",
            "coverage < 1 means the run was partial (missing members/folds); the frozen evaluator refuses partial gates",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({
        "dest": manifest["destination"],
        "oof_sha256": manifest["oof_sha256"][:16],
        "coverage": info["oof"]["coverage"],
        "test": bool(info["test"]),
        "unit_files": len(unit_files),
        "meta_files": n_meta,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
