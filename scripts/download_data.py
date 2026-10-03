"""Download all data required by the campaign. Idempotent; safe to re-run.

Usage:  python scripts/download_data.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ORIG = ROOT / "data" / "original"

COMP = "playground-series-s6e10"
ORIGINALS = [
    "arseniyshutko/binary-aviation-satisfaction-129k",  # the generator source (verified)
    "teejmahal20/airline-passenger-satisfaction",      # canonical attribution
    "mysarahmadbhat/airline-passenger-satisfaction",
    "nilanjansamanta1210/airline-passenger-satisfaction",
    "raminhuseyn/airline-customer-satisfaction",
    "yakhyojon/customer-satisfaction-in-airline",
    "binaryjoker/airline-passenger-satisfaction",
]


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(f"failed: {cmd}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if not (RAW / "train.csv").exists():
        sh(["kaggle", "competitions", "download", "-c", COMP, "-p", str(RAW)])
        import zipfile

        zipfile.ZipFile(RAW / f"{COMP}.zip").extractall(RAW)
    for ref in ORIGINALS:
        out = ORIG / ref.replace("/", "__")
        if not any(out.glob("*.csv")):
            out.mkdir(parents=True, exist_ok=True)
            sh(["kaggle", "datasets", "download", "-d", ref, "-p", str(out), "--unzip"])
    print("done")


if __name__ == "__main__":
    sys.exit(main())