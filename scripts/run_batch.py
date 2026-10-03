"""Sequential, resumable batch driver.

Every job is idempotent: the underlying runners skip experiments already present in the
prediction store, so this can be re-run safely after an interruption.

Usage:
  python scripts/run_batch.py --plan core      # the standard diversity batch
  python scripts/run_batch.py --plan eval      # finalists on 10 folds
  python scripts/run_batch.py --jobs "a|b|c"  # ad-hoc, '|' separated
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

JOBS: dict[str, list[list[str]]] = {
    "core": [
        ["scripts/run_models.py", "--view", "full", "--model", "realmlp", "--folds", "primary",
         "--tag", "prod5", "--epochs", "6", "--ens", "8", "--seed", "7", "--save-test"],
        ["scripts/run_models.py", "--view", "full", "--model", "realmlp", "--folds", "primary",
         "--tag", "prod5", "--epochs", "6", "--ens", "8", "--seed", "123", "--save-test"],
        ["scripts/run_models.py", "--view", "full", "--model", "realmlp", "--folds", "primary",
         "--tag", "prod5", "--epochs", "6", "--ens", "8", "--seed", "2024", "--save-test"],
        ["scripts/run_views.py", "--views", "core3_ogsurf", "--models", "lgbm,xgb,cat",
         "--folds", "primary", "--tag", "prod5", "--save-test"],
        ["scripts/run_models.py", "--view", "core3_ogsurf", "--model", "realmlp",
         "--folds", "primary", "--tag", "prod5", "--epochs", "6", "--ens", "8", "--seed", "11",
         "--save-test"],
    ],
    "tenfold": [
        ["scripts/run_views.py", "--views", "full", "--models", "lgbm,xgb,cat",
         "--folds", "block10", "--tag", "prod10", "--save-test"],
        ["scripts/run_models.py", "--view", "full", "--model", "realmlp", "--folds", "block10",
         "--tag", "prod10", "--epochs", "6", "--ens", "8", "--seed", "1", "--save-test"],
    ],
    "diverse": [
        # a deliberately different representation for extra ensemble diversity
        ["scripts/run_views.py", "--views", "raw_ext", "--models", "lgbm,xgb,cat",
         "--folds", "primary", "--tag", "div", "--save-test"],
        ["scripts/run_views.py", "--views", "raw_trans", "--models", "xgb",
         "--folds", "primary", "--tag", "div", "--save-test"],
        ["scripts/run_models.py", "--view", "core3", "--model", "realmlp", "--folds", "primary",
         "--tag", "div", "--epochs", "6", "--ens", "8", "--seed", "5", "--save-test"],
    ],
}


def run(job: list[str]) -> tuple[int, str]:
    t0 = time.time()
    r = subprocess.run([PY] + job, cwd=ROOT, capture_output=True, text=True, timeout=36 * 3600)
    tail = "\n".join(
        [ln for ln in (r.stdout or "").splitlines()
         if "==>" in ln or "SUMMARY" in ln or "Traceback" in ln or "Error" in ln]
    )
    if r.returncode != 0:
        tail += "\n" + "\n".join((r.stderr or "").splitlines()[-12:])
    return r.returncode, tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="core")
    ap.add_argument("--jobs", default="")
    args = ap.parse_args()

    if args.jobs:
        jobs = [j.split(" ") for j in args.jobs.split("|")]
    else:
        jobs = JOBS[args.plan]
    print(f"### batch plan={args.plan or 'adhoc'}  {len(jobs)} jobs")
    for j in jobs:
        print("  $ " + " ".join(j))
    fails = 0
    for j in jobs:
        print(f"\n===== {time.strftime('%H:%M:%S')}  {' '.join(j[1:4])}", flush=True)
        rc, tail = run(j)
        print(tail)
        print(f"  -> rc={rc} ({time.time()-0:.0f})")
        if rc != 0:
            fails += 1
    print(f"\n### batch finished, {fails} job(s) failed")


if __name__ == "__main__":
    sys.exit(main())