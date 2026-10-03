"""Submission tracker: counts daily usage against the 10/day cap and records public scores."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.common import SUBMISSIONS

LEDGER = SUBMISSIONS / "kaggle_submissions.csv"
CAP = 10


def _load() -> pd.DataFrame:
    return pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(
        columns=["ts", "file", "description", "ref", "status", "publicScore", "privateScore"]
    )


def used_today() -> int:
    df = _load()
    if df.empty:
        return 0
    today = pd.Timestamp.utcnow().date().isoformat()
    return int((df["ts"].astype(str).str[:10] == today).sum())


def remaining() -> int:
    return CAP - used_today()


def record(file: str, description: str, ref: str = "", status: str = "submitted",
           public: float | None = None) -> None:
    df = _load()
    row = {"ts": pd.Timestamp.utcnow().isoformat(), "file": file, "description": description,
           "ref": ref, "status": status, "publicScore": public, "privateScore": None}
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(LEDGER, index=False)


def poll(submission_file: str | None = None, tries: int = 12, wait: int = 60) -> pd.DataFrame:
    """Poll Kaggle until the named (or most recent) submission is scored."""
    import time

    df = _load()
    if submission_file is None:
        submission_file = df.iloc[-1]["file"]
    ref = df.loc[df["file"] == submission_file, "ref"]
    ref = ref.iloc[-1] if len(ref) else ""
    for _ in range(tries):
        r = subprocess.run(
            ["kaggle", "competitions", "submissions", "-c", "playground-series-s6e10"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(r.stderr[-1500:])
            return df
        txt = r.stdout
        print(txt[:4000])
        for line in txt.splitlines():
            if submission_file in line or (ref and str(ref) in line):
                print("MATCH:", line)
        # try to parse a numeric score next to the file name
        if "complete" in txt.lower() or "0.9" in txt:
            break
        time.sleep(wait)
    return df


def submit(file: Path, description: str) -> str:
    if remaining() <= 0:
        raise SystemExit(f"REFUSING: daily submission cap reached ({used_today()}/{CAP})")
    cmd = ["kaggle", "competitions", "submit", "-c", "playground-series-s6e10",
           "-f", str(file), "-m", description]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-2000:])
    print(r.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit("submit failed")
    ref = ""
    for tok in r.stdout.split():
        if tok.count("-") >= 2 and any(c.isdigit() for c in tok):
            ref = tok
    record(Path(file).name, description, ref=ref)
    return ref


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        print(f"used today: {used_today()}/{CAP}   remaining: {remaining()}")
        print(_load().to_string(index=False))
    elif cmd == "submit":
        submit(Path(sys.argv[2]), sys.argv[3])
    elif cmd == "poll":
        poll(*sys.argv[2:])