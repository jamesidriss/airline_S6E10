"""Submission tracker: counts daily usage against the 10/day cap and records public scores."""

from __future__ import annotations

import json
import contextlib
import io
import math
import sys
from pathlib import Path

import pandas as pd

from src.common import SUBMISSIONS

LEDGER = SUBMISSIONS / "kaggle_submissions.csv"
CAP = 10
COMPETITION = "playground-series-s6e10"


def _load() -> pd.DataFrame:
    return pd.read_csv(LEDGER, dtype={"ref": "string"}) if LEDGER.exists() else pd.DataFrame(
        columns=["ts", "file", "description", "ref", "status", "publicScore", "privateScore"]
    )


def _save(df: pd.DataFrame) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    temporary = LEDGER.with_suffix(".csv.tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(LEDGER)


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        api.authenticate()
    return api


def used_today() -> int:
    df = _load()
    if df.empty:
        return 0
    today = pd.Timestamp.utcnow().date().isoformat()
    return int((df["ts"].astype(str).str[:10] == today).sum())


def remaining() -> int:
    return CAP - used_today()


def record(file: str, description: str, ref: str = "", status: str = "submitted",
           public: float | None = None) -> int:
    df = _load()
    row = {"ts": pd.Timestamp.utcnow().isoformat(), "file": file, "description": description,
           "ref": ref, "status": status, "publicScore": public, "privateScore": None}
    _save(pd.concat([df, pd.DataFrame([row])], ignore_index=True))
    return len(df)


def poll(submission_file: str | None = None, tries: int = 12, wait: int = 60) -> pd.DataFrame:
    """Poll Kaggle until the named (or most recent) submission is scored."""
    import time

    df = _load()
    if df.empty:
        raise ValueError("No recorded submission to poll")
    if tries < 1 or not 0 <= wait <= 60:
        raise ValueError("Use at least one poll and a wait between0 and60 seconds")
    if submission_file is None:
        submission_file = df.iloc[-1]["file"]
    selected = df.loc[df["file"] == submission_file]
    if selected.empty or pd.isna(selected.iloc[-1]["ref"]):
        raise ValueError("Polling requires the exact recorded numeric Kaggle ref")
    ref = str(selected.iloc[-1]["ref"])
    if not ref.isdecimal() or int(ref) <= 0:
        raise ValueError("Polling requires the exact recorded numeric Kaggle ref")
    api = _api()
    for attempt in range(tries):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            submissions = api.competition_submissions(COMPETITION, page_size=100) or []
        matches = [s for s in submissions if s is not None and str(s.ref) == ref]
        if len(matches) > 1:
            raise ValueError("Ambiguous Kaggle submission reference")
        if matches:
            submission = matches[0]
            if submission.file_name != Path(submission_file).name:
                raise ValueError("Kaggle ref and filename disagree")
            status = getattr(submission.status, "name", str(submission.status)).lower()
            updates = {"status": status}
            for field, column in (("public_score", "publicScore"), ("private_score", "privateScore")):
                value = getattr(submission, field, None)
                if value not in (None, ""):
                    score = float(value)
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("Invalid Kaggle AUC score")
                    updates[column] = score
            current = _load()
            mask = current["ref"].fillna("").astype(str) == ref
            if int(mask.sum()) != 1:
                raise ValueError("Recorded submission reference is missing or ambiguous")
            for column, value in updates.items():
                current.loc[mask, column] = value
            _save(current)
            print(json.dumps({"ref": ref, **updates}))
            if status in {"complete", "error", "cancelled", "canceled"}:
                return current
        if attempt + 1 < tries:
            time.sleep(wait)
    return _load()


def submit(file: Path, description: str) -> str:
    if remaining() <= 0:
        raise SystemExit(f"REFUSING: daily submission cap reached ({used_today()}/{CAP})")
    file = Path(file)
    if not file.is_file():
        raise FileNotFoundError(file)
    api = _api()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        limits = api.competition_get_submission_limits(COMPETITION)
    if limits.num_allowed_now <= 0:
        raise SystemExit("REFUSING: Kaggle reports no submissions available")
    # Reserve before the network mutation. An uncertain response must still
    # consume local budget; never automatically retry an upload.
    row = record(file.name, description, status="upload_started_response_unknown")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        response = api.competition_submit(str(file), description, COMPETITION, quiet=True)
    ref = str(response.ref)
    if not ref.isdecimal() or int(ref) <= 0:
        raise RuntimeError("Submission has no confirmed ref; preserve the reserved ledger row and reconcile before retrying")
    df = _load()
    df.loc[row, "ref"] = ref
    df.loc[row, "status"] = "submitted"
    _save(df)
    print(json.dumps({"ref": ref, "status": "submitted", "file": file.name}))
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
