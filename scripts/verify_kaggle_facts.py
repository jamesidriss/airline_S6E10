"""Verify official competition facts from live Kaggle sources.

SECURITY CONTRACT
-----------------
This script talks to `www.kaggle.com`, whose frontend HTML embeds public Google API keys
(`window.kaggleStackdriverConfig`). A previous version of this file persisted raw response
bodies, which copied that key into `research/raw/kaggle_facts.json` and from there into git
history. GitHub secret scanning flagged it.

Rules now enforced:
  * **Only JSON responses are ever persisted, and only the fields we whitelist.**
  * Non-JSON responses store safe metadata only: HTTP status, content type, body length and a
    sanitised error classification (never the body, never a fragment of it).
  * Defensive final scrub of every persisted string.
  * Never prints or stores credentials.

Writes `research/raw/kaggle_facts.json` (safe by construction).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SLUG = "playground-series-s6e10"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "raw" / "kaggle_facts.json"

# Anything that looks like a credential is removed from anything we persist, regardless of where
# it came from. Belt-and-braces on top of the "JSON only" rule.
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"ya29\.[0-9A-Za-z_-]{20,}"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(?:bearer\s+)[A-Za-z0-9._~+/=-]{20,}"),
]

# Fields we keep when a JSON response is stored. Everything else is dropped.
SAFE_KEYS = {
    "id", "title", "briefDescription", "dateEnabled", "deadline", "hasLeaderboard",
    "leaderboardPercentage", "maxDailySubmissions", "numScoredSubmissions", "maxTeamSize",
    "totalTeams", "totalCompetitors", "totalSubmissions", "rowIdColumnName",
    "evaluationAlgorithm", "reward", "license", "numPrizes", "forumId", "totalSolutionRows",
    "submissionSizeLimitMb", "organization", "categories", "hostName", "hasSolution",
    "competitionName", "dateCreated", "scoreTruncationNumDecimals", "totalJoinedUsers",
    "pages", "name", "id", "order",
}


def scrub(value):
    """Recursively drop secret-shaped substrings from anything on its way to disk."""
    if isinstance(value, str):
        for pat in SECRET_PATTERNS:
            value = pat.sub("[REDACTED]", value)
        return value
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def classify_error(status: int, content_type: str) -> str:
    """A short, non-revealing classification of a response we will not persist."""
    if 200 <= status < 300:
        return "ok"
    if status == 401:
        return "unauthenticated"
    if status == 403:
        return "forbidden"
    if status == 404:
        return "not_found_or_wrong_endpoint"
    if status == 429:
        return "rate_limited"
    if 500 <= status < 600:
        return "server_error"
    if "html" in (content_type or "").lower():
        return "unexpected_html_response"
    return f"http_{status}"


def build_session():
    """Return a bare opener.

    These competition endpoints are public and need no authentication. This script therefore
    handles **no credentials at all** -- there is no token to leak, print, or store, which is a
    strictly smaller attack surface than the previous version that read the Kaggle token into an
    Authorization header.

    ``urllib`` is used rather than ``requests`` because it is the transport that this repository's
    other Kaggle fetchers already use successfully.
    """
    import urllib.request

    return urllib.request


def _post(url: str, payload: dict, timeout: int = 90):
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
        return r.status, dict(r.headers), body


def main() -> None:
    facts: dict = {"slug": SLUG, "note": "JSON-only, whitelisted fields, secret-scrubbed"}

    calls = [
        ("competition", "competitions.CompetitionService/GetCompetition", {"competitionName": SLUG}),
        ("pages", "competitions.PageService/ListPages", {"competitionId": 125224}),
    ]
    for name, svc, payload in calls:
        url = f"https://www.kaggle.com/api/i/{svc}"
        try:
            status, headers, body = _post(url, payload)
            ctype = headers.get("Content-Type", "")
            meta = {"status": status, "content_type": ctype, "body_len": len(body),
                    "classification": classify_error(status, ctype)}
            try:
                data = json.loads(body)
            except Exception:  # noqa: BLE001
                # NEVER persist the body of a non-JSON response.
                meta["persisted"] = False
                meta["reason"] = "non-JSON response; body intentionally not stored"
                facts[name] = meta
                print(f"{name}: HTTP {status} {ctype[:40]} len={len(body)} "
                      f"[{meta['classification']}] body NOT stored")
                continue
            if isinstance(data, dict):
                kept = {k: v for k, v in data.items() if k in SAFE_KEYS}
                meta["persisted"] = True
                meta["kept_fields"] = sorted(kept)
                facts[name] = meta | kept
                print(f"{name}: HTTP {status} len={len(body)} kept={len(kept)} fields")
            else:
                meta["persisted"] = False
                meta["reason"] = "JSON but not an object; body intentionally not stored"
                facts[name] = meta
                print(f"{name}: HTTP {status} JSON non-object; body NOT stored")
        except Exception as exc:  # noqa: BLE001
            facts[name] = {"error_type": type(exc).__name__, "classification": "request_exception"}
            print(f"{name}: ERROR {type(exc).__name__}")

    facts = scrub(facts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())