"""Fetch official competition pages (overview / rules / data) from Kaggle's internal API.

SECURITY CONTRACT
-----------------
Read-only, public competition metadata. Never prints credentials.
Only JSON API responses are persisted (competition page markdown). Every byte of remote content
still passes through `scrub` before it reaches disk, so a future endpoint change cannot
reintroduce a credential -- see scripts/verify_kaggle_facts.py for the full contract.
Output: research/raw/kaggle_pages.json + research/raw/kaggle_pages.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"ya29\.[0-9A-Za-z_-]{20,}"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(?:bearer\s+)[A-Za-z0-9._~+/=-]{20,}"),
]


def scrub(value):
    """Recursively drop secret-shaped substrings from anything on its way to disk."""
    if isinstance(value, str):
        for pat in _SECRET_PATTERNS:
            value = pat.sub("[REDACTED]", value)
        return value
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value

SLUG = "playground-series-s6e10"
COMP_ID = 125224
BASE = "https://www.kaggle.com/api/i/"
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "research" / "raw"


def post(svc: str, payload: dict) -> dict:
    import urllib.request

    req = urllib.request.Request(
        BASE + svc,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    facts = {
        "slug": SLUG,
        "competition_id": COMP_ID,
        "competition": post("competitions.CompetitionService/GetCompetition", {"competitionName": SLUG}),
        "pages": {},
    }
    try:
        facts["pages_list"] = post("competitions.PageService/ListPages", {"competitionId": COMP_ID})
    except Exception as exc:  # noqa: BLE001
        facts["pages_list"] = {"error": repr(exc)}

    for p in facts.get("pages_list", {}).get("pages", []):
        facts["pages"][p["name"]] = {"id": p["id"], "order": p.get("order"), "content": p.get("content", "")}

    # forum + leaderboard sanity
    for name, svc, payload in [
        ("topics", "forums.TopicService/ListTopics", {"forumId": facts["competition"].get("forumId"), "pageSize": 50, "sortBy": "hot"}),
        ("summary", "competitions.CompetitionService/GetCompetitionSummaryStats", {"competitionName": SLUG}),
    ]:
        try:
            facts[name] = post(svc, payload)
        except Exception as exc:  # noqa: BLE001
            facts[name] = {"error": repr(exc)}

    facts = scrub(facts)
    md = ["# Kaggle official pages: " + SLUG, ""]
    for name, p in facts.get("pages", {}).items():
        md += [f"## page: {name} (id={p['id']})", "", p.get("content", ""), ""]
    md = scrub(md)
    (OUTDIR / "kaggle_pages.json").write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")
    (OUTDIR / "kaggle_pages.md").write_text("\n".join(md), encoding="utf-8")

    print("pages:", list(facts["pages"]))
    for name, p in facts["pages"].items():
        print(f"  {name}: {len(p['content'])} chars")
    print("forum topics:", len(facts.get("topics", {}).get("topics", []) or []))


if __name__ == "__main__":
    sys.exit(main())