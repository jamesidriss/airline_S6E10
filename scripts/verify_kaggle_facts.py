"""Verify official competition facts from live Kaggle sources.

Never prints credentials. Reuses the Kaggle auth already configured on the machine.
Writes research/raw/kaggle_facts.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SLUG = "playground-series-s6e10"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "raw" / "kaggle_facts.json"


def build_session() -> "requests.Session":
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi

    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KaggleAPI/2.x"
    try:
        api = KaggleApi()
        api.authenticate()
        token = api.config.get("kaggle.api_token")
        if token:
            sess.headers["Authorization"] = f"Bearer {token}"
    except Exception as exc:  # noqa: BLE001
        print("warn: kaggle auth unavailable:", type(exc).__name__)
    return sess


def main() -> None:
    import requests

    sess = build_session()
    facts: dict = {"slug": SLUG}

    calls = [
        ("competition", "competitions.CompetitionService", "GetCompetition", {"competitionName": SLUG}),
        ("rules", "competitions.CompetitionService", "GetCompetitionRulesTabContent", {"competitionName": SLUG}),
        ("data_page", "competitions.CompetitionService", "ListDataFiles", {"competitionName": SLUG}),
        ("timeline", "competitions.CompetitionService", "ListCompetitionMilestones", {"competitionName": SLUG}),
        ("submissions", "competitions.SubmissionService", "ListUserSubmissions", {"competitionName": SLUG}),
    ]
    for name, svc, meth, payload in calls:
        url = f"https://www.kaggle.com/api/i/{svc}.{meth}"
        try:
            r = sess.post(url, json=payload, timeout=90)
            body = r.text
            facts[name] = {"status": r.status_code, "body": body[:40000]}
            print(f"{name}: HTTP {r.status_code} len={len(body)}")
        except Exception as exc:  # noqa: BLE001
            facts[name] = {"error": repr(exc)}
            print(f"{name}: ERROR {exc!r}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())