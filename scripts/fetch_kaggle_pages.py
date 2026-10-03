"""Fetch official competition pages (overview / rules / data) from Kaggle's internal API.

Read-only, public competition metadata. Never prints credentials.
Output: research/raw/kaggle_pages.json + research/raw/kaggle_pages.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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

    (OUTDIR / "kaggle_pages.json").write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")

    md = ["# Kaggle official pages: " + SLUG, ""]
    for name, p in facts["pages"].items():
        md += [f"## page: {name} (id={p['id']})", "", p["content"], ""]
    (OUTDIR / "kaggle_pages.md").write_text("\n".join(md), encoding="utf-8")

    print("pages:", list(facts["pages"]))
    for name, p in facts["pages"].items():
        print(f"  {name}: {len(p['content'])} chars")
    print("forum topics:", len(facts.get("topics", {}).get("topics", []) or []))


if __name__ == "__main__":
    sys.exit(main())