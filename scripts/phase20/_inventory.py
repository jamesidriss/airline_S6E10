"""Print the live catalogue status distribution and the NOT_STARTED queue.

Read-only inventory helper for campaign planning (S9/S30): recomputes the exact
list from local state instead of trusting any historical snapshot.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cat = json.loads((ROOT / "research" / "phase20_exhaustive" /
                  "experiment_catalogue_96.json").read_text(encoding="utf-8"))
rows = cat["experiments"] if "experiments" in cat else cat
print("TOTAL", len(rows))
print(dict(Counter(x["status"] for x in rows)))
ns = [x for x in rows if x["status"] == "NOT_STARTED"]
print(f"--- NOT_STARTED {len(ns)} ---")
for x in ns:
    print(f"{x['experiment_number']:3d} {x['research_family']:6s} "
          f"P{x['priority'][-1]} {x['canonical_name'][:70]}")
blocked = [x for x in rows if x["status"] in
           ("LICENSE_BLOCKED", "RESOURCE_BLOCKED", "DEPENDENCY_BLOCKED")]
if blocked:
    print("--- BLOCKED ---")
    for x in blocked:
        print(f"{x['experiment_number']:3d} {x['status']:18s} {x['canonical_name'][:60]}")
sys.exit(0)
