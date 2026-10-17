"""Print every non-NOT_STARTED catalogue row with fidelity + finding.

Companion to _inventory.py: shows what has been evaluated (and how far), so
S7's re-examination of partial/failed classifications can be done from live
state rather than memory.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cat = json.loads((ROOT / "research" / "phase20_exhaustive" /
                  "experiment_catalogue_96.json").read_text(encoding="utf-8"))
rows = cat["experiments"] if "experiments" in cat else cat
for x in rows:
    if x["status"] == "NOT_STARTED":
        continue
    print(f"{x['experiment_number']:3d} {x['status']:20s} {x['fidelity_reached']:4s} "
          f"{x['canonical_name'][:56]}")
    if x.get("finding") and x["status"] in ("PARTIAL_VALID", "COMPLETE_NEGATIVE"):
        print(f"      -> {x['finding'][:240]}")
