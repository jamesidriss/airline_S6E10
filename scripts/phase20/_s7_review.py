"""Print full catalogue rows for the S7 misclassification review.

Read-only: dumps blocking_reason + finding + next_action for the rows whose
negative status rests on an OOM, a proxy, or a single recipe variant.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WATCH = [15, 17, 30, 35, 36, 60, 63, 64, 65, 27, 29, 47, 13, 14, 12]
cat = json.loads((ROOT / "research" / "phase20_exhaustive" /
                  "experiment_catalogue_96.json").read_text(encoding="utf-8"))
rows = cat["experiments"] if "experiments" in cat else cat
by = {x["experiment_number"]: x for x in rows}
for n in WATCH:
    x = by.get(n)
    if not x:
        print(f"{n}: NOT IN CATALOGUE")
        continue
    print(f"=== {n} {x['canonical_name']} [{x['status']} / {x['fidelity_reached']}] ===")
    for k in ("blocking_reason", "finding", "next_action"):
        if x.get(k):
            print(f"  {k}: {x[k]}")
    print()
