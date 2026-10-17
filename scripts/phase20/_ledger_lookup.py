"""Read-only ledger lookup for the S7 fidelity review.

Prints full ledger entries matching given experiment numbers (or a substring
of exp_id/hypothesis_id), so the actual recorded fidelity of an OOM-marked or
proxy-marked negative can be re-read before reclassifying it.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
wanted = {int(a) for a in sys.argv[1:] if a.isdigit()}
substrs = [a.lower() for a in sys.argv[1:] if not a.isdigit()]

lines = (ROOT / "experiments" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
for ln in lines:
    if not ln.strip():
        continue
    d = json.loads(ln)
    en = d.get("experiment_number")
    hit = (en in wanted) if wanted else False
    if not hit and substrs:
        blob = (str(d.get("exp_id", "")) + str(d.get("hypothesis_id", ""))).lower()
        hit = any(s in blob for s in substrs)
    if hit:
        print(f"=== exp {en} {d.get('exp_id')} [{d.get('verdict')}] ===")
        res = d.get("results") or {}
        print("  results:", json.dumps(res)[:600])
        print("  finding:", str(d.get("finding"))[:900])
        print("  closes:", d.get("closes_rows"), "| prereg:", str(d.get("prereg"))[:160])
        print()
