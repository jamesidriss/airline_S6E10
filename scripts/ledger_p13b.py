"""Append the Phase 13B ledger entries."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()

ROWS = [
    {
        "phase": "p13b",
        "id": "p13b_AUX_GAINS_COMPOUND_ACROSS_SLOTS_BUT_STOP_AT_PLUS_0_745E_5",
        "kind": "EXPERIMENT",
        "claim": "Does the auxiliary-task gain COMPOUND across the six recoverable extra_trees slots "
                 "in v3? Each slot is retrained with the 13 aux features appended, with its OWN "
                 "original seed and parameters, and all six are spliced into v3 SIMULTANEOUSLY so "
                 "the result does not depend on the order they are visited.",
        "finding": "It compounds, but sub-linearly, and stops well short of the gate. All-slot "
                   "marginal on v3: +0.706, +1.164, +0.468, +0.224, +1.160e-5, mean +0.745e-5, "
                   "SE 0.187, t +3.99, 5/5 positive. That is 2.8x the single-slot +0.27e-5, so six "
                   "slots buy 2.8x rather than 6x.",
        "evidence": "Per-slot standalone gains on fold 0 ranged +26.03e-5 (xt_xt_d63) down to "
                    "-7.45e-5 (xt_xt_d255), and every slot's new prediction correlates 0.9980-0.9990 "
                    "with the member it replaces. The cumulative curve is NON-MONOTONIC in slot "
                    "order: fold 0 runs +0.31, +0.63, +0.69, +0.93, +0.79, +0.71, rising then "
                    "falling. Fold 3 is noisier still, reaching +0.003e-5 at five slots before "
                    "recovering to +0.224e-5 at six.",
        "verdict": "Gate: mean +0.745e-5 is 2.0x short of +1.5e-5; 5/5 positive PASSES; mean/SE 3.99x "
                   "PASSES. FAIL on effect size alone. NO ADMISSION, NO SUBMISSION. Extrapolating "
                   "to the remaining extra_trees members is REFUSED for the reason Phase 11R "
                   "withdrew arithmetic scaling: family weight times a per-slot number does not "
                   "predict blend gain, and the measured cumulative curve is non-monotonic, so any "
                   "extrapolation from it would invent precision. p13b_final.py therefore reports "
                   "only the order-independent all-slot marginal and attributes nothing to "
                   "individual slots.",
    },
    {
        "phase": "p13b",
        "id": "p13b_HEADLINE_RECOMPUTED_FROM_PREDICTION_VECTORS_AFTER_RUN_SCOPE_OVERWRITE",
        "kind": "CORRECTION",
        "claim": "Running p13b twice (folds 0,1,2 then folds 3,4) left reports/p13b_runs.json "
                 "holding only folds 3 and 4, so the headline number existed only in two log files.",
        "finding": "This is the run-scoped-report overwrite hazard already recorded in this campaign, "
                   "and it recurred. Anyone reading p13b_runs.json alone would have seen a 2-fold "
                   "+0.692e-5 and concluded the gate was missed by more than it is.",
        "evidence": "scripts/p13b_final.py recomputes the authoritative 5-fold all-slot marginal from "
                    "the per-fold .npy prediction vectors, which were never overwritten, and asserts "
                    "the v3 float32 geometry is bit-exact before reporting. The recomputed +0.745e-5 "
                    "matches the logs exactly, so no number changed; only the record was made durable.",
        "verdict": "FIXED by writing reports/p13b_final.json. Durable multi-fold records belong in the "
                   "*_final.json / *_swap.json / *_audit.json family, never in a per-invocation file.",
    },
    {
        "phase": "p13b",
        "id": "p13b_SILENTLY_WRONG_SLOT_SEED_FROM_STRING_PARSING",
        "kind": "CORRECTION",
        "claim": "run_phase13b.py derived each slot's seed from its exp_id with "
                 "int(s.rsplit('_s', 1)[1]), falling back to the loop counter when that failed.",
        "finding": "It crashes on xt_xt_d127_bin63 ('127_bin63' is not an int) and, worse, SILENTLY "
                   "gives the loop counter to any slot whose name contains no '_s'. So xt_xt_d255 was "
                   "trained with seed 2 when its recorded seed is 4. A silently wrong seed produces a "
                   "confident, plausible, wrong number.",
        "evidence": "This is the same failure that made Phase 12's fold-0 screen untrustworthy, where "
                    "two fold-index conventions both coincided on fold 0. It is the campaign's "
                    "recurring theme: anything derived rather than read is a place a wrong value can "
                    "hide.",
        "verdict": "FIXED. Seeds are stated explicitly in SLOT_SEEDS, cross-checked against "
                   "experiments/ledger.jsonl at startup, and the run raises SystemExit on any "
                   "disagreement.",
    },
]


def main() -> int:
    p = Path("experiments/ledger.jsonl")
    existing = p.read_text(encoding="utf-8")
    n0 = sum(1 for line in existing.splitlines() if line.strip())
    with p.open("a", encoding="utf-8") as f:
        for r in ROWS:
            if f'"{r["id"]}"' in existing:
                print("already present, skipping:", r["id"])
                continue
            f.write(json.dumps({"ts": NOW, **r}) + "\n")
    n1 = sum(1 for line in p.open(encoding="utf-8") if line.strip())
    bad = 0
    for i, line in enumerate(p.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except Exception as exc:
            bad += 1
            print("line", i, "INVALID:", exc)
    print(f"ledger lines {n0} -> {n1}; invalid: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())