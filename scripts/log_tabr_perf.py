"""Log the measured TabR performance diagnosis to experiments/ledger.jsonl.

Kept separate from log_phase4.py because these are PERFORMANCE findings, not model results: no
fold AUC was produced by any of them. Three successive hypotheses about the >60 min/epoch cost were
each stated, then each MEASURED and REFUTED, before the real driver was identified. That sequence is
worth recording precisely, because the refuted hypotheses are what a future reader would otherwise
re-derive.

Idempotent: an entry is skipped if its exp_id is already present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "experiments" / "ledger.jsonl"

ENTRIES = [
    {
        "exp_id": "tabr_perf_hypothesis1_candidate_encoding_batching",
        "oof_auc": None,
        "fold_aucs": [],
        "verdict": "REFUTED (real but minor)",
        "hypothesis": "candidate_encoding_batch_size=256 chunks the 503k-row retrieval database "
                      "into ~1969 pieces re-encoded every training step, i.e. ~242k GPU launches "
                      "per epoch, which dominates runtime",
        "measured": "profiler: memtopk (the retrieval top-k) totals 2.9 s per step; the GEMMs "
                    "~4 s. Candidate encoding is nowhere near dominant.",
        "action_taken": "still changed (single-pass encoding is exact and cheaper), but it was not "
                        "the cause",
    },
    {
        "exp_id": "tabr_perf_hypothesis2_process_wide_fp32",
        "oof_auc": None,
        "fold_aucs": [],
        "verdict": "REFUTED (real but minor)",
        "hypothesis": "the process-wide fp32 guard forced the encoder/predictor off TF32 tensor "
                      "cores",
        "measured": "kept TF32 for encoder/predictor; forward time barely moved",
        "action_taken": "kept anyway -- it is free and retrieval stays exact",
    },
    {
        "exp_id": "tabr_perf_hypothesis3_torch_isin",
        "oof_auc": None,
        "fold_aucs": [],
        "verdict": "REFUTED",
        "hypothesis": "tabr.py:506 calls torch.isin(train_indices, batch_indices) every training "
                      "step, and torch.isin loops over the 4096 test elements => ~4096 kernel "
                      "launches/step",
        "measured": "direct micro-benchmark at the real sizes (N=503739, B=4096): "
                    "torch.isin 50.8 ms/step = 6.2 s/epoch; an equivalent bool-mask scatter is "
                    "2.0 ms/step (25x faster) and returns a bit-identical index tensor. "
                    "get_Xy (candidate gather) is 0.040 s/step. So these are SECONDS per epoch, "
                    "not the missing ~60 min.",
        "note": "the bool-mask rewrite is still worth having, but it is not the fix",
    },
    {
        "exp_id": "tabr_perf_root_cause_context_size",
        "oof_auc": None,
        "fold_aucs": [],
        "verdict": "ROOT CAUSE FOUND",
        "measurement": {
            "method": "scripts/profile_tabr_step.py -- 3 real training steps on the real 503k-row "
                      "assembled view, timing TabrModel.forward vs the rest of training_step",
            "per_step_forward_seconds": {
                "ctx96_d128": 29.994,
                "ctx32_d128": 1.031,
                "ctx96_d64": 32.753,
            },
            "get_Xy_seconds_per_step": 0.040,
            "implied_epoch_minutes": {"ctx96_d128": 61.4, "ctx32_d128": 2.0, "ctx96_d64": 66.9},
        },
        "conclusion": (
            "Cost is driven by retrieval CONTEXT SIZE, not by the embedding width: cutting context "
            "96 -> 32 is a 29x speedup while halving d_main does nothing. A 3x smaller context "
            "should have given ~3x, so the superlinearity is a memory-pressure cliff rather than "
            "arithmetic -- at context 96 the process sat at 15.7 of 16.3 GB VRAM and the caching "
            "allocator thrashes, while at context 32 it fits comfortably. TabR's default context of "
            "96 (and the TabZilla-derived preset) is tuned for far smaller datasets than 504k rows."
        ),
        "action_taken": "fold-0 screening re-run at context_size=32, which is ~2 min/epoch and "
                        "makes a trusted multi-fold run feasible",
        "caveat": "context_size is also TabR's neighbourhood size, so 32 may cost accuracy; that "
                  "is an empirical question the screening run answers, and larger contexts can be "
                  "explored later now that the cost driver is known.",
    },
]


def main() -> int:
    existing = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line).get("exp_id"))
                except Exception:  # noqa: BLE001
                    pass
    added = 0
    with LEDGER.open("a", encoding="utf-8") as f:
        for e in ENTRIES:
            if e["exp_id"] in existing:
                print(f"  skip: {e['exp_id']}")
                continue
            e.setdefault("date", "2026-10-04")
            f.write(json.dumps(e) + "\n")
            added += 1
            print(f"  appended: {e['exp_id']} [{e['verdict']}]")
    print(f"\n{added} entries appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
