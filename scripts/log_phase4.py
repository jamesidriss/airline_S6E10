"""Append the Phase 4 / TabR campaign record to experiments/ledger.jsonl.

Kept as a script so the entries are written with the same field discipline as the rest of the ledger
and can be re-run idempotently (an entry is skipped if its exp_id is already present).
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
        "exp_id": "tabr_infra_gpu_retrieval_shim",
        "oof_auc": None,
        "fold_aucs": [],
        "verdict": "infrastructure",
        "note": (
            "faiss-cpu (only CPython 3.11 Windows wheel) has no GpuIndexFlatConfig/GpuIndexFlatL2, "
            "which pytabkit's TabrModel requires; CPU IndexFlatL2 over 503k x 128 per step is not "
            "viable. Built TorchExactL2Index: chunked torch matmuls, GPU-native, exact squared-L2, "
            "fp32 with TF32 disabled inside the top-k. install_faiss_gpu_shim() supplies the two "
            "missing faiss symbols so pytabkit's own code path runs unchanged."
        ),
        "evidence": {
            "vs_float64_reference_rel_err": 2.4e-07,
            "topk_set_overlap": "4800/4800 exact on 600k x 265",
            "peak_mem": "222 MiB peak vs 586 MiB full distance matrix",
            "retrieval_queries_exercised": 967882,
            "label_params_in_retrieval_path": 0,
        },
    },
    {
        "exp_id": "tabr_attempt1_memory_efficient_false",
        "oof_auc": None,
        "fold_aucs": [],
        "verdict": "INVALID (not negative)",
        "note": (
            "Died silently at end of epoch 0, twice, with no Python traceback; pagefile peaked at "
            "10.1 GB. Root cause: tabr.py:281-283 sets grad_enabled(is_grad_enabled() and not "
            "memory_efficient), so memory_efficient=False kept autograd ENABLED while encoding all "
            "~503k candidate rows on every training step -- a full autograd graph over the entire "
            "retrieval database each step. That is the RAM exhaustion and essentially all of the "
            "~9 min/epoch. Also --d-main was never wired into params (labelled d128 while "
            "d_main=265 was in force), and validation features came from vb.static_tr[val] "
            "(237 cols) instead of the assembled view (285 cols)."
        ),
    },
    {
        "exp_id": "tabr_attempt2_candidate_encode_batching",
        "oof_auc": None,
        "fold_aucs": [],
        "verdict": "INVALID (not negative)",
        "note": (
            "Pipeline verified correct end to end: train=(503739, 289) es=(55969, 289) "
            "val=(139927, 289) assembled=285 static=237 rss=4.43GB, and the recorder proved the "
            "inner-ES path. Still ~58 min/epoch, which makes 30 epochs x 5 folds (~100 h) "
            "impossible. Cause: candidate_encoding_batch_size=256 split the 503k-row database into "
            "~1969 chunks re-encoded EVERY training step (~242k GPU launches/epoch), and the "
            "process-wide fp32 guard was also forcing the encoder/predictor off TF32 tensor cores. "
            "Fixed with single-pass candidate encoding (exact, cheap: 503k x d_main x 4 B) plus "
            "enable_tf32_process_wide() for the model while full_fp32_matmul() still guards only "
            "the top-k search."
        ),
    },
    {
        "exp_id": "tabr_pysupplied_protocol_audit",
        "oof_auc": None,
        "fold_aucs": [],
        "verdict": "protocol-verified",
        "note": (
            "Confirmed against source that the outer fold is a genuine evaluation fold. "
            "sklearn_base.py:443-454 shows n_refit=0 => alg_interface_ = cv_alg_interface_, i.e. "
            "NO refit on train+val. Early stopping uses _inner_es_split, an inner 10% split carved "
            "from the outer-FIT rows only; outer validation is scored once, after training, so no "
            "outer target influences training duration or checkpoint selection. Caveat recorded: "
            "pytabkit monitors val_ACCURACY, not AUC -- fold-safe but a weaker proxy than the "
            "inner-ES AUC curve we now persist."
        ),
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
                print(f"  skip (already present): {e['exp_id']}")
                continue
            e.setdefault("date", "2026-10-04")
            f.write(json.dumps(e) + "\n")
            added += 1
            print(f"  appended: {e['exp_id']} [{e['verdict']}]")
    print(f"\n{added} entries appended to {LEDGER.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
