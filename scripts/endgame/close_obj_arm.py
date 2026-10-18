"""Close one OBJ37-41 objective arm: ledger entry + catalogue + scheduler sync.

Reads the aggregator artifact (reports/endgame/obj37_41_results.json), appends an
append-only ledger entry for the arm, sets the catalogue row to its honest state,
and records the verdict on the scheduler job.  The frozen gate decides; nothing
here re-interprets the metrics.

Usage:
  ./.venv/Scripts/python.exe scripts/endgame/close_obj_arm.py --arm supcon
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.phase20 import scheduler as S  # noqa: E402

RESULTS = ROOT / "reports" / "endgame" / "obj37_41_results.json"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
LEDGER = ROOT / "experiments" / "ledger.jsonl"
ARROW = {"supcon": 37, "auc_surrogate": 38, "hard_mining": 39, "gce": 40, "ema": 41}
JOB = {37: "J037", 38: "J038", 39: "J039", 40: "J040", 41: "J041"}
PREREG = "configs/phase20/obj37_41_objectives_prereg.json"
PREREG_SHA = "94561fcea7ebb4315aac24a9928240ef6d464be9c13a4d35930be92c83a112b2"
QUESTIONS = {
    "supcon": "Can BCE + supervised contrastive loss improve ranking accuracy and complement the champion?",
    "auc_surrogate": "Can a differentiable pairwise AUC surrogate outperform BCE for ROC-AUC?",
    "hard_mining": "Can FIT-only hard-example mining correct useful ranking errors?",
    "gce": "Can a noise-robust GCE objective improve generalization to hard generator rows?",
    "ema": "Does weight EMA stabilize optimization and improve ranking robustness?",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARROW))
    args = ap.parse_args()
    arm = args.arm
    row = ARROW[arm]
    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    rec = res["arms"][arm]
    if rec.get("status") == "PENDING" or not rec.get("frozen_gate"):
        raise SystemExit(f"{arm}: no completed result to close")

    gate = rec["frozen_gate"]
    promising = bool(gate["promising"])
    verdict = "PROMISING_SCREEN" if promising else "COMPLETE_NEGATIVE"
    now = datetime.now(timezone.utc).isoformat()
    gpu_h = round(float(rec.get("total_seconds") or 0) / 3600.0, 3)

    entry = {
        "kind": "EXPERIMENT",
        "exp_id": f"p20_obj{row}_{arm}",
        "hypothesis_id": f"OBJ{row}",
        "date": "2026-10-19",
        "ts": now,
        "family": "OBJ",
        "experiment_number": row,
        "fold_scheme": "primary fold-0 (F2 screen; frozen gate vs route_ref proxy)",
        "question": QUESTIONS[arm],
        "prereg": {"path": PREREG, "sha256": PREREG_SHA},
        "results": {
            "standalone_auc": rec["standalone_auc_recomputed"],
            "auc_recompute_delta": rec["auc_delta_recompute"],
            "delta_vs_control": rec["delta_vs_control"],
            "true_champion_fold0_auc": res["baseline"]["true_champion_fold0_auc"],
            "delta_vs_v7_standalone": rec["delta_vs_v7_standalone"],
            "blend_delta_vs_v7": rec["blend_delta_vs_v7"],
            "spearman_vs_v7": rec["spearman_vs_v7"],
            "logit_pearson_vs_v7": rec["logit_pearson_vs_v7"],
            "frozen_gate": gate,
            "fit_rows": rec["fit_rows"], "eval_rows": rec["eval_rows"],
            "train_rows": rec["train_rows"], "val_rows": rec["val_rows"],
            "features": rec["features"], "arch_cfg": rec["arch_cfg"], "train_cfg": rec["train_cfg"],
            "arm_params": rec["arm_params"],
            "prediction_sha256": rec["prediction_sha256"],
            "prediction_sha256_matches_report": rec["prediction_sha256_matches_report"],
            "total_seconds": rec["total_seconds"], "peak_vram_gb": rec["peak_vram_gb"],
            "artifact": f"reports/phase20/obj37_41_objectives/{arm}_fold0.json",
        },
        "finding": (
            f"{arm}: standalone {rec['standalone_auc_recomputed']:.6f} "
            f"(delta vs BCE control {rec['delta_vs_control']:+.2e}); true v7 fold-0 "
            f"{res['baseline']['true_champion_fold0_auc']:.6f} -> standalone delta "
            f"{rec['delta_vs_v7_standalone']:+.2e}, equal-logit blend delta vs v7 "
            f"{rec['blend_delta_vs_v7']:+.2e}; frozen gate port_delta {gate['port_delta']:+.2e} "
            f"(need >= +5e-5), spearman vs route_ref {gate['spearman_vs_route_ref']:.4f} "
            f"(need < 0.995) -> {'PASS' if promising else 'FAIL'}. "
            + ("Screen passed: promotion to 5-fold OOF is justified under the frozen protocol."
               if promising else
               "Weak-family ceiling stands: the objective cannot lift the fold-0 signal past the frozen screen; terminal.")
        ),
        "verdict": verdict,
        "closes_rows": [] if promising else [row],
        "gpu_hours": gpu_h,
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for r in cat["experiments"]:
        if r["experiment_number"] == row:
            r["fidelity_reached"] = "F2"
            if promising:
                r["status"] = "PROMISING"
                r["phase20_finding"] = entry["finding"]
                r["next_action"] = "promote to 5-fold primary OOF (frozen gate passed)"
            else:
                r["status"] = "COMPLETE_NEGATIVE"
                r["blocking_reason"] = "-"
                r["phase20_finding"] = entry["finding"]
                r["next_action"] = "closed: objective variant negative on the frozen fold-0 screen"
    cat["updated_utc"] = now
    CAT.write_text(json.dumps(cat, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    doc = S.load()
    job = doc["jobs"][JOB[row]]
    job["result"]["verdict"] = verdict
    job["result"]["report"] = f"reports/phase20/obj37_41_objectives/{arm}_fold0.json"
    if not promising:
        S.save(doc)
        S.cmd_block(type("A", (), {"job": JOB[row], "state": "CLOSED_NEGATIVE",
                                   "why": f"frozen gate FAIL: port_delta {gate['port_delta']:+.2e} "
                                          f"(need +5e-5); standalone {rec['standalone_auc_recomputed']:.6f}"})())
    else:
        job["history"].append({"ts": now, "event": "gate-pass", "detail": entry["finding"][:200]})
        S.save(doc)

    print(f"{arm} (row {row}): {verdict}")
    print(f"  standalone {rec['standalone_auc_recomputed']:.6f} | d_ctrl {rec['delta_vs_control']:+.2e} "
          f"| d_v7 {rec['delta_vs_v7_standalone']:+.2e} | blend_d_v7 {rec['blend_delta_vs_v7']:+.2e}")
    print(f"  gate port_delta {gate['port_delta']:+.2e} spearman {gate['spearman_vs_route_ref']:.4f} "
          f"-> promising={promising} | {rec['total_seconds']:.0f}s ({gpu_h}h) vram {rec['peak_vram_gb']}GB")
    print(f"  sha {rec['prediction_sha256'][:16]} verified={rec['prediction_sha256_matches_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())