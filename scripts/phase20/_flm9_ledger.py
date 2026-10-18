"""Ledger + catalogue + scheduler for FLM9 (Xiaomi-TabLDM screen, both arms)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
SCHED = ROOT / "research" / "phase20_exhaustive" / "scheduler.json"
REP_DIR = ROOT / "reports" / "phase20" / "flm9_tabldm"
PREREG = "configs/phase20/flm9_tabldm_prereg.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(p: str) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main() -> int:
    arms = {}
    for arm in ("A_ctx20k", "B_ctx50k"):
        rep = json.loads((REP_DIR / f"tabldm_{arm}_fold0.json").read_text(encoding="utf-8"))
        fr = rep["frozen_rule"]
        arms[arm] = {
            "context_rows": rep["context_rows"],
            "standalone_auc": round(rep["metrics"]["standalone_auc"], 6),
            "route_ref_auc": round(fr["route_ref_auc"], 6),
            "blend_auc": round(fr["blend_auc"], 6),
            "port_delta": round(fr["port_delta"], 8),
            "spearman_vs_route_ref": round(fr["spearman_vs_route_ref"], 4),
            "pearson_vs_portfolio": round(rep["metrics"]["logit_pearson_vs_portfolio"], 4),
            "promising": fr["promising"],
            "fit_s": round(rep["fit_seconds"], 1),
            "predict_s": round(rep["predict_seconds"], 1),
            "peak_vram_gb": rep["peak_vram_gb"],
            "guard": rep.get("resource_guard_note", "library defaults"),
        }

    entry = {
        "exp_id": "p20_flm9_xiaomi_tabldm",
        "hypothesis_id": "H9-XIAOMI-TABLDM",
        "date": "2026-10-18",
        "ts": utcnow(),
        "family": "FLM",
        "experiment_number": 9,
        "fold_scheme": "primary fold-0 (F2 screen, frozen rule vs route_ref)",
        "prereg": {"path": PREREG, "sha256": sha(PREREG)},
        "model_card": {
            "project": "github.com/xiaomi-research/xiaomi-tabldm (official, pip 0.1.0)",
            "hf": "occams/Xiaomi-TabLDM rev 712e5635, clf_default.ckpt sha256 7f8ec8b9",
            "license": "LICENSE file = Apache-2.0 (Xiaomi 2026); README News bullet says "
                       "'Non-Commercial License' but links to the same Apache file; "
                       "contradiction recorded, LICENSE treated as operative",
            "pretraining": "synthetic SCMs; context coverage hundreds to tens of thousands "
                           "of rows; test-time ensemble scaling",
        },
        "results": arms,
        "library_workarounds": [
            "kv_cache=False (conflicts with enhance_candidates=True default)",
            "enhance_candidates=False (gaussian-rank perturbation broadcast bug)",
            "device='cuda:0' (torch.cuda.set_device rejects indexless 'cuda')",
            "arm B: COL output buffer off pinned memory + async D2H staging pool disabled "
            "(both hardcode pinned buffers that OOM against a full GPU; disk offload "
            "forbidden by mission rules) + member_batch=1 (activation memory; members are "
            "independent forwards, only averaged) -- placement only, numerics unchanged",
        ],
        "finding": "NEGATIVE on both preregistered arms. Arm A (20k ctx): standalone "
                   "0.953351, blend 0.958932, port_delta -2.795e-3, Spearman 0.8858. "
                   "Arm B (50k ctx): standalone 0.956457, blend 0.960742, port_delta "
                   "-9.853e-4, Spearman 0.9160. Both arms fail the port limb by "
                   "0.9e-3 to 2.8e-3. Context scaling is real (+3.1e-3 AUC from 2.5x "
                   "context) and the extrapolated trend is monotone, but reaching the "
                   "portfolio would require O(1e5+) context rows while the architecture "
                   "is O(n^2): full-FIT (560k) context is infeasible on 16GB and beyond "
                   "the checkpoint's documented pretraining coverage (hundreds to tens "
                   "of thousands), so the gap cannot be closed under this protocol. "
                   "Spearman 0.89-0.92 is the most genuine decorrelation measured this "
                   "phase (vs NODE 0.9026 at far weaker AUC) -- the SCM-synthetic "
                   "pretraining does produce a differently-opinionated model, but its "
                   "route-view standalone skill (0.953-0.956) is 5e-3 to 8e-3 below the "
                   "portfolio, unusable as an 11th member. Verdict: Xiaomi-TabLDM closed "
                   "for this competition; large-ICL context scaling cannot be exploited "
                   "at this dataset size without disk offload (forbidden).",
        "verdict": "COMPLETE_NEGATIVE",
        "closes_rows": [9],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for r in cat["experiments"]:
        if r["experiment_number"] == 9:
            r["status"] = "COMPLETE_NEGATIVE"
            r["fidelity_reached"] = "F2"
            r["next_action"] = ("closed: both arms NEGATIVE; context scaling real but "
                                "O(n^2) architecture cannot reach portfolio-strength "
                                "context on 16GB without disk offload")
            r["phase20_finding"] = (
                "F2 fold-0 (Apache-2.0, official repo, rev-pinned ckpt): arm A ctx20k "
                "AUC 0.953351 / port_delta -2.795e-3 / Spearman 0.8858; arm B ctx50k "
                "AUC 0.956457 / -9.853e-4 / 0.9160. Both NEGATIVE; most genuine "
                "decorrelation of the phase but 5e-3..8e-3 too weak standalone; "
                "scaling trend cannot be extrapolated (O(n^2), full-FIT infeasible)."
            )
    cat["updated_utc"] = utcnow()
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")

    sched = json.loads(SCHED.read_text(encoding="utf-8"))
    j = sched["jobs"]["J009"]
    j["state"] = "CLOSED_NEGATIVE"
    j["state_note"] = ("F2 both arms: best ctx50k AUC 0.956457 port_delta -9.853e-4; "
                       "genuine decorrelation (Spearman 0.92) but unusable standalone; "
                       "O(n^2) context scaling infeasible at this data size")
    j["prereg"] = {"path": PREREG, "sha256": sha(PREREG)}
    j["result"] = {"verdict": "COMPLETE_NEGATIVE",
                   "report": "reports/phase20/flm9_tabldm/tabldm_B_ctx50k_fold0.json"}
    j["history"].append({"ts": utcnow(), "event": "f2_screen_done", "from": "NOT_STARTED"})
    sched["updated_utc"] = utcnow()
    SCHED.write_text(json.dumps(sched, indent=1) + "\n", encoding="utf-8")

    print(f"ledger +1 -> {len(LEDGER.read_text(encoding='utf-8').splitlines())} entries; "
          f"row 9 -> COMPLETE_NEGATIVE; J009 -> CLOSED_NEGATIVE")
    for a, v in arms.items():
        print(f"  {a}: auc={v['standalone_auc']} port_delta={v['port_delta']:+.3e} "
              f"sp={v['spearman_vs_route_ref']} promising={v['promising']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
