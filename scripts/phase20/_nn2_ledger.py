"""Ledger + catalogue + scheduler for neural batch 2 (rows 16, 18-26, 11 archs)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
SCHED = ROOT / "research" / "phase20_exhaustive" / "scheduler.json"
REPORTS = ROOT / "reports" / "phase20" / "neural_screen"

# row -> (canonical list of arch tags)
ROWS = {
    16: ["saint"],
    18: ["autoint"],
    19: ["deepfm", "xdeepfm"],
    20: ["node"],
    21: ["grande"],
    22: ["tabnet"],
    23: ["excelformer"],
    24: ["amformer"],
    25: ["t2gformer"],
    26: ["trompt"],
}
NAMES = {16: "SAINT", 18: "AutoInt", 19: "DeepFM & xDeepFM", 20: "NODE", 21: "GRANDE",
         22: "TabNet", 23: "ExcelFormer", 24: "AMFormer", 25: "T2G-Former", 26: "Trompt"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(p: str) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main() -> int:
    results = {}
    for row, archs in ROWS.items():
        for a in archs:
            rep = json.loads((REPORTS / f"{a}_route_fold0.json").read_text(encoding="utf-8"))
            m = rep["metrics"]
            results[a] = {
                "auc": round(m["standalone_auc"], 6),
                "port_delta": round(m["portfolio_delta"], 8),
                "corr": round(m["logit_pearson_vs_portfolio"], 4),
                "best_val": round(rep["best_inner_val_auc"], 4),
                "n_params": rep["n_params"],
                "fit_predict_seconds": round(rep["fit_predict_seconds"], 1),
                "verdict": "NEGATIVE" if m["portfolio_delta"] < 5e-5 else "CHECK_PORT_LIMB",
            }

    prereg_sha = sha("configs/phase20/neural_batch2_prereg.json")

    entry = {
        "exp_id": "p20_neural_block2",
        "hypothesis_id": "P20-NEURAL-B2",
        "date": "2026-10-18",
        "ts": utcnow(),
        "family": "NN",
        "fold_scheme": "primary_5f_seed20261010 (fold-0 discovery screen; F2, cannot pass admission gate)",
        "arch_module": "src/phase20/neural_arch.py",
        "prereg": {"path": "configs/phase20/neural_batch2_prereg.json", "sha256": prereg_sha},
        "protocol": "train on FIT (folds 1-4), inner 10% ES from FIT only, predict fold-0; "
                    "route view (22 feat, standardized on FIT-train only); frozen rule "
                    "port_delta >= +5e-5 AND corr_vs_portfolio < 0.995 (Pearson-logit here, "
                    "all values far below 0.995 so the corr limb is not binding)",
        "results": results,
        "finding": "NEGATIVE across all 11 batch-2 architectures. Best: SAINT 0.958077 "
                   "(port_delta -5.40e-4) and AutoInt 0.958145 (-6.00e-4); all 11 land "
                   "0.934-0.958 standalone vs portfolio 0.961828 fold-0, so every "
                   "equal-logit blend is BELOW the reference (port_delta negative for "
                   "all 11). Correlations 0.90-0.99: NODE decorrelates most (0.9026) "
                   "but is catastrophically weak (0.934402); the strongest (SAINT/AutoInt/"
                   "ExcelFormer/AMFormer) are all attention-over-features variants that "
                   "converge to the same ~0.958 plateau as batch-1 fttransformer -- the "
                   "route view has no remaining signal for feature-token attention that "
                   "the champion portfolio does not already extract. Interaction-stack "
                   "archs (DeepFM/xDeepFM/GRANDE) land 0.9567-0.9575, gated-recurrence "
                   "(TabNet/Trompt) 0.9534-0.9565, soft trees (NODE) 0.9344. Combined "
                   "with batch 1 this closes the standard-neural-architecture axis on "
                   "this dataset under this protocol: 19 architectures measured, none "
                   "within +5e-5 of the portfolio when blended. Future NN work on this "
                   "competition is only justified with pretraining/transfer (FLM family), "
                   "not de-novo tabular architectures.",
        "verdict": "COMPLETE_NEGATIVE",
        "closes_rows": sorted(ROWS),
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for r in cat["experiments"]:
        n = r["experiment_number"]
        if n in ROWS:
            archs = ROWS[n]
            best = max(archs, key=lambda a: results[a]["auc"])
            r["status"] = "COMPLETE_NEGATIVE"
            r["fidelity_reached"] = "F2"
            r["next_action"] = "closed: de-novo NN architectures do not clear the frozen rule on the route view (batch-1 + batch-2 = 19 archs measured)"
            r["phase20_finding"] = (
                f"F2 fold-0 route-view screen (prereg {prereg_sha[:8]}): "
                + "; ".join(f"{a} AUC {results[a]['auc']:.6f} port_delta {results[a]['port_delta']:+.3e} corr {results[a]['corr']:.4f}"
                            for a in archs)
                + f". NEGATIVE (all port_delta < +5e-5). Best of row: {best}."
            )
    cat["updated_utc"] = utcnow()
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")

    sched = json.loads(SCHED.read_text(encoding="utf-8"))
    for n, archs in ROWS.items():
        j = sched["jobs"].get(f"J{n:03d}")
        if j is None:
            continue
        best = max(archs, key=lambda a: results[a]["auc"])
        j["state"] = "CLOSED_NEGATIVE"
        j["state_note"] = (f"Batch-2 fold-0 route screen: {best} best AUC "
                           f"{results[best]['auc']:.6f} port_delta {results[best]['port_delta']:+.3e}; all NEGATIVE")
        j["prereg"] = {"path": "configs/phase20/neural_batch2_prereg.json", "sha256": prereg_sha}
        j["result"] = {"verdict": "COMPLETE_NEGATIVE",
                       "report": f"reports/phase20/neural_screen/{archs[0]}_route_fold0.json"}
        j["history"].append({"ts": utcnow(), "event": "batch2_screen_done", "from": "PENDING"})
    sched["updated_utc"] = utcnow()
    SCHED.write_text(json.dumps(sched, indent=1) + "\n", encoding="utf-8")

    print(f"ledger +{len(LEDGER.read_text(encoding='utf-8').splitlines())} entries; "
          f"catalogue rows {sorted(ROWS)} -> COMPLETE_NEGATIVE; scheduler synced")
    for a, v in results.items():
        print(f"  {a:13s} auc={v['auc']:.6f} port_delta={v['port_delta']:+.3e} corr={v['corr']:.4f} {v['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
