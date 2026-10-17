"""One-shot record keeper for the H20-TABICL-FT fb100k NEGATIVE result.

Not a test, not a runner: pure bookkeeping (catalogue row 31, ledger append,
campaign state). Kept as a file so the exact mutation is auditable and
re-runnable-safe (idempotent on the ledger entry id).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import git_commit  # noqa: E402

now = datetime.now(timezone.utc).isoformat()
git = git_commit()

# ---- catalogue row 31 ----
p = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
cat = json.loads(p.read_text(encoding="utf-8"))
rows = cat["experiments"] if "experiments" in cat else cat
r = [x for x in rows if x["experiment_number"] == 31][0]
if r["status"] == "IN_PROGRESS":
    r["status"] = "COMPLETE_NEGATIVE"
r["fidelity_reached"] = "F2"
r["blocking_reason"] = None
r["next_action"] = (
    "closed: reduced-dose FT NEGATIVE (standalone 0.957404 < 0.958343 bar; port_delta -8.17e-4; "
    "fine-tune gained +1.06e-3 strength but DECORRELATION COLLAPSED corr_route 0.9843->0.9881). "
    "Full-dose arm remains RESOURCE_BLOCKED (GPU driver crash under sustained load + "
    "~7.5 GPU-h/screen); reopening needs a driver-stable full-10-epoch run under an extended "
    "compute plan, not a new recipe")
r["finding"] = (
    "H20-TABICL-FT (items 31+13) NEGATIVE. Fine-tuned TabICL-v2 on route view: standalone "
    "fold-0 0.957404 (+1.061e-3 vs zero-shot 0.956343) but port_delta -8.169e-4 (champion "
    "0.961828 fold-0 portfolio) and corr_vs_route 0.9881 vs zero-shot 0.9843 -- fine-tuning "
    "moved the model TOWARD the champion while staying weaker: both strength AND decorrelation "
    "fail. Confirms the structural frontier independently for the in-context family. Both "
    "fine-tune processes were killed by GPU driver crashes (nvlddmkm 153), NOT allocator "
    "fragmentation; recovered via epoch-8 best checkpoint (preregistered recovery, no foreign "
    "weight update). Reduced-dose arm closed.")
p.write_text(json.dumps(cat, indent=1), encoding="utf-8")

# ---- ledger append (append-only) ----
entry = {
    "exp_id": "p20_tabicl_ft_fb100k_result",
    "hypothesis_id": "H20-TABICL-FT-FALLBACK",
    "date": now[:10], "ts": now, "git": git,
    "family": "OBJ", "experiment_number": 31, "fold_scheme": "primary fold-0 screen",
    "prereg": {"path": "configs/phase20/tabicl_ft_recovery_prereg.json",
               "sha256": "a9383ab92ae9997b7b6b33520bd46daf670fb921eb50b12dbe95e02c30a4c393",
               "fallback_prereg": "configs/phase20/tabicl_ft_fallback_prereg.json",
               "parent_prereg": "configs/phase20/tabicl_ft_prereg.json"},
    "view": "route (champion raw22, label-free)",
    "arch": "tabicl-v2-20260212 fine-tuned (AdamW, lr 1e-5, 8/10 epochs, recovered)",
    "results": {"standalone_auc": 0.957404, "portfolio_delta": -8.169e-4,
                "delta_vs_zeroshot": 1.061e-3, "corr_vs_portfolio": 0.9892,
                "corr_vs_route": 0.9881, "zeroshot_reference": 0.956343,
                "champion_fold0_portfolio": 0.961828,
                "inner_val_best": 0.9579433793073343, "inner_val_baseline": 0.9578,
                "prediction_sha256": "f3c97c47627903d2..."},
    "finding": (
        "NEGATIVE on the frozen rule (standalone 0.957404 < zero-shot+2e-3=0.958343). "
        "Fine-tuning added strength (+1.061e-3 over zero-shot) but destroyed decorrelation: "
        "corr_vs_route rose 0.9843->0.9881 and port_delta stayed strongly negative (-8.17e-4). "
        "The in-context family reproduces the structural frontier found across 29 experiments: "
        "every gain in strength pulls correlation with the boosted-tree champion up. "
        "CAUSE CORRECTION: both fine-tune processes were killed by GPU driver crashes "
        "(nvlddmkm Event 153 at 15:07:04 and 17:07:32; LiveKernelEvent 141/144/19e at "
        "14:42:45), NOT allocator fragmentation -- that earlier hypothesis is WITHDRAWN as "
        "unestablished. Recovered by preregistered checkpoint reload (epoch-8 best, library "
        "model_path path, no foreign weight update); deviation: 8/10 epochs. Full-dose arm "
        "remains RESOURCE_BLOCKED (driver instability under sustained load; ~7.5 GPU-h/screen)."),
    "verdict": "NEGATIVE",
    "negative_rule": (
        "reduced-dose FT closed: standalone < zero-shot+2e-3 AND decorrelation collapsed; "
        "reopening requires a driver-stable full-10-epoch run under an extended compute plan"),
    "closes_rows": [31],
    "budget": {"gpu_h": 0.3, "submissions": 0},
}
led = ROOT / "experiments" / "ledger.jsonl"
existing = led.read_text(encoding="utf-8").splitlines()
assert not any((json.loads(ln).get("exp_id") == entry["exp_id"]) for ln in existing if ln.strip()), \
    "ledger entry already present"
with open(led, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry) + "\n")

# ---- campaign state ----
cp = ROOT / "research" / "phase20_exhaustive" / "campaign_state.json"
cs = json.loads(cp.read_text(encoding="utf-8"))
cs["gpu_hours_used"] = round(float(cs.get("gpu_hours_used", 5.5)) + 0.3, 2)
cs["tabicl_ft"] = {
    "status": "CLOSED_NEGATIVE",
    "result": "reports/phase20/tabicl_ft/tabicl_ft_route_fold0_fb100k.json",
    "standalone": 0.957404, "port_delta": -8.169e-4, "corr_route": 0.9881,
    "cause_correction": "GPU driver crash (nvlddmkm 153); allocator-fragmentation hypothesis withdrawn",
    "full_dose": "RESOURCE_BLOCKED (driver instability + ~7.5 GPU-h/screen)",
    "recovery_prereg_sha256": "a9383ab92ae9997b7b6b33520bd46daf670fb921eb50b12dbe95e02c30a4c393",
}
cs["test_suite"] = "run_tests.py registers phase20 + 6 previously-unregistered legacy modules (223 tests)"
cp.write_text(json.dumps(cs, indent=1), encoding="utf-8")

n_led = len([ln for ln in existing if ln.strip()]) + 1
print(f"OK catalogue row31 -> COMPLETE_NEGATIVE | ledger lines now {n_led} | campaign_state updated")
