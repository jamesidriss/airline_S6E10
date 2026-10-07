"""Append audit/diagnostic evidence and synchronize scored submission metadata."""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import REPORTS, file_sha256, git_commit, save_json


def main():
    audit = json.loads((REPORTS / "sol_s0_audit.json").read_text(encoding="utf-8"))
    sim = json.loads((REPORTS / "sol_private_sim.json").read_text(encoding="utf-8"))
    ledger = Path("experiments/ledger.jsonl")
    existing = [json.loads(s) for s in ledger.read_text(encoding="utf-8").splitlines() if s.strip()]
    ids = {d.get("id", d.get("exp_id")) for d in existing}
    now = datetime.now(timezone.utc).isoformat()
    events = [
        {"id": "sol_s0_independent_audit", "kind": "AUDIT", "verdict": "INVALID_TEST_POLICY_REPAIR_REQUIRED",
         "oof_auc": audit["v5_auc"], "paired_fold_deltas": audit["paired_fold_deltas"],
         "corr_with_champion": audit["oof_logit_corr"], "report_sha256": file_sha256(REPORTS / "sol_s0_audit.json"),
         "defects": audit["defects"], "finding": "v5 OOF arithmetic reproduces; submitted inference does not match the ten OOF configurations. Preserve banked artifacts; repair before promotion."},
        {"id": "sol_s0_admission_arithmetic_correction", "kind": "CORRECTION", "verdict": "OOF_GATE_PASS_TEST_CONTRACT_FAIL",
         "finding": "Admission rule uses mean paired fold gain, not pooled AUC gain. Exact mean +1.5354500604281006e-5 >=1.5e-5; SE 3.7590797e-6, t=4.084646, 5/5 positive. Historical claim that +1.491165e-5 misses the gate used the wrong statistic.",
         "paired_fold_deltas": audit["paired_fold_deltas"], "paired_t": audit["paired_t"]},
        {"id": "sol_s0_private_simulator", "kind": "DIAGNOSTIC", "verdict": "ROBUST_OOF_DIAGNOSTIC_NOT_FORECAST",
         "seed": sim["seed"], "population": sim["population"],
         "split_definitions_sha256": sim["split_definitions_sha256"],
         "v3_vs_v5": sim["candidates"]["v3_final"], "report_sha256": file_sha256(REPORTS / "sol_private_sim.json"),
         "finding": "820 pseudo 20/80 splits at actual test population size. v4 deliberately marked CV proxy. Existing v5 test defects prevent transferring OOF robustness directly to the scored CSV."},
    ]
    with ledger.open("a", encoding="utf-8") as f:
        for event in events:
            if event["id"] not in ids:
                f.write(json.dumps({"phase": "sol_s0", "ts": now, "git": git_commit(), **event}) + "\n")
    # Authenticated CLI read; only public-safe scored metadata is persisted.
    r = subprocess.run([sys.executable, "-m", "kaggle", "competitions", "submissions", "-c", "playground-series-s6e10", "--csv"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if r.returncode == 0:
        history = pd.read_csv("submissions/kaggle_submissions.csv")
        scored = []
        for row in csv.DictReader(io.StringIO(r.stdout)):
            mask = history["file"] == row["fileName"]
            if mask.any():
                history.loc[mask, "ref"] = float(row["ref"])
                history.loc[mask, "status"] = "complete"
                if row.get("publicScore"):
                    history.loc[mask, "publicScore"] = float(row["publicScore"])
            scored.append({k: row.get(k) for k in ("ref", "fileName", "date", "status", "publicScore")})
        history.to_csv("submissions/kaggle_submissions.csv", index=False)
        save_json({"accessed_utc": now, "submissions": scored}, REPORTS / "sol_submission_refresh.json")
    save_json({"updated_utc": now, "BEST_A": {"name": "v5_aux_cross", "ref": 56918343,
               "oof": audit["v5_auc"], "public": .96103, "status": "PROVISIONAL_TEST_REPAIR_REQUIRED"},
               "BEST_B_HEDGE": None, "fallback_B": {"name": "v4_fulldata", "ref": 56845975,
               "public": .961, "status": "NOT_A_TRUE_HEDGE; v3 OOF proxy only"},
               "final_selection_locked": False}, "experiments/private_finalists.json")
    print("S0 evidence recorded; scored submission history synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
