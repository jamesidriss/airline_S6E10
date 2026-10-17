"""H19-B recording: append the append-only ledger entry + fill the scoreboard row.

Reads the measured eval JSON (reports/phase19/h19b_bag5_f4.json). Never overwrites an
existing H19-B ledger entry; never fabricates numbers. Run AFTER h19b_bag5_eval.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import REPORTS, file_sha256, git_commit, save_json

LEDGER = Path("experiments/ledger.jsonl")
SCORE_JSON = REPORTS / "phase19" / "experiment_scoreboard.json"
EVAL_JSON = REPORTS / "phase19" / "h19b_bag5_f4.json"
EXP_ID = "p19_h19b_bag5_deployment"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    r = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    assert r["hypothesis_id"] == "H19-B"
    # Older ledger entries use other key schemas (id / experiment), so guard with .get().
    if any(json.loads(l).get("exp_id") == EXP_ID for l in LEDGER.read_text(encoding="utf-8").splitlines()
           if l.strip()) and not args.force:
        raise FileExistsError("H19-B ledger entry already present (append-only; use --force to re-record)")

    pf = r["pairs"]
    port = r["portfolio"]
    aux = r["aux"]
    decision = r["decision"]

    entry = {
        "exp_id": EXP_ID,
        "hypothesis_id": "H19-B",
        "date": time.strftime("%Y-%m-%d"),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "git": git_commit(),
        "family": "deployment_policy",
        "fold_scheme": "primary_5f_seed20261010 (pseudo-test=fold4, FIT=folds0-3)",
        "prereg": {"path": "configs/phase19/h19b_prereg.json",
                   "sha256": file_sha256(Path("configs/phase19/h19b_prereg.json")),
                   "commit": "7f2a427", "note": "committed BEFORE any treatment evaluation"},
        "complete_folds": [4],
        "full_oof": False,
        "standard": "PSEUDO_TEST (cannot pass OOF admission gate; finalist-selection evidence only)",
        "control_policy": "single full-FIT refit (certified aux10, tag sol_clean_aux10_primary), "
                          "bit-exact vs phase17_B1/f4/portfolio.npy (identity gap 0.0)",
        "treatment_policy": "bag-of-five inner-fold members per role, equal mean-logit "
                            "(StratifiedKFold(5,20261022) on FIT; each bag ~80% of FIT)",
        "fit_rows": r["rows"]["fit"], "pseudotest_rows": r["rows"]["pseudotest"],
        "aux_ctrl_auc": aux["ctrl_auc"], "aux_bag5_auc": aux["bag5_auc"], "aux_delta": aux["delta"],
        "port_ctrl_auc": port["ctrl_auc"], "port_bag5_auc": port["bag5_auc"],
        "port_delta": port["delta"], "port_paired_se": port["paired_se"],
        "port_delta_over_se": decision["sig_multiple"],
        "pairs": {k: pf[k] for k in ("pairs", "ctrl_right", "ctrl_tie", "ctrl_wrong",
                                     "treat_right", "treat_tie", "treat_wrong",
                                     "net_credit", "rescue_credit", "damage_credit",
                                     "identity_abs_error")},
        "correlations": r["correlations"],
        "prediction_scale": r["prediction_scale"],
        "best_segment_delta": r["segments"][0]["delta"] if r["segments"] else None,
        "worst_segment_delta": r["segments"][-1]["delta"] if r["segments"] else None,
        "tests": {"baseline": 159, "h19b_added": 7, "total_expected": 166},
        "report": "reports/phase19/h19b_bag5_f4.json",
        "report_sha256": file_sha256(EVAL_JSON),
        "budget": {"cpu_hours_estimate": 8, "gpu_hours": 0, "submissions_used": 0,
                   "note": "CPU-classical; light cuda XGB preserved unchanged"},
        "verdict": r["verdict"],
        "status": f"{r['verdict']}_PSEUDO_TEST_NOT_OOF_PROMOTION",
        "notes": ("Deployment-policy A/B on frozen fold-4 pseudo-test; control reuses certified "
                  "full-FIT members (no retraining), treatment trains 10x5 bag members "
                  "fold-4-safe. " + " ".join(r["limitations"])),
    }
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"ledger appended {EXP_ID} verdict={r['verdict']}")

    # scoreboard row
    sb = json.loads(SCORE_JSON.read_text(encoding="utf-8"))
    for e in sb["experiments"]:
        if e["id"] == "H19-B":
            e["experiment"] = ("fold-4 pseudo-test: certified single full-FIT aux10 vs bag-of-five "
                               "inner-fold mean-logit (10 roles x 5 members); control bit-exact vs "
                               "phase17_B1/f4; exact pair flips; frozen decision policy")
            e["result"] = {
                "aux_ctrl_auc_f4": aux["ctrl_auc"], "aux_bag5_auc_f4": aux["bag5_auc"],
                "aux_delta": aux["delta"], "port_ctrl_auc_f4": port["ctrl_auc"],
                "port_bag5_auc_f4": port["bag5_auc"], "port_delta": port["delta"],
                "port_paired_se": port["paired_se"], "delta_over_se": decision["sig_multiple"],
                "pair_net_credit": pf["net_credit"], "pair_identity_err": pf["identity_abs_error"],
                "spearman": r["correlations"]["spearman"],
                "logit_pearson": r["correlations"]["logit_pearson"],
                "mean_abs_dp": r["prediction_scale"]["mean_abs"],
            }
            e["verdict"] = (f"{r['verdict']} (port Δ {port['delta']:+.3e}, "
                            f"{decision['sig_multiple']:+.1f}×SE; pseudo-test standard, "
                            f"cannot pass OOF gate)")
            e["note"] = "ledger p19_h19b_bag5_deployment; prereg 7f2a427 before results; report reports/phase19/h19b_bag5_f4.json"
            break
    else:
        raise RuntimeError("H19-B row missing from scoreboard")
    sb["updated_utc"] = time.strftime("%Y-%m-%d")
    save_json(sb, SCORE_JSON)
    print("scoreboard H19-B row filled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
