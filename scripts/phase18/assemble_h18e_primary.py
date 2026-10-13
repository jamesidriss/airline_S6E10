"""H18-E — five-fold aggregate and admission gate for the 4-estimator portfolio.

Requires all five per-fold evaluations (scripts/phase18/evaluate_h18e_fold.py).
Re-verifies every hash chain, pools the OOF vectors, and applies the frozen
Phase 18 admission gate against champion v7:
    mean paired portfolio gain >= 1.5e-5
    mean >= 2.5 x paired fold SE
    positive in >= 4 of 5 folds
No fitting, no weight selection, no public-LB input.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, arr_sha256, file_sha256, save_json  # noqa: E402
from scripts.phase16_common import bank  # noqa: E402
from scripts.phase16_pair_metrics import pair_rescue_damage  # noqa: E402
from scripts.score_sol_foundation import correlations, paired_bootstrap  # noqa: E402

MEAN_MIN, SE_MULT, POS_FOLDS = 1.5e-5, 2.5, 4


def main() -> int:
    out_path = ROOT / "reports" / "phase18_h18e_primary.json"
    if out_path.exists():
        raise FileExistsError("Preserve measured result")

    tr, _, y, folds, _, _, proof = bank()
    ids = tr.id.to_numpy()
    stage = ARTIFACTS / "phase18_h18e"
    evals, route4_oof, portfolio4_oof = {}, np.full(len(ids), np.nan, dtype="float32"), \
        np.full(len(ids), np.nan, dtype="float32")
    champion_oof = np.full(len(ids), np.nan, dtype="float32")
    deltas, fold_aucs = [], []
    for k in range(5):
        e = json.loads((ROOT / "reports" / f"phase18_h18e_f{k}_evaluation.json").read_text(encoding="utf-8"))
        assert e["hypothesis_id"] == "H18-E" and e["fold"] == k
        evals[k] = e
        m = np.flatnonzero(folds == k)
        p4 = np.load(stage / f"portfolio_f{k}.npy")
        assert arr_sha256(p4) == e["portfolio_prediction_sha256"]
        assert p4.shape == (len(m),) and np.isfinite(p4).all()
        portfolio4_oof[m] = p4
        champion = np.load(ARTIFACTS / "phase17_B1" / f"f{k}" / "portfolio.npy")
        assert arr_sha256(champion) == e["champion_fold_portfolio_sha256"]
        champion_oof[m] = champion
        run_root = (ARTIFACTS / "phase17_B2" / "f0") if k == 0 else (ARTIFACTS / "phase18_h18e" / f"f{k}")
        r4 = np.load(run_root / "prediction.npy")
        route4_oof[m] = r4
        deltas.append(e["portfolio_delta"])
        fold_aucs.append(e["portfolio_auc"])

    assert np.isfinite(portfolio4_oof).all() and np.isfinite(champion_oof).all()
    b1_primary = json.loads((ROOT / "reports" / "phase17_B1_primary.json").read_text(encoding="utf-8"))
    assert arr_sha256(champion_oof) == b1_primary["portfolio_sha256"] == \
        "1981745c3d4f5b743c3f02c23f92a1e8975692ca473bc13e3156165bda1a7552", "Champion OOF hash drift"
    assert not np.array_equal(portfolio4_oof, champion_oof)

    auc4 = float(roc_auc_score(y, portfolio4_oof))
    auc_champ = float(roc_auc_score(y, champion_oof))
    auc_route4 = float(roc_auc_score(y, route4_oof))
    pooled_delta = auc4 - auc_champ
    # pooled delta must equal the sum of fold-restricted deltas in expectation;
    # the exact per-fold deltas are the admission statistics
    mean_delta = float(np.mean(deltas))
    se = float(np.std(deltas, ddof=1) / np.sqrt(5))
    positive = int(sum(d > 0 for d in deltas))
    gate = (mean_delta >= MEAN_MIN and mean_delta >= SE_MULT * se and positive >= POS_FOLDS)
    corr = correlations(portfolio4_oof, champion_oof, y)
    pairs = pair_rescue_damage(y, portfolio4_oof, champion_oof)
    assert abs(pairs["net_auc_gain"] - pooled_delta) < 1e-12

    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis_id": "H18-E",
        "champion": "v7_sol_tabpfn_internal2_aux10",
        "bank": proof,
        "internal_estimators": 4,
        "folds": fold_aucs,
        "full_pooled_route4_auc": auc_route4,
        "full_pooled_oof_auc": auc4,
        "champion_pooled_oof_auc": auc_champ,
        "pooled_delta": pooled_delta,
        "paired_fold_deltas": deltas,
        "mean_paired_gain": mean_delta,
        "paired_se": se,
        "positive_folds": positive,
        "gate_thresholds": {"mean_min": MEAN_MIN, "se_multiple": SE_MULT, "positive_folds_min": POS_FOLDS},
        "performance_admission": bool(gate),
        "correlation_vs_champion": corr,
        "pair_rescue_damage": pairs,
        "bootstrap": paired_bootstrap(y, {"candidate": portfolio4_oof, "champion": champion_oof},
                                      reference="champion"),
        "route4_oof_sha256": arr_sha256(route4_oof),
        "portfolio4_oof_sha256": arr_sha256(portfolio4_oof),
        "champion_oof_sha256": arr_sha256(champion_oof),
        "distance_to_stretch": 0.9621 - auc4,
        "stretch_reached": bool(auc4 >= 0.9621),
        "per_fold_reports": {str(k): file_sha256(ROOT / "reports" / f"phase18_h18e_f{k}_evaluation.json")
                             for k in range(5)},
        "test_and_shadow_required": not bool(gate),
        "source_sha256": {q: file_sha256(q) for q in (
            "scripts/phase18/assemble_h18e_primary.py", "scripts/phase18/evaluate_h18e_fold.py")},
        "verdict": ("H18E_ADMISSION_GATE_PASS_SHADOW_AND_TEST_REQUIRED" if gate
                    else "H18E_ADMISSION_GATE_FAIL_NO_PROMOTION"),
    }
    save_json(result, out_path)
    np.save(stage / "portfolio4_primary_oof.npy", portfolio4_oof)
    np.save(stage / "route4_primary_oof.npy", route4_oof)
    with (ROOT / "experiments" / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({
            "exp_id": "phase18_h18e_primary", "kind": "PHASE18_H18E_AGGREGATE",
            "ts": result["utc"], "report": str(out_path.relative_to(ROOT)),
            "report_sha256": file_sha256(out_path),
            "paired_fold_deltas": deltas, "mean_paired_gain": mean_delta,
            "paired_se": se, "positive_folds": positive,
            "correlation_with_champion": corr, "auc": auc4,
            "complete_folds": [0, 1, 2, 3, 4], "full_oof": True,
            "verdict": result["verdict"]}) + "\n")
    print(json.dumps({k: result[k] for k in ("full_pooled_oof_auc", "pooled_delta",
                                             "mean_paired_gain", "paired_se", "positive_folds",
                                             "performance_admission", "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
