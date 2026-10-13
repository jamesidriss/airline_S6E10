"""H18-A stage S3 — assemble the 90%-coverage route OOF and apply the gate.

Requires all ten S3 context reports + prediction arrays (scripts/phase18/
run_h18a_s3_context.py), hash-verified. The candidate portfolio is the exact
champion composition with only the route replaced:
    portfolio90 = 0.5 * route90 + 0.5 * aux10   (logit-space, float32 at the end)
where aux10 logits are recomputed per PRIMARY fold via the certified classical()
path (the exact same component v7 used). Champion v7 OOF is reassembled from
the preserved per-fold portfolio.npy files and hash-checked against the
certified 1981745c aggregate.

Frozen admission gate vs champion v7 on primary folds:
    mean paired portfolio gain >= 1.5e-5, >= 2.5 x paired SE, positive >= 4/5.
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
from src.validation.compare import logit  # noqa: E402
from scripts.phase16_common import bank  # noqa: E402
from scripts.phase16_pair_metrics import pair_rescue_damage  # noqa: E402
from scripts.score_sol_foundation import correlations, paired_bootstrap  # noqa: E402
from scripts.replay_sol_classical import frozen_roles  # noqa: E402
from scripts.evaluate_sol_foundation import classical  # noqa: E402

MEAN_MIN, SE_MULT, POS_FOLDS = 1.5e-5, 2.5, 4
CHAMPION_PORTFOLIO_SHA256 = "1981745c3d4f5b743c3f02c23f92a1e8975692ca473bc13e3156165bda1a7552"


def main() -> int:
    out_path = ROOT / "reports" / "phase18_h18a_s3_primary.json"
    if out_path.exists():
        raise FileExistsError("Preserve measured result")

    tr, _, y, folds, _, _, proof = bank()
    ids = tr.id.to_numpy()
    n = len(ids)
    route90 = np.full(n, np.nan, dtype="float32")
    champion = np.full(n, np.nan, dtype="float32")
    aux_logits_full = np.full(n, np.nan, dtype="float64")

    # exact v7 auxiliary component, primary-fold granularity (certified classical path)
    roles = [x for x in frozen_roles() if x["aux_arm"]]
    assert len(roles) == 10
    aux_proofs = None
    for k in range(5):
        fl = np.flatnonzero(folds == k)
        al, aux_proofs = classical(k, "sol_clean_aux10_primary", roles, ids, folds, y,
                                   proof["data_sha256"])
        aux_logits_full[fl] = al
        champ = np.load(ARTIFACTS / "phase17_B1" / f"f{k}" / "portfolio.npy")
        b1 = json.loads((ROOT / "reports" / f"phase17_B1_f{k}.json").read_text(encoding="utf-8"))
        route2 = np.load(ARTIFACTS / "phase17_B1" / f"f{k}" / "prediction.npy")
        assert arr_sha256(route2) == b1["prediction_sha256"]
        recomposed = (1 / (1 + np.exp(-(logit(route2) + al) / 2))).astype("float32")
        assert np.array_equal(recomposed, champ), f"champion recomposition drift at fold {k}"
        champion[fl] = champ
    assert np.isfinite(aux_logits_full).all() and np.isfinite(champion).all()
    assert arr_sha256(champion) == CHAMPION_PORTFOLIO_SHA256, "Champion OOF hash drift"

    # candidate route90 OOF from the ten verified contexts
    stage = ARTIFACTS / "phase18"
    context_reports = []
    for k in range(10):
        rep = json.loads((ROOT / "reports" / f"h18a_s3_f{k}.json").read_text(encoding="utf-8"))
        assert rep["hypothesis_id"] == "H18-A" and rep["context_fold_block10"] == k
        assert rep["status"] in ("COMPLETE", "COMPLETE_REUSED_S1_TREATMENT")
        assert rep["context_rows"] == 629_671
        p = np.load(stage / f"h18a_s3_f{k}" / "prediction.npy")
        assert arr_sha256(p) == rep["prediction_sha256"]
        ev_ids = np.load(stage / f"h18a_s3_f{k}" / "eval_ids.npy")
        block10 = np.load(ROOT / "data/cache/folds/block10_10f_seed20261010.npy")
        ev = np.flatnonzero(block10 == k)
        assert np.array_equal(ev_ids, ids[ev]) and p.shape == (len(ev),)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        route90[ev] = p
        context_reports.append(rep)
    assert np.isfinite(route90).all(), "route90 OOF incomplete"

    portfolio90 = (1 / (1 + np.exp(-(logit(route90) + aux_logits_full) / 2))).astype("float32")
    assert not np.array_equal(portfolio90, champion)

    auc_p90 = float(roc_auc_score(y, portfolio90))
    auc_champ = float(roc_auc_score(y, champion))
    auc_route90 = float(roc_auc_score(y, route90))
    pooled_delta = auc_p90 - auc_champ

    # paired deltas on immutable primary folds
    deltas, fold_aucs, fold_route_aucs = [], [], []
    for k in range(5):
        fl = np.flatnonzero(folds == k)
        fa = float(roc_auc_score(y[fl], portfolio90[fl]))
        fold_aucs.append(fa)
        deltas.append(fa - float(roc_auc_score(y[fl], champion[fl])))
        fold_route_aucs.append(float(roc_auc_score(y[fl], route90[fl])))
    mean_delta = float(np.mean(deltas))
    se = float(np.std(deltas, ddof=1) / np.sqrt(5))
    positive = int(sum(d > 0 for d in deltas))
    gate = (mean_delta >= MEAN_MIN and mean_delta >= SE_MULT * se and positive >= POS_FOLDS)

    # informational conversion diagnostics (route-level change vs champion route)
    route2_full = np.full(n, np.nan, dtype="float32")
    for k in range(5):
        fl = np.flatnonzero(folds == k)
        route2_full[fl] = np.load(ARTIFACTS / "phase17_B1" / f"f{k}" / "prediction.npy")
    assert np.isfinite(route2_full).all()
    route_delta_pooled = auc_route90 - float(roc_auc_score(y, route2_full))

    corr = correlations(portfolio90, champion, y)
    pairs = pair_rescue_damage(y, portfolio90, champion)
    assert abs(pairs["net_auc_gain"] - pooled_delta) < 1e-12
    segments = {}
    for col in ("Class", "Type of Travel", "Customer Type", "Gender"):
        segments[col] = {}
        for value in sorted(tr[col].astype(str).unique()):
            mask = tr[col].astype(str).to_numpy() == value
            if len(np.unique(y[mask])) != 2:
                continue
            segments[col][value] = {
                "rows": int(mask.sum()),
                "auc": float(roc_auc_score(y[mask], portfolio90[mask])),
                "delta": float(roc_auc_score(y[mask], portfolio90[mask])
                                - roc_auc_score(y[mask], champion[mask])),
            }

    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis_id": "H18-A",
        "stage": "S3_primary_aggregate",
        "champion": "v7_sol_tabpfn_internal2_aux10",
        "coverage": "0.90 (ten block10 complement contexts, 629,671 rows each)",
        "bank": proof,
        "context_reports": {str(i): file_sha256(ROOT / "reports" / f"h18a_s3_f{i}.json")
                            for i in range(10)},
        "full_pooled_route90_auc": auc_route90,
        "full_pooled_route2_auc": float(roc_auc_score(y, route2_full)),
        "route_delta_pooled": route_delta_pooled,
        "full_pooled_oof_auc": auc_p90,
        "champion_pooled_oof_auc": auc_champ,
        "pooled_delta": pooled_delta,
        "paired_fold_deltas": deltas,
        "fold_portfolio_aucs": fold_aucs,
        "fold_route90_aucs": fold_route_aucs,
        "mean_paired_gain": mean_delta,
        "paired_se": se,
        "positive_folds": positive,
        "gate_thresholds": {"mean_min": MEAN_MIN, "se_multiple": SE_MULT,
                            "positive_folds_min": POS_FOLDS},
        "performance_admission": bool(gate),
        "conversion_observed": (pooled_delta / route_delta_pooled
                                if abs(route_delta_pooled) > 1e-9 else None),
        "correlation_vs_champion": corr,
        "pair_rescue_damage": pairs,
        "bootstrap": paired_bootstrap(y, {"candidate": portfolio90, "champion": champion},
                                      reference="champion"),
        "segments": segments,
        "worst_segment_delta": min(s["delta"] for v in segments.values() for s in v.values()),
        "route90_oof_sha256": arr_sha256(route90),
        "portfolio90_oof_sha256": arr_sha256(portfolio90),
        "champion_oof_sha256": arr_sha256(champion),
        "aux_proofs_last_fold": aux_proofs,
        "distance_to_stretch": 0.9621 - auc_p90,
        "stretch_reached": bool(auc_p90 >= 0.9621),
        "test_and_shadow_required": not bool(gate),
        "source_sha256": {q: file_sha256(q) for q in (
            "scripts/phase18/assemble_h18a_s3.py", "scripts/phase18/run_h18a_s3_context.py")},
        "verdict": ("H18A_S3_ADMISSION_GATE_PASS_SHADOW_AND_TEST_REQUIRED" if gate
                    else "H18A_S3_ADMISSION_GATE_FAIL_NO_PROMOTION"),
    }
    save_json(result, out_path)
    np.save(stage / "h18a_s3_route90_oof.npy", route90)
    np.save(stage / "h18a_s3_portfolio90_oof.npy", portfolio90)
    with (ROOT / "experiments" / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({
            "exp_id": "phase18_h18a_s3_primary", "kind": "PHASE18_H18A_S3_AGGREGATE",
            "ts": result["utc"], "report": str(out_path.relative_to(ROOT)),
            "report_sha256": file_sha256(out_path),
            "paired_fold_deltas": deltas, "mean_paired_gain": mean_delta,
            "paired_se": se, "positive_folds": positive,
            "correlation_with_champion": corr, "auc": auc_p90,
            "complete_folds": [0, 1, 2, 3, 4], "full_oof": True,
            "verdict": result["verdict"]}) + "\n")
    print(json.dumps({k: result[k] for k in ("full_pooled_oof_auc", "pooled_delta",
                                             "mean_paired_gain", "paired_se", "positive_folds",
                                             "conversion_observed", "performance_admission",
                                             "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
