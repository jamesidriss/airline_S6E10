"""H18-E — per-fold evaluation of the 4-estimator route against champion v7.

Fold 0 reuses the certified phase17_B2 fold-0 artifacts (bit-identical rerun is
guaranteed by frozen params/seeds, so no GPU is wasted). Folds 1-4 read
artifacts/phase18_h18e/f{k} produced by scripts/run_phase17_tabpfn.py
(--estimators 4 --tag phase18_h18e), whose frozen gates are satisfied by the
evaluation files this script writes.

Semantics differ from evaluate_phase17_fold.py in ONE respect: the comparator
is champion v7 (portfolio = 0.5*internal2 route + 0.5*aux10, OOF AUC
0.9618529228157235, hashes in reports/phase17_B1_primary.json), not v6.
The auxiliary component and composition are recomputed and array_equal-verified
against the certified per-fold champion portfolio.npy files.
"""
from __future__ import annotations

import argparse
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

DISCOVERY_PORTFOLIO_GAIN_MIN = 3e-5  # phase17 scope; continuation bar only


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, choices=range(5), required=True)
    ap.add_argument("--source-tag", default="phase18_h18e",
                    help="tag of the fold-k 4-estimator run (fold 0 always uses phase17_B2)")
    args = ap.parse_args()
    fold = args.fold
    path = ROOT / "reports" / f"phase18_h18e_f{fold}_evaluation.json"
    if path.exists():
        raise FileExistsError("Preserve measured result")

    if fold == 0:
        run_report_path = ROOT / "reports" / "phase17_B2_f0.json"
        root = ARTIFACTS / "phase17_B2" / "f0"
        provenance = ("fold0 evidence = certified phase17_B2 fold-0 run "
                      "(4-estimator primary-context route); staged, not re-run")
    else:
        run_report_path = ROOT / "reports" / f"{args.source_tag}_f{fold}.json"
        root = ARTIFACTS / args.source_tag / f"f{fold}"
        provenance = f"fold{fold} evidence = {args.source_tag} run via run_phase17_tabpfn.py"
    r = json.loads(run_report_path.read_text(encoding="utf-8"))
    c = r["contract"]
    assert r["status"] == "COMPLETE_FROZEN_PRIMARY_FOLD"
    assert c["fold"] == fold and c["scheme"] == "primary" and c["seed"] == 1201
    assert c["params"]["n_estimators"] == 4 and c["params"]["random_state"] == 1201
    assert c["params"]["memory_saving_mode"] is True
    assert not c["outer_labels_used_for_fit_or_configuration"]

    tr, _, y, folds, vectors, _, proof = bank()
    ids = tr.id.to_numpy()
    va = np.flatnonzero(folds == fold)
    assert c["apply_rows"] == len(va) and c["apply_ids_sha256"] == arr_sha256(ids[va])

    route4 = np.load(root / "prediction.npy")
    apply_ids = np.load(root / "apply_ids.npy")
    assert np.array_equal(apply_ids, ids[va])
    assert arr_sha256(route4) == r["prediction_sha256"]
    assert route4.shape == (len(va),) and np.isfinite(route4).all()
    assert ((route4 >= 0) & (route4 <= 1)).all()

    # identical auxiliary component as champion; recomputed, not trusted
    roles = [x for x in frozen_roles() if x["aux_arm"]]
    assert len(roles) == 10
    aux_logits, aux_proofs = classical(fold, "sol_clean_aux10_primary", roles, ids, folds, y,
                                       proof["data_sha256"])
    assert np.array_equal((1 / (1 + np.exp(-aux_logits))).astype("float32"),
                          vectors["aux10"][va]), "Auxiliary10 component drift"

    # champion v7 fold portfolio: recompose from the certified internal2 route
    # and assert array_equal against the preserved per-fold portfolio.npy
    champion_root = ARTIFACTS / "phase17_B1" / f"f{fold}"
    route2 = np.load(champion_root / "prediction.npy")
    b1_report = json.loads((ROOT / "reports" / f"phase17_B1_f{fold}.json").read_text(encoding="utf-8"))
    assert arr_sha256(route2) == b1_report["prediction_sha256"]
    compose = lambda a: (1 / (1 + np.exp(-(logit(a) + aux_logits) / 2))).astype("float32")
    champion = np.load(champion_root / "portfolio.npy")
    assert np.array_equal(compose(route2), champion), "Champion recomposition drift"

    p = compose(route4)
    labels = y[va]
    route4_auc = float(roc_auc_score(labels, route4))
    route2_auc = float(roc_auc_score(labels, route2))
    portfolio_auc = float(roc_auc_score(labels, p))
    standalone_delta = route4_auc - route2_auc
    portfolio_delta = portfolio_auc - float(roc_auc_score(labels, champion))
    assert abs(route4_auc - r["auc"]) < 1e-14
    corr = correlations(p, champion, labels)
    pairs = pair_rescue_damage(labels, p, champion)
    assert abs(pairs["net_auc_gain"] - portfolio_delta) < 1e-14

    segments = {}
    for col in ("Class", "Type of Travel", "Customer Type", "Gender"):
        segments[col] = {}
        for value in sorted(tr[col].astype(str).unique()):
            mask = tr.iloc[va][col].astype(str).to_numpy() == value
            if len(np.unique(labels[mask])) != 2:
                continue
            segments[col][value] = {
                "rows": int(mask.sum()),
                "auc": float(roc_auc_score(labels[mask], p[mask])),
                "delta": float(roc_auc_score(labels[mask], p[mask])
                                - roc_auc_score(labels[mask], champion[mask])),
            }

    performance = portfolio_delta >= DISCOVERY_PORTFOLIO_GAIN_MIN
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis_id": "H18-E",
        "fold": fold,
        "provenance": provenance,
        "source_report_sha256": file_sha256(run_report_path),
        "bank": proof,
        "champion": "v7_sol_tabpfn_internal2_aux10",
        "champion_fold_portfolio_sha256": arr_sha256(champion),
        "champion_fold_route_sha256": arr_sha256(route2),
        "route4_auc": route4_auc,
        "route2_auc": route2_auc,
        "standalone_delta": standalone_delta,
        "portfolio_auc": portfolio_auc,
        "portfolio_delta": portfolio_delta,
        "oof_correlation_vs_champion": corr,
        "exact_pair_rescue_damage_vs_champion": pairs,
        "bootstrap": paired_bootstrap(labels, {"candidate": p, "champion": champion},
                                      reference="champion"),
        "segments": segments,
        "worst_segment_delta": min(s["delta"] for v in segments.values() for s in v.values()),
        "performance_discovery_pass": bool(performance),
        "discovery_pass": bool(performance),
        "hedge_discovery_pass": False,
        "replication_pass": bool(portfolio_delta > 0),
        "portfolio_prediction_sha256": arr_sha256(p),
        "exact_auxiliary_logits_sha256": arr_sha256(aux_logits),
        "unchanged_auxiliary_proofs": aux_proofs,
        "runtime_seconds": r.get("seconds"),
        "source_sha256": {q: file_sha256(q) for q in (
            "scripts/phase18/evaluate_h18e_fold.py", "scripts/phase16_pair_metrics.py")},
        "verdict": ("DISCOVERY_PASS_CONTINUATION_GATE" if performance
                    else "DISCOVERY_BELOW_3E5_CONTINUE_ONLY_IF_EARLIER_GATES_ALLOW"),
    }
    stage_dir = ARTIFACTS / "phase18_h18e"
    stage_dir.mkdir(parents=True, exist_ok=True)
    np.save(stage_dir / f"portfolio_f{fold}.npy", p)
    save_json(result, path)
    with (ROOT / "experiments" / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({
            "exp_id": f"phase18_h18e_f{fold}", "kind": "PHASE18_H18E_FOLD",
            "ts": result["utc"], "report": str(path.relative_to(ROOT)),
            "report_sha256": file_sha256(path),
            "paired_fold_deltas": [portfolio_delta],
            "correlation_with_champion": corr, "auc": portfolio_auc,
            "complete_folds": [fold], "full_oof": False,
            "verdict": result["verdict"]}) + "\n")
    print(json.dumps({k: result[k] for k in ("fold", "route4_auc", "standalone_delta",
                                             "portfolio_auc", "portfolio_delta",
                                             "discovery_pass", "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
