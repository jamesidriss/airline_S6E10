"""Append the S7 fidelity-repair result entries to the experiment ledger (append-only).

Called AFTER the measurements exist as artifacts; entries cite their JSON paths and
exact metrics. Prereg: configs/phase20/s7_fidelity_repair_prereg.json (sha256
c10cf3fec2efa1e35eb561009afcf72f717dbc0274490df46fb049145e79e5ed, commit ebd7bac).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
PREREG = "configs/phase20/s7_fidelity_repair_prereg.json"
SHA = "c10cf3fec2efa1e35eb561009afcf72f717dbc0274490df46fb049145e79e5ed"
GIT = "e9ffc82"
NOW = datetime.now(timezone.utc).isoformat()

ENTRIES = [
    {
        "exp_id": "p20_s7_dcnv2full",
        "hypothesis_id": "H20-NEURAL-DCNV2-CANONICAL",
        "date": "2026-10-17",
        "ts": NOW,
        "git": GIT,
        "family": "NN",
        "experiment_number": 17,
        "fold_scheme": "primary fold-0 discovery screen (F2, cannot pass admission gate)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "view": "route (22 feat) + full (285 feat)",
        "arch": "DCNv2Full: canonical cross layer x_{l+1}=x_0*(W_l x_l + b_l)+x_l with FULL "
                "W_l (285x285), deep MLP + concat head otherwise identical to the scalar proxy",
        "results": {
            "route": {"standalone_auc": 0.957458, "portfolio_delta": -0.0009818,
                      "corr": 0.9862, "artifact": "reports/phase20/neural_screen/dcnv2full_route_fold0.json"},
            "full": {"standalone_auc": 0.959041, "portfolio_delta": -0.0009779,
                     "corr": 0.9799, "artifact": "reports/phase20/neural_screen/dcnv2full_full_fold0.json"},
            "scalar_proxy_reference": {"route_auc": 0.957580, "route_pd": -0.0009073,
                                       "full_auc": 0.959898, "full_pd": -0.0005827},
        },
        "finding": "Canonical full-W DCNv2 measured directly (S7 item-17 repair: the prior "
                   "negative came from a scalar-cross variant with W = 1*w^T). Canonical is "
                   "if anything WEAKER than the scalar gate (full-view 0.959041 vs 0.959898, "
                   "port_delta -9.78e-4 vs -5.83e-4). Negative CONFIRMED for canonical DCNv2 "
                   "on both views; cross-network family closed under this protocol.",
        "verdict": "NEGATIVE_CANONICAL_CONFIRMED",
        "closes_rows": [17],
        "verdict_note": "F2 screen only -- cannot pass admission gate; closure is of the "
                        "fold-0 discovery hypothesis, not an admission decision.",
    },
    {
        "exp_id": "p20_s7_factlog",
        "hypothesis_id": "H20-CLASS-FACTLOG",
        "date": "2026-10-17",
        "ts": NOW,
        "git": GIT,
        "family": "NN",
        "experiment_number": 65,
        "fold_scheme": "primary fold-0 discovery screen (F2, cannot pass admission gate)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "view": "full (285 feat, fold-safe TE via ViewBuilder.assemble)",
        "arch": "FactLog: logit = w.x + sum_{r=1..16} (u_r.x)(v_r.x); AdamW lr 3e-3, wd 1e-4, "
                "bs 8192, max 60 epochs, inner-val ES patience 10",
        "results": {
            "standalone_auc": 0.959788, "portfolio_delta": -0.0006815, "corr": 0.9868,
            "best_inner_val_auc": 0.9594,
            "artifact": "reports/phase20/neural_screen/factlog_full_fold0.json",
        },
        "finding": "S7 item-65 repair: the row had been closed by an ADDITIVE logreg negative "
                   "(0.925819) -- a different model. Factorized (rank-16 bilinear) "
                   "interactions measured directly: AUC 0.959788, port_delta -6.815e-4 "
                   "(far below +5e-5), corr 0.9868. Decorrelated but too weak -- NEGATIVE by "
                   "direct measurement. Prior auto-close CORRECTED.",
        "verdict": "NEGATIVE_DIRECT_MEASUREMENT",
        "closes_rows": [65],
    },
    {
        "exp_id": "p20_s7_rffsvm",
        "hypothesis_id": "H20-CLASS-RFFSVM",
        "date": "2026-10-17",
        "ts": NOW,
        "git": GIT,
        "family": "CLASS",
        "experiment_number": 64,
        "fold_scheme": "primary fold-0 discovery screen (F2, cannot pass admission gate)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "view": "full (285 feat, fold-safe TE)",
        "arch": "RBFSampler(n_components=512, gamma=0.01, seed 1) -> LinearSVC(C=1.0); "
                "StandardScaler fit on FIT-train only; predict_proba = sigmoid(decision_function)",
        "results": {
            "standalone_auc": 0.929997, "portfolio_delta": -0.0021749, "corr": 0.8671,
            "fit_predict_seconds": 17.5,
            "artifact": "reports/phase20/classical_screen/rffsvm_full_fold0.json",
        },
        "finding": "S7 item-64 repair: the row had been closed by the classical-block addendum "
                   "WITHOUT ever being run (no artifact existed). Measured directly: RFF "
                   "approximate RBF SVM on fold-0 full view -- AUC 0.929997, port_delta "
                   "-2.175e-3, corr 0.8671. Strongly decorrelated but far too weak -- NEGATIVE "
                   "by direct measurement. Prior auto-close CORRECTED.",
        "verdict": "NEGATIVE_DIRECT_MEASUREMENT",
        "closes_rows": [64],
    },
    {
        "exp_id": "p20_s7_fidelity_relabel",
        "hypothesis_id": "S7-FIDELITY-REVIEW",
        "date": "2026-10-17",
        "ts": NOW,
        "git": GIT,
        "family": "REVIEW",
        "experiment_number": "35,36,60,63",
        "fold_scheme": "n/a (classification repair, no new measurement)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "view": "n/a",
        "arch": "n/a",
        "results": {
            "rows_reclassified": [35, 36, 60, 63],
            "fidelity_F1_to_F2": [35, 36, 60, 63],
            "scope_clarified": "35/36: masked-CORAL RECIPE negative, not all MTL -- reopening "
                               "needs a different auxiliary signal. 60: projection-forest PROXY "
                               "(Gaussian projection + ExtraTrees), not classwise-PCA rotation "
                               "forest -- exact method untested. 63: 8k-subsample exact "
                               "KernelRidge, not Nystrom/RFF approximation -- that is item 64, "
                               "now measured separately.",
        },
        "finding": "S7 re-examination found 4 rows labeled F1 despite holding controlled "
                   "outer-fold measurements (F2) and 3 rows whose negative scope exceeded the "
                   "measured recipe/proxy. All reclassified in "
                   "research/phase20_exhaustive/experiment_catalogue_96.json via "
                   "scripts/phase20/_s7_catalogue_repair.py; no measurement deleted, no status "
                   "flipped to positive.",
        "verdict": "RELABEL_ONLY",
        "closes_rows": [],
    },
]

def main() -> int:
    with LEDGER.open("a", encoding="utf-8") as fh:
        for e in ENTRIES:
            fh.write(json.dumps(e) + "\n")
    print(f"appended {len(ENTRIES)} entries to {LEDGER.relative_to(ROOT).as_posix()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
