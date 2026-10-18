"""Champion residual integration (FEAT 51 → full view) ledger + catalogue."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"


def main() -> int:
    entry = {
        "exp_id": "p20_champ_residual_integration",
        "hypothesis_id": "H51-CHAMPION-INTEGRATION",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "FEAT",
        "experiment_number": 51,
        "fold_scheme": "primary 5-fold (F3, full OOF admission gate)",
        "prereg": {"path": "configs/phase20/feat51_residual_ratings_prereg.json",
                   "sha256": "a1013332af66f003af42a8db7aea15a69cdd02c199ac05360121d45b9a266eb0"},
        "arch": "Champion LGBM full view (285 feat) + 13 cross-fit residual ratings (FEAT 51). "
                "Residuals: 5-fold cross-fit HistGB teachers per rating, predicting from other 20 covariates. "
                "LGBM xt_xt_f10 recipe (seed 1, 6000 est, ES 250).",
        "results": {
            "oof_auc_5f": 0.961079,
            "champion_oof": 0.961853,
            "delta_vs_champion": -0.0007743,
            "fold_aucs": [0.960982, 0.961137, 0.960984, 0.960494, 0.961830],
            "mean_fold_auc": 0.961085,
            "std_fold_auc": 0.000459,
            "lgbm_full_baseline": 0.960904,
            "marginal_gain_vs_full": +0.000175,
            "admission_gate": {"required": 1.5e-5, "observed": -7.74e-4, "passes": False},
            "artifact": "reports/phase20/champion_residual_integration/champion_residual_integration.json",
        },
        "finding": (
            "Integrating the 13 cross-fit residual rating features into the champion's "
            "285-feature rich view yields 5-fold OOF 0.961079 — **below the champion's "
            "0.961853 (−7.74e-4)**. Marginal gain vs LGBM full (0.960904) is only "
            "+1.75e-4. The residual signal (strong on minimal base, +3.94e-3 on 21raw; "
            "decorrelated Spearman 0.929) is **already largely captured** by the existing "
            "rich-view blocks (TE on ratings, external features, token features, teacher "
            "predictions). The fold-0 blend with champion proxy (+3.32e-3) does not "
            "translate to full-5 OOF because the full view's TE/external/token blocks "
            "already absorb the rating-inconsistency signal. Residual block = genuinely "
            "new signal but redundant in the full feature set. Not an admission-gate pass."),
        "verdict": "NEGATIVE_MARGINAL (signal absorbed by existing rich view)",
        "closes_rows": [],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "Champion integration (full-5 OOF): 13 cross-fit residual rating features "
        "added to 285-feature rich view. 5-fold OOF 0.961079 vs champion 0.961853 "
        "(delta −7.74e-4). Marginal gain vs LGBM full (0.960904) = +1.75e-4 — below "
        "admission gate +1.5e-5 but negative vs champion. The residual signal (+3.94e-3 "
        "on 21raw, decorrelated Spearman 0.929, fold-0 blend +3.32e-3) is ALREADY "
        "CAPTURED by the existing rich-view blocks (TE on ratings, external, token, "
        "teacher). Fold-0 blend +3.32e-3 does not translate to full-5 OOF because the "
        "rich view already absorbs the signal. NEGATIVE for marginal champion upgrade. "
        "Recommendation: FEAT 51 is a PROMISING feature block for NEW pipelines with "
        "weaker base views, but redundant in the current rich-view champion."
    )
    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for row in cat["experiments"]:
        if row.get("experiment_number") == 51:
            row["phase20_finding"] = finding
            row["status"] = "COMPLETE_NEGATIVE"
            print("row 51: status -> COMPLETE_NEGATIVE (marginal negative)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())