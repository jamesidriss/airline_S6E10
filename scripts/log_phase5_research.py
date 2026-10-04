"""Log two Phase 5 research records: the public-notebook audit and the original-domain audit.

The notebook audit is also a data/rule-compliance record, so it carries source, access date and
what was actually done with it.

Idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "experiments" / "ledger.jsonl"

ENTRIES = [
    {
        "exp_id": "public_intel_audit_nina2025_s6e10",
        "oof_auc": None,
        "fold_aucs": [],
        "verdict": "NO ADOPTABLE MECHANISM -- top notebooks are blends of imported public submissions",
        "compliance": {
            "sources": [
                {"what": "notebook metadata + source",
                 "url": "https://www.kaggle.com/api/v1/kernels/pull/nina2025/ps-s6e10-h-blend-3",
                 "access_date": "2026-10-04", "access": "public, authenticated API read only",
                 "license": "public Kaggle notebook", "use": "read and analysed; nothing copied"},
                {"what": "notebook metadata + source",
                 "url": "https://www.kaggle.com/api/v1/kernels/pull/"
                        "nina2025/ps-s6e10-rank-top-vanillacatboost-honest-round",
                 "access_date": "2026-10-04", "access": "public, authenticated API read only",
                 "license": "public Kaggle notebook", "use": "read and analysed; nothing copied"},
                {"what": "author notebook index",
                 "url": "https://www.kaggle.com/api/v1/kernels/list?user=nina2025",
                 "access_date": "2026-10-04",
                 "note": "250 notebooks fetched; 8 are s6e10"},
            ],
            "decision": ("NOTHING was imported into any model, blend or submission. No external "
                         "prediction file, private label or leaked artefact was used. The "
                         "prediction CSVs those notebooks depend on were deliberately NOT downloaded."),
        },
        "findings": {
            "ps-s6e10-h-blend-3": {
                "claimed_public": 0.96159, "gpu": False, "version": 4,
                "inputs": ["nina2025/ps-s6e10-04", "playground-series-s6e10"],
                "what_it_actually_does": (
                    "rank-space blend of three PRE-EXISTING submission CSVs whose filenames are "
                    "literally their public scores: weights 0.00 on '0.96152', 0.07 on '0.96156', "
                    "0.93 on '0.96159'. It then adds an ascending/descending rank hedge with "
                    "type_sort 0.30/0.70 and per-rank corrections [-0.03, +0.02, +0.01]. No model "
                    "is trained."),
                "is_it_a_model": False,
                "reproduce_honestly": False,
                "why": ("the reported score is the INPUT FILE's own public score propagated; the "
                        "93/7 weights were chosen against public scores, so this is public-LB hill "
                        "climbing by construction. Reproducing it would mean importing other "
                        "participants' predictions."),
            },
            "ps-s6e10-rank-top-vanillacatboost-honest-round": {
                "votes": 28, "gpu": False,
                "what_it_actually_does": (
                    "trains a plain CatBoostRegressor(max_depth=7) on ordinal-encoded features, "
                    "then submits 1.001*imported_0.96160.csv - 0.001*own_catboost. Its own model "
                    "carries a weight of -0.1%. Feature names are truncated to feat[0:21], "
                    "consistent with blending the competition's own feature set."),
                "is_it_a_model": "trains one, but it contributes -0.1% of the submission",
                "reproduce_honestly": False,
            },
        },
        "usable_signal": (
            "The only transferable idea is the rank-space ascending/descending hedge geometry as an "
            "alternative blend space. That is a genuine technique, but it is a blend-geometry "
            "variant, which this campaign has already measured as saturated territory "
            "(equal-logit beats probability and rank alternatives), so its expected value is low."
        ),
        "board_top": (
            "Useful by-product: public submissions scoring 0.96152-0.96160 exist, so the top of the "
            "board is ~0.9616 against our v3_final public of 0.960980 -- a real gap of about 6e-4, "
            "not a blend artefact."
        ),
    },
    {
        "exp_id": "original_domain_audit",
        "oof_auc": None,
        "fold_aucs": [],
        "verdict": "GENERATOR FINGERPRINT FOUND -- the generator rewrote the delay distributions",
        "measured": {
            "n_competition": 699635, "n_original": 129880,
            "domain_classifier_auc": 0.7225,
            "nn_dist_original_to_comp": 3.2115,
            "nn_dist_comp_to_comp": 0.8930,
            "nn_density_ratio": 3.596,
            "max_categorical_total_variation": 0.0296,
            "categorical_tv": {"Class": 0.0296, "Type of Travel": 0.0204,
                               "Gender": 0.0101, "Customer Type": 0.0078},
            "worst_numeric_marginals": {
                "Departure Delay in Minutes": {"quantile_gap_sd": 1.6452,
                                                "frac_original_outside_comp_p01_p99": 0.1393},
                "Flight Distance": {"quantile_gap_sd": 0.1756,
                                    "frac_original_outside_comp_p01_p99": 0.0517},
                "Baggage handling": {"quantile_gap_sd": 0.1463,
                                     "frac_original_outside_comp_p01_p99": 0.0},
            },
        },
        "reading": (
            "The categories survived the generator essentially intact (total variation <= 0.030) and "
            "so did most 0-5 rating scales, but Departure Delay in Minutes did not: its quantile gap "
            "is 1.65 SD and 13.9% of original values fall outside the competition's central 98%. "
            "Original rows also sit 3.6x further from the competition set than competition rows sit "
            "from each other, i.e. they occupy genuinely sparser regions of a shared support, and a "
            "logistic domain classifier separates the two at AUC 0.72."
        ),
        "why_it_matters": (
            "This is a concrete, actionable generator fingerprint rather than a vague domain-shift "
            "claim, and it explains the already-measured harm from appending original rows: about 14% "
            "of them sit in delay regions the competition generator almost never produces. It also "
            "refines the Family C hypothesis -- original rows are not simply 'different', they are "
            "specifically wrong in the delay tail, so if they are used at all they should be filtered "
            "or importance-weighted on delay support rather than used wholesale."
        ),
        "bug_found_first": (
            "The first run reported a nearest-neighbour density ratio of 4.4e6, which is an artefact: "
            "the retrieval index was built on the whole competition set and then queried with a "
            "sample of that same set, so every query matched itself at distance 0. Fixed by splitting "
            "the competition set into disjoint index and query halves, which gives the sensible 3.60."
        ),
    },
]


def main() -> int:
    existing = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line).get("exp_id"))
                except Exception:  # noqa: BLE001
                    pass
    added = 0
    with LEDGER.open("a", encoding="utf-8") as f:
        for e in ENTRIES:
            if e["exp_id"] in existing:
                print(f"  skip: {e['exp_id']}")
                continue
            e["date"] = "2026-10-04"
            f.write(json.dumps(e) + "\n")
            added += 1
            print(f"  appended: {e['exp_id']} [{e['verdict'][:58]}]")
    print(f"\n{added} entries appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
