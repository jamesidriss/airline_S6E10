"""Append the Phase-24 external-OOF addendum to the append-only ledger (idempotent by exp_id)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments/ledger.jsonl"
TS = "2026-10-23T00:00:00+00:00"
GIT = "775dbab"

ENTRIES = [
    {
        "kind": "EXPERIMENT",
        "exp_id": "p24_20261023_E8_external_oof_pool_and_candidate_x",
        "ts": TS,
        "git": GIT,
        "prereg": "configs/phase24/candidate_x_prereg.json (frozen before the public measurement, which was never spent)",
        "scope": "27 downloadable external OOF vectors (11 Deotte TFM, 8 goodpjw2008 TabPFN/TabICL, 7 najiama blends, plus the v7 combinations), measured on our immutable primary and shadow folds",
        "results": {
            "deotte_tfm_individual": {
                "range": [0.957075, 0.961145],
                "delta_vs_v7_range": [-4.79e-3, -7.15e-4],
                "folds_positive": "0/5 for all 11",
            },
            "tfm_nested_stack": {"pooled_auc": 0.961236, "delta_vs_v7": -6.21e-4, "t": -28.12},
            "v7_plus_tfm_nested_stack": {"pooled_auc": 0.961933, "delta_vs_v7": 8.20e-5, "t": 5.94, "folds_positive": 5},
            "nj05_alone": {"pooled_auc": 0.9622339, "delta_vs_v7": 3.747e-4, "t": 11.36, "folds_positive": 5},
            "candidate_X_085nj05_015v7": {
                "pooled_auc": 0.9622466,
                "delta_vs_v7_primary": 3.886e-4,
                "t_primary": 13.38,
                "folds_positive_primary": 5,
                "delta_vs_v7_shadow": 3.944e-4,
                "t_shadow": 9.48,
                "folds_positive_shadow": 5,
                "test_sha256": "7032b4d928911cdbe23eb76433e1e9b1c4d32d73fe10405cd6ed8df40f1427f1",
            },
            "full_27_member_stack": {"pooled_auc": 0.962196, "delta_vs_v7": 3.413e-4, "t": 13.28},
        },
        "finding": "Our own model store caps at 0.96185, but the honest public frontier is ~0.9622: 27 external vectors were measured on our folds, the strongest honest public blend reaches 0.962234 (+3.75e-4, t 11.4, 5/5), and a fixed 0.85/0.15 blend with v7 reaches 0.962247 (+3.89e-4, t 13.4, primary and shadow 5/5). Deotte's 11 TFMs are individually and jointly worse than v7 (best individual 0.961145; nested stack 0.961236), so the gain is not 'a better TFM' but a specific diverse blend. The public leaderboard top (0.96341 on 2026-10-20, 0.96370 reported 2026-10-23) is public-surface selection, not model quality: the top teams have 128-133 submissions, and the public notebooks' own text says part of the gain is public noise and no independent CV improvement is claimed.",
        "verdict": "MEASURED_POSITIVE_NOT_DEPLOYED_USER_STOPPED",
    },
    {
        "kind": "DECISION",
        "exp_id": "p24_20261023_user_stop_no_upload",
        "ts": TS,
        "git": GIT,
        "decision": "NO_UPLOAD",
        "reason": "user closed the programme; the Phase-22 policy (external vectors are diagnostic only) was never overridden",
        "state": {
            "champion": {"name": "v7_sol_tabpfn_internal2_aux10", "ref": 57090508, "oof": 0.9618529228157235, "public": 0.96146},
            "finalists_recommended": [{"slot": "A", "ref": 57090508}, {"slot": "B", "ref": 56951906}],
            "candidate_X": "certified, not uploaded, kept in artifacts/phase24_candidateX and reports/phase24/candidate_x_certification.json",
            "submissions_used": 0,
            "gpu_hours_used": 0.0,
        },
        "open_user_action": "select up to two final submissions in the Kaggle web UI before 2026-10-31T23:59Z",
    },
]


def main() -> None:
    existing = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line).get("exp_id"))
                except json.JSONDecodeError:
                    pass
    added = 0
    with LEDGER.open("a", encoding="utf-8", newline="\n") as f:
        for e in ENTRIES:
            if e["exp_id"] in existing:
                continue
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
            added += 1
    total = sum(1 for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip())
    print(json.dumps({"added": added, "ledger_lines": total}))


if __name__ == "__main__":
    main()
