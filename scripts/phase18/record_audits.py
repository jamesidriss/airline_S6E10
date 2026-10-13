"""Append Phase 18 audit records to the experiments ledger + phase18 registry.

Idempotent: skips any (exp_id) already present in the ledger.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import arr_sha256, file_sha256  # noqa: E402
from experiments_ledger import LEDGER, Record, append  # noqa: E402
from src.phase18 import registry  # noqa: E402
from src.phase18.fingerprint import git_head  # noqa: E402

import numpy as np  # noqa: E402


def existing_ids() -> set[str]:
    if not LEDGER.exists():
        return set()
    out = set()
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            if "exp_id" in d:
                out.add(d["exp_id"])
    return out


def main() -> None:
    atlas = json.loads((ROOT / "reports/phase18/error_atlas.json").read_text(encoding="utf-8"))
    internal = json.loads((ROOT / "reports/phase18/tabpfn_internal_audit.json").read_text(encoding="utf-8"))
    internal4 = json.loads((ROOT / "reports/phase18/tabpfn_internal4_fold0_audit.json").read_text(encoding="utf-8"))

    ids = existing_ids()
    new = []
    if "p18_error_atlas" not in ids:
        new.append(Record(
            exp_id="p18_error_atlas", ts="", git=git_head(), family="diagnostic",
            featureset="none (certified v7/v6 OOF vectors)", fold_scheme="primary_5f_seed20261010",
            params={"kind": "exact_pair_credit_atlas"}, seed=20261010,
            oof_auc=atlas["auc"]["v7"], fold_aucs=[f["auc_v7"] for f in atlas["folds"]],
            oof_std=0.0, delta_vs_champion=0.0, paired_fold_deltas=[],
            corr_with_champion=1.0, duration_s=0.0,
            oof_hash=atlas["inputs"]["v7_portfolio_sha256"], test_hash=None,
            data_hash="2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36",
            notes=("Zero-training diagnostic. Exact pair identity verified: net credit "
                   "reproduces the v7-v6 AUC delta to <1e-12. Eco Plus: pooled +1.8e-5, "
                   "2/5 negative fold cells among 45 segmentxfold cells => multiple-"
                   "comparison candidates, no reproducible weakness."),
            verdict="INFORMATIONAL_NO_ADMISSION_DECISION",
        ))
    if "p18_internal2_audit" not in ids:
        a = internal["auc"]
        new.append(Record(
            exp_id="p18_internal2_audit", ts="", git=git_head(), family="diagnostic",
            featureset="none (certified B1 member logits)", fold_scheme="primary_5f_seed20261010",
            params={"kind": "tabpfn_internal_estimator_decomposition", "n_estimators": 2},
            seed=1201, oof_auc=a["route_internal2"],
            fold_aucs=[f["route"] for f in internal["fold_aucs"]],
            oof_std=0.0, delta_vs_champion=0.0, paired_fold_deltas=[],
            corr_with_champion=1.0, duration_s=0.0,
            oof_hash="bcb2811329d3686882450e93435647f8f7ab65abfc48c1a1b51db1e9d1ef95e0",
            test_hash=None, data_hash="2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36",
            notes=("Route = exact mean of the two member probabilities (max gap 1.2e-7). "
                   "Single members lose to the 2-mean on ALL 5 folds (m0 -23.7..-33.0e-5, "
                   "m1 -11.2..-20.2e-5). Averaging is a real consistent gain; a single-"
                   "member anchor would cost 11-33e-5. Diagnostic only; frozen config."),
            verdict="INFORMATIONAL_NO_ADMISSION_DECISION",
        ))
    if "p18_internal4_fold0_audit" not in ids:
        new.append(Record(
            exp_id="p18_internal4_fold0_audit", ts="", git=git_head(), family="diagnostic",
            featureset="none (certified B2 fold0 member logits)", fold_scheme="block_fold0_only",
            params={"kind": "tabpfn_4_estimator_fold0_decomposition", "n_estimators": 4},
            seed=1201, oof_auc=internal4["auc_route4"], fold_aucs=[internal4["auc_route4"]],
            oof_std=0.0, delta_vs_champion=0.0, paired_fold_deltas=[],
            corr_with_champion=1.0, duration_s=0.0,
            oof_hash=None, test_hash=None,
            data_hash="2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36",
            notes=("Wave-1 discovery evidence for H18-E: route4 - route2 = +5.34e-5 on fold 0 "
                   "ONLY. All 4 single members lose to the 4-mean. m0-m3 Spearman 0.9793. "
                   "Folds 1-4 were RESOURCE_BLOCKED (disk), not model-rejected. No "
                   "promotion claim from one fold."),
            verdict="DISCOVERY_FOLD0_POSITIVE_REPLICATION_REQUIRED",
        ))
    for rec in new:
        append(rec)
        print("ledger:", rec.exp_id)

    # phase18 registry: fingerprint the audit scripts for reproducibility records
    for exp_id, script, status in [
        ("p18_error_atlas", "scripts/phase18/build_error_atlas.py", "COMPLETE"),
        ("p18_internal2_audit", "scripts/phase18/audit_tabpfn_internals.py", "COMPLETE"),
        ("p18_internal4_fold0_audit", "scripts/phase18/audit_tabpfn_internal4_fold0.py", "COMPLETE"),
    ]:
        prior = registry.lookup(exp_id)
        if not prior:
            registry.append_record({
                "exp_id": exp_id, "fingerprint": exp_id,  # diagnostics: id-keyed, no training fingerprint
                "status": status,
                "script_sha256": file_sha256(ROOT / script),
                "git": git_head(),
            })
            print("registry:", exp_id)


if __name__ == "__main__":
    main()
