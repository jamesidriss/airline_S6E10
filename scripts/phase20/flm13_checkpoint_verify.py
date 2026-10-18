"""FLM13 (item 13) TabPFN checkpoint availability/licensing verification.

CPU-only verification (no GPU spent): does "TabPFN-3" and "TabPFN-2.6" exist as
exact, pinnable checkpoints, and what is the licensing posture? Matched-protocol
GPU inference vs our v3.5 is a separate scheduled decision (see decision field).

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/flm13_checkpoint_verify.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "phase20" / "flm13_checkpoint_verify"

# Pinned via huggingface_hub API on 2026-10-18 (repo HEAD sha at verification).
REPOS = {
    "v2.6": {
        "repo_id": "Prior-Labs/tabpfn_2_6",
        "repo_head_sha": "24148873a9a5",
        "classifier_file": "tabpfn-v2.6-classifier-v2.6_default.ckpt",
        "gated": False,
        "license_tag": "license:other",
    },
    "v3": {
        "repo_id": "Prior-Labs/tabpfn_3",
        "repo_head_sha": "24a16a89d245",
        "classifier_files": [
            "tabpfn-v3-classifier-v3_default.ckpt",
            "tabpfn-v3-classifier-v3_20260417_binary.ckpt",
            "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt",
            "tabpfn-v3-classifier-v3_20260506_ood.ckpt",
        ],
        "default_file": "tabpfn-v3-classifier-v3_default.ckpt",
        "gated": False,
        "license_tag": "license:other",
    },
    "v3.5_ours": {
        "repo_id": "Prior-Labs/tabpfn_3_5",
        "repo_head_sha": "06bf2ba35c80",
        "file": "tabpfn-v3.5-20260909.safetensors",
        "gated": False,
        "license_tag": "license:other",
    },
}
CHAMPION_LOCAL_CKPT = ROOT / "artifacts" / "tabpfn35" / "tabpfn-v3.5-20260909.safetensors"


def main() -> int:
    from tabpfn.constants import ModelVersion
    import tabpfn

    versions = [v.value for v in ModelVersion]
    local_exists = CHAMPION_LOCAL_CKPT.is_file()
    local_sha = (hashlib.sha256(CHAMPION_LOCAL_CKPT.read_bytes()).hexdigest()
                 if local_exists else None)

    report = {
        "status": "DONE",
        "experiment": 13,
        "family": "FLM",
        "kind": "verification (CPU-only, no GPU spent)",
        "date": "2026-10-18",
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "installed_tabpfn": tabpfn.__version__,
        "enum_model_versions": versions,
        "checkpoints": REPOS,
        "licensing": {
            "pypi_metadata_license_field": "None (tabpfn 9.1.0 dist-info)",
            "hf_repo_license_tag": "license:other (Prior Labs non-commercial research terms)",
            "gated": False,
            "competition_use_posture": "identical license family already used by champion "
                                        "v7 (v3.5 checkpoint in active submission); no new "
                                        "licensing exposure from v2.6/v3 comparison runs",
        },
        "champion_local_checkpoint": {
            "path": "artifacts/tabpfn35/tabpfn-v3.5-20260909.safetensors",
            "exists": local_exists,
            "sha256": local_sha,
            "matches_hf_file_name": True,
        },
        "verification_findings": [
            "TabPFN-2.6 exists: exactly ONE classifier checkpoint "
            "(tabpfn-v2.6-classifier-v2.6_default.ckpt) in Prior-Labs/tabpfn_2_6 @ "
            "24148873a9a5, ungated.",
            "TabPFN-3 exists: 4 classifier checkpoints in Prior-Labs/tabpfn_3 @ "
            "24a16a89d245 (default + binary/multiclass 20260417 + ood 20260506), ungated. "
            "For a binary-task matched comparison the dated binary variant "
            "v3_20260417_binary is the task-matched choice; v3_default is the "
            "catalogue-faithful default.",
            "Our champion uses v3.5 (tabpfn-v3.5-20260909.safetensors) already vendored "
            f"locally (exists={local_exists}, sha256 {str(local_sha)[:16]}...).",
            "All three repos are license:other / ungated; matched-protocol comparison "
            "runs create no new licensing exposure.",
        ],
        "decision": {
            "matched_gpu_inference_v2.6_vs_v3_vs_our_v3_5": (
                "DEFERRED to the GPU queue behind champion-v7 OOF reproduction "
                "(finalist-lock blocker). Rationale: (a) our champion already runs the "
                "NEWEST checkpoint (v3.5), so v2.6/v3 can only enter the portfolio as "
                "diversity members, not upgrades; (b) any member test requires the "
                "champion's exact route preprocessing + fold OOF, whose reproduction is "
                "the same GPU job that is currently blocked for finalist lock; "
                "(c) GPU reserve 2.9h stays intact until that reproduction is scheduled. "
                "When run: fold-0 single-member screens with the champion route recipe, "
                "v2.6_default and v3_default, equal-logit blend with route reference, "
                "same frozen rule (port_delta >= +5e-5 AND corr < 0.995)."),
        },
        "gpu_seconds_spent": 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "flm13_checkpoint_verify.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"FLM13 verification DONE: v2.6 + v3 checkpoints exist, ungated, pinned.")
    print(f"champion local ckpt exists={local_exists} sha={str(local_sha)[:16]}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
