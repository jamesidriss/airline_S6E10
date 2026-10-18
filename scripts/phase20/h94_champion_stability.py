"""H94 champion-stability completion (prereg amendment P20-FINALIST-READINESS-H94-CHAMPION).

Same audit as reports/phase20/finalist_readiness/finalist_readiness_audit.json
h94_stability, with the champion reference upgraded from the route proxy to the
now-archived true champion v7_sol_tabpfn_internal2_aux10.  Variant set copied
verbatim from the original audit report; stability rule unchanged (every fold:
blend_delta >= -5e-5 AND spearman > 0.98).  F1 on saved OOF arrays only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import git_commit, save_json  # noqa: E402
from src.submission.store import load_oof  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PREREG = "configs/phase20/finalist_readiness_h94_champion_amendment.json"
PREREG_SHA = "ee260bd43c7dc4108e1cbc576842d1e89edbe8f1c28eddf32bad4a08b7fae413"
ORIGINAL = "reports/phase20/finalist_readiness/finalist_readiness_audit.json"
CHAMPION = "v7_sol_tabpfn_internal2_aux10"
CHAMPION_OOF_SHA = "1981745c3d4f5b743c3f02c23f92a1e8975692ca473bc13e3156165bda1a7552"
RULE = "every fold: blend_delta >= -5e-5 AND spearman > 0.98"
OUT = ROOT / "reports/phase20/finalist_readiness/h94_champion_stability.json"


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def main() -> None:
    sha = hashlib.sha256((ROOT / PREREG).read_bytes()).hexdigest()
    assert sha == PREREG_SHA, f"prereg sha mismatch {sha}"
    orig = json.loads((ROOT / ORIGINAL).read_text(encoding="utf-8"))
    variant_names = sorted(orig["h94_stability"]["variants"].keys())

    from src.common import load_cached_parquet

    tr, _te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    folds = get_scheme("primary", y, tr["id"].to_numpy()).folds

    champ = load_oof(CHAMPION)
    champ_auc = float(roc_auc_score(y, champ))
    champ_fold_auc = [float(roc_auc_score(y[folds == k], champ[folds == k])) for k in range(5)]
    print(f"champion {CHAMPION} auc={champ_auc:.12f} fold_auc={[round(a, 6) for a in champ_fold_auc]}")

    results = {}
    for name in variant_names:
        var = load_oof(name)
        per_fold = []
        for k in range(5):
            m = folds == k
            c_auc = float(roc_auc_score(y[m], champ[m]))
            blend = (0.5 * (logit(champ[m]) + logit(var[m]))).astype(np.float32)
            b_auc = float(roc_auc_score(y[m], blend))
            from scipy.stats import spearmanr

            sp = float(spearmanr(champ[m], var[m]).statistic)
            per_fold.append(
                {
                    "fold": k,
                    "champion_auc": c_auc,
                    "blend_auc": b_auc,
                    "blend_delta": b_auc - c_auc,
                    "spearman": sp,
                }
            )
        min_delta = min(r["blend_delta"] for r in per_fold)
        min_sp = min(r["spearman"] for r in per_fold)
        stable = bool(min_delta >= -5e-5 and min_sp > 0.98)
        results[name] = {
            "per_fold": per_fold,
            "min_blend_delta": min_delta,
            "min_spearman": min_sp,
            "stable": stable,
        }
        print(f"  {name:40s} min_delta={min_delta:+.3e} min_spearman={min_sp:.6f} stable={stable}")

    champion_stable = all(r["stable"] for r in results.values())
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git": git_commit(),
        "prereg": PREREG,
        "prereg_sha256": sha,
        "original_audit": ORIGINAL,
        "champion": CHAMPION,
        "champion_oof_sha256": CHAMPION_OOF_SHA,
        "champion_overall_auc": champ_auc,
        "champion_fold_auc": champ_fold_auc,
        "stability_rule": RULE,
        "variants": results,
        "champion_stable": champion_stable,
        "risk_disclosure": sorted(n for n, r in results.items() if not r["stable"]),
    }
    save_json(report, OUT)
    print(f"champion_stable={champion_stable} -> {OUT}")


if __name__ == "__main__":
    main()
