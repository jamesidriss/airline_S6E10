"""Phase18 Axis-E part 2: 4-estimator member analysis from saved B2 fold0 artifacts.

Zero GPU. Only reads artifacts/phase17_B2/f0/member_{0..3}_raw_logits.npy.
B2 f0 is the only completed 4-estimator fold (f1+ were RESOURCE_BLOCKED by the
disk guard, recorded as such). This is diagnostic evidence about estimator
diversity, NOT a promotion path and NOT a five-fold claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score  # noqa: E402

OUT = ROOT / "reports" / "phase18"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ROOT / "data/cache/train.parquet")
    y = df["satisfaction"].to_numpy()

    ids = np.load(ROOT / "artifacts/phase17_B2/f0/apply_ids.npy")
    yv = y[ids]
    probs = []
    for mi in range(4):
        raw = np.load(ROOT / f"artifacts/phase17_B2/f0/member_{mi}_raw_logits.npy")[0]
        p = 1.0 / (1.0 + np.exp(-(raw[:, 1].astype(np.float64) - raw[:, 0])))
        probs.append(p)
    route4 = np.load(ROOT / "artifacts/phase17_B2/f0/prediction.npy").astype(np.float64)
    port4 = np.load(ROOT / "artifacts/phase17_B2/f0/portfolio.npy").astype(np.float64)

    aucs = [roc_auc_score(yv, p) for p in probs]
    mean4 = np.mean(probs, axis=0)
    # pairwise member correlations
    corr = np.corrcoef(np.array(probs))
    ranks = np.apply_along_axis(lambda v: pd.Series(v).rank().to_numpy(), 1, np.array(probs))
    spear = np.corrcoef(ranks)

    # B1 (2-estimator) same-fold reference
    ids1 = np.load(ROOT / "artifacts/phase17_B1/f0/apply_ids.npy")
    assert np.array_equal(ids, ids1), "fold0 apply ids differ between B1 and B2"
    route2 = np.load(ROOT / "artifacts/phase17_B1/f0/prediction.npy").astype(np.float64)
    a_route2 = roc_auc_score(yv, route2)
    a_route4 = roc_auc_score(yv, route4)

    out = {
        "fold": 0, "rows": int(len(ids)),
        "member_aucs": aucs,
        "member_mean_minus_route4": roc_auc_score(yv, mean4) - a_route4,
        "route4_vs_mean4_maxabs": float(np.abs(mean4 - route4).max()),
        "auc_route4": a_route4, "auc_route2": a_route2,
        "route4_minus_route2": a_route4 - a_route2,
        "portfolio4_auc": roc_auc_score(yv, port4),
        "member_pair_pearson": corr.tolist(),
        "member_pair_spearman": spear.tolist(),
        "m0_m3_pearson": float(corr[0, 3]),
        "m0_m3_spearman": float(spear[0, 3]),
        "notes": [
            "f0 is the ONLY completed 4-estimator fold; f1+ were RESOURCE_BLOCKED (disk guard), not model-rejected.",
            "No promotion claim: a single fold cannot establish a 4-estimator gain.",
            "Diagnostic purpose: measure whether estimator diversity (not just count) grows with member index.",
        ],
    }
    (OUT / "tabpfn_internal4_fold0_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("member_aucs", "auc_route4", "auc_route2",
                                          "route4_minus_route2", "m0_m3_pearson", "m0_m3_spearman")}, indent=2))


if __name__ == "__main__":
    main()
