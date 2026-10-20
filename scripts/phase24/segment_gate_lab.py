"""Phase 24 -- segment-gated composition test (H24-C2), frozen design.

Mechanism under test
--------------------
The phase-19 atlas reports port AUC 0.9503 on the 254,260 rows whose
(13 ratings + Class + Type of Travel + Customer Type) key occurs exactly once,
versus ~0.962 pooled.  Those rows cannot benefit from any key-level lookup and
must be ranked by a smooth function of the remaining covariates.  A family whose
inductive bias is smoother than the champion's deep trees may therefore rank
them better, even when it is worse pooled.  Segment gating on a *label-free*
feature-frequency covariate is a one-parameter composition per segment, so the
selection risk is far smaller than a learned gate over hundreds of covariates
(ENS72, which failed).

Frozen family (deterministic rule, no score-based choice of vectors):
  8 families, each contributing the stored OOF vector with the best standalone
  pooled AUC among vectors whose name starts with that family prefix.
  2 segment definitions.  => 16 tests, Holm over 16.

Gate: mean paired fold delta >= +1.5e-5, t >= 2.5, 5/5 folds positive,
Holm <= 0.05.  Otherwise NO_ADMISSION and the segment-gating axis is closed.

    .\\.venv\\Scripts\\python.exe scripts\\phase24\\segment_gate_lab.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import arr_sha256, file_sha256, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.validation.paired_auc import holm_adjust, paired_auc  # noqa: E402

OUT = ROOT / "reports/phase24/segment_gate_lab.json"
PREREG = ROOT / "configs/phase24/segment_gate_prereg.json"
PRED = ROOT / "artifacts/predictions"
CHAMP = "v7_sol_tabpfn_internal2_aux10"
GRID = np.round(np.arange(0.0, 0.5001, 0.05), 3)
RATING_KEY = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness", "Class", "Type of Travel", "Customer Type",
]
FAMILIES = {
    "lgbm": ("z4_lgb", "xt_xt", "prod5_lgbm", "view_lgbm"),
    "xgb": ("z4_xgb", "prod5_xgb", "xt_xgb", "zoo_xgb"),
    "cat": ("z3_cat", "prod5_cat", "zoo_cat"),
    "realmlp": ("z3_rm", "z5_rm", "nn_realmlp"),
    "tabm": ("z5_tabm",),
    "blend": ("blend_v4", "blend_v3", "cand_"),
    "conf": ("conf_",),
    "resid": ("resid_",),
}


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    tr, _ = load_cached_parquet()
    y = tr.satisfaction.to_numpy(dtype="int8")
    folds = get_scheme("primary", y, tr["id"].to_numpy()).folds
    v7 = np.load(PRED / f"{CHAMP}_oof.npy").astype(np.float64)
    base_logit = logit(v7)
    base_auc = float(roc_auc_score(y, v7))

    key = pd.util.hash_pandas_object(tr[RATING_KEY], index=False).to_numpy(np.uint64)
    freq = pd.Series(key).map(pd.Series(key).value_counts()).to_numpy()
    segs = {"S1_unique_key": freq == 1, "S2_sparse_key": freq <= 3}

    # deterministic family representatives: best standalone AUC inside each family
    fam_best = {}
    for fam, prefixes in FAMILIES.items():
        best = None
        for p in sorted(PRED.glob("*_oof.npy")):
            name = p.name[: -len("_oof.npy")]
            if name == CHAMP or not any(name.startswith(x) for x in prefixes):
                continue
            a = np.load(p)
            if a.shape != v7.shape or not np.isfinite(a).all():
                continue
            pv = a.astype(np.float64)
            if pv.min() < -1e-6 or pv.max() > 1 + 1e-6:
                pv = sigmoid(np.clip(pv, -60, 60))
            sc = float(roc_auc_score(y, pv))
            if best is None or sc > best[1]:
                best = (name, sc, pv)
        assert best is not None, f"no vector for family {fam}"
        fam_best[fam] = best

    tests = []
    for fam, (name, sc, pv) in fam_best.items():
        for sname, mask in segs.items():
            cand = base_logit.copy()
            weights = []
            for k in range(5):
                fit = (folds != k) & mask
                ev = (folds == k) & mask
                zr = base_logit[fit]
                zc = logit(pv)[fit]
                best_w, best_v = 0.0, -np.inf
                for w in GRID:
                    v = roc_auc_score(y[fit], (1 - w) * zr + w * zc)
                    if v > best_v:
                        best_v, best_w = v, float(w)
                weights.append(best_w)
                cand[ev] = (1 - best_w) * base_logit[ev] + best_w * logit(pv)[ev]
            cand_p = sigmoid(cand)
            d = np.array([roc_auc_score(y[folds == k], cand_p[folds == k])
                          - roc_auc_score(y[folds == k], v7[folds == k]) for k in range(5)])
            se = float(d.std(ddof=1) / np.sqrt(5))
            tests.append({
                "family": fam, "vector": name, "vector_standalone_auc": sc,
                "segment": sname, "segment_rows": int(mask.sum()),
                "weights_per_fold": weights,
                "pooled_auc": float(roc_auc_score(y, cand_p)),
                "mean_fold_delta": float(d.mean()), "fold_se": se,
                "t": float(d.mean() / se) if se > 0 else float("nan"),
                "folds_positive": int((d > 0).sum()),
                "per_fold_deltas": d.tolist(),
                "pooled_delong": paired_auc(y, cand_p, v7),
                "vector_sha256": arr_sha256(cand_p.astype(np.float32)),
            })

    adj = holm_adjust([t["pooled_delong"]["one_sided_p"] for t in tests])
    for t, p in zip(tests, adj):
        t["holm_adjusted_p"] = float(p)
        t["passes"] = bool(t["mean_fold_delta"] >= 1.5e-5 and t["t"] >= 2.5
                           and t["folds_positive"] == 5 and p <= 0.05)
    tests.sort(key=lambda t: -t["mean_fold_delta"])

    report = {
        "status": "COMPLETE",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prereg_sha256": file_sha256(PREREG) if PREREG.exists() else None,
        "source_sha256": file_sha256(Path(__file__)),
        "champion_auc": base_auc,
        "champion_sha256": file_sha256(PRED / f"{CHAMP}_oof.npy"),
        "segment_sizes": {k: int(v.sum()) for k, v in segs.items()},
        "family_representatives": {k: {"vector": v[0], "auc": v[1]} for k, v in fam_best.items()},
        "n_tests": len(tests),
        "tests": tests,
        "any_pass": bool(any(t["passes"] for t in tests)),
        "reading": (
            "Segment gating is a one-parameter composition per segment on a label-free "
            "covariate. A pass would mean the champion is measurably worse on a covariate-"
            "defined slice and that a smoother family repairs it. A failure closes the "
            "segment axis with a measured number rather than an assumption."
        ),
    }
    save_json(report, OUT)
    print(json.dumps({
        "segment_sizes": report["segment_sizes"],
        "reps": report["family_representatives"],
        "top": [(t["family"], t["segment"], round(t["mean_fold_delta"], 7), t["folds_positive"],
                 round(t["t"], 2), round(t["holm_adjusted_p"], 4), t["passes"]) for t in tests[:10]],
        "any_pass": report["any_pass"],
    }, indent=1))


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"refuse to overwrite {OUT}")
    main()
