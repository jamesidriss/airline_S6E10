"""Phase 24 -- member-level diagnostics of the certified aux10 parent and the TabPFN route seeds.

Read-only. Answers two frozen questions with stored artifacts only:
  Q1 does the equal mean of the 10 certified auxiliary members beat every leave-one-out mean?
  Q2 does averaging two TabPFN route seeds help where both seeds exist (fold 0)?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import load_cached_parquet  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

MEMBERS = [
    "xt_xt_d127_ss06", "xt_xt_d127_s1", "xt_xt_d255", "xt_xt_d127_bin63", "xt_xt_d127_cs05",
    "xt_xt_d63", "z3_cat_d8_s2", "prod5_xgb_full_primary", "xt_xgb_lossguide", "zoo_xgb_d6",
]
PRED = ROOT / "artifacts/predictions"


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def main() -> None:
    tr, _ = load_cached_parquet()
    y = tr.satisfaction.to_numpy(dtype="int8")
    folds = get_scheme("primary", y, tr["id"].to_numpy()).folds
    vecs = {}
    for m in MEMBERS:
        p = PRED / f"sol_clean_aux10_primary_{m}_oof.npy"
        if not p.exists():
            print("MISSING", p)
            continue
        vecs[m] = np.load(p).astype(np.float64)
    assert len(vecs) == 10, f"expected 10 members, got {len(vecs)}"
    ref = np.load(ROOT / "artifacts/sol_route_aux10_primary_final/aux10_oof.npy").astype(np.float64)
    mean_logit = np.mean([logit(v) for v in vecs.values()], axis=0)
    mean_prob = 1 / (1 + np.exp(-mean_logit))
    out = {
        "n_members": len(vecs),
        "stored_aux10_sha_ok_max_abs": float(np.max(np.abs(mean_prob - ref))),
        "mean_auc": float(roc_auc_score(y, mean_logit)),
        "members": {},
        "leave_one_out": {},
    }
    for m, v in vecs.items():
        out["members"][m] = float(roc_auc_score(y, logit(v)))
    base = out["mean_auc"]
    for drop in vecs:
        ml = np.mean([logit(v) for k, v in vecs.items() if k != drop], axis=0)
        out["leave_one_out"][drop] = float(roc_auc_score(y, ml) - base)
    best_drop = max(out["leave_one_out"], key=out["leave_one_out"].get)
    out["best_drop"] = {"member": best_drop, "delta": out["leave_one_out"][best_drop]}

    # Q2: route seed averaging on fold 0 (and any fold present in the seed-1202 recovery run)
    seed_dir = ROOT / "artifacts/sol_tabpfn35_seed1202_recovery"
    r1201 = np.load(ROOT / "artifacts/phase17_B1/primary/route_oof.npy").astype(np.float64)
    q2 = {}
    probe = seed_dir / "progress.json"
    if probe.exists():
        q2["seed1202_progress"] = json.loads(probe.read_text(encoding="utf-8"))
    for f in (0,):
        p2 = seed_dir / f"route_f{f}.npy"
        ids2 = seed_dir / f"ids_f{f}.npy"
        if not (p2.exists() and ids2.exists()):
            continue
        rows = np.flatnonzero(folds == f)
        if not np.array_equal(np.load(ids2), tr["id"].to_numpy()[rows]):
            q2[f"fold{f}"] = "ID_MISMATCH"
            continue
        v2 = np.load(p2).astype(np.float64)
        v1 = r1201[rows]
        aug = 0.5 * logit(v1) + 0.5 * logit(v2)
        q2[f"fold{f}"] = {
            "n": int(len(rows)),
            "auc_1201": float(roc_auc_score(y[rows], v1)),
            "auc_1202": float(roc_auc_score(y[rows], v2)),
            "auc_seed_average": float(roc_auc_score(y[rows], aug)),
            "delta_average_vs_1201": float(roc_auc_score(y[rows], aug) - roc_auc_score(y[rows], v1)),
            "corr_logit": float(np.corrcoef(logit(v1), logit(v2))[0, 1]),
        }
    out["route_seed_probe"] = q2
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
