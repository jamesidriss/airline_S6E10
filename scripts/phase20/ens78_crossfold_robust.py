"""P20-ENS78-CROSS-FOLD-ROBUST-ENSEMBLE -- prereg 86c623ff, frozen before results.

Saved-OOF study, no new training. Greedy forward selection (max 8 members)
maximizing the MINIMUM per-fold AUC of the equal-logit blend across folds 1-4;
fold 0 never used for selection. Then evaluate the robust blend on fold 0.

Frozen decision rule: port_delta >= +5e-5 AND corr_vs_portfolio < 0.995.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/ens78_crossfold_robust.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402
from src.submission.store import list_all, load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "ens78_crossfold_robust"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"
PREREG_SHA = "86c623ff8d2f9c88692bbd9ea9469b08f33f125905840bc3a8d8b08ce3739bf2"
PORTFOLIO_REF = "sol_route_aux10_primary_final"
MAX_MEMBERS = 8
ELIGIBLE_FAMILIES = {"lgbm", "xgb", "cat", "realmlp", "tabm", "lgbm_soft"}


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def main() -> int:
    ds = load_raw()
    y = ds.train[TARGET].to_numpy()
    folds = np.load(FOLDS_PATH)
    eval_m = folds == 0

    # Build member pool: eligible single-model OOFs, no duplicates, no blends.
    pool = {}
    for it in list_all():
        if it.get("family") not in ELIGIBLE_FAMILIES:
            continue
        if it.get("dup"):
            continue
        eid = it["exp_id"]
        try:
            oof = load_oof(eid)
        except Exception:
            continue
        if oof is None or len(oof) != len(y):
            continue
        if not np.all(np.isfinite(oof)) or oof.min() <= 0 or oof.max() >= 1:
            continue
        pool[eid] = logit(oof.astype(np.float64))
    # also include the portfolio reference members? No: route_ref is the benchmark.
    pool.pop(PORTFOLIO_REF, None)
    ids = sorted(pool)
    print(f"member pool: {len(ids)}")

    train_folds = [1, 2, 3, 4]

    def fold_aucs(mat: np.ndarray) -> np.ndarray:
        out = []
        for k in train_folds:
            m = folds == k
            out.append(roc_auc_score(y[m], mat[m]))
        return np.array(out)

    # Precompute fold masks for speed
    masks = {k: folds == k for k in train_folds}

    # Greedy forward selection maximizing minimum fold AUC (folds 1-4 only).
    chosen: list[str] = []
    cur = None
    best_min = -1.0
    trace = []
    for step in range(MAX_MEMBERS):
        cand_best, cand_auc, cand_cur = None, best_min, None
        for eid in ids:
            if eid in chosen:
                continue
            trial = np.column_stack([pool[eid]]) if cur is None else np.concatenate(
                [cur, pool[eid][:, None]], axis=1)
            blended = trial.mean(axis=1)
            fa = np.array([roc_auc_score(y[masks[k]], blended[masks[k]]) for k in train_folds])
            mn = float(fa.min())
            if mn > cand_auc + 1e-9:
                cand_auc, cand_best, cand_cur, cand_fa = mn, eid, trial, fa
        if cand_best is None:
            break
        chosen.append(cand_best)
        cur = cand_cur
        best_min = cand_auc
        trace.append({"step": step + 1, "added": cand_best,
                      "min_fold_auc_folds1_4": round(cand_auc, 6),
                      "fold_aucs_folds1_4": [round(v, 6) for v in cand_fa.tolist()]})
    robust_blend_train = cur.mean(axis=1)

    # Fold-0 evaluation of the robust selection.
    robust_fold0 = float(roc_auc_score(y[eval_m], robust_blend_train[eval_m]))
    port = load_oof(PORTFOLIO_REF)[eval_m]
    auc_port = float(roc_auc_score(y[eval_m], port))
    pd78 = robust_fold0 - auc_port
    corr = float(spearmanr(robust_blend_train[eval_m], port).statistic)
    prom = bool(pd78 >= 5e-5 and corr < 0.995)

    # Diagnostic-only: same-size subset greedily selected ON fold-0 (cheating).
    cheat: list[str] = []
    ccur = None
    for _ in range(len(chosen)):
        best, best_auc, bcur = None, -1.0, None
        for eid in ids:
            if eid in cheat:
                continue
            trial = np.column_stack([pool[eid]]) if ccur is None else np.concatenate(
                [ccur, pool[eid][:, None]], axis=1)
            auc = roc_auc_score(y[eval_m], trial.mean(axis=1)[eval_m])
            if auc > best_auc:
                best_auc, best, bcur = auc, eid, trial
        cheat.append(best)
        ccur = bcur
    cheat_fold0 = float(roc_auc_score(y[eval_m], ccur.mean(axis=1)[eval_m]))

    rep = {
        "status": "DONE",
        "prereg": "configs/phase20/ens78_crossfold_robust_prereg.json",
        "prereg_sha256": PREREG_SHA,
        "metrics": {
            "pool_size": len(ids),
            "n_selected": len(chosen),
            "selected_members": chosen,
            "min_fold_auc_folds1_4": round(best_min, 6),
            "robust_blend_fold0_auc": round(robust_fold0, 6),
            "fold0_greedy_cheat_auc_diagnostic_only": round(cheat_fold0, 6),
            "robustness_vs_peak_gap": round(cheat_fold0 - robust_fold0, 6),
            "portfolio_fold0_auc": round(auc_port, 6),
            "port_delta": round(pd78, 8),
            "candidate_spearman_vs_portfolio": round(corr, 6),
            "promising_frozen_rule": prom,
            "eval_rows": int(eval_m.sum()),
        },
        "eval_rows": int(eval_m.sum()),
        "selection_trace": trace,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ens78_crossfold_robust.json").write_text(
        json.dumps(rep, indent=1) + "\n", encoding="utf-8")

    m = rep["metrics"]
    print(f"selected {m['n_selected']} members, min-fold(1-4) AUC {m['min_fold_auc_folds1_4']}")
    print(f"robust fold0={m['robust_blend_fold0_auc']} cheat_fold0={m['fold0_greedy_cheat_auc_diagnostic_only']} "
          f"gap={m['robustness_vs_peak_gap']:+.4f}")
    print(f"port_delta={m['port_delta']:+.3e} corr={m['candidate_spearman_vs_portfolio']} -> PROMISING={prom}")
    print(f"report -> {(OUT / 'ens78_crossfold_robust.json').relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
