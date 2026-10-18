"""P20-CLASS61-GENUINE-EBM -- prereg 259dfa2e, frozen before results.

Genuine EBM (ExplainableBoostingClassifier, interpret 0.7.8) with 16 real
pairwise interactions, trained on outer-FIT (folds 1-4) only, evaluated on
fold-0; equal-logit blend with route reference as the frozen-rule candidate.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/class61_genuine_ebm.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "class61_genuine_ebm"
ART = ROOT / "artifacts" / "phase20" / "class61_genuine_ebm"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"
PREREG_SHA = "259dfa2ed06a766c36e66a2a33360df11970c042af1e26185a7b63a62226d619"
PORTFOLIO_REF = "sol_route_aux10_primary_final"

ALL_COV = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
    "Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
    "Gender", "Customer Type", "Type of Travel", "Class",
]
CAT_COLS = {"Gender", "Customer Type", "Type of Travel", "Class"}


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def main() -> int:
    from interpret.glassbox import ExplainableBoostingClassifier
    sha = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    assert sha == PREREG_SHA, f"prereg sha mismatch {sha}"

    ds = load_raw()
    tr = ds.train
    y = tr[TARGET].to_numpy()
    folds = np.load(FOLDS_PATH)
    eval_m = folds == 0
    fit_m = ~eval_m

    X = np.zeros((len(tr), 21), dtype=np.float32)
    for j, c in enumerate(ALL_COV):
        col = tr[c]
        if col.dtype == object or str(col.dtype).startswith("str"):
            X[:, j] = col.astype("category").cat.codes.astype(np.float32)
        else:
            X[:, j] = col.astype(np.float32)
    for j in range(21):
        bad = ~np.isfinite(X[:, j])
        if bad.any():
            X[bad, j] = np.nanmedian(X[~bad, j])

    t0 = time.time()
    m = ExplainableBoostingClassifier(
        max_bins=256, interactions=16, max_interaction_bins=16,
        learning_rate=0.01, max_leaves=11, min_samples_leaf=6,
        outer_bags=4, inner_bags=0, random_state=1, n_jobs=8)
    m.fit(X[fit_m], y[fit_m])
    fit_secs = round(time.time() - t0, 1)

    pred = m.predict_proba(X[eval_m])[:, 1]
    auc = float(roc_auc_score(y[eval_m], pred))

    port = load_oof(PORTFOLIO_REF)[eval_m]
    y_e = y[eval_m]
    auc_port = float(roc_auc_score(y_e, port))
    blend = (logit(pred) + logit(port)) / 2.0
    auc_blend = float(roc_auc_score(y_e, blend))
    pd61 = auc_blend - auc_port
    corr = float(spearmanr(pred, port).statistic)
    prom = bool(pd61 >= 5e-5 and corr < 0.995)

    rep = {
        "status": "DONE",
        "prereg": "configs/phase20/class61_genuine_ebm_prereg.json",
        "prereg_sha256": sha,
        "protocol": "EBM (interpret 0.7.8) interactions=16, outer_bags=4, outer-FIT only; "
                    "fold-0 evaluation; equal-logit blend with route reference.",
        "metrics": {
            "ebm_fold0_auc": round(auc, 6),
            "portfolio_fold0_auc": round(auc_port, 6),
            "equal_logit_blend_fold0_auc": round(auc_blend, 6),
            "port_delta": round(pd61, 8),
            "candidate_spearman_vs_portfolio": round(corr, 6),
            "promising_frozen_rule": prom,
            "eval_rows": int(eval_m.sum()),
            "fit_secs": fit_secs,
        },
        "eval_rows": int(eval_m.sum()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "class61_genuine_ebm.json").write_text(
        json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    ART.mkdir(parents=True, exist_ok=True)
    np.save(ART / "fold0_candidate.npy", pred.astype(np.float32))

    mm = rep["metrics"]
    print(f"EBM fold0={mm['ebm_fold0_auc']} port={mm['portfolio_fold0_auc']} "
          f"blend={mm['equal_logit_blend_fold0_auc']}")
    print(f"port_delta={mm['port_delta']:+.3e} corr={mm['candidate_spearman_vs_portfolio']} "
          f"-> PROMISING={prom} (fit {fit_secs}s)")
    print(f"report -> {(OUT / 'class61_genuine_ebm.json').relative_to(ROOT).as_posix()}")
    return 0


PREREG_PATH = ROOT / "configs" / "phase20" / "class61_genuine_ebm_prereg.json"

if __name__ == "__main__":
    raise SystemExit(main())
