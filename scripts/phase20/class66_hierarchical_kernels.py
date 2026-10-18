"""P20-CLASS66-HIERARCHICAL-KERNELS -- prereg 83982b50, frozen before results.

Per-segment Nadaraya-Watson kernel regression on the 21 standardized covariates:
bandwidth per (Type of Travel x Class) segment = median 50-NN distance within
segment, computed on outer-FIT rows only. Fold-0 rows only query.

Candidate = equal-logit blend of kernel probability with route reference.
Frozen decision rule: port_delta >= +5e-5 AND corr < 0.995.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/class66_hierarchical_kernels.py
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "class66_hierarchical_kernels"
ART = ROOT / "artifacts" / "phase20" / "class66_hierarchical_kernels"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"
PREREG = ROOT / "configs" / "phase20" / "class66_hierarchical_kernels_prereg.json"
PREREG_SHA = "83982b504f9ee4a904fa3bddbec775eb8a2642d9bc5974d26f71ef4f59aa8c50"
PORTFOLIO_REF = "sol_route_aux10_primary_final"
K = 50
SEG_COLS = ["Type of Travel", "Class"]

ALL_COV = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
    "Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
    "Gender", "Customer Type", "Type of Travel", "Class",
]


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def main() -> int:
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
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

    seg = np.zeros(len(tr), dtype=np.int64)
    for i, c in enumerate(SEG_COLS):
        seg = seg * 8 + tr[c].astype("category").cat.codes.to_numpy().astype(np.int64)

    scaler = StandardScaler().fit(X[fit_m])
    Xs = scaler.transform(X).astype(np.float32)

    p_kernel = np.zeros(int(eval_m.sum()), dtype=np.float32)
    seg_info = {}
    t0 = time.time()
    eval_idx = np.where(eval_m)[0]
    pos = 0
    for s in np.unique(seg[fit_m]):
        f_idx = np.where(fit_m & (seg == s))[0]
        e_local = np.where(seg[eval_m] == s)[0]
        if len(e_local) == 0:
            continue
        nn = NearestNeighbors(n_neighbors=min(K, len(f_idx)), n_jobs=8)
        nn.fit(Xs[f_idx])
        d, ind = nn.kneighbors(Xs[eval_idx[e_local]], return_distance=True)
        h = float(np.median(d[:, -1]))  # in-segment median kNN distance = bandwidth
        w = np.exp(-(d ** 2) / max(h ** 2, 1e-9))
        w /= w.sum(axis=1, keepdims=True)
        # row-wise Nadaraya-Watson weighted mean (NOT a matrix product)
        p = (w * y[f_idx][ind].astype(np.float64)).sum(axis=1)
        p_kernel[e_local] = p.astype(np.float32)
        seg_info[str(int(s))] = {"fit_rows": int(len(f_idx)), "eval_rows": int(len(e_local)),
                                 "bandwidth": round(h, 4)}
        pos += len(e_local)
    assert pos == int(eval_m.sum()), f"coverage {pos} != {int(eval_m.sum())}"
    kernel_secs = round(time.time() - t0, 1)

    y_e = y[eval_m]
    auc_kernel = float(roc_auc_score(y_e, p_kernel))
    port = load_oof(PORTFOLIO_REF)[eval_m]
    auc_port = float(roc_auc_score(y_e, port))
    blend = (logit(p_kernel) + logit(port)) / 2.0
    auc_blend = float(roc_auc_score(y_e, blend))
    pd66 = auc_blend - auc_port
    corr = float(spearmanr(p_kernel, port).statistic)
    prom = bool(pd66 >= 5e-5 and corr < 0.995)

    rep = {
        "status": "DONE",
        "prereg": "configs/phase20/class66_hierarchical_kernels_prereg.json",
        "prereg_sha256": sha,
        "protocol": "per-segment (Type of Travel x Class) Nadaraya-Watson kernel regression, "
                    "K=50, bandwidth = in-segment median kNN distance, outer-FIT only; "
                    "equal-logit blend with route reference.",
        "metrics": {
            "kernel_fold0_auc": round(auc_kernel, 6),
            "portfolio_fold0_auc": round(auc_port, 6),
            "equal_logit_blend_fold0_auc": round(auc_blend, 6),
            "port_delta": round(pd66, 8),
            "candidate_spearman_vs_portfolio": round(corr, 6),
            "promising_frozen_rule": prom,
            "eval_rows": int(eval_m.sum()),
            "kernel_secs": kernel_secs,
            "segments": seg_info,
        },
        "eval_rows": int(eval_m.sum()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "class66_hierarchical_kernels.json").write_text(
        json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    ART.mkdir(parents=True, exist_ok=True)
    np.save(ART / "fold0_candidate.npy", p_kernel)

    mm = rep["metrics"]
    print(f"kernel fold0={mm['kernel_fold0_auc']} port={mm['portfolio_fold0_auc']} "
          f"blend={mm['equal_logit_blend_fold0_auc']}")
    print(f"port_delta={mm['port_delta']:+.3e} corr={mm['candidate_spearman_vs_portfolio']} "
          f"-> PROMISING={prom} ({kernel_secs}s)")
    print(f"report -> {(OUT / 'class66_hierarchical_kernels.json').relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
