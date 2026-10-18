"""P20-GEN84 + P20-GEN85 -- preregs 9d604575 / d0ff0d4a, frozen before results.

GEN84: conditional transfer from the ORIGINAL dataset. A source LGBM is trained
  on all 129,880 original rows (original labels only; zero competition labels).
  Its prediction on competition fold-0 rows is transferred with a per-row weight
  w = clip(1 - d20/d95, 0, 1) from a 20-NN distance to the original training
  set (label-free gate). candidate = w*logit(p_src) + (1-w)*logit(p_route).
GEN85: per-segment source-trained specialists (Type of Travel x Class), each
  trained on in-segment original rows only, stitched onto fold-0 rows, equal
  logit blend with route reference.

Frozen decision rule (both): port_delta >= +5e-5 AND corr_vs_portfolio < 0.995.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/gen84_gen85_original_transfer.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402
from src.features.s6e10 import load_original  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT84 = ROOT / "reports" / "phase20" / "gen84_original_transfer"
OUT85 = ROOT / "reports" / "phase20" / "gen85_source_specialists"
ART = ROOT / "artifacts" / "phase20" / "gen84_gen85"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"

COV21 = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
    "Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
    "Gender", "Customer Type", "Type of Travel", "Class",
]
PORTFOLIO_REF = "sol_route_aux10_primary_final"
SEG_COLS = ["Type of Travel", "Class"]

PR84 = "9d604575cbfbff9010a80496ee93d25f652e3f2d96e351ccea7a95504891fee3"
PR85 = "d0ff0d4a2de79ac995e047b86f1d227ebd9d083795c9a39827f09d63772cbe14"


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def encode_pair(og, comb):
    """Joint integer encoding of original + competition frames (object cols)."""
    Xo = np.zeros((len(og), 21), dtype=np.float32)
    Xc = np.zeros((len(comb), 21), dtype=np.float32)
    for j, c in enumerate(COV21):
        o, cc = og[c], comb[c]
        if o.dtype == object or str(o.dtype).startswith("str"):
            cats = np.unique(np.concatenate([o.astype(str).values, cc.astype(str).values]))
            lut = {v: i for i, v in enumerate(cats)}
            Xo[:, j] = o.astype(str).map(lut).to_numpy(np.float32)
            Xc[:, j] = cc.astype(str).map(lut).to_numpy(np.float32)
        else:
            Xo[:, j] = o.to_numpy(np.float32)
            Xc[:, j] = cc.to_numpy(np.float32)
    for arr in (Xo, Xc):
        for j in range(21):
            bad = ~np.isfinite(arr[:, j])
            if bad.any():
                med = np.nanmedian(arr[~bad, j]) if (~bad).any() else 0.0
                arr[bad, j] = med
    return Xo, Xc


def src_lgbm(Xo, yo, Xc, seed: int = 1, min_rows: int = 300) -> np.ndarray:
    from lightgbm import LGBMClassifier
    if len(yo) < min_rows:
        return None
    rng = np.random.default_rng(seed)
    parts = []
    for cls in (0, 1):
        ci = np.where(yo == cls)[0]
        nv = max(1, int(round(0.1 * len(ci))))
        parts.append(rng.choice(ci, nv, replace=False))
    inner_val = np.concatenate(parts)
    inner_train = np.setdiff1d(np.arange(len(yo)), inner_val)
    m = LGBMClassifier(n_estimators=6000, early_stopping_rounds=250, **{
        "learning_rate": 0.02, "num_leaves": 127, "extra_trees": True,
        "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
        "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "verbose": -1,
        "n_jobs": 8, "random_state": seed})
    m.fit(Xo[inner_train], yo[inner_train],
          eval_set=[(Xo[inner_val], yo[inner_val])], eval_metric="auc")
    return m.predict_proba(Xc)[:, 1]


def main() -> int:
    ds = load_raw()
    comb = ds.train
    y = comb[TARGET].to_numpy()
    folds = np.load(FOLDS_PATH)
    eval_m = folds == 0

    og = load_original()
    yo = og[TARGET].to_numpy().astype(int)
    Xo, Xc = encode_pair(og, comb)

    # --- GEN84: global source model + kNN conditional gate ---
    p_src_all = src_lgbm(Xo, yo, Xc)
    p_src_eval = p_src_all[eval_m]

    scaler = StandardScaler().fit(Xo)
    nn = NearestNeighbors(n_neighbors=20, n_jobs=8)
    nn.fit(scaler.transform(Xo))
    d20, _ = nn.kneighbors(scaler.transform(Xc[eval_m]), return_distance=True)
    d20 = d20[:, -1]
    d95 = float(np.percentile(d20, 95))
    w = np.clip(1.0 - d20 / max(d95, 1e-9), 0.0, 1.0)

    port = load_oof(PORTFOLIO_REF)[eval_m]
    y_e = y[eval_m]
    auc_port = float(roc_auc_score(y_e, port))

    cand84 = w * logit(p_src_eval) + (1.0 - w) * logit(port)
    auc84 = float(roc_auc_score(y_e, cand84))
    blend_equal = (logit(p_src_eval) + logit(port)) / 2.0
    auc_equal = float(roc_auc_score(y_e, blend_equal))
    auc_src_only = float(roc_auc_score(y_e, p_src_eval))
    pd84 = auc84 - auc_port
    corr84 = float(spearmanr(cand84, port).statistic)
    prom84 = bool(pd84 >= 5e-5 and corr84 < 0.995)

    rep84 = {
        "status": "DONE",
        "prereg": "configs/phase20/gen84_original_transfer_prereg.json",
        "prereg_sha256": PR84,
        "protocol": "source LGBM on all 129,880 original rows (original labels only); "
                    "kNN(20) distance gate on covariates (label-free); no competition label "
                    "used in training or gating.",
        "metrics": {
            "source_model_fold0_auc": round(auc_src_only, 6),
            "conditional_candidate_fold0_auc": round(auc84, 6),
            "equal_blend_contrast_auc": round(auc_equal, 6),
            "portfolio_fold0_auc": round(auc_port, 6),
            "port_delta": round(pd84, 8),
            "candidate_spearman_vs_portfolio": round(corr84, 6),
            "gate_weight_mean": round(float(w.mean()), 6),
            "gate_weight_zero_frac": round(float((w == 0).mean()), 6),
            "promising_frozen_rule": prom84,
            "eval_rows": int(eval_m.sum()),
        },
        "eval_rows": int(eval_m.sum()),
    }
    OUT84.mkdir(parents=True, exist_ok=True)
    (OUT84 / "gen84_original_transfer.json").write_text(
        json.dumps(rep84, indent=1) + "\n", encoding="utf-8")

    # --- GEN85: per-segment source-trained specialists ---
    seg_o = np.zeros(len(og), dtype=np.int64)
    seg_c = np.zeros(len(comb), dtype=np.int64)
    for i, c in enumerate(SEG_COLS):
        cats = np.unique(np.concatenate([og[c].astype(str).values,
                                         comb[c].astype(str).values]))
        lut = {v: k for k, v in enumerate(cats)}
        seg_o = seg_o * 8 + og[c].astype(str).map(lut).to_numpy(np.int64)
        seg_c = seg_c * 8 + comb[c].astype(str).map(lut).to_numpy(np.int64)

    p_spec = np.full(len(comb), np.nan, dtype=np.float32)
    seg_report = {}
    global_p = p_src_all
    for s in np.unique(seg_c):
        om = seg_o == s
        cm = seg_c == s
        if om.sum() >= 300:
            p_spec[cm] = src_lgbm(Xo[om], yo[om], Xc[cm])
            seg_report[str(int(s))] = {"orig_rows": int(om.sum()), "model": "specialist"}
        else:
            p_spec[cm] = global_p[cm]
            seg_report[str(int(s))] = {"orig_rows": int(om.sum()), "model": "global-fallback"}
    p_spec_eval = p_spec[eval_m]

    cand85 = (logit(p_spec_eval) + logit(port)) / 2.0
    auc85 = float(roc_auc_score(y_e, cand85))
    auc_spec_only = float(roc_auc_score(y_e, p_spec_eval))
    pd85 = auc85 - auc_port
    corr85 = float(spearmanr(cand85, port).statistic)
    prom85 = bool(pd85 >= 5e-5 and corr85 < 0.995)

    rep85 = {
        "status": "DONE",
        "prereg": "configs/phase20/gen85_source_specialists_prereg.json",
        "prereg_sha256": PR85,
        "protocol": "per-(Type of Travel x Class) LGBM trained on in-segment original rows "
                    "(< 300 rows -> global source fallback); stitched onto fold-0; equal-logit "
                    "blend with route reference. Zero competition labels.",
        "metrics": {
            "specialists_fold0_auc": round(auc_spec_only, 6),
            "candidate_fold0_auc": round(auc85, 6),
            "portfolio_fold0_auc": round(auc_port, 6),
            "port_delta": round(pd85, 8),
            "port_delta_vs_gen84_equal": round(pd85 - (auc_equal - auc_port), 8),
            "candidate_spearman_vs_portfolio": round(corr85, 6),
            "promising_frozen_rule": prom85,
            "eval_rows": int(eval_m.sum()),
            "segments": seg_report,
        },
        "eval_rows": int(eval_m.sum()),
    }
    OUT85.mkdir(parents=True, exist_ok=True)
    (OUT85 / "gen85_source_specialists.json").write_text(
        json.dumps(rep85, indent=1) + "\n", encoding="utf-8")
    ART.mkdir(parents=True, exist_ok=True)
    np.save(ART / "gen84_fold0_candidate.npy", cand84.astype(np.float32))
    np.save(ART / "gen85_fold0_candidate.npy", cand85.astype(np.float32))

    print("GEN84: src_only=%.6f cond=%.6f equal=%.6f port=%.6f pd=%+.3e corr=%.4f -> %s"
          % (auc_src_only, auc84, auc_equal, auc_port, pd84, corr84, prom84))
    print("GEN85: spec_only=%.6f cand=%.6f port=%.6f pd=%+.3e corr=%.4f -> %s"
          % (auc_spec_only, auc85, auc_port, pd85, corr85, prom85))
    print(f"reports -> {OUT84.name}/, {OUT85.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
