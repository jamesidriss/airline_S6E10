"""P20-FEAT52-RATING-FACTORIZATION (item 52) -- prereg c53890fd, frozen before results.

Supervised low-rank rating factorization: learn passenger embeddings (rank 8)
from the 13 ratings, supervised by satisfaction. Cross-fit 5-fold.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/feat52_rating_factorization.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "feat52_rating_factorization"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"

RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
ALL_COV = RATING_COLS + ["Age", "Flight Distance", "Departure Delay in Minutes",
                         "Arrival Delay in Minutes", "Gender", "Customer Type",
                         "Type of Travel", "Class"]
assert len(ALL_COV) == 21

LGBM_PARAMS = {
    "learning_rate": 0.02, "num_leaves": 127, "extra_trees": True,
    "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
    "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "verbose": -1,
    "n_jobs": 8, "random_state": 1,
}

CHAMPION_REF = 0.9618529228157235


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class RatingFactorizer(nn.Module):
    """Supervised low-rank factorization: logit = w·x + u·v·x, rank 8."""
    def __init__(self, n_ratings: int, rank: int = 8):
        super().__init__()
        # rating embeddings (13, rank)
        self.U = nn.Parameter(torch.randn(n_ratings, rank) * 0.1)
        # satisfaction head
        self.w = nn.Linear(rank, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 13) ratings
        # embed: (B, 13) @ (13, rank) = (B, rank)
        emb = x @ self.U
        return self.w(emb).squeeze(-1)


def build_design(tr, folds: np.ndarray, rank: int = 8, epochs: int = 50, lr: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_raw [n,21], X_emb [n, rank] cross-fit passenger embeddings)."""
    n = len(tr)
    X_raw = np.zeros((len(tr), 21), dtype=np.float32)
    for j, c in enumerate(ALL_COV):
        col = tr[c]
        if col.dtype == object or str(col.dtype).startswith("str"):
            X_raw[:, j] = col.astype("category").cat.codes.astype(np.float32)
        else:
            X_raw[:, j] = col.astype(np.float32)
    for j in range(21):
        col = X_raw[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad])

    rating_idx = [ALL_COV.index(c) for c in ALL_COV[:13]]  # first 13 are ratings
    X_r = X_raw[:, :13].astype(np.float32)  # first 13 cols are ratings

    X_emb = np.zeros((len(tr), 8), dtype=np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for k in range(5):
        eval_m = np.load(FOLDS_PATH) == k
        fit_m = ~eval_m

        X_fit = torch.from_numpy(X_r[np.where(fit_m)[0]]).to(device).float()
        y_fit = torch.from_numpy(ds.train[TARGET].to_numpy()[fit_m]).to(device).float()
        X_eval = torch.from_numpy(X_r[np.where(eval_m)[0]]).to(device).float()

        model = RatingFactorizer(13, rank=8).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for _ in range(50):
            model.train()
            opt.zero_grad()
            logits = model(X_fit)
            loss = F.binary_cross_entropy_with_logits(logits, y_fit)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            eval_emb = (X_eval @ model.U).cpu().numpy().astype(np.float32)
            X_emb[np.where(eval_m)[0]] = eval_emb

    # Also get train embeddings for the fit fold (use a model trained on all train)
    # Not needed for fold-0 screen; fold-0 eval uses cross-fit embeddings

    return X_raw, X_emb


def lgbm_fold0(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, float]:
    from lightgbm import LGBMClassifier
    eval_m = folds == 0
    fit_m = ~eval_m
    rng = np.random.default_rng(1)
    yfit = y[fit_m]
    val_parts = []
    for cls in (0, 1):
        ci = np.where(yfit == cls)[0]
        nv = max(1, int(round(0.1 * len(ci))))
        val_parts.append(rng.choice(ci, nv, replace=False))
    inner_val = np.concatenate(val_parts)
    inner_train = np.setdiff1d(np.arange(fit_m.sum()), inner_val)
    X_fit = X[fit_m]
    m = LGBMClassifier(n_estimators=6000, early_stopping_rounds=250, **LGBM_PARAMS)
    m.fit(X_fit[inner_train], yfit[inner_train],
          eval_set=[(X_fit[inner_val], yfit[inner_val])], eval_metric="auc")
    pred = m.predict_proba(X[eval_m])[:, 1]
    return pred, float(roc_auc_score(y[eval_m], pred))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    global ds
    ds = load_raw()
    tr = ds.train
    y = tr[TARGET].to_numpy()
    folds = np.load(FOLDS_PATH)

    X_raw, X_emb = build_design(tr, np.load(FOLDS_PATH))
    X_aug = np.column_stack([X_raw, X_emb[:, :8]]).astype(np.float32)

    eval_m = np.load(FOLDS_PATH) == 0
    y_e = ds.train[TARGET].to_numpy()[eval_m]

    pred_base, auc_base = lgbm_fold0(X_raw, y, folds)
    pred_aug, auc_aug = lgbm_fold0(X_aug, y, folds)

    delta = auc_aug - auc_base
    promising = delta >= 5e-5

    from scipy.stats import spearmanr
    from src.submission.store import load_oof
    champ_ref = load_oof("sol_route_aux10_primary_final")[eval_m]
    spear = float(spearmanr(pred_aug, champ_ref).statistic)
    blend = (logit(pred_aug) + logit(champ_ref)) / 2.0
    auc_blend = float(roc_auc_score(y_e, blend))
    blend_delta = auc_blend - auc_aug

    report = {
        "prereg": "configs/phase20/feat52_rating_factorization_prereg.json",
        "prereg_sha256": "c53890fd05de98a989d599d1a55463556d480cb145fb110dd18de75dd64b0acc",
        "method": "Supervised low-rank factorization (rank 8) of 13 ratings, cross-fit 5-fold, PyTorch/AdamW",
        "fold0_results": {
            "base_21raw_auc": round(auc_base, 6),
            "augmented_21raw_8emb_auc": round(auc_aug, 6),
            "augmentation_delta": round(auc_aug - auc_base, 8),
            "spearman_aug_vs_champion": round(spear, 6),
            "equal_blend_auc": round(auc_blend, 6),
            "blend_delta_vs_aug": round(auc_blend - auc_aug, 8),
            "champion_ref": CHAMPION_REF,
            "port_delta_vs_champion": round(auc_aug - CHAMPION_REF, 8),
            "promising": bool(promising),
        },
    }
    path = OUT / "feat52_rating_factorization.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    f0 = report["fold0_results"]
    print(f"fold-0: base(21raw)={f0['base_21raw_auc']} "
          f"aug(+8emb)={f0['augmented_21raw_8emb_auc']} "
          f"delta={f0['augmentation_delta']:+.2e}")
    print(f"spearman_aug_vs_champ={f0['spearman_aug_vs_champion']} "
          f"blend_delta={f0['blend_delta_vs_aug']:+.2e}")
    print(f"port_delta vs champion={f0['port_delta_vs_champion']:+.2e} "
          f"promising={f0['promising']}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())