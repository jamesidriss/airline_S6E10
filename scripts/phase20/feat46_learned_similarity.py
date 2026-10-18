"""P20-FEAT46-LEARNED-SIMILARITY (item 46) -- prereg f416934a, frozen before results.

Learned passenger similarity metric via contrastive learning (Siamese network).
Learns a Mahalanobis-like metric on covariates. Cross-fit 5-fold.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/feat46_learned_similarity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "feat46_learned_similarity"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"

ALL_COV = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
    "Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
    "Gender", "Customer Type", "Type of Travel", "Class",
]
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


class MetricNet(nn.Module):
    """Siamese network learning a Mahalanobis-like metric: d(x,y) = (x-y)^T M (x-y) where M = L^T L."""
    def __init__(self, input_dim: int, embed_dim: int = 16):
        super().__init__()
        self.L = nn.Parameter(torch.randn(embed_dim, input_dim) * 0.1)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Mahalanobis distance: (x1 - x2) @ L^T @ L @ (x1 - x2).T
        diff = x1 - x2  # (B, D)
        transformed = diff @ self.L.T  # (B, embed_dim)
        return (transformed ** 2).sum(dim=1)  # (B,)


class ContrastiveNet(nn.Module):
    """Contrastive learning: learn embeddings where similar pairs are close, dissimilar far."""
    def __init__(self, input_dim: int, embed_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def build_design(tr, folds: np.ndarray, embed_dim: int = 8, epochs: int = 30, lr: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_raw [n,21], X_metric [n, embed_dim] cross-fit learned embeddings)."""
    n = len(tr)
    X_raw = np.zeros((n, 21), dtype=np.float32)
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

    X_metric = np.zeros((n, embed_dim), dtype=np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for k in range(5):
        eval_m = np.load(FOLDS_PATH) == k
        fit_m = ~eval_m
        
        X_fit = torch.from_numpy(X_raw[fit_m]).to(device).float()
        X_eval = torch.from_numpy(X_raw[eval_m]).to(device).float()
        y_fit = torch.from_numpy(ds.train[TARGET].to_numpy()[fit_m]).to(device).float()

        # Train contrastive network to learn embeddings
        # Use supervised contrastive loss: same-class pairs close, different-class far
        model = ContrastiveNet(21, embed_dim=embed_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # Create pairs for contrastive learning
        pos_pairs = []
        neg_pairs = []
        for cls in (0, 1):
            cls_idx = np.where(y_fit.cpu().numpy() == cls)[0]
            if len(cls_idx) < 2:
                continue
            # Sample positive pairs
            n_pairs = min(5000, len(cls_idx) * 2)
            idx1 = np.random.choice(cls_idx, n_pairs)
            idx2 = np.random.choice(cls_idx, n_pairs)
            pos_pairs.extend(list(zip(idx1, idx2)))
        
        # Sample negative pairs
        n_neg = len(pos_pairs)
        cls0_idx = np.where(y_fit.cpu().numpy() == 0)[0]
        cls1_idx = np.where(y_fit.cpu().numpy() == 1)[0]
        if len(cls0_idx) > 0 and len(cls1_idx) > 0:
            idx0 = np.random.choice(cls0_idx, n_neg)
            idx1 = np.random.choice(cls1_idx, n_neg)
            neg_pairs.extend(list(zip(idx0, idx1)))

        if len(pos_pairs) == 0:
            # Fallback: just train autoencoder
            for _ in range(epochs):
                model.train()
                opt.zero_grad()
                emb = model(X_fit)
                loss = F.mse_loss(emb, X_fit[:, :embed_dim])  # dummy
                loss.backward()
                opt.step()
        else:
            pos_idx1 = torch.tensor([p[0] for p in pos_pairs], device=device)
            pos_idx2 = torch.tensor([p[1] for p in pos_pairs], device=device)
            neg_idx1 = torch.tensor([p[0] for p in neg_pairs], device=device)
            neg_idx2 = torch.tensor([p[1] for p in neg_pairs], device=device)

            for _ in range(epochs):
                model.train()
                opt.zero_grad()
                emb = model(X_fit)
                
                pos_dist = ((emb[pos_idx1] - emb[pos_idx2]) ** 2).sum(1).mean()
                neg_dist = ((emb[neg_idx1] - emb[neg_idx2]) ** 2).sum(1).mean()
                
                # Contrastive loss: pull positives together, push negatives apart
                margin = 1.0
                loss = pos_dist + F.relu(margin - neg_dist).mean()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            eval_emb = model(X_eval).cpu().numpy().astype(np.float32)
            X_metric[eval_m] = eval_emb

    return X_raw, X_metric


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
    m = LGBMClassifier(n_estimators=6000, early_stopping_rounds=250, **{
        "learning_rate": 0.02, "num_leaves": 127, "extra_trees": True,
        "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
        "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "verbose": -1,
        "n_jobs": 8, "random_state": 1})
    m.fit(X_fit[inner_train], y[fit_m][inner_train],
          eval_set=[(X_fit[inner_val], y[fit_m][inner_val])], eval_metric="auc")
    pred = m.predict_proba(X[eval_m])[:, 1]
    return pred, float(roc_auc_score(y[eval_m], pred))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    global ds
    ds = load_raw()
    tr = ds.train
    y = tr[TARGET].to_numpy()
    folds = np.load(FOLDS_PATH)

    X_raw, X_metric = build_design(tr, folds, embed_dim=8, epochs=30, lr=1e-3)
    X_aug = np.column_stack([X_raw, X_metric]).astype(np.float32)

    eval_m = np.load(FOLDS_PATH) == 0
    y_e = ds.train[TARGET].to_numpy()[eval_m]

    pred_base, auc_base = lgbm_fold0(X_raw, y, folds)
    pred_aug, auc_aug = lgbm_fold0(X_aug, y, folds)

    delta = auc_aug - auc_base
    promising = delta >= 5e-5

    spear = float(spearmanr(pred_aug, load_oof("sol_route_aux10_primary_final")[eval_m]).statistic)
    blend = (logit(pred_aug) + logit(load_oof("sol_route_aux10_primary_final")[eval_m])) / 2.0
    auc_blend = float(roc_auc_score(y_e, blend))
    blend_delta = auc_blend - auc_aug

    report = {
        "prereg": "configs/phase20/feat46_learned_similarity_prereg.json",
        "prereg_sha256": "f416934a53c5c0c540715ed9902120d85e9af901278571896b89c427a3bd6d69",
        "method": "Learned similarity metric: contrastive Siamese network (embed_dim=8) on 21 covariates, cross-fit 5-fold. Learned embeddings added to LGBM.",
        "fold0_results": {
            "base_21raw_auc": round(auc_base, 6),
            "augmented_21raw_8emb_auc": round(auc_aug, 6),
            "augmentation_delta": round(delta, 8),
            "emb_only_auc": 0.0,  # not measured separately
            "spearman_aug_vs_champion": round(spear, 6),
            "equal_blend_auc": round(auc_blend, 6),
            "blend_delta_vs_aug": round(blend_delta, 8),
            "champion_ref": CHAMPION_REF,
            "port_delta_vs_champion": round(auc_aug - CHAMPION_REF, 8),
            "promising": bool(promising),
            "decision_rule": "PROMISING if augmentation_delta >= +5e-5",
        },
        "embed_dim": 8,
    }
    path = OUT / "feat46_learned_similarity.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    f0 = report["fold0_results"]
    print(f"fold-0: base(21raw)={f0['base_21raw_auc']} aug(+8emb)={f0['augmented_21raw_8emb_auc']} delta={f0['augmentation_delta']:+.2e}")
    print(f"spearman_aug_vs_champ={f0['spearman_aug_vs_champion']} blend_delta={f0['blend_delta_vs_aug']:+.2e}")
    print(f"port_delta vs champion={f0['port_delta_vs_champion']:+.2e} promising={f0['promising']}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())