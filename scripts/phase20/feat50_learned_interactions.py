"""P20-FEAT50-LEARNED-INTERACTIONS (item 50) -- prereg 7c5e79bb, frozen before results.

Factorization Machine interaction layer on rich view: learns pairwise
interactions via low-rank factorization. Cross-fit 5-fold on rich view.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/feat50_learned_interactions.py
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

OUT = ROOT / "reports" / "phase20" / "feat50_learned_interactions"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"

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


class FactorizationMachine(nn.Module):
    """FM interaction layer: learns pairwise interactions via rank-R factorization."""
    def __init__(self, input_dim: int, rank: int = 8):
        super().__init__()
        self.V = nn.Parameter(torch.randn(input_dim, rank) * 0.01)
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        # FM: w·x + Σ_{i<j} (V_i·V_j) x_i x_j
        # Efficient computation: 0.5 * ( (x·V)^2 - Σ (x_i^2 * V_i^2) )
        xV = x @ self.V  # (B, R)
        linear = self.linear(x).squeeze(-1)
        interaction = 0.5 * (xV.pow(2).sum(1) - (x.pow(2) @ (self.V ** 2).sum(1)))
        return linear + interaction


def build_rich_view(tr, te) -> tuple[np.ndarray, np.ndarray]:
    from src.features.view import ViewBuilder
    from src.common import load_raw
    ds = load_raw()
    vb = ViewBuilder(ds.train, ds.test, "full")
    vb.build_static()
    return vb.static_tr, ds.train["satisfaction"].to_numpy()


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
    ds = load_raw()
    tr, te = ds.train, ds.test
    y = tr["satisfaction"].to_numpy()
    folds = np.load(FOLDS_PATH)

    # Build rich view
    X_rich, y = build_rich_view(ds.train, ds.test)
    n_features = X_rich.shape[1]

    fit_m = folds > 0
    eval_m = folds == 0
    y_fit = y[fit_m]
    y_eval = y[eval_m]

    X_fit = X_rich[fit_m]
    X_eval = X_rich[eval_m]

    # Train FM interaction layer on FIT folds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_features = X_rich.shape[1]

    X_emb = np.zeros((len(y), 8), dtype=np.float32)
    
    for k in range(5):
        eval_m = folds == k
        fit_m = ~eval_m
        
        X_fit_t = torch.from_numpy(X_rich[fit_m]).to(device).float()
        y_fit_t = torch.from_numpy(y[fit_m]).to(device).float()
        X_eval_t = torch.from_numpy(X_rich[eval_m]).to(device).float()

        fm = FactorizationMachine(n_features, rank=8).to(device)
        opt = torch.optim.AdamW(fm.parameters(), lr=1e-3, weight_decay=1e-4)

        for _ in range(50):
            fm.train()
            opt.zero_grad()
            logits = fm(X_fit_t)
            loss = F.binary_cross_entropy_with_logits(logits, y_fit_t)
            loss.backward()
            opt.step()

        fm.eval()
        with torch.no_grad():
            # Get embedding features: xV = X @ V (B, R)
            xV = X_eval_t @ fm.V  # (B, R)
            X_emb[eval_m] = xV.cpu().numpy().astype(np.float32)

    X_aug = np.column_stack([X_rich, X_emb]).astype(np.float32)

    eval_m = folds == 0
    y_e = y[eval_m]

    pred_base, auc_base = lgbm_fold0(X_rich, y, folds)
    pred_aug, auc_aug = lgbm_fold0(X_aug, y, folds)

    delta = auc_aug - auc_base
    promising = delta >= 5e-5

    spear = float(spearmanr(pred_aug, load_oof("sol_route_aux10_primary_final")[folds == 0]).statistic)
    blend = (logit(pred_aug) + logit(load_oof("sol_route_aux10_primary_final")[folds == 0])) / 2.0
    auc_blend = float(roc_auc_score(y[eval_m], blend))
    blend_delta = auc_blend - auc_aug

    report = {
        "prereg": "configs/phase20/feat50_learned_interactions_prereg.json",
        "prereg_sha256": "7c5e79bb063c6ce33ce10fa5c67a604678ed2a4d16c3236a094d786f934ebe3a",
        "method": "FM interaction layer (rank 8) on 285-feature rich view, cross-fit 5-fold. Interaction features added to LGBM.",
        "fold0_results": {
            "base_rich_auc": round(auc_base, 6),
            "augmented_rich_fm_auc": round(auc_aug, 6),
            "augmentation_delta": round(delta, 8),
            "spearman_aug_vs_champion": round(spear, 6),
            "equal_blend_auc": round(auc_blend, 6),
            "blend_delta_vs_aug": round(blend_delta, 8),
            "champion_ref": 0.961853,
            "port_delta_vs_champion": round(auc_aug - 0.961853, 8),
            "promising": bool(promising),
        },
        "rank": 8,
    }
    path = OUT / "feat50_learned_interactions.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    f0 = report["fold0_results"]
    print(f"fold-0: base_rich={f0['base_rich_auc']} aug(+FM)={f0['augmented_rich_fm_auc']} delta={f0['augmentation_delta']:+.2e}")
    print(f"spearman={f0['spearman_aug_vs_champion']} blend_delta={f0['blend_delta_vs_aug']:+.2e}")
    print(f"port_delta vs champion={f0['port_delta_vs_champion']:+.2e} promising={f0['promising']}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())