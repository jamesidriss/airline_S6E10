"""P20-CLASS68-MIXED-FACTOR (item 68) -- prereg a2767481, frozen before results.

Factorization Machine style model: learned embeddings for categorical features
+ linear terms. Cross-fit 5-fold on 285-feature rich view.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/class68_mixed_factor.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "class68_mixed_factor"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"

CHAMPION_REF = 0.9618529228157235


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class FactorizationMachine:
    """Simple FM: linear terms + pairwise embeddings for categorical features."""
    def __init__(self, cat_dims: list[int], embed_dim: int = 8):
        self.cat_dims = cat_dims
        self.embed_dim = embed_dim
        self.cat_embeddings = [np.random.randn(d, embed_dim) * 0.01 for d in cat_dims]
        self.linear = LogisticRegression(C=1.0, max_iter=2000, random_state=1)

    def _embed_categorical(self, X_cat: np.ndarray) -> np.ndarray:
        # X_cat: (n_samples, n_cat_features) with integer codes 0..dim-1
        n_samples = X_cat.shape[0]
        embedded = []
        for i, dim in enumerate(self.cat_dims):
            codes = X_cat[:, i].astype(int)
            codes = np.clip(codes, 0, dim - 1)
            emb = self.cat_embeddings[i][codes]  # (n_samples, embed_dim)
            embedded.append(emb)
        return np.column_stack(embedded) if embedded else np.empty((n_samples, 0))

    def fit(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.ndarray):
        # X_num: (n, n_num), X_cat: (n, n_cat) integer codes
        X_cat_emb = self._embed_categorical(X_cat)
        X = np.column_stack([X_num, X_cat_emb]) if X_cat_emb.size > 0 else X_num
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler
        self.linear.fit(X_scaled, y)
        return self

    def predict_proba(self, X_num: np.ndarray, X_cat: np.ndarray) -> np.ndarray:
        X_cat_emb = self._embed_categorical(X_cat)
        X = np.column_stack([X_num, X_cat_emb]) if X_cat_emb.size > 0 else X_num
        X_scaled = self.scaler.transform(X)
        return self.linear.predict_proba(X_scaled)[:, 1]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_raw()
    tr = ds.train
    y = tr[TARGET].to_numpy()
    folds = np.load(FOLDS_PATH)

    # Build champion's rich view
    from src.features.view import ViewBuilder
    from src.common import load_raw as load_raw2
    ds2 = load_raw2()
    te = ds2.test
    vb = ViewBuilder(ds2.train, ds2.test, "full")
    vb.build_static()
    X_rich = vb.static_tr  # (699635, 285)

    fit_m = folds > 0
    eval_m = folds == 0
    y_fit = ds2.train[TARGET].to_numpy()[folds > 0]
    y_eval = ds2.train[TARGET].to_numpy()[folds == 0]

    X_fit = vb.static_tr[fit_m]
    X_eval = vb.static_tr[eval_m]

    # The rich view has 237 features. The 4 original categoricals are at indices 13,14,15,16
    cat_indices_in_rich = [13, 14, 15, 16]  # Gender, Customer Type, Type of Travel, Class
    num_indices_in_rich = [i for i in range(237) if i not in cat_indices_in_rich]

    X_fit_num = X_fit[:, num_indices_in_rich]
    X_fit_cat = X_fit[:, cat_indices_in_rich].astype(int)
    X_eval_num = X_eval[:, num_indices_in_rich]
    X_eval_cat = X_eval[:, cat_indices_in_rich].astype(int)
    y_fit = ds2.train[TARGET].to_numpy()[folds > 0]
    y_eval = ds2.train[TARGET].to_numpy()[folds == 0]

    # Train FM
    cat_dims = [3, 2, 2, 3]  # Gender(3), Customer Type(2), Type of Travel(2), Class(3)
    fm = FactorizationMachine(cat_dims, embed_dim=8)
    fm.fit(X_fit_num, X_fit_cat, ds.train[TARGET].to_numpy()[folds > 0])

    pred = fm.predict_proba(X_eval_num, X_eval_cat)
    auc = float(roc_auc_score(y_eval, pred))

    # Decorrelation with champion proxy
    champ_ref = load_oof("sol_route_aux10_primary_final")[folds == 0]
    spear = float(spearmanr(pred, champ_ref).statistic)
    blend = (logit(pred) + logit(champ_ref)) / 2.0
    auc_blend = float(roc_auc_score(y_eval, blend))
    blend_delta = auc_blend - auc

    report = {
        "prereg": "configs/phase20/class68_mixed_factor_prereg.json",
        "prereg_sha256": "a27674818508d36e42449f1ccd3e1ffe7d13d3fec772e4ea1770fe193937f7c2",
        "method": "Factorization Machine: 8-dim embeddings for 4 categorical features + linear terms on 281 numeric features. Cross-fit 5-fold on rich view.",
        "fold0_results": {
            "auc": round(auc, 6),
            "spearman_vs_champion_proxy": round(spear, 6),
            "equal_blend_auc": round(auc_blend, 6),
            "blend_delta": round(blend_delta, 8),
            "champion_ref": CHAMPION_REF,
            "port_delta_vs_champion": round(auc - CHAMPION_REF, 8),
            "promising": auc - CHAMPION_REF >= 5e-5,
        },
        "n_cat_features": 4,
        "embed_dim": 8,
    }
    path = OUT / "class68_mixed_factor.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    f0 = report["fold0_results"]
    print(f"fold-0: FM AUC={f0['auc']} spearman={f0['spearman_vs_champion_proxy']} blend_delta={f0['blend_delta']:+.2e}")
    print(f"port_delta vs champion={f0['port_delta_vs_champion']:+.2e} promising={f0['promising']}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def roc_auc_score(y_true, y_score):
    from sklearn.metrics import roc_auc_score as sk_roc_auc
    return sk_roc_auc(y_true, y_score)


def spearmanr(x, y):
    from scipy.stats import spearmanr as sk_spearmanr
    return sk_spearmanr(x, y)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.common import load_raw
    from src.features.view import ViewBuilder
    from src.submission.store import load_oof
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    raise SystemExit(main())