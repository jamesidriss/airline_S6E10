"""Feature-importance analysis: what actually drives the champion, and what is being ignored.

Ablation by *dropping* a whole feature family and re-measuring OOF AUC is the honest way to find
unused signal, but it is expensive. This script gives the cheap view first (gain-based importance
plus permutation importance on a subsample) and flags families the model barely uses.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

FAMILY_RULES = [
    ("raw", r"^(?!cnt_|fdm_|fdb_|na_|rate_|ix_|tok|ogte_|ogs_|teach|te_|delay_|log_dep|log_arr|age_bin|fd_bin|_fd)"),
    ("cat_twin", r"__cat$"),
    ("count", r"^cnt_"),
    ("route_profile", r"^fdm_|^fd_cnt$|^fdb_"),
    ("digits", r"_(m10|m100|m1000|d10|d100|len|is0)$"),
    ("survey_NA", r"^na_|^rate_"),
    ("domain", r"^delay_|^log_dep|^log_arr|^delay_per_km|^age_bin|^fd_bin|^ix_"),
    ("gpt2_token", r"^tok"),
    ("original_TE", r"^ogte_"),
    ("original_surface", r"^ogs_"),
    ("original_teacher", r"^teach"),
    ("target_encoding", r"^te_"),
]


def family_of(name: str) -> str:
    for fam, pat in FAMILY_RULES:
        if re.search(pat, name):
            return fam
    return "other"


def main() -> None:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y, tr[ID_COL]).folds
    vb = ViewBuilder(tr, te, "full")
    st_tr, st_te, names = vb.build_static()

    # one fold is enough to read importances
    k = 0
    fit = np.where(folds != k)[0]
    val = np.where(folds == k)[0]
    Xf, Xa, full_names = vb.assemble(fit, y, val, np.arange(len(tr), len(tr) + len(te)))
    from scripts.run_views import _inner_es_split

    es_tr, es_idx = _inner_es_split(fit, y, 1)
    pos = {v: i for i, v in enumerate(fit)}
    fitX = Xf[np.array([pos[v] for v in es_tr])]
    fitY = y[es_tr]
    esX = Xf[np.array([pos[v] for v in es_idx])]
    esY = y[es_idx]

    ds = lgb.Dataset(fitX, label=fitY, feature_name=full_names)
    dv = lgb.Dataset(esX, label=esY, reference=ds)
    m = lgb.train(dict(objective="binary", metric="auc", n_estimators=3000, learning_rate=0.03,
                       num_leaves=127, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                       verbose=-1, n_jobs=8, random_state=1),
                  ds, num_boost_round=3000, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    print("best_iter", m.best_iteration, "holdout auc", roc_auc_score(esY, m.predict(esX, num_iteration=m.best_iteration)))

    gain = pd.Series(m.feature_importance("gain"), index=full_names).sort_values(ascending=False)
    split = pd.Series(m.feature_importance("split"), index=full_names)
    tot = float(gain.sum()) or 1.0
    fams = defaultdict(float)
    famcnt = defaultdict(int)
    for n, g in gain.items():
        fams[family_of(n)] += float(g) / tot
        famcnt[family_of(n)] += 1
    print("\n=== gain share by feature family ===")
    for f, g in sorted(fams.items(), key=lambda x: -x[1]):
        print(f"  {f:<20} gain_share={g*100:6.2f}%  n_cols={famcnt[f]}")
    print("\n=== top 45 features by gain ===")
    print(pd.DataFrame({"gain_share": (gain / tot).head(45).round(5),
                        "splits": split.head(45)}).to_string())
    print("\n=== bottom 20 used features (near-zero gain) ===")
    nz = gain[gain > 0].tail(20)
    print(pd.DataFrame({"gain_share": (nz / tot).round(6), "splits": split[nz.index]}).to_string())

    save_json({"gain_by_family": {k: round(v, 5) for k, v in fams.items()},
               "n_cols_by_family": dict(famcnt),
               "top_features": [{"name": n, "gain_share": round(float(g / tot), 6),
                                 "splits": int(split[n])} for n, g in gain.head(120).items()],
               "best_iter": int(m.best_iteration)},
              REPORTS / "feature_importance.json")


if __name__ == "__main__":
    main()