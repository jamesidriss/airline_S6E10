"""Does any public OOF vector add to candidate X under a fold-safe two-parameter blend?

For every member, a logistic blend logit(X) x logit(member) is fitted on four folds and
scored on the fifth. The reported gain is the pooled out-of-fold gain over X alone, plus
the number of positive folds. A member can only help if the gain is consistent across all
five folds, which is what the admission gate requires.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import TRAIN_CSV  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

POOL = Path(r"C:\Users\James\AppData\Local\Temp\opencode\s6e10_oof_pool\_pool\pool.npz")
X_OOF = ROOT / "artifacts" / "phase24_candidateX" / "candidate_x_oof.npy"


def main() -> None:
    train = pd.read_csv(TRAIN_CSV, usecols=["id", "satisfaction"])
    y = train["satisfaction"].to_numpy(dtype="int8")
    folds = get_scheme("primary", y, train["id"]).folds
    P = np.load(POOL)
    x = logit(np.clip(np.load(X_OOF).astype("float64"), 1e-6, 1 - 1e-6))
    base_folds = [roc_auc_score(y[folds == k], x[folds == k]) for k in range(5)]
    base = float(roc_auc_score(y, x))
    rows = []
    for name in sorted(k[5:] for k in P.files if k.startswith("oof__")):
        v = logit(np.clip(P["oof__" + name].astype("float64"), 1e-6, 1 - 1e-6))
        if np.allclose(v, x, atol=1e-9):
            continue
        pred = np.zeros(len(y))
        w = []
        for k in range(5):
            tr, te = folds != k, folds == k
            m = LogisticRegression(C=1e6, max_iter=200)
            m.fit(np.column_stack([x[tr], v[tr]]), y[tr])
            pred[te] = m.decision_function(np.column_stack([x[te], v[te]]))
            w.append(float(m.coef_[0, 1]))
        f = [roc_auc_score(y[folds == k], pred[folds == k]) for k in range(5)]
        d = [a - b for a, b in zip(f, base_folds)]
        rows.append({"member": name, "pooled": roc_auc_score(y, pred), "delta": roc_auc_score(y, pred) - base,
                     "mean_fold_delta": float(np.mean(d)), "pos": int(sum(x_ > 0 for x_ in d)),
                     "w_member": float(np.mean(w)), "fold_deltas": d})
    rows.sort(key=lambda r: -r["mean_fold_delta"])
    out = {"base_x_pooled": base, "base_x_folds": base_folds, "n_members": len(rows),
           "top": rows[:25], "n_pos5": sum(r["pos"] == 5 for r in rows),
           "n_gate": sum(r["pos"] == 5 and r["mean_fold_delta"] >= 1.5e-5 for r in rows)}
    print(json.dumps({k: out[k] for k in ("base_x_pooled", "n_members", "n_pos5", "n_gate")}, indent=1))
    for r in rows[:12]:
        print(f"{r['mean_fold_delta']*1e4:+.3f}e-4  pos{r['pos']}  w={r['w_member']:+.2f}  {r['member']}")
    (ROOT / "reports" / "phase25").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "phase25" / "pairwise_screen_vs_X.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
