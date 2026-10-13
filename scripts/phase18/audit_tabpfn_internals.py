"""Phase18 Axis-E audit: are v7's two official TabPFN internal estimators complementary?

Zero GPU. Uses only saved raw member logits from artifacts/phase17_B1/f{0..4}
and the certified route/portfolio vectors. Answers:
  1. Are the internal members complementary (rescue/damage each other)?
  2. Does one member correct a specific ranking failure of the other?
  3. Would a fixed single-member (anchored) route be competitive with the 2-member route?
  4. Does member diversity survive to the portfolio level (vs aux10)?
Are the observed gains mostly averaging noise? (measured: member disagreement vs route-member gap)
No estimator selection on OOF labels is performed or implied: both members are
predeclared official members of the frozen config; this is a diagnostic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score  # noqa: E402
from scripts.phase16_pair_metrics import pair_rescue_damage  # noqa: E402
from scripts.phase16_common import error_pairs  # noqa: E402

OUT = ROOT / "reports" / "phase18"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ROOT / "data/cache/train.parquet")
    y = df["satisfaction"].to_numpy()
    n = len(y)

    route = np.zeros(n, dtype=np.float32)
    m0 = np.zeros(n, dtype=np.float32)
    m1 = np.zeros(n, dtype=np.float32)
    for k in range(5):
        ids = np.load(ROOT / f"artifacts/phase17_B1/f{k}/apply_ids.npy")
        route[ids] = np.load(ROOT / f"artifacts/phase17_B1/f{k}/prediction.npy")
        for mi in (0, 1):
            raw = np.load(ROOT / f"artifacts/phase17_B1/f{k}/member_{mi}_raw_logits.npy")
            # (1, N, 2) logits -> official probability aggregation = softmax per estimator,
            # then mean over estimators, then expit of logit? reproduce route exactly below.
            logits = raw[0]  # (N, 2)
            p1 = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))  # softmax of 2 logits = sigmoid(diff)
            if mi == 0:
                m0[ids] = p1.astype(np.float32)
            else:
                m1[ids] = p1.astype(np.float32)

    internal_mean = ((m0.astype(np.float64) + m1.astype(np.float64)) / 2.0)
    gap = float(np.abs(internal_mean - route.astype(np.float64)).max())
    # route is the official internal aggregation; check it equals the mean of member probabilities
    print(json.dumps({"route_vs_member_mean_max_abs": gap}))

    a_route = roc_auc_score(y, route)
    a_m0 = roc_auc_score(y, m0)
    a_m1 = roc_auc_score(y, m1)
    a_mean = roc_auc_score(y, internal_mean)

    prd_01 = pair_rescue_damage(y, m1, m0)  # m1 as candidate vs m0 as baseline
    e0, tot = error_pairs(y, m0)
    e1, _ = error_pairs(y, m1)

    folds = np.load(ROOT / "data/cache/folds/primary_5f_seed20261010.npy")
    fold_aucs = []
    for k in range(5):
        mk = folds == k
        fold_aucs.append({
            "fold": k, "route": roc_auc_score(y[mk], route[mk]),
            "m0": roc_auc_score(y[mk], m0[mk]), "m1": roc_auc_score(y[mk], m1[mk]),
            "m0_minus_route": roc_auc_score(y[mk], m0[mk]) - roc_auc_score(y[mk], route[mk]),
            "m1_minus_route": roc_auc_score(y[mk], m1[mk]) - roc_auc_score(y[mk], route[mk]),
        })

    member_gap = np.abs(m0.astype(np.float64) - m1.astype(np.float64))
    corr = float(np.corrcoef(m0, m1)[0, 1])
    spearman = float(pd.Series(m0).corr(pd.Series(m1), method="spearman"))

    out = {
        "auc": {"route_internal2": a_route, "member0_only": a_m0, "member1_only": a_m1,
                "member_mean": a_mean,
                "m0_minus_route": a_m0 - a_route, "m1_minus_route": a_m1 - a_route,
                "member_mean_minus_route": a_mean - a_route},
        "pair_rescue_m1_vs_m0": prd_01,
        "error_pairs": {"m0": e0, "m1": e1, "total": tot},
        "members": {"pearson": corr, "spearman": spearman,
                    "mean_abs_gap": float(member_gap.mean()),
                    "p99_abs_gap": float(np.quantile(member_gap, 0.99))},
        "fold_aucs": fold_aucs,
        "interpretation_limits": [
            "This is a diagnostic on the frozen config; no member was selected using these numbers.",
            "A single-member route would be a deployment-policy change requiring its own matched experiment.",
            "Internal-member AUCs here are OOF via the certified five FIT contexts, same geometry as route.",
        ],
    }
    (OUT / "tabpfn_internal_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["auc"], indent=2))
    print(json.dumps(out["members"], indent=2))


if __name__ == "__main__":
    main()
