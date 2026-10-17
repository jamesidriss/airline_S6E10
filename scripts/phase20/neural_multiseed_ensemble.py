"""Multiseed PLR-MLP equal-logit ensemble fold-0 screen (prereg: neural_multiseed_prereg)."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import ARTIFACTS, REPORTS, load_cached_parquet, save_json  # noqa: E402
from src.validation.compare import logit  # noqa: E402
from src.phase19.foundation_screen import screen_metrics  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.phase19.foundation_screen_fold0 import load_champion_fold0  # noqa: E402

ART = ARTIFACTS / "phase20" / "neural_screen"
OUT = REPORTS / "phase20" / "neural_screen"
EPS = 1e-7


def main():
    tr, _ = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds
    EVAL = np.flatnonzero(folds == 0)
    route_all, port_all = load_champion_fold0(folds)

    seeds = [1, 2, 3, 4]
    preds = []
    for s in seeds:
        tag = "plrmlp_full_fold0" if s == 1 else f"plrmlp_full_fold0_s{s}"
        p = ART / f"{tag}.npy"
        assert p.exists(), f"missing {p}"
        preds.append(np.load(p).astype(np.float64))

    ls = np.stack([logit(np.clip(p, EPS, 1 - EPS)) for p in preds], 0)
    blend = 1.0 / (1.0 + np.exp(-ls.mean(0)))

    port0 = port_all[EVAL]
    route0 = route_all[EVAL]
    y_eval = y[EVAL]
    m = screen_metrics(y_eval, blend, port0, route0)

    n = len(seeds)
    corr = np.corrcoef(ls)
    off = [corr[i, j] for i in range(n) for j in range(i + 1, n)]
    mean_pair = float(np.mean(off))
    bar = m["portfolio_delta"] >= 5e-5 and m["logit_pearson_vs_portfolio"] < 0.995

    save_json({
        "kind": "plrmlp_multiseed_equal_logit", "seeds": seeds, "n_members": n,
        "mean_pairwise_member_logit_corr": round(mean_pair, 4), "eval_fold": 0,
        "fit_folds": [1, 2, 3, 4], "utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "equal-logit avg of 4 PLR-MLP full-view fold-0 preds (seeds 1-4, each "
                    "FIT-trained, inner-10% ES, champion hash-asserted); SCREEN only",
        "metrics": m,
    }, OUT / "plrmlp_multiseed4_full_fold0.json")
    np.save(ART / "plrmlp_multiseed4_full_fold0.npy", blend.astype("float32"))

    print(f"AUC={m['standalone_auc']:.6f} port_delta={m['portfolio_delta']:+.3e} "
          f"corr_port={m['logit_pearson_vs_portfolio']:.4f} mean_member_corr={mean_pair:.4f} "
          f"PROMISING={bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
