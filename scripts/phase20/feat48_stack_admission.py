"""ENS-stage stack admission test for the FEAT48-F3 member (saved arrays only).

The F3 passed the FEATURE-level gate (mean +2.87e-4 vs the 21raw host, t=4.92).
AGENTS.md admission is defined against the CURRENT STACK: equal-logit blend of
champion v7 + the augmented member, per fold, same three conditions (mean >=
1.5e-5, >= 2.5*SE, >= 4/5 positive).  Frozen equal-logit, no weight search.
Descriptive metrics never decide; this IS the decision test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "feat48_hierarchical_ratings"
CHAMPION = "v7_sol_tabpfn_internal2_aux10"
ADMISSION_GAIN = 1.5e-5


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def main() -> int:
    ds = load_raw()
    tr = ds.train
    y = tr[TARGET].to_numpy()
    folds = get_scheme("primary", y, tr[ID_COL].to_numpy()).folds

    champ = load_oof(CHAMPION)
    member = np.load(OUT / "f3_oof_treatment.npy").astype(np.float32)
    assert len(champ) == len(member) == len(y)

    champ_fold = []
    blend_fold = []
    for k in range(5):
        m = folds == k
        blend = (0.5 * (logit(champ[m]) + logit(member[m]))).astype(np.float32)
        champ_fold.append(float(roc_auc_score(y[m], champ[m])))
        blend_fold.append(float(roc_auc_score(y[m], blend)))

    paired = np.array(blend_fold) - np.array(champ_fold)
    mean_delta = float(paired.mean())
    se_delta = float(paired.std(ddof=1) / np.sqrt(5))
    t_stat = mean_delta / se_delta if se_delta > 0 else 0.0
    pos_folds = int((paired > 0).sum())
    passes = bool(mean_delta >= ADMISSION_GAIN and mean_delta >= 2.5 * se_delta and pos_folds >= 4)
    sp = float(spearmanr(champ, member).statistic)

    report = {
        "test": "ENS-stage stack admission (AGENTS.md gate vs current stack, frozen equal-logit)",
        "champion": CHAMPION,
        "member": "feat48_f3_oof_treatment (21raw+13hier LGBM OOF)",
        "champion_fold_auc": [round(v, 6) for v in champ_fold],
        "blend_fold_auc": [round(v, 6) for v in blend_fold],
        "paired_deltas": [round(d, 8) for d in paired],
        "mean_paired_delta": round(mean_delta, 8),
        "se": round(se_delta, 8),
        "t": round(t_stat, 4),
        "positive_folds": pos_folds,
        "spearman_vs_champion": round(sp, 6),
        "passes_stack_gate": passes,
    }
    path = OUT / "feat48_stack_admission.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"champion fold auc {[round(v, 6) for v in champ_fold]}")
    print(f"blend fold auc    {[round(v, 6) for v in blend_fold]}")
    print(f"paired {[round(d, 8) for d in paired]} mean {mean_delta:+.3e} se {se_delta:.3e} t={t_stat:.2f} pos={pos_folds}/5")
    print(f"spearman {sp:.6f}  STACK GATE: {passes}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
