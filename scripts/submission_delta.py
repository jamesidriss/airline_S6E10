"""Why did a +1.65e-4 OOF gain not move the public LB?

Compares the test predictions of two submissions: Spearman / Pearson correlation of the raw
scores and, more importantly, the *rank* agreement and the AUC of one against the other as if
one were truth. A near-1 Spearman means the induced ordering barely changed, so no amount of
OOF improvement in that direction can show up on a 60k-row public split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.submission import store  # noqa: E402


def main() -> None:
    ids = sys.argv[1:] or ["blend_v1_equal4", "blend_v2_equal34"]
    T = {i: store.load_test(i).astype("float64") for i in ids}
    O = {i: store.load_oof(i).astype("float64") for i in ids}
    print(f"{'pair':<50} {'spearman':>10} {'pearson':>10} {'top-decile overlap':>20}")
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            x, z = T[ids[a]], T[ids[b]]
            sp = spearmanr(x, z).statistic
            pe = float(np.corrcoef(x, z)[0, 1])
            # rank agreement on the most confident deciles, where the AUC actually lives
            tx = np.argsort(np.argsort(-x))[: len(x) // 10]
            tz = set(np.argsort(np.argsort(-z))[: len(z) // 10].tolist())
            ov = len(set(tx.tolist()) & tz) / max(1, len(tx))
            print(f"{ids[a]+' vs '+ids[b]:<50} {sp:>10.6f} {pe:>10.6f} {ov:>19.4f}")

    # OOF-space check: does the OOF *ordering* move more than the test ordering?
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            sp = spearmanr(O[ids[a]], O[ids[b]]).statistic
            print(f"OOF  {ids[a]+' vs '+ids[b]:<44} {sp:>10.6f}")


if __name__ == "__main__":
    main()