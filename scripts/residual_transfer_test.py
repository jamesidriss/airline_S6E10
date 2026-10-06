"""Does the Phase 10A residual correction transfer to the actual v3 ensemble?

Why this test exists
--------------------
The nested scan (scripts/residual_nested.py) found ONE predeclared key passing its gate:
`raw:On-board service`, probability semantics +2.00e-5 (4/5 folds), logit semantics +1.92e-5 (4/5),
sign agreement 0.720. Two numbers argue for caution rather than celebration:

  replication r = 0.540            modest, not strong
  mean bias gap = -20.45e-5         the CONFIRMATION-side group bias is 20.45e-5 SMALLER than the
                                    discovery-side estimate

That gap is ten times the size of the +2.00e-5 gain it produces. It is the signature of an estimated
group offset that does not carry its magnitude out of sample: the ranking still improves slightly
because even a shrunken offset points the right way, but the correction is mostly noise.

And the nested scan was run on a single champion lgbm surrogate, not on the 59-member v3 blend that
we actually submit. A +2.00e-5 gain on the surrogate need not survive on v3, whose residuals could
differ.

Legitimacy of this test
-----------------------
For outer fold k:
  * the correction table is fitted ONLY on META_TRAIN rows, using residuals from the nested
    surrogate's inner cross-fit, so the table has seen no META_VALIDATION label by either route;
  * it is applied to v3's OOF predictions on META_VALIDATION rows, and v3's OOF for fold k is
    out-of-fold for fold k by construction.
So the applied prediction is honest. What this measures is TRANSFER -- surrogate to v3 -- which is a
genuinely different question from the nested scan's and cannot be answered any other way.

Three things are reported, because they can disagree:
  1. the nested surrogate's own delta (already known, for reference)
  2. the same correction applied to v3, per fold
  3. the CORRECTION MAGNITUDE RATIO, confirmation vs discovery, which is the honest measure of
     whether the fitted offset transfers at all

A correction that helps the surrogate by +2e-5 and hurts v3 has told us the surrogate's residuals
were the thing being fitted, not a shared bias.

Usage: python scripts/residual_transfer_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.residual_nested import (EPS, MIN_N, PRIOR_N, apply_table, as_group_series,  # noqa: E402
                                     fit_logit_offset, fit_prob_bias, logit, observed_bias)

CANDIDATE = "raw:On-board service"
EPS_CLIP = EPS


def main() -> None:
    tr, _te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    folds = get_scheme("primary", tr[TARGET].values.astype("int8"), tr[ID_COL]).folds
    v3 = None
    from src.submission import store
    v3 = store.load_oof("blend_v3_final").astype("float64")

    key_all = as_group_series(tr[CANDIDATE.split(":", 1)[1]].astype(str))
    print("RESIDUAL CORRECTION TRANSFER TEST -- surrogate (nested) -> v3 (59-member blend)")
    print(f"  candidate key: {CANDIDATE}")
    print(f"  levels: {key_all.nunique()}   pre-declared PRIOR_N={PRIOR_N}  MIN_N={MIN_N}\n")

    rows = []
    print(f"  {'fold':>5}{'surrogate d':>13}{'v3 d(prob)':>13}{'v3 d(logit)':>13}"
          f"{'bias gap':>11}{'size ratio':>12}")
    print(f"  {'-'*70}")
    for k in range(5):
        cache = REPORTS / f"p10a_surrogate_fold{k}.npz"
        if not cache.exists():
            print(f"  fold {k}: no cached surrogate, skipping")
            continue
        z = np.load(cache)
        p_val, p_oof = z["p_val"], z["p_oof"]
        meta_val, meta_fit = z["meta_val"], z["meta_fit"]
        assert list(folds[meta_val]) == [k] * len(meta_val), "fold mismatch"

        gf = key_all[meta_fit].reset_index(drop=True)
        gv = key_all[meta_val].reset_index(drop=True)
        yf, yv = y[meta_fit], y[meta_val]

        # fit on META_TRAIN using NESTED surrogate residuals only
        pbias = fit_prob_bias(gf, yf, p_oof, np.arange(len(yf)))
        loff = fit_logit_offset(gf, yf, p_oof, np.arange(len(yf)))

        # surrogate delta (reference)
        s0 = float(roc_auc_score(yv, p_val))
        s1 = float(roc_auc_score(yv, np.clip(p_val + apply_table(gv, np.arange(len(yv)), pbias),
                                             EPS_CLIP, 1 - EPS_CLIP)))
        d_sur = (s1 - s0) * 1e5

        # transfer to v3 on the SAME held-out rows
        v = v3[meta_val]
        v0 = float(roc_auc_score(yv, v))
        v1 = float(roc_auc_score(yv, np.clip(v + apply_table(gv, np.arange(len(yv)), pbias),
                                             EPS_CLIP, 1 - EPS_CLIP)))
        v2 = float(roc_auc_score(yv, logit(v) + apply_table(gv, np.arange(len(yv)), loff)))
        d_v3p = (v1 - v0) * 1e5
        d_v3l = (v2 - v0) * 1e5

        # does the fitted OFFSET transfer in magnitude?
        disc = np.array(list(pbias.values()))
        conf = observed_bias(gv, yv - v, np.arange(len(yv)))
        common = [u for u in pbias if u in conf]
        gap = (float(np.mean([conf[u] for u in common]))
               - float(np.mean([pbias[u] for u in common]))) * 1e5
        ratio = (float(np.mean(np.abs([conf[u] for u in common])))
                 / max(float(np.mean(np.abs([pbias[u] for u in common]))), 1e-12))
        rows.append({"fold": k, "surrogate_delta_e5": d_sur, "v3_delta_prob_e5": d_v3p,
                     "v3_delta_logit_e5": d_v3l, "v3_base_auc": v0, "surrogate_base_auc": s0,
                     "bias_gap_e5": gap, "magnitude_ratio": ratio,
                     "sign_agreement": float(np.mean(
                         np.sign([pbias[u] for u in common]) == np.sign([conf[u] for u in common])))})
        print(f"  {k:>5}{d_sur:>+12.2f}e{d_v3p:>+12.2f}e{d_v3l:>+12.2f}e"
              f"{gap:>+10.2f}e{ratio:>11.2f}x")

    if not rows:
        print("no folds available")
        return
    sp = float(np.mean([r["surrogate_delta_e5"] for r in rows]))
    vp = float(np.mean([r["v3_delta_prob_e5"] for r in rows]))
    vl = float(np.mean([r["v3_delta_logit_e5"] for r in rows]))
    gapm = float(np.mean([r["bias_gap_e5"] for r in rows]))
    ratio_m = float(np.mean([r["magnitude_ratio"] for r in rows]))
    signm = float(np.mean([r["sign_agreement"] for r in rows]))
    print(f"  {'-'*70}")
    print(f"  {'mean':>5}{sp:>+12.2f}e{vp:>+12.2f}e{vl:>+12.2f}e{gapm:>+10.2f}e{ratio_m:>11.2f}x")
    print(f"\n  v3 deltas: probability {vp:+.2f}e-5 positive in "
          f"{sum(r['v3_delta_prob_e5'] > 0 for r in rows)}/{len(rows)} folds")
    print(f"              logit      {vl:+.2f}e-5 positive in "
          f"{sum(r['v3_delta_logit_e5'] > 0 for r in rows)}/{len(rows)} folds")
    print(f"  fitted offset transfers at {ratio_m:.2f}x its fitted magnitude "
          f"(mean bias gap {gapm:+.2f}e-5, sign agreement {signm:.3f})")

    gate = 1.5
    transfer_ok = (max(vp, vl) >= gate
                   and sum(r["v3_delta_prob_e5"] > 0 for r in rows) >= 4
                   and sum(r["v3_delta_logit_e5"] > 0 for r in rows) >= 4)
    if transfer_ok:
        verdict = (f"TRANSFERS to v3: {max(vp, vl):+.2f}e-5 with >=4/5 folds positive under both "
                   f"semantics. A specialist is justified.")
    elif max(vp, vl) > 0:
        verdict = (f"PARTIAL: the correction helps v3 by only {max(vp, vl):+.2f}e-5, below the "
                   f"{gate}e-5 gate, and/or is not positive in 4/5 folds. Do NOT promote. The "
                   f"surrogate's {sp:+.2f}e-5 does not carry over.")
    else:
        verdict = ("DOES NOT TRANSFER: the correction is neutral-to-negative on v3. The nested "
                   f"surrogate's +{sp:.2f}e-5 was fitting that surrogate's residuals, not a shared "
                   "bias in v3.")
    print(f"\nVERDICT: {verdict}")

    save_json({"candidate": CANDIDATE, "folds": rows,
               "mean_surrogate_e5": sp, "mean_v3_prob_e5": vp, "mean_v3_logit_e5": vl,
               "mean_bias_gap_e5": gapm, "mean_magnitude_ratio": ratio_m,
               "mean_sign_agreement": signm, "gate_e5": gate, "verdict": verdict},
              REPORTS / "p10a_transfer.json")
    print("wrote", REPORTS / "p10a_transfer.json")


if __name__ == "__main__":
    main()
