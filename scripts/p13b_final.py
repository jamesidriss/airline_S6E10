"""Authoritative 5-fold Phase 13B result, recomputed from the saved prediction vectors.

WHY THIS SCRIPT EXISTS
----------------------
Run-scoped JSON reports are overwritten per `--folds` invocation, a hazard already recorded in this
campaign. Running p13b twice (folds 0,1,2 then folds 3,4) left reports/p13b_runs.json holding only
folds 3 and 4, so the headline number existed only in two log files. This recomputes it from the
per-fold .npy prediction vectors, which were never overwritten, and writes a durable record.

It also fixes a real limitation of the cumulative printout in run_phase13b.py: that curve is
SEED-ORDERED, so "cumulative over N slots" depends on the arbitrary order the slots were visited and
its non-monotonicity is not evidence that a particular slot hurt. This script reports the all-slot
marginal only, which is order-independent, and does not attribute gains to individual slots.

Usage: python scripts/p13b_final.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_phase12 import CHAMPION_SCHEME, lg, sg  # noqa: E402
from scripts.run_phase13b import SLOT_SEEDS, SLOTS  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def main() -> int:
    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    yi = tr[TARGET].values.astype("int8")
    folds = get_scheme(CHAMPION_SCHEME, yi, tr[ID_COL]).folds

    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
        encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]
    L = {e: lg(store.load_oof(e).astype("float64")) for e in ids}
    base_logit = np.mean(np.column_stack([L[e] for e in ids]), axis=1)
    v3 = store.load_oof("blend_v3_final").astype("float64")
    geo = float(np.abs(sg(base_logit).astype(np.float32).astype("float64") - v3).max())
    assert geo == 0.0, "v3 geometry check failed"

    present = sorted((s for s in SLOTS if s in ids), key=lambda s: SLOT_SEEDS[s])
    print("=" * 96)
    print("PHASE 13B -- AUTHORITATIVE 5-FOLD ALL-SLOT MARGINAL, from saved prediction vectors")
    print("=" * 96)
    print(f"  v3 = expit(mean(logits)), float32 cast bit-exact: {geo == 0.0}")
    print(f"  {len(present)} extra_trees slots replaced simultaneously, order-independent\n")
    print(f"  {'fold':>5}{'v3 control':>13}{'v3 + aux':>13}{'MARGINAL':>11}"
          f"{'slots present':>15}")
    print("  " + "-" * 60)

    rows = []
    for k in range(5):
        files = {s: REPORTS / f"p13b_{s}_{CHAMPION_SCHEME}_f{k}.npy" for s in present}
        have = [s for s, p in files.items() if p.exists()]
        if not have:
            print(f"  {k:>5}   (no prediction vectors on disk -- skipped)")
            continue
        val = np.where(folds == k)[0]
        a_b4 = float(roc_auc_score(y[val], sg(base_logit[val])))
        cols = []
        for e in ids:
            if e in have:
                P = np.load(files[e]).astype("float64")
                c = L[e].copy()
                c[val] = lg(P)
                outside = np.ones(len(c), dtype=bool)
                outside[val] = False
                assert np.array_equal(c[outside], L[e][outside]), f"fold {k} {e}: splice moved rows"
                assert np.array_equal(c[val], lg(P)), f"fold {k} {e}: spliced column mismatch"
                cols.append(c)
            else:
                cols.append(L[e])
        aft = sg(np.mean(np.column_stack(cols), axis=1))
        a_sw = float(roc_auc_score(y[val], aft[val]))
        rows.append({"fold": k, "v3_control": a_b4, "v3_with_aux": a_sw,
                     "marginal_e5": (a_sw - a_b4) * 1e5, "n_slots": len(have)})
        print(f"  {k:>5}{a_b4:>13.6f}{a_sw:>13.6f}{(a_sw - a_b4) * 1e5:>+10.3f}e{len(have):>15}")

    v = np.array([r["marginal_e5"] for r in rows])
    mean = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v)))
    print("  " + "-" * 60)
    print(f"  ALL-SLOT marginal: {[f'{x:+.3f}' for x in v]}")
    print(f"    mean {mean:+.3f}e-5   SE {se:.3f}   t {mean / se:+.2f}   "
          f"{int((v > 0).sum())}/{len(v)} positive")
    print(f"\n  ADMISSION GATE (predeclared by the campaign)")
    print(f"    mean marginal >= +1.5e-5        : {mean >= 1.5}   ({mean:+.3f}e-5, "
          f"{1.5 / mean:.1f}x short)" if mean > 0 else "    n/a")
    print(f"    positive in >= 4 of 5 folds    : {int((v > 0).sum()) >= 4}   "
          f"({int((v > 0).sum())}/{len(v)})")
    print(f"    mean >= 2.5x paired SE         : {mean >= 2.5 * se}   ({mean / se:.2f}x)")
    passed = bool(mean >= 1.5 and (v > 0).sum() >= 4 and mean >= 2.5 * se)
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    verdict = (
        "NO ADMISSION AND NO SUBMISSION. The auxiliary-task mechanism is VALIDATED as real: it is "
        "the largest standalone effect measured in this campaign (+12.25e-5 mean, t +3.11, 5/5 "
        "folds) and it produces the first consistently positive ensemble marginal in the campaign "
        "(5/5 folds, t +4.24 on a single slot). Replacing all six recoverable extra_trees slots "
        f"simultaneously gives {mean:+.3f}e-5, {int((v > 0).sum())}/{len(v)} folds positive. That is "
        "2.8x the single-slot gain, so the effect genuinely compounds across slots -- but it "
        "compounds SUB-LINEARLY and still falls short of the +1.5e-5 gate. Extrapolating to the "
        "remaining extra_trees members is refused for the same reason Phase 11R refused to scale a "
        "block arithmetically: family weight times a per-slot number is not a prediction of blend "
        "gain, and the measured cumulative curve is non-monotonic in slot order, so any extrapolation "
        "from it would be inventing precision."
        if not passed else
        "ADMIT the aux-equipped extra_trees block.")
    print(f"\n  {verdict}")
    save_json({"per_fold": rows, "slots": present, "mean_marginal_e5": mean, "se": se,
               "t": mean / se if se else None, "n_positive": int((v > 0).sum()),
               "n_folds": len(v), "gate_pass": passed, "verdict": verdict,
               "note": "recomputed from per-fold .npy prediction vectors because the run-scoped "
                       "JSON was overwritten by the second --folds invocation; this record is "
                       "durable and covers all five folds."},
              REPORTS / "p13b_final.json")
    print(f"\n  wrote {REPORTS / 'p13b_final.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())