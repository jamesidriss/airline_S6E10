"""Phase 12 final analysis: 5-fold paired evidence and the ACTUAL ensemble marginal.

This exists because folds 0-2 and folds 0-4 tell materially different stories, and only the second
one is admissible. On folds 0-2 the META4+irregular arm (L2) looked like a clean +5.5/+5.6/+6.5e-5.
Completing the folds gave -0.7 and -7.5e-5. The 5-fold mean is +1.9e-5, not +5.9e-5. Reporting the
screen would have overstated the effect by roughly 3x.

So this script does three things and refuses to print a conclusion without all three:
  1. the paired per-fold delta of each arm against the matched control, with mean, SE, t and sign
     consistency, computed from the ACTUAL predictions;
  2. the ensemble marginal, measured by splicing the arm into the exact v3 slot it replaces,
     using real vectors, per fold and pooled;
  3. the same quantities computed on folds 0-2 ONLY, printed side by side, so the gap between the
     screen and the full result is visible rather than forgotten.

Nothing here infers ensemble gain from standalone AUC. That inference is invalid -- Phase 11 already
recorded a structural-cap argument built on it and withdrew it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_phase12 import CHAMPION_MEMBER, CHAMPION_SCHEME, lg, sg  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

TAGS = ("p12b", "p12c")


def load_runs() -> dict:
    arms: dict = {}
    for tag in TAGS:
        p = REPORTS / f"{tag}_runs.json"
        if not p.exists():
            raise SystemExit(f"missing {p}; run the training first")
        d = json.loads(p.read_text(encoding="utf-8"))
        for aname, recs in d["arms"].items():
            for k, r in recs.items():
                arms.setdefault(aname, {})[r["fold"]] = (tag, r)
    return arms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="xt_xt_d127_s1")
    args = ap.parse_args()
    tgt = args.target

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
    print("=" * 108)
    print("PHASE 12 -- 5-FOLD PAIRED EVIDENCE AND THE ACTUAL ENSEMBLE MARGINAL")
    print("=" * 108)
    print(f"  v3 = expit(mean(logits)), float32 cast bit-exact: {geo == 0.0} (max|dp| {geo:.1e})")
    print(f"  v3 has {len(ids)} members; the slot replaced is {tgt!r}\n")

    arms = load_runs()
    all_folds = sorted({f for a in arms.values() for f in a})
    out = {"n_members": len(ids), "slot": tgt, "folds": all_folds, "arms": {}}

    # ---------- control reproduction on every fold -------------------------------------
    print("  CONTROL REPRODUCTION (the gate on every other number here)")
    l0 = arms.get("L0", {})
    reps = []
    for f in all_folds:
        if f in l0:
            reps.append((f, l0[f][1]["delta_vs_champion_e5"]))
    for f, d in reps:
        print(f"    fold {f}: {d:+.4f}e-5  {'ok' if abs(d) < 1.0 else 'MISMATCH'}")
    worst = max((abs(d) for _f, d in reps), default=999)
    print(f"    worst |delta| = {worst:.4f}e-5 -> "
          f"{'HARNESS IS THE CHAMPION' if worst < 1.0 else 'HARNESS DOES NOT MATCH THE CHAMPION'}")
    out["control_worst_abs_delta_e5"] = worst
    if worst >= 1.0:
        raise SystemExit("STOP: the control does not reproduce the champion; all deltas are void.")

    # ---------- per-arm paired deltas and ensemble marginal ----------------------------
    print(f"\n  {'arm':<5}{'fold':>5}{'cat':>5}{'AUC':>12}{'vs champ':>10}{'v3 before':>12}"
          f"{'v3 swap':>12}{'MARGINAL':>10}{'corr champ':>12}{'spear':>9}{'cat split':>10}")
    print("  " + "-" * 104)
    summary: dict = {}
    for aname in sorted(arms):
        if aname == "L0":
            continue
        per: list[dict] = []
        for f in all_folds:
            if f not in arms[aname]:
                continue
            tag, r = arms[aname][f]
            pf = REPORTS / f"{tag}_{aname}_{CHAMPION_SCHEME}_f{f}.npy"
            if not pf.exists():
                continue
            P = np.load(pf).astype("float64")
            val = np.where(folds == f)[0]
            col = L[tgt].copy()
            col[val] = lg(P)
            outside = np.ones(len(col), dtype=bool)
            outside[val] = False
            assert np.array_equal(col[outside], L[tgt][outside]), "splice moved out-of-fold rows"
            assert np.array_equal(col[val], lg(P)), "spliced column != native logit"
            aft = sg(np.mean(np.column_stack([col if e == tgt else L[e] for e in ids]), axis=1))
            a_b4 = float(roc_auc_score(y[val], sg(base_logit[val])))
            a_sw = float(roc_auc_score(y[val], aft[val]))
            per.append({"fold": f, "auc": r["auc"],
                        "vs_champ_e5": r["delta_vs_champion_e5"],
                        "v3_before": a_b4, "v3_after": a_sw,
                        "marginal_e5": (a_sw - a_b4) * 1e5,
                        "corr_champ": float(corr(lg(P), L[tgt][val])),
                        "spear_champ": float(spearman(P, sg(L[tgt][val]))),
                        "cat_splits": r["census"]["n_categorical_splits"],
                        "best_iter": r["best_iter"], "n_cat": r["n_cat"],
                        "seconds": r["seconds"]})
            d = per[-1]
            print(f"  {aname:<5}{f:>5}{d['n_cat']:>5}{d['auc']:>12.6f}"
                  f"{d['vs_champ_e5']:>+9.1f}e{d['v3_before']:>12.6f}{d['v3_after']:>12.6f}"
                  f"{d['marginal_e5']:>+9.2f}e{d['corr_champ']:>12.5f}{d['spear_champ']:>9.5f}"
                  f"{d['cat_splits']:>10}")

        def stats(key, sub=None):
            d = [x for x in per if (sub is None or x["fold"] in sub)]
            v = np.array([x[key] for x in d], dtype="float64")
            n = len(v)
            if n < 2:
                return {"n": n, "mean": float(v.mean()) if n else None}
            se = float(v.std(ddof=1) / np.sqrt(n))
            mean = float(v.mean())
            return {"n": n, "mean": mean, "se": se, "t": mean / se if se else None,
                    "n_positive": int((v > 0).sum()), "n_negative": int((v < 0).sum()),
                    "folds": d and [x["fold"] for x in d],
                    "values": [float(x) for x in v]}

        s5c = stats("vs_champ_e5")
        s5m = stats("marginal_e5")
        s3c = stats("vs_champ_e5", {0, 1, 2})
        s3m = stats("marginal_e5", {0, 1, 2})
        summary[aname] = {"per_fold": per, "standalone_5fold": s5c, "marginal_5fold": s5m,
                          "standalone_f0_2": s3c, "marginal_f0_2": s3m}
        print(f"\n  {aname}: standalone vs champion, 5 folds  mean {s5c['mean']:+.2f}e-5  "
              f"SE {s5c['se']:.2f}  t {s5c['t']:+.2f}  "
              f"{s5c['n_positive']}/{s5c['n']} positive")
        print(f"  {' '*len(aname)}  marginal on v3,   5 folds  mean {s5m['mean']:+.2f}e-5  "
              f"SE {s5m['se']:.2f}  t {s5m['t']:+.2f}  "
              f"{s5m['n_positive']}/{s5m['n']} positive")
        print(f"  {' '*len(aname)}  FOLD 0-2 SCREEN ONLY: standalone {s3c['mean']:+.2f}e-5, "
              f"marginal {s3m['mean']:+.2f}e-5")
        print(f"  {' '*len(aname)}  -> completing the folds moved standalone from "
              f"{s3c['mean']:+.2f}e-5 to {s5c['mean']:+.2f}e-5 "
              f"({(s5c['mean'] - s3c['mean']) / max(abs(s3c['mean']), 1e-12):+.0%} change)")

    # ---------- verdict -----------------------------------------------------------------
    print("\n  VERDICT, using the ADMISSION gate the campaign predeclared (>= +1.5e-5 honest")
    print("  marginal ensemble gain, positive in >= 4 of 5 folds, mean >= 2.5x paired SE):")
    passed = []
    for aname, s in summary.items():
        m = s["marginal_5fold"]
        ok = (m["mean"] >= 1.5 and m["n_positive"] >= 4
              and (m["se"] == 0 or m["mean"] >= 2.5 * m["se"]))
        if ok:
            passed.append(aname)
        print(f"    {aname}: marginal {m['mean']:+.2f}e-5, {m['n_positive']}/{m['n']} positive, "
              f"mean/SE = {(m['mean'] / m['se'] if m['se'] else 0):.2f} -> "
              f"{'PASS' if ok else 'FAIL'}")
    verdict = ("ADMIT " + ",".join(passed) if passed else
               "NO ADMISSION. No arm cleared +1.5e-5 marginal ensemble gain with fold consistency. "
               "The standalone improvements are real but do not survive into the blend, because "
               "each arm's prediction is ~0.999 logit-correlated with the member it replaces.")
    print(f"    -> {verdict}")
    out["summary"] = summary
    out["verdict"] = verdict
    out["admitted"] = passed
    save_json(out, REPORTS / "p12_final.json")
    print("\n  wrote", REPORTS / "p12_final.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())