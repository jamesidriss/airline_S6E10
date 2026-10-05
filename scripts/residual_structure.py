"""Diagnostic: after the champion predicts p(x), is E[y - p | group] reproducible?

This is a falsifiable test, not another feature dump
----------------------------------------------------
Phase 8 added target-encoding features and both variants lost. The natural follow-up question is
whether the champion has left *any* systematic bias behind that a correction could recover. We
compute the residual r = y - p_oof on training rows and ask whether its conditional mean depends on
anything in a way that replicates out of sample.

The subtlety that makes this honest
-----------------------------------
p_oof is out-of-fold for every training row, so (y, p) pairs on training rows are themselves
honest. But a *correction* estimated on the same rows it is applied to would still be a leak, and
it would look spectacular. So the design is strictly two-sided:

  DISCOVERY     estimate per-group residual bias using only the discovery rows' (y, p)
  CONFIRMATION  apply those frozen numbers to confirmation rows and score the AUC change there

Two numbers are reported for every grouping, and the second is the one that decides:

  signed-bias replication   correlation between the bias estimated on discovery and the bias
                            actually observed on confirmation, over groups present in both, plus
                            the sign-agreement rate. If a structure is real it replicates with a
                            positive correlation. If it is noise-fitted, discovery bias and
                            confirmation bias are uncorrelated.
  cross-fitted AUC delta    for each fold k, the bias is estimated on folds != k and applied to
                            fold k only. This is the number that would matter operationally.

Aggregation is by fold, never pooled, so every reported delta is out-of-sample for the rows it is
measured on.

Shrinkage is pre-declared, not tuned: bias_g = sum_g r / (n_g + PRIOR_N) with PRIOR_N = 50, and
groups below MIN_N = 200 rows are dropped. Both constants are fixed before any result is seen, so
the "many tiny groups fit noise" failure mode cannot be selected for post hoc.

Usage:
  python scripts/residual_structure.py --keys raw,te,pattern,route,teacher
  python scripts/residual_structure.py --calibration
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.s6e10 import META4, NUMS, SURVEY13, TE_KEYS_DEFAULT  # noqa: E402
from src.features.view import RAW21  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

PRIOR_N = 50        # pre-declared shrinkage strength; NOT tuned
MIN_N = 200         # pre-declared minimum group size; NOT tuned
N_BINS = 24         # quantile bins for continuous keys
FD, AGE = "Flight Distance", "Age"


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_teacher_columns(view: str = "full") -> dict[str, np.ndarray]:
    """Pull the original-data teacher columns out of the cached static feature matrix.

    They are NOT columns on the raw train frame -- `ogte_*` / `ogs_*` are assembled inside the
    feature view from original-dataset label statistics -- so reading them off `tr` silently yields
    an empty set and the teacher grouping would be skipped without any error. Reading them from the
    view cache is the only correct source. The view is static and fold-independent, so using it
    here introduces no leakage.
    """
    import json as _json

    from src.common import FEATURES
    npz, jsn = FEATURES / f"static_{view}.npz", FEATURES / f"static_{view}.json"
    if not npz.exists():
        print(f"  [warn] no cached static view '{view}' -> teacher keys skipped")
        return {}
    names = _json.loads(jsn.read_text(encoding="utf-8"))
    names = names["names"] if isinstance(names, dict) and "names" in names else names
    want = [i for i, n in enumerate(names) if n.startswith(("ogte_", "ogs_", "teach_"))]
    if not want:
        print(f"  [warn] cached view '{view}' has no ogte_/ogs_/teach_ columns")
        return {}
    z = np.load(npz)
    trm = z["tr"]
    out = {names[i]: np.asarray(trm[:, i], dtype="float64") for i in want}
    print(f"  teacher columns from cached view '{view}': {len(out)} "
          f"({sorted(out)[:4]} ...)")
    return out


# --------------------------------------------------------------------------- grouping keys
def make_keys(tr: pd.DataFrame, p: np.ndarray, sets: set[str],
              extra_cols: dict[str, np.ndarray] | None = None) -> dict[str, pd.Series]:
    """Build label-free grouping keys. Nothing here may touch the target."""
    keys: dict[str, pd.Series] = {}

    if "raw" in sets:
        for c in RAW21:
            s = tr[c]
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > N_BINS:
                # quantile bins keep group sizes comparable; ties collapse, which is fine
                keys[f"raw:{c}"] = pd.qcut(s, N_BINS, duplicates="drop").astype(str)
            else:
                keys[f"raw:{c}"] = s.astype(str)

    if "te" in sets:
        # the exact keys the champion's existing TE block conditions on
        cols = {c for cs in TE_KEYS_DEFAULT.values() for c in cs} - {"_fd_bin500", "_fd_bin1000",
                                                                     "_fd_m1000", "_fd_bin250"}
        cols |= {FD, AGE, "Class"}
        for c in sorted(cols):
            if c not in tr.columns:
                continue
            s = tr[c]
            keys[f"te:{c}"] = (pd.qcut(s, N_BINS, duplicates="drop").astype(str)
                               if pd.api.types.is_numeric_dtype(s) and s.nunique() > N_BINS
                               else s.astype(str))
        # the two cross keys the champion actually uses
        for nm, cs in (("te:age_class_trip_cust", [AGE, "Class", "Type of Travel", "Customer Type"]),
                       ("te:fd_class", [FD, "Class"]),
                       ("te:fd_trip", [FD, "Type of Travel"])):
            keys[nm] = tr[cs].astype(str).agg("|".join, axis=1)

    if "pattern" in sets:
        # which survey columns are zero -- a coverage pattern, not a level
        z = (tr[SURVEY13] == 0).astype("int8")
        keys["pat:survey_zero_signature"] = z.astype(str).agg("".join, axis=1)
        keys["pat:n_zero"] = z.sum(axis=1).astype(str)
        keys["pat:n_max5"] = (tr[SURVEY13] == 5).sum(axis=1).astype(str)
        keys["pat:mean_survey"] = pd.qcut(tr[SURVEY13].mean(axis=1), N_BINS,
                                          duplicates="drop").astype(str)
        keys["pat:delay_any"] = ((tr["Departure Delay in Minutes"] > 0).astype(str)
                                 + "_" + (tr["Arrival Delay in Minutes"] > 0).astype(str))

    if "route" in sets:
        fdq = pd.qcut(tr[FD], 12, duplicates="drop").astype(str)
        keys["route:fd_x_class"] = fdq + "|" + tr["Class"].astype(str)
        keys["route:fd_x_trip"] = fdq + "|" + tr["Type of Travel"].astype(str)
        keys["route:fd_x_cust"] = fdq + "|" + tr["Customer Type"].astype(str)
        keys["route:class_x_trip_x_gender"] = (tr["Class"].astype(str) + "|"
                                               + tr["Type of Travel"].astype(str) + "|"
                                               + tr["Gender"].astype(str))
        keys["route:age_x_class"] = (pd.qcut(tr[AGE], 8, duplicates="drop").astype(str) + "|"
                                     + tr["Class"].astype(str))

    if "teacher" in sets:
        og = (extra_cols or {})
        if og:
            # a single original-teacher composite, binned
            comp = pd.DataFrame(og).mean(axis=1)
            keys["teach:composite"] = pd.qcut(comp, N_BINS, duplicates="drop").astype(str)
            # and the strongest few individually
            ranked = (pd.DataFrame(og).corrwith(pd.Series(p, name="p")).abs()
                      .sort_values(ascending=False).index.tolist())
            for c in ranked[:6]:
                v = pd.Series(og[c])
                keys[f"teach:{c}"] = (pd.qcut(v, N_BINS, duplicates="drop").astype(str)
                                      if v.nunique() > N_BINS else v.round(4).astype(str))
        else:
            print("  [warn] teacher keys requested but no teacher columns were available")

    if "conf" in sets:
        # the champion's own confidence: residual structure that lives only where the model is
        # unsure would be invisible to raw-column keys
        keys["conf:p_decile"] = pd.qcut(p, 10, labels=False).astype(str)
        keys["conf:p_decile_x_class"] = (pd.qcut(p, 10, labels=False).astype(str) + "|"
                                         + tr["Class"].astype(str))

    return keys


# --------------------------------------------------------------------------- bias machinery
def estimate_bias(g: pd.Series, r: np.ndarray, rows: np.ndarray) -> tuple[dict, float]:
    """Shrinkage-averaged residual bias per group, from `rows` only."""
    sub = g.iloc[rows]
    rs = r[rows]
    codes, uniq = pd.factorize(sub, sort=True)
    cnt = np.bincount(codes, minlength=len(uniq)).astype("float64")
    tot = np.bincount(codes, weights=rs, minlength=len(uniq))
    bias = tot / (cnt + PRIOR_N)
    keep = cnt >= MIN_N
    return {u: float(b) for u, b, k in zip(uniq, bias, keep) if k}, float(np.abs(bias[keep]).sum())


def apply_bias(g: pd.Series, rows: np.ndarray, table: dict) -> np.ndarray:
    if not table:
        return np.zeros(len(rows))
    return np.array([table.get(u, 0.0) for u in g.iloc[rows]], dtype="float64")


def observed_bias(g: pd.Series, r: np.ndarray, rows: np.ndarray) -> dict:
    sub = g.iloc[rows]
    rs = r[rows]
    codes, uniq = pd.factorize(sub, sort=True)
    cnt = np.bincount(codes, minlength=len(uniq)).astype("float64")
    tot = np.bincount(codes, weights=rs, minlength=len(uniq))
    return {u: float(t / n) for u, t, n in zip(uniq, tot, cnt) if n >= MIN_N}


def evaluate_key(name: str, g: pd.Series, y: np.ndarray, p: np.ndarray, folds: np.ndarray,
                 idx: np.ndarray, base_auc: float) -> dict:
    r = y - p
    fold_ids = sorted(set(folds[idx].tolist()))
    deltas, reps, signs = [], [], []
    for k in fold_ids:
        disc = idx[folds[idx] != k]      # discovery rows
        conf = idx[folds[idx] == k]      # confirmation rows
        table, _ = estimate_bias(g, r, disc)
        add = apply_bias(g, conf, table)
        a0 = float(roc_auc_score(y[conf], p[conf]))
        a1 = float(roc_auc_score(y[conf], logit(p[conf]) + add))
        deltas.append(a1 - a0)
        # replication of the SIGNED bias on the same groups, confirmation side
        obs = observed_bias(g, r, conf)
        common = [u for u in table if u in obs]
        if len(common) >= 5:
            d = np.array([table[u] for u in common])
            o = np.array([obs[u] for u in common])
            reps.append(float(np.corrcoef(d, o)[0, 1]))
            signs.append(float(np.mean(np.sign(d) == np.sign(o))))
    if not deltas:
        return {"key": name, "n_groups": int(g.iloc[idx].nunique()), "verdict": "no usable groups"}
    return {
        "key": name,
        "n_groups": int(g.iloc[idx].nunique()),
        "groups_kept": int(sum(1 for u in g.iloc[idx].unique()
                               if (g.iloc[idx] == u).sum() >= MIN_N)),
        "mean_delta_e5": float(np.mean(deltas) * 1e5),
        "fold_deltas_e5": [float(d * 1e5) for d in deltas],
        "folds_positive": int(sum(d > 0 for d in deltas)),
        "n_folds": len(deltas),
        "bias_replication_r": float(np.mean(reps)) if reps else None,
        "sign_agreement": float(np.mean(signs)) if signs else None,
    }


def calibration_check(y, p, folds, idx):
    """Is the champion miscalibrated, and would an honest recalibration recover AUC at all?

    AUC is invariant to any strictly increasing monotone transform, so recalibration can only help
    through NON-monotone behaviour. A monotone fit is included precisely as a control: it must move
    AUC by ~0, which validates that the harness can detect a real effect.
    """
    res = {"base_auc": float(roc_auc_score(y[idx], p[idx])), "mean_residual": float((y[idx] - p[idx]).mean())}

    # cross-fitted Platt scaling
    pl = np.zeros(len(idx))
    for k in sorted(set(folds[idx].tolist())):
        d = idx[folds[idx] != k]
        c = idx[folds[idx] == k]
        z = logit(p[d])
        X = np.column_stack([np.ones(len(d)), z])
        w = np.linalg.solve(X.T @ X + 1e-6 * np.eye(2), X.T @ (y[d] - p[d]))
        pl[np.isin(idx, c)] = logit(p[c]) * w[1]
    res["platt_crossfit_auc"] = float(roc_auc_score(y[idx], pl))
    res["platt_delta_e5"] = (res["platt_crossfit_auc"] - res["base_auc"]) * 1e5

    # cross-fitted isotonic (non-monotone-in-principle, still a control on whether non-monotone
    # corrections have any room at all)
    from sklearn.isotonic import IsotonicRegression
    iso = np.zeros(len(idx))
    for k in sorted(set(folds[idx].tolist())):
        d = idx[folds[idx] != k]
        c = idx[folds[idx] == k]
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p[d], y[d])
        iso[np.isin(idx, c)] = ir.predict(p[c])
    res["isotonic_crossfit_auc"] = float(roc_auc_score(y[idx], iso))
    res["isotonic_delta_e5"] = (res["isotonic_crossfit_auc"] - res["base_auc"]) * 1e5
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="blend_v3_final")
    ap.add_argument("--scheme", default="primary")
    ap.add_argument("--view", default="full")
    ap.add_argument("--keys", default="raw,te,pattern,route,teacher,conf")
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--tag", default="residual_structure")
    args = ap.parse_args()

    t0 = time.time()
    tr, _te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    y_int = tr[TARGET].values.astype("int8")
    folds = get_scheme(args.scheme, y_int, tr[ID_COL]).folds
    p = store.load_oof(args.pred).astype("float64")
    idx = np.arange(len(y))

    base = float(roc_auc_score(y, p))
    print(f"base OOF ({args.pred}) = {base:.6f}   rows={len(y):,}   scheme={args.scheme}")
    print(f"pre-declared constants: PRIOR_N={PRIOR_N}  MIN_N={MIN_N}  N_BINS={N_BINS}  "
          f"(fixed before results, not tuned)\n")

    sets = {s.strip() for s in args.keys.split(",") if s.strip()}
    extra_cols = load_teacher_columns(args.view) if "teacher" in sets else {}
    keys = make_keys(tr, p, sets, extra_cols)
    print(f"{len(keys)} candidate groupings\n")

    cal = calibration_check(y, p, folds, idx)
    print("CALIBRATION (cross-fitted; the monotone rows are controls that must move ~0)")
    print(f"  mean residual            = {cal['mean_residual']:+.6f}")
    print(f"  Platt   (monotone)       = {cal['platt_crossfit_auc']:.6f}  "
          f"{cal['platt_delta_e5']:+.2f}e-5   <- control, should be ~0")
    print(f"  Isotonic                 = {cal['isotonic_crossfit_auc']:.6f}  "
          f"{cal['isotonic_delta_e5']:+.2f}e-5\n")

    print(f"scanning {len(keys)} groupings ...", flush=True)
    rows = []
    for i, (nm, g) in enumerate(keys.items()):
        rows.append(evaluate_key(nm, g, y, p, folds, idx, base))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(keys)}", flush=True)

    scored = [r for r in rows if "mean_delta_e5" in r]
    scored.sort(key=lambda r: -r["mean_delta_e5"])
    print(f"\n{'='*118}")
    print(f"TOP {args.top} BY CROSS-FITTED AUC DELTA   (gate: mean >= +1.5e-5 AND "
          f">=4/5 folds positive AND bias replicates)")
    print("=" * 118)
    print(f"  {'key':<34}{'groups':>7}{'kept':>7}{'mean d':>10}{'pos':>6}{'repl r':>9}"
          f"{'sign agr':>10}  fold deltas (e-5)")
    print(f"  {'-'*118}")
    for r in scored[:args.top]:
        rr = f"{r['bias_replication_r']:.3f}" if r["bias_replication_r"] is not None else "  -"
        sa = f"{r['sign_agreement']:.3f}" if r["sign_agreement"] is not None else "  -"
        fd = " ".join(f"{d:+.1f}" for d in r["fold_deltas_e5"])
        print(f"  {r['key'][:33]:<34}{r['n_groups']:>7}{r['groups_kept']:>7}"
              f"{r['mean_delta_e5']:>+9.2f}e{r['folds_positive']}/{r['n_folds']:<4}"
              f"{rr:>9}{sa:>10}  {fd}")

    # --- the actual verdict: replicate AND gain, not gain alone ---
    winners = [r for r in scored
               if r["mean_delta_e5"] >= 1.5
               and r["folds_positive"] >= 4
               and r["bias_replication_r"] is not None and r["bias_replication_r"] >= 0.30]
    print(f"\n{'='*118}")
    if winners:
        print("GROUPS WHILE BIAS REPLICATES AND CLEAR THE GATE:")
        for r in winners:
            print(f"  {r['key']}  mean {r['mean_delta_e5']:+.2f}e-5  "
                  f"{r['folds_positive']}/{r['n_folds']} folds  "
                  f"replication r={r['bias_replication_r']:.3f}  "
                  f"sign agreement {r['sign_agreement']:.3f}")
        verdict = "RESIDUAL STRUCTURE FOUND -- targeted correction is worth building"
    else:
        best_repl = max((r["bias_replication_r"] for r in scored
                         if r["bias_replication_r"] is not None), default=None)
        print("RESIDUAL STRUCTURE EXHAUSTED.")
        print("  No grouping simultaneously (a) gains >= +1.5e-5 cross-fitted, (b) is positive in")
        print("  >=4 of 5 folds, and (c) shows signed-bias replication with the champion's OOF")
        print(f"  prediction. Best replication correlation seen across all {len(scored)} groupings: "
              f"{best_repl if best_repl is None else round(best_repl, 3)}")
        print("  Larger cross-fitted gains exist only where replication is absent or negative, which")
        print("  is the signature of fitting group noise rather than finding structure.")
        verdict = "RESIDUAL STRUCTURE EXHAUSTED -- do not build a correction"

    out = {"pred": args.pred, "scheme": args.scheme, "base_auc": base, "calibration": cal,
           "constants": {"PRIOR_N": PRIOR_N, "MIN_N": MIN_N, "N_BINS": N_BINS},
           "n_groupings": len(rows), "top": scored[:args.top], "winners": winners,
           "verdict": verdict, "seconds": round(time.time() - t0, 1),
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()[:12]}
    save_json(out, REPORTS / f"{args.tag}.json")
    print(f"\nVERDICT: {verdict}")
    print("wrote", REPORTS / f"{args.tag}.json", f"({out['seconds']}s)")


if __name__ == "__main__":
    main()
