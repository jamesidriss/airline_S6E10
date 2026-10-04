"""Fold-0 decorrelation probe for TabR, per the predeclared decision gate.

The predeclared gate on primary fold 0 is:

    < 0.959301                    stop TabR
    0.959301 .. 0.9603            weak but potentially orthogonal -> CONTINUE ONLY IF the
                                   predictions are unusually decorrelated
    0.9603 .. 0.9610              promising -> proceed to trusted multi-fold
    >= 0.9610                     very strong
    >= 0.96130                    major result

Measured TabR fold-0 AUC is 0.960282, i.e. the second band. The rule for that band is explicit and
does not rest on standalone AUC: it requires measuring decorrelation against the finalist, the best
extra_trees LightGBM and the best RealMLP, and checking whether FIXED small blend weights (2%, 5%,
10%) show plausible positive marginal behaviour.

These fixed weights are DIAGNOSTIC ONLY and are evaluated on the same fold they are chosen on, so
they carry no evidential weight of their own -- a positive result here only justifies spending compute
on a full 5-fold where the marginal value can be measured honestly. Nothing here is a substitute for
the nested meta-validation that scripts/evaluate_tabr.py performs once a full OOF exists.

Usage:
  python scripts/tabr_fold0_decorrelation.py --tabr-dir reports/tabr_runs/<exp>/fold0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.ensemble import lab  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

REFERENCES = {
    "finalist_v3": "blend_v3_final",
    "xt_lightgbm": "z4_xt_f10_s4",
    "realmlp": "prod5_realmlp_full_primary_s7_e6",
    "tabm": "z5_tabm_e25",
}

# Predeclared gate boundaries -- recorded so they cannot drift after seeing the result.
GATE = {"stop": 0.959301, "weak_top": 0.9603, "promising_top": 0.9610, "major": 0.96130}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabr-dir", required=True, help=".../<exp_id>/fold0 containing oof.npy")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--fixed-weights", default="0.02,0.05,0.10")
    args = ap.parse_args()

    d = Path(args.tabr_dir)
    oof_all = np.load(d / "oof.npy").astype("float64")
    meta = json.loads((d / "fold_complete.json").read_text(encoding="utf-8"))
    val_idx = np.where(get_scheme(args.folds,
                                  load_cached_parquet()[0][TARGET].values.astype("int8"),
                                  load_cached_parquet()[0][ID_COL]).folds == args.fold)[0]
    if len(oof_all) != len(val_idx):
        raise SystemExit(f"oof.npy has {len(oof_all)} rows but fold {args.fold} has {len(val_idx)}")

    tr, _te = load_cached_parquet()
    y_all = tr[TARGET].values.astype("int8")
    y = y_all[val_idx]
    tabr = oof_all
    auc = float(roc_auc_score(y, tabr))

    print("=" * 92)
    print(f"TabR fold-{args.fold} decorrelation probe   exp_id={meta['cfg_hash']}")
    print("=" * 92)
    print(f"  TabR fold AUC        = {auc:.6f}")
    print(f"  predeclared stop     = {GATE['stop']:.6f}   weak/decorrelated band = "
          f"[{GATE['stop']:.6f}, {GATE['weak_top']:.4f})")

    # ---------------------------------------------------------------- correlations
    print("\n--- correlation vs existing families (same rows) ---")
    corrs = {}
    for label, name in REFERENCES.items():
        try:
            ref = store.load_oof(name).astype("float64")[val_idx]
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<14} UNAVAILABLE ({type(exc).__name__})")
            continue
        row = {"member": name,
               "auc": float(roc_auc_score(y, ref)),
               "pearson_prob": corr(tabr, ref),
               "pearson_logit": corr(lab.tform(tabr, "logit"), lab.tform(ref, "logit")),
               "spearman": spearman(tabr, ref)}
        corrs[label] = row
        print(f"  {label:<14} AUC={row['auc']:.6f}  pearson_prob={row['pearson_prob']:.5f}  "
              f"pearson_logit={row['pearson_logit']:.5f}  spearman={row['spearman']:.5f}")

    # Context: how correlated are the EXISTING families with each other? If TabR's correlation sits
    # inside that band it is just another near-clone; if it is clearly below it, it is genuinely
    # different. This comparison is what makes the numbers interpretable.
    try:
        pairs = []
        keys = [k for k in REFERENCES if k in corrs]
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                ra = store.load_oof(REFERENCES[a]).astype("float64")[val_idx]
                rb = store.load_oof(REFERENCES[b]).astype("float64")[val_idx]
                pairs.append((f"{a} vs {b}", corr(lab.tform(ra, "logit"), lab.tform(rb, "logit"))))
        print("\n--- for scale: logit correlation AMONG existing families ---")
        for nm, c in pairs:
            print(f"  {nm:<34} {c:.5f}")
        inter = [c for _, c in pairs]
        print(f"  range among existing pairs: {min(inter):.5f} .. {max(inter):.5f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (pairwise scale unavailable: {type(exc).__name__})")
        inter = []

    # ---------------------------------------------------------------- fixed-weight diagnostics
    fin = store.load_oof(REFERENCES["finalist_v3"]).astype("float64")[val_idx]
    base = float(roc_auc_score(y, fin))
    print("\n--- DIAGNOSTIC fixed-weight blends on the finalist (same fold, so indicative only) ---")
    print(f"  finalist alone        = {base:.6f}")
    blends = {}
    for w in [float(x) for x in args.fixed_weights.split(",")]:
        for space in ("logit", "prob"):
            if space == "logit":
                mix = w * lab.tform(tabr, "logit") + (1 - w) * lab.tform(fin, "logit")
            else:
                mix = w * tabr + (1 - w) * fin
            a = float(roc_auc_score(y, mix))
            blends[f"w={w}_{space}"] = a
            print(f"  +{w:>5.0%} TabR ({space:<5}) = {a:.6f}  ({a - base:+.6f})")

    best = max(blends.items(), key=lambda kv: kv[1])
    verdict = _verdict(auc, corrs, blends, inter, base)

    out = {"tabr_fold": int(args.fold), "tabr_auc": auc, "gate": GATE,
           "correlations": corrs, "existing_family_logit_corr_range": inter,
           "finalist_auc": base, "fixed_weight_diagnostics": blends,
           "best_diagnostic": {"name": best[0], "auc": best[1], "delta": best[1] - base},
           "verdict": verdict["verdict"], "reasons": verdict["reasons"]}
    save_json(out, REPORTS / f"tabr_fold{args.fold}_decorrelation.json")
    print(f"\nVERDICT: {verdict['verdict']}")
    for r in verdict["reasons"]:
        print(f"  - {r}")
    print("\nwrote", REPORTS / f"tabr_fold{args.fold}_decorrelation.json")


def _verdict(auc, corrs, blends, inter, base):
    reasons = []
    if auc < GATE["stop"]:
        return {"verdict": "STOP (catastrophically weak)",
                "reasons": [f"fold AUC {auc:.6f} < predeclared stop line {GATE['stop']:.6f}"]}
    reasons.append(f"fold AUC {auc:.6f} clears the predeclared stop line {GATE['stop']:.6f}")

    fin = corrs.get("finalist_v3", {})
    fl = fin.get("pearson_logit")
    band = ""
    decorrelated = False
    if fl is not None and inter:
        lo, hi = min(inter), max(inter)
        band = f"existing-family logit corr spans {lo:.5f}..{hi:.5f}"
        decorrelated = fl < lo
        reasons.append(f"TabR vs finalist logit corr {fl:.5f}; {band}")
        if decorrelated:
            reasons.append("TabR is MORE decorrelated from the finalist than any existing pair is "
                           "-- genuinely different errors, which is the point of running it")
        else:
            reasons.append("TabR sits INSIDE the existing correlation band, so on decorrelation "
                           "alone it is another near-clone")
    else:
        reasons.append("correlation scale unavailable")

    best_delta = max(v - base for v in blends.values())
    reasons.append(f"best fixed-weight diagnostic moves the finalist by {best_delta:+.6f}")
    if best_delta > 0:
        reasons.append("a fixed small TabR weight improves the finalist, so a trusted 5-fold is "
                       "warranted to measure the marginal value honestly")
    else:
        reasons.append("NO fixed small weight improves the finalist")

    if auc >= GATE["major"]:
        v = "PROCEED IMMEDIATELY (major: matches/beats the strongest single model on this fold)"
    elif best_delta > 0 and (decorrelated or fl is None):
        v = "PROCEED to trusted 5-fold (weak standalone but shows marginal value / decorrelation)"
    elif decorrelated:
        v = "BORDERLINE: run the 5-fold only if compute is cheap; do not treat as promising"
    else:
        v = "STOP (weak AND not decorrelated -- no reason to spend 5-fold compute)"
    return {"verdict": v, "reasons": reasons}


if __name__ == "__main__":
    main()
