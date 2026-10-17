"""H19-B evaluation: aux/portfolio AUC, paired delta, exact pair flips, segments, decision.

Reads only the saved prediction vectors from the bag-of-five run (no retraining). Applies
the frozen decision policy in configs/phase19/h19b_prereg.json. Recomputes every metric
from actual vectors; portfolio gain is never inferred from auxiliary standalone gain.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import (ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit,
                        load_cached_parquet, save_json)
from src.validation.compare import logit
from src.validation.folds import get_scheme
from src.phase19.h19b_bag5 import PSEUDOTEST_FOLD, exact_pair_flips, paired_se_auc

PREREG = Path("configs/phase19/h19b_prereg.json")
TAG = "phase19_h19b_bag5"
# frozen thresholds (mirror of prereg; asserted equal at runtime)
MEANINGFUL_PORT = 3e-5
MEANINGFUL_AUX = 3e-5
MIN_SIG_MULT = 3.0


def log(msg):
    print(f"[h19b-eval] {msg}", flush=True)


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def quartile_segments(col):
    """Equal-count quartile bins (label-free); returns list of boolean masks."""
    qs = np.quantile(col, [0.25, 0.5, 0.75])
    bounds = [-np.inf, qs[0], qs[1], qs[2], np.inf]
    return [((col > bounds[i]) & (col <= bounds[i + 1])) for i in range(4)]


def build_segments(tr, val_idx):
    """Predeclared label-free segments over the pseudo-test rows."""
    segs = {}
    col = lambda name: tr[name].to_numpy()[val_idx]
    for name in ("Gender", "Customer Type", "Type of Travel", "Class"):
        c = col(name)
        for v in np.unique(c):
            segs[f"{name}={v}"] = c == v
    age = col("Age").astype(np.float64)
    dist = col("Flight Distance").astype(np.float64)
    dep = col("Departure Delay in Minutes").astype(np.float64)
    arr = col("Arrival Delay in Minutes").astype(np.float64)
    for i, m in enumerate(quartile_segments(age)):
        segs[f"Age_q{i + 1}"] = m
    for i, m in enumerate(quartile_segments(dist)):
        segs[f"FlightDistance_q{i + 1}"] = m
    segs["DepartureDelay>0"] = dep > 0
    segs["DepartureDelay==0"] = dep == 0
    segs["ArrivalDelay>0"] = arr > 0
    segs["ArrivalDelay==0"] = arr == 0
    return segs


def decide(port_delta, port_se, aux_delta):
    sig = port_se > 0 and port_delta >= MIN_SIG_MULT * port_se
    if port_delta < -MEANINGFUL_PORT or (port_delta <= -MEANINGFUL_PORT
                                         and abs(port_delta) >= MIN_SIG_MULT * port_se):
        cls = "NEGATIVE"
    elif port_delta >= MEANINGFUL_PORT and sig and aux_delta > 0:
        cls = "PROMISING_DIAGNOSTIC"
    else:
        cls = "NEUTRAL"
    return cls, {"port_delta": port_delta, "port_se": port_se,
                 "sig_multiple": (port_delta / port_se if port_se > 0 else float("inf")),
                 "aux_delta": aux_delta, "classification": cls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = REPORTS / "phase19"
    out_json = out_dir / "h19b_bag5_f4.json"
    if out_json.exists() and not args.force:
        raise FileExistsError("Preserve measured H19-B result (use --force to overwrite)")

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    frozen = prereg["decision_policy"]["frozen_thresholds"]
    assert frozen["meaningful_portfolio_delta"] == MEANINGFUL_PORT
    assert frozen["meaningful_aux_delta"] == MEANINGFUL_AUX
    assert frozen["min_significant_se_multiple"] == MIN_SIG_MULT
    log(f"prereg thresholds verified ({file_sha256(PREREG)[:12]})")

    root = ARTIFACTS / TAG
    aux_ctrl_logits = np.load(root / "aux_ctrl_logits.npy")
    aux_bag_logits = np.load(root / "aux_bag_logits.npy")
    port_ctrl = np.load(root / "port_ctrl.npy")
    port_treat = np.load(root / "port_treat.npy")
    pids = np.load(root / "pseudotest_ids.npy")

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds
    val_idx = np.flatnonzero(folds == PSEUDOTEST_FOLD)
    assert np.array_equal(pids, ids[val_idx]), "pseudo-test row-ID order drift"
    yv = y[val_idx]

    # AUCs
    aux_ctrl_prob = 1.0 / (1.0 + np.exp(-aux_ctrl_logits))
    aux_bag_prob = 1.0 / (1.0 + np.exp(-aux_bag_logits))
    auc = roc_auc_score
    aux_ctrl_auc = float(auc(yv, aux_ctrl_prob))
    aux_bag_auc = float(auc(yv, aux_bag_prob))
    port_ctrl_auc = float(auc(yv, port_ctrl))
    port_treat_auc = float(auc(yv, port_treat))
    aux_delta = aux_bag_auc - aux_ctrl_auc
    port_delta = port_treat_auc - port_ctrl_auc
    log(f"aux  ctrl={aux_ctrl_auc:.10f} bag5={aux_bag_auc:.10f} delta={aux_delta:+.3e}")
    log(f"port ctrl={port_ctrl_auc:.10f} bag5={port_treat_auc:.10f} delta={port_delta:+.3e}")

    # paired SE (significance expression only; not for selection)
    port_se = paired_se_auc(yv, port_ctrl, port_treat)
    log(f"paired portfolio SE={port_se:.3e}  delta/SE={port_delta/port_se:+.2f}")

    # exact pair rescue/damage
    pf = exact_pair_flips(yv, port_ctrl, port_treat)
    # The identity (exact pair credit == roc_auc delta * total pairs) holds to float64
    # precision. roc_auc_score uses the Mann-Whitney rank-sum path while the pair counts
    # accumulate exact credits; at ~4.83e9 pairs these two equal-in-the-limit paths differ
    # by ~3e-13 relative (abs ~4.5e-7). Assert RELATIVE identity -- a real arithmetic bug
    # would show ~1e0 relative error -- rather than an absolute 1e-9 that float noise at
    # this scale legitimately exceeds. This is an internal sanity check, not a decision
    # threshold; the frozen decision (>=3e-5 meaningful, >=3x paired SE) is untouched.
    _rel = pf["identity_abs_error"] / max(1.0, abs(pf["net_credit"]))
    assert _rel < 1e-9, {**pf, "identity_rel_error": _rel}
    pf["identity_rel_error"] = _rel
    log(f"pair net_credit={pf['net_credit']:+.1f} identity_abs={pf['identity_abs_error']:.2e} "
        f"identity_rel={_rel:.2e}")

    # correlations
    sp = spearmanr(port_ctrl, port_treat).statistic
    lp = pearson(logit(np.clip(port_ctrl, 1e-7, 1 - 1e-7)),
                 logit(np.clip(port_treat, 1e-7, 1 - 1e-7)))

    # prediction-scale changes
    dprob = port_treat.astype(np.float64) - port_ctrl.astype(np.float64)
    adp = np.abs(dprob)
    scale = {"mean_abs": float(adp.mean()), "median_abs": float(np.median(adp)),
             "p95_abs": float(np.quantile(adp, 0.95)), "max_abs": float(adp.max()),
             "p99_abs": float(np.quantile(adp, 0.99)),
             "sign_flip_frac": float(((port_treat - 0.5) * (port_ctrl - 0.5) < 0).mean())}

    # segments
    segs = build_segments(tr, val_idx)
    seg_rows = []
    for name, m in segs.items():
        m = np.asarray(m)
        if m.sum() < 200 or yv[m].min() == yv[m].max():
            continue
        pc = float(auc(yv[m], port_ctrl[m]))
        pt = float(auc(yv[m], port_treat[m]))
        seg_rows.append({"segment": name, "rows": int(m.sum()), "pos_rate": float(yv[m].mean()),
                         "port_ctrl_auc": pc, "port_bag5_auc": pt, "delta": pt - pc})
    seg_rows.sort(key=lambda r: r["delta"])

    cls, detail = decide(port_delta, port_se, aux_delta)

    result = {
        "hypothesis_id": "H19-B",
        "verdict": cls,
        "prereg_sha256": file_sha256(PREREG),
        "git": git_commit(),
        "pseudotest": "primary fold 4",
        "rows": {"fit": int((folds != PSEUDOTEST_FOLD).sum()),
                 "pseudotest": int(len(val_idx))},
        "control": {"policy": "single full-FIT refit", "tag": "sol_clean_aux10_primary",
                    "bit_exact_vs_phase17": True, "identity_gap": 0.0},
        "treatment": {"policy": "bag-of-five inner-fold mean-logit", "roles": 10, "bags": 5,
                      "tag": TAG},
        "aux": {"ctrl_auc": aux_ctrl_auc, "bag5_auc": aux_bag_auc,
                "delta": aux_delta},
        "portfolio": {"ctrl_auc": port_ctrl_auc, "bag5_auc": port_treat_auc,
                      "delta": port_delta, "paired_se": port_se},
        "decision": detail,
        "pairs": pf,
        "correlations": {"spearman": float(sp), "logit_pearson": lp},
        "prediction_scale": scale,
        "segments": seg_rows,
        "limitations": [
            "Pseudo-test standard: OOF is unchanged; this cannot pass the admission gate.",
            "Fold 4 is part of the primary OOF and was used in earlier research; not a hidden holdout.",
            "A pseudo-test gain is not proof of private performance; S6E9 shows policy deltas do not transfer cleanly.",
        ],
        "budget": {"cpu_hours_estimate": 8, "gpu": "CPU-classical (light cuda XGB preserved)"},
        "submissions_used": 0,
    }
    save_json(result, out_json)

    # markdown
    md = out_dir / "h19b_bag5_f4.md"
    lines = ["# H19-B — aux10 deployment policy: single full-FIT vs bag-of-five (fold 4)", ""]
    lines.append(f"**Verdict:** `{cls}` · portfolio Δ = {port_delta:+.3e} "
                 f"(paired SE {port_se:.2e}, {port_delta/port_se:+.1f}×SE)")
    lines.append("")
    lines.append("| metric | control (full-FIT) | treatment (bag-of-five) | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(f"| aux10 standalone AUC | {aux_ctrl_auc:.10f} | {aux_bag_auc:.10f} | {aux_delta:+.3e} |")
    lines.append(f"| portfolio AUC | {port_ctrl_auc:.10f} | {port_treat_auc:.10f} | {port_delta:+.3e} |")
    lines.append("")
    lines.append(f"- pair net credit {pf['net_credit']:+.1f} (identity err {pf['identity_abs_error']:.1e}); "
                 f"ctrl right/tie/wrong {pf['ctrl_right']}/{pf['ctrl_tie']}/{pf['ctrl_wrong']}, "
                 f"bag5 {pf['treat_right']}/{pf['treat_tie']}/{pf['treat_wrong']}")
    lines.append(f"- Spearman {sp:.6f} · logit-Pearson {lp:.6f}")
    lines.append(f"- |Δp| mean {scale['mean_abs']:.2e} · median {scale['median_abs']:.2e} · "
                 f"p95 {scale['p95_abs']:.2e} · max {scale['max_abs']:.2e} · "
                 f"sign-flip {scale['sign_flip_frac']:.4f}")
    lines.append("")
    lines.append("## Segment effects (portfolio Δ, sorted)")
    lines.append("| segment | rows | pos rate | Δ |")
    lines.append("|---|---|---|---|")
    for r in seg_rows:
        lines.append(f"| {r['segment']} | {r['rows']} | {r['pos_rate']:.3f} | {r['delta']:+.3e} |")
    lines.append("")
    lines.append("## Limitations")
    for l in result["limitations"]:
        lines.append(f"- {l}")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {out_json} and {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
