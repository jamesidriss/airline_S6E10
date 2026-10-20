"""Phase-24 composition-geometry lab (H24-C1).

Discovery on the immutable primary folds, sensitivity on the certified shadow folds.
Frozen by configs/phase24/composition_lab_prereg.json; refuses to overwrite outputs;
no model fits, no GPU, no submissions.

    .\\.venv\\Scripts\\python.exe scripts\\phase24\\composition_lab.py --mode discovery
    .\\.venv\\Scripts\\python.exe scripts\\phase24\\composition_lab.py --mode shadow   # only on a discovery pass
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "phase24"))

import composition_rules as R  # noqa: E402
from src.common import arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.validation.paired_auc import holm_adjust, paired_auc  # noqa: E402

PREREG = ROOT / "configs/phase24/composition_lab_prereg.json"
OUT_DISCOVERY = ROOT / "reports/phase24/composition_lab.json"
OUT_SHADOW = ROOT / "reports/phase24/composition_lab_shadow.json"
ROUTE_OOF = ROOT / "artifacts/phase17_B1/primary/route_oof.npy"
AUX_OOF = ROOT / "artifacts/sol_route_aux10_primary_final/aux10_oof.npy"
PORT_OOF = ROOT / "artifacts/phase17_B1/primary/portfolio_oof.npy"
SHADOW_ROUTE = ROOT / "artifacts/phase17_shadow_B1"
SHADOW_AUX = ROOT / "artifacts/phase17_shadow_aux10"

GATE_DELTA = 1.5e-05
GATE_SE = 2.5
GATE_HOLM = 0.05
IDENTITY_TOL = 1e-6
SHADOW_REUSE_TOL = 2e-6


def score_block(y, s):
    return float(roc_auc_score(y, s))


def evaluate(name, y, folds, base, cand, fold_list):
    per = {}
    for k in fold_list:
        m = folds == k
        per[str(k)] = {
            "n": int(m.sum()),
            "auc_base": score_block(y[m], base[m]),
            "auc_cand": score_block(y[m], cand[m]),
        }
        per[str(k)]["delta"] = per[str(k)]["auc_cand"] - per[str(k)]["auc_base"]
    d = np.array([per[str(k)]["delta"] for k in fold_list])
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    pooled = paired_auc(y, cand, base)
    return {
        "name": name,
        "pooled_auc": score_block(y, cand),
        "mean_fold_delta": float(d.mean()),
        "fold_se": se,
        "t": float(d.mean() / se) if se and np.isfinite(se) else float("nan"),
        "folds_positive": int((d > 0).sum()),
        "n_folds": len(fold_list),
        "per_fold": per,
        "pooled_delong": pooled,
        "logit_corr_with_control": float(np.corrcoef(R.logit(cand), R.logit(base))[0, 1]),
        "vector_sha256": arr_sha256(np.asarray(cand, dtype=np.float32)),
    }


def discovery(spec):
    tr, _ = load_cached_parquet()
    y = tr.satisfaction.to_numpy(dtype="int8")
    ids = tr.id.to_numpy()
    folds = get_scheme("primary", y, ids).folds
    assert arr_sha256(folds) == spec["design"]["discovery"]["fold_sha256"], "primary fold drift"

    route = np.load(ROUTE_OOF).astype(np.float64)
    aux = np.load(AUX_OOF).astype(np.float64)
    port = np.load(PORT_OOF).astype(np.float64)
    assert route.shape == aux.shape == port.shape == (len(y),)
    base = R.c0_control(route, aux)
    identity = float(np.max(np.abs(base - port)))
    assert identity <= IDENTITY_TOL, f"C0 identity failed: {identity}"

    z_r, z_a = R.logit(route), R.logit(aux)
    out = {name: np.empty(len(y)) for name in R.RULE_NAMES}
    fits = {name: {} for name in R.RULE_NAMES}

    for k in range(5):
        fit = folds != k
        ev = folds == k
        assert not np.any(fit & ev)
        # C1
        w, fit_auc = R.c1_fit_weight(z_r[fit], z_a[fit], y[fit])
        out["C1_nested_weight"][ev] = R.c1_apply(w, z_r[ev], z_a[ev])
        fits["C1_nested_weight"][str(k)] = {"w": w, "fit_auc": fit_auc}
        # C2
        out["C2_block_rank"][ev] = R.c2_block_rank(route[ev], aux[ev])
        # C3
        srt_r, srt_a = R.c3_fit_cdf(route[fit]), R.c3_fit_cdf(aux[fit])
        out["C3_fit_cdf"][ev] = R.c3_apply(srt_r, route[ev]) + R.c3_apply(srt_a, aux[ev])
        fits["C3_fit_cdf"][str(k)] = {"n_fit": int(fit.sum())}
        # C4
        m4 = R.c4_fit(z_r[fit], z_a[fit], y[fit])
        out["C4_confidence_logistic"][ev] = m4.predict_proba(R.c4_features(z_r[ev], z_a[ev]))[:, 1]
        fits["C4_confidence_logistic"][str(k)] = {"coef": m4.coef_[0].tolist(), "intercept": float(m4.intercept_[0])}
        # C5
        cr, br = R.c5_fit_platt(z_r[fit], y[fit])
        ca, ba = R.c5_fit_platt(z_a[fit], y[fit])
        out["C5_platt_average"][ev] = R.sigmoid(0.5 * (R.c5_apply(z_r[ev], cr, br) + R.c5_apply(z_a[ev], ca, ba)))
        fits["C5_platt_average"][str(k)] = {"coef_route": cr, "int_route": br, "coef_aux": ca, "int_aux": ba}
        # C6 / C7
        out["C6_probability_mean"][ev] = R.c6_probability_mean(route[ev], aux[ev])
        out["C7_geometric_mean"][ev] = R.c7_geometric_mean(route[ev], aux[ev])

    results = [evaluate(n, y, folds, base, out[n], range(5)) for n in R.RULE_NAMES]
    pvals = [r["pooled_delong"]["one_sided_p"] for r in results]
    adj = holm_adjust(pvals)
    for r, p in zip(results, adj):
        r["holm_adjusted_p"] = float(p)
        r["gate"] = {
            "delta_ok": bool(r["mean_fold_delta"] >= GATE_DELTA),
            "se_ok": bool(r["t"] >= GATE_SE),
            "folds_ok": bool(r["folds_positive"] == 5),
            "holm_ok": bool(p <= GATE_HOLM),
        }
        r["passes"] = bool(all(r["gate"].values()))

    ranked = sorted(results, key=lambda r: -r["mean_fold_delta"])
    report = {
        "status": "COMPLETE",
        "protocol": spec["protocol"],
        "hypothesis_id": spec["hypothesis_id"],
        "prereg_sha256": file_sha256(PREREG),
        "source_sha256": file_sha256(Path(__file__)),
        "rules_source_sha256": file_sha256(ROOT / "scripts/phase24/composition_rules.py"),
        "git": git_commit(),
        "utc_run": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "identity_check_max_abs_vs_certified_portfolio": identity,
        "control": {
            "pooled_auc": score_block(y, base),
            "vector_sha256": arr_sha256(base.astype(np.float32)),
        },
        "fits": fits,
        "results": results,
        "ranking": [r["name"] for r in ranked],
        "gate": {
            "delta_min": GATE_DELTA,
            "se_multiple": GATE_SE,
            "folds_positive": "5/5",
            "holm_max": GATE_HOLM,
        },
        "any_pass": bool(any(r["passes"] for r in results)),
        "best": ranked[0]["name"],
        "limitations": [
            "Conditional on saved parent predictions; no refitting uncertainty.",
            "Primary folds have been used for selection across ~600 historical experiments; Holm covers these 7 rules only.",
            "FIT-row stacking features come from other fold models than eval-row features (standard stacking mismatch).",
        ],
    }
    save_json(report, OUT_DISCOVERY)
    print(json.dumps({
        "identity": identity,
        "control_auc": report["control"]["pooled_auc"],
        "ranking": [(r["name"], round(r["mean_fold_delta"], 9), r["folds_positive"], round(r["t"], 2),
                     round(r["holm_adjusted_p"], 4), r["passes"]) for r in ranked],
        "any_pass": report["any_pass"],
    }, indent=1))
    return 0


def shadow(spec):
    tr, _ = load_cached_parquet()
    y_all = tr.satisfaction.to_numpy(dtype="int8")
    ids = tr.id.to_numpy()
    folds = get_scheme("shadow", y_all, ids).folds
    assert arr_sha256(folds) == spec["design"]["confirmation_sensitivity"]["fold_sha256"], "shadow fold drift"
    fold_list = spec["design"]["confirmation_sensitivity"]["folds"]

    parts = {}
    for f in fold_list:
        rows = np.flatnonzero(folds == f)
        apply_ids = np.load(SHADOW_ROUTE / f"f{f}/apply_ids.npy")
        assert np.array_equal(rows, apply_ids), f"shadow row alignment mismatch f{f}"
        aux_logit = np.load(SHADOW_AUX / f"aux10_mean_logits_f{f}.npy").astype(np.float64)
        r_logits = np.stack([
            (lambda a: a[0, :, 1] - a[0, :, 0])(np.load(SHADOW_ROUTE / f"f{f}/member_{m}_raw_logits.npy"))
            for m in range(2)
        ])
        r_prob = np.mean(expit(r_logits), axis=0)
        base = expit(0.5 * R.logit(r_prob) + 0.5 * aux_logit)
        certified = np.load(SHADOW_ROUTE / f"f{f}/portfolio.npy").astype(np.float64)
        reuse = float(np.max(np.abs(base - certified)))
        assert reuse <= SHADOW_REUSE_TOL, f"shadow reuse gate failed f{f}: {reuse}"
        parts[f] = {"rows": rows, "y": y_all[rows], "r": r_prob, "a": expit(aux_logit),
                    "a_logit": aux_logit, "base": base, "reuse": reuse}

    out = {name: {} for name in R.RULE_NAMES}
    fits = {name: {} for name in R.RULE_NAMES}
    for f in fold_list:
        other = [g for g in fold_list if g != f][0]
        fit, ev = parts[other], parts[f]
        zr_f, za_f, y_f = R.logit(fit["r"]), R.logit(fit["a"]), fit["y"]
        zr_e, za_e = R.logit(ev["r"]), R.logit(ev["a"])
        w, _ = R.c1_fit_weight(zr_f, za_f, y_f)
        out["C1_nested_weight"][f] = R.c1_apply(w, zr_e, za_e)
        fits["C1_nested_weight"][str(f)] = {"w": w, "fit_fold": other}
        out["C2_block_rank"][f] = R.c2_block_rank(ev["r"], ev["a"])
        out["C3_fit_cdf"][f] = R.c3_apply(R.c3_fit_cdf(fit["r"]), ev["r"]) + R.c3_apply(R.c3_fit_cdf(fit["a"]), ev["a"])
        m4 = R.c4_fit(zr_f, za_f, y_f)
        out["C4_confidence_logistic"][f] = m4.predict_proba(R.c4_features(zr_e, za_e))[:, 1]
        fits["C4_confidence_logistic"][str(f)] = {"coef": m4.coef_[0].tolist(), "fit_fold": other}
        cr, br = R.c5_fit_platt(zr_f, y_f)
        ca, ba = R.c5_fit_platt(za_f, y_f)
        out["C5_platt_average"][f] = R.sigmoid(0.5 * (R.c5_apply(zr_e, cr, br) + R.c5_apply(za_e, ca, ba)))
        fits["C5_platt_average"][str(f)] = {"coef_route": cr, "coef_aux": ca, "fit_fold": other}
        out["C6_probability_mean"][f] = R.c6_probability_mean(ev["r"], ev["a"])
        out["C7_geometric_mean"][f] = R.c7_geometric_mean(ev["r"], ev["a"])

    y = np.concatenate([parts[f]["y"] for f in fold_list])
    base = np.concatenate([parts[f]["base"] for f in fold_list])
    results = []
    for name in R.RULE_NAMES:
        cand = np.concatenate([out[name][f] for f in fold_list])
        d = [score_block(parts[f]["y"], out[name][f]) - score_block(parts[f]["y"], parts[f]["base"]) for f in fold_list]
        results.append({
            "name": name,
            "pooled_auc": score_block(y, cand),
            "mean_fold_delta": float(np.mean(d)),
            "fold_deltas": {str(f): float(x) for f, x in zip(fold_list, d)},
            "both_positive": bool(all(x > 0 for x in d)),
            "pooled_delong": paired_auc(y, cand, base),
            "vector_sha256": arr_sha256(cand.astype(np.float32)),
        })
    ranked = sorted(results, key=lambda r: -r["mean_fold_delta"])
    report = {
        "status": "COMPLETE",
        "protocol": spec["protocol"] + "_shadow_sensitivity",
        "prereg_sha256": file_sha256(PREREG),
        "source_sha256": file_sha256(Path(__file__)),
        "git": git_commit(),
        "utc_run": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "folds": fold_list,
        "reuse_check_max_abs": {str(f): parts[f]["reuse"] for f in fold_list},
        "n_rows": int(len(y)),
        "control_pooled_auc": score_block(y, base),
        "fits": fits,
        "results": results,
        "ranking": [r["name"] for r in ranked],
        "limitations": [
            "Two shadow folds only (279,854 rows) and the same underlying competition data.",
            "Parameters are fitted leave-one-shadow-fold-out, so each arm has ~139,927 fitting rows only.",
            "Sensitivity analysis, not independent proof.",
        ],
    }
    save_json(report, OUT_SHADOW)
    print(json.dumps({"reuse": report["reuse_check_max_abs"],
                      "ranking": [(r["name"], round(r["mean_fold_delta"], 9), r["both_positive"]) for r in ranked]},
                     indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["discovery", "shadow"], required=True)
    args = ap.parse_args()
    spec = json.loads(PREREG.read_text(encoding="utf-8"))
    if args.mode == "discovery":
        if OUT_DISCOVERY.exists():
            raise SystemExit(f"refuse to overwrite {OUT_DISCOVERY}")
        return discovery(spec)
    if not OUT_DISCOVERY.exists():
        raise SystemExit("discovery report missing; shadow is only for a discovery pass")
    disc = json.loads(OUT_DISCOVERY.read_text(encoding="utf-8"))
    if not disc["any_pass"]:
        raise SystemExit("discovery gate did not pass; frozen protocol forbids the shadow arm")
    if OUT_SHADOW.exists():
        raise SystemExit(f"refuse to overwrite {OUT_SHADOW}")
    return shadow(spec)


if __name__ == "__main__":
    raise SystemExit(main())
