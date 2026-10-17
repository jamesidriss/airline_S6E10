"""S7 fidelity repair: update catalogue rows with correct fidelity/findings/status.

Applied AFTER prereg P20-S7-FIDELITY-REPAIR (sha c10cf3fe, commit ebd7bac) and the
resulting measurements. Rows 15/30 are NOT touched here (they wait for the bounded
rich-view re-runs). Each update is a correction of a misclassification found by the
S7 re-examination, never a deletion of history -- old text is preserved in the
finding where it matters.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"

# (row): dict of field updates
UPDATES = {
    17: {
        "fidelity_reached": "F2",
        "status": "COMPLETE_NEGATIVE",
        "phase20_finding": (
            "S7 repair (item 17): the earlier fold-0 screens used a SCALAR-cross variant "
            "(cross = x_0 * (1*(w.x)) + b, i.e. W = 1*w^T -- u fixed at ones), not canonical "
            "DCNv2 (full W) nor DCNv2-low-rank (learned u,v). Canonical DCNv2Full (full "
            "n_feat x n_feat cross matrix, deep branch + concat head otherwise identical) "
            "measured fold-0: route AUC 0.957458 / port_delta -9.818e-4 / corr 0.9862 "
            "(vs scalar proxy 0.957580 / -9.073e-4); full AUC 0.959041 / port_delta -9.779e-4 "
            "/ corr 0.9799 (vs scalar proxy 0.959898 / -5.827e-4). The canonical cross matrix "
            "is if anything WEAKER than the scalar gate on the rich view -- the negative is "
            "confirmed for canonical DCNv2 too, on both views. Cross-network family closed "
            "for this protocol (route plateau + rich-view port_delta < +5e-5)."
        ),
    },
    35: {
        "fidelity_reached": "F2",
        "status": "COMPLETE_NEGATIVE",
        "phase20_finding": (
            "S7 relabel: this is a TESTED-RECIPE negative (masked-rating CORAL aux on PLR "
            "trunk, fold-0 H20-MTL screen: 0.959963/port_delta -5.009e-4 and 0.959813/"
            "-5.576e-4 vs plain-PLR reference 0.960197/-4.191e-4, corr ~0.993 unchanged), "
            "NOT a negative on multi-task learning as a class. The coupling objective "
            "neither lifts nor decorrelates; reopening requires a DIFFERENT auxiliary "
            "signal (not masked ratings), per mandate S7. Fidelity was mislabeled F1 "
            "despite a full controlled outer-fold measurement -- corrected to F2."
        ),
    },
    36: {
        "fidelity_reached": "F2",
        "status": "COMPLETE_NEGATIVE",
        "phase20_finding": (
            "S7 relabel: recipe-scoped negative (fold-safe CORAL ordinal aux, measured with "
            "item 35 in the H20-MTL fold-0 screen; both arms slightly worse than plain PLR, "
            "corr unchanged ~0.993). This closes the ORDINAL-CORAL recipe, not ordinal "
            "auxiliary learning in general -- a different auxiliary formulation (e.g. "
            "explicit-quantity regression, ranking aux) is untested and may be reopened. "
            "Fidelity corrected F1 -> F2 (controlled outer-fold measurement exists)."
        ),
    },
    60: {
        "fidelity_reached": "F2",
        "status": "COMPLETE_NEGATIVE",
        "phase20_finding": (
            "S7 relabel: measured as a PROJECTION-FOREST PROXY (random Givens/Gaussian "
            "projection + ExtraTrees, n_components=64), NOT a classwise-PCA rotation forest. "
            "Fold-0 full view: AUC 0.947691 / port_delta -5.204e-3 / corr 0.8901 -- very "
            "decorrelated but far too weak. This negative applies to the projection-forest "
            "proxy only; an exact classwise-PCA rotation forest remains untested (blocked "
            "by memory cost of per-class PCA on 559k x 285, not by evidence). Fidelity "
            "corrected F1 -> F2 (controlled outer-fold measurement exists)."
        ),
    },
    63: {
        "fidelity_reached": "F2",
        "status": "COMPLETE_NEGATIVE",
        "phase20_finding": (
            "S7 relabel: measured as RBF KernelRidge on an 8000-row FIT SUBSAMPLE (40k-row "
            "kernel matrix was MemoryError), i.e. an exact method on a small subsample, not "
            "a Nyström/RFF approximation on full FIT. Fold-0 full view: AUC 0.955484 / corr "
            "0.7633 (very decorrelated, far too weak). Negative applies to the 8k-subsample "
            "proxy; the RFF/Nyström approximation on full FIT is a DIFFERENT experiment "
            "(item 64 -- now measured separately). Fidelity corrected F1 -> F2."
        ),
    },
    64: {
        "fidelity_reached": "F2",
        "status": "COMPLETE_NEGATIVE",
        "next_action": "CLOSED by direct measurement (was previously auto-closed without one)",
        "phase20_finding": (
            "S7 repair (item 64): this row had been closed by the classical-block addendum "
            "WITHOUT ever being run (only oblique + kernelridge artifacts existed) -- an "
            "auto-close from a related negative, which mandate S7 forbids. Now measured for "
            "real: RFF approximate RBF SVM (RBFSampler n_components=512, gamma=0.01, "
            "StandardScaler fit on FIT-train only -> LinearSVC C=1.0; predict_proba = "
            "sigmoid(decision_function)) on fold-0 full view: AUC 0.929997 / port_delta "
            "-2.175e-3 / corr 0.8671. Strongly decorrelated but far too weak -- NEGATIVE by "
            "direct measurement. Approximate-kernel family (via random features) closed."
        ),
    },
    65: {
        "fidelity_reached": "F2",
        "status": "COMPLETE_NEGATIVE",
        "next_action": "CLOSED by direct measurement (was previously auto-closed by an additive-logreg negative)",
        "phase20_finding": (
            "S7 repair (item 65): previously closed by the additive LogisticRegression "
            "negative (AUC 0.9258) -- a DIFFERENT model; factorized (bilinear) interactions "
            "were never tested. Now measured for real: FactLog, logit = w.x + sum_{r=1..16} "
            "(u_r.x)(v_r.x), trained end-to-end (AdamW lr 3e-3, wd 1e-4, bs 8192, inner-val "
            "ES) on fold-0 full view: AUC 0.959788 / port_delta -6.815e-4 / corr 0.9868. "
            "The rank-16 interaction logistic is decorrelated (corr < 0.995) but its "
            "port_delta is far below +5e-5 -- NEGATIVE by direct measurement. Factorized-"
            "interaction logistic closed under this protocol."
        ),
    },
}

def main() -> int:
    cat = json.loads(CAT.read_text(encoding="utf-8"))
    n = 0
    for row in cat["experiments"]:
        num = row.get("experiment_number")
        if num in UPDATES:
            old = {k: row.get(k) for k in UPDATES[num]}
            row.update(UPDATES[num])
            n += 1
            print(f"row {num} ({row['canonical_name']}): fidelity {old['fidelity_reached']}"
                  f" -> {UPDATES[num]['fidelity_reached']}, status {old['status']}"
                  f" -> {UPDATES[num]['status']}")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    print(f"updated {n} rows in {CAT.relative_to(ROOT).as_posix()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
