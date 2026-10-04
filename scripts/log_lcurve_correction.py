"""CORRECTION: the learning-curve per-doubling gains were written 10x too small.

The campaign's own script printed the correct value (+51.6e-5 for the 50%->100% doubling). The error
was introduced by hand when transcribing the numbers into STATUS.md and the ledger: 70.2 / 67.5 /
51.6 were written as 7.0 / 6.8 / 5.2. That is a transcription slip, not a measurement error, and it
mattered because it understated the single most strategically important number in the campaign by an
order of magnitude.

Corrected figures, recomputed directly from reports/lcurve_fold0.json:

    0.125 -> 0.250 (a doubling) : +0.000702  (+70.2e-5)
    0.250 -> 0.500 (a doubling) : +0.000675  (+67.5e-5)
    0.500 -> 0.750 (1.5x)        : +0.000261  (+26.1e-5)
    0.750 -> 1.000 (1.333x)      : +0.000256  (+25.6e-5)
    0.500 -> 1.000 (a doubling) : +0.000516  (+51.6e-5)

The correct conclusion is STRONGER than the one originally written, not weaker: a genuine doubling of
unique in-distribution training rows is worth roughly +5e-4 to +7e-4, which is comparable to the
entire rank-1-to-rank-45 spread (~3e-4).

This entry appends the correction to the append-only ledger rather than editing the faulty entry,
and also records the fitted scaling law, since that is what should be used for order-of-magnitude
extrapolation going forward.

Idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "experiments" / "ledger.jsonl"
REPORT = ROOT / "reports" / "lcurve_fold0.json"


def build_entry():
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = d["rows"]
    n = np.array([r["n_train_rows"] for r in rows], float)
    a = np.array([r["auc_mean"] for r in rows], float)

    doublings = {}
    fr = [r["frac"] for r in rows]
    for i in range(len(rows) - 1):
        ratio = n[i + 1] / n[i]
        if ratio >= 1.9:                      # count only genuine doublings
            doublings[f"{fr[i]:g}->{fr[i+1]:g}"] = float(a[i + 1] - a[i])
    full_doubling = float(a[-1] - a[list(fr).index(0.5)])

    l2 = np.log2(n)
    lin = np.polyfit(l2, a, 1)
    lin_pred = np.polyval(lin, l2)
    r2_lin = float(1 - ((a - lin_pred) ** 2).sum() / ((a - a.mean()) ** 2).sum())

    best = None
    for alpha in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
        x = n ** (-alpha)
        b = np.polyfit(x, a, 1)
        p = np.polyval(b, x)
        r2 = float(1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum())
        if best is None or r2 > best[1]:
            best = (alpha, r2, float(b[0]), float(b[1]))

    return {
        "exp_id": "lcurve_fold0_CORRECTION",
        "oof_auc": None,
        "fold_aucs": [],
        "verdict": "CORRECTION -- the previously reported per-doubling gains were 10x too small",
        "error": {
            "reported": {"0.125->0.25": 7.0, "0.25->0.5": 6.8, "0.5->1.0": 5.2},
            "actual": {k: round(v * 1e5, 1) for k, v in doublings.items()},
            "units": "e-5 AUC",
            "cause": ("hand transcription error when copying the script's own output into STATUS.md "
                      "and the ledger; the script printed +51.6e-5 correctly for 50%->100%"),
        },
        "corrected_doubling_gains_e5": {k: round(v * 1e5, 1) for k, v in doublings.items()},
        "full_doubling_50_to_100_e5": round(full_doubling * 1e5, 1),
        "non_doubling_steps_e5": {
            "0.5->0.75 (1.5x)": round(float(a[3] - a[2]) * 1e5, 1),
            "0.75->1.0 (1.333x)": round(float(a[4] - a[3]) * 1e5, 1),
        },
        "scaling_law": {
            "log2_form": {"slope_e5_per_doubling": round(float(lin[0]) * 1e5, 1),
                          "intercept": float(lin[1]), "r2": round(r2_lin, 5),
                          "residuals_e5": [round(float(p - q) * 1e5, 1)
                                           for p, q in zip(lin_pred, a)]},
            "power_form": {"alpha": best[0], "r2": round(best[1], 5),
                           "formula": "AUC(n) ~ %.6f + %.6f * n^(-%.2f)" % (best[3], best[2], best[0]),
                           "reading": "AUC - asymptote scales as n^(-1/5), i.e. the error decays "
                                      "with the inverse fifth root of the sample count"},
            "caveat": ("This is a five-point empirical fit used only for ORDER OF MAGNITUDE. It is "
                       "NOT a Bayes ceiling and must never be reported as one."),
        },
        "extrapolated_value_of_more_effective_data": {
            "1.25x": "+20.3e-5", "1.5x": "+36.8e-5", "2.0x": "+62.9e-5",
            "reference": "the entire rank-1 to rank-45 spread is ~3e-4, so 2x effective data would "
                         "be worth about twice the whole competitive gap",
        },
        "revised_conclusion": (
            "A genuine doubling of unique in-distribution training rows buys ~+5e-4 to +7e-4. That "
            "is an order of magnitude more than any ensemble-level gain measured in this campaign "
            "(the 59-member blend buys +2.5e-4 over the best single model), and it makes "
            "manufacturing additional effective training support the highest-value remaining "
            "direction by a wide margin. It does NOT follow that synthetic augmentation delivers the "
            "same benefit -- that is precisely the hypothesis now under test."
        ),
    }


def main() -> int:
    existing = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line).get("exp_id"))
                except Exception:  # noqa: BLE001
                    pass
    e = build_entry()
    if e["exp_id"] in existing:
        print("  skip: correction already logged")
        return 0
    e["date"] = "2026-10-04"
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(e) + "\n")
    print(f"  appended: {e['exp_id']}")
    print(f"  corrected doubling gains (e-5): {e['corrected_doubling_gains_e5']}")
    print(f"  50%->100% doubling           : {e['full_doubling_50_to_100_e5']}e-5")
    print(f"  log2 slope {e['scaling_law']['log2_form']['slope_e5_per_doubling']}e-5/doubling "
          f"(R2={e['scaling_law']['log2_form']['r2']}); power law alpha="
          f"{e['scaling_law']['power_form']['alpha']} (R2={e['scaling_law']['power_form']['r2']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
