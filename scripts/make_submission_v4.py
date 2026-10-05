"""Build the v4 candidate: v3_final's blend geometry with full-data test predictions.

What this is, stated precisely
-------------------------------
v3_final is an equal-logit blend of 59 members. Its stored TEST prediction for each member is the
average over K fold models, each fitted on that fold's OUTER-FIT subset -- 559,708 rows (80.00% of
the labels) at K=5, 629,672 (90.00%) at K=10. No path in this repository trains a test model on 100%
of the labels, so 20%/10% of the real labels are never shown to any member that votes on the test
set, although we already hold them and test-time inference needs no held-out fold.

v4 keeps the blend's identity completely intact -- the same 59 members, the same equal weights, the
same OOF predictions, the same feature pipeline and hyperparameters -- and substitutes, where a
full-data fit exists, the member's test prediction with a fit on 100% of the labelled rows using an
identical configuration and seed.

What the reported number is, and is not
----------------------------------------
The OOF AUC printed here is **v3_final's OOF, unchanged to the last decimal**, because the OOF
predictions are untouched by construction. That number is reported so the file is internally
consistent and comparable, and it is NOT evidence for v4. v4's evidence is:

  * a cross-validated training-policy gain at the member level: `scripts/run_fullfit.py` measured
    +2.2e-5 mean / +4.0e-5 best fold for the analogous substitution of 72% -> 80%, 2/2 folds;
  * an a-priori validated law for the training fraction: `scripts/validate_curve_on_folds.py`
    predicted the ALREADY-MEASURED 5 -> 10 fold gain to 7.4% from training fraction alone;
  * the law OVER-states gains at the top of the range, so the 80% -> 100% step is treated as a
    low-single-digit-e-5 effect rather than the +20.2e-5 the raw extrapolation suggests.

Reporting this as an OOF gain would be exactly the fabrication the full-data refit was designed to
avoid, so it is labelled `training_policy_gain`, never `oof`.

Pre-flight
----------
The existing submission gate is reused: column order, row count, id equality, finiteness, range.

Usage: python scripts/make_submission_v4.py --name v4_fulldata --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.ensemble import lab  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

MANIFEST = Path("artifacts") / "fulldata" / "members_fulldata.json"
FINALIST = REPORTS / "finalist_v3_final.json"


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v4_fulldata")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y, tr["id"]).folds

    final = json.loads(FINALIST.read_text(encoding="utf-8"))
    members = final["members"]
    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} not found -- run scripts/refit_members_fulldata.py first")
    fd = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fd_by_id = {m["exp_id"]: m for m in fd["members"]}

    from scipy.special import expit

    oof_cols, test_cols, used = [], [], []
    for m in members:
        eid = m["exp_id"]
        oof_cols.append(lab.tform(store.load_oof(eid).astype("float64"), "logit"))
        if eid in fd_by_id:
            p = np.load(fd_by_id[eid]["path"]).astype("float64")
            used.append(eid)
        else:
            p = store.load_test(eid).astype("float64")
        test_cols.append(lab.tform(p, "logit"))

    O = np.column_stack(oof_cols)
    T = np.column_stack(test_cols)
    w = np.full(len(members), 1.0 / len(members))       # v3_final is equal_all
    oof_blend = expit(O @ w)
    test_blend = expit(T @ w)

    auc = lab.auc(y, oof_blend)
    fa = lab.fold_aucs(y, oof_blend, folds)
    v3_test = store.load_test("blend_v3_final").astype("float64")

    print(f"  members                       : {len(members)}")
    print(f"  refit on 100% of labels       : {len(used)}")
    print(f"  OOF AUC (v3's, UNCHANGED)     : {auc:.6f}   folds={[round(x,6) for x in fa]}")
    print(f"  v3_final OOF for comparison   : {final['blend']['oof_auc']:.6f}")
    print(f"  v4 vs v3 test: spearman       : {spearman(test_blend, v3_test):.6f}")
    print(f"  v4 vs v3 test: logit corr     : {corr(logit(test_blend), logit(v3_test)):.6f}")
    print(f"  test range                    : "
          f"[{test_blend.min():.6f}, {test_blend.max():.6f}]  finite={np.isfinite(test_blend).all()}")

    # ---- pre-flight gate ----
    problems = []
    if len(test_blend) != len(te):
        problems.append(f"row count {len(test_blend)} != {len(te)}")
    if not np.isfinite(test_blend).all():
        problems.append("non-finite predictions")
    if test_blend.min() < 0 or test_blend.max() > 1:
        problems.append(f"out of range [{test_blend.min()}, {test_blend.max()}]")
    if problems:
        raise SystemExit("PRE-FLIGHT FAILED: " + "; ".join(problems))
    print("  pre-flight                    : PASS")

    digest = hashlib.sha256(test_blend.astype("float64").tobytes()).hexdigest()
    meta = {
        "family": "blend", "featureset": "equal_logit", "auc": round(auc, 6),
        "members": [m["exp_id"] for m in members],
        "weights": "equal (1/59), identical to blend_v3_final",
        "n_members_refit_on_100pct": len(used),
        "refit_member_ids": used,
        "oof_is_v3s_unchanged": True,
        "evidence": "cross-validated training-policy gain, NOT an OOF gain",
        "spearman_vs_v3_final": spearman(test_blend, v3_test),
        "test_sha256": digest,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()[:12],
    }
    store.save(f"blend_{args.name}", oof_blend, test_blend, fold_scheme="primary", meta=meta)

    out = {"name": args.name, "n_members": len(members), "n_refit_on_100pct": len(used),
           "refit_member_ids": used,
           "oof_auc_of_v3_members_unchanged": auc,
           "oof_equals_v3_final": bool(abs(auc - final["blend"]["oof_auc"]) < 1e-12),
           "fold_aucs": fa,
           "spearman_vs_v3_final": spearman(test_blend, v3_test),
           "logit_corr_vs_v3_final": corr(logit(test_blend), logit(v3_test)),
           "test_range": [float(test_blend.min()), float(test_blend.max())],
           "test_sha256": digest,
           "evidence": ("cross-validated training-policy gain: run_fullfit.py measured +2.2e-5 mean "
                        "/ +4.0e-5 best fold for the analogous 72pct->80pct substitution 2/2 folds; "
                        "validate_curve_on_folds.py showed the training-fraction law predicts the "
                        "already-measured 5->10 fold gain to 7.4pct. The 80pct->100pct step is an "
                        "EXTRAPOLATION and the law over-states gains at the top of the range, so it "
                        "is claimed as low-single-digit-e-5, not as the +20.2e-5 raw extrapolation."),
           "not_an_oof_gain": ("the OOF printed here belongs to v3_final's members and is unchanged "
                               "by construction; it is shown for internal consistency only and is "
                               "NOT evidence for this candidate"),
           "git_commit": meta["git_commit"]}
    save_json(out, REPORTS / f"submission_{args.name}.json")

    if not args.dry_run:
        # Reuse the established builder rather than hand-writing the CSV, so the pre-flight gate
        # (column order, row count against BOTH sample and test, id equality, finiteness, range)
        # and the manifest row are identical to every other submission in this campaign. A bespoke
        # writer here would be one more thing that could silently differ from the validated path.
        from src.submission.make import build
        build(test_blend, name=args.name,
              notes=(f"exp=blend_{args.name}; kind=equal_logit; members=v3_final's 59 with "
                     f"{len(used)} refit on 100pct of labels; OOF shown is v3_final's and is "
                     f"UNCHANGED -- this candidate's evidence is a cross-validated training-policy "
                     f"gain, not an OOF gain"),
              oof_auc=round(auc, 6), members=[m["exp_id"] for m in members])

    print("\nwrote", REPORTS / f"submission_{args.name}.json")
    print("  NOTE: the OOF above is v3_final's, unchanged. v4's case is a cross-validated "
          "training-policy\n        gain and must never be quoted as an OOF improvement.")


if __name__ == "__main__":
    main()
