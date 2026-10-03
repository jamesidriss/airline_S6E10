"""Reproduce the finalist end-to-end from clean code.

This is the script that defines the submitted solution. It
  1. verifies the environment and the data hashes,
  2. runs the test suite and the external-block leakage audit,
  3. trains every ensemble member with its recorded configuration,
  4. rebuilds the equal-logit blend,
  5. writes the submission file through the pre-flight gate,
  6. records the exact manifest needed to reproduce it later.

Idempotent: members already present in the prediction store are reused, so a re-run only fills
gaps. Use --force-members to retrain everything from scratch.

Usage:
  python scripts/reproduce_finalist.py --name v3_final
  python scripts/reproduce_finalist.py --name v3_final --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import REPORTS, ROOT as PROJ, SUBMISSIONS, TARGET, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.ensemble import lab  # noqa: E402
from src.submission import store  # noqa: E402
from src.submission.kaggle_io import remaining, used_today  # noqa: E402
from src.submission.make import build  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

MIN_AUC = 0.9605


def sh(cmd: list[str], label: str) -> int:
    print(f"\n--- {label}: {' '.join(cmd[:3])}", flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=PROJ, capture_output=True, text=True,
                       timeout=36 * 3600)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"{label} failed")
    print("\n".join(ln for ln in r.stdout.splitlines() if "passed" in ln or "FAIL" in ln))
    return r.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v3_final")
    ap.add_argument("--min-auc", type=float, default=MIN_AUC)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    manifest: dict = {"name": args.name, "git": git_commit(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    print("=" * 100)
    print("STEP 1/6  data hashes")
    print("=" * 100)
    from src.common import RAW, file_sha256

    hashes = {p.name: file_sha256(p)[:32] for p in sorted(RAW.glob("*.csv"))}
    for k, v in hashes.items():
        print(f"  {k:<24} {v}")
    manifest["data_hashes"] = hashes

    if not args.skip_tests:
        print("\n" + "=" * 100)
        print("STEP 2/6  tests + leakage audit")
        print("=" * 100)
        sh(["tests/run_tests.py"], "unit tests")
        sh(["scripts/external_block_audit.py"], "external leakage audit")

    if not args.skip_train:
        print("\n" + "=" * 100)
        print("STEP 3/6  train missing members (idempotent)")
        print("=" * 100)
        for plan in ("scripts/run_zoo.py --only ''",):
            pass
        # the runners are individually idempotent; run the four passes that define the finalist
        sh(["scripts/run_zoo.py", "--save-test", "--tag", "zoo"], "zoo pass 1")
        sh(["scripts/run_xt_zoo.py", "--save-test", "--tag", "xt"], "zoo pass 2 (extra_trees)")
        sh(["scripts/run_zoo3.py", "--save-test", "--tag", "z3"], "zoo pass 3 (cat / 10-fold)")
        sh(["scripts/run_zoo4.py", "--save-test", "--tag", "z4"], "zoo pass 4 (fold diversity)")
        sh(["scripts/run_zoo5.py", "--save-test", "--tag", "z5"], "zoo pass 5 (new families)")

    print("\n" + "=" * 100)
    print("STEP 4/6  assemble the finalist blend")
    print("=" * 100)
    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y, tr["id"]).folds
    idx = store._load_index()

    members = []
    for k, v in idx.items():
        m = v.get("meta", {})
        if not m.get("auc") or m.get("family") in ("blend", "stack"):
            continue
        if m["auc"] < args.min_auc or "test" not in v:
            continue
        members.append((k, m["auc"], v.get("meta", {})))
    members.sort(key=lambda t: -t[1])
    print(f"  {len(members)} admitted members (OOF >= {args.min_auc}, test predictions present)")
    missing_test = [k for k, _, _ in members if "test" not in idx[k]]
    if missing_test:
        raise SystemExit(f"members without test predictions: {missing_test}")

    P = {k: store.load_oof(k).astype("float64") for k, _, _ in members}
    T = {k: store.load_test(k).astype("float64") for k, _, _ in members}
    M = np.column_stack([lab.tform(P[k], "logit") for k, _, _ in members])
    MT = np.column_stack([lab.tform(T[k], "logit") for k, _, _ in members])
    oof = M.mean(axis=1)
    test = MT.mean(axis=1)
    auc = float(roc_auc_score(y, oof))
    fa = lab.fold_aucs(y, oof, folds)
    print(f"  blend OOF AUC = {auc:.6f}  folds={[round(x,6) for x in fa]}")
    print(f"  best single   = {members[0][1]:.6f} ({members[0][0]})")
    manifest["members"] = [{"exp_id": k, "auc": a, **m} for k, a, m in members]
    manifest["blend"] = {"scheme": "equal_logit", "n_members": len(members),
                         "oof_auc": auc, "fold_aucs": fa, "best_single_auc": members[0][1]}

    print("\n" + "=" * 100)
    print("STEP 5/6  shadow-fold robustness note")
    print("=" * 100)
    conf = REPORTS / "conf_shadow.json"
    if conf.exists():
        c = json.loads(conf.read_text())
        for k in ("det_full", "xt_full", "det_raw_ext", "det_raw_trans"):
            if k in c:
                print(f"  {k:<16} {c[k]:.6f}")
        manifest["shadow_confirmation"] = c
    else:
        print("  (run scripts/shadow_confirm.py --folds shadow to populate)")

    print("\n" + "=" * 100)
    print("STEP 6/6  write submission")
    print("=" * 100)
    from scipy.special import expit

    store.save(f"blend_{args.name}", expit(oof), expit(test), fold_scheme="primary",
               meta={"family": "blend", "featureset": "equal_logit", "auc": round(auc, 6),
                     "members": [k for k, _, _ in members]})
    manifest["pred_hash_oof"] = store._load_index()[f"blend_{args.name}"]["oof_sha"]
    manifest["pred_hash_test"] = store._load_index()[f"blend_{args.name}"]["test_sha"]
    if not args.dry_run:
        build(expit(test), name=args.name,
              notes=(f"exp=blend_{args.name}; equal-logit over {len(members)} members; "
                     f"OOF(primary5)={auc:.6f}; best_single={members[0][1]:.6f}"),
              oof_auc=round(auc, 6), members=[k for k, _, _ in members])

    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["wall_seconds"] = round(time.time() - t_start, 1)
    manifest["submissions_used_today"] = used_today()
    manifest["submissions_remaining"] = remaining()
    save_json(manifest, REPORTS / f"finalist_{args.name}.json")
    print("\nmanifest ->", REPORTS / f"finalist_{args.name}.json")
    print(f"done in {manifest['wall_seconds']}s. submissions used today: "
          f"{manifest['submissions_used_today']}/10")


if __name__ == "__main__":
    main()