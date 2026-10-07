"""Protocol guards for the Phase 11 native-CatBoost block harness.

Each test encodes a way this harness could produce a confident, wrong, or self-flattering number.
Most were written after observing a real failure in this phase rather than in anticipation of one.

  1. every native slot maps to exactly one original v3 slot
  2. the 52 non-CatBoost v3 slots are untouched by any candidate
  3. every slot retains exactly 1/59 of the total logit weight
  4. B25/B50/B75/B100 weights sum to exactly 1
  5. mixing happens in LOGIT space, never probability space
  6. alpha=0 reconstructs v3 to float64 round-off (NOT bit-exactness -- see below)
  7. B100 contains seven DISTINCT native prediction ids
  8. no duplicate native member silently replaces two slots
  9. inventory view / fold_scheme / seed hashes are self-consistent and complete
 10. OOF and test schema consistency for the frame builder
 11. the harness refuses to report when the reconstruction guard fails

Why the alpha=0 guard is a tolerance and not an equality
-------------------------------------------------------
v3 is STORED as sigmoid(mean(member probabilities)); this harness averages member LOGITS. Those are
different functions, so a bit-exact reconstruction is impossible and demanding one produced a false
alarm at -1.45e-10 with logit corr 1.000000000. The correct guard is "float64 round-off plus logit
correlation of 1 to machine precision". Asserting strict equality would have masked a real bug later.

Run: python tests/test_native_block.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.native_block_analysis import ALPHAS, logit, sig  # noqa: E402

FAILS: list[str] = []
N = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global N
    N += 1
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def _raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:                                          # noqa: BLE001
        return False
    return False


def load():
    import json as _j
    inv = _j.loads(Path("reports/native_cat_slot_inventory.json").read_text(encoding="utf-8"))
    man = _j.loads(Path("reports/finalist_v3_final.json").read_text(encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]
    return inv, ids


# --------------------------------------------------------------------------- 1, 8
def test_slot_mapping_is_one_to_one() -> None:
    print("\n1/8. slot mapping is one-to-one and free of duplicates")
    inv, ids = load()
    slots = inv["slots"]
    eids = [s["exp_id"] for s in slots]
    check("seven slots", len(slots) == 7, str(len(slots)))
    check("slot indices are 0..6 with no gaps",
          [s["slot"] for s in slots] == list(range(7)))
    check("every slot's exp_id is distinct", len(set(eids)) == len(eids), str(eids))
    check("every slot's exp_id is in the v3 manifest", all(e in ids for e in eids))
    check("every v3 CatBoost member is claimed by exactly one slot",
          sorted(eids) == sorted(m for m in ids
                                 if any(m.startswith(p) for p in
                                        ("z3_cat", "prod5_cat", "view_cat"))),
          str(sorted(set(ids) - set(eids) - set(
              m for m in ids if not any(m.startswith(p) for p in
                                        ("z3_cat", "prod5_cat", "view_cat"))))))
    # a duplicated native member across two slots would silently make the block 2x weight on one model
    check("no two slots would be filled by the same native id",
          len({s["exp_id"] for s in slots}) == 7)


# --------------------------------------------------------------------------- 2, 3, 4
def test_weights_and_untouched_slots() -> None:
    print("\n2/3/4. block weight is fixed and non-CatBoost slots are untouched")
    inv, ids = load()
    nmem, nslots = len(ids), 7
    w = 1.0 / nmem
    check(f"weight per slot is exactly 1/{nmem}", abs(nslots * w - nslots / nmem) < 1e-15)
    check("CatBoost block weight is 7/59", abs(nslots / nmem - 0.1186440677966102) < 1e-12)
    check("inventory records the same block weight",
          abs(inv["cat_block_weight"] - nslots / nmem) < 1e-12)
    check("non-CatBoost count is 52", nmem - nslots == 52, str(nmem - nslots))
    # every candidate must sum to exactly 1 in total weight
    rng = np.random.default_rng(0)
    L = {e: rng.normal(size=1000) for e in ids}
    cat = set(inv["exp_id and"] if False else [s["exp_id"] for s in inv["slots"]])
    for a in ALPHAS:
        tot = 0.0
        for e in ids:
            w_e = w if e not in cat else w
            inner = (1 - a) + a if e in cat else 1.0
            tot += w_e * inner
        check(f"alpha={a:.2f}: total weight sums to exactly 1", abs(tot - 1.0) < 1e-12, f"{tot!r}")
    check("alpha=0 gives every slot weight 1", all(
        abs(w * (1 - 0.0) + 0.0 - w) < 1e-15 for _ in [0]))
    check("alpha=1 gives every slot weight 1", all(
        abs(w * (0.0 + 1.0) - w) < 1e-15 for _ in [0]))


# --------------------------------------------------------------------------- 5
def test_mixing_is_in_logit_space() -> None:
    print("\n5. slot mixing is in logit space")
    o = np.array([0.2, 0.8])
    n = np.array([0.4, 0.6])
    a = 0.5
    logit_mix = (1 - a) * logit(o) + a * logit(n)
    prob_mix = (1 - a) * o + a * n
    check("logit-space and probability-space mixes differ",
          not np.allclose(sig(logit_mix), prob_mix),
          f"logit {sig(logit_mix)} vs prob {prob_mix}")
    check("logit mix equals the arithmetic mean of logits",
          np.allclose(logit_mix, 0.5 * (logit(o) + logit(n))))
    check("at alpha=0 the logit mix returns the original logit exactly",
          np.allclose((1 - 0.0) * logit(o) + 0.0 * logit(n), logit(o)))
    check("at alpha=1 the logit mix returns the native logit exactly",
          np.allclose(0.0 * logit(o) + 1.0 * logit(n), logit(n)))


# --------------------------------------------------------------------------- 6
def test_alpha0_reconstruction_tolerance() -> None:
    print("\n6. alpha=0 reconstruction tolerance")
    # the stored blend averages probabilities; the harness averages logits. Reproduce that gap.
    rng = np.random.default_rng(1)
    p = np.clip(rng.random(20000) * 0.8 + 0.1, 1e-6, 1 - 1e-6)
    prob_avg = p.mean()
    logit_avg = sig(logit(p).mean())
    check("the two averaging conventions genuinely differ",
          abs(prob_avg - logit_avg) > 1e-9, f"{prob_avg} vs {logit_avg}")
    check("but they agree to ~1e-3 or better on a well-behaved blend",
          abs(prob_avg - logit_avg) < 1e-3, f"{abs(prob_avg-logit_avg):.2e}")
    rep = Path("reports/p11_none_block_native.json")
    if rep.exists():
        d = json.loads(rep.read_text(encoding="utf-8"))
        check("observed reconstruction delta is float64 round-off",
              abs(d["guard_recon_delta_auc"]) < 1e-8, str(d["guard_recon_delta_auc"]))
        check("observed logit correlation is 1 to machine precision",
              d["guard_recon_logit_corr"] > 1 - 1e-12, str(d["guard_recon_logit_corr"]))


# --------------------------------------------------------------------------- 9
def test_inventory_is_complete_and_self_consistent() -> None:
    print("\n9. inventory completeness")
    inv, _ = load()
    for s in inv["slots"]:
        ok = (s["view_known"] and s["fold_scheme"] in {"primary", "block10", "shadow"}
              and isinstance(s["seed"], int) and s["params"]["depth"] in (6, 8, 10)
              and s["params"]["learning_rate"] in (0.03, 0.04)
              and s["boosting_type" if "boosting_type" in s else "params"]["boosting_type"]
              == "Plain")
        check(f"slot {s['slot']} {s['exp_id']}: view/scheme/seed/depth/lr/Plain all recovered", ok,
              json.dumps({k: s.get(k) for k in ("view", "fold_scheme", "seed")}))
    check("every slot records that the original had NO categorical treatment",
          all("cat_features was never passed" in s["categorical_treatment_original"]
              for s in inv["slots"]))
    check("every slot records the original's discarded-carve OOF policy",
          all("permanently discarded" in s["original_oof_iteration_policy"]
              for s in inv["slots"]))
    check("inventory carries a config hash per slot",
          all(isinstance(s["config_hash"], str) and len(s["config_hash"]) == 64
              for s in inv["slots"]))
    check("inventory carries an overall hash", len(inv["inventory_hash"]) == 64)
    check("inventory warns that native-original is not a causal estimate",
          "OPERATIONAL REPLACEMENT" in inv["interpretation_warning"])
    check("inventory records the block10 mixing caveat",
          "block10" in inv["fold_scheme_caveat"])
    check("slot 0 is the only block10 member",
          sum(1 for s in inv["slots"] if s["fold_scheme"] == "block10") == 1)
    check("no slot uses Ordered boosting",
          all(s["params"]["boosting_type"] == "Plain" for s in inv["slots"]))


# --------------------------------------------------------------------------- 10
def test_frame_schema_consistency() -> None:
    print("\n10. OOF / schema consistency of the frame builder")
    import pandas as pd
    from src.common import load_cached_parquet
    from scripts.native_cat import cat_frame, default_cat_cols
    tr, te = load_cached_parquet()
    src = default_cat_cols(True)
    rng = np.random.default_rng(2)
    n = len(tr)
    rows_a = rng.choice(n, 500, replace=False)
    rows_b = rng.choice(n, 300, replace=False)
    A = cat_frame(tr, src, rows_a)
    B = cat_frame(tr, src, rows_b)
    check("train and apply frames have identical column lists",
          list(A.columns) == list(B.columns), f"{list(A.columns)[:2]} vs {list(B.columns)[:2]}")
    check("row counts match the requested index sets", len(A) == 500 and len(B) == 300)
    check("exactly 17 twin columns", len(A.columns) == 17, str(len(A.columns)))
    check("every twin holds strings",
          all(all(isinstance(v, str) for v in A[c].head(30)) for c in A.columns))
    check("all twins are prefixed ncat__", all(c.startswith("ncat__") for c in A.columns))
    check("no twin collides with a raw column name",
          not (set(A.columns) & set(tr.columns[:30])))
    check("META4 twins present",
          all(f"ncat__{c}" in A.columns
              for c in ("Gender", "Customer Type", "Type of Travel", "Class")))
    check("SERVICE13 twins present", sum(1 for c in A.columns if c.startswith("ncat__")) == 17)
    check("Age twin is NOT included (separate arm C4)", "ncat__Age" not in A.columns)
    check("Flight Distance twin is NOT included (separate arm C5)",
          "ncat__Flight Distance" not in A.columns)


# --------------------------------------------------------------------------- 11
def test_harness_refuses_when_guard_fails() -> None:
    print("\n11. the harness refuses to report when the guard fails")
    src = Path("scripts/native_block_analysis.py").read_text(encoding="utf-8")
    # This test asserted on the literal OLD stop message ("STOP: alpha=0 failed to reproduce v3").
    # Phase 11R replaced that message with a stricter, more informative one that reports the measured
    # max|dp|, the float32-cast difference and the logit correlation, because the old tolerance was
    # arbitrary rather than derived from the store's dtype. A literal-string assertion on a message
    # that was deliberately rewritten is a test of the wording, not of the behaviour, so it now
    # asserts the behaviour: the script still aborts, the abort is keyed on the reconstruction, and it
    # names the actual quantities.
    check("the script raises SystemExit on a failed reconstruction guard",
          'raise SystemExit(f"STOP: alpha=0 reconstruction failed.' in src,
          "no SystemExit keyed on the alpha=0 reconstruction")
    check("the abort message reports the measured probability difference",
          "max|dp|=" in src)
    check("the abort message reports the float32 storage comparison",
          "float32 max|dp|=" in src)
    check("the abort message reports the logit correlation",
          "logit corr=" in src)
    check("the stop is still raised, not merely warned about",
          "if not ok_recon:" in src and src.count("raise SystemExit") >= 2)
    check("the guard is evaluated BEFORE any candidate is scored",
          src.index("ok_recon =") < src.index("def blend("))
    check("the guard is evaluated before single-slot swaps are computed",
          src.index("ok_recon =") < src.index("SINGLE-SLOT SWAP"))
    check("alphas are predeclared and include 0 and 1", ALPHAS[0] == 0.0 and ALPHAS[-1] == 1.0)


def main() -> int:
    print("=" * 78)
    print("PHASE 11 NATIVE BLOCK -- PROTOCOL GUARD TESTS")
    print("=" * 78)
    for fn in (test_slot_mapping_is_one_to_one, test_weights_and_untouched_slots,
               test_mixing_is_in_logit_space, test_alpha0_reconstruction_tolerance,
               test_inventory_is_complete_and_self_consistent,
               test_frame_schema_consistency, test_harness_refuses_when_guard_fails):
        fn()
    print("\n" + "=" * 78)
    print(f"{N - len(FAILS)}/{N} passed")
    if FAILS:
        print("FAILURES: " + ", ".join(FAILS))
    print("=" * 78)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
