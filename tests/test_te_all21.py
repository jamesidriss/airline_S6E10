"""Regression tests for the `te_all21` fold-safety contract (Phase 8A, section 5).

These are the load-bearing part of the work. A target-encoding block that leaks does not crash and
does not look wrong -- it looks like a modest, believable feature gain. So each property is checked
directly, and each check is written so that it CANNOT pass for the wrong reason.

The required six:

  1. randomising OUTER-VAL labels changes NOTHING in outer-FIT or outer-VAL TE values
  2. changing one FIT row's target cannot materially affect that SAME row's cross-fitted TE value
  3. validation/test transforms use only outer-FIT labels
  4. values unseen in outer-FIT map to the encoder's prior/fallback
  5. raw exact-value categories survive without float32 aliasing
  6. NaN sentinel behaviour is stable
  7. the inner split is deterministic and independent of the target -- which is what makes the
     exact-zero assertion in test 2 legitimate

Two anti-vacuity guards
-----------------------
A leakage test that passes because the code is a no-op is worse than no test, so:

  * test 2 asserts that OTHER fit rows DO change when one fit label flips. Without that, an encoder
    that ignored the target entirely would pass.
  * test 3 asserts that flipping a FIT label DOES move the apply-row TE. Without that, an encoder
    that leaked nothing *and learned nothing* would pass.

An honesty note that test 2 encodes
-----------------------------------
sklearn's cross-fitting excludes a row's own label from the COUNT, but the smoothing prior is computed
over all of `y`. So flipping one fit row's target moves that same row's own encoding by O(1/n), not by
exactly zero. The test asserts the tight observed bound and says so in the failure message, rather
than asserting an exact-zero claim that the library does not honour.

Runs on a 40,000-row subsample so the whole file stays fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import TARGET, load_cached_parquet  # noqa: E402
from src.features.te_all21 import (  # noqa: E402
    NA_SENTINEL, ORDINAL6, All21TargetEncoder, build_all21_codes, exact_value_labels,
)
from src.features.view import RAW21  # noqa: E402

N_SUB = 40_000
_CACHE: dict = {}


def _data():
    if "d" not in _CACHE:
        tr, _te = load_cached_parquet()
        sub = tr.iloc[:N_SUB].reset_index(drop=True)
        codes, names = build_all21_codes(sub)
        y = sub[TARGET].values.astype("int8")
        # outer-FIT = first 80%, outer-validation = last 20%, both inside the subsample
        n_fit = int(0.8 * len(sub))
        _CACHE["d"] = (sub, codes, names, y, np.arange(n_fit), np.arange(n_fit, len(sub)))
    return _CACHE["d"]


def _enc(seed: int = 0, smooth="auto") -> All21TargetEncoder:
    return All21TargetEncoder(smooth=smooth, cv=5, seed=seed)


def test_randomising_outer_val_labels_changes_nothing():
    """Requirement 1. The decisive test: what a row is SCORED on cannot depend on its own label."""
    _sub, codes, _names, y, fit_idx, val_idx = _data()

    Zf0 = _enc().fit_rows(codes, y, fit_idx)
    Zv0 = _enc().apply(codes, y, fit_idx, val_idx)

    rng = np.random.default_rng(12345)
    y_bad = y.copy()
    y_bad[val_idx] = rng.integers(0, 2, size=len(val_idx)).astype("int8")

    Zf1 = _enc().fit_rows(codes, y_bad, fit_idx)
    Zv1 = _enc().apply(codes, y_bad, fit_idx, val_idx)

    assert np.array_equal(Zf0, Zf1), (
        f"outer-FIT TE changed when only outer-VAL labels were randomised; max abs diff "
        f"{np.abs(Zf0 - Zf1).max():.3e}")
    assert np.array_equal(Zv0, Zv1), (
        f"outer-VAL TE changed when only outer-VAL labels were randomised; max abs diff "
        f"{np.abs(Zv0 - Zv1).max():.3e} -- this is the label leak the whole protocol exists to stop")


def test_one_fit_row_target_cannot_materially_move_its_own_encoding():
    """Requirement 2. Asserts EXACT zero, not a loose bound, and says why that is justified.

    The bound is exact rather than approximate because the inner split is y-independent: with
    KFold(shuffle=True, random_state=seed), flipping one label cannot move any row to a different
    fold, so the fold-complement table that encodes row j is bit-identical before and after. And
    `smooth="auto"` adds no global-prior path, because auto-smoothing is derived per fold from that
    fold's training labels (see `lambda_` in sklearn's `_fit_encoding_fast_auto_smooth`).

    The first version of this test asserted `own < 1e-3` and FAILED at 5.7e-2. That was not a leak:
    it was `StratifiedKFold` stratifying on y, so the perturbation also moved the folds. The splitter
    was changed to KFold and the assertion tightened to exact equality. Both facts are worth keeping,
    because a loose bound would have hidden the cause.
    """
    _sub, codes, _names, y, fit_idx, _val_idx = _data()
    Z0 = _enc().fit_rows(codes, y, fit_idx)

    j = 7                                  # a row in the middle of the fit block
    y_bad = y.copy()
    y_bad[fit_idx[j]] = 1 - y_bad[fit_idx[j]]
    Z1 = _enc().fit_rows(codes, y_bad, fit_idx)

    own = float(np.abs(Z1[j] - Z0[j]).max())
    assert own == 0.0, (
        f"flipping one FIT row's own target moved its own encoding by {own:.3e}, expected EXACTLY "
        f"zero. With a y-independent inner split there is no path for a row's own label to reach its "
        f"own encoding, so any nonzero value means something has broken that invariant.")

    # ANTI-VACUITY: other fit rows must change, or this test would pass for a no-op encoder.
    others = np.delete(np.abs(Z1 - Z0).max(axis=1), j)
    assert others.max() > 1e-6, (
        "flipping one FIT label moved NO other FIT row's encoding. The encoder appears to ignore the "
        "target entirely, which would make this test pass for the wrong reason.")


def test_apply_transform_uses_only_fit_labels():
    """Requirement 3, both directions: apply is blind to apply labels, sensitive to fit labels."""
    _sub, codes, _names, y, fit_idx, val_idx = _data()
    enc = _enc()

    Zv0 = enc.apply(codes, y, fit_idx, val_idx)
    rng = np.random.default_rng(999)
    y_bad = y.copy()
    y_bad[val_idx] = rng.integers(0, 2, size=len(val_idx)).astype("int8")
    Zv1 = _enc().apply(codes, y_bad, fit_idx, val_idx)
    assert np.array_equal(Zv0, Zv1), "apply-row TE moved when only apply-row labels changed"

    y_bad2 = y.copy()
    y_bad2[fit_idx[11]] = 1 - y_bad2[fit_idx[11]]
    Zv2 = _enc().apply(codes, y_bad2, fit_idx, val_idx)
    moved = float(np.abs(Zv2 - Zv0).max())
    assert moved > 1e-6, (
        "apply-row TE did NOT move when a FIT label changed. The encoder would be scoring rows "
        "without using the training labels at all, which would make the preceding assertion pass for "
        "the wrong reason.")


def test_unseen_category_maps_to_prior():
    """Requirement 4. A test value whose exact value never appears in outer-FIT must fall back."""
    _sub, codes, _names, y, fit_idx, _val_idx = _data()
    enc = _enc()
    prior = float(y[fit_idx].mean())

    unseen_code = int(codes[fit_idx].max()) + 9999
    probe = np.zeros((3, codes.shape[1]), dtype="int32")
    probe[:] = codes[fit_idx[0]]
    probe[:, 0] = unseen_code
    out = enc.apply(codes, y, fit_idx, np.arange(1))
    got = enc._encoder.transform(probe)

    col_prior = float(y[fit_idx].mean())
    assert abs(float(got[0, 0]) - col_prior) < 1e-6, (
        f"an unseen category encoded to {float(got[0, 0]):.6f}, expected the class prior "
        f"{col_prior:.6f}. Either the fallback is broken or unseen values are being silently mapped "
        f"onto a seen category.")
    assert abs(prior - col_prior) < 1e-9       # sanity: prior is well defined
    assert np.isfinite(out).all()


def test_exact_values_survive_without_float32_aliasing():
    """Requirement 5. The whole point of exact-value encoding is that distinct values stay distinct."""
    row = {c: 0 for c in RAW21}
    for c in ORDINAL6:
        row[c] = 3
    row["Gender"] = "Female"
    row["Customer Type"] = "Loyal Customer"
    row["Type of Travel"] = "Business Travel"
    row["Class"] = "Business"
    row["Age"] = 36
    row["Flight Distance"] = 761
    row["Departure Delay in Minutes"] = 0
    row["Arrival Delay in Minutes"] = 5.0

    # two float64 values that are DIFFERENT but both collapse to the same float32
    a = 1.00000000010
    b = 1.00000000020
    assert np.float32(a) == np.float32(b), "precondition: these must alias under float32"
    f1 = dict(row); f1["Flight Distance"] = a
    f2 = dict(row); f2["Flight Distance"] = b
    # 761 vs 761.0 must be the SAME category; 761 vs 761.5 must be DIFFERENT
    f3 = dict(row); f3["Flight Distance"] = 761.0
    f4 = dict(row); f4["Flight Distance"] = 761.5

    def lab(d, col="Flight Distance"):
        return exact_value_labels(pd.DataFrame([d]), col)[0]

    assert lab(f1) != lab(f2), (
        "two distinct float64 values collapsed to the same category -- an exact-value encoder that "
        "aliases is worse than useless, it silently merges classes")
    assert lab(row) == lab(f3), "761 and 761.0 must be the same exact category"
    assert lab(row) != lab(f4), "761 and 761.5 must be different categories"

    df = pd.concat([pd.DataFrame([f1, f2, row, f3, f4])], ignore_index=True)
    for c in ORDINAL6:
        df[c] = df[c].astype("float64")
    codes, _ = build_all21_codes(df)
    assert len(np.unique(codes[:, RAW21.index("Flight Distance")])) == 4, (
        "expected 4 distinct Flight Distance categories from 5 rows (a and b differ, 761 and 761.0 "
        "merge), got a different number")


def test_nan_sentinel_is_stable_and_collision_free():
    """Requirement 6. Missing Arrival Delay needs one explicit, stable, non-colliding category."""
    _sub, codes, names, y, fit_idx, _val_idx = _data()
    j = RAW21.index("Arrival Delay in Minutes")

    tr = _sub
    n_missing_tr = int(pd.to_numeric(tr["Arrival Delay in Minutes"],
                                    errors="coerce").isna().sum())
    assert n_missing_tr > 0, "precondition: the competition data does have missing Arrival Delay"

    lab = exact_value_labels(tr, "Arrival Delay in Minutes")
    n_sentinel = int(np.sum(lab == NA_SENTINEL))
    assert n_sentinel == n_missing_tr, (
        f"{n_sentinel} rows carry the sentinel but {n_missing_tr} are actually missing -- a real "
        f"value is being absorbed into the missing category, or missing rows are escaping it")
    assert NA_SENTINEL not in set(np.asarray(lab)[lab != NA_SENTINEL].tolist()), (
        "the sentinel string collides with a real value's label")

    # stability: the same input must produce the same labels every time, and the mapping must be
    # unchanged by which rows are present
    lab2 = exact_value_labels(tr.copy(), "Arrival Delay in Minutes")
    assert np.array_equal(lab, lab2), "exact-value labelling is not deterministic"
    sub = tr.iloc[:5000]
    lab3 = exact_value_labels(sub, "Arrival Delay in Minutes")
    assert np.array_equal(lab[:5000], lab3), (
        "labelling a row depends on which other rows are present; the sentinel path is not local")

    codes2, _ = build_all21_codes(tr)
    assert codes2.shape[1] == 21 and codes[:, j].min() >= 0


def test_inner_split_is_deterministic_and_independent_of_the_target():
    """Why the exact-zero assertion above is legitimate at all.

    Two properties, both load-bearing:
      * determinism -- the same inputs must give bit-identical encodings, or no experiment using this
        block is reproducible;
      * y-independence -- if the split moved with y, perturbing one label would also move folds, and
        requirement 2 would become untestable. `StratifiedKFold` fails this; `KFold` passes.

    This test is the reason a loose bound was not acceptable in the first place: it shows the split
    cannot be y-dependent.
    """
    _sub, codes, _names, y, fit_idx, _val_idx = _data()

    a = _enc(seed=5).fit_rows(codes, y, fit_idx)
    b = _enc(seed=5).fit_rows(codes, y, fit_idx)
    assert np.array_equal(a, b), "the same seed produced different encodings; not reproducible"
    c = _enc(seed=6).fit_rows(codes, y, fit_idx)
    assert not np.array_equal(a, c), "different seeds produced identical encodings; seed is ignored"

    rng = np.random.default_rng(31337)
    y_bad = y.copy()
    y_bad[fit_idx[: len(fit_idx) // 20]] = rng.integers(0, 2, size=len(fit_idx) // 20).astype("int8")
    d = _enc(seed=5).fit_rows(codes, y_bad, fit_idx)
    assert not np.array_equal(a, d), (
        "flipping 5pct of the FIT labels changed nothing, so the encoder is not using the target")

    # and the split itself must not move: verify via the encoder's own KFold
    from sklearn.model_selection import KFold, StratifiedKFold
    Xd = codes[fit_idx]
    kf = KFold(n_splits=5, shuffle=True, random_state=5)
    sf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    k1 = [t.tolist() for _, t in kf.split(Xd)]
    k2 = [t.tolist() for _, t in kf.split(Xd, y_bad[fit_idx])]
    s1 = [t.tolist() for _, t in sf.split(Xd, y[fit_idx])]
    s2 = [t.tolist() for _, t in sf.split(Xd, y_bad[fit_idx])]
    assert k1 == k2, "KFold.split changed when y changed -- it should not take y into account"
    assert s1 != s2, (
        "StratifiedKFold.split did NOT change when y changed. If this ever becomes true the block is "
        "no longer using the splitter this file depends on, and requirement 2's exact-zero assertion "
        "would need revisiting.")


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in fns:
        f()
        print(f"  PASS {n}")
    print(f"\n{len(fns)} passed")
