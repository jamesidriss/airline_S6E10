"""Regression test: a view must contain EXACTLY the blocks it declares, and no others.

The bug this guards against
---------------------------
While wiring Phase 8B, a duplicated `te_all21` body was left nested INSIDE the `if "te_cond"` branch
of `ViewBuilder.assemble`. Consequence: the view `full_tec`, which declares `te` and `te_cond` but
NOT `te_all21`, silently received 21 extra `te_all21` columns. The assembled matrix looked entirely
reasonable -- 378 columns, no NaN, every block's rows plausible -- and the mismatch only surfaced
because the column count did not match arithmetic:

    237 static + 48 (te) + 72 (te_cond) = 357, but the matrix had 378.

Had the arithmetic not been checked, Phase 8B would have measured `full + te + te_cond + te_all21`
and reported it as conditional TE. The number would have been plausible and the attribution wrong.

Why a composition test is the right shape for this
---------------------------------------------------
Feature-block ablations are only interpretable if the only thing differing between two arms is the
block under test. That is a structural property of the view definitions, so it is checked
structurally: every declared block contributes columns, and every column is attributed to a block the
view actually declares.

It also pins the exact counts, so a block silently changing size (a key list edited, a smoothing
default changed) shows up as a failure rather than as a few e-5 of unexplained drift.

Runs on fold 0 of the primary scheme, which builds one assembled matrix per view. A few minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_cached_parquet  # noqa: E402
from src.features.view import ViewBuilder, VIEWS  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

# expected (n_old_te, n_te_conditional, n_te_all21) per view. The static count is read from the builder
# rather than hard-coded, because it depends on how many transductive/external blocks the cache holds.
EXPECTED = {
    "full":           (48, 0, 0),
    "full_te21":      (48, 0, 21),
    "full_all21te":   (0, 0, 21),
    "full_tec":       (48, 72, 0),
    "full_tec_swap":  (0, 72, 0),
    "core3":          (0, 0, 0),
    "core3_te":       (48, 0, 0),
    "core3_tec":      (48, 72, 0),
    "core3_tec_swap": (0, 72, 0),
}

_CACHE: dict = {}


def _folds():
    if "d" not in _CACHE:
        tr, te = load_cached_parquet()
        y = tr[TARGET].values.astype("int8")
        f = get_scheme("primary", y, tr[ID_COL]).folds
        _CACHE["d"] = (tr, te, y, np.where(f != 0)[0], np.where(f == 0)[0])
    return _CACHE["d"]


def _counts(view):
    tr, te, y, fit, val = _folds()
    vb = ViewBuilder(tr, te, view)
    vb.build_static()
    _Xf, _Xa, names = vb.assemble(fit, y, val, None, inner_seed=0)
    n_old = sum(1 for n in names if n.startswith("te_")
                and not n.startswith("tec") and not n.startswith("te_all21"))
    n_tec = sum(1 for n in names if n.startswith("tec"))
    n_21 = sum(1 for n in names if n.startswith("te_all21"))
    n_static = vb.static_tr.shape[1]
    return n_static, n_old, n_tec, n_21, len(names)


def test_view_composition_matches_declared_blocks():
    """Every view assembles to exactly the blocks it declares, with the expected column counts."""
    tr, te, y, fit, val = _folds()
    problems = []
    for view, (e_old, e_tec, e_21) in EXPECTED.items():
        n_static, n_old, n_tec, n_21, total = _counts(view)
        blocks = VIEWS[view]

        # a view may not contain a block it did not declare -- the bug this file exists for
        if n_21 and "te_all21" not in blocks:
            problems.append(f"{view}: has {n_21} te_all21 columns but does not declare the block "
                            f"(declares {blocks})")
        if n_tec and "te_cond" not in blocks:
            problems.append(f"{view}: has {n_tec} te_cond columns but does not declare it")
        if n_old and "te" not in blocks:
            problems.append(f"{view}: has {n_old} te columns but does not declare it")
        # and the converse: a declared dynamic block must actually contribute
        if "te" in blocks and n_old == 0:
            problems.append(f"{view}: declares `te` but contributed 0 columns")
        if "te_cond" in blocks and n_tec == 0:
            problems.append(f"{view}: declares `te_cond` but contributed 0 columns")
        if "te_all21" in blocks and n_21 == 0:
            problems.append(f"{view}: declares `te_all21` but contributed 0 columns")

        if (n_old, n_tec, n_21) != (e_old, e_tec, e_21):
            problems.append(f"{view}: blocks gave (te={n_old}, tec={n_tec}, all21={n_21}), "
                            f"expected (te={e_old}, tec={e_tec}, all21={e_21})")
        if total != n_static + n_old + n_tec + n_21:
            problems.append(f"{view}: {total} named columns but "
                            f"{n_static}+{n_old}+{n_tec}+{n_21}={n_static+n_old+n_tec+n_21} "
                            f"attributed -- some columns belong to no block")
        print(f"  {view:<16} static={n_static:<4} te={n_old:<3} tec={n_tec:<3} all21={n_21:<3} "
              f"total={total}")
    assert not problems, "\n".join(problems)


def test_conditional_and_all21_blocks_are_mutually_exclusive_by_default():
    """`full` must remain the unmodified champion view.

    The 59 members of v3_final were built from `full` (and `full_ogsurf`, `full_ogm`, `core3`). If a
    change to `assemble` altered what any of those views produce, every recorded member AUC and the
    finalist's OOF would silently stop being reproducible -- and nothing downstream would notice,
    because nothing re-derives those numbers. This test pins the champion view's column count so that
    kind of drift fails loudly instead.
    """
    tr, te, y, fit, val = _folds()
    for view, expected_total in (("full", 285), ("core3", 225)):
        vb = ViewBuilder(tr, te, view)
        vb.build_static()
        _Xf, _Xa, names = vb.assemble(fit, y, val, None, inner_seed=0)
        assert len(names) == expected_total, (
            f"champion view `{view}` now assembles {len(names)} columns, expected {expected_total}. "
            f"v3_final's recorded member AUCs were measured on the old composition, so a silent "
            f"change here invalidates every comparison against them.")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("\nall view-composition checks passed")
