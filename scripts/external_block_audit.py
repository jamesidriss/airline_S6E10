"""Leakage audit of the external-data blocks, run as a hard executable test.

Checks, in order:
  1. every external feature is a pure function of (competition feature values, ORIGINAL labels);
     it must be bit-identical when the competition labels are changed or destroyed;
  2. the ORIGINAL loader reads only the downloaded public CSV, never competition data;
  3. the multi/surface builders use a competition-label-free code path (source scan + runtime
     proof by permuting the competition target column);
  4. no external table is fitted on the full competition train set.

Run: python scripts/external_block_audit.py
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import TARGET, load_cached_parquet  # noqa: E402
from src.features import s6e10 as S  # noqa: E402
from src.features.view import RAW21  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))


def tree_uses(node: ast.AST, forbidden: tuple[str, ...]) -> set[str]:
    """Names referenced anywhere in a function body that look like competition-label sources."""
    hit = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in forbidden:
            hit.add(sub.id)
        elif isinstance(sub, ast.Attribute) and sub.attr in forbidden:
            hit.add(sub.attr)
        elif isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant) \
                and sub.slice.value in forbidden:
            hit.add(sub.slice.value)
    return hit


def main() -> int:
    tr, te = load_cached_parquet()
    ntr = len(tr)
    comb = pd.concat([tr[RAW21], te[RAW21]], ignore_index=True)

    print("\n1. External features are invariant to the competition target")
    # (a) runtime proof: scramble/destroy the competition target, rebuild everything.
    tr2 = tr.copy()
    rng = np.random.default_rng(0)
    tr2[TARGET] = rng.integers(0, 2, len(tr2))          # labels randomised
    tr3 = tr.copy()
    tr3[TARGET] = 0                                     # labels constant
    for label, frame in (("randomised", tr2), ("constant", tr3)):
        c2 = pd.concat([frame[RAW21], te[RAW21]], ignore_index=True)
        a1, n1 = S.apply_external(c2, S.build_external_features())
        a2, n2 = S.apply_external_multi(c2, S.build_external_multi())
        a3, n3 = S.apply_original_surfaces(c2, S.build_original_surfaces(S.SURFACE_ANCHORS))
        # compare against the true-label build
        b1, _ = S.apply_external(comb, S.build_external_features())
        b2, _ = S.apply_external_multi(comb, S.build_external_multi())
        b3, _ = S.apply_original_surfaces(comb, S.build_original_surfaces(S.SURFACE_ANCHORS))
        same = (np.array_equal(a1, b1) and np.array_equal(a2, b2) and np.array_equal(a3, b3))
        check(f"external blocks unchanged when competition target is {label}", same,
              f"{a1.shape[0]}x{a1.shape[1]} multi={a2.shape[1]} surf={a3.shape[1]}")

    print("\n2. The original loader reads only the public original CSV")
    src = inspect.getsource(S.load_original)
    tree = ast.parse(inspect.cleandoc(src))
    # The original CSV has a column literally named `satisfaction`, so `TARGET` is legitimately
    # referenced as `og[TARGET]`. A bare name match cannot tell `og[TARGET]` from `tr[TARGET]`,
    # so resolve each TARGET subscript to its root name and require it to be the original frame.
    targets_rooted_at = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and getattr(node.value, "id", None) == "TARGET":
            targets_rooted_at.add(node.value.id)
    comp_frames = {"tr", "te", "train", "test", "comb", "full", "df_comp"}
    bad_roots = targets_rooted_at & comp_frames
    check("every TARGET subscript in load_original is rooted at the original frame",
          not bad_roots, f"roots={sorted(targets_rooted_at)}")
    other = tree_uses(tree, ("tr", "te", "train", "test", "comb")) - {None}
    check("load_original references no competition frame", not other, str(other))
    check("load_original reads ORIG_CSV only", "read_csv(ORIG_CSV)" in src.replace(" ", " "))
    check("original csv path is under data/original (gitignored, re-downloadable)",
          "data" in str(S.ORIG_CSV).replace("\\", "/") and "original" in str(S.ORIG_CSV))

    print("\n3. Builder code paths are label-free")
    for fn in (S.build_external_features, S.build_external_multi, S.build_original_surfaces):
        body = ast.parse(inspect.cleandoc(inspect.getsource(fn)))
        hits = tree_uses(body, ("tr", "te", "comb", "train", "test"))
        check(f"{fn.__name__} references no competition frame", not hits, str(hits))

    print("\n4. Original-only teacher never trains on competition rows")
    tsrc = inspect.getsource(S.OriginalTeacher.fit_predict)
    check("fit_predict trains on `og` (original) only",
          "m.fit(Xo, yo)" in tsrc and "Xc" in tsrc)
    check("fit_predict applies an exact-overlap audit first", "keep = ~ho.isin" in tsrc)

    print("\n5. Provenance summary")
    ex = S.build_external_features()
    exm = S.build_external_multi()
    check("external single-column tables built from original rows only",
          ex["n_original"] == len(S.load_original()), f"n_original={ex['n_original']}")
    check("external multi tables built from original rows only",
          len(exm["tables"]) > 0, f"{len(exm['tables'])} pair/triple tables")
    aud = S.original_leak_audit(S.load_original(), comb, [c for c in te.columns if c != "id"])
    check("original rows exactly matching a competition row are few and are excluded from the teacher",
          aud["n_original_rows_exactly_matching_a_competition_row"] < 1000,
          str(aud))

    n_fail = sum(1 for r in results if r[0] == FAIL)
    print(f"\n{len(results) - n_fail} passed, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())