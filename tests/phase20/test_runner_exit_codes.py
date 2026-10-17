"""Phase 20 infrastructure contract: a FAILED screen must not report success.

Mandate S8: "Fix experiment runners that return exit code zero after a FAILED
run. Require a successful result manifest for DONE status."

These are static (source-inspection) tests -- no GPU, no data -- so they run in
the dependency-free suite and catch the defect at commit time rather than at
scheduler time. A runner that sets status="FAILED" in an except-branch and then
unconditionally `return 0` is exactly the bug that let a killed experiment look
healthy to the orchestration layer.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE20 = ROOT / "scripts" / "phase20"

# Runners that own a DONE/FAILED status branch and must propagate failure.
SCREEN_RUNNERS = (
    "tabicl_ft_fold0.py",
    "tabicl_ft_recover.py",
    "neural_screen_fold0.py",
    "mtl_screen_fold0.py",
    "kumo_screen_fold0.py",
    "classical_screen_fold0.py",
)


def _final_return_of(path: Path) -> ast.Return:
    """The return statement of main(), i.e. the last top-level Return in the AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rets = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    assert rets, f"{path.name} has no return statement"
    # main()'s own return is the one whose value is an IfExp or Compare guarding
    # status; safest structural proxy: the Return with the highest lineno.
    return max(rets, key=lambda n: n.lineno)


def _mentions_status_failed(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    # Two idioms in the fleet: `status = "FAILED"` and tuple-assignment
    # `pred, m, status, err = None, None, "FAILED", repr(exc)`.
    return 'status = "FAILED"' in src or '"FAILED",' in src


def test_screen_runners_set_failed_status():
    for name in SCREEN_RUNNERS:
        p = PHASE20 / name
        assert p.exists(), f"missing runner {name}"
        assert _mentions_status_failed(p), f"{name} no longer has a FAILED branch; update SCREEN_RUNNERS"


def test_screen_runners_exit_nonzero_on_failed():
    for name in SCREEN_RUNNERS:
        p = PHASE20 / name
        final = _final_return_of(p)
        val = final.value
        # Accept: ternary `0 if (...) else 1`, or an explicit nonzero literal.
        if isinstance(val, ast.IfExp):
            # The success value must be the true-branch (0) and failure 1.
            true_n, false_n = val.orelse, val.body
            assert isinstance(true_n, ast.Constant) and true_n.value == 1, (
                f"{name}: final return must yield 1 on failure (false-branch), got AST {ast.dump(true_n)}")
            assert isinstance(false_n, ast.Constant) and false_n.value == 0, (
                f"{name}: final return must yield 0 on success (true-branch)")
            # The guard must reference the status variable, not a constant.
            assert "status" in ast.unparse(val.test), (
                f"{name}: failure guard does not test `status`; got {ast.unparse(val.test)}")
        elif isinstance(val, ast.Constant):
            assert val.value != 0, f"{name}: unconditional `return {val.value}` reports FAILED as success"
        else:
            raise AssertionError(f"{name}: unrecognised final return form: {ast.unparse(val)}")


def test_tabicl_recover_requires_verified_manifest_for_done():
    # The recovery runner's DONE path must be conditioned on a saved, hash-
    # checked prediction; a bare status string is not enough.
    src = (PHASE20 / "tabicl_ft_recover.py").read_text(encoding="utf-8")
    assert "prediction_sha256" in src, "recovery runner no longer records a prediction hash"
    assert "arr_sha256" in src, "recovery runner no longer hash-verifies its prediction array"
    assert "eval_rows" in src, "recovery runner no longer checks the evaluated row count"
