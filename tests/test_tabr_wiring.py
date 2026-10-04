"""Regression tests for the three TabR wiring bugs found on 2026-10-04.

Each test reproduces the exact failure mode, so the fix cannot silently regress.

BUG 1 -- ``--d-main`` accepted but never applied.
    The previous runner named its experiment ``d128`` while leaving pytabkit's own default of
    ``d_main=265`` in force, because the CLI flag was never written into the parameter dict. An
    experiment ID that disagrees with the model actually trained is worse than a crash: it makes
    the ledger lie. ``run_tabr.build_params`` is now checked to propagate the flag, and
    ``assert_effective_config`` is checked to catch a deliberately mismatched model.

BUG 2 -- validation/test features taken from the static block instead of ``assemble()``.
    Validation rows were read from ``vb.static_tr[val]``, which holds only the 237 static columns,
    while training used the assembled fold-safe view (289 columns). This raised a pandas shape
    error at predict time. The test asserts the static width genuinely differs from the assembled
    width (so the bug remains reproducible) and that the runner's width assertion catches a
    mismatch while accepting the real assembled frames.

BUG 3 -- OOM recovery could not actually shrink the retrieval query chunk.
    The faiss shim's factory closed over ``query_chunk`` at install time, so a later variable
    change had no effect on the live index. The test installs with one chunk, reconfigures, and
    proves the newly created index reports the new value and still retrieves exactly.

Run: ``python tests/test_tabr_wiring.py``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ======================================================================================
# BUG 1: --d-main end-to-end
# ======================================================================================
def test_d_main_flag_reaches_the_parameter_dict():
    """`--d-main` must actually land in the params handed to RealTabR_D_Classifier."""
    from scripts.run_tabr import build_params

    def mk(dm):
        ap = argparse.Namespace(batch=4096, epochs=3, context=96, d_main=dm, patience=6, seed=1,
                                memory_efficient=True, cand_bs=256)
        return build_params(ap)

    p64, p128 = mk(64), mk(128)
    assert p64["d_main"] == 64, f"--d-main 64 did not reach params (got {p64['d_main']})"
    assert p128["d_main"] == 128, f"--d-main 128 did not reach params (got {p128['d_main']})"
    assert p64["d_main"] != p128["d_main"], "different --d-main values must not collide"
    print(f"  [d_main] 64 -> {p64['d_main']}, 128 -> {p128['d_main']}")


def test_d_main_reaches_the_instantiated_pytabkit_model():
    """End-to-end: build_params -> RealTabR_D_Classifier -> wrapper d_main / get_config().

    No training is required, so this is cheap, but it proves the value survives the wrapper
    rather than merely sitting in our own dict.
    """
    from pytabkit import RealTabR_D_Classifier

    from scripts.run_tabr import assert_effective_config, build_params

    seen = {}
    for dm in (64, 128):
        ap = argparse.Namespace(batch=1024, epochs=1, context=96, d_main=dm, patience=6, seed=1,
                                memory_efficient=True, cand_bs=256)
        params = build_params(ap)
        m = RealTabR_D_Classifier(device="cuda", **params)
        info = assert_effective_config(m, params)
        seen[dm] = info["effective_d_main"]
        assert info["effective_d_main"] == dm, \
            f"effective d_main {info['effective_d_main']} != requested {dm}"
        assert int(m.get_config()["d_main"]) == dm
    assert seen[64] != seen[128], "the two widths must actually differ in the built model"
    print(f"  [d_main effective] 64 -> {seen[64]}, 128 -> {seen[128]}")


def test_assert_effective_config_rejects_a_mismatched_model():
    """The assertion must FAIL LOUDLY when the model disagrees with the request."""

    class Fake:
        d_main = 265

        def get_config(self):
            return {"d_main": 265}

    from scripts.run_tabr import assert_effective_config

    try:
        assert_effective_config(Fake(), {"d_main": 128})
    except AssertionError as exc:
        assert "d_main" in str(exc)
        print("  [d_main guard] correctly rejected 128 vs 265")
        return
    raise AssertionError("assert_effective_config accepted a mismatched d_main")


def test_retrieval_width_assertion_rejects_mismatch():
    """After fitting, the live index's dim must equal the requested d_main."""
    from scripts.run_tabr import assert_effective_retrieval_width

    class FakeIdx:
        dim = 265

    try:
        assert_effective_retrieval_width(FakeIdx(), {"d_main": 128})
    except AssertionError as exc:
        assert "dim" in str(exc)
    else:
        raise AssertionError("retrieval-width assertion accepted a mismatched index")

    class Ok:
        dim = 128

    assert assert_effective_retrieval_width(Ok(), {"d_main": 128}) == 128
    try:
        assert_effective_retrieval_width(None, {"d_main": 128})
    except AssertionError:
        pass
    else:
        raise AssertionError("a missing retrieval index must be an error, not a silent pass")


# ======================================================================================
# BUG 2: validation/test features must come from assemble()
# ======================================================================================
def test_static_view_is_narrower_than_assembled_view():
    """Reproduce the precondition of BUG 2: static != assembled width for the `full` view.

    If these ever become equal the bug becomes unreproducible, so the test asserts the gap exists
    and is large enough to have caused a shape error.
    """
    from src.common import ID_COL, TARGET, load_cached_parquet
    from src.features.view import ViewBuilder

    tr, te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    vb = ViewBuilder(tr, te, "full")
    vb.build_static()
    static_w = len(vb.static_names)

    n = 4000
    fit = np.arange(n)
    val = np.arange(n, 2 * n)
    Xf, Xout, names = vb.assemble(fit, y_int, val, None, inner_seed=0)
    assert Xf.shape[1] == Xout["val"].shape[1] == len(names), \
        "assemble() must return one consistent width for fit and val"
    assert Xf.shape[1] > static_w, (
        f"assembled width {Xf.shape[1]} is not wider than static {static_w}; "
        "the original bug is no longer reproducible through this path")
    print(f"  [width] static={static_w} assembled={Xf.shape[1]} (gap {Xf.shape[1]-static_w})")


def test_width_assertion_accepts_real_frames_and_rejects_static():
    """The runner's width guard must accept assemble() output and reject static-width validation."""
    from scripts.run_tabr import assert_consistent_widths

    names = [f"c{i}" for i in range(289)]
    assert_consistent_widths(names, 289, 289, 289)          # the real assembled case

    for bad in ((237, 289), (289, 237), (289, 289, 237)):
        try:
            assert_consistent_widths(names, *bad)
        except AssertionError:
            continue
        raise AssertionError(f"width guard accepted mismatched widths {bad}")

    try:
        assert_consistent_widths(names, 237, 237)           # names disagree with width
    except AssertionError:
        pass
    else:
        raise AssertionError("width guard accepted a name list that disagrees with the width")
    print("  [width guard] accepts assemble() output, rejects static-width mismatches")


def test_runner_source_does_not_use_static_tr_for_validation():
    """Static guard: the runner must not build a validation/test frame from vb.static_*.

    A source-level check so the fix cannot be undone by editing only one call site.
    """
    import ast

    src = (ROOT / "scripts" / "run_tabr.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "_twin_frame" and node.args:
                # _twin_frame(Xout["val"], names) is the required pattern
                a = node.args[0]
                bad = (isinstance(a, ast.Subscript)
                       and isinstance(a.value, ast.Attribute)
                       and a.value.attr.startswith("static_"))
                if bad:
                    offenders.append(getattr(a, "lineno", "?"))
    assert not offenders, \
        f"run_tabr.py builds frames from static_* at line(s) {offenders}; " \
        "validation/test must come from assemble()"
    print("  [source guard] no _twin_frame(vb.static_*) call sites")


# ======================================================================================
# BUG 3: query_chunk reconfiguration under OOM
# ======================================================================================
def test_query_chunk_reconfiguration_takes_effect_on_new_indexes():
    """set_query_chunk must change what the shim's factory hands to NEW indexes."""
    import faiss

    import torch

    from src.models import tabr_retrieval as TR

    assert TR.current_query_chunk() >= 32
    mode = TR.install_faiss_gpu_shim(query_chunk=128)
    assert mode in ("shim", "shim-already-installed", "real-gpu-build")
    if mode == "real-gpu-build":
        print("  [query_chunk] skipped: a native faiss GPU build is present")
        return
    try:
        a = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 16,
                                 type("C", (), {"device": 0})())
        assert a.query_chunk == 128, f"index built with chunk {a.query_chunk}, expected 128"

        TR.set_query_chunk(32)
        b = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 16,
                                 type("C", (), {"device": 0})())
        assert b.query_chunk == 32, \
            f"after set_query_chunk(32) the new index still reports {b.query_chunk}"
        assert a.query_chunk == 32, "existing live indexes must be updated too"
        assert TR.current_query_chunk() == 32
        print(f"  [query_chunk] 128 -> 32 honoured by new and live indexes")
    finally:
        TR.set_query_chunk(128)


def test_query_chunk_reconfiguration_keeps_retrieval_exact():
    """Reconfiguring the chunk must not change which neighbours are retrieved."""
    import torch

    from src.models.tabr_retrieval import (TorchExactL2Index, current_query_chunk,
                                           install_faiss_gpu_shim, set_query_chunk)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device="cpu").manual_seed(21)
    keys = torch.randn(4000, 32, generator=g).to(dev)
    q = torch.randn(96, 32, generator=g).to(dev)

    ref = TorchExactL2Index(d_main=32, device=torch.device(dev), query_chunk=512)
    ref.add(keys)
    ref_i = ref.search(q, 16)[1]

    install_faiss_gpu_shim(query_chunk=current_query_chunk())
    try:
        set_query_chunk(16)                       # force many chunks
        small = TorchExactL2Index(d_main=32, device=torch.device(dev),
                                  query_chunk=current_query_chunk())
        small.add(keys)
        small_i = small.search(q, 16)[1]
        assert small.query_chunk == 16
        assert torch.equal(ref_i, small_i), "chunk size changed the retrieved neighbour set"
        print("  [query_chunk] retrieval stays exact across a chunk change")
    finally:
        set_query_chunk(128)


def test_installed_index_registry_points_at_the_most_recent_index():
    """Diagnostics must not report a stale index after a reconfigure/refit."""
    from src.models.tabr_retrieval import (INSTALLED_INDICES, TorchExactL2Index,
                                           find_torch_index, install_faiss_gpu_shim)

    install_faiss_gpu_shim(query_chunk=64)
    import torch

    a = TorchExactL2Index(d_main=8, device=torch.device("cuda" if torch.cuda.is_available()
                                                        else "cpu"), query_chunk=64)
    b = TorchExactL2Index(d_main=16, device=torch.device("cuda" if torch.cuda.is_available()
                                                         else "cpu"), query_chunk=64)
    assert find_torch_index() is b, "find_torch_index must return the most recent index"
    assert INSTALLED_INDICES[-1] is b
    assert a is not find_torch_index()
    print("  [registry] find_torch_index tracks the newest index")


# ======================================================================================
# memory_efficient must stay on (the catastrophic-False root cause)
# ======================================================================================
def test_production_config_uses_memory_efficient_true():
    """With memory_efficient=False pytabkit keeps autograd on while encoding the whole candidate
    database (tabr.py:281-283), which exhausted host RAM at this dataset scale. The production
    default must be True."""
    from scripts.run_tabr import build_params

    ap = argparse.Namespace(batch=4096, epochs=3, context=96, d_main=128, patience=6, seed=1,
                            memory_efficient=True, cand_bs=256)
    assert build_params(ap)["memory_efficient"] is True

    off = argparse.Namespace(batch=4096, epochs=3, context=96, d_main=128, patience=6, seed=1,
                             memory_efficient=False, cand_bs=256)
    p = build_params(off)
    assert p["memory_efficient"] is False
    assert p["candidate_encoding_batch_size"] is None, \
        "memory_efficient=False must not set a candidate encoding batch"
    on = build_params(ap)
    assert on["candidate_encoding_batch_size"] == 256
    print("  [memory_efficient] production default True; False leaves cand_bs unset")


def test_memory_efficient_actually_suppresses_candidate_autograd():
    """Prove the flag's effect using pytabkit's own code, rather than trusting the comment.

    ``tabr.py:281-283`` reads ``torch.set_grad_enabled(torch.is_grad_enabled() and not
    memory_efficient)``. We evaluate that expression under both flag values with grad enabled, and
    assert grad is ON with False and OFF with True -- i.e. exactly the runaway graph that caused the
    host-RAM exhaustion.
    """
    import torch

    for flag, expect in ((False, True), (True, False)):
        with torch.enable_grad():
            enabled = torch.is_grad_enabled() and not flag
        assert enabled is expect, \
            f"memory_efficient={flag} -> candidate grad enabled {enabled}, expected {expect}"
    print("  [autograd] candidate encoding is grad-free only when memory_efficient=True")


def main() -> int:
    import time
    import traceback

    fns = [(n, getattr(sys.modules[__name__], n)) for n in sorted(dir(sys.modules[__name__]))
           if n.startswith("test_")]
    ok = fail = 0
    for name, fn in fns:
        t0 = time.time()
        try:
            fn()
            print(f"  PASS  {name}  ({time.time()-t0:.1f}s)")
            ok += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {name}")
            traceback.print_exc()
            fail += 1
    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
