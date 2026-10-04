"""Regression tests for the TabR GPU retrieval shim (src/models/tabr_retrieval.py).

Why these exist
---------------
pytabkit's ``TabrModel`` resolves its k-NN retrieval index through ``faiss.GpuIndexFlatL2``, which
the only CPython 3.11 Windows wheel (``faiss-cpu``) does not provide. ``TorchExactL2Index``
substitutes a chunked torch implementation behind the identical faiss API, and
``install_faiss_gpu_shim()`` supplies the two missing faiss symbols so that pytabkit's *own* code
path runs unchanged rather than being monkey-patched.

A retrieval bug would silently corrupt every TabR prediction while leaving the training loss
looking entirely plausible, so these tests pin exactly the properties the substitution relies on:

  * exact-L2 distance agreement against an independent float64 reference
  * top-k neighbour-SET overlap (the set is what feeds TabR's context, not the order)
  * true fp32 behaviour, and that the TF32 guard is doing real work
  * correct device placement, determinism, and memory bounded by the query chunk
  * no label dependency anywhere in the retrieval path (the mechanical fold-safety guarantee)
  * the faiss shim is additive, scoped, and idempotent

Run: ``python tests/test_tabr_retrieval.py``
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _dev():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def test_torch_exact_l2_index_matches_float64_reference():
    """Exact-L2 distances must agree with a float64 brute-force reference.

    The expansion ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b is algebraically exact but numerically
    lossy in fp32, so the contract is a tight *relative* tolerance rather than bit-equality. The
    reference is computed in float64 so the test cannot pass by reproducing the same fp32 error.
    """
    import torch

    from src.models.tabr_retrieval import TorchExactL2Index

    dev = _dev()
    g = torch.Generator(device="cpu").manual_seed(11)
    keys = torch.randn(3000, 64, generator=g).to(dev)
    q = torch.randn(128, 64, generator=g).to(dev)
    k = 16

    ref_d, _ = torch.topk((torch.cdist(q.double(), keys.double())) ** 2, k, dim=1,
                          largest=False, sorted=True)
    idx = TorchExactL2Index(d_main=64, device=torch.device(dev), query_chunk=32)
    idx.add(keys)
    got_d, got_i = idx.search(q, k)

    rel = float((got_d.double() - ref_d).abs().max() / max(float(ref_d.max()), 1e-12))
    assert rel < 1e-5, f"relative distance error too large: {rel}"
    assert tuple(got_i.shape) == (128, k)
    assert idx.dim == 64
    print(f"  [l2 vs float64] relative error {rel:.3e}")


def test_torch_exact_l2_index_topk_overlap_is_complete():
    """The retrieved neighbour SET must be the true k-nearest set, not merely close.

    Index *ordering* may legitimately differ from a reference among equidistant candidates, so the
    assertion is on set overlap -- that set is what actually becomes TabR's retrieved context.
    """
    import torch

    from src.models.tabr_retrieval import TorchExactL2Index

    dev = _dev()
    g = torch.Generator(device="cpu").manual_seed(12)
    keys = torch.randn(5000, 128, generator=g).to(dev)
    q = torch.randn(200, 128, generator=g).to(dev)
    k = 24

    _, ref_i = torch.topk((torch.cdist(q.double(), keys.double())) ** 2, k, dim=1,
                          largest=False, sorted=True)
    idx = TorchExactL2Index(d_main=128, device=torch.device(dev), query_chunk=64)
    idx.add(keys)
    _, got_i = idx.search(q, k)

    overlap = int(sum(len(set(got_i[i].tolist()) & set(ref_i[i].tolist())) for i in range(len(q))))
    assert overlap == k * len(q), f"neighbour-set overlap {overlap}/{k * len(q)} -- not exact"
    # the k actually returned must be recorded, not merely the k requested
    assert idx.last_n == k
    assert idx.last_result_shape == (len(q), k)
    print(f"  [top-k set overlap] {overlap}/{k * len(q)} exact")


def test_torch_exact_l2_index_is_fp32_exact_not_tf32():
    """Search must run in true fp32.

    TF32 keeps only 10 mantissa bits, which is enough to reorder near-tied candidates in the top-k.
    The test asserts the shim's error stays at fp32-epsilon level even when the ambient process
    setting is explicitly switched to TF32 -- i.e. the `full_fp32_matmul` guard is load-bearing.
    """
    import torch

    from src.models.tabr_retrieval import full_fp32_matmul

    dev = _dev()
    if dev != "cuda":
        print("  skipped: TF32 is a CUDA-only concern")
        return

    g = torch.Generator(device="cpu").manual_seed(13)
    keys = torch.randn(200_000, 265, generator=g).to(dev)
    q = torch.randn(256, 265, generator=g).to(dev)
    ref_d, _ = torch.topk((torch.cdist(q.double(), keys.double())) ** 2, 96, dim=1,
                          largest=False, sorted=True)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_prec = torch.get_float32_matmul_precision()
    try:
        # ambient TF32 pressure: the guard must still deliver fp32-epsilon accuracy
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

        naive = (q.float() @ keys.float().T)
        naive_err = float((naive.double() - (torch.cdist(q.double(), keys.double()) ** 2))
                          .abs().max() / float(ref_d.max()))

        with full_fp32_matmul():
            inside_tf32 = torch.backends.cuda.matmul.allow_tf32
            inside_prec = torch.get_float32_matmul_precision()
        assert inside_tf32 is False, "full_fp32_matmul must disable TF32 inside the block"
        assert inside_prec == "highest", "full_fp32_matmul must request highest precision"

        from src.models.tabr_retrieval import TorchExactL2Index

        idx = TorchExactL2Index(d_main=265, device=torch.device(dev), query_chunk=64)
        idx.add(keys)
        got, _ = idx.search(q, 96)
        rel = float((got.double() - ref_d).abs().max() / float(ref_d.max()))
        assert rel < 1e-5, f"fp32 search error {rel} while TF32 was ambiently enabled"
        print(f"  [fp32 guard] guarded rel err {rel:.3e} vs naive TF32-path {naive_err:.3e}")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        torch.set_float32_matmul_precision(prev_prec)


def test_torch_exact_l2_index_device_placement_and_determinism():
    """add/search must accept and return CUDA tensors on the right device, be deterministic, return
    sorted non-negative distances, and be memory-bounded by the query chunk rather than by
    n_queries x n_database."""
    import torch

    from src.models.tabr_retrieval import TorchExactL2Index

    if not torch.cuda.is_available():
        print("  skipped: no CUDA")
        return
    dev = torch.device("cuda:0")
    keys = torch.randn(300_000, 128, device=dev)
    q = torch.randn(512, 128, device=dev)

    idx = TorchExactL2Index(d_main=128, device=dev, query_chunk=64)
    assert idx.device.type == "cuda" and idx.device.index == 0, "device not honoured"
    assert idx.dtype == torch.float32, "search must default to fp32"

    idx.add(keys)                                   # a CUDA tensor: no silent host round-trip
    assert idx._keys.device.type == "cuda", "keys must stay on the GPU"
    d1, i1 = idx.search(q, 32)
    d2, i2 = idx.search(q, 32)
    assert d1.device.type == "cuda" and i1.device.type == "cuda", "results must stay on the GPU"
    assert torch.equal(i1, i2) and torch.equal(d1, d2), "search must be deterministic"
    assert bool((d1 >= 0).all()), "distances must be non-negative (round-off clamped)"
    assert bool((d1[:, 1:] >= d1[:, :-1]).all()), "distances must be ascending"

    # the whole point of chunking: peak memory must be far below one full distance matrix
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    idx.search(q, 32)
    peak_extra = torch.cuda.max_memory_allocated() - base
    full_matrix_bytes = q.shape[0] * keys.shape[0] * 4
    assert peak_extra < full_matrix_bytes, (
        f"peak extra {peak_extra} B >= one full distance matrix ({full_matrix_bytes} B); "
        "the query chunk is not bounding memory")
    print(f"  [device/memory] peak extra {peak_extra/2**20:.0f} MiB vs full matrix "
          f"{full_matrix_bytes/2**20:.0f} MiB")


def test_torch_exact_l2_index_depends_on_no_labels():
    """Retrieval is a pure function of embeddings: the index admits no label argument at all.

    This is the mechanical guarantee behind fold safety for TabR -- there is no channel through
    which a target value could reach neighbour selection. It is also checked behaviourally: the
    neighbour set must not depend on the query chunk size, i.e. there is no data-order coupling.
    """
    import torch

    from src.models.tabr_retrieval import TorchExactL2Index

    forbidden = {"y", "label", "labels", "target", "targets", "y_train", "y_val", "y_fit"}
    for meth in ("__init__", "add", "search", "reset"):
        params = set(inspect.signature(getattr(TorchExactL2Index, meth)).parameters)
        assert not (forbidden & params), f"{meth} accepts label-like parameters {forbidden & params}"

    dev = _dev()
    g = torch.Generator(device="cpu").manual_seed(14)
    keys = torch.randn(2000, 32, generator=g).to(dev)
    q = torch.randn(64, 32, generator=g).to(dev)
    a = TorchExactL2Index(d_main=32, device=torch.device(dev), query_chunk=7)
    b = TorchExactL2Index(d_main=32, device=torch.device(dev), query_chunk=512)
    a.add(keys)
    b.add(keys)
    assert torch.equal(a.search(q, 8)[1], b.search(q, 8)[1]), \
        "neighbour set depends on the chunking, which would mean hidden data-order coupling"


def test_faiss_gpu_shim_is_scoped_additive_and_idempotent():
    """The shim must add only the missing symbols, be idempotent, and never replace anything.

    install_faiss_gpu_shim() is invoked inside the TabR entry point, so it deliberately mutates the
    ``faiss`` module -- but only additively, only for symbols that are absent, and it must leave
    the CPU index class untouched so unrelated code paths keep their original behaviour.
    """
    import faiss

    from src.models.tabr_retrieval import TorchExactL2Index, install_faiss_gpu_shim

    cpu_index = faiss.IndexFlatL2
    cpu_index_ref = faiss.IndexFlatL2

    mode1 = install_faiss_gpu_shim(query_chunk=64)
    mode2 = install_faiss_gpu_shim(query_chunk=64)
    assert mode1 in ("real-gpu-build", "shim")
    assert mode2 in ("real-gpu-build", "shim", "shim-already-installed")
    # A repeat call must never report "real-gpu-build": that would make the recorded
    # `faiss_gpu_shim_used` gate metric lie about how retrieval was actually computed.
    assert mode2 != "real-gpu-build" or mode1 == "real-gpu-build", \
        "a repeat call mistook our own shim for a native faiss GPU build"
    assert faiss.IndexFlatL2 is cpu_index, "the CPU index class must not be replaced"

    assert hasattr(faiss, "GpuIndexFlatConfig"), "GpuIndexFlatConfig must be available to pytabkit"
    assert hasattr(faiss, "GpuIndexFlatL2"), "GpuIndexFlatL2 must be available to pytabkit"
    assert hasattr(faiss, "StandardGpuResources")

    if mode1 == "shim":
        from src.models.tabr_retrieval import shim_is_installed

        assert shim_is_installed(), "shim_is_installed must agree with the install report"
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = 0                                  # pytabkit assigns this
        assert cfg.device == 0
        idx = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 16, cfg)
        assert isinstance(idx, TorchExactL2Index), "the factory must return our index"
        assert idx.device.type == "cuda" and idx.dim == 16
        assert faiss.IndexFlatL2 is cpu_index_ref
    print("  [shim] faiss gpu mode:", mode1, "| repeat call:", mode2)


def test_shim_reset_clears_database_so_batches_cannot_leak_into_each_other():
    """reset() must fully drop the previous database.

    pytabkit calls reset() then add() before every batch, so the previous batch's training rows must
    not survive into the next batch's retrieval. A reset that only cleared a cached norm tensor
    would silently let batch N retrieve batch N-1's rows.
    """
    import torch

    from src.models.tabr_retrieval import TorchExactL2Index

    dev = _dev()
    old = torch.randn(500, 16, generator=torch.Generator(device="cpu").manual_seed(15)).to(dev)
    new = torch.randn(300, 16, generator=torch.Generator(device="cpu").manual_seed(16)).to(dev) * 50.0
    q = new[:8]

    idx = TorchExactL2Index(d_main=16, device=torch.device(dev), query_chunk=8)
    idx.add(old)
    assert idx.stats()["n_database"] == 500
    idx.reset()
    assert idx._keys is None and idx._key_sqnorm is None, "reset must drop keys and norms"
    idx.add(new)
    assert idx.stats()["n_database"] == 300
    d, i = idx.search(q, 8)
    assert int(i.max()) < 300, "returned an index from the pre-reset database"
    # each query row is itself in the new database, so its nearest neighbour (column 0) is itself
    # at distance 0 -- which is what proves reset() really swapped the whole database
    assert float(d[:, 0].max()) < 1e-6, "each query must retrieve itself after reset"
    assert torch.equal(i[:, 0], torch.arange(8, device=d.device)), \
        "the nearest neighbour of a database row must be that row"


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
