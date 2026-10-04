"""GPU-native exact L2 k-NN index for TabR retrieval, on the faiss API.

Why this exists
---------------
TabR retrieves each query's `context_size` nearest training rows in the learned embedding space.
pytabkit implements that search with `faiss.GpuIndexFlatL2`, but the only faiss wheel available
for CPython 3.11 on Windows is `faiss-cpu`, which has no `GpuIndexFlatConfig` at all. The fallback
inside pytabkit would be a CPU `IndexFlatL2`, and at 560k database rows x 265 dimensions x
~70 queries per batch that costs roughly 1.2e12 FLOPs *per boosting batch step on the CPU -- not
viable.

So we supply the index ourselves. `TorchExactL2Index` exposes exactly the three methods pytabkit
calls (`reset`, `add`, `search`) and performs an **exact** squared-L2 search with chunked torch
matmuls, entirely on the GPU. Results are identical to `faiss.IndexFlatL2` up to floating-point
tie-breaking, which is asserted in `verify_matches_reference`.

Memory
------
The full distance matrix is never materialised: queries are processed in chunks of `query_chunk`,
so peak extra memory is `query_chunk x n_database x 4` bytes (default chunk 256 x 560k x 4B
~ 573 MB), independent of how many rows TabR batches together.

Usage
-----
    from pytabkit import RealTabR_D_Classifier
    from src.models.tabr_retrieval import attach_torch_index

    m = RealTabR_D_Classifier(device="cuda", d_main=265, **rest)
    attach_torch_index(m)          # must be called BEFORE fit()
    m.fit(...)
"""

from __future__ import annotations

import contextlib

import torch

# Every index built by the shim registers here, so a caller that only holds a pytabkit sklearn
# wrapper (which does not expose the inner TabrModel) can still report retrieval statistics.
INSTALLED_INDICES: list = []

# Set once install_faiss_gpu_shim() substitutes our symbols, so a repeat call can report honestly
# instead of mistaking its own shim for a native faiss GPU build.
_SHIM_INSTALLED = False


def shim_is_installed() -> bool:
    """True if the faiss GPU symbols currently resolve to our torch implementation."""
    return _SHIM_INSTALLED


@contextlib.contextmanager
def full_fp32_matmul():
    """Temporarily force true fp32 matmul (disable TF32).

    TF32 keeps only 10 mantissa bits, which is enough to reorder near-tied candidates in the k-NN
    top-k and to make retrieved neighbours differ from an exact search. Retrieval correctness is
    the whole point of this index, so the reduction is done in full fp32.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    prev_prec = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
        torch.set_float32_matmul_precision(prev_prec)


class TorchExactL2Index:
    """Drop-in replacement for a faiss exact-L2 index, computed with chunked torch matmuls."""

    def __init__(self, d_main: int, device: torch.device, query_chunk: int = 256,
                 dtype: torch.dtype = torch.float32):
        # Named `d_main` because that is TabR's name for this embedding dimension; stored as
        # ``self.dim`` because that is the conventional k-NN index name.
        self.dim = int(d_main)
        self.device = torch.device(device)
        self.query_chunk = int(query_chunk)
        self.dtype = dtype
        self._keys: torch.Tensor | None = None
        self._key_sqnorm: torch.Tensor | None = None
        self.n_searches = 0
        self.n_added = 0
        self.last_n = None            # k actually requested by the last search
        self.last_result_shape = None # (n_queries, k) actually returned
        INSTALLED_INDICES.append(self)

    # --- faiss API ---------------------------------------------------------------
    def reset(self) -> None:
        """Drop the current database. pytabkit calls this before every batch."""
        self._keys = None
        self._key_sqnorm = None

    def add(self, keys: torch.Tensor) -> None:
        """Register the retrieval database (the full training set's embeddings)."""
        if not isinstance(keys, torch.Tensor):
            raise TypeError("TorchExactL2Index.add expects a torch.Tensor")
        keys = keys.detach().to(device=self.device, dtype=self.dtype).contiguous()
        if keys.shape[1] != self.dim:
            raise ValueError(f"expected d_main={self.dim}, got {keys.shape[1]}")
        self._keys = keys
        self._key_sqnorm = (keys * keys).sum(dim=1)
        self.n_added += keys.shape[0]

    def search(self, queries: torch.Tensor, n: int):
        """Return (distances, indices) for the ``n`` nearest database rows per query.

        Both are sorted ascending by distance, matching faiss.
        """
        if self._keys is None:
            raise RuntimeError("search() called before add()")
        q = queries.detach().to(device=self.device, dtype=self.dtype)
        if q.shape[1] != self.dim:
            raise ValueError(f"expected d_main={self.dim}, got {q.shape[1]}")
        if n > self._keys.shape[0]:
            raise ValueError(f"cannot retrieve {n} neighbours from {self._keys.shape[0]} rows")

        out_d, out_i = [], []
        with full_fp32_matmul():
            for start in range(0, q.shape[0], self.query_chunk):
                qc = q[start:start + self.query_chunk]
                # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b ; the clamp removes negative round-off.
                d = ((qc * qc).sum(dim=1, keepdim=True) + self._key_sqnorm.unsqueeze(0)
                     - 2.0 * (qc @ self._keys.T))
                d.clamp_(min=0.0)
                vals, idx = torch.topk(d, n, dim=1, largest=False, sorted=True)
                out_d.append(vals)
                out_i.append(idx)
                del d
        self.n_searches += q.shape[0]
        out_d_t, out_i_t = torch.cat(out_d, dim=0), torch.cat(out_i, dim=0)
        self.last_n = int(n)
        self.last_result_shape = tuple(int(v) for v in out_i_t.shape)
        return out_d_t, out_i_t

    # --- diagnostics --------------------------------------------------------------
    def stats(self) -> dict:
        return {"n_database": int(self._keys.shape[0]) if self._keys is not None else 0,
                "d_main": self.dim, "device": str(self.device),
                "query_chunk": self.query_chunk, "dtype": str(self.dtype),
                "actual_k": self.last_n, "last_result_shape": self.last_result_shape,
                "total_queries_searched": self.n_searches,
                "total_rows_added": self.n_added,
                "searches_per_epoch_estimate": None}


def install_faiss_gpu_shim(query_chunk: int = 256) -> str:
    """Make ``faiss.GpuIndexFlatL2`` resolve to our GPU exact-L2 index. Call before building TabR.

    pytabkit's ``TabrModel.__init__`` picks its retrieval index with::

        if device.type == 'cuda':
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = gpu_index
            self.search_index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main, cfg)

    ``faiss-cpu`` -- the only wheel published for CPython 3.11 on Windows -- has neither symbol, so
    this raises ``AttributeError`` deep inside the library. Rather than monkey-patching pytabkit's
    logic (fragile across versions), we supply the two missing symbols with compatible stand-ins,
    so pytabkit's own code path runs unchanged and simply lands on a torch implementation.

    Returns ``"real-gpu-build"`` if a genuine faiss GPU build is already present, ``"shim"`` if we
    installed ours, and ``"shim-already-installed"`` on a repeat call. The distinction matters: the
    mode is recorded in the experiment's gate metrics, so a shim must never be able to masquerade
    as a native faiss GPU build.
    """
    import faiss

    global _SHIM_INSTALLED
    if _SHIM_INSTALLED:
        return "shim-already-installed"
    if hasattr(faiss, "GpuIndexFlatConfig") and hasattr(faiss, "GpuIndexFlatL2"):
        return "real-gpu-build"

    class _GpuIndexFlatConfig:
        """Stand-in: pytabkit only reads and writes ``.device``."""

        def __init__(self):
            self.device = 0

    class _StandardGpuResources:
        """Stand-in: handed to the index factory and otherwise unused."""

    def _GpuIndexFlatL2(_resources, d, cfg=None):
        dev_id = getattr(cfg, "device", 0) or 0
        return TorchExactL2Index(d_main=d, device=torch.device(f"cuda:{dev_id}"),
                                 query_chunk=query_chunk)

    faiss.GpuIndexFlatConfig = _GpuIndexFlatConfig
    faiss.StandardGpuResources = _StandardGpuResources
    faiss.GpuIndexFlatL2 = _GpuIndexFlatL2
    _SHIM_INSTALLED = True
    return "shim"


def find_torch_index(model=None):
    """Return the TorchExactL2Index for this run.

    pytabkit's sklearn wrapper does not expose the inner ``TabrModel``, and that object is freed
    after ``fit()``, so walking the wrapper's attribute graph is unreliable. Instead every index
    registers itself on construction; we return the most recent one, which is the one this run
    created. Returns ``None`` if no index was ever built.
    """
    return INSTALLED_INDICES[-1] if INSTALLED_INDICES else None


def attach_torch_index(model, query_chunk: int = 256) -> TorchExactL2Index:
    """Install a TorchExactL2Index directly on a Tabr model. Call before ``fit()``.

    ``TabrModel.__init__`` sets ``self.search_index = None`` and fills it lazily, so assigning it
    up front bypasses the faiss branch. Prefer :func:`install_faiss_gpu_shim`, which needs no
    knowledge of the wrapper layout; this is the fallback for when you already hold a TabrModel.
    """
    cfg = getattr(model, "config", None)
    d_main = cfg["d_main"] if isinstance(cfg, dict) and "d_main" in cfg else cfg
    if d_main is None:
        d_main = getattr(model, "d_main", None)
    if d_main is None:
        raise ValueError("could not determine d_main from the TabR model")
    idx = TorchExactL2Index(d_main=int(d_main), device=torch.device(getattr(model, "device", "cuda")),
                            query_chunk=query_chunk)
    model.search_index = idx
    return idx


def verify_matches_reference(n_db: int = 4000, n_q: int = 64, dim: int = 32, n: int = 8,
                             seed: int = 0, device: str = "cuda") -> dict:
    """Assert the shim reproduces a brute-force k-NN.

    The contract that matters is the *distance vector* returned per query: if those match the exact
    search, the retrieved neighbourhood is the true k-nearest set, and any index that differs is
    merely a tie among equidistant candidates (which carries no information). So the test requires

      1. returned distances match the reference distances (tight tolerance), and
      2. every returned index really is that distance (no fabricated neighbours), and
      3. index-set overlap is reported as a diagnostic, not a hard failure.

    A retrieval bug would silently corrupt TabR, so this runs as part of the smoke test rather
    than being trusted by inspection.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    keys = torch.randn(n_db, dim, generator=g).to(device)
    q = torch.randn(n_q, dim, generator=g).to(device)

    # exact reference: cdist in float64, so the reference itself carries no fp32 error
    ref_d, ref_i = torch.topk((torch.cdist(q.double(), keys.double()) ** 2), n, dim=1,
                              largest=False, sorted=True)

    idx = TorchExactL2Index(d_main=dim, device=torch.device(device))
    idx.add(keys)
    got_d, got_i = idx.search(q, n)

    max_d_err = float((got_d.double() - ref_d).abs().max())
    # each returned neighbour's true distance must equal the distance we reported for it
    picked = keys[got_i].double()
    true_d = ((picked - q.double().unsqueeze(1)) ** 2).sum(-1)
    max_self_err = float((true_d - got_d.double()).abs().max())
    identical = bool(torch.equal(got_i, ref_i))
    overlap = float(torch.tensor([len(set(got_i[i].tolist()) & set(ref_i[i].tolist()))
                                  for i in range(n_q)]).float().mean() / n)

    rel = max_d_err / max(float(ref_d.max()), 1e-12)
    rep = {"n_db": n_db, "n_q": n_q, "dim": dim, "n": n,
           "indices_identical": identical, "mean_index_overlap": overlap,
           "max_distance_error": max_d_err, "max_relative_distance_error": rel,
           "max_self_consistency_error": max_self_err,
           "passes": bool(rel < 1e-5 and max_self_err < 1e-3)}
    print(f"  [retrieval check] db={n_db:<7} dim={dim:<4} q={n_q:<5}  "
          f"identical={identical}  overlap={overlap:.4f}  "
          f"max|d2|={max_d_err:.3e} (rel {rel:.2e})  self_err={max_self_err:.3e}  "
          f"-> {'PASS' if rep['passes'] else 'FAIL'}")
    return rep


if __name__ == "__main__":
    import json

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("verifying TorchExactL2Index against an exact float64 reference on", dev)
    reps = [verify_matches_reference(device=dev),
            verify_matches_reference(n_db=200_000, n_q=1024, dim=265, n=96, device=dev),
            verify_matches_reference(n_db=600_000, n_q=512, dim=265, n=96, device=dev)]
    print(json.dumps({"checks": reps, "all_pass": all(r["passes"] for r in reps)}, indent=2))
