"""Name the hot op inside Tabr's forward with the PyTorch profiler.

Already established by measurement:
  * per training step: TabrModel.forward = 38.3 s, get_Xy = 0.04 s
  * implied epoch time = 78 min for 123 steps at batch 4096, d_main 128, context 96
  * three earlier hypotheses (candidate-encoding chunking, process-wide fp32, torch.isin) were each
    measured and are each only seconds per epoch -- so reasoning is not identifying this

This profiles a bounded number of real steps and prints the top CUDA ops by total time, which names
the culprit instead of guessing at it. Never touches predictions or the prediction store.

Usage:
  python scripts/profile_tabr_ops.py --steps 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, TARGET, load_cached_parquet, set_seed  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.models.realmlp import _twin_frame  # noqa: E402
from src.models.tabr_retrieval import (enable_tf32_process_wide, install_faiss_gpu_shim,  # noqa: E402
                                       verify_matches_reference)
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_tabr import TABR_PRESET  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--d-main", type=int, default=128)
    ap.add_argument("--context", type=int, default=96)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--query-chunk", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--top", type=int, default=18)
    args = ap.parse_args()

    if not verify_matches_reference(device="cuda")["passes"]:
        raise SystemExit("retrieval verification failed")
    install_faiss_gpu_shim(query_chunk=args.query_chunk)
    enable_tf32_process_wide()

    tr, te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y_int, tr[ID_COL]).folds
    fit = np.where(folds != args.fold)[0]
    inner_tr, inner_es = _inner_es_split(fit, y_int, 1 + args.fold)

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()
    Xf, Xout, names = vb.assemble(inner_tr, y_int, inner_es, None, inner_seed=args.fold)
    d = _twin_frame(Xf, names)
    val_small = _twin_frame(Xf[:2048], names)
    print(f"train={d.shape} val_small={val_small.shape}", flush=True)

    params = dict(TABR_PRESET)
    params.update(batch_size=args.batch, n_epochs=1, context_size=args.context,
                  d_main=args.d_main, patience=1, random_state=1, n_cv=1, n_refit=0, verbosity=0,
                  memory_efficient=True, candidate_encoding_batch_size=None)

    from pytabkit import RealTabR_D_Classifier
    from pytabkit.models.nn_models.tabr import TabrLightning

    seen = {"n": 0}
    real_step = TabrLightning.training_step

    class Done(Exception):
        pass

    def step_then_stop(self, batch, batch_idx):
        # let the first `steps-1` steps run unprofiled, then profile the last one
        if seen["n"] < args.steps - 1:
            seen["n"] += 1
            return real_step(self, batch, batch_idx)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            out = real_step(self, batch, batch_idx)
        prof.export_chrome_trace(str(Path("reports") / "tabr_step_trace.json"))
        evs = prof.key_averages()
        rows = sorted(evs, key=lambda e: -e.self_device_time_total)[: args.top]
        print("\n=== top ops by CUDA self time (last profiled step) ===")
        print(f"{'op':<52}{'cuda_s':>10}{'calls':>8}{'cuda_mem_MB':>14}")
        for e in rows:
            if e.self_device_time_total <= 0:
                continue
            print(f"{e.key[:50]:<52}{e.self_device_time_total/1e6:>10.3f}"
                  f"{e.count:>8}{e.self_device_memory_usage/2**20:>14.1f}")
        tot = sum(e.self_device_time_total for e in evs) / 1e6
        print(f"\ntotal CUDA time in this step: {tot:.3f} s")
        raise Done()

    TabrLightning.training_step = step_then_stop
    set_seed(1)
    model = RealTabR_D_Classifier(device="cuda", **params)
    try:
        model.fit(d, y_int[inner_tr], X_val=val_small, y_val=y_int[inner_tr][:2048])
    except Done:
        print("\n(profiling complete)")
    except BaseException as exc:  # noqa: BLE001
        print(f"stopped: {type(exc).__name__}: {exc}")
    finally:
        TabrLightning.training_step = real_step
    print("wrote reports/tabr_step_trace.json")


if __name__ == "__main__":
    main()
