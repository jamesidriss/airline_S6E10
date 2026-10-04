"""Time-boxed profile of one TabR training step, to locate the real hotspot.

Motivation
----------
Three successive hypotheses about why an epoch took >60 min were each WRONG when measured:
  * candidate-encoding chunking (~242k launches/epoch) -- real overhead, but small
  * process-wide fp32 forcing the encoder off TF32   -- real, but small
  * torch.isin over the candidate pool each step     -- measured at 6.2 s/epoch, not 60 min

So the remaining ~545x is unaccounted for. Rather than reason further, this runs a handful of REAL
training steps on the real 503k-row assembled view and splits the time three ways:

    t_getxy      gathering the candidate pool (tabr.py:474 get_Xy)
    t_model      the TabrModel forward: encode candidates, search, predictor (tabr.py:265)
    t_step       everything else in training_step (logging, metric updates, loss)

and, on top of that, reports where the model's own time goes via torch.profiler so the culprit is
named rather than guessed.

Runs a bounded number of steps and exits; it never touches predictions or the prediction store.

Usage:
  python scripts/profile_tabr_step.py --steps 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

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
    ap.add_argument("--cand-bs", type=int, default=0)
    ap.add_argument("--query-chunk", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--profile-last", type=int, default=1)
    args = ap.parse_args()

    rc = verify_matches_reference(device="cuda")
    if not rc["passes"]:
        raise SystemExit("retrieval verification failed")
    install_faiss_gpu_shim(query_chunk=args.query_chunk)
    enable_tf32_process_wide()

    tr, te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y_int, tr[ID_COL]).folds
    fit = np.where(folds != args.fold)[0]
    inner_tr, _es = _inner_es_split(fit, y_int, 1 + args.fold)

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()
    t0 = time.time()
    Xf, _Xout, names = vb.assemble(inner_tr, y_int, _es, None, inner_seed=args.fold)
    print(f"assemble: {time.time()-t0:.1f}s  shape={Xf.shape}", flush=True)
    d = _twin_frame(Xf, names)
    print(f"twin frame: {time.time()-t0:.1f}s  shape={d.shape}", flush=True)

    params = dict(TABR_PRESET)
    params.update(batch_size=args.batch, n_epochs=1, context_size=args.context,
                  d_main=args.d_main, patience=1, random_state=1, n_cv=1, n_refit=0, verbosity=0,
                  memory_efficient=True,
                  candidate_encoding_batch_size=None if args.cand_bs <= 0 else args.cand_bs)

    from pytabkit import RealTabR_D_Classifier
    from pytabkit.models.nn_models.tabr import TabrLightning

    timings = {"getxy": 0.0, "model": 0.0, "rest": 0.0}
    n_calls = {"n": 0}
    orig_getxy = TabrLightning.get_Xy
    orig_forward = TabrModel_forward = None

    def timed_getxy(self, part, idx):
        torch.cuda.synchronize()
        s = time.time()
        out = orig_getxy(self, part, idx)
        torch.cuda.synchronize()
        timings["getxy"] += time.time() - s
        return out

    TabrLightning.get_Xy = timed_getxy

    model = RealTabR_D_Classifier(device="cuda", **params)
    # reach the inner TabrModel to time its forward
    inner = None
    try:
        inner = model.alg_interface_.cv_alg_interface_ if hasattr(model, "alg_interface_") else None
    except Exception:  # noqa: BLE001
        pass

    # fit one epoch, then time N fresh steps by re-entering training_step on random batches
    set_seed(0)
    train_ds = None

    print("\n--- driving real training steps ---", flush=True)

    # Faithful path: run fit with n_epochs=1 under timing hooks on get_Xy, the TabrModel forward
    # and training_step.
    holder = {}

    def make_timed_forward(real_forward):
        def fwd(self, **kw):
            torch.cuda.synchronize()
            s = time.time()
            out = real_forward(self, **kw)
            torch.cuda.synchronize()
            timings["model"] += time.time() - s
            n_calls["n"] += 1
            if n_calls["n"] == args.steps:
                holder["steps"] = n_calls["n"]
            return out
        return fwd

    import pytabkit.models.nn_models.tabr as tabr_mod
    real_fwd = tabr_mod.TabrModel.forward
    tabr_mod.TabrModel.forward = make_timed_forward(real_fwd)

    real_step = TabrLightning.training_step
    step_times = []

    def timed_step(self, batch, batch_idx):
        torch.cuda.synchronize()
        s = time.time()
        out = real_step(self, batch, batch_idx)
        torch.cuda.synchronize()
        step_times.append(time.time() - s)
        if len(step_times) >= args.steps:
            raise KeyboardInterrupt("enough steps measured")
        return out

    TabrLightning.training_step = timed_step

    set_seed(1)
    try:
        model.fit(d, y_int[inner_tr], X_val=_twin_frame(Xf[:2000], names),
                  y_val=y_int[inner_tr][:2000])
    except BaseException as exc:  # noqa: BLE001
        print(f"stopped early: {type(exc).__name__}: {exc}")
    finally:
        TabrLightning.training_step = real_step
        tabr_mod.TabrModel.forward = real_fwd
        TabrLightning.get_Xy = orig_getxy

    n = max(len(step_times), 1)
    print("\n=== per-step breakdown (seconds) ===")  # always report, even on early stop
    print(f"  steps measured        : {len(step_times)}")
    print(f"  get_Xy (candidate gather) : {timings['getxy']/n:.3f}")
    print(f"  TabrModel.forward         : {timings['model']/n:.3f}")
    est_rest = (sum(step_times) - timings["getxy"] - timings["model"]) / n
    print(f"  rest of training_step     : {est_rest:.3f}")
    if step_times:
        print(f"  TOTAL per step        : {sum(step_times)/len(step_times):.3f}")
        n_steps_epoch = int(np.ceil(len(inner_tr) / args.batch))
        print(f"  -> implied epoch time : {sum(step_times)/len(step_times)*n_steps_epoch/60:.1f} min "
              f"({n_steps_epoch} steps)")


if __name__ == "__main__":
    main()
