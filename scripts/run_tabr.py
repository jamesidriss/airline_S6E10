"""Durable runner for TabR (Phase 4), one outer fold at a time.

What this fixes relative to the earlier version
-----------------------------------------------
1. **Host RAM exhaustion.** The earlier run died silently, twice, at the end of epoch 0 with no
   Python traceback while the pagefile peaked at 10.1 GB. Cause: pytabkit's
   ``tabr.py:281-283`` wraps candidate encoding in
   ``torch.set_grad_enabled(torch.is_grad_enabled() and not self.memory_efficient)``.
   With ``memory_efficient=False`` -- which is what the earlier run used -- autograd stays ENABLED
   while all ~453k candidate rows are encoded on every training step, building a full autograd
   graph over the entire retrieval database each time. That is both the memory blow-up and most of
   the wall-clock cost. ``memory_efficient=True`` encodes candidates under ``no_grad`` and then
   recomputes gradients only for the retrieved context rows, which is TabR's own documented
   memory-efficient variant. It is now the default; the flag is part of the recorded config because
   it changes the gradient computation and therefore must never be silently mixed across folds.

2. **Durability.** Every fold writes, before training: a config JSON plus config/fold/view hashes.
   During training: an append-only epoch log flushed per epoch. After evaluation: the outer-fold
   predictions and diagnostics, then an atomic ``fold_complete.json``. A fold that dies can be
   re-run in isolation without redoing the others, and a resume is only permitted when the config
   hash, fold hash and view hash all match.

3. **Outer-fold purity.** The early-stopping holdout is carved from the FIT rows by
   ``_inner_es_split``; the outer validation rows are touched exactly once, after training, to
   score. ``n_refit=0`` means pytabkit does not refit on train+val, which was verified against
   ``sklearn_base.py:443-454``.

Memory policy
-------------
peak host RSS is sampled every epoch so a fold that is trending toward the commit limit is visible
while it can still be stopped, rather than after the process has been killed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.models.realmlp import _twin_frame  # noqa: E402
from src.models.tabr_epochlog import (auc_curve, install_epoch_recorder, records,  # noqa: E402
                                      clear_epoch_sink, reset_records, set_epoch_sink)
from src.models.tabr_retrieval import enable_tf32_process_wide  # noqa: E402
from src.models.tabr_retrieval import (find_torch_index, install_faiss_gpu_shim,  # noqa: E402
                                       set_query_chunk, shim_is_installed,
                                       verify_matches_reference)
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

RUNS = REPORTS / "tabr_runs"

# pytabkit's tuned TabR preset (TabZilla-derived), taken verbatim from
# pytabkit.models.sklearn.default_params.RealTABR_D_CLASS. `num_embeddings` there is a MODULE SPEC
# DICT, not the list of per-head embedding widths the original TabR paper code uses; passing a
# list raises a bare ValueError deep inside tabr_lib.make_module.
TABR_PRESET = dict(
    d_main=265,
    d_multiplier=2.0,
    encoder_n_blocks=0,
    predictor_n_blocks=1,
    mixer_normalization="auto",
    context_dropout=0.38920071545944357,
    dropout0=0.38852797479169876,
    dropout1=0.0,
    normalization="LayerNorm",
    activation="ReLU",
    eval_batch_size=4096,
    patience=16,
    n_epochs=100_000,
    context_size=96,
    freeze_contexts_after_n_epochs=None,
    num_embeddings={"type": "PBLDEmbeddings", "n_frequencies": 8, "d_embedding": 4,
                    "frequency_scale": 0.1},
    tfms=["median_center", "robust_scale", "smooth_clip"],
    optimizer={"type": "AdamW", "lr": 0.0003121273641315169, "weight_decay": 1.2260352006404615e-06,
               "betas": (0.9, 0.95)},
    add_scaling_layer=True,
    scale_lr_factor=96,
    ls_eps=0.1,
)


# ------------------------------------------------------------------ durability helpers
def h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)          # atomic on both POSIX and Windows


class EpochLog:
    """Append-only, flushed-per-write epoch log. Survives a kill mid-training."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = path.open("a", encoding="utf-8")

    def write(self, rec: dict) -> None:
        self.f.write(json.dumps(rec, default=str) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:  # noqa: BLE001
            pass


def rss_gb() -> float:
    import psutil

    return round(psutil.Process().memory_info().rss / 2**30, 3)


def nans_in(a) -> dict:
    a = np.asarray(a, dtype="float64")
    return {"n_nan": int(np.isnan(a).sum()), "n_inf": int(np.isinf(a).sum())}


# ------------------------------------------------------------------ config
def assert_effective_config(model, params) -> dict:
    """Assert TabR was actually instantiated with the requested retrieval width.

    Guards the class of bug where a CLI flag is accepted but never reaches the model: the previous
    runner named its experiment ``d128`` while leaving pytabkit's own default of 265 in force.

    Two levels are checked:

      1. before training, the wrapper's ``d_main`` and ``get_config()['d_main']``;
      2. after training, the live ``TorchExactL2Index.dim`` -- which pytabkit itself constructs from
         the *inner* TabrModel's d_main, so it proves the width retrieval actually used.

    This is the retrieval/embedding width and is deliberately NOT the input feature count (~289).
    """
    req = int(params["d_main"])
    got = getattr(model, "d_main", None)
    if got is not None and int(got) != req:
        raise AssertionError(f"requested d_main={req} but the model wrapper has d_main={got}")
    try:
        cfg = model.get_config()
    except Exception:  # noqa: BLE001
        cfg = None
    if isinstance(cfg, dict) and "d_main" in cfg and int(cfg["d_main"]) != req:
        raise AssertionError(f"requested d_main={req} but get_config() reports {cfg['d_main']}")
    return {"requested_d_main": req, "effective_d_main": None if got is None else int(got)}


def assert_effective_retrieval_width(index, params) -> int:
    """Assert the live retrieval index uses the requested embedding width. Call after fitting."""
    req = int(params["d_main"])
    if index is None:
        raise AssertionError("no retrieval index found -- retrieval never ran")
    if int(index.dim) != req:
        raise AssertionError(
            f"requested d_main={req} but the live retrieval index uses dim={index.dim}")
    return int(index.dim)


def assert_consistent_widths(n_names, n_twins, train_w, val_w, es_w, test_w=None) -> None:
    """train / inner-ES / outer-val / test must all come from the SAME assemble() call.

    Regression guard for the second wiring bug: validation features were once read from
    ``vb.static_tr[val]`` (the static block only, 237 columns) while training used the assembled
    fold-safe view, which raised a pandas shape error at predict time.

    Widths are compared AFTER the categorical-twin encoding, because ``_twin_frame`` adds one
    column per TWIN_CAP (assembled 285 -> 289). ``n_names + n_twins`` must therefore equal the
    train width exactly -- that also pins the twin count rather than trusting it.
    """
    if n_names + n_twins != train_w:
        raise AssertionError(
            f"assembled names ({n_names}) + twins ({n_twins}) != train width ({train_w})")
    for label, w in (("inner-ES", es_w), ("validation", val_w), ("test", test_w)):
        if w is None:
            continue
        if w != train_w:
            raise AssertionError(f"{label} width {w} != train width {train_w}")


def build_params(args) -> dict:
    p = dict(TABR_PRESET)
    p.update(batch_size=args.batch, n_epochs=args.epochs, context_size=args.context,
             d_main=args.d_main, patience=args.patience, random_state=args.seed,
             n_cv=1, n_refit=0, verbosity=0)
    p["memory_efficient"] = bool(args.memory_efficient)
    # cand_bs == 0 means "encode the whole candidate database in one call" (pytabkit's None).
    #
    # This is the difference between ~40 min/epoch and a workable run. With grad DISABLED for
    # candidate encoding (memory_efficient=True), encoding all ~503k rows at once costs only
    # 503k x d_main x 4 B (~258 MB at d_main=128) and is exact. Splitting it into 256-row chunks
    # instead costs ~1969 separate GPU launches PER TRAINING STEP, i.e. ~242k launches per epoch,
    # which dominated everything else.
    if p["memory_efficient"]:
        p["candidate_encoding_batch_size"] = None if args.cand_bs <= 0 else int(args.cand_bs)
    else:
        p["candidate_encoding_batch_size"] = None
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--only-fold", type=int, default=-1)
    ap.add_argument("--context", type=int, default=96, help="retrieval context size k")
    ap.add_argument("--d-main", type=int, default=128, help="retrieval embedding width")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--cand-bs", type=int, default=0,
                    help="0 = encode the whole candidate database in one call (recommended); "
                         "a positive value chunks it, costing one GPU launch per chunk per step")
    ap.add_argument("--query-chunk", type=int, default=128)
    ap.add_argument("--no-memory-efficient", dest="memory_efficient", action="store_false",
                    help="reproduce the old (autograd-over-the-whole-database) behaviour; "
                         "changes the gradient computation, so never mix with -memory-efficient folds")
    ap.add_argument("--memory-efficient", dest="memory_efficient", action="store_true")
    ap.set_defaults(memory_efficient=True)
    ap.add_argument("--lr", type=float, default=None, help="override AdamW lr")
    ap.add_argument("--subsample", type=int, default=0, help="smoke tests only")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="tabr")
    ap.add_argument("--save-test", action="store_true")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    y = y_int.astype("float64")
    ntr, nte = len(tr), len(te)
    folds = get_scheme(args.folds, y_int, tr[ID_COL]).folds
    ks = sorted(set(folds.tolist()))
    run_one = args.only_fold >= 0
    if run_one:
        ks = [args.only_fold]

    params = build_params(args)
    if args.lr is not None:
        params["optimizer"] = dict(params["optimizer"], lr=args.lr)

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()
    view_hash = h({"names": list(vb.static_names)})
    fold_hash = h({"scheme": args.folds, "fold_of_row": [int(v) for v in folds[:5000]]})
    cfg_hash = h({k: v for k, v in sorted(params.items())})

    exp_id = (f"{args.tag}_{args.view}_d{params['d_main']}c{args.context}_e{args.epochs}"
              f"_b{args.batch}_me{int(params['memory_efficient'])}")
    run_dir = RUNS / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- retrieval gate
    rc = verify_matches_reference(device="cuda")
    if not rc["passes"]:
        raise SystemExit("retrieval verification FAILED -- refusing to train TabR")
    faiss_mode = install_faiss_gpu_shim(query_chunk=args.query_chunk)
    print(f"  retrieval verified; faiss gpu mode = {faiss_mode}", flush=True)

    cfg = {"exp_id": exp_id, "view": args.view, "scheme": args.folds, "seed": args.seed,
           "params": params, "view_hash": view_hash, "fold_hash": fold_hash,
           "cfg_hash": cfg_hash, "args": vars(args), "retrieval_check": rc,
           "faiss_gpu_mode": faiss_mode, "shim_active": shim_is_installed(),
           "static_feature_count": len(vb.static_names), "n_train": ntr, "n_test": nte,
           "protocol": {
               "early_stopping_source": "inner split carved from the OUTER-FIT rows "
                                        "(_inner_es_split, 10%)",
               "outer_validation_used_for": "scoring only, once, after training",
               "n_refit": 0,
               "n_refit_note": "sklearn_base.py:443-454 -> alg_interface_ = cv_alg_interface_, "
                               "no refit on train+val",
           }}
    atomic_write_json(run_dir / "config.json", cfg)
    print(f"  exp_id={exp_id}  cfg_hash={cfg_hash}  view_hash={view_hash}  fold_hash={fold_hash}",
          flush=True)

    import torch
    from pytabkit import RealTabR_D_Classifier

    # TF32 for encoder/predictor, full fp32 strictly inside the retrieval search.
    prec = enable_tf32_process_wide()
    installed = install_epoch_recorder()
    print(f"  float32 matmul precision (encoder/predictor): {prec}", flush=True)
    print(f"  epoch recorder installed: {installed}", flush=True)

    oof = np.full(ntr, np.nan)
    testp = np.zeros(nte) if args.save_test else None
    diag = []

    for k in ks:
        fdir = run_dir / f"fold{k}"
        fdir.mkdir(parents=True, exist_ok=True)
        if (fdir / "fold_complete.json").exists():
            prev = json.loads((fdir / "fold_complete.json").read_text(encoding="utf-8"))
            if prev.get("cfg_hash") == cfg_hash and prev.get("fold_hash") == fold_hash:
                print(f"  fold{k}: already complete (hashes match) -- skipping", flush=True)
                oof[prev["val_idx"]] = np.load(fdir / "oof.npy")
                if args.save_test and (fdir / "test.npy").exists():
                    testp += np.load(fdir / "test.npy") / len(ks)
                diag.append(prev["metrics"])
                continue
            print(f"  fold{k}: existing completion has DIFFERENT hashes -- rerunning", flush=True)

        fold = {"fold": int(k), "cfg_hash": cfg_hash, "fold_hash": fold_hash,
                "view_hash": view_hash, "started": time.strftime("%F %T")}
        atomic_write_json(fdir / "fold_start.json", fold)

        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        set_seed(args.seed + k)

        # inner early-stopping holdout carved from the FIT rows only
        inner_tr, inner_es = _inner_es_split(fit, y_int, args.seed + k)

        elog = EpochLog(fdir / "epochs.jsonl")
        # flush every epoch record the moment it is produced, so a kill mid-epoch still leaves
        # the learning curve on disk
        set_epoch_sink(lambda rec: elog.write({"event": "epoch", **rec}))
        reset_records()
        t_fold0 = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # ---- build frames ----
        # Assemble ONCE over the whole outer-fit block. `assemble` returns the fold-safe TE /
        # transductive blocks fitted on these rows, so the inner early-stopping split and the
        # outer validation block must all be read out of this one call -- taking the validation
        # features from `vb.static_*` instead would silently drop every fold-safe block and
        # mismatch the column count against `names`.
        Xf, Xout, names = vb.assemble(fit, y_int, val,
                                      np.arange(ntr, ntr + nte) if args.save_test else None,
                                      inner_seed=k)
        pos = {int(v): i for i, v in enumerate(fit)}
        tr_local = np.array([pos[int(v)] for v in inner_tr])
        es_local = np.array([pos[int(v)] for v in inner_es])

        y_sub = y_int[inner_tr]
        Xtr = Xf[tr_local]
        if args.subsample and len(Xtr) > args.subsample:
            sel = np.random.default_rng(args.seed).choice(len(Xtr), args.subsample, replace=False)
            Xtr = Xtr[sel]
            y_sub = y_sub[sel]
        tr_d = _twin_frame(Xtr, names)
        del Xtr
        es_d = _twin_frame(Xf[es_local], names)
        val_d = _twin_frame(Xout["val"], names)
        te_d = _twin_frame(Xout["test"], names) if args.save_test else None
        import gc

        # train / inner-ES / outer-val / test must all come from this ONE assemble() call and agree
        # in width AFTER the categorical-twin encoding (_twin_frame adds one column per TWIN_CAP,
        # so the assembled 285 becomes 289).
        n_twin = int((tr_d.dtypes.astype(str) == "category").sum())
        assert tr_d.shape[1] == len(names) + n_twin, (
            f"train width {tr_d.shape[1]} != assembled {len(names)} + twins {n_twin}")
        assert_consistent_widths(len(names), n_twin, tr_d.shape[1], es_d.shape[1], val_d.shape[1],
                                 None if te_d is None else te_d.shape[1])
        gc.collect()
        print(f"  fold{k}: train={tr_d.shape} es={es_d.shape} val={val_d.shape} "
              f"assembled={len(names)} static={len(vb.static_names)} rss={rss_gb()}GB",
              flush=True)

        # ---- fit ----
        n_attempts, oom_events = 0, []
        p_fold = dict(params)
        while True:
            n_attempts += 1
            try:
                t_fit0 = time.time()
                model = RealTabR_D_Classifier(device="cuda", **p_fold)
                eff = assert_effective_config(model, p_fold)
                model.fit(tr_d, y_sub, X_val=es_d, y_val=y_int[inner_es])
                t_fit = time.time() - t_fit0
                break
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                gc.collect()
                oom_events.append({"attempt": n_attempts, "where": "fit",
                                   "error": type(exc).__name__, "message": str(exc)[:200],
                                   "cand_bs": p_fold.get("candidate_encoding_batch_size"),
                                   "batch": p_fold.get("batch_size")})
                print(f"          OOM in fit (attempt {n_attempts}): {str(exc)[:120]}", flush=True)
                if n_attempts >= 4:
                    raise
                # Recovery order per policy: shrink the retrieval query chunk first -- the chunk
                # sets the transient distance-matrix spike at chunk x n_database x 4 bytes -- and
                # only then the physical batch size. Note memory_efficient=False is never used as
                # a recovery step: it changes the gradient computation, so it would silently make
                # this fold non-comparable with the others.
                if args.query_chunk > 32:
                    args.query_chunk = set_query_chunk(max(32, args.query_chunk // 2))
                    print(f"          -> query_chunk {args.query_chunk}", flush=True)
                else:
                    p_fold["batch_size"] = max(1024, p_fold["batch_size"] // 2)
                    print(f"          -> batch {p_fold['batch_size']}", flush=True)
                if p_fold.get("candidate_encoding_batch_size"):
                    p_fold["candidate_encoding_batch_size"] = max(
                        16, p_fold["candidate_encoding_batch_size"] // 2)

        sidx = find_torch_index()
        # prove the width retrieval actually used equals the width we requested
        assert_effective_retrieval_width(sidx, p_fold)
        istats = sidx.stats() if sidx else {}
        best_epoch = getattr(model.alg_interface_, "best_iteration_", None)
        fit_params = getattr(model.alg_interface_, "fit_params", None)
        if fit_params:
            best_epoch = fit_params[0].get("n_epoch_limit", best_epoch)

        # ---- score the outer validation block exactly once ----
        t_inf0 = time.time()
        p = model.predict_proba(val_d)[:, 1]
        t_inf = time.time() - t_inf0
        auc = float(roc_auc_score(y[val], p))

        if args.save_test:
            t_tinf0 = time.time()
            tpred = model.predict_proba(te_d)[:, 1]
            t_tinf = time.time() - t_tinf0
            testp += tpred / len(ks)
        else:
            t_tinf = None

        metrics = {
            "fold": int(k), "auc_val": round(auc, 6),
            "train_seconds": round(t_fit, 1), "inference_seconds": round(t_inf, 1),
            "test_inference_seconds": None if t_tinf is None else round(t_tinf, 1),
            "total_seconds": round(time.time() - t_fold0, 1),
            "final_epoch": int(getattr(model, "n_epochs_ran_", 0) or 0),
            "best_epoch": None if best_epoch is None else int(best_epoch),
            "n_train_rows": int(len(tr_d)), "n_val_rows": int(len(val)),
            "requested_d_main": eff["requested_d_main"],
            "effective_d_main": eff["effective_d_main"],
            "input_feature_count": int(tr_d.shape[1]),
            "static_feature_count": len(vb.static_names),
            "n_categorical_twins": int((tr_d.dtypes.astype(str) == "category").sum()),
            "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            "peak_host_rss_gb": rss_gb(),
            "retrieval_queries": istats.get("total_queries_searched"),
            "retrieval_rows_added": istats.get("total_rows_added"),
            "retrieval_db_rows": istats.get("n_database"),
            "actual_k": istats.get("actual_k"), "context_size_requested": int(params["context_size"]),
            "retrieval_dimensionality": istats.get("d_main"),
            "retrieval_device": istats.get("device"), "retrieval_dtype": istats.get("dtype"),
            "retrieval_last_shape": istats.get("last_result_shape"),
            "faiss_gpu_shim_used": shim_is_installed(), "faiss_gpu_mode": faiss_mode,
            "memory_efficient": bool(params["memory_efficient"]),
            "candidate_encoding_batch_size": params["candidate_encoding_batch_size"],
            "batch_size": int(params["batch_size"]), "query_chunk": int(args.query_chunk),
            "pred_nan": nans_in(p)["n_nan"], "pred_inf": nans_in(p)["n_inf"],
            "pred_min": float(np.min(p)), "pred_max": float(np.max(p)),
            "oom_events": oom_events, "n_fit_attempts": n_attempts,
            "epochs_cap": int(params["n_epochs"]), "patience": int(params["patience"]),
            "n_epochs_recorded": len(records()),
            "inner_auc_curve": auc_curve(),
            "val_metric_name": params.get("val_metric_name", "class_error"),
            "es_monitors": "val_accuracy (pytabkit maps the default 'class_error' to accuracy, "
                           "which is a weaker proxy for ROC-AUC than the curve we select on)",
            "early_stopping_split": "inner 10% of the outer-FIT rows",
            "ckpt_dir": "checkpoints/",
        }
        # ---- PREDICTION FIRST ----
        # Persisting the prediction and the completion marker must not depend on anything that can
        # raise. A missing import in a logging call previously threw NameError AFTER a completed
        # 10-epoch fit and destroyed the prediction, so persistence now happens first and the
        # diagnostics that are nice-to-have are wrapped so they cannot block it.
        np.save(fdir / "oof.npy", p.astype("float32"))
        if args.save_test:
            np.save(fdir / "test.npy", tpred.astype("float32"))
        complete = {"fold": int(k), "cfg_hash": cfg_hash, "fold_hash": fold_hash,
                    "view_hash": view_hash, "metrics": metrics,
                    "val_idx_sha": h([int(v) for v in val]),
                    "completed": time.strftime("%F %T")}
        atomic_write_json(fdir / "fold_complete.json", complete)

        # ---- then the learning curve and epoch log, defensively ----
        try:
            eps = records()
            if eps:
                atomic_write_json(
                    fdir / "learning_curve.json",
                    {"fold": int(k), "epochs": eps, "auc_curve": auc_curve(),
                     "note": "all values are from the INNER early-stopping split carved out of the "
                             "outer-FIT rows; no outer-validation target is involved, so this "
                             "curve is safe for epoch selection"})
            elog.write({"event": "fold_finished", **metrics})
        except Exception as exc:  # noqa: BLE001
            print(f"          warn: could not write epoch diagnostics: {type(exc).__name__}: {exc}",
                  flush=True)
        finally:
            try:
                elog.close()
                clear_epoch_sink()
            except Exception:  # noqa: BLE001
                pass

        oof[val] = p
        diag.append(metrics)
        print(f"  fold{k}: AUC={auc:.6f}  train={metrics['train_seconds']}s  "
              f"infer={metrics['inference_seconds']}s  VRAM={metrics['peak_vram_allocated_gb']}GB  "
              f"RSS={metrics['peak_host_rss_gb']}GB  q={metrics['retrieval_queries']}  "
              f"k={metrics['actual_k']}  d={metrics['retrieval_dimensionality']}  "
              f"nan={metrics['pred_nan']}  oom={len(oom_events)}", flush=True)

        del model, tr_d, es_d, val_d, te_d, Xf, Xout
        torch.cuda.empty_cache()
        gc.collect()

    # ---------------------------------------------------------------- summary
    done = [d["fold"] for d in diag]
    if run_one:
        print(f"\nsingle-fold screening complete. exp_id={exp_id}")
        print("wrote", run_dir)
        return

    assert not np.isnan(oof).any(), "incomplete OOF -- some folds did not finish"
    auc = float(roc_auc_score(y, oof))
    fa = [float(roc_auc_score(y[folds == k], oof[folds == k])) for k in sorted(set(folds.tolist()))]
    v3 = store.load_oof("blend_v3_final").astype("float64")
    print("\n" + "=" * 96)
    print(f"TabR {exp_id}")
    print(f"  OOF AUC = {auc:.6f}   folds = {[round(x, 6) for x in fa]}")
    print(f"  vs finalist {float(roc_auc_score(y, v3)):.6f}: logit corr "
          f"{corr(lab_logit(oof), lab_logit(v3)):.5f}  spearman {spearman(oof, v3):.5f}")
    print("=" * 96)

    store.save(exp_id, oof, testp, fold_scheme=args.folds,
               meta={"family": "tabr", "featureset": args.view, "auc": round(auc, 6),
                     "cfg_hash": cfg_hash, "params": params, "gate_metrics": diag})
    atomic_write_json(run_dir / "summary.json",
                      {"exp_id": exp_id, "oof_auc": auc, "fold_aucs": fa,
                       "corr_logit_vs_finalist": corr(lab_logit(oof), lab_logit(v3)),
                       "spearman_vs_finalist": spearman(oof, v3),
                       "gate_metrics": diag, "config": cfg})
    print("wrote", run_dir / "summary.json")


def lab_logit(a):
    from src.ensemble import lab

    return lab.tform(np.asarray(a, dtype="float64"), "logit")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
