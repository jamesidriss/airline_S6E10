"""Endgame Champion-B screen: TabPFN-2.6 route member under the MATCHED champion
route protocol (only model_path differs).  Prereg 4ff41f64.

Matched protocol values are read from the CERTIFIED v3.5 route contract
(reports/sol_tabpfn35_route/route_f0.json), not guessed:
  n_estimators=1, fit_with_cache, int8 kv cache, autocast(bf16), icl_bf16,
  chunk_cells=262144, decoder_inplace_gelu, windows-MQA backend with
  reuse_query_output, batch_size=1024, gpu_memory_fraction=0.85,
  prefit reserve 4 GiB, prediction reserve 2 GiB, seed 1201.

Modes
  --probe        Step-0 feasibility probe: stratified 50k context, 4096 eval rows,
                 batch 2048.  Records time/VRAM only; NO scientific claim.
  --context N    Fit context bound (default: all FIT rows).  The only preregistered
                 amendment is context=100000 with batch 1024, committed before use.
  (default)      Full screen on primary fold 0, evaluated by the frozen rule.

Usage:
  ./.venv/Scripts/python.exe scripts/endgame/tabpfn26_screen.py --probe
  ./.venv/Scripts/python.exe scripts/endgame/tabpfn26_screen.py
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import (ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit,  # noqa: E402
                        load_cached_parquet, save_json)
from src.validation.folds import get_scheme  # noqa: E402
from src.submission.store import load_oof  # noqa: E402
from scripts.run_sol_tabpfn import create_model, frames  # noqa: E402

PREREG = "configs/phase20/endgame_tabpfn26_screen_prereg.json"
PREREG_SHA = "4ff41f646f4b166b7342b55eeaa2d49cc1b239b9eeb941b7471e0c8425b33631"
CKPT = ROOT / "artifacts/tabpfn26/tabpfn-v2.6-classifier-v2.6_default.ckpt"
CKPT_SHA = "0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a"
PROV = "research/raw/tabpfn26_provenance.json"
ROUTE_REF = "sol_route_aux10_primary_final"
CHAMPION = "v7_sol_tabpfn_internal2_aux10"
V7_F0_AUC = 0.9618278142504795
MIN_STANDALONE = 0.960457          # v6 route f0 0.961457 minus 1e-3 (prereg floor)
BOOT_B, BOOT_SEED = 2000, 7
PROMOTED_SPEARMAN = 0.995


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def matched_params(cats: list[int], ckpt: Path | None = None) -> dict:
    return dict(model_path=str((ckpt or CKPT).resolve()), n_estimators=1, auto_scale_n_estimators=False,
                random_state=1201, device="cuda", ignore_pretraining_limits=True,
                fit_mode="fit_with_cache", kv_cache_precision="int8", keep_cache_on_device=True,
                inference_precision="autocast", categorical_features_indices=cats,
                memory_saving_mode=True, n_preprocessing_jobs=1, tuning_config=None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="Step-0 feasibility probe (no scientific claim)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--context", type=int, default=None, help="fit-context bound (preregistered amendment)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--tag", default="tabpfn26_screen")
    ap.add_argument("--checkpoint", default=None, help="override checkpoint path (checkpoint-arm comparison)")
    ap.add_argument("--ckpt-sha", default=None, help="expected checkpoint sha256")
    ap.add_argument("--prereg", default=None, help="prereg path to verify before the run")
    ap.add_argument("--prereg-sha", default=None, help="expected prereg sha256")
    ap.add_argument("--revision", default=None, help="human-readable checkpoint revision (recorded only)")
    ap.add_argument("--family", default="TabPFN-2.6")
    args = ap.parse_args()

    prereg_path = args.prereg or PREREG
    prereg_sha = args.prereg_sha or PREREG_SHA
    ckpt = Path(args.checkpoint) if args.checkpoint else CKPT
    ckpt_sha = args.ckpt_sha or CKPT_SHA
    sha = hashlib.sha256((ROOT / prereg_path).read_bytes()).hexdigest()
    assert sha == prereg_sha, f"prereg sha mismatch {sha}"
    assert file_sha256(ckpt) == ckpt_sha, "checkpoint sha mismatch"

    root = ARTIFACTS / args.tag
    root.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_per_process_memory_fraction(0.85)
    from src.models.windows_attention import register, verify_gpu_equivalence

    gate = verify_gpu_equivalence(reuse_query_output=True)
    gate_bf16 = verify_gpu_equivalence(torch.bfloat16, reuse_query_output=True)
    register(reuse_query_output=True)

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds
    X, _, names, cats, maps = frames(tr, te, True)

    fi = np.flatnonzero(folds != args.fold)
    va = np.flatnonzero(folds == args.fold)
    assert len(fi) == len(y) - len(va) and not np.intersect1d(ids[fi], ids[va]).size

    contract = {
        "git": git_commit(), "tag": args.tag, "fold": args.fold, "mode": "probe" if args.probe else "screen",
        "model_family": args.family, "prereg": prereg_path, "prereg_sha256": sha,
        "checkpoint_sha256": ckpt_sha, "checkpoint_revision": args.revision or "24148873a9a5",
        "matched_source": "reports/sol_tabpfn35_route/route_f0.json (certified v3.5 route contract)",
        "amendment_a1": "configs/phase20/endgame_tabpfn26_screen_amendment_a1.json (icl_bf16 switch unavailable on TabPFNV2p6)",
        "params": matched_params(cats, ckpt), "seed": 1201,
        "icl_bf16": True, "chunk_cells": 262144, "decoder_inplace_gelu": True,
        "windows_mqa": True, "reuse_query_output": True, "batch": args.batch,
        "gpu_memory_fraction": 0.85, "backend_gate": {"fp16": gate, "bf16": gate_bf16},
        "fold_sha256": arr_sha256(folds), "fit_ids_sha256": arr_sha256(ids[fi]),
        "apply_ids_sha256": arr_sha256(ids[va]), "feature_names": names,
        "label_free_category_maps": maps, "feature_fit_sha256": arr_sha256(X[fi]),
        "feature_apply_sha256": arr_sha256(X[va]),
        "no_gradient_fitting": "pretrained inference only; no evaluation labels anywhere",
        "utc": datetime.now(timezone.utc).isoformat(),
    }

    # ---- probe / screen row selection ----
    if args.probe:
        rng = np.random.default_rng(1201)
        fit_idx = np.sort(rng.choice(fi, min(50000, len(fi)), replace=False))
        va_idx = va[:4096]
        batch = 2048
    else:
        fit_idx = fi if args.context is None else np.sort(
            np.random.default_rng(1201).choice(fi, min(args.context, len(fi)), replace=False))
        va_idx = va
        batch = args.batch
    contract.update(fit_rows=len(fit_idx), apply_rows=len(va_idx), context_bound=args.context)

    save_json({**contract, "status": "FITTING"}, root / "progress.json")
    start = time.monotonic()
    model = None
    try:
        params = matched_params(cats, ckpt)
        # Amendment A1 (prereg 4ff41f64 -> endgame_tabpfn26_screen_amendment_a1.json):
        # the v3.5-only icl_bf16 ICL switch does not exist on TabPFNV2p6 (introspected:
        # no enable_*/precision setter). Use the plain classifier path -- identical for
        # v3.5 except for that switch.  If a future checkpoint DOES expose it, the switch
        # is enabled and the deviation is recorded instead of silently taken.
        from tabpfn import TabPFNClassifier

        def _has_icl_bf16(path: str) -> bool:
            try:
                from tabpfn.base import initialize_tabpfn_model as _init

                ms, *_ = _init(path, "classifier", n_estimators_override=1,
                               devices=[torch.device("cpu")])
                return hasattr(ms[0], "enable_icl_bf16")
            except Exception:
                return False

        use_icl_bf16 = _has_icl_bf16(params["model_path"])
        if use_icl_bf16:
            model = create_model(params, True, 262144, None, None, True)
        else:
            model = TabPFNClassifier(**params)
        print(f"  model path={Path(params['model_path']).name} icl_bf16_available={use_icl_bf16}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        model.fit(X[fit_idx], y[fit_idx])
        fit_seconds = time.monotonic() - start
        gc.collect(); torch.cuda.empty_cache()
        parts = []
        for begin in range(0, len(va_idx), batch):
            stop = min(begin + batch, len(va_idx))
            parts.append(model.predict_proba(X[va_idx[begin:stop]])[:, 1])
            if stop == len(va_idx) or stop <= batch or stop // batch % 20 == 0:
                save_json({**contract, "status": "PREDICTING", "completed_rows": stop,
                           "fit_seconds": fit_seconds, "seconds": time.monotonic() - start},
                          root / "progress.json")
                print(f"  {stop}/{len(va_idx)} ({time.monotonic()-start:.0f}s)", flush=True)
        p = np.concatenate(parts).astype("float32")
        assert p.shape == (len(va_idx),) and np.isfinite(p).all()
        elapsed = time.monotonic() - start
        peak = torch.cuda.max_memory_allocated()
        np.save(root / "prediction.npy", p)
        np.save(root / "apply_ids.npy", ids[va_idx])

        if args.probe:
            rep = {**contract, "status": "PROBE_COMPLETE", "fit_seconds": fit_seconds,
                   "total_seconds": elapsed, "peak_gpu_gib": peak / 2**30,
                   "seconds_per_apply_row": elapsed / len(va_idx),
                   "scientific_claim": "NONE - resource feasibility only",
                   "prediction_sha256": arr_sha256(p)}
            save_json(rep, REPORTS / f"{args.tag}_probe.json")
            print(f"PROBE: fit {fit_seconds:.0f}s, total {elapsed:.0f}s, "
                  f"peak {peak/2**30:.2f} GiB, {elapsed/len(va_idx)*1000:.1f} ms/row")
            return 0

        # ---------------- scientific evaluation (frozen rule) ----------------
        from scipy.stats import spearmanr

        yv = y[va_idx]
        v7 = load_oof(CHAMPION)[va_idx]
        route = load_oof(ROUTE_REF)[va_idx]
        auc = float(roc_auc_score(yv, p))
        blend_v7 = 0.5 * (logit(p) + logit(v7))
        blend_route = 0.5 * (logit(p) + logit(route))
        port_v7 = float(roc_auc_score(yv, blend_v7)) - V7_F0_AUC
        port_route = float(roc_auc_score(yv, blend_route)) - float(roc_auc_score(yv, route))
        sp_v7 = float(spearmanr(p, v7).statistic)
        sp_route = float(spearmanr(p, route).statistic)
        pear_v7 = float(np.corrcoef(logit(p), logit(v7))[0, 1])

        rng = np.random.default_rng(BOOT_SEED)
        deltas = np.empty(BOOT_B)
        for b in range(BOOT_B):
            ix = rng.integers(0, len(yv), len(yv))
            deltas[b] = roc_auc_score(yv[ix], blend_v7[ix]) - roc_auc_score(yv[ix], v7[ix])
        ci_lo = float(np.quantile(deltas, 0.025))
        ci_hi = float(np.quantile(deltas, 0.975))

        promote = bool(ci_lo > 0 and sp_v7 < PROMOTED_SPEARMAN and auc >= MIN_STANDALONE)
        close = bool(ci_hi <= 0)
        decision = "PROMOTE_TO_5FOLD" if promote else ("CLOSE" if close else "AMBIGUOUS")

        rep = {
            **contract, "status": "SCREEN_COMPLETE", "fit_seconds": fit_seconds,
            "predict_seconds": elapsed - fit_seconds, "total_seconds": elapsed,
            "peak_gpu_gib": peak / 2**30, "prediction_sha256": arr_sha256(p),
            "metrics": {
                "standalone_auc": auc, "delta_vs_v7_standalone": auc - V7_F0_AUC,
                "port_delta_v7": port_v7, "port_delta_route_ref": port_route,
                "spearman_vs_v7": sp_v7, "spearman_vs_route_ref": sp_route,
                "logit_pearson_vs_v7": pear_v7,
                "paired_bootstrap": {"n": BOOT_B, "seed": BOOT_SEED, "ci_2.5": ci_lo, "ci_97.5": ci_hi,
                                     "p_delta_gt_0": float((deltas > 0).mean())},
            },
            "frozen_decision": {
                "rule": "PROMOTE iff ci_2.5 > 0 AND spearman_vs_v7 < 0.995 AND standalone >= 0.960457; "
                        "CLOSE iff ci_97.5 <= 0; else AMBIGUOUS",
                "decision": decision, "ci_2.5": ci_lo, "ci_97.5": ci_hi,
                "spearman_vs_v7": sp_v7, "standalone_auc": auc, "floor": MIN_STANDALONE,
            },
        }
        save_json(rep, REPORTS / f"{args.tag}_f{args.fold}.json")
        print(f"standalone {auc:.6f} | port_delta_v7 {port_v7:+.3e} | ci [{ci_lo:+.3e}, {ci_hi:+.3e}] "
              f"| spearman_v7 {sp_v7:.4f} -> {decision}")
        print(f"total {elapsed:.0f}s, peak {peak/2**30:.2f} GiB -> {REPORTS / f'{args.tag}_f{args.fold}.json'}")
        return 0
    except Exception as error:
        save_json({**contract, "status": "INVALID_RESOURCE_OR_IMPLEMENTATION", "error": repr(error),
                   "seconds": time.monotonic() - start}, REPORTS / f"{args.tag}_f{args.fold}.json")
        raise
    finally:
        if model is not None:
            try:
                del model
                gc.collect(); torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())