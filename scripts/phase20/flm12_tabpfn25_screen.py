"""Phase 20 -- FLM12 TabPFN-2.5 fold-0 screen (F2 discovery).

Screens the Prior-Labs/tabpfn_2_5 checkpoints (v2.5_real = the RealTabPFN
real-data-pretrained lineage; v2.5_default = descriptive mechanism arm) as
candidate ensemble members under the CHAMPION'S OWN fold-0 route protocol:
scripts/run_sol_tabpfn.py params verbatim (n_estimators=1, fit_with_cache,
int8 kv cache, autocast, ignore_pretraining_limits, categorical_features_indices
from frames()), ONLY the checkpoint differs (v2.5 vs champion v3.5).

Fit on a stratified 50,000-row context drawn from the primary FIT rows
(folds 1-4; prereg v2 - the full-FIT eager cache build is resource-infeasible
for v2.5 on 16 GB, v1 crash evidence), predict all fold-0 rows in batches of
512 (prereg v3 - v2.5 cache-mode predict materializes attention at batch x
context scale, v2 crash evidence), score against the hash-asserted champion
portfolio/route and the frozen-rule route_ref (sol_route_aux10_primary_final).

Checkpoints are vendored to artifacts/tabpfn25/ and sha256-asserted.

Frozen rule: PROMISING iff port_delta >= +5e-5 AND
Spearman(candidate, route_ref) < 0.995 (PRIMARY arm v2.5_real only).

Run:
  .\\.venv\\Scripts\\python.exe scripts/phase20/flm12_tabpfn25_screen.py --arm real
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, REPORTS, arr_sha256, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.phase19.foundation_screen import (  # noqa: E402
    CONTEXT_SEED,
    _cl,
    logit,
    screen_metrics,
    stratified_context,
)
from scripts.phase19.foundation_screen_fold0 import load_champion_fold0  # noqa: E402

OUT = REPORTS / "phase20" / "flm12_tabpfn25"
ART = ARTIFACTS / "phase20" / "flm12_tabpfn25"
VEND = ARTIFACTS / "tabpfn25"
PREREG = "configs/phase20/flm12_tabpfn25_prereg_v3.json"
# v3 = chunk-guard amendment of v2 (19ee7f70..., commit 3cd7d36); run v2 arm=real
# fit OK (50k context, 13.6s) but the first predict batch OOMed: v2.5 cache-mode
# predict materialized fp32 math-backend attention (50.42 GiB at 8192x50000;
# v2.5 attention not flash-eligible on this build). BATCH 8192 -> 512; chunking
# is exact in fit_with_cache mode (no test-test attention). See "delta_from_v2"
# inside the prereg file.
PREREG_SHA = "c85b24fa22e08801792104df4f200ba2de70e3dfdbe6e6c44b79d5de1cdfe905"  # pinned before the rerun (preregister-before-result)
ROUTE_REF = "sol_route_aux10_primary_final"
# v3: v2.5 cache-mode predict materializes attention at batch x context scale
# (50.42 GiB at 8192x50000; v2 crash evidence) -> small query batches. Chunking
# is exact in fit_with_cache mode (test rows are queries only).
BATCH = 512
# v2: stratified context bound (FLM9 precedent); full 559708-row fit is
# resource-infeasible for v2.5's per-column cache on 16 GB (v1 crash evidence).
CTX_ROWS = 50000

ARMS = {
    "real": {
        "file": "tabpfn-v2.5-classifier-v2.5_real.ckpt",
        "sha256": "b403ee3947766675a0a7c2fd9c31ee41a9c8d080e8b5e30bb31707ed744be267",
        "primary": True,
    },
    "default": {
        "file": "tabpfn-v2.5-classifier-v2.5_default.ckpt",
        "sha256": "5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12",
        "primary": False,
    },
}


def log(msg: str) -> None:
    print(f"[flm12] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    args = ap.parse_args()
    spec = ARMS[args.arm]

    sha = hashlib.sha256((ROOT / PREREG).read_bytes()).hexdigest()
    assert sha == PREREG_SHA, f"prereg sha mismatch {sha}"
    log(f"prereg sha256 {sha}")

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    VEND.mkdir(parents=True, exist_ok=True)

    # ---- vendor the checkpoint (revision-pinned download, sha-asserted) ----
    from huggingface_hub import hf_hub_download

    src = Path(hf_hub_download("Prior-Labs/tabpfn_2_5", spec["file"]))
    dst = VEND / spec["file"]
    if not dst.exists():
        shutil.copyfile(src, dst)
    got = hashlib.sha256(dst.read_bytes()).hexdigest()
    assert got == spec["sha256"], f"v2.5 checkpoint hash drift: {got}"
    log(f"checkpoint sha256 OK ({got[:16]}...) {spec['file']}")

    import torch
    from tabpfn import TabPFNClassifier
    import tabpfn

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds

    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)
    assert not np.intersect1d(FIT, EVAL).size

    route_all, port_all = load_champion_fold0(folds)

    from scripts.run_sol_tabpfn import frames

    Xall, _Xte, names, cats, maps = frames(tr, te, route=True)
    assert Xall.shape[0] == len(y), "feature/row misalignment"

    from src.submission.store import load_oof

    ref = load_oof(ROUTE_REF).astype(np.float64)
    assert np.isfinite(ref).all() and ref.min() >= 0.0 and ref.max() <= 1.0
    ref0 = ref[EVAL]
    route0, port0 = route_all[EVAL], port_all[EVAL]
    y_eval = y[EVAL]

    log(f"arm={args.arm} FIT={len(FIT)} EVAL={len(EVAL)} features={Xall.shape[1]} "
        f"cats={[int(c) for c in cats]}")

    # champion fold-0 route params verbatim; ONLY the checkpoint differs
    params = dict(model_path=str(dst.resolve()), n_estimators=1,
                  auto_scale_n_estimators=False, random_state=1201, device="cuda",
                  ignore_pretraining_limits=True, fit_mode="fit_with_cache",
                  kv_cache_precision="int8", keep_cache_on_device=True,
                  inference_precision="autocast",
                  categorical_features_indices=cats,
                  memory_saving_mode="auto", n_preprocessing_jobs=1, tuning_config=None)

    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    model = TabPFNClassifier(**params)
    # v2 (prereg flm12 v2): stratified context bound drawn from FIT ONLY; the
    # full-FIT eager cache build is resource-infeasible for v2.5 on 16 GB.
    ctx = stratified_context(FIT, y, CTX_ROWS, seed=CONTEXT_SEED)
    log(f"context: stratified {len(ctx)} rows from {len(FIT)} FIT rows "
        f"(seed={CONTEXT_SEED})")
    model.fit(Xall[ctx], y[ctx])
    fit_s = time.time() - t1
    log(f"fit done in {fit_s:.1f}s")

    t2 = time.time()
    chunks = []
    for begin in range(0, len(EVAL), BATCH):
        stop = min(begin + BATCH, len(EVAL))
        chunks.append(model.predict_proba(Xall[EVAL[begin:stop]])[:, 1])
        log(f"predicted {stop}/{len(EVAL)}")
    pred = np.concatenate(chunks).astype(np.float64)
    predict_s = time.time() - t2
    assert pred.shape == (len(EVAL),)
    assert np.isfinite(pred).all() and (pred >= 0).all() and (pred <= 1).all()
    log(f"predict done in {predict_s:.1f}s")

    m = screen_metrics(y_eval, pred, port0, route0)

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    auc_ref = float(roc_auc_score(y_eval, ref0))
    blend = 1.0 / (1.0 + np.exp(-(logit(_cl(pred)) + logit(_cl(ref0))) / 2.0))
    auc_blend_ref = float(roc_auc_score(y_eval, blend))
    port_delta = auc_blend_ref - auc_ref
    spearman = float(spearmanr(pred, ref0).statistic)
    prom = bool(spec["primary"] and port_delta >= 5e-5 and spearman < 0.995)

    pred_path = ART / f"tabpfn25_{args.arm}_fold0.npy"
    np.save(pred_path, pred.astype("float32"))

    meta = {
        "model": "tabpfn-2.5", "arm": args.arm, "primary_arm": bool(spec["primary"]),
        "status": "DONE", "utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "prereg": PREREG, "prereg_sha256": sha,
        "hf_repo": "Prior-Labs/tabpfn_2_5", "checkpoint": spec["file"],
        "checkpoint_sha256": got, "library_version": tabpfn.__version__,
        "recipe": "champion fold-0 route params verbatim (n_estimators=1, "
                  "fit_with_cache, int8 kv cache, autocast, "
                  "ignore_pretraining_limits, memory_saving_mode=auto); ONLY the "
                  "checkpoint differs from champion v3.5. v2: fit context = "
                  "stratified 50000-row subsample of FIT (prereg v2, v1 crash); "
                  "v3: predict batch 512 (prereg v3, v2 materialization crash)",
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in params.items()},
        "predict_batch": int(BATCH),
        "fit_source_rows": int(len(FIT)), "context_rows": int(len(ctx)),
        "context_seed": int(CONTEXT_SEED), "eval_fold": 0, "eval_rows": int(len(EVAL)),
        "fit_folds": [1, 2, 3, 4], "batch_size": BATCH,
        "features": list(names), "categorical_indices": [int(c) for c in cats],
        "label_free_category_maps": maps,
        "route_ref": ROUTE_REF, "route_ref_auc_fold0": auc_ref,
        "prediction_sha256": arr_sha256(np.load(pred_path)),
        "metrics": m,
        "frozen_rule": {
            "port_delta": port_delta, "route_ref_auc": auc_ref,
            "blend_auc": auc_blend_ref, "spearman_vs_route_ref": spearman,
            "promising": prom,
        },
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "fit_seconds": fit_s, "predict_seconds": predict_s,
        "total_seconds": time.time() - t0,
    }
    save_json(meta, OUT / f"tabpfn25_{args.arm}_fold0.json")

    fr = meta["frozen_rule"]
    log(f"standalone={m['standalone_auc']:.6f} (ref {auc_ref:.6f}) "
        f"blend={fr['blend_auc']:.6f} port_delta={port_delta:+.3e} "
        f"spearman={spearman:.4f} -> PROMISING={prom}")
    log(f"screen table: port_delta={m['portfolio_delta']:+.3e} "
        f"pearson_port={m['logit_pearson_vs_portfolio']:.4f}")
    log(f"peak vram {meta['peak_vram_gb']}GB; total {meta['total_seconds']:.1f}s")
    log(f"report -> {(OUT / f'tabpfn25_{args.arm}_fold0.json').relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
