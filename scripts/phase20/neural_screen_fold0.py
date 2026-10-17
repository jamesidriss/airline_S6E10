"""Phase 20 -- neural fold-0 screen (P2 block, discovery F2).

Screens canonical neural architectures (FT-Transformer attention-over-features,
DCNv2 explicit-crosses) as candidate 11th ensemble members under the SAME
phase-19 fold-0 protocol: the champion's raw21+coarse10 route representation
(built by frames()), train on ALL FIT rows (folds 1-4) with an inner 10% ES
split carved from FIT ONLY (never the eval fold), predict fold-0 rows, and
score against the hash-asserted champion portfolio/route.

Anti-leakage: train on FIT only; inner ES carved from FIT only; scaler fit on
FIT-train only; no target encoding (route view is label-free); champion
reference hash-asserted from phase17 artifacts.

This is a SCREEN, not a promotion: it cannot pass the OOF admission gate. It
decides whether an architecture merits a full 5-fold OOF + admission-gate run.

Run (certified .venv, GPU):
  .\\.venv\\Scripts\\python.exe scripts/phase20/neural_screen_fold0.py --model fttransformer
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, REPORTS, arr_sha256, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.phase19.foundation_screen import screen_metrics  # noqa: E402
from scripts.phase19.foundation_screen_fold0 import load_champion_fold0  # noqa: E402

OUT = REPORTS / "phase20" / "neural_screen"
ART = ARTIFACTS / "phase20" / "neural_screen"
PREREG = ROOT / "configs" / "phase20" / "neural_screen_prereg.json"
ARCH_MODULE = "src/phase20/neural_arch.py"


def log(msg: str) -> None:
    print(f"[neural] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_seed(seed)


def torch_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inner_es_split(fit_idx: np.ndarray, y: np.ndarray, frac: float, seed: int):
    """Stratified 10% of FIT as inner-val (seeded); returns (train_idx, val_idx).

    Fold-safe: both are subsets of fit_idx (never the eval fold).
    """
    rng = np.random.default_rng(seed)
    yf = y[fit_idx]
    val_parts = []
    for cls in (0, 1):
        cls_idx = fit_idx[yf == cls]
        n_val = max(1, int(round(frac * len(cls_idx))))
        val_parts.append(rng.choice(cls_idx, n_val, replace=False))
    val = np.sort(np.concatenate(val_parts))
    train = np.setdiff1d(fit_idx, val, assume_unique=False)
    assert np.isin(train, fit_idx).all() and np.isin(val, fit_idx).all()
    assert not np.intersect1d(train, val).size
    return train, val


def _fit_scaler(Xtr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean/unit-var scaler fit on FIT-train only (fold-safe)."""
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def train_and_predict(model, Xtr, ytr, Xval, yval, Xeval, cfg, device, seed):
    """AdamW + warmup/cosine + inner-val early stopping. Returns eval logits."""
    import torch
    import torch.nn as nn

    torch_seed(seed)
    dev = torch.device(device)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg.get("weight_decay", 0.0))
    crit = nn.BCEWithLogitsLoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xval_t = torch.tensor(Xval, dtype=torch.float32, device=dev)
    yval_np = np.asarray(yval, dtype=np.float64)
    Xeval_t = torch.tensor(Xeval, dtype=torch.float32, device=dev)

    n = Xtr_t.shape[0]
    bs = int(cfg["batch_size"])
    max_epochs = int(cfg["max_epochs"])
    patience = int(cfg["es_patience"])
    warmup = 5

    from sklearn.metrics import roc_auc_score

    best_val = -1.0
    best_state = None
    since_best = 0
    for epoch in range(max_epochs):
        # linear warmup then cosine decay
        if epoch < warmup:
            lr_scale = (epoch + 1) / warmup
        else:
            t = (epoch - warmup) / max(1, max_epochs - warmup)
            lr_scale = 0.5 * (1.0 + np.cos(np.pi * t))
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * lr_scale
        model.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                logits = model(Xtr_t[idx])
                loss = crit(logits, ytr_t[idx])
            loss.backward()
            opt.step()
        # inner-val AUC (never the eval fold)
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast(dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                vlog = model(Xval_t).float().cpu().numpy()
        v_auc = roc_auc_score(yval_np, vlog)
        if v_auc > best_val + 1e-6:
            best_val = v_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = []
        for s in range(0, Xeval_t.shape[0], 65536):
            chunk = Xeval_t[s:s + 65536]
            with torch.amp.autocast(dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                out.append(model(chunk).float().cpu().numpy())
    return np.concatenate(out), best_val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fttransformer",
                    choices=["fttransformer", "dcnv2", "plrmlp", "tabresnet", "moe",
                             "interactionnet"])
    ap.add_argument("--eval-rows", type=int, default=0,  # 0 = all fold-0 rows
                    help="0 uses every fold-0 row (139927); >0 subsamples first N")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    cfg = prereg["architectures"][args.model].copy()
    recipe = cfg.pop("recipe", "")
    # Split the frozen config into architecture kwargs (-> build) and training kwargs (-> loop).
    TRAIN_KEYS = {"lr", "weight_decay", "batch_size", "max_epochs", "es_patience"}
    arch_cfg = {k: v for k, v in cfg.items() if k not in TRAIN_KEYS}
    train_cfg = {k: v for k, v in cfg.items() if k in TRAIN_KEYS}
    tag = args.tag or f"{args.model}_fold0"

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    import torch

    from src.phase20 import neural_arch
    from scripts.run_sol_tabpfn import frames

    set_seed(args.seed)

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds

    Xtr_all, _Xte, names, cats, maps = frames(tr, te, route=True)
    assert Xtr_all.shape[0] == len(y), "feature/row misalignment"

    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)
    assert not np.intersect1d(FIT, EVAL).size

    route_all, port_all = load_champion_fold0(folds)
    eval_idx = EVAL if (not args.eval_rows or args.eval_rows >= len(EVAL)) else EVAL[:args.eval_rows]
    route0, port0 = route_all[eval_idx], port_all[eval_idx]
    y_eval = y[eval_idx]

    train_idx, val_idx = inner_es_split(FIT, y, frac=0.1, seed=args.seed)
    mu, sd = _fit_scaler(Xtr_all[train_idx])
    Xtr_s = (Xtr_all[train_idx] - mu) / sd
    Xval_s = (Xtr_all[val_idx] - mu) / sd
    Xeval_s = (Xtr_all[eval_idx] - mu) / sd
    ytr = y[train_idx].astype(np.float64)
    yval = y[val_idx].astype(np.float64)

    log(f"rows={len(y)} FIT={len(FIT)} train={len(train_idx)} val={len(val_idx)} "
        f"EVAL={len(eval_idx)} features={Xtr_all.shape[1]} model={args.model} "
        f"device={args.device} cfg={cfg}")

    pred_path = ART / f"{tag}.npy"
    meta_path = OUT / f"{tag}.json"
    if pred_path.exists() and meta_path.exists():
        saved = np.load(pred_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") == "DONE" and arr_sha256(saved) == meta.get("prediction_sha256") \
                and meta.get("eval_rows") == int(len(eval_idx)):
            log("verified prediction exists; reusing")
            m = meta["metrics"]
            log(f"AUC={m['standalone_auc']:.6f} port_delta={m['portfolio_delta']:+.3e}")
            return 0

    model = neural_arch.build(args.model, Xtr_all.shape[1], **arch_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"training {args.model} ({n_params:,} params) on {len(train_idx)} -> predict {len(eval_idx)}")
    t1 = time.time()
    try:
        logits, best_val = train_and_predict(model, Xtr_s, ytr, Xval_s, yval, Xeval_s,
                                             train_cfg, args.device, args.seed)
        pred = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        np.save(pred_path, pred.astype("float32"))
        m = screen_metrics(y_eval, pred, port0, route0)
        status = "DONE"
        err = None
    except Exception as exc:
        pred = None
        m = None
        best_val = None
        status = "FAILED"
        err = repr(exc)
    secs = time.time() - t1

    meta = {
        "model": args.model, "arch_module": ARCH_MODULE, "status": status, "error": err,
        "recipe": recipe, "arch_cfg": arch_cfg, "train_cfg": train_cfg,
        "utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "eval_rows": int(len(eval_idx)), "eval_fold": 0, "fit_folds": [1, 2, 3, 4],
        "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx)),
        "inner_es_seed": args.seed, "scaler": "zero-mean/unit-var fit on FIT-train only",
        "device": args.device, "seed": args.seed, "tag": tag, "n_params": int(n_params),
        "best_inner_val_auc": None if best_val is None else float(best_val),
        "features": list(names), "categorical_indices": list(cats),
        "protocol": "train on FIT, inner 10% ES from FIT only, predict fold-0; champion hash-asserted",
        "prediction_sha256": arr_sha256(np.load(pred_path)) if pred is not None else None,
        "metrics": m, "fit_predict_seconds": secs,
        "prereg": {"path": PREREG.relative_to(ROOT).as_posix()},
    }
    save_json(meta, meta_path)
    if m is not None:
        log(f"AUC={m['standalone_auc']:.6f} port_delta={m['portfolio_delta']:+.3e} "
            f"corr_port={m['logit_pearson_vs_portfolio']:.4f} best_val={best_val:.4f} "
            f"({secs:.1f}s)")
    else:
        log(f"FAILED: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
