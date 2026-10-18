"""Phase 20 -- OBJ37-41 neural training-objective batch (F2 discovery).

Holds architecture + config + seed FIXED (canonical FT-Transformer, frozen
batch-1 hyperparameters, seed 1) and varies ONLY the training objective /
sampling, so any fold-0 delta is attributable to the objective alone:

  bce_control    plain BCE through the new code path (parity arm: must
                 reproduce committed control AUC 0.9576563038026185 within
                 1e-4 or the batch is void)
  supcon         BCE + 0.5*SupCon on a 128-d projection of the CLS embedding
                 (temperature 0.1, FIT-train labels only)          [OBJ37]
  auc_surrogate  pure pairwise RankNet surrogate over 4096 sampled
                 (pos,neg) FIT-train pairs per step; no BCE anchor  [OBJ38]
  hard_mining    dynamic hard-example mining: per-sample BCE EMA hardness
                 on FIT-train only; batch = 50% current top-20% hardest
                 + 50% uniform (re-estimated every epoch)           [OBJ39]
  gce            generalized cross-entropy noise-robust loss, p=0.7  [OBJ40]
  ema            plain BCE; weight EMA (decay 0.999) used for inner-val
                 evaluation AND final prediction                    [OBJ41]

Anti-leakage: train on FIT only; inner 10% ES from FIT only; supcon/hard
use FIT-TRAIN labels only (never val/eval); route view label-free; scaler
fit on FIT-train only; champion references hash-asserted.

Frozen rule: PROMISING iff port_delta >= +5e-5 AND
Spearman(candidate, route_ref) < 0.995.

Run (one arm per invocation):
  .\\.venv\\Scripts\\python.exe scripts/phase20/obj37_41_objectives.py --arm bce_control
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, REPORTS, arr_sha256, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.phase19.foundation_screen import _cl, logit, screen_metrics  # noqa: E402
from scripts.phase19.foundation_screen_fold0 import load_champion_fold0  # noqa: E402

OUT = REPORTS / "phase20" / "obj37_41_objectives"
ART = ARTIFACTS / "phase20" / "obj37_41_objectives"
PREREG = "configs/phase20/obj37_41_objectives_prereg.json"
PREREG_SHA = "94561fcea7ebb4315aac24a9928240ef6d464be9c13a4d35930be92c83a112b2"  # pinned before the first run (preregister-before-result)
ROUTE_REF = "sol_route_aux10_primary_final"
CONTROL_AUC_COMMITTED = 0.9576563038026185  # batch-1 fttransformer_fold0

# frozen batch-1 FT-Transformer config
ARCH_CFG = {"d_token": 192, "n_layers": 3, "n_heads": 8, "dropout": 0.1,
            "head_hidden": 192}
TRAIN_CFG = {"lr": 1e-3, "weight_decay": 1e-5, "batch_size": 4096,
             "max_epochs": 100, "es_patience": 15}
ARMS = ("bce_control", "supcon", "auc_surrogate", "hard_mining", "gce", "ema")
INFER_CHUNK = 65536
SEED = 1


def log(msg: str) -> None:
    print(f"[obj37-41] {msg}", flush=True)


def torch_seed(seed: int) -> None:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inner_es_split(fit_idx: np.ndarray, y: np.ndarray, frac: float, seed: int):
    """Byte-faithful copy of batch-1's inner_es_split (neural_screen_fold0.py).

    Fold-safe: both returns are subsets of fit_idx (never the eval fold).
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


def fit_scaler(Xtr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    arm = args.arm

    sha = hashlib.sha256((ROOT / PREREG).read_bytes()).hexdigest()
    assert sha == PREREG_SHA, f"prereg sha mismatch {sha}"
    log(f"prereg sha256 {sha}")

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn as nn

    from src.phase20 import neural_arch

    import random

    random.seed(SEED)        # byte-faithful to batch-1's set_seed(args.seed)
    np.random.seed(SEED)
    torch_seed(SEED)

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
    X_fit = Xall[FIT]
    X_eval_all = Xall[EVAL]
    n_feat = X_fit.shape[1]

    from src.submission.store import load_oof

    ref = load_oof(ROUTE_REF).astype(np.float64)
    assert np.isfinite(ref).all() and ref.min() >= 0.0 and ref.max() <= 1.0
    ref0 = ref[EVAL]
    route0, port0 = route_all[EVAL], port_all[EVAL]
    y_eval = y[EVAL]

    fit_pos = np.arange(len(FIT))
    train_pos, val_pos = inner_es_split(fit_pos, y[FIT], frac=0.1, seed=SEED)
    y_fit = y[FIT].astype(np.float64)
    mu, sd = fit_scaler(X_fit[train_pos])
    Xtr_s = (X_fit[train_pos] - mu) / sd
    Xval_s = (X_fit[val_pos] - mu) / sd
    Xeval_s = (X_eval_all - mu) / sd
    ytr = y_fit[train_pos]
    yval = y_fit[val_pos]

    log(f"arm={arm} FIT={len(FIT)} train={len(train_pos)} val={len(val_pos)} "
        f"EVAL={len(EVAL)} features={n_feat} device={args.device}")

    dev = torch.device(args.device)

    # ---- model: identical init to batch-1; supcon arm needs the CLS embedding ----
    if arm == "supcon":
        class FTTransformerEmb(neural_arch.FTTransformer):
            """Same modules/init; forward returns (logit, CLS embedding)."""

            def forward(self, x):  # type: ignore[override]
                bsz = x.shape[0]
                tok = x.unsqueeze(-1) * self.W + self.b
                cls = self.cls.expand(bsz, -1, -1)
                h = self.encoder(torch.cat([cls, tok], dim=1))
                h = self.norm(h[:, 0])
                return self.head(h).squeeze(-1), h

        model = FTTransformerEmb(n_feat, **ARCH_CFG)
        proj = nn.Linear(ARCH_CFG["d_token"], 128).to(dev)
    else:
        model = neural_arch.build("fttransformer", n_feat, **ARCH_CFG)
        proj = None
    n_params = sum(p.numel() for p in model.parameters())
    log(f"built fttransformer ({n_params:,} params) for arm {arm}")

    model = model.to(dev)
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, lr=TRAIN_CFG["lr"],
                            weight_decay=TRAIN_CFG["weight_decay"])

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xval_t = torch.tensor(Xval_s, dtype=torch.float32, device=dev)
    yval_np = np.asarray(yval, dtype=np.float64)
    Xeval_t = torch.tensor(Xeval_s, dtype=torch.float32, device=dev)

    n = Xtr_t.shape[0]
    bs = TRAIN_CFG["batch_size"]
    max_epochs = TRAIN_CFG["max_epochs"]
    patience = TRAIN_CFG["es_patience"]
    warmup = 5

    from sklearn.metrics import roc_auc_score

    # hardness state (hard_mining): EMA of per-sample BCE on FIT-TRAIN only.
    # numpy on CPU: deterministic (unique-index writes) and cheap (k/topk per epoch).
    hardness = np.zeros(n, dtype=np.float32)
    HARD_ALPHA = 0.3     # EMA decay toward the newest epoch's losses
    HARD_FRAC = 0.2      # top fraction counted as hard
    HARD_MIX = 0.5       # fraction of each batch drawn from the hard pool

    def bce_none(logits: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(logits, yt, reduction="none")

    def supcon_loss(z: torch.Tensor, yt: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
        """Canonical supervised contrastive (Khosla et al. 2020) on L2-normalized
        projections; self-similarity EXCLUDED from the denominator (the self term
        would be the constant 1/temp after normalization and swamp the denominator);
        FIT-train labels only."""
        z = nn.functional.normalize(z.float(), dim=1)
        sim = z @ z.t() / temp
        yl = yt.long()
        mask_pos = (yl.unsqueeze(0) == yl.unsqueeze(1))
        mask_pos.fill_diagonal_(0.0)           # positives exclude self
        eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        sim_off = sim.masked_fill(eye, float("-inf"))
        rowmax = sim_off.max(dim=1, keepdim=True).values
        exp = torch.exp(sim_off - rowmax)       # diag -> exp(-inf) = 0
        denom = exp.sum(dim=1)
        logp = (sim - rowmax) - torch.log(denom + 1e-12)
        logp = logp.masked_fill(eye, 0.0)       # kill diag garbage before masking
        pos_count = mask_pos.sum(dim=1)
        valid = pos_count > 0
        if not valid.any():
            return z.sum() * 0.0
        mean_lp = (mask_pos.float() * logp).sum(dim=1)[valid] / pos_count[valid]
        return -mean_lp.mean()

    def gce_loss(logits: torch.Tensor, yt: torch.Tensor, p: float = 0.7) -> torch.Tensor:
        pt = torch.sigmoid(logits) * yt + (1.0 - torch.sigmoid(logits)) * (1.0 - yt)
        return ((1.0 - pt.pow(p)) / p).mean()

    # EMA state
    ema_decay = 0.999
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def ema_update() -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
                else:
                    ema_state[k].copy_(v)  # non-float buffers track the live value

    def eval_logits(X: torch.Tensor, use_ema: bool) -> np.ndarray:
        if use_ema:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state)
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, X.shape[0], INFER_CHUNK):
                chunk = X[s:s + INFER_CHUNK]
                with torch.amp.autocast(dev.type, dtype=torch.bfloat16,
                                        enabled=(dev.type == "cuda")):
                    o = model(chunk)
                out.append((o[0] if isinstance(o, tuple) else o).float().cpu().numpy())
        if use_ema:
            model.load_state_dict(backup)
        return np.concatenate(out)

    # Parity with batch-1: train_and_predict() re-seeds torch right before the
    # epoch loop (model construction consumed RNG for init); the first
    # randperm stream must therefore start from a fresh seed(SEED) state.
    torch_seed(SEED)

    best_val = -1.0
    best_state = None
    best_ema_state = None
    since_best = 0
    for epoch in range(max_epochs):
        if epoch < warmup:
            lr_scale = (epoch + 1) / warmup
        else:
            t = (epoch - warmup) / max(1, max_epochs - warmup)
            lr_scale = 0.5 * (1.0 + np.cos(np.pi * t))
        for g in opt.param_groups:
            g["lr"] = TRAIN_CFG["lr"] * lr_scale
        model.train()
        if proj is not None:
            proj.train()

        # batch index sampler per arm
        if arm == "hard_mining" and epoch > 0:
            # Each batch: HARD_MIX fraction drawn (with replacement) from the
            # current top-HARD_FRAC hardest FIT-train rows, remainder uniform;
            # deterministic per-epoch generator. Full-length perm so the batch
            # loop below walks every train row once per epoch.
            k = max(1, int(HARD_FRAC * n))
            top = np.argpartition(-hardness, k - 1)[:k]
            rng_e = np.random.default_rng(SEED * 1000 + epoch)
            perm_np = np.empty(n, dtype=np.int64)
            for s0 in range(0, n, bs):
                L = min(bs, n - s0)
                n_hard = int(HARD_MIX * L)
                perm_np[s0:s0 + n_hard] = top[rng_e.integers(0, k, n_hard)]
                perm_np[s0 + n_hard:s0 + L] = rng_e.integers(0, n, L - n_hard)
            perm = torch.from_numpy(perm_np).to(dev)
        else:
            perm = torch.randperm(n, device=dev)

        epoch_losses = [] if arm == "hard_mining" else None
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev.type, dtype=torch.bfloat16,
                                    enabled=(dev.type == "cuda")):
                if arm == "supcon":
                    logits, h = model(Xtr_t[idx])
                    bce = nn.functional.binary_cross_entropy_with_logits(
                        logits, ytr_t[idx])
                    z = proj(h)
                    # contrastive similarity matmul in fp32 (autocast would
                    # downcast z @ z.T to bf16, coarsening temp=0.1 similarities)
                    with torch.autocast(dev.type, enabled=False):
                        loss = bce + 0.5 * supcon_loss(z.float(), ytr_t[idx])
                elif arm == "auc_surrogate":
                    logits = model(Xtr_t[idx])
                    yt = ytr_t[idx]
                    pos = torch.where(yt > 0.5)[0]
                    neg = torch.where(yt < 0.5)[0]
                    if len(pos) and len(neg):
                        pi = pos[torch.randint(0, len(pos), (4096,), device=dev)]
                        ni = neg[torch.randint(0, len(neg), (4096,), device=dev)]
                        diff = (logits[pi] - logits[ni]).float()
                        loss = nn.functional.softplus(-diff).mean()
                    else:  # degenerate batch: fall back to BCE (recorded as rare)
                        loss = nn.functional.binary_cross_entropy_with_logits(
                            logits, yt)
                elif arm == "hard_mining":
                    logits = model(Xtr_t[idx])
                    per = bce_none(logits, ytr_t[idx])
                    loss = per.mean()
                    if epoch_losses is not None:
                        epoch_losses.append(
                            (idx.detach().cpu().numpy(),
                             per.detach().float().cpu().numpy()))
                elif arm == "gce":
                    logits = model(Xtr_t[idx])
                    loss = gce_loss(logits.float(), ytr_t[idx])
                else:  # bce_control, ema
                    logits = model(Xtr_t[idx])
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, ytr_t[idx])
            loss.backward()
            opt.step()
            if arm == "ema":
                ema_update()

        # update hardness EMA on FIT-TRAIN only (hard_mining); with-replacement
        # draws can duplicate rows within an epoch -> exactly one deterministic
        # update per row per epoch (first-occurrence loss).
        if arm == "hard_mining" and epoch_losses:
            all_idx = np.concatenate([e[0] for e in epoch_losses])
            all_loss = np.concatenate([e[1] for e in epoch_losses])
            order = np.argsort(all_idx, kind="stable")
            s_idx, s_loss = all_idx[order], all_loss[order]
            first = np.empty(len(s_idx), dtype=bool)
            first[0] = True
            first[1:] = s_idx[1:] != s_idx[:-1]
            u_idx, u_loss = s_idx[first], s_loss[first]
            hardness[u_idx] = (hardness[u_idx] * (1.0 - HARD_ALPHA)
                               + u_loss * HARD_ALPHA)

        # inner-val AUC (never the eval fold), bounded minibatches
        use_ema = (arm == "ema")
        vlog = eval_logits(Xval_t, use_ema=use_ema)
        v_auc = roc_auc_score(yval_np, vlog)
        if v_auc > best_val + 1e-6:
            best_val = v_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if arm == "ema":
                best_ema_state = {k: v.detach().clone() for k, v in ema_state.items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if arm == "ema" and best_ema_state is not None:
        ema_state = best_ema_state

    t1 = time.time()
    logits_eval = eval_logits(Xeval_t, use_ema=(arm == "ema"))
    predict_s = time.time() - t1
    pred = 1.0 / (1.0 + np.exp(-logits_eval.astype(np.float64)))
    assert pred.shape == (len(EVAL),)
    assert np.isfinite(pred).all() and (pred >= 0).all() and (pred <= 1).all()

    m = screen_metrics(y_eval, pred, port0, route0)

    auc_ref = float(roc_auc_score(y_eval, ref0))
    blend = 1.0 / (1.0 + np.exp(-(logit(_cl(pred)) + logit(_cl(ref0))) / 2.0))
    auc_blend_ref = float(roc_auc_score(y_eval, blend))
    port_delta = auc_blend_ref - auc_ref
    from scipy.stats import spearmanr
    spearman = float(spearmanr(pred, ref0).statistic)
    prom = bool(port_delta >= 5e-5 and spearman < 0.995)

    parity_ok = None
    if arm == "bce_control":
        parity_ok = bool(abs(m["standalone_auc"] - CONTROL_AUC_COMMITTED) <= 1e-4)
        log(f"parity gate: control {m['standalone_auc']:.6f} vs committed "
            f"{CONTROL_AUC_COMMITTED:.6f} -> {parity_ok}")

    pred_path = ART / f"{arm}_fold0.npy"
    np.save(pred_path, pred.astype("float32"))

    meta = {
        "model": "fttransformer", "arm": arm, "status": "DONE",
        "utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "prereg": PREREG, "prereg_sha256": sha,
        "recipe": f"FT-Transformer frozen batch-1 config, objective arm={arm}",
        "arch_cfg": ARCH_CFG, "train_cfg": TRAIN_CFG,
        "arm_params": {
            "supcon": "BCE + 0.5*SupCon, proj 128-d, temp 0.1, FIT-train labels only",
            "auc_surrogate": "pure RankNet surrogate, 4096 sampled pos/neg pairs/step",
            "hard_mining": "hardness EMA alpha=0.3, top-20% pool, 50% mix, FIT-train only",
            "gce": "generalized CE p=0.7",
            "ema": "weight EMA decay 0.999 (val eval + final prediction)",
            "bce_control": "plain BCE (batch-1 objective parity)",
        }[arm],
        "fit_rows": int(len(FIT)), "eval_fold": 0, "eval_rows": int(len(EVAL)),
        "fit_folds": [1, 2, 3, 4], "train_rows": int(len(train_pos)),
        "val_rows": int(len(val_pos)), "inner_es_seed": SEED, "seed": SEED,
        "scaler": "zero-mean/unit-var fit on FIT-train only",
        "features": list(names), "categorical_indices": list(cats),
        "control_reference_auc": CONTROL_AUC_COMMITTED if arm == "bce_control" else None,
        "control_parity_ok": parity_ok,
        "route_ref": ROUTE_REF, "route_ref_auc_fold0": auc_ref,
        "prediction_sha256": arr_sha256(np.load(pred_path)),
        "metrics": m,
        "frozen_rule": {
            "port_delta": port_delta, "route_ref_auc": auc_ref,
            "blend_auc": auc_blend_ref, "spearman_vs_route_ref": spearman,
            "promising": prom,
        },
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "predict_seconds": predict_s,
        "total_seconds": time.time() - t0,
    }
    save_json(meta, OUT / f"{arm}_fold0.json")

    fr = meta["frozen_rule"]
    log(f"standalone={m['standalone_auc']:.6f} (ref {auc_ref:.6f}) "
        f"blend={fr['blend_auc']:.6f} port_delta={port_delta:+.3e} "
        f"spearman={spearman:.4f} -> PROMISING={prom}")
    log(f"total {meta['total_seconds']:.1f}s")
    log(f"report -> {(OUT / f'{arm}_fold0.json').relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
