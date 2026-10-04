"""Per-epoch recording for TabR, so training progress survives a kill and drives decisions.

Why
---
pytabkit wires Lightning's logger to ``DummyLogger`` (tabr_interface.py:324), so every
``self.log('val_accuracy', ...)`` in ``TabrLightning`` is discarded. That means:

  * there is no learning curve to look at when deciding the epoch budget;
  * an epoch-by-epoch history that would justify the budget choice is never written to disk;
  * a run that dies at epoch 20 leaves no evidence of what happened during those 20 epochs.

Rather than reconfigure pytabkit's Lightning stack, this attaches an
``on_validation_epoch_end`` hook. ``TabrLightning`` does not define that method (verified), so
adding it cannot shadow or disable any existing behaviour -- Lightning calls it only if present.
The hook records, per epoch:

  * wall-clock seconds since the previous epoch (so epoch time is measured, not guessed)
  * peak host RSS and peak VRAM, sampled inside the run rather than only afterwards
  * the inner-validation metrics Lightning aggregated (``val_accuracy``, ``val_loss``)
  * inner-validation ROC-AUC computed from the inner-ES predictions, which is what we actually
    optimise -- pytabkit early-stops on accuracy, a much weaker proxy for AUC

Fold safety
-----------
Everything recorded here comes from the INNER early-stopping split carved out of the outer-FIT
rows. No outer-validation row and no outer-validation target passes through this recorder, so the
resulting curve is exactly the quantity a trusted protocol is allowed to use for epoch selection.

Reversibility
-------------
``uninstall_epoch_recorder()`` restores the original class state, and calling ``install`` twice does
not stack hooks.
"""

from __future__ import annotations

import time

import numpy as np

# Set by install_epoch_recorder(); read by the runner after fit() and/or mid-run.
EPOCH_RECORDS: list = []
# Scratch buffers, keyed by nothing: validation runs one batch at a time and we flush at epoch end.
_SCRATCH: dict = {"epoch": None, "t0": None, "last": None, "y_true": [], "y_pred": [],
                  "y_prob": []}

_INSTALLED = False

# Optional per-epoch sink, set by the runner so each record is flushed to disk IMMEDIATELY rather
# than buffered until the fold finishes. Without it, a run killed at epoch 20 leaves an empty log --
# exactly the failure mode this module exists to prevent.
_EPOCH_SINK = None


def set_epoch_sink(fn) -> None:
    """Register a callable invoked with each epoch record as it is produced."""
    global _EPOCH_SINK
    _EPOCH_SINK = fn


def clear_epoch_sink() -> None:
    global _EPOCH_SINK
    _EPOCH_SINK = None


def _epoch_now(module) -> int:
    try:
        return int(module.current_epoch)
    except Exception:  # noqa: BLE001
        return -1


def _on_validation_epoch_end(self) -> None:
    """Append one record per validation epoch, then flush the per-batch scratch buffers."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return

    now = time.time()
    ep = _epoch_now(self)

    # Lightning aggregates the logged scalars into trainer.callback_metrics. Reading
    # self.val_accuracy instead returns the torchmetrics Metric OBJECT, not a float, which is why
    # an earlier version of this recorder logged null accuracy/loss for every epoch.
    cb = {}
    try:
        cb = dict(getattr(self.trainer, "callback_metrics", {}) or {})
    except Exception:  # noqa: BLE001
        cb = {}

    def _f(key):
        v = cb.get(key)
        try:
            return None if v is None else float(v)
        except Exception:  # noqa: BLE001
            return None

    rec = {
        "epoch": ep,
        "epoch_seconds": None if _SCRATCH["last"] is None else round(now - _SCRATCH["last"], 1),
        "cumulative_seconds": None if _SCRATCH["t0"] is None else round(now - _SCRATCH["t0"], 1),
        "val_loss": _f("val_loss"),
        "val_accuracy": _f("val_accuracy"),
    }

    # Lightning runs a validation pass over the eval set BEFORE the first training step. Firing at
    # that point produced a 1.6-second "epoch 0" with every metric null, which is indistinguishable
    # from a real epoch unless it is labelled. global_step == 0 identifies it unambiguously.
    try:
        rec["global_step"] = int(self.global_step)
    except Exception:  # noqa: BLE001
        rec["global_step"] = None
    pre_training = rec["global_step"] == 0
    rec["pre_training_sanity_check"] = pre_training
    rec["kind"] = "sanity_check" if pre_training else "epoch"

    # inner-validation ROC-AUC from the predictions this epoch produced
    from sklearn.metrics import roc_auc_score

    y_true = _SCRATCH["y_true"]
    y_prob = _SCRATCH["y_prob"]
    if len(y_true) > 100 and len(set(y_true)) > 1:
        try:
            rec["val_auc_inner"] = round(float(roc_auc_score(y_true, y_prob)), 6)
        except Exception:  # noqa: BLE001
            rec["val_auc_inner"] = None
    else:
        rec["val_auc_inner"] = None
    rec["inner_val_rows_scored"] = len(y_true)

    try:
        rec["peak_vram_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        rec["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil

        rec["host_rss_gb"] = round(psutil.Process().memory_info().rss / 2**30, 3)
    except Exception:  # noqa: BLE001
        pass

    EPOCH_RECORDS.append(rec)
    if _EPOCH_SINK is not None:
        try:
            _EPOCH_SINK(rec)
        except Exception:  # noqa: BLE001
            pass
    _SCRATCH["y_true"].clear()
    _SCRATCH["y_prob"].clear()
    _SCRATCH["y_pred"].clear()
    _SCRATCH["last"] = now


def _validation_step(self, batch, batch_idx):
    """Wrap TabrLightning.validation_step to stash labels/preds, then defer to the original."""
    from sklearn.metrics import roc_auc_score  # noqa: F401  (ensures sklearn is importable early)

    import torch

    ep = _epoch_now(self)
    if _SCRATCH["epoch"] != ep:
        _SCRATCH["epoch"] = ep
        if _SCRATCH["t0"] is None:
            _SCRATCH["t0"] = time.time()
            _SCRATCH["last"] = _SCRATCH["t0"]
        _SCRATCH["y_true"].clear()
        _SCRATCH["y_prob"].clear()
        _SCRATCH["y_pred"].clear()

    out = _ORIGINAL_VALIDATION_STEP(self, batch, batch_idx)
    try:
        with torch.no_grad():
            # TabrDataset.__getitem__ returns {"indices": idx}, so the collated batch is a DICT and
            # the labels live under "Y" -- an earlier version looked for batch[1], which is why
            # val_auc_inner was always null.
            y = batch["Y"] if isinstance(batch, dict) and "Y" in batch else None
            logits = out[0] if isinstance(out, (list, tuple)) else out
            if y is not None and logits is not None and torch.is_tensor(logits):
                yb = y.detach().reshape(-1).cpu().numpy()
                lg = logits.detach()
                if lg.dim() == 2 and lg.shape[-1] == 2:
                    probs = torch.softmax(lg, dim=-1)[:, 1].reshape(-1).cpu().numpy()
                else:
                    probs = torch.sigmoid(lg.reshape(-1)).cpu().numpy()
                _SCRATCH["y_true"].extend(int(v) for v in yb)
                _SCRATCH["y_prob"].extend(float(v) for v in probs)
    except Exception:  # noqa: BLE001
        pass
    return out


_ORIGINAL_VALIDATION_STEP = None


def install_epoch_recorder() -> bool:
    """Attach the recorder to TabrLightning. Idempotent; returns True if newly installed."""
    global _INSTALLED, _ORIGINAL_VALIDATION_STEP
    from pytabkit.models.nn_models.tabr import TabrLightning

    if _INSTALLED:
        return False
    assert "on_validation_epoch_end" not in TabrLightning.__dict__, \
        "pytabkit now defines on_validation_epoch_end; the recorder must not shadow it"
    _ORIGINAL_VALIDATION_STEP = TabrLightning.validation_step
    TabrLightning.validation_step = _validation_step
    TabrLightning.on_validation_epoch_end = _on_validation_epoch_end
    _INSTALLED = True
    return True


def uninstall_epoch_recorder() -> None:
    global _INSTALLED
    from pytabkit.models.nn_models.tabr import TabrLightning

    if not _INSTALLED:
        return
    TabrLightning.validation_step = _ORIGINAL_VALIDATION_STEP
    TabrLightning.__dict__.pop("on_validation_epoch_end", None)
    _INSTALLED = False


def reset_records() -> None:
    EPOCH_RECORDS.clear()
    _SCRATCH["epoch"] = None
    _SCRATCH["t0"] = None
    _SCRATCH["last"] = None
    _SCRATCH["y_true"].clear()
    _SCRATCH["y_pred"].clear()
    _SCRATCH["y_prob"].clear()


def records() -> list:
    return list(EPOCH_RECORDS)


def auc_curve() -> list:
    """[(epoch, auc)] for real training epochs only -- sanity checks are excluded."""
    return [(r["epoch"], r.get("val_auc_inner")) for r in EPOCH_RECORDS
            if r.get("val_auc_inner") is not None and not r.get("pre_training_sanity_check")]


def epoch_times() -> list:
    """Per-epoch wall-clock seconds for real training epochs only."""
    return [(r["epoch"], r.get("epoch_seconds")) for r in EPOCH_RECORDS
            if not r.get("pre_training_sanity_check") and r.get("epoch_seconds") is not None]
