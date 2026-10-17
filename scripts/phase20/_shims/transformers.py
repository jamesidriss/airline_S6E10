"""Minimal auditable shim of the four transformers LR-scheduler functions that
tabicl.train._optim imports. The certified .venv does not contain `transformers`
and installing it in place would alter the certified stack (forbidden); these
implementations are the standard HuggingFace LambdaLR semantics, reproduced
verbatim in spirit (warmup then decay), and are sufficient for the fine-tuner's
only used scheduler, `cosine_warmup` (tabicl/_finetune/base.py line ~863).

Usage: insert this directory at the FRONT of sys.path before importing tabicl.
"""
from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1) -> LambdaLR:
    return LambdaLR(optimizer, lambda step: 1.0, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int,
                                    num_training_steps: int, last_epoch: int = -1) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_cosine_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int,
                                    num_training_steps: int, num_cycles: float = 0.5,
                                    last_epoch: int = -1) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_polynomial_decay_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int,
                                              num_training_steps: int, lr_end: float = 1e-7,
                                              power: float = 1.0, last_epoch: int = -1) -> LambdaLR:
    lr_init = optimizer.defaults.get("lr", 0.0)

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > num_training_steps:
            return lr_end / lr_init  # clip beyond training
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return ((1.0 - progress) ** power) * (1.0 - lr_end / lr_init) + lr_end / lr_init
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
