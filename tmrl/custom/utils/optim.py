"""Gradient stabilisation utilities for deep RL training."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class GradientStabilizer:
    """Replace hard gradient clipping with an EMA-based magnitude stabiliser.

    Instead of truncating every gradient whose norm exceeds a fixed threshold,
    this module tracks the running average of the gradient norm and rescales
    the current gradient so its magnitude matches the EMA whenever the
    instantaneous norm exceeds the running estimate.  The gradient *direction*
    is always preserved.

    Usage (drop-in replacement for ``clip_grad_norm_``)::

        stabilizer = GradientStabilizer(ema_decay=0.995)
        ...
        loss.backward()
        grad_norm = stabilizer.step(model.parameters())
        optimizer.step()
    """

    def __init__(self, ema_decay: float = 0.995, eps: float = 1e-8, warmup: int = 100) -> None:
        self.ema_decay = ema_decay
        self.eps = eps
        self._ema: float | None = None
        self._warmup = warmup
        self._call_count = 0

    def step(self, parameters: Iterable[nn.Parameter]) -> float:
        """Stabilise gradients in-place and return the (pre-stabilisation) grad norm."""
        params = [p for p in parameters if p.grad is not None]
        if not params:
            return 0.0

        total_norm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
        current_norm = float(total_norm)

        if self._ema is None:
            self._ema = current_norm
            self._call_count = 1
            return current_norm

        self._call_count += 1
        self._ema = self.ema_decay * self._ema + (1.0 - self.ema_decay) * current_norm

        if self._call_count <= self._warmup:
            return current_norm

        if current_norm > self._ema + self.eps:
            scale = self._ema / (current_norm + self.eps)
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(scale)

        return current_norm

    @property
    def ema_norm(self) -> float:
        return self._ema if self._ema is not None else 0.0
