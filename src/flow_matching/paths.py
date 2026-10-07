from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def _time_column(t: Tensor, batch: int) -> Tensor:
    if t.ndim == 1:
        t = t[:, None]
    if t.shape[0] != batch:
        raise ValueError("t must contain one time per sample")
    return t


@dataclass(frozen=True)
class LinearConditionalPath:
    """Straight conditional path from Gaussian noise x0 to data x1."""

    sigma_min: float = 0.0

    def sample(self, x0: Tensor, x1: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        if x0.shape != x1.shape:
            raise ValueError("x0 and x1 must have identical shapes")
        t = _time_column(t, x0.shape[0])
        scale = 1.0 - (1.0 - self.sigma_min) * t
        x_t = scale * x0 + t * x1
        velocity = -(1.0 - self.sigma_min) * x0 + x1
        return x_t, velocity


@dataclass(frozen=True)
class TrigonometricConditionalPath:
    """Variance-preserving trigonometric path.

    x_t = cos(pi t / 2) x0 + sin(pi t / 2) x1.

    Unlike the straight path, the squared interpolation coefficients sum to one,
    giving a controlled curved path for matched-budget ablations.
    """

    def sample(self, x0: Tensor, x1: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        if x0.shape != x1.shape:
            raise ValueError("x0 and x1 must have identical shapes")
        t = _time_column(t, x0.shape[0])
        angle = 0.5 * math.pi * t
        c, s = torch.cos(angle), torch.sin(angle)
        x_t = c * x0 + s * x1
        velocity = 0.5 * math.pi * (-s * x0 + c * x1)
        return x_t, velocity
