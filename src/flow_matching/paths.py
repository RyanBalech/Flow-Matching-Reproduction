from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class LinearConditionalPath:
    """Straight conditional path from Gaussian noise x0 to data x1.

    x_t = (1 - t) x0 + t x1
    dx_t / dt = x1 - x0
    """

    sigma_min: float = 0.0

    def sample(self, x0: Tensor, x1: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        if x0.shape != x1.shape:
            raise ValueError("x0 and x1 must have identical shapes")
        if t.ndim == 1:
            t = t[:, None]
        if t.shape[0] != x0.shape[0]:
            raise ValueError("t must contain one time per sample")

        # A small terminal noise floor is useful for later ablations while sigma_min=0
        # recovers the exact straight-line path.
        scale = 1.0 - (1.0 - self.sigma_min) * t
        x_t = scale * x0 + t * x1
        velocity = -(1.0 - self.sigma_min) * x0 + x1
        return x_t, velocity
