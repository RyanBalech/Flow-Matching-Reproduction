from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

VectorField = Callable[[Tensor, Tensor], Tensor]


def _time_batch(value: float, x: Tensor) -> Tensor:
    return torch.full((x.shape[0],), value, device=x.device, dtype=x.dtype)


@torch.no_grad()
def euler_integrate(field: VectorField, x0: Tensor, steps: int = 100) -> Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    x = x0.clone()
    dt = 1.0 / steps
    for index in range(steps):
        t = _time_batch(index * dt, x)
        x = x + dt * field(t, x)
    return x


@torch.no_grad()
def rk4_integrate(field: VectorField, x0: Tensor, steps: int = 25) -> Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    x = x0.clone()
    dt = 1.0 / steps
    for index in range(steps):
        time = index * dt
        k1 = field(_time_batch(time, x), x)
        k2 = field(_time_batch(time + dt / 2, x), x + dt * k1 / 2)
        k3 = field(_time_batch(time + dt / 2, x), x + dt * k2 / 2)
        k4 = field(_time_batch(time + dt, x), x + dt * k3)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x
