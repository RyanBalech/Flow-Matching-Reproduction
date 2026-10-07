from __future__ import annotations

import math

import torch
from torch import Tensor


def sample_eight_gaussians(n: int, *, generator: torch.Generator | None = None) -> Tensor:
    if n <= 0:
        raise ValueError("n must be positive")
    angles = torch.arange(8, dtype=torch.float32) * (2 * math.pi / 8)
    centers = 4.0 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    indices = torch.randint(0, 8, (n,), generator=generator)
    noise = 0.25 * torch.randn((n, 2), generator=generator)
    return centers[indices] + noise


def sample_checkerboard(n: int, *, generator: torch.Generator | None = None) -> Tensor:
    if n <= 0:
        raise ValueError("n must be positive")
    x = torch.rand((n,), generator=generator) * 4.0 - 2.0
    cell = torch.randint(0, 2, (n,), generator=generator).float()
    y = torch.rand((n,), generator=generator) - cell * 2.0
    y = y + (torch.floor(x) % 2)
    return torch.stack([x, y], dim=1) * 2.0
