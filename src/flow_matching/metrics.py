from __future__ import annotations

import torch
from torch import Tensor


def squared_mmd_rbf(x: Tensor, y: Tensor, bandwidth: float | None = None) -> Tensor:
    """Biased RBF maximum mean discrepancy; deterministic and stable for small studies."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be 2-D tensors with the same feature dimension")
    joined = torch.cat([x, y], dim=0)
    distances = torch.cdist(joined, joined).pow(2)
    if bandwidth is None:
        positive = distances[distances > 0]
        bandwidth = float(torch.median(positive).item()) if positive.numel() else 1.0
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")

    def kernel(a: Tensor, b: Tensor) -> Tensor:
        return torch.exp(-torch.cdist(a, b).pow(2) / (2.0 * bandwidth))

    return kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean()
