from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding dimension must be even")
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        if t.ndim == 1:
            t = t[:, None]
        half = self.dim // 2
        frequencies = torch.exp(
            torch.linspace(0.0, math.log(1000.0), half, device=t.device, dtype=t.dtype)
        )
        angles = 2.0 * math.pi * t * frequencies[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class VectorFieldMLP(nn.Module):
    def __init__(self, data_dim: int = 2, hidden_dim: int = 128, time_dim: int = 32) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.network = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        time_features = self.time_embedding(t)
        return self.network(torch.cat([x, time_features], dim=-1))
