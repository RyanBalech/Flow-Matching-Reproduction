from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .paths import LinearConditionalPath


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 2_000
    batch_size: int = 512
    learning_rate: float = 2e-3
    seed: int = 0


def train_flow(
    model: nn.Module,
    sample_data,
    *,
    path: LinearConditionalPath | None = None,
    config: TrainConfig = TrainConfig(),
    device: str = "cpu",
) -> list[float]:
    if config.steps <= 0 or config.batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    torch.manual_seed(config.seed)
    model.to(device)
    path = path or LinearConditionalPath()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []

    generator = torch.Generator().manual_seed(config.seed)
    for _ in range(config.steps):
        x1 = sample_data(config.batch_size, generator=generator).to(device)
        x0 = torch.randn(x1.shape, generator=generator, device="cpu").to(device)
        t = torch.rand((config.batch_size,), generator=generator, device="cpu").to(device)
        x_t, target_velocity = path.sample(x0, x1, t)
        prediction = model(t, x_t)
        loss = torch.mean((prediction - target_velocity) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return losses
