import math

import torch

from flow_matching.paths import LinearConditionalPath, TrigonometricConditionalPath


def test_linear_path_endpoints() -> None:
    x0 = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    x1 = torch.tensor([[4.0, -2.0], [2.0, 1.0]])
    path = LinearConditionalPath()
    start, velocity = path.sample(x0, x1, torch.zeros(2))
    end, _ = path.sample(x0, x1, torch.ones(2))
    assert torch.allclose(start, x0)
    assert torch.allclose(end, x1)
    assert torch.allclose(velocity, x1 - x0)


def test_trigonometric_path_endpoints_and_midpoint() -> None:
    x0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    x1 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    path = TrigonometricConditionalPath()
    start, _ = path.sample(x0, x1, torch.zeros(2))
    end, _ = path.sample(x0, x1, torch.ones(2))
    mid, velocity = path.sample(x0, x1, torch.full((2,), 0.5))
    assert torch.allclose(start, x0)
    assert torch.allclose(end, x1, atol=1e-6)
    expected = (x0 + x1) / math.sqrt(2.0)
    assert torch.allclose(mid, expected, atol=1e-6)
    assert velocity.shape == x0.shape
