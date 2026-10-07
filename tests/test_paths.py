import torch

from flow_matching.paths import LinearConditionalPath


def test_linear_path_endpoints() -> None:
    x0 = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    x1 = torch.tensor([[4.0, -2.0], [2.0, 1.0]])
    path = LinearConditionalPath()

    start, velocity = path.sample(x0, x1, torch.zeros(2))
    end, _ = path.sample(x0, x1, torch.ones(2))

    assert torch.allclose(start, x0)
    assert torch.allclose(end, x1)
    assert torch.allclose(velocity, x1 - x0)
