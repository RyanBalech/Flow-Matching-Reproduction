import torch

from flow_matching.metrics import squared_mmd_rbf


def test_mmd_is_zero_for_identical_samples() -> None:
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])
    assert torch.isclose(squared_mmd_rbf(x, x), torch.tensor(0.0), atol=1e-6)


def test_mmd_detects_shift() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn((100, 2), generator=generator)
    y = x + 4.0
    assert squared_mmd_rbf(x, y) > 0.1
