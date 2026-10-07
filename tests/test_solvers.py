import torch

from flow_matching.solvers import euler_integrate, rk4_integrate


def constant_field(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    del t
    return torch.ones_like(x)


def test_euler_integrates_constant_field() -> None:
    x0 = torch.zeros((4, 2))
    assert torch.allclose(euler_integrate(constant_field, x0, steps=10), torch.ones_like(x0))


def test_rk4_integrates_constant_field() -> None:
    x0 = torch.zeros((4, 2))
    assert torch.allclose(rk4_integrate(constant_field, x0, steps=5), torch.ones_like(x0))
