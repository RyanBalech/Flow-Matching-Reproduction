import math

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


def test_solver_convergence_on_analytic_exponential_and_actual_nfe() -> None:
    # Constant fields cannot distinguish Euler from a genuine fourth-order solver.
    x0 = torch.ones((1, 1), dtype=torch.float64)
    for solver, steps, minimum_ratio, expected_calls in [
        (euler_integrate, 16, 1.8, 16), (rk4_integrate, 8, 14., 32)
    ]:
        times = []

        def field(t, x, times=times):
            times.append(float(t[0]))
            return x

        coarse = solver(field, x0, steps=steps)
        assert len(times) == expected_calls
        assert min(times) >= 0. and max(times) <= 1.
        fine = solver(field, x0, steps=steps * 2)
        ratio = abs(float(coarse) - math.e) / abs(float(fine) - math.e)
        assert ratio > minimum_ratio
