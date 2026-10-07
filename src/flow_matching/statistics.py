from __future__ import annotations

import random
from statistics import mean


def paired_bootstrap_ci(
    deltas: list[float], *, samples: int = 10_000, seed: int = 2026, confidence: float = 0.95
) -> dict[str, float]:
    if not deltas:
        raise ValueError("deltas cannot be empty")
    rng = random.Random(seed)
    n = len(deltas)
    estimates = [
        mean(deltas[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    ]
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": mean(deltas),
        "ci_low": estimates[int(alpha * (samples - 1))],
        "ci_high": estimates[int((1.0 - alpha) * (samples - 1))],
        "confidence": confidence,
    }
