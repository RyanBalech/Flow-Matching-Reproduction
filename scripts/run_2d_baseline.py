from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from flow_matching.data import sample_eight_gaussians
from flow_matching.metrics import squared_mmd_rbf
from flow_matching.models import VectorFieldMLP
from flow_matching.solvers import euler_integrate, rk4_integrate
from flow_matching.training import TrainConfig, train_flow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/eight_gaussians_baseline.json")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = VectorFieldMLP()
    config = TrainConfig(steps=args.steps, seed=args.seed)
    losses = train_flow(model, sample_eight_gaussians, config=config)

    generator = torch.Generator().manual_seed(args.seed + 1)
    base = torch.randn((args.samples, 2), generator=generator)
    target = sample_eight_gaussians(args.samples, generator=generator)

    generated_euler = euler_integrate(model, base, steps=50)
    generated_rk4 = rk4_integrate(model, base, steps=12)

    result = {
        "experiment": "eight_gaussians_baseline",
        "seed": args.seed,
        "training": {
            "steps": args.steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "final_loss": losses[-1],
            "mean_last_100_loss": sum(losses[-100:]) / min(100, len(losses)),
        },
        "evaluation": {
            "samples": args.samples,
            "euler": {"steps": 50, "nfe": 50, "mmd2_rbf": float(squared_mmd_rbf(generated_euler, target))},
            "rk4": {"steps": 12, "nfe": 48, "mmd2_rbf": float(squared_mmd_rbf(generated_rk4, target))},
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "warning": "Single-seed baseline. Do not interpret as a paper-level result.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
