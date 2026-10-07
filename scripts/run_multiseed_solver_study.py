from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import torch

from flow_matching.data import sample_eight_gaussians
from flow_matching.metrics import squared_mmd_rbf
from flow_matching.models import VectorFieldMLP
from flow_matching.solvers import euler_integrate, rk4_integrate
from flow_matching.training import TrainConfig, train_flow


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/eight_gaussians_solver_multiseed.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    runs = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = VectorFieldMLP()
        config = TrainConfig(steps=args.steps, seed=seed)
        losses = train_flow(model, sample_eight_gaussians, config=config)

        generator = torch.Generator().manual_seed(seed + 10_000)
        base = torch.randn((args.samples, 2), generator=generator)
        target = sample_eight_gaussians(args.samples, generator=generator)
        with torch.no_grad():
            euler = euler_integrate(model, base, steps=50)
            rk4 = rk4_integrate(model, base, steps=12)
        euler_mmd = float(squared_mmd_rbf(euler, target))
        rk4_mmd = float(squared_mmd_rbf(rk4, target))
        runs.append({
            "seed": seed,
            "mean_last_100_loss": sum(losses[-100:]) / min(100, len(losses)),
            "euler_mmd2": euler_mmd,
            "rk4_mmd2": rk4_mmd,
            "rk4_minus_euler": rk4_mmd - euler_mmd,
        })

    euler_values = [r["euler_mmd2"] for r in runs]
    rk4_values = [r["rk4_mmd2"] for r in runs]
    deltas = [r["rk4_minus_euler"] for r in runs]
    result = {
        "experiment": "eight_gaussians_solver_multiseed",
        "training_steps": args.steps,
        "evaluation_samples": args.samples,
        "matched_compute": {"euler_nfe": 50, "rk4_nfe": 48},
        "runs": runs,
        "summary": {
            "euler_mmd2": summarize(euler_values),
            "rk4_mmd2": summarize(rk4_values),
            "paired_rk4_minus_euler": summarize(deltas),
        },
        "warning": "Synthetic 2-D study; lower MMD^2 is better. Interpret solver differences across seeds, not from one run.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
