from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import torch

from flow_matching.data import sample_eight_gaussians
from flow_matching.metrics import squared_mmd_rbf
from flow_matching.models import VectorFieldMLP
from flow_matching.paths import LinearConditionalPath, TrigonometricConditionalPath
from flow_matching.solvers import rk4_integrate
from flow_matching.statistics import paired_bootstrap_ci
from flow_matching.training import TrainConfig, train_flow


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/path_ablation_multiseed.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--samples", type=int, default=1000)
    args = p.parse_args()

    paths = {
        "linear": LinearConditionalPath(),
        "variance_preserving_trigonometric": TrigonometricConditionalPath(),
    }
    runs = []
    for seed in args.seeds:
        generator = torch.Generator().manual_seed(seed + 20_000)
        base = torch.randn((args.samples, 2), generator=generator)
        target = sample_eight_gaussians(args.samples, generator=generator)
        distances = torch.pdist(target).square()
        bandwidth = float(distances[distances > 0].median())
        row = {"seed": seed, "bandwidth": bandwidth}
        for name, path in paths.items():
            torch.manual_seed(seed)
            model = VectorFieldMLP()
            losses = train_flow(
                model,
                sample_eight_gaussians,
                path=path,
                config=TrainConfig(steps=args.steps, seed=seed),
            )
            with torch.no_grad():
                generated = rk4_integrate(model, base, steps=12)
            row[name] = {
                "mmd2": float(squared_mmd_rbf(generated, target, bandwidth=bandwidth)),
                "mean_last_100_loss": sum(losses[-100:]) / min(100, len(losses)),
            }
        row["vp_minus_linear_mmd2"] = (
            row["variance_preserving_trigonometric"]["mmd2"] - row["linear"]["mmd2"]
        )
        runs.append(row)

    linear = [r["linear"]["mmd2"] for r in runs]
    vp = [r["variance_preserving_trigonometric"]["mmd2"] for r in runs]
    delta = [r["vp_minus_linear_mmd2"] for r in runs]
    result = {
        "experiment": "matched_budget_probability_path_ablation",
        "training_steps_per_path": args.steps,
        "evaluation_samples": args.samples,
        "mmd_bandwidth": "median squared target distance; shared across methods within seed",
        "solver": {"name": "rk4", "steps": 12, "nfe": 48},
        "runs": runs,
        "summary": {
            "linear_mmd2": summary(linear),
            "variance_preserving_mmd2": summary(vp),
            "paired_vp_minus_linear_mmd2": summary(delta),
            "paired_vp_minus_linear_mmd2_ci": paired_bootstrap_ci(delta),
        },
        "interpretation": (
            "Negative paired differences favor the variance-preserving path; positive "
            "differences favor the linear path. Conclusions should use paired behavior "
            "across seeds rather than a single run."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
