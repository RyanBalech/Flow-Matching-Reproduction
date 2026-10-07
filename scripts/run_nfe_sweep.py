"""Separate sample-distribution error from numerical integration error.

Every solver/budget sees the same trained field, initial noise and target samples.
The high-budget RK4 solution is a numerical reference, not the true transport.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import torch

from flow_matching.data import sample_eight_gaussians
from flow_matching.metrics import squared_mmd_rbf
from flow_matching.models import VectorFieldMLP
from flow_matching.solvers import euler_integrate, rk4_integrate
from flow_matching.training import TrainConfig, train_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--nfe", nargs="+", type=int, default=[16, 32, 48, 96, 192, 384])
    parser.add_argument("--reference-nfe", type=int, default=1024)
    parser.add_argument("--output", default="results/nfe_sweep.json")
    parser.add_argument("--figure", default="docs/figures/nfe_sweep.svg")
    args = parser.parse_args()
    if any(n <= 0 or n % 4 for n in [*args.nfe, args.reference_nfe]):
        parser.error("NFE budgets must be positive multiples of four")
    if max(args.nfe) >= args.reference_nfe:
        parser.error("reference budget must exceed every sweep budget")
    if args.steps <= 0 or args.samples < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("positive steps, at least two samples and distinct seeds are required")
    visual_nfe = 48 if 48 in args.nfe else args.nfe[0]

    runs, visual = [], None
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = VectorFieldMLP()
        config = TrainConfig(steps=args.steps, seed=seed)
        losses = train_flow(model, sample_eight_gaussians, config=config)
        model.eval()
        generator = torch.Generator().manual_seed(seed + 10_000)
        base = torch.randn((args.samples, 2), generator=generator)
        target = sample_eight_gaussians(args.samples, generator=generator)
        distances = torch.pdist(target).square()
        bandwidth = float(distances[distances > 0].median())
        reference = rk4_integrate(model, base, steps=args.reference_nfe // 4)
        finer = rk4_integrate(model, base, steps=args.reference_nfe // 2)
        rows = []
        for nfe in args.nfe:
            for name, integrate, steps in [("euler", euler_integrate, nfe),
                                           ("rk4", rk4_integrate, nfe // 4)]:
                start = perf_counter()
                generated = integrate(model, base, steps=steps)
                elapsed = perf_counter() - start
                rows.append({"solver": name, "nfe": nfe,
                             "mmd2": float(squared_mmd_rbf(generated, target, bandwidth)),
                             "endpoint_mse_to_reference": float((generated - reference).square().mean()),
                             "integration_seconds": elapsed})
                if visual is None and nfe == visual_nfe and name == "rk4":
                    visual = (base.numpy(), target.numpy(), generated.numpy())
        state_hash = hashlib.sha256()
        for tensor in model.state_dict().values():
            state_hash.update(tensor.cpu().numpy().tobytes())
        runs.append({"seed": seed, "bandwidth": bandwidth,
                     "model_state_sha256": state_hash.hexdigest(),
                     "mean_last_100_loss": sum(losses[-100:]) / min(100, len(losses)),
                     "reference_mmd2": float(squared_mmd_rbf(reference, target, bandwidth)),
                     "reference_doubling_endpoint_mse": float((reference - finer).square().mean()),
                     "measurements": rows})
        print(f"seed {seed} completed", flush=True)
    result = {"experiment": "paired_nfe_sweep", "training_steps": args.steps,
              "samples_per_seed": args.samples, "nfe_budgets": args.nfe,
              "reference_nfe": args.reference_nfe,
              "reference_check_nfe": 2 * args.reference_nfe,
              "bandwidth_rule": "median positive target-only squared distance, shared within seed",
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "threads": torch.get_num_threads()},
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
              "runs": runs}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for name in ["euler", "rk4"]:
        for metric, ax in [("mmd2", axes[0]), ("endpoint_mse_to_reference", axes[1])]:
            values = np.array([[r[metric] for r in run["measurements"] if r["solver"] == name]
                               for run in runs])
            for row in values:
                ax.plot(args.nfe, row, alpha=0.18, color="C0" if name == "euler" else "C1")
            ax.plot(args.nfe, values.mean(0), marker="o", label=name)
    axes[0].set_ylabel("Biased MMD² to target")
    axes[1].set_ylabel("Endpoint MSE to RK4 reference")
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Function evaluations")
        ax.legend(frameon=False)
    assert visual is not None
    axes[2].scatter(visual[1][:, 0], visual[1][:, 1], s=3, alpha=.3, label="target")
    axes[2].scatter(visual[2][:, 0], visual[2][:, 1], s=3, alpha=.3, label=f"RK4, {visual_nfe} NFE")
    axes[2].set_title(f"Seed {args.seeds[0]}")
    axes[2].set_aspect("equal")
    axes[2].legend(frameon=False, markerscale=3)
    fig.tight_layout()
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, metadata={"Date": None})


if __name__ == "__main__":
    main()
