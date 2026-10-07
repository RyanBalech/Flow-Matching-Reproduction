"""Render paired seed results from saved JSON; no training or cherry-picked samples."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/published"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures/paired_studies.svg"))
    args = parser.parse_args()
    solver = json.loads((args.results / "solver_exact_nfe48.json").read_text())
    paths = json.loads((args.results / "path_ablation_multiseed.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    studies = [
        (solver, ["Euler", "RK4"], lambda r: [r["euler_mmd2"], r["rk4_mmd2"]],
         "Solvers: 48 evaluations each"),
        (paths, ["Linear", "Trigonometric"],
         lambda r: [r["linear"]["mmd2"], r["variance_preserving_trigonometric"]["mmd2"]],
         "Paths: RK4, 48 evaluations"),
    ]
    for ax, (data, labels, values, title) in zip(axes, studies):
        for row in data["runs"]:
            ax.plot([0, 1], values(row), "o-", alpha=0.8, label=f"Seed {row['seed']}")
        ax.set_xticks([0, 1], labels)
        ax.set_title(title)
        ax.set_ylabel("MMD² (lower is better)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
