# Controlled 2-D flow-matching experiments

## Scope and protocol

These are independent synthetic experiments with the core conditional flow-matching
objective, not a reproduction of the original paper's image-generation benchmarks.
Each model uses 2,000 training steps, a batch of 512, AdamW at 0.002, and seeds 0–4.
Each evaluation compares 1,000 generated samples against 1,000 target samples from
eight Gaussians. Methods within each seed share base noise and target samples.
The biased RBF MMD² uses the median positive squared distance in the target sample
as its bandwidth parameter, fixed across the methods being compared.

## Equal-budget solver study

Both solvers use exactly 48 vector-field evaluations: Euler has 48 steps; RK4 has 12.
Both use the same trained linear-path model within each seed.

| Solver | Mean MMD² | Sample standard deviation |
|---|---:|---:|
| Euler | 0.00094724 | 0.00060761 |
| RK4 | 0.00109940 | 0.00080713 |

The paired mean RK4 minus Euler is +0.00015216. Its seed-bootstrap 95% interval is
[+0.00000403, +0.00031817]. This small study favors Euler under this metric and budget;
five seeds provide limited evidence, and higher integration order does not guarantee
better samples from an approximate learned vector field. No general solver claim follows.

## Probability-path study

Both paths have identical initialization and training budgets within each seed.
Generation uses RK4 with 48 evaluations. The trigonometric path is a controlled
alternative, not a claim to reproduce every diffusion path in the paper.

| Path | Mean MMD² | Sample standard deviation |
|---|---:|---:|
| Linear | 0.00128229 | 0.00077063 |
| Trigonometric | 0.00134284 | 0.00088508 |

The paired trigonometric minus linear difference is +0.00006056, with a 95% seed-bootstrap
interval of [-0.00062013, +0.00070984]. The direction changes across seeds; neither path
has an established advantage here. This study uses a different evaluation seed offset
from the solver study, so compare methods within each study, not across the two tables.

![Paired results for every seed](figures/paired_studies.svg)

## Reproduce

From the repository root:

```bash
pip install -e ".[dev,plots]"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_multiseed_solver_study.py --nfe 48 --output results/published/solver_exact_nfe48.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_path_ablation.py --output results/published/path_ablation_multiseed.json
python scripts/plot_studies.py
pytest -q
```

Raw results and runtime versions are in `results/published/`. The earlier
`eight_gaussians_solver_multiseed.json` is preserved from GitHub Actions run
[37564755938](https://github.com/RyanBalech/Flow-Matching-Reproduction/actions/runs/37564755938)
at commit `1a3b4f6a20cf77e5665bbd1ed1192daf3beb969d`. It used Euler 50 versus RK4 48
evaluations and separately estimated bandwidths. Its apparent RK4 advantage is
superseded by the controlled comparison above; the old artifact is retained for audit.

Next research steps: more seeds, more target distributions, a predeclared bandwidth
sensitivity sweep, and sample/trajectory plots. These results are a project report,
not a peer-reviewed publication or evidence of high-dimensional performance.
