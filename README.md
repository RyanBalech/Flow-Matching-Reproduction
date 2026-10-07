# Flow Matching: Reproduction, Ablations & Extensions

Independent PyTorch reproduction of the core ideas in **Flow Matching for Generative Modeling** (Lipman et al., ICLR 2023).

This repository studies continuous normalizing flows trained by conditional flow matching. The goal is not to mirror an existing implementation: it builds the method from the equations, validates it on controlled 2-D distributions, and uses matched-compute experiments to study probability paths and ODE integration.

## Research questions

1. Can conditional flow matching learn known 2-D transports from a standard Gaussian?
2. How do linear/optimal-transport-style paths compare with variance-preserving diffusion-style paths under the same training budget?
3. How sensitive is sample quality to the ODE solver and number of function evaluations?
4. Which choices improve quality per unit of training/inference compute?

## Method

For data sample x1 and noise x0, a conditional probability path defines x_t and a target velocity u_t. A neural vector field v_theta(t, x) is trained with

    E ||v_theta(t, x_t) - u_t(x_t | x1)||^2.

At generation time, samples are drawn from the base distribution and transported by integrating the learned ODE from t=0 to t=1.

## Status

- [x] Independent conditional-flow-matching core
- [x] Time-conditioned MLP vector field
- [x] Euler and RK4 ODE solvers
- [x] Reproducible synthetic distributions
- [x] Unit tests and CI
- [ ] Reproduce 2-D transport experiments
- [x] Linear vs variance-preserving path ablation pipeline
- [x] Matched-NFE Euler vs RK4 ablation pipeline
- [x] Multi-seed paired comparison pipelines
- [ ] Research report with figures and failure analysis

## Reproducibility

Experiments use explicit seeds and save configuration with metrics. Headline comparisons will use multiple seeds and matched training budgets. Results are not added to this README until produced by reproducible runs.

## References

- Yaron Lipman et al. *Flow Matching for Generative Modeling*. ICLR 2023.
- Meta FAIR. *Flow Matching Guide and Code*. 2024. Used as a conceptual/correctness reference, not copied implementation.


## Implemented ablations

Two controlled studies are now implemented:

- **Probability path:** straight linear transport versus a variance-preserving trigonometric path, with the same architecture, optimizer, training steps, seeds, evaluation samples, and RK4 inference budget.
- **ODE solver:** Euler at 50 function evaluations versus RK4 at 48 function evaluations, evaluated on the same trained model and paired samples across five seeds.

Both studies write per-seed metrics and paired differences to JSON. Numerical conclusions remain intentionally separate from implementation status until the reproducible runs complete.
