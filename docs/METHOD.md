# Method and experimental choices

## Objective

Draw independent noise x₀ ~ N(0, I), a target x₁, and t ~ Uniform[0, 1]. The linear
conditional path is xₜ = (1−t)x₀ + tx₁ and its conditional velocity is x₁−x₀.
Regress a time-conditioned MLP onto that velocity with mean squared error. Sampling
solves dx/dt = vθ(t, x) from t=0 to 1.

Independent endpoint pairing is **not minibatch optimal transport**. The straight
conditional interpolation does not imply that the learned marginal coupling is
optimal. The trigonometric alternative uses the derivative of its own interpolant.
Neither experiment establishes the full paper's results.

## Why these controls?

- Compare Euler with N steps to RK4 with N/4 steps: a classical RK4 step calls the field four times.
- Share the trained field and initial noise: changing models or samples would confound solver effects.
- Estimate bandwidth from target samples only: a solver cannot change the kernel used to judge itself.
- Pair seeds when bootstrapping: compare differences within each independently trained model.
- Keep every seed: no favorable-seed selection or post-hoc solver ranking from a single run.

The biased MMD estimator includes diagonal terms and has a finite-sample floor.
Lower integration error therefore need not translate into a detectable MMD improvement.
The bootstrap has only five independent training seeds; its resolution is limited.

## Numerical reference

The NFE sweep uses RK4 at 8,192 function evaluations as a numerical reference and
checks it against 16,384. This is an approximation to the learned ODE solution,
not the true data-generating transport. Endpoint MSE compares paired trajectories'
final states; MMD compares distributions. They answer different questions.

The model's sinusoidal time embedding spans frequencies from 1 to 1,000 cycles over
[0, 1]. Coarse integration can undersample this time dependence. A smooth-field
fourth-order convergence test is therefore a correctness check, not a guarantee
that RK4 beats Euler at equal low NFE for this trained field. Reducing the embedding
frequency range is a useful next architectural ablation; it has not been measured.

## Reading the implementation

The [Meta 2-D example](https://github.com/facebookresearch/flow_matching/blob/main/examples/2d_flow_matching.ipynb)
provides a reference for separating path, velocity model and integration. This repo
keeps those pieces small and adds paired numerical studies. The original paper and
Meta library are references, not evidence of authorship or affiliation.
