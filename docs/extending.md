# Extending the package

## Add an allocation rule

An acquisition function receives `ctx` and returns one score per candidate.
Higher scores are preferred; `-inf` excludes a candidate. The decorator checks
the scores and applies the common toxicity and protocol eligibility masks.

```python
import numpy as np
import dose_combination_bo as bo

@bo.acquisition("gated_ucb")
def gated_ucb(ctx):
    mean, variance = ctx.latent_efficacy()
    return mean + 2.0 * np.sqrt(np.maximum(variance, 0.0))
```

| Context field | Meaning |
|---|---|
| `latent_efficacy()` | Posterior efficacy mean and variance |
| `mu_g`, `sd_g` | Posterior toxicity mean and SD |
| `pf`, `standardized_feasibility`, `gate_safe` | Toxicity qualification probabilities, standardized margins and strict criterion mask |
| `candidate_mask`, `safe` | Protocol eligibility and its intersection with the criterion mask |
| `tmse` | Pointwise toxicity-boundary score |
| `restrict(scores)` | Apply the common masks; the decorator already does this |
| `normalize_safe(scores)` | Range-normalize over candidates passing both masks |
| `me`, `le`, `mt`, `lt` | Efficacy and toxicity GPs and likelihoods |
| `g_dagger`, `gamma` | Mean-toxicity limit and probability cutoff (`tau` in the paper) |
| `r_k`, `step`, `rng` | Cohort size, adaptive-cohort index and policy RNG |

Only `ctx` is passed to the callback. To compare parameter values, register a
separate named rule for each value. The complete example is
[`add_your_own_acquisition.py`](../examples/add_your_own_acquisition.py).

## Add a response surface

A surface factory maps a stratum `z` to a dictionary with:

| Key | Value |
|---|---|
| `eff`, `tox` | Functions `(d1, d2) -> float` giving true mean responses |
| `gd` | Mean-toxicity limit |
| `dopt`, `fopt` | Declared constrained reference point and its efficacy |
| `sf`, `sg` | Outcome SDs at `kap=1` |

The package checks these values and builds a square `grid_n × grid_n` grid,
with `grid_n=5` by default. See
[`add_your_own_surface.py`](../examples/add_your_own_surface.py) for registration
and a small comparison. `kap` scales both outcome SDs for a new simulation;
the paper's analysis-only toxicity-SD sensitivity is a separate experiment.

## Supported inputs

The current grid supports exactly two agents with equal numbers of dose levels.
Arbitrary candidate sets, a one-agent grid, additional agents and longitudinal
regimens require implementation changes. Holding one coordinate constant still
creates duplicate points on the square grid and is not a one-agent interface.

The GP kernel and likelihood have no plug-in interface. Use a new output file
after changing a custom callback: checkpoint matching checks the package source,
not the callback code or its captured values.
The common fallback and stopping rules are described in
[Scope and limitations](scope_and_limitations.md).
