# How cKG is calculated

cKG scores a cohort by the expected change in the posterior mean efficacy of
the combination chosen by the final-selection rule. That rule selects the best
qualifying combination, or the one with the largest standardized toxicity
margin if none qualifies.

`dose_combination_bo.ckg_one_step_gated_exact` evaluates one candidate;
`ckg_scores_exact(ctx)` evaluates all candidates and applies the assignment mask.
Both response models update hypothetically, with GP hyperparameters held fixed.
GPs are refitted after an observed cohort. The lookahead covers one cohort on
the currently available grid; it does not anticipate later opening or stopping.

## Calculation

For candidate query `d`, the independent efficacy and toxicity updates have form

```text
mu_e,i + b_e,i Z_e     and     mu_g,i + b_g,i Z_g,
```

where `Z_e` and `Z_g` are independent standard normals. The evaluator:

1. partitions the toxicity-observation axis where combinations enter or leave
   the qualifying set;
2. on each interval with a nonempty set, integrates the upper envelope of the
   qualifying affine efficacy updates;
3. where the set is empty, partitions the upper envelope of standardized
   toxicity margins to identify the fallback; and
4. integrates over the toxicity observation and returns `E[V(post)] - V(now)`.

Under the finite-grid Gaussian model, this is analytic integration up to
floating-point error. Current and updated latent toxicity variances have a
`1e-12` lower bound before standardization. Equal-slope merging uses exact
floating-point equality; tail and narrow-interval probabilities use stable
log-domain calculations. See [normal interval probabilities](normal_interval_stability.md).

The score can be negative when a toxicity update removes valuable combinations
from the qualifying set. It is used with its sign: if all eligible scores are
negative, cKG selects the least expected decline.

## Toxicity criterion and candidate sets

The strict criterion is

```text
z_i = (g_dagger - mu_g,i) / s_g,i > Phi_inverse(tau).
```

`Phi(z_i)` is the posterior probability that **mean toxicity** is at or below its
limit. An empty qualifying set invokes `argmax z_i`. A fallback selection fails
the criterion.

Current and hypothetical values use the available grid `G_t`, including untried
combinations. Allocation is restricted to protocol-eligible candidates `E_t`;
`ctx.restrict` applies that mask. The simulator requires no minimum
exposure before final selection. [Settings and scope](scope_and_limitations.md)
describes allocation, terminal selection and stopping.

## Implementation and checks

All formal cKG simulations in the paper used this deterministic evaluator.
The public name `cKG` calls `ckg_scores_exact`. The earlier Monte Carlo scorer
remains under `cKG-legacy512` (`cKG1fix` is its compatibility alias).
The final-selection objective here differs from
[Chen et al.'s c-KG](https://doi.org/10.48550/arXiv.2101.08743) and the
[nonnegative constrained KG of Ungredda and Branke](https://doi.org/10.1145/3641544).

The [saved validation panel](https://github.com/clairexinzhuwang/dose-combination-bo/blob/main/results/ckg_piecewise_exact_validation/ckg_piecewise_exact_summary.md)
contains 720 posterior states and 18,000 query scores. A 64-state subset was
checked with eight independently scrambled Sobol integrations per state,
131,072 points each; the maximum absolute discrepancy was `9.39845e-5`.
Including 28 numerical sentinels, standardized-margin and probability-based
comparisons agreed on qualification and fallback in all 748 states.

Maintained tests cover empty sets, crossings, close slopes, tails, subnormal
masses, adjacent thresholds, invariances, signed values, cohort noise and
candidate masks. The [README](../README.md) gives the test commands.
To rebuild the historical panel from the complete research archive, enter its
`study/` directory and run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
  python paper/run_ckg_piecewise_exact_validation.py
```
