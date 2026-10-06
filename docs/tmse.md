# How tMSE chooses a combination

tMSE favors combinations whose mean toxicity is uncertain and close to the
limit. It learns about that boundary without using efficacy. The method's
name stands for targeted mean squared error.

Use `dose_combination_bo.run_trial("tmse", ...)` in Python, or
`--acquisitions tmse` with `dose-combination-bo run`.
Assignment uses the same eligibility restrictions, toxicity criterion and
fallback rule as the other methods.

## Calculation

```text
alpha_tMSE(d) = sigma_g(d) exp{-z_g(d)^2 / 2},
z_g(d) = {g_dagger - mu_g(d)} / sigma_g(d).
```

`d` is a dose combination; `mu_g` and `sigma_g` are the posterior mean and
standard deviation of its mean toxicity. `g_dagger` is the toxicity limit.
The score rises with uncertainty and falls as the posterior mean moves away
from the limit. It is proportional to the zero-bandwidth tMSE criterion
used in the paper; the omitted positive constant does not change rankings.
tMSE does not calculate how a new cohort would change the final selection.
