# How cEI chooses a combination

cEI favors combinations that could improve efficacy, giving more weight to
those more likely to meet the mean-toxicity limit. It compares them with the
best current posterior mean efficacy among qualifying combinations.
Previously assigned combinations count in this comparison.

Use `dose_combination_bo.run_trial("cEI", ...)` in Python, or
`--acquisitions cEI` with `dose-combination-bo run`.
Assignment uses the same eligibility restrictions, toxicity criterion and
fallback rule as the other methods.

## Calculation

```text
alpha_cEI(d) = EI{mu_e(d), sigma_e(d); V_t} PF(d).
```

`d` is a dose combination; `mu_e` and `sigma_e` are the posterior mean and
standard deviation of its mean efficacy. `EI` is the expected positive
efficacy gain over `V_t`, the best posterior mean efficacy among currently
available qualifying combinations. If none qualifies, `V_t` is the posterior
mean efficacy at the available combination with the largest standardized
toxicity margin. `PF(d)` is the probability that mean toxicity is at or below
its limit; it is not an individual's
adverse-event probability. The fitted models and `V_t` are fixed for this
calculation. cEI does not calculate how a new cohort would change the final
selection.
