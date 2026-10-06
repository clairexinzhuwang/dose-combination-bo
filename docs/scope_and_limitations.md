# Settings and scope

The package compares four allocation rules in two-agent simulations with
continuous efficacy and toxicity. Within each protocol, the rules share GP
models, dose restrictions, stopping and final selection.

## Grid and protocols

The default grid contains `5 × 5` standardized dose pairs in `[0,1]^2`. Zero
labels the lowest dose; it need not mean placebo.

| Protocol | Initialization | Budget |
|---|---|---|
| `lhs_fixed` | Four observations generated at continuous dose locations | 40 records: four initialization observations plus 36 assignments in 18 two-person cohorts; full grid, no early stop |
| `start_low_expansion` | Two participants at `(0,0)` | Up to 40 participants, including the first cohort; gradual availability and possible early stop |

Cohort `q` in the gradual protocol can use combinations with
`d1 + d2 <= 0.25*q`; the full grid opens at `q=8`. Before then, visited
combinations are excluded while an available unvisited combination remains.
The schedule is fixed, not guided by outcomes. Initialization, repeat assignments
and stopping also differ between protocols, so their comparison cannot isolate
the effect of opening doses gradually.

## Toxicity, fallback and stopping

A combination qualifies when
`(g_dagger - mu_g)/sd_g > Phi^{-1}(gamma)`, where `gamma` is the paper's `tau`.
This is a probability criterion for unknown mean toxicity, not an individual's
adverse-event risk. A larger cutoff makes the criterion stricter.

If no eligible combination qualifies, fallback selects the one with the largest
standardized toxicity margin. In the gradual protocol, the first two consecutive
failures use fallback; the third stops before assignment. A qualifying assignment
resets the count.

At completion, choose the available qualifying combination with the highest
posterior mean efficacy. If none qualifies, use fallback. The selected combination
may be untried. A stopped trial returns `stopped_no_selection` and recommends
nothing. Short gradual runs can select only combinations already opened.

## Scenarios

- `osa`: a synthetic OSA-derived scenario with separate efficacy surfaces and
  toxicity limits for two strata; the toxicity surface is shared.
- `mariposa`: a synthetic surface motivated qualitatively by published arm
  summaries.
- `logistic` (historical name `efftox`): increasing efficacy and toxicity, with
  a small acceptable region and its optimum near the toxicity limit.
- `gbump`: Gaussian-shaped efficacy peaks and nonmonotone toxicity. The lowest
  dose exceeds the mean-toxicity limit in both strata.

Each scenario has at least one acceptable grid combination. Outcome scales
and SDs are synthetic. The OSA strata have separate trials with no borrowing
between them.

## Clinical use

A clinical protocol would need justified outcomes, toxicity limits,
initialization and safety rules, followed by evaluation of the complete design.
The simulations use continuous Gaussian responses, fixed outcome variances and
independent GPs. They omit delayed or missing outcomes, permanent dose
elimination, independent safety monitoring, minimum efficacy or exposure
requirements, and a randomized follow-up stage.

Report efficacy, allocation, final-selection toxicity, fallback and stopping
together. See [Output and reporting](reporting.md) for denominators and the
[GP model](gp_surrogate.md) for assumptions.
