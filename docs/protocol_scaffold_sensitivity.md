# Historical dose-availability sensitivity

This secondary three-policy analysis uses 4,800 saved OSA rows: two protocols,
two strata, cEI/cKG/cEI–tMSE, `tau=0.7,0.9` and 200 replicate identifiers.
Half the rows reuse full-grid main-study records; half are gradual runs.
The current manuscript compares four rules; the hybrid remains a historical
schedule sensitivity. The unchanged plan is linked [here](protocol_scaffold_sensitivity_prespec.md).

Full-grid runs contain four initialization observations and 36
participant assignments. Gradual runs start with two participants at `(0,0)`,
open the grid on a fixed schedule and can stop early. Details are in
[Settings and scope](scope_and_limitations.md).

## cKG minus the cEI–tMSE hybrid under gradual availability

Intervals use estimate ± 1.96 replicate-level Monte Carlo SE. Efficacy and
reference-point distance use pairs with a selection under both policies;
above-limit final selection includes all initiated trials.

| `tau` | Outcome | Estimate | 95% MC interval | Eligible pairs / replicates |
|---:|---|---:|---:|---:|
| 0.7 | True mean efficacy at final selection, among trials with a selection | -0.3559 | [-0.7910, 0.0791] | 385 / 200 |
| 0.7 | Probability that the final selection exceeds the toxicity limit, all initiated trials | +10.25 pp | [5.63, 14.87] | 400 / 200 |
| 0.7 | Legacy reference-point distance among final selections | +0.1962 | [0.0808, 0.3116] | 385 / 200 |
| 0.9 | True mean efficacy at final selection, among trials with a selection | -0.2562 | [-0.6798, 0.1674] | 322 / 180 |
| 0.9 | Probability that the final selection exceeds the toxicity limit, all initiated trials | +2.50 pp | [0.12, 4.88] | 400 / 200 |
| 0.9 | Legacy reference-point distance among final selections | +0.0935 | [-0.0251, 0.2120] | 322 / 180 |

## Secondary contrasts at `tau=0.7`

| Outcome | Estimate | MCSE | 95% MC interval | Eligible pairs / replicates |
|---|---:|---:|---:|---:|
| Stopping (pp) | +1.75 | 0.895 | [-0.004, 3.504] | 400 / 200 |
| Legacy post-initialization assignments to above-threshold combinations | -2.610 | 0.253 | [-3.1065, -2.1135] | 400 / 200 |
| Sample size | -0.495 | 0.252 | [-0.9894, -0.0006] | 400 / 200 |

The legacy post-initialization count excludes the two starting participants.
Both rules start below the limit, so their above-limit count contrast equals
the all-enrollment contrast. Stopping rates for cEI, cKG and the hybrid are
1.0%, 3.0% and 1.25% at `tau=0.7`, and 8.25%, 16.75% and 11.5% at `tau=0.9`.

## Change in the contrast between protocols

The interaction is `(cKG - hybrid)_gradual - (cKG - hybrid)_full_grid`.

| `tau` | Outcome | Interaction | 95% MC interval | Eligible pairs / replicates |
|---:|---|---:|---:|---:|
| 0.7 | Final-selection efficacy among selections | -0.7643 | [-1.3231, -0.2054] | 385 / 200 |
| 0.9 | Final-selection efficacy among selections | -0.2867 | [-0.8246, 0.2511] | 322 / 180 |
| 0.7 | Final selection above toxicity limit, percentage points | -2.50 | [-8.74, 3.74] | 400 / 200 |
| 0.9 | Final selection above toxicity limit, percentage points | +0.75 | [-2.96, 4.46] | 400 / 200 |
| 0.7 | Legacy reference-point distance among selections | +0.2135 | [0.0604, 0.3666] | 385 / 200 |
| 0.9 | Legacy reference-point distance among selections | +0.0721 | [-0.0763, 0.2205] | 322 / 180 |

The protocols jointly change initialization, availability, repeat eligibility,
stopping and enrollment. The interactions therefore describe that combined
change. Reference-point distance is a legacy continuous-domain diagnostic,
not distance to the actual finite-grid target.

[Detailed tables and figure](https://github.com/clairexinzhuwang/dose-combination-bo/blob/main/results/protocol_scaffold_analysis/protocol_scaffold_sensitivity.md)
and [saved records](https://github.com/clairexinzhuwang/dose-combination-bo/blob/main/results/protocol_scaffold_sensitivity.json) retain the
complete results and provenance.
