# Paired aggregate-OC cKG fantasy sensitivity

This analysis is **post hoc** and concerns numerical stability, not accuracy,
Monte Carlo convergence, or a universally adequate fantasy count. The two OSA
strata are averaged within each of 200 paired trial seeds. Intervals are
pointwise 95% paired t Monte Carlo intervals and are not multiplicity-adjusted.
The 1,200 reused reference rows and 800 new executions are computational
diagnostics and contribute zero rows to the 55,480 logical trial-record inventory.
Trial seeds are the only random unit. Results are conditional on the two
evaluated fixed fantasy banks; two banks do not estimate a between-bank
seed distribution.

## Direct numerical contrasts

| 1024 arm minus production cKG-512 | Recommended efficacy | Threshold-exceeding final recommendation (pp) | Above-boundary assignments (pp of 40) | Dose-location error |
|---|---:|---:|---:|---:|
| cKG-1024-seed0 | 0.096 [-0.254, 0.447] | -7.250 [-11.637, -2.863] | 1.988 [0.859, 3.116] | -0.053 [-0.141, 0.035] |
| cKG-1024-seed1 | -0.014 [-0.308, 0.279] | 0.500 [-3.131, 4.131] | -0.275 [-0.853, 0.303] | 0.021 [-0.049, 0.091] |

At the same 1,024-fantasy count, seed1 minus seed0 was: -0.111 [-0.464, 0.243]; 7.750 [3.669, 11.831]; -2.263 [-3.250, -1.275]; 0.074 [-0.024, 0.172] (efficacy; terminal pp; assignment pp; dose error).

Recommendation rate was 100% and stop rate was 0% for every arm because
the full-panel `lhs_fixed` design has no early-stopping branch. These are
design facts, not empirical evidence of fantasy-count stability.

## Interpretation

The boundary-related operating characteristics were not uniformly stable.
For seed0, doubling from 512 to 1,024 changed threshold-exceeding final
recommendations by -7.25 percentage points and above-boundary assignments
by +1.99 percentage points, with both pointwise intervals excluding zero.
The two 1,024 banks also differed on both outcomes. These evaluated runs
therefore do not support describing the aggregate OCs as stable or converged.

Every comparator direction whose production interval excluded zero remained
interval-supported under both 1,024 arms. Among production-unresolved cells,
seed0 produced a new nominal positive assignment separation versus cEI; the
other evaluated intervals remained unresolved. This is directional
preservation evidence, not evidence that the absolute OCs are stable.

## Primary comparator-direction audit

| Contrast audited | cKG implementation | Estimate [95% interval] | Production interval | Current interval status |
|---|---|---:|:---:|:---:|
| Recommended true efficacy versus cEI | cKG-512-seed0 | 0.872 [0.432, 1.311] | excluded zero | supported positive direction |
| Recommended true efficacy versus cEI | cKG-1024-seed0 | 0.968 [0.583, 1.353] | excluded zero | supported positive direction |
| Recommended true efficacy versus cEI | cKG-1024-seed1 | 0.858 [0.448, 1.267] | excluded zero | supported positive direction |
| All-trial threshold-exceeding final recommendation versus cEI | cKG-512-seed0 | 12.250 [7.283, 17.217] | excluded zero | supported positive direction |
| All-trial threshold-exceeding final recommendation versus cEI | cKG-1024-seed0 | 5.000 [0.749, 9.251] | excluded zero | supported positive direction |
| All-trial threshold-exceeding final recommendation versus cEI | cKG-1024-seed1 | 12.750 [8.010, 17.490] | excluded zero | supported positive direction |
| Above-boundary simulated assignments versus cEI | cKG-512-seed0 | 0.362 [-1.181, 1.906] | included zero (unresolved) | unresolved |
| Above-boundary simulated assignments versus cEI | cKG-1024-seed0 | 2.350 [0.940, 3.760] | included zero (unresolved) | new nominal positive separation |
| Above-boundary simulated assignments versus cEI | cKG-1024-seed1 | 0.087 [-1.370, 1.545] | included zero (unresolved) | unresolved |
| Dose-location error versus cEI | cKG-512-seed0 | -0.207 [-0.317, -0.097] | excluded zero | supported negative direction |
| Dose-location error versus cEI | cKG-1024-seed0 | -0.260 [-0.362, -0.158] | excluded zero | supported negative direction |
| Dose-location error versus cEI | cKG-1024-seed1 | -0.187 [-0.294, -0.079] | excluded zero | supported negative direction |
| Recommended true efficacy versus specified 1:1 cEI-tMSE | cKG-512-seed0 | 0.415 [0.016, 0.814] | excluded zero | supported positive direction |
| Recommended true efficacy versus specified 1:1 cEI-tMSE | cKG-1024-seed0 | 0.511 [0.153, 0.870] | excluded zero | supported positive direction |
| Recommended true efficacy versus specified 1:1 cEI-tMSE | cKG-1024-seed1 | 0.401 [0.020, 0.781] | excluded zero | supported positive direction |
| All-trial threshold-exceeding final recommendation versus specified 1:1 cEI-tMSE | cKG-512-seed0 | 15.250 [10.448, 20.052] | excluded zero | supported positive direction |
| All-trial threshold-exceeding final recommendation versus specified 1:1 cEI-tMSE | cKG-1024-seed0 | 8.000 [3.781, 12.219] | excluded zero | supported positive direction |
| All-trial threshold-exceeding final recommendation versus specified 1:1 cEI-tMSE | cKG-1024-seed1 | 15.750 [10.980, 20.520] | excluded zero | supported positive direction |
| Above-boundary simulated assignments versus specified 1:1 cEI-tMSE | cKG-512-seed0 | -7.013 [-8.339, -5.686] | excluded zero | supported negative direction |
| Above-boundary simulated assignments versus specified 1:1 cEI-tMSE | cKG-1024-seed0 | -5.025 [-6.280, -3.770] | excluded zero | supported negative direction |
| Above-boundary simulated assignments versus specified 1:1 cEI-tMSE | cKG-1024-seed1 | -7.287 [-8.566, -6.009] | excluded zero | supported negative direction |
| Dose-location error versus specified 1:1 cEI-tMSE | cKG-512-seed0 | -0.002 [-0.114, 0.109] | included zero (unresolved) | unresolved |
| Dose-location error versus specified 1:1 cEI-tMSE | cKG-1024-seed0 | -0.055 [-0.158, 0.047] | included zero (unresolved) | unresolved |
| Dose-location error versus specified 1:1 cEI-tMSE | cKG-1024-seed1 | 0.018 [-0.092, 0.129] | included zero (unresolved) | unresolved |

## Why the original 20 seeds were not enough

Across non-degenerate direct contrasts, first-20 pointwise intervals were a median 2.95 times as wide as the 200-seed intervals. The first-20 subset is retained for transparency, but it was not a sufficiently precise basis for the requested aggregate-OC check.

No equivalence margin was prespecified. Therefore, an interval containing zero
must not be translated into formal equivalence. The defensible interpretation
depends on the direct effect-size bounds and whether the manuscript's named
comparator directions remain intact.
