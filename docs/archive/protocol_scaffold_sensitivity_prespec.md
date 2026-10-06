# Prespecified dose-availability and stopping sensitivity

> **Historical specification.** This plan retains its original scientific
> design; only the terminology has been updated. The current rerun uses
> deterministic cKG and implements its fallback by
> choosing the eligible candidate with the largest standardized toxicity score
> `z` (equivalently, the highest posterior probability of meeting the toxicity
> rule), and current numerical results are reported in
> [`protocol_scaffold_sensitivity.md`](protocol_scaffold_sensitivity.md). The
> specified comparator is the schedule now described as the cEI–tMSE hybrid
> (frozen execution label `cEI-tMSE`); serialized IDs below are retained for
> provenance.
>
> **Current terminology.** The documentation calls that frozen comparator the
> *cEI–tMSE hybrid* and denotes the independent
> simulation-replicate count by \(M\). Here `tau` is the required posterior
> probability that mean toxicity does not exceed the prespecified threshold;
> larger values make this model-based toxicity rule more stringent, but are not
> clinically calibrated safety probabilities. If no eligible combination meets that
> requirement, the prespecified fallback chooses the eligible combination with
> the greatest such posterior probability. cEI and tMSE are
> literature-based acquisition criteria, but their pairing and frozen cohort
> schedule are not a published design or optimal-allocation rule. Policies use
> common random numbers within each replicate. Current prose uses *all candidate
> combinations available* and *gradual expansion*, and reports *participant
> allocation* and *final recommendation*.

Status: **the local development record shows that this plan was finalized
before the complete performance results were inspected**

No externally timestamped pre-run commit is available.
Freeze date: 2026-08-18  
Scope: OSA surface only; this is a dose-availability sensitivity analysis, not a new
acquisition search and not a reproduction of a complete clinical design.

## Question

How do true mean efficacy at recommendation, above-threshold final
recommendations, and OBD dose-location error compare when the combinations
available for allocation expand gradually rather than all being available from
the start?

Within a dose-availability design, the working response and observation models,
observed-data updates, model-based toxicity rule, eligible dose set,
initialization, cohort size, maximum budget, stopping rule, terminal
recommendation rule, and seed-indexed outcome draws are common across policies;
only the next-cohort acquisition score changes.

## Frozen design

Two dose-availability designs are compared. The code headings below are
serialized implementation values.

### All candidate combinations available (`lhs_fixed`)

- Four continuous numerical initialization observations: `d1` is stratified
  into four equal intervals and sampled within each, while `d2` is sampled
  independently and uniformly. These records initialize the response models;
  they are not participants.
- The full 5-by-5 candidate grid is available to every adaptive acquisition.
- Maximum total response records: 40 per stratum, leaving 36
  post-initialization participant assignments.
- Cohort size: 2.
- No early stopping.
- If no eligible combination meets the toxicity probability requirement, the
  prespecified fallback chooses the one with the greatest posterior probability
  that mean toxicity does not exceed the threshold.

### Gradual expansion (`start_low_expansion`)

- The first cohort contains two actual simulated participants assigned to `(0, 0)`.
- For cohort `q`, every policy is restricted to grid doses satisfying
  `d1 + d2 <= 0.25 * q`; the full grid is available at `q = 8`.
- Before all combinations become available, an already evaluated grid dose cannot be
  repeated while an unevaluated dose remains in the current region.
- The first and second consecutive occasions on which no eligible dose meets
  the toxicity probability requirement use the eligible dose with the greatest
  posterior probability that mean toxicity does not exceed the threshold.
- On the third consecutive occasion with no eligible dose meeting the toxicity
  probability requirement, the stratum stops before another
  cohort is assigned and returns the serialized stop code `NO_FEASIBLE_DOSE`
  (a feasibility stop with no recommendation).
- Once an eligible dose meets the requirement, the consecutive-fallback counter resets.
- Maximum enrollment is 40 simulated participants per stratum; cohort size remains 2.

This is a **Willard-inspired gradual-expansion sensitivity**. It deliberately
borrows only initialization at the lowest combination, gradual expansion, and a
common feasibility stop; it is not a
reproduction of a complete source design, and no acquisition-value stopping
rule is used.

## Frozen experiment matrix

- Surface: `osa` only.
- Strata: `0` and `1`.
- Policies: `cEI`, `cKG`, and `cEI-tMSE`.
- Required posterior probabilities for mean toxicity not to exceed the threshold:
  `tau = 0.7` (primary) and `tau = 0.9` (sensitivity).
- Dose-availability designs: all candidate combinations available and gradual
  expansion (serialized as `lhs_fixed` and
  `start_low_expansion`).
- Seeds: integers `0, ..., 199`.
- Toxicity probabilities use latent-function uncertainty (serialized as
  `mode: latent`).
- Analysis-model within-combination outcome SDs: fixed at their data-generating
  values.
- Limit: 40 response records in the all-available design and maximum enrollment
  of 40 participants under gradual expansion.
- Cohort size: 2.
- Grid: 5-by-5.

The factorial defines

`2 dose-availability designs x 2 strata x 3 policies x 2 tau values x 200 seeds = 4,800`
trial-record cells. In the archived analysis, 2,400 cells with all combinations
available reuse the corresponding corrected main-sweep records; 2,400 gradual-expansion cells are additional
simulations. This provenance note describes record reuse; it does not change the frozen
factorial or estimands.

No setting will be altered after examining results. In particular, the region
step, number of consecutive occasions without a dose meeting the toxicity
requirement before stopping, `tau` values, seed count,
and policy set will
not be retuned. No interim subset of seeds will be used to decide whether to
complete the prespecified 200 seeds.

## Estimands

In these simulations, an above-threshold combination is one whose true mean
toxicity exceeds the prespecified threshold; the term does not refer to an
observed toxicity event.

### Outcomes over all initiated trials

Every simulated trial remains in the denominator:

- probability that a recommendation is made;
- probability of a feasibility stop (serialized as `NO_FEASIBLE_DOSE`; a false
  stop on the OSA truth, which has feasible grid doses);
- probability of recommending an above-threshold combination at trial end;
- probability of recommending within one grid unit of the OBD;
- expected total response records or enrolled sample size, as appropriate for
  the dose-availability design;
- expected number of above-threshold records or participant assignments, with
  the denominator identified explicitly;
- initialization and post-initialization values reported separately: the
  all-available initialization values are numerical records, whereas the
  gradual-expansion starting cohort contains actual participants;
- number of unique combinations evaluated; and
- number of fallback events because no eligible dose met the toxicity rule.

### Recommendation-conditional outcomes

These are evaluated only when a recommendation exists:

- true mean efficacy at the recommended combination;
- distance from the OBD in grid-dose units; and
- probability that the recommended combination is above threshold.

The serialized stop outcome `NO_FEASIBLE_DOSE` is never assigned an artificial
dose, efficacy, distance, or
utility.

## Frozen comparisons

The primary analysis is at `tau = 0.7` under gradual expansion
(`start_low_expansion` in the serialized API). The paired within-replicate
contrast is `cKG - cEI-tMSE` for:

1. true mean efficacy at the recommended combination among paired trials in
   which both policies make a recommendation; and
2. the initiated-trial indicator for an above-threshold final recommendation.

Key secondary contrasts concern recommendation-conditional dose-location error
in grid-dose units, false-stop probability, post-initialization assignments to
above-threshold combinations, and enrolled sample size.

For outcome `r`, the dose-availability interaction is

`[(cKG - cEI-tMSE)_start_low] - [(cKG - cEI-tMSE)_lhs_fixed]`.

It is reported for true mean efficacy at recommendation, above-threshold final
recommendations, and dose-location error with a
seed-clustered paired standard error (and a paired bootstrap interval where
generated). Analyses are retained regardless of p-values.

## Interpretation fixed in advance

- A similar point-estimate direction across dose-availability designs was to be described,
  not treated as proof that the contrast is invariant to dose availability.
- A smaller contrast under gradual expansion was to be described as attenuation,
  not attributed causally to dose availability without interaction evidence.
- A reversal was to motivate further joint study of dose availability, stopping, and
  acquisition, not a universal acquisition rule.

The analysis retains all three outcomes; none triggers retrospective tuning.

## How the results are interpreted

The three qualitative categories above describe directions, not statistical
decision rules. No numerical magnitude threshold or uncertainty-based retention
rule was specified. Every planned estimate and Monte Carlo interval is therefore
retained, without using a sign alone as the headline interpretation across the
two dose-availability designs.
