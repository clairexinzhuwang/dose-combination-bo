# Dose availability and stopping sensitivity

This page is generated from the complete OSA sensitivity records. It compares a design with all combinations available after initialization with a Willard-inspired gradual-expansion design and a common feasibility-stopping rule; it is not a reconstruction of the source design algorithm.
According to the local development record, the analysis specification was finalized before the complete performance results were inspected; no externally timestamped pre-run commit is available.

## Analysis specification

- 200 independent simulation replicates; each includes two strata, and the three policies are compared within replicate using common random numbers.
- Gate thresholds tau = 0.7 (representative intermediate value) and 0.9 (stricter sensitivity); neither is a clinical toxicity cutoff.
- Maximum sample size 40; cohort size 2; 5x5 candidate dose-combination grid.
- With all combinations available after initialization, four initial assignments leave 18 scheduled adaptive-cohort positions: 9 cEI and 9 tMSE. With two initialization assignments, a nonstopped maximum-length gradual trial has 19 scheduled positions: 10 cEI and 9 tMSE; stopping can truncate that sequence. The gradual design is not labelled 1:1.
- An empty eligible gate can invoke the common greatest-feasibility fallback instead of the scheduled acquisition.
- Gradual-expansion region: d1 + d2 <= 0.25 q; no repeat while an unvisited candidate remains before full expansion.
- The first two consecutive empty gates use the most-feasible eligible dose; the third stops before allocation and is recorded as a feasibility stop.

## Result

**Under gradual expansion in the OSA simulation, at threshold 0.7, the recommended-efficacy interval included zero, cKG had more true-boundary-exceeding final recommendations and the interval excluded zero, and cKG had greater dose-location error and the interval excluded zero; at threshold 0.9, the recommended-efficacy interval included zero, cKG had more true-boundary-exceeding final recommendations and the interval excluded zero, and the dose-location-error interval included zero.**

At tau = 0.7, the cKG minus author-specified cEI–tMSE schedule within-replicate differences (averaging eligible paired OSA strata within each simulation replicate) were:

| Dose-availability design | Efficacy | True-boundary-exceeding final recommendation (percentage points) | Dose-location error (dose units) |
|---|---:|---:|---:|
| All combinations available | 0.448 | 12.8 | -0.02 |
| Gradual expansion | -0.356 | 10.2 | 0.20 |

## Principal and key secondary paired contrasts

The contrast is cKG minus cEI–tMSE under gradual expansion, tau = 0.7. Monte Carlo intervals are mean +/- 1.96 Monte Carlo SE computed across independent simulation replicates; no p-value was used to retain or discard an analysis.

| Outcome | Mean | MCSE | 95% MC interval | Paired trials | Simulation replicates |
|---|---:|---:|---:|---:|---:|
| Recommended efficacy among recommendations | -0.3559 | 0.2220 | [-0.7910, 0.0791] | 385 | 200 |
| True-boundary-exceeding final recommendation, all trials | 0.1025 | 0.0236 | [0.0563, 0.1487] | 400 | 200 |
| Dose-location error (dose units) among recommendations | 0.1962 | 0.0589 | [0.0808, 0.3116] | 385 | 200 |
| Feasibility stop (incorrect under OSA truth) | 0.0175 | 0.0090 | [-0.0000, 0.0350] | 400 | 200 |
| Post-initialization above-boundary simulated assignments | -2.6100 | 0.2533 | [-3.1065, -2.1135] | 400 | 200 |
| Sample size | -0.4950 | 0.2522 | [-0.9894, -0.0006] | 400 | 200 |

## Protocol interactions

Each interaction is the change in the cKG minus cEI–tMSE contrast when moving from the all-combinations-available design to gradual expansion.

| tau | Outcome | Interaction | MCSE | 95% MC interval | Paired trials | Simulation replicates |
|---:|---|---:|---:|---:|---:|---:|
| 0.7 | Recommended efficacy among recommendations | -0.7643 | 0.2851 | [-1.3231, -0.2054] | 385 | 200 |
| 0.9 | Recommended efficacy among recommendations | -0.2867 | 0.2744 | [-0.8246, 0.2511] | 322 | 180 |
| 0.7 | True-boundary-exceeding final recommendation, all trials | -0.0250 | 0.0319 | [-0.0874, 0.0374] | 400 | 200 |
| 0.9 | True-boundary-exceeding final recommendation, all trials | 0.0075 | 0.0189 | [-0.0296, 0.0446] | 400 | 200 |
| 0.7 | Dose-location error (dose units) among recommendations | 0.2135 | 0.0781 | [0.0604, 0.3666] | 385 | 200 |
| 0.9 | Dose-location error (dose units) among recommendations | 0.0721 | 0.0757 | [-0.0763, 0.2205] | 322 | 180 |

Recommendation-conditional outcomes exclude stopped trials and report their own denominators in the machine-readable OC table. Stopped trials are never assigned a fictitious efficacy or dose-location-error value. Because feasible doses exist on the OSA truth, every feasibility stop is an incorrect operational stop under the simulation truth rather than evidence of clinical infeasibility.
