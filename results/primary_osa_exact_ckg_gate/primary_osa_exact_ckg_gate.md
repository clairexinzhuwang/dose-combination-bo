# Exact-cKG OSA primary operating-characteristic review

The authenticated 400-cell primary result is complete: 200 paired trial seeds
and two OSA strata at tau = 0.7. The two strata are averaged within seed;
trial seed is the independent Monte Carlo unit. Values are estimate [pointwise
95% t Monte Carlo interval]; difference columns use paired seed-level
contrasts. Intervals are unadjusted.

| Metric | Exact cKG | Exact - cKG-512 | Exact - cEI | Exact - specified 1:1 cEI-tMSE |
|---|---:|---:|---:|---:|
| Recommended true efficacy (OSA efficacy units) | 7.586 [7.193, 7.978] | 0.033 [-0.334, 0.401] | 0.905 [0.505, 1.305] | 0.448 [0.061, 0.835] |
| Above-threshold final recommendation (percentage points) | 19.500 [15.437, 23.563] | -2.500 [-7.330, 2.330] | 9.750 [5.183, 14.317] | 12.750 [8.010, 17.490] |
| Above-boundary assignments (assignments out of 40) | 7.567 [7.086, 8.049] | 0.695 [0.351, 1.039] | 0.840 [0.248, 1.432] | -2.110 [-2.605, -1.615] |
| Dose-location error (panel dose units) | 1.694 [1.589, 1.798] | -0.022 [-0.114, 0.069] | -0.230 [-0.332, -0.127] | -0.025 [-0.133, 0.083] |
| Recommendation rate (percentage points) | 100.000 [100.000, 100.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## Inference limits

The cKG-512 column is the direct numerical-change comparison. The cEI and
specified 1:1 cEI-tMSE columns are contextual policy contrasts. For
`rec_true_eff` and `dose_units`, a paired contrast is defined only where
both policies recommend; the JSON records the eligible pair counts. For
`rec_unsafe`, a trial without a recommendation would count as zero; `toxic`
is the number of above-boundary assignments out of 40; recommendation rate
uses all trials.

No numerical or clinical equivalence margins were prespecified. Therefore,
a pointwise interval containing zero is not evidence of equivalence or
convergence, and this analysis does not automatically authorize the broader
formal run. That decision requires scientific review of the estimates, MCSEs,
and intervals in the JSON artifact.

## Provenance gate

PASS: canonical 400-row coverage, current `package-pf-stable-z-v2` harness,
exact evaluator, zero stable-z-versus-current-Context pass-set or
operational-fallback differences, historical CDF rounding ties retained
as diagnostics, terminal gate and recommendation audits, source hashes,
manifest hash, exact-validation hash, and the
authenticated frozen OSA reference were all checked before analysis.
