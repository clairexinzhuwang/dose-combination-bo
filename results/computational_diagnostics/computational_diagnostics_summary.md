# Computational diagnostics

These OSA diagnostics are **post hoc and descriptive**. They benchmark one captured
CPU/software environment and do not establish portable wall-clock performance.

## Timing

The two strata were averaged within each seed. Values are medians [Q1, Q3] across
10 seed averages; ratios are paired to cEI within seed.

| Variant | Acquisition s/trial | Total s/trial | Acquisition ratio to cEI | Total ratio to cEI |
|---|---:|---:|---:|---:|
| cEI | 0.3474 [0.3461, 0.3607] | 5.3233 [5.2802, 5.4228] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| tMSE | 0.0003 [0.0003, 0.0003] | 4.9590 [4.6989, 4.9875] | 0.0009 [0.0008, 0.0009] | 0.9208 [0.8969, 0.9325] |
| cEI-tMSE | 0.1722 [0.1642, 0.1726] | 5.1144 [4.8577, 5.1252] | 0.4934 [0.4725, 0.4986] | 0.9515 [0.9344, 0.9642] |
| cKG-128 | 0.6175 [0.5863, 0.6211] | 5.5800 [5.2856, 5.5848] | 1.7743 [1.6943, 1.7819] | 1.0419 [1.0061, 1.0504] |
| cKG-256 | 0.6712 [0.6679, 0.6863] | 5.6644 [5.4660, 5.7964] | 1.9381 [1.9224, 1.9717] | 1.0644 [1.0503, 1.0741] |
| cKG-512 | 0.7367 [0.7293, 0.7387] | 5.6875 [5.6210, 5.7306] | 2.0837 [2.0464, 2.1250] | 1.0558 [1.0369, 1.0740] |

Acquisition-only time is the registered score-callable boundary. Total time is the
entire trial, including GP fitting, allocation bookkeeping, final refits, and metrics.

## Fantasy-count stability

Agreement uses cKG-512 as the reference and averages the two strata within each of
20 seeds before reporting means and Monte Carlo SEs.

| Comparison | Same-state selected dose | Full path exact | Cohort-position agreement | Terminal recommendation | Unsafe status |
|---|---:|---:|---:|---:|---:|
| cKG-128 vs cKG-512 | 43.8% +/- 3.3% | 2.5% +/- 2.5% | 23.9% +/- 3.2% | 17.5% +/- 6.6% | 72.5% +/- 9.2% |
| cKG-256 vs cKG-512 | 57.6% +/- 4.2% | 2.5% +/- 2.5% | 36.8% +/- 3.6% | 35.0% +/- 9.0% | 82.5% +/- 7.5% |

Agreement measures numerical stability, not decision accuracy. Lower fantasy counts
use channel-wise prefixes of the production-compatible 512-sample Zf/Zg banks.

See `computational_diagnostics_raw.json` and its metadata sidecar for every timing
call, allocation path, same-state choice, and the captured hardware/software record.
