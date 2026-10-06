# Fixed-data computation benchmark

The saved timings below were measured on 1 October 2026 with version 0.2.1rc4, before the package was renamed. They use three fixed OSA group-0 datasets and are not trial runtimes. The current package is `dose-combination-bo` (Python import: `dose_combination_bo`), version 0.2.1rc5. The rename does not change the calculations; the saved timing files have been kept as recorded.

On an Apple M4 Mac mini with 16 GB RAM, using one numerical/PyTorch thread:

| Records | Two-GP fit (s) | cKG (ms) | tMSE (ms) | Entropy (ms) | cEI (ms) |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.136 | 112.024 | 0.423 | 12.905 | 14.951 |
| 20 | 0.136 | 104.681 | 0.446 | 13.033 | 14.865 |
| 40 | 0.147 | 113.949 | 0.421 | 13.212 | 15.085 |

Scoring times are medians of five repetitions after one warm-up, including fresh context and gate construction. Fitted-GP prediction caches remain warm; score vectors are not reused. Fitting times are single measurements of both response models after a separate warm-up pair. All rules share the fitted models within each dataset.

The fixed plan uses a 25-combination grid, tau 0.7, cohort size two, and seed 20261001. The smaller datasets are prefixes of the same 40 generated observations. Each GP uses 120 Adam steps at learning rate 0.1 and the specified individual-outcome variance. The gate admits 25, 10 and 12 combinations at the three record counts. All 72 score-vector calls passed; repeated vectors were identical. These are fixed-data timings, not full-trial runtime estimates.

## Run the same timing check

Install the current wheel and its dependencies in a separate environment. From outside the source folder, run:

```sh
python /path/to/benchmark_installed.py --wheel /path/to/dose_combination_bo-0.2.1rc5-py3-none-any.whl --output new_run
```

The script checks the installed package against the supplied wheel, saves the plan and observations before fitting, and stops after 300 seconds. It does not overwrite an existing result. Add `--repo /path/to/dose-combination-bo` to compare installed modules with the source. Elapsed times will vary by machine and load.

`benchmark_plan.json` records the fixed design; `fixed_observations.json` contains its data; `benchmark_results.json` contains versions, fitted posteriors, scores and individual timings; `timing_summary.csv` contains the displayed summaries. The original run took 4.87 seconds wall time. Environment: Python 3.12.7, NumPy 2.1.3, SciPy 1.15.3, PyTorch 2.10.0, GPyTorch 1.15.2, macOS 15.0 arm64. No GPU or parallel workers were used.
