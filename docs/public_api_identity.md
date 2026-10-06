# Rule and scenario names

Install `dose-combination-bo`, import `dose_combination_bo` in Python, and run
`dose-combination-bo` in the terminal.

Use `cKG`, `cEI`, `tmse` and `entropy` in new code. Saved records keep their
original names.

| Rule | Older or alternative names |
|---|---|
| `cKG` | Deterministic evaluator; historical runs use `cKG-exact-formal` |
| `cEI` | Feasibility-weighted expected improvement |
| `tmse` | Historical runs use `tMSE-only`; `straddle` is deprecated |
| `entropy` | `feasibility-entropy` is an alias; `qBIG` is retained and `SUR` is deprecated |
| `cKG-legacy512` | Former 512-draw Monte Carlo evaluator; alias `cKG1fix` |
| `cEI-tMSE` | Fixed alternation used in historical sensitivity analyses |

The paper compares the first four. Entropy learns which available combinations
meet the mean-toxicity limit. See the [calculation](qbig_sur_approximation.md).

| Scenario | Meaning |
|---|---|
| `osa` | Synthetic OSA-derived surface |
| `mariposa` | Synthetic MARIPOSA-motivated surface |
| `logistic`, `efftox` | Synthetic continuous logistic surface; `efftox` is the historical name |
| `gbump` | Synthetic Gaussian-bump surface |

See [Reproduction](reproducibility.md) for how the current package relates to
the saved study results.
