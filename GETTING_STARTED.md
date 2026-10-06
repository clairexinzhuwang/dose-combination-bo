# Software guide

Start with the [installation and example](README.md#install-and-run).
[What the four rules do](docs/objective_alignment.md) explains cKG, cEI, tMSE
and entropy in one place. All four use the same models and trial settings.

## Settings and output

In the Python example, `gamma` is the required chance, according to the model,
that a combination's mean toxicity is below the limit. The paper calls it
`tau`. `z` chooses patient group 0 or 1. With the default settings, `budget=12`
means four simulated starting observations followed by eight participant
assignments. More settings are described [here](docs/scope_and_limitations.md).

| Output | Meaning |
|---|---|
| `selection_status` | Whether the final choice met the toxicity requirement, used the fallback, or the trial stopped without a choice |
| `rec_true_eff`, `rec_true_tox` | Simulated mean efficacy and toxicity at the final choice |
| `rec_passed_criterion` | Whether the final choice met the model's toxicity requirement |
| `rec_direct_observations` | Number of observations at the chosen combination |
| `grid_dose_units` | Distance from the best acceptable combination on the dose grid, measured in grid steps |

When none meets the requirement, the fallback chooses the combination most
likely to meet the mean-toxicity limit according to the model. This does not
mean it is known to be safe. Stopped trials make no choice. Full output
definitions are [here](docs/reporting.md).

## Add your own rule or scenario

After installation, run either example from the repository directory:

```bash
python examples/add_your_own_acquisition.py
python examples/add_your_own_surface.py
```

Each adds a rule or a dose-response scenario and runs a small comparison.
[Adding your own code](docs/extending.md) explains how they work.
For parallel runs, install `.[parallel]` and omit `--serial`.

## Check the results and software

The paper's results were generated with the original study code.
[Checking the paper's results](docs/reproducibility.md) explains which records
are included and how to run the analysis scripts. The saved summaries are
described [here](docs/extended_results.md).

To run the software tests:

```bash
python -m pip install -c requirements-lock.txt -r requirements-test.txt ".[dev]"
python -m pytest -q tests
python -m pytest -q verification/test_numerics_and_controller.py
```

See the [test results](ACCEPTANCE_RESULT.md), [version changes](CHANGELOG.md)
and [release status](RELEASE.md). The model and calculation details are linked
from the page explaining the four rules.
