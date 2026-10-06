# Reproduction and testing

## Test the software

The [README](../README.md) gives installation, example and test commands. The
[verification report](../distributions/VERIFICATION_REPORT.md) records the tested
environment and results.

The maintained suite tests real GP fitting, the API and CLI, cache/resume,
trial control and Ray execution. The separate
`verification/test_numerics_and_controller.py` tests decisions using supplied
posteriors; it does not fit GPs. Both commands run their complete included
suites. Tests that need the full historical archive stay with that archive.

## Reanalyse saved records

The repository includes [34,000 selected trial records](../results/comparator_family_analysis/comparator_family_selected_records.json.zst).
The two scripts below read these records and saved summaries. They need NumPy,
SciPy and zstandard and do not fit GPs or simulate trials.

Run from the repository root:

```bash
python3 -m venv /tmp/dose-combination-bo-reanalysis-env
source /tmp/dose-combination-bo-reanalysis-env/bin/activate
python -m pip install numpy scipy zstandard
python paper/revision_analysis/20260930/derive_efficacy.py --output-dir scratch/reanalysis/efficacy
python paper/revision_analysis/20261006/derive_near_optimal.py --output-dir scratch/reanalysis/near_optimal
```

The efficacy script recalculates paired efficacy contrasts for all four
scenarios and efficacy among selecting trials under gradual availability. For
the enrollment/noise analysis, it checks the saved marginal summaries against
CSV and JSON copies; it does not reconstruct the missing paths or paired
intervals.

The near-optimal script recalculates 80 selection summaries from 8,000 OSA
records. It compares each selection with the best truly acceptable combination
on the 25-point grid, using efficacy margins of 0, 0.5, 1 and 2 events/hour.
These are descriptive margins, not clinical targets. See the
[analysis notes](../paper/revision_analysis/20261006/README.txt).

The commands write to `scratch/reanalysis/`, leaving `results/` unchanged.
Compare the outputs with the saved [efficacy](../paper/revision_analysis/20260930/)
and [selection-margin](../paper/revision_analysis/20261006/) files.
[Saved results](extended_results.md) identifies the files used by the paper.

## Full archive

The calibration indexes refer to 24,000 paths and 72,000 snapshots. The path
shards, complete formal projection and original source environments are kept
in a separate archive. They are needed to reconstruct all calibration contrasts
and saved posterior predictions.

`study_archive/frozen_core/` contains reference code for decision tests, not the
full archive. Public access to the complete archive and a persistent identifier
remain to be arranged.

## Software versions

The paper's trial results come from the original study code. The maintained
package adds numerical and trial-control fixes. In 31 matched cases, rc4 kept
rc3's allocations and final selections. Against the historical code, all 24
full-budget paper cases agreed within the specified tolerances; one short-budget
case retained the known correction to the available final-selection doses.

These checks do not replace a full simulation study. Original workflows used
different software environments, recorded in their metadata; one dependency
lock cannot guarantee identical historical bytes for all results.
