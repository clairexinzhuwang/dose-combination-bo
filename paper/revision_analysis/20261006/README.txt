Acceptable near-optimal selection — OSA record reanalysis, 6 October 2026

This analysis reads the frozen comparator-family selected-record bundle already
used by the manuscript. It does not fit GPs or run adaptive trials.

The required input is included in this repository:
  results/comparator_family_analysis/comparator_family_selected_records.json.zst

With NumPy and zstandard installed, run from the repository root:
  python paper/revision_analysis/20261006/derive_near_optimal.py --output-dir scratch/reanalysis/near_optimal

The script infers the repository root from its location. Outputs must be outside
the historical results/ directory. No separate full archive or GP dependencies
are needed for this record reanalysis.

analysis_plan.json fixes the descriptive delta range and reporting scope.
near_optimal_summary.csv contains all 80 means/MCSEs, using 200 independent
replicate sets with the two strata averaged within each seed.
near_optimal_trials.csv provides 8,000 truth-based endpoint derivations.
near_optimal_table.tex is inserted verbatim into Appendix sec:nearoptimal of
PAPER_revised_singlefile.tex; all 20 rule/cutoff rows and all 4 margins are shown.
analysis_checks.json records input consistency, coverage, grid targets and truth
checks. table_traceability.csv maps all 80 table cells to the input, script,
summary and manuscript label. Input records and original acquisition paths are
read-only.

The margins 0, 0.5, 1, 2 are simulated AHI4 events/hour, not clinical minimally
important differences. The first three give the same rates because no other
acceptable grid point is within 1 event/hour of the optimum. Unsafe selections
never count as successes, including those with high efficacy. All 400 initiated
full-grid trials contribute per rule/cutoff. The legacy dense-grid eff_reg
field is not used. No gradual/stopped-trial conditional outcome is pooled
into this full-grid display.
