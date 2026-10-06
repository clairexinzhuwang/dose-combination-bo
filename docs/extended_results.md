# Saved results

`results/` contains the original study outputs. `paper/revision_analysis/`
contains later analyses of the same records. Additional raw paths and historical
source code are kept in the separate research archive.

## Files used by the current paper

| Results | Data |
|---|---|
| Allocation and final-selection contrasts | [By scenario](../results/comparator_family_analysis/comparator_family_surface_bivariate.csv), [by cutoff](../results/comparator_family_analysis/comparator_family_cellwise_contrasts.csv) |
| Full-grid operating characteristics | [By rule, scenario and cutoff](../results/comparator_family_analysis/comparator_family_full_panel_oc.csv) |
| Efficacy contrasts | [By scenario](../paper/revision_analysis/20260930/full_grid_efficacy_contrasts.csv), [by cutoff](../results/comparator_family_analysis/comparator_family_secondary_diagnostics.csv) |
| Gradual-availability results | [Toxicity, stopping and enrollment](../results/comparator_family_analysis/comparator_family_gradual_oc.csv), [paired contrasts](../results/comparator_family_analysis/comparator_family_gradual_contrasts.csv), [efficacy](../paper/revision_analysis/20260930/gradual_efficacy_absolute.csv) |
| Selected trial records | [34,000 records](../results/comparator_family_analysis/comparator_family_selected_records.json.zst), including historical comparators |
| Acceptable OSA selection within an efficacy margin | [Summary](../paper/revision_analysis/20261006/near_optimal_summary.csv), [trial endpoints](../paper/revision_analysis/20261006/near_optimal_trials.csv), [table mapping](../paper/revision_analysis/20261006/table_traceability.csv) |
| Enrollment/noise contrasts | [Allocation excluding initialization](../results/budget_toxicity_calibration_family_A_post_initialization/budget_toxicity_calibration_family_A_post_initialization.csv), [original contrast tables](../results/budget_toxicity_calibration_publication/budget_toxicity_calibration_all_118.csv) |
| Enrollment/noise efficacy | [Marginal summaries](../paper/revision_analysis/20260930/nested_efficacy_absolute.csv), checked against the original [CSV](../results/budget_toxicity_calibration_analysis/budget_toxicity_calibration_secondary.csv) and [JSON](../results/budget_toxicity_calibration_analysis/budget_toxicity_calibration_secondary.json) |

For gradual-availability efficacy, use `selection_only_trial_ratio` in the
revision CSV: total selected efficacy divided by the number of selecting
trials. The file also retains the older mean over eligible observations within
each replicate. These means use different weights.

For Family A allocation contrasts, use the file excluding initialization rather
than the allocation fields in the older contrast table. The paper compares
cKG, cEI, tMSE and entropy; stored records also retain older comparator names.
[Rule names](public_api_identity.md) explains them.

## Earlier analyses

These files remain available for reference; they are not a complete index of
the current paper.

| Analysis | Data |
|---|---|
| Pooled comparator summaries, including the former three-scenario average | [Toxicity contrasts](../results/comparator_family_analysis/comparator_family_aggregate_contrasts.csv), [secondary contrasts](../results/comparator_family_analysis/comparator_family_secondary_aggregate.csv) |
| Earlier manuscript facts | [Fact record](../results/manuscript_result_facts/manuscript_result_facts.json) |
| Earlier gradual-availability comparisons | [Paired results](../results/protocol_scaffold_analysis/protocol_scaffold_paired_grid.csv), [operating characteristics](../results/protocol_scaffold_analysis/protocol_scaffold_oc.csv) |
| Earlier terminal comparisons across scenarios | [Contrasts](../results/cross_testbed_analysis/cross_testbed_terminal_contrasts.csv) |
| cEI–tMSE schedule sensitivity | [Paired contrasts](../results/schedule_ratio_analysis/schedule_ratio_paired_contrasts.csv) |
| Numerical checks | [cKG](../results/ckg_piecewise_exact_validation/ckg_piecewise_exact_summary.json), [entropy](../results/qbig_quadrature_validation/qbig_quadrature_representative_audit.json) |

Keep the metadata alongside each historical output. Column names and
denominators differ between analyses. See [Output and reporting](reporting.md)
for definitions and [Reproduction](reproducibility.md) for runnable commands and
the limits of the included data.
