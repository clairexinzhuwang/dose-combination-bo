# Output and reporting

## Selection fields

| Field | Meaning |
|---|---|
| `selection_status` | `qualified_selection`, `fallback_selection` or `stopped_no_selection` |
| `recommendation_made` | Whether a final selection exists |
| `rec_true_eff`, `rec_true_tox` | True simulated means at the selected combination |
| `rec_passed_criterion` | Whether the selection passed the posterior toxicity criterion |
| `rec_direct_observations`, `rec_ever_assigned` | Direct-observation count and whether the selected combination was ever assigned |
| `grid_dose_units` | Distance to the best acceptable combination on the assignment grid |
| `dose_units` | Distance to the historical continuous-domain reference |
| `terminal_candidate_count`, `terminal_n_gate_pass` | Available combinations at completion and how many qualify |
| `n_gate_pass`, `n_gate_pass_safe` | Qualifying combinations and truly acceptable qualifying combinations on the full grid |

For short gradual runs, use the terminal fields: unopened combinations cannot
be selected. Stopped trials have no selected dose; their grid summaries describe
the posterior only. A selection may be untried, and a fallback fails the
posterior criterion. See [Settings and scope](scope_and_limitations.md).

## Allocation

An above-limit combination has true simulated mean toxicity above its limit.
The count measures assignments, not adverse events.

| Setting | Summary |
|---|---|
| Full grid with `R` records | Exclude four initialization records; divide by `R-4` |
| Main analysis, `R=40` | Divide above-limit assignments by 36 |
| Snapshots at `R=20,40,80` | Divide by 16, 36 or 76 |
| Gradual availability | Include the first two participants; report mean above-limit count and enrollment per initiated trial |

Historical `toxic` and `C_A` fields, and some `pct_patients_above` records,
include initialization. Use the corrected outputs linked in
[Saved results](extended_results.md). The historical field `100*N_above/40`
uses planned maximum enrollment, not the number actually enrolled.

## Final selection

The all-trial above-limit percentage uses all initiated trials as its denominator.
Stopped trials contribute no above-limit selection. The conditional percentage
uses selecting trials and is undefined when none selects. Do not assign efficacy
to a stopped trial.

The paper's gradual-availability efficacy mean weights selecting trials equally.
Use `selection_only_trial_ratio` in the [revision CSV](../paper/revision_analysis/20260930/gradual_efficacy_absolute.csv).
The older mean over eligible observations within each replicate is also retained
and labeled. Paired conditional contrasts use only pairs where both rules select;
report that pair count.

Report stopping, enrollment and selection rates with the conditional outcomes.

## Paired summaries

`M` is the number of independent replicate sets, `seed` their identifier, `R`
the full-grid record count, and `N_obs` realized enrollment. Rules share random
numbers within a replicate, although their dose paths may differ.

For each scenario, average the paired differences over two strata and five
cutoffs within each replicate. OSA and Gaussian bump use `M=200`; logistic and
MARIPOSA-motivated scenarios use `M=100`. The MCSE is the sample SD of replicate
differences divided by `sqrt(M)`. Pointwise intervals use the Student-t critical
value with `M-1` degrees of freedom. The separate noise analysis uses simultaneous
max-t intervals.

The current paper reports scenarios separately. Older files contain an equal
three-scenario average using identifiers 0–99. The current paper does not pool
efficacy across scenarios because its units differ.

For cKG minus a comparator, positive efficacy is favorable, positive above-limit
selection is adverse, and negative above-limit allocation means less exposure.

## Classification and record counts

Grid `recall` and `precision` measure sensitivity and positive predictive value
for classifying combinations below the limit. They do not measure final
selection or participant exposure.

The historical archive has 55,480 logical rows and 8,800 separately stored
extension runs. The 34,000-row comparator set uses 25,200 formal rows plus those
8,800 extension rows. These sets overlap; do not add their counts as independent
trials. See [Saved results](extended_results.md).
