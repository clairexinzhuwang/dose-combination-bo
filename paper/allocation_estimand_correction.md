# Allocation-estimand correction

This correction was recorded on 6 September 2026 after a mathematical and
reporting audit. It is post hoc and does not alter any frozen simulation path.

In the full-grid simulations, each 40-record analysis dataset begins with four
synthetic observations at continuous locations used only for numerical
initialization. Those locations are not administrable dose combinations. Earlier
summaries nevertheless included those four records when describing the
percentage of participants assigned to combinations above the true toxicity
threshold.

The corrected full-grid allocation outcome excludes the four initialization
records. Its numerator is the number of above-threshold assignments after
initialization, and its denominator is the number of participants assigned after
initialization. The main 40-record analysis therefore has 36 participant
assignments. In the nested analysis, the 20-, 40-, and 80-record snapshots
correspond to 16, 36, and 76 participant assignments.

Within each simulation replicate, every policy used the same initialization.
The initialization contribution therefore cancels exactly from paired allocation
contrasts; changing the denominator from the total record count to the
post-initialization participant count is an exact deterministic transformation.
For absolute full-grid summaries, initialization counts are recovered only from
authenticated allocation-history prefixes in the committed analysis records.
They are not regenerated from a seed alone.

The gradual-availability design is different: its two initial assignments are
actual participants treated at the administrable grid combination `(0, 0)`.
Those assignments remain in its allocation numerator and denominator.

Recommendation, efficacy, dose-location, stopping, and posterior-calibration
outcomes are unchanged. The original simulation records and the earlier frozen
analysis specification remain unchanged so that the correction is auditable.
