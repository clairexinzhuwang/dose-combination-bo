Efficacy summaries added in the 30 September 2026 manuscript revision

Run from the repository root with NumPy, SciPy and zstandard installed:
  python paper/revision_analysis/20260930/derive_efficacy.py --output-dir /tmp/dual-combo-efficacy

Full-grid contrasts use paired replicate means over two strata and five cutoffs.
The new Table 4 efficacy column uses selection_only_trial_ratio: total true
efficacy divided by n_sel, with no efficacy assigned to stopped trials. These
conditional summaries concern different policy-specific selecting subsets.
Nested means and MCSEs reproduce the archived, authenticated CSV and JSON;
raw path shards are not included in this repository, so paired nested efficacy
intervals are not reconstructed. The original 54 toxicity contrasts are unchanged.
No new trial simulations were run. Historical records are read-only inputs.
