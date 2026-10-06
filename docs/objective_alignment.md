# What the four rules do

The four rules use the same prediction models and the same rules for stopping
and choosing a final combination. They choose the next group differently.
The limit concerns a combination's mean toxicity.

| Rule | What it looks for |
|---|---|
| `cKG` | How the next group could improve the final choice |
| `cEI` | Higher efficacy, giving more weight to combinations likely to meet the toxicity limit |
| `tmse` | Uncertain toxicity at combinations predicted to be near the limit |
| `entropy` | Information about which available combinations meet the toxicity limit |

Entropy does not score efficacy.
It learns whether combinations meet the toxicity limit.
This can help high-efficacy combinations qualify for final selection.
Better toxicity information can therefore improve final efficacy.

cKG learns both efficacy and toxicity. It calculates the next-group value
exactly under the fitted model; this does not guarantee the best result over
a whole trial.

## What the simulations found

- cKG had higher estimated final efficacy than cEI at every toxicity cutoff in
  all four scenarios, with smaller gains at stricter cutoffs. Some confidence
  intervals included no difference. In the sleep-apnea (OSA) scenario average,
  cKG also made more above-limit assignments and final selections.
- Compared with tMSE, cKG made fewer above-limit assignments but more above-limit
  final selections in all four scenario averages.
- In the OSA average, entropy had higher final efficacy and cKG made fewer
  above-limit assignments. Their difference in final-selection toxicity was
  uncertain. Changing the model's assumed toxicity variation changed the ranking
  of the rules; the true mean responses and their variation stayed fixed.

See [Saved results](extended_results.md) and [Output and reporting](reporting.md)
for the data and definitions.

Earlier analyses also used `cEI-tMSE`, which alternates those two rules; see
[Rule names](public_api_identity.md).

See [how the model uses trial observations](gp_surrogate.md).
How each rule is calculated: [cKG](ckg_piecewise_exact.md), [cEI](cei.md),
[tMSE](tmse.md), [entropy](qbig_sur_approximation.md).
