# Where the allocation rule fits

A trial needs rules for opening doses, modeling outcomes, assigning cohorts,
stopping and choosing a final combination. This package changes the allocation
rule while keeping the other rules the same within each protocol.

The GP models use observed outcomes to predict responses at other combinations.
The acquisition function scores eligible combinations for the next cohort.
cKG asks how that cohort would change the final choice; cEI looks for efficacy
improvement; tMSE and entropy learn toxicity.

Other designs cover more trial decisions. [BOIN12](https://doi.org/10.1200/PO.20.00257)
and [Comb-BOIN12](https://doi.org/10.1080/19466315.2024.2370403) use utility-based
allocation, [DROID](https://doi.org/10.1111/biom.13840) identifies a dose range
before randomized comparison, and [BARD](https://doi.org/10.1177/17407745251350596)
combines backfill with adaptive randomization. They are not implemented here.

See [Allocation rules](objective_alignment.md) for the scores and
[Settings and scope](scope_and_limitations.md) for the protocol.
