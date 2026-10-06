# cKG-1024 numerical-stability anchor

This OSA analysis is **post hoc and descriptive**. The production-compatible
cKG-512 streams are retained as channel-wise prefixes; cKG-1024 appends fresh
efficacy and toxicity draws. It is a custom prefix-preserving anchor, not
the package's conventional nmc=1024 draw order or a package default.

The two strata were averaged within each of 20 trial seeds. Values are
means +/- seed-clustered SE across those seed averages.

| Comparison | Same-state selected dose | Full path exact | Cohort-position agreement | Terminal recommendation | Unsafe status |
|---|---:|---:|---:|---:|---:|
| cKG-1024 vs cKG-512 | 54.2% +/- 4.3% | 5.0% +/- 3.4% | 31.7% +/- 3.8% | 35.0% +/- 7.3% | 82.5% +/- 6.6% |

Excluding decisions with an empty outer gate, same-state selected-dose agreement was 50.2% +/- 4.6%. The forced fallback applied at 55 of 720 states.

Unsafe recommendation rates were 17.5% +/- 7.5% for cKG-1024 and 20.0% +/- 7.6% for cKG-512. Discordant directions were 7.5% +/- 4.1% for 1024-unsafe/512-safe and 10.0% +/- 5.8% for 1024-safe/512-unsafe.

Agreement measures numerical stability relative to cKG-512, not decision
accuracy or Monte Carlo convergence. The fantasy seed is fixed at 0 across
trial seeds. No timing or GPU claim is made.
