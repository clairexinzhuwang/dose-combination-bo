# How the entropy rule is calculated

The entropy rule chooses the next cohort to learn which combinations have
mean toxicity at or below the limit. Use `entropy` or
`feasibility-entropy`; `qBIG` is a historical alias and `SUR` is deprecated.

Let `PF(x)` be the posterior probability of meeting the mean-toxicity limit,
and `h(p) = -p log(p) - (1-p) log(1-p)`, with `0 log(0) = 0`. The score is

```text
alpha_ent(d) = sum_{x in G_t} [h(PF(x)) - E{h(PF+(x)) | D_t}].
```

The expectation averages over the hypothetical toxicity cohort mean at `d`.
`PF+` uses the resulting GP update, holding fitted parameters and the available
grid `G_t` fixed. The score sums reductions in marginal binary entropy. It is
not the entropy of the joint dose labels or of the optimal combination.
Nine-node Gauss–Hermite quadrature approximates the one-dimensional expectation.

This criterion adapts GP contour learning, using binary labels, zero margin
and equal weights on the finite grid. Related work includes
[Marques et al.'s contour-entropy criterion](https://papers.neurips.cc/paper_files/paper/2018/hash/01a0683665f38d8e5e567b3b15ca98bf-Abstract.html)
and [stepwise uncertainty reduction](https://doi.org/10.1007/s11222-011-9241-4).
It differs from the published CLoVER criterion and serves here as an acquisition
comparator.

## Quadrature check

The archived comparison used 432 saved posterior states: replicate identifiers
0, 37 and 99; both strata; response-record states 4, 10, 16, 22, 30 and 38;
and 12 surface-by-`tau` configurations. Each state had 25 candidates,
giving 10,800 scores.

Nine-node scores were compared with a 128-node reference. Adaptive quadrature
also checked the 20 candidates with the largest discrepancies.

| Check | Result |
|---|---:|
| Maximum absolute nine-node versus 128-node score difference | `0.0019135063` |
| Maximum normalized statewise infinity error | about `0.004306` (0.431%) |
| Maximum 128-node versus adaptive difference in the 20 checked cases | `6.5e-15` |
| Same next assignment after qualification and fallback | 432 / 432 states |

One raw top-score ranking differed, in a state where no eligible combination
qualified and fallback determined the assignment. The actual choices agreed
in all 281 states with qualifying candidates and all 151 fallback states.
This is evidence for the checked states; nine-node integration remains an
approximation without a uniform error bound.

The [saved comparison](https://github.com/clairexinzhuwang/dose-combination-bo/blob/main/results/qbig_quadrature_validation/qbig_quadrature_representative_audit.json)
records these checks. The complete research archive also contains the state-level
CSV, execution record and `paper/validate_qbig_quadrature.py`, which reconstructs
the comparison from saved states.
