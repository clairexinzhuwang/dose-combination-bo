# Comparator-family completion: frozen analysis specification

Status: **FROZEN BEFORE GENERATION OF THE MISSING EXTENSION CELLS**  
Freeze date: 2026-08-25  
Scope: simulation-methodology extension for the current acquisition-component paper

## Temporal boundary and purpose

This extension was specified after inspection of the original three-policy
simulations and of the already archived, incomplete tMSE/SUR comparisons. It is
therefore not a preregistered or wholly prospective experiment. The archived
records are retrospective pilot evidence. A disjoint set of missing execution
cells will be generated only after this document and the machine-readable design
manifest have been frozen. The final paper will describe the entire family
analysis as exploratory/descriptive and will identify which rows were reused and
which were newly executed.

The extension answers one bounded question: is the reported separation between
terminal recommendations and trial assignments specific to the author-specified
cEI--tMSE hybrid, or is an analogous finite-design pattern also present against
two single-objective boundary-learning acquisitions?

It does not add a clinical bridge, a terminal-objective factorial, extra
surfaces, or a leaderboard of additional acquisitions. Those are separate future
studies.

## Common design and policy hierarchy

Within each access protocol, every policy uses the same surrogate models,
posterior-feasibility gate, dose panel, initialization, cohort size, enrollment
limit, empty-gate rule, and terminal recommendation rule.

The principal family contains:

1. **cKG**: the manuscript's gate-aligned, hard-gated one-step
   recommendation-value KG, evaluated deterministically on the finite panel;
2. **tMSE-only**: gated localized targeted-MSE toxicity-boundary sampling;
3. **entropy-SUR**: gated one-step integrated contour-entropy reduction over
   the finite panel.

Two secondary comparators are retained:

4. **cEI**: current feasible-improvement sampling;
5. **cEI--tMSE**: an author-specified fixed hybrid sensitivity.

The terms “recommendation-value” and “boundary-learning” are conceptual labels
defined for this paper, not a standard taxonomy or a controlled factorial.
tMSE-only and entropy-SUR differ in localization and lookahead, and the common
gate and fallback can override every scheduled acquisition.

The tMSE score is literature-grounded, but the cEI--tMSE pairing is not a
published clinical design. Under full-panel access, the 18 scheduled adaptive
cohort positions yield nine cEI and nine tMSE positions solely by arithmetic.
This 1:1 scheduled equality is not an optimality result and does not apply
globally under gradual access or stopping.

## Reused and newly generated records

All simulations use latent gating, fixed data-generating observation variances,
`kap=1`, a 5-by-5 panel, `N_max=40`, cohort size 2, and the prespecified
greatest-posterior-feasibility empty-gate rule.

The completed family will contain 34,000 unique trial records:

- 25,200 authenticated records reused from the stable-z formal execution bundle;
- 8,800 newly executed records, held outside the immutable 55,480-row formal
  projection and committed as a separately authenticated extension.

The new records are exactly:

1. 4,000 full-panel tMSE-only/entropy-SUR trials on OSA and Gaussian bump,
   thresholds 0.5--0.9, strata 0/1, replicate identifiers 100--199;
2. 3,200 full-panel tMSE-only/entropy-SUR trials on Logistic and
   MARIPOSA-motivated surfaces, thresholds 0.5, 0.6, 0.8, and 0.9, strata 0/1,
   replicate identifiers 0--99 (threshold 0.7 already exists);
3. 1,600 gradual-access tMSE-only/entropy-SUR OSA trials at thresholds 0.7 and
   0.9, strata 0/1, replicate identifiers 0--199.

No random-policy, predictive-gate, noise-multiplier, schedule-composition, or
new cKG execution is part of this extension.

The independent Monte Carlo unit is a matched replicate set, denoted by `M` in
reader-facing material. Full-panel OSA and Gaussian bump use `M=200`; Logistic
and MARIPOSA-motivated use `M=100`; gradual OSA uses `M=200`. `N_max=40` is the
per-trial enrollment limit and must never be used for the Monte Carlo count.

## Primary finite-design estimands

OSA remains the primary testbed. For trial record `j`, define

\[
Y_{T,j}=I_{\mathrm{rec},j}
 I\{g(\widehat d_j,z_j)>g^\dagger(z_j)\}
\]

and, under full-panel access,

\[
Y_{A,j}=100\,C_{A,j}/N_{\max},\qquad N_{\max}=40,
\]

where `C_A` is the number of above-boundary simulated assignments. The latter is
a maximum-enrollment-scaled simulated-assignment percentage, not a patient-level
toxicity rate.

For each boundary comparator `b` in {tMSE-only, entropy-SUR} and matched
replicate set `r`, average the cKG-minus-`b` difference equally over the two
strata and five thresholds:

\[
D^{\mathrm{OSA}}_{k,b,r}
=\frac{1}{10}\sum_{z=0}^1\sum_{\tau\in\{.5,.6,.7,.8,.9\}}
\{Y_{k,r}(\mathrm{cKG})-Y_{k,r}(b)\},
\quad k\in\{T,A\}.
\]

The primary estimands are the four Monte Carlo means of these replicate-level
quantities. Report paired Monte Carlo SEs and two-sided 95% `t` intervals.
Gate-specific displays use pointwise paired intervals and are descriptive; no
cell vote count is a decision rule.

The objective-family directional pattern is considered interval-supported in
OSA only if all four intervals lie strictly in the prespecified directions:

\[
\Delta_{T,\mathrm{tMSE}}>0,\quad
\Delta_{A,\mathrm{tMSE}}<0,\quad
\Delta_{T,\mathrm{SUR}}>0,\quad
\Delta_{A,\mathrm{SUR}}<0.
\]

This conjunction was fixed after partial pilot results were known, so it governs
interpretation but is not presented as a confirmatory type-I-error-controlled
test. No absolute-maximum quantile, adaptive Monte Carlo stopping, or
result-dependent threshold is used.

## Supporting finite-design estimands

### Binding-boundary family

The prespecified supporting binding family comprises OSA, Logistic, and the
MARIPOSA-motivated surface. For this aggregate, use replicate identifiers 0--99
on every surface, give each surface weight 1/3, and within surface weight the two
strata and five thresholds equally. The same four directions and paired-interval
rule are applied. A class-level phrase is allowed only if both the OSA primary
aggregate and this binding-family aggregate support both boundary comparators.

### Multimodal nonbinding stress test

Gaussian bump is analyzed separately with `M=200`. It is a multimodal
nonbinding stress test, not a controlled test of bindingness. A reversal limits
the scope of the binding-family statement but does not identify bindingness as
the cause.

### cEI and hybrid

cKG versus cEI is used to contrast one-step recommendation value with current
improvement. The cEI--tMSE hybrid remains a secondary schedule sensitivity and
cannot represent the boundary-learning family.

## Secondary outcomes and mechanism qualifications

The same matched blocks will report:

- recommendation-conditional true efficacy and dose-location error, with the
  eligible pair count shown;
- recommendation probability;
- final admitted-set false-admission count and admitted-dose recall/precision,
  with empty-gate frequency shown where available;
- feasibility probability and posterior SD at the panel dose nearest the stored
  OBD reference;
- empty-gate/fallback event counts where available.

Objective alignment may be used as a mechanism interpretation only if the
boundary acquisitions improve the specified boundary diagnostics while cKG
improves a terminal-value outcome. Otherwise the result is described as an
empirical policy contrast, not proof of a general mechanism.

## Gradual access, stopping, and enrollment

Gradual-access OSA is secondary whole-protocol sensitivity at thresholds 0.7
and 0.9. Report for all five policies:

- all-trial `Y_T`;
- recommendation probability and feasibility-stop probability;
- expected observed enrollment `E(N_obs)`;
- above-boundary assignment count per initiated trial and its
  `100*C_A/N_max` scaling;
- recommendation-conditional efficacy and dose-location error, with eligible
  matched-pair counts.

At the stratum-specific trial level, `I_rec=1-I_stop`, hence

\[
E(Y_T)=P(I_{\mathrm{rec}}=1)
P\{g(\widehat d,z)>g^\dagger(z)\mid I_{\mathrm{rec}}=1\}.
\]

Stopped trials contribute zero to `Y_T`; this must not be interpreted as a
conditional safety improvement. Under the OSA truth, feasible panel doses exist,
so these are incorrect operational stops in the simulation truth.

The gradual protocol jointly changes initialization, deterministic panel
expansion, temporary no-repeat eligibility, fallback opportunities, stopping,
and realized enrollment. Its contrasts are whole-protocol contrasts, not causal
effects of dose escalation alone. The serialized empty-gate stop means three
consecutive empty eligible gates; it does not assert that the accessible gate or
the true feasible set is empty.

## Frozen interpretation ladder

1. Both boundary comparators satisfy the OSA conjunction and the binding-family
   conjunction: a manuscript-defined recommendation-value versus
   boundary-learning directional pattern may be reported within the evaluated
   binding-boundary finite design.
2. OSA satisfies the conjunction but the binding family does not: report an OSA
   objective-alignment pattern only.
3. Only tMSE-only or only entropy-SUR satisfies it: reject the family claim and
   keep the two boundary objectives separate.
4. Only the hybrid shows the separation: retain the original hybrid-specific
   trade-off; no acquisition-family upgrade.
5. Only the terminal direction holds: report a terminal difference, not a
   terminal--operational discordance.
6. Only the assignment direction holds: report an operational exposure
   difference, not the proposed trade-off.
7. Aggregate support with heterogeneous cells: report support for the defined
   finite-design average, not uniformity.
8. Gaussian-bump reversal: qualify transfer to the multimodal nonbinding stress
   test; do not replace the unified objective-alignment logic with unlimited
   case-by-case storytelling.

## Planned manuscript hierarchy

If the family criterion is supported, the main manuscript will use an OSA
comparator-family figure and a terminal-versus-assignment quadrant map; the
original hybrid profiles move to the Supporting Information. If it is not
supported, the main manuscript will explicitly retain a comparator-specific
claim and show the family analysis as falsification evidence. In either case,
the paper will add a concise “why these acquisition objectives” subsection,
demote the hybrid to secondary sensitivity, and avoid policy leaderboards.

