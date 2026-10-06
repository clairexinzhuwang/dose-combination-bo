# Budget and toxicity-noise calibration audit: frozen specification

Status: **SCIENTIFIC DESIGN AND THIRD OUTCOME-BLIND AMENDMENT FROZEN BEFORE THIRD CLEAN RESTART**  
Scientific-design freeze date: 2026-08-26  
Numerical-amendment date: 2026-08-26  
Amended execution-binding freeze date: 2026-08-26  
Baseline-serialization amendment date: 2026-08-27  
Second amended execution-binding freeze date: 2026-08-27  
Scientific-equivalence and conservative numerical-guard amendment date: 2026-08-27  
Third amended execution-binding freeze date: 2026-08-27  
Scope: post-result, major-revision sensitivity analysis for the current acquisition-component paper

## Temporal boundary and scientific question

The original acquisition results and the completed comparator-family results were
known before this audit was designed.  This is therefore an exploratory,
analysis-specified major-revision experiment, not a preregistered or wholly
prospective study.  No new audit path may be generated until this document, the
machine-readable design manifest, the independent runner, the blinded precision
monitor, and their source/runtime bindings have all been frozen.

Here, a "new audit path" means an \(N=80\) production-design path in the frozen
seed blocks. Small-budget construction fixtures used to test injection cleanup,
prefix identity, and schema rejection are not scientific audit records, are not
eligible for analysis, and are never written into a production artifact.

### Pre-outcome numerical amendment and clean-restart boundary

An initial authenticated execution began only after the scientific design above
had been frozen.  It stopped before completing the analytical \(M=200\) tranche
because the exact-cKG integrator encountered a floating-point monotonicity
failure: on an exceptionally narrow interval with strictly positive normal
mass, the installed
SciPy/libm stack returned endpoint `ndtr` and `log_ndtr` values in the wrong
order.  The error was localized using only the exception class, the two numerical
endpoints, runner progress, and pass/fail construction diagnostics.  No policy
contrast, outcome mean, effect sign, interval, ranking, or precision result from
that attempt was calculated or inspected.

Before this amendment, the failed execution directory was authenticated as 33
regular files (60,548,406 bytes): eight complete 400-path shard envelopes and
nine atomic checkpoints, of which eight repeated the committed 400-path shards
and one contained 100 of 400 paths.  All files carried launch fingerprint
`5718e01a44bd7392ddf596d7fa5cb7657eb0a57f4f8cfa1f3251f94edbd1327c`.
The SHA-256 of the compact canonical JSON inventory of `path`, `bytes`, and
`sha256`, sorted lexicographically by relative path with sorted object keys and
no insignificant whitespace, was
`df56909ddb9d26fad327141305888dec89e67e4e3f051234242d25943755fbee`.
The entire directory was moved without changing any byte to
`results/budget_toxicity_calibration_staging__ABORTED_NUMERIC_INTERVAL__launch_5718e01a44bd7392__20260826`.
It is an aborted provenance record, not an analysis input.

No shard or checkpoint from the aborted fingerprint may be resumed, rebound,
copied into, or indexed by the replacement execution, including shards from
policies that did not call exact cKG.  After new source/runtime/design bindings
are frozen, the analytical tranche restarts from seed 0 in a new empty staging
directory and regenerates the complete 24-shard factorial.  The seed blocks,
scientific estimands, multiplicity procedure, blinded precision rule, baseline
identity gate, and every other scientific-design choice remain unchanged.

### Frozen narrow-interval repair

The repair is confined to the new audit core; none of the 14 hash-bound formal
or package sources is edited.  In particular, the frozen exact-cKG source retains
SHA-256
`b0689cdfb1bc95f0bb6da12745962fb51f42cfc5e8fb15f39b07de702527484d`.
Only while the audit-only exact-cKG wrapper is active, the core pairs its
ordinary and log normal-interval probability calculations.  Both injected
helpers are restored in `finally`, and an external patch, re-entrant use, a
restoration failure, or unequal ordinary/log fallback counts is a hard
construction failure.  Non-cKG paths never activate the repair.

Let \(\ell<h\), \(w=h-\ell\), and \(m=(\ell+h)/2\).  An interval calculation
is eligible for the paired local replacement only when all of the following
pre-specified, outcome-independent geometric conditions hold:

- \(\ell\) and \(h\) are finite and \(w\) is finite and positive;
- \(w\le 64\max\{\operatorname{ulp}(\ell),\operatorname{ulp}(h)\}\), where
  `ulp` denotes the positive spacing to a neighboring binary64 value;
- the endpoints converted exactly to `Decimal` at precision 80 give
  \(|m|\le50\); and
- \(\delta=|m|w/2+w^2/8\le10^{-12}\); and
- the precision-80 value \(p_0=w\phi(m)\) converts to a finite, strictly
  positive binary64 value.  If it underflows on conversion, the interval is
  ineligible; the repair never clamps it to the smallest subnormal value.

Inside this set, both probability representations use the certified local
calculation before calling the installed CDF backend.  This paired rule also
prevents a silent negative, zero, or inaccurate positive CDF difference from
escaping merely because it did not raise an exception.  Outside this set, the
frozen helpers are used unchanged and retain their original
hard failure.  Within it, the paired replacement uses

\[
p_0=w\phi(m),\qquad
\log p_0=\log w-\frac{m^2}{2}-\frac12\log(2\pi).
\]

For \(x=m+t\) with \(|t|\le w/2\), the normal-density ratio satisfies
\(|\log\{\phi(m+t)/\phi(m)\}|\le\delta\).  Consequently,

\[
e^{-\delta}p_0\le \Pr(\ell<Z<h)\le e^{\delta}p_0,
\qquad
|\log\Pr(\ell<Z<h)-\log p_0|\le\delta,
\]

and the analytic relative-mass error is at most
`expm1(delta)`.  Under \(|m|\le50\), the precision-80 Decimal remainder is below
\(10^{-70}\); conversion of either final value to binary64 contributes at most
one-half ulp in addition to the stated analytic bound.  Taking an absolute CDF
difference, swapping reversed endpoint values, or assigning zero mass is not an
admissible repair.

Every raw path stores the following construction-audit fields:

- `narrow_interval_probability_fallback_count`;
- `narrow_interval_log_probability_fallback_count`;
- `narrow_interval_fallback_max_analytic_log_error_bound` (zero if unused);
- `narrow_interval_fallback_analytic_log_error_cap` (fixed at \(10^{-12}\)); and
- `narrow_interval_fallback_pair_count_identity_checked`.

A successful exact-cKG path must have identical ordinary and log fallback counts
and a true pair-count check.  Every non-exact policy must record zero for both
counts.  These fields diagnose construction only and are never scientific
outcomes or inputs to the precision top-up.

### Pre-analysis baseline-serialization amendment and second clean-restart boundary

The replacement execution bound to launch fingerprint
`5c0d05492e367c1bc52045dbcb8525daa672bf09868b69ecedf872c3278f89cb`
completed the full \(M=200\) block: all 24 immutable shard envelopes, 9,600
maximum-budget paths, and 28,800 nested snapshots.  The required baseline-only
identity gate was then invoked.  It failed before any precision-monitor or
scientific-analysis invocation; no effect estimate, sign, interval, \(p\)-value,
policy ranking, or top-up decision was calculated or inspected.

The construction-only diagnosis found that all 3,040 historical allocation
histories available for comparison agreed exactly.  The other 160 cells were
exactly the pre-specified cEI, \(\tau=0.7\) historical exception for which no
allocation history had been serialized.  Across the 3,200 common-field cells,
all fields other than three legacy scalar posterior readouts agreed exactly,
and the 25-dose batched posterior state used for gates and recommendations was
unchanged.  The only differences were last-binary64-bit query-shape effects:

- `rpsel` differed in 687 of 3,200 cells, with maximum absolute difference
  \(3.424\times10^{-12}\) (unrounded audit value
  `3.4239278079439828e-12`);
- `obd_pf` differed in 1,692 of 3,200 cells, with maximum absolute difference
  \(2.230\times10^{-13}\) (unrounded audit value
  `2.2304380564719395e-13`); and
- `obd_sdg` differed in 1,662 of 3,200 cells, with maximum absolute difference
  \(1.749\times10^{-12}\) (unrounded audit value
  `1.748934330692009e-12`).

The frozen historical harness computed those three serialized scalars from
singleton posterior queries, whereas the first audit core extracted them from
components of a batched 25-dose posterior query.  Although this did not alter
the allocation sequence, batched posterior state, gate, or recommendation, the
baseline contract deliberately requires exact available-field identity.  The
repair therefore preserves the batched 25-dose state for every panel-wide
quantity and decision, retains each fitted model until snapshot serialization,
and reproduces the frozen singleton query order, shapes, and formulas only for
`rpsel`, `obd_pf`, and `obd_sdg`.

During implementation diagnosis, two in-memory cEI maximum-budget construction
replays (seeds 0 and 1, maximum budget 80) were used only to check prefix,
legacy-envelope, and pre-frozen scientific-state-hash identity.  After the
repair, separate in-memory construction regressions (seeds 1 and 4, maximum
budget 42) compared their \(N=40\) prefixes with independently stopped frozen
\(N=40\) serializer results.  All four fixtures were immediately discarded,
were never written as production artifacts, and no scientific effect was
calculated or inspected from them.

Before this second amendment, the completed replacement directory was
authenticated as 99 regular files totaling 180,462,355 bytes.  The SHA-256 of
the compact canonical JSON inventory of `path`, `bytes`, and `sha256`, sorted
lexicographically by relative path with sorted object keys and no insignificant
whitespace, was
`09e6416d0a75de5610f7d88ab1785d6b5d9b87df0408776b84bfd77a3031fec9`.
The entire directory was moved without changing any byte to
`results/budget_toxicity_calibration_staging__ABORTED_BASELINE_SINGLETON_SERIALIZER__launch_5c0d05492e367c1b__20260827`.
It is an aborted construction-provenance record, not an analysis input.

No shard or checkpoint from that launch may be resumed, rebound, copied into,
or indexed by the second replacement execution.  After the amended core,
prespecification, manifest, and execution identity are refrozen, execution
restarts from seed 0 in a new empty staging directory and regenerates the
complete 24-shard (M=200) factorial.  This amendment changes only exact legacy
serialization and source/provenance bindings.  It does not change a scientific
estimand, seed block, policy, budget, calibration factor, multiplicity rule,
precision rule, baseline oracle, or interpretation rule.

### Outcome-blind construction-reproducibility audit, scientific-equivalence diagnostic, and third clean-restart boundary

The second replacement execution, bound to launch fingerprint
`f40e4e7d8f2d4f0b192babc1474556998bbf82b9061310a1d13e31cedb227acf`,
was stopped during its incomplete analytical tranche when exact-cKG
narrow-interval fallback counts were found not to be bitwise reproducible
across otherwise identically bound fresh processes.  The counts are
construction diagnostics rather than estimands.  Before deciding whether any
of that launch could be interpreted, a three-version, sealed,
outcome-blind scientific-equivalence diagnostic was committed.  At no point in
this diagnosis was a policy contrast, effect estimate, sign, confidence band,
ranking, primary-family precision result, or top-up decision inspected.

The following files and immutable artifact envelopes bind the complete
diagnostic history.  Every digest below is SHA-256.

- **V1.** Specification
  `paper/budget_toxicity_calibration_scientific_equivalence_diagnostic_spec.json`
  is bound at
  `3dd10218c7f4fec9677dd8fd711e8d987c310b895749609a478d7b0bed510d86`;
  runner
  `paper/run_budget_toxicity_calibration_scientific_equivalence_diagnostic.py`
  at
  `acffbe38d147b9eb15e57baec7e9bb1ce5448622af75f7d7bd9d73785ac8d48c`;
  and test
  `tests/test_budget_toxicity_calibration_scientific_equivalence_diagnostic.py`
  at
  `e73572be2e6947a2e9ac3a4b64b0945f0452a06760fa7953a89f5bf903b195af`.
  The artifact, metadata, and commit files are
  `results/budget_toxicity_calibration_scientific_equivalence_diagnostic/budget_toxicity_calibration_scientific_equivalence.json.zst`,
  `results/budget_toxicity_calibration_scientific_equivalence_diagnostic/budget_toxicity_calibration_scientific_equivalence.json.zst.metadata.json`,
  and
  `results/budget_toxicity_calibration_scientific_equivalence_diagnostic/budget_toxicity_calibration_scientific_equivalence.json.zst.commit.json`;
  their hashes are, respectively,
  `ef58ef01c39e6e2d35455b979d2f217f9a9bd38a388a36c73003819f4edd956e`,
  `be7acfbcde0581b6fc56c542b31065b12b29f5475c1d2773a615f155a9ca8aae`,
  and
  `46b4f5165e861bbb8d1d1bee9b682e6ee54ab5659ae6cf42823addf878f64f70`.
  V1 committed `INCONCLUSIVE_REPEAT_INSUFFICIENCY` with zero completed
  scientific paths.  Before any path began, its containment order replaced
  `socket.socket` before the bound core was imported; the Torch dependency
  chain imported `asyncio` and `ssl`, and `ssl` then failed because the socket
  object was no longer a class.  This was an infrastructure-only import
  failure and admits no scientific interpretation.

- **V2.** Specification
  `paper/budget_toxicity_calibration_scientific_equivalence_diagnostic_v2_spec.json`
  is bound at
  `87c96baba5c6a09f7284e7fd96ec25baaa271b3b7dc2f1bfa7f7a212f3b041a5`;
  runner
  `paper/run_budget_toxicity_calibration_scientific_equivalence_diagnostic_v2.py`
  at
  `f8cdae5d00382f19419b0eece8192f5a77c0a6144bb292c0dfa0e703aaea0f77`;
  and test
  `tests/test_budget_toxicity_calibration_scientific_equivalence_diagnostic_v2.py`
  at
  `6e6e8979340c2a21c4f173f55bc24d0439dd62f85085d9b9d12fdad2bbb3cf05`.
  The artifact, metadata, and commit files are
  `results/budget_toxicity_calibration_scientific_equivalence_diagnostic_v2/budget_toxicity_calibration_scientific_equivalence_v2.json.zst`,
  `results/budget_toxicity_calibration_scientific_equivalence_diagnostic_v2/budget_toxicity_calibration_scientific_equivalence_v2.json.zst.metadata.json`,
  and
  `results/budget_toxicity_calibration_scientific_equivalence_diagnostic_v2/budget_toxicity_calibration_scientific_equivalence_v2.json.zst.commit.json`;
  their hashes are, respectively,
  `fe3ba4418a7bb928760028596c89dce00369b5e5a28a7035676fe9326517db16`,
  `b3817a3edcd93b039d9ede39dcc401439fdfd0d1a426f35fa760d2ca6d95f334`,
  and
  `de47ee7253d0e58f027f007be79c8b11a3ef64cfced7ec9398eafa0ca96d07a3`.
  V2 completed the fixed 23-path fresh workload and both stored loaders
  authenticated and selected all six fixed keys, but it committed
  `INCONCLUSIVE_AUTHENTICATION` before any semantic group comparison.  The
  loaders returned role names `artifact_sha256`, `metadata_sha256`, and
  `commit_sha256`, whereas the sealed comparator expected the bare names
  `artifact`, `metadata`, and `commit`; exact dictionary equality therefore
  failed.  Under the fail-closed schema, the committed summary exported zero
  completed comparison paths rather than the completed fresh workload; the
  pre-execution V3 amendment record binds the 23-path execution fact.  No
  semantic comparison was released, and no V2 scientific result is
  interpretable.

- **V3.** Specification
  `paper/budget_toxicity_calibration_scientific_equivalence_diagnostic_v3_spec.json`
  is bound at
  `008e4c67f488ff0daa0a62b6867621ac94d75596da1d06258577a1ad2fb4b9eb`;
  runner
  `paper/run_budget_toxicity_calibration_scientific_equivalence_diagnostic_v3.py`
  at
  `cb0a7fd2686a2767467ba83d056747c3e375c98b04ff072147af6d9224960e30`;
  and test
  `tests/test_budget_toxicity_calibration_scientific_equivalence_diagnostic_v3.py`
  at
  `c1dfc6ac315dda099e673aa6a7ebd08824e8de643fa0fd950117fc70fa67bf78`.
  The artifact, metadata, and commit files are
  `results/budget_toxicity_calibration_scientific_equivalence_diagnostic_v3/budget_toxicity_calibration_scientific_equivalence_v3.json.zst`,
  `results/budget_toxicity_calibration_scientific_equivalence_diagnostic_v3/budget_toxicity_calibration_scientific_equivalence_v3.json.zst.metadata.json`,
  and
  `results/budget_toxicity_calibration_scientific_equivalence_diagnostic_v3/budget_toxicity_calibration_scientific_equivalence_v3.json.zst.commit.json`;
  their hashes are, respectively,
  `42ae32a6c4ed38a6f20dcc3f4f17154df3be0e0b68f158dedf324c95c2a6625a`,
  `0e3397512e25e01c90474f8f27ba89e96e9d0dd4add2c47be45517f8ecb75576`,
  and
  `2e58ea20d9e7397a4e88ccdc8a6fd9ab5a564558156313d9f71db43e9333ac99`.
  V3 changed only the stored-loader role normalization and committed
  `PASS_SCIENTIFIC_EQUIVALENCE_WITH_VARIABLE_CONSTRUCTION_COUNT`.  All 23 of
  23 fresh paths ran in distinct processes; the sensitive K01 assay exercised
  more than one native fallback-count signature, while the prespecified S
  control retained exactly one signature.  Every one of the six authenticated
  old--current stored-key pairs exhibited fallback-count drift.  Across the
  sealed comparisons, all categorical decisions and primary discrete endpoint
  contributions were exact, all continuous classes were within the frozen
  η tolerance below, and all frozen decision-margin certificates passed.
  Only these construction/equivalence flags left the sealed comparator; no
  policy effect was inspected.

The V3 PASS establishes only that the observed construction-count variability
was scientifically inert for these fixed keys on the hash-bound runtime and
platform, under the frozen tolerance and margin rules.  It does not establish
bitwise determinism, eliminate or repair the underlying cross-process
construction-count variability, or prove robustness to other runtimes, keys,
models, likelihoods, or forms of misspecification.

Before the third amendment, the incomplete second-replacement staging
directory was authenticated as 29 regular files totaling 52,894,420 bytes.
The SHA-256 of the compact canonical JSON inventory of `path`, `bytes`, and
`sha256`, sorted lexicographically by relative path with sorted object keys and
no insignificant whitespace, was
`f1ad4bd634aa186562ced1c8360344f3559b07d02d2fd9a26a1f444505034870`.
The entire directory was moved without changing any byte to
`results/budget_toxicity_calibration_staging__ABORTED_CONSTRUCTION_REPRODUCIBILITY__launch_f40e4e7d8f2d4f0__20260827`.
The default `results/budget_toxicity_calibration_staging` path is absent.  The
quarantine is provenance only: none of its shards or checkpoints may be
resumed, rebound, copied, or indexed.  After this prespecification and all
source/runtime/design bindings are refrozen, the complete \(M=200\) factorial
must restart from seed 0 in a new empty default staging directory.  No shard
reuse is permitted.

This third amendment leaves the scientific design, all 118 primary estimand
IDs, seed blocks, policies, budgets, calibration factors, baseline oracle,
multiplicity families, and whole-factorial top-up structure unchanged.  It
adds only the conservative numerical guard defined below and the corresponding
source/provenance bindings.

The audit asks two finite-design questions:

1. Does the previously observed terminal-recommendation versus assignment
   discordance persist when the toxicity observation-noise standard deviation
   used by the fitted model is wrong?
2. Over the evaluated budgets, does additional information reduce an observed
   toxicity-posterior calibration penalty, or can the common hard gate continue
   to exclude the true panel optimum?

It does **not** estimate a clinical minimum sample size, an acquisition-specific
convergence rate, general model-misspecification robustness, a safety guarantee,
or performance for binary DLT outcomes.  Response geometry, likelihood family,
kernel family, stationarity, homoscedasticity, and efficacy--toxicity independence
are not varied here.

## Frozen design

The data-generating truth is the existing OSA surface, evaluated separately in
strata 0 and 1.  True efficacy and toxicity observation-noise standard deviations
remain at the OSA values, 7.68 and 1.29.  The efficacy analysis model remains
correctly calibrated.  The only misspecification axis is

\[
c_\sigma=
\frac{\text{assumed toxicity observation-noise SD}}
     {\text{true toxicity observation-noise SD}}
\in\{0.5,1,2\}.
\]

Thus the fitted toxicity likelihood variances are multiplied by
\(\{0.25,1,4\}\), while the observations are always generated with the same true
noise.  The existing `kap` parameter is not used because it changes both the
data-generating and fitted noise.

The remaining design is fixed as follows:

- policies: deterministic finite-panel exact cKG, tMSE-only, finite-panel
  feasibility-entropy reduction (internal registry name `qBIG`), and cEI;
- gates: latent strict stable-\(z\) gates at \(\tau\in\{0.7,0.9\}\), admitting a
  dose only when \(z_g(d)>\Phi^{-1}(\tau)\);
- access: full \(5\times5\) panel throughout;
- initialization: the existing four-patient stratified-random warm-up;
- cohort size: 2;
- maximum physical path budget: 80 patients;
- authenticated terminal snapshots: \(N\in\{20,40,80\}\);
- empty-gate action: the common greatest-standardized-feasibility rule;
- no stopping under full-panel access.

Posterior feasibility probabilities are descriptive quantities.  Operational
admission is always decided by the strict stable-\(z\) comparison, never by the
floating-point comparison `PF > tau`.

The hybrid policy and gradual-access protocol are not rerun.  Existing gradual
results remain a correctly calibrated whole-protocol sensitivity and are not
pooled with this audit.

## One physical path, three nested budget snapshots

For every key

\[
(r,z,\tau,p,c_\sigma),
\]

one physical path is run to \(N=80\), and terminal readouts are saved at
\(N=20,40,80\).  None of the four policies receives the remaining horizon as an
input, so the shorter-budget paths must be exact prefixes.  Before production,
tests must prove that independently stopping the same path at 20 or 40 gives the
same allocation prefix, posterior gate, recommendation, and common operating
characteristics as the corresponding nested snapshot.

At final Monte Carlo size \(M\), the audit therefore contains \(48M\) independent
maximum-budget stratum-specific paths and \(144M\) nested budget snapshots.  The
three snapshots from a path are not independent trials.

Full-panel paths must enroll exactly to every snapshot and must produce a
terminal recommendation.  Any stop or missing recommendation is an implementation
failure, not an operating characteristic.

## Monte Carlo unit, common random numbers, and baseline identity

The independent Monte Carlo unit is the seed-labelled replicate.  The two strata
are averaged equally within replicate; they are not treated as \(2M\) independent
replicates.  Within a seed, warm-up draws and the ordered efficacy/toxicity
Gaussian innovations are common across policy, \(c_\sigma\), \(\tau\), and nested
budget.  Acquisition histories may diverge, but the innovation stream does not.

Seed blocks are immutable and nested:

- analytical tranche: 0--199 (\(M=200\));
- first possible top-up: 200--499 (cumulative \(M=500\));
- final possible top-up: 500--999 (cumulative \(M=1000\)).

The \(c_\sigma=1,N=40\) snapshots must deep-equal the authenticated OSA records
for all four policies, both gates, both strata, and seeds 0--199 on every field
that the historical record serialized.  The selected 34,000-record master gives
the common terminal operating-characteristic fields for all 3,200 overlapping
cells.  The authenticated formal decision master and comparator-extension raw
master additionally give complete allocation histories for 3,040 of those
cells.  The remaining 160 cEI cells at \(\tau=0.7\), seeds 0--79, were historically
projected without allocation histories, so they are checked on their complete
available field envelope rather than against nonexistent bytes.  Independently,
construction tests compare the new wrapper with the frozen harness and verify
prefix identity.  The old records are pass/fail construction oracles only; they
are not mixed into the new audit estimands.  Any available-field mismatch blocks
finalization and all scientific interpretation.

The machine-readable manifest fixes the compressed artifact and sidecar hashes
of all three historical oracles before execution.  A same-named replacement or
an uncommitted local copy is not an admissible identity oracle.

The exact cKG evaluator, stable-\(z\) gate audit, operational empty-gate rule, and
independent terminal replay must match the existing validated implementation.
The hash-bound formal and comparator-family execution sources must remain
unchanged; the assumed-noise perturbation is implemented only in new audit files.

## True panel optimum

For stratum \(z\), define the target as

\[
d^*_{\mathrm{panel},z}
=\arg\max_{d\in\mathcal G:g_z(d)\le g_z^\dagger}f_z(d),
\]

with the first canonical panel index used only to break an exact tie.  It is not
the panel dose nearest a stored continuous optimum.  The frozen OSA targets are:

- stratum 0: panel index 8, dose \((0.25,0.75)\), efficacy
  7.67921875, toxicity 1.2816796875, threshold 1.5;
- stratum 1: panel index 13, dose \((0.50,0.75)\), efficacy
  13.20546875, toxicity 1.62234375, threshold 2.0.

The runner and analyzer independently recompute these targets from the frozen
surface code and reject any discrepancy.

## Required serialized state

Each maximum-budget raw path must contain design identifiers, source/runtime
bindings, truth and assumed noise fields, the complete allocation and observed
efficacy/toxicity histories, and the true above-boundary label for every enrolled
patient.

At every posterior checkpoint \(n=4,6,\ldots,80\), store the true panel target's
toxicity posterior mean, latent SD, stable \(z\), feasibility probability, and
strict admission indicator.  At \(N=20,40,80\), additionally store all 25 panel
doses' efficacy posterior mean/SD, toxicity posterior mean/SD/stable-\(z\)/PF,
strict gate mask, fitted kernel parameters, gate counts, terminal recommendation,
and an independent terminal-rule audit.  The analyzer must be able to recompute
every reported metric from these serialized arrays without fitting a GP.

## Outcomes

All percentage outcomes used in contrasts are on a 0--100 scale.

### Primary operating characteristics

\[
Y_T=100I\{g(\widehat d_N)>g^\dagger\},
\qquad
C_A=\sum_{i=1}^{N}I\{g(d_i)>g^\dagger\},
\qquad
Y_A=100C_A/N.
\]

Report both \(Y_A\) and cumulative count \(C_A\).  Because the four warm-up
patients occupy different fractions of the three budgets, also report the
post-initialization above-boundary percentage as a supporting outcome.

### Gate failure and recovery

Let \(G_j(d^*)\) be the strict admission indicator for the true panel optimum at
posterior checkpoint \(j\).  Required outcomes are:

- terminal target exclusion, \(100\{1-G_K(d^*)\}\);
- ever admitted and finally admitted;
- the mutually exclusive partition: never admitted; admitted previously but
  excluded finally; admitted finally;
- exclusion occupancy, the fraction of checkpoints at which the target is
  excluded;
- terminal exclusion-run fraction;
- first-admission enrollment;
- recovery after a preterminal exclusion, reported unconditionally and with its
  conditioning denominator.

Assignment by the empty-gate rule does not count as gate admission.

### Posterior calibration

For terminal posterior feasibility probability \(PF_N(d)\),

\[
B_N=\frac1{25}\sum_{d\in\mathcal G}
\left[PF_N(d)-I\{g(d)\le g^\dagger\}\right]^2.
\]

Store \(B_N\) on 0--1 and report \(100B_N\).  Also report the proportion of the
25 true latent toxicity values falling in the plug-in intervals
\(\mu_g(d)\pm1.96\sigma_g(d)\), target-dose interval coverage, and mean interval
width.  These are empirical plug-in diagnostics, not Bayesian calibration
guarantees.

### Dose selection and additional diagnostics

Report panel correct-selection probability, grid-unit distance from the true
panel optimum, true recommended efficacy, final gate size and empty-gate
frequency, false admissions, feasible-dose recall, and precision conditional on
a nonempty gate.  True-feasible recommendation simple regret is conditional on
both compared recommendations being truly feasible; unsafe recommendations are
never assigned zero regret, and every conditional contrast must report the
matched eligible denominator.

## Locked primary contrast families

For all primary contrasts, first average each outcome equally over the two strata
and two gates within seed.  Gate- and stratum-specific displays are secondary.

### Family A: policy contrasts (54)

For comparator \(b\in\{\mathrm{tMSE},\mathrm{Entropy},\mathrm{cEI}\}\), compute
cKG minus \(b\) for \(Y_T\) and \(Y_A\) at each of 3 budgets and 3 assumed-noise
factors: \(3\times2\times3\times3=54\) estimands.  cKG--tMSE and cKG--Entropy are
the scientific acquisition comparisons.  cEI is the shared-gate reference and
does not create a four-policy ranking.

### Family B: calibration contrasts (48)

For each policy, budget, and erroneous factor \(c\in\{0.5,2\}\), compute
\(Y_p(c)-Y_p(1)\) for terminal target exclusion and \(100B_N\):
\(4\times3\times2\times2=48\) estimands.

### Family C: budget-recovery interactions (16)

For each policy, erroneous factor, and the two Family-B outcomes, compute

\[
\{Y_{80}(c)-Y_{80}(1)\}
-\{Y_{20}(c)-Y_{20}(1)\},
\]

giving \(4\times2\times2=16\) estimands.  A negative value means that the excess
calibration penalty observed at \(N=20\) is smaller at \(N=80\); it is not a
convergence theorem.

Absolute \(N=80-N=20\) budget contrasts and policy-versus-cEI calibration
difference-in-differences are required secondary diagnostics but do not govern
Monte Carlo top-up.

## Frozen numerical-equivalence guard

The V3 diagnostic fixed, before its execution,

\[
\eta=\sqrt{\epsilon_{64}}=2^{-26}
=1.4901161193847656\times10^{-8}.
\]

For feasibility probabilities \(p,q\in[0,1]\) and truth label
\(y\in\{0,1\}\),

\[
\left|(p-y)^2-(q-y)^2\right|\le 2|p-q|.
\]

Therefore the V3 probability tolerance gives the following frozen upper bound
for one reported panel-Brier contribution:

\[
\delta_0=200\eta
=2.9802322387695312\times10^{-6}
\quad\text{percentage points}.
\]

The row-level replicate-contrast bounds are consequently

\[
b_B=2\delta_0=400\eta
=5.9604644775390625\times10^{-6}
\quad\text{percentage points}
\]

for a Family-B Brier row and

\[
b_C=4\delta_0=800\eta
=1.1920928955078125\times10^{-5}
\quad\text{percentage points}
\]

for a Family-C Brier interaction.  Define \(b_k=b_B\) or \(b_C\) for those
rows as applicable and \(b_k=0\) for every discrete row, including all Family-A
rows and the target-exclusion rows in Families B and C.  At cumulative Monte
Carlo size \(M\), the corresponding rowwise MCSE guard is

\[
a_k(M)=\frac{b_k}{\sqrt{M-1}}.
\]

These constants may not be enlarged, reduced, or selected after inspecting an
effect.  They do not convert continuous results into exact equality; they only
propagate the already-frozen V3 tolerance conservatively into the locked
precision and sign rules.

## Blinded precision top-up

Before the first precision-monitor invocation, the complete \(M=200\) baseline
identity audit must commit an authenticated PASS-only artifact with metadata and
commit sidecars.  That artifact exposes only the pass status and frozen audit
counts, not mismatch values or scientific effect estimates.  Every precision
decision must include the hashes of this three-file baseline envelope among its
input commitments; a missing, partial, changed, or failed baseline envelope
blocks monitoring and therefore blocks any top-up decision.

After each eligible tranche, a separate precision monitor may read only the
fields required to calculate paired MCSE for the 118 locked primary estimands.
For row \(k\), its decision quantity is the guarded value

\[
\operatorname{MCSE}^{\mathrm{guard}}_k
=\operatorname{MCSE}_k+a_k(M).
\]

Its committed output may contain only estimand ID, cumulative \(M\), raw MCSE,
the fixed row guard, guarded MCSE, 1.5-percentage-point threshold pass/fail,
the raw and guarded maxima, next-\(M\) decision, and input hashes.  It must not
expose estimates, signs, confidence intervals, \(p\)-values, or policy rankings.

1. Start at \(M=200\).
2. A row is certified precise only when its guarded MCSE is at most 1.5
   percentage points.  If any guarded primary MCSE exceeds 1.5, add the entire
   balanced 200--499 block to reach \(M=500\).
3. If any guarded primary MCSE still exceeds 1.5, add the entire balanced
   500--999 block to reach \(M=1000\).
4. Stop at \(M=1000\) regardless of achieved precision and disclose every
   unresolved guarded precision target.

No cell-, policy-, outcome-, or significance-selective top-up is permitted.

## Intervals and multiplicity

Families A, B, and C each receive seed-clustered simultaneous 95% max-\(t\)
Monte Carlo bands, together with paired pointwise 95% \(t\) intervals as
precision summaries.  For each family, let \(D\) be the \(M\times K\) matrix of
replicate-level paired contrasts after the frozen within-replicate aggregation.
Using 100,000 independent Rademacher multiplier vectors and fixed RNG seeds
202608261, 202608262, and 202608263 for Families A, B, and C, respectively,
compute

\[
T_k^{(b)}=
\frac{\sum_{r=1}^M\xi_r^{(b)}(D_{rk}-\bar D_k)}
{\{\sum_{r=1}^M(D_{rk}-\bar D_k)^2\}^{1/2}}.
\]

The critical value is the empirical 0.95 quantile, using the conservative
``higher`` order-statistic rule, of \(\max_k|T_k^{(b)}|\).  A zero-variance
column contributes zero to the multiplier maximum and has a zero-width Monte
Carlo interval.  This resampling quantifies Monte Carlo uncertainty across
independent seeds; it is not sampling inference about a patient population.
For a row \(k\) in family \(F\), let \(q_F\) denote that frozen max-\(t\)
critical value and let \([L_k,U_k]\) be its raw simultaneous band.  The
additional numerical half-width is

\[
h_k=b_k+q_Fa_k(M),
\]

and the committed guarded band is
\([L_k-h_k,U_k+h_k]\).  Thus discrete rows retain their raw bands exactly.
All sign-based primary predicates and strong family conclusions use the
guarded simultaneous band and the frozen aggregate; raw bands may be retained
for transparent numerical audit.  A primary contrast is positive only if
\(L_k-h_k>0\), negative only if \(U_k+h_k<0\), and otherwise unresolved.

Secondary profiles remain pointwise and use the same outcome-blind propagation
rule.  For an absolute panel-Brier profile set \(b_k=\delta_0=200\eta\); for a
within-policy \(N=80-N=20\) panel-Brier contrast set
\(b_k=2\delta_0=400\eta\); and for a four-term policy-versus-cEI calibration
difference-in-differences in panel Brier set
\(b_k=4\delta_0=800\eta\).  Every non-Brier secondary row has \(b_k=0\).
With \(a_k(M)=b_k/\sqrt{M-1}\), pointwise 95% critical value \(t_{M-1,0.975}\),
and raw pointwise interval \([L_k,U_k]\), its fixed numerical half-width is

\[
h_k^{\mathrm{secondary}}
=b_k+t_{M-1,0.975}a_k(M),
\]

and its guarded interval is
\([L_k-h_k^{\mathrm{secondary}},U_k+h_k^{\mathrm{secondary}}]\).  Raw
pointwise intervals remain in the audit record, but every secondary sign label,
table, and figure uses the guarded interval.  Strict positivity requires a
guarded lower endpoint greater than zero and strict negativity requires a
guarded upper endpoint less than zero; an endpoint equal to zero is unresolved.
There is no cell vote, equivalence claim from an interval covering zero, or
result-defined favorable subgroup.

## Frozen interpretation ladder

1. The pathwise available-field identity audit at \(c_\sigma=1,N=40\) must pass
   before any effect is estimated or interpreted. A failure is a construction
   error, irrespective of the direction or interval of any aggregate contrast.
2. The discordance is called calibration-robust at \(N=40\) only if both
   directions are supported at all three assumed-noise factors.
3. It is called budget-robust within a calibration setting only if both
   directions are supported at all three budgets.
4. It is called robust over the evaluated audit grid only if both directions are
   supported in all nine budget--calibration cells.  “Uniform” or general
   robustness is prohibited.
5. If only one direction is supported, report only that terminal or assignment
   difference.  Entropy is judged separately; family language requires tMSE and
   Entropy to meet the same frozen scope.
6. Additional budget is said only to attenuate an observed calibration penalty
   over the evaluated \(N=20\) to \(N=80\) range when the corresponding
   erroneous-versus-correct penalty is present at \(N=20\) and the recovery
   interaction is negative under the locked, numerically guarded inference.
   “Repair” is prohibited.
7. Brier improvement without target-exclusion improvement means panel-wide
   calibration improvement without recovery of the decision target.  The reverse
   means target-local recovery without panel-wide calibration improvement.
8. An unresolved interaction means no Monte Carlo-resolved attenuation over
   20--80, not nonconvergence.  A positive interaction means the penalty
   increased over this finite range, not mathematical divergence.
9. Similar deterioration across policies with unresolved policy-versus-cEI
   differential sensitivity is described as compatible with shared-gate
   fragility, not proof of causation.  Resolved differentials show that
   policy-induced histories modify sensitivity.

Counts can rise while percentages fall because counts accumulate with budget.
Neither trend alone establishes improved safety.

Every attenuation statement is confined to this finite, fixed-design,
fixed-runtime simulation audit.  It is not evidence of deterministic execution, a
general numerical repair, robustness to model misspecification outside the
single assumed toxicity-noise factor, an acquisition convergence rate, or a
clinical sample-size requirement.

## Relationship to the earlier noise sweep and theory

The historical `kappa` sweep changed both true and fitted noise and compared
only cKG with cEI.  It is a truth-noise difficulty sensitivity, not a calibration
audit.  Its immutable records remain in the archive for provenance, but its
scientific role in the Supporting Information will be replaced by the present
audit after authenticated results are available.

The existing local boundary-shift result is an information-theoretic lower
bound.  Its \(J^{-1/2}\) floor means that halving that lower bound requires four
times as many equally informative observations; it does not say that the actual
algorithm attains the floor or that four times as many patients halve its error.

All evaluated budgets are finite simulation budgets.  They do not add across
strata to a clinical sample-size recommendation.

## Authentication and artifacts

Raw paths are stored as 24 immutable shard envelopes per seed block. A small
authenticated seed-block index commits those shards without copying their rows;
after the blinded precision decision, a final master index commits the eligible
block indices and precision history, again without copying rows. Precision
decisions and analysis artifacts use the same separate
`budget_toxicity_calibration` namespace and atomic compressed-data/metadata/commit
envelopes. Partial checkpoints are explicitly not for analysis. The monitor and
analyzer authenticate and process the indexed shards one at a time. The final
analyzer refuses missing commits,
incomplete factorials, hash/runtime/source mismatches, nonfinite arrays, failed
stable-\(z\) audits, failed terminal replays, failed baseline identity, or an
invalid top-up history.

The old 55,480-row formal projection, 8,800-row comparator extension, 34,000-row
selected family master, and historical `kappa` records are never overwritten or
merged into the new raw master.
