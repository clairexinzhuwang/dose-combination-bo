# Deterministic piecewise-analytic cKG validation panel

Status: `PASS_DETERMINISTIC_PIECEWISE_ANALYTIC_VALIDATION_EXACT_UP_TO_FLOATING_POINT_STABLE_Z_HARNESS_REGISTERED_CKG_EVALUATOR_UNCHANGED`.

The primary panel reconstructs 720 posterior states (40 existing trajectories x 18 decisions) and saves all 18,000 piecewise-analytic query scores (exact up to floating-point arithmetic) together with the nested 512- and prefix-preserving 1024-fantasy vectors.
Both archived decision grids were reconstructed exactly (720/720 for 512; 720/720 for 1024).
The current Context implements the stable standardized gate and full/eligible fallback at every audited state; historical CDF rounding differences are retained as diagnostics across all 720 primary states.

| Approximation | Exact selection agreement | Nonempty-gate agreement | Max exact regret | Max full-grid score error |
|---|---:|---:|---:|---:|
| legacy_mc_512_seed0 | 466/720 | 411/665 | 0.0275538 | 0.357119 |
| prefix_preserving_mc_1024_seed0 | 537/720 | 482/665 | 0.015958 | 0.245651 |

The stratified Sobol audit contains 64 primary states and 512 independently scrambled checks (131,072 points each); maximum absolute error was 9.39845e-05.
The additive sentinel panel contains 20 four-testbed/five-gate states, 4 predictive-gate states, and 4 reconstructed gradual-access states.

This validates a deterministic piecewise-analytic evaluator, exact up to floating-point arithmetic, on frozen/reconstructed posterior states and numerical sentinels. It does not run formal OC trials, prove a Monte Carlo convergence rate, or authorize changing the registered cKG default.
