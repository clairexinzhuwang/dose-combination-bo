# Stable normal interval probabilities

CDF subtraction can lose small interval probabilities. For example,
(-1e-17, 1e-17) has probability about 7.978845608028654e-18, but both CDF values
round to 0.5. The earlier helper then attempted log(0). The same problem can
affect closely spaced endpoints on one side of zero.

Both probability interfaces now use the same log-mass calculation. For finite
narrow intervals let w = hi-lo, h = w/2, and m be the midpoint. After factoring
out w phi(m), the residual integral is the uniform average of
exp(q(s)), where q(s) = -A s - C s^2, A = m h, C = h^2/2, and -1 <= s <= 1.

When B = |A|+C <= 0.001, integrate the degree-four Taylor polynomial in q
analytically. Its correction to one is

    -C/3 + A^2/6 + C^2/10 - A^2 C/10 - C^3/42
    + A^4/120 + A^2 C^2/28 + C^4/216.

The exponential remainder is at most exp(B) B^5/120. Since the exact average
is at least exp(-B), the relative error is at most exp(2B) B^5/120; the
corresponding log error is below 8.4e-18 at the branch bound. This is a
truncation bound, not a bound on binary64 rounding. The implementation also
accounts for midpoint rounding at adjacent floating-point endpoints and
computes log(w) before any density multiplication can underflow. The branch
condition does not depend on endpoint signs or an endpoint-ULP count.

Outside that small-density-variation region, same-sign intervals use the
scaled complementary error function. For 0 <= lo < hi, the survival ratio is

    exp(-(hi-lo)(hi+lo)/2) * erfcx(hi/sqrt(2)) / erfcx(lo/sqrt(2)).

Use log_ndtr(-lo) for the leading tail and expm1 for one minus the ratio.
This avoids subtracting nearly equal large negative log-CDF values. Negative
intervals are reflected. Intervals crossing zero use the sum of positive erf
masses; infinite endpoints retain the exact limiting definitions.

Ordinary probability is `exp(log_probability)` and may underflow to zero;
the finite log mass remains available to the signed-log integrator. Empty
intervals have zero probability. The calculation preserves small intervals
and signed cKG values without probability floors.

Tests include ten regression cases and sixty independent 100-decimal
integral references spanning cross-zero and same-sign intervals, subnormal
masses, tails, infinite endpoints and the expansion threshold. The complete
narrow-interval example returns `1.5`. See the
[verification report](https://github.com/clairexinzhuwang/dose-combination-bo/blob/main/distributions/VERIFICATION_REPORT.md).

References: [SciPy log_ndtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_ndtr.html),
[SciPy erf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erf.html),
and [SciPy erfcx](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfcx.html).
