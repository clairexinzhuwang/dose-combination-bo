"""Independent normal-density integration at exact binary64 endpoints."""
import math
import mpmath as mp
import pytest
from dose_combination_bo.ckg_exact import (
    _normal_interval_probability, _log_normal_interval_probability,
)

MIN_SUBNORMAL = math.ulp(0.0)
CASES = [(-math.inf, math.inf), (-math.inf, 0.0), (0.0, math.inf),
         (-math.inf, -40.0), (40.0, math.inf), (-40.0, -39.0),
         (40.0, 41.0), (-1.0, 1.0), (-0.1, 0.7),
         (-MIN_SUBNORMAL, MIN_SUBNORMAL), (0.0, MIN_SUBNORMAL),
         (-1e-300, 2e-300), (-1e-100, 1e-100), (-1e-17, 1e-17)]
for center in (-1e6, -1000.0, -40.0, -8.0, -1.0, -0.1,
               0.1, 1.0, 8.0, 40.0, 1000.0, 1e6):
    for ulps in (1, 4, 64):
        lo = center
        hi = lo
        for _ in range(ulps):
            hi = math.nextafter(hi, math.inf)
        CASES.append((lo, hi))
# Exercise both sides of the local-density expansion's error-bound threshold.
for center in (-40.0, -1.0, 0.0, 1.0, 40.0):
    half = 0.002 / (math.sqrt(center * center + 0.002) + abs(center))
    for factor in (0.99, 1.01):
        CASES.append((center - factor * half, center + factor * half))


def oracle(lo, hi):
    with mp.workdps(100):
        l, h = mp.mpf(lo), mp.mpf(hi)
        if math.isinf(lo):
            p = mp.erfc(-h / mp.sqrt(2)) / 2
            logp = mp.log(p)
        elif math.isinf(hi):
            p = mp.erfc(l / mp.sqrt(2)) / 2
            logp = mp.log(p)
        else:
            width, midpoint = h - l, (h + l) / 2
            scaled_mass = mp.quad(
                lambda t: mp.exp(-midpoint * width * t - (width * t)**2 / 2),
                [-mp.mpf('0.5'), mp.mpf('0.5')],
            )
            logp = (mp.log(width) - midpoint**2 / 2
                    - mp.log(2 * mp.pi) / 2 + mp.log(scaled_mass))
            p = mp.exp(logp)
        return float(p), float(logp)


@pytest.mark.parametrize('lo,hi', CASES)
def test_probability_and_log_against_independent_density_integral(lo, hi):
    expected_p, expected_logp = oracle(lo, hi)
    p = _normal_interval_probability(lo, hi)
    logp = _log_normal_interval_probability(lo, hi)
    assert math.isfinite(logp)
    assert logp == pytest.approx(expected_logp, rel=0.0,
                                 abs=max(2e-12, 4 * math.ulp(expected_logp)))
    assert p == pytest.approx(expected_p, rel=3e-12,
                             abs=2 * MIN_SUBNORMAL)
    # Both interfaces describe the same mass, including true underflow.
    assert p == math.exp(logp)
