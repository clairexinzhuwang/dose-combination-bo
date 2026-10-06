"""New regressions for the maintained cKG numerical backend.

Run against the truly installed wheel, from outside both source trees:
    python -m pytest -q /path/to/test_rc3_interval_regression.py

These assertions describe correct numerical behavior. Do not invert them into
an expected-crash test, skip missing dependencies, or replace the integrator.
The central-interval oracle has relative truncation error O(eps**2), far below
binary64 precision at the tested eps values. The same-sign oracle was computed
independently with 90-decimal normal-density integration at exact float bounds.
"""
import math

import pytest
from dose_combination_bo.ckg_exact import (
    _normal_interval_probability,
    _log_normal_interval_probability,
    expected_terminal_value_exact,
)


@pytest.mark.parametrize("eps", [1e-17, 1e-20, 1e-30])
def test_cross_zero_interval_has_positive_mass_and_finite_log(eps):
    expected_p = (2.0 * eps) / math.sqrt(2.0 * math.pi)
    expected_logp = math.log(2.0 * eps) - 0.5 * math.log(2.0 * math.pi)
    p = _normal_interval_probability(-eps, eps)
    logp = _log_normal_interval_probability(-eps, eps)
    assert p > 0.0
    assert math.isfinite(logp)
    assert p == pytest.approx(expected_p, rel=2e-13, abs=0.0)
    assert logp == pytest.approx(expected_logp, rel=0.0, abs=2e-13)
    assert math.exp(logp) == pytest.approx(p, rel=2e-13, abs=0.0)


@pytest.mark.parametrize("eps", [1e-17, 1e-20, 1e-30])
def test_complete_terminal_evaluator_accepts_narrow_gate_partition(eps):
    value = expected_terminal_value_exact(
        [1.0, 2.0], [0.0, 0.0], [-eps, eps],
        [1.0, 1.0], [1.0, 1.0], 0.0, 0.5,
    )
    # Exact expression: 1 + Phi(-eps), which rounds to 1.5 at these eps.
    assert math.isfinite(value)
    assert value == pytest.approx(1.5, rel=0.0, abs=2e-14)


def test_near_coincident_same_sign_interval_matches_independent_oracle():
    lo = float.fromhex("-0x1.e4b499763f335p-1")
    hi = float.fromhex("-0x1.e4b499763f331p-1")
    expected_p = float("1.1317956281629873300442331323661600043563174635957e-16")
    expected_logp = float("-36.717556064915196882343392105113584080531785833616")
    p = _normal_interval_probability(lo, hi)
    logp = _log_normal_interval_probability(lo, hi)
    assert p == pytest.approx(expected_p, rel=2e-13, abs=0.0)
    assert logp == pytest.approx(expected_logp, rel=0.0, abs=2e-13)
    assert math.exp(logp) == pytest.approx(p, rel=2e-13, abs=0.0)


def test_underflowed_mass_retains_finite_log_mass():
    p = _normal_interval_probability(40.0, 41.0)
    logp = _log_normal_interval_probability(40.0, 41.0)
    assert p == 0.0  # Legitimate binary64 underflow; do not add an epsilon.
    assert math.isfinite(logp)
    assert logp < -800.0


@pytest.mark.parametrize("lo,hi", [(0.0, 0.0), (1.0, 0.0)])
def test_empty_interval_contract_is_unchanged(lo, hi):
    assert _normal_interval_probability(lo, hi) == 0.0
    assert _log_normal_interval_probability(lo, hi) == -math.inf
