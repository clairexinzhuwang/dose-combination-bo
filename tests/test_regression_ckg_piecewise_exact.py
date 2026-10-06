# Migrated from study/tests/test_ckg_piecewise_exact.py; see REGRESSION_MIGRATION.csv.
"""Validation tests for the opt-in deterministic hard-gated cKG evaluator."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm, qmc

import dose_combination_bo.ckg_exact as exact_module
from dose_combination_bo.acquisitions import ckg_one_step_gated
from dose_combination_bo.ckg_exact import (
    _affine_upper_envelope,
    _affine_max_expectation,
    _normal_interval_first_moment_log,
    _scale_by_log_probability,
    ckg_one_step_gated_exact,
    ckg_scores_exact,
    expected_terminal_value_exact,
)
from dose_combination_bo.gp import kg_lines


def _direct_value(mu_f, b_f, mu_g, b_g, sd_g_post, gd, gamma, zf, zg):
    muf = mu_f[None, :] + zf[:, None] * b_f[None, :]
    mug = mu_g[None, :] + zg[:, None] * b_g[None, :]
    z_feasibility = (gd - mug) / sd_g_post[None, :]
    gate = z_feasibility > norm.ppf(gamma)
    values = np.where(gate, muf, -np.inf).max(axis=1)
    empty = ~np.isfinite(values)
    if empty.any():
        values[empty] = muf[empty][
            np.arange(int(empty.sum())), z_feasibility[empty].argmax(axis=1)
        ]
    return values


def _adaptive_quadrature_reference(mu_f, b_f, mu_g, b_g, sd, gd, gamma):
    """Independent direct quadrature of the defining max/fallback expression."""
    qgamma = norm.ppf(gamma)
    alpha = (gd - mu_g) / sd
    beta = -b_g / sd

    def inner(zg):
        gate = alpha + beta * zg > qgamma
        if not gate.any():
            j = int(np.argmax(alpha + beta * zg))
            fn = lambda ze: (mu_f[j] + b_f[j] * ze) * norm.pdf(ze)
            return quad(fn, -9.0, 9.0, epsabs=2e-12, epsrel=2e-12, limit=200)[0]
        aa, bb = mu_f[gate], b_f[gate]
        crossings = []
        for i in range(len(aa)):
            for j in range(i + 1, len(aa)):
                if bb[i] != bb[j]:
                    x = (aa[j] - aa[i]) / (bb[i] - bb[j])
                    if -9.0 < x < 9.0:
                        crossings.append(float(x))
        fn = lambda ze: float(np.max(aa + bb * ze)) * norm.pdf(ze)
        return quad(
            fn, -9.0, 9.0, points=sorted(set(crossings)),
            epsabs=2e-11, epsrel=2e-11, limit=300,
        )[0]

    outer_points = []
    for a, b in zip(alpha, beta):
        if b != 0.0:
            x = (qgamma - a) / b
            if -9.0 < x < 9.0:
                outer_points.append(float(x))
    for i in range(len(alpha)):
        for j in range(i + 1, len(alpha)):
            if beta[i] != beta[j]:
                x = (alpha[j] - alpha[i]) / (beta[i] - beta[j])
                if -9.0 < x < 9.0:
                    outer_points.append(float(x))
    fn = lambda zg: inner(zg) * norm.pdf(zg)
    return quad(
        fn, -9.0, 9.0, points=sorted(set(outer_points)),
        epsabs=5e-10, epsrel=5e-10, limit=500,
    )[0]


def test_fixed_gate_reduces_to_existing_kg_lines():
    mu_f = np.array([-0.4, 0.7, 0.1])
    b_f = np.array([0.2, -0.8, 0.5])
    value, diagnostics = expected_terminal_value_exact(
        mu_f, b_f,
        mu_g=np.array([-2.0, -3.0, 4.0]),
        toxicity_update=np.zeros(3),
        sd_g_post=np.ones(3),
        g_dagger=0.0,
        gamma=0.7,
        return_diagnostics=True,
    )
    assert value == pytest.approx(kg_lines(mu_f[:2], b_f[:2]), abs=1e-14)
    assert diagnostics["gate_regime_count"] == 1
    assert diagnostics["nonempty_gate_regime_count"] == 1
    assert diagnostics["fallback_segment_count"] == 0


def test_affine_envelope_does_not_merge_distinct_close_slopes():
    delta = 9.0e-6
    expected = delta / np.sqrt(2.0 * np.pi)
    assert _affine_max_expectation([0.0, 0.0], [1.0, 1.0 + delta]) == pytest.approx(
        expected, rel=2e-12, abs=0.0
    )
    # This locks the reason the exact path cannot call the frozen legacy helper.
    assert kg_lines([0.0, 0.0], [1.0, 1.0 + delta]) == 0.0


def test_affine_envelope_integrates_extreme_upper_tail_stably():
    # E[max(0, 1e30*(Z-8))] = 1e30 * (phi(8) - 8*sf(8)).
    expected = 7.5502624119e13
    value = _affine_max_expectation([0.0, -8.0e30], [0.0, 1.0e30])
    assert value == pytest.approx(expected, rel=7e-12)
    assert kg_lines([0.0, -8.0e30], [0.0, 1.0e30]) < 0.0


def test_affine_envelope_single_large_constant_is_exact():
    assert _affine_max_expectation([1.0e16], [0.0]) == 1.0e16
    assert _scale_by_log_probability(1.0e16, 0.0) == 1.0e16


def test_affine_crossing_beyond_float_range_is_warning_free():
    # On platforms where long double has the same range as float64, the formal
    # crossing is +inf.  That is the correct envelope geometry and must not emit
    # a RuntimeWarning during a trial.
    with np.errstate(all="raise"):
        envelope = _affine_upper_envelope([0.0, 1.0], [0.0, 3.0e-310])
        value = _affine_max_expectation([0.0, 1.0], [0.0, 3.0e-310])
    # The crossing is -infinity here, so the second line dominates every finite
    # normal draw and the first line is removed from the upper hull.
    assert len(envelope) == 1
    assert envelope[0][0] == 1
    assert envelope[0][3] == -np.inf
    assert value == 1.0


def test_outer_tail_probability_can_resuscitate_a_large_conditional_value():
    log_tail = norm.logsf(38.0)
    value = _scale_by_log_probability(1.0e300, log_tail)
    assert value == pytest.approx(
        np.exp(np.log(1.0e300) + log_tail), rel=2e-14, abs=0.0
    )
    assert value > 0.0


def test_full_evaluator_resuscitates_an_outer_gate_beyond_subnormal_mass():
    # Dose 0 is admitted only for Z_g>38; dose 1 lies exactly on the strict gate
    # boundary and is the empty-gate fallback below 38.
    expected = np.exp(np.log(1.0e300) + norm.logsf(38.0))
    value = expected_terminal_value_exact(
        mu_f=[1.0e300, 0.0], efficacy_update=[0.0, 0.0],
        mu_g=[38.0, 0.0], toxicity_update=[-1.0, 0.0],
        sd_g_post=[1.0, 1.0], g_dagger=0.0, gamma=0.5,
    )
    assert value == pytest.approx(expected, rel=2e-14, abs=0.0)


def test_affine_envelope_deep_tail_uses_log_moments_not_subnormal_products():
    # 1e300 * E[(Z-38)+], from a high-precision independent oracle.
    value = _affine_max_expectation(
        [0.0, -38.0e300], [0.0, 1.0e300]
    )
    assert value == pytest.approx(7.5827518145492083e-18, rel=5e-9, abs=0.0)


def test_first_moment_handles_enormous_threshold_without_overflow():
    assert _normal_interval_first_moment_log(1.0e308, np.inf) == (0.0, -np.inf)


def test_always_empty_uses_most_feasible_fallback_and_integrates_efficacy():
    value, diagnostics = expected_terminal_value_exact(
        mu_f=[100.0, 2.5, -40.0],
        efficacy_update=[99.0, -17.0, 25.0],
        mu_g=[4.0, 2.0, 3.0],
        toxicity_update=[-0.0, 0.0, 0.0],
        sd_g_post=[1.0, 1.0, 1.0],
        g_dagger=0.0,
        gamma=0.9,
        return_diagnostics=True,
    )
    assert value == pytest.approx(2.5, abs=1e-14)
    assert diagnostics["empty_gate_regime_count"] == 1
    assert diagnostics["fallback_segment_count"] == 1
    assert diagnostics["regimes"][0]["fallback_index"] == 1


def test_empty_gate_fallback_crossing_is_partitioned():
    # The central empty-gate region switches its most-feasible line at zero.  The
    # two tail gates and the two fallback regions are symmetric, so E[V+] = 2.
    value, diagnostics = expected_terminal_value_exact(
        mu_f=[1.0, 3.0],
        efficacy_update=[4.0, -7.0],
        mu_g=[1.0, 1.0],
        toxicity_update=[1.0, -1.0],
        sd_g_post=[1.0, 1.0],
        g_dagger=0.0,
        gamma=0.5,
        return_diagnostics=True,
    )
    assert value == pytest.approx(2.0, abs=2e-14)
    assert diagnostics["empty_gate_regime_count"] == 1
    assert diagnostics["fallback_segment_count"] == 2


def test_adjacent_float_gate_thresholds_are_classified_symbolically(monkeypatch):
    lower = 1.0
    upper = np.nextafter(lower, np.inf)

    # This regression isolates symbolic regime membership when consecutive
    # float64 thresholds leave no representable probe.  Some SciPy/libm builds
    # reverse the two one-ulp-separated ``log_ndtr`` values in the later
    # interval-mass calculation.  Inject the positive local normal mass for
    # this single interval so backend-specific integration arithmetic cannot
    # mask the classification contract.  This fixture does not claim
    # cross-platform portability of ``_log_normal_interval_probability``.
    original_log_probability = exact_module._log_normal_interval_probability

    def log_probability(lo, hi):
        if lo == lower and hi == upper:
            return float(norm.logpdf(lower) + np.log(upper - lower))
        return original_log_probability(lo, hi)

    monkeypatch.setattr(
        exact_module, "_log_normal_interval_probability", log_probability
    )
    _value, diagnostics = expected_terminal_value_exact(
        mu_f=[1.0, 2.0],
        efficacy_update=[0.0, 0.0],
        # Standardised feasibility lines are z-lower and upper-z.
        mu_g=[lower, -upper],
        toxicity_update=[-1.0, 1.0],
        sd_g_post=[1.0, 1.0],
        g_dagger=0.0,
        gamma=0.5,
        return_diagnostics=True,
    )
    middle = next(
        row for row in diagnostics["regimes"]
        if row["z_g_lo"] == lower and row["z_g_hi"] == upper
    )
    assert middle["gate_indices"] == [0, 1]


def test_negative_zero_covariance_is_a_fixed_line():
    positive = expected_terminal_value_exact(
        [0.0, 1.0], [0.1, 0.2], [-2.0, 3.0], [0.0, 0.0], [1.0, 1.0],
        0.0, 0.7, return_diagnostics=True,
    )
    negative = expected_terminal_value_exact(
        [0.0, 1.0], [0.1, 0.2], [-2.0, 3.0], [-0.0, 0.0], [1.0, 1.0],
        0.0, 0.7, return_diagnostics=True,
    )
    assert negative[0] == positive[0]
    assert negative[1]["gate_threshold_count"] == 0


def test_permutation_invariance_without_positive_measure_ties():
    args = dict(
        mu_f=np.array([0.2, 1.4, -0.8, 0.7]),
        efficacy_update=np.array([-0.4, 0.1, 0.8, -0.2]),
        mu_g=np.array([-0.5, 0.8, 1.1, -0.1]),
        toxicity_update=np.array([0.3, -0.6, 0.2, -0.1]),
        sd_g_post=np.array([0.9, 1.2, 0.7, 1.1]),
        g_dagger=0.25,
        gamma=0.77,
    )
    expected = expected_terminal_value_exact(**args)
    permutation = np.array([2, 0, 3, 1])
    permuted = expected_terminal_value_exact(
        **{key: (value[permutation] if isinstance(value, np.ndarray) else value)
           for key, value in args.items()}
    )
    assert permuted == pytest.approx(expected, abs=2e-14)


def test_offset_invariance_of_signed_value():
    kwargs = dict(
        efficacy_update=np.array([-0.4, 0.1, 0.8]),
        mu_g=np.array([-0.5, 0.8, 1.1]),
        toxicity_update=np.array([0.3, -0.6, 0.2]),
        sd_g_post=np.array([0.9, 1.2, 0.7]),
        g_dagger=0.25,
        gamma=0.77,
    )
    mu_f = np.array([0.2, 1.4, -0.8])
    current = float(mu_f[0])
    base = expected_terminal_value_exact(mu_f=mu_f, **kwargs) - current
    shifted = (
        expected_terminal_value_exact(mu_f=mu_f + 1.0e6, **kwargs)
        - (current + 1.0e6)
    )
    assert shifted == pytest.approx(base, abs=2e-10)


def test_independent_adaptive_quadrature_matches_piecewise_exact():
    mu_f = np.array([-0.4, 1.2, 0.3, 0.9])
    b_f = np.array([0.8, -0.2, 0.45, -0.7])
    mu_g = np.array([-0.2, 0.9, 1.4, 0.1])
    b_g = np.array([0.35, -0.6, 0.18, -0.27])
    sd = np.array([0.8, 1.1, 0.7, 0.95])
    gd, gamma = 0.4, 0.82
    exact = expected_terminal_value_exact(mu_f, b_f, mu_g, b_g, sd, gd, gamma)
    reference = _adaptive_quadrature_reference(mu_f, b_f, mu_g, b_g, sd, gd, gamma)
    assert exact == pytest.approx(reference, abs=2e-9)


@pytest.mark.parametrize("scramble_seed", range(8))
def test_scrambled_sobol_reference_checks(scramble_seed):
    # Eight independent scramblings are a replication check, not a convergence
    # claim.  They directly evaluate the legacy max/fallback definition.
    mu_f = np.array([-0.3, 0.9, 1.1, 0.25, -0.6])
    b_f = np.array([0.55, -0.35, 0.12, 0.8, -0.2])
    mu_g = np.array([-0.4, 0.2, 1.0, 0.6, -0.1])
    b_g = np.array([0.25, -0.7, 0.16, -0.35, 0.42])
    sd = np.array([0.75, 1.05, 0.8, 1.2, 0.9])
    gd, gamma = 0.35, 0.78
    target = expected_terminal_value_exact(mu_f, b_f, mu_g, b_g, sd, gd, gamma)
    uniforms = qmc.Sobol(2, scramble=True, seed=scramble_seed).random_base2(17)
    normals = norm.ppf(np.clip(uniforms, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
    reference = float(_direct_value(
        mu_f, b_f, mu_g, b_g, sd, gd, gamma,
        normals[:, 0], normals[:, 1],
    ).mean())
    assert reference == pytest.approx(target, abs=1.2e-3)


class _Noise:
    def __init__(self, value):
        self._value = float(value)

    def item(self):
        return self._value


def test_model_wrapper_preserves_cohort_noise_gate_mode_and_signed_value(monkeypatch):
    mu_f = torch.tensor([4.0, -1.0, 0.5])
    mu_g = torch.tensor([-0.2, 0.7, 1.1])
    var_g = torch.tensor([0.9, 1.2, 0.8])
    cov_f = torch.tensor([-0.4, 0.2, 0.6])
    cov_g = torch.tensor([0.3, -0.5, -0.0])
    nf, ng = 0.8, 0.6
    s2f_full, s2g_full = 1.9, 1.7

    def fake_joint(model, likelihood, Xset, d):
        if model == "eff":
            return mu_f, torch.ones(3), cov_f, s2f_full
        return mu_g, var_g, cov_g, s2g_full

    def fake_post_latent(model, Xset):
        assert model == "tox"
        return mu_g, torch.tensor([0.5, 0.7, 0.9])

    monkeypatch.setattr(exact_module, "joint", fake_joint)
    monkeypatch.setattr(exact_module, "post_latent", fake_post_latent)
    Xset = torch.zeros((3, 2))
    le, lt = SimpleNamespace(noise=_Noise(nf)), SimpleNamespace(noise=_Noise(ng))

    for gate_mode in ("latent", "predictive"):
        for r_k in (1, 3):
            score, diagnostics = ckg_one_step_gated_exact(
                Xset[0], Xset, "eff", le, "tox", lt,
                0.25, 0.75, r_k=r_k, gate_mode=gate_mode,
                return_diagnostics=True,
            )
            s2f = (s2f_full - nf) + nf / r_k
            s2g = (s2g_full - ng) + ng / r_k
            obs = ng if gate_mode == "predictive" else 0.0
            sd_post = np.sqrt(np.maximum(var_g.numpy() - cov_g.numpy() ** 2 / s2g, 1e-12) + obs)
            expected_post = expected_terminal_value_exact(
                mu_f.numpy(), cov_f.numpy() / np.sqrt(s2f),
                mu_g.numpy(), cov_g.numpy() / np.sqrt(s2g),
                sd_post, 0.25, 0.75,
            )
            sd_now = np.sqrt(np.array([0.5, 0.7, 0.9]) + obs)
            z_now = (0.25 - mu_g.numpy()) / sd_now
            gate_now = z_now > norm.ppf(0.75)
            current = (mu_f.numpy()[gate_now].max() if gate_now.any()
                       else mu_f.numpy()[z_now.argmax()])
            assert score == pytest.approx(expected_post - current, abs=1e-14)
            assert diagnostics["r_k"] == r_k
            assert diagnostics["gate_mode"] == gate_mode


def test_current_empty_gate_semantics_are_most_feasible(monkeypatch):
    mu_f = torch.tensor([50.0, 2.0, -10.0])
    mu_g = torch.tensor([3.0, 1.0, 2.0])

    def fake_joint(model, likelihood, Xset, d):
        if model == "eff":
            return mu_f, torch.ones(3), torch.zeros(3), 2.0
        return mu_g, torch.ones(3), torch.zeros(3), 2.0

    monkeypatch.setattr(exact_module, "joint", fake_joint)
    monkeypatch.setattr(
        exact_module, "post_latent",
        lambda model, Xset: (mu_g, torch.ones(3)),
    )
    likelihood = SimpleNamespace(noise=_Noise(1.0))
    Xset = torch.zeros((3, 2))
    score, diagnostics = ckg_one_step_gated_exact(
        Xset[0], Xset, "eff", likelihood, "tox", likelihood,
        0.0, 0.9, return_diagnostics=True,
    )
    assert diagnostics["current_gate_indices"] == []
    assert diagnostics["current_fallback_index"] == 1
    assert score == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize(
    ("gamma", "z_now", "mu_f", "expected_gate", "expected_fallback"),
    [
        (0.9, np.array([-40.0, -39.0]), np.array([10.0, 0.0]), [], 1),
        (
            0.1,
            np.array([norm.ppf(0.1), norm.ppf(0.1) + 1.0]),
            np.array([5.0, 0.0]),
            [1],
            None,
        ),
    ],
)
def test_zero_information_is_invariant_in_saturated_and_strict_gate_cases(
    monkeypatch, gamma, z_now, mu_f, expected_gate, expected_fallback
):
    mu_f_t = torch.tensor(mu_f)
    mu_g_t = torch.tensor(-z_now)  # g_dagger=0 and sd=1 -> z_now exactly

    def fake_joint(model, likelihood, Xset, d):
        if model == "eff":
            return mu_f_t, torch.ones(2), torch.zeros(2), 2.0
        return mu_g_t, torch.ones(2), torch.zeros(2), 2.0

    monkeypatch.setattr(exact_module, "joint", fake_joint)
    monkeypatch.setattr(
        exact_module, "post_latent",
        lambda model, Xset: (mu_g_t, torch.ones(2)),
    )
    likelihood = SimpleNamespace(noise=_Noise(1.0))
    Xset = torch.zeros((2, 2))
    score, diagnostics = ckg_one_step_gated_exact(
        Xset[0], Xset, "eff", likelihood, "tox", likelihood,
        0.0, gamma, return_diagnostics=True,
    )
    assert diagnostics["current_gate_indices"] == expected_gate
    assert diagnostics["current_fallback_index"] == expected_fallback
    assert score == pytest.approx(0.0, abs=1e-14)








def test_score_is_signed_not_clamped(monkeypatch):
    # The current high-efficacy dose is feasible, but the fantasy frequently drops
    # it from the gate.  The exact evaluator must retain the resulting negative KG.
    mu_f = torch.tensor([5.0, 0.0])
    mu_g = torch.tensor([-1.0, 2.0])

    def fake_joint(model, likelihood, Xset, d):
        if model == "eff":
            return mu_f, torch.ones(2), torch.zeros(2), 1.0
        return mu_g, torch.ones(2), torch.tensor([1.2, 0.0]), 2.0

    monkeypatch.setattr(exact_module, "joint", fake_joint)
    monkeypatch.setattr(
        exact_module, "post_latent",
        lambda model, Xset: (mu_g, torch.full((2,), 0.05)),
    )
    le = SimpleNamespace(noise=_Noise(0.5))
    lt = SimpleNamespace(noise=_Noise(0.5))
    Xset = torch.zeros((2, 2))
    score = ckg_one_step_gated_exact(
        Xset[0], Xset, "eff", le, "tox", lt, 0.0, 0.7,
    )
    assert score < 0.0


def test_accessible_grid_is_fully_scored_before_eligibility_mask(monkeypatch):
    calls = []

    def fake_score(d, *args, **kwargs):
        calls.append(float(d[0]))
        return float(d[0])

    monkeypatch.setattr(exact_module, "ckg_one_step_gated_exact", fake_score)

    class FakeContext:
        Xset = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        me = le = mt = lt = object()
        g_dagger = 0.0
        gamma = 0.7
        r_k = 2
        gate_mode = "latent"

        @staticmethod
        def restrict(values):
            return np.where([True, False, True], values, -np.inf)

    scores = ckg_scores_exact(FakeContext())
    assert calls == [0.0, 0.5, 1.0]
    assert np.array_equal(scores, np.array([0.0, -np.inf, 1.0]))


def test_legacy_mc_evaluator_remains_explicitly_available():
    import dose_combination_bo as sdb

    assert callable(ckg_one_step_gated)
    assert "cKG" in sdb.list_acquisitions()
    assert "cKG-legacy512" in sdb.list_acquisitions()
    assert sdb.get_acquisition("cKG") is sdb.get_acquisition("cKG-exact")
    assert (
        sdb.get_acquisition("cKG-legacy512")
        is sdb.get_acquisition("cKG1fix")
    )


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"gamma": 1.0},
        {"gamma": 0.0},
        {"sd_g_post": [1.0, 0.0]},
        {"toxicity_update": [0.0]},
    ],
)
def test_input_validation(bad_kwargs):
    kwargs = dict(
        mu_f=[0.0, 1.0], efficacy_update=[0.0, 0.0],
        mu_g=[0.0, 1.0], toxicity_update=[0.0, 0.0],
        sd_g_post=[1.0, 1.0], g_dagger=0.0, gamma=0.7,
    )
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        expected_terminal_value_exact(**kwargs)
