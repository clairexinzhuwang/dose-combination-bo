# Migrated from study/tests/test_stable_z_gate_semantics.py; see REGRESSION_MIGRATION.csv.
"""Regression tests for the shared strict-z feasibility-gate semantics.

The chance constraint is evaluated as ``z > ppf(tau)`` and empty gates fall
back to ``argmax(z)``.  CDF values remain useful reports, but floating-point
rounding in ``Phi(z)`` must not change an allocation or recommendation.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.stats import norm

import dose_combination_bo.acquisitions as acquisition_module
import dose_combination_bo.context as context_module
import dose_combination_bo.metrics as metrics_module
import dose_combination_bo.trial as trial_module
from dose_combination_bo.context import Context


torch.set_default_dtype(torch.double)


def _force_reported_cdf_tie(monkeypatch, value=0.25):
    """Make the reporting-scale CDF tie deterministic across SciPy builds.

    These tests exercise the rule that operational choices use standardized
    feasibility rather than a rounded feasibility probability.  Whether the
    particular reproduced tail values round to one or two float64 CDF values is
    an implementation detail of the installed SciPy build, so construct the
    reporting tie explicitly instead of relying on that detail.
    """

    def tied_cdf(values):
        array = np.asarray(values, dtype=float)
        result = np.full(array.shape, float(value), dtype=float)
        return float(result) if result.ndim == 0 else result

    monkeypatch.setattr(context_module.ss.norm, "cdf", tied_cdf)


def _context_from_z(
    monkeypatch,
    z_values,
    *,
    gamma=0.7,
    gate_mode="latent",
    candidate_mask=None,
):
    z = np.asarray(z_values, dtype=float)
    mu = torch.tensor(-z, dtype=torch.double)
    variance = torch.ones(len(z), dtype=torch.double)

    def posterior(model, points):
        del model
        assert int(points.shape[0]) == len(z)
        return mu.clone(), variance.clone()

    monkeypatch.setattr(context_module, "post_latent", posterior)
    monkeypatch.setattr(context_module, "post_predictive", lambda model, likelihood, points: posterior(model, points))
    return Context(
        Xset=torch.zeros((len(z), 2), dtype=torch.double),
        me=object(),
        le=object(),
        mt=object(),
        lt=object(),
        g_dagger=0.0,
        gamma=gamma,
        gate_mode=gate_mode,
        rng=np.random.default_rng(123),
        candidate_mask=candidate_mask,
    )


@pytest.mark.parametrize("gate_mode", ["latent", "predictive"])
def test_context_uses_strict_z_gate_at_equality_and_adjacent_floats(
    monkeypatch, gate_mode
):
    gamma = 0.7
    _force_reported_cdf_tie(monkeypatch, value=gamma)
    q = float(norm.ppf(gamma))
    z = np.array(
        [
            np.nextafter(q, -np.inf),
            q,
            np.nextafter(q, np.inf),
        ]
    )
    ctx = _context_from_z(
        monkeypatch, z, gamma=gamma, gate_mode=gate_mode
    )

    assert ctx.gate_quantile == q
    np.testing.assert_array_equal(ctx.standardized_feasibility, z)
    np.testing.assert_array_equal(ctx.gate_safe, [False, False, True])
    np.testing.assert_array_equal(ctx.safe, [False, False, True])
    # Phi rounds all three adjacent inputs to tau on this runtime.  The upper
    # adjacent z is nevertheless strictly above q and must be admitted.
    np.testing.assert_array_equal(ctx.pf > gamma, [False, False, False])


def test_gbump_seed55_rounding_tie_uses_z_not_cdf_for_operational_fallback(
    monkeypatch,
):
    # Reproduced acquisition state: gbump, seed 55, stratum 1, latent gate,
    # tau=.6, step 16.  The three CDF values are exactly tied in float64 even
    # though the middle standardized feasibility is larger.
    z = np.array(
        [
            -1.0382382718584444,
            -1.0382382718584440,
            -1.0382382718584444,
        ]
    )
    _force_reported_cdf_tie(monkeypatch)
    ctx = _context_from_z(monkeypatch, z, gamma=0.6)

    assert ctx.gate_empty is True
    assert ctx.most_feasible_full_index == 1
    assert ctx.most_feasible_index == 1
    assert len(set(ctx.pf.tolist())) == 1
    assert int(np.argmax(ctx.pf)) == 0


def test_full_and_protocol_eligible_fallbacks_are_separate_z_argmaxes(monkeypatch):
    # All three CDF values round to the same float.  Index 1 is best overall but
    # is protocol-ineligible; index 2 is the best eligible dose.
    z = np.array(
        [
            -1.0382382718584444,
            -1.0382382718584440,
            -1.0382382718584442,
        ]
    )
    _force_reported_cdf_tie(monkeypatch)
    candidate = np.array([True, False, True])
    ctx = _context_from_z(
        monkeypatch, z, gamma=0.6, candidate_mask=candidate
    )

    assert ctx.gate_empty is True
    assert ctx.most_feasible_full_index == 1
    assert ctx.most_feasible_index == 2
    assert int(np.argmax(np.where(candidate, ctx.pf, -np.inf))) == 0

    scores = acquisition_module._random(ctx)
    np.testing.assert_array_equal(np.isfinite(scores), [False, False, True])
    assert int(np.argmax(scores)) == 2


def test_exact_z_ties_preserve_numpy_first_index_tie_break(monkeypatch):
    ctx = _context_from_z(
        monkeypatch,
        [-2.0, -2.0, -3.0],
        gamma=0.9,
        candidate_mask=[False, True, True],
    )
    assert ctx.most_feasible_full_index == 0
    assert ctx.most_feasible_index == 1


@pytest.mark.parametrize("gate_mode", ["latent", "predictive"])
def test_recommend_obd_excludes_equality_and_admits_nextafter(
    monkeypatch, gate_mode
):
    gamma = 0.7
    q = float(norm.ppf(gamma))
    z = np.array([q, np.nextafter(q, np.inf), q - 1.0])
    efficacy = torch.tensor([100.0, 1.0, 50.0])
    if gate_mode == "latent":
        tox_variance = torch.ones(3)
        tox_noise = 0.0
        tox_mean = torch.tensor(-z)
    else:
        tox_variance = torch.full((3,), 3.0)
        tox_noise = 1.0
        tox_mean = torch.tensor(-2.0 * z)

    eff_model, tox_model = object(), object()

    def posterior(model, points):
        assert int(points.shape[0]) == 3
        if model is eff_model:
            return efficacy.clone(), torch.ones(3)
        assert model is tox_model
        return tox_mean.clone(), tox_variance.clone()

    monkeypatch.setattr(metrics_module, "post_latent", posterior)
    Xset = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    recommendation, index = metrics_module.recommend_obd(
        Xset,
        eff_model,
        object(),
        tox_model,
        SimpleNamespace(noise=tox_noise),
        0.0,
        gamma=gamma,
        gate_mode=gate_mode,
    )

    assert index == 1
    np.testing.assert_array_equal(recommendation, [0.5, 0.0])


def test_recommend_obd_empty_gate_uses_gbump_z_tie_break(monkeypatch):
    z = np.array(
        [
            -1.0382382718584444,
            -1.0382382718584440,
            -1.0382382718584444,
        ]
    )
    _force_reported_cdf_tie(monkeypatch)
    eff_model, tox_model = object(), object()

    def posterior(model, points):
        assert int(points.shape[0]) == 3
        if model is eff_model:
            return torch.tensor([100.0, -5.0, 50.0]), torch.ones(3)
        assert model is tox_model
        return torch.tensor(-z), torch.ones(3)

    monkeypatch.setattr(metrics_module, "post_latent", posterior)
    Xset = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    recommendation, index = metrics_module.recommend_obd(
        Xset,
        eff_model,
        object(),
        tox_model,
        SimpleNamespace(noise=0.0),
        0.0,
        gamma=0.6,
    )

    assert index == 1
    np.testing.assert_array_equal(recommendation, [0.5, 0.0])
    assert int(np.argmax(norm.cdf(z))) == 0


def test_terminal_gate_pass_counts_use_strict_z_not_rounded_cdf(monkeypatch):
    gamma = 0.7
    _force_reported_cdf_tie(monkeypatch, value=gamma)
    q = float(norm.ppf(gamma))
    z = np.array(
        [q, np.nextafter(q, np.inf), np.nextafter(q, -np.inf), -1.0]
    )
    models = iter([("eff", object()), ("tox", object())])

    monkeypatch.setattr(trial_module, "fit_gp", lambda *args, **kwargs: next(models))
    monkeypatch.setattr(
        trial_module,
        "get_acquisition",
        lambda policy: (lambda ctx: np.zeros(ctx.Xset.shape[0])),
    )
    monkeypatch.setattr(
        trial_module,
        "resolve_surface",
        lambda sim, stratum: {
            "gd": 0.0,
            "dopt": np.array([0.0, 0.0]),
            "fopt": 0.0,
            "sf": 0.1,
            "sg": 0.1,
            "eff": lambda d1, d2: 0.0,
            "tox": lambda d1, d2: -1.0,
        },
    )
    monkeypatch.setattr(
        trial_module,
        "recommend_obd",
        lambda Xset, *args, **kwargs: (Xset[0].cpu().numpy(), 0),
    )

    def posterior(model, points):
        n = int(points.shape[0])
        if model == "eff":
            return torch.zeros(n), torch.ones(n)
        assert model == "tox"
        use_z = z if n == len(z) else np.repeat(z[0], n)
        return torch.tensor(-use_z), torch.ones(n)

    monkeypatch.setattr(trial_module, "post_latent", posterior)
    monkeypatch.setattr(
        trial_module,
        "post_predictive",
        lambda model, likelihood, points: posterior(model, points),
    )

    result = trial_module.run_trial(
        "stable-z-test",
        seed=0,
        z=0,
        gamma=gamma,
        sim="test-surface",
        budget=2,
        warmup=2,
        r_k=2,
        grid_n=2,
    )

    assert int(np.count_nonzero(norm.cdf(z) > gamma)) == 0
    assert result["n_gate_pass"] == 1
    assert result["n_gate_pass_safe"] == 1


@pytest.mark.parametrize("gate_mode", ["latent", "predictive"])
def test_mc_ckg_current_and_fantasy_fallbacks_match_stable_z_definition(
    monkeypatch, gate_mode
):
    _force_reported_cdf_tie(monkeypatch)
    gamma = 0.6
    z_now_target = np.array(
        [-1.0382382718584444, -1.0382382718584440]
    )
    mu_f = torch.tensor([10.0, 0.0])
    var_f = torch.ones(2)
    cov_f = torch.zeros(2)
    var_g = torch.ones(2)
    cov_g = torch.tensor([0.4, -0.4])
    nf = ng = 1.0
    obs_g = ng if gate_mode == "predictive" else 0.0
    sd_now = np.sqrt(1.0 + obs_g)
    mu_g = torch.tensor(-z_now_target * sd_now)
    s2f_full = s2g_full = 2.0

    def fake_joint(model, likelihood, Xset, d):
        del likelihood, Xset, d
        if model == "eff":
            return mu_f, var_f, cov_f, s2f_full
        assert model == "tox"
        return mu_g, var_g, cov_g, s2g_full

    def fake_post_latent(model, Xset):
        assert model == "tox"
        return mu_g, torch.ones(int(Xset.shape[0]))

    monkeypatch.setattr(acquisition_module, "joint", fake_joint)
    monkeypatch.setattr(acquisition_module, "post_latent", fake_post_latent)
    likelihood = SimpleNamespace(noise=torch.tensor(1.0))
    Xset = torch.zeros((2, 2))
    nmc, seed = 257, 55

    observed = acquisition_module.ckg_one_step_gated(
        Xset[0],
        Xset,
        "eff",
        likelihood,
        "tox",
        likelihood,
        0.0,
        gamma,
        r_k=1,
        nmc=nmc,
        seed=seed,
        gate_mode=gate_mode,
    )

    s2f = (s2f_full - nf) + nf
    s2g = (s2g_full - ng) + ng
    wf = cov_f.numpy() / s2f
    wg = cov_g.numpy() / s2g
    sf, sg = np.sqrt(s2f), np.sqrt(s2g)
    sd_post = np.sqrt(
        np.maximum(var_g.numpy() - cov_g.numpy() ** 2 / s2g, 1e-12)
        + obs_g
    )
    rng = np.random.default_rng(seed)
    zf = rng.standard_normal((nmc, 1))
    zg = rng.standard_normal((nmc, 1))
    muf2 = mu_f.numpy()[None, :] + wf[None, :] * sf * zf
    mug2 = mu_g.numpy()[None, :] + wg[None, :] * sg * zg
    z2 = (0.0 - mug2) / sd_post[None, :]
    gate2 = z2 > float(norm.ppf(gamma))
    future = np.where(gate2, muf2, -np.inf).max(axis=1)
    empty = ~np.isfinite(future)
    future[empty] = muf2[empty][
        np.arange(int(empty.sum())), z2[empty].argmax(axis=1)
    ]
    z_now = (0.0 - mu_g.numpy()) / sd_now
    gate_now = z_now > float(norm.ppf(gamma))
    current = (
        float(mu_f.numpy()[gate_now].max())
        if gate_now.any()
        else float(mu_f.numpy()[z_now.argmax()])
    )
    expected = float(future.mean() - current)

    # A CDF fallback picks index 0 in the reproduced tie, while z picks index 1.
    # The ten-unit discrepancy makes this a regression test for both the current
    # and imagined empty-gate branches, not just a near-equality smoke check.
    cdf_current = float(mu_f.numpy()[norm.cdf(z_now).argmax()])
    cdf_reference = float(future.mean() - cdf_current)
    assert int(z_now.argmax()) == 1
    assert int(norm.cdf(z_now).argmax()) == 0
    assert observed == pytest.approx(expected, abs=1e-14)
    assert abs(observed - cdf_reference) == pytest.approx(10.0, abs=1e-14)


def test_mc_ckg_fantasy_empty_gate_fallback_uses_z_under_cdf_tie(monkeypatch):
    _force_reported_cdf_tie(monkeypatch)
    target_fantasy_z = np.array(
        [-1.0382382718584444, -1.0382382718584440]
    )
    sd_post = np.array([1.0, 0.5])
    mu_g = torch.tensor(-target_fantasy_z * sd_post)
    cov_g = torch.tensor([0.0, np.sqrt(1.5)])
    mu_f = torch.tensor([10.0, 0.0])

    def fake_joint(model, likelihood, Xset, d):
        del likelihood, Xset, d
        if model == "eff":
            return mu_f, torch.ones(2), torch.zeros(2), 2.0
        assert model == "tox"
        return mu_g, torch.ones(2), cov_g, 2.0

    monkeypatch.setattr(acquisition_module, "joint", fake_joint)
    monkeypatch.setattr(
        acquisition_module,
        "post_latent",
        lambda model, Xset: (mu_g, torch.ones(2)),
    )

    class ZeroRng:
        @staticmethod
        def standard_normal(shape):
            return np.zeros(shape)

    monkeypatch.setattr(
        acquisition_module.np.random,
        "default_rng",
        lambda seed: ZeroRng(),
    )
    likelihood = SimpleNamespace(noise=torch.tensor(1.0))
    Xset = torch.zeros((2, 2))
    observed = acquisition_module.ckg_one_step_gated(
        Xset[0],
        Xset,
        "eff",
        likelihood,
        "tox",
        likelihood,
        0.0,
        0.6,
        nmc=3,
        seed=0,
        gate_mode="latent",
    )

    current_z = -mu_g.numpy()
    assert int(current_z.argmax()) == 1
    assert int(target_fantasy_z.argmax()) == 1
    assert int(norm.cdf(target_fantasy_z).argmax()) == 0
    assert observed == pytest.approx(0.0, abs=1e-14)


def test_mc_ckg_fantasy_gate_is_strict_at_ppf_equality(monkeypatch):
    gamma = 0.1
    _force_reported_cdf_tie(
        monkeypatch, value=float(np.nextafter(gamma, np.inf))
    )
    q = float(norm.ppf(gamma))
    target_fantasy_z = np.array([q, np.nextafter(q, np.inf)])
    sd_post = np.array([1.0, 0.5])
    mu_g = torch.tensor(-target_fantasy_z * sd_post)
    cov_g = torch.tensor([0.0, np.sqrt(1.5)])
    mu_f = torch.tensor([10.0, 0.0])

    def fake_joint(model, likelihood, Xset, d):
        del likelihood, Xset, d
        if model == "eff":
            return mu_f, torch.ones(2), torch.zeros(2), 2.0
        assert model == "tox"
        return mu_g, torch.ones(2), cov_g, 2.0

    monkeypatch.setattr(acquisition_module, "joint", fake_joint)
    monkeypatch.setattr(
        acquisition_module,
        "post_latent",
        lambda model, Xset: (mu_g, torch.ones(2)),
    )

    class ZeroRng:
        @staticmethod
        def standard_normal(shape):
            return np.zeros(shape)

    monkeypatch.setattr(
        acquisition_module.np.random,
        "default_rng",
        lambda seed: ZeroRng(),
    )
    likelihood = SimpleNamespace(noise=torch.tensor(1.0))
    Xset = torch.zeros((2, 2))
    observed = acquisition_module.ckg_one_step_gated(
        Xset[0],
        Xset,
        "eff",
        likelihood,
        "tox",
        likelihood,
        0.0,
        gamma,
        nmc=3,
        seed=0,
        gate_mode="latent",
    )

    np.testing.assert_array_equal(target_fantasy_z > q, [False, True])
    # At this lower-tail equality Phi(q) rounds above tau.  A CDF gate would
    # incorrectly admit the high-efficacy first dose in every fantasy.
    assert norm.cdf(target_fantasy_z[0]) > gamma
    assert observed == pytest.approx(0.0, abs=1e-14)
