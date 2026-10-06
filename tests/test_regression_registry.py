# Migrated from study/tests/test_registry.py; see REGRESSION_MIGRATION.csv.
"""Registry + plug-in API tests (no dependency on the original paper code)."""
import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest

import dose_combination_bo as sdb


def test_builtins_registered():
    acqs = sdb.list_acquisitions()
    for name in [
        "cEI", "cKG", "cKG-legacy512", "cEI-tMSE", "tmse", "utmse",
        "entropy", "random", "sKG3", "SPW",
    ]:
        assert name in acqs
    # Compatibility aliases resolve, but discovery presents reader-facing names.
    assert "qBIG" not in acqs
    assert "SUR" not in acqs
    assert "feasibility-entropy" not in acqs
    surfaces = sdb.list_surfaces()
    assert set(surfaces) >= {"osa", "mariposa", "logistic", "gaussian_bump"}
    assert "efftox" not in surfaces
    assert "gbump" not in surfaces


def test_alias_resolves():
    exact = sdb.get_acquisition("cKG")
    assert sdb.get_acquisition("cKG-exact") is exact
    # cKG1fix is retained only as a compatibility alias for the old MC rule.
    assert (
        sdb.get_acquisition("cKG1fix")
        is sdb.get_acquisition("cKG-legacy512")
    )
    assert sdb.get_acquisition("cKG1fix") is not sdb.get_acquisition("cKG")
    entropy = sdb.get_acquisition("entropy")
    assert sdb.get_acquisition("feasibility-entropy") is entropy
    assert sdb.get_acquisition("qBIG") is entropy
    with pytest.warns(DeprecationWarning, match="not a standard SUR"):
        assert sdb.get_acquisition("SUR") is entropy
    assert sdb.get_surface("logistic") is sdb.get_surface("efftox")
    assert sdb.get_surface("gaussian_bump") is sdb.get_surface("gbump")


def test_public_acquisition_decorator_applies_common_assignment_mask():
    from dose_combination_bo.registry import acquisition as registry_acquisition

    assert registry_acquisition is sdb.acquisition

    class DummyContext:
        Xset = np.zeros((3, 2))

        def __init__(self):
            self.seen = None

        def restrict(self, values):
            self.seen = np.asarray(values, float).copy()
            return np.where([True, False, True], values, -np.inf)

    @registry_acquisition("raw_score_mask_test")
    def raw_scores(ctx):
        return np.array([1.0, 100.0, 2.0])

    ctx = DummyContext()
    observed = sdb.get_acquisition("raw_score_mask_test")(ctx)
    np.testing.assert_array_equal(ctx.seen, [1.0, 100.0, 2.0])
    np.testing.assert_array_equal(observed, [1.0, -np.inf, 2.0])


def test_public_acquisition_decorator_validates_score_shape_before_assignment():
    class DummyContext:
        Xset = np.zeros((3, 2))

        @staticmethod
        def restrict(values):  # pragma: no cover - validation must fail first
            raise AssertionError("restrict should not be called")

    @sdb.acquisition("wrong_score_shape_test")
    def wrong_shape(ctx):
        return np.array([1.0, 2.0])

    with pytest.raises(ValueError, match=r"scores must have shape \(3,\)"):
        sdb.get_acquisition("wrong_score_shape_test")(DummyContext())


def test_public_acquisition_rejects_accidentally_matrix_shaped_scores():
    class DummyContext:
        Xset = np.zeros((4, 2))

        @staticmethod
        def restrict(values):  # pragma: no cover - validation must fail first
            raise AssertionError("restrict should not be called")

    @sdb.acquisition("matrix_score_shape_test")
    def matrix_scores(ctx):
        return np.ones((2, 2))

    with pytest.raises(ValueError, match=r"got \(2, 2\)"):
        sdb.get_acquisition("matrix_score_shape_test")(DummyContext())


@pytest.mark.parametrize("name", ["entropy", "feasibility-entropy", "qBIG"])
def test_entropy_alias_trial_records_preserve_caller_identifier(name):
    record = sdb.run_trial(
        name,
        seed=0,
        z=0,
        gamma=0.7,
        sim="osa",
        budget=4,
    )
    assert record["policy"] == name


def test_sur_dispatch_warns_but_remains_compatible():
    with pytest.warns(DeprecationWarning, match="use 'entropy'"):
        record = sdb.run_trial(
            "SUR",
            seed=0,
            z=0,
            gamma=0.7,
            sim="osa",
            budget=4,
        )
    assert record["policy"] == "SUR"


def test_sur_sweep_preflight_warns_before_compute(monkeypatch):
    sweep_module = importlib.import_module("dose_combination_bo.sweep")
    monkeypatch.setattr(
        sweep_module,
        "run_trial",
        lambda policy, *args, **kwargs: {"policy": policy},
    )
    with pytest.warns(DeprecationWarning, match="use 'entropy'"):
        records = sweep_module.sweep(
            ["SUR"],
            ["osa"],
            seeds=[0],
            gammas=[0.7],
            modes=["latent"],
            strata=[0],
            budget=4,
            parallel=False,
            verbose=False,
        )
    assert records == [{"policy": "SUR"}]


def test_release_bindings_install_accurate_runtime_help():
    import dose_combination_bo.acquisitions as acquisitions
    import dose_combination_bo.ckg_exact as exact

    assert "cKG-exact" in acquisitions.__doc__
    assert "cKG-legacy512" in acquisitions.__doc__
    assert "entropy" in acquisitions.__doc__
    assert "deprecated compatibility alias" in acquisitions.__doc__
    assert "public ``cKG``" in exact.__doc__
    assert "backs public ``cKG``" in exact.ckg_scores_exact.__doc__




def test_cei_tmse_is_cei_tmse_alternation_not_ckg():
    """Pin the construction identity independently of the GP trial machinery."""
    class DummyContext:
        def __init__(self, step):
            self.step = step
            self.tmse = np.array([0.0, 5.0])

        @staticmethod
        def cei_scores():
            return np.array([7.0, 0.0])

        @staticmethod
        def restrict(values):
            return values

    rule = sdb.get_acquisition("cEI-tMSE")
    np.testing.assert_array_equal(rule(DummyContext(step=0)), [7.0, 0.0])
    np.testing.assert_array_equal(rule(DummyContext(step=1)), [0.0, 5.0])


def test_deprecated_straddle_aliases_still_resolve():
    # 'straddle'/'ustrad' were misnomers for the tMSE criterion; kept as aliases
    assert sdb.get_acquisition("straddle") is sdb.get_acquisition("tmse")
    assert sdb.get_acquisition("ustrad") is sdb.get_acquisition("utmse")


def test_ctx_straddle_property_matches_tmse():
    import numpy as np
    seen = {}

    @sdb.acquisition("ctx_alias_probe")
    def _probe(ctx):
        seen["equal"] = np.array_equal(ctx.straddle, ctx.tmse)
        return ctx.restrict(ctx.tmse)

    sdb.run_trial("ctx_alias_probe", seed=0, z=0, gamma=0.7, sim="osa", budget=12)
    assert seen["equal"]


def test_unknown_names_raise():
    with pytest.raises(KeyError):
        sdb.get_acquisition("does_not_exist")
    with pytest.raises(KeyError):
        sdb.get_surface("does_not_exist")


def test_custom_acquisition_runs():
    policy_draws = []

    @sdb.acquisition("greedy_efficacy_test")
    def greedy(ctx):
        mu_e, _ = ctx.latent_efficacy()
        return ctx.restrict(mu_e)

    @sdb.acquisition("greedy_rng_consumer_test")
    def greedy_rng_consumer(ctx):
        policy_draws.extend(ctx.rng.random(17))  # must not advance outcome randomness
        mu_e, _ = ctx.latent_efficacy()
        return ctx.restrict(mu_e)

    @sdb.acquisition("nan_score_test")
    def nan_score(ctx):
        scores = np.zeros(len(ctx.pf))
        scores[0] = np.nan
        return ctx.restrict(scores)

    assert "greedy_efficacy_test" in sdb.list_acquisitions()
    r = sdb.run_trial("greedy_efficacy_test", seed=0, z=0, gamma=0.7, sim="osa", budget=12)
    r2 = sdb.run_trial("greedy_rng_consumer_test", seed=0, z=0, gamma=0.7, sim="osa", budget=12)
    assert np.isfinite(r["dose_units"]) and r["dose_units"] >= 0
    assert r["policy"] == "greedy_efficacy_test"
    r.pop("policy"); r2.pop("policy")
    assert r == r2
    expected_rng = np.random.default_rng(np.random.SeedSequence([0, 823451]))
    assert np.array_equal(np.asarray(policy_draws), expected_rng.random(len(policy_draws)))
    with pytest.raises(ValueError, match="must not contain NaN"):
        sdb.run_trial("nan_score_test", seed=0, z=0, gamma=0.7, sim="osa", budget=12)


def test_custom_surface_runs():
    @sdb.surface("quad_test")
    def quad(z):
        eff = lambda a, b: -((a - 0.5) ** 2 + (b - 0.5) ** 2)
        tox = lambda a, b: a + b
        return dict(gd=1.0, dopt=(0.5, 0.5), fopt=0.0, eff=eff, tox=tox, sf=0.2, sg=0.2)

    r = sdb.run_trial("cEI", seed=0, z=0, gamma=0.7, sim="quad_test", budget=12, grid_n=3)
    assert np.isfinite(r["rpsel"])
    expected_du = np.linalg.norm(np.array([r["rec_d1"], r["rec_d2"]]) - np.array([0.5, 0.5])) / 0.5
    assert np.isclose(r["dose_units"], expected_du)
