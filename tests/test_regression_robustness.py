# Migrated from study/tests/test_robustness.py; see REGRESSION_MIGRATION.csv.
"""Regression tests for the review's findings: fail-loud contracts, no silent
surface fallback, (n,1) tolerance, design-aware resume, and Ray worker registry."""
import numpy as np
import pytest

import dose_combination_bo as sdb


def test_unknown_surface_raises_not_silent_fallback():
    # previously fell back to gbump and mislabeled the result
    with pytest.raises(KeyError):
        sdb.run_trial("cEI", seed=0, z=0, gamma=0.7, sim="efftx_typo", budget=12)


def test_acquisition_wrong_shape_raises():
    @sdb.acquisition("bad_shape_test")
    def bad(ctx):
        return np.zeros(3)  # grid is 25, not 3

    with pytest.raises(ValueError, match="scores"):
        sdb.run_trial("bad_shape_test", seed=0, z=0, gamma=0.7, sim="osa", budget=12)


def test_acquisition_nan_raises():
    @sdb.acquisition("nan_test")
    def nan_acq(ctx):
        v = np.zeros(ctx.Xset.shape[0]); v[0] = np.nan
        return v

    with pytest.raises(ValueError, match="NaN"):
        sdb.run_trial("nan_test", seed=0, z=0, gamma=0.7, sim="osa", budget=12)

    @sdb.acquisition("positive_inf_test")
    def positive_inf_acq(ctx):
        v = np.zeros(ctx.Xset.shape[0]); v[0] = np.inf
        return v

    with pytest.raises(ValueError, match=r"\+inf"):
        sdb.run_trial("positive_inf_test", seed=0, z=0, gamma=0.7, sim="osa", budget=12)


def test_column_vector_return_tolerated():
    @sdb.acquisition("colvec_test")
    def colvec(ctx):
        mu_e, _ = ctx.latent_efficacy()
        return mu_e.reshape(-1, 1)  # the public wrapper accepts (n,1) without broadcasting

    r = sdb.run_trial("colvec_test", seed=0, z=0, gamma=0.7, sim="osa", budget=12)
    assert np.isfinite(r["dose_units"])


def test_surface_missing_key_raises():
    @sdb.surface("incomplete_test")
    def incomplete(z):
        return dict(gd=1.0, eff=lambda a, b: 0.0, tox=lambda a, b: 0.0)  # missing dopt/fopt/sf/sg

    with pytest.raises(KeyError):
        sdb.run_trial("cEI", seed=0, z=0, gamma=0.7, sim="incomplete_test", budget=12)

    @sdb.surface("inconsistent_test")
    def inconsistent(z):
        return dict(gd=1.0, dopt=(0.5, 0.5), fopt=1.0,
                    eff=lambda a, b: 0.0, tox=lambda a, b: a + b, sf=0.2, sg=0.2)

    with pytest.raises(ValueError, match="maximisation convention"):
        sdb.run_trial("cEI", seed=0, z=0, gamma=0.7, sim="inconsistent_test", budget=12)


def test_sweep_preflight_rejects_typos(tmp_path):
    with pytest.raises(KeyError):
        sdb.sweep(["cEI", "cKGG_typo"], surfaces=["osa"], seeds=1, gammas=[0.7],
                 strata=[0], parallel=False, budget=12, verbose=False)
    with pytest.raises(ValueError, match="empty_gate"):
        sdb.sweep(["cEI"], surfaces=["osa"], seeds=1, gammas=[0.7], strata=[0],
                  parallel=False, budget=12, empty_gate="typo", verbose=False)
    for kwargs, message in [({"mode": "typo"}, "mode"), ({"noise": "typo"}, "noise"),
                            ({"kap": 0.0}, "kap"),
                            ({"gamma": 1.0}, "gamma"), ({"grid_n": 1}, "grid_n")]:
        args = dict(policy="cEI", seed=0, z=0, gamma=0.7, sim="osa", budget=12)
        args.update(kwargs)
        with pytest.raises(ValueError, match=message):
            sdb.run_trial(**args)

    checkpoint = tmp_path / "strict_resume.json"
    sdb.sweep(["cEI"], surfaces=["osa"], seeds=1, gammas=[0.7], strata=[0],
              parallel=False, budget=12, grid_n=5, out=str(checkpoint), verbose=False)
    with pytest.raises(ValueError, match="design does not match"):
        sdb.sweep(["cEI"], surfaces=["osa"], seeds=1, gammas=[0.7], strata=[0],
                  parallel=False, budget=12, grid_n=3, out=str(checkpoint), verbose=False)
    with pytest.raises(ValueError, match="design does not match"):
        sdb.sweep(["cEI"], surfaces=["osa"], seeds=1, gammas=[0.7], strata=[0],
                  parallel=False, budget=12, grid_n=5, kap=2.0,
                  out=str(checkpoint), verbose=False)


def test_kap_scales_observations_and_fixed_likelihood_together():
    def register(name, sf, sg):
        @sdb.surface(name)
        def noise_surface(z):
            eff = lambda a, b: -((a - 0.5) ** 2 + (b - 0.5) ** 2)
            tox = lambda a, b: a + b
            return dict(gd=1.0, dopt=(0.5, 0.5), fopt=0.0,
                        eff=eff, tox=tox, sf=sf, sg=sg)
        return noise_surface

    register("kap_base_test", 0.1, 0.2)
    register("kap_scaled_test", 0.2, 0.4)
    scaled_by_input = sdb.run_trial(
        "cEI", seed=7, z=0, gamma=0.7, sim="kap_base_test", budget=12, kap=2.0,
    )
    scaled_in_surface = sdb.run_trial(
        "cEI", seed=7, z=0, gamma=0.7, sim="kap_scaled_test", budget=12, kap=1.0,
    )
    assert scaled_by_input["kap"] == 2.0
    assert scaled_by_input["eff_noise_sd"] == pytest.approx(0.2)
    assert scaled_by_input["tox_noise_sd"] == pytest.approx(0.4)
    for key in ("dose_units", "rpsel", "rec_unsafe", "toxic", "n_gate_pass",
                "n_gate_pass_safe", "rec_d1", "rec_d2", "rec_true_eff",
                "rec_true_tox", "obd_pf", "obd_sdg"):
        assert scaled_by_input[key] == pytest.approx(scaled_in_surface[key])


@pytest.mark.ray
def test_ray_workers_see_custom_acquisition():
    import ray  # Required here: missing Ray is an import error, not a skip.

    @sdb.acquisition("worker_visibility_test")
    def wv(ctx):
        mu_e, _ = ctx.latent_efficacy()
        return mu_e

    # parallel=True must not KeyError on the user acquisition inside a fresh worker
    res = sdb.sweep(["worker_visibility_test"], surfaces=["osa"], seeds=2, gammas=[0.7],
                   strata=[0], parallel=True, budget=12, verbose=False)
    assert len(res) == 2
    assert all(np.isfinite(r["dose_units"]) for r in res)


@pytest.mark.ray
def test_ray_workers_see_custom_surface():
    import ray  # Required here: missing Ray is an import error, not a skip.

    @sdb.surface("worker_surface_test")
    def ws(z):
        eff = lambda a, b: -((a - 0.4) ** 2 + (b - 0.6) ** 2)
        tox = lambda a, b: a + b
        return dict(gd=1.0, dopt=(0.4, 0.6), fopt=0.0, eff=eff, tox=tox, sf=0.2, sg=0.2)

    res = sdb.sweep(["cEI"], surfaces=["worker_surface_test"], seeds=2, gammas=[0.7],
                   strata=[0], parallel=True, budget=12, verbose=False)
    assert len(res) == 2
    # must NOT be the silent-gbump fallback: results are labeled with the real surface
    assert all(r["sim"] == "worker_surface_test" for r in res)
