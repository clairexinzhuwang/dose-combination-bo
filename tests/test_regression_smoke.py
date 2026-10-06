# Migrated from study/tests/test_smoke.py; see REGRESSION_MIGRATION.csv.
"""Smoke tests: every built-in acquisition runs on every surface and returns a
well-formed result. Fast (small budget); no dependency on the original paper code."""
import numpy as np
import pytest

import dose_combination_bo as sdb

ACQS = ["cEI", "cKG", "cEI-tMSE", "straddle", "ustrad", "SPW", "entropy", "random", "sKG3"]
SIMS = ["osa", "mariposa", "efftox", "gbump"]

REQUIRED = ["policy", "dose_units", "rpsel", "rec_unsafe", "toxic",
            "n_gate_pass", "n_gate_pass_safe", "recs", "obd_pf", "kap",
            "eff_noise_sd", "tox_noise_sd", "protocol_scaffold", "stop_reason",
            "recommendation_made", "n_enrolled", "n_unique_doses",
            "n_empty_gate_events", "initialization_patients_above",
            "post_initialization_patients_above", "initialization_size",
            "region_q_at_stop", "allocation_history"]


@pytest.mark.parametrize("acq", ACQS)
def test_acquisition_runs_on_osa(acq):
    r = sdb.run_trial(acq, seed=0, z=0, gamma=0.7, sim="osa", budget=12,
                      allow_ungated_diagnostic=(acq == "ustrad"))
    for k in REQUIRED:
        assert k in r
    assert np.isfinite(r["dose_units"]) and r["dose_units"] >= 0
    assert np.isfinite(r["rpsel"])
    assert r["rec_unsafe"] in (0, 1)
    assert len(r["recs"]) == 6


@pytest.mark.parametrize("sim", SIMS)
def test_surface_runs(sim):
    r = sdb.run_trial("cEI", seed=0, z=0, gamma=0.7, sim=sim, budget=12)
    assert np.isfinite(r["dose_units"])
    spec = sdb.get_surface(sim)(0)
    assert np.isclose(spec["fopt"], spec["eff"](*spec["dopt"]), rtol=1e-10, atol=1e-10)
    assert spec["tox"](*spec["dopt"]) <= spec["gd"] + 1e-10


def test_synth_family_runs():
    r = sdb.run_trial("cKG", seed=0, z=0, gamma=0.7, sim="synth:k=1.0;s=0.5", budget=12)
    assert np.isfinite(r["dose_units"])


def test_sweep_and_summarize():
    res = sdb.sweep(["cEI", "cEI-tMSE"], surfaces=["osa"], seeds=3, gammas=[0.7],
                   strata=[0], parallel=False, budget=12, empty_gate="pf", verbose=False)
    assert len(res) == 2 * 3  # 2 acquisitions x 3 seeds
    assert {r["empty_gate"] for r in res} == {"pf"}
    tab = sdb.summarize(res, metric="dose_units")
    # summarize returns a DataFrame (pandas present) or list of dicts
    assert len(tab) == 2
    contrast = sdb.paired_contrast(res, "cEI", "cEI-tMSE", metric="dose_units")
    rows = contrast.to_dict("records") if hasattr(contrast, "to_dict") else contrast
    assert len(rows) == 1 and rows[0]["n"] == 3
    assert np.isfinite(rows[0]["difference_mean"])
    with pytest.raises(ValueError, match="distinct policies"):
        sdb.paired_contrast(res, "cEI", "cEI")
    with pytest.raises(ValueError, match="unpaired policy cells"):
        sdb.paired_contrast(res[:-1], "cEI", "cEI-tMSE")
    malformed = [dict(r) for r in res]
    malformed[0].pop("empty_gate")
    with pytest.raises(ValueError, match="missing fields"):
        sdb.paired_contrast(malformed, "cEI", "cEI-tMSE")

    mixed_kap = res + [dict(r, kap=2.0) for r in res]
    with pytest.raises(ValueError, match="different 'kap'"):
        sdb.summarize(mixed_kap, metric="dose_units")
    with pytest.raises(ValueError, match="different 'kap'"):
        sdb.leaderboard(mixed_kap, metric="dose_units", sim="osa")
    with pytest.raises(ValueError, match="different 'kap'"):
        sdb.paired_contrast(mixed_kap, "cEI", "cEI-tMSE", metric="dose_units")

    by_kap = sdb.summarize(
        mixed_kap, metric="dose_units", by=("sim", "policy", "gamma", "kap")
    )
    assert len(by_kap) == 4
