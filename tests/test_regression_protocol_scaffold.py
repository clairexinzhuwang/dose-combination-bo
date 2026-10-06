# Migrated from study/tests/test_protocol_scaffold.py; see REGRESSION_MIGRATION.csv.
"""Protocol-scaffold contracts that do not depend on fitted GP thresholds.

Geometry and repeat exclusion are tested through the pure protocol helpers.  The
stopping tests replace only the fitted-model boundary with a scripted Context, so
they exercise the real trial loop, allocation timing, and returned record without
paying for (or depending on) Gaussian-process fitting.
"""
import json

import numpy as np
import pytest
import torch

import dose_combination_bo as sdb
from dose_combination_bo.version import implementation_fingerprint
from dose_combination_bo.protocol import eligible_mask, initialization_size, region_mask
import dose_combination_bo.trial as trial_module


def _run_scripted_start_low(monkeypatch, gate_empty_sequence):
    """Run the real start-low loop against a deterministic sequence of gate states."""
    events = iter(gate_empty_sequence)

    class ScriptedContext:
        def __init__(self, *, Xset, candidate_mask=None, **kwargs):
            del kwargs
            self.Xset = Xset
            if candidate_mask is None:
                candidate_mask = np.ones(Xset.shape[0], dtype=bool)
            self.candidate_mask = np.asarray(candidate_mask, dtype=bool)
            self.gate_empty = bool(next(events))
            self.gate_safe = np.full(len(Xset), not self.gate_empty, dtype=bool)
            self.standardized_feasibility = np.zeros(len(Xset))
            eligible = np.flatnonzero(self.candidate_mask)
            assert len(eligible), "the protocol must always expose an eligible fallback"
            self.most_feasible_index = int(eligible[0])

    def no_gp_fit(*args, **kwargs):
        del args, kwargs
        return object(), object()

    def flat_posterior(model, points, *args, **kwargs):
        del model, args, kwargs
        n = int(points.shape[0])
        return torch.zeros(n, dtype=torch.double), torch.ones(n, dtype=torch.double)

    def no_terminal_recommendation(*args, **kwargs):
        del args, kwargs
        raise AssertionError("a stopped trial must not construct a terminal recommendation")

    monkeypatch.setattr(trial_module, "Context", ScriptedContext)
    monkeypatch.setattr(trial_module, "fit_gp", no_gp_fit)
    monkeypatch.setattr(trial_module, "post_latent", flat_posterior)
    monkeypatch.setattr(trial_module, "post_predictive", flat_posterior)
    monkeypatch.setattr(trial_module, "recommend_obd", no_terminal_recommendation)
    monkeypatch.setattr(
        trial_module,
        "get_acquisition",
        lambda policy: (lambda ctx: np.zeros(ctx.Xset.shape[0], dtype=float)),
    )

    return trial_module.run_trial(
        "scripted-test-policy",
        seed=0,
        z=0,
        gamma=0.7,
        sim="osa",
        budget=40,
        r_k=2,
        protocol_scaffold="start_low_expansion",
        region_step=0.25,
        empty_gate_stop_after=3,
        exclude_repeats_during_expansion=True,
    )


def _paired_records():
    common = {
        "seed": 0,
        "sim": "osa",
        "stratum": 0,
        "gamma": 0.7,
        "mode": "latent",
        "budget": 40,
        "r_k": 2,
        "grid_n": 5,
        "noise": "fixed",
        "kap": 1.0,
        "warmup": 4,
        "empty_gate": "pf",
        "protocol_scaffold": "lhs_fixed",
        "region_step": 0.25,
        "empty_gate_stop_after": 3,
        "exclude_repeats_during_expansion": True,
        "dose_units": 1.0,
    }
    return [dict(common, policy="cKG"), dict(common, policy="cEI-tMSE")]


def _checkpoint_record():
    return {
        "policy": "cEI",
        "software_version": sdb.__version__,
        "core_source_sha256": implementation_fingerprint(),
        "allow_ungated_diagnostic": False,
        "seed": 0,
        "sim": "osa",
        "stratum": 0,
        "gamma": 0.7,
        "mode": "latent",
        "budget": 12,
        "r_k": 2,
        "grid_n": 5,
        "noise": "fixed",
        "kap": 1.0,
        "warmup": 4,
        "empty_gate": "pf",
        "protocol_scaffold": "lhs_fixed",
        "region_step": 0.25,
        "empty_gate_stop_after": 3,
        "exclude_repeats_during_expansion": True,
    }


def test_start_low_initialization_is_two_patients_at_origin(monkeypatch):
    assert initialization_size("start_low_expansion", warmup=4, r_k=2) == 2
    result = _run_scripted_start_low(monkeypatch, [True, True, True])
    assert result["initialization_size"] == 2
    assert result["allocation_history"][:2] == [[0.0, 0.0], [0.0, 0.0]]


@pytest.mark.parametrize("q", range(1, 8))
def test_start_low_region_never_admits_dose_above_expansion_boundary(q):
    points = sdb.grid(5)
    allowed = region_mask(points, q=q, region_step=0.25)
    assert np.all(points[allowed].sum(axis=1) <= 0.25 * q + 1e-12)
    assert not allowed.all()
    assert region_mask(points, q=8, region_step=0.25).all()


def test_start_low_excludes_repeats_until_partial_region_is_exhausted():
    points = sdb.grid(5)
    q = 2
    region = region_mask(points, q=q, region_step=0.25)
    region_points = points[region]
    history = np.repeat(region_points[:1], repeats=2, axis=0)

    eligible = eligible_mask(points, history, q=q, region_step=0.25)
    assert not eligible[np.flatnonzero(region)[0]]
    assert eligible.sum() == region.sum() - 1

    exhausted_history = np.repeat(region_points, repeats=2, axis=0)
    eligible_after_exhaustion = eligible_mask(
        points, exhausted_history, q=q, region_step=0.25
    )
    np.testing.assert_array_equal(eligible_after_exhaustion, region)

    # Once q=8 opens the full panel, expansion-stage repeat exclusion is disabled.
    assert eligible_mask(points, history, q=8, region_step=0.25).all()


def test_third_consecutive_empty_gate_stops_before_next_cohort(monkeypatch):
    result = _run_scripted_start_low(monkeypatch, [True, True, True])
    assert result["stop_reason"] == "NO_FEASIBLE_DOSE"
    assert result["n_empty_gate_events"] == 3
    assert result["region_q_at_stop"] == 4
    # Initial cohort plus only the first two empty-gate fallback cohorts.
    assert result["n_enrolled"] == 6
    assert len(result["allocation_history"]) == 6


def test_nonempty_gate_resets_consecutive_empty_counter(monkeypatch):
    result = _run_scripted_start_low(
        monkeypatch, [True, True, False, True, True, True]
    )
    assert result["stop_reason"] == "NO_FEASIBLE_DOSE"
    assert result["n_empty_gate_events"] == 5
    assert result["region_q_at_stop"] == 7
    # Five cohorts were allocated after initialization; the sixth check stopped.
    assert result["n_enrolled"] == 12


def test_stopped_trial_has_no_fictitious_recommendation(monkeypatch):
    result = _run_scripted_start_low(monkeypatch, [True, True, True])
    assert result["recommendation_made"] is False
    assert result["stop_reason"] == "NO_FEASIBLE_DOSE"
    for field in (
        "dose_units",
        "rpsel",
        "rec_unsafe",
        "rec_d1",
        "rec_d2",
        "rec_true_eff",
        "rec_true_tox",
    ):
        assert result[field] is None
    assert result["recs"] == []


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("protocol_scaffold", "start_low_expansion"),
        ("region_step", 0.5),
        ("empty_gate_stop_after", 4),
        ("exclude_repeats_during_expansion", False),
    ],
)
def test_paired_contrast_rejects_protocol_config_mismatch(field, different_value):
    rows = _paired_records()
    rows[1][field] = different_value
    with pytest.raises(ValueError, match=field):
        sdb.paired_contrast(rows, "cKG", "cEI-tMSE")


@pytest.mark.parametrize(
    "requested_config",
    [
        {"protocol_scaffold": "start_low_expansion"},
        {"region_step": 0.5},
        {"empty_gate_stop_after": 4},
        {"exclude_repeats_during_expansion": False},
    ],
)
def test_sweep_checkpoint_key_rejects_protocol_config_mismatch(
    tmp_path, requested_config
):
    checkpoint = tmp_path / "protocol-checkpoint.json"
    checkpoint.write_text(json.dumps([_checkpoint_record()]))
    kwargs = dict(
        acquisitions=["cEI"],
        surfaces=["osa"],
        seeds=1,
        gammas=[0.7],
        strata=[0],
        parallel=False,
        budget=12,
        out=str(checkpoint),
        verbose=False,
    )
    kwargs.update(requested_config)
    with pytest.raises(ValueError, match="design does not match"):
        sdb.sweep(**kwargs)
