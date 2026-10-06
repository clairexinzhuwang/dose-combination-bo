"""Dose-access-protocol helpers for the single-trial harness.

The acquisition functions deliberately know nothing about escalation.  This module
defines the common candidate region and no-repeat rule that the harness applies to
every policy.  Keeping these operations pure also makes the protocol contract easy
to test without fitting Gaussian processes.
"""
from __future__ import annotations

from typing import Literal

import numpy as np


ProtocolScaffold = Literal["lhs_fixed", "start_low_expansion"]
PROTOCOL_SCAFFOLDS = ("lhs_fixed", "start_low_expansion")
PROTOCOL_CONFIG_FIELDS = (
    "protocol_scaffold",
    "region_step",
    "empty_gate_stop_after",
    "exclude_repeats_during_expansion",
)
PROTOCOL_DEFAULTS = {
    "protocol_scaffold": "lhs_fixed",
    "region_step": 0.25,
    "empty_gate_stop_after": 3,
    "exclude_repeats_during_expansion": True,
}


def validate_protocol_config(
    protocol_scaffold: ProtocolScaffold,
    region_step,
    empty_gate_stop_after,
    exclude_repeats_during_expansion,
    *,
    budget,
    warmup,
    r_k,
    grid_n,
    empty_gate,
):
    """Validate access-protocol-specific inputs.

    ``warmup`` remains part of the public API because it configures ``lhs_fixed``.
    The gradual-access option instead has one fixed initial cohort, whose size is
    ``r_k`` (and is required to be two for the prespecified sensitivity).
    """
    if protocol_scaffold not in PROTOCOL_SCAFFOLDS:
        raise ValueError(
            "protocol_scaffold must be 'lhs_fixed' or 'start_low_expansion'"
        )
    if (
        not np.isscalar(region_step)
        or not np.isfinite(float(region_step))
        or float(region_step) <= 0
    ):
        raise ValueError("region_step must be a finite positive scalar")
    if (
        not isinstance(empty_gate_stop_after, (int, np.integer))
        or isinstance(empty_gate_stop_after, (bool, np.bool_))
        or int(empty_gate_stop_after) < 1
    ):
        raise ValueError("empty_gate_stop_after must be an integer >= 1")
    if not isinstance(exclude_repeats_during_expansion, (bool, np.bool_)):
        raise ValueError("exclude_repeats_during_expansion must be boolean")

    if protocol_scaffold == "lhs_fixed":
        if warmup > budget:
            raise ValueError("warmup must not exceed budget")
        if (budget - warmup) % r_k:
            raise ValueError("budget - warmup must be divisible by r_k")
        return

    if r_k != 2:
        raise ValueError("start_low_expansion requires r_k=2")
    if budget < r_k:
        raise ValueError("budget must accommodate the two-patient initial cohort")
    if (budget - r_k) % r_k:
        raise ValueError("budget - r_k must be divisible by r_k")
    if empty_gate != "pf":
        raise ValueError(
            "start_low_expansion requires empty_gate='pf'; the legacy ungated "
            "fallback is not permitted"
        )


def initialization_size(protocol_scaffold: ProtocolScaffold, warmup, r_k):
    """Number of patients assigned before acquisition-guided cohorts begin."""
    if protocol_scaffold == "lhs_fixed":
        return int(warmup)
    if protocol_scaffold == "start_low_expansion":
        return int(r_k)
    raise ValueError("unknown protocol_scaffold")


def cohort_region_q(n_enrolled, r_k):
    """One-based index of the next equal-sized cohort.

    After the fixed start-low cohort (``n_enrolled == r_k``), this returns two.
    Thus the default ``region_step=.25`` opens the full two-agent square at q=8.
    """
    if (
        not isinstance(n_enrolled, (int, np.integer))
        or isinstance(n_enrolled, (bool, np.bool_))
        or int(n_enrolled) < 0
    ):
        raise ValueError("n_enrolled must be a non-negative integer")
    if (
        not isinstance(r_k, (int, np.integer))
        or isinstance(r_k, (bool, np.bool_))
        or int(r_k) < 1
    ):
        raise ValueError("r_k must be a positive integer")
    if int(n_enrolled) % int(r_k):
        raise ValueError("n_enrolled must be divisible by r_k")
    return int(n_enrolled) // int(r_k) + 1


def _points_array(points):
    """Convert numpy/torch-like dose points to a validated float array."""
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be a finite (n, 2) array")
    return points


def region_mask(points, q, region_step=0.25):
    """Mask doses in ``d1 + d2 <= region_step * q``."""
    points = _points_array(points)
    if (
        not isinstance(q, (int, np.integer))
        or isinstance(q, (bool, np.bool_))
        or int(q) < 1
    ):
        raise ValueError("q must be an integer >= 1")
    if (
        not np.isscalar(region_step)
        or not np.isfinite(float(region_step))
        or float(region_step) <= 0
    ):
        raise ValueError("region_step must be a finite positive scalar")
    boundary = float(region_step) * int(q)
    return np.asarray(points.sum(axis=1) <= boundary + 1e-12, dtype=bool)


def unvisited_mask(points, allocation_history):
    """Return a mask of grid candidates never assigned in prior cohorts."""
    points = _points_array(points)
    history = np.asarray(allocation_history, dtype=float)
    if history.size == 0:
        return np.ones(points.shape[0], dtype=bool)
    if history.ndim != 2 or history.shape[1] != 2 or not np.isfinite(history).all():
        raise ValueError("allocation_history must be a finite (n, 2) array")
    visited = np.any(
        np.all(np.isclose(points[:, None, :], history[None, :, :], rtol=0, atol=1e-12), axis=2),
        axis=1,
    )
    return ~visited


def eligible_mask(
    points,
    allocation_history,
    q,
    region_step=0.25,
    exclude_repeats_during_expansion=True,
):
    """Common administration mask for a start-low acquisition step.

    Before the region is the full panel, previously visited candidates are excluded
    whenever the current region contains at least one unvisited candidate.  Once the
    full panel is open, or once every candidate in a partial region has been visited,
    repeats are allowed.
    """
    points = _points_array(points)
    region = region_mask(points, q, region_step)
    if not isinstance(exclude_repeats_during_expansion, (bool, np.bool_)):
        raise ValueError("exclude_repeats_during_expansion must be boolean")
    if not exclude_repeats_during_expansion or region.all():
        return region
    unvisited = unvisited_mask(points, allocation_history)
    if np.any(region & unvisited):
        return region & unvisited
    return region


__all__ = [
    "ProtocolScaffold",
    "PROTOCOL_SCAFFOLDS",
    "PROTOCOL_CONFIG_FIELDS",
    "PROTOCOL_DEFAULTS",
    "validate_protocol_config",
    "initialization_size",
    "cohort_region_q",
    "region_mask",
    "unvisited_mask",
    "eligible_mask",
]
