"""Public decorators and discovery names.

Scores returned by a custom acquisition are validated and restricted to
combinations allowed by the model-based toxicity and dose-availability rules.
The ``empty_gate="ungated"`` setting remains only for backward compatibility.
"""
from __future__ import annotations

from functools import wraps

import numpy as np

from . import registry as _registry
from .registry import ACQUISITIONS, SURFACES
from .registry import acquisition as _register_acquisition


def acquisition(name: str, *, aliases: tuple[str, ...] = ()):
    """Register a custom acquisition under the common assignment rules.

    The decorated function returns one raw score per currently available
    candidate combination. Higher is better and ``-inf`` may exclude a
    combination. The public wrapper validates the scores and applies
    ``ctx.restrict``; applying it inside the function as well is harmless but
    unnecessary.
    The explicit legacy ``empty_gate="ungated"`` option retains its historical
    exception to the model-based toxicity rule.
    """

    def decorate(fn):
        @wraps(fn)
        def restricted(ctx):
            raw_values = np.asarray(fn(ctx), dtype=float)
            # Lightweight contexts used by downstream acquisition tests may
            # expose only the score inputs and ``restrict``.  In that case the
            # returned score vector supplies the only available candidate count.
            candidate_set = getattr(ctx, "Xset", None)
            expected = (
                int(candidate_set.shape[0])
                if candidate_set is not None
                else int(raw_values.shape[0]) if raw_values.ndim else 0
            )
            if raw_values.shape not in {(expected,), (expected, 1)}:
                raise ValueError(
                    f"acquisition {name!r} scores must have shape ({expected},) or "
                    f"({expected}, 1), got {raw_values.shape}"
                )
            values = raw_values.reshape(-1)
            if np.isnan(values).any() or np.isposinf(values).any():
                raise ValueError("acquisition scores must not contain NaN or +inf")
            return ctx.restrict(values)

        restricted.raw_acquisition = fn  # type: ignore[attr-defined]
        return _register_acquisition(name, aliases=aliases)(restricted)

    return decorate


def install_reader_facing_surface_names() -> None:
    """Keep historical aliases while showing descriptive surface names."""

    for historical, reader_name in (
        ("efftox", "logistic"),
        ("gbump", "gaussian_bump"),
    ):
        surface_fn = SURFACES.get(historical)
        if surface_fn is not None:
            surface_fn.surface_name = reader_name  # type: ignore[attr-defined]


# Importing ``dose_combination_bo.registry`` still executes the package initializer first.
# Rebinding here keeps that extension path under the same assignment rules without
# changing built-ins registered earlier in package initialization.
_registry.acquisition = acquisition


__all__ = ["acquisition", "install_reader_facing_surface_names"]
