"""Plug-in registries for acquisitions and surfaces.

An **acquisition** is any callable ``fn(ctx) -> np.ndarray`` that scores every dose
on the grid (higher is better; use ``-inf`` to exclude a dose). The trial harness
handles the posterior-feasibility gate, the argmax, cohort assignment, and every operating
characteristic. So to benchmark a new acquisition against the paper's baselines you
write one function and decorate it::

    from dose_combination_bo import acquisition

    @acquisition("my_rule")
    def my_rule(ctx):
        mu_e, _ = ctx.latent_efficacy()          # posterior efficacy mean over the grid
        return ctx.restrict(mu_e + 2.0 * ctx.sd_g)   # gate + a boundary bonus

    # then: dose_combination_bo.sweep(acquisitions=["cEI", "cKG", "my_rule"], surfaces=["osa"])

A **surface** is a callable ``fn(z) -> dict`` returning the data-generating spec for
stratum ``z`` (keys: ``gd, dopt, fopt, eff, tox, sf, sg``); see ``dose_combination_bo.surfaces``.
"""
from __future__ import annotations
import warnings
from typing import Callable, Dict

ACQUISITIONS: Dict[str, Callable] = {}
SURFACES: Dict[str, Callable] = {}


def acquisition(name: str, *, aliases: tuple[str, ...] = ()):
    """Register an acquisition under ``name`` (and optional ``aliases``)."""
    def deco(fn: Callable) -> Callable:
        _register(ACQUISITIONS, name, fn, aliases, "acquisition")
        fn.acquisition_name = name  # type: ignore[attr-defined]
        return fn
    return deco


def surface(name: str, *, aliases: tuple[str, ...] = ()):
    """Register a surface factory under ``name`` (and optional ``aliases``)."""
    def deco(fn: Callable) -> Callable:
        _register(SURFACES, name, fn, aliases, "surface")
        fn.surface_name = name  # type: ignore[attr-defined]
        return fn
    return deco


def _register(reg, name, fn, aliases, kind):
    for key in (name, *aliases):
        if key in reg and reg[key] is not fn:
            warnings.warn(f"{kind} {key!r} re-registered (overwriting the previous one)", stacklevel=3)
        reg[key] = fn


def get_acquisition(name: str) -> Callable:
    try:
        return ACQUISITIONS[name]
    except KeyError:
        raise KeyError(
            f"unknown acquisition {name!r}; registered: {sorted(set(ACQUISITIONS))}"
        ) from None


def get_surface(name: str) -> Callable:
    try:
        return SURFACES[name]
    except KeyError:
        raise KeyError(
            f"unknown surface {name!r}; registered: {sorted(set(SURFACES))}"
        ) from None


def list_acquisitions() -> list[str]:
    """Canonical acquisition names (deduplicated, aliases collapsed)."""
    return sorted({fn.acquisition_name for fn in ACQUISITIONS.values()})


def list_surfaces() -> list[str]:
    return sorted({fn.surface_name for fn in SURFACES.values()})
