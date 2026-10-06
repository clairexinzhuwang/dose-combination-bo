"""Release-time public registry bindings for exact and legacy cKG.

The formal execution sources are hash-frozen, including the module that originally
registered ``cKG`` as the 512-fantasy approximation.  This release-only bridge keeps
those historical bytes unchanged while making the manuscript evaluator the public
default:

* ``cKG`` and ``cKG-exact`` resolve to the deterministic piecewise-analytic scorer;
* ``cKG-legacy512`` resolves to the former public implementation; and
* ``cKG1fix`` remains an alias of ``cKG-legacy512`` for old scripts and records.

The frozen formal runners still use their private process-local policy names, but
their scorer and the public exact scorer both call :func:`ckg_scores_exact`.
"""
from __future__ import annotations

from . import ckg_exact as _ckg_exact_module
from .ckg_exact import ckg_scores_exact
from .registry import ACQUISITIONS


_legacy512 = ACQUISITIONS.get("cKG-legacy512") or ACQUISITIONS.get("cKG1fix")
if _legacy512 is None:  # pragma: no cover - import-order contract
    raise RuntimeError("legacy cKG must be registered before public cKG bindings")

# Preserve the old callable itself under an explicit canonical identity.  Direct
# assignment matches the process-local registration pattern used by the frozen
# formal runner and avoids a misleading re-registration warning at package import.
_legacy512.acquisition_name = "cKG-legacy512"  # type: ignore[attr-defined]
ACQUISITIONS["cKG-legacy512"] = _legacy512
ACQUISITIONS["cKG1fix"] = _legacy512


def _exact_ckg(ctx):
    """Score all public cKG queries with the manuscript's exact evaluator."""
    return ckg_scores_exact(ctx)


_exact_ckg.acquisition_name = "cKG"  # type: ignore[attr-defined]
ACQUISITIONS["cKG"] = _exact_ckg
ACQUISITIONS["cKG-exact"] = _exact_ckg


# The exact implementation source is part of the authenticated execution chain,
# so correct its public help text at the release boundary without changing those
# pinned bytes or their recorded hashes.
_ckg_exact_module.__doc__ = """Deterministic piecewise-analytic evaluation of
the finite-grid hard-gated one-step cKG.

After normal package initialization, public ``cKG`` and ``cKG-exact`` both use
``ckg_scores_exact``. The historical fixed-512-fantasy approximation remains
available as ``cKG-legacy512`` and compatibility alias ``cKG1fix``. The source
bytes of this module remain unchanged because they authenticate the completed
simulation record; this runtime documentation states the effective release API.
"""

ckg_scores_exact.__doc__ = """Score every accessible query with the deterministic
piecewise-analytic evaluator and apply the context assignment mask.

This scorer backs the public ``cKG`` and ``cKG-exact`` acquisitions. The legacy
``cKG-legacy512`` / ``cKG1fix`` path uses the frozen Monte Carlo approximation
instead.
"""


__all__: list[str] = []
