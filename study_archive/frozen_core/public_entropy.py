"""Release-time public names for finite-panel feasibility-entropy reduction.

The authenticated execution source registered this unchanged scorer as ``qBIG``
and also carried the historically misleading alias ``SUR``. This binding layer
does not rewrite those hash-frozen bytes. After package initialization it:

* exposes ``entropy`` as the canonical reader-facing acquisition name;
* exposes ``feasibility-entropy`` as a descriptive public alias;
* keeps ``qBIG`` as a provenance-compatible alias for archived scripts/records;
* keeps ``SUR`` for one compatibility cycle, but emits ``DeprecationWarning``
  because the scorer is not a standard named SUR acquisition; and
* installs accurate runtime documentation on the frozen implementation module.

All four accepted strings resolve to the same numerical scorer. Trial records
continue to retain the exact policy string supplied by the caller, so historical
``qBIG`` identifiers and scientific outputs are not rewritten.
"""
from __future__ import annotations

import importlib
import warnings

_frozen_acquisitions = importlib.import_module(".acquisitions", __package__)
_registry = importlib.import_module(".registry", __package__)
_sweep = importlib.import_module(".sweep", __package__)
_trial = importlib.import_module(".trial", __package__)
from .registry import ACQUISITIONS


_entropy = ACQUISITIONS.get("qBIG")
if _entropy is None:  # pragma: no cover - import-order contract
    raise RuntimeError("qBIG must be registered before public entropy bindings")

# Canonicalize discovery without changing the scorer or caller-supplied record ID.
_entropy.acquisition_name = "entropy"  # type: ignore[attr-defined]
ACQUISITIONS["entropy"] = _entropy
ACQUISITIONS["feasibility-entropy"] = _entropy
ACQUISITIONS["qBIG"] = _entropy
ACQUISITIONS["SUR"] = _entropy


_original_get_acquisition = getattr(
    _registry.get_acquisition,
    "_safedosebo_unwrapped",
    _registry.get_acquisition,
)


def get_acquisition(name: str):
    """Resolve a public acquisition name, warning on the legacy ``SUR`` alias."""
    if name == "SUR":
        warnings.warn(
            "'SUR' is a deprecated compatibility alias for the study-specific "
            "finite-panel feasibility-entropy criterion; use 'entropy' (or "
            "'feasibility-entropy'). The implementation is not a standard named "
            "SUR acquisition.",
            DeprecationWarning,
            stacklevel=2,
        )
    return _original_get_acquisition(name)


get_acquisition._safedosebo_unwrapped = _original_get_acquisition  # type: ignore[attr-defined]

# ``trial`` and ``sweep`` imported the frozen resolver before this release layer
# was installed. Rebind their module-local references so every supported public
# dispatch path provides the same deprecation behavior. Numerical dispatch for
# every non-SUR name is unchanged.
_registry.get_acquisition = get_acquisition
_trial.get_acquisition = get_acquisition
_sweep.get_acquisition = get_acquisition


_frozen_acquisitions.__doc__ = """Authenticated acquisition implementations.

This module's source bytes are frozen execution provenance, so its historical
source-level catalog is not rewritten. The effective release API is installed
after import by :mod:`safedosebo.public_ckg` and
:mod:`safedosebo.public_entropy`:

* ``cKG`` / ``cKG-exact``: deterministic piecewise-analytic hard-gated cKG;
* ``cKG-legacy512`` / ``cKG1fix``: historical fixed-512-fantasy approximation;
* ``tmse``: localized gated targeted-MSE boundary score;
* ``entropy`` / ``feasibility-entropy``: finite-panel marginal feasibility-
  entropy reduction, with ``qBIG`` retained for provenance; and
* ``cEI``: myopic constrained expected improvement.

The author-specified ``cEI-tMSE`` hybrid remains available as a secondary
sensitivity policy. ``SUR`` is a deprecated compatibility alias for ``entropy``
and does not identify a standard named SUR algorithm.
"""

_frozen_acquisitions.ckg_one_step_gated.__doc__ = """Historical fixed-fantasy
Monte Carlo evaluator of the hard-gated one-step cKG score.

The public ``cKG`` and ``cKG-exact`` acquisitions use the deterministic
piecewise-analytic evaluator. This function remains the implementation behind
``cKG-legacy512`` and its compatibility alias ``cKG1fix``.
"""

_frozen_acquisitions.qbig_value.__doc__ = """Approximate the expected reduction
in the sum of marginal finite-panel feasibility entropies.

The scalar predictive expectation uses fixed nine-node Gauss--Hermite
quadrature. The public acquisition names are ``entropy`` and
``feasibility-entropy``; ``qBIG`` is retained as a provenance alias. This is not
exact integration or a standard named SUR acquisition.
"""


__all__ = ["get_acquisition"]
