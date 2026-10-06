"""Public names for an author-specified toxicity-classification entropy rule.

Earlier releases exposed this scorer as ``qBIG`` and ``SUR``. The current API:

* exposes ``entropy`` as the main public acquisition name;
* exposes ``feasibility-entropy`` as a descriptive public alias;
* keeps ``qBIG`` for backward compatibility; and
* deprecates ``SUR`` because the scorer is not a standard SUR acquisition.

All four supported identifiers resolve to the same numerical scorer. Trial records
retain the exact policy string supplied by the caller.
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
    "_dose_combination_bo_unwrapped",
    _registry.get_acquisition,
)


def get_acquisition(name: str):
    """Resolve a public acquisition name, warning on the legacy ``SUR`` alias."""
    if name == "SUR":
        warnings.warn(
            "'SUR' is a deprecated alias for the author-specified finite-grid "
            "toxicity-classification entropy-reduction "
            "acquisition used in this package; use 'entropy' or 'feasibility-entropy'. "
            "This implementation is not a standard SUR acquisition.",
            DeprecationWarning,
            stacklevel=2,
        )
    return _original_get_acquisition(name)


get_acquisition._dose_combination_bo_unwrapped = _original_get_acquisition  # type: ignore[attr-defined]

# Rebind the resolver references imported by ``trial`` and ``sweep`` so every
# public dispatch path provides the same deprecation behavior. Numerical dispatch
# for every non-SUR name is unchanged.
_registry.get_acquisition = get_acquisition
_trial.get_acquisition = get_acquisition
_sweep.get_acquisition = get_acquisition


_frozen_acquisitions.__doc__ = """Built-in acquisition functions.

The public API provides the following names:

* ``cKG`` / ``cKG-exact``: deterministic finite-set cKG;
* ``cKG-legacy512`` / ``cKG1fix``: historical fixed-512-draw Monte Carlo
  approximation;
* ``tmse``: pointwise targeted-MSE boundary score under the common toxicity rule;
* ``entropy`` / ``feasibility-entropy``: author-specified finite-grid marginal
  toxicity-classification entropy reduction, with ``qBIG`` retained for
  backward compatibility; and
* ``cEI``: myopic constrained expected improvement.

The ``cEI-tMSE`` hybrid remains available for sensitivity analyses. ``SUR`` is
a deprecated compatibility alias for ``entropy`` and does not identify a
standard named SUR algorithm.
"""

_frozen_acquisitions.ckg_one_step_gated.__doc__ = """Historical fixed-draw
Monte Carlo evaluator of the signed one-cohort cKG score.

Public ``cKG`` and its ``cKG-exact`` compatibility alias use the deterministic
finite-set evaluator. This function remains the implementation behind
``cKG-legacy512`` and its compatibility alias ``cKG1fix``.
"""

_frozen_acquisitions.qbig_value.__doc__ = """Approximate the expected reduction
in the sum of marginal finite-grid toxicity-classification entropies.

The scalar predictive expectation uses fixed nine-node Gauss–Hermite
quadrature. The public acquisition names are ``entropy`` and
``feasibility-entropy``; ``qBIG`` is retained as a backward-compatible alias.
This is not exact integration or a standard named SUR acquisition.
"""


__all__ = ["get_acquisition"]
