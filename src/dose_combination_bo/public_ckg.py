"""Public names for the current and legacy cKG implementations.

Earlier releases registered ``cKG`` as the 512-draw Monte Carlo approximation.
The current public API uses the deterministic finite-set evaluator:

* ``cKG`` is the public name for the deterministic finite-set scorer;
* ``cKG-exact`` remains a compatibility alias for that scorer;
* ``cKG-legacy512`` provides the earlier approximation; and
* ``cKG1fix`` remains its compatibility alias for older scripts and records.

The implemented cKG score is the expected one-cohort change in the value used
by the final-selection rule and can be negative. A combination meets the model-based
toxicity rule when the posterior probability that its mean toxicity does not
exceed the limit is greater than ``tau``; larger ``tau`` values make this rule
more stringent. This is not a clinically calibrated safety guarantee. This
implementation is distinct from previously published constrained-KG formulations.
"""
from __future__ import annotations

from . import ckg_exact as _ckg_exact_module
from .ckg_exact import ckg_scores_exact
from .registry import ACQUISITIONS


_legacy512 = ACQUISITIONS.get("cKG-legacy512") or ACQUISITIONS.get("cKG1fix")
if _legacy512 is None:  # pragma: no cover - import-order contract
    raise RuntimeError("legacy cKG must be registered before public cKG bindings")

# Preserve the earlier callable under an explicit compatibility name. Direct
# assignment avoids a misleading re-registration warning during package import.
_legacy512.acquisition_name = "cKG-legacy512"  # type: ignore[attr-defined]
ACQUISITIONS["cKG-legacy512"] = _legacy512
ACQUISITIONS["cKG1fix"] = _legacy512


def _exact_ckg(ctx):
    """Score candidate combinations with the deterministic cKG evaluator."""
    return ckg_scores_exact(ctx)


_exact_ckg.acquisition_name = "cKG"  # type: ignore[attr-defined]
ACQUISITIONS["cKG"] = _exact_ckg
ACQUISITIONS["cKG-exact"] = _exact_ckg


# Update the imported implementation's runtime documentation without modifying
# its source file.
_ckg_exact_module.__doc__ = """Deterministic finite-set evaluation of cKG.

The score is the expected one-cohort change in final-selection value and
can be negative. It applies a model-based toxicity rule: the posterior
probability that mean toxicity does not exceed its limit must be greater than
``tau``. Larger ``tau`` values make the rule more stringent; they are not
clinically calibrated safety probabilities. This implementation is distinct
from previously published constrained-KG formulations. The value includes the
prespecified largest-margin fallback when no combination meets the rule; that
fallback does not meet the model-based toxicity criterion.

The public ``cKG`` and its compatibility alias ``cKG-exact`` both use
``ckg_scores_exact``. The historical fixed-512-draw
Monte Carlo approximation remains available as ``cKG-legacy512`` and
its compatibility alias ``cKG1fix``.
"""

_ckg_exact_module.ckg_one_step_gated_exact.__doc__ = """Compute one candidate
combination's raw cKG score over the currently available dose set.

This low-level function does not apply the common assignment restrictions. Use
``ckg_scores_exact`` to score all currently available combinations and apply the
toxicity and dose-availability restrictions used for participant assignment.
Conditional on the implementation's
``1e-12`` latent-toxicity-variance floor, independent Gaussian response
channels, and fixed GP hyperparameters during the hypothetical update, the
finite-partition calculation is exact up to floating-point arithmetic.
"""

ckg_scores_exact.__doc__ = """Score every currently available combination with
the deterministic finite-set evaluator and apply the common toxicity and
dose-availability restrictions.

This scorer backs public ``cKG`` and its ``cKG-exact`` compatibility alias. The
legacy ``cKG-legacy512`` / ``cKG1fix`` path uses the earlier Monte Carlo
approximation instead.
"""


__all__: list[str] = []
