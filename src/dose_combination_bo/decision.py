"""Validated decision primitives for research simulations.

A posterior probability criterion for a latent *mean* is not an individual adverse
-event risk limit. A fallback index is explicitly labelled and is not a qualified
clinical recommendation. These helpers do not replace protocol calibration.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy.stats import norm


def _vector(name: str, value, n: int | None = None) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.ndim != 1 or out.size == 0 or (n is not None and out.size != n):
        raise ValueError(f"{name} must be a nonempty one-dimensional vector" +
                         (f" of length {n}" if n is not None else ""))
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain finite values")
    return out


def _mask(name: str, value, n: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (n,) or raw.dtype.kind != 'b':
        raise ValueError(f"{name} must be a Boolean vector of length {n}")
    return raw


@dataclass(frozen=True)
class Selection:
    index: int | None
    status: Literal['qualified_selection', 'fallback_selection', 'no_qualified_selection']
    n_passing: int
    standardized_feasibility: float | None
    posterior_probability: float | None


def select_from_posterior(mu_eff, mu_tox, var_tox, g_dagger: float,
                          gamma: float = 0.5, *, observation_variance: float = 0.0,
                          on_empty: Literal['fallback', 'none'] = 'fallback') -> Selection:
    """Select within the supplied terminal candidate domain, in its input order.

    Ties use the first candidate, as in the archived study. ``on_empty='none'``
    is an optional terminal readout, NOT the terminal value used by the evaluated
    cKG policy. Using that readout as an adaptive policy requires changing its
    lookahead objective and running a new simulation study.
    """
    eff = _vector('mu_eff', mu_eff)
    tox = _vector('mu_tox', mu_tox, eff.size)
    var = _vector('var_tox', var_tox, eff.size)
    if np.any(var < 0):
        raise ValueError('var_tox must be nonnegative')
    if not np.isscalar(g_dagger) or not np.isfinite(float(g_dagger)):
        raise ValueError('g_dagger must be a finite scalar')
    if not np.isscalar(gamma) or not np.isfinite(float(gamma)) or not 0 < float(gamma) < 1:
        raise ValueError('gamma must be finite and strictly between zero and one')
    if (not np.isscalar(observation_variance) or
        not np.isfinite(float(observation_variance)) or float(observation_variance) < 0):
        raise ValueError('observation_variance must be finite and nonnegative')
    if on_empty not in ('fallback', 'none'):
        raise ValueError("on_empty must be 'fallback' or 'none'")
    sd = np.sqrt(np.maximum(var + float(observation_variance), 1e-12))
    standardized = (float(g_dagger) - tox) / sd
    if not np.isfinite(standardized).all():
        raise FloatingPointError('standardized feasibility is nonfinite')
    passed = standardized > norm.ppf(float(gamma))
    n_pass = int(passed.sum())
    if n_pass:
        index = int(np.argmax(np.where(passed, eff, -np.inf)))
        status = 'qualified_selection'
    elif on_empty == 'none':
        return Selection(None, 'no_qualified_selection', 0, None, None)
    else:
        index = int(np.argmax(standardized))
        status = 'fallback_selection'
    return Selection(index, status, n_pass, float(standardized[index]),
                     float(norm.cdf(standardized[index])))


@dataclass(frozen=True)
class Allocation:
    index: int
    passed_criterion: bool
    used_fallback: bool
    diagnostic_ungated: bool


def choose_allocation(scores, admitted_mask, eligible_mask, standardized_feasibility,
                      *, empty_gate: Literal['pf', 'ungated'] = 'pf',
                      diagnostic_ungated: bool = False) -> Allocation:
    """Enforce the assignment mask independently of an acquisition plug-in.

    NaN, +inf and wrong-length scores are errors, not empty-gate events. When at
    least one eligible candidate passes the criterion, all -inf scores are also
    an error. Legitimate negative cKG scores remain signed and rank normally.
    The 'ungated' controls are explicit historical/diagnostic modes only.
    """
    z = _vector('standardized_feasibility', standardized_feasibility)
    admitted = _mask('admitted_mask', admitted_mask, z.size)
    eligible = _mask('eligible_mask', eligible_mask, z.size)
    if not eligible.any():
        raise ValueError('at least one administration must be eligible')
    vals = np.asarray(scores, dtype=float)
    if vals.shape != z.shape or np.isnan(vals).any() or np.isposinf(vals).any():
        raise ValueError('scores must have one entry per candidate, without NaN or +inf')
    if empty_gate not in ('pf', 'ungated'):
        raise ValueError("empty_gate must be 'pf' or 'ungated'")
    if not isinstance(diagnostic_ungated, (bool, np.bool_)):
        raise ValueError('diagnostic_ungated must be Boolean')
    passing = admitted & eligible
    if diagnostic_ungated:
        rankable = np.where(eligible, vals, -np.inf)
    elif passing.any():
        rankable = np.where(passing, vals, -np.inf)
    elif empty_gate == 'ungated':
        rankable = np.where(eligible, vals, -np.inf)
    else:
        idx = int(np.argmax(np.where(eligible, z, -np.inf)))
        return Allocation(idx, False, True, False)
    if not np.isfinite(rankable).any():
        raise ValueError('acquisition has no finite score on its permitted set; '
                         'this is a numerical/plug-in failure, not a toxicity stopping event')
    idx = int(np.argmax(rankable))
    return Allocation(idx, bool(admitted[idx]), False, bool(diagnostic_ungated))
