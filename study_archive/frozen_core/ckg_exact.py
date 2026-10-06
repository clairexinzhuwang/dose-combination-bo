"""Deterministic piecewise-exact evaluation of the hard-gated one-step cKG.

This module is deliberately opt-in.  The registered ``cKG`` acquisition continues
to use the frozen Monte Carlo evaluator in :mod:`safedosebo.acquisitions`; importing
or calling the functions here does not change a trial default.

For a fixed query, the imagined toxicity observation changes each grid dose's
standardised feasibility through an affine function of one standard normal variate.
Its gate crossings therefore partition the real line into regimes with a fixed
admitted set. On a nonempty regime, a strict-slope upper-envelope routine integrates
the maximum of the admitted affine efficacy lines over the independent efficacy
fantasy. On
an empty regime, pairwise crossings of the standardised-feasibility lines partition
the most-feasible-dose fallback; integrating the selected efficacy line over the
independent efficacy fantasy leaves its intercept.  Normal interval probabilities
then integrate over the toxicity fantasy.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtr
from scipy.stats import norm

from .gp import joint, post_latent


_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_LOG_FLOAT_MAX = math.log(np.finfo(float).max)
_FLOAT_TINY = np.finfo(float).tiny


def _as_vector(name: str, value: Any, n: int | None = None) -> np.ndarray:
    out = np.asarray(value, dtype=float).reshape(-1)
    if not len(out):
        raise ValueError(f"{name} must be nonempty")
    if n is not None and len(out) != n:
        raise ValueError(f"{name} must have length {n}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain only finite values")
    return out


def _normal_interval_probability(lo: float, hi: float) -> float:
    """Return P(lo < Z < hi) without avoidable upper-tail cancellation."""
    if not lo < hi:
        return 0.0
    if lo >= 0.0:
        return float(ndtr(-lo) - ndtr(-hi))
    if hi <= 0.0:
        return float(ndtr(hi) - ndtr(lo))
    return float(ndtr(hi) - ndtr(lo))


def _log_difference(log_large: float, log_small: float) -> float:
    """Return ``log(exp(log_large)-exp(log_small))`` stably."""
    if log_small == -np.inf:
        return float(log_large)
    if log_small > log_large:
        raise ValueError("log_small must not exceed log_large")
    if log_small == log_large:
        return -np.inf
    return float(log_large + np.log(-np.expm1(log_small - log_large)))


def _log_normal_interval_probability(lo: float, hi: float) -> float:
    """Log P(lo < Z < hi), retaining mass below double underflow."""
    if not lo < hi:
        return -np.inf
    if lo >= 0.0:
        return _log_difference(float(log_ndtr(-lo)), float(log_ndtr(-hi)))
    if hi <= 0.0:
        return _log_difference(float(log_ndtr(hi)), float(log_ndtr(lo)))
    return float(math.log(float(ndtr(hi) - ndtr(lo))))


def _normal_interval_first_moment_log(lo: float, hi: float) -> tuple[float, float]:
    """Return sign and log-absolute first normal moment over an interval."""
    square_limit = math.sqrt(np.finfo(float).max)
    log_pdf_lo = (
        -np.inf if np.isneginf(lo) or abs(lo) > square_limit
        else -0.5 * float(lo) ** 2 - _LOG_SQRT_2PI
    )
    log_pdf_hi = (
        -np.inf if np.isposinf(hi) or abs(hi) > square_limit
        else -0.5 * float(hi) ** 2 - _LOG_SQRT_2PI
    )
    if log_pdf_lo == log_pdf_hi:
        return 0.0, -np.inf
    if log_pdf_lo > log_pdf_hi:
        return 1.0, _log_difference(log_pdf_lo, log_pdf_hi)
    return -1.0, _log_difference(log_pdf_hi, log_pdf_lo)


def _affine_interval_integral(a: float, b: float, lo: float, hi: float) -> float:
    """Integrate ``(a+b*z) phi(z)`` using signed log-domain moments."""
    if np.isneginf(lo) and np.isposinf(hi):
        # Preserve the exact identity E[a+bZ] = a.  Besides avoiding an
        # unnecessary log/exp round trip, this keeps large constant offsets exact.
        return float(a)
    log_probability = _log_normal_interval_probability(lo, hi)
    moment_sign, log_moment = _normal_interval_first_moment_log(lo, hi)

    # Ordinary intervals should use ordinary multiplication: it is both faster
    # and more accurate for large intercepts.  Fall through to signed-log
    # arithmetic only when an interval moment has underflowed or a product would
    # overflow, which is the case that needs tail resuscitation.
    probability = _normal_interval_probability(lo, hi)
    moment = (
        0.0 if moment_sign == 0.0 or log_moment == -np.inf
        else math.copysign(math.exp(log_moment), moment_sign)
    )
    probability_represented = (
        probability >= _FLOAT_TINY
        or (probability == 0.0 and log_probability == -np.inf)
    )
    moment_represented = (
        abs(moment) >= _FLOAT_TINY
        or (moment == 0.0 and log_moment == -np.inf)
    )
    if a != 0.0 and probability_represented and moment_represented:
        direct_a = float(a) * probability
        direct_b = float(b) * moment
        if np.isfinite(direct_a) and np.isfinite(direct_b):
            return float(math.fsum((direct_a, direct_b)))

    log_terms: list[float] = []
    signs: list[float] = []
    if a != 0.0 and log_probability != -np.inf:
        log_terms.append(math.log(abs(a)) + log_probability)
        signs.append(math.copysign(1.0, a))
    if b != 0.0 and moment_sign != 0.0 and log_moment != -np.inf:
        log_terms.append(math.log(abs(b)) + log_moment)
        signs.append(math.copysign(1.0, b) * moment_sign)
    if not log_terms:
        return 0.0
    log_absolute, sign = logsumexp(
        np.asarray(log_terms), b=np.asarray(signs), return_sign=True
    )
    if sign == 0.0:
        return 0.0
    if log_absolute > _LOG_FLOAT_MAX:
        return math.copysign(np.inf, float(sign))
    return float(sign * math.exp(float(log_absolute)))


def _scale_by_log_probability(value: float, log_probability: float) -> float:
    """Multiply by an interval probability without losing a resuscitated tail."""
    if value == 0.0 or log_probability == -np.inf:
        return 0.0
    if log_probability == 0.0:
        return float(value)
    probability = math.exp(log_probability)
    if probability >= _FLOAT_TINY:
        direct = float(value) * probability
        if np.isfinite(direct):
            return direct
    log_absolute = math.log(abs(value)) + log_probability
    if log_absolute > _LOG_FLOAT_MAX:
        return math.copysign(np.inf, value)
    return math.copysign(math.exp(log_absolute), value)


def _affine_crossing(
    left_intercept: float,
    left_slope: float,
    right_intercept: float,
    right_slope: float,
) -> float:
    """Return the crossing of two distinct-slope lines without noisy overflow.

    A crossing beyond the floating-point range is correctly represented by an
    infinity: one line then dominates throughout every finite normal-fantasy
    value.  NumPy otherwise emits a RuntimeWarning for that valid limiting case,
    so silence only the arithmetic warning while retaining the signed infinity.
    """
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        crossing = (
            (np.longdouble(left_intercept) - np.longdouble(right_intercept))
            / (np.longdouble(right_slope) - np.longdouble(left_slope))
        )
    return float(crossing)


def _affine_upper_envelope(a: Any, b: Any):
    """Return ``(source_index, a, b, lo, hi)`` upper-envelope segments."""
    a = _as_vector("affine intercepts", a)
    b = _as_vector("affine slopes", b, len(a))
    order = np.argsort(b, kind="stable")
    sorted_a, sorted_b = a[order], b[order]

    unique_a: list[float] = []
    unique_b: list[float] = []
    unique_index: list[int] = []
    for source_index, intercept, slope in zip(order, sorted_a, sorted_b):
        if unique_b and slope == unique_b[-1]:
            if intercept > unique_a[-1]:
                unique_a[-1] = float(intercept)
                unique_index[-1] = int(source_index)
            continue
        unique_a.append(float(intercept))
        unique_b.append(float(slope))
        unique_index.append(int(source_index))

    envelope_a: list[float] = []
    envelope_b: list[float] = []
    envelope_index: list[int] = []
    lower_bounds: list[float] = []
    for source_index, intercept, slope in zip(unique_index, unique_a, unique_b):
        if not envelope_a:
            envelope_a.append(intercept)
            envelope_b.append(slope)
            envelope_index.append(source_index)
            lower_bounds.append(-np.inf)
            continue
        while envelope_a:
            crossing = _affine_crossing(
                envelope_a[-1], envelope_b[-1], intercept, slope
            )
            if crossing > lower_bounds[-1]:
                break
            envelope_a.pop()
            envelope_b.pop()
            envelope_index.pop()
            lower_bounds.pop()
        lower_bound = -np.inf if not envelope_a else _affine_crossing(
            envelope_a[-1], envelope_b[-1], intercept, slope
        )
        envelope_a.append(intercept)
        envelope_b.append(slope)
        envelope_index.append(source_index)
        lower_bounds.append(lower_bound)

    upper_bounds = [*lower_bounds[1:], np.inf]
    return list(zip(
        envelope_index, envelope_a, envelope_b, lower_bounds, upper_bounds
    ))


def _affine_max_expectation(a: Any, b: Any) -> float:
    """Robustly compute E[max_i(a_i+b_i Z)] for finite affine lines.

    The frozen legacy ``gp.kg_lines`` helper merges merely close slopes via
    ``np.isclose`` and subtracts upper-tail CDFs. The opt-in exact path instead
    merges only exactly equal float slopes and evaluates interval probability and
    first-moment terms in the survival/log domain. The legacy helper remains
    untouched for historical reproduction.
    """
    envelope = _affine_upper_envelope(a, b)
    pieces = [
        _affine_interval_integral(intercept, slope, lo, hi)
        for _source_index, intercept, slope, lo, hi in envelope
    ]
    value = float(math.fsum(pieces))
    if not np.isfinite(value):
        raise FloatingPointError("affine-envelope expectation is not finite")
    return value


def expected_terminal_value_exact(
    mu_f: Any,
    efficacy_update: Any,
    mu_g: Any,
    toxicity_update: Any,
    sd_g_post: Any,
    g_dagger: float,
    gamma: float,
    *,
    return_diagnostics: bool = False,
    include_regimes: bool = True,
):
    """Integrate the imagined hard-gated terminal value exactly, piecewise.

    Parameters use the same one-dimensional fantasy representation as the legacy
    evaluator: ``mu_f + efficacy_update * Z_f`` and
    ``mu_g + toxicity_update * Z_g``, with independent standard-normal fantasies.
    ``sd_g_post`` is the fixed post-fantasy gate standard deviation, including
    observation noise only for a predictive gate.
    """
    mu_f = _as_vector("mu_f", mu_f)
    n = len(mu_f)
    efficacy_update = _as_vector("efficacy_update", efficacy_update, n)
    mu_g = _as_vector("mu_g", mu_g, n)
    toxicity_update = _as_vector("toxicity_update", toxicity_update, n)
    sd_g_post = _as_vector("sd_g_post", sd_g_post, n)
    if (sd_g_post <= 0.0).any():
        raise ValueError("sd_g_post must be strictly positive")
    if not np.isfinite(float(g_dagger)):
        raise ValueError("g_dagger must be finite")
    if not np.isfinite(float(gamma)) or not 0.0 < float(gamma) < 1.0:
        raise ValueError("gamma must be finite and strictly between 0 and 1")

    q_gamma = float(norm.ppf(gamma))
    feasibility_intercept = (float(g_dagger) - mu_g) / sd_g_post
    feasibility_slope = -toxicity_update / sd_g_post

    gate_crossings: list[float] = []
    gate_thresholds = np.full(n, np.nan, dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for index, (intercept, slope) in enumerate(zip(
            feasibility_intercept, feasibility_slope
        )):
            # Equality includes +0.0 and -0.0 and preserves fixed-gate semantics.
            if slope == 0.0:
                continue
            crossing = (q_gamma - intercept) / slope
            gate_thresholds[index] = float(crossing)
            if np.isfinite(crossing):
                gate_crossings.append(float(crossing))
    gate_crossings = sorted(set(gate_crossings))
    gate_bounds = [-np.inf, *gate_crossings, np.inf]
    fallback_envelope = _affine_upper_envelope(
        feasibility_intercept, feasibility_slope
    )

    expected_pieces: list[float] = []
    probability_mass = 0.0
    nonempty_regimes = 0
    empty_regimes = 0
    fallback_segments = 0
    regime_rows: list[dict[str, Any]] = []

    for gate_lo, gate_hi in zip(gate_bounds[:-1], gate_bounds[1:]):
        # Classify from line direction and the interval bounds. This remains exact
        # when two thresholds are adjacent floats and no representable probe lies
        # strictly between them.
        gate = np.empty(n, dtype=bool)
        for index, (intercept, slope) in enumerate(zip(
            feasibility_intercept, feasibility_slope
        )):
            if slope == 0.0:
                gate[index] = intercept > q_gamma
            elif slope > 0.0:
                gate[index] = gate_lo >= gate_thresholds[index]
            else:
                gate[index] = gate_hi <= gate_thresholds[index]
        gate_probability = _normal_interval_probability(gate_lo, gate_hi)
        log_gate_probability = _log_normal_interval_probability(gate_lo, gate_hi)
        probability_mass += gate_probability

        if gate.any():
            conditional_value = _affine_max_expectation(
                mu_f[gate], efficacy_update[gate]
            )
            expected_pieces.append(_scale_by_log_probability(
                conditional_value, log_gate_probability
            ))
            nonempty_regimes += 1
            if return_diagnostics and include_regimes:
                regime_rows.append({
                    "z_g_lo": float(gate_lo),
                    "z_g_hi": float(gate_hi),
                    "probability": gate_probability,
                    "gate_indices": np.flatnonzero(gate).astype(int).tolist(),
                    "fallback_index": None,
                    "conditional_value": float(conditional_value),
                })
            continue

        empty_regimes += 1
        for fallback_index, _a, _b, envelope_lo, envelope_hi in fallback_envelope:
            fallback_lo = max(gate_lo, envelope_lo)
            fallback_hi = min(gate_hi, envelope_hi)
            if not fallback_lo < fallback_hi:
                continue
            segment_probability = _normal_interval_probability(fallback_lo, fallback_hi)
            log_segment_probability = _log_normal_interval_probability(
                fallback_lo, fallback_hi
            )
            # Conditional on Z_g, the selected line is
            # mu_f[j] + efficacy_update[j] * Z_f.  The two legacy fantasy channels
            # are independent and E[Z_f]=0, hence the exact inner integral is mu_f[j].
            conditional_value = float(mu_f[fallback_index])
            expected_pieces.append(_scale_by_log_probability(
                conditional_value, log_segment_probability
            ))
            fallback_segments += 1
            if return_diagnostics and include_regimes:
                regime_rows.append({
                    "z_g_lo": float(fallback_lo),
                    "z_g_hi": float(fallback_hi),
                    "probability": segment_probability,
                    "gate_indices": [],
                    "fallback_index": fallback_index,
                    "conditional_value": conditional_value,
                })

    diagnostics = {
        "gate_threshold_count": len(gate_crossings),
        "gate_regime_count": len(gate_bounds) - 1,
        "nonempty_gate_regime_count": nonempty_regimes,
        "empty_gate_regime_count": empty_regimes,
        "fallback_segment_count": fallback_segments,
        "integrated_segment_count": nonempty_regimes + fallback_segments,
        "probability_mass": float(probability_mass),
        "regime_rows_included": bool(return_diagnostics and include_regimes),
        "regimes": regime_rows,
    }
    if not np.isclose(probability_mass, 1.0, rtol=0.0, atol=2e-14):
        raise FloatingPointError(
            f"piecewise normal probability mass is {probability_mass}, not one"
        )
    value = float(math.fsum(expected_pieces))
    return (value, diagnostics) if return_diagnostics else value


def ckg_one_step_gated_exact(
    d,
    Xset,
    m_eff,
    l_eff,
    m_tox,
    l_tox,
    g_dagger,
    gamma,
    r_k=1,
    gate_mode="latent",
    *,
    return_diagnostics: bool = False,
    include_regimes: bool = True,
):
    """Deterministic evaluator of the same finite-grid hard-gated cKG score.

    The return value is the signed ``E[V(post)] - V(now)``.  No non-negativity
    clamp is applied.  The query grid and current/imagined empty-gate fallbacks are
    identical to :func:`safedosebo.acquisitions.ckg_one_step_gated`.
    """
    if gate_mode not in {"latent", "predictive"}:
        raise ValueError("gate_mode must be 'latent' or 'predictive'")
    if not isinstance(r_k, (int, np.integer)) or int(r_k) < 1:
        raise ValueError("r_k must be an integer >= 1")

    mu_f_t, _var_f_t, covf_t, s2f_full = joint(m_eff, l_eff, Xset, d)
    mu_g_t, var_g_t, covg_t, s2g_full = joint(m_tox, l_tox, Xset, d)
    nf = float(l_eff.noise.item())
    ng = float(l_tox.noise.item())
    s2f = (float(s2f_full) - nf) + nf / int(r_k)
    s2g = (float(s2g_full) - ng) + ng / int(r_k)
    if not np.isfinite(s2f) or not np.isfinite(s2g) or s2f <= 0.0 or s2g <= 0.0:
        raise FloatingPointError("cohort fantasy variances must be finite and positive")

    mu_f = mu_f_t.detach().cpu().numpy()
    mu_g = mu_g_t.detach().cpu().numpy()
    var_g = var_g_t.detach().cpu().numpy()
    covf = covf_t.detach().cpu().numpy()
    covg = covg_t.detach().cpu().numpy()
    efficacy_update = covf / np.sqrt(s2f)
    toxicity_update = covg / np.sqrt(s2g)
    obs_g = ng if gate_mode == "predictive" else 0.0
    residual_g = np.maximum(var_g - covg ** 2 / s2g, 1e-12)
    sd_g_post = np.sqrt(residual_g + obs_g)

    _mu_now, var_g_now_t = post_latent(m_tox, Xset)
    sd_g_now = np.sqrt(
        np.maximum(var_g_now_t.detach().cpu().numpy(), 1e-12) + obs_g
    )
    q_gamma = float(norm.ppf(float(gamma)))
    z_now = (float(g_dagger) - mu_g) / sd_g_now
    gate_now = z_now > q_gamma
    current_fallback_index = None
    if gate_now.any():
        current_value = float(mu_f[gate_now].max())
    else:
        current_fallback_index = int(np.argmax(z_now))
        current_value = float(mu_f[current_fallback_index])

    # Center before integrating so the signed KG is not obtained by subtracting
    # two large offset-shifted expectations after the fact.
    centered_mu_f = mu_f - current_value
    imagined = expected_terminal_value_exact(
        centered_mu_f,
        efficacy_update,
        mu_g,
        toxicity_update,
        sd_g_post,
        float(g_dagger),
        float(gamma),
        return_diagnostics=return_diagnostics,
        include_regimes=include_regimes,
    )
    if return_diagnostics:
        centered_expected_value, diagnostics = imagined
        diagnostics.update({
            "current_value": current_value,
            "expected_post_value": float(centered_expected_value + current_value),
            "current_gate_indices": np.flatnonzero(gate_now).astype(int).tolist(),
            "current_fallback_index": current_fallback_index,
            "gate_mode": gate_mode,
            "r_k": int(r_k),
        })
        return float(centered_expected_value), diagnostics
    return float(imagined)


def ckg_scores_exact(ctx, *, return_diagnostics: bool = False,
                     include_regimes: bool = True):
    """Evaluate every accessible query and apply the context's assignment mask.

    This mirrors the registered ``cKG`` wrapper but remains an explicit diagnostic
    call: it is not registered as a production acquisition.  ``ctx.Xset`` is the
    accessible grid ``G_t``; ``ctx.restrict`` applies both the gate and protocol
    eligibility only after all query scores have been evaluated.
    """
    raw_scores: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    for i in range(ctx.Xset.shape[0]):
        result = ckg_one_step_gated_exact(
            ctx.Xset[i],
            ctx.Xset,
            ctx.me,
            ctx.le,
            ctx.mt,
            ctx.lt,
            ctx.g_dagger,
            ctx.gamma,
            r_k=ctx.r_k,
            gate_mode=ctx.gate_mode,
            return_diagnostics=return_diagnostics,
            include_regimes=include_regimes,
        )
        if return_diagnostics:
            score, detail = result
            raw_scores.append(score)
            diagnostics.append(detail)
        else:
            raw_scores.append(result)
    raw = np.asarray(raw_scores, dtype=float)
    restricted = ctx.restrict(raw)
    if return_diagnostics:
        return restricted, {"raw_scores": raw.tolist(), "queries": diagnostics}
    return restricted


__all__ = [
    "expected_terminal_value_exact",
    "ckg_one_step_gated_exact",
    "ckg_scores_exact",
]
