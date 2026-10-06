"""Recommendation rule and operating-characteristic metrics."""
import numpy as np
from scipy.stats import norm

from .gp import post_latent


def recommend_obd(Xset, m_eff, l_eff, m_tox, l_tox, g_dagger, gamma=0.5, gate_mode='latent'):
    """Recommend the believed-best feasible dose: argmax latent efficacy over the
    gate-passed set (fall back to the most-feasible dose if the gate is empty)."""
    mu_f, _ = post_latent(m_eff, Xset)
    mu_g, var_g = post_latent(m_tox, Xset)              # gate on latent (default) or predictive tox SD
    if gate_mode == 'predictive':
        var_g = var_g + float(l_tox.noise)
    z = ((g_dagger - mu_g) / var_g.clamp_min(1e-12).sqrt()).cpu().numpy()
    q = float(norm.ppf(float(gamma)))
    mu_f = mu_f.cpu().numpy()
    safe = z > q
    idx = np.where(safe, mu_f, -np.inf).argmax() if safe.any() else z.argmax()
    return Xset[idx].cpu().numpy(), idx


def dose_units(d_hat, d_opt, spacing=0.25):
    """Selection error: Euclidean distance from the recommendation to the supplied OBD reference,
    in grid units. ``spacing=0.25`` is the public 5-by-5-grid convenience default;
    the trial harness always passes its actual ``1 / (grid_n - 1)`` spacing so this
    metric remains correct for non-default grids. Built-in surfaces use the fixed numerical
    references documented with the study. Lower is better."""
    if not np.isscalar(spacing) or not np.isfinite(float(spacing)) or float(spacing) <= 0:
        raise ValueError("spacing must be a positive finite scalar")
    return float(np.linalg.norm(np.asarray(d_hat) - np.asarray(d_opt)) / spacing)


def n_toxic(tox_history, g_dagger):
    """Number of simulated assignments whose latent toxicity exceeded the threshold."""
    return int(np.sum(np.asarray(tox_history) > g_dagger))
