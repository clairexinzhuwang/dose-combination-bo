"""Recommendation rule and operating-characteristic metrics."""
import numpy as np
from scipy.stats import norm

from .gp import post_latent
from .decision import select_from_posterior


def recommend_obd(Xset, m_eff, l_eff, m_tox, l_tox, g_dagger, gamma=0.5, gate_mode='latent'):
    """Recommend the believed-best feasible dose: argmax latent efficacy over the
    gate-passed set (fall back to the most-feasible dose if the gate is empty)."""
    if gate_mode not in {"latent", "predictive"}:
        raise ValueError("gate_mode must be 'latent' or 'predictive'")
    if Xset.ndim != 2 or Xset.shape[1] != 2 or len(Xset) == 0:
        raise ValueError("Xset must be a nonempty n-by-2 dose tensor")
    if not np.isfinite(Xset.detach().cpu().numpy()).all():
        raise ValueError("Xset must contain finite doses")
    mu_f, _ = post_latent(m_eff, Xset)
    mu_g, var_g = post_latent(m_tox, Xset)
    decision = select_from_posterior(
        mu_f.detach().cpu().numpy(), mu_g.detach().cpu().numpy(),
        var_g.detach().cpu().numpy(), g_dagger, gamma,
        observation_variance=float(l_tox.noise) if gate_mode == 'predictive' else 0.0,
    )
    idx = decision.index
    return Xset[idx].detach().cpu().numpy(), idx


def dose_units(d_hat, d_opt, spacing=0.25):
    """Selection error: Euclidean distance from the recommendation to the supplied OBD reference,
    in grid units. ``spacing=0.25`` is the public 5-by-5-grid convenience default;
    the trial harness always passes its actual ``1 / (grid_n - 1)`` spacing so this
    metric remains correct for non-default grids. Built-in surfaces use the fixed numerical
    references documented with the study. Lower is better."""
    if not np.isscalar(spacing) or not np.isfinite(float(spacing)) or float(spacing) <= 0:
        raise ValueError("spacing must be a positive finite scalar")
    estimated, target = np.asarray(d_hat, float), np.asarray(d_opt, float)
    if estimated.shape != (2,) or target.shape != (2,) or not np.isfinite(estimated).all() or not np.isfinite(target).all():
        raise ValueError("d_hat and d_opt must be finite dose pairs")
    return float(np.linalg.norm(estimated - target) / spacing)


def n_toxic(tox_history, g_dagger):
    """Number of simulated assignments whose latent toxicity exceeded the threshold."""
    history = np.asarray(tox_history, dtype=float)
    if history.ndim != 1 or not np.isfinite(history).all() or not np.isfinite(float(g_dagger)):
        raise ValueError("tox_history must be a finite vector and g_dagger finite")
    return int(np.sum(history > float(g_dagger)))
