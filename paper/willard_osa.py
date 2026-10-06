"""Historical simulator implementation retained for identity checks.

This is an independent Python implementation of the OSA response-surface
specification in Willard, Golchi, and Moodie (2024), arXiv:2404.11323v1. It
implements the published seven-term polynomial equations with the rounded
parameter values printed in the preprint. It contains no patient-level data and
does not reproduce the authors' R code or full trial algorithm. The current
package-facing implementation is ``src/dose_combination_bo/surfaces.py``.

Functional form (their Table 1, "OSA" rows), 7-term polynomial basis:
    h(d,z) = c0 + c1 d1 + c2 d2 + c3 d1 d2 + c4 d1^2 + c5 d2^2 + c6 d1^2 d2^2
  efficacy f uses beta (stratum-specific); toxicity g uses theta (shared).
  doses standardized to [0,1]^2; smaller f and g are BETTER (they MINIMIZE f
  s.t. g <= g_dagger). NOTE: their convention minimizes f; our ckg_core
  MAXIMIZES efficacy, so we return NEGATED f for use as "efficacy" if desired
  (see to_efficacy_maximization()).

Parameters (from the paper text):
  beta_{z=0} = (-1.38, -4.08, -0.48, -4.23, 2.45, -7.51, -1.56)   mild/moderate
  beta_{z=1} = ( 1.05,-11.28, -8.32,-17.02, 8.17,  2.34,  4.61)   severe
  theta      = (-0.59,  1.83,  2.26, -4.05, 1.79,  0.47,  2.91)   (shared)
  thresholds: g_dagger = 1.5 (mild/moderate, z=0),  2.0 (severe, z=1)
  The ``noise_sd`` helper below is a later diagnostic based on response-surface
  SD; it is not the source paper's observation-noise calibration.
"""
import numpy as np

BETA = {0: np.array([-1.38, -4.08, -0.48, -4.23, 2.45, -7.51, -1.56]),
        1: np.array([ 1.05,-11.28, -8.32,-17.02, 8.17, 2.34,  4.61])}
THETA = np.array([-0.59, 1.83, 2.26, -4.05, 1.79, 0.47, 2.91])
G_DAGGER = {0: 1.5, 1: 2.0}

def _basis(d1, d2):
    return np.array([np.ones_like(d1), d1, d2, d1*d2, d1**2, d2**2, d1**2 * d2**2])

def f_osa(d1, d2, z=0):
    """Efficacy surface (their convention: SMALLER is better = bigger AHI reduction)."""
    return np.tensordot(BETA[z], _basis(d1, d2), axes=([0],[0]))

def g_osa(d1, d2, z=0):
    """Toxicity surface (log AE-burden); feasible if g <= g_dagger[z]."""
    return np.tensordot(THETA, _basis(d1, d2), axes=([0],[0]))

def true_obd_osa(z=0, grid=200):
    """Dense-grid approximation to min f subject to g<=g_dagger on [0,1]^2."""
    xs = np.linspace(0, 1, grid); G1, G2 = np.meshgrid(xs, xs)
    F = f_osa(G1, G2, z); G = g_osa(G1, G2, z)
    Fm = np.where(G <= G_DAGGER[z], F, np.inf)   # minimize f
    i = np.unravel_index(Fm.argmin(), Fm.shape)
    return np.array([G1[i], G2[i]]), F[i]

def noise_sd(z=0, ses=1.0):
    """Return a response-surface-SD noise diagnostic.

    ``ses`` is the per-observation signal-to-noise ratio, so the returned noise
    SD is ``std(h) / ses``. This helper is not the source paper's observation-
    noise calibration.
    """
    xs = np.linspace(0, 1, 100); G1, G2 = np.meshgrid(xs, xs)
    f_std = float(f_osa(G1, G2, z).std())
    g_std = float(g_osa(G1, G2, z).std())
    return f_std / ses, g_std / ses

if __name__ == "__main__":
    for z in (0, 1):
        dopt, fopt = true_obd_osa(z)
        sf, sg = noise_sd(z)
        print(f"stratum z={z} (g_dagger={G_DAGGER[z]}): "
              f"true OBD={dopt.round(3)} f_opt={fopt:.2f} "
              f"g_at_opt={float(g_osa(dopt[0],dopt[1],z)):.2f}  sigma_f={sf:.2f} sigma_g={sg:.2f}")
