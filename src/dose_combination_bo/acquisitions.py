"""Acquisition functions.

Two layers:

* **Low-level math** (``cei``, ``ckg_one_step``, ``ckg_one_step_gated``,
  ``qbig_value``) copied verbatim from the paper code so numbers match exactly.
* **Registered plug-ins** -- thin ``@acquisition``-decorated wrappers with the
  benchmark's uniform ``fn(ctx) -> scores`` signature. These are the names you pass
  to ``dose_combination_bo.run_trial`` / ``dose_combination_bo.sweep``. Add your own the same way (see
  ``dose_combination_bo.registry`` and ``examples/add_your_own_acquisition.py``).

Built-in acquisitions
---------------------
    cEI       myopic constrained expected improvement (Willard baseline)
    cKG       deterministic one-step lookahead (bound by public_ckg)
    cKG-legacy512  historical 512-fantasy approximation [alias: cKG1fix]
    cEI-tMSE  fixed lookahead-free cEI/tMSE alternation; cKG is separate
    tmse      gated tMSE boundary score (alias: straddle, deprecated misnomer)
    utmse     ungated tMSE diagnostic (gate ignored; alias: ustrad)
    SPW       equal-weight, range-normalised cEI + tMSE
    qBIG      marginal feasibility-entropy reduction (9-node quadrature)          [alias: SUR]
    random    uniform over the gate-passed set (gated chance control, not a lower bound)
    sKG3      gate-projected cKG (information target vs assignment decoupled)
"""
import numpy as np
import torch
from scipy.stats import norm

from .gp import post_predictive, post_latent, joint, kg_lines
from .registry import acquisition

# ==========================================================================
# Low-level acquisition math (verbatim from ckg_core.py / gate_ab_smoke.py)
# ==========================================================================

def cei(d, Xset, m_eff, l_eff, m_tox, l_tox, g_dagger, f_star, gate_mode='latent'):
    """Constrained expected improvement (Willard closed form): maximize f s.t. g<=g_dagger."""
    mu_f, var_f = post_latent(m_eff, d.unsqueeze(0))    # EI is improvement of the LATENT efficacy
    mu_g, var_g = post_latent(m_tox, d.unsqueeze(0))    # feasibility: latent (default) or predictive g
    if gate_mode == 'predictive':
        var_g = var_g + float(l_tox.noise)              # gp.predict-style sigma_g (adds obs noise)
    sd_f = var_f.clamp_min(1e-12).sqrt(); sd_g = var_g.clamp_min(1e-12).sqrt()
    z = (mu_f - f_star) / sd_f
    N = torch.distributions.Normal(0., 1.)
    ei = ((mu_f - f_star) * N.cdf(z) + sd_f * torch.exp(N.log_prob(z))).clamp_min(0.)
    pf = N.cdf((g_dagger - mu_g) / sd_g)
    return float((ei * pf).item())


def ckg_one_step(d, Xset, m_eff, l_eff, m_tox, l_tox, g_dagger, r_k=1, gh=48, var_g_current=None):
    """Soft-value one-step constrained KG (feasibility-weighted; offset-sensitive).
    Kept for completeness; the gated version below is what the paper's cKG uses."""
    if var_g_current is None:
        _, vgc = post_predictive(m_tox, l_tox, Xset)
        var_g_current = vgc.clamp_min(1e-9).cpu().numpy()
    mu_f, _, covf, s2f_full = joint(m_eff, l_eff, Xset, d)
    mu_g, var_g, covg, s2g_full = joint(m_tox, l_tox, Xset, d)
    nf = l_eff.noise.item(); ng = l_tox.noise.item()
    s2f = (s2f_full - nf) + nf / r_k          # cohort fantasy variance
    s2g = (s2g_full - ng) + ng / r_k
    mu_f = mu_f.cpu().numpy(); mu_g = mu_g.cpu().numpy()
    wf = (covf / s2f).cpu().numpy(); wg = (covg / s2g).cpu().numpy()
    sd_resid = np.sqrt(np.maximum((var_g - covg**2 / s2g).cpu().numpy(), 1e-12))
    sf = np.sqrt(s2f); sg = np.sqrt(s2g)
    x, w = np.polynomial.hermite_e.hermegauss(gh); w = w / np.sqrt(2 * np.pi)
    E = 0.0
    for zg, wq in zip(x, w):
        c = norm.cdf((g_dagger - (mu_g + wg * sg * zg)) / sd_resid)
        E += wq * kg_lines(c * mu_f, c * wf * sf)
    V_now = np.max(norm.cdf((g_dagger - mu_g) / np.sqrt(var_g_current)) * mu_f)
    return E - V_now


def ckg_one_step_gated(d, Xset, m_eff, l_eff, m_tox, l_tox, g_dagger, gamma, r_k=1,
                       nmc=512, seed=0, gate_mode='latent'):
    """CORRECTED discrete constrained KG. The value is TERMINAL-RECOMMENDATION-ALIGNED:
        V(post) = max_{x : P(g(x) <= g_dagger) > gamma} mu_f(x)
    evaluated numerically as the equivalent strict standardized gate
    ``z > Phi^{-1}(gamma)``. This is the believed-best *feasible* efficacy that
    ``recommend_obd`` would pick. The
    hard-gate max is OFFSET-INVARIANT (gate is on g; shifting mu_f shifts V by a
    constant so KG is unchanged), unlike the soft feasibility-weighted value."""
    mu_f, var_f, covf, s2f_full = joint(m_eff, l_eff, Xset, d)
    mu_g, var_g, covg, s2g_full = joint(m_tox, l_tox, Xset, d)
    nf = l_eff.noise.item(); ng = l_tox.noise.item()
    s2f = (s2f_full - nf) + nf / r_k; s2g = (s2g_full - ng) + ng / r_k
    mu_f = mu_f.cpu().numpy(); mu_g = mu_g.cpu().numpy()
    wf = (covf / s2f).cpu().numpy(); wg = (covg / s2g).cpu().numpy()
    sf = np.sqrt(s2f); sg = np.sqrt(s2g)
    obs_g = ng if gate_mode == 'predictive' else 0.0     # predictive gate adds obs noise to sigma_g
    resid_g = np.maximum((var_g - covg**2 / s2g).cpu().numpy(), 1e-12)
    sd_g_post = np.sqrt(resid_g + obs_g)                  # post-fantasy gate SD
    _, vgc = post_latent(m_tox, Xset); sd_g_now = (vgc.clamp_min(1e-12).cpu().numpy() + obs_g) ** 0.5
    q_gamma = float(norm.ppf(float(gamma)))
    z_now = (g_dagger - mu_g) / sd_g_now
    gate_now = z_now > q_gamma
    V_now = mu_f[gate_now].max() if gate_now.any() else mu_f[z_now.argmax()]
    rng = np.random.default_rng(seed)
    Zf = rng.standard_normal((nmc, 1)); Zg = rng.standard_normal((nmc, 1))
    muf2 = mu_f[None, :] + wf[None, :] * sf * Zf
    mug2 = mu_g[None, :] + wg[None, :] * sg * Zg
    z2 = (g_dagger - mug2) / sd_g_post[None, :]
    gate2 = z2 > q_gamma
    Vp = np.where(gate2, muf2, -np.inf).max(axis=1)
    bad = ~np.isfinite(Vp)
    if bad.any():
        Vp[bad] = muf2[bad][np.arange(int(bad.sum())), z2[bad].argmax(axis=1)]
    return float(Vp.mean() - V_now)


def _hb(p):
    """Bernoulli entropy (feasibility uncertainty)."""
    p = np.clip(np.asarray(p), 1e-12, 1 - 1e-12); return -(p * np.log(p) + (1 - p) * np.log(1 - p))


_QX, _QW = np.polynomial.hermite_e.hermegauss(9); _QW = _QW / np.sqrt(2 * np.pi)   # E_{N(0,1)}[f]


def qbig_value(d, Xset, m_tox, l_tox, g_dagger, gate_mode, r_k):
    """One-step boundary information gain: H(feasibility) - E_y[H(feasibility | obs at d)],
    summed over the grid. Exact realization of the alpha_BIG (SUR) channel."""
    mu_g, var_g, covg, s2g_full = joint(m_tox, l_tox, Xset, d)
    ng = l_tox.noise.item(); s2g = (s2g_full - ng) + ng / r_k        # cohort fantasy variance at d
    mu_g = mu_g.cpu().numpy(); var_g = var_g.cpu().numpy(); covg = covg.cpu().numpy()
    obs = ng if gate_mode == 'predictive' else 0.0
    sd_now = np.sqrt(var_g + obs)
    resid = np.maximum(var_g - covg**2 / s2g, 1e-12); sd_post = np.sqrt(resid + obs)
    H_now = _hb(norm.cdf((g_dagger - mu_g) / sd_now))
    zeta0 = (g_dagger - mu_g) / sd_post; tau = (np.abs(covg) / np.sqrt(s2g)) / sd_post
    EH = np.zeros_like(zeta0)
    for xq, wq in zip(_QX, _QW):
        EH += wq * _hb(norm.cdf(zeta0 + tau * xq))
    return float(np.sum(H_now - EH))


# ==========================================================================
# Registered plug-ins: fn(ctx) -> per-dose scores (higher better; -inf excludes)
# ==========================================================================

@acquisition("cEI")
def _cEI(ctx):
    return ctx.restrict(ctx.cei_scores())


@acquisition("cKG-legacy512", aliases=("cKG1fix",))
def _cKG(ctx):
    vals = np.array([
        ckg_one_step_gated(ctx.Xset[i], ctx.Xset, ctx.me, ctx.le, ctx.mt, ctx.lt,
                           ctx.g_dagger, ctx.gamma, r_k=ctx.r_k, nmc=ctx.ckg_nmc,
                           seed=ctx.ckg_seed, gate_mode=ctx.gate_mode)
        for i in range(ctx.Xset.shape[0])])
    return ctx.restrict(vals)


@acquisition("cEI-tMSE")
def _cei_tmse(ctx):
    """Fixed gated cEI/tMSE alternation; it does not invoke cKG."""
    return ctx.restrict(ctx.cei_scores()) if (ctx.step % 2 == 0) else ctx.restrict(ctx.tmse)


@acquisition("tmse", aliases=("straddle",))
def _tmse(ctx):
    """Gated tMSE boundary score. ``straddle`` is a deprecated alias: the score is the
    targeted-MSE criterion (Picheny et al. 2010; Lyu et al. 2021), not Bryan et al.'s
    (2005) additive straddle."""
    return ctx.restrict(ctx.tmse)


@acquisition("utmse", aliases=("ustrad",))
def _utmse(ctx):
    """Ungated tMSE diagnostic (gate ignored) that isolates the gate's role."""
    return ctx.tmse


@acquisition("SPW")
def _SPW(ctx):
    cv = ctx.cei_scores()
    return ctx.restrict(ctx.normalize_safe(cv) + ctx.normalize_safe(ctx.tmse))


@acquisition("qBIG", aliases=("SUR",))
def _qBIG(ctx):
    vals = np.array([
        qbig_value(ctx.Xset[i], ctx.Xset, ctx.mt, ctx.lt, ctx.g_dagger, ctx.gate_mode, ctx.r_k)
        for i in range(ctx.Xset.shape[0])])
    return ctx.restrict(vals)


@acquisition("random")
def _random(ctx):
    si = np.where(ctx.safe)[0]
    pick = (
        int(si[ctx.rng.integers(len(si))])
        if len(si)
        else int(ctx.most_feasible_index)
    )
    vals = np.full(ctx.Xset.shape[0], -np.inf); vals[pick] = 0.0
    return vals


@acquisition("sKG3")
def _sKG3(ctx):
    a = np.array([
        ckg_one_step_gated(ctx.Xset[i], ctx.Xset, ctx.me, ctx.le, ctx.mt, ctx.lt,
                           ctx.g_dagger, ctx.gamma, r_k=ctx.r_k, nmc=ctx.ckg_nmc,
                           seed=ctx.ckg_seed, gate_mode=ctx.gate_mode)
        for i in range(ctx.Xset.shape[0])])
    istar = int(np.argmax(a))
    with torch.no_grad():
        Kp = ctx.mt(ctx.Xset).covariance_matrix.detach().cpu().numpy()
    ng = ctx.lt.noise.item()
    vr = Kp[istar, :]**2 / (np.clip(np.diag(Kp), 1e-12, None) + ng)
    return ctx.restrict(vr)
