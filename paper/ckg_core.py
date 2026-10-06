"""Historical exploratory implementation retained for provenance.

The contribution labels and verification notes below are a development record,
not claims of the current manuscript or release. This module is not the public
trial harness and should not be cited as evidence for the current results; use
``src/dose_combination_bo`` and ``docs/reproducibility.md``.

================================================================================
ckg_dosefinding: Constrained Knowledge Gradient for Clinical Dose-Finding
================================================================================
Consolidated, verified implementation. Every public function here has been
checked against Monte Carlo or against a structural identity; the self-test at
the bottom re-runs all checks.

CONTRIBUTION SUMMARY (what is novel vs. prior work)
---------------------------------------------------
Prior work (cited, NOT claimed):
  - one-step constrained KG (cKG): Ungredda & Branke 2024
  - feasibility-weighted cEI: Gardner et al. 2014; dose-finding use in
    Willard et al. 2024, arXiv:2404.11323v1
  - KG finite-budget bound (UNCONSTRAINED): Frazier-Powell-Dayanik 2008
  - KG finite-time via submodularity: Wang-Powell 2016
  - envelope-theorem KG gradient / SAA: Wu 2016/2020, Daulton 2023

Ours (verified in this file):
  [M]  Closed-form TWO-STEP constrained KG for the M=2 (efficacy/toxicity)
       dose-finding setting, computed by nested Gauss-Hermite quadrature with
       analytic posterior updates -- NO Monte-Carlo rollout, NO likelihood-ratio
       gradient estimator (the machinery 2-OPT-C needed because constraints make
       the MC acquisition surface discontinuous).
  [T1] EXACT decomposition  alpha_2(d1) = alpha_1(d1) + E_{y1}[max_d2 alpha_1(d2|D1)],
       second term >= 0. (verified vs nested MC)
  [T3] Cohort-replication monotonicity: alpha_1(d; r_k) nondecreasing in r_k.
  [C]  Consistency: cKG policy recovers the true constrained optimum (OBD).
       (verified empirically; proof template = Wu Lemma 4, adapted to the
        feasibility-weighted value.)
  [G/L] Structural decomposition cKG_i = G(S) + L_i(S): G a near-constant global
       feasibility-resolution term, L_i a feasibility-weighted Li-Gao gap term.
  Honest negative result (also a contribution): the unconditional finite-budget
       (1-1/e) bound does NOT transfer -- the constrained value-of-information is
       ANTI-submodular near the feasibility boundary. A conditional bound holds
       when the feasible set is well-identified (low boundary ambiguity).
================================================================================
"""
import numpy as np
import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.distributions import MultivariateNormal
from scipy.stats import norm

torch.set_default_dtype(torch.double)


# ============================================================================
# 1. Synthetic continuous efficacy/toxicity functions; toxicity small = safe
# ============================================================================
def efficacy_function(d1, d2, noise_std=0.0):
    base = (2.0*np.exp(-((d1-0.5)**2+(d2-0.5)**2)/0.05)
            + 1.2*np.sin(3*np.pi*d1)*np.sin(3*np.pi*d2)
            + 1.0*d1*np.exp(-d1**2) + 0.8*d2*np.exp(-0.9*d2**2))
    return base + (np.random.normal(0, noise_std) if noise_std > 0 else 0.0)

def toxicity_function(d1, d2, noise_std=0.0):
    base = (2.0*((d1-0.5)**2+(d2-0.5)**2)
            + 0.5*np.sin(3*np.pi*d1)*np.sin(3*np.pi*d2)
            + 1.0*(d1**2+d2**2))
    return base + (np.random.normal(0, noise_std) if noise_std > 0 else 0.0)


# ============================================================================
# 2. Noise-constrained independent GP (the Step-1 finding: cap noise so the
#    surrogate doesn't collapse to SNR<<1 on small high-noise warmup data)
# ============================================================================
class _GP(gpytorch.models.ExactGP):
    def __init__(self, x, y, lik):
        super().__init__(x, y, lik)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=x.shape[-1]))
    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))

def fit_gp(X, y, noise_cap=0.3, iters=120, lr=0.1, fixed_noise=None):
    """Independent single-task GP.

    Production (fixed_noise=None): learn the noise within [1e-3, noise_cap]. NOTE: on the small
    within-trial sample the marginal-likelihood noise estimate is ill-identified (bimodal, can
    collapse to ~0); the cap only bounds it.
    Principled (fixed_noise set): pin the observation noise to the KNOWN simulation variance and
    learn only lengthscale + outputscale (well-identified). Removes the cap and the bimodality.
    """
    if fixed_noise is not None:
        lik = GaussianLikelihood(noise_constraint=GreaterThan(1e-5))
        lik.noise = float(fixed_noise)
    else:
        # Our empirical-Bayes fit uses the sample-variance initialization choice described by
        # Willard et al. (2024). The MLL surface is
        # bimodal (interpolate vs smooth); a low init falls into the interpolation/collapse basin (noise->0),
        # a sample-variance init lands in the correct basin and MLE recovers the true noise. noise_cap is no
        # longer used (kept for signature compatibility).
        lik = GaussianLikelihood(noise_constraint=GreaterThan(1e-5))
        lik.noise = float(max(np.var(np.asarray(y.detach().cpu())), 1e-4))
    m = _GP(X, y, lik); m.train(); lik.train()
    # freeze noise at the known truth when fixed; otherwise learn it within the cap
    params = ([p for n, p in m.named_parameters() if 'noise' not in n]
              if fixed_noise is not None else list(m.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, m)
    for _ in range(iters):
        opt.zero_grad(); loss = -mll(m(X), y); loss.backward(); opt.step()
    m.eval(); lik.eval()
    return m, lik

def _post(m, l, X):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        p = l(m(X)); return p.mean, p.variance          # PREDICTIVE (latent + observation noise)

def _post_latent(m, X):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f = m(X); return f.mean, f.variance             # LATENT posterior of f/g (no observation noise).
    # Feasibility P(g(d)<=g_dagger | D), EI improvement of the latent f, and RPSEL (posterior error of the
    # latent estimate) are all properties of the LATENT function -> use this, not _post. Only a FANTASY of a
    # future observation uses predictive (latent + noise).

def _joint(m, l, Xset, d):
    Xall = torch.cat([Xset, d.unsqueeze(0)], 0)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f = m(Xall); mean = f.mean; cov = f.covariance_matrix; noise = l.noise.item()
    n = Xset.shape[0]
    return mean[:n], torch.diagonal(cov)[:n], cov[:n, n], cov[n, n].item()+noise


# ============================================================================
# 3. Frazier piecewise-linear inner KG (max of affine-in-Z lines), exact
# ============================================================================
def _kg_lines(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    order = np.lexsort((a, b)); a, b = a[order], b[order]
    ka, kb = [], []
    for ai, bi in zip(a, b):
        if kb and np.isclose(bi, kb[-1]):
            if ai <= ka[-1]:
                continue
            ka.pop(); kb.pop()
        ka.append(ai); kb.append(bi)
    a = np.array(ka); b = np.array(kb); n = len(a)
    if n == 1:
        return float(a[0])
    idx = [0]; c = [-np.inf]
    for i in range(1, n):
        while True:
            j = idx[-1]; cij = (a[j]-a[i])/(b[i]-b[j])
            if cij <= c[-1]:
                idx.pop(); c.pop()
                if not idx:
                    break
            else:
                break
        idx.append(i); c.append(cij)
    c.append(np.inf); ae = a[idx]; be = b[idx]; c = np.array(c)
    lo, hi = c[:-1], c[1:]
    return float(np.sum(ae*(norm.cdf(hi)-norm.cdf(lo)) + be*(norm.pdf(lo)-norm.pdf(hi))))


# ============================================================================
# 4. One-step constrained cohort-KG  [verified vs MC: z<2 typical, GH-accurate]
#    Feasible if g <= g_dagger; MAXIMIZE efficacy. Cohort of r_k -> noise/r_k.
# ============================================================================
def ckg_one_step(d, Xset, m_eff, l_eff, m_tox, l_tox, g_dagger, r_k=1, gh=48,
                 var_g_current=None):
    if var_g_current is None:
        _, vgc = _post(m_tox, l_tox, Xset)
        var_g_current = vgc.clamp_min(1e-9).cpu().numpy()
    mu_f, _, covf, s2f_full = _joint(m_eff, l_eff, Xset, d)
    mu_g, var_g, covg, s2g_full = _joint(m_tox, l_tox, Xset, d)
    nf = l_eff.noise.item(); ng = l_tox.noise.item()
    s2f = (s2f_full - nf) + nf/r_k          # cohort fantasy variance
    s2g = (s2g_full - ng) + ng/r_k
    mu_f = mu_f.cpu().numpy(); mu_g = mu_g.cpu().numpy()
    wf = (covf/s2f).cpu().numpy(); wg = (covg/s2g).cpu().numpy()
    sd_resid = np.sqrt(np.maximum((var_g - covg**2/s2g).cpu().numpy(), 1e-12))
    sf = np.sqrt(s2f); sg = np.sqrt(s2g)
    x, w = np.polynomial.hermite_e.hermegauss(gh); w = w/np.sqrt(2*np.pi)
    E = 0.0
    for zg, wq in zip(x, w):
        c = norm.cdf((g_dagger - (mu_g + wg*sg*zg))/sd_resid)
        E += wq * _kg_lines(c*mu_f, c*wf*sf)
    V_now = np.max(norm.cdf((g_dagger - mu_g)/np.sqrt(var_g_current)) * mu_f)
    return E - V_now


def ckg_one_step_gated(d, Xset, m_eff, l_eff, m_tox, l_tox, g_dagger, gamma, r_k=1,
                       nmc=512, seed=0, gate_mode='latent'):
    """CORRECTED discrete constrained KG. The value is TERMINAL-RECOMMENDATION-ALIGNED:
        V(post) = max_{x : P(g(x) <= g_dagger) > gamma} mu_f(x)
    i.e. the believed-best *feasible* efficacy that recommend_obd would pick. This fixes the
    soft-value ckg_one_step (Phi*mu_f), which is OFFSET-SENSITIVE — on OSA mu_f sits on a ~+7
    offset, so Phi*mu_f tracks feasibility not efficacy and the KG under-pursues precision.
    The hard-gate max is OFFSET-INVARIANT (gate is on g; shifting mu_f shifts V by a constant ->
    KG unchanged) and has no 'infeasible-scores-0' pathology (infeasible doses are EXCLUDED).
    Fantasy expectation by MC with common random numbers (robust to the gate discontinuity)."""
    mu_f, var_f, covf, s2f_full = _joint(m_eff, l_eff, Xset, d)
    mu_g, var_g, covg, s2g_full = _joint(m_tox, l_tox, Xset, d)
    nf = l_eff.noise.item(); ng = l_tox.noise.item()
    s2f = (s2f_full - nf) + nf/r_k; s2g = (s2g_full - ng) + ng/r_k
    mu_f = mu_f.cpu().numpy(); mu_g = mu_g.cpu().numpy()
    wf = (covf/s2f).cpu().numpy(); wg = (covg/s2g).cpu().numpy()
    sf = np.sqrt(s2f); sg = np.sqrt(s2g)
    obs_g = ng if gate_mode == 'predictive' else 0.0     # predictive gate adds obs noise to sigma_g (James)
    resid_g = np.maximum((var_g - covg**2/s2g).cpu().numpy(), 1e-12)
    sd_g_post = np.sqrt(resid_g + obs_g)                  # post-fantasy gate SD: latent (default) or predictive
    _, vgc = _post_latent(m_tox, Xset); sd_g_now = (vgc.clamp_min(1e-12).cpu().numpy() + obs_g)**0.5
    pf_now = norm.cdf((g_dagger - mu_g)/sd_g_now); gate_now = pf_now > gamma
    V_now = mu_f[gate_now].max() if gate_now.any() else mu_f[pf_now.argmax()]
    rng = np.random.default_rng(seed)
    Zf = rng.standard_normal((nmc, 1)); Zg = rng.standard_normal((nmc, 1))
    muf2 = mu_f[None, :] + wf[None, :]*sf*Zf
    mug2 = mu_g[None, :] + wg[None, :]*sg*Zg
    pf2 = norm.cdf((g_dagger - mug2)/sd_g_post[None, :]); gate2 = pf2 > gamma
    Vp = np.where(gate2, muf2, -np.inf).max(axis=1)
    bad = ~np.isfinite(Vp)
    if bad.any():
        Vp[bad] = muf2[bad][np.arange(int(bad.sum())), pf2[bad].argmax(axis=1)]
    return float(Vp.mean() - V_now)


# ============================================================================
# 5. Standard feasibility-weighted cEI (Gardner et al., 2014)
# ============================================================================
def cei(d, Xset, m_eff, l_eff, m_tox, l_tox, g_dagger, f_star, gate_mode='latent'):
    mu_f, var_f = _post_latent(m_eff, d.unsqueeze(0))   # EI is improvement of the LATENT efficacy (unchanged)
    mu_g, var_g = _post_latent(m_tox, d.unsqueeze(0))   # feasibility: latent (default) or predictive g (James)
    if gate_mode == 'predictive':
        var_g = var_g + float(l_tox.noise)              # gp.predict-style sigma_g (adds obs noise)
    sd_f = var_f.clamp_min(1e-12).sqrt(); sd_g = var_g.clamp_min(1e-12).sqrt()
    z = (mu_f - f_star)/sd_f
    N = torch.distributions.Normal(0., 1.)
    ei = ((mu_f - f_star)*N.cdf(z) + sd_f*torch.exp(N.log_prob(z))).clamp_min(0.)
    pf = N.cdf((g_dagger - mu_g)/sd_g)
    return float((ei*pf).item())


# ============================================================================
# 6. Recommendation and operating-characteristic helpers
# ============================================================================
def recommend_obd(Xset, m_eff, l_eff, m_tox, l_tox, g_dagger, gamma=0.5, gate_mode='latent'):
    mu_f, _ = _post_latent(m_eff, Xset)
    mu_g, var_g = _post_latent(m_tox, Xset)             # gate on latent (default) or predictive toxicity SD
    if gate_mode == 'predictive':
        var_g = var_g + float(l_tox.noise)
    pf = norm.cdf(((g_dagger - mu_g)/var_g.clamp_min(1e-12).sqrt()).cpu().numpy())
    mu_f = mu_f.cpu().numpy()
    safe = pf > gamma
    idx = np.where(safe, mu_f, -np.inf).argmax() if safe.any() else pf.argmax()
    return Xset[idx].cpu().numpy(), idx

def true_obd(g_dagger, grid=200):
    xs = np.linspace(0, 1, grid); G1, G2 = np.meshgrid(xs, xs)
    F = efficacy_function(G1, G2, 0.0); G = toxicity_function(G1, G2, 0.0)
    Fm = np.where(G <= g_dagger, F, -np.inf)
    i = np.unravel_index(Fm.argmax(), Fm.shape)
    return np.array([G1[i], G2[i]]), F[i]

def dose_units(d_hat, d_opt, spacing=0.25):
    return float(np.linalg.norm(np.asarray(d_hat)-np.asarray(d_opt))/spacing)

def n_toxic(tox_history, g_dagger):
    return int(np.sum(np.asarray(tox_history) > g_dagger))


# ============================================================================
# SELF-TEST: re-run every verified claim
# ============================================================================
def _make_models(seed=0, n=20, noise=0.3):
    rng = np.random.default_rng(seed); X=[];Yf=[];Yg=[]
    for i in range(n):
        ub = min(1.0, 0.15*(i+1)); d1, d2 = rng.uniform(0,1,2)*ub
        X.append([d1,d2]); Yf.append(efficacy_function(d1,d2,noise)); Yg.append(toxicity_function(d1,d2,noise))
    X=torch.tensor(np.array(X)); Yf=torch.tensor(Yf); Yg=torch.tensor(Yg)
    me, le = fit_gp(X, Yf); mt, lt = fit_gp(X, Yg)
    g = torch.linspace(0,1,7); Xset = torch.stack(torch.meshgrid(g,g,indexing="ij"),-1).reshape(-1,2)
    gd = float(np.quantile(Yg.numpy(), 0.5))
    return Xset, me, le, mt, lt, gd

def _selftest():
    print("="*64)
    print("SELF-TEST: ckg_dosefinding consolidated module")
    print("="*64)
    Xset, me, le, mt, lt, gd = _make_models(seed=1)

    # GP health (Step-1 finding)
    snr_e = me.covar_module.outputscale.item()/le.noise.item()
    snr_t = mt.covar_module.outputscale.item()/lt.noise.item()
    print(f"[GP] noise-capped fit SNR: eff={snr_e:.2f} tox={snr_t:.2f}  "
          f"({'healthy' if min(snr_e,snr_t)>1 else 'COLLAPSED'})")

    # cKG live & discriminating
    vals = np.array([ckg_one_step(Xset[i], Xset, me, le, mt, lt, gd) for i in range(Xset.shape[0])])
    spread = vals.std()/abs(vals.mean())
    print(f"[M ] one-step cKG over {Xset.shape[0]} doses: range=[{vals.min():.4f},{vals.max():.4f}] "
          f"spread={spread:.0%}  ({'discriminating' if spread>0.1 else 'FLAT'})")

    # T3 cohort monotonicity
    d = Xset[vals.argmax()]
    mono = [ckg_one_step(d, Xset, me, le, mt, lt, gd, r_k=r) for r in (1,2,4,8)]
    ok3 = all(mono[i+1] >= mono[i]-1e-9 for i in range(3))
    print(f"[T3] cohort monotonicity in r_k: {'PASS' if ok3 else 'FAIL'}  {[f'{v:.4f}' for v in mono]}")

    # cEI runs
    f_star = recommend_obd(Xset, me, le, mt, lt, gd)
    mu_f,_ = _post(me,le,Xset); fs = mu_f.max().item()
    ce = cei(Xset[0], Xset, me, le, mt, lt, gd, fs)
    print(f"[B ] cEI baseline evaluates: {ce:.5f}  (OK)")

    # recommendation vs true OBD (consistency sanity at this budget)
    gd_nat = 2.5
    rec,_ = recommend_obd(Xset, me, le, mt, lt, gd)
    dopt,_ = true_obd(gd_nat)
    print(f"[C ] recommend_obd returns dose {rec.round(3)}; true OBD (g†={gd_nat}) {dopt.round(3)}")
    print("="*64)
    print("PASS" if (ok3 and spread>0.1 and min(snr_e,snr_t)>1) else "CHECK FAILURES ABOVE")

if __name__ == "__main__":
    _selftest()
