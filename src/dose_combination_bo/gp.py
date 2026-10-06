"""Gaussian-process surrogate and posterior helpers.

Uses the study's GP specification and 120-step Adam fitting recipe. This release
adds input and numerical-failure checks. Historical replay must use the separately
retained frozen source rather than assume byte-identical outputs from this release. Two independent
single-task GPs are fit, one for efficacy and one for toxicity, reproducing the paper's fixed
working-surrogate specification. This benchmark does not compare independent and multi-output
GPs; exact autokrigeability is a noise-free separable-kernel result and does not establish
equivalence for the noisy outcomes used here.

Public helpers
--------------
fit_gp(X, y, ...)        -> (model, likelihood)   fit one GP
post_predictive(m, l, X) -> (mean, var)           latent + observation noise
post_latent(m, X)        -> (mean, var)           latent function only
joint(m, l, Xset, d)     -> (mu, var, cov, s2)    grid-vs-query joint (for KG)
kg_lines(a, b)           -> float                 Frazier exact inner KG
"""
import numpy as np
import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from scipy.stats import norm

torch.set_default_dtype(torch.double)


class _GP(gpytorch.models.ExactGP):
    def __init__(self, x, y, lik):
        super().__init__(x, y, lik)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=x.shape[-1]))

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


def fit_gp(X, y, noise_cap=0.3, iters=120, lr=0.1, fixed_noise=None):
    """Fit an independent single-task GP (Matern-5/2, ARD, constant mean).

    ``fixed_noise=None``  -> learn positive noise, initialised at the sample
    variance (Willard empirical-Bayes; the marginal-likelihood noise is
    ill-identified on small within-trial samples and can collapse toward zero, so
    the sample-variance initialization matters). ``noise_cap`` is retained only for
    historical signature compatibility and is not applied.

    ``fixed_noise`` set   -> pin the observation noise to the KNOWN simulation
    variance and learn the constant mean, lengthscales and output scale. Their
    small-sample identification is not guaranteed. This is
    the paper's default (``noise='fixed'``).
    """
    if X.ndim != 2 or X.shape[1] != 2 or len(X) == 0 or y.shape != (len(X),):
        raise ValueError("X and y must have shapes (n, 2) and (n,), with n > 0")
    if not bool(torch.isfinite(X).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("GP observations must be finite")
    if isinstance(iters, (bool, np.bool_)) or not isinstance(iters, (int, np.integer)) or iters < 1:
        raise ValueError("iters must be an integer >= 1")
    if not np.isfinite(float(lr)) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if fixed_noise is not None and (not np.isfinite(float(fixed_noise)) or fixed_noise <= 1e-5):
        raise ValueError("fixed_noise must be finite and exceed the likelihood's 1e-5 lower bound")
    if fixed_noise is not None:
        lik = GaussianLikelihood(noise_constraint=GreaterThan(1e-5))
        lik.noise = float(fixed_noise)
    else:
        lik = GaussianLikelihood(noise_constraint=GreaterThan(1e-5))
        lik.noise = float(max(np.var(np.asarray(y.detach().cpu())), 1e-4))
    m = _GP(X, y, lik); m.train(); lik.train()
    params = ([p for n, p in m.named_parameters() if 'noise' not in n]
              if fixed_noise is not None else list(m.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, m)
    for _ in range(iters):
        opt.zero_grad()
        loss = -mll(m(X), y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite GP marginal-likelihood loss")
        loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
            raise FloatingPointError("nonfinite GP gradient")
        opt.step()
    m.eval(); lik.eval()
    return m, lik


def post_predictive(m, l, X):
    """Predictive posterior: latent + observation noise. Use only for a fantasy
    of a future observation (see ``joint``)."""
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        p = l(m(X)); return p.mean, p.variance


def post_latent(m, X):
    """Latent posterior of f/g (no observation noise). Feasibility, EI improvement,
    and RPSEL are all properties of the latent function -> use this, not
    ``post_predictive``."""
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f = m(X); return f.mean, f.variance


def joint(m, l, Xset, d):
    """Return (mu_grid, var_grid, cov_grid_query, s2_query+noise) for the joint
    latent posterior over ``Xset`` and a single query dose ``d``."""
    Xall = torch.cat([Xset, d.unsqueeze(0)], 0)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f = m(Xall); mean = f.mean; cov = f.covariance_matrix; noise = l.noise.item()
    n = Xset.shape[0]
    return mean[:n], torch.diagonal(cov)[:n], cov[:n, n], cov[n, n].item() + noise


def kg_lines(a, b):
    """Frazier piecewise-linear inner knowledge gradient: E[max_i (a_i + b_i Z)]
    for Z ~ N(0,1), computed exactly over the upper envelope of the affine lines."""
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
            j = idx[-1]; cij = (a[j] - a[i]) / (b[i] - b[j])
            if cij <= c[-1]:
                idx.pop(); c.pop()
                if not idx:
                    break
            else:
                break
        idx.append(i); c.append(cij)
    c.append(np.inf); ae = a[idx]; be = b[idx]; c = np.array(c)
    lo, hi = c[:-1], c[1:]
    return float(np.sum(ae * (norm.cdf(hi) - norm.cdf(lo)) + be * (norm.pdf(lo) - norm.pdf(hi))))


# Backwards-compatible private aliases (match ckg_core names used across the paper code)
_post = post_predictive
_post_latent = post_latent
_joint = joint
_kg_lines = kg_lines
