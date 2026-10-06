"""Per-cohort decision context handed to every acquisition.

The trial harness rebuilds a ``Context`` before each cohort and passes it to the
chosen acquisition. It bundles the fitted GPs, the dose grid, the gate, and the
gate-dependent quantities every acquisition tends to need (feasibility probability,
the admitted-and-eligible mask, the toxicity posterior, the tMSE boundary score), plus small
helpers (``restrict``, ``latent_efficacy``, ``cei_scores``, ``normalize_safe``).

An acquisition reads whatever it needs off this object and returns a score per
grid dose. It never has to touch GP internals, the gate wiring, or the metrics.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.stats as ss
import torch

from .gp import post_predictive, post_latent
from . import acquisitions as _acq  # for cei / cei_scores reuse


@dataclass
class Context:
    Xset: torch.Tensor          # (n_grid, 2) candidate doses
    me: object                  # efficacy GP model
    le: object                  # efficacy likelihood
    mt: object                  # toxicity GP model
    lt: object                  # toxicity likelihood
    g_dagger: float             # gate threshold (feasible iff tox <= g_dagger)
    gamma: float                # gate strictness tau (strictly admit iff z > Phi^{-1}(tau))
    gate_mode: str = "latent"   # "latent" or "predictive"
    r_k: int = 2                # cohort size
    step: int = 0               # cohort index since warmup (for alternating rules)
    rng: Optional[np.random.Generator] = None  # policy-only stream; never used for outcomes
    ckg_nmc: int = 512
    ckg_seed: int = 0
    empty_gate: str = "pf"      # "pf" (correct) | "ungated" (legacy pre-correction path)
    candidate_mask: Optional[np.ndarray] = None  # protocol-eligible administrations in Xset

    # cached gate quantities (filled in __post_init__)
    mu_g: np.ndarray = field(init=False)
    sd_g: np.ndarray = field(init=False)
    standardized_feasibility: np.ndarray = field(init=False)
    gate_quantile: float = field(init=False)
    pf: np.ndarray = field(init=False)
    gate_safe: np.ndarray = field(init=False)
    safe: np.ndarray = field(init=False)
    gate_empty: bool = field(init=False)
    most_feasible_full_index: int = field(init=False)
    most_feasible_index: int = field(init=False)
    tmse: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.gate_mode not in {"latent", "predictive"}:
            raise ValueError("gate_mode must be 'latent' or 'predictive'")
        if not np.isfinite(float(self.gamma)) or not 0 < float(self.gamma) < 1:
            raise ValueError("gamma must be finite and strictly between 0 and 1")
        if self.empty_gate not in {"pf", "ungated"}:
            raise ValueError("empty_gate must be 'pf' or 'ungated'")
        mu_g_t, vg_t = (post_predictive(self.mt, self.lt, self.Xset)
                        if self.gate_mode == "predictive" else post_latent(self.mt, self.Xset))
        self.mu_g = mu_g_t.cpu().numpy()
        self.sd_g = vg_t.clamp_min(1e-12).sqrt().cpu().numpy()
        # Use one numerically stable ordering for both the strict chance gate and
        # the most-feasible fallback.  Phi(z) is retained for reporting and for
        # criteria that genuinely use a probability, but it can round distinct
        # z values to the same float in the tails and therefore must not decide
        # the fallback index.
        self.standardized_feasibility = (
            self.g_dagger - self.mu_g
        ) / self.sd_g
        self.gate_quantile = float(ss.norm.ppf(float(self.gamma)))
        self.pf = ss.norm.cdf(self.standardized_feasibility)
        if self.candidate_mask is None:
            self.candidate_mask = np.ones(self.Xset.shape[0], dtype=bool)
        else:
            self.candidate_mask = np.asarray(self.candidate_mask, dtype=bool).reshape(-1)
            if self.candidate_mask.shape != (self.Xset.shape[0],):
                raise ValueError("candidate_mask must have one entry per dose in Xset")
            if not self.candidate_mask.any():
                raise ValueError("candidate_mask must admit at least one dose")
        self.gate_safe = self.standardized_feasibility > self.gate_quantile
        # ``safe`` retains its historical API name for the assignment mask used by
        # built-in acquisitions.  It implements the mathematical strict gate in
        # standardized space; unlike ``pf > gamma``, it is not vulnerable to CDF
        # rounding or saturation.  Under expansion it also enforces eligibility.
        self.safe = self.gate_safe & self.candidate_mask
        self.gate_empty = not bool(self.safe.any())
        self.most_feasible_full_index = int(
            np.argmax(self.standardized_feasibility)
        )
        eligible_z = np.where(
            self.candidate_mask, self.standardized_feasibility, -np.inf
        )
        self.most_feasible_index = int(np.argmax(eligible_z))
        # Targeted mean-squared-error (tMSE) boundary score, sqrt(2*pi) * sd_g * phi(zeta):
        # the pointwise tMSE of Lyu, Binois & Ludkovski (2021, Eqs. 3.9-3.10), a
        # localized eps=0 form of Picheny et al.'s (2010) target-region criterion. It is
        # proportional to the leading-order small-window term of Bichon et al.'s (2008)
        # expected-feasibility criterion when eps=alpha*sd_g; the unnormalised criterion
        # itself tends to zero as alpha->0. NOT Bryan et al.'s (2005) additive straddle
        # 1.96*sd - |mu - t|, which is a different criterion.
        self.tmse = self.sd_g * np.exp(-0.5 * ((self.g_dagger - self.mu_g) / self.sd_g) ** 2)

    @property
    def straddle(self) -> np.ndarray:
        """Deprecated alias for :attr:`tmse`.

        Kept for backwards compatibility. The score is the tMSE criterion, not the
        additive straddle of Bryan et al. (2005); the old name was a misnomer.
        """
        return self.tmse

    # ---- helpers -----------------------------------------------------------
    def restrict(self, v: np.ndarray) -> np.ndarray:
        """Apply the posterior-feasibility gate: excluded doses get ``-inf``. If the gate empties
        the grid, return all ``-inf`` so the caller falls back to the largest
        standardized feasibility score --
        matching ``recommend_obd`` and the paper's stated rule. Set ``empty_gate="ungated"``
        to restore the legacy behaviour that generated the pre-correction tables."""
        v = np.asarray(v, float).reshape(-1)   # accept (n,) or (n,1) without silent broadcast
        if np.isnan(v).any() or np.isposinf(v).any():
            raise ValueError("acquisition scores must not contain NaN or +inf")
        m = np.where(self.safe, v, -np.inf)
        if np.isfinite(m).any():
            return m
        # Gate is empty. Correct behaviour: return all -inf so the harness falls back to
        # argmax z (the most-feasible dose), as the paper describes. The "ungated"
        # legacy path returns the UNGATED scores, i.e. doses at the unconstrained
        # acquisition maximiser -- that is what produced the pre-correction numbers, and it
        # administers a truly-toxic dose far more often. Kept only for reproduction.
        return v if self.empty_gate == "ungated" else m

    def latent_efficacy(self):
        """Latent posterior (mean, var) of efficacy over the grid, as numpy arrays."""
        mu, var = post_latent(self.me, self.Xset)
        return mu.cpu().numpy(), var.cpu().numpy()

    def cei_scores(self) -> np.ndarray:
        """Ungated constrained-EI score at every grid dose (before the gate)."""
        muf, _ = post_latent(self.me, self.Xset)
        muf = muf.cpu().numpy()
        # Improvement is measured against the best gate-passed dose in the whole
        # current region, including already visited candidates. The no-repeat rule
        # constrains where the next cohort may be assigned, not what is incumbent.
        fstar = (muf[self.gate_safe].max() if self.gate_safe.any()
                 else muf[self.most_feasible_full_index])
        return np.array([
            _acq.cei(self.Xset[i], self.Xset, self.me, self.le, self.mt, self.lt,
                     self.g_dagger, fstar, gate_mode=self.gate_mode)
            for i in range(self.Xset.shape[0])
        ])

    def normalize_safe(self, v: np.ndarray) -> np.ndarray:
        """Range-normalise ``v`` over the admitted-and-eligible set to [0, 1]."""
        v = np.asarray(v, float).reshape(-1)
        vs = v[self.safe] if self.safe.any() else v
        lo, hi = float(np.min(vs)), float(np.max(vs))
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
