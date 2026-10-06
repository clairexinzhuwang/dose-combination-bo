"""Add your own acquisition and compare it with the paper's reference policies.

Run:  python examples/add_your_own_acquisition.py

An acquisition is one function ``fn(ctx) -> scores`` (higher is better; ``-inf``
excludes a dose). The public decorator validates the scores and applies the common
model-based toxicity criterion and protocol-eligibility mask before the harness picks a dose.
You read the fitted state you need from ``ctx`` without changing GP or trial internals.
"""
import numpy as np

import dose_combination_bo as bo


# ---------------------------------------------------------------------------
# Example 1: gated upper-confidence-bound on efficacy (a classic BO acquisition).
# The harness calls acq_fn(ctx) with no extra arguments, so any hyperparameter is
# baked in here. To *sweep* a hyperparameter, register one acquisition per value
# with a factory (see gated_ucb_b* below).
# ---------------------------------------------------------------------------
@bo.acquisition("gated_ucb")
def gated_ucb(ctx):
    beta = 2.0
    mu_e, var_e = ctx.latent_efficacy()          # posterior efficacy mean + variance over the grid
    return mu_e + beta * np.sqrt(np.maximum(var_e, 0.0))


# Sweeping a hyperparameter: register one named acquisition per beta value.
def register_gated_ucb(beta):
    @bo.acquisition(f"gated_ucb_b{beta}")
    def _ucb(ctx, beta=beta):
        mu_e, var_e = ctx.latent_efficacy()
        return mu_e + beta * np.sqrt(np.maximum(var_e, 0.0))
    return f"gated_ucb_b{beta}"


for _b in (1.0, 2.0, 3.0):
    register_gated_ucb(_b)


# ---------------------------------------------------------------------------
# Example 2: a boundary-aware rule -- efficacy UCB plus a bonus for doses whose
# toxicity is uncertain near the threshold (the tMSE boundary score is precomputed).
# ---------------------------------------------------------------------------
@bo.acquisition("ucb_plus_boundary")
def ucb_plus_boundary(ctx):
    mu_e, var_e = ctx.latent_efficacy()
    ucb = ctx.normalize_safe(mu_e + 2.0 * np.sqrt(np.maximum(var_e, 0.0)))
    boundary = ctx.normalize_safe(ctx.tmse)  # sd_g * exp(-1/2 z^2): high where the gate is uncertain
    return ucb + 0.5 * boundary


if __name__ == "__main__":
    print("registered acquisitions:", bo.list_acquisitions())
    print("\nrunning an installation-scale sweep on the OSA surface (seeds=3, budget=12)...\n")

    res = bo.sweep(
        acquisitions=["cKG", "tmse", "entropy", "cEI", "gated_ucb", "ucb_plus_boundary"],
        surfaces=["osa"],
        seeds=3,
        gammas=[0.7],
        strata=[0],
        parallel=False,     # set True if you have ray installed
        budget=12,
        verbose=True,
    )

    print("\n=== selection precision (dose-units, lower is better) ===")
    bo.leaderboard(res, metric="dose_units", sim="osa", lower_is_better=True)

    print("\n=== final selections above the toxicity limit (lower is better) ===")
    bo.leaderboard(res, metric="rec_unsafe", sim="osa", lower_is_better=True)
