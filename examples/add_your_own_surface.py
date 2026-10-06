"""Add a dose-finding surface (data-generating scenario) and compare policies on it.

Run:  python examples/add_your_own_surface.py

A surface is one function ``fn(z) -> spec`` returning the true efficacy/toxicity
surfaces, the gate threshold, a declared OBD reference, and the observation-noise SDs. Efficacy
is MAXIMISED; a dose is feasible iff ``tox(d) <= gd``. The ``sf`` and ``sg`` values
are the baseline SDs at ``kap=1``; pass ``kap`` to ``run_trial`` or ``sweep`` for a
common sensitivity multiplier without redefining the surface.
"""
import numpy as np

import dose_combination_bo as bo


@bo.surface("shifted_gaussian")
def shifted_gaussian(z):
    """A single-peak Gaussian efficacy bump whose optimum is pushed against a linear
    toxicity boundary, so the OBD binds the constraint. Stratum z shifts the peak."""
    peak = (0.4, 0.6) if z == 0 else (0.6, 0.4)

    def eff(a, b):
        return float(np.exp(-((a - peak[0]) ** 2 + (b - peak[1]) ** 2) / 0.08))

    def tox(a, b):
        return float(1.2 * a + 0.8 * b)

    gd = 1.0
    # declare a fixed fine-grid approximation to the constrained optimum
    g = np.linspace(0, 1, 80)
    G = np.array([(x, y) for x in g for y in g])
    fv = np.array([eff(*p) for p in G]); gv = np.array([tox(*p) for p in G])
    feas = gv <= gd
    i = np.where(feas)[0][fv[feas].argmax()] if feas.any() else int(gv.argmin())
    dopt = tuple(G[i]); fopt = float(fv[i])
    return dict(gd=gd, dopt=dopt, fopt=fopt, eff=eff, tox=tox, sf=0.15, sg=0.15)


if __name__ == "__main__":
    print("registered surfaces:", bo.list_surfaces())
    print("\ninstallation-scale comparison on the new surface (seeds=3, budget=12)...\n")

    res = bo.sweep(
        acquisitions=["cKG", "tmse", "entropy", "cEI"],
        surfaces=["shifted_gaussian"],
        seeds=3,
        gammas=[0.7],
        strata=[0],
        parallel=False,
        budget=12,
        verbose=True,
    )
    print("\n=== selection precision on shifted_gaussian (dose-units) ===")
    bo.leaderboard(res, metric="dose_units", sim="shifted_gaussian", lower_is_better=True)
