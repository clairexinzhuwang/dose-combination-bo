"""Dose-finding testbeds (data-generating surfaces).

A surface factory maps a stratum ``z in {0, 1}`` to a spec dict:

    gd    float                gate threshold g-dagger (feasible iff tox <= gd)
    dopt  (d1, d2)             supplied constrained-OBD reference
    fopt  float                efficacy at that reference (efficacy is MAXIMISED)
    eff   (d1, d2) -> float    true efficacy surface
    tox   (d1, d2) -> float    true toxicity surface
    sf    float                efficacy observation-noise SD
    sg    float                toxicity observation-noise SD

Four built-in surfaces reproduce the paper:

    "osa"      dual-agent obstructive-sleep-apnea polynomial (Willard et al. 2024;
               aroxybutynin + atomoxetine, the AD109 combination)   [aliases: none]
    "mariposa" an aggregate-informed synthetic calibration using published
               phase-2b MARIPOSA arm summaries
    "efftox"   a logistic (Thall-Cook EffTox-style) response geometry, interior
               efficacy peak pushed infeasible so the OBD binds the MTD    [alias: logistic]
    "gbump"    a Gaussian-bump surface with multiple local optima         [alias: gaussian_bump]

plus a parametric ``synth:...`` family used for the precision-reversal experiments.

The numeric definitions below preserve the study implementation. Built-in OSA uses a
fixed 200-by-200 approximation to the continuous constrained optimum; MARIPOSA,
EffTox-style logistic, and Gaussian-bump testbeds use fixed 80-by-80 approximations.
"""
import numpy as np

from .registry import surface, SURFACES

# --------------------------------------------------------------------------
# OSA polynomial (Willard, Golchi & Moodie 2024, arXiv:2404.11323;
# their Table 1 "OSA")
# --------------------------------------------------------------------------
_BETA = {0: np.array([-1.38, -4.08, -0.48, -4.23, 2.45, -7.51, -1.56]),
         1: np.array([1.05, -11.28, -8.32, -17.02, 8.17, 2.34, 4.61])}
_THETA = np.array([-0.59, 1.83, 2.26, -4.05, 1.79, 0.47, 2.91])
G_DAGGER = {0: 1.5, 1: 2.0}


def _basis(d1, d2):
    return np.array([np.ones_like(d1), d1, d2, d1 * d2, d1**2, d2**2, d1**2 * d2**2])


def f_osa(d1, d2, z=0):
    """OSA efficacy in the paper's convention (SMALLER is better = bigger AHI reduction)."""
    return np.tensordot(_BETA[z], _basis(d1, d2), axes=([0], [0]))


def g_osa(d1, d2, z=0):
    """OSA toxicity (log AE-burden); feasible iff g <= g_dagger[z]."""
    return np.tensordot(_THETA, _basis(d1, d2), axes=([0], [0]))


def true_obd_osa(z=0, grid=200):
    """Dense-grid approximation to the continuous constrained benchmark target.

    Willard et al. (2024), Table 1, report standardized panel OBDs
    (0.25, 0.75) for z=0 and (0.5, 0.75) for z=1.
    """
    xs = np.linspace(0, 1, grid); G1, G2 = np.meshgrid(xs, xs)
    F = f_osa(G1, G2, z); G = g_osa(G1, G2, z)
    Fm = np.where(G <= G_DAGGER[z], F, np.inf)  # minimize f
    i = np.unravel_index(Fm.argmin(), Fm.shape)
    return np.array([G1[i], G2[i]]), F[i]


# --------------------------------------------------------------------------
# Gaussian-bump surfaces (multiple local optima)
# --------------------------------------------------------------------------
def gb_eff(d1, d2, z):
    if z == 0:
        return (3.2 * np.exp(-((d1 - 0.3)**2 / 0.03 + (d2 - 0.7)**2 / 0.1)) + 1.0 * np.sin(2 * np.pi * d1) * np.cos(3 * np.pi * d2)
                + 1.4 * d1 * np.exp(-1.2 * d1**2) + 0.9 * d2 * np.exp(-0.6 * d2**2))
    return (2.0 * np.exp(-((d1 - 0.5)**2 + (d2 - 0.5)**2) / 0.05) + 1.2 * np.sin(3 * np.pi * d1) * np.sin(3 * np.pi * d2)
            + 1.0 * d1 * np.exp(-1.0 * d1**2) + 0.8 * d2 * np.exp(-0.9 * d2**2))


def gb_tox(d1, d2, z):
    if z == 0:
        return 3.0 * ((d1 - 0.5)**2 + (d2 - 0.5)**2) + 0.5 * np.sin(3 * np.pi * d1) * np.sin(3 * np.pi * d2) + 1.5 * (d1**2 + d2**2)
    return 4.0 * ((d1 - 0.5)**2 + (d2 - 0.5)**2) + 0.7 * np.sin(3 * np.pi * d1) * np.sin(3 * np.pi * d2) + 2.0 * (d1**2 + d2**2)


def _fine(n=80):
    g = np.linspace(0, 1, n); return np.array([(a, b) for a in g for b in g])


def gb_thresh(z):
    G = _fine(); g = np.array([gb_tox(x[0], x[1], z) for x in G]); return float(np.percentile(g, 50))


def gb_obd(z, gd):
    G = _fine(); f = np.array([gb_eff(x[0], x[1], z) for x in G]); g = np.array([gb_tox(x[0], x[1], z) for x in G])
    feas = g <= gd; i = (np.where(feas)[0][f[feas].argmax()]) if feas.any() else int(g.argmin()); return G[i], float(f[i])


def gb_std(z):
    G = _fine(60); return (np.array([gb_eff(x[0], x[1], z) for x in G]).std(),
                           np.array([gb_tox(x[0], x[1], z) for x in G]).std())


# --------------------------------------------------------------------------
# Verbatim spec builder (== gate_ab_smoke.sim_spec)
# --------------------------------------------------------------------------
def _builtin_spec(sim, z):
    if sim == 'osa':
        gd = G_DAGGER[z]; dopt, _ = true_obd_osa(z)
        fopt = -float(f_osa(np.array(dopt[0]), np.array(dopt[1]), z))   # efficacy (e=-f) at the OBD
        eff = lambda a, b: -float(f_osa(np.array(a), np.array(b), z))   # efficacy = -f, MAXIMIZE
        tox = lambda a, b: float(g_osa(np.array(a), np.array(b), z))
        sf, sg = 7.68, 1.29
    elif sim == 'mariposa':
        # Aggregate-informed synthetic calibration using published MARIPOSA arm summaries
        # (Schweitzer 2023): AD109 = aroxybutynin
        # (d1; 0-5 mg -> [0,1]) + atomoxetine (d2; 0-75 mg -> [0,1]). Efficacy = % AHI reduction
        # (maximize); toxicity = log adverse-event burden (antimuscarinic, steeper in aroxybutynin).
        eff = lambda a, b: 50.0 * (1 - np.exp(-(1.5 * a + 0.6 * b + 1.8 * a * b)))
        tox = lambda a, b: 0.5 + 1.4 * a + 0.5 * b + 0.6 * a * b
        gd = 2.0
        Gf = _fine(80); fv = np.array([eff(x[0], x[1]) for x in Gf]); gv = np.array([tox(x[0], x[1]) for x in Gf])
        feas = gv <= gd; i = (np.where(feas)[0][fv[feas].argmax()]) if feas.any() else int(gv.argmin())
        dopt = Gf[i]; fopt = float(fv[i])
        es = np.array([eff(x[0], x[1]) for x in _fine(60)]).std(); gs = np.array([tox(x[0], x[1]) for x in _fine(60)]).std()
        sf, sg = es / 0.55, gs / 0.72
    elif sim == 'efftox':
        # Distinct response geometry: Thall-Cook/EffTox-style logistic dose-response.
        # It is still observed through the same Gaussian-noise model used by every testbed.
        # Both channels are probability-shaped continuous geometries, observed through
        # Gaussian noise; no binary DLT or response outcomes are generated. The efficacy
        # surface has an interior peak beyond the g <= 0.30 region, so the constrained
        # numerical target lies on that boundary.
        A0, A1, A2, A12 = (-2.0, 3.0, 2.5, 1.0) if z == 0 else (-2.0, 2.0, 1.8, 0.5)  # z=0 steep, z=1 gentle
        sig = lambda u: 1.0 / (1.0 + np.exp(-u))
        tox = lambda a, b: float(sig(A0 + A1 * a + A2 * b + A12 * a * b))
        eff = lambda a, b: float(sig(-1.0 + 2.6 * a + 2.2 * b - 1.1 * a * a - 0.9 * b * b + 0.3 * a * b))
        gd = 0.30
        Gf = _fine(80); fv = np.array([eff(x[0], x[1]) for x in Gf]); gv = np.array([tox(x[0], x[1]) for x in Gf])
        feas = gv <= gd; i = (np.where(feas)[0][fv[feas].argmax()]) if feas.any() else int(gv.argmin())
        dopt = Gf[i]; fopt = float(fv[i])
        es = np.array([eff(x[0], x[1]) for x in _fine(60)]).std(); gs = np.array([tox(x[0], x[1]) for x in _fine(60)]).std()
        sf, sg = es / 0.55, gs / 0.72
    elif sim.startswith('synth:'):
        # Controlled 2-parameter family for the precision sign-flip experiment.
        # Toxicity g=a(d1+d2): boundary at d1+d2=t. Efficacy beta(d1+d2) - kappa*((d1-d2)/2)^2 rises
        # with total dose (optimum BINDS the boundary), tangential curvature kappa pins it at d1=d2=t/2.
        p = dict(kv.split('=') for kv in sim.split(':', 1)[1].split(';'))
        kap = float(p.get('k', 1.0)); a = float(p.get('a', 3.0)); t = float(p.get('t', 0.9)); beta = float(p.get('b', 1.0))
        crv = float(p.get('c', 0.0))   # toxicity-boundary curvature
        eff = lambda x, y: float(beta * (x + y) - kap * ((x - y) / 2.0)**2)
        tox = lambda x, y: float(a * (x + y) + crv * ((x - y) / 2.0)**2)
        gd = a * t; dopt = (t / 2.0, t / 2.0); fopt = float(beta * t)
        es = np.array([eff(x[0], x[1]) for x in _fine(60)]).std()
        sf = float(p['sf']) if 'sf' in p else es / 0.55
        sg = float(p['s']) if 's' in p else (np.array([tox(x[0], x[1]) for x in _fine(60)]).std()) / 0.72
    else:
        gd = gb_thresh(z); dopt, fopt = gb_obd(z, gd)
        eff = lambda a, b: float(gb_eff(a, b, z)); tox = lambda a, b: float(gb_tox(a, b, z))
        fstd, gstd = gb_std(z); sf, sg = fstd / 0.55, gstd / 0.72              # SNR-match to OSA
    # Internal truth must be self-consistent: reference efficacy is evaluated
    # at the supplied OBD reference, and that point is feasible. A mismatch corrupts every
    # selection-error and efficacy-regret result downstream, so fail immediately.
    eff_at_opt = float(eff(float(dopt[0]), float(dopt[1])))
    tox_at_opt = float(tox(float(dopt[0]), float(dopt[1])))
    if not np.isclose(float(fopt), eff_at_opt, rtol=1e-10, atol=1e-10):
        raise AssertionError(f"{sim} z={z}: fopt={fopt} != eff(dopt)={eff_at_opt}")
    if tox_at_opt > float(gd) + 1e-10:
        raise AssertionError(f"{sim} z={z}: dopt is infeasible ({tox_at_opt} > {gd})")
    return dict(gd=gd, dopt=dopt, fopt=fopt, eff=eff, tox=tox, sf=sf, sg=sg)


# --------------------------------------------------------------------------
# Register the four built-in surfaces (thin wrappers over _builtin_spec)
# --------------------------------------------------------------------------
@surface("osa")
def _osa(z):
    return _builtin_spec('osa', z)


@surface("mariposa")
def _mariposa(z):
    return _builtin_spec('mariposa', z)


@surface("efftox", aliases=("logistic",))
def _efftox(z):
    return _builtin_spec('efftox', z)


@surface("gbump", aliases=("gaussian_bump",))
def _gbump(z):
    return _builtin_spec('gbump', z)


_REQUIRED_KEYS = ("gd", "dopt", "fopt", "eff", "tox", "sf", "sg")


def _validate_spec(spec, sim):
    """Check a surface spec is well-formed (helps users who register their own)."""
    miss = [k for k in _REQUIRED_KEYS if k not in spec]
    if miss:
        raise KeyError(f"surface {sim!r} spec missing {miss}; required keys: {list(_REQUIRED_KEYS)}")
    dopt = np.asarray(spec["dopt"], float)
    if dopt.shape != (2,):
        raise ValueError(f"surface {sim!r} 'dopt' must be length-2 (d1, d2), got {spec['dopt']!r}")
    if not np.isfinite(dopt).all() or not ((0 <= dopt).all() and (dopt <= 1).all()):
        raise ValueError(f"surface {sim!r} 'dopt' must be finite and lie in [0, 1]^2")
    if not (callable(spec["eff"]) and callable(spec["tox"])):
        raise TypeError(f"surface {sim!r} 'eff'/'tox' must be callables (d1, d2) -> float")
    for name in ("gd", "fopt", "sf", "sg"):
        if not np.isscalar(spec[name]) or not np.isfinite(float(spec[name])):
            raise ValueError(f"surface {sim!r} {name!r} must be a finite scalar")
    if float(spec["sf"]) <= 0 or float(spec["sg"]) <= 0:
        raise ValueError(f"surface {sim!r} 'sf' and 'sg' must be positive")
    try:
        eff_at_opt = float(spec["eff"](*dopt))
        tox_at_opt = float(spec["tox"](*dopt))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"surface {sim!r} 'eff'/'tox' must return scalar values") from exc
    if not np.isfinite(eff_at_opt) or not np.isfinite(tox_at_opt):
        raise ValueError(f"surface {sim!r} 'eff'/'tox' must be finite at dopt")
    if not np.isclose(float(spec["fopt"]), eff_at_opt, rtol=1e-10, atol=1e-10):
        raise ValueError(
            f"surface {sim!r} fopt={spec['fopt']} != eff(dopt)={eff_at_opt}; "
            "efficacy must use the maximisation convention"
        )
    if tox_at_opt > float(spec["gd"]) + 1e-10:
        raise ValueError(f"surface {sim!r} dopt is infeasible ({tox_at_opt} > {spec['gd']})")
    _check_dopt_is_optimal(spec, sim, dopt, eff_at_opt)
    return spec


#: Resolution of the dense scan used to catch a mis-declared optimum.
_OPT_SCAN_N = 101

#: A declared OBD is rejected when a feasible point beats it by more than this
#: fraction of the feasible efficacy RANGE. The tolerance is relative because the
#: built-in OBDs are themselves finite grid searches (80x80 for mariposa, efftox
#: and gbump; 200x200 for osa), so any independent scan will find sub-resolution
#: improvements -- measured at 0.18% of range for mariposa and 0.03% for efftox.
#: Those are discretization, not error. A genuinely mis-declared OBD is off by a
#: large fraction of the range, so 1% separates the two by two orders of
#: magnitude without being tuned to any particular surface.
_OPT_REL_TOL = 0.01

def _check_dopt_is_optimal(spec, sim, dopt, eff_at_opt):
    """Reject a supplied reference that is materially suboptimal on a validation scan.

    Self-consistency (``fopt == eff(dopt)``, ``dopt`` feasible) does not catch the
    failure that matters: a *feasible but materially suboptimal* reference silently
    corrupts every selection-error and efficacy-regret number downstream, because
    those are measured as distance and shortfall relative to ``dopt``. Nothing
    downstream can detect it, so it is checked here, once, at registration.

    The scan is coarse by design -- it is a mistake detector, not an exact optimizer, and
    a surface whose optimum hides between grid points at this resolution is too
    sharp for the benchmark's own 5x5 dose grid to resolve anyway.
    """
    lin = np.linspace(0.0, 1.0, _OPT_SCAN_N)
    d1, d2 = np.meshgrid(lin, lin, indexing="ij")
    try:
        eff = np.vectorize(lambda a, b: float(spec["eff"](a, b)))(d1, d2)
        tox = np.vectorize(lambda a, b: float(spec["tox"](a, b)))(d1, d2)
    except (TypeError, ValueError):
        return  # not vectorizable point-by-point; the self-consistency checks stand
    feasible = np.isfinite(eff) & np.isfinite(tox) & (tox <= float(spec["gd"]) + 1e-10)
    if not feasible.any():
        raise ValueError(f"surface {sim!r} has no feasible dose on a {_OPT_SCAN_N}x{_OPT_SCAN_N} scan")
    best = float(eff[feasible].max())
    span = float(eff[feasible].max() - eff[feasible].min())
    tol = max(_OPT_REL_TOL * span, 1e-9)
    if best > eff_at_opt + tol:
        k = np.argmax(np.where(feasible, eff, -np.inf))
        b1, b2 = d1.ravel()[k], d2.ravel()[k]
        raise ValueError(
            f"surface {sim!r} 'dopt' {tuple(np.round(dopt, 4))} is feasible but materially suboptimal on the validation scan: "
            f"({b1:.4f}, {b2:.4f}) is feasible with efficacy {best:.6g} > {eff_at_opt:.6g}. "
            "Selection error and efficacy regret are measured against 'dopt', so a "
            "suboptimal 'dopt' silently corrupts every downstream number."
        )


def resolve_surface(sim, z):
    """Return the (validated) spec dict for surface ``sim`` at stratum ``z``.

    Resolution order: parametric ``synth:...`` family, then the registry
    (built-in + user-registered surfaces). An unknown name raises, rather than
    silently falling back to a default surface -- a silent fallback would mislabel
    results and, under Ray, quietly corrupt a benchmark.
    """
    if isinstance(sim, str) and sim.startswith('synth:'):
        return _validate_spec(_builtin_spec(sim, z), sim)
    if sim in SURFACES:
        return _validate_spec(SURFACES[sim](z), sim)
    raise KeyError(f"unknown surface {sim!r}; registered: {sorted(set(SURFACES))} (or use a 'synth:...' spec)")
