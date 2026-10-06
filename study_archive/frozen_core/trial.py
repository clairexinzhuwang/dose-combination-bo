"""Single-trial harness.

``run_trial`` runs one posterior-feasibility-gated dose-optimization simulation with a chosen acquisition
and returns a dict of operating characteristics. It is a faithful refactor of the
paper's ``gate_ab_smoke.run_trial``: the same warmup, GP fits, cohort loop,
recommendation, and metrics, with acquisitions looked up in the plug-in registry.
The corrected default uses the documented most-feasible-dose empty-gate fallback
and isolates acquisition randomness from observation randomness. The historical
pre-correction bit-for-bit path is available only with ``empty_gate="ungated"`` (see
``tests/test_identity.py``).
"""
import numpy as np
import torch
import scipy.stats as ss

from .gp import fit_gp, post_latent, post_predictive
from .metrics import recommend_obd, dose_units
from .surfaces import resolve_surface
from .context import Context
from .registry import get_acquisition
from .protocol import (
    ProtocolScaffold,
    cohort_region_q,
    eligible_mask,
    initialization_size,
    region_mask,
    validate_protocol_config,
)

torch.set_default_dtype(torch.double)


def grid(n=5):
    g = np.linspace(0, 1, n)
    return np.array([(a, b) for a in g for b in g])


def _validate_trial_args(seed, z, gamma, mode, noise, budget, warmup, r_k, grid_n,
                         empty_gate, kap=1.0,
                         protocol_scaffold: ProtocolScaffold = 'lhs_fixed',
                         region_step=0.25, empty_gate_stop_after=3,
                         exclude_repeats_during_expansion=True):
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if z not in {0, 1}:
        raise ValueError("z must be 0 or 1")
    if not np.isscalar(gamma) or not np.isfinite(float(gamma)) or not 0 < float(gamma) < 1:
        raise ValueError("gamma must be finite and strictly between 0 and 1")
    if mode not in {"latent", "predictive"}:
        raise ValueError("mode must be 'latent' or 'predictive'")
    if noise not in {"fixed", "learned"}:
        raise ValueError("noise must be 'fixed' or 'learned'")
    if not np.isscalar(kap) or not np.isfinite(float(kap)) or float(kap) <= 0:
        raise ValueError("kap must be a finite positive observation-noise multiplier")
    for name, value, lower in (("budget", budget, 1), ("warmup", warmup, 1),
                               ("r_k", r_k, 1), ("grid_n", grid_n, 2)):
        if not isinstance(value, (int, np.integer)) or int(value) < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if empty_gate not in {"pf", "ungated"}:
        raise ValueError("empty_gate must be 'pf' or 'ungated'")
    validate_protocol_config(
        protocol_scaffold,
        region_step,
        empty_gate_stop_after,
        exclude_repeats_during_expansion,
        budget=budget,
        warmup=warmup,
        r_k=r_k,
        grid_n=grid_n,
        empty_gate=empty_gate,
    )


def run_trial(policy, seed, z, gamma, sim, mode='latent', noise='fixed', traj=False,
              budget=40, warmup=4, r_k=2, grid_n=5, empty_gate='pf', kap=1.0,
              protocol_scaffold: ProtocolScaffold = 'lhs_fixed', region_step=0.25,
              empty_gate_stop_after=3, exclude_repeats_during_expansion=True):
    """Run one simulated trial record.

    Parameters
    ----------
    policy : str          registered acquisition name (e.g. "cEI", "cKG", "cEI-tMSE", or your own)
    seed : int            RNG seed (controls warmup design and observation noise;
                          acquisition randomness uses an isolated derived stream)
    z : int               stratum (0 or 1)
    gamma : float         gate strictness tau (implemented as strict z > Phi^{-1}(tau))
    sim : str             surface name ("osa", "mariposa", "efftox", "gbump", "synth:...")
    mode : str            gate mode, "latent" (default) or "predictive"
    noise : str           noise-fitting mode: "fixed" pins the data-generating variance;
                          "learned" estimates it by the Willard empirical-Bayes MLE
    kap : float           positive multiplier applied to both surface observation-noise SDs;
                          1.0 uses the surface calibration
    traj : bool           also record the per-cohort recommendation trajectory
    budget, warmup, r_k, grid_n : simulation-design settings (paper defaults)
    protocol_scaffold : str  ``"lhs_fixed"`` (default) or ``"start_low_expansion"``
    region_step : float      start-low expansion increment in d1+d2 per cohort
    empty_gate_stop_after : int  consecutive empty eligible gates before stopping
    exclude_repeats_during_expansion : bool  avoid revisiting candidates before full expansion

    Returns
    -------
    dict of operating characteristics (dose_units, rpsel, rec_unsafe, toxic,
    gate calibration, per-gamma recommendation readout, ...).
    """
    _validate_trial_args(seed, z, gamma, mode, noise, budget, warmup, r_k, grid_n,
                         empty_gate, kap, protocol_scaffold, region_step,
                         empty_gate_stop_after, exclude_repeats_during_expansion)
    torch.set_default_dtype(torch.double)              # ensure double in Ray workers
    acq_fn = get_acquisition(policy)                    # fail before any simulation work
    S = resolve_surface(sim, z); gd = S['gd']; dopt = S['dopt']
    kap = float(kap)
    sf, sg = kap * float(S['sf']), kap * float(S['sg'])
    nzf = sf**2 if noise == 'fixed' else None          # 'fixed'=true noise pinned; 'learned'=EB MLE
    nzg = sg**2 if noise == 'fixed' else None
    tr_n, tr_du, tr_rp, tr_tx = [], [], [], []         # per-cohort trajectory
    rng = np.random.default_rng(seed)
    # A policy may consume arbitrarily many random draws without changing warmup or
    # observation noise. This makes seed-paired policy comparisons fair.
    # Keep this derivation identical to the frozen paper harness so randomized
    # acquisitions (notably random-safe) reproduce its seed-labelled trials.
    policy_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 823451]))
    Xset = torch.tensor(grid(grid_n)); X, Yf, Yg = [], [], []
    spacing = 1.0 / (grid_n - 1)
    region_step = float(region_step)
    empty_gate_stop_after = int(empty_gate_stop_after)
    exclude_repeats_during_expansion = bool(exclude_repeats_during_expansion)
    init_size = initialization_size(protocol_scaffold, warmup, r_k)

    def adm(d1, d2):
        X.append([d1, d2]); Yf.append(S['eff'](d1, d2) + rng.normal(0, sf)); Yg.append(S['tox'](d1, d2) + rng.normal(0, sg))

    if protocol_scaffold == 'lhs_fixed':
        # Keep this initialization byte-for-byte equivalent to the frozen harness.
        cut = np.linspace(0, 1, warmup + 1)
        for i in range(warmup):
            adm(float(cut[i] + rng.uniform() * (cut[1] - cut[0])), float(rng.uniform()))
    else:
        for _ in range(r_k):
            adm(0.0, 0.0)

    n = init_size
    empty_gate_streak = 0
    n_empty_gate_events = 0
    stop_reason = None
    region_q_at_stop = None
    while n < budget:
        if protocol_scaffold == 'start_low_expansion':
            region_q = cohort_region_q(n, r_k)
            in_region = region_mask(Xset, region_q, region_step)
            active_Xset = Xset[torch.as_tensor(in_region, dtype=torch.bool)]
            eligible_full = eligible_mask(
                Xset,
                X,
                region_q,
                region_step,
                exclude_repeats_during_expansion,
            )
            candidate_mask = eligible_full[in_region]
        else:
            region_q = None
            active_Xset = Xset
            candidate_mask = None

        Xt = torch.tensor(np.array(X))
        me, le = fit_gp(Xt, torch.tensor(Yf), fixed_noise=nzf); mt, lt = fit_gp(Xt, torch.tensor(Yg), fixed_noise=nzg)
        if traj:                                          # log current recommendation's criteria at this cohort
            rt, _ = recommend_obd(active_Xset, me, le, mt, lt, gd, gamma=gamma, gate_mode=mode)
            mfr, vfr = post_latent(me, torch.tensor(np.array([rt])))
            tr_n.append(int(n)); tr_du.append(float(dose_units(rt, dopt, spacing=spacing)))
            tr_rp.append(float(np.sqrt((float(mfr.reshape(-1)[0]) - S['fopt'])**2 + float(vfr.reshape(-1)[0]))))
            tr_tx.append(int(sum(1 for x in X if S['tox'](x[0], x[1]) > gd)))
        ctx = Context(Xset=active_Xset, me=me, le=le, mt=mt, lt=lt, g_dagger=gd, gamma=gamma,
                      gate_mode=mode, r_k=r_k, step=(n - init_size) // r_k, rng=policy_rng,
                      empty_gate=empty_gate, candidate_mask=candidate_mask)
        if ctx.gate_empty:
            n_empty_gate_events += 1
            if protocol_scaffold == 'start_low_expansion':
                empty_gate_streak += 1
                if empty_gate_streak >= empty_gate_stop_after:
                    stop_reason = 'NO_FEASIBLE_DOSE'
                    region_q_at_stop = int(region_q)
                    break
        elif protocol_scaffold == 'start_low_expansion':
            empty_gate_streak = 0

        vals = np.asarray(acq_fn(ctx), dtype=float)
        if vals.shape != (active_Xset.shape[0],):
            raise ValueError(f"acquisition {policy!r} must return {active_Xset.shape[0]} scores "
                             f"(one per grid dose), got shape {vals.shape}")
        if np.isnan(vals).any() or np.isposinf(vals).any():
            raise ValueError(f"acquisition {policy!r} returned NaN or +inf scores")
        vals = np.where(ctx.candidate_mask, vals, -np.inf)
        if protocol_scaffold == 'start_low_expansion' and ctx.gate_empty:
            # The first empty events use the prespecified most-feasible eligible
            # fallback. The stopping event above never reaches allocation.
            d = active_Xset[ctx.most_feasible_index]
        else:
            d = (active_Xset[ctx.most_feasible_index] if not np.isfinite(vals).any()
                 else active_Xset[int(vals.argmax())])
        for _ in range(r_k):
            adm(float(d[0]), float(d[1]))
        n += r_k

    Xt = torch.tensor(np.array(X))
    me, le = fit_gp(Xt, torch.tensor(Yf), fixed_noise=nzf); mt, lt = fit_gp(Xt, torch.tensor(Yg), fixed_noise=nzg)
    recommendation_made = stop_reason is None
    if recommendation_made:
        rec, _ = recommend_obd(Xset, me, le, mt, lt, gd, gamma=gamma, gate_mode=mode)
        du = float(dose_units(rec, dopt, spacing=spacing)); rec_tox = float(S['tox'](rec[0], rec[1]))
        muf_rec, varf_rec = post_latent(me, torch.tensor(np.array([rec])))   # RPSEL = posterior efficacy RMSE at rec
        rpsel = float(np.sqrt((float(muf_rec.reshape(-1)[0]) - S['fopt'])**2 + float(varf_rec.reshape(-1)[0])))
        rec_unsafe = int(rec_tox > gd)
        rec_d1, rec_d2 = float(rec[0]), float(rec[1])
        rec_true_eff = float(S['eff'](rec[0], rec[1]))
        rec_true_tox = rec_tox
    else:
        du = rpsel = rec_unsafe = None
        rec_d1 = rec_d2 = rec_true_eff = rec_true_tox = None
    toxic = int(sum(1 for x in X if S['tox'](x[0], x[1]) > gd))
    mu_gf, vgf = (post_predictive(mt, lt, Xset) if mode == 'predictive' else post_latent(mt, Xset))
    z_f = (gd - mu_gf.cpu().numpy()) / vgf.clamp_min(1e-12).sqrt().cpu().numpy()
    pf_f = ss.norm.cdf(z_f)
    passed = z_f > float(ss.norm.ppf(float(gamma)))
    g_true = np.array([S['tox'](float(x[0]), float(x[1])) for x in Xset.numpy()])
    npass = int(passed.sum()); nsafe = int((passed & (g_true <= gd)).sum())
    # recommendation read out across a gamma_rec grid (decouples sampling-gamma from recommend-gamma)
    recs = []
    if recommendation_made:
        for grx in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            rcx, _ = recommend_obd(Xset, me, le, mt, lt, gd, gamma=grx, gate_mode=mode)
            recs.append(dict(grec=float(grx), du=float(dose_units(rcx, dopt, spacing=spacing)),
                             unsafe=int(float(S['tox'](rcx[0], rcx[1])) > gd), eff_reg=float(S['fopt'] - float(S['eff'](rcx[0], rcx[1])))))
    iobd = int(np.argmin(np.linalg.norm(Xset.numpy() - np.asarray(dopt, float), axis=1)))   # panel dose nearest stored OBD reference
    mu_go, vgo = post_latent(mt, Xset[iobd:iobd + 1]); sdo = float(vgo.clamp_min(1e-12).sqrt().reshape(-1)[0])
    obd_pf = float(ss.norm.cdf((gd - float(mu_go.reshape(-1)[0])) / sdo))   # feasibility probability at nearest panel reference
    above = [int(S['tox'](x[0], x[1]) > gd) for x in X]
    n_unique_doses = len({(float(x[0]), float(x[1])) for x in X})
    return dict(policy=policy, seed=int(seed), sim=sim, stratum=int(z), gamma=float(gamma), mode=mode,
                budget=int(budget), r_k=int(r_k), grid_n=int(grid_n), noise=noise, kap=kap,
                eff_noise_sd=sf, tox_noise_sd=sg, warmup=int(warmup),
                empty_gate=empty_gate,
                protocol_scaffold=protocol_scaffold, region_step=region_step,
                empty_gate_stop_after=empty_gate_stop_after,
                exclude_repeats_during_expansion=exclude_repeats_during_expansion,
                stop_reason=stop_reason, recommendation_made=recommendation_made,
                n_enrolled=int(n), n_unique_doses=int(n_unique_doses),
                n_empty_gate_events=int(n_empty_gate_events),
                initialization_patients_above=int(sum(above[:init_size])),
                post_initialization_patients_above=int(sum(above[init_size:])),
                initialization_size=int(init_size), region_q_at_stop=region_q_at_stop,
                allocation_history=[[float(x[0]), float(x[1])] for x in X],
                dose_units=du, rpsel=rpsel, rec_unsafe=rec_unsafe, toxic=toxic, n_gate_pass=npass, n_gate_pass_safe=nsafe,
                rec_d1=rec_d1, rec_d2=rec_d2, rec_true_eff=rec_true_eff, rec_true_tox=rec_true_tox,
                recs=recs, obd_pf=obd_pf, obd_sdg=sdo,
                traj=(dict(n=tr_n, du=tr_du, rpsel=tr_rp, toxic=tr_tx) if traj else None))
