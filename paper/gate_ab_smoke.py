"""Historical pre-correction gate-mode A/B analysis runner.

This file is retained for identity checks and legacy figure helpers.  Its empty-gate
path can revert to ungated scoring, so it is not the current public trial harness and
must not be used to support current numerical claims.  Use ``dose_combination_bo.run_trial``
with ``empty_gate="pf"`` for the documented most-feasible fallback.

OUR acquisition (cei, ckg_one_step_gated -- verified == botorch to machine precision), UNCHANGED.
FIXED noise (true sim variance pinned in fit_gp). gate_mode in {latent, predictive} threaded
CONSISTENTLY through: safe-set + cEI feasibility + cKG gate + recommendation.
  latent     = our current choice (sigma_g = latent posterior SD, shrinks with data)
  predictive = James gp.predict (sigma_g^2 = latent var + obs noise; floored at the noise level)
Both simulators (OSA + Gaussian-bump), budget 40 (iterations UNCHANGED), 100 seeds. Ray, resume-safe.
"""
import argparse, json, os, time
import numpy as np, torch
import scipy.stats as ss
from ckg_core import fit_gp, _post, _post_latent, _joint, cei, ckg_one_step_gated, recommend_obd, dose_units
import willard_osa as OSA
torch.set_default_dtype(torch.double)

# ---- Gaussian-bump surfaces (verbatim from run_demo_transfer.py) ----
def gb_eff(d1, d2, z):
    if z == 0:
        return (3.2*np.exp(-((d1-0.3)**2/0.03+(d2-0.7)**2/0.1))+1.0*np.sin(2*np.pi*d1)*np.cos(3*np.pi*d2)
                + 1.4*d1*np.exp(-1.2*d1**2)+0.9*d2*np.exp(-0.6*d2**2))
    return (2.0*np.exp(-((d1-0.5)**2+(d2-0.5)**2)/0.05)+1.2*np.sin(3*np.pi*d1)*np.sin(3*np.pi*d2)
            + 1.0*d1*np.exp(-1.0*d1**2)+0.8*d2*np.exp(-0.9*d2**2))
def gb_tox(d1, d2, z):
    if z == 0:
        return 3.0*((d1-0.5)**2+(d2-0.5)**2)+0.5*np.sin(3*np.pi*d1)*np.sin(3*np.pi*d2)+1.5*(d1**2+d2**2)
    return 4.0*((d1-0.5)**2+(d2-0.5)**2)+0.7*np.sin(3*np.pi*d1)*np.sin(3*np.pi*d2)+2.0*(d1**2+d2**2)
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

def sim_spec(sim, z):
    if sim == 'osa':
        gd = OSA.G_DAGGER[z]; dopt, _ = OSA.true_obd_osa(z)
        fopt = -float(OSA.f_osa(np.array(dopt[0]), np.array(dopt[1]), z))   # efficacy (e=-f) at the OBD, for RPSEL
        eff = lambda a, b: -float(OSA.f_osa(np.array(a), np.array(b), z))   # efficacy = -f, MAXIMIZE
        tox = lambda a, b:  float(OSA.g_osa(np.array(a), np.array(b), z))
        sf, sg = 7.68, 1.29
    elif sim == 'mariposa':
        # Aggregate-informed synthetic calibration using published MARIPOSA arm summaries
        # (Schweitzer 2023): AD109 = aroxybutynin
        # (d1; 0-5 mg -> [0,1]) + atomoxetine (d2; 0-75 mg -> [0,1]). Efficacy = % AHI reduction
        # (maximize); toxicity = log adverse-event burden (antimuscarinic, steeper in aroxybutynin).
        # Anchored so: atomoxetine monotherapy ~22%, AD109 2.5/75 ~45%, and 5/75 ~49%
        # (plateau). The assumed synthetic threshold places its numerical target near
        # (0.5,1.0), the coordinates of the published 2.5/75 mg reference arm; this is
        # not patient-level validation or an estimate of a clinical OBD.
        eff = lambda a, b: 50.0*(1-np.exp(-(1.5*a + 0.6*b + 1.8*a*b)))
        tox = lambda a, b: 0.5 + 1.4*a + 0.5*b + 0.6*a*b
        gd = 2.0
        Gf = _fine(80); fv = np.array([eff(x[0], x[1]) for x in Gf]); gv = np.array([tox(x[0], x[1]) for x in Gf])
        feas = gv <= gd; i = (np.where(feas)[0][fv[feas].argmax()]) if feas.any() else int(gv.argmin())
        dopt = Gf[i]; fopt = float(fv[i])
        es = np.array([eff(x[0], x[1]) for x in _fine(60)]).std(); gs = np.array([tox(x[0], x[1]) for x in _fine(60)]).std()
        sf, sg = es/0.55, gs/0.72
    elif sim == 'efftox':
        # Distinct Thall-Cook/EffTox-inspired logistic geometry (separate from the OSA
        # polynomial and Gaussian bump), but observed with the same Gaussian response
        # model used elsewhere. Both channels are probability-shaped continuous scores;
        # no binary DLT or response outcomes are generated. The efficacy peak lies beyond
        # the g <= 0.30 region, so the constrained numerical target binds that boundary.
        A0, A1, A2, A12 = (-2.0, 3.0, 2.5, 1.0) if z == 0 else (-2.0, 2.0, 1.8, 0.5)  # z=0 steep, z=1 gentle boundary
        sig = lambda u: 1.0/(1.0 + np.exp(-u))
        tox = lambda a, b: float(sig(A0 + A1*a + A2*b + A12*a*b))
        eff = lambda a, b: float(sig(-1.0 + 2.6*a + 2.2*b - 1.1*a*a - 0.9*b*b + 0.3*a*b))
        gd = 0.30
        Gf = _fine(80); fv = np.array([eff(x[0], x[1]) for x in Gf]); gv = np.array([tox(x[0], x[1]) for x in Gf])
        feas = gv <= gd; i = (np.where(feas)[0][fv[feas].argmax()]) if feas.any() else int(gv.argmin())
        dopt = Gf[i]; fopt = float(fv[i])
        es = np.array([eff(x[0], x[1]) for x in _fine(60)]).std(); gs = np.array([tox(x[0], x[1]) for x in _fine(60)]).std()
        sf, sg = es/0.55, gs/0.72
    elif sim.startswith('synth:'):
        # Controlled 2-parameter family for the precision sign-flip experiment (Move #3).
        # Toxicity g=a(d1+d2): boundary at d1+d2=t (feasible size set by t; |grad g|=a*sqrt2).
        # Efficacy beta(d1+d2) - kappa*((d1-d2)/2)^2 rises with total dose (the optimum BINDS the
        # boundary) and has tangential curvature kappa pinning the optimum at d1=d2=t/2. Knobs:
        #   k = kappa (tangential efficacy curvature; sharpness of the optimum ALONG the boundary)
        #   s = sigma_g (toxicity noise => boundary identifiability a*sqrt2/s; lower s = sharper boundary)
        # Optional: a (tox gradient, def 3), t (feasible size, def 0.9 -> OBD off-grid), b=beta (def 1), sf override.
        p = dict(kv.split('=') for kv in sim.split(':', 1)[1].split(';'))
        kap = float(p.get('k', 1.0)); a = float(p.get('a', 3.0)); t = float(p.get('t', 0.9)); beta = float(p.get('b', 1.0))
        crv = float(p.get('c', 0.0))   # toxicity-boundary curvature: tox tapers the feasible region to a tip at s=0,
        #                                so with k=0 the OBD's tangential position is set by the CONSTRAINT (cKG's domain)
        eff = lambda x, y: float(beta*(x+y) - kap*((x-y)/2.0)**2)
        tox = lambda x, y: float(a*(x+y) + crv*((x-y)/2.0)**2)
        gd = a*t; dopt = (t/2.0, t/2.0); fopt = float(beta*t)
        es = np.array([eff(x[0], x[1]) for x in _fine(60)]).std()
        sf = float(p['sf']) if 'sf' in p else es/0.55
        sg = float(p['s']) if 's' in p else (np.array([tox(x[0], x[1]) for x in _fine(60)]).std())/0.72
    else:
        gd = gb_thresh(z); dopt, fopt = gb_obd(z, gd)
        eff = lambda a, b: float(gb_eff(a, b, z)); tox = lambda a, b: float(gb_tox(a, b, z))
        fstd, gstd = gb_std(z); sf, sg = fstd/0.55, gstd/0.72              # SNR-match to OSA
    return dict(gd=gd, dopt=dopt, fopt=fopt, eff=eff, tox=tox, sf=sf, sg=sg)

def grid(n=5):
    g = np.linspace(0, 1, n); return np.array([(a, b) for a in g for b in g])

def gate(mt, lt, X, gd, mode):                       # feasibility pf under gate_mode
    mu, var = (_post(mt, lt, X) if mode == 'predictive' else _post_latent(mt, X))
    return ss.norm.cdf((gd - mu.cpu().numpy())/var.clamp_min(1e-12).sqrt().cpu().numpy())

def _gn(model, x, h=0.05):                            # ||grad of latent posterior mean|| at x (central diff)
    x = np.asarray(x, float)
    pts = torch.tensor(np.array([[x[0]+h, x[1]], [x[0]-h, x[1]], [x[0], x[1]+h], [x[0], x[1]-h]]))
    m, _ = _post_latent(model, pts); m = m.cpu().numpy()
    return float(np.hypot((m[0]-m[1])/(2*h), (m[2]-m[3])/(2*h)))

def _hb(p):                                           # Bernoulli entropy (feasibility uncertainty)
    p = np.clip(np.asarray(p), 1e-12, 1-1e-12); return -(p*np.log(p) + (1-p)*np.log(1-p))
_QX, _QW = np.polynomial.hermite_e.hermegauss(9); _QW = _QW/np.sqrt(2*np.pi)   # E_{N(0,1)}[f]=sum _QW f(_QX)

def qbig_value(d, Xset, m_tox, l_tox, g_dagger, gate_mode, r_k):
    """One-step boundary information gain: H(feasibility) - E_y[H(feasibility | obs at d)], summed over grid.
    Exact realization of the alpha_BIG channel; straddle is its leading-order point approximation."""
    mu_g, var_g, covg, s2g_full = _joint(m_tox, l_tox, Xset, d)
    ng = l_tox.noise.item(); s2g = (s2g_full - ng) + ng/r_k        # cohort fantasy variance at d
    mu_g = mu_g.cpu().numpy(); var_g = var_g.cpu().numpy(); covg = covg.cpu().numpy()
    obs = ng if gate_mode == 'predictive' else 0.0
    sd_now = np.sqrt(var_g + obs)
    resid = np.maximum(var_g - covg**2/s2g, 1e-12); sd_post = np.sqrt(resid + obs)   # deterministic post-obs SD
    H_now = _hb(ss.norm.cdf((g_dagger - mu_g)/sd_now))
    zeta0 = (g_dagger - mu_g)/sd_post; tau = (np.abs(covg)/np.sqrt(s2g))/sd_post     # zeta+ ~ N(zeta0, tau^2)
    EH = np.zeros_like(zeta0)
    for xq, wq in zip(_QX, _QW): EH += wq*_hb(ss.norm.cdf(zeta0 + tau*xq))
    return float(np.sum(H_now - EH))

def run_trial(policy, seed, z, gamma, sim, mode, noise='fixed', traj=False, budget=40, warmup=4, r_k=2, grid_n=5):
    torch.set_default_dtype(torch.double)              # ensure double in Ray workers
    S = sim_spec(sim, z); gd = S['gd']; dopt = S['dopt']; sf, sg = S['sf'], S['sg']
    nzf = sf**2 if noise == 'fixed' else None          # 'fixed'=true noise pinned; 'learned'=our sample-var-init EB fit
    nzg = sg**2 if noise == 'fixed' else None
    tr_n, tr_du, tr_rp, tr_tx = [], [], [], []         # per-cohort recommendation and criterion trajectory
    rng = np.random.default_rng(seed); Xset = torch.tensor(grid(grid_n)); X, Yf, Yg = [], [], []
    def adm(d1, d2):
        X.append([d1, d2]); Yf.append(S['eff'](d1, d2)+rng.normal(0, sf)); Yg.append(S['tox'](d1, d2)+rng.normal(0, sg))
    cut = np.linspace(0, 1, warmup+1)
    for i in range(warmup): adm(float(cut[i]+rng.uniform()*(cut[1]-cut[0])), float(rng.uniform()))
    n = warmup
    while n < budget:
        Xt = torch.tensor(np.array(X))
        me, le = fit_gp(Xt, torch.tensor(Yf), fixed_noise=nzf); mt, lt = fit_gp(Xt, torch.tensor(Yg), fixed_noise=nzg)
        if traj:                                          # log current recommendation's criteria at this cohort
            rt, _ = recommend_obd(Xset, me, le, mt, lt, gd, gamma=gamma, gate_mode=mode)
            mfr, vfr = _post_latent(me, torch.tensor(np.array([rt])))
            tr_n.append(int(n)); tr_du.append(float(dose_units(rt, dopt)))
            tr_rp.append(float(np.sqrt((float(mfr.reshape(-1)[0])-S['fopt'])**2+float(vfr.reshape(-1)[0]))))
            tr_tx.append(int(sum(1 for x in X if S['tox'](x[0], x[1]) > gd)))
        mu_g_t, vg_t = (_post(mt, lt, Xset) if mode == 'predictive' else _post_latent(mt, Xset))
        mu_g = mu_g_t.cpu().numpy(); sd_g = vg_t.clamp_min(1e-12).sqrt().cpu().numpy()
        pf = ss.norm.cdf((gd - mu_g)/sd_g); safe = pf > gamma
        strad = sd_g*np.exp(-0.5*((gd-mu_g)/sd_g)**2)         # straddle / boundary score (leading-order BIG)
        def restrict(v):
            m = np.where(safe, v, -np.inf); return m if np.isfinite(m).any() else v
        def cei_vals():
            muf, _ = _post_latent(me, Xset); fstar = muf.numpy()[safe].max() if safe.any() else muf.numpy()[pf.argmax()]
            return np.array([cei(Xset[i], Xset, me, le, mt, lt, gd, fstar, gate_mode=mode) for i in range(Xset.shape[0])])
        if policy == 'cEI':
            vals = restrict(cei_vals())
        elif policy == 'cKG1fix':
            vals = restrict(np.array([ckg_one_step_gated(Xset[i], Xset, me, le, mt, lt, gd, gamma, r_k=r_k, gate_mode=mode) for i in range(Xset.shape[0])]))
        elif policy == 'straddle':                            # pure gated straddle (boundary only)
            vals = restrict(strad)
        elif policy == 'ustrad':                              # ungated all-boundary sampler (straddle, gate ignored)
            vals = strad
        elif policy == 'SPW':                                 # equal-weight, range-normalized cEI + boundary score
            cv = cei_vals()
            def _nrm(v):
                vs = v[safe] if safe.any() else v
                lo, hi = float(np.min(vs)), float(np.max(vs))
                return (v - lo)/(hi - lo) if hi > lo else np.zeros_like(v)
            vals = restrict(_nrm(cv) + _nrm(strad))
        elif policy == 'qBIG':                                # boundary information gain: exact alpha_BIG channel
            vals = restrict(np.array([qbig_value(Xset[i], Xset, mt, lt, gd, mode, r_k) for i in range(Xset.shape[0])]))
        elif policy == 'random':                              # random-safe FLOOR: uniform over the gate-passed set
            si = np.where(safe)[0]
            pick = int(si[rng.integers(len(si))]) if len(si) else int(pf.argmax())
            vals = np.full(Xset.shape[0], -np.inf); vals[pick] = 0.0
        elif policy == 'sKG3':                                # safe-projected cKG: info-target / administration DECOUPLING
            a = np.array([ckg_one_step_gated(Xset[i], Xset, me, le, mt, lt, gd, gamma, r_k=r_k, gate_mode=mode) for i in range(Xset.shape[0])])
            istar = int(np.argmax(a))                         # WHERE info is most valuable (cKG's pick; may be unsafe)
            with torch.no_grad():
                Kp = mt(Xset).covariance_matrix.detach().cpu().numpy()   # latent posterior cov of toxicity over grid
            ng = lt.noise.item()
            vr = Kp[istar, :]**2 / (np.clip(np.diag(Kp), 1e-12, None) + ng)   # exact var-reduction at istar from each obs
            vals = restrict(vr)                               # administer the SAFE dose best resolving istar via the kernel
        else:  # GBE: alternating gated cEI / straddle, lookahead-free
            vals = restrict(cei_vals()) if ((n-warmup)//r_k) % 2 == 0 else restrict(strad)
        d = Xset[pf.argmax()] if not np.isfinite(vals).any() else Xset[int(vals.argmax())]
        for _ in range(r_k): adm(float(d[0]), float(d[1]))
        n += r_k
    Xt = torch.tensor(np.array(X))
    me, le = fit_gp(Xt, torch.tensor(Yf), fixed_noise=nzf); mt, lt = fit_gp(Xt, torch.tensor(Yg), fixed_noise=nzg)
    rec, _ = recommend_obd(Xset, me, le, mt, lt, gd, gamma=gamma, gate_mode=mode)
    du = float(dose_units(rec, dopt)); rec_tox = float(S['tox'](rec[0], rec[1]))
    muf_rec, varf_rec = _post_latent(me, torch.tensor(np.array([rec])))   # RPSEL = posterior efficacy RMSE at rec
    rpsel = float(np.sqrt((float(muf_rec.reshape(-1)[0]) - S['fopt'])**2 + float(varf_rec.reshape(-1)[0])))
    toxic = int(sum(1 for x in X if S['tox'](x[0], x[1]) > gd))
    pf_f = gate(mt, lt, Xset, gd, mode); passed = pf_f > gamma
    g_true = np.array([S['tox'](float(x[0]), float(x[1])) for x in Xset.numpy()])
    npass = int(passed.sum()); nsafe = int((passed & (g_true <= gd)).sum())
    # recommendation read out across a gamma_rec grid (decouples sampling-gamma from recommend-gamma) + OBD-safety diagnostic
    recs = []
    for grx in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        rcx, _ = recommend_obd(Xset, me, le, mt, lt, gd, gamma=grx, gate_mode=mode)
        recs.append(dict(grec=float(grx), du=float(dose_units(rcx, dopt)),
                         unsafe=int(float(S['tox'](rcx[0], rcx[1])) > gd), eff_reg=float(S['fopt'] - float(S['eff'](rcx[0], rcx[1])))))
    iobd = int(np.argmin(np.linalg.norm(Xset.numpy() - np.asarray(dopt, float), axis=1)))   # grid dose nearest true OBD
    mu_go, vgo = _post_latent(mt, Xset[iobd:iobd+1]); sdo = float(vgo.clamp_min(1e-12).sqrt().reshape(-1)[0])
    obd_pf = float(ss.norm.cdf((gd - float(mu_go.reshape(-1)[0]))/sdo))   # higher = near-OBD more confidently safe
    return dict(policy=policy, seed=int(seed), sim=sim, stratum=int(z), gamma=float(gamma), mode=mode,
                dose_units=du, rpsel=rpsel, rec_unsafe=int(rec_tox > gd), toxic=toxic, n_gate_pass=npass, n_gate_pass_safe=nsafe,
                rec_d1=float(rec[0]), rec_d2=float(rec[1]), rec_true_eff=float(S['eff'](rec[0], rec[1])), rec_true_tox=rec_tox,
                recs=recs, obd_pf=obd_pf, obd_sdg=sdo,
                traj=(dict(n=tr_n, du=tr_du, rpsel=tr_rp, toxic=tr_tx) if traj else None))

POLS = ['cEI', 'cKG1fix', 'GBE']
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=100)
    ap.add_argument('--gammas', default='0.5,0.6,0.7,0.8,0.9'); ap.add_argument('--out', default='gate_ab_smoke.json')
    ap.add_argument('--pols', default=','.join(POLS)); ap.add_argument('--modes', default='latent,predictive')
    ap.add_argument('--sims', default='osa,gbump'); ap.add_argument('--noise', default='fixed')
    ap.add_argument('--traj', action='store_true'); ap.add_argument('--budget', type=int, default=40); ap.add_argument('--rk', type=int, default=2); ap.add_argument('--cpus', type=int, default=0); ap.add_argument('--grid', type=int, default=5); a = ap.parse_args()
    gammas = [float(x) for x in a.gammas.split(',')]; pols = a.pols.split(','); modes = a.modes.split(','); sims = a.sims.split(',')
    res = json.load(open(a.out)) if os.path.exists(a.out) and os.path.getsize(a.out) else []
    dk = {(r['policy'], r['seed'], r['sim'], r['stratum'], r['mode'], round(r['gamma'], 3)) for r in res}
    jobs = [(p, s, sim, z, mode, g) for g in gammas for sim in sims for mode in modes
            for z in (0, 1) for p in pols for s in range(a.seeds)
            if (p, s, sim, z, mode, round(g, 3)) not in dk]
    print(f"{len(jobs)} NEW trials  (gammas={gammas}, {a.seeds} seeds x 2 sims x 2 modes x 2 strata x 3 pols)")
    t0 = time.time()
    import ray; ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=a.cpus if a.cpus > 0 else None)
    rr = ray.remote(num_cpus=1)(run_trial)
    futs = [rr.remote(p, s, z, g, sim, mode, a.noise, a.traj, a.budget, 4, a.rk, a.grid) for (p, s, sim, z, mode, g) in jobs]
    while futs:
        ready, futs = ray.wait(futs, num_returns=min(60, len(futs)))
        res += ray.get(ready); json.dump(res, open(a.out, 'w')); print(f"  {len(res)} done [{time.time()-t0:.0f}s]", flush=True)
    ray.shutdown(); analyze(res, gammas)

def analyze(res, gammas):
    def cell(sim, mode, z, p, g, key='dose_units'):
        return np.array([r[key] for r in res if r['policy'] == p and r['sim'] == sim and r['stratum'] == z
                         and r['mode'] == mode and abs(r['gamma']-g) < 1e-9])
    def gaprow(sim, mode, z, pa, pb):
        out = []
        for g in gammas:
            a_, b_ = cell(sim, mode, z, pa, g), cell(sim, mode, z, pb, g)
            if len(a_) and len(b_):
                gap = a_.mean()-b_.mean(); se = np.hypot(a_.std()/np.sqrt(len(a_)), b_.std()/np.sqrt(len(b_)))
                out.append(f"{gap:+.2f}{'*' if abs(gap) > 2*se else ' '}")
            else: out.append("  -- ")
        return out
    print("\n" + "="*88)
    print(f"GATE A/B  γ-SWEEP  (fixed noise, OUR verified acquisition, 100 seeds)   * = |gap|>2SE")
    print("="*88)
    hdr = "    z | " + "  ".join(f"γ={g}" for g in gammas)
    for sim in ('osa', 'gbump'):
        print(f"\n################  {sim.upper()}  ################")
        for mode in ('latent', 'predictive'):
            print(f"  ==== gate_mode = {mode} ====")
            print(f"   cEI−cKG  (+ ⇒ cKG better):")
            print(hdr)
            for z in (0, 1): print(f"    {z} | " + "  ".join(f"{c:<5}" for c in gaprow(sim, mode, z, 'cEI', 'cKG1fix')))
            print(f"   cKG−GBE  (− ⇒ GBE better ⇒ lookahead moot):")
            for z in (0, 1): print(f"    {z} | " + "  ".join(f"{c:<5}" for c in gaprow(sim, mode, z, 'cKG1fix', 'GBE')))
    # toxic-exposure: latent vs predictive (averaged over γ, cKG)
    print("\n-- cKG toxic exposure, latent vs predictive (mean over γ) --")
    for sim in ('osa', 'gbump'):
        for z in (0, 1):
            tl = np.mean([r['toxic'] for r in res if r['policy'] == 'cKG1fix' and r['sim'] == sim and r['stratum'] == z and r['mode'] == 'latent'])
            tp = np.mean([r['toxic'] for r in res if r['policy'] == 'cKG1fix' and r['sim'] == sim and r['stratum'] == z and r['mode'] == 'predictive'])
            print(f"   {sim} z={z}: latent={tl:.1f}  predictive={tp:.1f}  (Δ={tp-tl:+.1f})")

if __name__ == '__main__':
    main()
