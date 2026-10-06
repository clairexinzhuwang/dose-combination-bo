#!/usr/bin/env python3
"""Historical pre-correction table helper retained for provenance only.

This file is *not* the source of current manuscript tables: it predates the corrected
empty-gate rule and seed-clustered paired analyses. Follow ``docs/reproducibility.md``
and use the corrected analysis copies in the public reproducibility archive.

Data fields used (present in every results JSON):
  dose_units, rpsel (= efficacy RMSE), rec_unsafe (0/1), toxic (count of toxic
  doses among 40 patients), n_gate_pass, n_gate_pass_safe.
  toxic-exposure% = toxic/40*100 ; gate-precision% = sum(safe)/sum(pass)*100.

Policy code 'cKG1fix' is printed as 'cKG' (paper name).
"""
import json, sys

NAME = {'cEI': 'cEI', 'cKG1fix': 'cKG', 'GBE': 'GBE',
        'straddle': 'straddle', 'qBIG': 'SUR', 'random': 'random-safe'}

def load(path, sim=None, mode='latent'):
    d = json.load(open(path))
    return [r for r in d if r.get('mode') == mode and (sim is None or r['sim'] == sim)]

def sel(rows, policy, gamma, stratum=None):
    return [r for r in rows if r['policy'] == policy and abs(r['gamma'] - gamma) < 1e-9
            and (stratum is None or r['stratum'] == stratum)]

def mean(xs): return sum(xs) / len(xs) if xs else float('nan')

def cell(rows, policy, gamma, stratum=None):
    rs = sel(rows, policy, gamma, stratum)
    if not rs: return None
    npass = sum(r['n_gate_pass'] for r in rs); nsafe = sum(r['n_gate_pass_safe'] for r in rs)
    return dict(
        du=mean([r['dose_units'] for r in rs]),
        rmse=mean([r['rpsel'] for r in rs]),
        toxct=mean([r['toxic'] for r in rs]),           # mean toxic count (out of 40)
        exp=mean([r['toxic'] for r in rs]) / 40 * 100,  # toxic exposure %
        unsafe=mean([r['rec_unsafe'] for r in rs]) * 100,
        gprec=(100 * nsafe / npass) if npass else float('nan'),
        n=len(rs))

def demo_table(path='gb_detail.json'):
    """tab:demo -- Gaussian-bump, per (stratum, gamma, policy)."""
    rows = load(path, 'gbump')
    print("% ---- tab:demo (Gaussian-bump) from", path, "----")
    for z in (0, 1):
        for i, g in enumerate((0.5, 0.7, 0.9)):
            for p in ('cEI', 'cKG1fix', 'GBE'):
                c = cell(rows, p, g, z)
                lead = f"$z{{=}}{z}$" if (i == 0 and p == 'cEI') else "       "
                print(f"{lead} & {g}, {NAME[p]:3} & {c['du']:.2f} & {c['rmse']:.2f} & "
                      f"{c['toxct']:.1f} & {round(c['unsafe'])}\\% & {round(c['gprec'])} \\\\")
    # ranges for the main-text claim
    duck = [cell(rows, 'cKG1fix', g, z)['du'] for z in (0, 1) for g in (0.5, 0.7, 0.9)]
    ducei = [cell(rows, 'cEI', g, z)['du'] for z in (0, 1) for g in (0.5, 0.7, 0.9)]
    dugbe = [cell(rows, 'GBE', g, z)['du'] for z in (0, 1) for g in (0.5, 0.7, 0.9)]
    print(f"% cKG du range {min(duck):.2f}-{max(duck):.2f} | cEI {min(ducei):.2f}-{max(ducei):.2f} | GBE {min(dugbe):.2f}-{max(dugbe):.2f}")

def pooled_table(path, sim, label):
    """efftox / mariposa style: pooled over strata, per gamma; dose-units + unsafe%."""
    rows = load(path, sim)
    print(f"% ---- {label} (pooled) from {path} ----")
    for p in ('cEI', 'cKG1fix', 'GBE'):
        du = [cell(rows, p, g)['du'] for g in (0.5, 0.7, 0.9)]
        un = [round(cell(rows, p, g)['unsafe']) for g in (0.5, 0.7, 0.9)]
        print(f"{NAME[p]:3} & " + " & ".join(f"{x:.2f}" for x in du) + " & " +
              " & ".join(str(x) for x in un) + " \\\\")

def osa_tradeoff(path='osa_detail.json'):
    """tab:tradeoff -- OSA, per (gamma, stratum): unsafe% and exposure%."""
    rows = load(path, 'osa')
    print("% ---- tab:tradeoff (OSA unsafe% | exposure%) from", path, "----")
    for g in (0.5, 0.7, 0.9):
        for z in (0, 1):
            u = {p: cell(rows, p, g, z)['unsafe'] for p in ('cEI', 'cKG1fix', 'GBE')}
            e = {p: cell(rows, p, g, z)['exp'] for p in ('cEI', 'cKG1fix', 'GBE')}
            print(f"{g} & {z} & {u['cEI']:.1f} & {u['cKG1fix']:.1f} & {u['GBE']:.1f} & "
                  f"{e['cEI']:.1f} & {e['cKG1fix']:.1f} & {e['GBE']:.1f} \\\\")

def efficacy(path='efficacy_check.json'):
    """Recommended-dose TRUE efficacy at gamma=0.7, per surface (needs rec_true_eff field)."""
    import os
    if not os.path.exists(path):
        print(f"% (efficacy: {path} not present; run gate_ab_smoke.py --gammas 0.7 --sims osa,gbump,efftox,mariposa)")
        return
    d = [r for r in json.load(open(path)) if r.get('mode') == 'latent' and abs(r['gamma'] - 0.7) < 1e-9
         and 'rec_true_eff' in r]
    print("% ---- recommended-dose TRUE efficacy (gamma=0.7, pooled strata) ----")
    print(f"% {'surface':10} {'cEI':>9} {'cKG':>9} {'GBE':>9}  (cKG-cEI)")
    for sim in ('osa', 'gbump', 'efftox', 'mariposa'):
        v = {}
        for p in ('cEI', 'cKG1fix', 'GBE'):
            rs = [r for r in d if r['sim'] == sim and r['policy'] == p]
            v[p] = mean([r['rec_true_eff'] for r in rs]) if rs else float('nan')
        print(f"% {sim:10} {v['cEI']:9.3f} {v['cKG1fix']:9.3f} {v['GBE']:9.3f}  ({v['cKG1fix']-v['cEI']:+.2f})")

def baselines(osa='gate_ab_smoke.json', bm='baselines_efftox_mariposa.json'):
    """tab:baselines (Supplement S1) -- rec-unsafe % at gamma=0.7, 5 policies x 3 surfaces."""
    import os
    print("% ---- tab:baselines (rec-unsafe %, gamma=0.7, latent) ----")
    pols = ['cEI', 'cKG1fix', 'GBE', 'straddle', 'qBIG']
    print("% " + " & ".join(['Surface'] + [NAME.get(p, p) for p in pols]) + " \\\\")
    for lbl, path, sim in [('OSA', osa, 'osa'), ('Logistic', bm, 'efftox'), ('MARIPOSA', bm, 'mariposa')]:
        if not os.path.exists(path):
            print(f"% {lbl}: {path} missing"); continue
        d = [r for r in json.load(open(path)) if r['sim'] == sim and r.get('mode') == 'latent'
             and abs(r['gamma'] - 0.7) < 1e-9]
        cells = []
        for p in pols:
            rs = [r for r in d if r['policy'] == p]
            cells.append(str(round(100*sum(r['rec_unsafe'] for r in rs)/len(rs))) if rs else "-")
        print(f"% {lbl:9} & " + " & ".join(cells) + " \\\\")

def signflip(path='curvsweep.json'):
    """tab:signflip (Supplement S5) -- Family B paired cKG-cEI dose-units gap vs curvature."""
    import os
    if not os.path.exists(path):
        print(f"% (signflip: {path} missing; run curvsweep.py)"); return
    d = json.load(open(path)); CURVS = [1.0, 3.0, 6.0, 12.0]; SIGS = [0.5, 1.5, 3.0]
    print("% ---- tab:signflip (Family B paired gap = cKG-cEI dose-units; k=0) ----")
    for c in CURVS:
        cells = []
        for s in SIGS:
            ce = {r['seed']: r['dose_units'] for r in d if r['curv']==c and r['sigg']==s and r['policy']=='cEI'}
            ck = {r['seed']: r['dose_units'] for r in d if r['curv']==c and r['sigg']==s and r['policy']=='cKG1fix'}
            seeds = sorted(set(ce) & set(ck)); g = mean([ck[i]-ce[i] for i in seeds])
            cells.append(f"{g:+.2f}")
        print(f"% c={int(c):<3} & " + " & ".join(cells) + " \\\\")

def kappa(path='kappa_sweep_fixed.json'):
    """Historical kappa helper: dose-location error and above-threshold recommendations."""
    import os
    if not os.path.exists(path):
        print(f"% (kappa: {path} missing; run kappa_sweep_fixed.py)"); return
    d = json.load(open(path)); KAPS = [0.25, 0.5, 1.0, 1.5, 2.0]
    print("% ---- tab:kappa (fixed-noise; du=dose-units precision, uns=rec-unsafe rate; pooled over gate+strata) ----")
    print("% kappa & cEI_du & cKG_du & cEI_uns & cKG_uns \\\\")
    for k in KAPS:
        def mm(pol, f):
            return mean([r[f] for r in d if r['policy'] == pol and abs(r['kap'] - k) < 1e-9])
        print(f"% {k:g} & {mm('cEI','dose_units'):.2f} & {mm('cKG1fix','dose_units'):.2f} & "
              f"{mm('cEI','rec_unsafe'):.2f} & {mm('cKG1fix','rec_unsafe'):.2f} \\\\")

if __name__ == '__main__':
    demo_table()
    print()
    pooled_table('efftox_results.json', 'efftox', 'tab:efftox')
    print()
    pooled_table('mariposa_results.json', 'mariposa', 'tab:mariposa_results')
    print()
    osa_tradeoff()
    print()
    efficacy()
    print()
    baselines()
    print()
    signflip()
    print()
    kappa()
