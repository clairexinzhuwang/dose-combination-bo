#!/usr/bin/env python3
"""Historical falsification experiment; not a source of current manuscript claims.

curvsweep.py -- Move #3, attempt 2: does cKG have lower dose-location error when the OBD's location is set
by the TOXICITY BOUNDARY (which cKG resolves), not by the efficacy ridge?

Attempt 1 (signflip_sweep.py) falsified the (kappa, sigma_g) hypothesis: cKG lost everywhere
because the optimum's tangential position was pinned by EFFICACY curvature kappa, which cKG (a
toxicity-boundary resolver) cannot exploit. Here we set kappa=0 (efficacy purely increasing, no
tangential efficacy info) and add toxicity-boundary curvature c, so the feasible region tapers to
a tip and the OBD location is determined ENTIRELY by the constraint geometry cKG resolves.

PREDICTION (falsifiable): gap := mean[dose-units(cKG) - dose-units(cEI)] (paired) becomes NEGATIVE
(cKG localizes BETTER) as boundary curvature c grows (sharper tip => OBD more sharply determined by
the boundary), with the cKG edge largest at non-trivial sigma_g (where cEI's sigma_g-blindness costs
most). FALSIFIED if cKG does not win at high c, i.e. resolving the OBD-determining constraint is still
not enough -> the OSA/MARIPOSA localization wins are not reproduced by constraint geometry either.

Run:  python curvsweep.py    # -> curvsweep.json
"""
import json, sys, time
import numpy as np
import gate_ab_smoke as G

SMOKE = '--smoke' in sys.argv
CURVS = [3.0] if SMOKE else [1.0, 3.0, 6.0, 12.0]
SIGS  = [1.5] if SMOKE else [0.5, 1.5, 3.0]
SEEDS = 4 if SMOKE else 40
GAMMA = 0.7
POLS  = ['cEI', 'cKG1fix']
A, T, B, K = 3.0, 0.9, 1.0, 0.0      # k=0: no tangential efficacy curvature -> OBD set by constraint only

def simstr(c, s): return f"synth:k={K};c={c};s={s};a={A};t={T};b={B}"

def main():
    jobs = [(p, seed, simstr(c, s), c, s)
            for c in CURVS for s in SIGS for p in POLS for seed in range(SEEDS)]
    print(f"{len(jobs)} trials | curvature={CURVS} sigma_g={SIGS} seeds={SEEDS} (k=0, gamma={GAMMA})")
    t0 = time.time()
    if SMOKE:
        out = [dict(G.run_trial(p, seed, 0, GAMMA, sim, 'latent', 'fixed', False, 40, 4, 2, 5), curv=c, sigg=s)
               for (p, seed, sim, c, s) in jobs]
    else:
        import ray; ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=8)
        rr = ray.remote(num_cpus=1)(G.run_trial)
        futs = [rr.remote(p, seed, 0, GAMMA, sim, 'latent', 'fixed', False, 40, 4, 2, 5)
                for (p, seed, sim, c, s) in jobs]
        out = [dict(r, curv=c, sigg=s) for r, (p, seed, sim, c, s) in zip(ray.get(futs), jobs)]
    json.dump(out, open('curvsweep.json', 'w'))
    print(f"done in {time.time()-t0:.0f}s -> curvsweep.json")
    analyze(out)

def analyze(out):
    print("\n=== paired cKG-cEI dose-units gap (NEGATIVE = cKG localizes BETTER), k=0 ===")
    print("rows=toxicity-boundary curvature c (sharper OBD via constraint), cols=sigma_g")
    print("curv\\sig | " + " | ".join(f"{s:>14}" for s in SIGS))
    for c in CURVS:
        cells = []
        for s in SIGS:
            ce = {r['seed']: r['dose_units'] for r in out if r['curv']==c and r['sigg']==s and r['policy']=='cEI'}
            ck = {r['seed']: r['dose_units'] for r in out if r['curv']==c and r['sigg']==s and r['policy']=='cKG1fix'}
            seeds = sorted(set(ce) & set(ck))
            d = np.array([ck[i]-ce[i] for i in seeds])
            mg = d.mean(); se = d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else float('nan')
            cells.append(f"{mg:+.3f}+-{se:.3f}{'***' if abs(mg)>2*se else '':<3}")
        print(f"{c:>8} | " + " | ".join(f"{x:>14}" for x in cells))
    print("\n(Prediction: gap goes NEGATIVE as c grows. If it stays >=0, cKG's localization win is "
          "not reproduced by constraint geometry either -> honest negative.)")

if __name__ == '__main__':
    main()
