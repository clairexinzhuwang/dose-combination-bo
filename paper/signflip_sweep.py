#!/usr/bin/env python3
"""Historical falsification experiment; not a source of current manuscript claims.

signflip_sweep.py -- Move #3: is the cKG-vs-cEI dose-location sign-flip predictable?

The paper reports that cKG localizes BETTER than cEI on OSA but WORSE on the logistic
surface, and concedes it cannot name the controlling scalar. Here we test a falsifiable
2-parameter hypothesis on a controlled family (gate_ab_smoke sim='synth:...'):

  knob 1  kappa = tangential efficacy curvature  (sharpness of the optimum ALONG the boundary)
  knob 2  sigma_g = toxicity noise               (boundary identifiability = a*sqrt2/sigma_g; lower = sharper)

PREDICTION (stated up front, falsifiable):
  gap(d) := mean[ dose-units(cKG) - dose-units(cEI) ]  (paired by seed)
  is POSITIVE (cKG worse) at LOW kappa & HIGH sigma_g  (flat ridge, poorly-resolved boundary
     = poorly-identified optimum), and NEGATIVE (cKG better) at HIGH kappa & LOW sigma_g
     (sharply-identified optimum), with a MONOTONE sign boundary in the (kappa, sigma_g) plane.
  FALSIFIED if: the sign does not order monotonically with (kappa, sigma_g), or neither knob moves it.

Run:  python signflip_sweep.py            # full sweep -> signflip_sweep.json
      python signflip_sweep.py --smoke    # 2x2 cells x 6 seeds, serial, sanity only
"""
import json, sys, time
import numpy as np
import gate_ab_smoke as G

SMOKE = '--smoke' in sys.argv
KAPPAS = [0.5, 2.0] if SMOKE else [0.0, 0.5, 1.0, 2.0, 4.0]
SIGS   = [0.5, 2.0] if SMOKE else [0.5, 1.0, 2.0, 4.0]
SEEDS  = 6 if SMOKE else 40
GAMMA  = 0.7
POLS   = ['cEI', 'cKG1fix']
A, T, B = 3.0, 0.9, 1.0       # tox gradient, feasible size (OBD off-grid at 0.45,0.45), beta

def simstr(k, s): return f"synth:k={k};s={s};a={A};t={T};b={B}"

def main():
    jobs = [(p, seed, simstr(k, s), k, s)
            for k in KAPPAS for s in SIGS for p in POLS for seed in range(SEEDS)]
    print(f"{len(jobs)} trials | kappa={KAPPAS} sigma_g={SIGS} seeds={SEEDS} (gamma={GAMMA})")
    t0 = time.time()
    if SMOKE:
        out = [dict(G.run_trial(p, seed, 0, GAMMA, sim, 'latent', 'fixed', False, 40, 4, 2, 5),
                    kappa=k, sigg=s) for (p, seed, sim, k, s) in jobs]
    else:
        import ray; ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=8)
        rr = ray.remote(num_cpus=1)(G.run_trial)
        futs = [rr.remote(p, seed, 0, GAMMA, sim, 'latent', 'fixed', False, 40, 4, 2, 5)
                for (p, seed, sim, k, s) in jobs]
        res = ray.get(futs)
        out = [dict(r, kappa=k, sigg=s) for r, (p, seed, sim, k, s) in zip(res, jobs)]
    json.dump(out, open('signflip_sweep.json', 'w'))
    print(f"done in {time.time()-t0:.0f}s -> signflip_sweep.json")
    analyze(out)

def analyze(out):
    print("\n=== paired cKG-cEI dose-units gap (NEGATIVE = cKG localizes BETTER) ===")
    print("rows=kappa (tangential curvature), cols=sigma_g (higher=poorer boundary id)")
    hdr = "kappa\\sig | " + " | ".join(f"{s:>14}" for s in SIGS); print(hdr)
    for k in KAPPAS:
        cells = []
        for s in SIGS:
            ce = {r['seed']: r['dose_units'] for r in out if r['kappa']==k and r['sigg']==s and r['policy']=='cEI'}
            ck = {r['seed']: r['dose_units'] for r in out if r['kappa']==k and r['sigg']==s and r['policy']=='cKG1fix'}
            seeds = sorted(set(ce) & set(ck))
            d = np.array([ck[i]-ce[i] for i in seeds])
            mg = d.mean(); se = d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else float('nan')
            sig = '***' if abs(mg) > 2*se else ''
            cells.append(f"{mg:+.3f}+-{se:.3f}{sig:<3}")
        print(f"{k:>9} | " + " | ".join(f"{c:>14}" for c in cells))
    print("\n(* gap exceeds 2 SE. Prediction: top-right (low kappa, high sigma_g) POSITIVE; "
          "bottom-left (high kappa, low sigma_g) NEGATIVE; monotone sign boundary.)")

if __name__ == '__main__':
    main()
