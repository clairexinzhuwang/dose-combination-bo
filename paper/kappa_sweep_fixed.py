"""Fixed-noise observation-noise-multiplier sweep for the cEI-versus-cKG comparison.

Re-run full-panel OSA with both observation SDs scaled by kappa in
{0.25, 0.5, 1, 1.5, 2}; kappa=1 uses the study calibration. The reported
``dose_units`` metric is distance to the stored 200-by-200 numerical
continuous-domain OBD reference. Output: ``kappa_sweep_fixed.json``.
"""
import json, os, time, argparse
import numpy as np
from dose_combination_bo import run_trial

KAPS = [0.25, 0.5, 1.0, 1.5, 2.0]
GAMMAS = [0.5, 0.7, 0.9]
POLS = ['cEI', 'cKG1fix']
SIM, MODE = 'osa', 'latent'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=100)
    ap.add_argument('--out', default='kappa_sweep_fixed.json')
    ap.add_argument('--cpus', type=int, default=0)
    a = ap.parse_args()
    if a.seeds < 1:
        raise ValueError('--seeds must be a positive integer')
    res = json.load(open(a.out)) if os.path.exists(a.out) and os.path.getsize(a.out) else []
    for index, row in enumerate(res):
        expected = {
            'sim': SIM, 'mode': MODE, 'noise': 'fixed', 'empty_gate': 'pf',
            'budget': 40, 'warmup': 4, 'r_k': 2, 'grid_n': 5,
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise ValueError(
                f'{a.out} row {index} has incompatible design metadata; use a fresh output path'
            )
        if (row.get('policy') not in POLS or row.get('stratum') not in (0, 1)
                or row.get('gamma') not in GAMMAS or row.get('kap') not in KAPS
                or not 0 <= int(row.get('seed', -1)) < a.seeds):
            raise ValueError(
                f'{a.out} row {index} lies outside the requested sweep; use a fresh output path'
            )
    dk = {(r['policy'], r['seed'], r['stratum'], round(r['gamma'], 3), round(r['kap'], 3)) for r in res}
    if len(dk) != len(res):
        raise ValueError(f'{a.out} contains duplicate design cells')
    jobs = [(p, s, z, g, kap) for kap in KAPS for g in GAMMAS for z in (0, 1) for p in POLS
            for s in range(a.seeds) if (p, s, z, round(g, 3), round(kap, 3)) not in dk]
    print(f"{len(jobs)} NEW trials (KAPS={KAPS}, GAMMAS={GAMMAS}, {a.seeds} seeds x 2 strata x 2 pols)", flush=True)
    t0 = time.time()
    import ray
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=a.cpus if a.cpus > 0 else None)
    # Use the current public harness, not the vendored pre-correction identity
    # script.  In particular, this makes the observation-noise multiplier and
    # the documented most-feasible empty-gate fallback explicit.
    rr = ray.remote(num_cpus=1)(run_trial)
    futs = {
        rr.remote(
            p, s, z, g, SIM, mode=MODE, noise='fixed', traj=False,
            budget=40, warmup=4, r_k=2, grid_n=5, empty_gate='pf', kap=kap,
        ): (p, s, z, g, kap)
            for (p, s, z, g, kap) in jobs}
    fl = list(futs)
    while fl:
        ready, fl = ray.wait(fl, num_returns=min(60, len(fl)))
        for fut in ready:
            r = ray.get(fut); p, s, z, g, kap = futs[fut]; r['kap'] = float(kap); res.append(r)
        json.dump(res, open(a.out, 'w')); print(f"  {len(res)} done [{time.time()-t0:.0f}s]", flush=True)
    ray.shutdown()
    analyze(res)

def analyze(res):
    def clustered(kap, policy, gamma=None):
        by_seed = {}
        for row in res:
            if (row['policy'] == policy and abs(row['kap'] - kap) < 1e-9
                    and (gamma is None or abs(row['gamma'] - gamma) < 1e-9)):
                by_seed.setdefault(int(row['seed']), []).append(float(row['dose_units']))
        return {seed: float(np.mean(values)) for seed, values in by_seed.items()}

    def paired(kap, gamma=None):
        ce = clustered(kap, 'cEI', gamma)
        ck = clustered(kap, 'cKG1fix', gamma)
        if not ce or set(ce) != set(ck):
            raise ValueError(f'unpaired kappa cells for kap={kap}, gamma={gamma}')
        seeds = sorted(ce)
        diff = np.asarray([ce[seed] - ck[seed] for seed in seeds], dtype=float)
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
        return ce, ck, diff, se

    print("\n" + "=" * 74)
    print("KAPPA (noise-multiplier) SWEEP  fixed noise  cEI-cKG dose-units gap (+ = cKG more precise)")
    print("=" * 74)
    for g in GAMMAS:
        print(f"\n gamma={g}")
        print(f"  {'kappa':>6} {'cEI_du':>7} {'cKG_du':>7} {'gap':>8} {'2SE':>6}  n")
        for kap in KAPS:
            ce, ck, diff, se = paired(kap, g)
            gap = diff.mean()
            star = '  *' if abs(gap) > 2*se else ''
            print(f"  {kap:>6} {np.mean(list(ce.values())):7.3f} "
                  f"{np.mean(list(ck.values())):7.3f} {gap:+8.3f} {2*se:6.3f}  "
                  f"{len(diff)}{star}")
    print("\n POOLED over gamma  (+ = cKG more precise):")
    print(f"  {'kappa':>6} {'gap':>8} {'2SE':>6}  n")
    for kap in KAPS:
        _, _, diff, se = paired(kap)
        gap = diff.mean()
        star = '  *' if abs(gap) > 2*se else ''
        print(f"  {kap:>6} {gap:+8.3f} {2*se:6.3f}  {len(diff)}{star}")

if __name__ == '__main__':
    main()
