"""Derive exploratory acceptable near-optimal OSA selections from frozen records.

No GP fitting, allocation updates or new trials are performed. Run with the
repository's scientific Python environment; historical inputs are read-only.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import numpy as np
import zstandard


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    inferred = next((p for p in Path(__file__).resolve().parents
                     if (p / 'results/comparator_family_analysis').is_dir()), None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=inferred)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.repo is None:
        parser.error('Pass --repo pointing to the complete study repository.')
    root, out = args.repo.resolve(), args.output_dir.resolve()
    if out == root / 'results' or root / 'results' in out.parents:
        parser.error('Outputs must be outside the historical results directory.')
    out.mkdir(parents=True, exist_ok=True)
    source = root / 'results/comparator_family_analysis/comparator_family_selected_records.json.zst'
    raw = source.read_bytes()
    metadata = json.loads(Path(str(source) + '.metadata.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == metadata['artifact_sha256']
    records = json.loads(zstandard.ZstdDecompressor().decompress(raw))
    assert len(records) == 34000
    policies = ['cKG', 'tmse', 'qBIG', 'cEI']
    names = dict(cKG='cKG', tmse='tMSE', qBIG='Entropy rule', cEI='cEI')
    taus, deltas = [.5, .6, .7, .8, .9], [0., .5, 1., 2.]
    selected = [r for r in records if r['sim'] == 'osa'
                and r.get('protocol_scaffold', 'lhs_fixed') == 'lhs_fixed'
                and r['policy'] in policies]
    keyed = {(r['policy'], float(r['gamma']), int(r['stratum']), int(r['seed'])): r
             for r in selected}
    expected = {(p, t, z, seed) for p in policies for t in taus
                for z in [0, 1] for seed in range(200)}
    assert len(selected) == len(keyed) == 8000 and set(keyed) == expected
    assert all(r.get('recommendation_made', True) for r in selected)
    grid = np.array([(a, b) for a in np.linspace(0, 1, 5)
                     for b in np.linspace(0, 1, 5)])
    a, b = grid.T
    basis = np.column_stack([np.ones(25), a, b, a*b, a*a, b*b, a*a*b*b])
    beta = np.array([[-1.38,-4.08,-.48,-4.23,2.45,-7.51,-1.56],
                     [1.05,-11.28,-8.32,-17.02,8.17,2.34,4.61]])
    theta = np.array([-.59,1.83,2.26,-4.05,1.79,.47,2.91])
    efficacy, toxicity = -basis @ beta.T, basis @ theta
    limits = [1.5, 2.]
    dose_index = {tuple(d): i for i, d in enumerate(grid)}
    safe = np.column_stack([toxicity <= limit for limit in limits])
    best = np.array([efficacy[safe[:, z], z].max() for z in [0, 1]])
    optima = [set(np.flatnonzero(safe[:, z] & (efficacy[:, z] == best[z])))
              for z in [0, 1]]
    assert [len(x) for x in optima] == [1, 1]
    trial_rows, indicators = [], {}
    max_eff_error = max_tox_error = 0.
    for key in sorted(keyed):
        policy, tau, z, seed = key
        row = keyed[key]
        point = (float(row['rec_d1']), float(row['rec_d2']))
        assert point in dose_index
        i = dose_index[point]
        e, g = float(efficacy[i, z]), float(toxicity[i])
        ee, ge = abs(e-float(row['rec_true_eff'])), abs(g-float(row['rec_true_tox']))
        max_eff_error, max_tox_error = max(max_eff_error, ee), max(max_tox_error, ge)
        assert ee < 1e-10 and ge < 1e-10
        acceptable = bool(safe[i, z])
        assert bool(row['rec_unsafe']) == (not acceptable)
        gap = float(best[z]-e)
        if acceptable:
            assert gap >= 0
        values = [int(acceptable and (i in optima[z] if d == 0 else gap <= d))
                  for d in deltas]
        assert values == sorted(values) and all(v <= acceptable for v in values)
        item = dict(policy=policy, tau=tau, stratum=z, seed=seed,
                    selected_d1=point[0], selected_d2=point[1], true_efficacy=e,
                    true_toxicity=g, true_acceptable=int(acceptable), grid_best_efficacy=float(best[z]),
                    efficacy_gap=gap)
        for d, value in zip(deltas, values):
            indicators[key + (d,)] = value
            item[f'acceptable_within_{d:g}'] = value
        trial_rows.append(item)
    summaries = []
    for tau in taus:
        for policy in policies:
            for delta in deltas:
                blocks = np.array([50.*sum(indicators[policy,tau,z,seed,delta] for z in [0,1])
                                   for seed in range(200)])
                count = int(sum(indicators[policy,tau,z,seed,delta]
                                for z in [0,1] for seed in range(200)))
                mean, se = float(blocks.mean()), float(blocks.std(ddof=1)/math.sqrt(200))
                assert math.isclose(mean, 100*count/400, abs_tol=1e-12)
                summaries.append(dict(tau=tau, policy=policy, policy_label=names[policy],
                                      delta_events_per_hour=delta, successful_trials=count,
                                      initiated_trials=400, estimate_pct=mean, mcse_pp=se,
                                      independent_replicate_sets=200))
    write_csv(out/'near_optimal_trials.csv', trial_rows)
    write_csv(out/'near_optimal_summary.csv', summaries)
    cells = {(r['tau'],r['policy'],r['delta_events_per_hour']): r for r in summaries}
    tex = [r'\begin{table}[H]', r'\revisioncolor', r'\centering', r'\footnotesize',
           r'\setlength{\tabcolsep}{5pt}',
           r'\caption{Truly acceptable selections within an efficacy margin of the grid optimum in OSA, under full-grid availability.}',
           r'\label{tab:nearoptimal}', r'\begin{tabular}{@{}clcccc@{}}', r'\toprule',
           r'$\tau$ & Rule & $\delta=0$ & $\delta=0.5$ & $\delta=1$ & $\delta=2$ \\',
           r'\midrule']
    for tau in taus:
        for policy in policies:
            formatted = [f"{cells[tau,policy,d]['estimate_pct']:.2f} ({cells[tau,policy,d]['mcse_pp']:.2f})"
                         for d in deltas]
            tex.append(f'{tau:g} & {names[policy]} & ' + ' & '.join(formatted) + r' \\')
        if tau != taus[-1]:
            tex.append(r'\addlinespace[2pt]')
    tex.extend([r'\bottomrule', r'\end{tabular}', r'\vspace{3pt}',
                r'\begin{minipage}{0.98\linewidth}', r'\scriptsize\raggedright',
                r'\textit{Note.} Entries are percentages (MCSE in pp), averaging the two stratum indicators within each of $M=200$ independent replicate sets. Each rule--cutoff row uses all 400 initiated trials, retaining fallback selections. Acceptability uses true mean toxicity, not the posterior criterion. Margins are in simulated AHI4 events/hour and are descriptive, not clinically calibrated. $\delta=0$ denotes selection of the exact acceptable grid optimum. These post hoc summaries reuse existing records; no adaptive trials were rerun.',
                r'\end{minipage}', r'\end{table}'])
    (out/'near_optimal_table.tex').write_text('\n'.join(tex)+'\n')
    checks = dict(source_authenticated=True, input_bundle_rows=34000, selected_full_grid_osa_rows=8000,
                  complete_four_policy_factorial=True, all_full_grid_trials_select=True,
                  grid_optima=[dict(stratum=z, dose=grid[next(iter(optima[z]))].tolist(),
                                    efficacy=float(best[z]), acceptable_pairs=int(safe[:,z].sum())) for z in [0,1]],
                  maximum_stored_truth_efficacy_error=max_eff_error, maximum_stored_truth_toxicity_error=max_tox_error,
                  all_stored_unsafe_flags_match_truth=True, indicators_monotone_in_delta=True,
                  unsafe_selections_never_successes=True, legacy_dense_grid_regret_not_used=True,
                  mcse_unit='200 independent seeds with paired strata; not400 independent Bernoulli trials',
                  new_trials_run=0)
    (out/'analysis_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
    print(f'Derived {len(summaries)} summaries from {len(trial_rows)} existing trials.')


if __name__ == '__main__':
    main()
