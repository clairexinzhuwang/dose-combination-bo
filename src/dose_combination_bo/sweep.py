"""Run a grid of trials and aggregate operating characteristics.

``sweep`` fans out over acquisitions x surfaces x strata x gammas x modes x seeds
(Ray-parallel if available, else serial and resume-safe), returning a flat list of
per-trial result dicts. ``summarize`` collapses those into a tidy per-cell table
(mean +/- seed-clustered Monte Carlo SE of any metric). ``paired_contrast`` uses
the shared seed/design cells for policy differences. ``leaderboard`` prints a
seed-clustered descriptive ranking.
"""
import json
import os
import time

import numpy as np

from .trial import run_trial, _validate_trial_args
from .version import VERSION, implementation_fingerprint
from .registry import get_acquisition, get_surface, ACQUISITIONS, SURFACES
from .protocol import PROTOCOL_CONFIG_FIELDS, PROTOCOL_DEFAULTS


def _dump(res, out):
    """Atomically write results to ``out`` (temp file + os.replace) so a crash
    mid-write never truncates an existing checkpoint."""
    if not out:
        return
    tmp = f"{out}.tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    try:
        dfd = os.open(os.path.dirname(os.path.abspath(out)), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


_DESIGN_DEFAULTS = {
    "kap": 1.0,
    "software_version": "historical_unversioned",
    "core_source_sha256": "historical_unversioned",
    "allow_ungated_diagnostic": False,
    **PROTOCOL_DEFAULTS,
}


def _design_key(job, budget, r_k, grid_n, noise, warmup, empty_gate, kap,
                protocol_scaffold, region_step, empty_gate_stop_after,
                exclude_repeats_during_expansion, allow_ungated_diagnostic=False):
    p, s, sim, z, mode, g = job
    return (p, s, sim, z, mode, float(g), budget, r_k, grid_n, noise,
            warmup, empty_gate, float(kap), protocol_scaffold,
            float(region_step), int(empty_gate_stop_after),
            bool(exclude_repeats_during_expansion),
            bool(allow_ungated_diagnostic), VERSION, implementation_fingerprint())


def _cached_key(r):
    required = ("policy", "seed", "sim", "stratum", "mode", "gamma", "budget", "r_k",
                "grid_n", "noise", "warmup", "empty_gate")
    required += ("software_version", "core_source_sha256", "allow_ungated_diagnostic")
    missing = [k for k in required if k not in r]
    if missing:
        raise ValueError(f"cached result is missing design metadata {missing}")
    return (r["policy"], r["seed"], r["sim"], r["stratum"], r["mode"],
            float(r["gamma"]), r["budget"], r["r_k"], r["grid_n"], r["noise"],
            r["warmup"], r["empty_gate"], float(r.get("kap", 1.0)),
            r.get("protocol_scaffold", "lhs_fixed"),
            float(r.get("region_step", 0.25)),
            int(r.get("empty_gate_stop_after", 3)),
            bool(r.get("exclude_repeats_during_expansion", True)),
            bool(r["allow_ungated_diagnostic"]), r["software_version"], r["core_source_sha256"])


def sweep(acquisitions, surfaces, seeds=100, gammas=(0.5, 0.6, 0.7, 0.8, 0.9),
          modes=("latent",), strata=(0, 1), noise="fixed", budget=40, r_k=2, grid_n=5,
          warmup=4, empty_gate="pf", parallel=True, out=None, verbose=True, kap=1.0,
          protocol_scaffold="lhs_fixed", region_step=0.25, empty_gate_stop_after=3,
          exclude_repeats_during_expansion=True, allow_ungated_diagnostic=False):
    """Run every (acquisition, surface, stratum, gamma, mode, seed) combination.

    Returns a list of result dicts (one per trial). If ``out`` is given, results are
    checkpointed to that JSON file and already-completed cells are skipped on rerun.
    ``noise`` selects whether the GP likelihood variance is fixed or learned; ``kap``
    is a positive multiplier on both data-generating observation-noise SDs. The resume
    key includes both plus the serialized dose-availability-design value and its tuning constants, so reusing
    one ``out`` across different designs never silently mixes them. Exact floating-point
    cutoffs and the core-source fingerprint are part of the key. Old unversioned
    checkpoints are refused. External plug-in changes are not authenticated by the
    core fingerprint: use a fresh output file after changing a plug-in.
    """
    acquisitions = [acquisitions] if isinstance(acquisitions, str) else list(acquisitions)
    surfaces = [surfaces] if isinstance(surfaces, str) else list(surfaces)
    if isinstance(seeds, (bool, np.bool_)):
        raise ValueError("seeds must be a positive integer count or an explicit integer sequence")
    if isinstance(seeds, (int, np.integer)):
        if seeds < 1: raise ValueError("seeds count must be positive")
        seeds = list(range(int(seeds)))
    else:
        seeds = list(seeds)
    if not isinstance(allow_ungated_diagnostic, (bool, np.bool_)):
        raise ValueError("allow_ungated_diagnostic must be Boolean")
    gammas, modes, strata = list(gammas), list(modes), list(strata)
    dimensions = {"acquisitions": acquisitions, "surfaces": surfaces, "seeds": seeds,
                  "gammas": gammas, "modes": modes, "strata": strata}
    for name, values in dimensions.items():
        if not values:
            raise ValueError(f"{name} must not be empty")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} contains duplicate values")

    # preflight: fail fast on an unknown acquisition/surface name (before any compute)
    for p in acquisitions:
        fn = get_acquisition(p)
        if getattr(fn, "acquisition_name", p) == "utmse" and not allow_ungated_diagnostic:
            raise ValueError("ungated diagnostic requires allow_ungated_diagnostic=True")
    for sim in surfaces:
        if not (isinstance(sim, str) and sim.startswith("synth:")):
            get_surface(sim)
    for seed in seeds:
        for z in strata:
            for g in gammas:
                for mode in modes:
                    _validate_trial_args(seed, z, g, mode, noise, budget, warmup, r_k,
                                         grid_n, empty_gate, kap, protocol_scaffold,
                                         region_step, empty_gate_stop_after,
                                         exclude_repeats_during_expansion)

    expected = {_design_key((p, s, sim, z, mode, g), budget, r_k, grid_n, noise,
                            warmup, empty_gate, kap, protocol_scaffold, region_step,
                            empty_gate_stop_after, exclude_repeats_during_expansion, allow_ungated_diagnostic)
                for g in gammas for sim in surfaces for mode in modes
                for z in strata for p in acquisitions for s in seeds}

    res = []
    if out and os.path.exists(out) and os.path.getsize(out):
        with open(out) as fh:
            res = json.load(fh)
        if not isinstance(res, list):
            raise ValueError("cached result file must contain a JSON list")
    cached_keys = [_cached_key(r) for r in res]
    if len(cached_keys) != len(set(cached_keys)):
        raise ValueError("cached result file contains duplicate design cells")
    unexpected = set(cached_keys) - expected
    if unexpected:
        example = next(iter(unexpected))
        raise ValueError(f"cached result design does not match requested sweep; example: {example}")
    done = set(cached_keys)
    jobs = [(p, s, sim, z, mode, g)
            for g in gammas for sim in surfaces for mode in modes
            for z in strata for p in acquisitions for s in seeds
            if _design_key((p, s, sim, z, mode, g), budget, r_k, grid_n, noise, warmup,
                           empty_gate, kap, protocol_scaffold, region_step,
                           empty_gate_stop_after,
                           exclude_repeats_during_expansion, allow_ungated_diagnostic) not in done]
    if verbose:
        print(f"{len(jobs)} new trials ({len(res)} cached)")
    if not jobs:
        return res

    t0 = time.time()
    if parallel:
        try:
            import ray
            ray.init(ignore_reinit_error=True, include_dashboard=False)
            # Ship the driver's registries into fresh worker processes; without this,
            # user-registered acquisitions/surfaces are absent in workers.
            _ea, _es = dict(ACQUISITIONS), dict(SURFACES)

            def _run(*a, _ea=_ea, _es=_es):
                from .registry import ACQUISITIONS as A, SURFACES as S
                A.update(_ea); S.update(_es)
                return run_trial(*a)

            rr = ray.remote(num_cpus=1)(_run)
            futs = [rr.remote(p, s, z, g, sim, mode, noise, False, budget, warmup, r_k, grid_n,
                              empty_gate, kap, protocol_scaffold, region_step,
                              empty_gate_stop_after, exclude_repeats_during_expansion, allow_ungated_diagnostic)
                    for (p, s, sim, z, mode, g) in jobs]
            while futs:
                ready, futs = ray.wait(futs, num_returns=min(60, len(futs)))
                res += ray.get(ready)
                _dump(res, out)
                if verbose:
                    print(f"  {len(res)} done [{time.time() - t0:.0f}s]", flush=True)
            ray.shutdown()
            return res
        except ImportError:
            if verbose:
                print("  ray not available; running serially")

    for k, (p, s, sim, z, mode, g) in enumerate(jobs, 1):
        res.append(run_trial(p, s, z, g, sim, mode, noise, False, budget, warmup, r_k, grid_n,
                             empty_gate, kap, protocol_scaffold, region_step,
                             empty_gate_stop_after, exclude_repeats_during_expansion, allow_ungated_diagnostic))
        if out and k % 25 == 0:
            _dump(res, out)
        if verbose and k % 25 == 0:
            print(f"  {k}/{len(jobs)} [{time.time() - t0:.0f}s]", flush=True)
    _dump(res, out)
    return res


def _mean_se(values):
    a = np.asarray(values, float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("metric values must be non-empty and finite")
    se = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return float(a.mean()), se


_COMPARABILITY_FIELDS = (
    "sim", "mode", "noise", "kap", "budget", "r_k", "grid_n", "warmup",
    "empty_gate", "software_version", "core_source_sha256", "allow_ungated_diagnostic", *PROTOCOL_CONFIG_FIELDS,
)


def _result_value(row, field):
    """Read design metadata with explicit defaults for legacy ``lhs_fixed`` records."""
    value = row.get(field, _DESIGN_DEFAULTS.get(field))
    if value is None and field not in row:
        raise KeyError(field)
    return float(value) if field in {"kap", "region_step"} else value


def _guard_group_designs(results, by, caller):
    """Refuse to average different trial designs inside one displayed group."""
    groups = {}
    for i, row in enumerate(results):
        missing = [field for field in by if field not in row]
        missing += [field for field in _COMPARABILITY_FIELDS
                    if field not in _DESIGN_DEFAULTS and field not in row]
        if missing:
            raise ValueError(f"result row {i} is missing fields {sorted(set(missing))}")
        key = tuple(row[field] for field in by)
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        for field in _COMPARABILITY_FIELDS:
            if field in by:
                continue
            values = {_result_value(row, field) for row in rows}
            if len(values) > 1:
                raise ValueError(
                    f"{caller} cannot pool different {field!r} values; include {field!r} "
                    "in by or filter the input records"
                )


def summarize(results, metric="dose_units", by=("sim", "policy", "gamma")):
    """Mean and seed-clustered Monte Carlo SE of ``metric`` grouped by ``by``.

    Multiple strata, gates, or other rows from one seed are averaged before the
    SE is computed. Returns a pandas DataFrame if available, else a list of dicts.
    """
    results = list(results)
    _guard_group_designs(results, tuple(by), "summarize")
    groups = {}
    for r in results:
        key = tuple(r[b] for b in by)
        groups.setdefault(key, {}).setdefault(r["seed"], []).append(r[metric])
    rows = []
    for key, seed_values in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        vals = [float(np.mean(v)) for _, v in sorted(seed_values.items())]
        mean, se = _mean_se(vals)
        row = dict(zip(by, key))
        row[f"{metric}_mean"] = mean
        row[f"{metric}_se"] = se
        row["n"] = len(vals)
        rows.append(row)
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows


def leaderboard(results, metric="dose_units", sim=None, lower_is_better=True,
                recommendation_conditional=False):
    """Print a descriptive policy ranking with seed-clustered Monte Carlo SE.

    Set ``recommendation_conditional=True`` for a metric such as ``dose_units``
    that is undefined after a stop with no final selection. Such rows are excluded rather
    than imputed. The all-trial unsafe indicator treats a stopped trial as zero.
    """
    rs = [r for r in results if (sim is None or r["sim"] == sim)]
    _guard_group_designs(rs, (), "leaderboard")
    agg = {}
    for r in rs:
        value = r.get(metric)
        if metric == "rec_unsafe" and not bool(r.get("recommendation_made", True)):
            value = 0.0
        if value is None:
            if recommendation_conditional:
                continue
            raise ValueError(
                f"metric {metric!r} is undefined for at least one trial; set "
                "recommendation_conditional=True or use oc_table()"
            )
        if not np.isfinite(float(value)):
            raise ValueError(f"metric {metric!r} must be finite")
        agg.setdefault(r["policy"], {}).setdefault(r["seed"], []).append(float(value))
    ranked = []
    for policy, seed_values in agg.items():
        vals = [float(np.mean(v)) for _, v in sorted(seed_values.items())]
        mean, se = _mean_se(vals)
        ranked.append((policy, mean, se, len(vals)))
    if not ranked:
        raise ValueError("no eligible metric values remain after conditioning")
    ranked.sort(key=lambda t: t[1], reverse=not lower_is_better)
    title = f"{metric}  ({'lower' if lower_is_better else 'higher'} is better)"
    if sim:
        title += f"  [surface={sim}]"
    print(title)
    print("-" * len(title))
    for p, m, se, n in ranked:
        print(f"  {p:<12} {m:8.3f} +/- {se:.3f}   (n={n} seeds)")
    return ranked


def paired_contrast(results, policy_a, policy_b, metric="dose_units",
                    by=("sim", "stratum", "gamma"), *,
                    recommendation_conditional=False):
    """Return paired ``policy_a - policy_b`` contrasts and seed-clustered MCSE.

    Pairing uses the complete trial-design identity (including seed, surface,
    stratum, gate, mode, and design knobs). Missing or duplicate partners fail
    loudly. If ``by`` pools multiple cells per seed, their differences are averaged
    before the Monte Carlo SE is computed. Set ``recommendation_conditional=True``
    for selection-conditional estimands; a seed/design cell then contributes only if
    both policies made a final selection. ``n_pairs`` reports the eligible
    seed-by-design cells and ``n`` reports independent seed-level averages. For the
    all-trial ``rec_unsafe`` indicator,
    a stopped trial contributes zero even though its stored recommendation field is
    ``None``.
    """
    if policy_a == policy_b:
        raise ValueError("paired_contrast requires two distinct policies")
    design_fields = ("seed", "sim", "stratum", "gamma", "mode", "budget", "r_k",
                     "grid_n", "noise", "kap", "warmup", "empty_gate",
                     "protocol_scaffold", "region_step", "empty_gate_stop_after",
                     "exclude_repeats_during_expansion")
    chosen = [r for r in results if r.get("policy") in {policy_a, policy_b}]
    _guard_group_designs(chosen, tuple(by), "paired_contrast")
    cells = {}
    for r in chosen:
        missing = [k for k in (*design_fields, metric)
                   if k not in _DESIGN_DEFAULTS and k not in r]
        if missing:
            raise ValueError(f"paired result row is missing fields {missing}")
        key = tuple(_result_value(r, k) for k in design_fields)
        slot = cells.setdefault(key, {})
        if r["policy"] in slot:
            raise ValueError(f"duplicate paired cell for {r['policy']!r}: {key}")
        slot[r["policy"]] = r
    if not cells:
        raise ValueError("no rows found for requested policies")
    incomplete = [k for k, v in cells.items() if set(v) != {policy_a, policy_b}]
    if incomplete:
        raise ValueError(f"unpaired policy cells; example: {incomplete[0]}")
    def metric_value(row):
        made = bool(row.get("recommendation_made", True))
        if recommendation_conditional and not made:
            return None
        value = row.get(metric)
        if metric == "rec_unsafe" and not made:
            return 0.0
        if value is None or not np.isfinite(float(value)):
            raise ValueError(f"paired metric {metric!r} must be finite for eligible rows")
        return float(value)

    groups = {}
    all_groups = set()
    for key, pair in cells.items():
        ref = pair[policy_a]
        group = tuple(ref[k] for k in by)
        all_groups.add(group)
        seed = ref["seed"]
        left, right = metric_value(pair[policy_a]), metric_value(pair[policy_b])
        if left is None or right is None:
            continue
        diff = left - right
        groups.setdefault(group, {}).setdefault(seed, []).append(diff)
    rows = []
    for key in sorted(all_groups, key=lambda value: [str(x) for x in value]):
        seed_values = groups.get(key, {})
        vals = [float(np.mean(v)) for _, v in sorted(seed_values.items())]
        n_pairs = sum(len(v) for v in seed_values.values())
        mean, se = _mean_se(vals) if vals else (np.nan, np.nan)
        row = dict(zip(by, key))
        row.update(policy_a=policy_a, policy_b=policy_b, metric=metric,
                   recommendation_conditional=bool(recommendation_conditional),
                   difference_mean=mean, difference_se=se, n=len(vals),
                   n_pairs=n_pairs)
        rows.append(row)
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows


def scaffold_interaction(results, policy_a, policy_b, metric="dose_units", *,
                         scaffold_a="start_low_expansion", scaffold_b="lhs_fixed",
                         by=("sim", "stratum", "gamma"),
                         recommendation_conditional=False):
    """Paired difference-in-differences across two dose-availability designs.

    The returned contrast is ``[(policy_a-policy_b)_scaffold_a] -
    [(policy_a-policy_b)_scaffold_b]``. Every included seed must contain the full
    two-policy by two-scaffold block under otherwise identical design metadata.
    Selection-conditional interactions require all four trials to make a
    final selection. ``n_pairs`` and ``n`` report eligible blocks and independent
    seed-level averages, respectively.
    """
    if policy_a == policy_b:
        raise ValueError("scaffold_interaction requires two distinct policies")
    if scaffold_a == scaffold_b:
        raise ValueError("scaffold_interaction requires two distinct access protocols")
    chosen = [r for r in results
              if r.get("policy") in {policy_a, policy_b}
              and r.get("protocol_scaffold", "lhs_fixed") in {scaffold_a, scaffold_b}]
    if not chosen:
        raise ValueError("no rows found for requested policies and dose-availability designs")
    _guard_group_designs(chosen, tuple((*by, "protocol_scaffold")),
                         "scaffold_interaction")
    base_fields = ("seed", "sim", "stratum", "gamma", "mode", "budget", "r_k",
                   "grid_n", "noise", "kap", "warmup", "empty_gate", "region_step",
                   "empty_gate_stop_after", "exclude_repeats_during_expansion")
    cells = {}
    for row in chosen:
        missing = [field for field in (*base_fields, metric)
                   if field not in _DESIGN_DEFAULTS and field not in row]
        if missing:
            raise ValueError(f"interaction row is missing fields {missing}")
        key = tuple(_result_value(row, field) for field in base_fields)
        slot = cells.setdefault(key, {})
        scaffold = row.get("protocol_scaffold", "lhs_fixed")
        cell_name = (scaffold, row["policy"])
        if cell_name in slot:
            raise ValueError(f"duplicate interaction cell {cell_name}: {key}")
        slot[cell_name] = row
    required = {(s, p) for s in (scaffold_a, scaffold_b)
                for p in (policy_a, policy_b)}
    incomplete = [key for key, values in cells.items() if set(values) != required]
    if incomplete:
        raise ValueError(f"incomplete scaffold interaction block; example: {incomplete[0]}")

    def value(row):
        made = bool(row.get("recommendation_made", True))
        if recommendation_conditional and not made:
            return None
        raw = row.get(metric)
        if metric == "rec_unsafe" and not made:
            return 0.0
        if raw is None or not np.isfinite(float(raw)):
            raise ValueError(f"interaction metric {metric!r} must be finite for eligible rows")
        return float(raw)

    groups, all_groups = {}, set()
    for block in cells.values():
        ref = block[(scaffold_a, policy_a)]
        group = tuple(ref[field] for field in by)
        all_groups.add(group)
        values = [value(block[(s, p)]) for s in (scaffold_a, scaffold_b)
                  for p in (policy_a, policy_b)]
        if any(item is None for item in values):
            continue
        a_a, a_b, b_a, b_b = values
        interaction = (a_a - a_b) - (b_a - b_b)
        groups.setdefault(group, {}).setdefault(ref["seed"], []).append(interaction)

    rows = []
    for key in sorted(all_groups, key=lambda item: [str(x) for x in item]):
        seed_values = groups.get(key, {})
        values = [float(np.mean(v)) for _, v in sorted(seed_values.items())]
        n_pairs = sum(len(v) for v in seed_values.values())
        mean, se = _mean_se(values) if values else (np.nan, np.nan)
        row = dict(zip(by, key))
        row.update(policy_a=policy_a, policy_b=policy_b, metric=metric,
                   scaffold_a=scaffold_a, scaffold_b=scaffold_b,
                   recommendation_conditional=bool(recommendation_conditional),
                   interaction_mean=mean, interaction_se=se, n=len(values),
                   n_pairs=n_pairs)
        rows.append(row)
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows
