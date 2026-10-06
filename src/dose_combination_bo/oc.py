"""Operating-characteristic summaries for simulated trials.

This module is deliberately post-processing only.  It consumes the per-trial
records returned by :func:`dose_combination_bo.sweep` and never fits a model, changes a
final selection, or runs a simulation.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import warnings

import numpy as np

from .surfaces import resolve_surface
from .trial import grid
from .protocol import PROTOCOL_DEFAULTS


_OC_METRICS = (
    "recommendation_pct",
    "no_feasible_dose_pct",
    "pcs",
    "pcs_within1",
    "rec_unsafe_pct",
    "rec_efficacy",
    "efficacy_regret",
    "dose_units",
    "unsafe_given_recommendation_pct",
    "expected_n",
    "patients_above",
    "pct_patients_above",
    "initialization_patients_above",
    "post_initialization_patients_above",
    "n_unique_doses",
    "n_empty_gate_events",
    "overdose60",
    "overdose80",
    "gate_size",
    "gate_empty_pct",
    "gate_precision",
)

_OC_REQUIRED = {
    "policy", "seed", "sim", "stratum", "gamma", "grid_n", "budget",
    "dose_units", "rec_d1", "rec_d2", "rec_true_eff", "rec_true_tox",
    "rec_unsafe", "toxic", "n_gate_pass", "n_gate_pass_safe",
}

_SELECTION_REQUIRED = {
    "policy", "seed", "sim", "stratum", "gamma", "grid_n", "rec_d1", "rec_d2",
}

_DESIGN_FIELDS = (
    ("sim", ("sim",)),
    ("mode", ("mode",)),
    ("noise", ("noise",)),
    ("kap", ("kap", "kappa", "κ")),
    ("budget", ("budget",)),
    ("r_k", ("r_k",)),
    ("grid_n", ("grid_n",)),
    ("warmup", ("warmup",)),
    ("empty_gate", ("empty_gate",)),
    ("protocol_scaffold", ("protocol_scaffold",)),
    ("region_step", ("region_step",)),
    ("empty_gate_stop_after", ("empty_gate_stop_after",)),
    ("exclude_repeats_during_expansion", ("exclude_repeats_during_expansion",)),
)

_DESIGN_DEFAULTS = {
    "kap": 1.0,
    **PROTOCOL_DEFAULTS,
}

_MISSING = object()


def _records(results):
    if hasattr(results, "to_dict"):
        rows = results.to_dict("records")
    else:
        try:
            rows = list(results)
        except TypeError as exc:
            raise TypeError("results must be an iterable of per-trial mappings") from exc
    if not rows:
        raise ValueError("results must contain at least one per-trial record")
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("every result row must be a mapping")
    return rows


def _require_fields(rows, required):
    for i, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"result row {i} is missing fields {missing}")


def _same_number(left, right):
    try:
        return bool(np.isclose(float(left), float(right), rtol=0, atol=1e-12))
    except (TypeError, ValueError):
        return False


def _row_design_value(row, aliases):
    values = [row[name] for name in aliases if name in row]
    if not values:
        return _MISSING
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"result row has conflicting aliases for {aliases[0]!r}")
    return first


def _filter(rows, *, sim=None, gamma=None, z=None, mode=None, noise=None, kap=None,
            budget=None, r_k=None, grid_n=None, warmup=None, empty_gate=None,
            protocol_scaffold=None, region_step=None, empty_gate_stop_after=None,
            exclude_repeats_during_expansion=None):
    if gamma is not None and (not np.isscalar(gamma) or not np.isfinite(float(gamma))):
        raise ValueError("gamma must be a finite scalar")
    if kap is not None and (not np.isscalar(kap) or not np.isfinite(float(kap))
                            or float(kap) <= 0):
        raise ValueError("kap must be a finite positive scalar")
    design_filters = {
        "mode": mode,
        "noise": noise,
        "budget": budget,
        "r_k": r_k,
        "grid_n": grid_n,
        "warmup": warmup,
        "empty_gate": empty_gate,
        "protocol_scaffold": protocol_scaffold,
        "region_step": region_step,
        "empty_gate_stop_after": empty_gate_stop_after,
        "exclude_repeats_during_expansion": exclude_repeats_during_expansion,
    }
    selected = []
    for row in rows:
        if sim is not None and row.get("sim") != sim:
            continue
        if gamma is not None:
            try:
                same_gamma = np.isclose(float(row.get("gamma")), float(gamma), rtol=0, atol=1e-12)
            except (TypeError, ValueError):
                same_gamma = False
            if not same_gamma:
                continue
        if z is not None and row.get("stratum") != z:
            continue
        row_kap = _row_design_value(row, ("kap", "kappa", "κ"))
        if row_kap is _MISSING:
            row_kap = 1.0  # records created before the public kap input used the calibration
        if kap is not None and not _same_number(row_kap, kap):
            continue
        mismatch = False
        for field, value in design_filters.items():
            if value is None:
                continue
            row_value = row.get(field, _DESIGN_DEFAULTS.get(field, _MISSING))
            if field == "region_step":
                if not _same_number(row_value, value):
                    mismatch = True
                    break
            elif row_value != value:
                mismatch = True
                break
        if mismatch:
            continue
        selected.append(row)
    if not selected:
        details = [f"sim={sim!r}" if sim is not None else None,
                   f"z={z!r}" if z is not None else None,
                   f"gamma={gamma!r}" if gamma is not None else None,
                   f"kap={kap!r}" if kap is not None else None]
        details.extend(f"{field}={value!r}" for field, value in design_filters.items()
                       if value is not None)
        raise ValueError("no result rows match " + ", ".join(x for x in details if x))
    return selected


def _group_fields(by):
    if isinstance(by, str):
        raise TypeError("by must be a sequence of field names, not a string")
    try:
        fields = tuple(by)
    except TypeError as exc:
        raise TypeError("by must be a sequence of field names") from exc
    if not all(isinstance(field, str) and field for field in fields):
        raise TypeError("every by field must be a non-empty string")
    if len(fields) != len(set(fields)):
        raise ValueError("by contains duplicate fields")
    return fields if "policy" in fields else ("policy", *fields)


def _hashable(value):
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _guard_comparable_designs(rows, fields):
    """Refuse accidental pooling over trial designs with different semantics."""
    grouped = set(fields)
    for canonical, aliases in _DESIGN_FIELDS:
        if grouped.intersection(aliases):
            continue
        values = [_row_design_value(row, aliases) for row in rows]
        if canonical in _DESIGN_DEFAULTS:
            values = [_DESIGN_DEFAULTS[canonical] if value is _MISSING else value
                      for value in values]
        present = [value for value in values if value is not _MISSING]
        if not present:
            continue
        distinct = {_hashable(value) for value in present}
        if len(present) != len(values):
            distinct.add("<missing>")
        if len(distinct) > 1:
            raise ValueError(
                f"cannot pool rows with different {canonical!r} values; "
                f"filter {canonical!r} or include it in by"
            )


def _mean_mcse(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("summary values must be non-empty and finite")
    mcse = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return float(values.mean()), mcse


def _policy_label(policy):
    labels = {
        "random": "random among eligible combinations",
        "cKG": "cKG",
        "cKG-exact": "cKG",
        "cKG-exact-formal": "cKG",
        "cEI-tMSE": "cEI–tMSE",
        "cEI-tMSE-1to1": "cEI–tMSE",
        "cEI-tMSE-2to1": "cEI-heavy (12 cEI, 6 tMSE cohorts)",
    }
    return labels.get(policy, str(policy))


def _warn_without_random(rows):
    if not any(row["policy"] == "random" for row in rows):
        warnings.warn(
            "no 'random' policy rows were supplied; the summary has no random-policy row",
            UserWarning,
            stacklevel=3,
        )


def _grid_index(row, points):
    rec = np.asarray((row["rec_d1"], row["rec_d2"]), dtype=float)
    if rec.shape != (2,) or not np.isfinite(rec).all():
        raise ValueError("final selected doses must be finite two-coordinate values")
    distances = np.linalg.norm(points - rec, axis=1)
    index = int(np.argmin(distances))
    if distances[index] > 1e-9:
        raise ValueError(
            f"final selected dose {tuple(rec)} is not on the reported {row['grid_n']}x{row['grid_n']} grid"
        )
    return index


def _derive(row, caches):
    try:
        z = int(row["stratum"])
        grid_n = int(row["grid_n"])
        budget = int(row["budget"])
        toxic = int(row["toxic"])
        n_gate_pass = int(row["n_gate_pass"])
        n_gate_pass_safe = int(row["n_gate_pass_safe"])
    except (TypeError, ValueError) as exc:
        raise ValueError("summary fields must be numeric") from exc
    recommendation_made = row.get("recommendation_made", True)
    if not isinstance(recommendation_made, (bool, np.bool_)):
        raise ValueError("recommendation_made must be boolean")
    recommendation_made = bool(recommendation_made)
    try:
        n_enrolled = int(row.get("n_enrolled", budget))
    except (TypeError, ValueError) as exc:
        raise ValueError("n_enrolled must be an integer") from exc
    if grid_n < 2 or budget <= 0 or not 0 < n_enrolled <= budget:
        raise ValueError("grid_n must be at least 2 and budget must be positive")
    if not 0 <= toxic <= n_enrolled:
        raise ValueError("toxic must lie between zero and n_enrolled")
    if not 0 <= n_gate_pass_safe <= n_gate_pass <= grid_n ** 2:
        raise ValueError("gate counts are inconsistent with grid_n")

    spec_key = (row["sim"], z)
    specs = caches.setdefault("spec", {})
    if spec_key not in specs:
        specs[spec_key] = resolve_surface(row["sim"], z)
    spec = specs[spec_key]

    stop_reason = row.get("stop_reason")
    no_feasible = stop_reason == "NO_FEASIBLE_DOSE"
    if recommendation_made and no_feasible:
        raise ValueError("a NO_FEASIBLE_DOSE trial cannot have a final selection")
    if not recommendation_made and stop_reason not in {"NO_FEASIBLE_DOSE"}:
        raise ValueError("a trial without a final selection must report NO_FEASIBLE_DOSE")

    rec_index = target_index = None
    dose_units = rec_true_eff = rec_true_tox = rec_unsafe = None
    if recommendation_made:
        try:
            dose_units = float(row["dose_units"])
            rec_true_eff = float(row["rec_true_eff"])
            rec_true_tox = float(row["rec_true_tox"])
            rec_unsafe = int(row["rec_unsafe"])
        except (TypeError, ValueError) as exc:
            raise ValueError("final-selection fields must be numeric when a final selection exists") from exc
        if not np.isfinite([dose_units, rec_true_eff, rec_true_tox]).all() or dose_units < 0:
            raise ValueError("dose_units and final-selection truth fields must be finite")
        if rec_unsafe not in (0, 1):
            raise ValueError("rec_unsafe must be binary")
        if rec_unsafe != int(rec_true_tox > float(spec["gd"])):
            raise ValueError("rec_unsafe is inconsistent with rec_true_tox and the surface threshold")
        grids = caches.setdefault("grid", {})
        if grid_n not in grids:
            grids[grid_n] = grid(grid_n)
        points = grids[grid_n]
        rec_index = _grid_index(row, points)
        target_key = (row["sim"], z, grid_n)
        targets = caches.setdefault("target", {})
        if target_key not in targets:
            targets[target_key] = int(np.argmin(
                np.linalg.norm(points - np.asarray(spec["dopt"], dtype=float), axis=1)
            ))
        target_index = targets[target_key]
    else:
        for field in ("dose_units", "rec_d1", "rec_d2", "rec_true_eff",
                      "rec_true_tox", "rpsel"):
            if row.get(field) is not None:
                raise ValueError(f"stopped trial must not contain a terminal {field}")
        if row.get("rec_unsafe") not in (None, 0, False):
            raise ValueError(
                "stopped trial cannot contain an above-limit final selection"
            )

    def optional_int(name, default=None, lower=0, upper=None):
        raw = row.get(name, default)
        if raw is None:
            return None
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer")
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if value != raw or value < lower or (upper is not None and value > upper):
            raise ValueError(f"{name} is outside its valid range")
        return value

    initialization_size = optional_int(
        "initialization_size", row.get("warmup"), lower=1, upper=n_enrolled
    )
    initialization_above = optional_int(
        "initialization_patients_above", lower=0, upper=initialization_size
    ) if initialization_size is not None else None
    post_above = optional_int(
        "post_initialization_patients_above", lower=0,
        upper=(n_enrolled - initialization_size if initialization_size is not None else n_enrolled),
    )
    if initialization_above is not None and post_above is not None:
        if initialization_above + post_above != toxic:
            raise ValueError("initialization and post-initialization exposure must sum to toxic")
    n_unique = optional_int("n_unique_doses", lower=1, upper=n_enrolled)
    n_empty = optional_int("n_empty_gate_events", lower=0)

    patient_fraction = toxic / n_enrolled
    return {
        "recommendation_pct": int(recommendation_made),
        "no_feasible_dose_pct": int(no_feasible),
        "pcs": int(recommendation_made and rec_index == target_index),
        "pcs_within1": int(recommendation_made and dose_units <= 1.0),
        "rec_unsafe_pct": int(rec_unsafe) if recommendation_made else 0,
        "rec_efficacy": rec_true_eff,
        "efficacy_regret": (float(spec["fopt"]) - rec_true_eff
                            if recommendation_made else None),
        "dose_units": dose_units,
        "unsafe_given_recommendation_pct": rec_unsafe,
        "expected_n": n_enrolled,
        "patients_above": toxic,
        "pct_patients_above": 100.0 * patient_fraction,
        "initialization_patients_above": initialization_above,
        "post_initialization_patients_above": post_above,
        "n_unique_doses": n_unique,
        "n_empty_gate_events": n_empty,
        "overdose60": int(patient_fraction > 0.60),
        "overdose80": int(patient_fraction > 0.80),
        "gate_size": n_gate_pass,
        "gate_empty_pct": int(n_gate_pass == 0),
        "gate_precision": (n_gate_pass_safe / n_gate_pass if n_gate_pass else None),
    }


def _table(rows):
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows


def oc_table(results, sim=None, gamma=None, by=("policy",), *, mode=None, noise=None,
             kap=None, budget=None, r_k=None, grid_n=None, warmup=None,
             empty_gate=None, protocol_scaffold=None, region_step=None,
             empty_gate_stop_after=None, exclude_repeats_during_expansion=None):
    """Summarize operating characteristics from saved simulated-trial records.

    ``policy`` is always a grouping field; any additional fields in ``by`` subdivide
    each policy. Percent-valued metrics and their MCSEs are on a 0--100 scale.
    Rates defined over initiated trials include every simulated trial in their
    denominator. In particular, a stopped trial contributes zero to ``pcs``,
    ``pcs_within1``, and ``rec_unsafe_pct``. An above-threshold combination is one
    whose true mean toxicity exceeds the prespecified simulation threshold;
    above-threshold outputs do not count observed toxicity events. True mean efficacy
    at final selection, dose-location error, and the selection-conditional
    probability of an above-threshold final selection exclude stopped trials and carry
    explicit ``*_n``/``*_n_rows`` denominators. ``gate_precision`` is on a 0--1 scale:
    for each trial with at
    least one combination classified as acceptable, it computes the proportion of
    those combinations whose true mean toxicity does not exceed the threshold, then
    averages within seed and across seeds.

    The returned object is a pandas DataFrame when pandas is installed, otherwise a
    list of dictionaries. ``pcs``, ``pcs_within1``, and ``dose_units`` use the
    stored continuous-domain reference supplied by the surface and are retained
    as legacy location diagnostics; they do not define the finite-grid trial
    target. Values are not rounded. Rows sharing a ``seed`` are averaged
    before the MCSE is computed, and ``n`` is the number of independent seeds (``n_rows``
    is the raw row count). The ``random`` policy, when supplied, is labelled
    ``random among eligible combinations``; it is not a lower bound on achievable
    performance. Scheduled final-selection outputs include the largest-margin
    empty-set fallback. That fallback does not meet the model-based toxicity criterion,
    and any final selection may be untried because no minimum-exposure rule is imposed.
    """
    rows = _filter(
        _records(results), sim=sim, gamma=gamma, mode=mode, noise=noise, kap=kap,
        budget=budget, r_k=r_k, grid_n=grid_n, warmup=warmup, empty_gate=empty_gate,
        protocol_scaffold=protocol_scaffold, region_step=region_step,
        empty_gate_stop_after=empty_gate_stop_after,
        exclude_repeats_during_expansion=exclude_repeats_during_expansion,
    )
    _require_fields(rows, _OC_REQUIRED)
    fields = _group_fields(by)
    _guard_comparable_designs(rows, fields)
    for i, row in enumerate(rows):
        missing = [field for field in fields
                   if field not in row and field not in _DESIGN_DEFAULTS]
        if missing:
            raise ValueError(f"result row {i} is missing grouping fields {missing}")
    _warn_without_random(rows)

    caches = {}
    groups = {}
    for row in rows:
        key = tuple(row.get(field, _DESIGN_DEFAULTS.get(field)) for field in fields)
        groups.setdefault(key, defaultdict(list))[row["seed"]].append(_derive(row, caches))

    output = []
    sort_key = lambda item: (item[0][0] == "random", *[str(x) for x in item[0]])
    for key, seed_groups in sorted(groups.items(), key=sort_key):
        result = dict(zip(fields, key))
        result["policy_label"] = _policy_label(result["policy"])
        result["chance_control"] = result["policy"] == "random"
        result["n"] = len(seed_groups)
        result["n_rows"] = sum(len(values) for values in seed_groups.values())
        for metric in ("recommendation_pct", "no_feasible_dose_pct", "pcs",
                       "pcs_within1", "rec_unsafe_pct", "overdose60", "overdose80",
                       "gate_empty_pct"):
            per_seed = [np.mean([value[metric] for value in values])
                        for values in seed_groups.values()]
            result[metric], result[f"{metric}_mcse"] = _mean_mcse(per_seed)
            result[metric] *= 100.0
            result[f"{metric}_mcse"] *= 100.0
        for metric in ("expected_n", "patients_above", "pct_patients_above", "gate_size"):
            per_seed = [np.mean([value[metric] for value in values])
                        for values in seed_groups.values()]
            result[metric], result[f"{metric}_mcse"] = _mean_mcse(
                per_seed
            )

        def add_subset_metric(metric, *, percent=False):
            per_seed = []
            n_rows = 0
            for values in seed_groups.values():
                valid = [value[metric] for value in values if value[metric] is not None]
                if valid:
                    per_seed.append(float(np.mean(valid)))
                    n_rows += len(valid)
            result[f"{metric}_n"] = len(per_seed)
            result[f"{metric}_n_rows"] = n_rows
            if per_seed:
                value, mcse = _mean_mcse(per_seed)
                scale = 100.0 if percent else 1.0
                result[metric] = scale * value
                result[f"{metric}_mcse"] = scale * mcse
            else:
                result[metric] = np.nan
                result[f"{metric}_mcse"] = np.nan

        for metric in ("rec_efficacy", "efficacy_regret", "dose_units"):
            add_subset_metric(metric)
        add_subset_metric("unsafe_given_recommendation_pct", percent=True)
        for metric in ("initialization_patients_above",
                       "post_initialization_patients_above", "n_unique_doses",
                       "n_empty_gate_events"):
            add_subset_metric(metric)
        gate_precision = []
        gate_precision_n_rows = 0
        for values in seed_groups.values():
            nonempty = [value["gate_precision"] for value in values
                        if value["gate_precision"] is not None]
            if nonempty:
                gate_precision.append(float(np.mean(nonempty)))
                gate_precision_n_rows += len(nonempty)
        result["gate_precision_n"] = len(gate_precision)
        result["gate_precision_n_rows"] = gate_precision_n_rows
        if gate_precision:
            result["gate_precision"], result["gate_precision_mcse"] = _mean_mcse(
                gate_precision
            )
        else:
            result["gate_precision"] = np.nan
            result["gate_precision_mcse"] = np.nan
        output.append(result)
    return _table(output)


def selection_table(results, sim, z, gamma, *, mode=None, noise=None, kap=None,
                    budget=None, r_k=None, grid_n=None, warmup=None,
                    empty_gate=None, protocol_scaffold=None, region_step=None,
                    empty_gate_stop_after=None,
                    exclude_repeats_during_expansion=None):
    """Return a wide table of final-selection frequencies by dose combination.

    There is one row per combination on the administrable grid and a final-selection
    percentage plus seed-clustered MCSE column for each supplied policy. The
    ``random`` policy column, when present, is labelled ``random among eligible
    combinations``, not a performance lower bound. Returned values are unrounded.
    Policy-prefixed ``*_n`` and ``*_n_rows`` columns give the independent-seed and
    selection counts. The stored ``true_obd`` flag marks the grid point nearest
    the legacy continuous-domain reference, not the finite-grid constrained optimum.
    Stopped trials are excluded from the final-selection distribution rather than
    assigned a fictitious combination; ``*_trial_n_rows`` and
    ``*_recommendation_pct`` retain the initiated-trial denominator.
    Scheduled final selections include the largest-margin empty-set fallback. That
    fallback does not meet the model-based toxicity criterion, and any final selection may
    be untried because no minimum-exposure rule is imposed.
    """
    rows = _filter(
        _records(results), sim=sim, gamma=gamma, z=z, mode=mode, noise=noise,
        kap=kap, budget=budget, r_k=r_k, grid_n=grid_n, warmup=warmup,
        empty_gate=empty_gate,
        protocol_scaffold=protocol_scaffold, region_step=region_step,
        empty_gate_stop_after=empty_gate_stop_after,
        exclude_repeats_during_expansion=exclude_repeats_during_expansion,
    )
    _require_fields(rows, _SELECTION_REQUIRED)
    _guard_comparable_designs(rows, ())
    _warn_without_random(rows)
    grid_sizes = {int(row["grid_n"]) for row in rows}
    if len(grid_sizes) != 1:
        raise ValueError("selection_table requires one common grid_n")
    grid_n = next(iter(grid_sizes))
    if grid_n < 2:
        raise ValueError("grid_n must be at least 2")

    spec = resolve_surface(sim, int(z))
    points = grid(grid_n)
    target_index = int(np.argmin(np.linalg.norm(
        points - np.asarray(spec["dopt"], dtype=float), axis=1
    )))
    output = []
    for index, (d1, d2) in enumerate(points):
        true_tox = float(spec["tox"](float(d1), float(d2)))
        output.append({
            "d1": float(d1),
            "d2": float(d2),
            "true_eff": float(spec["eff"](float(d1), float(d2))),
            "true_tox": true_tox,
            "feasible": bool(true_tox <= float(spec["gd"])),
            "true_obd": index == target_index,
        })

    policies = sorted({row["policy"] for row in rows}, key=lambda p: (p == "random", str(p)))
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        recommendation_rows = [row for row in policy_rows
                               if bool(row.get("recommendation_made", True))]
        seed_rows = defaultdict(list)
        for row in recommendation_rows:
            seed_rows[row["seed"]].append(row)
        seed_fractions = []
        for values in seed_rows.values():
            counts = np.zeros(len(points), dtype=float)
            for row in values:
                counts[_grid_index(row, points)] += 1.0
            seed_fractions.append(counts / len(values))
        seed_fractions = np.asarray(seed_fractions, dtype=float)
        n = len(seed_rows)
        label = _policy_label(policy)
        pct_name = f"{label}_selected_pct"
        mcse_name = f"{label}_selected_pct_mcse"
        recommendation_pct = 100.0 * len(recommendation_rows) / len(policy_rows)
        for index in range(len(points)):
            if n:
                p, mcse = _mean_mcse(seed_fractions[:, index])
                output[index][pct_name] = 100.0 * p
                output[index][mcse_name] = 100.0 * mcse
            else:
                output[index][pct_name] = np.nan
                output[index][mcse_name] = np.nan
            output[index][f"{label}_n"] = n
            output[index][f"{label}_n_rows"] = len(recommendation_rows)
            output[index][f"{label}_trial_n_rows"] = len(policy_rows)
            output[index][f"{label}_recommendation_pct"] = recommendation_pct
        if n and not np.isclose(sum(row[pct_name] for row in output), 100.0,
                                rtol=0, atol=1e-10):
            raise AssertionError(
                f"final-selection percentages do not sum to 100 for {policy!r}"
            )
    return _table(output)


def _latex_escape(value):
    text = str(value)
    for source, target in (("\\", r"\textbackslash{}"), ("_", r"\_"),
                           ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(source, target)
    return text


def oc_latex(table):
    """Print and return a compact LaTeX tabular for :func:`oc_table` output."""
    rows = _records(table)
    first = rows[0]
    metric_columns = set(_OC_METRICS)
    metric_columns.update(f"{name}_mcse" for name in _OC_METRICS)
    excluded = metric_columns | {
        "policy_label", "chance_control", "n", "n_rows", "gate_precision_n",
        "gate_precision_n_rows",
    }
    excluded.update(name for name in first if name.endswith("_n") or name.endswith("_n_rows"))
    groups = [name for name in first if name not in excluded]
    display_groups = [name for name in groups if name != "policy"]
    if "policy" in groups:
        display_groups.insert(0, "policy_label")
    headers = [name.replace("_", " ") for name in display_groups]
    display_metrics = (
        "recommendation_pct", "no_feasible_dose_pct", "pcs", "pcs_within1",
        "rec_unsafe_pct", "rec_efficacy", "dose_units", "pct_patients_above",
        "expected_n", "n_unique_doses", "n_empty_gate_events",
    )
    headers += [
        "Final selection made (\\%)", "Stopped, no final selection (\\%)",
        "Legacy exact-reference match (\\%)",
        "Within 1 grid unit of legacy reference (\\%)",
        "Final selection above toxicity limit (\\%)",
        "True mean efficacy at final selection", "Legacy reference distance",
        "Above-limit records or participants (design dependent; \\%)",
        "Mean response records or enrollment (design dependent)",
        "Distinct response locations or assigned combinations (design dependent)",
        "Empty qualifying-set occasions",
    ]
    align = "l" * len(display_groups) + "r" * len(display_metrics)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\hline", " & ".join(headers) + r" \\", "\\hline"]
    digits = {
        "recommendation_pct": 1, "no_feasible_dose_pct": 1, "pcs": 1,
        "pcs_within1": 1, "rec_unsafe_pct": 1, "rec_efficacy": 3,
        "dose_units": 2, "pct_patients_above": 1, "expected_n": 1,
        "n_unique_doses": 1, "n_empty_gate_events": 1,
    }
    for row in rows:
        cells = [_latex_escape(row[name]) for name in display_groups]
        for metric in display_metrics:
            value, mcse = float(row[metric]), float(row[f"{metric}_mcse"])
            if np.isnan(value) or np.isnan(mcse):
                cells.append("--")
            else:
                d = digits[metric]
                cells.append(f"${value:.{d}f}\\pm{mcse:.{d}f}$")
        lines.append(" & ".join(cells) + r" \\")
    lines += ["\\hline", "\\end{tabular}"]
    latex = "\n".join(lines)
    print(latex)
    return latex


__all__ = ["oc_table", "selection_table", "oc_latex"]
