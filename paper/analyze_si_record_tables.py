#!/usr/bin/env python3
"""Regenerate every record-derived supporting-information table.

The input is one authenticated formal record projection.  This module performs
post-processing only: it neither imports the trial harness nor runs a
simulation.  Every displayed estimate is derived from the projected records,
and every output receives a portable provenance sidecar.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

import dose_combination_bo as sdb
from dose_combination_bo.surfaces import resolve_surface

try:
    from paper.analysis_record_bundle import ProjectionBundle, load_projection_bundle
except ModuleNotFoundError:  # direct execution from paper/
    from analysis_record_bundle import ProjectionBundle, load_projection_bundle


MAIN_POLICIES = ("cEI", "cKG", "cEI-tMSE")
DISPLAY_GATES = (0.5, 0.7, 0.9)
FULL_GATES = (0.5, 0.6, 0.7, 0.8, 0.9)
STRATA = (0, 1)
FORMAL_BUDGET = 40
KAPPA_LEVELS = (0.25, 0.5, 1.0, 1.5, 2.0)

OSA_RECORD = "osa_main.json.zst"
GBUMP_RECORD = "gbump_main.json.zst"
EFFTOX_RECORD = "efftox_main.json.zst"
MARIPOSA_RECORD = "mariposa_main.json.zst"
SIXWAY_RECORD = "gate_sixway_supplemental.json.zst"
BASELINE_RECORD = "efftox_mariposa_baselines_supplemental.json.zst"
KAPPA_RECORD = "kappa_sweep_fixed_supplemental.json.zst"
USED_RECORDS = (
    OSA_RECORD,
    GBUMP_RECORD,
    EFFTOX_RECORD,
    MARIPOSA_RECORD,
    SIXWAY_RECORD,
    BASELINE_RECORD,
    KAPPA_RECORD,
)

RAW_SIXWAY_POLICIES = ("cEI", "cKG1fix", "GBE", "straddle", "qBIG", "random")
RAW_BASELINE_POLICIES = ("cEI", "cKG1fix", "GBE", "straddle", "qBIG")
RAW_KAPPA_POLICIES = ("cEI", "cKG1fix")
SIXWAY_POLICIES = ("cKG", "SUR", "cEI-tMSE", "tMSE", "cEI", "random-admitted")
BASELINE_POLICIES = ("cEI", "cKG", "cEI-tMSE", "tMSE", "SUR")

POLICY_CANONICAL = {
    "cEI": "cEI",
    "cKG": "cKG",
    "cKG1fix": "cKG",
    "cEI-tMSE": "cEI-tMSE",
    "GBE": "cEI-tMSE",
    "straddle": "tMSE",
    "qBIG": "SUR",
    "random": "random-admitted",
}
POLICY_TEX = {
    "cEI": "cEI",
    "cKG": "cKG",
    "cEI-tMSE": r"cEI--tMSE",
    "tMSE": "tMSE",
    "SUR": "SUR",
    "random-admitted": "random-admitted",
}


@dataclass
class TableSpec:
    """One SI table and the authenticated record subset supporting it."""

    table_id: str
    rows: list[dict[str, Any]]
    fieldnames: tuple[str, ...]
    tex: str
    selected_records: list[dict[str, Any]]
    source_names: tuple[str, ...]
    estimand: dict[str, Any]
    auxiliary: dict[str, Any] | None = None


Metric = str | Callable[[Mapping[str, Any]], float]


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _policy(row: Mapping[str, Any]) -> str:
    try:
        return POLICY_CANONICAL[str(row["policy"])]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown policy code {row.get('policy')!r}") from exc


def _key_value(field: str, value: Any) -> Any:
    if field in {"seed", "stratum", "budget"}:
        return int(value)
    if field in {"gamma", "kap"}:
        return float(value)
    return str(value)


def _validate_factorial(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    key_fields: Sequence[str],
    expected: set[tuple[Any, ...]],
) -> None:
    """Require one and only one row for every declared factorial cell."""

    observed: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        try:
            key = tuple(_key_value(field, row[field]) for field in key_fields)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} row {index} has an invalid design identity") from exc
        if key in observed:
            raise ValueError(f"{label} contains duplicate factorial cell {key}")
        observed.add(key)
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        example = next(iter(missing or extra))
        raise ValueError(
            f"{label} does not equal its declared factorial; "
            f"missing={len(missing)}, extra={len(extra)}, example={example}"
        )


def _require_integer(value: Any, *, label: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label} must be a finite integer")
    return int(number)


def _validate_trial_fields(rows: Sequence[Mapping[str, Any]], *, label: str) -> None:
    """Validate fixed formal-design fields and metric bounds without result pins."""

    for index, row in enumerate(rows):
        prefix = f"{label} row {index}"
        budget = _require_integer(row.get("budget"), label=f"{prefix} budget")
        if budget != FORMAL_BUDGET:
            raise ValueError(
                f"{prefix} budget differs from the frozen formal design: {budget}"
            )
        grid_n = _require_integer(row.get("grid_n"), label=f"{prefix} grid_n")
        warmup = _require_integer(row.get("warmup"), label=f"{prefix} warmup")
        r_k = _require_integer(row.get("r_k"), label=f"{prefix} r_k")
        if (grid_n, warmup, r_k) != (5, 4, 2):
            raise ValueError(f"{prefix} has changed frozen panel/cohort fields")
        if row.get("noise") != "fixed" or row.get("empty_gate") != "pf":
            raise ValueError(f"{prefix} has changed noise or empty-gate semantics")

        toxic = _require_integer(row.get("toxic"), label=f"{prefix} toxic")
        passed = _require_integer(
            row.get("n_gate_pass"), label=f"{prefix} n_gate_pass"
        )
        passed_safe = _require_integer(
            row.get("n_gate_pass_safe"), label=f"{prefix} n_gate_pass_safe"
        )
        if not 0 <= toxic <= budget:
            raise ValueError(f"{prefix} toxic count is outside its row budget")
        if not 0 <= passed_safe <= passed <= grid_n**2:
            raise ValueError(f"{prefix} final-gate counts are inconsistent")
        if float(row.get("rec_unsafe")) not in (0.0, 1.0):
            raise ValueError(f"{prefix} rec_unsafe must be binary")
        for field in (
            "dose_units",
            "rpsel",
            "rec_true_eff",
            "rec_true_tox",
            "rec_d1",
            "rec_d2",
        ):
            value = float(row.get(field))
            if not math.isfinite(value):
                raise ValueError(f"{prefix} {field} must be finite")
        if not (0.0 <= float(row["rec_d1"]) <= 1.0):
            raise ValueError(f"{prefix} rec_d1 lies outside [0,1]")
        if not (0.0 <= float(row["rec_d2"]) <= 1.0):
            raise ValueError(f"{prefix} rec_d2 lies outside [0,1]")


def _validate_used_records(bundle: ProjectionBundle) -> None:
    """Validate the seven exact formal factorials used by the SI tables."""

    for filename in USED_RECORDS:
        _validate_trial_fields(bundle.records[filename], label=filename)

    main_specs = (
        (OSA_RECORD, "osa", 200),
        (GBUMP_RECORD, "gbump", 200),
        (EFFTOX_RECORD, "efftox", 100),
        (MARIPOSA_RECORD, "mariposa", 100),
    )
    main_fields = ("policy", "seed", "sim", "stratum", "mode", "gamma", "budget")
    for filename, sim, seed_count in main_specs:
        expected = {
            (policy, seed, sim, stratum, "latent", gamma, FORMAL_BUDGET)
            for policy in MAIN_POLICIES
            for seed in range(seed_count)
            for stratum in STRATA
            for gamma in FULL_GATES
        }
        _validate_factorial(
            bundle.records[filename],
            label=filename,
            key_fields=main_fields,
            expected=expected,
        )

    sixway_fields = (
        "policy",
        "seed",
        "sim",
        "stratum",
        "mode",
        "gamma",
        "kap",
        "budget",
    )
    sixway_expected = {
        (policy, seed, sim, stratum, mode, gamma, 1.0, FORMAL_BUDGET)
        for policy in RAW_SIXWAY_POLICIES
        for seed in range(100)
        for sim in ("osa", "gbump")
        for stratum in STRATA
        for mode in ("latent", "predictive")
        for gamma in FULL_GATES
    }
    _validate_factorial(
        bundle.records[SIXWAY_RECORD],
        label=SIXWAY_RECORD,
        key_fields=sixway_fields,
        expected=sixway_expected,
    )

    baseline_fields = (
        "policy",
        "seed",
        "sim",
        "stratum",
        "mode",
        "gamma",
        "kap",
        "budget",
    )
    baseline_expected = {
        (policy, seed, sim, stratum, "latent", 0.7, 1.0, FORMAL_BUDGET)
        for policy in RAW_BASELINE_POLICIES
        for seed in range(100)
        for sim in ("efftox", "mariposa")
        for stratum in STRATA
    }
    _validate_factorial(
        bundle.records[BASELINE_RECORD],
        label=BASELINE_RECORD,
        key_fields=baseline_fields,
        expected=baseline_expected,
    )

    kappa_fields = (
        "policy",
        "seed",
        "sim",
        "stratum",
        "mode",
        "gamma",
        "kap",
        "budget",
    )
    kappa_expected = {
        (policy, seed, "osa", stratum, "latent", gamma, kap, FORMAL_BUDGET)
        for policy in RAW_KAPPA_POLICIES
        for seed in range(100)
        for stratum in STRATA
        for gamma in DISPLAY_GATES
        for kap in KAPPA_LEVELS
    }
    _validate_factorial(
        bundle.records[KAPPA_RECORD],
        label=KAPPA_RECORD,
        key_fields=kappa_fields,
        expected=kappa_expected,
    )


def _select(
    rows: Iterable[Mapping[str, Any]],
    *,
    policy: str | None = None,
    sim: str | None = None,
    mode: str | None = None,
    tau: float | None = None,
    stratum: int | None = None,
    kap: float | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if policy is not None and _policy(row) != policy:
            continue
        if sim is not None and row.get("sim") != sim:
            continue
        if mode is not None and row.get("mode") != mode:
            continue
        if tau is not None and float(row.get("gamma")) != float(tau):
            continue
        if stratum is not None and int(row.get("stratum")) != int(stratum):
            continue
        if kap is not None and float(row.get("kap")) != float(kap):
            continue
        selected.append(dict(row))
    return selected


def _metric_value(row: Mapping[str, Any], metric: Metric) -> float:
    value = metric(row) if callable(metric) else row[metric]
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("table metric must be finite")
    return number


def _seed_values(rows: Iterable[Mapping[str, Any]], metric: Metric) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(_metric_value(row, metric))
    if not grouped:
        raise ValueError("table cell has no records")
    return {
        seed: float(np.mean(values)) for seed, values in sorted(grouped.items())
    }


def _mean_se(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("table cell has no finite seed estimates")
    se = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return float(array.mean()), se, int(len(array))


def _paired_maps(
    left: Mapping[int, float], right: Mapping[int, float]
) -> dict[str, float | int]:
    if not left or set(left) != set(right):
        raise ValueError("paired comparison has missing or unequal seed labels")
    differences = np.asarray(
        [float(left[seed]) - float(right[seed]) for seed in sorted(left)], dtype=float
    )
    se = (
        float(differences.std(ddof=1) / np.sqrt(len(differences)))
        if len(differences) > 1
        else 0.0
    )
    return {"difference": float(differences.mean()), "mcse": se, "n": len(differences)}


def _paired(
    rows: Iterable[Mapping[str, Any]],
    left: str,
    right: str,
    metric: Metric,
) -> dict[str, float | int]:
    materialized = list(rows)
    return _paired_maps(
        _seed_values((row for row in materialized if _policy(row) == left), metric),
        _seed_values((row for row in materialized if _policy(row) == right), metric),
    )


def _assignment_pct(row: Mapping[str, Any]) -> float:
    """Above-boundary assignment percentage using that row's stored budget."""

    budget = float(row["budget"])
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("assignment percentage requires a positive row budget")
    toxic = float(row["toxic"])
    if not 0.0 <= toxic <= budget:
        raise ValueError("above-boundary count lies outside its row budget")
    return 100.0 * toxic / budget


def _gate_precision(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, int]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(row)
    if not grouped:
        raise ValueError("gate-precision cell has no records")
    passed = np.asarray(
        [sum(float(row["n_gate_pass"]) for row in grouped[seed]) for seed in sorted(grouped)],
        dtype=float,
    )
    safe = np.asarray(
        [sum(float(row["n_gate_pass_safe"]) for row in grouped[seed]) for seed in sorted(grouped)],
        dtype=float,
    )
    if passed.sum() <= 0.0 or passed.mean() <= 0.0:
        raise ValueError("pooled gate precision is undefined with no admissions")
    ratio = float(safe.sum() / passed.sum())
    influence = safe - ratio * passed
    se = (
        float(influence.std(ddof=1) / (np.sqrt(len(influence)) * passed.mean()))
        if len(influence) > 1
        else 0.0
    )
    return 100.0 * ratio, 100.0 * se, len(grouped)


def _cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("table cell has no records")
    output: dict[str, float | int] = {}
    metrics: tuple[tuple[str, Metric, float], ...] = (
        ("dose_units", "dose_units", 1.0),
        ("rpsel", "rpsel", 1.0),
        ("above_boundary_assignments", "toxic", 1.0),
        ("above_boundary_assignments_pct", _assignment_pct, 1.0),
        ("above_threshold_recommendation_pct", "rec_unsafe", 100.0),
    )
    for prefix, metric, multiplier in metrics:
        mean, se, n = _mean_se(_seed_values(rows, metric).values())
        output[f"{prefix}_mean"] = multiplier * mean
        output[f"{prefix}_mcse"] = multiplier * se
        output["independent_seeds"] = n
    precision, precision_se, precision_n = _gate_precision(rows)
    if precision_n != int(output["independent_seeds"]):
        raise ValueError("gate precision and outcome metrics use different seed counts")
    output["gate_precision_pct_mean"] = precision
    output["gate_precision_pct_mcse"] = precision_se
    return output


def _pm(mean: float, se: float, digits: int) -> str:
    body = f"{mean:.{digits}f}{{\\pm}}{se:.{digits}f}"
    return f"${body}$"


def _table_tex(column_spec: str, headers: Sequence[str], body: Sequence[str]) -> str:
    lines = [f"\\begin{{tabular}}{{{column_spec}}}", r"\toprule"]
    lines.extend(headers)
    lines.append(r"\midrule")
    lines.extend(body)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _build_demo(bundle: ProjectionBundle) -> TableSpec:
    records = bundle.records[GBUMP_RECORD]
    rows: list[dict[str, Any]] = []
    for stratum in STRATA:
        for tau in DISPLAY_GATES:
            cell_records = _select(records, sim="gbump", mode="latent", tau=tau, stratum=stratum)
            for policy in MAIN_POLICIES:
                cell = _cell([row for row in cell_records if _policy(row) == policy])
                rows.append({
                    "stratum": stratum,
                    "tau": tau,
                    "policy": policy,
                    **cell,
                })
    body = []
    for row in rows:
        lead = f"$z{{=}}{row['stratum']}$" if row["tau"] == DISPLAY_GATES[0] and row["policy"] == MAIN_POLICIES[0] else ""
        tau_policy = f"{row['tau']:.1f}, {POLICY_TEX[row['policy']]}"
        body.append(
            f"{lead} & {tau_policy} & "
            + " & ".join((
                _pm(row["dose_units_mean"], row["dose_units_mcse"], 2),
                _pm(row["rpsel_mean"], row["rpsel_mcse"], 2),
                _pm(row["above_boundary_assignments_mean"], row["above_boundary_assignments_mcse"], 1),
                _pm(row["above_threshold_recommendation_pct_mean"], row["above_threshold_recommendation_pct_mcse"], 1),
                _pm(row["gate_precision_pct_mean"], row["gate_precision_pct_mcse"], 1),
            ))
            + r" \\"
        )
    fields = (
        "stratum", "tau", "policy", "dose_units_mean", "dose_units_mcse",
        "rpsel_mean", "rpsel_mcse", "above_boundary_assignments_mean",
        "above_boundary_assignments_mcse", "above_boundary_assignments_pct_mean",
        "above_boundary_assignments_pct_mcse", "above_threshold_recommendation_pct_mean",
        "above_threshold_recommendation_pct_mcse", "gate_precision_pct_mean",
        "gate_precision_pct_mcse", "independent_seeds",
    )
    tex = _table_tex(
        "@{}llrrrrr@{}",
        (
            r"stratum & $\tau$, policy & dose-location error (dose-units) & posterior efficacy error & \shortstack{above-boundary simulated\\assignments} & \shortstack{true-boundary-exceeding\\final recommendation (\%)} & pooled gate precision (\%) \\",
        ),
        body,
    )
    selected = _select(records, sim="gbump", mode="latent")
    selected = [row for row in selected if float(row["gamma"]) in DISPLAY_GATES]
    return TableSpec(
        "gaussian_bump_detailed", rows, fields, tex, selected, (GBUMP_RECORD,),
        {
            "aggregation": "one row per seed in each policy-by-stratum-by-threshold cell",
            "above_boundary_assignment_unit": "simulated assignment count",
            "displayed_thresholds": list(DISPLAY_GATES),
        },
    )


def _build_pooled_geometry(
    bundle: ProjectionBundle, *, record_name: str, sim: str, table_id: str
) -> TableSpec:
    records = bundle.records[record_name]
    rows: list[dict[str, Any]] = []
    for tau in DISPLAY_GATES:
        tau_records = _select(records, sim=sim, mode="latent", tau=tau)
        for policy in MAIN_POLICIES:
            cell = _cell([row for row in tau_records if _policy(row) == policy])
            rows.append({
                "sim": sim,
                "policy": policy,
                "tau": tau,
                "dose_units_mean": cell["dose_units_mean"],
                "dose_units_mcse": cell["dose_units_mcse"],
                "above_threshold_recommendation_pct_mean": cell["above_threshold_recommendation_pct_mean"],
                "above_threshold_recommendation_pct_mcse": cell["above_threshold_recommendation_pct_mcse"],
                "independent_seeds": cell["independent_seeds"],
            })
    lookup = {(row["policy"], row["tau"]): row for row in rows}
    body = []
    for policy in MAIN_POLICIES:
        du_cells = [
            _pm(
                lookup[policy, tau]["dose_units_mean"],
                lookup[policy, tau]["dose_units_mcse"],
                2,
            )
            for tau in DISPLAY_GATES
        ]
        terminal_cells = [
            _pm(
                lookup[policy, tau]["above_threshold_recommendation_pct_mean"],
                lookup[policy, tau]["above_threshold_recommendation_pct_mcse"],
                1,
            )
            for tau in DISPLAY_GATES
        ]
        body.append(
            f"{POLICY_TEX[policy]} & " + " & ".join(du_cells + terminal_cells) + r" \\"
        )
    tex = _table_tex(
        "@{}lcccccc@{}",
        (
            r" & \multicolumn{3}{c}{dose-location error (dose-units)} & \multicolumn{3}{c}{\shortstack{true-boundary-exceeding\\final recommendation (\%)}} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r"policy & $\tau{=}0.5$ & $0.7$ & $0.9$ & $0.5$ & $0.7$ & $0.9$ \\",
        ),
        body,
    )
    selected = _select(records, sim=sim, mode="latent")
    selected = [row for row in selected if float(row["gamma"]) in DISPLAY_GATES]
    fields = (
        "sim", "policy", "tau", "dose_units_mean", "dose_units_mcse",
        "above_threshold_recommendation_pct_mean",
        "above_threshold_recommendation_pct_mcse", "independent_seeds",
    )
    return TableSpec(
        table_id, rows, fields, tex, selected, (record_name,),
        {
            "aggregation": "strata averaged within seed before Monte Carlo SE across seeds",
            "displayed_thresholds": list(DISPLAY_GATES),
        },
    )


def _build_osa_tradeoff(bundle: ProjectionBundle) -> TableSpec:
    records = bundle.records[OSA_RECORD]
    rows: list[dict[str, Any]] = []
    for tau in DISPLAY_GATES:
        for stratum in STRATA:
            selected_cell = _select(records, sim="osa", mode="latent", tau=tau, stratum=stratum)
            for policy in MAIN_POLICIES:
                cell = _cell([row for row in selected_cell if _policy(row) == policy])
                rows.append({
                    "tau": tau,
                    "stratum": stratum,
                    "policy": policy,
                    "above_threshold_recommendation_pct_mean": cell["above_threshold_recommendation_pct_mean"],
                    "above_threshold_recommendation_pct_mcse": cell["above_threshold_recommendation_pct_mcse"],
                    "above_boundary_assignments_pct_mean": cell["above_boundary_assignments_pct_mean"],
                    "above_boundary_assignments_pct_mcse": cell["above_boundary_assignments_pct_mcse"],
                    "independent_seeds": cell["independent_seeds"],
                })
    lookup = {(row["tau"], row["stratum"], row["policy"]): row for row in rows}
    body = []
    for tau in DISPLAY_GATES:
        for stratum in STRATA:
            cell_rows = [lookup[tau, stratum, policy] for policy in MAIN_POLICIES]
            terminal = [
                _pm(row["above_threshold_recommendation_pct_mean"], row["above_threshold_recommendation_pct_mcse"], 1)
                for row in cell_rows
            ]
            assignment = [
                _pm(row["above_boundary_assignments_pct_mean"], row["above_boundary_assignments_pct_mcse"], 1)
                for row in cell_rows
            ]
            body.append(f"{tau:.1f} & {stratum} & " + " & ".join(terminal + assignment) + r" \\")
    tex = _table_tex(
        "@{}llcccccc@{}",
        (
            r" & & \multicolumn{3}{c}{\shortstack{true-boundary-exceeding\\final recommendation (\%)}} & \multicolumn{3}{c}{\shortstack{above-boundary simulated\\assignments (\%)}} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"$\tau$ & $z$ & cEI & cKG & cEI--tMSE & cEI & cKG & cEI--tMSE \\",
        ),
        body,
    )
    selected = _select(records, sim="osa", mode="latent")
    selected = [row for row in selected if float(row["gamma"]) in DISPLAY_GATES]
    fields = (
        "tau", "stratum", "policy", "above_threshold_recommendation_pct_mean",
        "above_threshold_recommendation_pct_mcse", "above_boundary_assignments_pct_mean",
        "above_boundary_assignments_pct_mcse", "independent_seeds",
    )
    return TableSpec(
        "osa_tradeoff", rows, fields, tex, selected, (OSA_RECORD,),
        {
            "aggregation": "one row per seed in each policy-by-stratum-by-threshold cell",
            "assignment_denominator": "each trial's stored budget field",
            "assignment_unit": "percentage of simulated assignments",
        },
    )


def _build_osa_precision(bundle: ProjectionBundle) -> TableSpec:
    records = bundle.records[OSA_RECORD]
    policies = ("cEI", "cKG")
    rows: list[dict[str, Any]] = []
    for tau in DISPLAY_GATES:
        for stratum in STRATA:
            selected_cell = _select(records, sim="osa", mode="latent", tau=tau, stratum=stratum)
            for policy in policies:
                cell = _cell([row for row in selected_cell if _policy(row) == policy])
                rows.append({
                    "tau": tau,
                    "stratum": stratum,
                    "policy": policy,
                    "dose_units_mean": cell["dose_units_mean"],
                    "dose_units_mcse": cell["dose_units_mcse"],
                    "rpsel_mean": cell["rpsel_mean"],
                    "rpsel_mcse": cell["rpsel_mcse"],
                    "independent_seeds": cell["independent_seeds"],
                })
    lookup = {(row["tau"], row["stratum"], row["policy"]): row for row in rows}
    body = []
    for tau in DISPLAY_GATES:
        cells = []
        for metric in ("dose_units", "rpsel"):
            for stratum in STRATA:
                policy_cells = []
                for policy in policies:
                    row = lookup[tau, stratum, policy]
                    policy_cells.append(_pm(row[f"{metric}_mean"], row[f"{metric}_mcse"], 2))
                cells.append(" / ".join(policy_cells))
        body.append(f"{tau:.1f} & " + " & ".join(cells) + r" \\")
    tex = _table_tex(
        "@{}lcccc@{}",
        (
            r" & \multicolumn{2}{c}{dose-location error (dose-units)} & \multicolumn{2}{c}{posterior efficacy error} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"$\tau$ & $z{=}0$ & $z{=}1$ & $z{=}0$ & $z{=}1$ \\",
        ),
        body,
    )
    selected = [
        row for row in _select(records, sim="osa", mode="latent")
        if float(row["gamma"]) in DISPLAY_GATES and _policy(row) in policies
    ]
    fields = (
        "tau", "stratum", "policy", "dose_units_mean", "dose_units_mcse",
        "rpsel_mean", "rpsel_mcse", "independent_seeds",
    )
    return TableSpec(
        "osa_precision", rows, fields, tex, selected, (OSA_RECORD,),
        {"policies": list(policies), "displayed_thresholds": list(DISPLAY_GATES)},
    )


def _build_gate_calibration(bundle: ProjectionBundle) -> TableSpec:
    records = bundle.records[OSA_RECORD]
    rows: list[dict[str, Any]] = []
    for tau in DISPLAY_GATES:
        for stratum in STRATA:
            for policy in MAIN_POLICIES:
                cell_records = _select(
                    records, policy=policy, sim="osa", mode="latent", tau=tau, stratum=stratum
                )
                false_map = _seed_values(
                    cell_records,
                    lambda row: float(row["n_gate_pass"]) - float(row["n_gate_pass_safe"]),
                )
                false_mean, false_se, n = _mean_se(false_map.values())
                precision, precision_se, precision_n = _gate_precision(cell_records)
                if n != precision_n:
                    raise ValueError("gate calibration seed counts disagree")
                rows.append({
                    "tau": tau,
                    "stratum": stratum,
                    "policy": policy,
                    "false_admissions_mean": false_mean,
                    "false_admissions_mcse": false_se,
                    "gate_precision_pct_mean": precision,
                    "gate_precision_pct_mcse": precision_se,
                    "independent_seeds": n,
                })
    lookup = {(row["tau"], row["stratum"], row["policy"]): row for row in rows}
    body = []
    for tau in DISPLAY_GATES:
        for stratum in STRATA:
            current = [lookup[tau, stratum, policy] for policy in MAIN_POLICIES]
            false = [_pm(row["false_admissions_mean"], row["false_admissions_mcse"], 2) for row in current]
            precision = [_pm(row["gate_precision_pct_mean"], row["gate_precision_pct_mcse"], 1) for row in current]
            body.append(f"{tau:.1f} & {stratum} & " + " & ".join(false + precision) + r" \\")
    tex = _table_tex(
        "@{}llcccccc@{}",
        (
            r" & & \multicolumn{3}{c}{false admissions/trial} & \multicolumn{3}{c}{pooled gate precision (\%)} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"$\tau$ & $z$ & cEI & cKG & cEI--tMSE & cEI & cKG & cEI--tMSE \\",
        ),
        body,
    )
    selected = [
        row for row in _select(records, sim="osa", mode="latent")
        if float(row["gamma"]) in DISPLAY_GATES
    ]
    fields = (
        "tau", "stratum", "policy", "false_admissions_mean",
        "false_admissions_mcse", "gate_precision_pct_mean",
        "gate_precision_pct_mcse", "independent_seeds",
    )
    return TableSpec(
        "osa_gate_calibration", rows, fields, tex, selected, (OSA_RECORD,),
        {
            "precision": "ratio of total truly feasible admissions to total admissions",
            "precision_mcse": "matched-replicate-clustered influence-function Monte Carlo SE",
        },
    )


@lru_cache(maxsize=None)
def _true_safe_panel_count(stratum: int, grid_n: int) -> int:
    spec = resolve_surface("osa", stratum)
    axis = np.linspace(0.0, 1.0, grid_n)
    return sum(
        float(spec["tox"](float(d1), float(d2))) <= float(spec["gd"])
        for d1 in axis
        for d2 in axis
    )


def _build_gate_diagnostics(bundle: ProjectionBundle) -> TableSpec:
    records = bundle.records[OSA_RECORD]
    rows: list[dict[str, Any]] = []
    for tau in DISPLAY_GATES:
        for stratum in STRATA:
            for policy in MAIN_POLICIES:
                cell_records = _select(
                    records, policy=policy, sim="osa", mode="latent", tau=tau, stratum=stratum
                )
                grid_levels = {int(row["grid_n"]) for row in cell_records}
                if len(grid_levels) != 1:
                    raise ValueError("gate-diagnostic cell mixes panel sizes")
                grid_n = next(iter(grid_levels))
                true_safe = _true_safe_panel_count(stratum, grid_n)
                if true_safe <= 0:
                    raise ValueError("OSA truth has no feasible panel dose")
                admitted = _mean_se(_seed_values(cell_records, "n_gate_pass").values())
                empty = _mean_se(
                    _seed_values(cell_records, lambda row: float(row["n_gate_pass"] == 0)).values()
                )
                recall = _mean_se(
                    _seed_values(
                        cell_records,
                        lambda row: 100.0 * float(row["n_gate_pass_safe"]) / true_safe,
                    ).values()
                )
                if not (admitted[2] == empty[2] == recall[2]):
                    raise ValueError("gate-diagnostic seed counts disagree")
                rows.append({
                    "tau": tau,
                    "stratum": stratum,
                    "policy": policy,
                    "grid_n": grid_n,
                    "true_safe_panel_doses": true_safe,
                    "admitted_doses_mean": admitted[0],
                    "admitted_doses_mcse": admitted[1],
                    "final_empty_pct_mean": 100.0 * empty[0],
                    "final_empty_pct_mcse": 100.0 * empty[1],
                    "feasible_dose_recall_pct_mean": recall[0],
                    "feasible_dose_recall_pct_mcse": recall[1],
                    "independent_seeds": admitted[2],
                })
    body = [
        f"{row['tau']:.1f} & {row['stratum']} & {POLICY_TEX[row['policy']]} & "
        + " & ".join((
            _pm(row["admitted_doses_mean"], row["admitted_doses_mcse"], 2),
            _pm(row["final_empty_pct_mean"], row["final_empty_pct_mcse"], 1),
            _pm(row["feasible_dose_recall_pct_mean"], row["feasible_dose_recall_pct_mcse"], 1),
        ))
        + r" \\"
        for row in rows
    ]
    tex = _table_tex(
        "@{}cclrrr@{}",
        (r"$\tau$ & $z$ & policy & admitted doses & final empty (\%) & feasible-dose recall (\%) \\",),
        body,
    )
    selected = [
        row for row in _select(records, sim="osa", mode="latent")
        if float(row["gamma"]) in DISPLAY_GATES
    ]
    fields = (
        "tau", "stratum", "policy", "grid_n", "true_safe_panel_doses",
        "admitted_doses_mean", "admitted_doses_mcse", "final_empty_pct_mean",
        "final_empty_pct_mcse", "feasible_dose_recall_pct_mean",
        "feasible_dose_recall_pct_mcse", "independent_seeds",
    )
    return TableSpec(
        "osa_gate_diagnostics", rows, fields, tex, selected, (OSA_RECORD,),
        {
            "scope": "final-posterior gate only; not an ever-empty rate",
            "recall_denominator": "true feasible dose count on each record's stored panel size",
        },
    )


SOURCE_PANEL_TARGET = {
    0: np.asarray((0.25, 0.75), dtype=float),
    1: np.asarray((0.50, 0.75), dtype=float),
}


def _source_panel_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rescored = []
    for row in records:
        item = dict(row)
        grid_n = int(row["grid_n"])
        if grid_n < 2:
            raise ValueError("source-panel scoring requires at least two dose levels")
        spacing = 1.0 / (grid_n - 1)
        recommendation = np.asarray((row["rec_d1"], row["rec_d2"]), dtype=float)
        target = SOURCE_PANEL_TARGET[int(row["stratum"])]
        item["_source_panel_dose_units"] = float(
            np.linalg.norm(recommendation - target) / spacing
        )
        rescored.append(item)
    return rescored


def _build_source_panel(bundle: ProjectionBundle) -> TableSpec:
    original = [
        row for row in _select(bundle.records[OSA_RECORD], sim="osa", mode="latent")
        if float(row["gamma"]) in DISPLAY_GATES
    ]
    rescored = _source_panel_records(original)
    rows: list[dict[str, Any]] = []
    for tau in DISPLAY_GATES:
        for stratum in STRATA:
            cell_records = _select(rescored, sim="osa", mode="latent", tau=tau, stratum=stratum)
            output: dict[str, Any] = {"tau": tau, "stratum": stratum}
            for policy in MAIN_POLICIES:
                mean, se, n = _mean_se(
                    _seed_values(
                        (row for row in cell_records if _policy(row) == policy),
                        "_source_panel_dose_units",
                    ).values()
                )
                output[f"{policy}_mean"] = mean
                output[f"{policy}_mcse"] = se
                output["independent_seeds"] = n
            cei_ckg = _paired(cell_records, "cEI", "cKG", "_source_panel_dose_units")
            ckg_tmse = _paired(cell_records, "cKG", "cEI-tMSE", "_source_panel_dose_units")
            output.update({
                "cEI_minus_cKG_mean": cei_ckg["difference"],
                "cEI_minus_cKG_mcse": cei_ckg["mcse"],
                "cKG_minus_cEI_tMSE_mean": ckg_tmse["difference"],
                "cKG_minus_cEI_tMSE_mcse": ckg_tmse["mcse"],
            })
            rows.append(output)
    body = []
    for row in rows:
        cells = [
            _pm(row[f"{policy}_mean"], row[f"{policy}_mcse"], 2)
            for policy in MAIN_POLICIES
        ]
        contrasts = [
            _pm(row["cEI_minus_cKG_mean"], row["cEI_minus_cKG_mcse"], 3),
            _pm(row["cKG_minus_cEI_tMSE_mean"], row["cKG_minus_cEI_tMSE_mcse"], 3),
        ]
        body.append(f"{row['tau']:.1f} & {row['stratum']} & " + " & ".join(cells + contrasts) + r" \\")
    tex = _table_tex(
        "@{}llrrrrr@{}",
        (r"$\tau$ & $z$ & cEI & cKG & cEI--tMSE & cEI $-$ cKG & cKG $-$ cEI--tMSE \\",),
        body,
    )
    fields = (
        "tau", "stratum", "cEI_mean", "cEI_mcse", "cKG_mean", "cKG_mcse",
        "cEI-tMSE_mean", "cEI-tMSE_mcse", "cEI_minus_cKG_mean",
        "cEI_minus_cKG_mcse", "cKG_minus_cEI_tMSE_mean",
        "cKG_minus_cEI_tMSE_mcse", "independent_seeds",
    )
    return TableSpec(
        "osa_source_panel", rows, fields, tex, original, (OSA_RECORD,),
        {
            "operation": "saved final recommendations re-scored; no trial rerun",
            "targets": {"0": [0.25, 0.75], "1": [0.5, 0.75]},
            "distance_scale": "one over (grid_n minus one), read from each row",
        },
    )


def _build_baselines(bundle: ProjectionBundle) -> TableSpec:
    osa_records = [
        row
        for row in _select(
            bundle.records[SIXWAY_RECORD], sim="osa", mode="latent", tau=0.7
        )
        if _policy(row) in BASELINE_POLICIES
    ]
    other_records = _select(bundle.records[BASELINE_RECORD], mode="latent", tau=0.7)
    sources = {
        "OSA": osa_records,
        "Logistic": _select(other_records, sim="efftox"),
        "MARIPOSA-motivated": _select(other_records, sim="mariposa"),
    }
    rows: list[dict[str, Any]] = []
    for surface, records in sources.items():
        maps = {
            policy: _seed_values(
                (row for row in records if _policy(row) == policy), "rec_unsafe"
            )
            for policy in BASELINE_POLICIES
        }
        for policy in BASELINE_POLICIES:
            mean, se, n = _mean_se(maps[policy].values())
            rows.append({
                "surface": surface,
                "policy": policy,
                "above_threshold_recommendation_pct_mean": 100.0 * mean,
                "above_threshold_recommendation_pct_mcse": 100.0 * se,
                "independent_seeds": n,
            })
    lookup = {(row["surface"], row["policy"]): row for row in rows}
    body = []
    for surface in sources:
        cells = [
            _pm(
                lookup[surface, policy]["above_threshold_recommendation_pct_mean"],
                lookup[surface, policy]["above_threshold_recommendation_pct_mcse"],
                1,
            )
            for policy in BASELINE_POLICIES
        ]
        body.append(f"{surface} & " + " & ".join(cells) + r" \\")
    tex = _table_tex(
        "@{}lccccc@{}",
        (r"Surface & cEI & cKG & cEI--tMSE & tMSE & SUR \\",),
        body,
    )
    fields = (
        "surface", "policy", "above_threshold_recommendation_pct_mean",
        "above_threshold_recommendation_pct_mcse", "independent_seeds",
    )
    return TableSpec(
        "boundary_baselines", rows, fields, tex, osa_records + other_records,
        (SIXWAY_RECORD, BASELINE_RECORD),
        {
            "threshold": 0.7,
            "aggregation": "strata averaged within seed before Monte Carlo SE across seeds",
        },
    )


def _build_sixway(bundle: ProjectionBundle) -> TableSpec:
    records = bundle.records[SIXWAY_RECORD]
    columns = (
        ("osa", "latent"),
        ("osa", "predictive"),
        ("gbump", "latent"),
        ("gbump", "predictive"),
    )
    rows: list[dict[str, Any]] = []
    for sim, mode in columns:
        cell_records = _select(records, sim=sim, mode=mode)
        maps = {
            policy: _seed_values(
                (row for row in cell_records if _policy(row) == policy), "dose_units"
            )
            for policy in SIXWAY_POLICIES
        }
        for policy in SIXWAY_POLICIES:
            mean, se, n = _mean_se(maps[policy].values())
            rows.append({
                "sim": sim,
                "gate_mode": mode,
                "policy": policy,
                "dose_units_mean": mean,
                "dose_units_mcse": se,
                "independent_seeds": n,
            })
    lookup = {(row["policy"], row["sim"], row["gate_mode"]): row for row in rows}
    body = []
    for policy in SIXWAY_POLICIES:
        cells = [
            _pm(
                lookup[policy, sim, mode]["dose_units_mean"],
                lookup[policy, sim, mode]["dose_units_mcse"],
                2,
            )
            for sim, mode in columns
        ]
        body.append(f"{POLICY_TEX[policy]} & " + " & ".join(cells) + r" \\")
    tex = _table_tex(
        "@{}lcccc@{}",
        (
            r" & \multicolumn{2}{c}{OSA} & \multicolumn{2}{c}{Gaussian-bump} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"policy & latent & predictive & latent & predictive \\",
        ),
        body,
    )
    fields = (
        "sim", "gate_mode", "policy", "dose_units_mean", "dose_units_mcse",
        "independent_seeds",
    )
    return TableSpec(
        "six_policy", rows, fields, tex, list(records), (SIXWAY_RECORD,),
        {
            "aggregation": "two strata and five thresholds averaged within each seed",
            "thresholds": list(FULL_GATES),
        },
    )


def _build_kappa(bundle: ProjectionBundle) -> TableSpec:
    records = bundle.records[KAPPA_RECORD]
    policies = ("cEI", "cKG")
    rows: list[dict[str, Any]] = []
    threshold_contrasts = []
    for kap in KAPPA_LEVELS:
        kap_records = _select(records, sim="osa", mode="latent", kap=kap)
        maps_by_metric = {
            metric: {
                policy: _seed_values(
                    (row for row in kap_records if _policy(row) == policy), metric
                )
                for policy in policies
            }
            for metric in ("dose_units", "rec_unsafe")
        }
        for policy in policies:
            du = _mean_se(maps_by_metric["dose_units"][policy].values())
            unsafe = _mean_se(maps_by_metric["rec_unsafe"][policy].values())
            rows.append({
                "kappa": kap,
                "policy": policy,
                "dose_units_mean": du[0],
                "dose_units_mcse": du[1],
                "above_threshold_recommendation_pct_mean": 100.0 * unsafe[0],
                "above_threshold_recommendation_pct_mcse": 100.0 * unsafe[1],
                "independent_seeds": du[2],
            })
        for tau in DISPLAY_GATES:
            tau_records = _select(kap_records, tau=tau)
            contrast = _paired(tau_records, "cEI", "cKG", "dose_units")
            threshold_contrasts.append({
                "kappa": kap,
                "tau": tau,
                "contrast": "cEI minus cKG",
                "difference_mean": contrast["difference"],
                "difference_mcse": contrast["mcse"],
                "independent_seeds": contrast["n"],
            })
    lookup = {(row["kappa"], row["policy"]): row for row in rows}
    body = []
    for kap in KAPPA_LEVELS:
        current = [lookup[kap, policy] for policy in policies]
        du = [_pm(row["dose_units_mean"], row["dose_units_mcse"], 2) for row in current]
        terminal = [
            _pm(row["above_threshold_recommendation_pct_mean"], row["above_threshold_recommendation_pct_mcse"], 1)
            for row in current
        ]
        body.append(f"${kap:g}$ & " + " & ".join(du + terminal) + r" \\")
    tex = _table_tex(
        "@{}lcccc@{}",
        (
            r" & \multicolumn{2}{c}{dose-location error (dose-units)} & \multicolumn{2}{c}{\shortstack{true-boundary-exceeding\\final recommendation (\%)}} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"$\kappa$ & cEI & cKG & cEI & cKG \\",
        ),
        body,
    )
    fields = (
        "kappa", "policy", "dose_units_mean", "dose_units_mcse",
        "above_threshold_recommendation_pct_mean",
        "above_threshold_recommendation_pct_mcse", "independent_seeds",
    )
    return TableSpec(
        "kappa_sensitivity", rows, fields, tex, list(records), (KAPPA_RECORD,),
        {
            "aggregation": "two strata and three thresholds averaged within each seed",
            "thresholds": list(DISPLAY_GATES),
        },
        auxiliary={"threshold_specific_dose_unit_contrasts": threshold_contrasts},
    )


def _build_tables(bundle: ProjectionBundle) -> list[TableSpec]:
    return [
        _build_demo(bundle),
        _build_pooled_geometry(
            bundle,
            record_name=EFFTOX_RECORD,
            sim="efftox",
            table_id="logistic_geometry",
        ),
        _build_pooled_geometry(
            bundle,
            record_name=MARIPOSA_RECORD,
            sim="mariposa",
            table_id="mariposa_geometry",
        ),
        _build_osa_tradeoff(bundle),
        _build_osa_precision(bundle),
        _build_gate_calibration(bundle),
        _build_gate_diagnostics(bundle),
        _build_source_panel(bundle),
        _build_baselines(bundle),
        _build_sixway(bundle),
        _build_kappa(bundle),
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if not rows:
        raise ValueError("cannot write an empty SI table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_sidecar(
    *,
    artifact: Path,
    table: TableSpec,
    bundle: ProjectionBundle,
    artifact_type: str,
) -> Path:
    provenance = bundle.provenance
    record_files = provenance["record_files"]
    input_files = {}
    for filename in table.source_names:
        commitment = record_files[filename]
        input_files[filename] = {
            "rows": int(commitment["rows"]),
            "compressed_bytes": int(commitment["compressed_bytes"]),
            "compressed_sha256": commitment["compressed_sha256"],
            "uncompressed_bytes": int(commitment["uncompressed_bytes"]),
            "uncompressed_sha256": commitment["uncompressed_sha256"],
        }
    analyzer = Path(__file__).resolve()
    metadata = {
        "schema_version": 2,
        "artifact_type": artifact_type,
        "artifact_basename": artifact.name,
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": _sha256_file(artifact),
        "table_id": table.table_id,
        "analyzer": {
            "basename": analyzer.name,
            "sha256": _sha256_file(analyzer),
        },
        "projection": {
            "projection_metadata_path": provenance["projection_metadata_path"],
            "projection_metadata_sha256": provenance["projection_metadata_sha256"],
            "projection_fingerprint": provenance["projection_fingerprint"],
            "formal_manifest_sha256": provenance["formal_manifest_sha256"],
        },
        "input_files": input_files,
        "selected_record_count": len(table.selected_records),
        "selected_records_sha256": sdb.records_sha256(table.selected_records),
        "independent_seed_labels": sorted(
            {int(row["seed"]) for row in table.selected_records}
        ),
        "observed_budget_levels": sorted(
            {int(row["budget"]) for row in table.selected_records}
        ),
        "formal_factorial_validation": {
            "status": "PASS",
            "validated_record_files": list(USED_RECORDS),
            "budget_is_read_from_each_record_for_percentage_metrics": True,
        },
        "estimand": table.estimand,
    }
    raw_policies = {str(row.get("policy")) for row in table.selected_records}
    if raw_policies.intersection({"cEI-tMSE", "GBE"}):
        metadata["reader_facing_policy_alias"] = {"cEI-tMSE": "cEI–tMSE"}
        metadata["full_panel_reference_schedule"] = {
            "scope": "the selected records use full-panel access",
            "maximum_assignments": 40,
            "initialization_assignments": 4,
            "cohort_size": 2,
            "scheduled_adaptive_cohort_positions": 18,
            "cEI_scheduled_positions": 9,
            "tMSE_scheduled_positions": 9,
            "empty_gate_override": (
                "An empty eligible gate can invoke the common "
                "greatest-feasibility fallback instead of the scheduled acquisition."
            ),
        }
    path = artifact.with_suffix(artifact.suffix + ".metadata.json")
    path.write_bytes(_canonical_json_bytes(metadata))
    return path


def _write_table_artifacts(
    table: TableSpec, bundle: ProjectionBundle, output_dir: Path
) -> list[Path]:
    base = output_dir / f"si_{table.table_id}"
    tex_path = base.with_suffix(".tex")
    csv_path = base.with_suffix(".csv")
    json_path = base.with_suffix(".json")
    tex_path.write_text(table.tex, encoding="utf-8")
    _write_csv(csv_path, table.rows, table.fieldnames)
    payload = {
        "schema_version": 2,
        "table_id": table.table_id,
        "source_record_files": list(table.source_names),
        "estimand": table.estimand,
        "columns": list(table.fieldnames),
        "rows": table.rows,
    }
    if table.auxiliary is not None:
        payload["auxiliary"] = table.auxiliary
    json_path.write_bytes(_canonical_json_bytes(payload))

    artifacts = [tex_path, csv_path, json_path]
    for artifact, artifact_type in (
        (tex_path, "supporting_information_latex_table"),
        (csv_path, "supporting_information_table_data_csv"),
        (json_path, "supporting_information_table_data_json"),
    ):
        artifacts.append(
            _write_sidecar(
                artifact=artifact,
                table=table,
                bundle=bundle,
                artifact_type=artifact_type,
            )
        )
    return artifacts


def analyze_bundle(bundle: ProjectionBundle, output_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Validate one authenticated bundle and write all 11 SI tables."""

    _validate_used_records(bundle)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = _build_tables(bundle)
    if len(tables) != 11 or len({table.table_id for table in tables}) != 11:
        raise AssertionError("the record-derived SI table inventory must contain 11 unique tables")
    for table in tables:
        _write_table_artifacts(table, bundle, output)
    artifact_inventory = {}
    for table in tables:
        artifact_inventory[table.table_id] = {}
        for extension in ("tex", "csv", "json"):
            artifact = output / f"si_{table.table_id}.{extension}"
            sidecar = artifact.with_suffix(artifact.suffix + ".metadata.json")
            artifact_inventory[table.table_id][extension] = {
                "basename": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": _sha256_file(artifact),
                "metadata_basename": sidecar.name,
                "metadata_sha256": _sha256_file(sidecar),
            }
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "table_count": len(tables),
        "table_ids": [table.table_id for table in tables],
        "artifacts_per_table": ["tex", "csv", "json"],
        "sidecar_per_artifact": True,
        "projection_fingerprint": bundle.provenance["projection_fingerprint"],
        "formal_manifest_sha256": bundle.provenance["formal_manifest_sha256"],
        "artifacts": artifact_inventory,
    }
    manifest_path = output / "si_record_tables.manifest.json"
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    print(f"wrote {len(tables)} SI tables and provenance sidecars to {output}")
    return {table.table_id: table.rows for table in tables}


def analyze(projection_dir: str | Path, output_dir: str | Path):
    """Authenticate a formal projection once and regenerate the SI tables."""

    return analyze_bundle(load_projection_bundle(projection_dir), output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection-dir",
        required=True,
        help="authenticated formal_projection-<fingerprint> directory",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    analyze(args.projection_dir, args.out_dir)


if __name__ == "__main__":
    main()
