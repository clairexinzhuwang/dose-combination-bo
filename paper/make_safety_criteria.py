#!/usr/bin/env python3
"""Generate authenticated OSA/GBump SI safety and criteria profiles.

The generator validates both complete 6,000-row main factorials and derives all
displayed directions from the formal records. It does not run simulations or
modify recommendations.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from paper.analysis_record_bundle import ProjectionBundle, load_projection_bundle
    from paper.figure_artifacts import (
        finite_float,
        padded_limits,
        save_figure,
        write_artifact_sidecar,
        write_csv,
    )
except ModuleNotFoundError:  # Support direct script execution.
    from analysis_record_bundle import ProjectionBundle, load_projection_bundle
    from figure_artifacts import (
        finite_float,
        padded_limits,
        save_figure,
        write_artifact_sidecar,
        write_csv,
    )


RECORDS = {
    "osa": "osa_main.json.zst",
    "gbump": "gbump_main.json.zst",
}
POLICIES = ("cEI", "cKG", "cEI-tMSE")
POLICY_LABELS = {"cEI": "cEI", "cKG": "cKG", "cEI-tMSE": "cEI-tMSE"}
STRATA = (0, 1)
TAU_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)
SEEDS = tuple(range(200))
SAFETY_STEM = "si_osa_gate_and_recommendation_safety_by_threshold"
CRITERIA_STEM = "si_osa_gbump_operating_criteria_by_threshold"
ARTIFACT_CLASS = "journal_supplement_figure_candidate"
CRITERIA = (
    ("dose_units", "Dose-units", "dose-units", "lower"),
    ("rpsel", "Posterior efficacy error", "response units", "lower"),
    ("toxic", "Above-boundary simulated assignments", "assignments", "lower"),
)


def _validate_main_factorial(records: list[dict[str, Any]], simulator: str) -> None:
    """Validate one complete 3-policy x 200-seed x 2-stratum x 5-tau design."""

    expected = set(itertools.product(POLICIES, SEEDS, STRATA, TAU_LEVELS))
    observed: set[tuple[Any, ...]] = set()
    for index, row in enumerate(records):
        if row.get("sim") != simulator:
            raise ValueError(f"record {index} is not from simulator {simulator}")
        key = (
            row.get("policy"),
            row.get("seed"),
            row.get("stratum"),
            finite_float(row.get("gamma"), label=f"record {index} gamma"),
        )
        if key in observed:
            raise ValueError(f"duplicate {simulator} main design cell: {key}")
        observed.add(key)
        fixed = {
            "mode": "latent",
            "budget": 40,
            "grid_n": 5,
            "r_k": 2,
            "warmup": 4,
            "noise": "fixed",
            "empty_gate": "pf",
        }
        for field, expected_value in fixed.items():
            if row.get(field) != expected_value:
                raise ValueError(
                    f"record {index} has {field}={row.get(field)!r}; "
                    f"expected {expected_value!r}"
                )
        values = {
            field: finite_float(row.get(field), label=f"record {index} {field}")
            for field in (
                "dose_units",
                "rpsel",
                "toxic",
                "rec_unsafe",
                "n_gate_pass",
                "n_gate_pass_safe",
            )
        }
        if values["dose_units"] < 0.0 or values["rpsel"] < 0.0:
            raise ValueError("selection and efficacy-error criteria must be nonnegative")
        if not 0.0 <= values["toxic"] <= 40.0:
            raise ValueError("above-boundary assignments are outside the trial budget")
        if values["rec_unsafe"] not in (0.0, 1.0):
            raise ValueError("recommendation threshold indicator must be binary")
        if not 0.0 <= values["n_gate_pass_safe"] <= values["n_gate_pass"]:
            raise ValueError("safe gate admissions cannot exceed all gate admissions")
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        example = next(iter(missing or extra))
        raise ValueError(
            f"{simulator} main records do not contain the complete frozen factorial; "
            f"example mismatched cell: {example}"
        )


def _cell(records: list[dict[str, Any]], policy: str, stratum: int, tau: float) -> list[dict[str, Any]]:
    chosen = [
        row
        for row in records
        if row["policy"] == policy
        and row["stratum"] == stratum
        and float(row["gamma"]) == tau
    ]
    if len(chosen) != len(SEEDS) or {int(row["seed"]) for row in chosen} != set(SEEDS):
        raise ValueError(
            f"incomplete cell: policy={policy}, stratum={stratum}, tau={tau}"
        )
    return chosen


def _mean_mcse(rows: list[dict[str, Any]], field: str) -> tuple[float, float]:
    values = np.asarray(
        [finite_float(row[field], label=field) for row in rows], dtype=float
    )
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"{field} cell requires at least two finite values")
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def _gate_precision_mcse(rows: list[dict[str, Any]]) -> tuple[float, float, int, int]:
    admitted = np.asarray([row["n_gate_pass"] for row in rows], dtype=float)
    safe = np.asarray([row["n_gate_pass_safe"] for row in rows], dtype=float)
    total_admitted = int(admitted.sum())
    total_safe = int(safe.sum())
    if total_admitted <= 0:
        raise ValueError("pooled gate precision is undefined with zero total admissions")
    ratio = float(safe.sum() / admitted.sum())
    influence = (safe - ratio * admitted) / admitted.mean()
    mcse = float(influence.std(ddof=1) / np.sqrt(len(rows)))
    return 100.0 * ratio, 100.0 * mcse, total_admitted, total_safe


def _safety_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stratum in STRATA:
        for tau in TAU_LEVELS:
            for policy in POLICIES:
                rows = _cell(records, policy, stratum, tau)
                precision, precision_mcse, admitted, safe = _gate_precision_mcse(rows)
                rec_rate, rec_mcse = _mean_mcse(rows, "rec_unsafe")
                common = {
                    "simulator": "osa",
                    "dose_access": "full_panel",
                    "gate_mode": "latent",
                    "stratum": stratum,
                    "tau": tau,
                    "policy_id": policy,
                    "policy": POLICY_LABELS[policy],
                    "independent_seeds": len(rows),
                }
                output.extend(
                    (
                        {
                            **common,
                            "outcome_id": "pooled_gate_precision",
                            "outcome": "Admission-weighted pooled gate precision",
                            "unit": "percent",
                            "direction": "higher is better",
                            "estimate": precision,
                            "mcse": precision_mcse,
                            "gate_admissions": admitted,
                            "gate_admissions_within_boundary": safe,
                        },
                        {
                            **common,
                            "outcome_id": "threshold_exceeding_final_recommendation",
                            "outcome": "Threshold-exceeding final recommendation rate",
                            "unit": "percent",
                            "direction": "lower is better",
                            "estimate": 100.0 * rec_rate,
                            "mcse": 100.0 * rec_mcse,
                            "gate_admissions": "",
                            "gate_admissions_within_boundary": "",
                        },
                    )
                )
    expected = len(STRATA) * len(TAU_LEVELS) * len(POLICIES) * 2
    if len(output) != expected:
        raise ValueError("safety summary does not contain every factorial cell")
    return output


def _criteria_rows(records_by_simulator: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for simulator in ("osa", "gbump"):
        records = records_by_simulator[simulator]
        for stratum in STRATA:
            for tau in TAU_LEVELS:
                for policy in POLICIES:
                    rows = _cell(records, policy, stratum, tau)
                    for field, label, unit, direction in CRITERIA:
                        estimate, mcse = _mean_mcse(rows, field)
                        output.append(
                            {
                                "simulator": simulator,
                                "dose_access": "full_panel",
                                "gate_mode": "latent",
                                "stratum": stratum,
                                "tau": tau,
                                "policy_id": policy,
                                "policy": POLICY_LABELS[policy],
                                "outcome_id": field,
                                "outcome": label,
                                "unit": unit,
                                "direction": f"{direction} is better",
                                "independent_seeds": len(rows),
                                "estimate": estimate,
                                "mcse": mcse,
                            }
                        )
    expected = 2 * len(STRATA) * len(TAU_LEVELS) * len(POLICIES) * len(CRITERIA)
    if len(output) != expected:
        raise ValueError("criteria summary does not contain every factorial cell")
    return output


def _styles() -> tuple[dict[str, str], dict[int, tuple[str, str]]]:
    colors = {"cEI": "#D55E00", "cKG": "#0072B2", "cEI-tMSE": "#009E73"}
    strata = {0: ("-", "o"), 1: ("--", "s")}
    return colors, strata


def _configure_plotting() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9.2,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.unicode_minus": False,
        }
    )


def _plot_safety(rows: list[dict[str, Any]], pdf_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _configure_plotting()
    colors, strata_styles = _styles()
    outcomes = (
        (
            "pooled_gate_precision",
            "(a) Gate calibration",
            "Pooled gate precision (%)",
        ),
        (
            "threshold_exceeding_final_recommendation",
            "(b) Final recommendation threshold status",
            "Threshold-exceeding recommendation (%)",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.65))
    for axis, (outcome_id, title, ylabel) in zip(axes, outcomes):
        relevant = [row for row in rows if row["outcome_id"] == outcome_id]
        bounds = [
            float(row["estimate"]) + sign * float(row["mcse"])
            for row in relevant
            for sign in (-1.0, 1.0)
        ]
        axis.set_ylim(*padded_limits(bounds, minimum_span=5.0))
        for policy in POLICIES:
            for stratum in STRATA:
                chosen = [
                    row
                    for row in relevant
                    if row["policy_id"] == policy and row["stratum"] == stratum
                ]
                chosen.sort(key=lambda row: float(row["tau"]))
                linestyle, marker = strata_styles[stratum]
                axis.errorbar(
                    [row["tau"] for row in chosen],
                    [row["estimate"] for row in chosen],
                    yerr=[row["mcse"] for row in chosen],
                    color=colors[policy],
                    linestyle=linestyle,
                    marker=marker,
                    markersize=4.2,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    linewidth=1.5,
                    capsize=2.0,
                )
        axis.set_title(title, loc="left")
        axis.set_xlabel(r"Toxicity-rule stringency $\tau$ (larger is stricter)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(TAU_LEVELS)
        axis.set_xlim(0.475, 0.925)
        axis.grid(axis="y", color="#e2e2e2", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    legend = [
        Line2D([], [], color=colors[policy], linewidth=2.2, label=POLICY_LABELS[policy])
        for policy in POLICIES
    ] + [
        Line2D(
            [],
            [],
            color="#333333",
            linestyle=strata_styles[stratum][0],
            marker=strata_styles[stratum][1],
            markerfacecolor="white",
            label=f"stratum $z={stratum}$",
        )
        for stratum in STRATA
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        fontsize=8.1,
    )
    fig.text(
        0.5,
        0.005,
        "Full-panel OSA; points are means or pooled ratios over 200 seeds; bars are +/-1 MCSE.",
        ha="center",
        fontsize=7.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.90))
    save_figure(
        fig,
        pdf_path,
        png_path,
        title="OSA gate calibration and final recommendation threshold status",
        subject=(
            "Authenticated full-panel OSA pooled gate precision and "
            "threshold-exceeding final-recommendation profiles"
        ),
    )
    plt.close(fig)


def _plot_criteria(rows: list[dict[str, Any]], pdf_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _configure_plotting()
    colors, strata_styles = _styles()
    fig, axes = plt.subplots(2, 3, figsize=(7.55, 5.05))
    simulators = (("osa", "OSA simulator"), ("gbump", "Gaussian-bump simulator"))
    for row_index, (simulator, simulator_label) in enumerate(simulators):
        for column_index, (field, label, _unit, _direction) in enumerate(CRITERIA):
            axis = axes[row_index, column_index]
            relevant = [
                row
                for row in rows
                if row["simulator"] == simulator and row["outcome_id"] == field
            ]
            bounds = [
                float(row["estimate"]) + sign * float(row["mcse"])
                for row in relevant
                for sign in (-1.0, 1.0)
            ]
            axis.set_ylim(*padded_limits(bounds, minimum_span=0.25))
            for policy in POLICIES:
                for stratum in STRATA:
                    chosen = [
                        row
                        for row in relevant
                        if row["policy_id"] == policy and row["stratum"] == stratum
                    ]
                    chosen.sort(key=lambda row: float(row["tau"]))
                    linestyle, marker = strata_styles[stratum]
                    axis.plot(
                        [row["tau"] for row in chosen],
                        [row["estimate"] for row in chosen],
                        color=colors[policy],
                        linestyle=linestyle,
                        marker=marker,
                        markersize=3.7,
                        markerfacecolor="white",
                        markeredgewidth=0.9,
                        linewidth=1.35,
                    )
            if row_index == 0:
                axis.set_title(label)
            if row_index == 1:
                axis.set_xlabel(r"Threshold $\tau$")
            if column_index == 0:
                axis.set_ylabel(simulator_label, fontweight="semibold")
            axis.set_xticks(TAU_LEVELS)
            axis.set_xlim(0.475, 0.925)
            axis.grid(axis="y", color="#e5e5e5", linewidth=0.65)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
    legend = [
        Line2D([], [], color=colors[policy], linewidth=2.2, label=POLICY_LABELS[policy])
        for policy in POLICIES
    ] + [
        Line2D(
            [],
            [],
            color="#333333",
            linestyle=strata_styles[stratum][0],
            marker=strata_styles[stratum][1],
            markerfacecolor="white",
            label=f"stratum $z={stratum}$",
        )
        for stratum in STRATA
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=5,
        frameon=False,
        fontsize=8.0,
    )
    fig.text(
        0.5,
        0.002,
        "All criteria are lower-is-better. Lines are means over 200 seeds; MCSEs are in the source-data CSV.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.105, 1, 1))
    save_figure(
        fig,
        pdf_path,
        png_path,
        title="OSA and Gaussian-bump operating criteria across thresholds",
        subject=(
            "Authenticated full-panel dose-units, posterior efficacy-error, and "
            "above-boundary assignment profiles across two testbeds"
        ),
    )
    plt.close(fig)


def _write_group(
    bundle: ProjectionBundle,
    output_dir: Path,
    *,
    stem: str,
    rows: list[dict[str, Any]],
    plotter: Callable[[list[dict[str, Any]], Path, Path], None],
    record_names: tuple[str, ...],
    analysis: dict[str, Any],
) -> None:
    csv_path = write_csv(output_dir / f"{stem}.csv", rows)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    plotter(rows, pdf_path, png_path)
    generators = (
        Path(__file__),
        Path(__file__).with_name("figure_artifacts.py"),
        Path(__file__).with_name("analysis_record_bundle.py"),
    )
    for path, kind in (
        (csv_path, "figure_source_data_csv"),
        (pdf_path, "vector_figure_pdf"),
        (png_path, "raster_figure_png"),
    ):
        write_artifact_sidecar(
            path,
            artifact_type=kind,
            artifact_class=ARTIFACT_CLASS,
            provenance=bundle.provenance,
            record_names=record_names,
            generator_paths=generators,
            analysis=analysis,
        )


def analyze_bundle(bundle: ProjectionBundle, output_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    records_by_simulator = {
        simulator: bundle.records[record_name]
        for simulator, record_name in RECORDS.items()
    }
    for simulator, records in records_by_simulator.items():
        _validate_main_factorial(records, simulator)
    safety = _safety_rows(records_by_simulator["osa"])
    criteria = _criteria_rows(records_by_simulator)
    output_dir = Path(output_dir)

    _write_group(
        bundle,
        output_dir,
        stem=SAFETY_STEM,
        rows=safety,
        plotter=_plot_safety,
        record_names=(RECORDS["osa"],),
        analysis={
            "role": "candidate journal SI figure",
            "legacy_correspondence": "fig4_safety.pdf",
            "input_factorial_rows": len(records_by_simulator["osa"]),
            "displayed_summary_rows": len(safety),
            "simulator": "osa",
            "dose_access": "full_panel",
            "gate_mode": "latent",
            "policies": list(POLICIES),
            "strata": list(STRATA),
            "tau_levels": list(TAU_LEVELS),
            "independent_seeds_per_cell": len(SEEDS),
            "estimands": [
                "admission-weighted pooled gate precision",
                "all-trial threshold-exceeding final-recommendation percentage",
            ],
            "error_bars": "plus or minus one Monte Carlo standard error",
            "fallback_interpretation": (
                "the threshold-exceedance outcome is simulator-truth reporting; "
                "a fallback assignment is not treated as a safety certificate"
            ),
            "directional_assertion": None,
            "axis_limits": "derived from every displayed estimate plus or minus one MCSE",
        },
    )
    _write_group(
        bundle,
        output_dir,
        stem=CRITERIA_STEM,
        rows=criteria,
        plotter=_plot_criteria,
        record_names=(RECORDS["osa"], RECORDS["gbump"]),
        analysis={
            "role": "candidate journal SI figure",
            "legacy_correspondence": "fig_criteria.pdf",
            "input_factorial_rows": sum(len(value) for value in records_by_simulator.values()),
            "displayed_summary_rows": len(criteria),
            "simulators": ["osa", "gbump"],
            "dose_access": "full_panel",
            "gate_mode": "latent",
            "policies": list(POLICIES),
            "strata": list(STRATA),
            "tau_levels": list(TAU_LEVELS),
            "independent_seeds_per_cell": len(SEEDS),
            "estimands": [item[1] for item in CRITERIA],
            "directional_assertion": None,
            "axis_limits": "derived separately for every simulator-by-outcome panel",
        },
    )
    return {"safety": safety, "criteria": criteria}


def analyze(projection_dir: str | Path, output_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    return analyze_bundle(load_projection_bundle(projection_dir), output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection", help="authenticated formal_projection-* directory")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    analyze(args.projection, args.out_dir)


if __name__ == "__main__":
    main()
