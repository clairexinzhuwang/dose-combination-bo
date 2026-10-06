#!/usr/bin/env python3
"""Generate the OSA dose-selection precision SI figure from formal records.

This is authenticated post-processing only. It reads the complete six-policy
formal projection, verifies its frozen factorial, and plots the cEI and cKG
OSA latent-gate dose-units profiles. No outcome ordering is prespecified.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

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
except ModuleNotFoundError:  # Support ``python paper/make_fig2.py``.
    from analysis_record_bundle import ProjectionBundle, load_projection_bundle
    from figure_artifacts import (
        finite_float,
        padded_limits,
        save_figure,
        write_artifact_sidecar,
        write_csv,
    )


RECORD_NAME = "gate_sixway_supplemental.json.zst"
SIMULATORS = ("gbump", "osa")
MODES = ("latent", "predictive")
ALL_POLICIES = ("GBE", "cEI", "cKG1fix", "qBIG", "random", "straddle")
PLOTTED_POLICIES = ("cEI", "cKG1fix")
POLICY_LABELS = {"cEI": "cEI", "cKG1fix": "cKG"}
STRATA = (0, 1)
TAU_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)
SEEDS = tuple(range(100))
OUTPUT_STEM = "si_osa_dose_selection_precision_by_threshold"
ARTIFACT_CLASS = "journal_supplement_figure_candidate"


def _validate_full_factorial(records: list[dict[str, Any]]) -> None:
    """Validate the complete 24,000-row six-way formal design structurally."""

    expected = set(
        itertools.product(
            SIMULATORS, MODES, ALL_POLICIES, SEEDS, STRATA, TAU_LEVELS
        )
    )
    observed: set[tuple[Any, ...]] = set()
    for index, row in enumerate(records):
        key = (
            row.get("sim"),
            row.get("mode"),
            row.get("policy"),
            row.get("seed"),
            row.get("stratum"),
            finite_float(row.get("gamma"), label=f"record {index} gamma"),
        )
        if key in observed:
            raise ValueError(f"duplicate six-way design cell: {key}")
        observed.add(key)
        fixed = {
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
        dose_units = finite_float(
            row.get("dose_units"), label=f"record {index} dose_units"
        )
        toxic = finite_float(row.get("toxic"), label=f"record {index} toxic")
        rec_unsafe = finite_float(
            row.get("rec_unsafe"), label=f"record {index} rec_unsafe"
        )
        gate_pass = finite_float(
            row.get("n_gate_pass"), label=f"record {index} n_gate_pass"
        )
        gate_safe = finite_float(
            row.get("n_gate_pass_safe"),
            label=f"record {index} n_gate_pass_safe",
        )
        if dose_units < 0.0:
            raise ValueError("dose-units must be nonnegative")
        if not 0.0 <= toxic <= 40.0:
            raise ValueError("above-boundary assignments are outside the trial budget")
        if rec_unsafe not in (0.0, 1.0):
            raise ValueError("recommendation threshold indicator must be binary")
        if not 0.0 <= gate_safe <= gate_pass:
            raise ValueError("safe gate admissions cannot exceed all gate admissions")
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        example = next(iter(missing or extra))
        raise ValueError(
            "six-way records do not contain the complete frozen factorial; "
            f"example mismatched cell: {example}"
        )


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute dose-units means and Monte Carlo SEs without sign assertions."""

    selected = [
        row
        for row in records
        if row["sim"] == "osa"
        and row["mode"] == "latent"
        and row["policy"] in PLOTTED_POLICIES
    ]
    expected_rows = len(PLOTTED_POLICIES) * len(STRATA) * len(TAU_LEVELS) * len(SEEDS)
    if len(selected) != expected_rows:
        raise ValueError("OSA latent cEI/cKG selection is not the complete factorial")

    output: list[dict[str, Any]] = []
    for stratum in STRATA:
        for tau in TAU_LEVELS:
            for policy in PLOTTED_POLICIES:
                values = np.asarray(
                    [
                        finite_float(row["dose_units"], label="dose_units")
                        for row in selected
                        if row["stratum"] == stratum
                        and float(row["gamma"]) == tau
                        and row["policy"] == policy
                    ],
                    dtype=float,
                )
                if values.size != len(SEEDS) or not np.isfinite(values).all():
                    raise ValueError(
                        f"incomplete/non-finite OSA precision cell: "
                        f"stratum={stratum}, tau={tau}, policy={policy}"
                    )
                output.append(
                    {
                        "simulator": "osa",
                        "dose_access": "full_panel",
                        "gate_mode": "latent",
                        "stratum": stratum,
                        "tau": tau,
                        "policy_id": policy,
                        "policy": POLICY_LABELS[policy],
                        "independent_seeds": int(values.size),
                        "dose_units_mean": float(values.mean()),
                        "dose_units_mcse": float(
                            values.std(ddof=1) / np.sqrt(values.size)
                        ),
                    }
                )
    if len(output) != len(PLOTTED_POLICIES) * len(STRATA) * len(TAU_LEVELS):
        raise ValueError("precision summary does not contain every plotted cell")
    return output


def _plot(rows: list[dict[str, Any]], pdf_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_limits = [
        float(row["dose_units_mean"]) + sign * float(row["dose_units_mcse"])
        for row in rows
        for sign in (-1.0, 1.0)
    ]
    y_limits = padded_limits(all_limits, minimum_span=0.25)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    styles = {
        "cEI": {"color": "#D55E00", "marker": "o"},
        "cKG1fix": {"color": "#0072B2", "marker": "s"},
    }
    for axis, stratum in zip(axes, STRATA):
        for policy in PLOTTED_POLICIES:
            chosen = [
                row
                for row in rows
                if row["stratum"] == stratum and row["policy_id"] == policy
            ]
            chosen.sort(key=lambda row: float(row["tau"]))
            style = styles[policy]
            axis.errorbar(
                [row["tau"] for row in chosen],
                [row["dose_units_mean"] for row in chosen],
                yerr=[row["dose_units_mcse"] for row in chosen],
                label=POLICY_LABELS[policy],
                color=style["color"],
                marker=style["marker"],
                markerfacecolor="white",
                markeredgewidth=1.2,
                linewidth=1.6,
                capsize=2.5,
            )
        axis.set_title(f"OSA stratum $z={stratum}$", loc="left")
        axis.set_xlabel(r"Toxicity-rule stringency $\tau$ (larger is stricter)")
        axis.set_xticks(TAU_LEVELS)
        axis.set_xlim(0.475, 0.925)
        axis.set_ylim(*y_limits)
        axis.grid(axis="y", color="#e2e2e2", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Dose-units (lower is better)")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("OSA dose-selection precision across toxicity-rule settings", y=0.995)
    fig.text(
        0.5,
        0.005,
        "Points are means over 100 independent seeds; bars are +/-1 Monte Carlo SE.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    save_figure(
        fig,
        pdf_path,
        png_path,
        title="OSA dose-selection precision across toxicity-rule settings",
        subject=(
            "Authenticated full-panel latent-gate OSA dose-units means for cEI "
            "and cKG across five toxicity-rule settings and two strata"
        ),
    )
    plt.close(fig)


def analyze_bundle(bundle: ProjectionBundle, output_dir: str | Path) -> list[dict[str, Any]]:
    records = bundle.records[RECORD_NAME]
    _validate_full_factorial(records)
    rows = _summary_rows(records)
    output_dir = Path(output_dir)
    csv_path = write_csv(output_dir / f"{OUTPUT_STEM}.csv", rows)
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    _plot(rows, pdf_path, png_path)

    analysis = {
        "role": "candidate journal SI figure",
        "legacy_correspondence": "fig2_precision.pdf",
        "input_factorial_rows": len(records),
        "selected_trial_rows": 2_000,
        "displayed_summary_rows": len(rows),
        "simulator": "osa",
        "dose_access": "full_panel",
        "gate_mode": "latent",
        "policies": [POLICY_LABELS[value] for value in PLOTTED_POLICIES],
        "strata": list(STRATA),
        "tau_levels": list(TAU_LEVELS),
        "independent_seeds_per_cell": len(SEEDS),
        "estimand": "mean dose-units by policy, stratum, and threshold",
        "error_bars": "plus or minus one Monte Carlo standard error across seeds",
        "directional_assertion": None,
        "axis_limits": "derived from every displayed mean plus or minus one MCSE",
    }
    generator_paths = (
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
            record_names=(RECORD_NAME,),
            generator_paths=generator_paths,
            analysis=analysis,
        )
    return rows


def analyze(projection_dir: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
    return analyze_bundle(load_projection_bundle(projection_dir), output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection", help="authenticated formal_projection-* directory")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    analyze(args.projection, args.out_dir)


if __name__ == "__main__":
    main()
