#!/usr/bin/env python3
"""Generate the archive-only OSA trajectory diagnostic from formal records.

This artifact is intentionally excluded from the journal manuscript and SI.
It summarizes operating-characteristic evolution; it does not contain
acquisition values and cannot evaluate acquisition-threshold stopping rules.
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
        write_json,
    )
except ModuleNotFoundError:  # Support direct script execution.
    from analysis_record_bundle import ProjectionBundle, load_projection_bundle
    from figure_artifacts import (
        finite_float,
        padded_limits,
        save_figure,
        write_artifact_sidecar,
        write_csv,
        write_json,
    )


RECORD_NAME = "osa_trajectory_supplemental.json.zst"
POLICIES = ("cEI", "cKG1fix", "GBE")
POLICY_LABELS = {"cEI": "cEI", "cKG1fix": "cKG", "GBE": "cEI-tMSE"}
STRATA = (0, 1)
SEEDS = tuple(range(80))
TAU = 0.7
STORED_ENROLLMENT_STATES = tuple(range(4, 40, 2))
FINAL_ENROLLMENT = 40
OUTPUT_STEM = "archive_only_osa_trajectory_diagnostic"
ARTIFACT_CLASS = "archive_only_github_diagnostic_not_manuscript"
METRICS = (
    ("du", "dose_units", "Dose-units", "lower is better"),
    ("rpsel", "rpsel", "Posterior efficacy error", "lower is better"),
    (
        "toxic",
        "toxic",
        "Cumulative above-boundary assignments",
        "lower is better",
    ),
)


def _validate_trajectory_records(records: list[dict[str, Any]]) -> None:
    """Require exactly 480 rows and exactly 18 serialized states per row."""

    if len(records) != 480:
        raise ValueError(f"trajectory record must contain exactly 480 rows, not {len(records)}")
    expected = set(itertools.product(POLICIES, SEEDS, STRATA))
    observed: set[tuple[Any, ...]] = set()
    for index, row in enumerate(records):
        key = (row.get("policy"), row.get("seed"), row.get("stratum"))
        if key in observed:
            raise ValueError(f"duplicate trajectory design cell: {key}")
        observed.add(key)
        fixed = {
            "sim": "osa",
            "mode": "latent",
            "gamma": TAU,
            "budget": FINAL_ENROLLMENT,
            "grid_n": 5,
            "r_k": 2,
            "warmup": 4,
            "noise": "fixed",
            "empty_gate": "pf",
        }
        for field, expected_value in fixed.items():
            observed_value = row.get(field)
            if field == "gamma":
                observed_value = finite_float(observed_value, label=f"record {index} gamma")
            if observed_value != expected_value:
                raise ValueError(
                    f"trajectory record {index} has {field}={observed_value!r}; "
                    f"expected {expected_value!r}"
                )
        trajectory = row.get("traj")
        if not isinstance(trajectory, dict) or set(trajectory) != {
            "n",
            "du",
            "rpsel",
            "toxic",
        }:
            raise ValueError("every trajectory row must contain n/du/rpsel/toxic arrays")
        if tuple(trajectory["n"]) != STORED_ENROLLMENT_STATES:
            raise ValueError(
                "every trajectory row must contain the same 18 enrollment states "
                "from 4 through 38"
            )
        for trajectory_field, final_field, _label, _direction in METRICS:
            values = trajectory.get(trajectory_field)
            if not isinstance(values, list) or len(values) != 18:
                raise ValueError(
                    f"trajectory {trajectory_field} must contain exactly 18 states"
                )
            numeric = np.asarray(
                [
                    finite_float(value, label=f"record {index} {trajectory_field}")
                    for value in values
                ],
                dtype=float,
            )
            final = finite_float(row.get(final_field), label=f"record {index} {final_field}")
            if np.any(numeric < 0.0) or final < 0.0:
                raise ValueError("trajectory operating criteria must be nonnegative")
            if trajectory_field == "toxic":
                if np.any(np.diff(numeric) < 0.0) or final < numeric[-1]:
                    raise ValueError("cumulative above-boundary assignments must be monotone")
                if final > FINAL_ENROLLMENT:
                    raise ValueError("final above-boundary assignments exceed enrollment")
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        example = next(iter(missing or extra))
        raise ValueError(
            "trajectory records do not contain 80 seeds x 2 strata x 3 policies; "
            f"example mismatched cell: {example}"
        )


def _curve_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["policy"], int(row["seed"]), int(row["stratum"])): row
        for row in records
    }
    output: list[dict[str, Any]] = []
    enrollment = STORED_ENROLLMENT_STATES + (FINAL_ENROLLMENT,)
    for policy in POLICIES:
        for trajectory_field, final_field, label, direction in METRICS:
            seed_curves = []
            for seed in SEEDS:
                stratum_curves = []
                for stratum in STRATA:
                    row = indexed[(policy, seed, stratum)]
                    values = [float(value) for value in row["traj"][trajectory_field]]
                    values.append(float(row[final_field]))
                    stratum_curves.append(np.asarray(values, dtype=float))
                seed_curves.append(np.mean(stratum_curves, axis=0))
            matrix = np.asarray(seed_curves, dtype=float)
            if matrix.shape != (80, 19) or not np.isfinite(matrix).all():
                raise ValueError("pooled trajectory matrix is incomplete or non-finite")
            means = matrix.mean(axis=0)
            mcses = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
            for state_index, (n_enrolled, mean, mcse) in enumerate(
                zip(enrollment, means, mcses)
            ):
                output.append(
                    {
                        "scope": "archive_only_not_manuscript",
                        "simulator": "osa",
                        "dose_access": "full_panel",
                        "gate_mode": "latent",
                        "tau": TAU,
                        "policy_id": policy,
                        "policy": POLICY_LABELS[policy],
                        "outcome_id": trajectory_field,
                        "outcome": label,
                        "direction": direction,
                        "state_index": state_index,
                        "state_source": (
                            "serialized_trajectory" if n_enrolled < FINAL_ENROLLMENT else "final_trial_record"
                        ),
                        "patients_enrolled": n_enrolled,
                        "independent_seed_clusters": len(SEEDS),
                        "strata_averaged_within_seed": len(STRATA),
                        "mean": float(mean),
                        "mcse": float(mcse),
                    }
                )
    if len(output) != len(POLICIES) * len(METRICS) * 19:
        raise ValueError("trajectory summary does not contain all curve coordinates")
    return output


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints = []
    for policy in POLICIES:
        for metric, _final, label, _direction in METRICS:
            chosen = [
                row
                for row in rows
                if row["policy_id"] == policy and row["outcome_id"] == metric
            ]
            chosen.sort(key=lambda row: int(row["patients_enrolled"]))
            endpoints.append(
                {
                    "policy": POLICY_LABELS[policy],
                    "outcome": label,
                    "first_state": {
                        "patients_enrolled": chosen[0]["patients_enrolled"],
                        "mean": chosen[0]["mean"],
                        "mcse": chosen[0]["mcse"],
                    },
                    "final_state": {
                        "patients_enrolled": chosen[-1]["patients_enrolled"],
                        "mean": chosen[-1]["mean"],
                        "mcse": chosen[-1]["mcse"],
                    },
                }
            )
    return {
        "schema_version": 1,
        "scope": "archive_only_github_diagnostic_not_manuscript",
        "interpretation": (
            "Operating-characteristic evolution only. The records do not contain "
            "acquisition values and cannot evaluate acquisition-threshold stopping."
        ),
        "input_trial_rows": 480,
        "design": {
            "independent_seeds": 80,
            "strata": 2,
            "policies": 3,
            "serialized_states_per_trial_row": 18,
            "serialized_enrollment_states": list(STORED_ENROLLMENT_STATES),
            "final_state_from_trial_record": FINAL_ENROLLMENT,
        },
        "summary_rows": len(rows),
        "endpoints": endpoints,
    }


def _plot(rows: list[dict[str, Any]], pdf_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
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
    colors = {"cEI": "#D55E00", "cKG1fix": "#0072B2", "GBE": "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.55))
    for axis, (metric, _final, label, _direction) in zip(axes, METRICS):
        relevant = [row for row in rows if row["outcome_id"] == metric]
        bounds = [
            float(row["mean"]) + sign * float(row["mcse"])
            for row in relevant
            for sign in (-1.0, 1.0)
        ]
        axis.set_ylim(*padded_limits(bounds, minimum_span=0.25))
        for policy in POLICIES:
            chosen = [row for row in relevant if row["policy_id"] == policy]
            chosen.sort(key=lambda row: int(row["patients_enrolled"]))
            x = np.asarray([row["patients_enrolled"] for row in chosen], dtype=float)
            mean = np.asarray([row["mean"] for row in chosen], dtype=float)
            mcse = np.asarray([row["mcse"] for row in chosen], dtype=float)
            axis.plot(
                x,
                mean,
                color=colors[policy],
                label=POLICY_LABELS[policy],
                marker="o",
                markersize=2.7,
                linewidth=1.4,
            )
            axis.fill_between(x, mean - mcse, mean + mcse, color=colors[policy], alpha=0.14)
        axis.set_title(label)
        axis.set_xlabel("Patients enrolled")
        axis.grid(axis="y", color="#e3e3e3", linewidth=0.65)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.0)
    fig.suptitle(
        "ARCHIVE-ONLY DIAGNOSTIC: OSA operating-criterion trajectories",
        color="#8B1A1A",
        fontweight="semibold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        (
            "Not a manuscript or SI figure. Strata are averaged within each of 80 seeds; "
            "bands are +/-1 MCSE. These records contain no acquisition values."
        ),
        ha="center",
        fontsize=7.9,
        color="#8B1A1A",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    save_figure(
        fig,
        pdf_path,
        png_path,
        title="Archive-only OSA operating-criterion trajectories",
        subject=(
            "Not a manuscript figure; authenticated diagnostic from 480 formal "
            "trajectory trial records with 18 serialized states per row"
        ),
    )
    plt.close(fig)


def analyze_bundle(bundle: ProjectionBundle, output_dir: str | Path) -> list[dict[str, Any]]:
    records = bundle.records[RECORD_NAME]
    _validate_trajectory_records(records)
    rows = _curve_rows(records)
    summary = _summary(rows)
    output_dir = Path(output_dir)
    csv_path = write_csv(output_dir / f"{OUTPUT_STEM}.csv", rows)
    json_path = write_json(output_dir / f"{OUTPUT_STEM}_summary.json", summary)
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    _plot(rows, pdf_path, png_path)
    analysis = {
        "role": "archive/GitHub-only diagnostic; excluded from manuscript and SI",
        "legacy_correspondence": "fig_traj.pdf",
        "input_trial_rows": 480,
        "design_identity": "80 seeds x 2 strata x 3 policies",
        "serialized_states_per_trial_row": 18,
        "displayed_states_per_curve": 19,
        "final_state_source": "terminal fields in the trial record at n=40",
        "displayed_summary_rows": len(rows),
        "tau": TAU,
        "directional_assertion": None,
        "stopping_rule_claim": None,
        "axis_limits": "derived separately for every outcome from all means +/-1 MCSE",
    }
    generators = (
        Path(__file__),
        Path(__file__).with_name("figure_artifacts.py"),
        Path(__file__).with_name("analysis_record_bundle.py"),
    )
    for path, kind in (
        (csv_path, "diagnostic_source_data_csv"),
        (json_path, "diagnostic_summary_json"),
        (pdf_path, "diagnostic_vector_figure_pdf"),
        (png_path, "diagnostic_raster_figure_png"),
    ):
        write_artifact_sidecar(
            path,
            artifact_type=kind,
            artifact_class=ARTIFACT_CLASS,
            provenance=bundle.provenance,
            record_names=(RECORD_NAME,),
            generator_paths=generators,
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
