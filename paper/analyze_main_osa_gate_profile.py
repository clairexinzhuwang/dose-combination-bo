#!/usr/bin/env python3
"""Generate the full-panel OSA gate profile from the immutable main records.

This is post-processing only. It does not fit a model, alter a recommendation,
or run a simulation. The reader-facing figure compares cKG with a study-defined
reference schedule alternating cEI and tMSE across successive adaptive cohorts,
compactly labelled ``cEI–tMSE``, at every gate level in the corrected
main OSA sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import dose_combination_bo as sdb

from paper.analysis_record_bundle import load_projection_record

POLICIES = ("cKG", "cEI-tMSE")
CONTRAST_LABEL = "cKG minus cEI–tMSE"
GATE_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)
STRATA = (0, 1)
OSA_RECORD_NAME = "osa_main.json.zst"
BOOTSTRAP_RESAMPLES = 500_000
BOOTSTRAP_SEED = 20_260_822
BOOTSTRAP_BATCH_SIZE = 1_000
SIMULTANEOUS_LEVEL = 0.95
STRATUM_LABELS = {
    None: "OSA strata averaged within matched replicate set",
    0: "Mild/moderate OSA",
    1: "Severe OSA",
}
OUTCOMES = (
    (
        "final_threshold_exceeding_recommendation",
        "True-boundary-exceeding final recommendation",
        "_terminal_threshold_pct",
    ),
    (
        "above_boundary_simulated_assignments",
        "Above-boundary simulated assignments",
        "_above_boundary_assignment_pct",
    ),
)


def _rows(table):
    return table.to_dict("records") if hasattr(table, "to_dict") else list(table)


def _validate_projection_records(records):
    expected = {
        (policy, seed, stratum, tau)
        for policy in ("cEI", "cKG", "cEI-tMSE")
        for seed in range(200)
        for stratum in STRATA
        for tau in GATE_LEVELS
    }
    observed = set()
    for index, row in enumerate(records):
        if row.get("sim") != "osa":
            raise ValueError(f"record {index} is not an OSA trial")
        key = (
            row.get("policy"),
            row.get("seed"),
            row.get("stratum"),
            float(row.get("gamma")),
        )
        if key in observed:
            raise ValueError(f"duplicate OSA main design cell: {key}")
        observed.add(key)
    if observed != expected:
        raise ValueError("OSA main records do not contain the complete frozen factorial")


def _enriched(records, gate_levels=GATE_LEVELS):
    gate_levels = tuple(float(value) for value in gate_levels)
    selected = []
    for row in records:
        if (
            row.get("sim") != "osa"
            or row.get("policy") not in POLICIES
            or float(row.get("gamma")) not in gate_levels
        ):
            continue
        budget = float(row["budget"])
        if budget <= 0:
            raise ValueError("budget must be positive")
        terminal = float(row["rec_unsafe"])
        assignments = float(row["toxic"])
        if terminal not in (0.0, 1.0):
            raise ValueError("the terminal threshold indicator must be binary")
        if not 0 <= assignments <= budget:
            raise ValueError("the above-boundary assignment count is outside the budget")
        item = dict(row)
        item["_terminal_threshold_pct"] = 100.0 * terminal
        item["_above_boundary_assignment_pct"] = 100.0 * assignments / budget
        selected.append(item)
    return selected


def _summary_lookup(records, metric, by):
    output = {}
    for row in _rows(sdb.summarize(records, metric=metric, by=by)):
        key = tuple(row[field] for field in by)
        output[key] = row
    return output


def _profile_rows(records, gate_levels=GATE_LEVELS):
    """Return pooled and stratum-specific paired gate-profile estimates."""
    gate_levels = tuple(float(value) for value in gate_levels)
    selected = _enriched(records, gate_levels)
    if not selected:
        raise ValueError("no OSA gate-profile rows were selected")

    output = []
    for outcome_id, outcome_label, metric in OUTCOMES:
        pooled_contrasts = _rows(sdb.paired_contrast(
            selected,
            POLICIES[0],
            POLICIES[1],
            metric=metric,
            by=("sim", "gamma"),
        ))
        stratum_contrasts = _rows(sdb.paired_contrast(
            selected,
            POLICIES[0],
            POLICIES[1],
            metric=metric,
            by=("sim", "gamma", "stratum"),
        ))
        pooled_means = _summary_lookup(
            selected, metric, ("sim", "policy", "gamma")
        )
        stratum_means = _summary_lookup(
            selected, metric, ("sim", "policy", "gamma", "stratum")
        )

        for contrast in pooled_contrasts + stratum_contrasts:
            stratum = contrast.get("stratum")
            if stratum is None:
                mean_key = lambda policy: ("osa", policy, contrast["gamma"])
                lookup = pooled_means
                level = "pooled_strata"
            else:
                mean_key = lambda policy: (
                    "osa", policy, contrast["gamma"], int(stratum)
                )
                lookup = stratum_means
                level = "stratum"
            mean = float(contrast["difference_mean"])
            mcse = float(contrast["difference_se"])
            output.append({
                "analysis_level": level,
                "stratum": "" if stratum is None else int(stratum),
                "stratum_label": STRATUM_LABELS[
                    None if stratum is None else int(stratum)
                ],
                "tau": float(contrast["gamma"]),
                "outcome_id": outcome_id,
                "outcome": outcome_label,
                "contrast": CONTRAST_LABEL,
                "unit": "percentage points",
                "cKG_mean_pct": float(lookup[mean_key("cKG")][f"{metric}_mean"]),
                "cEI_tMSE_mean_pct": float(
                    lookup[mean_key("cEI-tMSE")][f"{metric}_mean"]
                ),
                "difference_mean_pp": mean,
                "difference_mcse_pp": mcse,
                "mc_interval_95_low_pp": mean - 1.96 * mcse,
                "mc_interval_95_high_pp": mean + 1.96 * mcse,
                "eligible_pairs": int(contrast["n_pairs"]),
                "independent_seeds": int(contrast["n"]),
            })

    level_order = {"pooled_strata": 0, "stratum": 1}
    outcome_order = {item[0]: index for index, item in enumerate(OUTCOMES)}
    output.sort(key=lambda row: (
        float(row["tau"]),
        outcome_order[row["outcome_id"]],
        level_order[row["analysis_level"]],
        -1 if row["stratum"] == "" else int(row["stratum"]),
    ))
    return output, selected


def _seed_contrast_matrix(records, gate_levels=GATE_LEVELS):
    """Return the plotted replicate-level contrast vector and its coordinate order.

    Each coordinate first contrasts the two policies within OSA stratum and then
    averages the two stratum differences within matched replicate set. Keeping the
    complete vector for each replicate preserves common-random-number dependence
    across thresholds
    and outcomes for simultaneous inference.
    """
    gate_levels = tuple(float(value) for value in gate_levels)
    selected = _enriched(records, gate_levels)
    seeds = sorted({int(row["seed"]) for row in selected})
    if len(seeds) < 2:
        raise ValueError("simultaneous bands require at least two replicate sets")

    indexed = {}
    for row in selected:
        key = (
            row["policy"], int(row["seed"]), int(row["stratum"]),
            float(row["gamma"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate OSA gate-profile cell: {key}")
        indexed[key] = row
    expected = {
        (policy, seed, stratum, tau)
        for policy in POLICIES
        for seed in seeds
        for stratum in STRATA
        for tau in gate_levels
    }
    if set(indexed) != expected:
        missing = expected - set(indexed)
        extra = set(indexed) - expected
        example = next(iter(missing or extra))
        raise ValueError(
            "incomplete replicate-level OSA gate-profile factorial; "
            f"example mismatched cell: {example}"
        )

    coordinates = [
        (outcome_id, tau, metric)
        for outcome_id, _outcome_label, metric in OUTCOMES
        for tau in gate_levels
    ]
    matrix = np.empty((len(seeds), len(coordinates)), dtype=float)
    for seed_index, seed in enumerate(seeds):
        for coordinate_index, (_outcome_id, tau, metric) in enumerate(coordinates):
            stratum_differences = []
            for stratum in STRATA:
                left = indexed[(POLICIES[0], seed, stratum, tau)][metric]
                right = indexed[(POLICIES[1], seed, stratum, tau)][metric]
                stratum_differences.append(float(left) - float(right))
            matrix[seed_index, coordinate_index] = float(
                np.mean(stratum_differences)
            )
    if not np.isfinite(matrix).all():
        raise ValueError("replicate-level OSA gate-profile contrasts must be finite")
    return seeds, coordinates, matrix


def _max_t_summary(
    seed_matrix,
    coordinates,
    *,
    n_resamples=BOOTSTRAP_RESAMPLES,
    rng_seed=BOOTSTRAP_SEED,
    batch_size=BOOTSTRAP_BATCH_SIZE,
):
    """Compute outcome-specific and joint paired-replicate bootstrap-t bands.

    Replicate-set rows are sampled with replacement as complete vectors. Each bootstrap
    draw is re-studentized coordinate by coordinate, and the empirical 0.95
    quantile of the absolute maximum t statistic supplies the single-step critical
    value. The joint family contains the ten plotted contrasts; outcome-specific
    five-threshold families are retained in the CSV as a transparent sensitivity.
    """
    matrix = np.asarray(seed_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("seed_matrix must have at least two rows and be two-dimensional")
    if matrix.shape[1] != len(coordinates):
        raise ValueError("seed_matrix columns do not match the coordinate list")
    if not np.isfinite(matrix).all():
        raise ValueError("seed_matrix must be finite")
    n_resamples = int(n_resamples)
    batch_size = int(batch_size)
    if n_resamples < 1 or batch_size < 1:
        raise ValueError("bootstrap resamples and batch size must be positive")

    n_seeds = matrix.shape[0]
    estimates = matrix.mean(axis=0)
    standard_errors = matrix.std(axis=0, ddof=1) / np.sqrt(n_seeds)
    if not np.isfinite(standard_errors).all() or np.any(standard_errors <= 0):
        raise ValueError("every simultaneous-band coordinate must have positive MCSE")

    outcome_ids = tuple(item[0] for item in OUTCOMES)
    family_indices = {
        outcome_id: np.asarray([
            index for index, coordinate in enumerate(coordinates)
            if coordinate[0] == outcome_id
        ], dtype=int)
        for outcome_id in outcome_ids
    }
    if any(len(indices) != len(GATE_LEVELS) for indices in family_indices.values()):
        raise ValueError("each outcome family must contain all five gate levels")

    rng = np.random.Generator(np.random.PCG64(int(rng_seed)))
    max_statistics = {
        outcome_id: np.empty(n_resamples, dtype=float)
        for outcome_id in outcome_ids
    }
    max_statistics["joint_10_contrasts"] = np.empty(n_resamples, dtype=float)
    for start in range(0, n_resamples, batch_size):
        count = min(batch_size, n_resamples - start)
        indices = rng.integers(0, n_seeds, size=(count, n_seeds))
        bootstrap = matrix[indices]
        bootstrap_means = bootstrap.mean(axis=1)
        bootstrap_ses = bootstrap.std(axis=1, ddof=1) / np.sqrt(n_seeds)
        if not np.isfinite(bootstrap_ses).all() or np.any(bootstrap_ses <= 0):
            raise ValueError("a bootstrap resample produced a zero or non-finite MCSE")
        absolute_t = np.abs((bootstrap_means - estimates) / bootstrap_ses)
        for outcome_id, columns in family_indices.items():
            max_statistics[outcome_id][start:start + count] = absolute_t[
                :, columns
            ].max(axis=1)
        max_statistics["joint_10_contrasts"][start:start + count] = (
            absolute_t.max(axis=1)
        )

    critical_values = {
        family: float(np.quantile(
            values, SIMULTANEOUS_LEVEL, method="higher"
        ))
        for family, values in max_statistics.items()
    }
    coordinate_rows = {}
    joint_critical = critical_values["joint_10_contrasts"]
    for index, (outcome_id, tau, _metric) in enumerate(coordinates):
        estimate = float(estimates[index])
        mcse = float(standard_errors[index])
        curve_critical = critical_values[outcome_id]
        coordinate_rows[(outcome_id, float(tau))] = {
            "estimate": estimate,
            "mcse": mcse,
            "curve_max_t_critical": curve_critical,
            "curve_simultaneous_95_low_pp": estimate - curve_critical * mcse,
            "curve_simultaneous_95_high_pp": estimate + curve_critical * mcse,
            "joint10_max_t_critical": joint_critical,
            "joint10_simultaneous_95_low_pp": estimate - joint_critical * mcse,
            "joint10_simultaneous_95_high_pp": estimate + joint_critical * mcse,
        }
    return {
        "coordinates": coordinate_rows,
        "critical_values": critical_values,
        "n_seed_clusters": int(n_seeds),
        "n_resamples": n_resamples,
        "rng": "numpy.random.Generator(PCG64)",
        "rng_seed": int(rng_seed),
        "batch_size": batch_size,
        "confidence_level": SIMULTANEOUS_LEVEL,
        "quantile_method": "higher",
        "studentization": "bootstrap-sample MCSE recomputed for each coordinate",
        "resampling_unit": "complete matched Monte Carlo replicate-set vector",
        "joint_family_size": len(coordinates),
        "post_hoc_descriptive": True,
    }


def _attach_simultaneous_bands(
    profile_rows,
    records,
    *,
    n_resamples=BOOTSTRAP_RESAMPLES,
    rng_seed=BOOTSTRAP_SEED,
    batch_size=BOOTSTRAP_BATCH_SIZE,
):
    """Attach simultaneous bands to the plotted pooled rows in place."""
    _seeds, coordinates, matrix = _seed_contrast_matrix(records)
    summary = _max_t_summary(
        matrix,
        coordinates,
        n_resamples=n_resamples,
        rng_seed=rng_seed,
        batch_size=batch_size,
    )
    band_fields = (
        "curve_max_t_critical",
        "curve_simultaneous_95_low_pp",
        "curve_simultaneous_95_high_pp",
        "joint10_max_t_critical",
        "joint10_simultaneous_95_low_pp",
        "joint10_simultaneous_95_high_pp",
    )
    seen = set()
    for row in profile_rows:
        for field in band_fields:
            row[field] = ""
        if row["analysis_level"] != "pooled_strata":
            continue
        key = (row["outcome_id"], float(row["tau"]))
        values = summary["coordinates"].get(key)
        if values is None or key in seen:
            raise ValueError(f"missing or duplicate pooled profile coordinate: {key}")
        if not np.isclose(
            float(row["difference_mean_pp"]), values["estimate"], atol=1e-12
        ) or not np.isclose(
            float(row["difference_mcse_pp"]), values["mcse"], atol=1e-12
        ):
            raise ValueError(f"simultaneous and pointwise estimands disagree at {key}")
        for field in band_fields:
            row[field] = values[field]
        seen.add(key)
    if seen != set(summary["coordinates"]):
        raise ValueError("the pooled profile does not contain the full joint family")
    return summary


def _plot(profile_rows, pdf_path, png_path, eps_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = [row for row in profile_rows if row["analysis_level"] == "pooled_strata"]
    if len(pooled) != len(GATE_LEVELS) * len(OUTCOMES):
        raise ValueError("the plotted profile must contain both outcomes at every gate")

    styles = {
        "final_threshold_exceeding_recommendation": {
            "color": "#0072B2",
            "marker": "o",
            "linestyle": "-",
            "label": "True-boundary-exceeding final recommendation",
        },
        "above_boundary_simulated_assignments": {
            "color": "#D55E00",
            "marker": "s",
            "linestyle": "--",
            "label": "Above-boundary simulated assignments",
        },
    }
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axis = plt.subplots(figsize=(6.95, 3.25), constrained_layout=True)
    for outcome_id, style in styles.items():
        rows = sorted(
            (row for row in pooled if row["outcome_id"] == outcome_id),
            key=lambda row: float(row["tau"]),
        )
        x = np.asarray([row["tau"] for row in rows], dtype=float)
        y = np.asarray([row["difference_mean_pp"] for row in rows], dtype=float)
        lower = np.asarray(
            [row["joint10_simultaneous_95_low_pp"] for row in rows], dtype=float
        )
        upper = np.asarray(
            [row["joint10_simultaneous_95_high_pp"] for row in rows], dtype=float
        )
        errors = np.vstack((y - lower, upper - y))
        axis.errorbar(
            x,
            y,
            yerr=errors,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.3,
            linewidth=1.8,
            elinewidth=1.1,
            capsize=3,
            label=style["label"],
        )
    axis.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axis.set_xticks(GATE_LEVELS)
    axis.set_xlim(min(GATE_LEVELS) - 0.025, max(GATE_LEVELS) + 0.025)
    axis.set_xlabel(r"Toxicity-rule stringency $\tau$ (larger is stricter)")
    axis.set_ylabel("cKG − cEI–tMSE (percentage points)")
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        borderaxespad=0.0,
    )
    pdf_path = Path(pdf_path)
    png_path = Path(png_path)
    eps_path = Path(eps_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    eps_path.parent.mkdir(parents=True, exist_ok=True)
    title = "Full-panel OSA gate profile: cKG − cEI–tMSE"
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Subject": (
                "cKG contrasts versus the study-defined cEI–tMSE schedule; under "
                "full-panel access its 18 scheduled adaptive-cohort positions assign "
                "nine to cEI and nine to tMSE"
            ),
            "Keywords": "dose optimization, gate profile, Monte Carlo",
            # Suppress clock-dependent PDF fields so identical inputs generate
            # byte-identical figures in a fixed software environment.
            "CreationDate": None,
            "ModDate": None,
        },
    )
    previous_source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        fig.savefig(
            eps_path,
            format="eps",
            bbox_inches="tight",
            metadata={"Creator": "dose-combination-bo figure generator"},
        )
    finally:
        if previous_source_date_epoch is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = previous_source_date_epoch
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": title,
            "Description": (
                "Generated from frozen full-panel OSA trial records; post hoc "
                "paired-replicate simultaneous max-t Monte Carlo bands."
            ),
        },
    )
    plt.close(fig)


def _write_metadata(path, selected, artifact_type, simultaneous, provenance):
    path = Path(path)
    metadata = sdb.write_artifact_metadata(
        path, selected, artifact_type=artifact_type,
    )
    commitment = provenance["selected_record_commitment"]
    metadata.update({
        "source_path": OSA_RECORD_NAME,
        "source_sha256": commitment["compressed_sha256"],
        "source_uncompressed_sha256": commitment["uncompressed_sha256"],
        "formal_projection": {
            "projection_metadata_path": provenance["projection_metadata_path"],
            "projection_metadata_sha256": provenance["projection_metadata_sha256"],
            "projection_fingerprint": provenance["projection_fingerprint"],
            "formal_manifest_sha256": provenance["formal_manifest_sha256"],
            "record": OSA_RECORD_NAME,
            "record_commitment": dict(commitment),
        },
    })
    metadata["estimand"] = {
        "contrast": CONTRAST_LABEL,
        "dose_access": "full-panel access",
        "reader_facing_policy_alias": {"cEI-tMSE": "cEI–tMSE"},
        "full_panel_reference_schedule": {
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
        },
        "gate_levels": list(GATE_LEVELS),
        "pairing": (
            "Policies are paired within Monte Carlo replicate set, OSA stratum, gate level, and the "
            "complete serialized trial-design identity."
        ),
        "pooled_summary": (
            "The two stratum-specific differences are averaged within matched replicate "
            "set before the Monte Carlo standard error is computed across M=200 sets."
        ),
        "pointwise_interval": (
            "point estimate plus or minus 1.96 replicate-level Monte Carlo SE; "
            "retained for the pooled and stratum-specific CSV rows"
        ),
        "plotted_interval": (
            "post hoc two-sided 95% paired-replicate bootstrap-t max-t band using "
            "one joint family of the two outcomes by five gate levels"
        ),
        "multiplicity": (
            "the plotted pooled profile is simultaneous across all 10 displayed "
            "contrasts; stratum-specific and other manuscript results remain "
            "exploratory and unadjusted"
        ),
        "public_field_mapping": {
            "true-boundary-exceeding final recommendation": "rec_unsafe",
            "above-boundary simulated assignments": "100 * toxic / budget",
            "posterior-feasibility threshold tau": "gamma",
        },
    }
    metadata["simultaneous_max_t"] = {
        key: value for key, value in simultaneous.items() if key != "coordinates"
    }
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def analyze(
    projection_dir,
    output_dir,
    *,
    n_resamples=BOOTSTRAP_RESAMPLES,
    rng_seed=BOOTSTRAP_SEED,
    batch_size=BOOTSTRAP_BATCH_SIZE,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, provenance = load_projection_record(projection_dir, OSA_RECORD_NAME)
    _validate_projection_records(records)
    profile_rows, selected = _profile_rows(records)
    simultaneous = _attach_simultaneous_bands(
        profile_rows,
        selected,
        n_resamples=n_resamples,
        rng_seed=rng_seed,
        batch_size=batch_size,
    )

    csv_path = output_dir / "osa_gate_profile_contrasts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)
    pdf_path = output_dir / "osa_gate_profile.pdf"
    png_path = output_dir / "osa_gate_profile.png"
    eps_path = output_dir / "osa_gate_profile.eps"
    _plot(profile_rows, pdf_path, png_path, eps_path)

    for artifact, kind in (
        (csv_path, "paired_gate_profile_data"),
        (pdf_path, "paired_gate_profile_figure_pdf"),
        (png_path, "paired_gate_profile_figure_png"),
        (eps_path, "paired_gate_profile_figure_eps"),
    ):
        _write_metadata(
            artifact.with_suffix(artifact.suffix + ".metadata.json"),
            selected,
            kind,
            simultaneous,
            provenance,
        )
    print(f"wrote {csv_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {eps_path}")
    return profile_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection", help="authenticated formal_projection-* directory")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    analyze(args.projection, args.out_dir)


if __name__ == "__main__":
    main()
