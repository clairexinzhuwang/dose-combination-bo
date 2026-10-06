#!/usr/bin/env python3
"""Plot supporting-surface terminal and assignment contrast profiles.

This script performs post-processing only. It authenticates a completed formal
projection, reads its four main-run record sets, and computes cKG minus a
study-defined reference schedule alternating cEI and tMSE across successive
adaptive cohorts (compact label: ``cEI–tMSE``) in (1) the all-trial indicator that the final
recommendation exceeds the simulated latent boundary and (2) the percentage of
assignments above that boundary. The two stratum differences are averaged
within each matched Monte Carlo replicate set before uncertainty is estimated.
The figure displays the three supporting surfaces; the OSA primary profile is
reported separately with simultaneous bands. No model is fitted and no trial is
rerun. Scientific-result summaries in the figure and sidecars are derived from
the supplied records, never encoded as acceptance assertions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import dose_combination_bo as sdb

try:  # Package import in tests and module execution.
    from paper.analysis_record_bundle import ProjectionBundle, load_projection_bundle
except ModuleNotFoundError:  # Direct execution from the paper/ directory.
    from analysis_record_bundle import ProjectionBundle, load_projection_bundle


POLICIES = ("cKG", "cEI-tMSE")
CONTRAST_LABEL = "cKG minus cEI–tMSE"
MAIN_POLICIES = ("cEI", "cKG", "cEI-tMSE")
GATE_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)
STRATA = (0, 1)
POINTWISE_CRITICAL = 1.96
TESTBEDS = (
    {"id": "osa", "label": "OSA", "record_file": "osa_main.json.zst"},
    {
        "id": "gbump",
        "label": "Gaussian bump",
        "record_file": "gbump_main.json.zst",
    },
    {
        "id": "efftox",
        "label": "Logistic",
        "record_file": "efftox_main.json.zst",
    },
    {
        "id": "mariposa",
        "label": "MARIPOSA-motivated",
        "record_file": "mariposa_main.json.zst",
    },
)
PLOT_TESTBEDS = tuple(item for item in TESTBEDS if item["id"] != "osa")
TESTBED_BY_ID = {item["id"]: item for item in TESTBEDS}
OUTCOMES = (
    {
        "id": "terminal_recommendation",
        "metric": "rec_unsafe",
        "derived_metric": "_terminal_recommendation_pct",
        "label": "True-boundary-exceeding final recommendation",
        "legend": "True-boundary-exceeding final recommendation",
        "annotation": "terminal-recommendation",
    },
    {
        "id": "assignment_percentage",
        "metric": "toxic",
        "derived_metric": "_assignment_percentage",
        "label": "Above-boundary simulated assignments",
        "legend": "Above-boundary simulated assignments",
        "annotation": "assignment",
    },
)
OUTCOME_BY_ID = {item["id"]: item for item in OUTCOMES}


def _rows(table):
    return table.to_dict("records") if hasattr(table, "to_dict") else list(table)


def _validate_main_records(records, testbed):
    """Validate one complete main-run factorial without asserting its results."""
    records = list(records)
    if not records:
        raise ValueError(f"{testbed['label']} main records are empty")

    required = {
        "policy",
        "seed",
        "sim",
        "stratum",
        "gamma",
        "mode",
        "budget",
        "rec_unsafe",
        "toxic",
    }
    observed = set()
    seeds = set()
    budgets = set()
    for index, row in enumerate(records):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"record {index} in {testbed['label']} is missing fields {missing}"
            )
        if row["sim"] != testbed["id"]:
            raise ValueError(
                f"record {index} in {testbed['label']} has an unexpected testbed id"
            )
        if row["mode"] != "latent":
            raise ValueError(
                f"record {index} in {testbed['label']} is not a latent main-run row"
            )
        try:
            seed = int(row["seed"])
            stratum = int(row["stratum"])
            tau = float(row["gamma"])
            budget = int(row["budget"])
            rec_unsafe = float(row["rec_unsafe"])
            toxic = float(row["toxic"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"record {index} in {testbed['label']} has invalid numeric fields"
            ) from exc
        if seed < 0:
            raise ValueError(f"record {index} has a negative seed")
        if stratum not in STRATA:
            raise ValueError(f"record {index} has an unexpected stratum {stratum}")
        if tau not in GATE_LEVELS:
            raise ValueError(f"record {index} has an unexpected threshold {tau}")
        if row["policy"] not in MAIN_POLICIES:
            raise ValueError(
                f"record {index} has an unexpected policy {row['policy']!r}"
            )
        if budget <= 0:
            raise ValueError(f"record {index} has a nonpositive assignment budget")
        if rec_unsafe not in {0.0, 1.0}:
            raise ValueError(f"record {index} has a nonbinary rec_unsafe value")
        if not np.isfinite(toxic) or toxic < 0.0 or toxic > budget:
            raise ValueError(f"record {index} has an invalid toxic-assignment count")
        if not toxic.is_integer():
            raise ValueError(f"record {index} has a noninteger toxic-assignment count")
        key = (row["policy"], seed, stratum, tau)
        if key in observed:
            raise ValueError(f"duplicate main-run design cell: {key}")
        observed.add(key)
        seeds.add(seed)
        budgets.add(budget)

    if len(seeds) < 2:
        raise ValueError(
            f"{testbed['label']} requires at least two replicate sets for an MCSE"
        )
    expected = {
        (policy, seed, stratum, tau)
        for policy in MAIN_POLICIES
        for seed in seeds
        for stratum in STRATA
        for tau in GATE_LEVELS
    }
    if observed != expected:
        example = next(iter((expected - observed) or (observed - expected)))
        raise ValueError(
            "main records do not contain the complete policy-by-seed-by-stratum-by-"
            f"threshold factorial; example inconsistent cell: {example}"
        )
    if len(budgets) != 1:
        raise ValueError(
            f"{testbed['label']} main factorial mixes assignment budgets {sorted(budgets)}"
        )
    return {
        "seed_count": len(seeds),
        "seeds": tuple(sorted(seeds)),
        "budget": next(iter(budgets)),
        "rows": len(records),
    }


def _analysis_records(records):
    """Add percentage metrics while preserving all stored design fields."""
    output = []
    for row in records:
        enriched = dict(row)
        made = bool(row.get("recommendation_made", True))
        terminal = 0.0 if not made else float(row["rec_unsafe"])
        enriched["_terminal_recommendation_pct"] = 100.0 * terminal
        enriched["_assignment_percentage"] = (
            100.0 * float(row["toxic"]) / float(row["budget"])
        )
        output.append(enriched)
    return output


def _profile_rows(records_by_testbed):
    """Return within-seed cross-testbed contrasts in percentage points."""
    output = []
    selected = []
    for testbed in TESTBEDS:
        sim = testbed["id"]
        records = list(records_by_testbed[sim])
        _validate_main_records(records, testbed)
        chosen_original = [row for row in records if row["policy"] in POLICIES]
        chosen = _analysis_records(chosen_original)
        selected.extend(chosen_original)

        for outcome in OUTCOMES:
            metric = outcome["derived_metric"]
            contrasts = _rows(
                sdb.paired_contrast(
                    chosen,
                    POLICIES[0],
                    POLICIES[1],
                    metric=metric,
                    by=("sim", "gamma"),
                )
            )
            means = {
                (row["policy"], float(row["gamma"])): row
                for row in _rows(
                    sdb.summarize(
                        chosen,
                        metric=metric,
                        by=("policy", "gamma"),
                    )
                )
            }
            if len(contrasts) != len(GATE_LEVELS):
                raise ValueError(
                    f"{testbed['label']} must contribute all five thresholds "
                    f"for {outcome['id']}"
                )

            for contrast in contrasts:
                tau = float(contrast["gamma"])
                estimate = float(contrast["difference_mean"])
                mcse = float(contrast["difference_se"])
                low = estimate - POINTWISE_CRITICAL * mcse
                high = estimate + POINTWISE_CRITICAL * mcse
                output.append(
                    {
                        "testbed_id": sim,
                        "testbed": testbed["label"],
                        "tau": tau,
                        "outcome_id": outcome["id"],
                        "outcome": outcome["label"],
                        "contrast": CONTRAST_LABEL,
                        "unit": "percentage points",
                        "cKG_mean_pct": float(means[("cKG", tau)][f"{metric}_mean"]),
                        "cEI_tMSE_mean_pct": float(
                            means[("cEI-tMSE", tau)][f"{metric}_mean"]
                        ),
                        "difference_mean_pp": estimate,
                        "difference_mcse_pp": mcse,
                        "pointwise_95_low_pp": low,
                        "pointwise_95_high_pp": high,
                        "independent_seeds": int(contrast["n"]),
                        "stratum_cells": int(contrast["n_pairs"]),
                        "interval_excludes_zero": bool(low > 0.0 or high < 0.0),
                    }
                )

    testbed_order = {item["id"]: index for index, item in enumerate(TESTBEDS)}
    outcome_order = {item["id"]: index for index, item in enumerate(OUTCOMES)}
    output.sort(
        key=lambda row: (
            outcome_order[row["outcome_id"]],
            testbed_order[row["testbed_id"]],
            row["tau"],
        )
    )
    _validate_profile_result(output)
    return output, selected


def _validate_profile_result(rows):
    """Protect the estimand and factorial shape without asserting result signs."""
    expected_cells = {
        (outcome["id"], testbed["id"], tau)
        for outcome in OUTCOMES
        for testbed in TESTBEDS
        for tau in GATE_LEVELS
    }
    observed_cells = {
        (row["outcome_id"], row["testbed_id"], float(row["tau"])) for row in rows
    }
    if len(rows) != len(expected_cells) or observed_cells != expected_cells:
        raise ValueError(
            "the cross-testbed profile must contain one unique contrast for every "
            "outcome-by-testbed-by-threshold cell"
        )

    seed_counts = {}
    for row in rows:
        estimate = float(row["difference_mean_pp"])
        mcse = float(row["difference_mcse_pp"])
        low = float(row["pointwise_95_low_pp"])
        high = float(row["pointwise_95_high_pp"])
        left = float(row["cKG_mean_pct"])
        right = float(row["cEI_tMSE_mean_pct"])
        values = (estimate, mcse, low, high, left, right)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("profile estimates and intervals must be finite")
        if mcse < 0.0:
            raise ValueError("profile MCSEs must be nonnegative")
        if not (0.0 <= left <= 100.0 and 0.0 <= right <= 100.0):
            raise ValueError("displayed policy rates must lie between 0 and 100")
        if not np.isclose(estimate, left - right, atol=1e-12, rtol=0.0):
            raise ValueError("a displayed contrast does not equal its two policy means")
        if not np.isclose(
            low, estimate - POINTWISE_CRITICAL * mcse, atol=1e-12, rtol=0.0
        ) or not np.isclose(
            high, estimate + POINTWISE_CRITICAL * mcse, atol=1e-12, rtol=0.0
        ):
            raise ValueError(
                "a displayed interval does not equal estimate +/- 1.96 MCSE"
            )
        expected_exclusion = low > 0.0 or high < 0.0
        if bool(row["interval_excludes_zero"]) != expected_exclusion:
            raise ValueError(
                "interval_excludes_zero is inconsistent with interval bounds"
            )
        n = int(row["independent_seeds"])
        n_pairs = int(row["stratum_cells"])
        if n < 2 or n_pairs != len(STRATA) * n:
            raise ValueError("unexpected seed or stratum-cell denominator")
        prior = seed_counts.setdefault(row["testbed_id"], n)
        if n != prior:
            raise ValueError("seed denominator varies within a testbed")


def _sign_count(rows):
    positive = sum(float(row["difference_mean_pp"]) > 0.0 for row in rows)
    negative = sum(float(row["difference_mean_pp"]) < 0.0 for row in rows)
    return {
        "positive": positive,
        "negative": negative,
        "zero": len(rows) - positive - negative,
    }


def _cell_summary(row):
    return {
        "testbed_id": row["testbed_id"],
        "testbed": row["testbed"],
        "tau": float(row["tau"]),
        "difference_mean_pp": float(row["difference_mean_pp"]),
        "pointwise_95_low_pp": float(row["pointwise_95_low_pp"]),
        "pointwise_95_high_pp": float(row["pointwise_95_high_pp"]),
    }


def _profile_summary(rows):
    """Derive sign and interval summaries used by both figure and sidecars."""
    _validate_profile_result(rows)
    by_outcome = {}
    for outcome in OUTCOMES:
        chosen = [row for row in rows if row["outcome_id"] == outcome["id"]]
        by_outcome[outcome["id"]] = {
            "cells": len(chosen),
            "sign_counts": _sign_count(chosen),
            "pointwise_intervals_excluding_zero": sum(
                bool(row["interval_excludes_zero"]) for row in chosen
            ),
            "pointwise_intervals_including_zero_cells": [
                _cell_summary(row)
                for row in chosen
                if not bool(row["interval_excludes_zero"])
            ],
        }

    pairs = {
        (row["testbed_id"], float(row["tau"])): row
        for row in rows
        if row["outcome_id"] == "terminal_recommendation"
    }
    assignment_pairs = {
        (row["testbed_id"], float(row["tau"])): row
        for row in rows
        if row["outcome_id"] == "assignment_percentage"
    }
    if set(pairs) != set(assignment_pairs):
        raise ValueError("terminal and assignment profiles do not share the same cells")
    return {
        "outcomes": by_outcome,
        "opposite_sign_point_estimate_cells": sum(
            float(pairs[key]["difference_mean_pp"])
            * float(assignment_pairs[key]["difference_mean_pp"])
            < 0.0
            for key in pairs
        ),
        "both_pointwise_intervals_excluding_zero": sum(
            bool(pairs[key]["interval_excludes_zero"])
            and bool(assignment_pairs[key]["interval_excludes_zero"])
            for key in pairs
        ),
    }


def _legend_labels(profile_rows):
    """Return outcome names only; sign/interval summaries belong in prose."""
    _validate_profile_result(profile_rows)
    return {outcome["id"]: outcome["legend"] for outcome in OUTCOMES}


def _minority_sign_annotations(profile_rows):
    """Return unique nonzero minority-sign estimates, if the data contain any."""
    annotations = []
    for outcome in OUTCOMES:
        rows = [row for row in profile_rows if row["outcome_id"] == outcome["id"]]
        signs = _sign_count(rows)
        if signs["positive"] == 1 and signs["negative"] > 1:
            direction = "positive"
        elif signs["negative"] == 1 and signs["positive"] > 1:
            direction = "negative"
        else:
            continue
        row = next(
            item
            for item in rows
            if (float(item["difference_mean_pp"]) > 0.0) == (direction == "positive")
        )
        annotations.append(
            {
                **_cell_summary(row),
                "outcome_id": outcome["id"],
                "direction": direction,
                "text": (
                    f"Only {direction} {outcome['annotation']} estimate\n"
                    f"{float(row['difference_mean_pp']):+.2f} "
                    f"[{float(row['pointwise_95_low_pp']):.2f}, "
                    f"{float(row['pointwise_95_high_pp']):.2f}]"
                ),
            }
        )
    return annotations


def _plot(profile_rows, pdf_path, png_path, eps_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _validate_profile_result(profile_rows)
    legend_labels = _legend_labels(profile_rows)
    plot_ids = {item["id"] for item in PLOT_TESTBEDS}
    plot_rows = [row for row in profile_rows if row["testbed_id"] in plot_ids]
    interval_limits = [
        float(row[field])
        for row in plot_rows
        for field in ("pointwise_95_low_pp", "pointwise_95_high_pp")
    ]
    lower = min(0.0, min(interval_limits))
    upper = max(0.0, max(interval_limits))
    span = max(upper - lower, 1.0)
    y_min = lower - 0.10 * span
    y_max = upper + 0.10 * span

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.1, 3.15),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = np.asarray(axes).reshape(-1)
    fig.subplots_adjust(
        # Reserve a distinct column for the shared y label so it cannot collide
        # with the first panel's tick labels after full-width journal scaling.
        left=0.145,
        right=0.985,
        bottom=0.19,
        top=0.69,
        wspace=0.08,
    )
    styles = {
        "terminal_recommendation": {
            "color": "#0072B2",
            "marker": "o",
            "linestyle": "-",
        },
        "assignment_percentage": {
            "color": "#D55E00",
            "marker": "s",
            "linestyle": "--",
        },
    }
    legend_handles = []
    for axis, testbed in zip(axes, PLOT_TESTBEDS):
        panel_rows = [row for row in profile_rows if row["testbed_id"] == testbed["id"]]
        for outcome in OUTCOMES:
            rows = [row for row in panel_rows if row["outcome_id"] == outcome["id"]]
            rows.sort(key=lambda row: float(row["tau"]))
            x = np.asarray([row["tau"] for row in rows], dtype=float)
            y = np.asarray([row["difference_mean_pp"] for row in rows], dtype=float)
            low = np.asarray([row["pointwise_95_low_pp"] for row in rows], dtype=float)
            high = np.asarray(
                [row["pointwise_95_high_pp"] for row in rows], dtype=float
            )
            style = styles[outcome["id"]]
            handle = axis.errorbar(
                x,
                y,
                yerr=np.vstack((y - low, high - y)),
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=5.0,
                markerfacecolor="white",
                markeredgewidth=1.3,
                linewidth=1.6,
                elinewidth=1.0,
                capsize=2.8,
                zorder=3,
            )
            if axis is axes[0]:
                legend_handles.append(handle)
        axis.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(testbed["label"], loc="left", fontweight="semibold")
        axis.set_xticks(GATE_LEVELS)
        axis.set_xlim(0.475, 0.925)
        axis.set_ylim(y_min, y_max)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.legend(
        legend_handles,
        [legend_labels[item["id"]] for item in OUTCOMES],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=1,
        frameon=False,
        fontsize=8.5,
    )
    fig.supxlabel(r"Toxicity-rule stringency $\tau$ (larger is stricter)", y=0.02)
    fig.supylabel(
        "cKG − cEI–tMSE (percentage points)",
        x=0.022,
    )

    pdf_path = Path(pdf_path)
    png_path = Path(png_path)
    eps_path = Path(eps_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    eps_path.parent.mkdir(parents=True, exist_ok=True)
    title = "Supporting-surface terminal-recommendation and assignment contrast profile"
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Subject": (
                "Within-matched-replicate-set cKG minus the study-defined "
                "cEI–tMSE schedule "
                "terminal-recommendation and "
                "assignment contrasts"
            ),
            "Keywords": (
                "dose optimization, terminal recommendation, assignments, Monte Carlo"
            ),
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
                "Generated from a validated formal record projection. "
                "Intervals are pointwise and unadjusted."
            ),
        },
    )
    plt.close(fig)


def _metadata(profile_rows, selected, bundle_provenance, artifact_path, artifact_type):
    """Write a stable sidecar with authenticated inputs and derived summaries."""
    artifact_path = Path(artifact_path)
    base = sdb.artifact_metadata(selected, artifact_type=artifact_type)
    base["source_path"] = None
    base["source_sha256_role"] = (
        "SHA-256 of the selected records in canonical compact-JSON form and "
        "testbed order; authenticated compressed and logical input commitments follow"
    )
    input_files = {}
    record_provenance = bundle_provenance["record_files"]
    records_by_name = {item["record_file"]: item for item in TESTBEDS}
    for filename, testbed in records_by_name.items():
        detail = record_provenance[filename]
        input_files[testbed["id"]] = {
            "source_path": filename,
            "compressed_sha256": detail["compressed_sha256"],
            "uncompressed_sha256": detail["uncompressed_sha256"],
            "trial_rows": int(detail["rows"]),
        }
    base["input_files"] = input_files
    base["projection"] = {
        "projection_fingerprint": bundle_provenance["projection_fingerprint"],
        "projection_metadata_path": bundle_provenance["projection_metadata_path"],
        "projection_metadata_sha256": bundle_provenance["projection_metadata_sha256"],
        "formal_manifest_sha256": bundle_provenance["formal_manifest_sha256"],
    }
    base["schema_version"] = 3
    budgets = sorted({int(row["budget"]) for row in selected})
    base["estimand"] = {
        "outcomes": {
            "terminal_recommendation": (
                "all-trial true-boundary-exceeding final-recommendation rate"
            ),
            "assignment_percentage": (
                "percentage of above-boundary simulated assignments"
            ),
        },
        "contrast": CONTRAST_LABEL,
        "reader_facing_policy_alias": {"cEI-tMSE": "cEI–tMSE"},
        "full_panel_reference_schedule": {
            "scope": "all displayed testbeds use full-panel access",
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
        "unit": "percentage points",
        "aggregation": (
            "The two stratum-specific policy differences are averaged within matched "
            "Monte Carlo replicate set before the standard error is computed across sets."
        ),
        "no_recommendation": (
            "A trial with no final recommendation contributes zero to the all-trial "
            "threshold-exceedance indicator."
        ),
        "assignment_denominator": (
            "Each above-boundary assignment count is divided by that trial's stored "
            f"assignment budget; observed budget levels are {budgets}."
        ),
        "interval": (
            "point estimate plus or minus 1.96 Monte Carlo standard errors; "
            "pointwise and unadjusted across all profile contrasts; the figure "
            "displays the supporting-surface subset"
        ),
        "scope": (
            "the profile data contain four evaluated two-dimensional continuous-outcome "
            "simulations; the figure displays only the three supporting surfaces because "
            "OSA is reported separately with simultaneous bands; no universal policy "
            "ordering or simultaneous cross-testbed analysis is claimed"
        ),
    }
    base["independent_seed_counts"] = {
        testbed["id"]: next(
            int(row["independent_seeds"])
            for row in profile_rows
            if row["testbed_id"] == testbed["id"]
        )
        for testbed in TESTBEDS
    }
    # Reader-facing notation uses M for independent Monte Carlo replicate sets.
    # Keep the legacy field above for artifact-schema compatibility.
    base["monte_carlo_replicate_sets_by_testbed"] = dict(
        base["independent_seed_counts"]
    )
    summary = _profile_summary(profile_rows)
    terminal = summary["outcomes"]["terminal_recommendation"]
    assignment = summary["outcomes"]["assignment_percentage"]
    base["profile_cells"] = len(TESTBEDS) * len(GATE_LEVELS)
    base["profile_contrasts"] = len(profile_rows)
    base["figure_testbeds"] = [item["id"] for item in PLOT_TESTBEDS]
    base["figure_cells"] = len(PLOT_TESTBEDS) * len(GATE_LEVELS)
    base["figure_contrasts"] = len(PLOT_TESTBEDS) * len(GATE_LEVELS) * len(OUTCOMES)
    base["result_summary"] = summary
    base["positive_terminal_point_estimates"] = terminal["sign_counts"]["positive"]
    base["negative_assignment_point_estimates"] = assignment["sign_counts"]["negative"]
    base["terminal_pointwise_intervals_excluding_zero"] = terminal[
        "pointwise_intervals_excluding_zero"
    ]
    base["assignment_pointwise_intervals_excluding_zero"] = assignment[
        "pointwise_intervals_excluding_zero"
    ]
    base["opposite_direction_point_estimate_cells"] = summary[
        "opposite_sign_point_estimate_cells"
    ]
    base["both_pointwise_intervals_excluding_zero"] = summary[
        "both_pointwise_intervals_excluding_zero"
    ]
    base["terminal_pointwise_interval_exceptions"] = terminal[
        "pointwise_intervals_including_zero_cells"
    ]
    base["assignment_pointwise_interval_exceptions"] = assignment[
        "pointwise_intervals_including_zero_cells"
    ]
    base["pointwise_interval_exceptions"] = base[
        "terminal_pointwise_interval_exceptions"
    ]
    positive_assignment_rows = [
        row
        for row in profile_rows
        if row["outcome_id"] == "assignment_percentage"
        and float(row["difference_mean_pp"]) > 0.0
    ]
    base["unique_positive_assignment_estimate"] = (
        _cell_summary(positive_assignment_rows[0])
        if len(positive_assignment_rows) == 1
        else None
    )
    base["minority_sign_annotations"] = _minority_sign_annotations(profile_rows)
    base["artifact_sha256"] = sdb.file_sha256(artifact_path)
    metadata_path = artifact_path.with_suffix(artifact_path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata_path


def analyze_bundle(bundle: ProjectionBundle, output_dir):
    """Analyze one already authenticated projection bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_testbed = {
        testbed["id"]: bundle.records[testbed["record_file"]] for testbed in TESTBEDS
    }
    profile_rows, selected = _profile_rows(records_by_testbed)
    csv_path = output_dir / "cross_testbed_terminal_contrasts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)

    pdf_path = output_dir / "cross_testbed_terminal_contrasts.pdf"
    png_path = output_dir / "cross_testbed_terminal_contrasts.png"
    eps_path = output_dir / "cross_testbed_terminal_contrasts.eps"
    _plot(profile_rows, pdf_path, png_path, eps_path)
    for artifact, kind in (
        (csv_path, "cross_testbed_terminal_assignment_contrast_data"),
        (pdf_path, "cross_testbed_terminal_assignment_contrast_figure_pdf"),
        (png_path, "cross_testbed_terminal_assignment_contrast_figure_png"),
        (eps_path, "cross_testbed_terminal_assignment_contrast_figure_eps"),
    ):
        _metadata(profile_rows, selected, bundle.provenance, artifact, kind)

    print(f"wrote {csv_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {eps_path}")
    return profile_rows


def analyze(projection_dir, output_dir):
    """Authenticate and analyze a completed formal projection."""
    return analyze_bundle(load_projection_bundle(projection_dir), output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection-dir",
        required=True,
        help=(
            "authenticated formal_projection-<fingerprint>/ directory containing "
            "formal_projection.metadata.json and records/"
        ),
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    analyze(args.projection_dir, args.out_dir)


if __name__ == "__main__":
    main()
