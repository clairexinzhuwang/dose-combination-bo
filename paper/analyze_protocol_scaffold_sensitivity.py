#!/usr/bin/env python3
"""Generate every dose-availability table, figure, card, and report."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import warnings

import numpy as np

import dose_combination_bo as sdb

try:
    from paper.analysis_record_bundle import load_projection_record
except ModuleNotFoundError:  # Direct execution as ``python paper/<script>.py``.
    from analysis_record_bundle import load_projection_record


POLICIES = ("cEI", "cKG", "cEI-tMSE")
SCAFFOLDS = ("lhs_fixed", "start_low_expansion")
SCAFFOLD_LABELS = {
    "lhs_fixed": "All combinations available",
    "start_low_expansion": "Gradual expansion",
}
GAMMAS = (0.7, 0.9)
STRATA = (0, 1)
PAIR_POLICIES = ("cKG", "cEI-tMSE")
TEX_POLICY_NAMES = {
    "cEI": "cEI",
    "cKG": "cKG",
    "cEI-tMSE": "cEI--tMSE",
}
PLOT_POLICY_NAMES = {
    "cEI": "cEI",
    "cKG": "cKG",
    "cEI-tMSE": "cEI–tMSE",
}
FROZEN = {
    "mode": "latent",
    "noise": "fixed",
    "kap": 1.0,
    "budget": 40,
    "warmup": 4,
    "r_k": 2,
    "grid_n": 5,
    "empty_gate": "pf",
    "region_step": 0.25,
    "empty_gate_stop_after": 3,
    "exclude_repeats_during_expansion": True,
}
PROJECTION_RECORD = "protocol_scaffold_sensitivity.json.zst"


def _rows(table):
    return table.to_dict("records") if hasattr(table, "to_dict") else list(table)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_artifact_sidecar(
    path,
    records,
    *,
    source_path,
    artifact_type,
    projection_provenance=None,
):
    """Write portable metadata, binding projected analyses to their source bundle."""
    metadata = sdb.artifact_metadata(
        records,
        source_path=source_path,
        artifact_type=artifact_type,
    )
    metadata["analyzer_sha256"] = sdb.file_sha256(Path(__file__))
    if projection_provenance is not None:
        commitment = projection_provenance["selected_record_commitment"]
        metadata["source_path"] = projection_provenance["selected_record"]
        metadata["source_sha256"] = commitment["uncompressed_sha256"]
        metadata["source_compressed_sha256"] = commitment["compressed_sha256"]
        metadata["formal_projection"] = {
            "projection_fingerprint": projection_provenance[
                "projection_fingerprint"
            ],
            "projection_metadata_sha256": projection_provenance[
                "projection_metadata_sha256"
            ],
            "formal_manifest_sha256": projection_provenance[
                "formal_manifest_sha256"
            ],
        }
    metadata["reader_facing_policy_alias"] = {"cEI-tMSE": "cEI–tMSE"}
    metadata["comparator_schedule_scope"] = {
        "full_panel_only": (
            "With N_max=40, four initialization assignments, and cohorts of two, "
            "the all-combinations-available design has 18 scheduled adaptive-cohort positions: "
            "nine cEI and nine tMSE."
        ),
        "gradual_access": (
            "With two initialization assignments, a nonstopped maximum-length "
            "gradual trial has 19 scheduled positions: ten cEI and nine tMSE; "
            "early stopping can truncate that sequence. It is not described as 1:1."
        ),
        "empty_gate_override": (
            "An empty eligible gate can invoke the common greatest-feasibility "
            "fallback instead of the scheduled acquisition."
        ),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _validate(records, require_full):
    expected_fields = {
        "policy", "seed", "sim", "stratum", "gamma", "protocol_scaffold",
        "recommendation_made", "stop_reason", "n_enrolled",
        "initialization_patients_above", "post_initialization_patients_above",
        "n_unique_doses", "n_empty_gate_events",
        "rec_unsafe", "rec_true_eff", "dose_units", "initialization_size",
        "region_q_at_stop", "allocation_history", "rec_d1", "rec_d2",
        "rec_true_tox", "rpsel", "recs",
    }
    keys = set()
    for index, row in enumerate(records):
        missing = expected_fields - set(row)
        if missing:
            raise ValueError(f"record {index} is missing {sorted(missing)}")
        if row["sim"] != "osa" or row["policy"] not in POLICIES:
            raise ValueError("records do not match the frozen OSA/policy matrix")
        if row["protocol_scaffold"] not in SCAFFOLDS:
            raise ValueError("records contain an unknown access protocol")
        if float(row["gamma"]) not in GAMMAS or int(row["stratum"]) not in STRATA:
            raise ValueError("records contain an unplanned gamma or stratum")
        if type(row["recommendation_made"]) is not bool:
            raise ValueError(f"record {index} recommendation_made must be boolean")
        try:
            n_enrolled = int(row["n_enrolled"])
            allocation_count = len(row["allocation_history"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"record {index} has invalid n_enrolled or allocation_history"
            ) from exc
        if n_enrolled != row["n_enrolled"] or allocation_count != n_enrolled:
            raise ValueError(
                f"record {index} allocation_history length {allocation_count} "
                f"does not equal n_enrolled={row['n_enrolled']!r}"
            )
        terminal_fields = (
            "dose_units", "rpsel", "rec_unsafe", "rec_d1", "rec_d2",
            "rec_true_eff", "rec_true_tox",
        )
        if row["recommendation_made"]:
            if row["stop_reason"] is not None:
                raise ValueError(
                    f"record {index} has a recommendation and non-null stop_reason"
                )
        else:
            if row["stop_reason"] != "NO_FEASIBLE_DOSE":
                raise ValueError(
                    f"record {index} without a recommendation must report "
                    "NO_FEASIBLE_DOSE"
                )
            populated = [field for field in terminal_fields if row[field] is not None]
            if populated:
                raise ValueError(
                    f"record {index} stopped trial has terminal fields {populated}"
                )
            if row["recs"] != []:
                raise ValueError(f"record {index} stopped trial must have empty recs")
        for field, expected_value in FROZEN.items():
            actual = row.get(field, 1.0 if field == "kap" else None)
            if actual != expected_value:
                raise ValueError(
                    f"record {index} has {field}={actual!r}; expected {expected_value!r}"
                )
        expected_initialization = 4 if row["protocol_scaffold"] == "lhs_fixed" else 2
        if int(row.get("initialization_size", -1)) != expected_initialization:
            raise ValueError(
                f"record {index} has initialization_size={row.get('initialization_size')!r}; "
                f"expected {expected_initialization}"
            )
        key = (
            row["protocol_scaffold"], float(row["gamma"]), int(row["stratum"]),
            row["policy"], int(row["seed"]),
        )
        if key in keys:
            raise ValueError(f"duplicate design cell: {key}")
        keys.add(key)
    seeds = sorted({int(row["seed"]) for row in records})
    expected = {
        (scaffold, gamma, stratum, policy, seed)
        for scaffold in SCAFFOLDS for gamma in GAMMAS for stratum in STRATA
        for policy in POLICIES for seed in seeds
    }
    if keys != expected:
        raise ValueError(f"incomplete factorial matrix: {len(expected - keys)} cells are missing")
    if require_full and (seeds != list(range(200)) or len(records) != 4800):
        raise ValueError("manuscript artifacts require exactly seeds 0..199 and 4,800 records")
    return seeds


def _enriched(records):
    output = []
    for row in records:
        item = dict(row)
        item["no_feasible_dose"] = int(row["stop_reason"] == "NO_FEASIBLE_DOSE")
        item["all_trial_unsafe"] = int(
            bool(row["recommendation_made"]) and bool(row["rec_unsafe"])
        )
        output.append(item)
    return output


def _eligible_counts(records, *, conditional, scaffolds):
    """Count eligible seed--stratum pairs and seeds for a frozen paired estimand."""
    scaffolds = tuple(scaffolds)
    required = {(scaffold, policy) for scaffold in scaffolds for policy in PAIR_POLICIES}
    blocks = {}
    for row in records:
        scaffold = row["protocol_scaffold"]
        policy = row["policy"]
        if scaffold not in scaffolds or policy not in PAIR_POLICIES:
            continue
        key = (int(row["seed"]), row["sim"], int(row["stratum"]), float(row["gamma"]))
        cell = (scaffold, policy)
        slot = blocks.setdefault(key, {})
        if cell in slot:
            raise ValueError(f"duplicate eligibility cell {cell}: {key}")
        slot[cell] = row
    incomplete = [key for key, values in blocks.items() if set(values) != required]
    if incomplete:
        raise ValueError(f"incomplete eligibility block; example: {incomplete[0]}")
    eligible = [
        key for key, values in blocks.items()
        if not conditional or all(bool(row["recommendation_made"]) for row in values.values())
    ]
    return len(eligible), len({key[0] for key in eligible})


def _contrast_rows(records):
    enriched = _enriched(records)
    output = []
    definitions = (
        ("rec_true_eff", True, "Recommended efficacy | recommend", "primary"),
        ("all_trial_unsafe", False, "True-boundary-exceeding final recommendation, all trials", "primary"),
        ("dose_units", True, "Dose-location error | recommend", "secondary"),
        ("no_feasible_dose", False, "Feasibility stop", "secondary"),
        ("post_initialization_patients_above", False, "Post-initialization above-boundary simulated assignments", "secondary"),
        ("n_enrolled", False, "Sample size", "secondary"),
    )
    primary = [row for row in enriched
               if row["protocol_scaffold"] == "start_low_expansion"
               and float(row["gamma"]) == 0.7]
    for metric, conditional, label, tier in definitions:
        result = _rows(sdb.paired_contrast(
            primary,
            "cKG",
            "cEI-tMSE",
            metric=metric,
            by=("sim", "gamma"),
            recommendation_conditional=conditional,
        ))[0]
        pair_n, seed_n = _eligible_counts(
            primary, conditional=conditional, scaffolds=("start_low_expansion",)
        )
        if int(result["n_pairs"]) != pair_n or int(result["n"]) != seed_n:
            raise ValueError("paired-contrast counts disagree with eligibility audit")
        result.update(
            label=label,
            tier=tier,
            contrast="cKG − cEI–tMSE",
            protocol_scaffold="start_low_expansion",
            eligible_pair_n=pair_n,
            eligible_seed_n=seed_n,
        )
        result["ci95_low"] = result["difference_mean"] - 1.96 * result["difference_se"]
        result["ci95_high"] = result["difference_mean"] + 1.96 * result["difference_se"]
        output.append(result)
    return output


def _interaction_rows(records):
    enriched = _enriched(records)
    output = []
    for metric, conditional, label in (
        ("rec_true_eff", True, "Recommended efficacy | recommend"),
        ("all_trial_unsafe", False, "True-boundary-exceeding final recommendation, all trials"),
        ("dose_units", True, "Dose-location error | recommend"),
    ):
        rows = _rows(sdb.scaffold_interaction(
            enriched,
            "cKG",
            "cEI-tMSE",
            metric=metric,
            by=("sim", "gamma"),
            recommendation_conditional=conditional,
        ))
        for row in rows:
            selected = [
                record for record in enriched
                if record["sim"] == row["sim"]
                and float(record["gamma"]) == float(row["gamma"])
            ]
            pair_n, seed_n = _eligible_counts(
                selected, conditional=conditional, scaffolds=SCAFFOLDS
            )
            if int(row["n_pairs"]) != pair_n or int(row["n"]) != seed_n:
                raise ValueError("interaction counts disagree with eligibility audit")
            row["label"] = label
            row["contrast"] = "cKG − cEI–tMSE"
            row["eligible_pair_n"] = pair_n
            row["eligible_seed_n"] = seed_n
            row["ci95_low"] = row["interaction_mean"] - 1.96 * row["interaction_se"]
            row["ci95_high"] = row["interaction_mean"] + 1.96 * row["interaction_se"]
            output.append(row)
    return output


def _paired_grid(records):
    enriched = _enriched(records)
    output = []
    for scaffold in SCAFFOLDS:
        for gamma in GAMMAS:
            selected = [row for row in enriched
                        if row["protocol_scaffold"] == scaffold
                        and float(row["gamma"]) == gamma]
            for metric, conditional, label in (
                ("rec_true_eff", True, "Recommended efficacy | recommend"),
                ("all_trial_unsafe", False, "True-boundary-exceeding final recommendation, all trials"),
                ("dose_units", True, "Dose-location error | recommend"),
            ):
                row = _rows(sdb.paired_contrast(
                    selected,
                    "cKG",
                    "cEI-tMSE",
                    metric=metric,
                    by=("sim", "gamma"),
                    recommendation_conditional=conditional,
                ))[0]
                pair_n, seed_n = _eligible_counts(
                    selected, conditional=conditional, scaffolds=(scaffold,)
                )
                if int(row["n_pairs"]) != pair_n or int(row["n"]) != seed_n:
                    raise ValueError("paired-grid counts disagree with eligibility audit")
                row.update(
                    protocol_scaffold=scaffold,
                    label=label,
                    contrast="cKG − cEI–tMSE",
                    eligible_pair_n=pair_n,
                    eligible_seed_n=seed_n,
                )
                output.append(row)
    return output


def _pm(value, se, digits=2):
    if not np.isfinite([value, se]).all():
        return "--"
    return f"{value:.{digits}f} $\\pm$ {se:.{digits}f}"


def _tex_policy(policy):
    try:
        return TEX_POLICY_NAMES[policy]
    except KeyError as exc:
        raise ValueError(f"unknown policy for LaTeX output: {policy!r}") from exc


def _latex_table(oc_rows, gamma, *, denominator_panels=False):
    chosen = [row for row in oc_rows if float(row["gamma"]) == float(gamma)]
    names = SCAFFOLD_LABELS
    ordered = []
    for scaffold in SCAFFOLDS:
        for policy in POLICIES:
            rows = [row for row in chosen
                    if row["protocol_scaffold"] == scaffold and row["policy"] == policy]
            if len(rows) != 1:
                raise ValueError("expected one replicate-level OC row across strata")
            ordered.append((scaffold, policy, rows[0]))

    if denominator_panels:
        lines = [
            r"\begin{tabular}{@{}llrrrrr@{}}",
            r"\multicolumn{7}{@{}l}{\textit{Panel A: all $2M=400$ stratum-specific simulated trials per policy--protocol cell}}\\[2pt]",
            r"\toprule",
            r"Protocol & Policy & Recommend (\%) & Stop (\%) & \shortstack{True-boundary-\\exceeding final rec. (\%)} & \shortstack{Post-init. above-\\boundary simulated\\assignments} & \shortstack{Mean\\$N_{\mathrm{obs}}$} \\",
            r"\midrule",
        ]
        for scaffold, policy, row in ordered:
            values = (
                _pm(row["recommendation_pct"], row["recommendation_pct_mcse"], 1),
                _pm(row["no_feasible_dose_pct"], row["no_feasible_dose_pct_mcse"], 1),
                _pm(row["rec_unsafe_pct"], row["rec_unsafe_pct_mcse"], 1),
                _pm(row["post_initialization_patients_above"],
                    row["post_initialization_patients_above_mcse"], 2),
                _pm(row["expected_n"], row["expected_n_mcse"], 1),
            )
            lines.append(
                f"{names[scaffold]} & {_tex_policy(policy)} & "
                + " & ".join(values) + r" \\"
            )
            if policy == POLICIES[-1] and scaffold != SCAFFOLDS[-1]:
                lines.append(r"\addlinespace")
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\par\vspace{5pt}",
            r"\begin{tabular}{@{}llrrr@{}}",
            r"\multicolumn{5}{@{}l}{\textit{Panel B: recommendation-conditional outcomes}}\\[2pt]",
            r"\toprule",
            r"Protocol & Policy & Recommendations & Rec. efficacy & Dose-location error \\",
            r"\midrule",
        ])
        for scaffold, policy, row in ordered:
            values = (
                str(int(row["rec_efficacy_n_rows"])),
                _pm(row["rec_efficacy"], row["rec_efficacy_mcse"], 3),
                _pm(row["dose_units"], row["dose_units_mcse"], 2),
            )
            lines.append(
                f"{names[scaffold]} & {_tex_policy(policy)} & "
                + " & ".join(values) + r" \\"
            )
            if policy == POLICIES[-1] and scaffold != SCAFFOLDS[-1]:
                lines.append(r"\addlinespace")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return "\n".join(lines) + "\n"

    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Protocol & Policy & Recommend (\%) & Stop (\%) & Rec. efficacy & True-boundary-exceeding final rec. (\%) & Dose-location error & Post-init. above-boundary simulated assignments & Mean $N_{\mathrm{obs}}$ \\",
        r"\midrule",
    ]
    for scaffold, policy, row in ordered:
        values = (
            _pm(row["recommendation_pct"], row["recommendation_pct_mcse"], 1),
            _pm(row["no_feasible_dose_pct"], row["no_feasible_dose_pct_mcse"], 1),
            _pm(row["rec_efficacy"], row["rec_efficacy_mcse"], 3),
            _pm(row["rec_unsafe_pct"], row["rec_unsafe_pct_mcse"], 1),
            _pm(row["dose_units"], row["dose_units_mcse"], 2),
            _pm(row["post_initialization_patients_above"],
                row["post_initialization_patients_above_mcse"], 2),
            _pm(row["expected_n"], row["expected_n_mcse"], 1),
        )
        lines.append(
            f"{names[scaffold]} & {_tex_policy(policy)} & "
            + " & ".join(values) + r" \\"
        )
        if policy == POLICIES[-1] and scaffold != SCAFFOLDS[-1]:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _supplement_program_table(oc_rows):
    lines = [
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Threshold & Protocol & Policy & Recommend (\%) & Stop (\%) & True-boundary-exceeding final rec. (\%) & Within one (\%) & Mean $N_{\mathrm{obs}}$ \\",
        r"\midrule",
    ]
    names = SCAFFOLD_LABELS
    for gamma in GAMMAS:
        for scaffold in SCAFFOLDS:
            for policy in POLICIES:
                matches = [row for row in oc_rows if float(row["gamma"]) == gamma
                           and row["protocol_scaffold"] == scaffold
                           and row["policy"] == policy]
                if len(matches) != 1:
                    raise ValueError("expected one pooled program OC row")
                row = matches[0]
                values = (
                    _pm(row["recommendation_pct"], row["recommendation_pct_mcse"], 1),
                    _pm(row["no_feasible_dose_pct"], row["no_feasible_dose_pct_mcse"], 1),
                    _pm(row["rec_unsafe_pct"], row["rec_unsafe_pct_mcse"], 1),
                    _pm(row["pcs_within1"], row["pcs_within1_mcse"], 1),
                    _pm(row["expected_n"], row["expected_n_mcse"], 1),
                )
                lines.append(
                    f"{gamma:.1f} & {names[scaffold]} & {_tex_policy(policy)} & "
                    + " & ".join(values) + r" \\"
                )
        if gamma != GAMMAS[-1]:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _supplement_operations_table(oc_rows):
    lines = [
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Threshold & Protocol & Policy & Init. above-boundary simulated assignments & Post-init. above-boundary simulated assignments & Total above-boundary simulated assignments & Unique doses & Empty events \\",
        r"\midrule",
    ]
    names = SCAFFOLD_LABELS
    for gamma in GAMMAS:
        for scaffold in SCAFFOLDS:
            for policy in POLICIES:
                matches = [row for row in oc_rows if float(row["gamma"]) == gamma
                           and row["protocol_scaffold"] == scaffold
                           and row["policy"] == policy]
                if len(matches) != 1:
                    raise ValueError("expected one pooled operations OC row")
                row = matches[0]
                values = tuple(
                    _pm(row[field], row[f"{field}_mcse"], 2)
                    for field in (
                        "initialization_patients_above",
                        "post_initialization_patients_above",
                        "patients_above",
                        "n_unique_doses",
                        "n_empty_gate_events",
                    )
                )
                lines.append(
                    f"{gamma:.1f} & {names[scaffold]} & {_tex_policy(policy)} & "
                    + " & ".join(values) + r" \\"
                )
        if gamma != GAMMAS[-1]:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _supplement_conditional_table(oc_rows):
    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Threshold & Protocol & Policy & Recommendations & Rec. efficacy & Rec. dose-location error & \shortstack{True-boundary-exceeding\\among recommendations (\%)} \\",
        r"\midrule",
    ]
    names = SCAFFOLD_LABELS
    for gamma in GAMMAS:
        for scaffold in SCAFFOLDS:
            for policy in POLICIES:
                matches = [row for row in oc_rows if float(row["gamma"]) == gamma
                           and row["protocol_scaffold"] == scaffold
                           and row["policy"] == policy]
                if len(matches) != 1:
                    raise ValueError("expected one pooled conditional OC row")
                row = matches[0]
                values = (
                    str(int(row["rec_efficacy_n_rows"])),
                    _pm(row["rec_efficacy"], row["rec_efficacy_mcse"], 3),
                    _pm(row["dose_units"], row["dose_units_mcse"], 2),
                    _pm(
                        row["unsafe_given_recommendation_pct"],
                        row["unsafe_given_recommendation_pct_mcse"],
                        1,
                    ),
                )
                lines.append(
                    f"{gamma:.1f} & {names[scaffold]} & {_tex_policy(policy)} & "
                    + " & ".join(values) + r" \\"
                )
        if gamma != GAMMAS[-1]:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _contrast_display(row, *, interaction=False):
    """Return publication-scale values for a paired contrast row.

    Binary trial indicators are displayed as percentage-point differences; all
    other outcomes retain their natural units.  The machine-readable CSV files
    continue to store the unscaled estimands.
    """
    label = row["label"]
    percentage_point = label in {
        "True-boundary-exceeding final recommendation, all trials",
        "Feasibility stop",
    }
    scale = 100.0 if percentage_point else 1.0
    prefix = "interaction" if interaction else "difference"
    mean = scale * float(row[f"{prefix}_mean"])
    se = scale * float(row[f"{prefix}_se"])
    low = scale * float(row["ci95_low"])
    high = scale * float(row["ci95_high"])
    latex_labels = {
        "Recommended efficacy | recommend": "Recommended efficacy among recommendations",
        "True-boundary-exceeding final recommendation, all trials": "True-boundary-exceeding final recommendation, all trials",
        "Dose-location error | recommend": "Dose-location error among recommendations",
        "Feasibility stop": "Feasibility stop",
        "Post-initialization above-boundary simulated assignments": "Post-init. above-boundary simulated assignments",
        "Sample size": "Sample size",
    }
    try:
        display_label = latex_labels[label]
    except KeyError as exc:
        raise ValueError(f"unknown contrast label for LaTeX output: {label!r}") from exc
    if percentage_point:
        display_label += " (pp)"
    return display_label, mean, se, low, high


def _supplement_contrast_table(contrast_rows):
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Outcome & Difference & MCSE & 95\% MC interval & Paired trials & Replicate sets \\",
        r"\midrule",
    ]
    for row in contrast_rows:
        label, mean, se, low, high = _contrast_display(row)
        lines.append(
            f"{label} & {mean:.3f} & {se:.3f} & "
            f"[{low:.3f}, {high:.3f}] & {int(row['eligible_pair_n'])} & "
            f"{int(row['eligible_seed_n'])} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _supplement_interaction_table(interaction_rows):
    lines = [
        r"\begin{tabular}{rlrrrrr}",
        r"\toprule",
        r"Threshold & Outcome & Interaction & MCSE & 95\% MC interval & Paired trials & Replicate sets \\",
        r"\midrule",
    ]
    for row in interaction_rows:
        label, mean, se, low, high = _contrast_display(row, interaction=True)
        lines.append(
            f"{float(row['gamma']):.1f} & {label} & {mean:.3f} & {se:.3f} & "
            f"[{low:.3f}, {high:.3f}] & {int(row['eligible_pair_n'])} & "
            f"{int(row['eligible_seed_n'])} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _plot(interaction_rows, output):
    """Plot gradual-minus-full changes in paired policy contrasts.

    This interaction plot is deliberately distinct from the primary OSA gate
    profile.  It shows how the cKG-minus-reference contrast changes with the
    surrounding protocol, using only authenticated four-cell interaction
    estimates and no additional simulation or model fit.
    """
    import matplotlib.pyplot as plt

    panels = (
        (
            "Recommended efficacy | recommend",
            "(a) Recommended efficacy",
            "Efficacy-contrast change",
            1.0,
        ),
        (
            "True-boundary-exceeding final recommendation, all trials",
            "(b) True-boundary-exceeding\nfinal recommendation",
            "Terminal-contrast change (pp)",
            100.0,
        ),
        (
            "Dose-location error | recommend",
            "(c) Dose-location error",
            "Error-contrast change\n(grid units)",
            1.0,
        ),
    )
    colors = {0.7: "#0072B2", 0.9: "#D55E00"}
    markers = {0.7: "o", 0.9: "s"}
    y_positions = {0.7: 1.0, 0.9: 0.0}

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8), constrained_layout=False)
    for axis, (label, title, xlabel, scale) in zip(axes, panels):
        for gamma in GAMMAS:
            rows = [
                row
                for row in interaction_rows
                if row["label"] == label and float(row["gamma"]) == gamma
            ]
            if len(rows) != 1:
                raise ValueError(
                    "expected one protocol interaction per outcome and gate"
                )
            mean = scale * float(rows[0]["interaction_mean"])
            error = scale * 1.96 * float(rows[0]["interaction_se"])
            face = colors[gamma] if gamma == 0.7 else "white"
            axis.errorbar(
                mean,
                y_positions[gamma],
                xerr=error,
                color=colors[gamma],
                marker=markers[gamma],
                markerfacecolor=face,
                markeredgecolor=colors[gamma],
                markeredgewidth=1.2,
                linestyle="none",
                capsize=3.0,
                markersize=5.5,
                elinewidth=1.2,
            )
        axis.axvline(0.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_yticks(
            [y_positions[gamma] for gamma in GAMMAS],
            [rf"$\tau={gamma:.1f}$" for gamma in GAMMAS],
        )
        axis.set_ylim(-0.6, 1.6)
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_xlabel(xlabel)
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Gradual expansion minus all combinations available: change in the cKG − cEI–tMSE contrast",
        fontsize=10.5,
        fontweight="semibold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.15)

    png_path = Path(output)
    pdf_path = png_path.with_suffix(".pdf")
    eps_path = png_path.with_suffix(".eps")
    title = "Protocol interaction in paired cKG-minus-reference contrasts"
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Subject": (
                "Between-design changes in paired cKG minus author-specified "
                "cEI–tMSE schedule contrasts across gate thresholds"
            ),
            "Keywords": "dose optimization, dose availability, Monte Carlo",
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
                "Generated from authenticated paired protocol-contrast records; "
                "intervals are pointwise and unadjusted."
            ),
        },
    )
    plt.close(fig)
    return png_path, pdf_path, eps_path


def _finding(paired_grid):
    """Summarize gradual-access contrasts without a sign-only headline."""
    def values_for(scaffold, gamma):
        def value(label):
            rows = [row for row in paired_grid
                    if row["protocol_scaffold"] == scaffold
                    and float(row["gamma"]) == float(gamma)
                    and row["label"] == label]
            if len(rows) != 1 or not np.isfinite(rows[0]["difference_mean"]):
                raise ValueError("paired finding is undefined")
            row = rows[0]
            mean = float(row["difference_mean"])
            se = float(row.get("difference_se", math.nan))
            return {
                "mean": mean,
                "low": mean - 1.96 * se if np.isfinite(se) else math.nan,
                "high": mean + 1.96 * se if np.isfinite(se) else math.nan,
            }
        return {
            "eff": value("Recommended efficacy | recommend"),
            "unsafe": value("True-boundary-exceeding final recommendation, all trials"),
            "du": value("Dose-location error | recommend"),
        }
    deltas = {scaffold: values_for(scaffold, 0.7) for scaffold in SCAFFOLDS}
    gradual = deltas["start_low_expansion"]

    def resolved_positive(item):
        return np.isfinite(item["low"]) and item["low"] > 0

    def resolved_negative(item):
        return np.isfinite(item["high"]) and item["high"] < 0

    def unresolved(item):
        return (
            np.isfinite(item["low"])
            and item["low"] <= 0 <= item["high"]
        )

    def classify(values):
        if unresolved(values["eff"]):
            efficacy = "the recommended-efficacy interval included zero"
        elif resolved_positive(values["eff"]):
            efficacy = (
                "cKG had higher recommended efficacy and the interval excluded zero"
            )
        elif resolved_negative(values["eff"]):
            efficacy = (
                "cKG had lower recommended efficacy and the interval excluded zero"
            )
        else:
            efficacy = "the recommended-efficacy contrast could not be classified"

        if resolved_positive(values["unsafe"]):
            safety = (
                "cKG had more true-boundary-exceeding final recommendations and the interval "
                "excluded zero"
            )
        elif resolved_negative(values["unsafe"]):
            safety = (
                "cKG had fewer true-boundary-exceeding final recommendations and the interval "
                "excluded zero"
            )
        else:
            safety = "the true-boundary-exceeding final-recommendation interval included zero"

        if resolved_positive(values["du"]):
            localization = (
                "cKG had greater dose-location error and the interval excluded zero"
            )
        elif resolved_negative(values["du"]):
            localization = (
                "cKG had smaller dose-location error and the interval excluded zero"
            )
        else:
            localization = "the dose-location-error interval included zero"
        return efficacy, safety, localization

    efficacy, safety, localization = classify(gradual)
    category = f"{efficacy}; {safety}; {localization}"
    has_09 = any(
        row["protocol_scaffold"] == "start_low_expansion"
        and float(row["gamma"]) == 0.9
        for row in paired_grid
    )
    if has_09:
        gradual_09 = values_for("start_low_expansion", 0.9)
        efficacy_09, safety_09, localization_09 = classify(gradual_09)
        sentence = (
            "Under gradual expansion in the OSA simulation, at threshold 0.7, "
            + efficacy
            + ", "
            + safety
            + ", and "
            + localization
            + "; at threshold 0.9, "
            + efficacy_09
            + ", "
            + safety_09
            + ", and "
            + localization_09
            + "."
        )
    else:
        sentence = (
            "Under gradual expansion in the OSA simulation at threshold 0.7, " + efficacy + ", "
            + safety + ", and " + localization + "."
        )
    return category, sentence, deltas


def _markdown(contrast_rows, interaction_rows, paired_grid, seeds):
    _category, sentence, deltas = _finding(paired_grid)
    md = lambda value: str(value).replace("|", r"\|")
    lines = [
        "# Dose availability and stopping sensitivity",
        "",
        "This page is generated from the complete OSA sensitivity records. It compares a design "
        "with all combinations available after initialization with a Willard-inspired "
        "gradual-expansion design and a common feasibility-stopping rule; it is not a "
        "reconstruction of the source design algorithm.",
        "According to the local development record, the analysis specification was finalized "
        "before the complete performance results were inspected; no externally timestamped "
        "pre-run commit is available.",
        "",
        "## Analysis specification",
        "",
        f"- {len(seeds)} independent simulation replicates; each includes two strata, "
        "and the three policies are compared within replicate using common random numbers.",
        "- Gate thresholds tau = 0.7 (representative intermediate value) and 0.9 "
        "(stricter sensitivity); neither is a clinical toxicity cutoff.",
        "- Maximum sample size 40; cohort size 2; 5x5 candidate dose-combination grid.",
        "- With all combinations available after initialization, four initial assignments leave 18 scheduled "
        "adaptive-cohort positions: 9 cEI and 9 tMSE. With two initialization "
        "assignments, a nonstopped maximum-length gradual trial has 19 scheduled "
        "positions: 10 cEI and 9 tMSE; stopping can truncate that sequence. The "
        "gradual design is not labelled 1:1.",
        "- An empty eligible gate can invoke the common greatest-feasibility fallback "
        "instead of the scheduled acquisition.",
        "- Gradual-expansion region: d1 + d2 <= 0.25 q; no repeat while an unvisited "
        "candidate remains before full expansion.",
        "- The first two consecutive empty gates use the most-feasible eligible dose; "
        "the third stops before allocation and is recorded as a feasibility stop.",
        "",
        "## Result",
        "",
        f"**{sentence}**",
        "",
        "At tau = 0.7, the cKG minus author-specified cEI–tMSE schedule "
        "within-replicate differences (averaging eligible paired OSA strata within "
        "each simulation replicate) were:",
        "",
        "| Dose-availability design | Efficacy | True-boundary-exceeding final recommendation (percentage points) | Dose-location error (dose units) |",
        "|---|---:|---:|---:|",
    ]
    protocol_names = SCAFFOLD_LABELS
    for scaffold in SCAFFOLDS:
        value = deltas[scaffold]
        lines.append(
            f"| {protocol_names[scaffold]} | {value['eff']['mean']:.3f} "
            f"| {100.0 * value['unsafe']['mean']:.1f} | {value['du']['mean']:.2f} |"
        )
    lines.extend([
        "",
        "## Principal and key secondary paired contrasts",
        "",
        "The contrast is cKG minus cEI–tMSE under gradual expansion, tau = 0.7. "
        "Monte Carlo intervals are mean +/- 1.96 Monte Carlo SE computed across "
        "independent simulation replicates; "
        "no p-value was used "
        "to retain or discard an analysis.",
        "",
        "| Outcome | Mean | MCSE | 95% MC interval | Paired trials | Simulation replicates |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    public_labels = {
        "Recommended efficacy | recommend": "Recommended efficacy among recommendations",
        "True-boundary-exceeding final recommendation, all trials": "True-boundary-exceeding final recommendation, all trials",
        "Dose-location error | recommend": "Dose-location error (dose units) among recommendations",
        "Feasibility stop": "Feasibility stop (incorrect under OSA truth)",
        "Post-initialization above-boundary simulated assignments": "Post-initialization above-boundary simulated assignments",
        "Sample size": "Sample size",
    }
    for row in contrast_rows:
        lines.append(
            f"| {md(public_labels[row['label']])} | {row['difference_mean']:.4f} | {row['difference_se']:.4f} "
            f"| [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] "
            f"| {row['eligible_pair_n']} | {row['eligible_seed_n']} |"
        )
    lines.extend([
        "",
        "## Protocol interactions",
        "",
        "Each interaction is the change in the cKG minus cEI–tMSE contrast when moving "
        "from the all-combinations-available design to gradual expansion.",
        "",
        "| tau | Outcome | Interaction | MCSE | 95% MC interval | Paired trials | Simulation replicates |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in interaction_rows:
        lines.append(
            f"| {row['gamma']:.1f} | {md(public_labels[row['label']])} | {row['interaction_mean']:.4f} "
            f"| {row['interaction_se']:.4f} | [{row['ci95_low']:.4f}, "
            f"{row['ci95_high']:.4f}] | {row['eligible_pair_n']} "
            f"| {row['eligible_seed_n']} |"
        )
    all_intervals_include_zero = all(
        float(row["ci95_low"]) <= 0 <= float(row["ci95_high"])
        for row in interaction_rows
    )
    if all_intervals_include_zero:
        lines.extend([
            "",
            "Every reported protocol-interaction interval includes zero. The analysis "
            "therefore does not establish that changing the dose-availability design caused the contrast "
            "changes described above.",
        ])
    lines.extend([
        "",
        "Recommendation-conditional outcomes exclude stopped trials and report their own "
        "denominators in the machine-readable OC table. Stopped trials are never assigned "
        "a fictitious efficacy or dose-location-error value. Because feasible doses exist on the "
        "OSA truth, every feasibility stop is an incorrect operational stop under the "
        "simulation truth rather than evidence of clinical infeasibility.",
        "",
    ])
    return "\n".join(lines)


def _analyze_records(
    records,
    output_dir,
    *,
    require_full=True,
    source_path=None,
    projection_provenance=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = _validate(records, require_full)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="no 'random' policy rows were supplied.*", category=UserWarning
        )
        oc_rows = _rows(sdb.oc_table(
            records,
            by=("policy", "protocol_scaffold", "gamma", "stratum"),
        ))
        pooled_oc_rows = _rows(sdb.oc_table(
            records,
            by=("policy", "protocol_scaffold", "gamma"),
        ))
    contrasts = _contrast_rows(records)
    interactions = _interaction_rows(records)
    paired_grid = _paired_grid(records)

    oc_csv = output_dir / "protocol_scaffold_oc.csv"
    pooled_oc_csv = output_dir / "protocol_scaffold_oc_pooled.csv"
    contrast_csv = output_dir / "protocol_scaffold_primary_secondary_contrasts.csv"
    interaction_csv = output_dir / "protocol_scaffold_interactions.csv"
    paired_grid_csv = output_dir / "protocol_scaffold_paired_grid.csv"
    _write_csv(oc_csv, oc_rows)
    _write_csv(pooled_oc_csv, pooled_oc_rows)
    _write_csv(contrast_csv, contrasts)
    _write_csv(interaction_csv, interactions)
    _write_csv(paired_grid_csv, paired_grid)

    table_primary = output_dir / "protocol_scaffold_table_tau07.tex"
    table_sensitivity = output_dir / "protocol_scaffold_table_tau09.tex"
    supplement_program = output_dir / "protocol_scaffold_supp_program.tex"
    supplement_operations = output_dir / "protocol_scaffold_supp_operations.tex"
    supplement_conditional = output_dir / "protocol_scaffold_supp_conditional.tex"
    supplement_contrasts = output_dir / "protocol_scaffold_supp_contrasts.tex"
    supplement_interactions = output_dir / "protocol_scaffold_supp_interactions.tex"
    table_primary.write_text(
        _latex_table(pooled_oc_rows, 0.7, denominator_panels=True),
        encoding="utf-8",
    )
    table_sensitivity.write_text(_latex_table(pooled_oc_rows, 0.9), encoding="utf-8")
    supplement_program.write_text(
        _supplement_program_table(pooled_oc_rows), encoding="utf-8"
    )
    supplement_operations.write_text(
        _supplement_operations_table(pooled_oc_rows), encoding="utf-8"
    )
    supplement_conditional.write_text(
        _supplement_conditional_table(pooled_oc_rows), encoding="utf-8"
    )
    supplement_contrasts.write_text(
        _supplement_contrast_table(contrasts), encoding="utf-8"
    )
    supplement_interactions.write_text(
        _supplement_interaction_table(interactions), encoding="utf-8"
    )
    figure = output_dir / "protocol_scaffold_sensitivity.png"
    figure_png, figure_pdf, figure_eps = _plot(interactions, figure)
    card = output_dir / "protocol_scaffold_policy_card.html"
    if source_path is not None:
        sdb.policy_card_from_file(source_path, card)
    else:
        card.write_text(sdb.build_policy_card(records), encoding="utf-8")
        _write_artifact_sidecar(
            card.with_suffix(card.suffix + ".metadata.json"),
            records,
            source_path=None,
            artifact_type="policy_card",
            projection_provenance=projection_provenance,
        )
    markdown = output_dir / "protocol_scaffold_sensitivity.md"
    markdown.write_text(
        _markdown(contrasts, interactions, paired_grid, seeds), encoding="utf-8"
    )

    primary_pair_records = [
        row for row in records
        if row["policy"] in PAIR_POLICIES
        and row["protocol_scaffold"] == "start_low_expansion"
        and float(row["gamma"]) == 0.7
    ]
    paired_grid_records = [row for row in records if row["policy"] in PAIR_POLICIES]
    tau07_records = [row for row in records if float(row["gamma"]) == 0.7]
    tau09_records = [row for row in records if float(row["gamma"]) == 0.9]
    artifacts = (
        (oc_csv, "table_data", records),
        (pooled_oc_csv, "pooled_table_data", records),
        (contrast_csv, "paired_contrasts", primary_pair_records),
        (interaction_csv, "scaffold_interactions", paired_grid_records),
        (paired_grid_csv, "paired_contrast_grid", paired_grid_records),
        (table_primary, "latex_table", tau07_records),
        (table_sensitivity, "latex_table", tau09_records),
        (supplement_program, "latex_table", records),
        (supplement_operations, "latex_table", records),
        (supplement_conditional, "latex_table", records),
        (supplement_contrasts, "latex_table", primary_pair_records),
        (supplement_interactions, "latex_table", paired_grid_records),
        (figure_png, "figure", paired_grid_records),
        (figure_pdf, "figure", paired_grid_records),
        (figure_eps, "figure", paired_grid_records),
        (markdown, "documentation", records),
    )
    for artifact, kind, metadata_records in artifacts:
        _write_artifact_sidecar(
            artifact.with_suffix(artifact.suffix + ".metadata.json"),
            metadata_records,
            source_path=source_path,
            artifact_type=kind,
            projection_provenance=projection_provenance,
        )
    generated_file_count = 2 * len(artifacts) + 2  # artifacts + sidecars + card pair
    print(f"generated {generated_file_count} files in {output_dir}")
    return {
        "oc": oc_rows,
        "pooled_oc": pooled_oc_rows,
        "contrasts": contrasts,
        "interactions": interactions,
        "paired_grid": paired_grid,
        "markdown": markdown,
        "figure": figure,
        "figure_pdf": figure_pdf,
        "figure_eps": figure_eps,
        "policy_card": card,
    }


def analyze(input_path, output_dir, *, require_full=True):
    """Analyze an explicit logical JSON file (retained for smoke tests)."""
    input_path = Path(input_path)
    records = json.loads(input_path.read_text(encoding="utf-8"))
    return _analyze_records(
        records,
        output_dir,
        require_full=require_full,
        source_path=input_path,
    )


def analyze_projection(projection_dir, output_dir, *, require_full=True):
    """Analyze the authenticated protocol record in a formal projection bundle."""
    records, provenance = load_projection_record(projection_dir, PROJECTION_RECORD)
    return _analyze_records(
        records,
        output_dir,
        require_full=require_full,
        source_path=None,
        projection_provenance=provenance,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        help=(
            "formal projection directory (with --projection) or a logical JSON "
            "file for smoke tests"
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--projection",
        action="store_true",
        help="authenticate all ten formal records and select the protocol record",
    )
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="permit fewer than 200 seeds for smoke-testing only")
    args = parser.parse_args(argv)
    if args.projection:
        analyze_projection(
            args.source,
            args.out_dir,
            require_full=not args.allow_incomplete,
        )
    else:
        analyze(
            args.source,
            args.out_dir,
            require_full=not args.allow_incomplete,
        )


if __name__ == "__main__":
    main()
