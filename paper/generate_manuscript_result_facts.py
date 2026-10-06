#!/usr/bin/env python3
"""Generate authenticated manuscript facts without hand-entered results.

This post-processing script loads one completed formal projection, recomputes
the manuscript-facing OSA, cross-testbed, protocol-scaffold, and schedule-ratio
summaries, and requires exact agreement with the corresponding generated CSV
artifacts.  Each CSV sidecar must point to the same projection fingerprint,
projection-metadata hash, and frozen formal-manifest hash.  The output is a
portable JSON fact record plus small LaTeX macro and prose-snippet files.

The script does not import a trial runner, fit a model, or run a simulation.
Scientific values, directions, interval exclusions, and exception lists are
derived from authenticated records on every invocation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np

# Direct ``python paper/...py`` execution puts paper/, not src/, on sys.path.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
for import_root in (PACKAGE_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import dose_combination_bo as sdb

from paper.analysis_record_bundle import ProjectionBundle, load_projection_bundle
from paper import analyze_cross_testbed_terminal_profile as cross
from paper import analyze_main_osa_gate_profile as gate
from paper import analyze_main_osa_primary as main_osa
from paper import analyze_protocol_scaffold_sensitivity as protocol
from paper import analyze_schedule_ratio_sensitivity as schedule
from paper import analyze_si_record_tables as si_tables


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "authenticated_manuscript_result_facts"
OUTPUT_JSON = "manuscript_result_facts.json"
OUTPUT_MACROS = "manuscript_result_facts.tex"
OUTPUT_SNIPPETS = "manuscript_result_snippets.tex"
OUTPUT_METADATA = "manuscript_result_facts.metadata.json"
EXACT_VALIDATION_SUMMARY = (
    PACKAGE_ROOT
    / "results"
    / "ckg_piecewise_exact_validation"
    / "ckg_piecewise_exact_summary.json"
)
EXACT_VALIDATION_METADATA = EXACT_VALIDATION_SUMMARY.with_suffix(
    EXACT_VALIDATION_SUMMARY.suffix + ".metadata.json"
)
PROTOCOL_TERMINAL_OUTCOME = (
    "True-boundary-exceeding final recommendation, all trials"
)
PROTOCOL_TERMINAL_OUTCOME_ALIASES = {
    PROTOCOL_TERMINAL_OUTCOME,
    "Above-threshold final recommendation, all trials",
}
PROTOCOL_ASSIGNMENT_OUTCOME = (
    "Post-initialization above-boundary simulated assignments"
)
PROTOCOL_ASSIGNMENT_OUTCOME_ALIASES = {
    PROTOCOL_ASSIGNMENT_OUTCOME,
    "Post-initialization above-boundary assignments",
}


class ResultFactsError(RuntimeError):
    """Raised when an input artifact or derived manuscript fact is inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_text(value) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _csv_bytes(rows) -> bytes:
    rows = list(rows)
    if not rows:
        raise ResultFactsError("cannot authenticate an empty analyzer CSV")
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _require_equal(observed, expected, label: str) -> None:
    if observed != expected:
        raise ResultFactsError(
            f"{label} differs: observed {observed!r}, expected {expected!r}"
        )


def _resolve_artifact(root: str | Path, *relative_candidates: str) -> Path:
    root = Path(root).resolve()
    candidates = [root / relative for relative in relative_candidates]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ResultFactsError(
            f"expected exactly one generated artifact below {root}: "
            + ", ".join(relative_candidates)
        )
    return matches[0]


def _projection_block(metadata: dict, label: str) -> dict:
    candidates = []
    if isinstance(metadata.get("formal_projection"), dict):
        candidates.append(metadata["formal_projection"])
    if isinstance(metadata.get("projection"), dict):
        candidates.append(metadata["projection"])
    inputs = metadata.get("inputs")
    if isinstance(inputs, dict) and isinstance(inputs.get("formal_projection"), dict):
        candidates.append(inputs["formal_projection"])
    if len(candidates) != 1:
        raise ResultFactsError(
            f"{label} must contain exactly one formal-projection provenance block"
        )
    return candidates[0]


def _validate_record_commitment(
    observed: dict,
    expected: dict,
    *,
    label: str,
) -> None:
    for field in (
        "rows",
        "compressed_bytes",
        "compressed_sha256",
        "uncompressed_bytes",
        "uncompressed_sha256",
    ):
        if field in observed:
            _require_equal(observed[field], expected[field], f"{label}.{field}")


def _validate_sidecar(
    artifact: Path,
    bundle: ProjectionBundle,
    *,
    expected_records: tuple[str, ...],
    schedule_manifest: bool = False,
) -> dict:
    metadata_path = (
        artifact.parent / "schedule_ratio_analysis.metadata.json"
        if schedule_manifest
        else artifact.with_suffix(artifact.suffix + ".metadata.json")
    )
    if not metadata_path.is_file():
        raise ResultFactsError(f"missing analyzer sidecar: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultFactsError(f"invalid analyzer sidecar JSON: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ResultFactsError(f"analyzer sidecar must be an object: {metadata_path}")

    provenance = _projection_block(metadata, metadata_path.name)
    expected_provenance = bundle.provenance
    for field in (
        "projection_fingerprint",
        "projection_metadata_sha256",
        "formal_manifest_sha256",
    ):
        _require_equal(
            provenance.get(field),
            expected_provenance[field],
            f"{metadata_path.name}.{field}",
        )

    expected_record_set = set(expected_records)
    if isinstance(provenance.get("selected_records"), dict):
        selected = provenance["selected_records"]
        _require_equal(
            set(selected), expected_record_set, f"{metadata_path.name}.selected_records"
        )
        for filename, detail in selected.items():
            _validate_record_commitment(
                detail,
                expected_provenance["record_files"][filename],
                label=f"{metadata_path.name}.{filename}",
            )
    elif isinstance(provenance.get("record_commitment"), dict):
        filename = provenance.get("record")
        _require_equal(
            {filename}, expected_record_set, f"{metadata_path.name}.record"
        )
        _validate_record_commitment(
            provenance["record_commitment"],
            expected_provenance["record_files"][filename],
            label=f"{metadata_path.name}.{filename}",
        )
    elif isinstance(metadata.get("input_files"), dict):
        observed_files = {}
        for detail in metadata["input_files"].values():
            if not isinstance(detail, dict) or "source_path" not in detail:
                raise ResultFactsError(
                    f"{metadata_path.name}.input_files contains an invalid entry"
                )
            observed_files[detail["source_path"]] = detail
        _require_equal(
            set(observed_files), expected_record_set, f"{metadata_path.name}.input_files"
        )
        for filename, detail in observed_files.items():
            expected = expected_provenance["record_files"][filename]
            mapped = {
                "rows": detail.get("trial_rows"),
                "compressed_sha256": detail.get("compressed_sha256"),
                "uncompressed_sha256": detail.get("uncompressed_sha256"),
            }
            _validate_record_commitment(
                mapped,
                expected,
                label=f"{metadata_path.name}.{filename}",
            )
    else:
        selected = provenance.get("selected_record")
        if selected is not None:
            _require_equal(
                {selected}, expected_record_set, f"{metadata_path.name}.selected_record"
            )
        elif len(expected_record_set) != 1:
            raise ResultFactsError(
                f"{metadata_path.name} does not identify its selected record files"
            )
        # Protocol sidecars carry the authenticated logical source digest directly.
        if len(expected_record_set) == 1 and metadata.get("source_sha256") is not None:
            filename = next(iter(expected_record_set))
            expected = expected_provenance["record_files"][filename]
            _require_equal(
                metadata["source_sha256"],
                expected["uncompressed_sha256"],
                f"{metadata_path.name}.source_sha256",
            )

    artifact_digest = _sha256(artifact)
    if metadata.get("artifact_sha256") is not None:
        _require_equal(
            metadata["artifact_sha256"],
            artifact_digest,
            f"{metadata_path.name}.artifact_sha256",
        )
    outputs = metadata.get("outputs")
    if isinstance(outputs, dict) and artifact.name in outputs:
        _require_equal(
            outputs[artifact.name].get("sha256"),
            artifact_digest,
            f"{metadata_path.name}.outputs.{artifact.name}.sha256",
        )

    return {
        "artifact": artifact.name,
        "artifact_sha256": artifact_digest,
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256(metadata_path),
        "projection_identity_verified": True,
    }


def _verify_csv(
    artifact: Path,
    rows,
    bundle: ProjectionBundle,
    *,
    expected_records: tuple[str, ...],
    schedule_manifest: bool = False,
) -> dict:
    expected = _csv_bytes(rows)
    observed = artifact.read_bytes()
    if observed != expected:
        raise ResultFactsError(
            f"{artifact.name} differs byte-for-byte from the record-derived recomputation"
        )
    detail = _validate_sidecar(
        artifact,
        bundle,
        expected_records=expected_records,
        schedule_manifest=schedule_manifest,
    )
    detail["recomputed_csv_byte_identity"] = True
    detail["rows"] = len(rows)
    return detail


def _direction(value: float) -> str:
    value = float(value)
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _interval_excludes_zero(low: float, high: float) -> bool:
    return bool(float(low) > 0.0 or float(high) < 0.0)


def _contrast_fact(
    row: dict,
    *,
    estimate_field: str,
    mcse_field: str,
    low_field: str | None = None,
    high_field: str | None = None,
    scale: float = 1.0,
    unit: str,
) -> dict:
    estimate = scale * float(row[estimate_field])
    mcse = scale * float(row[mcse_field])
    low = (
        scale * float(row[low_field])
        if low_field is not None
        else estimate - 1.96 * mcse
    )
    high = (
        scale * float(row[high_field])
        if high_field is not None
        else estimate + 1.96 * mcse
    )
    values = (estimate, mcse, low, high)
    if not all(math.isfinite(value) for value in values):
        raise ResultFactsError("a manuscript contrast contains a non-finite value")
    return {
        "estimate": estimate,
        "mcse": mcse,
        "pointwise_95_low": low,
        "pointwise_95_high": high,
        "direction": _direction(estimate),
        "pointwise_interval_excludes_zero": _interval_excludes_zero(low, high),
        "unit": unit,
    }


def _main_primary_rows(records):
    main_osa._validate_osa_main_records(records)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="no 'random' policy rows were supplied.*",
            category=UserWarning,
        )
        table = sdb.oc_table(
            records,
            sim="osa",
            gamma=main_osa.PRIMARY_TAU,
            by=("policy",),
        )
    rows = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    rows.sort(key=lambda row: main_osa.POLICIES.index(row["policy"]))
    return rows


def _main_facts(primary_rows, profile_rows, simultaneous):
    primary_fields = (
        "rec_efficacy",
        "dose_units",
        "pcs_within1",
        "rec_unsafe_pct",
        "pct_patients_above",
    )
    policies = {}
    for row in primary_rows:
        policies[row["policy"]] = {
            field: {
                "mean": float(row[field]),
                "mcse": float(row[f"{field}_mcse"]),
            }
            for field in primary_fields
        }

    pooled = [
        row for row in profile_rows if row["analysis_level"] == "pooled_strata"
    ]
    tau_primary = [
        row for row in pooled if float(row["tau"]) == main_osa.PRIMARY_TAU
    ]
    if len(tau_primary) != len(gate.OUTCOMES):
        raise ResultFactsError("OSA primary gate profile is incomplete")
    contrasts = {}
    for row in tau_primary:
        fact = _contrast_fact(
            row,
            estimate_field="difference_mean_pp",
            mcse_field="difference_mcse_pp",
            low_field="mc_interval_95_low_pp",
            high_field="mc_interval_95_high_pp",
            unit="percentage points",
        )
        joint_low = float(row["joint10_simultaneous_95_low_pp"])
        joint_high = float(row["joint10_simultaneous_95_high_pp"])
        fact.update(
            cKG_mean_pct=float(row["cKG_mean_pct"]),
            cEI_tMSE_mean_pct=float(row["cEI_tMSE_mean_pct"]),
            joint10_95_low=joint_low,
            joint10_95_high=joint_high,
            joint10_interval_excludes_zero=_interval_excludes_zero(
                joint_low, joint_high
            ),
            eligible_pairs=int(row["eligible_pairs"]),
            independent_seeds=int(row["independent_seeds"]),
        )
        contrasts[row["outcome_id"]] = fact

    pooled_summaries = {}
    for outcome_id, _label, _metric in gate.OUTCOMES:
        chosen = [row for row in pooled if row["outcome_id"] == outcome_id]
        signs = {
            direction: sum(_direction(row["difference_mean_pp"]) == direction for row in chosen)
            for direction in ("positive", "negative", "zero")
        }
        pooled_summaries[outcome_id] = {
            "cells": len(chosen),
            "sign_counts": signs,
            "joint10_intervals_excluding_zero": sum(
                _interval_excludes_zero(
                    row["joint10_simultaneous_95_low_pp"],
                    row["joint10_simultaneous_95_high_pp"],
                )
                for row in chosen
            ),
        }

    return {
        "sim": "osa",
        "dose_access": "full_panel",
        "primary_tau": main_osa.PRIMARY_TAU,
        "reader_facing_contrast": gate.CONTRAST_LABEL,
        "full_panel_reference_schedule": {
            "policy": "cEI-tMSE",
            "reader_facing_alias": "cEI–tMSE",
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
        "policy_operating_characteristics": policies,
        "cKG_minus_cEI_tMSE_at_primary_tau": contrasts,
        "pooled_five_threshold_summary": pooled_summaries,
        "joint10_max_t": {
            key: value
            for key, value in simultaneous.items()
            if key != "coordinates"
        },
    }


def _cross_facts(profile_rows):
    summary = cross._profile_summary(profile_rows)
    return {
        "testbeds": [item["id"] for item in cross.TESTBEDS],
        "gate_levels": list(cross.GATE_LEVELS),
        "displayed_testbed_threshold_cells": (
            len(cross.TESTBEDS) * len(cross.GATE_LEVELS)
        ),
        "displayed_contrasts": len(profile_rows),
        "pointwise_interval": "estimate plus or minus 1.96 paired MCSE; unadjusted",
        "summary": summary,
        "minority_sign_annotations": cross._minority_sign_annotations(profile_rows),
    }


def _posterior_learning_facts(bundle: ProjectionBundle) -> dict:
    """Derive the Table 4 OSA feasibility-learning summary from formal records."""
    rows = si_tables._build_gate_calibration(bundle).rows
    expected_cells = len(si_tables.DISPLAY_GATES) * len(si_tables.STRATA)
    expected_rows = expected_cells * len(si_tables.MAIN_POLICIES)
    _require_equal(len(rows), expected_rows, "OSA gate-calibration fact rows")
    lookup = {
        (float(row["tau"]), int(row["stratum"]), row["policy"]): row
        for row in rows
    }
    _require_equal(len(lookup), expected_rows, "unique OSA gate-calibration fact rows")

    both_improve = 0
    reference_best = 0
    for tau in si_tables.DISPLAY_GATES:
        for stratum in si_tables.STRATA:
            cei = lookup[float(tau), int(stratum), "cEI"]
            ckg = lookup[float(tau), int(stratum), "cKG"]
            reference = lookup[float(tau), int(stratum), "cEI-tMSE"]
            if (
                ckg["false_admissions_mean"] < cei["false_admissions_mean"]
                and reference["false_admissions_mean"]
                < cei["false_admissions_mean"]
                and ckg["gate_precision_pct_mean"]
                > cei["gate_precision_pct_mean"]
                and reference["gate_precision_pct_mean"]
                > cei["gate_precision_pct_mean"]
            ):
                both_improve += 1
            if (
                reference["false_admissions_mean"]
                < ckg["false_admissions_mean"]
                and reference["gate_precision_pct_mean"]
                > ckg["gate_precision_pct_mean"]
            ):
                reference_best += 1

    return {
        "sim": "osa",
        "dose_access": "full_panel",
        "gate_levels": list(si_tables.DISPLAY_GATES),
        "strata": list(si_tables.STRATA),
        "displayed_gate_by_stratum_cells": expected_cells,
        "point_estimate_summary": True,
        "both_cKG_and_reference_improve_over_cEI_on_both_diagnostics": (
            both_improve
        ),
        "reference_most_favorable_on_both_diagnostics": reference_best,
        "diagnostics": (
            "mean false admissions per trial and admission-weighted pooled gate "
            "precision"
        ),
    }


def _formal_projection_facts(
    projection_dir: str | Path,
    bundle: ProjectionBundle,
) -> dict:
    """Extract row/execution accounting already authenticated by the bundle loader."""
    metadata_path = Path(projection_dir).resolve() / "formal_projection.metadata.json"
    _require_equal(
        _sha256(metadata_path),
        bundle.provenance["projection_metadata_sha256"],
        "formal projection metadata SHA-256",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    counts = metadata.get("counts")
    validation = metadata.get("validation")
    if not isinstance(counts, dict) or not isinstance(validation, dict):
        raise ResultFactsError("formal projection metadata lacks counts or validation")
    total_rows = int(counts["total_rows"])
    approved_rows = int(counts["approved_replacement_rows"])
    unchanged_rows = int(counts["unchanged_rows"])
    file_counts = counts.get("files")
    if not isinstance(file_counts, dict):
        raise ResultFactsError("formal projection counts.files is missing")
    expected_names = set(bundle.records)
    _require_equal(set(file_counts), expected_names, "formal projection count file set")
    _require_equal(
        total_rows,
        sum(len(rows) for rows in bundle.records.values()),
        "formal projection logical row total",
    )
    all_files_replaced = all(
        int(detail["rows"]) == int(detail["approved_replacement_rows"])
        and int(detail["unchanged_rows"]) == 0
        for detail in file_counts.values()
    )
    all_rows_replaced = (
        approved_rows == total_rows
        and unchanged_rows == 0
        and all_files_replaced
    )
    if not all_rows_replaced:
        raise ResultFactsError("formal projection is not a complete row replacement")
    required_flags = (
        "all_target_rows_have_authenticated_formal_sources",
        "candidate_outputs_committed",
        "candidate_record_key_sets_unchanged",
        "formal_rows_canonically_ordered_and_complete",
        "unapproved_row_count_zero",
    )
    if not all(validation.get(field) is True for field in required_flags):
        raise ResultFactsError("formal projection validation flags are incomplete")

    decision_executions = int(counts["unique_decision_executions"])
    trajectory_executions = int(counts["unique_trajectory_replay_executions"])
    return {
        "logical_record_rows": total_rows,
        "approved_replacement_rows": approved_rows,
        "unchanged_rows": unchanged_rows,
        "record_files": len(file_counts),
        "all_rows_replaced": all_rows_replaced,
        "unique_decision_executions": decision_executions,
        "unique_trajectory_replay_executions": trajectory_executions,
        "unique_formal_executions": decision_executions + trajectory_executions,
        "execution_counts_are_not_logical_record_counts": True,
        "frozen_archive_read_only": bool(validation["frozen_archive_read_only"]),
        "not_available_from_projection_metadata": [
            "formal_stopped_execution_count",
            "full_execution stable-z/current operational-identity audit totals",
        ],
    }


_OPERATIONAL_AUDIT_FIELDS = (
    "eligible_fallback_index_difference_state_count",
    "eligible_gate_empty_difference_state_count",
    "eligible_gate_pass_set_difference_state_count",
    "full_fallback_index_difference_state_count",
    "gate_pass_set_difference_state_count",
    "operational_eligible_fallback_rounding_difference_state_count",
    "operational_full_fallback_rounding_difference_state_count",
)


def _exact_validation_facts(
    summary_path: str | Path = EXACT_VALIDATION_SUMMARY,
) -> dict:
    """Authenticate and summarize the tracked deterministic exact validation."""
    summary_path = Path(summary_path).resolve()
    metadata_path = summary_path.with_suffix(summary_path.suffix + ".metadata.json")
    if not summary_path.is_file() or not metadata_path.is_file():
        raise ResultFactsError("exact-validation summary or sidecar is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(metadata, dict):
        raise ResultFactsError("exact-validation summary and sidecar must be objects")
    _require_equal(metadata.get("artifact"), summary_path.name, "exact artifact name")
    _require_equal(
        metadata.get("artifact_sha256"),
        _sha256(summary_path),
        "exact-validation artifact SHA-256",
    )
    for source_field, expected_field in (
        ("exact_module", "exact_module_sha256"),
        ("generator", "generator_sha256"),
    ):
        source_name = metadata[source_field]
        candidates = (PACKAGE_ROOT / source_name, PACKAGE_ROOT / "paper" / source_name)
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            raise ResultFactsError(
                f"cannot resolve exact-validation {source_field}: {source_name}"
            )
        source = matches[0]
        _require_equal(
            _sha256(source), metadata[expected_field], f"exact validation {source_field}"
        )
    if metadata.get("production_harness_gate_semantics") != "stable_z_strict_ppf":
        raise ResultFactsError("exact-validation sidecar does not bind stable-z semantics")

    primary = summary["primary"]
    sentinels = summary["sentinels"]
    sobol = summary["sobol_audit"]
    audits = {
        "primary": primary["current_comparator_audit"],
        "sentinels": sentinels["current_comparator_audit"],
    }
    operational_counts = {
        scope: {field: int(audit[field]) for field in _OPERATIONAL_AUDIT_FIELDS}
        for scope, audit in audits.items()
    }
    if any(
        value != 0
        for scope in operational_counts.values()
        for value in scope.values()
    ):
        raise ResultFactsError(
            "tracked exact validation contains a stable/current operational difference"
        )
    total_operational_differences = sum(
        value
        for scope in operational_counts.values()
        for value in scope.values()
    )
    primary_states = int(primary["state_count"])
    sentinel_states = int(sentinels["state_count"])
    _require_equal(
        int(audits["primary"]["state_count"]), primary_states, "primary audit states"
    )
    _require_equal(
        int(audits["sentinels"]["state_count"]),
        sentinel_states,
        "sentinel audit states",
    )
    return {
        "status": summary["status"],
        "interpretation": summary["interpretation"],
        "primary_state_count": primary_states,
        "total_exact_query_evaluations": int(
            primary["total_exact_query_evaluations"]
        ),
        "full_score_vector_length": int(primary["full_score_vector_length"]),
        "sobol_audit": {
            "check_count": int(sobol["check_count"]),
            "state_count": int(sobol["state_count"]),
            "points_per_check": int(sobol["points_per_check"]),
            "max_abs_error": float(sobol["max_abs_error"]),
            "p95_abs_error": float(sobol["p95_abs_error"]),
        },
        "sentinels": {
            "state_count": sentinel_states,
            "sobol_check_count": int(sentinels["sobol_check_count"]),
            "sobol_max_abs_error": float(sentinels["sobol_max_abs_error"]),
        },
        "stable_z_current_operational_audit": {
            "audited_state_count": primary_states + sentinel_states,
            "difference_counts": operational_counts,
            "total_difference_count": total_operational_differences,
            "all_difference_counts_zero": True,
            "inactive_rounding_differences_are_not_operational": True,
        },
        "historical_approximation_comparisons": {
            "historical_only": True,
            "not_the_current_exact_implementation": True,
            "legacy_512_and_prefix_preserving_1024": primary["approximations"],
        },
        "production_default_changed_by_validation": bool(
            metadata["production_default_changed"]
        ),
        "production_harness_gate_semantics": metadata[
            "production_harness_gate_semantics"
        ],
        "provenance": {
            "summary": summary_path.name,
            "summary_sha256": metadata["artifact_sha256"],
            "metadata": metadata_path.name,
            "metadata_sha256": _sha256(metadata_path),
            "exact_module": metadata["exact_module"],
            "exact_module_sha256": metadata["exact_module_sha256"],
            "generator": metadata["generator"],
            "generator_sha256": metadata["generator_sha256"],
        },
    }


def _protocol_scale(label: str) -> tuple[float, str]:
    if label in PROTOCOL_TERMINAL_OUTCOME_ALIASES or label == "Feasibility stop":
        return 100.0, "percentage points"
    if label in {"Recommended efficacy | recommend"}:
        return 1.0, "efficacy units"
    if label in {"Dose-location error | recommend"}:
        return 1.0, "dose-location units"
    if label in PROTOCOL_ASSIGNMENT_OUTCOME_ALIASES or label == "Sample size":
        return 1.0, "assignments"
    raise ResultFactsError(f"unknown protocol outcome label: {label!r}")


def _protocol_row_fact(row: dict, *, interaction: bool = False) -> dict:
    scale, unit = _protocol_scale(row["label"])
    prefix = "interaction" if interaction else "difference"
    fact = _contrast_fact(
        row,
        estimate_field=f"{prefix}_mean",
        mcse_field=f"{prefix}_se",
        scale=scale,
        unit=unit,
    )
    outcome = (
        PROTOCOL_TERMINAL_OUTCOME
        if row["label"] in PROTOCOL_TERMINAL_OUTCOME_ALIASES
        else (
            PROTOCOL_ASSIGNMENT_OUTCOME
            if row["label"] in PROTOCOL_ASSIGNMENT_OUTCOME_ALIASES
            else row["label"]
        )
    )
    fact.update(
        outcome=outcome,
        tau=float(row["gamma"]),
        eligible_pairs=int(row.get("eligible_pair_n", row["n_pairs"])),
        independent_seeds=int(row.get("eligible_seed_n", row["n"])),
    )
    if "protocol_scaffold" in row:
        fact["protocol_scaffold"] = row["protocol_scaffold"]
    return fact


def _protocol_facts(contrasts, interactions, paired_grid):
    category, sentence, _deltas = protocol._finding(paired_grid)
    primary = [_protocol_row_fact(row) for row in contrasts]
    grid = [_protocol_row_fact(row) for row in paired_grid]
    interaction = [
        _protocol_row_fact(row, interaction=True) for row in interactions
    ]
    return {
        "contrast": "cKG minus cEI–tMSE",
        "primary_protocol_scaffold": "start_low_expansion",
        "primary_tau": 0.7,
        "primary_and_secondary_contrasts": primary,
        "paired_grid": grid,
        "scaffold_interactions": interaction,
        "derived_finding_category": category,
        "derived_finding_sentence": sentence,
    }


def _schedule_facts(summaries, contrasts):
    summary_rows = []
    for row in summaries:
        if row["outcome_id"] == "above_boundary_assignment_percentage":
            continue
        summary_rows.append(
            {
                "policy": row["policy"],
                "policy_label": row["policy_label"],
                "outcome_id": row["outcome_id"],
                "mean": float(row["mean"]),
                "mcse": float(row["mcse"]),
                "unit": row["unit"],
                "independent_seeds": int(row["independent_seeds"]),
            }
        )
    contrast_rows = []
    for row in contrasts:
        if row["outcome_id"] == "above_boundary_assignment_percentage":
            continue
        fact = _contrast_fact(
            row,
            estimate_field="difference_mean",
            mcse_field="difference_mcse",
            low_field="pointwise_95_low",
            high_field="pointwise_95_high",
            unit=row["unit"],
        )
        fact.update(
            contrast=row["contrast"],
            contrast_role=row["contrast_role"],
            policy_a=row["policy_a"],
            policy_b=row["policy_b"],
            outcome_id=row["outcome_id"],
            eligible_stratum_pairs=int(row["eligible_stratum_pairs"]),
            independent_seeds=int(row["independent_seeds"]),
        )
        contrast_rows.append(fact)
    return {
        "sim": "osa",
        "tau": 0.7,
        "dose_access": "full_panel",
        "post_hoc_descriptive": True,
        "ratio_tuned": False,
        "new_schedule_cycle": list(schedule.SCHEDULE_CYCLE),
        "operating_characteristics": summary_rows,
        "paired_contrasts": contrast_rows,
    }


def derive_and_verify(
    bundle: ProjectionBundle,
    *,
    projection_dir: str | Path,
    main_osa_dir: str | Path,
    main_osa_gate_dir: str | Path | None = None,
    cross_testbed_dir: str | Path,
    protocol_dir: str | Path,
    schedule_dir: str | Path,
    exact_validation_summary: str | Path = EXACT_VALIDATION_SUMMARY,
    max_t_resamples: int = gate.BOOTSTRAP_RESAMPLES,
    max_t_seed: int = gate.BOOTSTRAP_SEED,
    max_t_batch_size: int = gate.BOOTSTRAP_BATCH_SIZE,
) -> dict:
    """Recompute all facts and authenticate the generated analyzer artifacts."""
    verified_inputs = {}

    osa_records = bundle.records[main_osa.OSA_RECORD_NAME]
    primary_rows = _main_primary_rows(osa_records)
    profile_rows, gate_selected = gate._profile_rows(osa_records)
    simultaneous = gate._attach_simultaneous_bands(
        profile_rows,
        gate_selected,
        n_resamples=max_t_resamples,
        rng_seed=max_t_seed,
        batch_size=max_t_batch_size,
    )
    main_primary_csv = _resolve_artifact(
        main_osa_dir, "primary/main_osa_primary.csv", "main_osa_primary.csv"
    )
    gate_csv = _resolve_artifact(
        main_osa_gate_dir if main_osa_gate_dir is not None else main_osa_dir,
        "gate/osa_gate_profile_contrasts.csv",
        "osa_gate_profile_contrasts.csv",
    )
    verified_inputs["main_osa_primary"] = _verify_csv(
        main_primary_csv,
        primary_rows,
        bundle,
        expected_records=(main_osa.OSA_RECORD_NAME,),
    )
    verified_inputs["main_osa_gate_profile"] = _verify_csv(
        gate_csv,
        profile_rows,
        bundle,
        expected_records=(main_osa.OSA_RECORD_NAME,),
    )

    records_by_testbed = {
        testbed["id"]: bundle.records[testbed["record_file"]]
        for testbed in cross.TESTBEDS
    }
    cross_rows, _cross_selected = cross._profile_rows(records_by_testbed)
    cross_csv = _resolve_artifact(
        cross_testbed_dir, "cross_testbed_terminal_contrasts.csv"
    )
    verified_inputs["cross_testbed_profile"] = _verify_csv(
        cross_csv,
        cross_rows,
        bundle,
        expected_records=tuple(item["record_file"] for item in cross.TESTBEDS),
    )

    protocol_records = bundle.records[protocol.PROJECTION_RECORD]
    protocol._validate(protocol_records, require_full=True)
    protocol_contrasts = protocol._contrast_rows(protocol_records)
    protocol_interactions = protocol._interaction_rows(protocol_records)
    protocol_grid = protocol._paired_grid(protocol_records)
    for key, filename, rows in (
        (
            "protocol_primary_secondary",
            "protocol_scaffold_primary_secondary_contrasts.csv",
            protocol_contrasts,
        ),
        (
            "protocol_interactions",
            "protocol_scaffold_interactions.csv",
            protocol_interactions,
        ),
        (
            "protocol_paired_grid",
            "protocol_scaffold_paired_grid.csv",
            protocol_grid,
        ),
    ):
        artifact = _resolve_artifact(protocol_dir, filename)
        verified_inputs[key] = _verify_csv(
            artifact,
            rows,
            bundle,
            expected_records=(protocol.PROJECTION_RECORD,),
        )

    sixway = bundle.records[schedule.SIXWAY_RECORD_NAME]
    new_schedule = bundle.records[schedule.SCHEDULE_RECORD_NAME]
    schedule._validate_new(new_schedule, require_full=True)
    schedule_combined = schedule._combine(
        schedule._select_reference(sixway), new_schedule
    )
    schedule_summaries = schedule._summary_rows(schedule_combined)
    schedule_contrasts = schedule._contrast_rows(
        schedule_combined, schedule_summaries
    )
    for key, filename, rows in (
        (
            "schedule_operating_characteristics",
            "schedule_ratio_operating_characteristics.csv",
            schedule_summaries,
        ),
        (
            "schedule_paired_contrasts",
            "schedule_ratio_paired_contrasts.csv",
            schedule_contrasts,
        ),
    ):
        artifact = _resolve_artifact(schedule_dir, filename)
        verified_inputs[key] = _verify_csv(
            artifact,
            rows,
            bundle,
            expected_records=(
                schedule.SIXWAY_RECORD_NAME,
                schedule.SCHEDULE_RECORD_NAME,
            ),
            schedule_manifest=True,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "provenance": {
            "projection_fingerprint": bundle.provenance[
                "projection_fingerprint"
            ],
            "projection_metadata_sha256": bundle.provenance[
                "projection_metadata_sha256"
            ],
            "formal_manifest_sha256": bundle.provenance[
                "formal_manifest_sha256"
            ],
            "all_analyzer_csvs_recomputed_byte_identical": True,
            "all_analyzer_sidecars_match_projection": True,
            "verified_analysis_inputs": verified_inputs,
        },
        "formal_projection": _formal_projection_facts(projection_dir, bundle),
        "exact_validation": _exact_validation_facts(exact_validation_summary),
        "main_osa": _main_facts(primary_rows, profile_rows, simultaneous),
        "cross_testbed": _cross_facts(cross_rows),
        "final_posterior_feasibility_learning": _posterior_learning_facts(bundle),
        "protocol_scaffold": _protocol_facts(
            protocol_contrasts, protocol_interactions, protocol_grid
        ),
        "schedule_ratio": _schedule_facts(
            schedule_summaries, schedule_contrasts
        ),
    }


def _find_one(rows, **fields):
    matches = [
        row for row in rows
        if all(row.get(field) == value for field, value in fields.items())
    ]
    if len(matches) != 1:
        raise ResultFactsError(
            f"expected one fact row for {fields}, observed {len(matches)}"
        )
    return matches[0]


def _tex_number(value: float, digits: int, *, signed: bool = False) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ResultFactsError("cannot write a non-finite LaTeX numeric fact")
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _tex_scientific(value: float, digits: int = 2) -> str:
    mantissa, exponent = f"{float(value):.{digits}e}".split("e")
    return rf"\ensuremath{{{mantissa}\times 10^{{{int(exponent)}}}}}"


def _tex_macro(name: str, value: str) -> str:
    if not name.isalpha():
        raise ResultFactsError(f"invalid generated LaTeX macro name: {name}")
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _macro_text(facts: dict) -> str:
    main = facts["main_osa"]
    main_contrasts = main["cKG_minus_cEI_tMSE_at_primary_tau"]
    terminal = main_contrasts["final_threshold_exceeding_recommendation"]
    assignment = main_contrasts["above_boundary_simulated_assignments"]
    cross_summary = facts["cross_testbed"]["summary"]["outcomes"]
    osa_gate_summary = main["pooled_five_threshold_summary"]
    posterior_learning = facts["final_posterior_feasibility_learning"]
    protocol_rows = facts["protocol_scaffold"]["paired_grid"]
    gradual_eff = _find_one(
        protocol_rows,
        protocol_scaffold="start_low_expansion",
        tau=0.7,
        outcome="Recommended efficacy | recommend",
    )
    gradual_terminal = _find_one(
        protocol_rows,
        protocol_scaffold="start_low_expansion",
        tau=0.7,
        outcome=PROTOCOL_TERMINAL_OUTCOME,
    )
    gradual_dose = _find_one(
        protocol_rows,
        protocol_scaffold="start_low_expansion",
        tau=0.7,
        outcome="Dose-location error | recommend",
    )
    schedule_rows = facts["schedule_ratio"]["paired_contrasts"]
    ckg_2to1_terminal = _find_one(
        schedule_rows,
        policy_a="cKG",
        policy_b=schedule.NEW_POLICY,
        outcome_id="terminal_threshold_event",
    )
    ckg_2to1_assignment = _find_one(
        schedule_rows,
        policy_a="cKG",
        policy_b=schedule.NEW_POLICY,
        outcome_id="above_boundary_assignment_count",
    )

    macros = {
        "FormalProjectionRecordRows": str(
            facts["formal_projection"]["logical_record_rows"]
        ),
        "FormalProjectionReplacementRows": str(
            facts["formal_projection"]["approved_replacement_rows"]
        ),
        "FormalProjectionDecisionExecutions": str(
            facts["formal_projection"]["unique_decision_executions"]
        ),
        "FormalProjectionTrajectoryExecutions": str(
            facts["formal_projection"]["unique_trajectory_replay_executions"]
        ),
        "ExactValidationPrimaryStates": str(
            facts["exact_validation"]["primary_state_count"]
        ),
        "ExactValidationQueryEvaluations": str(
            facts["exact_validation"]["total_exact_query_evaluations"]
        ),
        "ExactValidationSobolChecks": str(
            facts["exact_validation"]["sobol_audit"]["check_count"]
        ),
        "ExactValidationSobolMaxAbsError": (
            _tex_scientific(
                facts["exact_validation"]["sobol_audit"]["max_abs_error"]
            )
        ),
        "ExactValidationSentinelStates": str(
            facts["exact_validation"]["sentinels"]["state_count"]
        ),
        "ExactValidationOperationalDifferences": str(
            facts["exact_validation"]["stable_z_current_operational_audit"][
                "total_difference_count"
            ]
        ),
        "OSAPrimaryTau": _tex_number(main["primary_tau"], 1),
        "OSAJointTenMaxTCritical": _tex_number(
            main["joint10_max_t"]["critical_values"]["joint_10_contrasts"], 3
        ),
        "OSAPrimaryTerminalDifferencePP": _tex_number(
            terminal["estimate"], 2, signed=True
        ),
        "OSAPrimaryTerminalJointLowPP": _tex_number(
            terminal["joint10_95_low"], 2, signed=True
        ),
        "OSAPrimaryTerminalJointHighPP": _tex_number(
            terminal["joint10_95_high"], 2, signed=True
        ),
        "OSAPrimaryAssignmentDifferencePP": _tex_number(
            assignment["estimate"], 2, signed=True
        ),
        "OSAPrimaryAssignmentJointLowPP": _tex_number(
            assignment["joint10_95_low"], 2, signed=True
        ),
        "OSAPrimaryAssignmentJointHighPP": _tex_number(
            assignment["joint10_95_high"], 2, signed=True
        ),
        "CrossTestbedCellCount": str(
            facts["cross_testbed"]["displayed_testbed_threshold_cells"]
        ),
        "CrossTerminalPositiveCount": str(
            cross_summary["terminal_recommendation"]["sign_counts"]["positive"]
        ),
        "CrossTerminalIntervalsExcludingZero": str(
            cross_summary["terminal_recommendation"][
                "pointwise_intervals_excluding_zero"
            ]
        ),
        "CrossAssignmentNegativeCount": str(
            cross_summary["assignment_percentage"]["sign_counts"]["negative"]
        ),
        "CrossAssignmentIntervalsExcludingZero": str(
            cross_summary["assignment_percentage"][
                "pointwise_intervals_excluding_zero"
            ]
        ),
        "OSAGateCount": str(
            osa_gate_summary["above_boundary_simulated_assignments"]["cells"]
        ),
        "OSAAssignmentJointBandExclusionCount": str(
            osa_gate_summary["above_boundary_simulated_assignments"][
                "joint10_intervals_excluding_zero"
            ]
        ),
        "OSATerminalJointBandExclusionCount": str(
            osa_gate_summary["final_threshold_exceeding_recommendation"][
                "joint10_intervals_excluding_zero"
            ]
        ),
        "OSAPosteriorDisplayedCellCount": str(
            posterior_learning["displayed_gate_by_stratum_cells"]
        ),
        "OSAPosteriorBothImproveCellCount": str(
            posterior_learning[
                "both_cKG_and_reference_improve_over_cEI_on_both_diagnostics"
            ]
        ),
        "OSAPosteriorReferenceBestCellCount": str(
            posterior_learning["reference_most_favorable_on_both_diagnostics"]
        ),
        "ProtocolGradualEfficacyDifference": _tex_number(
            gradual_eff["estimate"], 3, signed=True
        ),
        "ProtocolGradualEfficacyLow": _tex_number(
            gradual_eff["pointwise_95_low"], 3, signed=True
        ),
        "ProtocolGradualEfficacyHigh": _tex_number(
            gradual_eff["pointwise_95_high"], 3, signed=True
        ),
        "ProtocolGradualTerminalDifferencePP": _tex_number(
            gradual_terminal["estimate"], 2, signed=True
        ),
        "ProtocolGradualTerminalLowPP": _tex_number(
            gradual_terminal["pointwise_95_low"], 2, signed=True
        ),
        "ProtocolGradualTerminalHighPP": _tex_number(
            gradual_terminal["pointwise_95_high"], 2, signed=True
        ),
        "ProtocolGradualDoseErrorDifference": _tex_number(
            gradual_dose["estimate"], 3, signed=True
        ),
        "ProtocolGradualDoseErrorLow": _tex_number(
            gradual_dose["pointwise_95_low"], 3, signed=True
        ),
        "ProtocolGradualDoseErrorHigh": _tex_number(
            gradual_dose["pointwise_95_high"], 3, signed=True
        ),
        "ScheduleCKGTwoToOneTerminalDifferencePP": _tex_number(
            ckg_2to1_terminal["estimate"], 1, signed=True
        ),
        "ScheduleCKGTwoToOneTerminalLowPP": _tex_number(
            ckg_2to1_terminal["pointwise_95_low"], 1, signed=True
        ),
        "ScheduleCKGTwoToOneTerminalHighPP": _tex_number(
            ckg_2to1_terminal["pointwise_95_high"], 1, signed=True
        ),
        "ScheduleCKGTwoToOneAssignmentDifference": _tex_number(
            ckg_2to1_assignment["estimate"], 2, signed=True
        ),
        "ScheduleCKGTwoToOneAssignmentLow": _tex_number(
            ckg_2to1_assignment["pointwise_95_low"], 2, signed=True
        ),
        "ScheduleCKGTwoToOneAssignmentHigh": _tex_number(
            ckg_2to1_assignment["pointwise_95_high"], 2, signed=True
        ),
    }
    lines = [
        "% Generated by generate_manuscript_result_facts.py; do not hand-edit.",
        "% Every numeric value is regenerated from the formal simulation records.",
    ]
    lines.extend(_tex_macro(name, value) for name, value in macros.items())
    return "\n".join(lines) + "\n"


def _sign_phrase(counts: dict, direction: str, cells: int) -> str:
    count = int(counts[direction])
    if count == cells:
        return f"all {cells}"
    return f"{count} of {cells}"


def _snippet_text(facts: dict) -> str:
    main = facts["main_osa"]
    terminal = main["cKG_minus_cEI_tMSE_at_primary_tau"][
        "final_threshold_exceeding_recommendation"
    ]
    assignment = main["cKG_minus_cEI_tMSE_at_primary_tau"][
        "above_boundary_simulated_assignments"
    ]
    cross_facts = facts["cross_testbed"]
    cross_summary = cross_facts["summary"]["outcomes"]
    cells = cross_facts["displayed_testbed_threshold_cells"]
    terminal_signs = cross_summary["terminal_recommendation"]["sign_counts"]
    assignment_signs = cross_summary["assignment_percentage"]["sign_counts"]
    protocol_sentence = facts["protocol_scaffold"]["derived_finding_sentence"].replace(
        "above-threshold final recommendations",
        "true-boundary-exceeding final recommendations",
    )

    projection = facts["formal_projection"]
    projection_sentence = (
        f"The formal simulation projection contains "
        f"{projection['logical_record_rows']:,} logical record rows, with "
        f"{projection['approved_replacement_rows']:,} approved replacements and "
        f"{projection['unchanged_rows']:,} unchanged rows. Its "
        f"{projection['unique_decision_executions']:,} unique decision executions "
        f"and {projection['unique_trajectory_replay_executions']:,} trajectory "
        "replays are execution counts, not additional logical record rows."
    )
    exact = facts["exact_validation"]
    exact_sentence = (
        "The deterministic finite-set evaluator was checked on "
        f"{exact['primary_state_count']:,} posterior states "
        f"({exact['total_exact_query_evaluations']:,} query evaluations); "
        f"{exact['sobol_audit']['check_count']:,} independently scrambled Sobol checks had maximum absolute "
        f"error {_tex_scientific(exact['sobol_audit']['max_abs_error'])}, and "
        f"{exact['sentinels']['state_count']} numerical sentinels were included. "
        "Gate decisions and empty-gate choices agreed in every audited operational "
        "state (total differences: "
        f"{exact['stable_z_current_operational_audit']['total_difference_count']})."
    )

    main_sentence = (
        rf"At $\tau={main['primary_tau']:.1f}$, relative to the study-defined "
        rf"reference schedule alternating cEI and tMSE across successive adaptive "
        rf"cohorts (cEI–tMSE), the cKG "
        rf"difference was {terminal['estimate']:+.2f} percentage points for the "
        rf"true-boundary-exceeding final recommendation "
        rf"(joint max-$t$ 95\% interval "
        rf"[{terminal['joint10_95_low']:+.2f}, "
        rf"{terminal['joint10_95_high']:+.2f}]) and "
        rf"{assignment['estimate']:+.2f} percentage points for above-boundary "
        rf"simulated assignments (joint max-$t$ 95\% interval "
        rf"[{assignment['joint10_95_low']:+.2f}, "
        rf"{assignment['joint10_95_high']:+.2f}])."
    )
    cross_sentence = (
        "Across the evaluated testbed--threshold cells, "
        + _sign_phrase(terminal_signs, "positive", cells)
        + " terminal-recommendation point estimates were positive and "
        + _sign_phrase(assignment_signs, "negative", cells)
        + " assignment point estimates were negative; "
        + str(cross_summary["terminal_recommendation"][
            "pointwise_intervals_excluding_zero"
        ])
        + f"/{cells} and "
        + str(cross_summary["assignment_percentage"][
            "pointwise_intervals_excluding_zero"
        ])
        + f"/{cells} pointwise intervals excluded zero, respectively."
    )
    schedule_terminal = _find_one(
        facts["schedule_ratio"]["paired_contrasts"],
        policy_a="cKG",
        policy_b=schedule.NEW_POLICY,
        outcome_id="terminal_threshold_event",
    )
    schedule_assignment = _find_one(
        facts["schedule_ratio"]["paired_contrasts"],
        policy_a="cKG",
        policy_b=schedule.NEW_POLICY,
        outcome_id="above_boundary_assignment_count",
    )
    schedule_sentence = (
        "In the exploratory cEI-heavy schedule-composition check, cKG minus the "
        rf"cEI--cEI--tMSE cycle was {schedule_terminal['estimate']:+.1f} percentage "
        rf"points for a true-boundary-exceeding final recommendation "
        rf"([{schedule_terminal['pointwise_95_low']:+.1f}, "
        rf"{schedule_terminal['pointwise_95_high']:+.1f}]) and "
        rf"{schedule_assignment['estimate']:+.2f} above-boundary simulated "
        rf"assignments per 40-assignment trial "
        rf"([{schedule_assignment['pointwise_95_low']:+.2f}, "
        rf"{schedule_assignment['pointwise_95_high']:+.2f}])."
    )
    commands = {
        "FormalProjectionFactSentence": projection_sentence,
        "ExactValidationFactSentence": exact_sentence,
        "OSAPrimaryResultSentence": main_sentence,
        "CrossTestbedResultSentence": cross_sentence,
        "ProtocolScaffoldResultSentence": protocol_sentence.replace("%", r"\%"),
        "ScheduleRatioResultSentence": schedule_sentence,
    }
    lines = [
        "% Generated by generate_manuscript_result_facts.py; do not hand-edit.",
        "% These prose snippets are descriptive and inherit the scopes in the JSON fact record.",
    ]
    lines.extend(_tex_macro(name, value) for name, value in commands.items())
    return "\n".join(lines) + "\n"


def write_outputs(facts: dict, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / OUTPUT_JSON
    macros_path = output_dir / OUTPUT_MACROS
    snippets_path = output_dir / OUTPUT_SNIPPETS
    metadata_path = output_dir / OUTPUT_METADATA

    json_path.write_text(_json_text(facts), encoding="utf-8")
    macros_path.write_text(_macro_text(facts), encoding="utf-8")
    snippets_path.write_text(_snippet_text(facts), encoding="utf-8")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "manuscript_result_facts_output_manifest",
        "generator": Path(__file__).name,
        "generator_sha256": _sha256(Path(__file__)),
        "projection_fingerprint": facts["provenance"]["projection_fingerprint"],
        "projection_metadata_sha256": facts["provenance"][
            "projection_metadata_sha256"
        ],
        "formal_manifest_sha256": facts["provenance"][
            "formal_manifest_sha256"
        ],
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (json_path, macros_path, snippets_path)
        },
    }
    metadata_path.write_text(_json_text(metadata), encoding="utf-8")
    return {
        "json": json_path,
        "macros": macros_path,
        "snippets": snippets_path,
        "metadata": metadata_path,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-dir", required=True)
    parser.add_argument("--main-osa-dir", required=True)
    parser.add_argument(
        "--main-osa-gate-dir",
        help=(
            "directory containing the current gate-profile CSV/sidecar; defaults "
            "to --main-osa-dir"
        ),
    )
    parser.add_argument("--cross-testbed-dir", required=True)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--schedule-dir", required=True)
    parser.add_argument(
        "--exact-validation-summary",
        default=str(EXACT_VALIDATION_SUMMARY),
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    bundle = load_projection_bundle(args.projection_dir)
    facts = derive_and_verify(
        bundle,
        projection_dir=args.projection_dir,
        main_osa_dir=args.main_osa_dir,
        main_osa_gate_dir=args.main_osa_gate_dir,
        cross_testbed_dir=args.cross_testbed_dir,
        protocol_dir=args.protocol_dir,
        schedule_dir=args.schedule_dir,
        exact_validation_summary=args.exact_validation_summary,
    )
    outputs = write_outputs(facts, args.out_dir)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
