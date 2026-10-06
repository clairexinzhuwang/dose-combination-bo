#!/usr/bin/env python3
"""Authenticate and project a completed formal rerun into candidate records.

This is a post-processing program.  It imports no trial harness or acquisition
implementation and cannot run a simulation.  The default invocation validates
all inputs and constructs the candidate rows in memory without writing them.
``--write`` commits all ten records as one new hash-addressed staging directory.
The frozen archive is always a read-only merge base.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import zstandard as zstd

from paper import analyze_primary_osa_exact_ckg_gate as primary_analysis
from paper import formal_rerun_preflight as preflight


DECISION_NAME = "formal_decision_master.json"
TRAJECTORY_NAME = "formal_trajectory_replays.json"
DECISION_STATUS = "COMPLETE_FORMAL_DECISION_MASTER"
TRAJECTORY_STATUS = "COMPLETE_FORMAL_TRAJECTORY_REPLAYS"
FORMAL_ARTIFACT_CLASS = "formal_current_harness_staging_not_archive_records"
FORMAL_AUTHORIZATION = "FULL_FORMAL_RUN_GO"
PROJECTION_STATUS = "COMPLETE_AUTHENTICATED_FORMAL_RECORD_PROJECTION"
PROJECTION_ARTIFACT_CLASS = "formal_record_projection_staging_not_archive"
RECORD_FILENAMES = tuple(preflight._RECORD_KEYS)
CANONICAL_SORT_KEY = (
    "matrix_id", "policy", "sim", "mode", "protocol_scaffold",
    "kap", "gamma", "stratum", "seed",
)
FORMAL_SOURCE_PREFIX = (
    "paper/run_full_formal_rerun.py",
    "paper/formal_rerun_preflight.py",
    "paper/formal_rerun_manifest.json",
    "paper/run_primary_osa_exact_ckg_gate.py",
    "paper/analyze_primary_osa_exact_ckg_gate.py",
)
ZERO_GATE_FIELDS = (
    "gate_pass_set_difference_count",
    "eligible_gate_pass_set_difference_count",
    "operational_full_fallback_index_difference_count",
    "operational_eligible_fallback_index_difference_count",
)
SCHEDULE_POST_HOC_FIELDS: dict[str, Any] = {
    "post_hoc_descriptive": True,
    "schedule_tuned": False,
    "schedule_ratio": "2:1 cEI:tMSE",
    "schedule_cycle": ["cEI", "cEI", "tMSE"],
    "schedule_phase_origin": "first adaptive cohort",
    "schedule_review_status": (
        "fixed before this post-hoc review run after the original study results were known"
    ),
}


class ProjectionError(RuntimeError):
    """Raised when an authenticated projection invariant fails."""


@dataclass(frozen=True)
class FormalInput:
    path: Path
    payload: dict[str, Any]
    rows_by_key: dict[tuple[Any, ...], dict[str, Any]]
    artifact_sha256: str
    metadata_sha256: str


@dataclass
class PreparedProjection:
    candidate_rows: dict[str, list[dict[str, Any]]]
    metadata: dict[str, Any]
    fingerprint: str
    frozen_hashes: dict[str, str]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return preflight.file_sha256(Path(path))


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True,
        "allow_nan": False,
        "ensure_ascii": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ProjectionError(
            f"{label} differs: observed {observed!r}, expected {expected!r}"
        )


def _assert_finite_json(value: Any, *, label: str = "payload") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProjectionError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_json(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_json(child, label=f"{label}[{index}]")


def _runtime_identity(exact_panel_path: Path) -> dict[str, str]:
    panel = json.loads(Path(exact_panel_path).read_text(encoding="utf-8"))
    environment = panel.get("environment", {})
    identity = {
        key: environment.get(key)
        for key in ("python", "numpy", "scipy", "torch", "gpytorch")
    }
    if any(not isinstance(value, str) or not value for value in identity.values()):
        raise ProjectionError("exact-validation runtime identity is incomplete")
    return identity  # type: ignore[return-value]


def _required_source_files(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    execution = tuple(manifest["artifact_contract"]["execution_source_files"])
    required = tuple(dict.fromkeys((*FORMAL_SOURCE_PREFIX, *execution)))
    if len(required) != 14:
        raise ProjectionError("formal execution source contract must contain 14 files")
    return required


def _expected_source_hashes(
    manifest: Mapping[str, Any], *, manifest_path: Path
) -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in _required_source_files(manifest):
        source = (
            Path(manifest_path)
            if relative == "paper/formal_rerun_manifest.json"
            else ROOT / relative
        )
        if not source.is_file():
            raise ProjectionError(f"formal execution source is absent: {relative}")
        output[relative] = _sha256_file(source)
    return output


def _read_json(path: Path) -> tuple[bytes, Any]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectionError(f"{path}: not valid JSON") from exc
    _assert_finite_json(value, label=Path(path).name)
    return raw, value


def _validate_gate_audit(
    audit: Any,
    result: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(audit, dict):
        raise ProjectionError(f"{label}: missing gate-numerics audit")
    checked = audit.get("checked_state_count")
    if not isinstance(checked, int) or isinstance(checked, bool) or checked < 1:
        raise ProjectionError(f"{label}: no audited gate state")
    if audit.get("terminal_state_checked") is not True:
        raise ProjectionError(f"{label}: terminal gate state was not audited")
    for field in ZERO_GATE_FIELDS:
        if audit.get(field) != 0:
            raise ProjectionError(f"{label}: nonzero or missing {field}")
    for field in (
        "inactive_full_fallback_rounding_difference_count",
        "inactive_eligible_fallback_rounding_difference_count",
        "operational_full_fallback_rounding_difference_count",
        "operational_eligible_fallback_rounding_difference_count",
        "cdf_gate_pass_set_rounding_difference_count",
        "cdf_eligible_gate_pass_set_rounding_difference_count",
        "cdf_saturated_zero_total",
        "cdf_saturated_one_total",
    ):
        value = audit.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ProjectionError(f"{label}: malformed gate diagnostic {field}")
    for field in ("minimum_abs_z_minus_q", "maximum_abs_standardized_feasibility"):
        value = audit.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ProjectionError(f"{label}: malformed gate diagnostic {field}")
    if result.get("recommendation_made") is True:
        expected_index = audit.get("terminal_recommendation_expected_index")
        saved_index = audit.get("terminal_recommendation_saved_index")
        if (
            audit.get("terminal_recommendation_status") != "checked"
            or audit.get("terminal_recommendation_index_difference_count") != 0
            or not isinstance(expected_index, int)
            or isinstance(expected_index, bool)
            or not isinstance(saved_index, int)
            or isinstance(saved_index, bool)
            or expected_index != saved_index
            or not 0 <= expected_index < int(job["grid_n"]) ** 2
        ):
            raise ProjectionError(
                f"{label}: terminal recommendation audit is absent or differs"
            )
    elif result.get("recommendation_made") is False:
        if (
            audit.get("terminal_recommendation_status")
            != "not_applicable_stopped"
            or audit.get("terminal_recommendation_index_difference_count") is not None
            or audit.get("terminal_recommendation_expected_index") is not None
            or audit.get("terminal_recommendation_saved_index") is not None
        ):
            raise ProjectionError(
                f"{label}: stopped recommendation audit is not explicitly inapplicable"
            )
    else:
        raise ProjectionError(
            f"{label}: result lacks recommendation status for terminal audit"
        )


def _validate_result(result: Any, job: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(result, dict):
        raise ProjectionError(f"{label}: result is not an object")
    expected = {
        "policy": job["policy"],
        "seed": int(job["seed"]),
        "sim": job["sim"],
        "stratum": int(job["stratum"]),
        "gamma": float(job["gamma"]),
        "mode": job["mode"],
        "kap": float(job["kap"]),
        "budget": int(job["budget"]),
        "warmup": int(job["warmup"]),
        "r_k": int(job["r_k"]),
        "grid_n": int(job["grid_n"]),
        "noise": job["noise"],
        "empty_gate": job["empty_gate"],
        "protocol_scaffold": job["protocol_scaffold"],
        "region_step": float(job["region_step"]),
        "empty_gate_stop_after": int(job["empty_gate_stop_after"]),
        "exclude_repeats_during_expansion": bool(
            job["exclude_repeats_during_expansion"]
        ),
    }
    for field, value in expected.items():
        _require_equal(result.get(field), value, f"{label} result {field}")
    if job["policy"] == "cEI-tMSE-2to1":
        for field, value in SCHEDULE_POST_HOC_FIELDS.items():
            _require_equal(result.get(field), value, f"{label} result {field}")
    trajectory = result.get("traj")
    if bool(job.get("traj")) != (trajectory is not None):
        raise ProjectionError(f"{label}: trajectory presence differs from the job")
    if trajectory is not None:
        if not isinstance(trajectory, dict):
            raise ProjectionError(f"{label}: trajectory is not an object")
        for field in ("n", "du", "rpsel", "toxic"):
            values = trajectory.get(field)
            if not isinstance(values, list) or len(values) != 18:
                raise ProjectionError(f"{label}: trajectory {field!r} is incomplete")
    history = result.get("allocation_history")
    enrolled = result.get("n_enrolled")
    if (
        not isinstance(enrolled, int)
        or isinstance(enrolled, bool)
        or enrolled < 1
        or not isinstance(history, list)
        or len(history) != enrolled
    ):
        raise ProjectionError(f"{label}: allocation history/enrollment is malformed")
    initialization_size = (
        int(job["warmup"])
        if job["protocol_scaffold"] == "lhs_fixed"
        else int(job["r_k"])
    )
    if (
        enrolled < initialization_size
        or enrolled > int(job["budget"])
        or (enrolled - initialization_size) % int(job["r_k"])
    ):
        raise ProjectionError(f"{label}: enrollment is incompatible with its scaffold")
    if result.get("recommendation_made") is True:
        if result.get("stop_reason") is not None:
            raise ProjectionError(f"{label}: recommending row has a stop reason")
        if job["protocol_scaffold"] == "lhs_fixed" and enrolled != int(job["budget"]):
            raise ProjectionError(f"{label}: lhs-fixed trial did not enroll its full budget")
        for field in (
            "dose_units", "rpsel", "rec_d1", "rec_d2",
            "rec_true_eff", "rec_true_tox",
        ):
            value = result.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ProjectionError(f"{label}: {field} must be finite")
        if result.get("rec_unsafe") not in (0, 1, False, True):
            raise ProjectionError(f"{label}: rec_unsafe must be binary")
    elif result.get("recommendation_made") is False:
        if result.get("stop_reason") != "NO_FEASIBLE_DOSE":
            raise ProjectionError(f"{label}: no-recommendation row lacks its stop reason")
        for field in (
            "dose_units", "rpsel", "rec_unsafe", "rec_d1", "rec_d2",
            "rec_true_eff", "rec_true_tox",
        ):
            if result.get(field) is not None:
                raise ProjectionError(f"{label}: {field} must be null without recommendation")
    else:
        raise ProjectionError(f"{label}: recommendation_made must be boolean")


def _validate_formal_row(
    row: Any, job: Mapping[str, Any], *, label: str
) -> tuple[Any, ...]:
    if not isinstance(row, dict):
        raise ProjectionError(f"{label}: row is not an object")
    expected_key = tuple(preflight.execution_key(job))
    raw_key = row.get("formal_execution_key")
    if not isinstance(raw_key, list) or tuple(raw_key) != expected_key:
        raise ProjectionError(f"{label}: formal execution key differs from the manifest")
    if row.get("harness_id") != preflight.HARNESS_ID:
        raise ProjectionError(f"{label}: mixed harness id")
    if row.get("evaluator_id") != job["evaluator_id"]:
        raise ProjectionError(f"{label}: wrong evaluator id")
    if row.get("historical_wrapper_used") is not False:
        raise ProjectionError(f"{label}: historical wrapper provenance is prohibited")
    expected_row_fields = {
        "matrix_id": job["matrix_id"],
        "run_class": job["run_class"],
        "policy": job["policy"],
        "seed": int(job["seed"]),
        "stratum": int(job["stratum"]),
    }
    for field, expected in expected_row_fields.items():
        _require_equal(row.get(field), expected, f"{label} {field}")
    elapsed = row.get("elapsed_seconds_not_for_timing_inference")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ProjectionError(f"{label}: elapsed time is malformed")
    audit = row.get("gate_numerics_audit")
    result = row.get("result")
    _validate_result(result, job, label=label)
    _validate_gate_audit(audit, result, job, label=label)
    initialization_size = (
        int(job["warmup"])
        if job["protocol_scaffold"] == "lhs_fixed"
        else int(job["r_k"])
    )
    expected_acquisition_states = (
        (int(result["n_enrolled"]) - initialization_size) // int(job["r_k"])
        + int(result.get("stop_reason") is not None)
    )
    if (
        audit.get("acquisition_state_count") != expected_acquisition_states
        or audit.get("checked_state_count") != expected_acquisition_states + 1
    ):
        raise ProjectionError(f"{label}: gate-audit state count differs from enrollment")
    return expected_key


def _validate_formal_artifact(
    path: Path,
    *,
    kind: str,
    jobs: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
    exact_panel_sha256: str,
    exact_metadata_sha256: str,
    primary_sha256: str,
    primary_metadata_sha256: str,
    expected_sources: Mapping[str, str],
    runtime_identity: Mapping[str, str],
) -> FormalInput:
    if kind not in {"decision", "trajectory_replay"}:
        raise ValueError(f"unknown formal input kind {kind!r}")
    path = Path(path)
    expected_name = DECISION_NAME if kind == "decision" else TRAJECTORY_NAME
    if path.name != expected_name:
        raise ProjectionError(
            f"formal {kind} artifact must be named {expected_name!r}"
        )
    raw, payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ProjectionError(f"{path.name}: top level must be an object")
    status = DECISION_STATUS if kind == "decision" else TRAJECTORY_STATUS
    required_top = {
        "schema_version": 1,
        "status": status,
        "artifact_class": FORMAL_ARTIFACT_CLASS,
    }
    for field, expected in required_top.items():
        _require_equal(payload.get(field), expected, f"{path.name} {field}")
    design = payload.get("design")
    if not isinstance(design, dict):
        raise ProjectionError(f"{path.name}: missing design object")
    required_design = {
        "execution_kind": kind,
        "expected_rows": len(jobs),
        "canonical_sort_key": list(CANONICAL_SORT_KEY),
    }
    for field, expected in required_design.items():
        _require_equal(design.get(field), expected, f"{path.name} design {field}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ProjectionError(f"{path.name}: missing provenance object")
    required_provenance: dict[str, Any] = {
        "manifest_sha256": manifest_sha256,
        "exact_validation_panel_sha256": exact_panel_sha256,
        "exact_validation_metadata_sha256": exact_metadata_sha256,
        "historical_wrappers_used": False,
        "production_ckg_default_changed": False,
        "authorization": FORMAL_AUTHORIZATION,
        "runtime_identity": dict(runtime_identity),
    }
    if kind == "decision":
        required_provenance.update({
            "primary_raw_sha256": primary_sha256,
            "primary_raw_metadata_sha256": primary_metadata_sha256,
        })
    else:
        if "primary_raw_sha256" in provenance or "primary_raw_metadata_sha256" in provenance:
            raise ProjectionError(
                f"{path.name}: primary provenance is permitted only on the decision master"
            )
    for field, expected in required_provenance.items():
        _require_equal(provenance.get(field), expected, f"{path.name} provenance {field}")
    _require_equal(
        payload.get("source_sha256"), dict(expected_sources),
        f"{path.name} execution-source hashes",
    )
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        raise ProjectionError(f"{path.name}: missing execution environment")
    _require_equal(
        environment.get("python"), runtime_identity["python"],
        f"{path.name} environment python",
    )
    versions = environment.get("versions")
    if not isinstance(versions, dict):
        raise ProjectionError(f"{path.name}: missing environment versions")
    for library in ("numpy", "scipy", "torch", "gpytorch"):
        _require_equal(
            versions.get(library), runtime_identity[library],
            f"{path.name} environment {library}",
        )
    thread_environment = environment.get("thread_environment")
    thread_names = (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )
    if not isinstance(thread_environment, dict) or any(
        thread_environment.get(name) != "1" for name in thread_names
    ):
        raise ProjectionError(f"{path.name}: execution was not single-thread configured")
    if not isinstance(environment.get("privacy_note"), str):
        raise ProjectionError(f"{path.name}: environment lacks its privacy note")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(jobs):
        raise ProjectionError(f"{path.name}: wrong formal row count")
    rows_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for index, (row, job) in enumerate(zip(rows, jobs)):
        key = _validate_formal_row(row, job, label=f"{path.name} row {index}")
        if key in rows_by_key:
            raise ProjectionError(f"{path.name}: duplicate formal execution key")
        rows_by_key[key] = dict(row)

    sidecar_path = Path(str(path) + ".metadata.json")
    _, sidecar = _read_json(sidecar_path)
    if not isinstance(sidecar, dict):
        raise ProjectionError(f"{sidecar_path.name}: top level must be an object")
    required_sidecar: dict[str, Any] = {
        "schema_version": 1,
        "artifact": path.name,
        "artifact_sha256": _sha256_bytes(raw),
        "artifact_bytes": len(raw),
        "status": status,
        "row_count": len(jobs),
        "manifest_sha256": manifest_sha256,
        "exact_validation_panel_sha256": exact_panel_sha256,
        "exact_validation_metadata_sha256": exact_metadata_sha256,
        "source_sha256": dict(expected_sources),
        "runtime_identity": dict(runtime_identity),
    }
    if kind == "decision":
        required_sidecar.update({
            "primary_raw_sha256": primary_sha256,
            "primary_raw_metadata_sha256": primary_metadata_sha256,
        })
    elif (
        "primary_raw_sha256" in sidecar
        or "primary_raw_metadata_sha256" in sidecar
    ):
        raise ProjectionError(
            f"{sidecar_path.name}: primary hashes are decision-master-only fields"
        )
    for field, expected in required_sidecar.items():
        _require_equal(sidecar.get(field), expected, f"{sidecar_path.name} {field}")
    return FormalInput(
        path=path,
        payload=payload,
        rows_by_key=rows_by_key,
        artifact_sha256=_sha256_bytes(raw),
        metadata_sha256=_sha256_file(sidecar_path),
    )


def _parse_checksum_manifest(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        pieces = raw_line.split(maxsplit=1)
        if len(pieces) != 2:
            raise ProjectionError(f"checksum manifest line {line_number} is malformed")
        digest, relative = pieces
        relative = relative.lstrip("*")
        if len(digest) != 64:
            raise ProjectionError(f"checksum manifest line {line_number} has a bad digest")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ProjectionError(
                f"checksum manifest line {line_number} has a bad digest"
            ) from exc
        if relative in output:
            raise ProjectionError(f"checksum manifest duplicates {relative!r}")
        output[relative] = digest.lower()
    return output


def _authenticate_frozen_records(
    records: Path, checksum_manifest: Path
) -> dict[str, str]:
    records = Path(records).resolve()
    checksums = _parse_checksum_manifest(checksum_manifest)
    authenticated: dict[str, str] = {}
    for filename in RECORD_FILENAMES:
        relative = f"records/{filename}"
        if relative not in checksums:
            raise ProjectionError(f"checksum manifest omits {relative}")
        path = records / filename
        observed = _sha256_file(path)
        if observed != checksums[relative]:
            raise ProjectionError(f"frozen record hash mismatch: {filename}")
        authenticated[filename] = observed
    # These read-only audits pin row counts, key sets, approved boundaries, and
    # all documented cross-file duplicate identities before projection.
    preflight.audit_archive_deep_equality(records, records)
    preflight.audit_frozen_reuse_identity(records)
    return authenticated


def _locator(job: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        job["matrix_id"], job["policy"], job["sim"], job["mode"],
        job["protocol_scaffold"], float(job["kap"]), float(job["gamma"]),
        int(job["stratum"]), int(job["seed"]),
    )


def _row_locator(
    matrix_id: str, policy: str, row: Mapping[str, Any], *,
    scaffold: str | None = None, kap: float | None = None,
) -> tuple[Any, ...]:
    return (
        matrix_id,
        policy,
        row.get("sim", "osa"),
        row.get("mode", "latent"),
        scaffold if scaffold is not None else row.get("protocol_scaffold", "lhs_fixed"),
        float(row.get("kap", 1.0) if kap is None else kap),
        float(row["gamma"]),
        int(row["stratum"]),
        int(row["seed"]),
    )


def _main_current_source(
    row: Mapping[str, Any], formal_policy: str,
) -> tuple[str, tuple[Any, ...]]:
    """Use the trajectory execution for its identical captured decision design."""
    replay_fulfills = bool(
        formal_policy in {"cEI", "cEI-tMSE"}
        and row.get("sim", "osa") == "osa"
        and row.get("mode", "latent") == "latent"
        and int(row["seed"]) < 80
        and math.isclose(float(row["gamma"]), 0.7)
        and math.isclose(float(row.get("kap", 1.0)), 1.0)
        and row.get("protocol_scaffold", "lhs_fixed") == "lhs_fixed"
    )
    if replay_fulfills:
        return "trajectory", _row_locator(
            "osa_tau07_current_nonckg_trajectory_replay", formal_policy, row,
            scaffold="lhs_fixed", kap=1.0,
        )
    return "decision", _row_locator(
        "main_latent_current_cei_controls", formal_policy, row,
        scaffold="lhs_fixed", kap=1.0,
    )


def _source_locator_for_target(
    filename: str, row: Mapping[str, Any]
) -> tuple[str, tuple[Any, ...]] | None:
    policy = row.get("policy")
    if filename in preflight._MAIN_RECORD_FILES:
        if policy == "cKG":
            matrix, formal_policy = "main_full_panel_exact_ckg", "cKG-exact-formal"
        elif policy in {"cEI", "cEI-tMSE"}:
            return _main_current_source(row, str(policy))
        else:
            return None
        return "decision", _row_locator(
            matrix, formal_policy, row,
            scaffold="lhs_fixed", kap=1.0,
        )
    if filename == "gate_sixway_supplemental.json.zst":
        mode = row.get("mode")
        if policy == "cKG1fix":
            matrix = (
                "main_full_panel_exact_ckg"
                if mode == "latent" else "sixway_predictive_exact_ckg"
            )
            return "decision", _row_locator(
                matrix, "cKG-exact-formal", row, scaffold="lhs_fixed", kap=1.0
            )
        aliases = {"GBE": "cEI-tMSE", "straddle": "tmse"}
        formal_policy = aliases.get(str(policy), str(policy))
        if mode == "predictive":
            return "decision", _row_locator(
                "sixway_predictive_current_nonckg", formal_policy, row,
                scaffold="lhs_fixed", kap=1.0,
            )
        if mode == "latent" and policy in {"cEI", "GBE"}:
            return _main_current_source(row, formal_policy)
        if mode == "latent" and policy in {"straddle", "qBIG", "random"}:
            return "decision", _row_locator(
                "sixway_latent_current_boundary_controls", formal_policy, row,
                scaffold="lhs_fixed", kap=1.0,
            )
        return None
    if filename == "efftox_mariposa_baselines_supplemental.json.zst":
        if policy == "cKG1fix":
            return "decision", _row_locator(
                "main_full_panel_exact_ckg", "cKG-exact-formal", row,
                scaffold="lhs_fixed", kap=1.0,
            )
        if policy in {"cEI", "GBE"}:
            return _main_current_source(
                row, "cEI-tMSE" if policy == "GBE" else "cEI"
            )
        if policy in {"straddle", "qBIG"}:
            return "decision", _row_locator(
                "baseline_current_tmse_qbig",
                "tmse" if policy == "straddle" else "qBIG",
                row, scaffold="lhs_fixed", kap=1.0,
            )
        return None
    if filename == "kappa_sweep_fixed_supplemental.json.zst":
        reference = math.isclose(float(row["kap"]), 1.0)
        if policy == "cKG1fix":
            matrix = (
                "main_full_panel_exact_ckg"
                if reference else "kappa_nonreference_exact_ckg"
            )
            formal_policy = "cKG-exact-formal"
        elif policy == "cEI":
            if reference:
                return _main_current_source(row, "cEI")
            matrix = "kappa_nonreference_current_cei"
            formal_policy = "cEI"
        else:
            return None
        return "decision", _row_locator(
            matrix, formal_policy, row,
            scaffold="lhs_fixed", kap=float(row["kap"]),
        )
    if filename == "osa_trajectory_supplemental.json.zst":
        if policy == "cKG1fix":
            return "decision", _row_locator(
                "main_full_panel_exact_ckg", "cKG-exact-formal", row,
                scaffold="lhs_fixed", kap=1.0,
            )
        formal_policy = "cEI-tMSE" if policy == "GBE" else str(policy)
        return "trajectory", _row_locator(
            "osa_tau07_current_nonckg_trajectory_replay", formal_policy, row,
            scaffold="lhs_fixed", kap=1.0,
        )
    if filename == "protocol_scaffold_sensitivity.json.zst":
        formal_policy = {
            "cKG": "cKG-exact-formal",
            "cEI": "cEI",
            "cEI-tMSE": "cEI-tMSE",
        }.get(str(policy))
        if formal_policy is None:
            return None
        scaffold = str(row["protocol_scaffold"])
        if scaffold == "lhs_fixed":
            if policy != "cKG":
                return _main_current_source(row, formal_policy)
            matrix = "main_full_panel_exact_ckg"
        elif scaffold == "start_low_expansion":
            matrix = (
                "gradual_access_exact_ckg"
                if policy == "cKG" else "gradual_access_current_cei_controls"
            )
        else:
            return None
        return "decision", _row_locator(
            matrix, formal_policy, row,
            scaffold=scaffold, kap=1.0,
        )
    if filename == "schedule_ratio_sensitivity_2to1.json.zst":
        if policy != "cEI-tMSE-2to1":
            return None
        return "decision", _row_locator(
            "schedule_ratio_2to1_current", "cEI-tMSE-2to1", row,
            scaffold="lhs_fixed", kap=1.0,
        )
    raise ProjectionError(f"unknown record target {filename!r}")


def _expected_usage(
    decision_jobs: Sequence[Mapping[str, Any]],
    trajectory_jobs: Sequence[Mapping[str, Any]],
) -> tuple[Counter[tuple[Any, ...]], Counter[tuple[Any, ...]]]:
    decision = Counter({tuple(preflight.execution_key(job)): 1 for job in decision_jobs})
    trajectory = Counter({tuple(preflight.execution_key(job)): 1 for job in trajectory_jobs})
    job_by_key = {
        tuple(preflight.execution_key(job)): job for job in decision_jobs
    }
    for job in decision_jobs:
        if job["matrix_id"] not in {
            "main_full_panel_exact_ckg",
            "main_latent_current_cei_controls",
        }:
            continue
        extra = 0
        if job["sim"] in {"osa", "gbump"} and int(job["seed"]) < 100:
            extra += 1  # latent main-policy row in gate_sixway
        if job["sim"] in {"efftox", "mariposa"} and math.isclose(
            float(job["gamma"]), 0.7
        ):
            extra += 1  # main-policy row in the two-testbed baseline supplement
        if (
            job["policy"] in {"cKG-exact-formal", "cEI"}
            and
            job["sim"] == "osa"
            and int(job["seed"]) < 100
            and float(job["gamma"]) in {0.5, 0.7, 0.9}
        ):
            extra += 1  # kap=1 reference in kappa sweep
        if job["sim"] == "osa" and float(job["gamma"]) in {0.7, 0.9}:
            extra += 1  # lhs_fixed main-policy row in protocol sensitivity
        if (
            job["policy"] == "cKG-exact-formal"
            and
            job["sim"] == "osa"
            and math.isclose(float(job["gamma"]), 0.7)
            and int(job["seed"]) < 80
        ):
            extra += 1  # captured cKG trajectory
        decision[tuple(preflight.execution_key(job))] += extra
    for job in trajectory_jobs:
        # Each captured run supplies its trajectory record plus the matching
        # main, latent-sixway, and lhs-protocol rows. cEI additionally supplies
        # the kap=1 reference row.
        extra = 3 + int(job["policy"] == "cEI")
        trajectory[tuple(preflight.execution_key(job))] += extra
    if sum(decision.values()) != 54_040 or sum(trajectory.values()) != 1_440:
        raise ProjectionError("formal source-appearance plan no longer totals 55,480")
    duplicate_reuse = sum(value - 1 for value in decision.values()) + sum(
        value - 1 for value in trajectory.values()
    )
    if duplicate_reuse != 11_280:
        raise ProjectionError("formal duplicate-reuse plan no longer totals 11,280")
    ckg_reuse = sum(
        value - 1
        for key, value in decision.items()
        if job_by_key[key]["run_class"] == "exact_ckg"
    )
    if ckg_reuse != 3_960:
        raise ProjectionError("exact-cKG duplicate-reuse plan no longer totals 3,960")
    ckg_appearances = sum(
        value
        for key, value in decision.items()
        if job_by_key[key]["run_class"] == "exact_ckg"
    )
    if ckg_appearances != 15_160:
        raise ProjectionError("exact-cKG source-appearance plan no longer totals 15,160")
    if sum(decision.values()) - ckg_appearances + sum(trajectory.values()) != 40_320:
        raise ProjectionError("non-cKG source-appearance plan no longer totals 40,320")
    return decision, trajectory


def _project_result(
    original: Mapping[str, Any], result: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    missing = set(original) - set(result) - {"policy"}
    if missing:
        raise ProjectionError(f"{label}: formal result lacks target fields {sorted(missing)}")
    filename = label.split(":", 1)[0]
    projected = {}
    for field in original:
        if field == "policy":
            projected[field] = original[field]
        elif field == "traj" and filename != "osa_trajectory_supplemental.json.zst":
            # A captured run can fulfill its matching decision identity, but
            # trajectory payloads belong only in the dedicated trajectory file.
            projected[field] = original[field]
        else:
            projected[field] = result[field]
    key_fields = preflight._RECORD_KEYS[filename]
    original_key = tuple(original[field] for field in key_fields)
    projected_key = tuple(projected[field] for field in key_fields)
    if projected_key != original_key:
        raise ProjectionError(f"{label}: projection changed the target record key")
    return projected


def _row_sequence_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        encoded = _canonical_json_bytes(row)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _candidate_commitments(
    candidate_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Bind every candidate record's exact canonical logical bytes."""
    if set(candidate_rows) != set(RECORD_FILENAMES):
        raise ProjectionError("candidate commitment does not cover exactly ten records")
    commitments: dict[str, dict[str, Any]] = {}
    for filename in RECORD_FILENAMES:
        logical = _canonical_json_bytes(candidate_rows[filename])
        commitments[filename] = {
            "rows": len(candidate_rows[filename]),
            "uncompressed_bytes": len(logical),
            "uncompressed_sha256": _sha256_bytes(logical),
        }
    return commitments


def _build_candidate_rows(
    *,
    frozen_records: Path,
    decision: FormalInput,
    trajectory: FormalInput,
    decision_jobs: Sequence[Mapping[str, Any]],
    trajectory_jobs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    decision_locator = {_locator(job): tuple(preflight.execution_key(job)) for job in decision_jobs}
    trajectory_locator = {
        _locator(job): tuple(preflight.execution_key(job)) for job in trajectory_jobs
    }
    if len(decision_locator) != len(decision_jobs) or len(trajectory_locator) != len(trajectory_jobs):
        raise ProjectionError("formal manifest contains duplicate source locators")
    observed_decision: Counter[tuple[Any, ...]] = Counter()
    observed_trajectory: Counter[tuple[Any, ...]] = Counter()
    candidate: dict[str, list[dict[str, Any]]] = {}
    file_summary: dict[str, Any] = {}
    total = approved = unchanged = 0
    for filename in RECORD_FILENAMES:
        original_rows = preflight._load_zstd_json(Path(frozen_records) / filename)
        revised_rows: list[dict[str, Any]] = []
        unchanged_original: list[dict[str, Any]] = []
        unchanged_revised: list[dict[str, Any]] = []
        approved_here = 0
        for position, original in enumerate(original_rows):
            reference = _source_locator_for_target(filename, original)
            allowed = preflight._replacement_allowed(filename, original)
            if (reference is not None) != allowed:
                raise ProjectionError(
                    f"{filename} row {position}: mapping and approved boundary disagree"
                )
            if reference is None:
                revised = dict(original)
                unchanged_original.append(dict(original))
                unchanged_revised.append(revised)
            else:
                source_kind, locator = reference
                locator_map = decision_locator if source_kind == "decision" else trajectory_locator
                row_map = decision.rows_by_key if source_kind == "decision" else trajectory.rows_by_key
                try:
                    execution = locator_map[locator]
                    formal_row = row_map[execution]
                except KeyError as exc:
                    raise ProjectionError(
                        f"{filename} row {position}: no authenticated formal source for {locator}"
                    ) from exc
                revised = _project_result(
                    original, formal_row["result"], label=f"{filename}:row {position}"
                )
                if source_kind == "decision":
                    observed_decision[execution] += 1
                else:
                    observed_trajectory[execution] += 1
                approved_here += 1
            revised_rows.append(revised)
        if unchanged_original != unchanged_revised:
            raise AssertionError(f"{filename}: unapproved rows changed in memory")
        before_hash = _row_sequence_hash(unchanged_original)
        after_hash = _row_sequence_hash(unchanged_revised)
        if before_hash != after_hash:
            raise AssertionError(f"{filename}: unapproved logical row bytes changed")
        candidate[filename] = revised_rows
        unchanged_here = len(original_rows) - approved_here
        file_summary[filename] = {
            "rows": len(original_rows),
            "approved_replacement_rows": approved_here,
            "unchanged_rows": unchanged_here,
            "unchanged_rows_sha256": before_hash,
        }
        total += len(original_rows)
        approved += approved_here
        unchanged += unchanged_here
    expected_decision, expected_trajectory = _expected_usage(decision_jobs, trajectory_jobs)
    if observed_decision != expected_decision:
        raise ProjectionError("decision-master projection identities/counts are incomplete")
    if observed_trajectory != expected_trajectory:
        raise ProjectionError("trajectory-replay projection identities/counts are incomplete")
    if (total, approved, unchanged) != (55_480, 55_480, 0):
        raise ProjectionError(
            f"projection boundary differs: total={total}, approved={approved}, unchanged={unchanged}"
        )
    return candidate, {
        "total_rows": total,
        "approved_replacement_rows": approved,
        "unchanged_rows": unchanged,
        "unique_decision_executions": len(observed_decision),
        "unique_trajectory_replay_executions": len(observed_trajectory),
        "decision_source_appearances": sum(observed_decision.values()),
        "trajectory_source_appearances": sum(observed_trajectory.values()),
        "exact_ckg_decision_source_appearances": sum(
            value
            for key, value in observed_decision.items()
            if decision.rows_by_key[key]["run_class"] == "exact_ckg"
        ),
        "current_nonckg_source_appearances": sum(
            value
            for key, value in observed_decision.items()
            if decision.rows_by_key[key]["run_class"] != "exact_ckg"
        ) + sum(observed_trajectory.values()),
        "duplicate_decision_reuse_appearances": sum(
            value - 1 for value in observed_decision.values()
        ),
        "duplicate_trajectory_reuse_appearances": sum(
            value - 1 for value in observed_trajectory.values()
        ),
        "duplicate_total_reuse_appearances": sum(
            value - 1 for value in observed_decision.values()
        ) + sum(value - 1 for value in observed_trajectory.values()),
        "duplicate_exact_ckg_reuse_appearances": sum(
            value - 1
            for key, value in observed_decision.items()
            if decision.rows_by_key[key]["run_class"] == "exact_ckg"
        ),
        "files": file_summary,
    }


def _authenticate_primary(
    *,
    primary_raw: Path,
    exact_panel: Path,
    frozen_records: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    exact_panel_sha256: str,
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    try:
        _, frozen_osa = primary_analysis._load_frozen_osa(
            Path(frozen_records) / "osa_main.json.zst"
        )
        _, primary_provenance = primary_analysis._load_primary(
            Path(primary_raw),
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            exact_validation_sha256=exact_panel_sha256,
            exact_validation_path=Path(exact_panel),
            frozen_osa_provenance=frozen_osa,
        )
    except primary_analysis.PrimaryGateError as exc:
        raise ProjectionError(f"primary artifact authentication failed: {exc}") from exc
    payload = json.loads(Path(primary_raw).read_text(encoding="utf-8"))
    rows = {
        (int(row["seed"]), int(row["stratum"])): dict(row)
        for row in payload["rows"]
    }
    if len(rows) != 400:
        raise ProjectionError("authenticated primary artifact does not contain 400 unique rows")
    return rows, primary_provenance


def _assert_primary_identity(
    decision: FormalInput,
    decision_jobs: Sequence[Mapping[str, Any]],
    primary_rows: Mapping[tuple[int, int], Mapping[str, Any]],
) -> None:
    matched = 0
    for job in decision_jobs:
        if not (
            job["matrix_id"] == "main_full_panel_exact_ckg"
            and job["sim"] == "osa"
            and math.isclose(float(job["gamma"]), 0.7)
        ):
            continue
        formal = decision.rows_by_key[tuple(preflight.execution_key(job))]
        primary = primary_rows[(int(job["seed"]), int(job["stratum"]))]
        for field in (
            "harness_id", "evaluator_id", "historical_wrapper_used",
            "gate_numerics_audit", "elapsed_seconds_not_for_timing_inference",
            "result",
        ):
            _require_equal(
                formal.get(field), primary.get(field),
                f"primary/formal byte-identity field {field} seed={job['seed']} stratum={job['stratum']}",
            )
        matched += 1
    if matched != 400:
        raise ProjectionError("primary/formal identity check did not cover exactly 400 cells")


def prepare_projection(
    *,
    manifest_path: Path,
    exact_panel: Path,
    primary_raw: Path,
    decision_master: Path,
    trajectory_replays: Path,
    frozen_records: Path,
    frozen_checksums: Path,
) -> PreparedProjection:
    manifest_path = Path(manifest_path).resolve()
    exact_panel = Path(exact_panel).resolve()
    primary_raw = Path(primary_raw).resolve()
    frozen_records = Path(frozen_records).resolve()
    manifest = preflight.load_manifest(manifest_path)
    preflight.audit_exact_validation_artifact(exact_panel)
    decision_jobs, trajectory_jobs = preflight.expand_execution_jobs(manifest)
    manifest_sha256 = _sha256_file(manifest_path)
    exact_panel_sha256 = _sha256_file(exact_panel)
    exact_metadata_path = Path(str(exact_panel) + ".metadata.json")
    exact_metadata_sha256 = _sha256_file(exact_metadata_path)
    runtime_identity = _runtime_identity(exact_panel)
    frozen_hashes = _authenticate_frozen_records(frozen_records, frozen_checksums)
    primary_rows, primary_provenance = _authenticate_primary(
        primary_raw=primary_raw,
        exact_panel=exact_panel,
        frozen_records=frozen_records,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        exact_panel_sha256=exact_panel_sha256,
    )
    primary_sha256 = _sha256_file(primary_raw)
    primary_metadata_path = Path(str(primary_raw) + ".metadata.json")
    primary_metadata_sha256 = _sha256_file(primary_metadata_path)
    expected_sources = _expected_source_hashes(manifest, manifest_path=manifest_path)
    decision = _validate_formal_artifact(
        decision_master,
        kind="decision",
        jobs=decision_jobs,
        manifest_sha256=manifest_sha256,
        exact_panel_sha256=exact_panel_sha256,
        exact_metadata_sha256=exact_metadata_sha256,
        primary_sha256=primary_sha256,
        primary_metadata_sha256=primary_metadata_sha256,
        expected_sources=expected_sources,
        runtime_identity=runtime_identity,
    )
    trajectory = _validate_formal_artifact(
        trajectory_replays,
        kind="trajectory_replay",
        jobs=trajectory_jobs,
        manifest_sha256=manifest_sha256,
        exact_panel_sha256=exact_panel_sha256,
        exact_metadata_sha256=exact_metadata_sha256,
        primary_sha256=primary_sha256,
        primary_metadata_sha256=primary_metadata_sha256,
        expected_sources=expected_sources,
        runtime_identity=runtime_identity,
    )
    _assert_primary_identity(decision, decision_jobs, primary_rows)
    candidate_rows, counts = _build_candidate_rows(
        frozen_records=frozen_records,
        decision=decision,
        trajectory=trajectory,
        decision_jobs=decision_jobs,
        trajectory_jobs=trajectory_jobs,
    )
    inputs = {
        "manifest": {"basename": manifest_path.name, "sha256": manifest_sha256},
        "exact_validation_panel": {
            "basename": exact_panel.name,
            "sha256": exact_panel_sha256,
            "metadata_sha256": exact_metadata_sha256,
        },
        "primary_raw": {
            "basename": primary_raw.name,
            "sha256": primary_sha256,
            "metadata_sha256": primary_metadata_sha256,
            "authenticated_row_count": primary_provenance["row_count"],
        },
        "decision_master": {
            "basename": decision.path.name,
            "sha256": decision.artifact_sha256,
            "metadata_sha256": decision.metadata_sha256,
            "rows": len(decision.rows_by_key),
        },
        "trajectory_replays": {
            "basename": trajectory.path.name,
            "sha256": trajectory.artifact_sha256,
            "metadata_sha256": trajectory.metadata_sha256,
            "rows": len(trajectory.rows_by_key),
        },
        "frozen_checksum_manifest": {
            "basename": Path(frozen_checksums).name,
            "sha256": _sha256_file(frozen_checksums),
        },
        "frozen_records_compressed_sha256": dict(frozen_hashes),
        "formal_execution_source_sha256": dict(expected_sources),
        "projection_source_sha256": _sha256_file(Path(__file__)),
        "runtime_identity": dict(runtime_identity),
    }
    candidate_commitments = _candidate_commitments(candidate_rows)
    fingerprint = _sha256_bytes(_canonical_json_bytes({
        "inputs": inputs,
        "candidate_records": candidate_commitments,
    }))
    metadata = {
        "schema_version": 1,
        "status": PROJECTION_STATUS,
        "artifact_class": PROJECTION_ARTIFACT_CLASS,
        "projection_fingerprint": fingerprint,
        "created_at_utc": None,
        "inputs": inputs,
        "candidate_records_commitment": candidate_commitments,
        "counts": counts,
        "validation": {
            "formal_rows_canonically_ordered_and_complete": True,
            "primary_400_byte_identity_fields_match": True,
            "frozen_archive_read_only": True,
            "unapproved_row_count_zero": True,
            "all_target_rows_have_authenticated_formal_sources": True,
            "candidate_record_key_sets_unchanged": True,
            "cross_file_reuse_identities_verified": False,
            "candidate_outputs_committed": False,
        },
        "outputs": {},
        "publication_scope": (
            "new candidate record staging only; frozen and canonical archives unchanged"
        ),
    }
    return PreparedProjection(candidate_rows, metadata, fingerprint, frozen_hashes)


def _write_durable(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _authorization_is_valid(*, write: bool, token: str | None) -> bool:
    return not write or token == FORMAL_AUTHORIZATION


def commit_projection(
    prepared: PreparedProjection,
    *,
    frozen_records: Path,
    output_parent: Path,
    authorization_token: str,
) -> Path:
    if not _authorization_is_valid(write=True, token=authorization_token):
        raise ProjectionError(
            "candidate projection commit requires FULL_FORMAL_RUN_GO authorization"
        )
    frozen_records = Path(frozen_records).resolve()
    output_parent = Path(output_parent).resolve()
    final = output_parent / f"formal_projection-{prepared.fingerprint}"
    candidate_records = final / "records"
    observed_commitments = _candidate_commitments(prepared.candidate_rows)
    recorded_commitments = prepared.metadata.get("candidate_records_commitment")
    if observed_commitments != recorded_commitments:
        raise ProjectionError("candidate rows changed after authenticated preparation")
    expected_fingerprint = _sha256_bytes(_canonical_json_bytes({
        "inputs": prepared.metadata.get("inputs", {}),
        "candidate_records": recorded_commitments,
    }))
    if prepared.fingerprint != expected_fingerprint or prepared.metadata.get(
        "projection_fingerprint"
    ) != expected_fingerprint:
        raise ProjectionError("prepared projection fingerprint is not bound to its inputs")
    if len(prepared.fingerprint) != 64:
        raise ProjectionError("prepared projection fingerprint is malformed")
    if set(prepared.candidate_rows) != set(RECORD_FILENAMES):
        raise ProjectionError("prepared projection does not contain exactly ten record targets")
    if final.exists():
        raise ProjectionError(f"hash-addressed projection already exists: {final}")
    if candidate_records == frozen_records:
        raise ProjectionError("candidate and frozen record directories resolve to the same path")
    frozen_archive_root = frozen_records.parent
    if output_parent == frozen_archive_root or output_parent.is_relative_to(
        frozen_archive_root
    ):
        raise ProjectionError("output parent is inside the frozen archive")
    if preflight.is_quarantined_formal_staging(output_parent) or (
        preflight.is_quarantined_formal_staging(final)
    ):
        raise ProjectionError(
            "output parent is inside provenance-only historical formal staging"
        )
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=output_parent))
    try:
        records_dir = temporary / "records"
        records_dir.mkdir()
        outputs: dict[str, Any] = {}
        compressor = zstd.ZstdCompressor(level=10, threads=0, write_checksum=True)
        for filename in RECORD_FILENAMES:
            destination = records_dir / filename
            logical = _canonical_json_bytes(prepared.candidate_rows[filename])
            _write_durable(destination, compressor.compress(logical))
            outputs[filename] = {
                "rows": len(prepared.candidate_rows[filename]),
                "compressed_bytes": destination.stat().st_size,
                "compressed_sha256": _sha256_file(destination),
                "uncompressed_bytes": len(logical),
                "uncompressed_sha256": _sha256_bytes(logical),
            }
            if outputs[filename] != {
                **recorded_commitments[filename],
                "compressed_bytes": destination.stat().st_size,
                "compressed_sha256": _sha256_file(destination),
            }:
                raise ProjectionError(
                    f"materialized candidate differs from its commitment: {filename}"
                )
        # Validate the materialized records before the directory becomes visible.
        archive_summary = preflight.audit_archive_deep_equality(
            frozen_records, records_dir
        )
        reuse_summary = preflight.audit_frozen_reuse_identity(records_dir)
        _require_equal(
            archive_summary,
            {
                "total_rows": 55_480,
                "approved_replacement_rows": 55_480,
                "unchanged_rows": 0,
            },
            "materialized archive boundary",
        )
        _require_equal(
            reuse_summary["total_reuse_appearances"], 11_280,
            "materialized reuse appearances",
        )
        _require_equal(
            reuse_summary["ckg_reuse_appearances"], 3_960,
            "materialized cKG reuse appearances",
        )
        for filename, original_hash in prepared.frozen_hashes.items():
            if _sha256_file(frozen_records / filename) != original_hash:
                raise ProjectionError(f"frozen input changed during projection: {filename}")
        metadata = json.loads(json.dumps(prepared.metadata))
        metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["outputs"] = outputs
        metadata["validation"].update({
            "cross_file_reuse_identities_verified": True,
            "candidate_outputs_committed": True,
        })
        _write_durable(
            temporary / "formal_projection.metadata.json",
            _canonical_json_bytes(metadata, pretty=True),
        )
        _fsync_directory(records_dir)
        _fsync_directory(temporary)
        os.replace(temporary, final)
        _fsync_directory(output_parent)
        return final
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=preflight.MANIFEST_PATH)
    parser.add_argument("--exact-validation-panel", type=Path, required=True)
    parser.add_argument("--primary-raw", type=Path, required=True)
    parser.add_argument("--decision-master", type=Path, required=True)
    parser.add_argument("--trajectory-replays", type=Path, required=True)
    parser.add_argument("--frozen-records", type=Path, required=True)
    parser.add_argument("--frozen-checksums", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--authorization-token")
    args = parser.parse_args(argv)
    if bool(args.write) != bool(args.output_parent):
        parser.error("--write and --output-parent must be supplied together")
    if not _authorization_is_valid(
        write=bool(args.write), token=args.authorization_token
    ):
        parser.error("--write requires --authorization-token FULL_FORMAL_RUN_GO")
    prepared = prepare_projection(
        manifest_path=args.manifest,
        exact_panel=args.exact_validation_panel,
        primary_raw=args.primary_raw,
        decision_master=args.decision_master,
        trajectory_replays=args.trajectory_replays,
        frozen_records=args.frozen_records,
        frozen_checksums=args.frozen_checksums,
    )
    print(
        "[validated] decision_executions=43,880; trajectories=320; replacements=55,480; "
        f"unchanged=0; fingerprint={prepared.fingerprint}",
        flush=True,
    )
    if not args.write:
        print("[validated] no candidate files written", flush=True)
        return 0
    final = commit_projection(
        prepared,
        frozen_records=args.frozen_records,
        output_parent=args.output_parent,
        authorization_token=str(args.authorization_token),
    )
    print(f"[committed] {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
