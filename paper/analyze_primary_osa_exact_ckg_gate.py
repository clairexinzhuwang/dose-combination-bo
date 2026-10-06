#!/usr/bin/env python3
"""Analyze the completed 400-cell exact-cKG OSA primary gate.

This is a post-processing program.  It imports neither ``run_trial`` nor an
acquisition implementation and cannot execute a simulation.  Before computing
anything it requires a complete, canonically ordered primary artifact with
current-harness provenance, a passing exact-evaluator validation artifact, and
the exact frozen OSA reference record.

The two OSA strata are clustered within each of 200 paired trial seeds.  The
reported 95% t Monte Carlo intervals are pointwise and unadjusted; policy
differences are paired within seed and stratum.  No
equivalence margins were prespecified, so this analyzer deliberately does not
turn an interval containing zero into a numerical-stability PASS.

The source-path keys and checks in the original run contract are historical.
They require the original study archive and are not proof that the renamed
current package produced the saved results. Expected hashes are unchanged.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.stats import t as student_t

from paper import formal_rerun_preflight as preflight


EXPECTED_FROZEN_OSA_UNCOMPRESSED_SHA256 = (
    "62547e70a028fc4819b4acd57cec43c55d9e69195ecb0cac7eec2fb04138e9f8"
)
EXPECTED_FROZEN_OSA_COMPRESSED_SHA256 = (
    "204325858aba812f977757871d863b4babcad32f5b31fef266241bedf1b56afd"
)
PRIMARY_STATUS = "COMPLETE_PRIMARY_400_EXACT_CKG_GATE"
PRIMARY_ARTIFACT_CLASS = "formal_primary_gate_staging_not_archive_records"
ANALYSIS_STATUS = "DESCRIPTIVE_PRIMARY_OC_REVIEW_READY"
PRIMARY_MATRIX_ID = "primary_osa_tau07_exact_ckg"
EXACT_ARM = "exact cKG"
REFERENCE_ARMS = ("cKG-512", "cEI", "specified 1:1 cEI-tMSE")
ALL_ARMS = (EXACT_ARM, *REFERENCE_ARMS)
STRATA = (0, 1)
SEEDS = tuple(range(200))
EXPECTED_PRIMARY_ROWS = 400
EXPECTED_ADAPTIVE_STATES = 18
EXPECTED_GATE_STATES_WITH_TERMINAL = EXPECTED_ADAPTIVE_STATES + 1
PRIMARY_SOURCE_FILES = (
    "paper/run_primary_osa_exact_ckg_gate.py",
    "paper/formal_rerun_preflight.py",
    "paper/formal_rerun_manifest.json",
    "src/safedosebo/acquisitions.py",
    "src/safedosebo/ckg_exact.py",
    "src/safedosebo/context.py",
    "src/safedosebo/gp.py",
    "src/safedosebo/metrics.py",
    "src/safedosebo/protocol.py",
    "src/safedosebo/registry.py",
    "src/safedosebo/surfaces.py",
    "src/safedosebo/trial.py",
)


class PrimaryGateError(RuntimeError):
    """Raised when primary-gate provenance, completeness, or analysis fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, value: bytes) -> None:
    """Durably replace one artifact using a sibling temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json_payload(path: Path) -> tuple[bytes, Any]:
    path = Path(path)
    if path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:  # pragma: no cover - locked environment includes it
            raise PrimaryGateError("zstandard is required to read the frozen archive") from exc
        with path.open("rb") as stream:
            raw = zstd.ZstdDecompressor().stream_reader(stream).read()
    else:
        raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrimaryGateError(f"{path}: input is not valid JSON") from exc
    return raw, value


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise PrimaryGateError(f"{label} differs: observed {observed!r}, expected {expected!r}")


def _validate_result_metric_fields(row: Mapping[str, Any], *, label: str) -> None:
    made = row.get("recommendation_made", True)
    if not isinstance(made, (bool, np.bool_)):
        raise PrimaryGateError(f"{label}: recommendation_made must be boolean")
    toxic = row.get("toxic")
    if not isinstance(toxic, (int, np.integer)) or not 0 <= int(toxic) <= 40:
        raise PrimaryGateError(f"{label}: toxic must be an integer in [0, 40]")
    if made:
        for field in ("rec_true_eff", "dose_units"):
            value = row.get(field)
            if not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(
                float(value)
            ):
                raise PrimaryGateError(f"{label}: {field} must be finite when recommending")
        if float(row["dose_units"]) < 0.0:
            raise PrimaryGateError(f"{label}: dose_units cannot be negative")
        if row.get("rec_unsafe") not in (0, 1, False, True):
            raise PrimaryGateError(f"{label}: rec_unsafe must be binary when recommending")
    else:
        for field in ("rec_true_eff", "dose_units", "rec_unsafe"):
            if row.get(field) is not None:
                raise PrimaryGateError(f"{label}: {field} must be null without a recommendation")


def _load_frozen_osa(
    path: Path,
    *,
    expected_uncompressed_sha256: str = EXPECTED_FROZEN_OSA_UNCOMPRESSED_SHA256,
    expected_compressed_sha256: str = EXPECTED_FROZEN_OSA_COMPRESSED_SHA256,
) -> tuple[dict[str, dict[tuple[int, int], dict[str, Any]]], dict[str, Any]]:
    if Path(path).suffix != ".zst":
        raise PrimaryGateError("frozen OSA input must be the compressed osa_main.json.zst")
    compressed_sha256 = _sha256_file(path)
    if compressed_sha256 != expected_compressed_sha256:
        raise PrimaryGateError("frozen OSA compressed SHA-256 differs from the authenticated reference")
    raw, value = _read_json_payload(path)
    logical_sha256 = _sha256_bytes(raw)
    if logical_sha256 != expected_uncompressed_sha256:
        raise PrimaryGateError(
            "frozen OSA decompressed JSON SHA-256 differs from the authenticated reference"
        )
    if not isinstance(value, list) or len(value) != 6000:
        raise PrimaryGateError("frozen OSA record must contain exactly 6,000 rows")

    expected_policies = ("cEI", "cKG", "cEI-tMSE")
    expected_gammas = (0.5, 0.6, 0.7, 0.8, 0.9)
    index: dict[tuple[str, float, int, int], dict[str, Any]] = {}
    for position, raw_row in enumerate(value):
        if not isinstance(raw_row, dict):
            raise PrimaryGateError(f"frozen OSA row {position} is not an object")
        row = dict(raw_row)
        key = (
            str(row.get("policy")),
            float(row.get("gamma", float("nan"))),
            int(row.get("seed", -1)),
            int(row.get("stratum", -1)),
        )
        if key in index:
            raise PrimaryGateError(f"frozen OSA duplicate factorial key {key}")
        expected_design = {
            "sim": "osa",
            "mode": "latent",
            "budget": 40,
            "warmup": 4,
            "r_k": 2,
            "grid_n": 5,
            "noise": "fixed",
            "empty_gate": "pf",
        }
        for field, expected in expected_design.items():
            _require_equal(row.get(field), expected, f"frozen OSA row {position} {field}")
        _validate_result_metric_fields(row, label=f"frozen OSA row {position}")
        index[key] = row

    expected_keys = {
        (policy, gamma, seed, stratum)
        for policy in expected_policies
        for gamma in expected_gammas
        for seed in SEEDS
        for stratum in STRATA
    }
    if set(index) != expected_keys:
        missing = expected_keys - set(index)
        extra = set(index) - expected_keys
        raise PrimaryGateError(
            f"frozen OSA factorial differs: {len(missing)} missing, {len(extra)} extra"
        )

    by_arm: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    source_policy = {
        "cKG-512": "cKG",
        "cEI": "cEI",
        "specified 1:1 cEI-tMSE": "cEI-tMSE",
    }
    for arm, policy in source_policy.items():
        by_arm[arm] = {
            (seed, stratum): index[(policy, 0.7, seed, stratum)]
            for seed in SEEDS
            for stratum in STRATA
        }
    return by_arm, {
        "path_basename": Path(path).name,
        "compressed_sha256": compressed_sha256,
        "uncompressed_sha256": logical_sha256,
        "archive_rows": len(value),
        "selected_reference_rows": 1200,
    }


def _expected_execution_source_sha256(manifest_sha256: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in PRIMARY_SOURCE_FILES:
        if relative == "paper/formal_rerun_manifest.json":
            output[relative] = manifest_sha256
            continue
        source = ROOT / relative
        if not source.is_file():
            raise PrimaryGateError(f"manifest execution source is absent: {relative}")
        output[relative] = _sha256_file(source)
    return output


def _validate_gate_audit(audit: Any, *, label: str) -> None:
    if not isinstance(audit, dict):
        raise PrimaryGateError(f"{label}: missing gate-numerics audit")
    if audit.get("checked_state_count") != EXPECTED_GATE_STATES_WITH_TERMINAL:
        raise PrimaryGateError(
            f"{label}: exactly {EXPECTED_GATE_STATES_WITH_TERMINAL} gate states must be audited"
        )
    if audit.get("acquisition_state_count") != EXPECTED_ADAPTIVE_STATES:
        raise PrimaryGateError(
            f"{label}: exactly {EXPECTED_ADAPTIVE_STATES} acquisition states must be audited"
        )
    difference_fields = (
        "gate_pass_set_difference_count",
        "eligible_gate_pass_set_difference_count",
        "operational_full_fallback_index_difference_count",
        "operational_eligible_fallback_index_difference_count",
    )
    for field in difference_fields:
        if audit.get(field) != 0:
            raise PrimaryGateError(f"{label}: nonzero or missing {field}")
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
        if not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise PrimaryGateError(f"{label}: malformed diagnostic {field}")
    for field in ("minimum_abs_z_minus_q", "maximum_abs_standardized_feasibility"):
        value = audit.get(field)
        if not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(
            float(value)
        ) or float(value) < 0.0:
            raise PrimaryGateError(f"{label}: malformed diagnostic {field}")
    if audit.get("terminal_state_checked") is not True:
        raise PrimaryGateError(f"{label}: terminal recommendation gate was not audited")
    if audit.get("terminal_recommendation_status") != "checked":
        raise PrimaryGateError(f"{label}: terminal recommendation was not independently audited")
    if audit.get("terminal_recommendation_index_difference_count") != 0:
        raise PrimaryGateError(f"{label}: saved terminal recommendation differs from stable-z replay")
    expected_index = audit.get("terminal_recommendation_expected_index")
    saved_index = audit.get("terminal_recommendation_saved_index")
    if (
        not isinstance(expected_index, (int, np.integer))
        or not isinstance(saved_index, (int, np.integer))
        or not 0 <= int(expected_index) < 25
        or int(saved_index) != int(expected_index)
    ):
        raise PrimaryGateError(f"{label}: malformed terminal recommendation audit indices")


def _validate_trajectory(result: Mapping[str, Any], *, capture: bool, label: str) -> None:
    trajectory = result.get("traj")
    if not capture:
        if trajectory is not None:
            raise PrimaryGateError(f"{label}: unrequested trajectory payload is non-null")
        return
    if not isinstance(trajectory, dict):
        raise PrimaryGateError(f"{label}: required primary trajectory is absent")
    expected_n = list(range(4, 40, 2))
    if trajectory.get("n") != expected_n:
        raise PrimaryGateError(f"{label}: trajectory cohort labels are incomplete")
    for field in ("du", "rpsel", "toxic"):
        values = trajectory.get(field)
        if not isinstance(values, list) or len(values) != EXPECTED_ADAPTIVE_STATES:
            raise PrimaryGateError(f"{label}: trajectory field {field!r} is incomplete")


def _validate_primary_result(
    result: Any, expected_job: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise PrimaryGateError(f"{label}: result is not an object")
    expected_fields = {
        "policy": expected_job["policy"],
        "seed": int(expected_job["seed"]),
        "sim": expected_job["sim"],
        "stratum": int(expected_job["stratum"]),
        "gamma": float(expected_job["gamma"]),
        "mode": expected_job["mode"],
        "budget": int(expected_job["budget"]),
        "warmup": int(expected_job["warmup"]),
        "r_k": int(expected_job["r_k"]),
        "grid_n": int(expected_job["grid_n"]),
        "noise": expected_job["noise"],
        "kap": float(expected_job["kap"]),
        "empty_gate": expected_job["empty_gate"],
        "protocol_scaffold": expected_job["protocol_scaffold"],
        "region_step": float(expected_job["region_step"]),
        "empty_gate_stop_after": int(expected_job["empty_gate_stop_after"]),
        "exclude_repeats_during_expansion": bool(
            expected_job["exclude_repeats_during_expansion"]
        ),
    }
    for field, expected in expected_fields.items():
        _require_equal(result.get(field), expected, f"{label} result {field}")
    if result.get("recommendation_made") is not True:
        raise PrimaryGateError(f"{label}: lhs_fixed full-panel trial must recommend")
    if result.get("stop_reason") is not None:
        raise PrimaryGateError(f"{label}: lhs_fixed full-panel trial cannot stop early")
    if result.get("n_enrolled") != 40:
        raise PrimaryGateError(f"{label}: full-panel trial must enroll exactly 40")
    _validate_result_metric_fields(result, label=label)
    _validate_trajectory(result, capture=bool(expected_job["traj"]), label=label)
    return dict(result)


def validate_primary_payload(
    payload: Any,
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    exact_validation_sha256: str,
    exact_validation_basename: str,
    exact_validation_metadata_sha256: str,
    frozen_osa_provenance: Mapping[str, Any],
    exact_validation_runtime_identity: Mapping[str, str] | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Validate completeness, order, hashes, harness, and every primary row."""
    if not isinstance(payload, dict):
        raise PrimaryGateError("primary artifact must be a JSON object")
    required_top = {
        "schema_version": 1,
        "status": PRIMARY_STATUS,
        "artifact_class": PRIMARY_ARTIFACT_CLASS,
    }
    for field, expected in required_top.items():
        _require_equal(payload.get(field), expected, f"primary artifact {field}")
    design = payload.get("design")
    if not isinstance(design, dict):
        raise PrimaryGateError("primary artifact lacks a design object")
    required_design = {
        "matrix_id": PRIMARY_MATRIX_ID,
        "policy": "cKG-exact-formal",
        "surface": "osa",
        "tau": 0.7,
        "gate_mode": "latent",
        "protocol_scaffold": "lhs_fixed",
        "noise": "fixed",
        "kap": 1.0,
        "budget": 40,
        "warmup": 4,
        "cohort_size": 2,
        "grid_n": 5,
        "empty_gate": "pf",
        "seeds": list(SEEDS),
        "strata": list(STRATA),
        "expected_cells": EXPECTED_PRIMARY_ROWS,
        "trajectory_capture": "seeds 0..79 in both strata; 160 rows in place",
    }
    for field, expected in required_design.items():
        _require_equal(design.get(field), expected, f"primary design {field}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise PrimaryGateError("primary artifact lacks a provenance object")
    required_provenance = {
        "manifest_sha256": manifest_sha256,
        "exact_validation_panel": exact_validation_basename,
        "exact_validation_panel_sha256": exact_validation_sha256,
        "exact_validation_metadata_sha256": exact_validation_metadata_sha256,
        "frozen_osa": dict(frozen_osa_provenance),
        "authorization": "PRIMARY_RUN_GO",
        "historical_wrappers_used": False,
        "production_ckg_default_changed": False,
    }
    for field, expected in required_provenance.items():
        _require_equal(provenance.get(field), expected, f"primary provenance {field}")
    if exact_validation_runtime_identity is not None:
        _require_equal(
            provenance.get("runtime_identity"),
            dict(exact_validation_runtime_identity),
            "primary provenance exact-validation runtime identity",
        )
    environment = payload.get("environment")
    if not isinstance(environment, dict) or not environment:
        raise PrimaryGateError("primary artifact lacks execution-environment provenance")
    if exact_validation_runtime_identity is not None:
        if environment.get("python") != exact_validation_runtime_identity.get("python"):
            raise PrimaryGateError(
                "primary environment differs from its validated Python runtime"
            )
        versions = environment.get("versions")
        if not isinstance(versions, dict) or any(
            versions.get(name) != exact_validation_runtime_identity.get(name)
            for name in ("numpy", "scipy", "torch", "gpytorch")
        ):
            raise PrimaryGateError(
                "primary environment differs from its validated library runtime"
            )
        thread_environment = environment.get("thread_environment")
        thread_names = (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        if not isinstance(thread_environment, dict) or any(
            thread_environment.get(name) != "1" for name in thread_names
        ):
            raise PrimaryGateError(
                "primary environment is not the single-thread execution contract"
            )
        if not isinstance(environment.get("privacy_note"), str):
            raise PrimaryGateError(
                "primary environment lacks its privacy-safe provenance note"
            )
    checkpoint_fingerprint = provenance.get("checkpoint_fingerprint")
    if not isinstance(checkpoint_fingerprint, str) or len(checkpoint_fingerprint) != 64:
        raise PrimaryGateError("primary provenance checkpoint fingerprint is malformed")
    try:
        int(checkpoint_fingerprint, 16)
    except ValueError as exc:
        raise PrimaryGateError("primary provenance checkpoint fingerprint is malformed") from exc
    scope = design.get("scope")
    if not isinstance(scope, str) or "does not authorize" not in scope or "44,200" not in scope:
        raise PrimaryGateError("primary design scope does not retain the primary-only boundary")
    expected_sources = _expected_execution_source_sha256(manifest_sha256)
    _require_equal(
        payload.get("source_sha256"), expected_sources, "primary execution-source hashes"
    )

    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_PRIMARY_ROWS:
        raise PrimaryGateError("primary artifact must contain exactly 400 rows")
    jobs = preflight.primary_jobs(manifest)
    expected_keys = [(int(job["seed"]), int(job["stratum"])) for job in jobs]
    observed_keys = [
        (int(row.get("seed", -1)), int(row.get("stratum", -1)))
        if isinstance(row, dict)
        else None
        for row in rows
    ]
    if observed_keys != expected_keys:
        if len(set(observed_keys)) != len(observed_keys):
            raise PrimaryGateError("primary rows contain a duplicate seed-stratum key")
        raise PrimaryGateError("primary rows are incomplete, unexpected, or not canonically ordered")

    output: dict[tuple[int, int], dict[str, Any]] = {}
    for position, (row, job) in enumerate(zip(rows, jobs)):
        label = f"primary row {position} seed={job['seed']} stratum={job['stratum']}"
        if row.get("policy") != "cKG-exact-formal":
            raise PrimaryGateError(f"{label}: wrong primary policy")
        if row.get("harness_id") != preflight.HARNESS_ID:
            raise PrimaryGateError(f"{label}: mixed harness id")
        if row.get("evaluator_id") != preflight.EXACT_EVALUATOR_ID:
            raise PrimaryGateError(f"{label}: mixed evaluator id")
        if row.get("historical_wrapper_used") is not False:
            raise PrimaryGateError(f"{label}: historical wrapper provenance is prohibited")
        elapsed = row.get("elapsed_seconds_not_for_timing_inference")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or float(elapsed) < 0:
            raise PrimaryGateError(f"{label}: elapsed time must be finite and nonnegative")
        _validate_gate_audit(row.get("gate_numerics_audit"), label=label)
        result = _validate_primary_result(row.get("result"), job, label=label)
        key = (int(job["seed"]), int(job["stratum"]))
        if key in output:
            raise PrimaryGateError(f"duplicate primary seed-stratum cell {key}")
        output[key] = result
    expected_cells = {(seed, stratum) for seed in SEEDS for stratum in STRATA}
    if set(output) != expected_cells:
        raise PrimaryGateError("primary output does not cover the 200 x 2 seed-stratum grid")
    return output


def _load_primary(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    exact_validation_sha256: str,
    exact_validation_path: Path,
    frozen_osa_provenance: Mapping[str, Any],
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrimaryGateError("primary artifact is not valid JSON") from exc
    sidecar_path = Path(str(path) + ".metadata.json")
    if not sidecar_path.is_file():
        raise PrimaryGateError(f"primary artifact metadata is missing: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    exact_validation_metadata_sha256 = _sha256_file(
        Path(str(exact_validation_path) + ".metadata.json")
    )
    exact_panel = json.loads(exact_validation_path.read_text(encoding="utf-8"))
    exact_environment = exact_panel.get("environment", {})
    exact_runtime_identity = {
        key: exact_environment.get(key)
        for key in ("python", "numpy", "scipy", "torch", "gpytorch")
    }
    if any(not isinstance(value, str) or not value for value in exact_runtime_identity.values()):
        raise PrimaryGateError("exact-validation runtime identity is incomplete")
    required_sidecar = {
        "schema_version": 1,
        "artifact": path.name,
        "artifact_sha256": _sha256_bytes(raw),
        "artifact_bytes": len(raw),
        "status": PRIMARY_STATUS,
        "row_count": EXPECTED_PRIMARY_ROWS,
        "trajectory_row_count": 160,
        "manifest_sha256": manifest_sha256,
        "exact_validation_panel_sha256": exact_validation_sha256,
        "runtime_identity": exact_runtime_identity,
        "publication_scope": "staging primary gate only; frozen archive unchanged",
    }
    for field, expected in required_sidecar.items():
        _require_equal(sidecar.get(field), expected, f"primary sidecar {field}")
    expected_sources = _expected_execution_source_sha256(manifest_sha256)
    _require_equal(sidecar.get("source_sha256"), expected_sources, "primary sidecar source hashes")
    _require_equal(sidecar.get("source_sha256"), payload.get("source_sha256"), "primary/sidecar source hashes")
    lookup = validate_primary_payload(
        payload,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        exact_validation_sha256=exact_validation_sha256,
        exact_validation_basename=exact_validation_path.name,
        exact_validation_metadata_sha256=exact_validation_metadata_sha256,
        frozen_osa_provenance=frozen_osa_provenance,
        exact_validation_runtime_identity=exact_runtime_identity,
    )
    audit_summary = {
        field: int(
            sum(row["gate_numerics_audit"][field] for row in payload["rows"])
        )
        for field in (
            "gate_pass_set_difference_count",
            "eligible_gate_pass_set_difference_count",
            "operational_full_fallback_index_difference_count",
            "operational_eligible_fallback_index_difference_count",
            "inactive_full_fallback_rounding_difference_count",
            "inactive_eligible_fallback_rounding_difference_count",
            "operational_full_fallback_rounding_difference_count",
            "operational_eligible_fallback_rounding_difference_count",
            "cdf_gate_pass_set_rounding_difference_count",
            "cdf_eligible_gate_pass_set_rounding_difference_count",
            "cdf_saturated_zero_total",
            "cdf_saturated_one_total",
        )
    }
    return lookup, {
        "path_basename": path.name,
        "file_sha256": _sha256_bytes(raw),
        "metadata_basename": sidecar_path.name,
        "metadata_sha256": _sha256_file(sidecar_path),
        "manifest_sha256": manifest_sha256,
        "exact_validation_sha256": exact_validation_sha256,
        "exact_validation_metadata_sha256": exact_validation_metadata_sha256,
        "source_sha256": expected_sources,
        "row_count": len(lookup),
        "trajectory_row_count": sum(
            row["result"].get("traj") is not None for row in payload["rows"]
        ),
        "gate_numerics_audit_summary": audit_summary,
    }


def _recommended_efficacy(row: Mapping[str, Any]) -> float | None:
    return float(row["rec_true_eff"]) if bool(row.get("recommendation_made", True)) else None


def _unsafe_recommendation(row: Mapping[str, Any]) -> float:
    return 100.0 * float(
        bool(row.get("recommendation_made", True)) and bool(row.get("rec_unsafe", 0))
    )


def _toxic_assignments(row: Mapping[str, Any]) -> float:
    return float(row["toxic"])


def _dose_units(row: Mapping[str, Any]) -> float | None:
    return float(row["dose_units"]) if bool(row.get("recommendation_made", True)) else None


def _recommendation_rate(row: Mapping[str, Any]) -> float:
    return 100.0 * float(bool(row.get("recommendation_made", True)))


METRICS: dict[str, dict[str, Any]] = {
    "rec_true_eff": {
        "label": "Recommended true efficacy",
        "unit": "OSA efficacy units",
        "denominator": "trials producing a recommendation",
        "contrast_denominator": "seed-stratum pairs where both policies recommend",
        "favorable_direction": "higher",
        "fn": _recommended_efficacy,
    },
    "rec_unsafe": {
        "label": "Above-threshold final recommendation",
        "unit": "percentage points",
        "denominator": "all trials; no recommendation is counted as zero",
        "contrast_denominator": "all paired seed-stratum trials",
        "favorable_direction": "lower",
        "fn": _unsafe_recommendation,
    },
    "toxic": {
        "label": "Above-boundary assignments",
        "unit": "assignments out of 40",
        "denominator": "all trials",
        "contrast_denominator": "all paired seed-stratum trials",
        "favorable_direction": "lower",
        "fn": _toxic_assignments,
    },
    "dose_units": {
        "label": "Dose-location error",
        "unit": "panel dose units",
        "denominator": "trials producing a recommendation",
        "contrast_denominator": "seed-stratum pairs where both policies recommend",
        "favorable_direction": "lower",
        "fn": _dose_units,
    },
    "recommendation_made": {
        "label": "Recommendation rate",
        "unit": "percentage points",
        "denominator": "all trials",
        "contrast_denominator": "all paired seed-stratum trials",
        "favorable_direction": "higher",
        "fn": _recommendation_rate,
    },
}


def _interval(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if array.size < 2 or not np.isfinite(array).all():
        raise PrimaryGateError("interval requires at least two finite seed-level values")
    mean = float(array.mean())
    mcse = float(array.std(ddof=1) / math.sqrt(array.size))
    critical = float(student_t.ppf(0.975, array.size - 1))
    return {
        "independent_seed_clusters": int(array.size),
        "estimate": mean,
        "mcse": mcse,
        "ci95_low": mean - critical * mcse,
        "ci95_high": mean + critical * mcse,
        "critical_t": critical,
    }


def _absolute_seed_values(
    lookup: Mapping[tuple[int, int], Mapping[str, Any]], metric: str
) -> tuple[list[float], int]:
    fn: Callable[[Mapping[str, Any]], float | None] = METRICS[metric]["fn"]
    values: list[float] = []
    eligible_rows = 0
    for seed in SEEDS:
        observed = [
            float(value)
            for stratum in STRATA
            if (value := fn(lookup[(seed, stratum)])) is not None
        ]
        eligible_rows += len(observed)
        if observed:
            values.append(float(statistics.fmean(observed)))
    return values, eligible_rows


def _paired_seed_values(
    left: Mapping[tuple[int, int], Mapping[str, Any]],
    right: Mapping[tuple[int, int], Mapping[str, Any]],
    metric: str,
) -> tuple[list[float], int]:
    fn: Callable[[Mapping[str, Any]], float | None] = METRICS[metric]["fn"]
    values: list[float] = []
    eligible_pairs = 0
    for seed in SEEDS:
        within: list[float] = []
        for stratum in STRATA:
            a = fn(left[(seed, stratum)])
            b = fn(right[(seed, stratum)])
            if a is None or b is None:
                continue
            within.append(float(a) - float(b))
        eligible_pairs += len(within)
        if within:
            values.append(float(statistics.fmean(within)))
    return values, eligible_pairs


def _build_analysis(
    exact: Mapping[tuple[int, int], Mapping[str, Any]],
    references: Mapping[str, Mapping[tuple[int, int], Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_arm = {EXACT_ARM: exact, **references}
    absolute: list[dict[str, Any]] = []
    for arm in ALL_ARMS:
        for metric, specification in METRICS.items():
            values, eligible = _absolute_seed_values(by_arm[arm], metric)
            absolute.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "label": specification["label"],
                    "unit": specification["unit"],
                    "denominator_definition": specification["denominator"],
                    "favorable_direction": specification["favorable_direction"],
                    "eligible_seed_stratum_rows": eligible,
                    **_interval(values),
                }
            )

    contrasts: list[dict[str, Any]] = []
    for comparator in REFERENCE_ARMS:
        for metric, specification in METRICS.items():
            values, eligible = _paired_seed_values(exact, references[comparator], metric)
            interval = _interval(values)
            contrasts.append(
                {
                    "contrast": f"{EXACT_ARM} minus {comparator}",
                    "left_arm": EXACT_ARM,
                    "right_arm": comparator,
                    "metric": metric,
                    "label": specification["label"],
                    "unit": specification["unit"],
                    "denominator_definition": specification["contrast_denominator"],
                    "favorable_direction_for_exact_minus_comparator": (
                        "positive"
                        if specification["favorable_direction"] == "higher"
                        else "negative"
                    ),
                    "eligible_seed_stratum_pairs": eligible,
                    **interval,
                    "pointwise_ci95_contains_zero": bool(
                        interval["ci95_low"] <= 0.0 <= interval["ci95_high"]
                    ),
                    "seed_level_paired_differences": values,
                }
            )
    return absolute, contrasts


def _find_row(
    rows: Sequence[Mapping[str, Any]], *, metric: str, arm: str | None = None,
    right_arm: str | None = None,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["metric"] == metric
        and (arm is None or row.get("arm") == arm)
        and (right_arm is None or row.get("right_arm") == right_arm)
    ]
    if len(matches) != 1:
        raise PrimaryGateError("analysis row lookup is not unique")
    return matches[0]


def _format_interval(row: Mapping[str, Any], digits: int = 3) -> str:
    return (
        f"{float(row['estimate']):.{digits}f} "
        f"[{float(row['ci95_low']):.{digits}f}, {float(row['ci95_high']):.{digits}f}]"
    )


def _render_markdown(summary: Mapping[str, Any]) -> str:
    absolute = summary["absolute_operating_characteristics"]
    contrasts = summary["paired_contrasts"]
    lines = [
        "# Exact-cKG OSA primary operating-characteristic review",
        "",
        "The authenticated 400-cell primary result is complete: 200 paired trial seeds",
        "and two OSA strata at tau = 0.7. The two strata are averaged within seed;",
        "trial seed is the independent Monte Carlo unit. Values are estimate [pointwise",
        "95% t Monte Carlo interval]; difference columns use paired seed-level",
        "contrasts. Intervals are unadjusted.",
        "",
        "| Metric | Exact cKG | Exact - cKG-512 | Exact - cEI | Exact - specified 1:1 cEI-tMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, specification in METRICS.items():
        cells = [_format_interval(_find_row(absolute, metric=metric, arm=EXACT_ARM))]
        for comparator in REFERENCE_ARMS:
            cells.append(
                _format_interval(_find_row(contrasts, metric=metric, right_arm=comparator))
            )
        lines.append(f"| {specification['label']} ({specification['unit']}) | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Inference limits",
            "",
            "The cKG-512 column is the direct numerical-change comparison. The cEI and",
            "specified 1:1 cEI-tMSE columns are contextual policy contrasts. For",
            "`rec_true_eff` and `dose_units`, a paired contrast is defined only where",
            "both policies recommend; the JSON records the eligible pair counts. For",
            "`rec_unsafe`, a trial without a recommendation would count as zero; `toxic`",
            "is the number of above-boundary assignments out of 40; recommendation rate",
            "uses all trials.",
            "",
            "No numerical or clinical equivalence margins were prespecified. Therefore,",
            "a pointwise interval containing zero is not evidence of equivalence or",
            "convergence, and this analysis does not automatically authorize the broader",
            "formal run. That decision requires scientific review of the estimates, MCSEs,",
            "and intervals in the JSON artifact.",
            "",
            "## Provenance gate",
            "",
            f"PASS: canonical 400-row coverage, current `{preflight.HARNESS_ID}` harness,",
            "exact evaluator, zero stable-z-versus-current-Context pass-set or",
            "operational-fallback differences, historical CDF rounding ties retained",
            "as diagnostics, terminal gate and recommendation audits, source hashes,",
            "manifest hash, exact-validation hash, and the",
            "authenticated frozen OSA reference were all checked before analysis.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    primary_path: Path,
    frozen_osa_path: Path,
    exact_validation_path: Path,
    output_dir: Path,
    *,
    manifest_path: Path = preflight.MANIFEST_PATH,
    expected_frozen_sha256: str = EXPECTED_FROZEN_OSA_UNCOMPRESSED_SHA256,
    expected_frozen_compressed_sha256: str = EXPECTED_FROZEN_OSA_COMPRESSED_SHA256,
) -> dict[str, Any]:
    """Validate inputs, compute paired OCs, and atomically write three artifacts."""
    primary_path = Path(primary_path)
    frozen_osa_path = Path(frozen_osa_path)
    exact_validation_path = Path(exact_validation_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)

    manifest = preflight.load_manifest(manifest_path)
    manifest_sha256 = _sha256_file(manifest_path)
    preflight.audit_exact_validation_artifact(exact_validation_path, require_sidecar=True)
    exact_validation_sha256 = _sha256_file(exact_validation_path)
    references, frozen_provenance = _load_frozen_osa(
        frozen_osa_path,
        expected_uncompressed_sha256=expected_frozen_sha256,
        expected_compressed_sha256=expected_frozen_compressed_sha256,
    )
    exact, primary_provenance = _load_primary(
        primary_path,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        exact_validation_sha256=exact_validation_sha256,
        exact_validation_path=exact_validation_path,
        frozen_osa_provenance=frozen_provenance,
    )
    absolute, contrasts = _build_analysis(exact, references)

    generated_at = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": ANALYSIS_STATUS,
        "artifact_class": "formal_primary_gate_analysis",
        "primary_run_authorization_verified": True,
        "broader_formal_run_authorized": False,
        "automatic_stability_verdict": "NOT_DEFINED_NO_PRESPECIFIED_EQUIVALENCE_MARGINS",
        "design": {
            "matrix_id": PRIMARY_MATRIX_ID,
            "sim": "osa",
            "tau": 0.7,
            "mode": "latent",
            "protocol_scaffold": "lhs_fixed",
            "trial_seed_count": 200,
            "strata": list(STRATA),
            "primary_rows": EXPECTED_PRIMARY_ROWS,
        },
        "estimand": {
            "metrics": list(METRICS),
            "pairing": "seed x stratum; average both stratum differences within trial seed",
            "independent_unit": "trial seed",
            "uncertainty": "pointwise unadjusted 95% t Monte Carlo intervals; policy differences are paired within seed x stratum before clustering by seed",
            "multiplicity_adjusted": False,
            "equivalence_margin_prespecified": False,
            "scope": "numerical change versus frozen cKG-512 and contextual comparisons versus frozen cEI and specified 1:1 cEI-tMSE at the OSA tau=0.7 primary cell only",
        },
        "provenance_gate": {
            "status": "PASS",
            "primary": primary_provenance,
            "frozen_osa": frozen_provenance,
            "exact_validation": {
                "path_basename": exact_validation_path.name,
                "file_sha256": exact_validation_sha256,
                "metadata_sha256": _sha256_file(
                    Path(str(exact_validation_path) + ".metadata.json")
                ),
            },
            "manifest": {
                "path_basename": manifest_path.name,
                "file_sha256": manifest_sha256,
            },
            "operational_gate_numerics_difference_count": 0,
            "inactive_fallback_rounding_difference_count": (
                primary_provenance["gate_numerics_audit_summary"][
                    "inactive_full_fallback_rounding_difference_count"
                ]
                + primary_provenance["gate_numerics_audit_summary"][
                    "inactive_eligible_fallback_rounding_difference_count"
                ]
            ),
            "terminal_state_checked_for_all_rows": True,
        },
        "absolute_operating_characteristics": absolute,
        "paired_contrasts": contrasts,
        "generated_at_utc": generated_at,
    }
    markdown = _render_markdown(summary)

    json_path = output_dir / "primary_osa_exact_ckg_gate.json"
    markdown_path = output_dir / "primary_osa_exact_ckg_gate.md"
    metadata_path = output_dir / "primary_osa_exact_ckg_gate.metadata.json"
    json_bytes = _canonical_json_bytes(summary)
    markdown_bytes = (markdown.rstrip() + "\n").encode("utf-8")
    # Metadata is the commit marker and is replaced last. A crash before that
    # leaves artifact hashes inconsistent with the old marker and is detectable.
    _atomic_write(json_path, json_bytes)
    _atomic_write(markdown_path, markdown_bytes)
    metadata = {
        "schema_version": 1,
        "status": ANALYSIS_STATUS,
        "artifact_class": "formal_primary_gate_analysis_metadata",
        "metadata_is_commit_marker": True,
        "artifacts": {
            "json": {"basename": json_path.name, "sha256": _sha256_bytes(json_bytes)},
            "markdown": {
                "basename": markdown_path.name,
                "sha256": _sha256_bytes(markdown_bytes),
            },
        },
        "inputs": {
            "primary": primary_provenance,
            "frozen_osa": frozen_provenance,
            "exact_validation_sha256": exact_validation_sha256,
            "manifest_sha256": manifest_sha256,
        },
        "analysis_source_sha256": _sha256_file(Path(__file__)),
        "preflight_source_sha256": _sha256_file(ROOT / "paper/formal_rerun_preflight.py"),
        "primary_run_authorization_verified": True,
        "broader_formal_run_authorized": False,
        "automatic_stability_verdict": "NOT_DEFINED_NO_PRESPECIFIED_EQUIVALENCE_MARGINS",
        "generated_at_utc": generated_at,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    _atomic_write(metadata_path, _canonical_json_bytes(metadata))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--frozen-osa", type=Path, required=True)
    parser.add_argument("--exact-validation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=preflight.MANIFEST_PATH)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = analyze(
        args.primary,
        args.frozen_osa,
        args.exact_validation,
        args.out_dir,
        manifest_path=args.manifest,
    )
    print(f"primary provenance gate: {summary['provenance_gate']['status']}")
    print("formal run authorization: NOT GRANTED BY THIS ANALYSIS")


if __name__ == "__main__":
    main()
