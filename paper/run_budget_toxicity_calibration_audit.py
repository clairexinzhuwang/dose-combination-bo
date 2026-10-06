#!/usr/bin/env python3
"""Authenticated Ray runner for the budget--toxicity calibration audit.

The default invocation is a read-only preflight.  Any write requires the
production authorization token.  Physical paths are executed in three
immutable seed blocks, with 24 shards per block (policy x assumed toxicity
noise SD x gate).  Each Ray task receives exactly one CPU, a ten-CPU Ray
runtime is mandatory, at most twenty tasks may be in flight, and there is no
serial fallback.

A completed shard is an immutable raw artifact.  Each seed block is a small
authenticated index over exactly 24 such shard artifacts and never duplicates
their path rows.  The cumulative raw master is committed only after the blinded precision
monitor has authenticated a valid top-up history whose final decision has
``next_M = null``.  This file never runs a simulation when imported.

The source-path keys and checks in the original run contract are historical.
They require the original study archive and are not proof that the renamed
current package produced the saved results. Expected hashes are unchanged.
"""
from __future__ import annotations

import os

for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gpytorch
import matplotlib
import numpy as np
import ray
import scipy
import torch
import zstandard as zstd


AUTHORIZATION_TOKEN = "BUDGET_TOXICITY_CALIBRATION_AUDIT_GO"
RAY_NUM_CPUS = 10
WORKER_NUM_CPUS = 1
MAX_IN_FLIGHT = 20
CHECKPOINT_EVERY = 25
SHARDS_PER_BLOCK = 24
PHYSICAL_PATHS_PER_REPLICATE = 48
SNAPSHOTS_PER_PATH = 3

MANIFEST_PATH = ROOT / "paper/budget_toxicity_calibration_manifest.json"
ANALYSIS_SPEC_PATH = ROOT / "paper/budget_toxicity_calibration_prespec.md"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/budget_toxicity_calibration_staging"
).resolve()

IMPLEMENTATION_SOURCE_PATHS = (
    "paper/budget_toxicity_calibration_core.py",
    "paper/run_budget_toxicity_calibration_audit.py",
    "paper/monitor_budget_toxicity_calibration_precision.py",
    "paper/analyze_budget_toxicity_calibration_audit.py",
)
FROZEN_FORMAL_SOURCE_PATHS = (
    "paper/analyze_primary_osa_exact_ckg_gate.py",
    "paper/formal_rerun_manifest.json",
    "paper/formal_rerun_preflight.py",
    "paper/run_full_formal_rerun.py",
    "paper/run_primary_osa_exact_ckg_gate.py",
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
if len(FROZEN_FORMAL_SOURCE_PATHS) != 14:
    raise RuntimeError("the frozen formal source contract must contain 14 files")

PENDING_STATUS = "DESIGN_FROZEN_IMPLEMENTATION_BINDING_PENDING"
FROZEN_STATUS = "FROZEN_BEFORE_NEW_AUDIT_EXECUTION"
EXECUTION_IDENTITY_SENTINEL = "TO_BE_HASH_BOUND_BEFORE_FIRST_NEW_AUDIT_PATH"

CHECKPOINT_STATUS = "PARTIAL_BUDGET_TOXICITY_CALIBRATION_NOT_FOR_ANALYSIS"
SHARD_STATUS = "COMPLETE_IMMUTABLE_BUDGET_TOXICITY_CALIBRATION_SHARD"
SHARD_COMMIT_STATUS = "COMMITTED_IMMUTABLE_BUDGET_TOXICITY_CALIBRATION_SHARD"
SHARD_ARTIFACT_CLASS = "authenticated_immutable_shard_raw_paths"
BLOCK_STATUS = "COMPLETE_IMMUTABLE_BUDGET_TOXICITY_CALIBRATION_SEED_BLOCK"
BLOCK_COMMIT_STATUS = (
    "COMMITTED_IMMUTABLE_BUDGET_TOXICITY_CALIBRATION_SEED_BLOCK"
)
BLOCK_ARTIFACT_CLASS = "authenticated_seed_block_index_over_immutable_shards"
MASTER_STATUS = "COMPLETE_BUDGET_TOXICITY_CALIBRATION_RAW_MASTER"
MASTER_COMMIT_STATUS = "COMMITTED_BUDGET_TOXICITY_CALIBRATION_RAW_MASTER"
MASTER_ARTIFACT_CLASS = "authenticated_budget_toxicity_calibration_raw_master"
MASTER_STORAGE_MODEL = "authenticated_index_over_seed_block_indexes"
FINAL_MASTER_BASENAME = "budget_toxicity_calibration_raw_master.json.zst"
BASELINE_BASENAME = "budget_toxicity_calibration_baseline_identity_M0200.json.zst"
BASELINE_PASS_STATUS = "PASS_BUDGET_TOXICITY_CALIBRATION_BASELINE_IDENTITY"

POLICIES = ("cKG-exact-formal", "tmse", "qBIG", "cEI")
NOISE_FACTORS = (0.5, 1.0, 2.0)
GAMMAS = (0.7, 0.9)
STRATA = (0, 1)
SNAPSHOT_BUDGETS = (20, 40, 80)
MAXIMUM_BUDGET = 80

BLOCK_IDS = ("M0200", "M0500_TOPUP", "M1000_TOPUP")
BLOCK_BY_ID = {
    "M0200": {
        "block_id": "M0200",
        "seed_start": 0,
        "seed_stop_exclusive": 200,
        "cumulative_M": 200,
    },
    "M0500_TOPUP": {
        "block_id": "M0500_TOPUP",
        "seed_start": 200,
        "seed_stop_exclusive": 500,
        "cumulative_M": 500,
    },
    "M1000_TOPUP": {
        "block_id": "M1000_TOPUP",
        "seed_start": 500,
        "seed_stop_exclusive": 1000,
        "cumulative_M": 1000,
    },
}
BLOCKS_FOR_M = {
    200: ("M0200",),
    500: ("M0200", "M0500_TOPUP"),
    1000: ("M0200", "M0500_TOPUP", "M1000_TOPUP"),
}
EXPECTED_BLOCK_PATHS = {
    block_id: (
        int(block["seed_stop_exclusive"]) - int(block["seed_start"])
    )
    * PHYSICAL_PATHS_PER_REPLICATE
    for block_id, block in BLOCK_BY_ID.items()
}
CANONICAL_SORT_KEY = (
    "block_id",
    "policy",
    "assumed_tox_noise_sd_factor",
    "gamma",
    "stratum",
    "seed",
)
BLOCK_INDEX_FIELDS = frozenset(
    {
        "block_id",
        "artifact",
        "artifact_sha256",
        "uncompressed_sha256",
        "uncompressed_bytes",
        "metadata",
        "metadata_sha256",
        "commit",
        "commit_sha256",
        "path_count",
        "snapshot_count",
        "shard_count",
    }
)
SHARD_INDEX_FIELDS = frozenset(
    {
        "shard_id",
        "block_id",
        "artifact",
        "artifact_sha256",
        "uncompressed_sha256",
        "uncompressed_bytes",
        "metadata",
        "metadata_sha256",
        "commit",
        "commit_sha256",
        "path_count",
        "snapshot_count",
        "execution_keys_sha256",
        "first_execution_key",
        "last_execution_key",
    }
)
SHARD_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "artifact_class",
        "shard",
        "bindings",
        "canonical_sort_key",
        "checkpoint_sha256",
        "path_count",
        "snapshot_count",
        "rows",
    }
)
CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "artifact_class",
        "shard_id",
        "seed_block_id",
        "shard_design_sha256",
        "expected_path_count",
        "completed_path_count",
        "bindings",
        "rows",
    }
)
SHARD_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "artifact_class",
        "artifact",
        "artifact_sha256",
        "artifact_bytes",
        "uncompressed_sha256",
        "uncompressed_bytes",
        "shard",
        "path_count",
        "snapshot_count",
        "checkpoint_sha256",
        "bindings",
        "compression",
        "environment",
        "generated_at_utc",
    }
)
SHARD_COMMIT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "artifact",
        "artifact_sha256",
        "uncompressed_sha256",
        "metadata",
        "metadata_sha256",
        "shard_id",
        "block_id",
        "path_count",
        "snapshot_count",
        "execution_keys_sha256",
        "launch_fingerprint",
        "immutable",
        "committed_at_utc",
    }
)
BLOCK_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "artifact_class",
        "storage_model",
        "block",
        "bindings",
        "canonical_sort_key",
        "shard_commits",
        "path_count",
        "snapshot_count",
    }
)
BLOCK_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "artifact_class",
        "storage_model",
        "artifact",
        "artifact_sha256",
        "artifact_bytes",
        "uncompressed_sha256",
        "uncompressed_bytes",
        "block_id",
        "seed_start",
        "seed_stop_exclusive",
        "path_count",
        "snapshot_count",
        "shard_count",
        "bindings",
        "compression",
        "environment",
        "generated_at_utc",
    }
)
BLOCK_COMMIT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "artifact",
        "artifact_sha256",
        "uncompressed_sha256",
        "metadata",
        "metadata_sha256",
        "block_id",
        "path_count",
        "snapshot_count",
        "shard_count",
        "launch_fingerprint",
        "immutable",
        "committed_at_utc",
    }
)
PRECISION_INDEX_FIELDS = frozenset(
    {
        "cumulative_M",
        "artifact",
        "artifact_sha256",
        "metadata",
        "metadata_sha256",
        "commit",
        "commit_sha256",
        "next_M",
    }
)
TOP_UP_HISTORY_FIELDS = frozenset(
    {
        "cumulative_M",
        "precision_artifact",
        "precision_artifact_sha256",
        "next_M",
    }
)
BASELINE_INDEX_FIELDS = frozenset(
    {
        "status",
        "cumulative_M",
        "pass",
        "artifact",
        "artifact_sha256",
        "metadata",
        "metadata_sha256",
        "commit",
        "commit_sha256",
        "checked_common_cells",
        "checked_allocation_histories",
        "unavailable_allocation_histories",
    }
)
MASTER_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "artifact_class",
        "storage_model",
        "final_M",
        "cumulative_block_ids",
        "path_count",
        "snapshot_count",
        "bindings",
        "baseline_identity",
        "block_commits",
        "precision_decisions",
        "top_up_history",
        "canonical_sort_key",
    }
)
MASTER_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "artifact_class",
        "storage_model",
        "artifact",
        "artifact_sha256",
        "artifact_bytes",
        "uncompressed_sha256",
        "uncompressed_bytes",
        "final_M",
        "path_count",
        "snapshot_count",
        "bindings",
        "compression",
        "environment",
        "generated_at_utc",
    }
)
MASTER_COMMIT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "artifact",
        "artifact_sha256",
        "uncompressed_sha256",
        "metadata",
        "metadata_sha256",
        "final_M",
        "path_count",
        "snapshot_count",
        "launch_fingerprint",
        "committed_at_utc",
    }
)

COMPRESSION_CONTRACT = {
    "format": "zstandard",
    "level": 19,
    "threads": 0,
    "checksum": True,
    "content_size": True,
    "dict_id": False,
}


class CalibrationRunError(RuntimeError):
    """Raised when design, provenance, execution, or artifact checks fail."""


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_bytes_write(path: Path, value: bytes) -> None:
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


def _atomic_json_write(path: Path, value: Any) -> None:
    _atomic_bytes_write(Path(path), _canonical_json_bytes(value))


def _compress_json(value: Any, *, level: int) -> tuple[bytes, bytes]:
    uncompressed = _canonical_json_bytes(value)
    compressed = zstd.ZstdCompressor(
        level=int(level),
        threads=0,
        write_checksum=True,
        write_content_size=True,
        write_dict_id=False,
    ).compress(uncompressed)
    return uncompressed, compressed


def _assert_finite_json(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise CalibrationRunError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_json(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_json(child, label=f"{label}[{index}]")


def _read_canonical_json_object(path: Path, label: str) -> dict[str, Any]:
    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token}")

    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw.decode("utf-8"), parse_constant=reject_nonfinite)
    except BaseException as error:
        raise CalibrationRunError(f"{label} is not readable finite JSON") from error
    if not isinstance(value, dict):
        raise CalibrationRunError(f"{label} is not a JSON object")
    _assert_finite_json(value, label=label)
    if raw != _canonical_json_bytes(value):
        raise CalibrationRunError(f"{label} is not canonical JSON")
    return value


def _read_zstd_json(path: Path, label: str) -> tuple[dict[str, Any], bytes, bytes]:
    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token}")

    try:
        compressed = Path(path).read_bytes()
        uncompressed = zstd.ZstdDecompressor().decompress(compressed)
        value = json.loads(
            uncompressed.decode("utf-8"), parse_constant=reject_nonfinite
        )
    except BaseException as error:
        raise CalibrationRunError(f"{label} is not a readable zstd JSON artifact") from error
    if not isinstance(value, dict):
        raise CalibrationRunError(f"{label} payload is not a JSON object")
    _assert_finite_json(value, label=label)
    if uncompressed != _canonical_json_bytes(value):
        raise CalibrationRunError(f"{label} payload is not canonical JSON")
    return value, compressed, uncompressed


def _runtime_identity() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "gpytorch": gpytorch.__version__,
        "matplotlib": matplotlib.__version__,
        "ray": str(ray.__version__),
        "zstandard": str(zstd.__version__),
    }


def _privacy_safe_environment() -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "ray_contract": {
            "num_cpus": RAY_NUM_CPUS,
            "worker_num_cpus": WORKER_NUM_CPUS,
            "maximum_in_flight": MAX_IN_FLIGHT,
            "serial_fallback": False,
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "privacy_note": (
            "No username, home path, hostname, serial number, or persistent "
            "device identifier is recorded."
        ),
    }


def _source_hashes(relative_paths: Sequence[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in sorted(relative_paths):
        path = ROOT / relative
        if not path.is_file():
            raise CalibrationRunError(f"execution source is absent: {relative}")
        output[relative] = _sha256_file(path)
    if len(output) != len(relative_paths):
        raise CalibrationRunError("execution source contract contains duplicates")
    return output


def _design_payload_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash the frozen design with only execution identity replaced by its sentinel."""
    value = json.loads(json.dumps(dict(manifest), allow_nan=False))
    value["execution_identity"] = EXECUTION_IDENTITY_SENTINEL
    return _sha256_bytes(_canonical_json_bytes(value))


def _expected_execution_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "analysis_spec_sha256": _sha256_file(ANALYSIS_SPEC_PATH),
        "implementation_source_sha256": _source_hashes(
            IMPLEMENTATION_SOURCE_PATHS
        ),
        "frozen_formal_source_sha256": _source_hashes(
            FROZEN_FORMAL_SOURCE_PATHS
        ),
        "runtime_identity": _runtime_identity(),
        "design_sha256": _design_payload_sha256(manifest),
    }


def _validate_manifest_design(manifest: Mapping[str, Any]) -> None:
    design = manifest.get("design")
    expected_design = {
        "surface": "osa",
        "strata": [0, 1],
        "policies": list(POLICIES),
        "policy_display": {
            "cKG-exact-formal": "cKG",
            "tmse": "tMSE-only",
            "qBIG": "Entropy reduction",
            "cEI": "cEI",
        },
        "assumed_toxicity_noise_sd_factors": list(NOISE_FACTORS),
        "assumed_toxicity_noise_variance_factors": [0.25, 1.0, 4.0],
        "truth_efficacy_noise_sd_factor": 1.0,
        "truth_toxicity_noise_sd_factor": 1.0,
        "assumed_efficacy_noise_sd_factor": 1.0,
        "gammas": list(GAMMAS),
        "snapshot_budgets": list(SNAPSHOT_BUDGETS),
        "maximum_physical_budget": MAXIMUM_BUDGET,
        "mode": "latent",
        "noise_fitting": "fixed_assumed_variance",
        "protocol_scaffold": "lhs_fixed",
        "warmup": 4,
        "cohort_size": 2,
        "grid_n": 5,
        "empty_gate": "pf",
        "stopping": False,
        "stable_gate_rule": "z_g > Phi^{-1}(tau)",
    }
    if design != expected_design:
        raise CalibrationRunError("calibration manifest design changed")
    blocks = manifest.get("seed_blocks")
    if blocks != [BLOCK_BY_ID[block_id] for block_id in BLOCK_IDS]:
        raise CalibrationRunError("calibration manifest seed blocks changed")
    accounting = manifest.get("accounting")
    expected_accounting = {
        "physical_paths_per_replicate": 48,
        "nested_snapshots_per_replicate": 144,
        "M0200_physical_paths": 9_600,
        "M0200_nested_snapshots": 28_800,
        "M0500_cumulative_physical_paths": 24_000,
        "M0500_cumulative_nested_snapshots": 72_000,
        "M1000_cumulative_physical_paths": 48_000,
        "M1000_cumulative_nested_snapshots": 144_000,
    }
    if accounting != expected_accounting:
        raise CalibrationRunError("calibration manifest accounting changed")
    required_runtime = manifest.get("required_runtime")
    expected_runtime = {
        "python": "3.11.6",
        "numpy": "1.26.4",
        "scipy": "1.16.3",
        "torch": "2.4.1",
        "gpytorch": "1.14",
        "matplotlib": "3.10.7",
        "ray": "2.10.0",
        "zstandard": "0.19.0",
        "ray_num_cpus": RAY_NUM_CPUS,
        "worker_num_cpus": WORKER_NUM_CPUS,
        "maximum_in_flight": MAX_IN_FLIGHT,
        "serial_fallback": False,
    }
    if required_runtime != expected_runtime:
        raise CalibrationRunError("calibration manifest runtime contract changed")
    if manifest.get("analysis_specification") != (
        "paper/budget_toxicity_calibration_prespec.md"
    ):
        raise CalibrationRunError("calibration manifest analysis-spec path changed")
    execution_paths = manifest.get("execution_source_paths")
    if execution_paths != list(IMPLEMENTATION_SOURCE_PATHS):
        raise CalibrationRunError("calibration manifest implementation-source list changed")


def _load_frozen_manifest(
    manifest_path: Path = MANIFEST_PATH,
    analysis_spec_path: Path = ANALYSIS_SPEC_PATH,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    analysis_spec_path = Path(analysis_spec_path)
    if not manifest_path.is_file() or not analysis_spec_path.is_file():
        raise CalibrationRunError("frozen calibration manifest or prespec is absent")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationRunError("frozen calibration manifest is unreadable") from error
    if manifest.get("schema_version") != 1 or manifest.get("status") != FROZEN_STATUS:
        raise CalibrationRunError(
            "calibration execution remains blocked until manifest status is frozen"
        )
    _validate_manifest_design(manifest)
    identity = manifest.get("execution_identity")
    if not isinstance(identity, Mapping):
        raise CalibrationRunError("calibration manifest lacks a bound execution identity")

    # The default paths are themselves part of the frozen design.  Custom copies
    # are allowed only if byte-identical, which makes tests and offline audits
    # possible without weakening production provenance.
    if _sha256_file(manifest_path) != _sha256_file(MANIFEST_PATH):
        raise CalibrationRunError("design manifest hash differs from frozen input")
    if _sha256_file(analysis_spec_path) != _sha256_file(ANALYSIS_SPEC_PATH):
        raise CalibrationRunError("analysis specification hash differs from frozen input")
    expected = _expected_execution_identity(manifest)
    if dict(identity) != expected:
        observed_runtime = _runtime_identity()
        if identity.get("runtime_identity") != observed_runtime:
            raise CalibrationRunError(
                "execution runtime differs from the frozen manifest: "
                f"observed={observed_runtime!r}, "
                f"required={identity.get('runtime_identity')!r}"
            )
        raise CalibrationRunError(
            "execution source, prespec, or design hash differs from the frozen manifest"
        )
    return manifest


def _build_bindings(
    manifest_path: Path = MANIFEST_PATH,
    analysis_spec_path: Path = ANALYSIS_SPEC_PATH,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    identity = manifest.get("execution_identity")
    if not isinstance(identity, Mapping):
        raise CalibrationRunError("cannot bind an unfrozen calibration manifest")
    implementation = dict(identity.get("implementation_source_sha256", {}))
    frozen_formal = dict(identity.get("frozen_formal_source_sha256", {}))
    combined_sources = {**frozen_formal, **implementation}
    if len(combined_sources) != len(frozen_formal) + len(implementation):
        raise CalibrationRunError("implementation and frozen source paths overlap")
    bindings: dict[str, Any] = {
        "manifest_sha256": _sha256_file(Path(manifest_path)),
        "analysis_spec_sha256": _sha256_file(Path(analysis_spec_path)),
        "design_sha256": identity.get("design_sha256"),
        "implementation_source_sha256": implementation,
        "frozen_formal_source_sha256": frozen_formal,
        "source_sha256": combined_sources,
        "runtime_identity": dict(identity.get("runtime_identity", {})),
        "ray_contract": {
            "num_cpus": RAY_NUM_CPUS,
            "worker_num_cpus": WORKER_NUM_CPUS,
            "maximum_in_flight": MAX_IN_FLIGHT,
            "serial_fallback": False,
        },
    }
    bindings["launch_fingerprint"] = _sha256_bytes(
        _canonical_json_bytes(bindings)
    )
    return bindings


def _job_key(job: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(job["block_id"]),
        str(job["policy"]),
        float(job["assumed_tox_noise_sd_factor"]),
        float(job["gamma"]),
        int(job["stratum"]),
        int(job["seed"]),
    )


def _job_sort_key(job: Mapping[str, Any]) -> tuple[Any, ...]:
    return _job_key(job)


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    raw = row.get("execution_key")
    if not isinstance(raw, list) or len(raw) != len(CANONICAL_SORT_KEY):
        raise CalibrationRunError("calibration row lacks its execution key")
    return tuple(raw)


def _expand_jobs(
    manifest: Mapping[str, Any], block_id: str | None = None
) -> list[dict[str, Any]]:
    _validate_manifest_design(manifest)
    requested = BLOCK_IDS if block_id is None else (str(block_id),)
    unknown = set(requested) - set(BLOCK_IDS)
    if unknown:
        raise CalibrationRunError(f"unknown calibration seed block: {sorted(unknown)}")
    jobs: list[dict[str, Any]] = []
    for current in requested:
        block = BLOCK_BY_ID[current]
        for policy in POLICIES:
            for factor in NOISE_FACTORS:
                for gamma in GAMMAS:
                    for stratum in STRATA:
                        for seed in range(
                            int(block["seed_start"]),
                            int(block["seed_stop_exclusive"]),
                        ):
                            jobs.append(
                                {
                                    "design_id": "budget_toxicity_calibration_v1",
                                    "block_id": current,
                                    "policy": policy,
                                    "seed": int(seed),
                                    "stratum": int(stratum),
                                    "gamma": float(gamma),
                                    "assumed_tox_noise_sd_factor": float(factor),
                                    "maximum_budget": MAXIMUM_BUDGET,
                                    "snapshot_budgets": list(SNAPSHOT_BUDGETS),
                                }
                            )
    jobs.sort(key=_job_sort_key)
    keys = [_job_key(job) for job in jobs]
    if len(keys) != len(set(keys)):
        raise CalibrationRunError("calibration design contains duplicate physical paths")
    expected = sum(EXPECTED_BLOCK_PATHS[current] for current in requested)
    if len(jobs) != expected:
        raise CalibrationRunError(
            f"calibration path count changed: {len(jobs)} != {expected}"
        )
    return jobs


def _shard_id(job: Mapping[str, Any]) -> str:
    return (
        f"{job['block_id']}__{job['policy']}__"
        f"csigma{float(job['assumed_tox_noise_sd_factor']):g}__"
        f"tau{float(job['gamma']):g}"
    )


def _group_jobs(
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        output.setdefault(_shard_id(job), []).append(dict(job))
    output = {
        shard: sorted(values, key=_job_sort_key)
        for shard, values in sorted(output.items())
    }
    blocks = {str(job["block_id"]) for job in jobs}
    if len(output) != SHARDS_PER_BLOCK * len(blocks):
        raise CalibrationRunError(
            "calibration design must contain exactly 24 shards per seed block"
        )
    for shard, values in output.items():
        if len({str(job["block_id"]) for job in values}) != 1:
            raise CalibrationRunError(f"calibration shard crosses seed blocks: {shard}")
    return output


def _load_core() -> Any:
    try:
        return importlib.import_module("paper.budget_toxicity_calibration_core")
    except BaseException as error:
        raise CalibrationRunError("calibration core cannot be imported") from error


def _validate_row(row: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if row.get("execution_key") != list(_job_key(job)):
        raise CalibrationRunError("calibration row execution key differs from its job")
    if row.get("block_id") != job.get("block_id"):
        raise CalibrationRunError("calibration row seed-block identifier changed")
    result = row.get("result")
    if not isinstance(result, Mapping):
        raise CalibrationRunError("calibration row lacks its physical-path result")
    expected = {
        "design_id": "budget_toxicity_calibration_v1",
        "policy": job["policy"],
        "seed": int(job["seed"]),
        "stratum": int(job["stratum"]),
        "gamma": float(job["gamma"]),
        "assumed_tox_noise_sd_factor": float(
            job["assumed_tox_noise_sd_factor"]
        ),
        "maximum_budget": MAXIMUM_BUDGET,
        "snapshot_budgets": list(SNAPSHOT_BUDGETS),
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise CalibrationRunError(
                f"calibration result has incompatible {field}: "
                f"{result.get(field)!r} != {value!r}"
            )
    try:
        _load_core().validate_path_record(dict(result))
    except BaseException as error:
        if isinstance(error, CalibrationRunError):
            raise
        raise CalibrationRunError("calibration core rejected a path record") from error


def _canonical_rows(
    rows: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    *,
    complete: bool,
) -> list[dict[str, Any]]:
    job_by_key = {_job_key(job): dict(job) for job in jobs}
    output: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        raw_key = row.get("execution_key")
        if not isinstance(raw_key, list):
            raise CalibrationRunError("calibration row lacks an execution key")
        key = tuple(raw_key)
        if key not in job_by_key:
            raise CalibrationRunError("calibration row is outside its shard design")
        if key in output:
            raise CalibrationRunError("calibration rows contain a duplicate path")
        _validate_row(row, job_by_key[key])
        output[key] = row
    if complete and set(output) != set(job_by_key):
        raise CalibrationRunError(
            f"calibration coverage is incomplete: {len(output)}/{len(job_by_key)}"
        )
    return sorted(output.values(), key=_row_sort_key)


def _checkpoint_path(output_dir: Path, shard: str) -> Path:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in shard
    )
    if not safe or safe != shard:
        raise CalibrationRunError(f"unsafe calibration shard id: {shard!r}")
    return Path(output_dir) / "checkpoints" / f"{safe}.checkpoint.json.zst"


def _safe_shard_name(shard: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in shard
    )
    if not safe or safe != shard:
        raise CalibrationRunError(f"unsafe calibration shard id: {shard!r}")
    return safe


def _shard_paths(output_dir: Path, shard: str) -> tuple[Path, Path, Path]:
    safe = _safe_shard_name(shard)
    artifact = Path(output_dir) / "raw" / "shards" / f"{safe}.paths.json.zst"
    return (
        artifact,
        Path(str(artifact) + ".metadata.json"),
        Path(str(artifact) + ".commit.json"),
    )


def _shard_design_sha256(jobs: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(
        _canonical_json_bytes([dict(job) for job in sorted(jobs, key=_job_sort_key)])
    )


def _execution_keys_sha256(jobs: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            [list(_job_key(job)) for job in sorted(jobs, key=_job_sort_key)]
        )
    )


def _shard_descriptor(
    shard: str, jobs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not jobs or {_shard_id(job) for job in jobs} != {shard}:
        raise CalibrationRunError("shard descriptor received incompatible jobs")
    canonical = sorted(jobs, key=_job_sort_key)
    return {
        "shard_id": shard,
        "block_id": str(canonical[0]["block_id"]),
        "policy": str(canonical[0]["policy"]),
        "assumed_tox_noise_sd_factor": float(
            canonical[0]["assumed_tox_noise_sd_factor"]
        ),
        "gamma": float(canonical[0]["gamma"]),
        "strata": list(STRATA),
        "seed_start": min(int(job["seed"]) for job in canonical),
        "seed_stop_exclusive": max(int(job["seed"]) for job in canonical) + 1,
        "path_count": len(canonical),
        "snapshot_count": len(canonical) * SNAPSHOTS_PER_PATH,
        "execution_keys_sha256": _execution_keys_sha256(canonical),
        "first_execution_key": list(_job_key(canonical[0])),
        "last_execution_key": list(_job_key(canonical[-1])),
    }


def _write_checkpoint(
    path: Path,
    *,
    shard: str,
    rows: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> None:
    canonical = _canonical_rows(rows, jobs, complete=False)
    payload = {
        "schema_version": 1,
        "status": CHECKPOINT_STATUS,
        "artifact_class": "atomic_resume_checkpoint_not_for_analysis",
        "shard_id": shard,
        "seed_block_id": str(jobs[0]["block_id"]),
        "shard_design_sha256": _shard_design_sha256(jobs),
        "expected_path_count": len(jobs),
        "completed_path_count": len(canonical),
        "bindings": dict(bindings),
        "rows": canonical,
    }
    _, compressed = _compress_json(payload, level=3)
    _atomic_bytes_write(Path(path), compressed)


def _load_checkpoint(
    path: Path,
    *,
    shard: str,
    jobs: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    payload, _, _ = _read_zstd_json(path, "calibration checkpoint")
    expected = {
        "schema_version": 1,
        "status": CHECKPOINT_STATUS,
        "artifact_class": "atomic_resume_checkpoint_not_for_analysis",
        "shard_id": shard,
        "seed_block_id": str(jobs[0]["block_id"]),
        "shard_design_sha256": _shard_design_sha256(jobs),
        "expected_path_count": len(jobs),
        "bindings": dict(bindings),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise CalibrationRunError(
                f"calibration checkpoint {field} resume binding mismatch"
            )
    if set(payload) != CHECKPOINT_FIELDS:
        raise CalibrationRunError("calibration checkpoint fields changed")
    rows = payload.get("rows")
    if not isinstance(rows, list) or payload.get("completed_path_count") != len(rows):
        raise CalibrationRunError("calibration checkpoint path count is malformed")
    canonical = _canonical_rows(rows, jobs, complete=False)
    if rows != canonical:
        raise CalibrationRunError("calibration checkpoint rows are not canonical")
    return canonical


def _require_ray_capacity(resources: Mapping[str, Any]) -> None:
    cpu = float(resources.get("CPU", 0.0))
    if not math.isfinite(cpu) or not math.isclose(
        cpu, float(RAY_NUM_CPUS), rel_tol=0.0, abs_tol=1e-9
    ):
        raise CalibrationRunError(
            f"Ray must expose exactly {RAY_NUM_CPUS} CPUs; observed {cpu:g}"
        )


def _start_ray() -> None:
    if ray.is_initialized():
        raise CalibrationRunError(
            "a pre-existing Ray runtime is not accepted; the audit must initialize "
            "its frozen ten-CPU runtime"
        )
    try:
        ray.init(
            num_cpus=RAY_NUM_CPUS,
            include_dashboard=False,
            ignore_reinit_error=False,
            log_to_driver=False,
        )
        _require_ray_capacity(ray.cluster_resources())
        _require_ray_capacity(ray.available_resources())
    except BaseException:
        if ray.is_initialized():
            ray.shutdown()
        raise


def _ray_worker_run_job(job: Mapping[str, Any]) -> dict[str, Any]:
    core = _load_core()
    core.initialize_worker()
    try:
        result = core.run_calibration_path(
            str(job["policy"]),
            int(job["seed"]),
            int(job["stratum"]),
            float(job["gamma"]),
            float(job["assumed_tox_noise_sd_factor"]),
            maximum_budget=int(job["maximum_budget"]),
            snapshot_budgets=tuple(int(value) for value in job["snapshot_budgets"]),
        )
        row = {
            "execution_key": list(_job_key(job)),
            "block_id": str(job["block_id"]),
            "result": result,
        }
        _validate_row(row, job)
        return row
    finally:
        core.cleanup_worker()


def _run_jobs_ray(
    jobs: Sequence[Mapping[str, Any]],
    *,
    on_result: Callable[[dict[str, Any]], None],
) -> None:
    if not ray.is_initialized():
        raise CalibrationRunError("Ray is not initialized; serial fallback is prohibited")
    remote_job = ray.remote(num_cpus=WORKER_NUM_CPUS, max_retries=0)(
        _ray_worker_run_job
    )
    iterator = iter(jobs)
    pending: dict[Any, tuple[Any, ...]] = {}

    def fill() -> None:
        while len(pending) < MAX_IN_FLIGHT:
            try:
                job = next(iterator)
            except StopIteration:
                return
            reference = remote_job.remote(dict(job))
            pending[reference] = _job_key(job)

    fill()
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1)
        for reference in ready:
            key = pending.pop(reference)
            try:
                row = ray.get(reference)
            except BaseException as error:
                for remaining in pending:
                    ray.cancel(remaining, force=True)
                raise CalibrationRunError(
                    f"Ray calibration task failed: {key}"
                ) from error
            on_result(dict(row))
        fill()


def _write_shard_envelope(
    output_dir: Path,
    *,
    shard: str,
    jobs: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    checkpoint_sha256: str,
    bindings: Mapping[str, Any],
) -> Path:
    paths = _shard_paths(output_dir, shard)
    if _validate_envelope_absence_or_commit(paths, f"calibration shard {shard}"):
        return _validate_committed_shard(
            output_dir,
            shard=shard,
            jobs=jobs,
            bindings=bindings,
        )[0]
    if not _is_sha256(checkpoint_sha256):
        raise CalibrationRunError("shard checkpoint hash is malformed")
    checkpoint = _checkpoint_path(output_dir, shard)
    if not checkpoint.is_file() or _sha256_file(checkpoint) != checkpoint_sha256:
        raise CalibrationRunError(
            "shard commit requires the exact completed atomic checkpoint"
        )
    canonical = _canonical_rows(rows, jobs, complete=True)
    descriptor = _shard_descriptor(shard, jobs)
    payload = {
        "schema_version": 1,
        "status": SHARD_STATUS,
        "artifact_class": SHARD_ARTIFACT_CLASS,
        "shard": descriptor,
        "bindings": dict(bindings),
        "canonical_sort_key": list(CANONICAL_SORT_KEY),
        "checkpoint_sha256": checkpoint_sha256,
        "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
        "rows": canonical,
    }
    uncompressed, compressed = _compress_json(payload, level=19)
    artifact, metadata_path, commit_path = paths
    _atomic_bytes_write(artifact, compressed)
    metadata = {
        "schema_version": 1,
        "status": SHARD_STATUS,
        "artifact_class": SHARD_ARTIFACT_CLASS,
        "artifact": artifact.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(uncompressed),
        "uncompressed_bytes": len(uncompressed),
        "shard": descriptor,
        "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
        "checkpoint_sha256": checkpoint_sha256,
        "bindings": dict(bindings),
        "compression": dict(COMPRESSION_CONTRACT),
        "environment": _privacy_safe_environment(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json_write(metadata_path, metadata)
    _atomic_json_write(
        commit_path,
        {
            "schema_version": 1,
            "status": SHARD_COMMIT_STATUS,
            "artifact": artifact.name,
            "artifact_sha256": _sha256_file(artifact),
            "uncompressed_sha256": _sha256_bytes(uncompressed),
            "metadata": metadata_path.name,
            "metadata_sha256": _sha256_file(metadata_path),
            "shard_id": shard,
            "block_id": descriptor["block_id"],
            "path_count": descriptor["path_count"],
            "snapshot_count": descriptor["snapshot_count"],
            "execution_keys_sha256": descriptor["execution_keys_sha256"],
            "launch_fingerprint": bindings["launch_fingerprint"],
            "immutable": True,
            "committed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return _validate_committed_shard(
        output_dir,
        shard=shard,
        jobs=jobs,
        bindings=bindings,
    )[0]


def _validate_committed_shard_envelope(
    output_dir: Path,
    *,
    shard: str,
    jobs: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, str], dict[str, Any]]:
    artifact, metadata_path, commit_path = _shard_paths(output_dir, shard)
    if not all(path.is_file() for path in (artifact, metadata_path, commit_path)):
        raise CalibrationRunError(f"committed calibration shard is incomplete: {shard}")
    metadata = _read_canonical_json_object(
        metadata_path, f"calibration shard metadata {shard}"
    )
    commit = _read_canonical_json_object(
        commit_path, f"calibration shard commit {shard}"
    )
    if set(metadata) != SHARD_METADATA_FIELDS:
        raise CalibrationRunError(f"calibration shard metadata fields changed: {shard}")
    if set(commit) != SHARD_COMMIT_FIELDS:
        raise CalibrationRunError(f"calibration shard commit fields changed: {shard}")
    descriptor = _shard_descriptor(shard, jobs)
    artifact_sha256 = _sha256_file(artifact)
    metadata_sha256 = _sha256_file(metadata_path)
    commit_sha256 = _sha256_file(commit_path)
    expected_commit = {
        "schema_version": 1,
        "status": SHARD_COMMIT_STATUS,
        "artifact": artifact.name,
        "artifact_sha256": artifact_sha256,
        "uncompressed_sha256": metadata.get("uncompressed_sha256"),
        "metadata": metadata_path.name,
        "metadata_sha256": metadata_sha256,
        "shard_id": shard,
        "block_id": descriptor["block_id"],
        "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
        "execution_keys_sha256": descriptor["execution_keys_sha256"],
        "launch_fingerprint": bindings["launch_fingerprint"],
        "immutable": True,
    }
    for field, value in expected_commit.items():
        if commit.get(field) != value:
            raise CalibrationRunError(f"shard commit mismatch for {shard}:{field}")
    expected_metadata = {
        "schema_version": 1,
        "status": SHARD_STATUS,
        "artifact_class": SHARD_ARTIFACT_CLASS,
        "artifact": artifact.name,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": artifact.stat().st_size,
        "shard": descriptor,
        "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
        "bindings": dict(bindings),
        "compression": dict(COMPRESSION_CONTRACT),
    }
    for field, value in expected_metadata.items():
        if metadata.get(field) != value:
            raise CalibrationRunError(f"shard metadata mismatch for {shard}:{field}")
    logical_sha = metadata.get("uncompressed_sha256")
    logical_bytes = metadata.get("uncompressed_bytes")
    checkpoint_sha = metadata.get("checkpoint_sha256")
    if (
        not _is_sha256(logical_sha)
        or not isinstance(logical_bytes, int)
        or logical_bytes <= 0
        or not _is_sha256(checkpoint_sha)
    ):
        raise CalibrationRunError(f"shard logical/checkpoint commitments malformed: {shard}")
    hashes = {
        str(artifact.relative_to(output_dir)): artifact_sha256,
        str(metadata_path.relative_to(output_dir)): metadata_sha256,
        str(commit_path.relative_to(output_dir)): commit_sha256,
    }
    identity = {
        "shard_id": shard,
        "block_id": descriptor["block_id"],
        "artifact": str(artifact.relative_to(output_dir)),
        "artifact_sha256": artifact_sha256,
        "uncompressed_sha256": logical_sha,
        "uncompressed_bytes": logical_bytes,
        "metadata": str(metadata_path.relative_to(output_dir)),
        "metadata_sha256": metadata_sha256,
        "commit": str(commit_path.relative_to(output_dir)),
        "commit_sha256": commit_sha256,
        "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
        "execution_keys_sha256": descriptor["execution_keys_sha256"],
        "first_execution_key": descriptor["first_execution_key"],
        "last_execution_key": descriptor["last_execution_key"],
    }
    if set(identity) != SHARD_INDEX_FIELDS:
        raise CalibrationRunError("internal shard-index fields changed")
    return artifact, metadata, commit, hashes, identity


def _validate_committed_shard(
    output_dir: Path,
    *,
    shard: str,
    jobs: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> tuple[Path, list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    artifact, metadata, _commit, hashes, identity = (
        _validate_committed_shard_envelope(
            output_dir,
            shard=shard,
            jobs=jobs,
            bindings=bindings,
        )
    )
    payload, compressed, uncompressed = _read_zstd_json(
        artifact, f"calibration shard {shard}"
    )
    if metadata.get("artifact_sha256") != _sha256_bytes(compressed):
        raise CalibrationRunError(f"shard compressed hash changed: {shard}")
    if (
        metadata.get("uncompressed_sha256") != _sha256_bytes(uncompressed)
        or metadata.get("uncompressed_bytes") != len(uncompressed)
    ):
        raise CalibrationRunError(f"shard logical hash changed: {shard}")
    descriptor = _shard_descriptor(shard, jobs)
    expected = {
        "schema_version": 1,
        "status": SHARD_STATUS,
        "artifact_class": SHARD_ARTIFACT_CLASS,
        "shard": descriptor,
        "bindings": dict(bindings),
        "canonical_sort_key": list(CANONICAL_SORT_KEY),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise CalibrationRunError(f"shard payload mismatch for {shard}:{field}")
    if set(payload) != SHARD_PAYLOAD_FIELDS:
        raise CalibrationRunError(f"calibration shard payload fields changed: {shard}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise CalibrationRunError(f"calibration shard lacks rows: {shard}")
    canonical = _canonical_rows(rows, jobs, complete=True)
    if rows != canonical:
        raise CalibrationRunError(f"calibration shard rows are not canonical: {shard}")
    return artifact, canonical, hashes, identity


def _block_paths(output_dir: Path, block_id: str) -> tuple[Path, Path, Path]:
    if block_id not in BLOCK_BY_ID:
        raise CalibrationRunError(f"unknown calibration seed block: {block_id}")
    artifact = (
        Path(output_dir)
        / "raw"
        / f"budget_toxicity_calibration_{block_id}_block_index.json.zst"
    )
    metadata = Path(str(artifact) + ".metadata.json")
    commit = Path(str(artifact) + ".commit.json")
    return artifact, metadata, commit


def _master_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    artifact = Path(output_dir) / "raw" / FINAL_MASTER_BASENAME
    return (
        artifact,
        Path(str(artifact) + ".metadata.json"),
        Path(str(artifact) + ".commit.json"),
    )


def _block_descriptor(block_id: str) -> dict[str, Any]:
    block = BLOCK_BY_ID[block_id]
    return {
        **block,
        "path_count": EXPECTED_BLOCK_PATHS[block_id],
        "snapshot_count": EXPECTED_BLOCK_PATHS[block_id] * SNAPSHOTS_PER_PATH,
        "shard_count": SHARDS_PER_BLOCK,
    }


def _validate_envelope_absence_or_commit(
    paths: tuple[Path, Path, Path], label: str
) -> bool:
    exists = tuple(path.exists() for path in paths)
    if not any(exists):
        return False
    if not all(exists):
        raise CalibrationRunError(
            f"{label} has a partial artifact/metadata/commit envelope; "
            "refusing to overwrite it"
        )
    return True


def _write_block_envelope(
    output_dir: Path,
    *,
    block_id: str,
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    shard_commits: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> Path:
    artifact, metadata_path, commit_path = _block_paths(output_dir, block_id)
    if _validate_envelope_absence_or_commit(
        (artifact, metadata_path, commit_path), f"seed block {block_id}"
    ):
        return _validate_committed_block_envelope(
            output_dir,
            block_id=block_id,
            groups=groups,
            bindings=bindings,
        )[0]
    descriptor = _block_descriptor(block_id)
    expected_shards = tuple(groups)
    if (
        len(expected_shards) != SHARDS_PER_BLOCK
        or tuple(value.get("shard_id") for value in shard_commits) != expected_shards
        or any(set(value) != SHARD_INDEX_FIELDS for value in shard_commits)
        or sum(int(value["path_count"]) for value in shard_commits)
        != descriptor["path_count"]
        or sum(int(value["snapshot_count"]) for value in shard_commits)
        != descriptor["snapshot_count"]
    ):
        raise CalibrationRunError("seed-block shard index coverage/fields changed")
    payload = {
        "schema_version": 1,
        "status": BLOCK_STATUS,
        "artifact_class": BLOCK_ARTIFACT_CLASS,
        "storage_model": "authenticated_index_over_immutable_shards",
        "block": descriptor,
        "bindings": dict(bindings),
        "canonical_sort_key": list(CANONICAL_SORT_KEY),
        "shard_commits": [dict(value) for value in shard_commits],
        "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
    }
    uncompressed, compressed = _compress_json(payload, level=19)
    _atomic_bytes_write(artifact, compressed)
    metadata = {
        "schema_version": 1,
        "status": BLOCK_STATUS,
        "artifact_class": BLOCK_ARTIFACT_CLASS,
        "storage_model": "authenticated_index_over_immutable_shards",
        "artifact": artifact.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(uncompressed),
        "uncompressed_bytes": len(uncompressed),
        "block_id": block_id,
        "seed_start": descriptor["seed_start"],
        "seed_stop_exclusive": descriptor["seed_stop_exclusive"],
        "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
        "shard_count": SHARDS_PER_BLOCK,
        "bindings": dict(bindings),
        "compression": dict(COMPRESSION_CONTRACT),
        "environment": _privacy_safe_environment(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json_write(metadata_path, metadata)
    _atomic_json_write(
        commit_path,
        {
            "schema_version": 1,
            "status": BLOCK_COMMIT_STATUS,
            "artifact": artifact.name,
            "artifact_sha256": _sha256_file(artifact),
            "uncompressed_sha256": _sha256_bytes(uncompressed),
            "metadata": metadata_path.name,
            "metadata_sha256": _sha256_file(metadata_path),
            "block_id": block_id,
            "path_count": descriptor["path_count"],
            "snapshot_count": descriptor["snapshot_count"],
            "shard_count": SHARDS_PER_BLOCK,
            "launch_fingerprint": bindings["launch_fingerprint"],
            "immutable": True,
            "committed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return _validate_committed_block_envelope(
        output_dir,
        block_id=block_id,
        groups=groups,
        bindings=bindings,
    )[0]


def _validate_committed_block_envelope(
    output_dir: Path,
    *,
    block_id: str,
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    bindings: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, str], dict[str, Any]]:
    """Authenticate the small block index and all shard envelopes, not rows."""
    artifact, metadata_path, commit_path = _block_paths(output_dir, block_id)
    if not all(path.is_file() for path in (artifact, metadata_path, commit_path)):
        raise CalibrationRunError(f"committed seed block is incomplete: {block_id}")
    metadata = _read_canonical_json_object(
        metadata_path, f"seed-block metadata {block_id}"
    )
    commit = _read_canonical_json_object(
        commit_path, f"seed-block commit {block_id}"
    )
    if set(metadata) != BLOCK_METADATA_FIELDS:
        raise CalibrationRunError(f"seed-block metadata fields changed: {block_id}")
    if set(commit) != BLOCK_COMMIT_FIELDS:
        raise CalibrationRunError(f"seed-block commit fields changed: {block_id}")
    descriptor = _block_descriptor(block_id)
    artifact_sha256 = _sha256_file(artifact)
    metadata_sha256 = _sha256_file(metadata_path)
    commit_sha256 = _sha256_file(commit_path)
    expected_commit = {
        "schema_version": 1,
        "status": BLOCK_COMMIT_STATUS,
        "artifact": artifact.name,
        "artifact_sha256": artifact_sha256,
        "uncompressed_sha256": metadata.get("uncompressed_sha256"),
        "metadata": metadata_path.name,
        "metadata_sha256": metadata_sha256,
        "block_id": block_id,
        "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
        "shard_count": SHARDS_PER_BLOCK,
        "launch_fingerprint": bindings["launch_fingerprint"],
        "immutable": True,
    }
    for field, value in expected_commit.items():
        if commit.get(field) != value:
            raise CalibrationRunError(
                f"seed-block commit mismatch for {block_id}:{field}"
            )
    expected_metadata = {
        "schema_version": 1,
        "status": BLOCK_STATUS,
        "artifact_class": BLOCK_ARTIFACT_CLASS,
        "storage_model": "authenticated_index_over_immutable_shards",
        "artifact": artifact.name,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": artifact.stat().st_size,
        "block_id": block_id,
        "seed_start": descriptor["seed_start"],
        "seed_stop_exclusive": descriptor["seed_stop_exclusive"],
        "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
        "shard_count": SHARDS_PER_BLOCK,
        "bindings": dict(bindings),
        "compression": dict(COMPRESSION_CONTRACT),
    }
    for field, value in expected_metadata.items():
        if metadata.get(field) != value:
            raise CalibrationRunError(
                f"seed-block metadata mismatch for {block_id}:{field}"
            )
    payload, compressed, uncompressed = _read_zstd_json(
        artifact, f"seed-block index {block_id}"
    )
    logical_sha = metadata.get("uncompressed_sha256")
    logical_bytes = metadata.get("uncompressed_bytes")
    if (
        not isinstance(logical_sha, str)
        or len(logical_sha) != 64
        or any(character not in "0123456789abcdef" for character in logical_sha)
        or not isinstance(logical_bytes, int)
        or logical_bytes <= 0
    ):
        raise CalibrationRunError(
            f"seed-block logical hash/size commitment is malformed: {block_id}"
        )
    if (
        metadata.get("artifact_sha256") != _sha256_bytes(compressed)
        or logical_sha != _sha256_bytes(uncompressed)
        or logical_bytes != len(uncompressed)
    ):
        raise CalibrationRunError(f"seed-block index hashes changed: {block_id}")
    expected_payload = {
        "schema_version": 1,
        "status": BLOCK_STATUS,
        "artifact_class": BLOCK_ARTIFACT_CLASS,
        "storage_model": "authenticated_index_over_immutable_shards",
        "block": descriptor,
        "bindings": dict(bindings),
        "canonical_sort_key": list(CANONICAL_SORT_KEY),
        "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
    }
    for field, value in expected_payload.items():
        if payload.get(field) != value:
            raise CalibrationRunError(f"seed-block index mismatch: {block_id}:{field}")
    if set(payload) != BLOCK_PAYLOAD_FIELDS or "rows" in payload:
        raise CalibrationRunError("seed-block index fields changed or duplicate rows")
    indexed_shards = payload.get("shard_commits")
    if (
        not isinstance(indexed_shards, list)
        or len(indexed_shards) != SHARDS_PER_BLOCK
        or tuple(value.get("shard_id") for value in indexed_shards) != tuple(groups)
    ):
        raise CalibrationRunError(f"seed-block shard index coverage changed: {block_id}")
    hashes = {
        str(artifact.relative_to(output_dir)): artifact_sha256,
        str(metadata_path.relative_to(output_dir)): metadata_sha256,
        str(commit_path.relative_to(output_dir)): commit_sha256,
    }
    observed_shards: list[dict[str, Any]] = []
    for shard, shard_jobs in groups.items():
        _shard_artifact, _shard_metadata, _shard_commit, shard_hashes, shard_identity = (
            _validate_committed_shard_envelope(
                output_dir,
                shard=shard,
                jobs=shard_jobs,
                bindings=bindings,
            )
        )
        hashes.update(shard_hashes)
        observed_shards.append(shard_identity)
    if indexed_shards != observed_shards:
        raise CalibrationRunError(f"seed-block shard identities changed: {block_id}")
    if (
        sum(int(value["path_count"]) for value in observed_shards)
        != descriptor["path_count"]
        or sum(int(value["snapshot_count"]) for value in observed_shards)
        != descriptor["snapshot_count"]
    ):
        raise CalibrationRunError(f"seed-block shard counts changed: {block_id}")
    identity = {
        "block_id": block_id,
        "artifact": str(artifact.relative_to(output_dir)),
        "artifact_sha256": artifact_sha256,
        "uncompressed_sha256": logical_sha,
        "uncompressed_bytes": logical_bytes,
        "metadata": str(metadata_path.relative_to(output_dir)),
        "metadata_sha256": metadata_sha256,
        "commit": str(commit_path.relative_to(output_dir)),
        "commit_sha256": commit_sha256,
        "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
        "shard_count": SHARDS_PER_BLOCK,
    }
    if set(identity) != BLOCK_INDEX_FIELDS:
        raise CalibrationRunError("internal block-index fields changed")
    return artifact, metadata, commit, hashes, identity


def _finalize_block(
    output_dir: Path,
    *,
    block_id: str,
    jobs: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    bindings: Mapping[str, Any],
) -> Path:
    envelope = _block_paths(output_dir, block_id)
    if _validate_envelope_absence_or_commit(envelope, f"seed block {block_id}"):
        return _validate_committed_block_envelope(
            output_dir,
            block_id=block_id,
            groups=groups,
            bindings=bindings,
        )[0]
    if set(str(job["block_id"]) for job in jobs) != {block_id}:
        raise CalibrationRunError("block finalizer received cross-block jobs")
    shard_commits: list[dict[str, Any]] = []
    for shard, shard_jobs in groups.items():
        _artifact, _metadata, _commit, _hashes, identity = (
            _validate_committed_shard_envelope(
                output_dir,
                shard=shard,
                jobs=shard_jobs,
                bindings=bindings,
            )
        )
        shard_commits.append(identity)
    return _write_block_envelope(
        output_dir,
        block_id=block_id,
        groups=groups,
        shard_commits=shard_commits,
        bindings=bindings,
    )


def _precision_path(output_dir: Path, cumulative_m: int) -> Path:
    return (
        Path(output_dir)
        / "precision"
        / f"budget_toxicity_calibration_precision_M{int(cumulative_m):04d}.json.zst"
    )


def _baseline_path(output_dir: Path) -> Path:
    return Path(output_dir) / "baseline" / BASELINE_BASENAME


def _load_analysis_module() -> Any:
    try:
        return importlib.import_module(
            "paper.analyze_budget_toxicity_calibration_audit"
        )
    except BaseException as error:
        raise CalibrationRunError("calibration analyzer cannot be imported") from error


def _load_baseline_identity(
    output_dir: Path,
    *,
    expected_input_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    path = _baseline_path(output_dir)
    analyzer = _load_analysis_module()
    try:
        payload = analyzer.load_baseline_identity_pass(
            path,
            expected_input_hashes=dict(expected_input_hashes),
            expected_m=200,
        )
    except BaseException as error:
        raise CalibrationRunError(
            "baseline identity PASS artifact is absent or invalid; precision and "
            "master finalization are blocked"
        ) from error
    if not isinstance(payload, Mapping):
        raise CalibrationRunError("baseline identity loader returned a non-mapping")
    expected_payload = {
        "status": BASELINE_PASS_STATUS,
        "cumulative_M": 200,
        "pass": True,
        "checked_common_cells": 3_200,
        "checked_allocation_histories": 3_040,
        "unavailable_allocation_histories": 160,
    }
    for field, value in expected_payload.items():
        if payload.get(field) != value:
            raise CalibrationRunError(f"baseline identity PASS changed: {field}")
    metadata = Path(str(path) + ".metadata.json")
    commit = Path(str(path) + ".commit.json")
    if not all(candidate.is_file() for candidate in (path, metadata, commit)):
        raise CalibrationRunError("baseline identity PASS envelope is incomplete")
    index = {
        **expected_payload,
        "artifact": str(path.relative_to(output_dir)),
        "artifact_sha256": _sha256_file(path),
        "metadata": str(metadata.relative_to(output_dir)),
        "metadata_sha256": _sha256_file(metadata),
        "commit": str(commit.relative_to(output_dir)),
        "commit_sha256": _sha256_file(commit),
    }
    if set(index) != BASELINE_INDEX_FIELDS:
        raise CalibrationRunError("internal baseline identity index fields changed")
    hashes = {
        str(path.relative_to(output_dir)): index["artifact_sha256"],
        str(metadata.relative_to(output_dir)): index["metadata_sha256"],
        str(commit.relative_to(output_dir)): index["commit_sha256"],
    }
    return index, hashes


def _load_precision_monitor() -> Any:
    try:
        return importlib.import_module(
            "paper.monitor_budget_toxicity_calibration_precision"
        )
    except BaseException as error:
        raise CalibrationRunError("blinded precision monitor cannot be imported") from error


def _validate_top_up_history(
    decisions: Sequence[Mapping[str, Any]], final_m: int
) -> list[dict[str, Any]]:
    required_m = tuple(value for value in (200, 500, 1000) if value <= int(final_m))
    observed_m = tuple(int(value.get("cumulative_M", -1)) for value in decisions)
    if observed_m != required_m:
        raise CalibrationRunError(
            f"precision-decision history is incomplete or out of order: {observed_m}"
        )
    expected_next = {
        200: None if final_m == 200 else 500,
        500: None if final_m == 500 else 1000,
        1000: None,
    }
    history: list[dict[str, Any]] = []
    for decision in decisions:
        current = int(decision["cumulative_M"])
        next_m = decision.get("next_M")
        if next_m != expected_next[current]:
            if current == final_m:
                raise CalibrationRunError(
                    "final raw master is blocked until the precision decision "
                    "has next_M = null"
                )
            raise CalibrationRunError(
                f"invalid blinded top-up transition at M={current}: {next_m!r}"
            )
        history.append(
            {
                "cumulative_M": current,
                "precision_artifact": str(decision["precision_artifact"]),
                "precision_artifact_sha256": str(
                    decision["precision_artifact_sha256"]
                ),
                "next_M": next_m,
            }
        )
    return history


def _load_precision_history(
    output_dir: Path,
    *,
    final_m: int,
    cumulative_block_hashes: Mapping[int, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    monitor = _load_precision_monitor()
    decisions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for cumulative_m in tuple(value for value in (200, 500, 1000) if value <= final_m):
        path = _precision_path(output_dir, cumulative_m)
        try:
            payload = monitor.load_precision_decision(
                path,
                expected_raw_hashes=dict(cumulative_block_hashes[cumulative_m]),
                expected_M=int(cumulative_m),
            )
        except BaseException as error:
            raise CalibrationRunError(
                f"invalid blinded precision decision at M={cumulative_m}"
            ) from error
        if not isinstance(payload, Mapping):
            raise CalibrationRunError("precision monitor returned a non-mapping payload")
        decisions.append(
            {
                **dict(payload),
                "precision_artifact": str(path.relative_to(output_dir)),
                "precision_artifact_sha256": _sha256_file(path),
            }
        )
        summaries.append(
            {
                "cumulative_M": int(cumulative_m),
                "artifact": str(path.relative_to(output_dir)),
                "artifact_sha256": _sha256_file(path),
                "metadata": str(
                    Path(str(path) + ".metadata.json").relative_to(output_dir)
                ),
                "metadata_sha256": _sha256_file(
                    Path(str(path) + ".metadata.json")
                ),
                "commit": str(
                    Path(str(path) + ".commit.json").relative_to(output_dir)
                ),
                "commit_sha256": _sha256_file(
                    Path(str(path) + ".commit.json")
                ),
                "next_M": payload.get("next_M"),
            }
        )
    history = _validate_top_up_history(decisions, final_m)
    return history, summaries


def _validate_master_indexes(
    output_dir: Path,
    *,
    final_m: int,
    baseline_identity: Mapping[str, Any],
    block_commits: Sequence[Mapping[str, Any]],
    precision_decisions: Sequence[Mapping[str, Any]],
    top_up_history: Sequence[Mapping[str, Any]],
) -> None:
    if set(baseline_identity) != BASELINE_INDEX_FIELDS:
        raise CalibrationRunError("raw-master baseline identity fields changed")
    for field, value in {
        "status": BASELINE_PASS_STATUS,
        "cumulative_M": 200,
        "pass": True,
        "checked_common_cells": 3_200,
        "checked_allocation_histories": 3_040,
        "unavailable_allocation_histories": 160,
    }.items():
        if baseline_identity.get(field) != value:
            raise CalibrationRunError(f"raw-master baseline identity changed: {field}")
    baseline_artifact = _baseline_path(output_dir)
    baseline_metadata = Path(str(baseline_artifact) + ".metadata.json")
    baseline_commit = Path(str(baseline_artifact) + ".commit.json")
    for field, expected in {
        "artifact": str(baseline_artifact.relative_to(output_dir)),
        "metadata": str(baseline_metadata.relative_to(output_dir)),
        "commit": str(baseline_commit.relative_to(output_dir)),
    }.items():
        if baseline_identity.get(field) != expected:
            raise CalibrationRunError(f"raw-master baseline path changed: {field}")
    for path, hash_field in (
        (baseline_artifact, "artifact_sha256"),
        (baseline_metadata, "metadata_sha256"),
        (baseline_commit, "commit_sha256"),
    ):
        if (
            not _is_sha256(baseline_identity.get(hash_field))
            or not path.is_file()
            or _sha256_file(path) != baseline_identity[hash_field]
        ):
            raise CalibrationRunError(
                f"raw-master baseline identity file/hash mismatch: {path.name}"
            )

    expected_blocks = BLOCKS_FOR_M[int(final_m)]
    if tuple(value.get("block_id") for value in block_commits) != expected_blocks:
        raise CalibrationRunError("raw-master block index order/coverage changed")
    for value in block_commits:
        if set(value) != BLOCK_INDEX_FIELDS:
            raise CalibrationRunError("raw-master block index fields changed")
        block_id = str(value["block_id"])
        artifact, metadata, commit = _block_paths(output_dir, block_id)
        expected_paths = {
            "artifact": str(artifact.relative_to(output_dir)),
            "metadata": str(metadata.relative_to(output_dir)),
            "commit": str(commit.relative_to(output_dir)),
            "path_count": EXPECTED_BLOCK_PATHS[block_id],
            "snapshot_count": EXPECTED_BLOCK_PATHS[block_id] * SNAPSHOTS_PER_PATH,
            "shard_count": SHARDS_PER_BLOCK,
        }
        for field, expected in expected_paths.items():
            if value.get(field) != expected:
                raise CalibrationRunError(
                    f"raw-master block index mismatch for {block_id}:{field}"
                )
        for field in (
            "artifact_sha256",
            "uncompressed_sha256",
            "metadata_sha256",
            "commit_sha256",
        ):
            if not _is_sha256(value.get(field)):
                raise CalibrationRunError(
                    f"raw-master block index has malformed {field}"
                )
        for path, hash_field in (
            (artifact, "artifact_sha256"),
            (metadata, "metadata_sha256"),
            (commit, "commit_sha256"),
        ):
            if not path.is_file() or _sha256_file(path) != value[hash_field]:
                raise CalibrationRunError(
                    f"raw-master block index file/hash mismatch: {path.name}"
                )
        indexed_metadata = _read_canonical_json_object(
            metadata, "raw-master indexed block metadata"
        )
        if (
            indexed_metadata.get("uncompressed_sha256")
            != value["uncompressed_sha256"]
            or indexed_metadata.get("uncompressed_bytes")
            != value["uncompressed_bytes"]
        ):
            raise CalibrationRunError(
                "raw-master block logical commitments differ from metadata"
            )
        if not isinstance(value.get("uncompressed_bytes"), int) or int(
            value["uncompressed_bytes"]
        ) <= 0:
            raise CalibrationRunError("raw-master block logical size is malformed")

    expected_ms = tuple(value for value in (200, 500, 1000) if value <= final_m)
    if tuple(value.get("cumulative_M") for value in precision_decisions) != expected_ms:
        raise CalibrationRunError("raw-master precision index order/coverage changed")
    for value in precision_decisions:
        if set(value) != PRECISION_INDEX_FIELDS:
            raise CalibrationRunError("raw-master precision index fields changed")
        cumulative_m = int(value["cumulative_M"])
        artifact = _precision_path(output_dir, cumulative_m)
        metadata = Path(str(artifact) + ".metadata.json")
        commit = Path(str(artifact) + ".commit.json")
        for field, expected in {
            "artifact": str(artifact.relative_to(output_dir)),
            "metadata": str(metadata.relative_to(output_dir)),
            "commit": str(commit.relative_to(output_dir)),
        }.items():
            if value.get(field) != expected:
                raise CalibrationRunError(
                    f"raw-master precision index mismatch at M={cumulative_m}:{field}"
                )
        for field in ("artifact_sha256", "metadata_sha256", "commit_sha256"):
            if not _is_sha256(value.get(field)):
                raise CalibrationRunError(
                    f"raw-master precision index has malformed {field}"
                )
        for path, hash_field in (
            (artifact, "artifact_sha256"),
            (metadata, "metadata_sha256"),
            (commit, "commit_sha256"),
        ):
            if not path.is_file() or _sha256_file(path) != value[hash_field]:
                raise CalibrationRunError(
                    f"raw-master precision index file/hash mismatch: {path.name}"
                )

    if tuple(value.get("cumulative_M") for value in top_up_history) != expected_ms:
        raise CalibrationRunError("raw-master top-up history order/coverage changed")
    for value in top_up_history:
        if set(value) != TOP_UP_HISTORY_FIELDS:
            raise CalibrationRunError("raw-master top-up history fields changed")
        if not _is_sha256(value.get("precision_artifact_sha256")):
            raise CalibrationRunError("raw-master top-up history hash is malformed")
    expected_history = [
        {
            "cumulative_M": value["cumulative_M"],
            "precision_artifact": value["artifact"],
            "precision_artifact_sha256": value["artifact_sha256"],
            "next_M": value["next_M"],
        }
        for value in precision_decisions
    ]
    if [dict(value) for value in top_up_history] != expected_history:
        raise CalibrationRunError(
            "raw-master top-up history differs from precision commitments"
        )


def _write_master_envelope(
    output_dir: Path,
    *,
    final_m: int,
    bindings: Mapping[str, Any],
    baseline_identity: Mapping[str, Any],
    block_commits: Sequence[Mapping[str, Any]],
    precision_decisions: Sequence[Mapping[str, Any]],
    top_up_history: Sequence[Mapping[str, Any]],
) -> Path:
    _validate_master_indexes(
        output_dir,
        final_m=int(final_m),
        baseline_identity=baseline_identity,
        block_commits=block_commits,
        precision_decisions=precision_decisions,
        top_up_history=top_up_history,
    )
    path_count = PHYSICAL_PATHS_PER_REPLICATE * int(final_m)
    snapshot_count = path_count * SNAPSHOTS_PER_PATH
    if sum(int(value.get("path_count", -1)) for value in block_commits) != path_count:
        raise CalibrationRunError("raw-master block path counts differ from final M")
    if sum(int(value.get("snapshot_count", -1)) for value in block_commits) != snapshot_count:
        raise CalibrationRunError("raw-master block snapshot counts differ from final M")
    payload = {
        "schema_version": 1,
        "status": MASTER_STATUS,
        "artifact_class": MASTER_ARTIFACT_CLASS,
        "storage_model": MASTER_STORAGE_MODEL,
        "final_M": int(final_m),
        "cumulative_block_ids": list(BLOCKS_FOR_M[int(final_m)]),
        "path_count": path_count,
        "snapshot_count": snapshot_count,
        "bindings": dict(bindings),
        "baseline_identity": dict(baseline_identity),
        "block_commits": [dict(value) for value in block_commits],
        "precision_decisions": [dict(value) for value in precision_decisions],
        "top_up_history": [dict(value) for value in top_up_history],
        "canonical_sort_key": list(CANONICAL_SORT_KEY),
    }
    uncompressed, compressed = _compress_json(payload, level=19)
    artifact, metadata_path, commit_path = _master_paths(output_dir)
    if _validate_envelope_absence_or_commit(
        (artifact, metadata_path, commit_path), "calibration raw master"
    ):
        return _validate_committed_master(
            output_dir,
            final_m=final_m,
            bindings=bindings,
            baseline_identity=baseline_identity,
            block_commits=block_commits,
            precision_decisions=precision_decisions,
            top_up_history=top_up_history,
        )
    _atomic_bytes_write(artifact, compressed)
    metadata = {
        "schema_version": 1,
        "status": MASTER_STATUS,
        "artifact_class": MASTER_ARTIFACT_CLASS,
        "storage_model": MASTER_STORAGE_MODEL,
        "artifact": artifact.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(uncompressed),
        "uncompressed_bytes": len(uncompressed),
        "final_M": int(final_m),
        "path_count": path_count,
        "snapshot_count": snapshot_count,
        "bindings": dict(bindings),
        "compression": dict(COMPRESSION_CONTRACT),
        "environment": _privacy_safe_environment(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json_write(metadata_path, metadata)
    _atomic_json_write(
        commit_path,
        {
            "schema_version": 1,
            "status": MASTER_COMMIT_STATUS,
            "artifact": artifact.name,
            "artifact_sha256": _sha256_file(artifact),
            "uncompressed_sha256": _sha256_bytes(uncompressed),
            "metadata": metadata_path.name,
            "metadata_sha256": _sha256_file(metadata_path),
            "final_M": int(final_m),
            "path_count": path_count,
            "snapshot_count": snapshot_count,
            "launch_fingerprint": bindings["launch_fingerprint"],
            "committed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return _validate_committed_master(
        output_dir,
        final_m=final_m,
        bindings=bindings,
        baseline_identity=baseline_identity,
        block_commits=block_commits,
        precision_decisions=precision_decisions,
        top_up_history=top_up_history,
    )


def _validate_committed_master(
    output_dir: Path,
    *,
    final_m: int,
    bindings: Mapping[str, Any],
    baseline_identity: Mapping[str, Any],
    block_commits: Sequence[Mapping[str, Any]],
    precision_decisions: Sequence[Mapping[str, Any]],
    top_up_history: Sequence[Mapping[str, Any]],
) -> Path:
    _validate_master_indexes(
        output_dir,
        final_m=int(final_m),
        baseline_identity=baseline_identity,
        block_commits=block_commits,
        precision_decisions=precision_decisions,
        top_up_history=top_up_history,
    )
    artifact, metadata_path, commit_path = _master_paths(output_dir)
    if not all(path.is_file() for path in (artifact, metadata_path, commit_path)):
        raise CalibrationRunError("committed calibration raw master is incomplete")
    metadata = _read_canonical_json_object(metadata_path, "raw-master metadata")
    commit = _read_canonical_json_object(commit_path, "raw-master commit")
    if set(metadata) != MASTER_METADATA_FIELDS:
        raise CalibrationRunError("raw-master metadata fields changed")
    if set(commit) != MASTER_COMMIT_FIELDS:
        raise CalibrationRunError("raw-master commit fields changed")
    path_count = PHYSICAL_PATHS_PER_REPLICATE * int(final_m)
    snapshot_count = path_count * SNAPSHOTS_PER_PATH
    expected_commit = {
        "schema_version": 1,
        "status": MASTER_COMMIT_STATUS,
        "artifact": artifact.name,
        "artifact_sha256": _sha256_file(artifact),
        "uncompressed_sha256": metadata.get("uncompressed_sha256"),
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256_file(metadata_path),
        "final_M": int(final_m),
        "path_count": path_count,
        "snapshot_count": snapshot_count,
        "launch_fingerprint": bindings["launch_fingerprint"],
    }
    for field, value in expected_commit.items():
        if commit.get(field) != value:
            raise CalibrationRunError(f"raw-master commit mismatch for {field}")
    payload, compressed, uncompressed = _read_zstd_json(
        artifact, "calibration raw master"
    )
    expected_metadata = {
        "schema_version": 1,
        "status": MASTER_STATUS,
        "artifact_class": MASTER_ARTIFACT_CLASS,
        "storage_model": MASTER_STORAGE_MODEL,
        "artifact": artifact.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(uncompressed),
        "uncompressed_bytes": len(uncompressed),
        "final_M": int(final_m),
        "path_count": path_count,
        "snapshot_count": snapshot_count,
        "bindings": dict(bindings),
        "compression": dict(COMPRESSION_CONTRACT),
    }
    for field, value in expected_metadata.items():
        if metadata.get(field) != value:
            raise CalibrationRunError(f"raw-master metadata mismatch for {field}")
    expected_payload = {
        "schema_version": 1,
        "status": MASTER_STATUS,
        "artifact_class": MASTER_ARTIFACT_CLASS,
        "storage_model": MASTER_STORAGE_MODEL,
        "final_M": int(final_m),
        "cumulative_block_ids": list(BLOCKS_FOR_M[int(final_m)]),
        "path_count": path_count,
        "snapshot_count": snapshot_count,
        "bindings": dict(bindings),
        "baseline_identity": dict(baseline_identity),
        "block_commits": [dict(value) for value in block_commits],
        "precision_decisions": [dict(value) for value in precision_decisions],
        "top_up_history": [dict(value) for value in top_up_history],
        "canonical_sort_key": list(CANONICAL_SORT_KEY),
    }
    for field, value in expected_payload.items():
        if payload.get(field) != value:
            raise CalibrationRunError(f"raw-master payload mismatch for {field}")
    if set(payload) != MASTER_PAYLOAD_FIELDS:
        raise CalibrationRunError("raw-master index payload fields changed")
    if "rows" in payload:
        raise CalibrationRunError(
            "raw-master index must not duplicate immutable seed-block rows"
        )
    return artifact


def _finalize_master(
    output_dir: Path,
    *,
    final_m: int,
    manifest: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> Path:
    if int(final_m) not in BLOCKS_FOR_M:
        raise CalibrationRunError("final M must be exactly 200, 500, or 1000")
    _validate_manifest_design(manifest)
    cumulative_hashes: dict[str, str] = {
        "paper/budget_toxicity_calibration_manifest.json": str(
            bindings["manifest_sha256"]
        ),
        "paper/budget_toxicity_calibration_prespec.md": str(
            bindings["analysis_spec_sha256"]
        ),
    }
    hashes_by_m: dict[int, dict[str, str]] = {}
    block_commits: list[dict[str, Any]] = []
    for block_id in BLOCKS_FOR_M[int(final_m)]:
        groups = _group_jobs(_expand_jobs(manifest, block_id))
        artifact, _metadata, _commit, hashes, identity = (
            _validate_committed_block_envelope(
                output_dir,
                block_id=block_id,
                groups=groups,
                bindings=bindings,
            )
        )
        cumulative_hashes.update(hashes)
        cumulative_m = int(BLOCK_BY_ID[block_id]["cumulative_M"])
        hashes_by_m[cumulative_m] = dict(sorted(cumulative_hashes.items()))
        if identity["artifact"] != str(artifact.relative_to(output_dir)):
            raise CalibrationRunError("seed-block index artifact path changed")
        block_commits.append(identity)
    baseline_identity, baseline_hashes = _load_baseline_identity(
        output_dir,
        expected_input_hashes=hashes_by_m[200],
    )
    for cumulative_m in tuple(hashes_by_m):
        combined = {**hashes_by_m[cumulative_m], **baseline_hashes}
        hashes_by_m[cumulative_m] = dict(sorted(combined.items()))
    history, precision_decisions = _load_precision_history(
        output_dir,
        final_m=int(final_m),
        cumulative_block_hashes=hashes_by_m,
    )
    return _write_master_envelope(
        output_dir,
        final_m=int(final_m),
        bindings=bindings,
        baseline_identity=baseline_identity,
        block_commits=block_commits,
        precision_decisions=precision_decisions,
        top_up_history=history,
    )


def _assert_isolated_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    protected = (
        (ROOT / "results/formal_rerun_stable_z_staging").resolve(),
        (ROOT / "results/comparator_family_completion_staging").resolve(),
        (ROOT / "results/comparator_family_analysis").resolve(),
        (ROOT / "analysis/formal_execution_provenance").resolve(),
        (ROOT / "records/comparator_family_extension").resolve(),
    )
    if any(resolved == root or resolved.is_relative_to(root) for root in protected):
        raise CalibrationRunError(
            "calibration output must remain separate from immutable prior evidence"
        )
    return resolved


def _authorization_is_valid(*, mutate: bool, token: str | None) -> bool:
    return not mutate or token == AUTHORIZATION_TOKEN


def _select_shards(
    groups: Mapping[str, list[dict[str, Any]]], requested: Sequence[str]
) -> list[tuple[str, list[dict[str, Any]]]]:
    if requested:
        unknown = set(requested) - set(groups)
        if unknown:
            raise CalibrationRunError(f"unknown calibration shards: {sorted(unknown)}")
        names = list(dict.fromkeys(requested))
    else:
        names = list(groups)
    return [(name, groups[name]) for name in names]


def _execute_selected_shards(
    output_dir: Path,
    *,
    selected: Sequence[tuple[str, list[dict[str, Any]]]],
    bindings: Mapping[str, Any],
) -> None:
    pending_shards: list[tuple[str, list[dict[str, Any]]]] = []
    for shard, shard_jobs in selected:
        envelope = _shard_paths(output_dir, shard)
        if _validate_envelope_absence_or_commit(envelope, f"calibration shard {shard}"):
            _validate_committed_shard_envelope(
                output_dir,
                shard=shard,
                jobs=shard_jobs,
                bindings=bindings,
            )
            print(f"[committed-existing] {shard}", flush=True)
        else:
            pending_shards.append((shard, shard_jobs))
    if not pending_shards:
        return
    _start_ray()
    try:
        for shard, shard_jobs in pending_shards:
            path = _checkpoint_path(output_dir, shard)
            rows = _load_checkpoint(
                path,
                shard=shard,
                jobs=shard_jobs,
                bindings=bindings,
            )
            complete = {tuple(row["execution_key"]) for row in rows}
            remaining = [job for job in shard_jobs if _job_key(job) not in complete]
            print(
                f"[resume] {shard}: {len(rows)}/{len(shard_jobs)}; "
                f"remaining={len(remaining)}",
                flush=True,
            )
            new_since_checkpoint = 0

            def checkpoint() -> None:
                nonlocal new_since_checkpoint
                _write_checkpoint(
                    path,
                    shard=shard,
                    rows=rows,
                    jobs=shard_jobs,
                    bindings=bindings,
                )
                new_since_checkpoint = 0

            def accept(row: dict[str, Any]) -> None:
                nonlocal new_since_checkpoint
                rows.append(row)
                new_since_checkpoint += 1
                if new_since_checkpoint >= CHECKPOINT_EVERY:
                    checkpoint()

            if remaining:
                _run_jobs_ray(remaining, on_result=accept)
            if new_since_checkpoint or not path.exists():
                checkpoint()
            if len(rows) != len(shard_jobs):
                raise CalibrationRunError(f"selected shard did not finish: {shard}")
            _write_shard_envelope(
                output_dir,
                shard=shard,
                jobs=shard_jobs,
                rows=rows,
                checkpoint_sha256=_sha256_file(path),
                bindings=bindings,
            )
            print(f"[shard-committed] {shard}", flush=True)
    finally:
        ray.shutdown()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--analysis-spec", type=Path, default=ANALYSIS_SPEC_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--block-id", choices=BLOCK_IDS)
    parser.add_argument("--shard-id", action="append", default=[])
    parser.add_argument("--cumulative-M", type=int, choices=tuple(BLOCKS_FOR_M))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--finalize-block", action="store_true")
    action.add_argument("--finalize-master", action="store_true")
    parser.add_argument("--authorization-token")
    args = parser.parse_args(argv)

    mutating = bool(args.execute or args.finalize_block or args.finalize_master)
    if not _authorization_is_valid(mutate=mutating, token=args.authorization_token):
        parser.error(
            "execution/finalization requires --authorization-token "
            f"{AUTHORIZATION_TOKEN}"
        )
    if (args.execute or args.finalize_block) and args.block_id is None:
        parser.error("--execute/--finalize-block requires one --block-id")
    if args.finalize_master and args.cumulative_M is None:
        parser.error("--finalize-master requires --cumulative-M")
    if args.finalize_master and args.block_id is not None:
        parser.error("--block-id is incompatible with --finalize-master")
    if args.shard_id and not args.execute:
        parser.error("--shard-id is accepted only with --execute")
    if args.cumulative_M is not None and not args.finalize_master:
        parser.error("--cumulative-M is accepted only with --finalize-master")

    manifest_path = args.manifest.resolve()
    analysis_spec_path = args.analysis_spec.resolve()
    manifest = _load_frozen_manifest(manifest_path, analysis_spec_path)
    bindings = _build_bindings(manifest_path, analysis_spec_path)
    preview_blocks = (args.block_id,) if args.block_id else BLOCK_IDS
    preview_jobs = sum(EXPECTED_BLOCK_PATHS[value] for value in preview_blocks)
    print(
        f"[preflight] blocks={','.join(preview_blocks)}; "
        f"physical_paths={preview_jobs}; shards_per_block=24; "
        f"fingerprint={bindings['launch_fingerprint']}",
        flush=True,
    )
    print(
        "[preflight] Ray=required; CPUs=10; worker_CPU=1; max_in_flight=20; "
        "serial_fallback=prohibited; exact_runtime=pass",
        flush=True,
    )
    if not mutating:
        print("[preflight] no checkpoint or raw artifact written", flush=True)
        return 0

    output_dir = _assert_isolated_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.finalize_master:
        artifact = _finalize_master(
            output_dir,
            final_m=int(args.cumulative_M),
            manifest=manifest,
            bindings=bindings,
        )
        print(f"[committed] {artifact}", flush=True)
        return 0

    assert args.block_id is not None
    jobs = _expand_jobs(manifest, args.block_id)
    groups = _group_jobs(jobs)
    if args.execute:
        selected = _select_shards(groups, args.shard_id)
        artifact, metadata, commit = _block_paths(output_dir, args.block_id)
        if any(path.exists() for path in (artifact, metadata, commit)):
            if args.shard_id:
                raise CalibrationRunError(
                    "seed block is already committed; shard execution is immutable"
                )
            _validate_committed_block_envelope(
                output_dir,
                block_id=args.block_id,
                groups=groups,
                bindings=bindings,
            )
            print(f"[committed-existing] {artifact}", flush=True)
            return 0
        _execute_selected_shards(
            output_dir,
            selected=selected,
            bindings=bindings,
        )
        if not args.shard_id:
            artifact = _finalize_block(
                output_dir,
                block_id=args.block_id,
                jobs=jobs,
                groups=groups,
                bindings=bindings,
            )
            print(f"[committed] {artifact}", flush=True)
        return 0

    artifact = _finalize_block(
        output_dir,
        block_id=args.block_id,
        jobs=jobs,
        groups=groups,
        bindings=bindings,
    )
    print(f"[committed] {artifact}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
