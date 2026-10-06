#!/usr/bin/env python3
"""Blinded Monte Carlo precision monitor for the calibration audit.

The monitor implements only the frozen, whole-factorial top-up rule.  It
authenticates a committed cumulative raw master, projects the four outcomes
needed by the 118 locked control estimands, and writes MCSEs without effect
estimates, signs, intervals, p-values, or rankings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import zstandard as zstd


ROOT = Path(__file__).resolve().parents[1]
PRESPEC_PATH = ROOT / "paper/budget_toxicity_calibration_prespec.md"
MANIFEST_PATH = ROOT / "paper/budget_toxicity_calibration_manifest.json"

POLICIES = ("cKG-exact-formal", "tmse", "qBIG", "cEI")
COMPARATORS = ("tmse", "qBIG", "cEI")
FACTORS = (0.5, 1.0, 2.0)
ERRONEOUS_FACTORS = (0.5, 2.0)
GATES = (0.7, 0.9)
STRATA = (0, 1)
BUDGETS = (20, 40, 80)
ELIGIBLE_M = (200, 500, 1000)
MCSE_THRESHOLD_POINTS = 1.5

# Outcome-blind numerical-equivalence guard frozen before the production
# restart.  ETA is sqrt(binary64 epsilon); all derived bounds are percentage
# points and are deliberately computed from the exact frozen expression.
NUMERICAL_EQUIVALENCE_ETA = 2.0 ** -26
PANEL_BRIER_SNAPSHOT_BOUND_POINTS = 200.0 * NUMERICAL_EQUIVALENCE_ETA
FAMILY_B_BRIER_REPLICATE_BOUND_POINTS = (
    400.0 * NUMERICAL_EQUIVALENCE_ETA
)
FAMILY_C_BRIER_INTERACTION_BOUND_POINTS = (
    800.0 * NUMERICAL_EQUIVALENCE_ETA
)

GUARD_CLASS_EXACT_DISCRETE = "exact_discrete_primary_contribution"
GUARD_CLASS_FAMILY_B_BRIER = "family_B_panel_brier_difference"
GUARD_CLASS_FAMILY_C_BRIER = "family_C_panel_brier_difference_in_differences"
ROW_GUARD_CLASSES = (
    GUARD_CLASS_EXACT_DISCRETE,
    GUARD_CLASS_FAMILY_B_BRIER,
    GUARD_CLASS_FAMILY_C_BRIER,
)

SHARD_STATUS = "COMPLETE_IMMUTABLE_BUDGET_TOXICITY_CALIBRATION_SHARD"
SHARD_COMMIT_STATUS = "COMMITTED_IMMUTABLE_BUDGET_TOXICITY_CALIBRATION_SHARD"
SHARD_ARTIFACT_CLASS = "authenticated_immutable_shard_raw_paths"
BLOCK_STATUS = "COMPLETE_IMMUTABLE_BUDGET_TOXICITY_CALIBRATION_SEED_BLOCK"
BLOCK_COMMIT_STATUS = "COMMITTED_IMMUTABLE_BUDGET_TOXICITY_CALIBRATION_SEED_BLOCK"
BLOCK_ARTIFACT_CLASS = "authenticated_seed_block_index_over_immutable_shards"
SEED_BLOCKS = {
    "M0200": (0, 200),
    "M0500_TOPUP": (200, 500),
    "M1000_TOPUP": (500, 1000),
}
SHARD_INDEX_FIELDS = {
    "shard_id", "block_id", "artifact", "artifact_sha256",
    "uncompressed_sha256", "uncompressed_bytes", "metadata",
    "metadata_sha256", "commit", "commit_sha256", "path_count",
    "snapshot_count", "execution_keys_sha256", "first_execution_key",
    "last_execution_key",
}
SHARD_METADATA_FIELDS = {
    "schema_version", "status", "artifact_class", "artifact",
    "artifact_sha256", "artifact_bytes", "uncompressed_sha256",
    "uncompressed_bytes", "shard", "path_count", "snapshot_count",
    "checkpoint_sha256", "bindings", "compression", "environment",
    "generated_at_utc",
}
SHARD_COMMIT_FIELDS = {
    "schema_version", "status", "artifact", "artifact_sha256",
    "uncompressed_sha256", "metadata", "metadata_sha256", "shard_id",
    "block_id", "path_count", "snapshot_count", "execution_keys_sha256",
    "launch_fingerprint", "immutable", "committed_at_utc",
}
BLOCK_METADATA_FIELDS = {
    "schema_version", "status", "artifact_class", "storage_model", "artifact",
    "artifact_sha256", "artifact_bytes", "uncompressed_sha256",
    "uncompressed_bytes", "block_id", "seed_start", "seed_stop_exclusive",
    "path_count", "snapshot_count", "shard_count", "bindings", "compression",
    "environment", "generated_at_utc",
}
BLOCK_COMMIT_FIELDS = {
    "schema_version", "status", "artifact", "artifact_sha256",
    "uncompressed_sha256", "metadata", "metadata_sha256", "block_id",
    "path_count", "snapshot_count", "shard_count", "launch_fingerprint",
    "immutable", "committed_at_utc",
}
PRECISION_STATUS = (
    "COMPLETE_BLINDED_BUDGET_TOXICITY_CALIBRATION_PRECISION_DECISION"
)
PRECISION_COMMIT_STATUS = (
    "COMMITTED_BLINDED_BUDGET_TOXICITY_CALIBRATION_PRECISION_DECISION"
)
PRECISION_ARTIFACT_CLASS = "blind_precision_decision_no_effect_estimates"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_FROZEN_RUNTIME_VERSIONS = {
    "python": "3.11.6",
    "numpy": "1.26.4",
    "scipy": "1.16.3",
    "torch": "2.4.1",
    "gpytorch": "1.14",
    "ray": "2.10.0",
    "zstandard": "0.19.0",
    "matplotlib": "3.10.7",
}
_FROZEN_RUNTIME_CONTRACT = {
    **_FROZEN_RUNTIME_VERSIONS,
    "ray_num_cpus": 10,
    "worker_num_cpus": 1,
    "maximum_in_flight": 20,
    "serial_fallback": False,
}
COMPRESSION_CONTRACT = {
    "format": "zstandard", "level": 19, "threads": 0, "checksum": True,
    "content_size": True, "dict_id": False,
}

OUTCOME_TERMINAL = "terminal_true_boundary_exceedance_pct"
OUTCOME_ASSIGNMENT = "above_boundary_assignment_pct"
OUTCOME_EXCLUSION = "terminal_target_exclusion_pct"
OUTCOME_BRIER = "panel_brier_score_pct"

_OUTCOME_ALIASES = {
    OUTCOME_TERMINAL: (
        OUTCOME_TERMINAL,
        "terminal_boundary_exceedance_pct",
        "terminal_recommendation_pct",
        "terminal_true_boundary_exceedance",
    ),
    OUTCOME_ASSIGNMENT: (
        OUTCOME_ASSIGNMENT,
        "above_boundary_assignment_percentage",
        "assignment_pct_n",
        "assignment_pct_nmax",
    ),
    OUTCOME_EXCLUSION: (
        OUTCOME_EXCLUSION,
        "target_terminal_exclusion_pct",
        "terminal_true_panel_target_exclusion_pct",
    ),
    OUTCOME_BRIER: (
        OUTCOME_BRIER,
        "brier_score_pct",
        "terminal_panel_brier_score_pct",
        "panel_feasibility_brier_pct",
    ),
}


class PrecisionMonitorError(RuntimeError):
    """Raised when authentication, factorial, or blinding invariants fail."""


def _assert_exact_runtime() -> dict[str, str]:
    """Fail before a hash-bound write unless the frozen runtime is active."""

    manifest = _read_finite_json(MANIFEST_PATH.read_bytes(), label=MANIFEST_PATH.name)
    required = manifest.get("required_runtime") if isinstance(manifest, Mapping) else None
    if not isinstance(required, Mapping):
        raise PrecisionMonitorError("design manifest lacks required_runtime")
    if dict(required) != _FROZEN_RUNTIME_CONTRACT:
        raise PrecisionMonitorError("manifest required-runtime contract changed")
    observed = {"python": platform.python_version()}
    for distribution in (
        "numpy", "scipy", "torch", "gpytorch", "ray", "zstandard",
        "matplotlib",
    ):
        try:
            observed[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise PrecisionMonitorError(
                f"frozen runtime package is absent: {distribution}"
            ) from exc
    if observed != _FROZEN_RUNTIME_VERSIONS:
        raise PrecisionMonitorError(
            f"wrong runtime for hash-bound audit: observed={observed!r}, "
            f"required={_FROZEN_RUNTIME_VERSIONS!r}"
        )
    return observed


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_finite_json(raw: bytes, *, label: str) -> Any:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token}")

    try:
        value = json.loads(raw, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PrecisionMonitorError(f"{label} is not finite valid JSON") from exc
    _assert_finite(value, label=label)
    return value


def _assert_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PrecisionMonitorError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, label=f"{label}[{index}]")


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PrecisionMonitorError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _sidecar_paths(raw_path: Path) -> tuple[Path, Path]:
    return (
        Path(str(raw_path) + ".metadata.json"),
        Path(str(raw_path) + ".commit.json"),
    )


def _block_descriptor(block_id: str) -> dict[str, Any]:
    if block_id not in SEED_BLOCKS:
        raise PrecisionMonitorError("unknown seed block")
    start, stop = SEED_BLOCKS[block_id]
    paths = 48 * (stop - start)
    return {
        "block_id": block_id, "seed_start": start, "seed_stop_exclusive": stop,
        "cumulative_M": stop, "path_count": paths, "snapshot_count": 3 * paths,
        "shard_count": 24,
    }


def _execution_keys(block_id: str, policy: str, factor: float, gamma: float) -> list[list[Any]]:
    start, stop = SEED_BLOCKS[block_id]
    return [
        [block_id, policy, factor, gamma, stratum, seed]
        for stratum in STRATA for seed in range(start, stop)
    ]


def _expected_shard_descriptor(
    block_id: str, policy: str, factor: float, gamma: float
) -> dict[str, Any]:
    keys = _execution_keys(block_id, policy, factor, gamma)
    start, stop = SEED_BLOCKS[block_id]
    shard_id = f"{block_id}__{policy}__csigma{factor:g}__tau{gamma:g}"
    return {
        "shard_id": shard_id, "block_id": block_id, "policy": policy,
        "assumed_tox_noise_sd_factor": factor, "gamma": gamma,
        "strata": list(STRATA), "seed_start": start, "seed_stop_exclusive": stop,
        "path_count": len(keys), "snapshot_count": 3 * len(keys),
        "execution_keys_sha256": _sha256_bytes(_canonical_json_bytes(keys)),
        "first_execution_key": keys[0], "last_execution_key": keys[-1],
    }


def _validate_bindings(bindings: Mapping[str, Any], *, verify_local: bool) -> None:
    if not isinstance(bindings, Mapping):
        raise PrecisionMonitorError("execution bindings are absent")
    if not verify_local:
        return
    if bindings.get("analysis_spec_sha256") != _sha256_file(PRESPEC_PATH):
        raise PrecisionMonitorError("artifact is not bound to the local prespec")
    if bindings.get("manifest_sha256") != _sha256_file(MANIFEST_PATH):
        raise PrecisionMonitorError("artifact is not bound to the local manifest")
    source_hashes = bindings.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise PrecisionMonitorError("execution source-hash map is absent")
    for relative, expected in source_hashes.items():
        _require_sha(expected, label=f"source_sha256.{relative}")
        source = ROOT / str(relative)
        if not source.is_file() or _sha256_file(source) != expected:
            raise PrecisionMonitorError(f"bound execution source changed: {relative}")


def authenticate_seed_block(
    raw_path: str | Path,
    *,
    verify_local_bindings: bool = True,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Authenticate a small seed-block index; shard rows remain unread."""

    path = Path(raw_path).resolve()
    metadata_path, commit_path = _sidecar_paths(path)
    if not all(candidate.is_file() for candidate in (path, metadata_path, commit_path)):
        raise PrecisionMonitorError("seed-block index lacks metadata/commit sidecars")
    compressed, metadata_raw, commit_raw = (
        path.read_bytes(), metadata_path.read_bytes(), commit_path.read_bytes()
    )
    metadata = _read_finite_json(metadata_raw, label=metadata_path.name)
    commit = _read_finite_json(commit_raw, label=commit_path.name)
    if not isinstance(metadata, dict) or not isinstance(commit, dict):
        raise PrecisionMonitorError("seed-block metadata/commit must be objects")
    if set(metadata) != BLOCK_METADATA_FIELDS or set(commit) != BLOCK_COMMIT_FIELDS:
        raise PrecisionMonitorError("seed-block sidecar fields changed")
    try:
        logical = zstd.ZstdDecompressor().decompress(compressed)
    except zstd.ZstdError as exc:
        raise PrecisionMonitorError("seed-block index is not valid zstd") from exc
    payload = _read_finite_json(logical, label=path.name)
    if not isinstance(payload, dict) or logical != _canonical_json_bytes(payload):
        raise PrecisionMonitorError("seed-block index is not canonical JSON")
    block = payload.get("block")
    if not isinstance(block, Mapping):
        raise PrecisionMonitorError("seed-block descriptor is absent")
    block_id = str(block.get("block_id"))
    descriptor = _block_descriptor(block_id)
    expected_envelope = {
        "schema_version": 1, "status": BLOCK_STATUS,
        "artifact_class": BLOCK_ARTIFACT_CLASS,
        "storage_model": "authenticated_index_over_immutable_shards",
        "block": descriptor, "canonical_sort_key": [
            "block_id", "policy", "assumed_tox_noise_sd_factor", "gamma", "stratum", "seed"
        ],
        "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
    }
    for field, expected in expected_envelope.items():
        if payload.get(field) != expected:
            raise PrecisionMonitorError(f"seed-block payload mismatch: {field}")
    if set(payload) != {
        "schema_version", "status", "artifact_class", "storage_model", "block",
        "bindings", "canonical_sort_key", "shard_commits", "path_count", "snapshot_count",
    }:
        raise PrecisionMonitorError("seed-block payload fields changed")
    bindings = payload.get("bindings")
    _validate_bindings(bindings, verify_local=verify_local_bindings)
    for field, expected in {
        "schema_version": 1, "status": BLOCK_STATUS,
        "artifact_class": BLOCK_ARTIFACT_CLASS,
        "storage_model": "authenticated_index_over_immutable_shards",
        "artifact": path.name, "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed), "uncompressed_sha256": _sha256_bytes(logical),
        "uncompressed_bytes": len(logical), "block_id": block_id,
        "seed_start": descriptor["seed_start"], "seed_stop_exclusive": descriptor["seed_stop_exclusive"],
        "path_count": descriptor["path_count"], "snapshot_count": descriptor["snapshot_count"],
        "shard_count": 24, "bindings": dict(bindings),
        "compression": COMPRESSION_CONTRACT,
    }.items():
        if metadata.get(field) != expected:
            raise PrecisionMonitorError(f"seed-block metadata mismatch: {field}")
    for field, expected in {
        "schema_version": 1, "status": BLOCK_COMMIT_STATUS,
        "artifact": path.name, "artifact_sha256": _sha256_bytes(compressed),
        "uncompressed_sha256": _sha256_bytes(logical),
        "metadata": metadata_path.name, "metadata_sha256": _sha256_bytes(metadata_raw),
        "block_id": block_id, "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"], "shard_count": 24,
        "launch_fingerprint": bindings.get("launch_fingerprint"), "immutable": True,
    }.items():
        if commit.get(field) != expected:
            raise PrecisionMonitorError(f"seed-block commit mismatch: {field}")
    commits = payload.get("shard_commits")
    if not isinstance(commits, list) or len(commits) != 24:
        raise PrecisionMonitorError("seed-block does not index 24 shards")
    expected_descriptors = sorted(
        (_expected_shard_descriptor(block_id, policy, factor, gamma)
         for policy in POLICIES for factor in FACTORS for gamma in GATES),
        key=lambda value: value["shard_id"],
    )
    for item, expected in zip(commits, expected_descriptors, strict=True):
        if not isinstance(item, Mapping) or set(item) != SHARD_INDEX_FIELDS:
            raise PrecisionMonitorError("shard-index fields changed")
        for field in (
            "shard_id", "block_id", "path_count", "snapshot_count",
            "execution_keys_sha256", "first_execution_key", "last_execution_key",
        ):
            if item.get(field) != expected[field]:
                raise PrecisionMonitorError(f"shard-index design mismatch: {field}")
        shard_id = expected["shard_id"]
        expected_paths = {
            "artifact": f"raw/shards/{shard_id}.paths.json.zst",
            "metadata": f"raw/shards/{shard_id}.paths.json.zst.metadata.json",
            "commit": f"raw/shards/{shard_id}.paths.json.zst.commit.json",
        }
        for field, value in expected_paths.items():
            if item.get(field) != value:
                raise PrecisionMonitorError(f"shard-index path changed: {field}")
        if not isinstance(item.get("uncompressed_bytes"), int) or item["uncompressed_bytes"] <= 0:
            raise PrecisionMonitorError("shard-index logical byte count is invalid")
        for field in (
            "artifact_sha256", "uncompressed_sha256", "metadata_sha256", "commit_sha256"
        ):
            _require_sha(item.get(field), label=f"shard_index.{field}")
    provenance = {
        "artifact_sha256": _sha256_bytes(compressed),
        "uncompressed_sha256": _sha256_bytes(logical),
        "uncompressed_bytes": len(logical),
        "metadata_sha256": _sha256_bytes(metadata_raw),
        "commit_sha256": _sha256_bytes(commit_raw),
    }
    return payload, provenance, {"path": path, "metadata": metadata_path, "commit": commit_path}


def iter_authenticated_shard_results(
    staging_root: str | Path,
    shard_commitment: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
    validate_core: bool = True,
):
    """Load, authenticate, and yield one shard's records before releasing it."""

    root = Path(staging_root).resolve()
    artifact = (root / str(shard_commitment["artifact"])).resolve()
    metadata_path = (root / str(shard_commitment["metadata"])).resolve()
    commit_path = (root / str(shard_commitment["commit"])).resolve()
    if not all(path.is_relative_to(root) for path in (artifact, metadata_path, commit_path)):
        raise PrecisionMonitorError("shard commitment escapes the staging root")
    if not all(path.is_file() for path in (artifact, metadata_path, commit_path)):
        raise PrecisionMonitorError("indexed shard envelope is incomplete")
    compressed, metadata_raw, commit_raw = (
        artifact.read_bytes(), metadata_path.read_bytes(), commit_path.read_bytes()
    )
    for field, observed in (
        ("artifact_sha256", _sha256_bytes(compressed)),
        ("metadata_sha256", _sha256_bytes(metadata_raw)),
        ("commit_sha256", _sha256_bytes(commit_raw)),
    ):
        if shard_commitment.get(field) != observed:
            raise PrecisionMonitorError(f"indexed shard {field} differs")
    metadata = _read_finite_json(metadata_raw, label=metadata_path.name)
    commit = _read_finite_json(commit_raw, label=commit_path.name)
    if not isinstance(metadata, dict) or not isinstance(commit, dict):
        raise PrecisionMonitorError("shard metadata/commit must be objects")
    if set(metadata) != SHARD_METADATA_FIELDS or set(commit) != SHARD_COMMIT_FIELDS:
        raise PrecisionMonitorError("shard sidecar fields changed")
    try:
        logical = zstd.ZstdDecompressor().decompress(compressed)
    except zstd.ZstdError as exc:
        raise PrecisionMonitorError("indexed shard is not valid zstd") from exc
    payload = _read_finite_json(logical, label=artifact.name)
    if not isinstance(payload, dict) or logical != _canonical_json_bytes(payload):
        raise PrecisionMonitorError("indexed shard is not canonical JSON")
    if (
        _sha256_bytes(logical) != shard_commitment.get("uncompressed_sha256")
        or len(logical) != shard_commitment.get("uncompressed_bytes")
    ):
        raise PrecisionMonitorError("indexed shard logical commitment differs")
    shard_id = str(shard_commitment["shard_id"])
    parts = shard_id.split("__")
    if len(parts) != 4:
        raise PrecisionMonitorError("indexed shard ID is malformed")
    block_id, policy = parts[0], parts[1]
    factor = float(parts[2].removeprefix("csigma"))
    gamma = float(parts[3].removeprefix("tau"))
    descriptor = _expected_shard_descriptor(block_id, policy, factor, gamma)
    expected_payload = {
        "schema_version": 1, "status": SHARD_STATUS,
        "artifact_class": SHARD_ARTIFACT_CLASS, "shard": descriptor,
        "bindings": dict(bindings), "canonical_sort_key": [
            "block_id", "policy", "assumed_tox_noise_sd_factor", "gamma", "stratum", "seed"
        ], "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
    }
    for field, expected in expected_payload.items():
        if payload.get(field) != expected:
            raise PrecisionMonitorError(f"shard payload mismatch: {field}")
    if set(payload) != {
        "schema_version", "status", "artifact_class", "shard", "bindings",
        "canonical_sort_key", "checkpoint_sha256", "path_count", "snapshot_count", "rows",
    }:
        raise PrecisionMonitorError("shard payload fields changed")
    _require_sha(payload.get("checkpoint_sha256"), label="shard.checkpoint_sha256")
    for field, expected in {
        "schema_version": 1, "status": SHARD_STATUS,
        "artifact_class": SHARD_ARTIFACT_CLASS, "artifact": artifact.name,
        "artifact_sha256": _sha256_bytes(compressed), "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(logical), "uncompressed_bytes": len(logical),
        "shard": descriptor, "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"], "bindings": dict(bindings),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "compression": COMPRESSION_CONTRACT,
    }.items():
        if metadata.get(field) != expected:
            raise PrecisionMonitorError(f"shard metadata mismatch: {field}")
    for field, expected in {
        "schema_version": 1, "status": SHARD_COMMIT_STATUS,
        "artifact": artifact.name, "artifact_sha256": _sha256_bytes(compressed),
        "uncompressed_sha256": _sha256_bytes(logical), "metadata": metadata_path.name,
        "metadata_sha256": _sha256_bytes(metadata_raw), "shard_id": shard_id,
        "block_id": block_id, "path_count": descriptor["path_count"],
        "snapshot_count": descriptor["snapshot_count"],
        "execution_keys_sha256": descriptor["execution_keys_sha256"],
        "launch_fingerprint": bindings.get("launch_fingerprint"), "immutable": True,
    }.items():
        if commit.get(field) != expected:
            raise PrecisionMonitorError(f"shard commit mismatch: {field}")
    rows = payload.get("rows")
    expected_keys = _execution_keys(block_id, policy, factor, gamma)
    if not isinstance(rows, list) or len(rows) != len(expected_keys):
        raise PrecisionMonitorError("shard row count differs")
    core_validator = None
    if validate_core:
        try:
            from paper.budget_toxicity_calibration_core import validate_path_record
        except ModuleNotFoundError:  # Direct execution from paper/.
            from budget_toxicity_calibration_core import validate_path_record  # type: ignore
        core_validator = validate_path_record
    for wrapper, expected_key in zip(rows, expected_keys, strict=True):
        if not isinstance(wrapper, Mapping) or wrapper.get("execution_key") != expected_key:
            raise PrecisionMonitorError("shard row order/identity differs")
        if wrapper.get("block_id") != block_id or not isinstance(wrapper.get("result"), dict):
            raise PrecisionMonitorError("shard wrapper is malformed")
        result = dict(wrapper["result"])
        seed, result_policy, result_gamma, stratum, result_factor = _path_identifiers(result)
        if [block_id, result_policy, result_factor, result_gamma, stratum, seed] != expected_key:
            raise PrecisionMonitorError("shard wrapper/result identity differs")
        if core_validator is not None:
            core_validator(result)
        yield result


def authenticate_cumulative_seed_blocks(
    raw_paths: Sequence[str | Path],
    *,
    hash_root: str | Path | None = None,
    verify_local_bindings: bool = True,
) -> tuple[dict[tuple[int, str, float, int, float, int], dict[str, float]], dict[str, str], int]:
    """Authenticate block/shard indices and stream a four-outcome projection."""

    if not raw_paths:
        raise PrecisionMonitorError("at least one committed seed block is required")
    projected: dict[tuple[int, str, float, int, float, int], dict[str, float]] = {}
    payloads: list[dict[str, Any]] = []
    block_provenance: list[tuple[Path, dict[str, str], dict[str, Any]]] = []
    for candidate in raw_paths:
        path = Path(candidate).resolve()
        payload, provenance, _paths = authenticate_seed_block(
            path, verify_local_bindings=verify_local_bindings
        )
        payloads.append(payload)
        block_provenance.append((path, provenance, payload))
    block_ids = [str(payload["block"]["block_id"]) for payload in payloads]
    allowed_sequences = {
        200: ["M0200"],
        500: ["M0200", "M0500_TOPUP"],
        1000: ["M0200", "M0500_TOPUP", "M1000_TOPUP"],
    }
    final_m = max(SEED_BLOCKS[block_id][1] for block_id in block_ids)
    if block_ids != allowed_sequences.get(final_m):
        raise PrecisionMonitorError("seed blocks are not one ordered cumulative sequence")
    bindings = [payload.get("bindings") for payload in payloads]
    if not bindings or any(binding != bindings[0] for binding in bindings[1:]):
        raise PrecisionMonitorError("cumulative seed blocks have different execution bindings")
    root = (
        Path(hash_root).resolve() if hash_root is not None
        else Path(raw_paths[0]).resolve().parent.parent
    )
    hashes: dict[str, str] = {}
    for path, provenance, payload in block_provenance:
        if path.parent.parent != root:
            raise PrecisionMonitorError("cumulative block indices do not share one staging root")
        metadata_path, commit_path = _sidecar_paths(path)
        for file_path, kind in (
            (path, "artifact_sha256"),
            (metadata_path, "metadata_sha256"),
            (commit_path, "commit_sha256"),
        ):
            try:
                label = file_path.relative_to(root).as_posix()
            except ValueError as exc:
                raise PrecisionMonitorError("seed block lies outside --hash-root") from exc
            if label in hashes:
                raise PrecisionMonitorError("input-hash path labels are not unique")
            hashes[label] = provenance[kind]
        for shard in payload["shard_commits"]:
            for field, hash_field in (
                ("artifact", "artifact_sha256"),
                ("metadata", "metadata_sha256"),
                ("commit", "commit_sha256"),
            ):
                label = str(shard[field])
                if label in hashes or not (root / label).is_file():
                    raise PrecisionMonitorError("shard input-hash path is absent or duplicated")
                hashes[label] = str(shard[hash_field])
            for record in iter_authenticated_shard_results(
                root, shard, bindings=payload["bindings"], validate_core=True
            ):
                seed, policy, gamma, stratum, factor = _path_identifiers(record)
                for budget, snapshot in _snapshots(record).items():
                    key = (seed, policy, gamma, stratum, factor, budget)
                    if key in projected:
                        raise PrecisionMonitorError("streamed precision key is duplicated")
                    projected[key] = {
                        OUTCOME_TERMINAL: _outcome(snapshot, OUTCOME_TERMINAL),
                        OUTCOME_ASSIGNMENT: _outcome(snapshot, OUTCOME_ASSIGNMENT),
                        OUTCOME_EXCLUSION: _outcome(snapshot, OUTCOME_EXCLUSION),
                        OUTCOME_BRIER: _outcome(snapshot, OUTCOME_BRIER),
                    }
    hashes["paper/budget_toxicity_calibration_manifest.json"] = _sha256_file(
        MANIFEST_PATH
    )
    hashes["paper/budget_toxicity_calibration_prespec.md"] = _sha256_file(
        PRESPEC_PATH
    )
    expected = final_m * len(POLICIES) * len(GATES) * len(STRATA) * len(FACTORS) * len(BUDGETS)
    if len(projected) != expected or {key[0] for key in projected} != set(range(final_m)):
        raise PrecisionMonitorError("streamed precision projection is not the full factorial")
    return projected, hashes, final_m


def m0200_input_hashes(
    cumulative_hashes: Mapping[str, str],
) -> dict[str, str]:
    """Select the exact 77 raw/design hashes bound by the baseline PASS."""

    design = {
        "paper/budget_toxicity_calibration_manifest.json",
        "paper/budget_toxicity_calibration_prespec.md",
    }
    output = {
        str(label): str(digest)
        for label, digest in cumulative_hashes.items()
        if str(label) in design or "M0200" in Path(str(label)).name
    }
    if len(output) != 77 or not design.issubset(output):
        raise PrecisionMonitorError("cumulative inputs do not contain exact M0200 bindings")
    for label, digest in output.items():
        _require_sha(digest, label=f"m0200_input_hashes.{label}")
    return output


def _add_baseline_pass_hashes(
    hashes: dict[str, str],
    baseline_identity_path: str | Path,
    *,
    hash_root: str | Path,
) -> None:
    """Authenticate the pre-unblinding gate and append its committed trio."""

    try:
        from paper.analyze_budget_toxicity_calibration_audit import (
            load_baseline_identity_pass,
        )
    except ModuleNotFoundError:  # Direct execution from paper/.
        from analyze_budget_toxicity_calibration_audit import (  # type: ignore
            load_baseline_identity_pass,
        )
    load_baseline_identity_pass(
        baseline_identity_path,
        expected_input_hashes=m0200_input_hashes(hashes),
        expected_m=200,
    )
    root = Path(hash_root).resolve()
    artifact = Path(baseline_identity_path).resolve()
    metadata_path, commit_path = _sidecar_paths(artifact)
    for path in (artifact, metadata_path, commit_path):
        try:
            label = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise PrecisionMonitorError(
                "baseline PASS artifact is outside --hash-root"
            ) from exc
        if label in hashes:
            raise PrecisionMonitorError("baseline PASS hash label is duplicated")
        hashes[label] = _sha256_file(path)


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise PrecisionMonitorError(f"{label} must be an integer")
    try:
        output = int(value)
    except (TypeError, ValueError) as exc:
        raise PrecisionMonitorError(f"{label} must be an integer") from exc
    if output != value:
        raise PrecisionMonitorError(f"{label} must be an exact integer")
    return output


def _nested(record: Mapping[str, Any], *keys: str) -> Any:
    for container_name in (None, "identifiers", "design", "noise"):
        container: Mapping[str, Any]
        if container_name is None:
            container = record
        else:
            value = record.get(container_name)
            if not isinstance(value, Mapping):
                continue
            container = value
        for key in keys:
            if key in container:
                return container[key]
    return None


def _path_identifiers(record: Mapping[str, Any]) -> tuple[int, str, float, int, float]:
    seed = _integer(_nested(record, "seed"), label="record seed")
    policy = str(_nested(record, "policy"))
    gamma = float(_nested(record, "gamma", "tau"))
    stratum = _integer(_nested(record, "stratum"), label="record stratum")
    factor = float(
        _nested(
            record,
            "assumed_toxicity_noise_sd_factor",
            "assumed_toxicity_sd_factor",
            "assumed_tox_noise_sd_factor",
            "toxicity_noise_sd_factor",
        )
    )
    if policy not in POLICIES or gamma not in GATES or stratum not in STRATA or factor not in FACTORS:
        raise PrecisionMonitorError(
            "raw path key falls outside the frozen policy/gate/stratum/factor grid"
        )
    return seed, policy, gamma, stratum, factor


def _snapshots(record: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw = record.get("snapshots")
    if isinstance(raw, Mapping):
        candidates = list(raw.values())
    elif isinstance(raw, list):
        candidates = raw
    else:
        raise PrecisionMonitorError("raw path lacks its three snapshots")
    output: dict[int, Mapping[str, Any]] = {}
    for snapshot in candidates:
        if not isinstance(snapshot, Mapping):
            raise PrecisionMonitorError("snapshot must be an object")
        budget = _integer(snapshot.get("budget", snapshot.get("n")), label="snapshot budget")
        if budget in output:
            raise PrecisionMonitorError(f"duplicate nested snapshot at N={budget}")
        output[budget] = snapshot
    if set(output) != set(BUDGETS):
        raise PrecisionMonitorError("raw path does not contain exactly N=20,40,80")
    return output


def _outcome(snapshot: Mapping[str, Any], canonical: str) -> float:
    containers: list[Mapping[str, Any]] = [snapshot]
    for name in ("outcomes", "metrics", "derived_metrics"):
        value = snapshot.get(name)
        if isinstance(value, Mapping):
            containers.insert(0, value)
    found: list[Any] = []
    for container in containers:
        for alias in _OUTCOME_ALIASES[canonical]:
            if alias in container:
                found.append(container[alias])
    if not found:
        raise PrecisionMonitorError(f"snapshot lacks precision outcome {canonical}")
    values = [float(value) for value in found]
    if not all(math.isfinite(value) for value in values) or not all(
        abs(value - values[0]) <= 1e-10 for value in values[1:]
    ):
        raise PrecisionMonitorError(f"snapshot has inconsistent values for {canonical}")
    value = values[0]
    if canonical == OUTCOME_TERMINAL and value in (0.0, 1.0):
        # The canonical field is percentage-scaled.  A value of one is therefore
        # valid only if explicitly stored under a percentage alias; do not guess.
        pass
    if not 0.0 <= value <= 100.0:
        raise PrecisionMonitorError(f"precision outcome {canonical} is outside 0--100")
    return value


def precision_projection(
    records: Sequence[Mapping[str, Any]],
    *,
    final_m: int | None = None,
) -> dict[tuple[int, str, float, int, float, int], dict[str, float]]:
    """Project only identifiers and the four frozen precision outcomes."""

    projected: dict[tuple[int, str, float, int, float, int], dict[str, float]] = {}
    seeds: set[int] = set()
    for record in records:
        seed, policy, gamma, stratum, factor = _path_identifiers(record)
        seeds.add(seed)
        for budget, snapshot in _snapshots(record).items():
            key = (seed, policy, gamma, stratum, factor, budget)
            if key in projected:
                raise PrecisionMonitorError(f"duplicate physical/snapshot key {key!r}")
            projected[key] = {
                OUTCOME_TERMINAL: _outcome(snapshot, OUTCOME_TERMINAL),
                OUTCOME_ASSIGNMENT: _outcome(snapshot, OUTCOME_ASSIGNMENT),
                OUTCOME_EXCLUSION: _outcome(snapshot, OUTCOME_EXCLUSION),
                OUTCOME_BRIER: _outcome(snapshot, OUTCOME_BRIER),
            }
    observed_m = len(seeds)
    if final_m is not None and observed_m != final_m:
        raise PrecisionMonitorError(
            f"precision projection has {observed_m} seeds, expected M={final_m}"
        )
    if seeds != set(range(observed_m)) or observed_m not in ELIGIBLE_M:
        raise PrecisionMonitorError("seed set is not one frozen cumulative block")
    expected = observed_m * len(POLICIES) * len(GATES) * len(STRATA) * len(FACTORS) * len(BUDGETS)
    if len(projected) != expected:
        raise PrecisionMonitorError(
            f"precision projection has {len(projected)} snapshots, expected {expected}"
        )
    return projected


def _factor_label(value: float) -> str:
    return {0.5: "0p5", 1.0: "1", 2.0: "2"}[float(value)]


def primary_estimand_ids() -> list[str]:
    """Return the exact ordered 54+48+16 locked estimand identifiers."""

    output: list[str] = []
    for comparator in COMPARATORS:
        for outcome in (OUTCOME_TERMINAL, OUTCOME_ASSIGNMENT):
            for budget in BUDGETS:
                for factor in FACTORS:
                    output.append(
                        f"A|cKG-minus-{comparator}|{outcome}|N={budget}|c={_factor_label(factor)}"
                    )
    for policy in POLICIES:
        for budget in BUDGETS:
            for factor in ERRONEOUS_FACTORS:
                for outcome in (OUTCOME_EXCLUSION, OUTCOME_BRIER):
                    output.append(
                        f"B|{policy}|{outcome}|N={budget}|c={_factor_label(factor)}-minus-1"
                    )
    for policy in POLICIES:
        for factor in ERRONEOUS_FACTORS:
            for outcome in (OUTCOME_EXCLUSION, OUTCOME_BRIER):
                output.append(
                    f"C|{policy}|{outcome}|N80-minus-N20|c={_factor_label(factor)}-minus-1"
                )
    if len(output) != 118 or len(set(output)) != 118:
        raise AssertionError("locked precision family is not 118 unique estimands")
    return output


_LOCKED_PRIMARY_ESTIMANDS = frozenset(primary_estimand_ids())


def numerical_equivalence_guard_metadata() -> dict[str, Any]:
    """Return the exact, result-independent numerical-guard contract."""

    return {
        "schema_version": 1,
        "eta": NUMERICAL_EQUIVALENCE_ETA,
        "eta_formula": "2^-26 = sqrt(binary64_epsilon)",
        "panel_brier_snapshot_bound_points": (
            PANEL_BRIER_SNAPSHOT_BOUND_POINTS
        ),
        "panel_brier_snapshot_bound_formula": "200 * eta",
        "family_B_brier_replicate_bound_points": (
            FAMILY_B_BRIER_REPLICATE_BOUND_POINTS
        ),
        "family_B_brier_replicate_bound_formula": "400 * eta",
        "family_C_brier_interaction_bound_points": (
            FAMILY_C_BRIER_INTERACTION_BOUND_POINTS
        ),
        "family_C_brier_interaction_bound_formula": "800 * eta",
        "mcse_guard_formula": (
            "replicate_bound_points / sqrt(cumulative_M - 1)"
        ),
        "guarded_mcse_formula": "raw_mcse_points + mcse_guard_points",
        "threshold_points": MCSE_THRESHOLD_POINTS,
        "threshold_rule": (
            "all guarded_mcse_points <= threshold_points; top up the whole "
            "factorial from M=200 to 500 or from M=500 to 1000; at M=1000 "
            "stop and retain any rows over threshold"
        ),
        "whole_factorial_top_up": [
            {"from_M": 200, "to_M": 500},
            {"from_M": 500, "to_M": 1000},
        ],
        "maximum_M": 1000,
        "row_guard_classes": [
            {
                "row_guard_class": GUARD_CLASS_EXACT_DISCRETE,
                "applies_to": "all locked non-Brier primary rows",
                "replicate_bound_points": 0.0,
            },
            {
                "row_guard_class": GUARD_CLASS_FAMILY_B_BRIER,
                "applies_to": "Family B panel-Brier differences",
                "replicate_bound_points": (
                    FAMILY_B_BRIER_REPLICATE_BOUND_POINTS
                ),
            },
            {
                "row_guard_class": GUARD_CLASS_FAMILY_C_BRIER,
                "applies_to": (
                    "Family C panel-Brier differences-in-differences"
                ),
                "replicate_bound_points": (
                    FAMILY_C_BRIER_INTERACTION_BOUND_POINTS
                ),
            },
        ],
    }


def row_numerical_guard(
    estimand_id: str,
    cumulative_m: int,
) -> dict[str, Any]:
    """Return the frozen guard class and bounds for one locked primary row."""

    if estimand_id not in _LOCKED_PRIMARY_ESTIMANDS:
        raise PrecisionMonitorError("numerical guard received an unlocked estimand")
    if isinstance(cumulative_m, bool) or cumulative_m not in ELIGIBLE_M:
        raise PrecisionMonitorError("numerical guard received an ineligible M")
    if (
        estimand_id.startswith("B|")
        and f"|{OUTCOME_BRIER}|" in estimand_id
    ):
        guard_class = GUARD_CLASS_FAMILY_B_BRIER
        replicate_bound = FAMILY_B_BRIER_REPLICATE_BOUND_POINTS
    elif (
        estimand_id.startswith("C|")
        and f"|{OUTCOME_BRIER}|" in estimand_id
    ):
        guard_class = GUARD_CLASS_FAMILY_C_BRIER
        replicate_bound = FAMILY_C_BRIER_INTERACTION_BOUND_POINTS
    else:
        guard_class = GUARD_CLASS_EXACT_DISCRETE
        replicate_bound = 0.0
    mcse_guard = replicate_bound / math.sqrt(cumulative_m - 1)
    return {
        "row_guard_class": guard_class,
        "replicate_bound_points": replicate_bound,
        "mcse_guard_points": mcse_guard,
    }


def _aggregated_value(
    projected: Mapping[tuple[int, str, float, int, float, int], Mapping[str, float]],
    *,
    seed: int,
    policy: str,
    factor: float,
    budget: int,
    outcome: str,
) -> float:
    values = [
        projected[(seed, policy, gamma, stratum, factor, budget)][outcome]
        for gamma in GATES
        for stratum in STRATA
    ]
    return float(math.fsum(values) / 4.0)


def primary_replicate_vectors(
    projected: Mapping[tuple[int, str, float, int, float, int], Mapping[str, float]],
) -> tuple[list[str], np.ndarray]:
    """Build the paired seed vector matrix in locked estimand order."""

    seeds = sorted({key[0] for key in projected})
    ids = primary_estimand_ids()
    columns: list[np.ndarray] = []

    for comparator in COMPARATORS:
        for outcome in (OUTCOME_TERMINAL, OUTCOME_ASSIGNMENT):
            for budget in BUDGETS:
                for factor in FACTORS:
                    columns.append(np.asarray([
                        _aggregated_value(
                            projected, seed=seed, policy="cKG-exact-formal",
                            factor=factor, budget=budget, outcome=outcome,
                        )
                        - _aggregated_value(
                            projected, seed=seed, policy=comparator,
                            factor=factor, budget=budget, outcome=outcome,
                        )
                        for seed in seeds
                    ], dtype=float))

    for policy in POLICIES:
        for budget in BUDGETS:
            for factor in ERRONEOUS_FACTORS:
                for outcome in (OUTCOME_EXCLUSION, OUTCOME_BRIER):
                    columns.append(np.asarray([
                        _aggregated_value(
                            projected, seed=seed, policy=policy, factor=factor,
                            budget=budget, outcome=outcome,
                        )
                        - _aggregated_value(
                            projected, seed=seed, policy=policy, factor=1.0,
                            budget=budget, outcome=outcome,
                        )
                        for seed in seeds
                    ], dtype=float))

    for policy in POLICIES:
        for factor in ERRONEOUS_FACTORS:
            for outcome in (OUTCOME_EXCLUSION, OUTCOME_BRIER):
                columns.append(np.asarray([
                    (
                        _aggregated_value(
                            projected, seed=seed, policy=policy, factor=factor,
                            budget=80, outcome=outcome,
                        )
                        - _aggregated_value(
                            projected, seed=seed, policy=policy, factor=1.0,
                            budget=80, outcome=outcome,
                        )
                    )
                    - (
                        _aggregated_value(
                            projected, seed=seed, policy=policy, factor=factor,
                            budget=20, outcome=outcome,
                        )
                        - _aggregated_value(
                            projected, seed=seed, policy=policy, factor=1.0,
                            budget=20, outcome=outcome,
                        )
                    )
                    for seed in seeds
                ], dtype=float))

    matrix = np.column_stack(columns)
    if matrix.shape != (len(seeds), 118) or not np.isfinite(matrix).all():
        raise PrecisionMonitorError("primary replicate-vector matrix is invalid")
    return ids, matrix


def blind_precision_decision(
    records: Sequence[Mapping[str, Any]],
    *,
    final_m: int | None = None,
    input_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Calculate only MCSE information allowed by the frozen blind rule."""

    projection = precision_projection(records, final_m=final_m)
    return blind_precision_decision_from_projection(
        projection, final_m=final_m, input_hashes=input_hashes
    )


def blind_precision_decision_from_projection(
    projection: Mapping[tuple[int, str, float, int, float, int], Mapping[str, float]],
    *,
    final_m: int | None = None,
    input_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Calculate the blind decision from a streamed four-outcome projection."""

    ids, matrix = primary_replicate_vectors(projection)
    cumulative_m = matrix.shape[0]
    if final_m is not None and cumulative_m != final_m:
        raise PrecisionMonitorError("streamed projection has the wrong cumulative M")
    mcse = np.std(matrix, axis=0, ddof=1) / math.sqrt(cumulative_m)
    if not np.isfinite(mcse).all() or np.any(mcse < 0):
        raise PrecisionMonitorError("one or more paired MCSEs are invalid")
    estimand_rows: list[dict[str, Any]] = []
    for estimand_id, raw_mcse in zip(ids, mcse, strict=True):
        guard = row_numerical_guard(estimand_id, cumulative_m)
        guarded_mcse = float(raw_mcse) + float(guard["mcse_guard_points"])
        estimand_rows.append({
            "estimand_id": estimand_id,
            "cumulative_M": cumulative_m,
            **guard,
            "raw_mcse_points": float(raw_mcse),
            "guarded_mcse_points": guarded_mcse,
            "passes_guarded_1_5pp": bool(
                guarded_mcse <= MCSE_THRESHOLD_POINTS
            ),
        })
    maximum_raw = max(float(row["raw_mcse_points"]) for row in estimand_rows)
    maximum_guarded = max(
        float(row["guarded_mcse_points"]) for row in estimand_rows
    )
    all_pass = all(bool(row["passes_guarded_1_5pp"]) for row in estimand_rows)
    if cumulative_m == 200 and not all_pass:
        next_m: int | None = 500
    elif cumulative_m == 500 and not all_pass:
        next_m = 1000
    else:
        next_m = None
    hashes = dict(input_hashes or {})
    for field, value in hashes.items():
        _require_sha(value, label=f"input_hashes.{field}")
    return {
        "schema_version": 2,
        "status": PRECISION_STATUS,
        "artifact_class": PRECISION_ARTIFACT_CLASS,
        "cumulative_M": cumulative_m,
        "numerical_equivalence_guard": numerical_equivalence_guard_metadata(),
        "estimands": estimand_rows,
        "maximum_raw_mcse_points": maximum_raw,
        "maximum_guarded_mcse_points": maximum_guarded,
        "all_estimands_pass_guarded_1_5pp": all_pass,
        "top_up_limit_reached": bool(cumulative_m == 1000 and not all_pass),
        "next_M": next_m,
        "input_hashes": hashes,
    }


_PROHIBITED_PRECISION_TOKENS = (
    "estimate", "mean", "effect", "sign", "interval", "ci", "pvalue",
    "p_value", "rank", "winner", "lower", "upper",
)


def validate_blind_payload(payload: Mapping[str, Any]) -> None:
    """Reject additions capable of unblinding the precision decision."""

    allowed_top = {
        "schema_version", "status", "artifact_class", "cumulative_M",
        "numerical_equivalence_guard", "estimands",
        "maximum_raw_mcse_points", "maximum_guarded_mcse_points",
        "all_estimands_pass_guarded_1_5pp", "top_up_limit_reached",
        "next_M", "input_hashes",
    }
    if set(payload) != allowed_top:
        raise PrecisionMonitorError("blind payload contains a non-approved top-level field")
    if payload.get("schema_version") != 2 or payload.get("status") != PRECISION_STATUS:
        raise PrecisionMonitorError("blind payload envelope changed")
    if payload.get("artifact_class") != PRECISION_ARTIFACT_CLASS:
        raise PrecisionMonitorError("blind payload artifact class changed")
    cumulative_m = _integer(payload.get("cumulative_M"), label="cumulative_M")
    if cumulative_m not in ELIGIBLE_M:
        raise PrecisionMonitorError("blind payload cumulative M is not eligible")
    guard_metadata = payload.get("numerical_equivalence_guard")
    if (
        not isinstance(guard_metadata, Mapping)
        or _canonical_json_bytes(dict(guard_metadata))
        != _canonical_json_bytes(numerical_equivalence_guard_metadata())
    ):
        raise PrecisionMonitorError("blind numerical-guard contract changed")
    estimands = payload.get("estimands")
    if not isinstance(estimands, list) or len(estimands) != 118:
        raise PrecisionMonitorError("blind payload is not the 118-estimand family")
    if [row.get("estimand_id") for row in estimands if isinstance(row, dict)] != primary_estimand_ids():
        raise PrecisionMonitorError("blind payload estimand order changed")
    for row in estimands:
        if not isinstance(row, dict) or set(row) != {
            "estimand_id", "cumulative_M", "row_guard_class",
            "replicate_bound_points", "raw_mcse_points", "mcse_guard_points",
            "guarded_mcse_points", "passes_guarded_1_5pp",
        }:
            raise PrecisionMonitorError("blind estimand row contains a non-approved field")
        if row["cumulative_M"] != cumulative_m:
            raise PrecisionMonitorError("blind estimand cumulative M differs")
        expected_guard = row_numerical_guard(row["estimand_id"], cumulative_m)
        if (
            type(row["row_guard_class"]) is not str
            or type(row["replicate_bound_points"]) is not float
            or type(row["mcse_guard_points"]) is not float
            or any(row[field] != expected_guard[field] for field in expected_guard)
        ):
            raise PrecisionMonitorError("blind estimand numerical guard changed")
        if type(row["raw_mcse_points"]) is not float:
            raise PrecisionMonitorError("blind estimand MCSE is not a float")
        raw_mcse = row["raw_mcse_points"]
        if not math.isfinite(raw_mcse) or raw_mcse < 0.0:
            raise PrecisionMonitorError("blind estimand MCSE is invalid")
        expected_guarded = raw_mcse + float(expected_guard["mcse_guard_points"])
        if (
            type(row["guarded_mcse_points"]) is not float
            or row["guarded_mcse_points"] != expected_guarded
        ):
            raise PrecisionMonitorError("blind guarded MCSE is inconsistent")
        if (
            type(row["passes_guarded_1_5pp"]) is not bool
            or row["passes_guarded_1_5pp"] != (
                expected_guarded <= MCSE_THRESHOLD_POINTS
            )
        ):
            raise PrecisionMonitorError("blind estimand threshold flag is invalid")
    if type(payload.get("maximum_raw_mcse_points")) is not float:
        raise PrecisionMonitorError("blind maximum raw MCSE is not a float")
    maximum_raw = payload["maximum_raw_mcse_points"]
    expected_maximum_raw = max(float(row["raw_mcse_points"]) for row in estimands)
    if not math.isfinite(maximum_raw) or maximum_raw != expected_maximum_raw:
        raise PrecisionMonitorError("blind maximum raw MCSE is inconsistent")
    if type(payload.get("maximum_guarded_mcse_points")) is not float:
        raise PrecisionMonitorError("blind maximum guarded MCSE is not a float")
    maximum_guarded = payload["maximum_guarded_mcse_points"]
    expected_maximum_guarded = max(
        float(row["guarded_mcse_points"]) for row in estimands
    )
    if (
        not math.isfinite(maximum_guarded)
        or maximum_guarded != expected_maximum_guarded
    ):
        raise PrecisionMonitorError("blind maximum guarded MCSE is inconsistent")
    expected_all_pass = all(
        bool(row["passes_guarded_1_5pp"]) for row in estimands
    )
    if (
        type(payload.get("all_estimands_pass_guarded_1_5pp")) is not bool
        or payload["all_estimands_pass_guarded_1_5pp"] is not expected_all_pass
    ):
        raise PrecisionMonitorError("blind all-estimand threshold flag is invalid")
    expected_limit = cumulative_m == 1000 and not expected_all_pass
    if (
        type(payload.get("top_up_limit_reached")) is not bool
        or payload["top_up_limit_reached"] is not expected_limit
    ):
        raise PrecisionMonitorError("blind top-up limit flag is invalid")
    expected_next = (
        500 if cumulative_m == 200 and not expected_all_pass
        else 1000 if cumulative_m == 500 and not expected_all_pass
        else None
    )
    if payload.get("next_M") != expected_next:
        raise PrecisionMonitorError("blind next-M decision violates the frozen rule")
    hashes = payload.get("input_hashes")
    if not isinstance(hashes, dict):
        raise PrecisionMonitorError("blind input hashes are absent")
    for field, value in hashes.items():
        _require_sha(value, label=f"input_hashes.{field}")

    def mapping_keys(value: Any) -> Iterable[str]:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).lower()
                yield normalized
                # Input-hash labels are authenticated file names, not output
                # fields, and may legitimately contain substrings such as
                # "toxicity" that overlap the conservative field-name denylist.
                if normalized != "input_hashes":
                    yield from mapping_keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from mapping_keys(child)

    serialized_keys = " ".join(mapping_keys(payload))
    if any(token in serialized_keys for token in _PROHIBITED_PRECISION_TOKENS):
        raise PrecisionMonitorError("blind payload contains a prohibited result field")


def _atomic_write(path: Path, value: bytes) -> None:
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
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_blind_precision_decision(
    path: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Path]:
    """Atomically write a compressed decision plus authenticated sidecars."""

    _assert_exact_runtime()
    validate_blind_payload(payload)
    target = Path(path).resolve()
    if target.suffixes[-2:] != [".json", ".zst"]:
        raise PrecisionMonitorError("precision artifact must end in .json.zst")
    metadata_path, commit_path = _sidecar_paths(target)
    existence = tuple(
        candidate.is_file() for candidate in (target, metadata_path, commit_path)
    )
    if any(existence):
        if not all(existence):
            raise PrecisionMonitorError(
                "precision decision has a partial immutable envelope"
            )
        existing = load_blind_precision_decision(
            target, expected_input_hashes=payload["input_hashes"],
            expected_m=int(payload["cumulative_M"]),
        )
        if existing != dict(payload):
            raise PrecisionMonitorError(
                "committed precision decision differs; immutable overwrite refused"
            )
        return {"artifact": target, "metadata": metadata_path, "commit": commit_path}
    logical = _canonical_json_bytes(dict(payload))
    compressed = zstd.ZstdCompressor(
        level=19, threads=0, write_checksum=True, write_content_size=True,
        write_dict_id=False,
    ).compress(logical)
    metadata = {
        "schema_version": 2,
        "status": PRECISION_STATUS,
        "artifact_class": PRECISION_ARTIFACT_CLASS,
        "artifact": target.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(logical),
        "uncompressed_bytes": len(logical),
        "cumulative_M": payload["cumulative_M"],
        "estimand_count": 118,
        "numerical_equivalence_guard": payload["numerical_equivalence_guard"],
        "maximum_raw_mcse_points": payload["maximum_raw_mcse_points"],
        "maximum_guarded_mcse_points": payload["maximum_guarded_mcse_points"],
        "all_estimands_pass_guarded_1_5pp": payload[
            "all_estimands_pass_guarded_1_5pp"
        ],
        "top_up_limit_reached": payload["top_up_limit_reached"],
        "next_M": payload["next_M"],
        "input_hashes": dict(payload["input_hashes"]),
    }
    metadata_raw = _canonical_json_bytes(metadata)
    commit = {
        "schema_version": 2,
        "status": PRECISION_COMMIT_STATUS,
        "artifact": target.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256_bytes(metadata_raw),
        "cumulative_M": payload["cumulative_M"],
        "estimand_count": 118,
        "maximum_guarded_mcse_points": payload["maximum_guarded_mcse_points"],
        "all_estimands_pass_guarded_1_5pp": payload[
            "all_estimands_pass_guarded_1_5pp"
        ],
        "top_up_limit_reached": payload["top_up_limit_reached"],
        "next_M": payload["next_M"],
    }
    _atomic_write(target, compressed)
    _atomic_write(metadata_path, metadata_raw)
    _atomic_write(commit_path, _canonical_json_bytes(commit))
    return {"artifact": target, "metadata": metadata_path, "commit": commit_path}


def load_blind_precision_decision(
    path: str | Path,
    *,
    expected_input_hashes: Mapping[str, str] | None = None,
    expected_m: int | None = None,
) -> dict[str, Any]:
    """Authenticate a committed blind decision for a runner or analyzer."""

    target = Path(path).resolve()
    metadata_path, commit_path = _sidecar_paths(target)
    if not all(candidate.is_file() for candidate in (target, metadata_path, commit_path)):
        raise PrecisionMonitorError("precision decision lacks its metadata/commit sidecars")
    compressed = target.read_bytes()
    metadata_raw = metadata_path.read_bytes()
    metadata = _read_finite_json(metadata_raw, label=metadata_path.name)
    commit = _read_finite_json(commit_path.read_bytes(), label=commit_path.name)
    if not isinstance(metadata, dict) or not isinstance(commit, dict):
        raise PrecisionMonitorError("precision metadata/commit must be objects")
    if set(metadata) != {
        "schema_version", "status", "artifact_class", "artifact",
        "artifact_sha256", "artifact_bytes", "uncompressed_sha256",
        "uncompressed_bytes", "cumulative_M", "estimand_count",
        "numerical_equivalence_guard", "maximum_raw_mcse_points",
        "maximum_guarded_mcse_points", "all_estimands_pass_guarded_1_5pp",
        "top_up_limit_reached", "next_M", "input_hashes",
    }:
        raise PrecisionMonitorError("precision metadata fields changed")
    if set(commit) != {
        "schema_version", "status", "artifact", "artifact_sha256", "metadata",
        "metadata_sha256", "cumulative_M", "estimand_count",
        "maximum_guarded_mcse_points", "all_estimands_pass_guarded_1_5pp",
        "top_up_limit_reached", "next_M",
    }:
        raise PrecisionMonitorError("precision commit fields changed")
    if (
        commit.get("schema_version") != 2
        or commit.get("status") != PRECISION_COMMIT_STATUS
    ):
        raise PrecisionMonitorError("precision commit status is not complete")
    for field, expected in {
        "artifact": target.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256_bytes(metadata_raw),
    }.items():
        if commit.get(field) != expected:
            raise PrecisionMonitorError(f"precision commit mismatch: {field}")
    try:
        logical = zstd.ZstdDecompressor().decompress(compressed)
    except zstd.ZstdError as exc:
        raise PrecisionMonitorError("precision decision is not valid zstd data") from exc
    payload = _read_finite_json(logical, label=target.name)
    if not isinstance(payload, dict) or logical != _canonical_json_bytes(payload):
        raise PrecisionMonitorError("precision payload is not canonical JSON")
    validate_blind_payload(payload)
    for field, expected in {
        "schema_version": 2,
        "status": PRECISION_STATUS,
        "artifact_class": PRECISION_ARTIFACT_CLASS,
        "artifact": target.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(logical),
        "uncompressed_bytes": len(logical),
        "cumulative_M": payload["cumulative_M"],
        "estimand_count": 118,
        "numerical_equivalence_guard": payload["numerical_equivalence_guard"],
        "maximum_raw_mcse_points": payload["maximum_raw_mcse_points"],
        "maximum_guarded_mcse_points": payload["maximum_guarded_mcse_points"],
        "all_estimands_pass_guarded_1_5pp": payload[
            "all_estimands_pass_guarded_1_5pp"
        ],
        "top_up_limit_reached": payload["top_up_limit_reached"],
        "next_M": payload["next_M"],
        "input_hashes": payload["input_hashes"],
    }.items():
        if metadata.get(field) != expected:
            raise PrecisionMonitorError(f"precision metadata mismatch: {field}")
    for field in (
        "cumulative_M", "estimand_count", "maximum_guarded_mcse_points",
        "all_estimands_pass_guarded_1_5pp", "top_up_limit_reached", "next_M",
    ):
        if commit.get(field) != metadata.get(field):
            raise PrecisionMonitorError(f"precision commit mismatch: {field}")
    if expected_m is not None and payload["cumulative_M"] != expected_m:
        raise PrecisionMonitorError("precision decision has the wrong cumulative M")
    if expected_input_hashes is not None and payload["input_hashes"] != dict(
        expected_input_hashes
    ):
        raise PrecisionMonitorError("precision decision is bound to different inputs")
    return dict(payload)


def load_precision_decision(
    path: str | Path,
    *,
    expected_raw_hashes: Mapping[str, str] | None = None,
    expected_M: int | None = None,
) -> dict[str, Any]:
    """Runner-facing spelling of :func:`load_blind_precision_decision`."""

    return load_blind_precision_decision(
        path,
        expected_input_hashes=expected_raw_hashes,
        expected_m=expected_M,
    )


def run_monitor(
    raw_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    baseline_identity_path: str | Path,
    hash_root: str | Path | None = None,
) -> dict[str, Path]:
    _assert_exact_runtime()
    projection, hashes, final_m = authenticate_cumulative_seed_blocks(
        raw_paths, hash_root=hash_root
    )
    root = (
        Path(hash_root).resolve() if hash_root is not None
        else Path(raw_paths[0]).resolve().parent.parent
    )
    _add_baseline_pass_hashes(
        hashes, baseline_identity_path, hash_root=root
    )
    decision = blind_precision_decision_from_projection(
        projection, final_m=final_m, input_hashes=hashes
    )
    filename = f"budget_toxicity_calibration_precision_M{final_m:04d}.json.zst"
    return write_blind_precision_decision(Path(output_dir) / filename, decision)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-block", required=True, action="append", type=Path,
        help="committed seed block; repeat in M0200/M0500/M1000 order",
    )
    parser.add_argument(
        "--hash-root", type=Path,
        help="root used to make portable precision input-hash labels",
    )
    parser.add_argument(
        "--baseline-identity", required=True, type=Path,
        help="committed M0200 baseline-identity PASS artifact",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = run_monitor(
        args.raw_block, args.output_dir,
        baseline_identity_path=args.baseline_identity, hash_root=args.hash_root,
    )
    print(f"blind precision decision: {paths['artifact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAMILY_B_BRIER_REPLICATE_BOUND_POINTS",
    "FAMILY_C_BRIER_INTERACTION_BOUND_POINTS",
    "GUARD_CLASS_EXACT_DISCRETE",
    "GUARD_CLASS_FAMILY_B_BRIER",
    "GUARD_CLASS_FAMILY_C_BRIER",
    "MCSE_THRESHOLD_POINTS",
    "NUMERICAL_EQUIVALENCE_ETA",
    "OUTCOME_ASSIGNMENT",
    "OUTCOME_BRIER",
    "OUTCOME_EXCLUSION",
    "OUTCOME_TERMINAL",
    "PANEL_BRIER_SNAPSHOT_BOUND_POINTS",
    "PRECISION_COMMIT_STATUS",
    "PRECISION_STATUS",
    "PrecisionMonitorError",
    "ROW_GUARD_CLASSES",
    "authenticate_cumulative_seed_blocks",
    "authenticate_seed_block",
    "blind_precision_decision",
    "blind_precision_decision_from_projection",
    "load_blind_precision_decision",
    "load_precision_decision",
    "iter_authenticated_shard_results",
    "m0200_input_hashes",
    "numerical_equivalence_guard_metadata",
    "precision_projection",
    "primary_estimand_ids",
    "primary_replicate_vectors",
    "row_numerical_guard",
    "validate_blind_payload",
    "write_blind_precision_decision",
]
