#!/usr/bin/env python3
"""Authenticate and analyze the frozen budget/toxicity-calibration audit.

This is a post-processing program.  It never fits a Gaussian process or runs a
trial.  Every reported operating characteristic is recomputed from serialized
histories and posterior arrays, checked against the runner's saved scalar
fields, and then aggregated with the seed (not the stratum) as the independent
Monte Carlo unit.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import special, stats
import zstandard as zstd

try:
    from paper import monitor_budget_toxicity_calibration_precision as precision
except ModuleNotFoundError:  # Direct execution from paper/.
    import monitor_budget_toxicity_calibration_precision as precision


ROOT = Path(__file__).resolve().parents[1]
PRESPEC_PATH = ROOT / "paper/budget_toxicity_calibration_prespec.md"
MANIFEST_PATH = ROOT / "paper/budget_toxicity_calibration_manifest.json"
MASTER_STATUS = "COMPLETE_BUDGET_TOXICITY_CALIBRATION_RAW_MASTER"
MASTER_COMMIT_STATUS = "COMMITTED_BUDGET_TOXICITY_CALIBRATION_RAW_MASTER"
MASTER_ARTIFACT_CLASS = "authenticated_budget_toxicity_calibration_raw_master"
BASELINE_PASS_STATUS = "PASS_BUDGET_TOXICITY_CALIBRATION_BASELINE_IDENTITY"
BASELINE_PASS_COMMIT_STATUS = (
    "COMMITTED_BUDGET_TOXICITY_CALIBRATION_BASELINE_IDENTITY"
)
BASELINE_PASS_ARTIFACT_CLASS = "authenticated_baseline_identity_pass_only"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

POLICIES = precision.POLICIES
POLICY_LABELS = {
    "cKG-exact-formal": "cKG",
    "tmse": "tMSE-only",
    "qBIG": "Entropy reduction",
    "cEI": "cEI",
}
FACTORS = precision.FACTORS
ERRONEOUS_FACTORS = precision.ERRONEOUS_FACTORS
GATES = precision.GATES
STRATA = precision.STRATA
BUDGETS = precision.BUDGETS
GATE_QUANTILES = {gamma: float(stats.norm.ppf(gamma)) for gamma in GATES}

MAX_T_DRAWS = 100_000
MAX_T_SEEDS = {"A": 202_608_261, "B": 202_608_262, "C": 202_608_263}
CONFIDENCE_LEVEL = 0.95

_VECTOR_ALIASES = {
    "efficacy_mean": ("efficacy_mean", "efficacy_posterior_mean", "mu_efficacy"),
    "efficacy_sd": (
        "efficacy_sd", "efficacy_posterior_sd", "efficacy_posterior_latent_sd",
        "sd_efficacy",
    ),
    "toxicity_mean": ("toxicity_mean", "toxicity_posterior_mean", "mu_toxicity"),
    "toxicity_sd": (
        "toxicity_sd", "toxicity_latent_sd", "toxicity_posterior_sd",
        "toxicity_posterior_latent_sd", "sd_toxicity",
    ),
    "stable_z": ("stable_z", "toxicity_stable_z", "gate_stable_z"),
    "feasibility_probability": (
        "feasibility_probability", "feasibility_probabilities",
        "toxicity_feasibility_probability", "pf", "PF"
    ),
    "gate_mask": ("gate_mask", "strict_gate_mask", "admitted_mask"),
}


class CalibrationAnalysisError(RuntimeError):
    """Raised when authentication, replay, or estimand invariants fail."""


def _assert_exact_runtime() -> dict[str, str]:
    """Use the monitor's single frozen-runtime check for every bound write."""

    try:
        return precision._assert_exact_runtime()
    except precision.PrecisionMonitorError as exc:
        raise CalibrationAnalysisError(str(exc)) from exc


@dataclass(frozen=True)
class RawMaster:
    payload: dict[str, Any]
    provenance: dict[str, Any]
    staging_root: Path
    block_payloads: tuple[dict[str, Any], ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any, *, compact: bool = False) -> bytes:
    if compact:
        text = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
    else:
        text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _read_finite_json(raw: bytes, *, label: str) -> Any:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token}")

    try:
        value = json.loads(raw, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CalibrationAnalysisError(f"{label} is not finite valid JSON") from exc
    _assert_finite(value, label=label)
    return value


def _assert_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise CalibrationAnalysisError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, label=f"{label}[{index}]")


def _sidecars(path: Path) -> tuple[Path, Path]:
    return Path(str(path) + ".metadata.json"), Path(str(path) + ".commit.json")


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise CalibrationAnalysisError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_local_bindings(bindings: Mapping[str, Any]) -> None:
    prespec_sha = _sha256_file(PRESPEC_PATH)
    manifest_sha = _sha256_file(MANIFEST_PATH)
    observed_prespec = bindings.get(
        "prespec_sha256", bindings.get("analysis_spec_sha256")
    )
    observed_manifest = bindings.get(
        "manifest_sha256", bindings.get("design_manifest_sha256")
    )
    if observed_prespec != prespec_sha or observed_manifest != manifest_sha:
        raise CalibrationAnalysisError("raw master design bindings differ locally")
    source_hashes = bindings.get("source_hashes", bindings.get("source_sha256"))
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise CalibrationAnalysisError("raw master source-hash binding is absent")
    for relative, expected in source_hashes.items():
        _require_sha(expected, label=f"source_hashes.{relative}")
        source = ROOT / str(relative)
        if not source.is_file() or _sha256_file(source) != expected:
            raise CalibrationAnalysisError(
                f"raw-bound execution source is absent or changed: {relative}"
            )


def load_raw_master(
    raw_path: str | Path,
    *,
    block_paths: Sequence[str | Path] = (),
    verify_local_bindings: bool = True,
    validate_core: bool = True,
) -> RawMaster:
    """Authenticate the final master and unwrap its physical path records."""

    path = Path(raw_path).resolve()
    metadata_path, commit_path = _sidecars(path)
    if not all(candidate.is_file() for candidate in (path, metadata_path, commit_path)):
        raise CalibrationAnalysisError(
            "raw master requires the zstd artifact and adjacent metadata/commit files"
        )
    compressed = path.read_bytes()
    metadata_raw = metadata_path.read_bytes()
    commit_raw = commit_path.read_bytes()
    metadata = _read_finite_json(metadata_raw, label=metadata_path.name)
    commit = _read_finite_json(commit_raw, label=commit_path.name)
    if not isinstance(metadata, dict) or not isinstance(commit, dict):
        raise CalibrationAnalysisError("master metadata and commit must be objects")
    if set(metadata) != {
        "schema_version", "status", "artifact_class", "storage_model",
        "artifact", "artifact_sha256", "artifact_bytes", "uncompressed_sha256",
        "uncompressed_bytes", "final_M", "path_count", "snapshot_count",
        "bindings", "compression", "environment", "generated_at_utc",
    }:
        raise CalibrationAnalysisError("raw-master metadata fields changed")
    if set(commit) != {
        "schema_version", "status", "artifact", "artifact_sha256",
        "uncompressed_sha256", "metadata", "metadata_sha256", "final_M",
        "path_count", "snapshot_count", "launch_fingerprint", "committed_at_utc",
    }:
        raise CalibrationAnalysisError("raw-master commit fields changed")
    if commit.get("status") != MASTER_COMMIT_STATUS:
        raise CalibrationAnalysisError("raw-master commit status is not complete")
    for field, expected in {
        "artifact": path.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256_bytes(metadata_raw),
    }.items():
        if commit.get(field) != expected:
            raise CalibrationAnalysisError(f"raw-master commit mismatch: {field}")

    try:
        logical = zstd.ZstdDecompressor().decompress(compressed)
    except zstd.ZstdError as exc:
        raise CalibrationAnalysisError("raw master is not valid zstd data") from exc
    payload = _read_finite_json(logical, label=path.name)
    if not isinstance(payload, dict) or logical != _canonical_json_bytes(payload):
        raise CalibrationAnalysisError("raw master is not canonical JSON")
    for envelope, label in ((payload, "payload"), (metadata, "metadata")):
        if envelope.get("schema_version") != 1:
            raise CalibrationAnalysisError(f"master {label} schema version changed")
        if envelope.get("status") != MASTER_STATUS:
            raise CalibrationAnalysisError(f"master {label} status changed")
        if envelope.get("artifact_class") != MASTER_ARTIFACT_CLASS:
            raise CalibrationAnalysisError(f"master {label} artifact class changed")
        if envelope.get("storage_model") != "authenticated_index_over_seed_block_indexes":
            raise CalibrationAnalysisError(f"master {label} storage model changed")
    for field, expected in {
        "artifact": path.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(logical),
        "uncompressed_bytes": len(logical),
    }.items():
        if metadata.get(field) != expected:
            raise CalibrationAnalysisError(f"raw-master metadata mismatch: {field}")
    if metadata.get("compression") != precision.COMPRESSION_CONTRACT:
        raise CalibrationAnalysisError("raw-master compression contract changed")

    final_m = _exact_int(payload.get("final_M"), label="final_M")
    if final_m not in precision.ELIGIBLE_M:
        raise CalibrationAnalysisError("raw-master M is not one frozen cumulative size")
    expected_paths, expected_snapshots = 48 * final_m, 144 * final_m
    if payload.get("path_count") != expected_paths or payload.get("snapshot_count") != expected_snapshots:
        raise CalibrationAnalysisError("raw-master payload counts changed")
    for envelope, label in ((metadata, "metadata"), (commit, "commit")):
        for field, expected in (
            ("final_M", final_m), ("path_count", expected_paths),
            ("snapshot_count", expected_snapshots),
        ):
            if envelope.get(field) != expected:
                raise CalibrationAnalysisError(f"raw-master {label} mismatch: {field}")
    expected_master_fields = {
        "schema_version", "status", "artifact_class", "storage_model", "final_M",
        "cumulative_block_ids", "path_count", "snapshot_count", "bindings",
        "baseline_identity", "block_commits", "precision_decisions",
        "top_up_history", "canonical_sort_key",
    }
    if set(payload) != expected_master_fields or "rows" in payload:
        raise CalibrationAnalysisError("raw-master index fields changed or contain rows")
    if payload.get("storage_model") != "authenticated_index_over_seed_block_indexes":
        raise CalibrationAnalysisError("index-master storage model changed")
    if payload.get("canonical_sort_key") != [
        "block_id", "policy", "assumed_tox_noise_sd_factor", "gamma",
        "stratum", "seed",
    ]:
        raise CalibrationAnalysisError("index-master canonical order changed")
    expected_ids = {
        200: ["M0200"], 500: ["M0200", "M0500_TOPUP"],
        1000: ["M0200", "M0500_TOPUP", "M1000_TOPUP"],
    }[final_m]
    if payload.get("cumulative_block_ids") != expected_ids or len(block_paths) != len(expected_ids):
        raise CalibrationAnalysisError("index-master block sequence/inputs changed")
    staging_root = path.parent.parent
    observed_commits: list[dict[str, Any]] = []
    block_payloads: list[dict[str, Any]] = []
    for block_path, block_id in zip(block_paths, expected_ids, strict=True):
        resolved = Path(block_path).resolve()
        try:
            artifact_relative = resolved.relative_to(staging_root).as_posix()
        except ValueError as exc:
            raise CalibrationAnalysisError("block index is outside the staging root") from exc
        block_payload, hashes, _paths = precision.authenticate_seed_block(
            resolved, verify_local_bindings=verify_local_bindings
        )
        if block_payload["block"]["block_id"] != block_id:
            raise CalibrationAnalysisError("index-master block order changed")
        metadata_block, commit_block = _sidecars(resolved)
        observed_commits.append({
            "block_id": block_id, "artifact": artifact_relative,
            "artifact_sha256": hashes["artifact_sha256"],
            "uncompressed_sha256": hashes["uncompressed_sha256"],
            "uncompressed_bytes": hashes["uncompressed_bytes"],
            "metadata": metadata_block.relative_to(staging_root).as_posix(),
            "metadata_sha256": hashes["metadata_sha256"],
            "commit": commit_block.relative_to(staging_root).as_posix(),
            "commit_sha256": hashes["commit_sha256"],
            "path_count": block_payload["path_count"],
            "snapshot_count": block_payload["snapshot_count"],
            "shard_count": 24,
        })
        block_payloads.append(dict(block_payload))
    if payload.get("block_commits") != observed_commits:
        raise CalibrationAnalysisError("index-master block commitments differ")
    cumulative_raw_hashes = {
        "paper/budget_toxicity_calibration_manifest.json": _sha256_file(MANIFEST_PATH),
        "paper/budget_toxicity_calibration_prespec.md": _sha256_file(PRESPEC_PATH),
    }
    for block_commit, block_payload in zip(
        observed_commits, block_payloads, strict=True
    ):
        for field, hash_field in (
            ("artifact", "artifact_sha256"),
            ("metadata", "metadata_sha256"),
            ("commit", "commit_sha256"),
        ):
            cumulative_raw_hashes[str(block_commit[field])] = str(
                block_commit[hash_field]
            )
        for shard in block_payload["shard_commits"]:
            for field, hash_field in (
                ("artifact", "artifact_sha256"),
                ("metadata", "metadata_sha256"),
                ("commit", "commit_sha256"),
            ):
                cumulative_raw_hashes[str(shard[field])] = str(shard[hash_field])
    baseline_index = payload.get("baseline_identity")
    expected_baseline_fields = {
        "status", "cumulative_M", "pass", "artifact", "artifact_sha256",
        "metadata", "metadata_sha256", "commit", "commit_sha256",
        "checked_common_cells", "checked_allocation_histories",
        "unavailable_allocation_histories",
    }
    if not isinstance(baseline_index, Mapping) or set(baseline_index) != expected_baseline_fields:
        raise CalibrationAnalysisError("raw-master baseline identity index changed")
    baseline_path = (staging_root / str(baseline_index["artifact"])).resolve()
    if not baseline_path.is_relative_to(staging_root):
        raise CalibrationAnalysisError("raw-master baseline identity path escapes staging")
    load_baseline_identity_pass(
        baseline_path,
        expected_input_hashes=precision.m0200_input_hashes(cumulative_raw_hashes),
        expected_m=200,
    )
    baseline_metadata, baseline_commit = _sidecars(baseline_path)
    observed_baseline_index = {
        "status": BASELINE_PASS_STATUS, "cumulative_M": 200, "pass": True,
        "artifact": baseline_path.relative_to(staging_root).as_posix(),
        "artifact_sha256": _sha256_file(baseline_path),
        "metadata": baseline_metadata.relative_to(staging_root).as_posix(),
        "metadata_sha256": _sha256_file(baseline_metadata),
        "commit": baseline_commit.relative_to(staging_root).as_posix(),
        "commit_sha256": _sha256_file(baseline_commit),
        "checked_common_cells": 3_200, "checked_allocation_histories": 3_040,
        "unavailable_allocation_histories": 160,
    }
    if dict(baseline_index) != observed_baseline_index:
        raise CalibrationAnalysisError("raw-master baseline identity commitment differs")
    expected_decision_ms = {
        200: [200], 500: [200, 500], 1000: [200, 500, 1000]
    }[final_m]
    decision_fields = {
        "cumulative_M", "artifact", "artifact_sha256", "metadata",
        "metadata_sha256", "commit", "commit_sha256", "next_M",
    }
    declared_decisions = payload.get("precision_decisions")
    if (
        not isinstance(declared_decisions, list)
        or len(declared_decisions) != len(expected_decision_ms)
        or [item.get("cumulative_M") for item in declared_decisions
            if isinstance(item, Mapping)] != expected_decision_ms
        or any(not isinstance(item, Mapping) or set(item) != decision_fields
               for item in declared_decisions)
    ):
        raise CalibrationAnalysisError("index-master precision decision schema changed")
    history_fields = {
        "cumulative_M", "precision_artifact", "precision_artifact_sha256", "next_M"
    }
    declared_history = payload.get("top_up_history")
    if (
        not isinstance(declared_history, list)
        or len(declared_history) != len(expected_decision_ms)
        or any(not isinstance(item, Mapping) or set(item) != history_fields
               for item in declared_history)
    ):
        raise CalibrationAnalysisError("index-master top-up history schema changed")
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        raise CalibrationAnalysisError("raw-master execution bindings are absent")
    if commit.get("launch_fingerprint") != bindings.get("launch_fingerprint"):
        raise CalibrationAnalysisError("raw-master commit launch fingerprint changed")
    if verify_local_bindings:
        _validate_local_bindings(bindings)
    provenance = {
        "artifact": path.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256_bytes(metadata_raw),
        "commit": commit_path.name,
        "commit_sha256": _sha256_bytes(commit_raw),
        "uncompressed_sha256": _sha256_bytes(logical),
        "final_M": final_m,
        "path_count": expected_paths,
        "snapshot_count": expected_snapshots,
        "bindings": dict(bindings),
    }
    return RawMaster(
        payload=dict(payload), provenance=provenance, staging_root=staging_root,
        block_payloads=tuple(block_payloads),
    )


def _exact_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise CalibrationAnalysisError(f"{label} must be an integer")
    try:
        output = int(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationAnalysisError(f"{label} must be an integer") from exc
    if output != value:
        raise CalibrationAnalysisError(f"{label} must be an exact integer")
    return output


def _precision_input_hashes(master: RawMaster, cumulative_m: int) -> dict[str, str]:
    output = {
        "paper/budget_toxicity_calibration_manifest.json": _sha256_file(MANIFEST_PATH),
        "paper/budget_toxicity_calibration_prespec.md": _sha256_file(PRESPEC_PATH),
    }
    block_count = {200: 1, 500: 2, 1000: 3}[int(cumulative_m)]
    block_commits = master.payload["block_commits"][:block_count]
    block_payloads = master.block_payloads[:block_count]
    for block_commit, block_payload in zip(
        block_commits, block_payloads, strict=True
    ):
        for field, hash_field in (
            ("artifact", "artifact_sha256"),
            ("metadata", "metadata_sha256"),
            ("commit", "commit_sha256"),
        ):
            output[str(block_commit[field])] = str(block_commit[hash_field])
        for shard in block_payload["shard_commits"]:
            for field, hash_field in (
                ("artifact", "artifact_sha256"),
                ("metadata", "metadata_sha256"),
                ("commit", "commit_sha256"),
            ):
                output[str(shard[field])] = str(shard[hash_field])
    baseline = master.payload["baseline_identity"]
    for field, hash_field in (
        ("artifact", "artifact_sha256"),
        ("metadata", "metadata_sha256"),
        ("commit", "commit_sha256"),
    ):
        output[str(baseline[field])] = str(baseline[hash_field])
    expected_count = {200: 80, 500: 155, 1000: 230}[int(cumulative_m)]
    if len(output) != expected_count:
        raise CalibrationAnalysisError("precision input-hash commitment count changed")
    return output


def validate_top_up_history(
    master: RawMaster,
    decision_paths: Sequence[str | Path],
) -> list[dict[str, Any]]:
    """Authenticate every precision decision and the all-factorial history."""

    payload = master.payload
    final_m = _exact_int(payload.get("final_M"), label="final_M")
    expected_ms = {200: [200], 500: [200, 500], 1000: [200, 500, 1000]}[final_m]
    if len(decision_paths) != len(expected_ms):
        raise CalibrationAnalysisError("precision-decision count does not match final M")
    history = payload.get("top_up_history")
    if not isinstance(history, list) or len(history) != len(expected_ms):
        raise CalibrationAnalysisError("raw-master top-up history is incomplete")
    declared = payload.get("precision_decisions")
    if not isinstance(declared, list) or len(declared) != len(expected_ms):
        raise CalibrationAnalysisError("raw-master precision-decision commitments are absent")
    output: list[dict[str, Any]] = []
    for index, (cumulative_m, path) in enumerate(zip(expected_ms, decision_paths, strict=True)):
        decision_path = Path(path).resolve()
        try:
            artifact_relative = decision_path.relative_to(master.staging_root).as_posix()
        except ValueError as exc:
            raise CalibrationAnalysisError(
                "precision decision is outside the master staging root"
            ) from exc
        decision = precision.load_blind_precision_decision(
            decision_path, expected_m=cumulative_m,
            expected_input_hashes=_precision_input_hashes(master, cumulative_m),
        )
        expected_next = expected_ms[index + 1] if index + 1 < len(expected_ms) else None
        if decision["next_M"] != expected_next:
            raise CalibrationAnalysisError("precision history does not justify the next tranche")
        artifact_sha = _sha256_file(decision_path)
        metadata_path, commit_path = _sidecars(decision_path)
        observed_declaration = {
            "cumulative_M": cumulative_m,
            "artifact": artifact_relative,
            "artifact_sha256": artifact_sha,
            "metadata": metadata_path.relative_to(master.staging_root).as_posix(),
            "metadata_sha256": _sha256_file(metadata_path),
            "commit": commit_path.relative_to(master.staging_root).as_posix(),
            "commit_sha256": _sha256_file(commit_path),
            "next_M": expected_next,
        }
        if not isinstance(declared[index], Mapping) or dict(declared[index]) != observed_declaration:
            raise CalibrationAnalysisError("raw-master precision commitment differs")
        entry = history[index]
        expected_history = {
            "cumulative_M": cumulative_m,
            "precision_artifact": artifact_relative,
            "precision_artifact_sha256": artifact_sha,
            "next_M": expected_next,
        }
        if not isinstance(entry, Mapping) or set(entry) != set(expected_history):
            raise CalibrationAnalysisError("top-up history entry is not an object")
        if dict(entry) != expected_history:
            raise CalibrationAnalysisError("top-up history differs from committed precision index")
        output.append(decision)
    return output


def _mapping(record: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = record.get(name)
    if not isinstance(value, Mapping):
        raise CalibrationAnalysisError(f"path record lacks {name}")
    return value


def _first(container: Mapping[str, Any], names: Sequence[str], *, label: str) -> Any:
    for name in names:
        if name in container:
            return container[name]
    raise CalibrationAnalysisError(f"serialized state lacks {label}")


def _vector(snapshot: Mapping[str, Any], canonical: str, *, dtype: Any = float) -> np.ndarray:
    containers = [snapshot]
    for name in ("posterior", "panel", "serialized_arrays"):
        child = snapshot.get(name)
        if isinstance(child, Mapping):
            containers.insert(0, child)
    found: list[Any] = []
    for container in containers:
        for alias in _VECTOR_ALIASES[canonical]:
            if alias in container:
                found.append(container[alias])
    if not found:
        raise CalibrationAnalysisError(f"snapshot lacks 25-vector {canonical}")
    arrays = [np.asarray(value, dtype=dtype) for value in found]
    for array in arrays:
        expected_shape = (25,)
        if array.shape != expected_shape:
            raise CalibrationAnalysisError(
                f"snapshot {canonical} has shape {array.shape}, expected {expected_shape}"
            )
        if dtype is not bool and not np.isfinite(array).all():
            raise CalibrationAnalysisError(f"snapshot {canonical} contains nonfinite values")
    if any(not np.array_equal(array, arrays[0]) for array in arrays[1:]):
        raise CalibrationAnalysisError(f"snapshot aliases disagree for {canonical}")
    return arrays[0]


def _history(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(record, "history")


def _true_above_history(record: Mapping[str, Any]) -> np.ndarray:
    history = record.get("history")
    if isinstance(history, Mapping):
        values = _first(
            history,
            ("true_above_boundary", "true_above_boundary_history", "above_boundary"),
            label="true above-boundary history",
        )
    else:
        values = _first(
            record, ("above_boundary_labels",), label="true above-boundary history"
        )
    array = np.asarray(values, dtype=bool)
    if array.shape != (80,):
        raise CalibrationAnalysisError("true above-boundary history must have length 80")
    return array


def _snapshots(record: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    try:
        return precision._snapshots(record)
    except precision.PrecisionMonitorError as exc:
        raise CalibrationAnalysisError(str(exc)) from exc


def _target_index(record: Mapping[str, Any]) -> int:
    target = _mapping(record, "true_panel_target")
    return _exact_int(
        _first(target, ("canonical_index", "panel_index", "index"), label="target index"),
        label="target index",
    )


def _toxicity_threshold(record: Mapping[str, Any], snapshot: Mapping[str, Any]) -> float:
    for container in (snapshot, record.get("true_panel_target"), record.get("design"), record):
        if not isinstance(container, Mapping):
            continue
        for name in ("toxicity_threshold", "gd", "boundary"):
            if name in container:
                value = float(container[name])
                if math.isfinite(value):
                    return value
    raise CalibrationAnalysisError("path lacks its true toxicity threshold")


def _recommendation_index(snapshot: Mapping[str, Any]) -> int:
    recommendation = snapshot.get("recommendation")
    if not isinstance(recommendation, Mapping):
        raise CalibrationAnalysisError("full-panel snapshot lacks terminal recommendation")
    made = recommendation.get("made", recommendation.get("recommendation_made", True))
    if made is not True:
        raise CalibrationAnalysisError("full-panel audit produced no recommendation")
    return _exact_int(
        _first(
            recommendation,
            ("panel_index", "index", "recommendation_index", "recommended_index"),
            label="recommendation index",
        ),
        label="recommendation index",
    )


def _checkpoint_rows(record: Mapping[str, Any], budget: int) -> list[Mapping[str, Any]]:
    rows = record.get("checkpoints", record.get("checkpoint_history"))
    if not isinstance(rows, list):
        raise CalibrationAnalysisError("path lacks target checkpoint history")
    selected = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise CalibrationAnalysisError("checkpoint entry must be an object")
        n = _exact_int(row.get("n"), label="checkpoint n")
        if n <= budget:
            selected.append(row)
    expected_n = list(range(4, budget + 1, 2))
    if [int(row["n"]) for row in selected] != expected_n:
        raise CalibrationAnalysisError(f"checkpoint schedule through N={budget} changed")
    return selected


def _saved_outcomes(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    value = snapshot.get("outcomes", snapshot.get("metrics"))
    if not isinstance(value, Mapping):
        raise CalibrationAnalysisError("snapshot lacks saved outcomes")
    return value


def _saved_gate(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    value = snapshot.get("gate_diagnostics")
    return value if isinstance(value, Mapping) else {}


@lru_cache(maxsize=2)
def _osa_truth_for_stratum(
    stratum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    try:
        from dose_combination_bo.surfaces import resolve_surface
        from dose_combination_bo.trial import grid
    except ImportError as exc:
        raise CalibrationAnalysisError("frozen OSA surface code is unavailable") from exc
    surface = resolve_surface("osa", stratum)
    doses = np.asarray(grid(5), dtype=float)
    true_eff = np.asarray(
        [surface["eff"](float(dose[0]), float(dose[1])) for dose in doses], dtype=float
    )
    true_tox = np.asarray(
        [surface["tox"](float(dose[0]), float(dose[1])) for dose in doses], dtype=float
    )
    return doses, true_eff, true_tox, float(surface["gd"])


def _osa_truth(
    record: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    stratum = int(precision._path_identifiers(record)[3])
    return _osa_truth_for_stratum(stratum)


def _get_saved(
    containers: Sequence[Mapping[str, Any]], aliases: Sequence[str]
) -> tuple[bool, Any]:
    found = []
    for container in containers:
        for alias in aliases:
            if alias in container:
                found.append(container[alias])
    if not found:
        return False, None
    first = found[0]
    for value in found[1:]:
        if isinstance(first, (int, float)) and isinstance(value, (int, float)):
            if not math.isclose(float(first), float(value), rel_tol=0.0, abs_tol=1e-10):
                raise CalibrationAnalysisError(f"saved scalar aliases disagree: {aliases[0]}")
        elif value != first:
            raise CalibrationAnalysisError(f"saved scalar aliases disagree: {aliases[0]}")
    return True, first


def _check_saved(
    containers: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
    recomputed: Any,
    *,
    required: bool = True,
    tolerance: float = 1e-9,
) -> None:
    present, saved = _get_saved(containers, aliases)
    if not present:
        if required:
            raise CalibrationAnalysisError(f"saved metric {aliases[0]} is absent")
        return
    if recomputed is None:
        if saved is not None:
            raise CalibrationAnalysisError(f"saved metric {aliases[0]} should be null")
    elif isinstance(recomputed, bool):
        if type(saved) is not bool or saved != recomputed:
            raise CalibrationAnalysisError(f"saved metric {aliases[0]} differs from replay")
    elif not math.isclose(float(saved), float(recomputed), rel_tol=0.0, abs_tol=tolerance):
        raise CalibrationAnalysisError(f"saved metric {aliases[0]} differs from replay")


def recompute_snapshot_metrics(
    record: Mapping[str, Any], budget: int
) -> dict[str, Any]:
    """Replay all scalar outcomes from stored arrays, without fitting a model."""

    snapshot = _snapshots(record)[budget]
    doses, true_eff, true_tox, true_threshold = _osa_truth(record)
    eff_mean = _vector(snapshot, "efficacy_mean")
    eff_sd = _vector(snapshot, "efficacy_sd")
    tox_mean = _vector(snapshot, "toxicity_mean")
    tox_sd = _vector(snapshot, "toxicity_sd")
    stable_z = _vector(snapshot, "stable_z")
    pf = _vector(snapshot, "feasibility_probability")
    gate = _vector(snapshot, "gate_mask", dtype=bool)
    if np.any(eff_sd < 0) or np.any(tox_sd <= 0) or np.any((pf < 0) | (pf > 1)):
        raise CalibrationAnalysisError("posterior SD/PF arrays are outside their domains")
    target_index = _target_index(record)
    rec_index = _recommendation_index(snapshot)
    if not 0 <= target_index < 25 or not 0 <= rec_index < 25:
        raise CalibrationAnalysisError("target or recommendation index is off panel")
    serialized_threshold = _toxicity_threshold(record, snapshot)
    if not math.isclose(
        serialized_threshold, true_threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        raise CalibrationAnalysisError("serialized toxicity threshold differs from OSA truth")
    threshold = true_threshold
    expected_stable_z = (threshold - tox_mean) / tox_sd
    if not np.allclose(stable_z, expected_stable_z, rtol=1e-12, atol=1e-12):
        raise CalibrationAnalysisError("snapshot stable-z differs from posterior replay")
    expected_pf = special.ndtr(stable_z)
    if not np.allclose(pf, expected_pf, rtol=1e-12, atol=1e-12):
        raise CalibrationAnalysisError("snapshot feasibility probabilities differ from stable-z")
    gamma = float(precision._path_identifiers(record)[2])
    gate_quantile = float(snapshot.get("gate_quantile"))
    expected_quantile = GATE_QUANTILES[gamma]
    if not math.isclose(gate_quantile, expected_quantile, rel_tol=0.0, abs_tol=1e-12):
        raise CalibrationAnalysisError("snapshot gate quantile differs from gamma")
    if not np.array_equal(gate, stable_z > gate_quantile):
        raise CalibrationAnalysisError("snapshot gate mask differs from strict stable-z gate")
    true_feasible = true_tox <= threshold
    if not true_feasible[target_index]:
        raise CalibrationAnalysisError("frozen true panel target is not feasible")
    feasible_indices = np.flatnonzero(true_feasible)
    canonical_target = int(feasible_indices[np.argmax(true_eff[true_feasible])])
    if target_index != canonical_target:
        raise CalibrationAnalysisError("serialized true panel target is not canonical")
    target_serialized = _mapping(record, "true_panel_target")
    expected_target = {
        "canonical_index": target_index,
        "dose": doses[target_index].tolist(),
        "true_efficacy": float(true_eff[target_index]),
        "true_toxicity": float(true_tox[target_index]),
        "toxicity_threshold": threshold,
        "true_feasible_dose_count": int(np.count_nonzero(true_feasible)),
    }
    for field, expected in expected_target.items():
        observed = target_serialized.get(field)
        if isinstance(expected, list):
            equal = np.array_equal(np.asarray(observed, dtype=float), np.asarray(expected))
        elif isinstance(expected, float):
            equal = math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
        else:
            equal = observed == expected
        if not equal:
            raise CalibrationAnalysisError(f"serialized true target differs at {field}")

    labels = _true_above_history(record)[:budget]
    above_count = int(np.count_nonzero(labels))
    post_labels = labels[4:]
    checkpoints = _checkpoint_rows(record, budget)
    assumed_noise_variance = float(record["assumed_tox_noise_sd"]) ** 2
    checkpoint_sd = np.asarray(
        [row["target_toxicity_latent_sd"] for row in checkpoints], dtype=float
    )
    checkpoint_mean = np.asarray(
        [row["target_toxicity_mean"] for row in checkpoints], dtype=float
    )
    checkpoint_z = np.asarray(
        [row["target_stable_z"] for row in checkpoints], dtype=float
    )
    checkpoint_pf = np.asarray(
        [row["target_feasibility_probability"] for row in checkpoints], dtype=float
    )
    checkpoint_admitted = np.asarray(
        [row["target_admitted"] for row in checkpoints], dtype=bool
    )
    checkpoint_gate_count = np.asarray(
        [_exact_int(row["gate_count"], label="checkpoint gate count") for row in checkpoints],
        dtype=int,
    )
    checkpoint_empty = np.asarray(
        [row["gate_empty"] for row in checkpoints], dtype=bool
    )
    checkpoint_noise = np.asarray(
        [row["fitted_toxicity_noise_variance"] for row in checkpoints], dtype=float
    )
    if np.any(checkpoint_sd <= 0) or not np.allclose(
        checkpoint_z, (threshold - checkpoint_mean) / checkpoint_sd,
        rtol=1e-12, atol=1e-12,
    ):
        raise CalibrationAnalysisError("checkpoint target stable-z replay failed")
    if not np.allclose(
        checkpoint_pf, special.ndtr(checkpoint_z), rtol=1e-12, atol=1e-12
    ):
        raise CalibrationAnalysisError("checkpoint target feasibility probability failed")
    if not np.array_equal(checkpoint_admitted, checkpoint_z > gate_quantile):
        raise CalibrationAnalysisError("checkpoint target admission is not strict-gate based")
    if (
        np.any((checkpoint_gate_count < 0) | (checkpoint_gate_count > 25))
        or not np.array_equal(checkpoint_empty, checkpoint_gate_count == 0)
    ):
        raise CalibrationAnalysisError("checkpoint gate count/empty flag disagree")
    if not np.allclose(
        checkpoint_noise, assumed_noise_variance, rtol=0.0, atol=2e-5
    ):
        raise CalibrationAnalysisError("checkpoint fitted noise differs from assumption")
    admitted = np.asarray(
        [bool(row["target_admitted"]) for row in checkpoints], dtype=bool
    )
    if bool(admitted[-1]) != bool(gate[target_index]):
        raise CalibrationAnalysisError("checkpoint and terminal target admission disagree")
    never_admitted = not bool(np.any(admitted))
    finally_admitted = bool(admitted[-1])
    lost_after_admission = bool(np.any(admitted[:-1]) and not finally_admitted)
    partition_sum = int(never_admitted) + int(lost_after_admission) + int(finally_admitted)
    if partition_sum != 1:
        raise CalibrationAnalysisError("target gate partition is not mutually exclusive")
    exclusion = ~admitted
    final_exclusion_run = 0
    for value in exclusion[::-1]:
        if not value:
            break
        final_exclusion_run += 1
    preterminal_exclusion = bool(np.any(exclusion[:-1]))
    recovered_after_preterminal_exclusion = bool(preterminal_exclusion and finally_admitted)
    first_admission_n = (
        int(checkpoints[int(np.argmax(admitted))]["n"]) if np.any(admitted) else None
    )

    gate_count = int(np.count_nonzero(gate))
    safe_admitted = int(np.count_nonzero(gate & true_feasible))
    false_admissions = gate_count - safe_admitted
    rec_feasible = bool(true_feasible[rec_index])
    brier = float(np.mean((pf - true_feasible.astype(float)) ** 2))
    coverage = np.abs(true_tox - tox_mean) <= 1.96 * tox_sd
    grid_step = float(np.min(np.diff(np.unique(doses[:, 0]))))
    if not math.isfinite(grid_step) or grid_step <= 0:
        raise CalibrationAnalysisError("panel coordinates do not define a grid step")

    output: dict[str, Any] = {
        "above_boundary_assignment_count": above_count,
        precision.OUTCOME_ASSIGNMENT: 100.0 * above_count / budget,
        "post_initialization_above_boundary_assignment_pct": (
            100.0 * int(np.count_nonzero(post_labels)) / (budget - 4)
        ),
        precision.OUTCOME_TERMINAL: 100.0 * float(not rec_feasible),
        precision.OUTCOME_EXCLUSION: 100.0 * float(not finally_admitted),
        "target_ever_admitted_pct": 100.0 * float(np.any(admitted)),
        "target_finally_admitted_pct": 100.0 * float(finally_admitted),
        "target_never_admitted_pct": 100.0 * float(never_admitted),
        "target_admitted_then_excluded_pct": 100.0 * float(lost_after_admission),
        "target_exclusion_occupancy_pct": 100.0 * float(np.mean(exclusion)),
        "target_terminal_exclusion_run_fraction_pct": (
            100.0 * final_exclusion_run / len(checkpoints)
        ),
        "target_first_admission_enrollment": first_admission_n,
        "target_preterminal_exclusion_eligible": preterminal_exclusion,
        "target_recovered_after_preterminal_exclusion_pct": (
            100.0 * float(recovered_after_preterminal_exclusion)
        ),
        "target_admitted_but_not_selected_pct": (
            100.0 * float(finally_admitted and rec_index != target_index)
        ),
        precision.OUTCOME_BRIER: 100.0 * brier,
        "panel_brier_score": brier,
        "latent_95_interval_coverage_pct": 100.0 * float(np.mean(coverage)),
        "target_latent_95_interval_coverage_pct": 100.0 * float(coverage[target_index]),
        "mean_latent_95_interval_width": float(np.mean(3.92 * tox_sd)),
        "panel_correct_selection_pct": 100.0 * float(rec_index == target_index),
        "recommendation_grid_distance": float(
            np.linalg.norm(doses[rec_index] - doses[target_index]) / grid_step
        ),
        "true_recommended_efficacy": float(true_eff[rec_index]),
        "true_feasible_simple_regret": (
            float(true_eff[target_index] - true_eff[rec_index]) if rec_feasible else None
        ),
        "true_feasible_recommendation": rec_feasible,
        "final_gate_size": gate_count,
        "final_empty_gate_pct": 100.0 * float(gate_count == 0),
        "false_admissions": false_admissions,
        "feasible_dose_recall_pct": 100.0 * safe_admitted / int(np.count_nonzero(true_feasible)),
        "gate_precision_pct": (
            100.0 * safe_admitted / gate_count if gate_count else None
        ),
        "recommendation_index": rec_index,
        "target_index": target_index,
        "target_stable_z": float(stable_z[target_index]),
        "target_feasibility_probability_pct": 100.0 * float(pf[target_index]),
    }
    outcomes = _saved_outcomes(snapshot)
    gate_saved = _saved_gate(snapshot)
    calibration_saved = snapshot.get("posterior_calibration")
    if not isinstance(calibration_saved, Mapping):
        raise CalibrationAnalysisError("snapshot lacks posterior calibration diagnostics")
    target_saved = snapshot.get("target_gate")
    if not isinstance(target_saved, Mapping):
        raise CalibrationAnalysisError("snapshot lacks target-gate diagnostics")
    recommendation_saved = snapshot.get("recommendation")
    if not isinstance(recommendation_saved, Mapping):
        raise CalibrationAnalysisError("snapshot lacks recommendation diagnostics")
    containers = [outcomes, gate_saved, calibration_saved, target_saved]
    aliases = {
        "above_boundary_assignment_count": ("above_boundary_assignment_count", "assignment_count"),
        precision.OUTCOME_ASSIGNMENT: precision._OUTCOME_ALIASES[precision.OUTCOME_ASSIGNMENT],
        precision.OUTCOME_TERMINAL: precision._OUTCOME_ALIASES[precision.OUTCOME_TERMINAL],
        precision.OUTCOME_EXCLUSION: precision._OUTCOME_ALIASES[precision.OUTCOME_EXCLUSION],
        precision.OUTCOME_BRIER: precision._OUTCOME_ALIASES[precision.OUTCOME_BRIER],
        "post_initialization_above_boundary_assignment_pct": (
            "post_initialization_above_boundary_pct",
        ),
        "final_gate_size": ("final_gate_size", "gate_count", "n_gate_pass"),
        "false_admissions": ("false_admissions",),
        "panel_correct_selection_pct": ("panel_correct_selection_pct", "correct_selection_pct"),
        "mean_latent_95_interval_width": ("mean_latent_95_interval_width",),
    }
    for key, key_aliases in aliases.items():
        _check_saved(containers, key_aliases, output[key], required=True)
    _check_saved(
        [gate_saved], ("feasible_dose_recall",),
        output["feasible_dose_recall_pct"] / 100.0,
    )
    _check_saved(
        [calibration_saved], ("latent_95_interval_coverage_proportion",),
        output["latent_95_interval_coverage_pct"] / 100.0,
    )
    _check_saved(
        [calibration_saved], ("brier_score",), output["panel_brier_score"]
    )
    _check_saved(
        [calibration_saved], ("target_latent_95_interval_covered",),
        bool(output["target_latent_95_interval_coverage_pct"]),
    )
    target_checks = {
        "terminal_excluded": bool(output[precision.OUTCOME_EXCLUSION]),
        "ever_admitted": bool(output["target_ever_admitted_pct"]),
        "never_admitted": bool(output["target_never_admitted_pct"]),
        "admitted_then_excluded": bool(output["target_admitted_then_excluded_pct"]),
        "admitted_finally": bool(output["target_finally_admitted_pct"]),
        "admitted_but_not_selected": bool(output["target_admitted_but_not_selected_pct"]),
        "exclusion_occupancy": output["target_exclusion_occupancy_pct"] / 100.0,
        "terminal_exclusion_run_fraction": (
            output["target_terminal_exclusion_run_fraction_pct"] / 100.0
        ),
        "first_admission_enrollment": output["target_first_admission_enrollment"],
        "had_preterminal_exclusion": output["target_preterminal_exclusion_eligible"],
        "recovered_after_preterminal_exclusion": bool(
            output["target_recovered_after_preterminal_exclusion_pct"]
        ),
    }
    for saved_key, replayed in target_checks.items():
        _check_saved([target_saved], (saved_key,), replayed)
    if gate_count:
        _check_saved(
            [gate_saved], ("precision_if_nonempty",),
            output["gate_precision_pct"] / 100.0,
        )
    else:
        _check_saved([gate_saved], ("precision_if_nonempty",), None)
    recommendation_checks = {
        "index": rec_index,
        "true_efficacy": output["true_recommended_efficacy"],
        "true_feasible": output["true_feasible_recommendation"],
        "panel_correct_selection": bool(output["panel_correct_selection_pct"]),
        "grid_unit_distance_from_target": output["recommendation_grid_distance"],
        "simple_regret_if_true_feasible": output["true_feasible_simple_regret"],
    }
    for saved_key, replayed in recommendation_checks.items():
        _check_saved([recommendation_saved], (saved_key,), replayed)
    # The terminal replay is a construction invariant, not an outcome.
    audit = snapshot.get("terminal_rule_audit")
    if not isinstance(audit, Mapping):
        raise CalibrationAnalysisError("snapshot lacks terminal-rule audit")
    if (
        audit.get("status") != "checked"
        or audit.get("stable_z_gate_difference_count") != 0
        or audit.get("recommendation_index_difference_count") != 0
    ):
        raise CalibrationAnalysisError("terminal-rule replay did not pass")
    return output


def explode_and_validate(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one authenticated logical row per nested snapshot."""

    rows: list[dict[str, Any]] = []
    for record in records:
        seed, policy, gamma, stratum, factor = precision._path_identifiers(record)
        for budget in BUDGETS:
            metrics = recompute_snapshot_metrics(record, budget)
            rows.append({
                "seed": seed,
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "assumed_toxicity_noise_sd_factor": factor,
                "gamma": gamma,
                "stratum": stratum,
                "budget": budget,
                **metrics,
            })
    expected = len(records) * 3
    keys = {
        (row["seed"], row["policy"], row["assumed_toxicity_noise_sd_factor"],
         row["gamma"], row["stratum"], row["budget"])
        for row in rows
    }
    if len(rows) != expected or len(keys) != expected:
        raise CalibrationAnalysisError("logical snapshot expansion is incomplete or duplicated")
    return rows


def stream_master_analysis_state(
    master: RawMaster,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Authenticate one shard at a time and retain only derived logical rows."""

    logical_rows: list[dict[str, Any]] = []
    baseline_envelopes: list[dict[str, Any]] = []
    physical_keys: set[tuple[Any, ...]] = set()
    innovation_by_seed: dict[int, tuple[str, str, str]] = {}
    truth_noise_pairs: set[tuple[float, float]] = set()
    path_count = 0
    for block_payload in master.block_payloads:
        for shard in block_payload["shard_commits"]:
            for record in precision.iter_authenticated_shard_results(
                master.staging_root, shard, bindings=block_payload["bindings"],
                validate_core=True,
            ):
                seed, policy, gamma, stratum, factor = precision._path_identifiers(record)
                key = (seed, policy, gamma, stratum, factor)
                if key in physical_keys:
                    raise CalibrationAnalysisError("streamed physical path is duplicated")
                physical_keys.add(key)
                path_count += 1
                truth_pair = (
                    float(record["truth_eff_noise_sd"]),
                    float(record["truth_tox_noise_sd"]),
                )
                truth_noise_pairs.add(truth_pair)
                if not math.isclose(
                    float(record["assumed_eff_noise_sd"]), truth_pair[0],
                    rel_tol=0.0, abs_tol=1e-12,
                ) or not math.isclose(
                    float(record["assumed_tox_noise_sd"]), truth_pair[1] * factor,
                    rel_tol=0.0, abs_tol=1e-12,
                ):
                    raise CalibrationAnalysisError("truth/assumed noise fields are inconsistent")
                innovation = record.get("outcome_innovation_stream")
                if not isinstance(innovation, Mapping) or innovation.get(
                    "innovation_count_per_endpoint"
                ) != 80:
                    raise CalibrationAnalysisError("innovation-stream audit is incomplete")
                triple = tuple(
                    str(innovation[field])
                    for field in ("combined_sha256", "efficacy_sha256", "toxicity_sha256")
                )
                if any(SHA256_RE.fullmatch(value) is None for value in triple):
                    raise CalibrationAnalysisError("innovation-stream hash is malformed")
                previous = innovation_by_seed.setdefault(seed, triple)
                if previous != triple:
                    raise CalibrationAnalysisError(
                        "policy/gate/factor/stratum changed a seed's innovation stream"
                    )
                for budget in BUDGETS:
                    metrics = recompute_snapshot_metrics(record, budget)
                    logical_rows.append({
                        "seed": seed, "policy": policy,
                        "policy_label": POLICY_LABELS[policy],
                        "assumed_toxicity_noise_sd_factor": factor,
                        "gamma": gamma, "stratum": stratum, "budget": budget,
                        **metrics,
                    })
                if seed < 200 and factor == 1.0:
                    legacy = _snapshots(record)[40].get("frozen_run_trial_common")
                    if not isinstance(legacy, Mapping):
                        raise CalibrationAnalysisError("baseline legacy envelope is absent")
                    baseline_envelopes.append(dict(legacy))
            # The generator and its shard payload become unreachable here before
            # the next shard is decompressed.
    final_m = int(master.payload["final_M"])
    expected_physical = 48 * final_m
    expected_logical = 144 * final_m
    if path_count != expected_physical or len(physical_keys) != expected_physical:
        raise CalibrationAnalysisError("streamed physical factorial is incomplete")
    if len(logical_rows) != expected_logical:
        raise CalibrationAnalysisError("streamed logical snapshot factorial is incomplete")
    if set(innovation_by_seed) != set(range(final_m)):
        raise CalibrationAnalysisError("innovation-stream seed coverage is incomplete")
    if truth_noise_pairs != {(7.68, 1.29)}:
        raise CalibrationAnalysisError(
            f"data-generating noise changed across the audit: {truth_noise_pairs!r}"
        )
    if len(baseline_envelopes) != 3_200:
        raise CalibrationAnalysisError("streamed baseline cache is not 3,200 cells")
    audit = {
        "streaming_storage": "one_authenticated_shard_at_a_time",
        "raw_records_accumulated": False,
        "physical_paths_validated": expected_physical,
        "logical_snapshots_recomputed": expected_logical,
        "common_random_number_seeds": final_m,
        "innovation_hash_differences_across_factorial": 0,
        "truth_eff_noise_sd": 7.68,
        "truth_tox_noise_sd": 1.29,
    }
    return logical_rows, baseline_envelopes, audit


def _derived_projection(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, float, int, float, int], dict[str, float]]:
    output: dict[tuple[int, str, float, int, float, int], dict[str, float]] = {}
    for row in rows:
        key = (
            int(row["seed"]), str(row["policy"]), float(row["gamma"]),
            int(row["stratum"]), float(row["assumed_toxicity_noise_sd_factor"]),
            int(row["budget"]),
        )
        output[key] = {
            precision.OUTCOME_TERMINAL: float(row[precision.OUTCOME_TERMINAL]),
            precision.OUTCOME_ASSIGNMENT: float(row[precision.OUTCOME_ASSIGNMENT]),
            precision.OUTCOME_EXCLUSION: float(row[precision.OUTCOME_EXCLUSION]),
            precision.OUTCOME_BRIER: float(row[precision.OUTCOME_BRIER]),
        }
    return output


def guarded_scientific_sign(low: float, high: float) -> str:
    """Classify a guarded band using strict, exact zero-exclusion rules."""

    lower = float(low)
    upper = float(high)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise CalibrationAnalysisError("guarded band endpoints are invalid")
    if lower > 0.0:
        return "positive"
    if upper < 0.0:
        return "negative"
    return "unresolved"


def max_t_family_summary(
    matrix: np.ndarray,
    estimand_ids: Sequence[str],
    *,
    family: str,
    draws: int = MAX_T_DRAWS,
    random_seed: int | None = None,
    batch_size: int = 1_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen seed-clustered Rademacher max-t construction."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(estimand_ids):
        raise CalibrationAnalysisError("max-t input dimensions are invalid")
    if not np.isfinite(values).all() or family not in MAX_T_SEEDS:
        raise CalibrationAnalysisError("max-t input values/family are invalid")
    if draws <= 0 or batch_size <= 0:
        raise CalibrationAnalysisError("max-t draws and batch size must be positive")
    locked = frozenset(precision.primary_estimand_ids())
    if any(
        estimand_id not in locked
        or not estimand_id.startswith(f"{family}|")
        for estimand_id in estimand_ids
    ):
        raise CalibrationAnalysisError(
            "max-t family contains an unlocked or misclassified estimand"
        )
    n = values.shape[0]
    if n not in precision.ELIGIBLE_M:
        raise CalibrationAnalysisError(
            "max-t replicate count is not one frozen cumulative M"
        )
    if len(set(estimand_ids)) != len(estimand_ids):
        raise CalibrationAnalysisError("max-t estimand identifiers are duplicated")
    estimates = np.mean(values, axis=0)
    centered = values - estimates
    denominator = np.sqrt(np.sum(centered * centered, axis=0))
    zero_variance = denominator == 0.0
    mcse = np.std(values, axis=0, ddof=1) / math.sqrt(n)
    point_critical = float(stats.t.ppf(0.975, df=n - 1))
    rng = np.random.default_rng(
        MAX_T_SEEDS[family] if random_seed is None else int(random_seed)
    )
    maxima = np.empty(draws, dtype=float)
    filled = 0
    active = ~zero_variance
    while filled < draws:
        take = min(batch_size, draws - filled)
        xi = 2.0 * rng.integers(0, 2, size=(take, n), dtype=np.int8) - 1.0
        t_values = np.zeros((take, values.shape[1]), dtype=float)
        if np.any(active):
            t_values[:, active] = (
                xi @ centered[:, active]
            ) / denominator[active]
        maxima[filled : filled + take] = np.max(np.abs(t_values), axis=1)
        filled += take
    critical = float(np.quantile(maxima, CONFIDENCE_LEVEL, method="higher"))
    rows: list[dict[str, Any]] = []
    for index, estimand_id in enumerate(estimand_ids):
        estimate = float(estimates[index])
        se = float(mcse[index])
        raw_low = estimate - critical * se
        raw_high = estimate + critical * se
        guard = precision.row_numerical_guard(estimand_id, n)
        replicate_bound = float(guard["replicate_bound_points"])
        mcse_guard = float(guard["mcse_guard_points"])
        guard_halfwidth = replicate_bound + critical * mcse_guard
        guarded_low = raw_low - guard_halfwidth
        guarded_high = raw_high + guard_halfwidth
        sign = guarded_scientific_sign(guarded_low, guarded_high)
        rows.append({
            "family": family,
            "estimand_id": estimand_id,
            "estimate_points": estimate,
            "mcse_points": se,
            "pointwise_t_critical_95": point_critical,
            "pointwise_95_low_points": estimate - point_critical * se,
            "pointwise_95_high_points": estimate + point_critical * se,
            "family_max_t_critical_95": critical,
            "simultaneous_95_low_points": raw_low,
            "simultaneous_95_high_points": raw_high,
            "row_guard_class": guard["row_guard_class"],
            "replicate_numerical_bound_points": replicate_bound,
            "mcse_numerical_guard_points": mcse_guard,
            "simultaneous_numerical_halfwidth_guard_points": guard_halfwidth,
            "guarded_simultaneous_95_low_points": guarded_low,
            "guarded_simultaneous_95_high_points": guarded_high,
            "guarded_simultaneous_95_strictly_positive": sign == "positive",
            "guarded_simultaneous_95_strictly_negative": sign == "negative",
            "guarded_simultaneous_95_excludes_zero": sign != "unresolved",
            "guarded_scientific_sign": sign,
            "zero_variance": bool(zero_variance[index]),
            "monte_carlo_replicates": n,
        })
    guard_class_counts = {
        guard_class: sum(
            row["row_guard_class"] == guard_class for row in rows
        )
        for guard_class in precision.ROW_GUARD_CLASSES
        if any(row["row_guard_class"] == guard_class for row in rows)
    }
    metadata = {
        "family": family,
        "draws": int(draws),
        "random_seed": MAX_T_SEEDS[family] if random_seed is None else int(random_seed),
        "multiplier": "Rademacher",
        "statistic": "sum_i xi_i*(Y_i-Ybar)/sqrt(sum_i (Y_i-Ybar)^2)",
        "critical_statistic": "max absolute coordinate T",
        "quantile": 0.95,
        "quantile_method": "higher",
        "critical_value": critical,
        "zero_variance_rule": (
            "T=0 and zero-width raw interval; a nonzero numerical bound may "
            "still widen the guarded band"
        ),
        "numerical_equivalence_guard": (
            precision.numerical_equivalence_guard_metadata()
        ),
        "simultaneous_numerical_halfwidth_guard_formula": (
            "h = replicate_numerical_bound_points + "
            "family_max_t_critical_95 * mcse_numerical_guard_points"
        ),
        "guarded_simultaneous_band_formula": (
            "guarded_low = raw_low - h; guarded_high = raw_high + h"
        ),
        "guarded_scientific_sign_rule": (
            "positive iff guarded_low > 0; negative iff guarded_high < 0; "
            "otherwise unresolved, including an endpoint exactly equal to zero"
        ),
        "row_guard_class_counts": guard_class_counts,
        "estimand_count": len(estimand_ids),
        "monte_carlo_replicates": n,
    }
    return rows, metadata


def primary_analysis(
    logical_rows: Sequence[Mapping[str, Any]],
    *,
    max_t_draws: int = MAX_T_DRAWS,
    max_t_batch_size: int = 1_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Construct all 118 locked primary estimates and family bands."""

    ids, matrix = precision.primary_replicate_vectors(_derived_projection(logical_rows))
    output: list[dict[str, Any]] = []
    family_metadata: dict[str, Any] = {}
    for family in ("A", "B", "C"):
        indices = [index for index, estimand_id in enumerate(ids) if estimand_id.startswith(f"{family}|")]
        family_rows, details = max_t_family_summary(
            matrix[:, indices], [ids[index] for index in indices], family=family,
            draws=max_t_draws, batch_size=max_t_batch_size,
        )
        output.extend(family_rows)
        family_metadata[family] = details
    if len(output) != 118:
        raise CalibrationAnalysisError("primary analysis did not produce 118 estimands")
    return output, {
        "family_bands": family_metadata,
        "pointwise_method": "paired seed-level t interval",
        "aggregation": "equal weight over two strata and two gates within seed",
        "numerical_equivalence_guard": (
            precision.numerical_equivalence_guard_metadata()
        ),
        "simultaneous_numerical_halfwidth_guard_formula": (
            "h = replicate_numerical_bound_points + "
            "family_max_t_critical_95 * mcse_numerical_guard_points"
        ),
        "scientific_sign_source": "guarded simultaneous 95% band only",
        "cell_vote_used": False,
    }


ABSOLUTE_METRICS = (
    precision.OUTCOME_TERMINAL,
    "above_boundary_assignment_count",
    precision.OUTCOME_ASSIGNMENT,
    "post_initialization_above_boundary_assignment_pct",
    precision.OUTCOME_EXCLUSION,
    "target_ever_admitted_pct",
    "target_finally_admitted_pct",
    "target_never_admitted_pct",
    "target_admitted_then_excluded_pct",
    "target_exclusion_occupancy_pct",
    "target_terminal_exclusion_run_fraction_pct",
    "target_recovered_after_preterminal_exclusion_pct",
    "target_admitted_but_not_selected_pct",
    precision.OUTCOME_BRIER,
    "latent_95_interval_coverage_pct",
    "target_latent_95_interval_coverage_pct",
    "mean_latent_95_interval_width",
    "panel_correct_selection_pct",
    "recommendation_grid_distance",
    "true_recommended_efficacy",
    "final_gate_size",
    "final_empty_gate_pct",
    "false_admissions",
    "feasible_dose_recall_pct",
)


SECONDARY_ANALYSIS_ABSOLUTE_PROFILE = "absolute_equal_gate_stratum_profile"
SECONDARY_ANALYSIS_BUDGET_CONTRAST = "absolute_budget_contrast_N80_minus_N20"
SECONDARY_ANALYSIS_CALIBRATION_DID = (
    "policy_minus_cEI_calibration_difference_in_differences"
)

SECONDARY_GUARD_CLASS_NON_BRIER_ZERO = "secondary_non_brier_zero_guard"
SECONDARY_GUARD_CLASS_ABSOLUTE_BRIER = (
    "secondary_absolute_panel_brier_snapshot"
)
SECONDARY_GUARD_CLASS_BUDGET_BRIER = (
    "secondary_within_policy_N80_minus_N20_panel_brier_contrast"
)
SECONDARY_GUARD_CLASS_CALIBRATION_DID_BRIER = (
    "secondary_policy_vs_cEI_panel_brier_calibration_did"
)
SECONDARY_ROW_GUARD_CLASSES = (
    SECONDARY_GUARD_CLASS_NON_BRIER_ZERO,
    SECONDARY_GUARD_CLASS_ABSOLUTE_BRIER,
    SECONDARY_GUARD_CLASS_BUDGET_BRIER,
    SECONDARY_GUARD_CLASS_CALIBRATION_DID_BRIER,
)


def secondary_numerical_equivalence_guard_metadata() -> dict[str, Any]:
    """Return the frozen pointwise guard contract for secondary quantities."""

    return {
        "schema_version": 1,
        "eta": precision.NUMERICAL_EQUIVALENCE_ETA,
        "eta_expression": "2^-26",
        "unit": "percentage points for panel_brier_score_pct; zero otherwise",
        "row_guard_classes": {
            SECONDARY_GUARD_CLASS_NON_BRIER_ZERO: {
                "applies_to": "every non-Brier secondary row",
                "replicate_numerical_bound_points": 0.0,
                "bound_expression": "0",
            },
            SECONDARY_GUARD_CLASS_ABSOLUTE_BRIER: {
                "applies_to": SECONDARY_ANALYSIS_ABSOLUTE_PROFILE,
                "replicate_numerical_bound_points": (
                    precision.PANEL_BRIER_SNAPSHOT_BOUND_POINTS
                ),
                "bound_expression": "delta0 = 200*eta",
            },
            SECONDARY_GUARD_CLASS_BUDGET_BRIER: {
                "applies_to": SECONDARY_ANALYSIS_BUDGET_CONTRAST,
                "replicate_numerical_bound_points": (
                    2.0 * precision.PANEL_BRIER_SNAPSHOT_BOUND_POINTS
                ),
                "bound_expression": "2*delta0 = 400*eta",
            },
            SECONDARY_GUARD_CLASS_CALIBRATION_DID_BRIER: {
                "applies_to": SECONDARY_ANALYSIS_CALIBRATION_DID,
                "replicate_numerical_bound_points": (
                    4.0 * precision.PANEL_BRIER_SNAPSHOT_BOUND_POINTS
                ),
                "bound_expression": "4*delta0 = 800*eta",
            },
        },
        "mcse_numerical_guard_formula": "a = b/sqrt(M-1)",
        "pointwise_numerical_halfwidth_guard_formula": "h = b + t_crit*a",
        "guarded_pointwise_interval_formula": (
            "guarded_low = raw_ci95_low - h; "
            "guarded_high = raw_ci95_high + h"
        ),
        "raw_pointwise_interval_fields": ["ci95_low", "ci95_high"],
        "guarded_pointwise_interval_fields": [
            "guarded_ci95_low", "guarded_ci95_high",
        ],
        "guarded_scientific_sign_rule": (
            "positive iff guarded_low > 0; negative iff guarded_high < 0; "
            "otherwise unresolved, including an endpoint exactly equal to zero"
        ),
    }


def secondary_row_numerical_guard(
    analysis_type: str, metric: str, cumulative_m: int,
) -> dict[str, Any]:
    """Resolve the prespecified numerical bound for one secondary row."""

    try:
        normalized_m = int(cumulative_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalibrationAnalysisError(
            "secondary guard replicate count is not one frozen cumulative M"
        ) from exc
    if (
        isinstance(cumulative_m, bool)
        or normalized_m != cumulative_m
        or normalized_m not in precision.ELIGIBLE_M
    ):
        raise CalibrationAnalysisError(
            "secondary guard replicate count is not one frozen cumulative M"
        )
    if metric != precision.OUTCOME_BRIER:
        guard_class = SECONDARY_GUARD_CLASS_NON_BRIER_ZERO
        replicate_bound = 0.0
    else:
        brier_guards = {
            SECONDARY_ANALYSIS_ABSOLUTE_PROFILE: (
                SECONDARY_GUARD_CLASS_ABSOLUTE_BRIER,
                precision.PANEL_BRIER_SNAPSHOT_BOUND_POINTS,
            ),
            SECONDARY_ANALYSIS_BUDGET_CONTRAST: (
                SECONDARY_GUARD_CLASS_BUDGET_BRIER,
                2.0 * precision.PANEL_BRIER_SNAPSHOT_BOUND_POINTS,
            ),
            SECONDARY_ANALYSIS_CALIBRATION_DID: (
                SECONDARY_GUARD_CLASS_CALIBRATION_DID_BRIER,
                4.0 * precision.PANEL_BRIER_SNAPSHOT_BOUND_POINTS,
            ),
        }
        try:
            guard_class, replicate_bound = brier_guards[str(analysis_type)]
        except KeyError as exc:
            raise CalibrationAnalysisError(
                "secondary Brier row has no prespecified numerical guard"
            ) from exc
    replicate_bound = float(replicate_bound)
    return {
        "row_guard_class": guard_class,
        "replicate_numerical_bound_points": replicate_bound,
        "mcse_numerical_guard_points": (
            replicate_bound / math.sqrt(normalized_m - 1)
        ),
    }


def _apply_secondary_pointwise_guard(row: Mapping[str, Any]) -> dict[str, Any]:
    """Append guarded pointwise endpoints without replacing the raw interval."""

    required = {
        "analysis_type", "metric", "pointwise_t_critical_95",
        "ci95_low", "ci95_high", "monte_carlo_replicates",
    }
    if not required.issubset(row):
        raise CalibrationAnalysisError(
            "secondary row lacks fields required for the pointwise guard"
        )
    cumulative_m = row["monte_carlo_replicates"]
    guard = secondary_row_numerical_guard(
        str(row["analysis_type"]), str(row["metric"]), cumulative_m,
    )
    critical = float(row["pointwise_t_critical_95"])
    if not math.isfinite(critical) or critical < 0.0:
        raise CalibrationAnalysisError("secondary pointwise t critical is invalid")
    halfwidth_guard = float(guard["replicate_numerical_bound_points"]) + (
        critical * float(guard["mcse_numerical_guard_points"])
    )
    raw_low = row["ci95_low"]
    raw_high = row["ci95_high"]
    if raw_low is None or raw_high is None:
        if raw_low is not None or raw_high is not None:
            raise CalibrationAnalysisError(
                "secondary raw pointwise interval is only partially unavailable"
            )
        if halfwidth_guard != 0.0:
            raise CalibrationAnalysisError(
                "a guarded Brier interval cannot have unavailable endpoints"
            )
        guarded_low = None
        guarded_high = None
        sign = "unavailable"
    else:
        lower = float(raw_low)
        upper = float(raw_high)
        if (
            not math.isfinite(lower) or not math.isfinite(upper)
            or lower > upper
        ):
            raise CalibrationAnalysisError(
                "secondary raw pointwise interval endpoints are invalid"
            )
        guarded_low = lower - halfwidth_guard
        guarded_high = upper + halfwidth_guard
        sign = guarded_scientific_sign(guarded_low, guarded_high)
    return {
        **dict(row),
        **guard,
        "pointwise_numerical_halfwidth_guard_points": halfwidth_guard,
        "guarded_ci95_low": guarded_low,
        "guarded_ci95_high": guarded_high,
        "guarded_ci95_strictly_positive": sign == "positive",
        "guarded_ci95_strictly_negative": sign == "negative",
        "guarded_ci95_excludes_zero": sign in {"positive", "negative"},
        "guarded_pointwise_scientific_sign": sign,
    }


def _t_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise CalibrationAnalysisError("paired t summary needs at least two finite seeds")
    estimate = float(np.mean(array))
    mcse = float(np.std(array, ddof=1) / math.sqrt(array.size))
    critical = float(stats.t.ppf(0.975, df=array.size - 1))
    return {
        "estimate": estimate,
        "mcse": mcse,
        "pointwise_t_critical_95": critical,
        "ci95_low": estimate - critical * mcse,
        "ci95_high": estimate + critical * mcse,
        "monte_carlo_replicates": int(array.size),
    }


def _cluster_ratio_summary(
    numerator_by_seed: Sequence[float],
    denominator_by_seed: Sequence[float],
) -> dict[str, Any]:
    numerator = np.asarray(numerator_by_seed, dtype=float)
    denominator = np.asarray(denominator_by_seed, dtype=float)
    if (
        numerator.shape != denominator.shape or numerator.ndim != 1
        or numerator.size < 2 or not np.isfinite(numerator).all()
        or not np.isfinite(denominator).all() or np.any(denominator < 0)
    ):
        raise CalibrationAnalysisError("conditional cluster ratio has an invalid denominator")
    if float(np.sum(denominator)) == 0.0:
        critical = float(stats.t.ppf(0.975, df=numerator.size - 1))
        return {
            "estimate": None, "mcse": None, "ci95_low": None, "ci95_high": None,
            "pointwise_t_critical_95": critical,
            "monte_carlo_replicates": int(numerator.size),
            "eligible_cell_denominator": 0,
        }
    ratio = float(np.sum(numerator) / np.sum(denominator))
    mean_denominator = float(np.mean(denominator))
    influence = (numerator - ratio * denominator) / mean_denominator
    mcse = float(np.std(influence, ddof=1) / math.sqrt(influence.size))
    critical = float(stats.t.ppf(0.975, df=influence.size - 1))
    return {
        "estimate": ratio,
        "mcse": mcse,
        "pointwise_t_critical_95": critical,
        "ci95_low": ratio - critical * mcse,
        "ci95_high": ratio + critical * mcse,
        "monte_carlo_replicates": int(influence.size),
        "eligible_cell_denominator": int(np.sum(denominator)),
    }


def _cell_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, float, int, float, int], Mapping[str, Any]]:
    output = {
        (
            int(row["seed"]), str(row["policy"]), float(row["gamma"]),
            int(row["stratum"]), float(row["assumed_toxicity_noise_sd_factor"]),
            int(row["budget"]),
        ): row
        for row in rows
    }
    if len(output) != len(rows):
        raise CalibrationAnalysisError("logical rows have duplicate analysis keys")
    return output


def _seed_average(
    index: Mapping[tuple[int, str, float, int, float, int], Mapping[str, Any]],
    *,
    seed: int,
    policy: str,
    factor: float,
    budget: int,
    metric: str,
    gamma: float | None = None,
    stratum: int | None = None,
) -> float:
    gates = GATES if gamma is None else (gamma,)
    strata = STRATA if stratum is None else (stratum,)
    values = [
        index[(seed, policy, gate, z, factor, budget)][metric]
        for gate in gates for z in strata
    ]
    if any(value is None for value in values):
        raise CalibrationAnalysisError(f"unconditional metric {metric} is unexpectedly null")
    return float(math.fsum(float(value) for value in values) / len(values))


def secondary_analysis(
    logical_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return absolute profiles and locked supporting contrasts."""

    index = _cell_index(logical_rows)
    seeds = sorted({int(row["seed"]) for row in logical_rows})
    absolute: list[dict[str, Any]] = []
    for policy in POLICIES:
        for factor in FACTORS:
            for budget in BUDGETS:
                for metric in ABSOLUTE_METRICS:
                    summary = _t_summary(
                        _seed_average(
                            index, seed=seed, policy=policy, factor=factor,
                            budget=budget, metric=metric,
                        )
                        for seed in seeds
                    )
                    absolute.append({
                        "analysis_type": SECONDARY_ANALYSIS_ABSOLUTE_PROFILE,
                        "policy": policy,
                        "policy_label": POLICY_LABELS[policy],
                        "assumed_toxicity_noise_sd_factor": factor,
                        "budget": budget,
                        "metric": metric,
                        **summary,
                    })

    contrasts: list[dict[str, Any]] = []
    # Absolute finite-budget movement, N=80 minus N=20.
    for policy in POLICIES:
        for factor in FACTORS:
            for metric in (
                precision.OUTCOME_TERMINAL, precision.OUTCOME_ASSIGNMENT,
                precision.OUTCOME_EXCLUSION, precision.OUTCOME_BRIER,
            ):
                summary = _t_summary(
                    _seed_average(index, seed=seed, policy=policy, factor=factor, budget=80, metric=metric)
                    - _seed_average(index, seed=seed, policy=policy, factor=factor, budget=20, metric=metric)
                    for seed in seeds
                )
                contrasts.append({
                    "analysis_type": SECONDARY_ANALYSIS_BUDGET_CONTRAST,
                    "policy": policy, "policy_label": POLICY_LABELS[policy],
                    "assumed_toxicity_noise_sd_factor": factor,
                    "budget": "80-20", "metric": metric, **summary,
                })
    # Differential calibration sensitivity versus the shared-gate cEI reference.
    for policy in ("cKG-exact-formal", "tmse", "qBIG"):
        for factor in ERRONEOUS_FACTORS:
            for budget in BUDGETS:
                for metric in (precision.OUTCOME_EXCLUSION, precision.OUTCOME_BRIER):
                    summary = _t_summary(
                        (
                            _seed_average(index, seed=seed, policy=policy, factor=factor, budget=budget, metric=metric)
                            - _seed_average(index, seed=seed, policy=policy, factor=1.0, budget=budget, metric=metric)
                        )
                        - (
                            _seed_average(index, seed=seed, policy="cEI", factor=factor, budget=budget, metric=metric)
                            - _seed_average(index, seed=seed, policy="cEI", factor=1.0, budget=budget, metric=metric)
                        )
                        for seed in seeds
                    )
                    contrasts.append({
                        "analysis_type": SECONDARY_ANALYSIS_CALIBRATION_DID,
                        "policy": policy, "policy_label": POLICY_LABELS[policy],
                        "assumed_toxicity_noise_sd_factor": factor,
                        "budget": budget, "metric": metric, **summary,
                    })
    # Gate-specific cKG comparisons remain secondary and are never pooled by a vote.
    for comparator in precision.COMPARATORS:
        for factor in FACTORS:
            for budget in BUDGETS:
                for gamma in GATES:
                    for metric in (precision.OUTCOME_TERMINAL, precision.OUTCOME_ASSIGNMENT):
                        summary = _t_summary(
                            _seed_average(index, seed=seed, policy="cKG-exact-formal", factor=factor, budget=budget, metric=metric, gamma=gamma)
                            - _seed_average(index, seed=seed, policy=comparator, factor=factor, budget=budget, metric=metric, gamma=gamma)
                            for seed in seeds
                        )
                        contrasts.append({
                            "analysis_type": "gate_specific_cKG_minus_comparator",
                            "policy": "cKG-exact-formal", "comparator": comparator,
                            "assumed_toxicity_noise_sd_factor": factor,
                            "budget": budget, "gamma": gamma, "metric": metric, **summary,
                        })
    # Conditional recovery, gate precision, and true-feasible simple regret.
    for policy in POLICIES:
        for factor in FACTORS:
            for budget in BUDGETS:
                numerator, denominator = [], []
                precision_numerator, precision_denominator = [], []
                regret_numerator, regret_denominator = [], []
                first_numerator, first_denominator = [], []
                for seed in seeds:
                    cell_rows = [
                        index[(seed, policy, gamma, stratum, factor, budget)]
                        for gamma in GATES for stratum in STRATA
                    ]
                    eligible_recovery = [
                        row for row in cell_rows
                        if bool(row["target_preterminal_exclusion_eligible"])
                    ]
                    numerator.append(sum(
                        float(row["target_recovered_after_preterminal_exclusion_pct"])
                        for row in eligible_recovery
                    ))
                    denominator.append(len(eligible_recovery))
                    nonempty = [row for row in cell_rows if int(row["final_gate_size"]) > 0]
                    precision_numerator.append(sum(
                        float(row["gate_precision_pct"]) for row in nonempty
                    ))
                    precision_denominator.append(len(nonempty))
                    feasible = [
                        row for row in cell_rows if bool(row["true_feasible_recommendation"])
                    ]
                    regret_numerator.append(sum(
                        float(row["true_feasible_simple_regret"]) for row in feasible
                    ))
                    regret_denominator.append(len(feasible))
                    ever_admitted = [
                        row for row in cell_rows
                        if row["target_first_admission_enrollment"] is not None
                    ]
                    first_numerator.append(sum(
                        float(row["target_first_admission_enrollment"])
                        for row in ever_admitted
                    ))
                    first_denominator.append(len(ever_admitted))
                for metric, summary in (
                    (
                        "recovery_after_preterminal_exclusion_conditional_pct",
                        _cluster_ratio_summary(numerator, denominator),
                    ),
                    (
                        "gate_precision_pct_conditional_on_nonempty",
                        _cluster_ratio_summary(precision_numerator, precision_denominator),
                    ),
                    (
                        "simple_regret_conditional_on_true_feasible_recommendation",
                        _cluster_ratio_summary(regret_numerator, regret_denominator),
                    ),
                    (
                        "first_admission_enrollment_conditional_on_ever_admitted",
                        _cluster_ratio_summary(first_numerator, first_denominator),
                    ),
                ):
                    contrasts.append({
                        "analysis_type": "conditional_absolute_diagnostic",
                        "policy": policy, "policy_label": POLICY_LABELS[policy],
                        "assumed_toxicity_noise_sd_factor": factor,
                        "budget": budget, "metric": metric, **summary,
                    })

    # Paired simple-regret contrasts require both recommendations to be feasible.
    for comparator in precision.COMPARATORS:
        for factor in FACTORS:
            for budget in BUDGETS:
                numerator, denominator = [], []
                for seed in seeds:
                    differences = []
                    for gamma in GATES:
                        for stratum in STRATA:
                            ckg = index[(seed, "cKG-exact-formal", gamma, stratum, factor, budget)]
                            other = index[(seed, comparator, gamma, stratum, factor, budget)]
                            if (
                                bool(ckg["true_feasible_recommendation"])
                                and bool(other["true_feasible_recommendation"])
                            ):
                                differences.append(
                                    float(ckg["true_feasible_simple_regret"])
                                    - float(other["true_feasible_simple_regret"])
                                )
                    numerator.append(math.fsum(differences))
                    denominator.append(len(differences))
                summary = _cluster_ratio_summary(numerator, denominator)
                contrasts.append({
                    "analysis_type": "matched_true_feasible_simple_regret_contrast",
                    "policy": "cKG-exact-formal", "comparator": comparator,
                    "assumed_toxicity_noise_sd_factor": factor,
                    "budget": budget,
                    "metric": "cKG_minus_comparator_true_feasible_simple_regret",
                    **summary,
                })
    guarded_absolute = [
        _apply_secondary_pointwise_guard(row) for row in absolute
    ]
    guarded_contrasts = [
        _apply_secondary_pointwise_guard(row) for row in contrasts
    ]
    return guarded_absolute, guarded_contrasts


BASELINE_COMMON_FIELDS = (
    "policy", "seed", "sim", "stratum", "gamma", "mode", "budget", "r_k",
    "grid_n", "noise", "kap", "eff_noise_sd", "tox_noise_sd", "warmup",
    "empty_gate", "protocol_scaffold", "region_step", "empty_gate_stop_after",
    "exclude_repeats_during_expansion", "stop_reason", "recommendation_made",
    "n_enrolled", "n_unique_doses", "n_empty_gate_events",
    "initialization_patients_above", "post_initialization_patients_above",
    "initialization_size", "region_q_at_stop", "dose_units", "rpsel",
    "rec_unsafe", "toxic", "n_gate_pass", "n_gate_pass_safe", "rec_d1",
    "rec_d2", "rec_true_eff", "rec_true_tox", "recs", "obd_pf", "obd_sdg",
    "traj",
)


def _baseline_key(row: Mapping[str, Any]) -> tuple[str, int, float, int]:
    policy = str(row.get("policy"))
    if policy == "cKG-exact-formal":
        policy = "cKG"
    return policy, int(row["seed"]), float(row["gamma"]), int(row["stratum"])


def _baseline_oracle_specs() -> Mapping[str, Any]:
    manifest = _read_finite_json(MANIFEST_PATH.read_bytes(), label=MANIFEST_PATH.name)
    try:
        specs = manifest["baseline_identity"]["oracles"]
    except (KeyError, TypeError) as exc:
        raise CalibrationAnalysisError("design manifest lacks baseline oracle commitments") from exc
    if not isinstance(specs, Mapping):
        raise CalibrationAnalysisError("baseline oracle commitments are malformed")
    return specs


def _baseline_filter(row: Mapping[str, Any]) -> bool:
    return (
        row.get("sim") == "osa"
        and row.get("mode") == "latent"
        and row.get("protocol_scaffold", "lhs_fixed") == "lhs_fixed"
        and float(row.get("kap", 1.0)) == 1.0
        and int(row.get("budget", 40)) == 40
        and float(row.get("gamma")) in GATES
        and int(row.get("seed")) < 200
        and str(row.get("policy")) in {"cKG", "cKG-exact-formal", "tmse", "qBIG", "cEI"}
    )


def _unwrap_oracle_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        candidates = payload["rows"]
    else:
        raise CalibrationAnalysisError("baseline oracle payload has no rows")
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise CalibrationAnalysisError("baseline oracle row is not an object")
        result = candidate.get("result", candidate)
        if isinstance(result, Mapping):
            item = dict(result)
            matrix_id = candidate.get("matrix_id")
            execution_key = candidate.get("formal_execution_key")
            if matrix_id is None and isinstance(execution_key, list) and execution_key:
                matrix_id = execution_key[0]
            if matrix_id is not None:
                item["_matrix_id"] = matrix_id
            output.append(item)
    return output


def _formal_allocation_filter(row: Mapping[str, Any]) -> bool:
    if not _baseline_filter(row):
        return False
    expected_matrix = {
        "cKG": "main_full_panel_exact_ckg",
        "cEI": "main_latent_current_cei_controls",
        "tmse": "sixway_latent_current_boundary_controls",
        "qBIG": "sixway_latent_current_boundary_controls",
    }.get(_baseline_key(row)[0])
    return row.get("_matrix_id") == expected_matrix


def _extension_allocation_filter(row: Mapping[str, Any]) -> bool:
    return (
        _baseline_filter(row)
        and row.get("_matrix_id") == "family_full_osa_gbump_boundary_completion"
        and 100 <= int(row["seed"]) < 200
    )


def baseline_identity_audit(
    records: Sequence[Mapping[str, Any]],
    common_oracle_rows: Sequence[Mapping[str, Any]],
    allocation_oracle_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Strongly compare all historical fields that were actually serialized."""

    new_rows: dict[tuple[str, int, float, int], Mapping[str, Any]] = {}
    for record in records:
        if "snapshots" in record:
            seed, _policy, _gamma, _stratum, factor = precision._path_identifiers(record)
            if factor != 1.0 or seed >= 200:
                continue
            snapshot = _snapshots(record)[40]
            legacy = snapshot.get("frozen_run_trial_common")
            if not isinstance(legacy, Mapping):
                raise CalibrationAnalysisError("baseline snapshot lacks frozen common envelope")
        else:
            legacy = record
        key = _baseline_key(legacy)
        if key in new_rows:
            raise CalibrationAnalysisError("baseline new-audit key is duplicated")
        new_rows[key] = legacy
    if len(new_rows) != 3_200:
        raise CalibrationAnalysisError(
            f"baseline audit has {len(new_rows)} new cells, expected 3200"
        )

    common_index: dict[tuple[str, int, float, int], Mapping[str, Any]] = {}
    for row in common_oracle_rows:
        if not _baseline_filter(row):
            continue
        key = _baseline_key(row)
        if key in common_index:
            raise CalibrationAnalysisError("baseline common oracle key is duplicated")
        common_index[key] = row
    if set(common_index) != set(new_rows) or len(common_index) != 3_200:
        raise CalibrationAnalysisError("baseline common oracle coverage is not 3200 cells")
    observed_common_union = {
        field for row in common_index.values() for field in row
        if not field.startswith("_") and field != "allocation_history"
    }
    if observed_common_union != set(BASELINE_COMMON_FIELDS):
        raise CalibrationAnalysisError(
            "baseline historical common-field union changed: "
            f"missing={sorted(set(BASELINE_COMMON_FIELDS) - observed_common_union)}, "
            f"extra={sorted(observed_common_union - set(BASELINE_COMMON_FIELDS))}"
        )
    mismatch_examples: list[str] = []
    common_comparisons = 0
    for key, new in new_rows.items():
        old = common_index[key]
        serialized_fields = sorted(
            field for field in old
            if not field.startswith("_") and field != "allocation_history"
        )
        for field in serialized_fields:
            if field not in new:
                raise CalibrationAnalysisError(
                    f"new baseline envelope lacks historical field {field} at {key!r}"
                )
            common_comparisons += 1
            if (
                _canonical_json_bytes(new[field], compact=True)
                != _canonical_json_bytes(old[field], compact=True)
                and len(mismatch_examples) < 5
            ):
                mismatch_examples.append(f"{key!r}:{field}")
    if mismatch_examples:
        raise CalibrationAnalysisError(
            "baseline common identity failed; examples=" + ",".join(mismatch_examples)
        )

    allocation_index: dict[tuple[str, int, float, int], Any] = {}
    for row in allocation_oracle_rows:
        if not _baseline_filter(row) or "allocation_history" not in row:
            continue
        key = _baseline_key(row)
        if key in allocation_index:
            raise CalibrationAnalysisError("baseline allocation oracle key is duplicated")
        allocation_index[key] = row["allocation_history"]
    unavailable = {
        ("cEI", seed, 0.7, stratum)
        for seed in range(80) for stratum in STRATA
    }
    missing = set(new_rows) - set(allocation_index)
    extra = set(allocation_index) - set(new_rows)
    if missing != unavailable or extra:
        raise CalibrationAnalysisError(
            "baseline allocation availability does not match the frozen 160-cell exception"
        )
    allocation_mismatches: list[str] = []
    for key in sorted(set(new_rows) - unavailable):
        if (
            _canonical_json_bytes(
                new_rows[key].get("allocation_history"), compact=True
            )
            != _canonical_json_bytes(allocation_index[key], compact=True)
        ):
            if len(allocation_mismatches) < 5:
                allocation_mismatches.append(repr(key))
    if allocation_mismatches:
        raise CalibrationAnalysisError(
            "baseline allocation identity failed; examples=" + ",".join(allocation_mismatches)
        )
    return {
        "status": "PASS_AVAILABLE_FIELD_IDENTITY",
        "common_field_cells": 3_200,
        "locked_common_fields_per_cell": len(BASELINE_COMMON_FIELDS),
        "common_field_comparisons": common_comparisons,
        "allocation_history_cells": 3_040,
        "historical_cells_without_allocation_history": 160,
        "allocation_history_exception": {
            "policy": "cEI", "gamma": 0.7, "seed_start": 0,
            "seed_stop_exclusive": 80, "strata": list(STRATA),
        },
        "construction_prefix_identity": "validated_by_frozen_core_test_contract",
    }


def load_selected_common_oracle(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Authenticate 34,000 rows, retaining only the 3,200 locked cells."""

    target = Path(path).resolve()
    metadata_path = Path(str(target) + ".metadata.json")
    if not target.is_file() or not metadata_path.is_file():
        raise CalibrationAnalysisError("selected-family oracle lacks its sidecar")
    compressed = target.read_bytes()
    spec = _baseline_oracle_specs()["selected_comparator_family"]
    if (
        target.name != Path(str(spec.get("path"))).name
        or _sha256_bytes(compressed) != spec.get("artifact_sha256")
        or _sha256_file(metadata_path) != spec.get("metadata_sha256")
    ):
        raise CalibrationAnalysisError("selected-family oracle differs from manifest pin")
    metadata = _read_finite_json(metadata_path.read_bytes(), label=metadata_path.name)
    if not isinstance(metadata, Mapping):
        raise CalibrationAnalysisError("selected-family metadata is not an object")
    commitment = metadata.get("selected_family_commitment")
    if not isinstance(commitment, Mapping):
        raise CalibrationAnalysisError("selected-family logical commitment is absent")
    if metadata.get("artifact_sha256") != _sha256_bytes(compressed):
        raise CalibrationAnalysisError("selected-family compressed SHA differs")
    try:
        logical = zstd.ZstdDecompressor().decompress(compressed)
    except zstd.ZstdError as exc:
        raise CalibrationAnalysisError("selected-family oracle is not valid zstd") from exc
    rows = _read_finite_json(logical, label=target.name)
    if not isinstance(rows, list) or len(rows) != 34_000:
        raise CalibrationAnalysisError("selected-family oracle is not 34,000 rows")
    if logical != _canonical_json_bytes(rows, compact=True):
        raise CalibrationAnalysisError("selected-family logical bytes are not canonical")
    expected = {
        "rows": 34_000,
        "compressed_bytes": len(compressed),
        "compressed_sha256": _sha256_bytes(compressed),
        "uncompressed_bytes": len(logical),
        "uncompressed_sha256": _sha256_bytes(logical),
    }
    if any(commitment.get(field) != value for field, value in expected.items()):
        raise CalibrationAnalysisError("selected-family commitment differs")
    selected = [dict(row) for row in rows if _baseline_filter(row)]
    if len(selected) != 3_200:
        raise CalibrationAnalysisError(
            "selected-family oracle does not yield 3,200 locked baseline cells"
        )
    return selected, {
        "artifact_sha256": _sha256_bytes(compressed),
        "metadata_sha256": _sha256_file(metadata_path),
    }


def load_formal_allocation_oracle(
    raw_path: str | Path,
    metadata_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Authenticate the archived formal decision master with dual hashes."""

    target = Path(raw_path).resolve()
    sidecar = Path(metadata_path).resolve()
    if not target.is_file() or not sidecar.is_file():
        raise CalibrationAnalysisError("formal allocation oracle is absent")
    compressed = target.read_bytes()
    spec = _baseline_oracle_specs()["formal_decision_master"]
    if (
        target.name != Path(str(spec.get("archive_path"))).name
        or sidecar.name != Path(str(spec.get("metadata_path"))).name
        or _sha256_bytes(compressed) != spec.get("artifact_sha256")
        or _sha256_file(sidecar) != spec.get("metadata_sha256")
    ):
        raise CalibrationAnalysisError("formal oracle differs from manifest pin")
    metadata = _read_finite_json(sidecar.read_bytes(), label=sidecar.name)
    if not isinstance(metadata, Mapping) or metadata.get("status") != "COMPLETE_FORMAL_DECISION_MASTER":
        raise CalibrationAnalysisError("formal allocation metadata status changed")
    try:
        logical = zstd.ZstdDecompressor().decompress(compressed)
    except zstd.ZstdError as exc:
        raise CalibrationAnalysisError("formal allocation oracle is not valid zstd") from exc
    if metadata.get("artifact_sha256") != _sha256_bytes(logical):
        raise CalibrationAnalysisError("formal metadata logical SHA differs")
    if metadata.get("artifact_bytes") != len(logical):
        raise CalibrationAnalysisError("formal metadata logical byte count differs")
    payload = _read_finite_json(logical, label=target.name)
    if not isinstance(payload, Mapping) or payload.get("status") != "COMPLETE_FORMAL_DECISION_MASTER":
        raise CalibrationAnalysisError("formal decision payload status changed")
    rows = _unwrap_oracle_rows(payload)
    if len(rows) != int(metadata.get("row_count", -1)):
        raise CalibrationAnalysisError("formal decision row count differs")
    selected = [
        row for row in rows
        if _formal_allocation_filter(row) and "allocation_history" in row
    ]
    if len(selected) != 2_240:
        raise CalibrationAnalysisError(
            "formal oracle does not yield 2,240 locked latent allocation histories"
        )
    return selected, {
        "compressed_sha256": _sha256_bytes(compressed),
        "uncompressed_sha256": _sha256_bytes(logical),
        "metadata_sha256": _sha256_file(sidecar),
    }


def load_extension_allocation_oracle(
    path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reuse the existing strict comparator-extension authenticator."""

    target = Path(path).resolve()
    metadata_path, commit_path = _sidecars(target)
    spec = _baseline_oracle_specs()["comparator_extension"]
    if not all(candidate.is_file() for candidate in (target, metadata_path, commit_path)):
        raise CalibrationAnalysisError("comparator-extension oracle is incomplete")
    if (
        target.name != Path(str(spec.get("archive_path"))).name
        or _sha256_file(target) != spec.get("artifact_sha256")
        or _sha256_file(metadata_path) != spec.get("metadata_sha256")
        or _sha256_file(commit_path) != spec.get("commit_sha256")
    ):
        raise CalibrationAnalysisError("comparator-extension oracle differs from manifest pin")
    try:
        from paper.analyze_comparator_family import load_extension_records
    except ImportError as exc:
        raise CalibrationAnalysisError("comparator-extension authenticator is unavailable") from exc
    try:
        rows, provenance = load_extension_records(target, verify_local_bindings=True)
    except Exception as exc:
        raise CalibrationAnalysisError("comparator-extension oracle failed authentication") from exc
    selected = [
        dict(row) for row in rows
        if _extension_allocation_filter(row) and "allocation_history" in row
    ]
    if len(selected) != 800:
        raise CalibrationAnalysisError(
            "extension oracle does not yield 800 locked allocation histories"
        )
    return selected, dict(provenance)


def _baseline_oracle_hashes() -> dict[str, str]:
    """Flatten only the SHA commitments frozen in the local design manifest."""

    output: dict[str, str] = {}
    for oracle_name, spec in _baseline_oracle_specs().items():
        if not isinstance(spec, Mapping):
            raise CalibrationAnalysisError("baseline oracle specification is malformed")
        for field, value in spec.items():
            if not str(field).endswith("_sha256"):
                continue
            digest = _require_sha(value, label=f"baseline_identity.{oracle_name}.{field}")
            output[f"{oracle_name}.{field}"] = digest
    if len(output) != 7:
        raise CalibrationAnalysisError("baseline oracle SHA commitment count changed")
    return output


def _m0200_baseline_state(
    raw_block_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Authenticate all M0200 shards and retain only factor-one N=40 envelopes."""

    path = Path(raw_block_path).resolve()
    payload, provenance, paths = precision.authenticate_seed_block(
        path, verify_local_bindings=True
    )
    if payload["block"]["block_id"] != "M0200":
        raise CalibrationAnalysisError("baseline identity must use the M0200 block")
    staging_root = path.parent.parent
    input_hashes: dict[str, str] = {}
    for file_path, hash_field in (
        (paths["path"], "artifact_sha256"),
        (paths["metadata"], "metadata_sha256"),
        (paths["commit"], "commit_sha256"),
    ):
        try:
            label = Path(file_path).relative_to(staging_root).as_posix()
        except ValueError as exc:
            raise CalibrationAnalysisError(
                "M0200 block is outside its staging root"
            ) from exc
        input_hashes[label] = str(provenance[hash_field])

    baseline_envelopes: list[dict[str, Any]] = []
    authenticated_paths = 0
    for shard in payload["shard_commits"]:
        for field, hash_field in (
            ("artifact", "artifact_sha256"),
            ("metadata", "metadata_sha256"),
            ("commit", "commit_sha256"),
        ):
            label = str(shard[field])
            if label in input_hashes:
                raise CalibrationAnalysisError("baseline input-hash label is duplicated")
            input_hashes[label] = _require_sha(
                shard[hash_field], label=f"baseline_input_hashes.{label}"
            )
        for record in precision.iter_authenticated_shard_results(
            staging_root, shard, bindings=payload["bindings"], validate_core=True
        ):
            authenticated_paths += 1
            seed, _policy, _gamma, _stratum, factor = precision._path_identifiers(record)
            if seed < 200 and factor == 1.0:
                legacy = _snapshots(record)[40].get("frozen_run_trial_common")
                if not isinstance(legacy, Mapping):
                    raise CalibrationAnalysisError(
                        "M0200 baseline snapshot lacks its frozen common envelope"
                    )
                baseline_envelopes.append(dict(legacy))
    if authenticated_paths != 9_600 or len(baseline_envelopes) != 3_200:
        raise CalibrationAnalysisError(
            "M0200 baseline streaming coverage is not 9,600 paths / 3,200 cells"
        )
    input_hashes["paper/budget_toxicity_calibration_manifest.json"] = _sha256_file(
        MANIFEST_PATH
    )
    input_hashes["paper/budget_toxicity_calibration_prespec.md"] = _sha256_file(
        PRESPEC_PATH
    )
    if len(input_hashes) != 77:
        raise CalibrationAnalysisError("M0200 baseline input commitment count changed")
    return baseline_envelopes, input_hashes


def _validate_baseline_pass_payload(payload: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema_version", "status", "artifact_class", "cumulative_M",
        "input_hashes", "oracle_hashes", "checked_common_cells",
        "checked_allocation_histories", "unavailable_allocation_histories",
        "exception_definition", "pass",
    }
    if set(payload) != expected_fields:
        raise CalibrationAnalysisError("baseline PASS payload fields changed")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != BASELINE_PASS_STATUS
        or payload.get("artifact_class") != BASELINE_PASS_ARTIFACT_CLASS
        or payload.get("cumulative_M") != 200
        or payload.get("pass") is not True
        or payload.get("checked_common_cells") != 3_200
        or payload.get("checked_allocation_histories") != 3_040
        or payload.get("unavailable_allocation_histories") != 160
    ):
        raise CalibrationAnalysisError("baseline PASS envelope/counts changed")
    expected_exception = {
        "policy": "cEI", "gamma": 0.7, "seed_start": 0,
        "seed_stop_exclusive": 80, "strata": [0, 1], "count": 160,
    }
    if payload.get("exception_definition") != expected_exception:
        raise CalibrationAnalysisError("baseline allocation exception changed")
    hashes = payload.get("input_hashes")
    if not isinstance(hashes, Mapping) or len(hashes) != 77:
        raise CalibrationAnalysisError("baseline PASS input hashes are incomplete")
    for label, digest in hashes.items():
        _require_sha(digest, label=f"baseline_pass.input_hashes.{label}")
    if payload.get("oracle_hashes") != _baseline_oracle_hashes():
        raise CalibrationAnalysisError("baseline PASS oracle hashes differ from manifest")


def write_baseline_identity_pass(
    path: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Path]:
    """Write the authenticated pre-unblinding PASS-only artifact."""

    _assert_exact_runtime()
    _validate_baseline_pass_payload(payload)
    target = Path(path).resolve()
    if target.suffixes[-2:] != [".json", ".zst"]:
        raise CalibrationAnalysisError("baseline PASS artifact must end in .json.zst")
    metadata_path, commit_path = _sidecars(target)
    existence = tuple(
        candidate.is_file() for candidate in (target, metadata_path, commit_path)
    )
    if any(existence):
        if not all(existence):
            raise CalibrationAnalysisError(
                "baseline PASS has a partial immutable envelope"
            )
        existing = load_baseline_identity_pass(
            target, expected_input_hashes=payload["input_hashes"], expected_m=200
        )
        if existing != dict(payload):
            raise CalibrationAnalysisError(
                "committed baseline PASS differs; immutable overwrite refused"
            )
        return {"artifact": target, "metadata": metadata_path, "commit": commit_path}
    logical = _canonical_json_bytes(dict(payload))
    compressed = zstd.ZstdCompressor(
        level=19, threads=0, write_checksum=True, write_content_size=True,
        write_dict_id=False,
    ).compress(logical)
    shared = {
        "cumulative_M": 200, "pass": True, "checked_common_cells": 3_200,
        "checked_allocation_histories": 3_040,
        "unavailable_allocation_histories": 160,
    }
    metadata = {
        "schema_version": 1, "status": BASELINE_PASS_STATUS,
        "artifact_class": BASELINE_PASS_ARTIFACT_CLASS,
        "artifact": target.name, "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(logical),
        "uncompressed_bytes": len(logical),
        "input_hashes": dict(payload["input_hashes"]),
        "oracle_hashes": dict(payload["oracle_hashes"]), **shared,
    }
    metadata_raw = _canonical_json_bytes(metadata)
    commit = {
        "schema_version": 1, "status": BASELINE_PASS_COMMIT_STATUS,
        "artifact": target.name, "artifact_sha256": _sha256_bytes(compressed),
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256_bytes(metadata_raw), **shared,
    }
    _atomic_write(target, compressed)
    _atomic_write(metadata_path, metadata_raw)
    _atomic_write(commit_path, _canonical_json_bytes(commit))
    return {"artifact": target, "metadata": metadata_path, "commit": commit_path}


def load_baseline_identity_pass(
    path: str | Path,
    *,
    expected_input_hashes: Mapping[str, str],
    expected_m: int = 200,
) -> dict[str, Any]:
    """Authenticate the hard baseline gate used by runner and monitor."""

    _assert_exact_runtime()
    target = Path(path).resolve()
    metadata_path, commit_path = _sidecars(target)
    if not all(candidate.is_file() for candidate in (target, metadata_path, commit_path)):
        raise CalibrationAnalysisError("baseline PASS artifact is incomplete")
    compressed = target.read_bytes()
    metadata_raw, commit_raw = metadata_path.read_bytes(), commit_path.read_bytes()
    metadata = _read_finite_json(metadata_raw, label=metadata_path.name)
    commit = _read_finite_json(commit_raw, label=commit_path.name)
    if not isinstance(metadata, dict) or not isinstance(commit, dict):
        raise CalibrationAnalysisError("baseline PASS sidecars must be objects")
    try:
        logical = zstd.ZstdDecompressor().decompress(compressed)
    except zstd.ZstdError as exc:
        raise CalibrationAnalysisError("baseline PASS artifact is not valid zstd") from exc
    payload = _read_finite_json(logical, label=target.name)
    if not isinstance(payload, dict) or logical != _canonical_json_bytes(payload):
        raise CalibrationAnalysisError("baseline PASS artifact is not canonical JSON")
    _validate_baseline_pass_payload(payload)
    if expected_m != 200 or payload["cumulative_M"] != expected_m:
        raise CalibrationAnalysisError("baseline PASS has the wrong cumulative M")
    if payload["input_hashes"] != dict(expected_input_hashes):
        raise CalibrationAnalysisError("baseline PASS is bound to different M0200 inputs")
    shared = {
        "cumulative_M": 200, "pass": True, "checked_common_cells": 3_200,
        "checked_allocation_histories": 3_040,
        "unavailable_allocation_histories": 160,
    }
    expected_metadata = {
        "schema_version": 1, "status": BASELINE_PASS_STATUS,
        "artifact_class": BASELINE_PASS_ARTIFACT_CLASS,
        "artifact": target.name, "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(logical),
        "uncompressed_bytes": len(logical),
        "input_hashes": dict(payload["input_hashes"]),
        "oracle_hashes": dict(payload["oracle_hashes"]), **shared,
    }
    if metadata != expected_metadata:
        raise CalibrationAnalysisError("baseline PASS metadata differs")
    expected_commit = {
        "schema_version": 1, "status": BASELINE_PASS_COMMIT_STATUS,
        "artifact": target.name, "artifact_sha256": _sha256_bytes(compressed),
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256_bytes(metadata_raw), **shared,
    }
    if commit != expected_commit:
        raise CalibrationAnalysisError("baseline PASS commit differs")
    return dict(payload)


def run_baseline_only(
    *,
    raw_block_path: str | Path,
    selected_oracle_path: str | Path,
    formal_oracle_path: str | Path,
    formal_oracle_metadata_path: str | Path,
    extension_oracle_path: str | Path,
    staging_root: str | Path,
) -> dict[str, Path]:
    """Authenticate M0200 identity and commit only PASS/count information."""

    _assert_exact_runtime()
    resolved_staging_root = Path(staging_root).resolve()
    derived_staging_root = Path(raw_block_path).resolve().parent.parent
    if resolved_staging_root != derived_staging_root:
        raise CalibrationAnalysisError(
            "baseline output root must equal the M0200 block staging root"
        )
    baseline_envelopes, input_hashes = _m0200_baseline_state(raw_block_path)
    selected_rows, _selected_provenance = load_selected_common_oracle(
        selected_oracle_path
    )
    formal_rows, _formal_provenance = load_formal_allocation_oracle(
        formal_oracle_path, formal_oracle_metadata_path
    )
    extension_rows, _extension_provenance = load_extension_allocation_oracle(
        extension_oracle_path
    )
    audit = baseline_identity_audit(
        baseline_envelopes, selected_rows, [*formal_rows, *extension_rows]
    )
    if audit.get("status") != "PASS_AVAILABLE_FIELD_IDENTITY":
        raise CalibrationAnalysisError("baseline identity did not pass")
    payload = {
        "schema_version": 1, "status": BASELINE_PASS_STATUS,
        "artifact_class": BASELINE_PASS_ARTIFACT_CLASS, "cumulative_M": 200,
        "input_hashes": input_hashes, "oracle_hashes": _baseline_oracle_hashes(),
        "checked_common_cells": 3_200, "checked_allocation_histories": 3_040,
        "unavailable_allocation_histories": 160,
        "exception_definition": {
            "policy": "cEI", "gamma": 0.7, "seed_start": 0,
            "seed_stop_exclusive": 80, "strata": [0, 1], "count": 160,
        },
        "pass": True,
    }
    target = (
        resolved_staging_root / "baseline"
        / "budget_toxicity_calibration_baseline_identity_M0200.json.zst"
    )
    return write_baseline_identity_pass(target, payload)


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


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise CalibrationAnalysisError("cannot serialize an empty CSV")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.2f}"


def _primary_tex(rows: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [
        r"\begin{longtable}{llrrrrrl}",
        r"\toprule",
        r"Family & Estimand & Estimate & MCSE & Pointwise 95\% CI & Raw simultaneous 95\% band & Guarded simultaneous 95\% band & Guarded sign \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Family & Estimand & Estimate & MCSE & Pointwise 95\% CI & Raw simultaneous 95\% band & Guarded simultaneous 95\% band & Guarded sign \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        estimand = str(row["estimand_id"]).replace("_", r"\_").replace("|", r"\textbar{}")
        lines.append(
            f"{row['family']} & {estimand} & {_fmt(row['estimate_points'])} & "
            f"{_fmt(row['mcse_points'])} & "
            f"[{_fmt(row['pointwise_95_low_points'])}, {_fmt(row['pointwise_95_high_points'])}] & "
            f"[{_fmt(row['simultaneous_95_low_points'])}, {_fmt(row['simultaneous_95_high_points'])}] & "
            f"[{_fmt(row['guarded_simultaneous_95_low_points'])}, "
            f"{_fmt(row['guarded_simultaneous_95_high_points'])}] & "
            f"{row['guarded_scientific_sign']} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _secondary_tex(
    absolute: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]]
) -> bytes:
    headline_metrics = {
        precision.OUTCOME_TERMINAL, precision.OUTCOME_ASSIGNMENT,
        precision.OUTCOME_EXCLUSION, precision.OUTCOME_BRIER,
        "panel_correct_selection_pct",
    }
    selected = [row for row in absolute if row["metric"] in headline_metrics]
    lines = [
        r"\begin{longtable}{lllrrrrrl}", r"\toprule",
        r"Policy & $c_\sigma$ & Metric & $N$ & Estimate & MCSE & Raw 95\% MC interval & Guarded 95\% MC interval & Guarded sign \\",
        r"\midrule", r"\endfirsthead", r"\toprule",
        r"Policy & $c_\sigma$ & Metric & $N$ & Estimate & MCSE & Raw 95\% MC interval & Guarded 95\% MC interval & Guarded sign \\",
        r"\midrule", r"\endhead",
    ]
    for row in selected:
        metric = str(row["metric"]).replace("_", r"\_")
        lines.append(
            f"{row['policy_label']} & {row['assumed_toxicity_noise_sd_factor']:g} & "
            f"{metric} & {row['budget']} & {_fmt(row['estimate'])} & {_fmt(row['mcse'])} & "
            f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}] & "
            f"[{_fmt(row['guarded_ci95_low'])}, "
            f"{_fmt(row['guarded_ci95_high'])}] & "
            f"{row['guarded_pointwise_scientific_sign']} \\\\"
        )
    lines.extend([
        r"\bottomrule", r"\end{longtable}",
        f"% Complete supporting-contrast rows: {len(contrasts)}",
    ])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _artifact_sidecar(
    artifact: Path,
    *,
    artifact_type: str,
    provenance: Mapping[str, Any],
    baseline: Mapping[str, Any],
    analysis_details: Mapping[str, Any],
) -> Path:
    metadata = {
        "schema_version": 1,
        "status": "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_ANALYSIS_ARTIFACT",
        "artifact_type": artifact_type,
        "artifact": artifact.name,
        "artifact_sha256": _sha256_file(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "analyzer": "paper/analyze_budget_toxicity_calibration_audit.py",
        "analyzer_sha256": _sha256_file(Path(__file__)),
        "prespec_sha256": _sha256_file(PRESPEC_PATH),
        "manifest_sha256": _sha256_file(MANIFEST_PATH),
        "raw_provenance": dict(provenance),
        "baseline_identity": dict(baseline),
        "analysis": dict(analysis_details),
    }
    path = Path(str(artifact) + ".metadata.json")
    _atomic_write(path, _canonical_json_bytes(metadata))
    return path


def _save_figure(
    figure: Any,
    stem: Path,
    *,
    provenance: Mapping[str, Any],
    baseline: Mapping[str, Any],
    description: str,
) -> list[Path]:
    outputs: list[Path] = []
    fixed_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
    for suffix in (".png", ".pdf", ".eps"):
        path = stem.with_suffix(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".png":
            metadata = {
                "Title": description,
                "Author": "dose-combination-bo calibration analyzer",
                "Software": "Matplotlib 3.10.7",
            }
        elif suffix == ".pdf":
            metadata = {
                "Title": description,
                "Author": "dose-combination-bo calibration analyzer",
                "Creator": "dose-combination-bo calibration analyzer",
                "Producer": "Matplotlib 3.10.7",
                "CreationDate": fixed_date,
                "ModDate": fixed_date,
            }
        else:
            metadata = {"Creator": "dose-combination-bo calibration analyzer"}
        figure.savefig(
            path, dpi=600 if suffix == ".png" else None, bbox_inches="tight",
            facecolor="white", edgecolor="none", transparent=False,
            metadata=metadata,
        )
        _artifact_sidecar(
            path, artifact_type="calibration_audit_figure", provenance=provenance,
            baseline=baseline, analysis_details={"description": description},
        )
        outputs.append(path)
    return outputs


def make_figures(
    primary_rows: Sequence[Mapping[str, Any]],
    absolute_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    provenance: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> list[Path]:
    _assert_exact_runtime()
    previous_source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "946684800"
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
        "font.family": "DejaVu Sans",
    })
    import matplotlib.pyplot as plt

    generated: list[Path] = []
    colors = {0.5: "#2878B5", 1.0: "#666666", 2.0: "#D95319"}
    grid_color = "#D9D9D9"

    def draw_interval_line(
        ax: Any, x: Sequence[int], y: Sequence[float], low: Sequence[float],
        high: Sequence[float], *, factor: float, label: str,
    ) -> None:
        lower = np.maximum(0.0, np.asarray(y) - np.asarray(low))
        upper = np.maximum(0.0, np.asarray(high) - np.asarray(y))
        ax.errorbar(
            x, y, yerr=np.vstack((lower, upper)), marker="o", markersize=4,
            linewidth=1.4, elinewidth=0.9, capsize=2.2, capthick=0.9,
            color=colors[factor], label=label, zorder=2,
        )

    # Figure 1: the acquisition trade-off, preserving comparator and outcome identity.
    family_a = [row for row in primary_rows if row["family"] == "A"]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.0), sharex=True)
    for i, comparator in enumerate(precision.COMPARATORS):
        for j, outcome in enumerate((precision.OUTCOME_TERMINAL, precision.OUTCOME_ASSIGNMENT)):
            ax = axes[i, j]
            for factor in FACTORS:
                points = [
                    row for row in family_a
                    if f"cKG-minus-{comparator}|{outcome}" in row["estimand_id"]
                    and f"c={precision._factor_label(factor)}" in row["estimand_id"]
                ]
                points.sort(key=lambda row: int(row["estimand_id"].split("N=")[1].split("|")[0]))
                x = [int(row["estimand_id"].split("N=")[1].split("|")[0]) for row in points]
                y = [row["estimate_points"] for row in points]
                low = [
                    row["guarded_simultaneous_95_low_points"] for row in points
                ]
                high = [
                    row["guarded_simultaneous_95_high_points"] for row in points
                ]
                draw_interval_line(
                    ax, x, y, low, high, factor=factor, label=f"{factor:g}×"
                )
            ax.axhline(0.0, color="black", linewidth=0.8)
            short_outcome = {
                precision.OUTCOME_TERMINAL: "Terminal unsafe recommendation",
                precision.OUTCOME_ASSIGNMENT: "Above-boundary assignment",
            }[outcome]
            ax.set_title(
                f"cKG − {POLICY_LABELS[comparator]}: {short_outcome}", fontsize=8.5
            )
            ax.set_ylabel("Difference (percentage points)")
            ax.grid(color=grid_color, linewidth=0.6)
    for ax in axes[-1, :]:
        ax.set_xlabel("Patient budget N")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False,
        title="Assumed toxicity-noise SD / true SD",
    )
    fig.suptitle("Terminal recommendation and in-trial assignment contrasts", fontweight="semibold")
    fig.subplots_adjust(bottom=0.09, hspace=0.42)
    generated.extend(_save_figure(
        fig, output_dir / "figure_calibration_acquisition_tradeoff",
        provenance=provenance, baseline=baseline,
        description="Gate-and-stratum averaged cKG policy contrasts across budgets and assumed toxicity-noise factors",
    ))
    plt.close(fig)

    # Figure 2: erroneous-versus-correct calibration penalties (Family B).
    family_b = [row for row in primary_rows if row["family"] == "B"]
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.2), sharex=True)
    for column, policy in enumerate(POLICIES):
        for row_index, outcome in enumerate((precision.OUTCOME_EXCLUSION, precision.OUTCOME_BRIER)):
            ax = axes[row_index, column]
            for factor in ERRONEOUS_FACTORS:
                points = [
                    row for row in family_b
                    if f"B|{policy}|{outcome}" in row["estimand_id"]
                    and f"c={precision._factor_label(factor)}-minus-1" in row["estimand_id"]
                ]
                points.sort(key=lambda row: int(row["estimand_id"].split("N=")[1].split("|")[0]))
                x = [int(row["estimand_id"].split("N=")[1].split("|")[0]) for row in points]
                y = [row["estimate_points"] for row in points]
                low = [
                    row["guarded_simultaneous_95_low_points"] for row in points
                ]
                high = [
                    row["guarded_simultaneous_95_high_points"] for row in points
                ]
                draw_interval_line(
                    ax, x, y, low, high, factor=factor,
                    label=f"{factor:g}× vs 1×",
                )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(POLICY_LABELS[policy], fontsize=9)
            if column == 0:
                outcome_label = {
                    precision.OUTCOME_EXCLUSION: "Target-dose exclusion (%)",
                    precision.OUTCOME_BRIER: "Panel Brier score ×100",
                }[outcome]
                ax.set_ylabel(outcome_label + "\nDifference (points)")
            ax.grid(color=grid_color, linewidth=0.6)
    for ax in axes[-1, :]:
        ax.set_xlabel("Patient budget N")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False,
        title="Assumed toxicity-noise SD / true SD",
    )
    fig.suptitle("Sensitivity to assumed toxicity observation-noise calibration", fontweight="semibold")
    fig.subplots_adjust(bottom=0.12, hspace=0.36)
    generated.extend(_save_figure(
        fig, output_dir / "figure_calibration_penalties",
        provenance=provenance, baseline=baseline,
        description=(
            "Erroneous-versus-correct assumed toxicity-noise effects with "
            "numerically guarded familywise max-t bands"
        ),
    ))
    plt.close(fig)

    # Figure 3: absolute gate-target exclusion and panel-wide Brier profiles.
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.2), sharex=True)
    for column, policy in enumerate(POLICIES):
        for row_index, metric in enumerate((precision.OUTCOME_EXCLUSION, precision.OUTCOME_BRIER)):
            ax = axes[row_index, column]
            for factor in FACTORS:
                points = [
                    row for row in absolute_rows
                    if row["policy"] == policy and row["metric"] == metric
                    and float(row["assumed_toxicity_noise_sd_factor"]) == factor
                ]
                points.sort(key=lambda row: int(row["budget"]))
                x = [int(row["budget"]) for row in points]
                y = [row["estimate"] for row in points]
                low = [row["guarded_ci95_low"] for row in points]
                high = [row["guarded_ci95_high"] for row in points]
                draw_interval_line(
                    ax, x, y, low, high, factor=factor, label=f"{factor:g}×"
                )
            ax.set_title(POLICY_LABELS[policy], fontsize=9)
            if column == 0:
                ax.set_ylabel({
                    precision.OUTCOME_EXCLUSION: "Target-dose exclusion (%)",
                    precision.OUTCOME_BRIER: "Panel Brier score ×100",
                }[metric])
            ax.grid(color=grid_color, linewidth=0.6)
    for ax in axes[-1, :]:
        ax.set_xlabel("Patient budget N")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False,
        title="Assumed toxicity-noise SD / true SD",
    )
    fig.suptitle("Decision-target exclusion versus panel-wide posterior calibration", fontweight="semibold")
    fig.subplots_adjust(bottom=0.12, hspace=0.36)
    generated.extend(_save_figure(
        fig, output_dir / "figure_gate_failure_recovery_profiles",
        provenance=provenance, baseline=baseline,
        description=(
            "Absolute terminal target-exclusion and panel Brier profiles "
            "across finite budgets with numerically guarded pointwise intervals"
        ),
    ))
    plt.close(fig)
    if previous_source_date_epoch is None:
        os.environ.pop("SOURCE_DATE_EPOCH", None)
    else:
        os.environ["SOURCE_DATE_EPOCH"] = previous_source_date_epoch
    return generated


def write_analysis_artifacts(
    output_dir: str | Path,
    *,
    primary_rows: Sequence[Mapping[str, Any]],
    primary_details: Mapping[str, Any],
    absolute_rows: Sequence[Mapping[str, Any]],
    secondary_contrasts: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Write primary/secondary CSV, TeX, JSON, figures, and sidecars."""

    _assert_exact_runtime()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    secondary_rows = [
        {"section": "absolute", **dict(row)} for row in absolute_rows
    ] + [
        {"section": "contrast_or_conditional", **dict(row)}
        for row in secondary_contrasts
    ]
    artifacts: list[tuple[Path, str]] = []
    primary_csv = output / "budget_toxicity_calibration_primary.csv"
    primary_tex = output / "budget_toxicity_calibration_primary.tex"
    primary_json = output / "budget_toxicity_calibration_primary.json"
    secondary_csv = output / "budget_toxicity_calibration_secondary.csv"
    secondary_tex = output / "budget_toxicity_calibration_secondary.tex"
    secondary_json = output / "budget_toxicity_calibration_secondary.json"
    baseline_json = output / "budget_toxicity_calibration_baseline_identity.json"
    _atomic_write(primary_csv, _csv_bytes(primary_rows))
    _atomic_write(primary_tex, _primary_tex(primary_rows))
    _atomic_write(primary_json, _canonical_json_bytes({
        "schema_version": 2,
        "analysis": "locked_primary_118_estimands",
        "rows": list(primary_rows),
        "multiplicity": dict(primary_details),
    }))
    _atomic_write(secondary_csv, _csv_bytes(secondary_rows))
    _atomic_write(secondary_tex, _secondary_tex(absolute_rows, secondary_contrasts))
    _atomic_write(secondary_json, _canonical_json_bytes({
        "schema_version": 2,
        "analysis": "prespecified_secondary_diagnostics",
        "absolute_rows": list(absolute_rows),
        "contrast_and_conditional_rows": list(secondary_contrasts),
        "numerical_equivalence_guard": (
            secondary_numerical_equivalence_guard_metadata()
        ),
        "scientific_sign_source": "guarded pointwise 95% interval only",
        "cell_vote_used": False,
    }))
    _atomic_write(baseline_json, _canonical_json_bytes(dict(baseline)))
    artifacts.extend([
        (primary_csv, "primary_contrasts_csv"),
        (primary_tex, "primary_contrasts_tex"),
        (primary_json, "primary_contrasts_json"),
        (secondary_csv, "secondary_diagnostics_csv"),
        (secondary_tex, "secondary_diagnostics_tex"),
        (secondary_json, "secondary_diagnostics_json"),
        (baseline_json, "baseline_identity_json"),
    ])
    sidecars = []
    for artifact, artifact_type in artifacts:
        sidecars.append(_artifact_sidecar(
            artifact, artifact_type=artifact_type, provenance=provenance,
            baseline=baseline,
            analysis_details={
                "primary_estimand_count": 118,
                "secondary_absolute_row_count": len(absolute_rows),
                "secondary_contrast_row_count": len(secondary_contrasts),
                "max_t_draws": MAX_T_DRAWS,
                "max_t_random_seeds": MAX_T_SEEDS,
                "numerical_equivalence_guard": (
                    precision.numerical_equivalence_guard_metadata()
                ),
                "secondary_numerical_equivalence_guard": (
                    secondary_numerical_equivalence_guard_metadata()
                ),
                "scientific_sign_source": (
                    "guarded simultaneous 95% band only"
                ),
                "secondary_scientific_sign_source": (
                    "guarded pointwise 95% interval only"
                ),
                "cell_vote_used": False,
            },
        ))
    figures = make_figures(
        primary_rows, absolute_rows, output,
        provenance=provenance, baseline=baseline,
    )
    artifact_manifest = {
        path.name: {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "metadata": Path(str(path) + ".metadata.json").name,
            "metadata_sha256": _sha256_file(Path(str(path) + ".metadata.json")),
        }
        for path, _kind in artifacts
    }
    for path in figures:
        artifact_manifest[path.name] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "metadata": Path(str(path) + ".metadata.json").name,
            "metadata_sha256": _sha256_file(Path(str(path) + ".metadata.json")),
        }
    manifest_path = output / "budget_toxicity_calibration_analysis.metadata.json"
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_ANALYSIS",
        "artifact_class": "authenticated_postprocessing_analysis_not_raw_trials",
        "analyzer_sha256": _sha256_file(Path(__file__)),
        "prespec_sha256": _sha256_file(PRESPEC_PATH),
        "manifest_sha256": _sha256_file(MANIFEST_PATH),
        "raw_provenance": dict(provenance),
        "baseline_identity": dict(baseline),
        "primary_estimand_count": 118,
        "numerical_equivalence_guard": (
            precision.numerical_equivalence_guard_metadata()
        ),
        "secondary_numerical_equivalence_guard": (
            secondary_numerical_equivalence_guard_metadata()
        ),
        "scientific_sign_source": "guarded simultaneous 95% band only",
        "secondary_scientific_sign_source": (
            "guarded pointwise 95% interval only"
        ),
        "secondary_absolute_row_count": len(absolute_rows),
        "secondary_contrast_row_count": len(secondary_contrasts),
        "artifacts": artifact_manifest,
    }
    _atomic_write(manifest_path, _canonical_json_bytes(manifest))
    commit_path = output / "budget_toxicity_calibration_analysis.commit.json"
    _atomic_write(commit_path, _canonical_json_bytes({
        "schema_version": 1,
        "status": "COMMITTED_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_ANALYSIS",
        "metadata": manifest_path.name,
        "metadata_sha256": _sha256_file(manifest_path),
        "artifact_count": len(artifact_manifest),
        "baseline_identity_passed": baseline.get("status") == "PASS_AVAILABLE_FIELD_IDENTITY",
    }))
    return {
        "artifacts": artifact_manifest,
        "metadata": manifest_path,
        "commit": commit_path,
        "sidecar_count": len(sidecars) + len(figures),
    }


def run_analysis(
    *,
    raw_master_path: str | Path,
    raw_block_paths: Sequence[str | Path],
    precision_decision_paths: Sequence[str | Path],
    selected_oracle_path: str | Path,
    formal_oracle_path: str | Path,
    formal_oracle_metadata_path: str | Path,
    extension_oracle_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    _assert_exact_runtime()
    master = load_raw_master(
        raw_master_path, block_paths=raw_block_paths,
        verify_local_bindings=True, validate_core=True,
    )
    decisions = validate_top_up_history(master, precision_decision_paths)
    selected_rows, selected_provenance = load_selected_common_oracle(
        selected_oracle_path
    )
    formal_rows, formal_provenance = load_formal_allocation_oracle(
        formal_oracle_path, formal_oracle_metadata_path,
    )
    extension_rows, extension_provenance = load_extension_allocation_oracle(
        extension_oracle_path
    )
    logical_rows, baseline_envelopes, streaming_audit = stream_master_analysis_state(
        master
    )
    baseline = baseline_identity_audit(
        baseline_envelopes, selected_rows, [*formal_rows, *extension_rows]
    )
    baseline["streaming_and_crn_audit"] = streaming_audit
    baseline["oracle_provenance"] = {
        "selected_family": selected_provenance,
        "formal_decision_master": formal_provenance,
        "comparator_extension": extension_provenance,
    }
    primary_rows, primary_details = primary_analysis(logical_rows)
    absolute_rows, secondary_contrasts = secondary_analysis(logical_rows)
    provenance = dict(master.provenance)
    provenance["streaming_and_crn_audit"] = streaming_audit
    provenance["precision_decisions"] = [
        {
            "cumulative_M": decision["cumulative_M"],
            "maximum_raw_mcse_points": decision["maximum_raw_mcse_points"],
            "maximum_guarded_mcse_points": decision[
                "maximum_guarded_mcse_points"
            ],
            "all_estimands_pass_guarded_1_5pp": decision[
                "all_estimands_pass_guarded_1_5pp"
            ],
            "top_up_limit_reached": decision["top_up_limit_reached"],
            "numerical_equivalence_guard": decision[
                "numerical_equivalence_guard"
            ],
            "next_M": decision["next_M"],
            "input_hashes": decision["input_hashes"],
        }
        for decision in decisions
    ]
    return write_analysis_artifacts(
        output_dir,
        primary_rows=primary_rows,
        primary_details=primary_details,
        absolute_rows=absolute_rows,
        secondary_contrasts=secondary_contrasts,
        provenance=provenance,
        baseline=baseline,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="authenticate M0200 historical identity and write the mandatory PASS",
    )
    parser.add_argument("--raw-master", type=Path)
    parser.add_argument(
        "--raw-block", required=True, action="append", type=Path,
        help="committed raw seed block, repeated in cumulative order",
    )
    parser.add_argument(
        "--precision-decision", action="append", type=Path,
        help="committed blinded decision, repeated in cumulative order",
    )
    parser.add_argument("--baseline-selected", required=True, type=Path)
    parser.add_argument("--baseline-formal-raw", required=True, type=Path)
    parser.add_argument("--baseline-formal-metadata", required=True, type=Path)
    parser.add_argument("--baseline-extension", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--staging-root", type=Path,
        help="runner staging root; required only with --baseline-only",
    )
    args = parser.parse_args(argv)
    if args.baseline_only:
        if (
            len(args.raw_block) != 1 or args.raw_master is not None
            or args.precision_decision or args.staging_root is None
        ):
            parser.error(
                "--baseline-only requires exactly one --raw-block and --staging-root, "
                "and forbids --raw-master/--precision-decision"
            )
        paths = run_baseline_only(
            raw_block_path=args.raw_block[0],
            selected_oracle_path=args.baseline_selected,
            formal_oracle_path=args.baseline_formal_raw,
            formal_oracle_metadata_path=args.baseline_formal_metadata,
            extension_oracle_path=args.baseline_extension,
            staging_root=args.staging_root,
        )
        print(f"baseline identity PASS: {paths['artifact']}")
        return 0
    if (
        args.raw_master is None or not args.precision_decision
        or args.output_dir is None or args.staging_root is not None
    ):
        parser.error(
            "full analysis requires --raw-master, --precision-decision, and "
            "--output-dir; --staging-root is baseline-only"
        )
    result = run_analysis(
        raw_master_path=args.raw_master,
        raw_block_paths=args.raw_block,
        precision_decision_paths=args.precision_decision,
        selected_oracle_path=args.baseline_selected,
        formal_oracle_path=args.baseline_formal_raw,
        formal_oracle_metadata_path=args.baseline_formal_metadata,
        extension_oracle_path=args.baseline_extension,
        output_dir=args.output_dir,
    )
    print(f"analysis metadata: {result['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CalibrationAnalysisError",
    "MAX_T_DRAWS",
    "MAX_T_SEEDS",
    "RawMaster",
    "baseline_identity_audit",
    "explode_and_validate",
    "guarded_scientific_sign",
    "load_raw_master",
    "load_baseline_identity_pass",
    "max_t_family_summary",
    "primary_analysis",
    "recompute_snapshot_metrics",
    "run_baseline_only",
    "secondary_analysis",
    "secondary_numerical_equivalence_guard_metadata",
    "secondary_row_numerical_guard",
    "validate_top_up_history",
    "write_baseline_identity_pass",
    "write_analysis_artifacts",
]
