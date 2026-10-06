#!/usr/bin/env python3
"""Read and authenticate a committed formal-projection record bundle.

The loader is deliberately post-processing only: it reads the projection
metadata and compressed JSON records, verifies their content commitments, and
returns the authenticated logical records together with portable provenance.
It does not import the trial harness or run a simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import zstandard as zstd


PROJECTION_METADATA_NAME = "formal_projection.metadata.json"
PROJECTION_STATUS = "COMPLETE_AUTHENTICATED_FORMAL_RECORD_PROJECTION"
PROJECTION_ARTIFACT_CLASS = "formal_record_projection_staging_not_archive"
EXPECTED_FORMAL_MANIFEST_SHA256 = (
    "8fde3c59e83da0f8bce76f4d7d3caf6a7aa4e03c2fbb3d71f633d6563f207b24"
)
RECORD_FILENAMES = (
    "efftox_main.json.zst",
    "efftox_mariposa_baselines_supplemental.json.zst",
    "gate_sixway_supplemental.json.zst",
    "gbump_main.json.zst",
    "kappa_sweep_fixed_supplemental.json.zst",
    "mariposa_main.json.zst",
    "osa_main.json.zst",
    "osa_trajectory_supplemental.json.zst",
    "protocol_scaffold_sensitivity.json.zst",
    "schedule_ratio_sensitivity_2to1.json.zst",
)
EXPECTED_RECORD_ROWS = {
    "efftox_main.json.zst": 3_000,
    "efftox_mariposa_baselines_supplemental.json.zst": 2_000,
    "gate_sixway_supplemental.json.zst": 24_000,
    "gbump_main.json.zst": 6_000,
    "kappa_sweep_fixed_supplemental.json.zst": 6_000,
    "mariposa_main.json.zst": 3_000,
    "osa_main.json.zst": 6_000,
    "osa_trajectory_supplemental.json.zst": 480,
    "protocol_scaffold_sensitivity.json.zst": 4_800,
    "schedule_ratio_sensitivity_2to1.json.zst": 200,
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED_VALIDATION_FLAGS = (
    "all_target_rows_have_authenticated_formal_sources",
    "candidate_outputs_committed",
    "candidate_record_key_sets_unchanged",
    "cross_file_reuse_identities_verified",
    "formal_rows_canonically_ordered_and_complete",
    "frozen_archive_read_only",
    "primary_400_byte_identity_fields_match",
    "unapproved_row_count_zero",
)


class ProjectionBundleError(RuntimeError):
    """Raised when a formal-projection bundle fails authentication."""


@dataclass(frozen=True)
class ProjectionBundle:
    """Authenticated records and portable provenance for one projection."""

    records: dict[str, list[dict[str, Any]]]
    provenance: dict[str, Any]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON token {token}")


def _read_json_bytes(raw: bytes, *, label: str) -> Any:
    try:
        value = json.loads(raw, parse_constant=_reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProjectionBundleError(f"{label} is not finite valid JSON") from exc
    _assert_finite(value, label=label)
    return value


def _assert_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProjectionBundleError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, label=f"{label}[{index}]")


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ProjectionBundleError(
            f"{label} differs: observed {observed!r}, expected {expected!r}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProjectionBundleError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProjectionBundleError(f"{label} must be a JSON object")
    return value


def load_projection_bundle(projection_dir: str | Path) -> ProjectionBundle:
    """Authenticate all ten records in ``projection_dir`` and return them.

    Authentication covers the metadata envelope, its hash-addressed directory
    name and content-derived projection fingerprint, the externally pinned
    formal-manifest SHA-256, the exact record-file set, and every compressed and
    uncompressed byte/hash/row commitment.
    """

    root = Path(projection_dir).resolve()
    metadata_path = root / PROJECTION_METADATA_NAME
    records_dir = root / "records"
    if not root.is_dir() or not metadata_path.is_file() or not records_dir.is_dir():
        raise ProjectionBundleError(
            "projection must contain formal_projection.metadata.json and records/"
        )

    metadata_raw = metadata_path.read_bytes()
    metadata = _read_json_bytes(metadata_raw, label=PROJECTION_METADATA_NAME)
    metadata = _require_mapping(metadata, PROJECTION_METADATA_NAME)
    _require_equal(metadata.get("schema_version"), 1, "projection schema_version")
    _require_equal(metadata.get("status"), PROJECTION_STATUS, "projection status")
    _require_equal(
        metadata.get("artifact_class"),
        PROJECTION_ARTIFACT_CLASS,
        "projection artifact_class",
    )

    validation = _require_mapping(metadata.get("validation"), "validation")
    for field in _REQUIRED_VALIDATION_FLAGS:
        _require_equal(validation.get(field), True, f"validation.{field}")

    fingerprint = _require_sha256(
        metadata.get("projection_fingerprint"), "projection_fingerprint"
    )
    _require_equal(root.name, f"formal_projection-{fingerprint}", "projection dirname")
    inputs = _require_mapping(metadata.get("inputs"), "inputs")
    commitments = _require_mapping(
        metadata.get("candidate_records_commitment"),
        "candidate_records_commitment",
    )
    outputs = _require_mapping(metadata.get("outputs"), "outputs")
    expected_fingerprint = _sha256(_canonical_json_bytes({
        "inputs": inputs,
        "candidate_records": commitments,
    }))
    _require_equal(fingerprint, expected_fingerprint, "projection fingerprint")

    manifest = _require_mapping(inputs.get("manifest"), "inputs.manifest")
    _require_equal(
        manifest.get("basename"),
        "formal_rerun_manifest.json",
        "inputs.manifest.basename",
    )
    manifest_sha256 = _require_sha256(
        manifest.get("sha256"), "inputs.manifest.sha256"
    )
    _require_equal(
        manifest_sha256,
        EXPECTED_FORMAL_MANIFEST_SHA256,
        "formal-manifest SHA-256",
    )
    source_hashes = _require_mapping(
        inputs.get("formal_execution_source_sha256"),
        "inputs.formal_execution_source_sha256",
    )
    _require_equal(
        source_hashes.get("paper/formal_rerun_manifest.json"),
        manifest_sha256,
        "formal execution manifest SHA-256",
    )

    expected_names = set(RECORD_FILENAMES)
    actual_names = {path.name for path in records_dir.iterdir()}
    _require_equal(actual_names, expected_names, "projection record-file set")
    _require_equal(set(commitments), expected_names, "record commitment set")
    _require_equal(set(outputs), expected_names, "record output set")

    counts = _require_mapping(metadata.get("counts"), "counts")
    count_files = _require_mapping(counts.get("files"), "counts.files")
    _require_equal(set(count_files), expected_names, "counts.files set")

    records: dict[str, list[dict[str, Any]]] = {}
    record_provenance: dict[str, dict[str, Any]] = {}
    total_rows = 0
    decompressor = zstd.ZstdDecompressor()
    for filename in RECORD_FILENAMES:
        path = records_dir / filename
        if not path.is_file():
            raise ProjectionBundleError(f"projection record is not a file: {filename}")
        compressed = path.read_bytes()
        output = _require_mapping(outputs[filename], f"outputs.{filename}")
        commitment = _require_mapping(
            commitments[filename], f"candidate_records_commitment.{filename}"
        )
        _require_equal(
            set(output),
            {
                "rows", "compressed_bytes", "compressed_sha256",
                "uncompressed_bytes", "uncompressed_sha256",
            },
            f"{filename} output commitment fields",
        )
        _require_equal(
            set(commitment),
            {"rows", "uncompressed_bytes", "uncompressed_sha256"},
            f"{filename} logical commitment fields",
        )
        _require_equal(len(compressed), output.get("compressed_bytes"), f"{filename} compressed bytes")
        _require_equal(_sha256(compressed), output.get("compressed_sha256"), f"{filename} compressed SHA-256")
        try:
            logical = decompressor.decompress(compressed)
        except zstd.ZstdError as exc:
            raise ProjectionBundleError(f"{filename} is not valid zstd data") from exc
        _require_equal(len(logical), output.get("uncompressed_bytes"), f"{filename} uncompressed bytes")
        _require_equal(_sha256(logical), output.get("uncompressed_sha256"), f"{filename} uncompressed SHA-256")
        rows = _read_json_bytes(logical, label=filename)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ProjectionBundleError(f"{filename} must contain a JSON list of objects")
        _require_equal(logical, _canonical_json_bytes(rows), f"{filename} canonical logical bytes")
        _require_equal(len(rows), output.get("rows"), f"{filename} output rows")
        _require_equal(
            len(rows), EXPECTED_RECORD_ROWS[filename], f"{filename} declared rows"
        )
        _require_equal(
            dict(commitment),
            {
                "rows": output.get("rows"),
                "uncompressed_bytes": output.get("uncompressed_bytes"),
                "uncompressed_sha256": output.get("uncompressed_sha256"),
            },
            f"{filename} commitment",
        )
        _require_equal(
            count_files[filename].get("rows"), len(rows), f"{filename} count rows"
        )
        records[filename] = rows
        record_provenance[filename] = dict(output)
        total_rows += len(rows)

    _require_equal(total_rows, 55_480, "total authenticated record rows")
    _require_equal(counts.get("total_rows"), total_rows, "counts.total_rows")
    provenance = {
        "schema_version": 1,
        "projection_metadata_path": PROJECTION_METADATA_NAME,
        "projection_metadata_sha256": _sha256(metadata_raw),
        "projection_fingerprint": fingerprint,
        "formal_manifest_sha256": manifest_sha256,
        "record_files": record_provenance,
    }
    return ProjectionBundle(records=records, provenance=provenance)


def load_projection_record(
    projection_dir: str | Path, filename: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate the full bundle and return one named record plus provenance."""

    if filename not in RECORD_FILENAMES:
        raise ProjectionBundleError(f"unknown formal-projection record: {filename}")
    bundle = load_projection_bundle(projection_dir)
    provenance = dict(bundle.provenance)
    provenance["selected_record"] = filename
    provenance["selected_record_commitment"] = dict(
        bundle.provenance["record_files"][filename]
    )
    return bundle.records[filename], provenance


__all__ = [
    "EXPECTED_FORMAL_MANIFEST_SHA256",
    "EXPECTED_RECORD_ROWS",
    "PROJECTION_METADATA_NAME",
    "ProjectionBundle",
    "ProjectionBundleError",
    "RECORD_FILENAMES",
    "load_projection_bundle",
    "load_projection_record",
]
