"""Machine-readable design metadata for tables, figures, and archives."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from .protocol import PROTOCOL_CONFIG_FIELDS, PROTOCOL_DEFAULTS


_LEGACY_DEFAULTS = {
    "kap": 1.0,
    **PROTOCOL_DEFAULTS,
}

_FIELDS = (
    "policy", "sim", "stratum", "gamma", "mode", "noise", "kap", "budget",
    "r_k", "grid_n", "warmup", "empty_gate", *PROTOCOL_CONFIG_FIELDS,
)


def _rows(records):
    if hasattr(records, "to_dict"):
        records = records.to_dict("records")
    else:
        records = list(records)
    if not records:
        raise ValueError("records must contain at least one trial")
    if not all(isinstance(row, Mapping) for row in records):
        raise TypeError("every record must be a mapping")
    return records


def _normalise(value):
    if hasattr(value, "item"):
        value = value.item()
    return value


def _sorted_unique(values):
    values = {_normalise(value) for value in values}
    def key(value):
        if isinstance(value, bool):
            return (0, int(value))
        if isinstance(value, (int, float)):
            return (1, float(value))
        if isinstance(value, str):
            return (2, value)
        return (3, type(value).__name__, str(value))
    return sorted(values, key=key)


def records_sha256(records):
    """SHA-256 of records in their supplied order using canonical compact JSON."""
    rows = _rows(records)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    """Stream a file into SHA-256."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_metadata(records, *, source_path=None, artifact_type="analysis"):
    """Build the sidecar metadata required for every public table or figure.

    The metadata never silently chooses one value from a mixed design. Every design
    level is returned as a sorted list, including dose-availability design, gate threshold, budget,
    and seed count. ``source_sha256`` is the exact input-file hash when a source path
    is supplied and otherwise a canonical in-memory records hash. Only the portable
    source basename, never an absolute workstation path, is stored in the sidecar.
    """
    rows = _rows(records)
    for index, row in enumerate(rows):
        missing = [field for field in _FIELDS
                   if field not in row and field not in _LEGACY_DEFAULTS]
        if missing:
            raise ValueError(f"record {index} is missing design metadata {missing}")
        if "seed" not in row:
            raise ValueError(f"record {index} is missing seed")

    levels = {}
    for field in _FIELDS:
        levels[field] = _sorted_unique(
            row.get(field, _LEGACY_DEFAULTS.get(field)) for row in rows
        )
    source_file = Path(source_path) if source_path is not None else None
    source_path_text = source_file.name if source_file is not None else None
    source_hash = file_sha256(source_file) if source_file is not None else records_sha256(rows)
    return {
        "schema_version": 1,
        "artifact_type": str(artifact_type),
        "record_count": len(rows),
        "seed_count": len({int(row["seed"]) for row in rows}),
        "seeds": _sorted_unique(int(row["seed"]) for row in rows),
        "source_path": source_path_text,
        "source_sha256": source_hash,
        "design": levels,
        # Prominent aliases make the four required caption fields easy to audit.
        "protocol_scaffold": levels["protocol_scaffold"],
        "tau": levels["gamma"],
        "budget": levels["budget"],
    }


def write_artifact_metadata(path, records, *, source_path=None, artifact_type="analysis"):
    """Write :func:`artifact_metadata` as stable, indented JSON."""
    metadata = artifact_metadata(
        records, source_path=source_path, artifact_type=artifact_type
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


__all__ = [
    "artifact_metadata", "write_artifact_metadata", "records_sha256", "file_sha256",
]
