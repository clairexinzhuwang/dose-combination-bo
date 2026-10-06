#!/usr/bin/env python3
"""Deterministic artifact helpers for authenticated record-derived figures."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_float(value: Any, *, label: str) -> float:
    """Convert ``value`` to a finite float or raise a labelled error."""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def padded_limits(
    values: Iterable[float],
    *,
    minimum_span: float = 1.0,
    padding_fraction: float = 0.10,
) -> tuple[float, float]:
    """Return finite plot limits that contain every supplied value.

    The limits are data-derived so a changed formal result cannot be silently
    clipped by a historical hard-coded axis range.
    """

    numbers = [finite_float(value, label="axis value") for value in values]
    if not numbers:
        raise ValueError("cannot derive axis limits from an empty sequence")
    lower = min(numbers)
    upper = max(numbers)
    span = max(upper - lower, float(minimum_span))
    padding = span * float(padding_fraction)
    limits = (lower - padding, upper + padding)
    if not (limits[0] < lower <= upper < limits[1]):
        raise ValueError("derived axis limits do not contain all plotted values")
    return limits


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write a stable UTF-8 CSV with Unix newlines and fixed column order."""

    if not rows:
        raise ValueError("cannot write an empty artifact table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("artifact rows do not share one ordered schema")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: str | Path, value: Any) -> Path:
    """Write stable, finite, sorted JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def save_figure(fig, pdf_path: str | Path, png_path: str | Path, *, title: str, subject: str) -> tuple[Path, Path]:
    """Save byte-stable vector and raster versions of a Matplotlib figure."""

    pdf_path = Path(pdf_path)
    png_path = Path(png_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Subject": subject,
            "Creator": "dose-combination-bo post-processing",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": title, "Description": subject},
    )
    return pdf_path, png_path


def write_artifact_sidecar(
    artifact_path: str | Path,
    *,
    artifact_type: str,
    artifact_class: str,
    provenance: Mapping[str, Any],
    record_names: Sequence[str],
    generator_paths: Sequence[str | Path],
    analysis: Mapping[str, Any],
) -> Path:
    """Commit an output to its authenticated inputs and generator source."""

    artifact_path = Path(artifact_path)
    record_files = provenance["record_files"]
    inputs: dict[str, Any] = {}
    for name in record_names:
        detail = record_files[name]
        inputs[name] = {
            "rows": int(detail["rows"]),
            "compressed_bytes": int(detail["compressed_bytes"]),
            "compressed_sha256": detail["compressed_sha256"],
            "uncompressed_bytes": int(detail["uncompressed_bytes"]),
            "uncompressed_sha256": detail["uncompressed_sha256"],
        }
    generators = {
        Path(path).name: file_sha256(path)
        for path in sorted((Path(path).resolve() for path in generator_paths), key=lambda value: value.name)
    }
    metadata = {
        "schema_version": 1,
        "artifact_class": artifact_class,
        "artifact_type": artifact_type,
        "artifact_filename": artifact_path.name,
        "artifact_bytes": artifact_path.stat().st_size,
        "artifact_sha256": file_sha256(artifact_path),
        "projection": {
            "projection_fingerprint": provenance["projection_fingerprint"],
            "projection_metadata_path": provenance["projection_metadata_path"],
            "projection_metadata_sha256": provenance["projection_metadata_sha256"],
            "formal_manifest_sha256": provenance["formal_manifest_sha256"],
        },
        "input_records": inputs,
        "generator_files_sha256": generators,
        "analysis": dict(analysis),
    }
    return write_json(
        artifact_path.with_suffix(artifact_path.suffix + ".metadata.json"),
        metadata,
    )


__all__ = [
    "file_sha256",
    "finite_float",
    "padded_limits",
    "save_figure",
    "write_artifact_sidecar",
    "write_csv",
    "write_json",
]
