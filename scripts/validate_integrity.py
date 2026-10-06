#!/usr/bin/env python3
"""Validate the reproducibility archive without the GP stack.

This entry point deliberately does not import dose_combination_bo, torch, gpytorch, or
the historical analysis modules.  It needs only the Python standard library
and ``zstandard`` to verify the archive payload, decompress the trial records,
and audit their frozen design and terminal-state semantics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


sys.dont_write_bytecode = True

SCAFFOLDS = ("lhs_fixed", "start_low_expansion")
GAMMAS = (0.7, 0.9)
STRATA = (0, 1)
POLICIES = ("cEI", "cKG", "cEI-tMSE")
SEEDS = tuple(range(200))
LEGACY_TO_CANONICAL_POLICY = {
    "cEI": "cEI",
    "cKG1fix": "cKG",
    "GBE": "cEI-tMSE",
}
FROZEN_SENSITIVITY = {
    "sim": "osa",
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
COMMON_FIXED_DESIGN = {
    "mode": "latent",
    "noise": "fixed",
    "budget": 40,
    "warmup": 4,
    "r_k": 2,
    "grid_n": 5,
    "empty_gate": "pf",
}
TERMINAL_FIELDS = (
    "dose_units",
    "rpsel",
    "rec_unsafe",
    "rec_d1",
    "rec_d2",
    "rec_true_eff",
    "rec_true_tox",
)
REQUIRED_SENSITIVITY_FIELDS = {
    "allocation_history",
    "eff_noise_sd",
    "gamma",
    "initialization_patients_above",
    "initialization_size",
    "n_empty_gate_events",
    "n_enrolled",
    "n_gate_pass",
    "n_gate_pass_safe",
    "n_unique_doses",
    "obd_pf",
    "obd_sdg",
    "policy",
    "post_initialization_patients_above",
    "protocol_scaffold",
    "recommendation_made",
    "recs",
    "region_q_at_stop",
    "rpsel",
    "seed",
    "stop_reason",
    "stratum",
    "toxic",
    "tox_noise_sd",
    "traj",
    *FROZEN_SENSITIVITY,
    *TERMINAL_FIELDS,
}
SHA256_HEX = frozenset("0123456789abcdef")
EXPECTED_FORMAL_MANIFEST_SHA256 = (
    "8fde3c59e83da0f8bce76f4d7d3caf6a7aa4e03c2fbb3d71f633d6563f207b24"
)
FORMAL_PROJECTION_STATUS = "COMPLETE_AUTHENTICATED_FORMAL_RECORD_PROJECTION"
FORMAL_PROJECTION_METADATA_DEFAULT = "metadata/formal_projection.metadata.json"
FORMAL_RECORD_ROWS = {
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
FORMAL_PROJECTION_VALIDATION_FLAGS = (
    "all_target_rows_have_authenticated_formal_sources",
    "candidate_outputs_committed",
    "candidate_record_key_sets_unchanged",
    "cross_file_reuse_identities_verified",
    "formal_rows_canonically_ordered_and_complete",
    "frozen_archive_read_only",
    "primary_400_byte_identity_fields_match",
    "unapproved_row_count_zero",
)

COMPARATOR_EXTENSION_ROWS = 8_800
COMPARATOR_REUSED_ROWS = 25_200
COMPARATOR_FAMILY_ROWS = 34_000
COMPARATOR_EXTENSION_STATUS = "COMPLETE_COMPARATOR_FAMILY_RAW_EXTENSION"
COMPARATOR_EXTENSION_COMMIT_STATUS = "COMMITTED_COMPARATOR_FAMILY_RAW_EXTENSION"
COMPARATOR_EXTENSION_CLASS = (
    "authenticated_comparator_family_raw_extension_not_formal_projection"
)
COMPARATOR_EXTENSION_MATRIX_COUNTS = {
    "family_full_osa_gbump_boundary_completion": 4_000,
    "family_full_efftox_mariposa_boundary_completion": 3_200,
    "family_gradual_osa_boundary_completion": 1_600,
}
COMPARATOR_EXTENSION_SORT_KEY = (
    "matrix_id",
    "policy",
    "sim",
    "mode",
    "protocol_scaffold",
    "kap",
    "gamma",
    "stratum",
    "seed",
)
COMPARATOR_TERMINAL_FIELDS = (
    "dose_units",
    "rpsel",
    "rec_unsafe",
    "rec_d1",
    "rec_d2",
    "rec_true_eff",
    "rec_true_tox",
)
QBIG_AUDIT_STATUS = "COMPLETE_QBIG_QUADRATURE_REPRESENTATIVE_AUDIT"
QBIG_AUDIT_COMMIT_STATUS = "COMMITTED_QBIG_QUADRATURE_REPRESENTATIVE_AUDIT"
ANALYSIS_REGENERATION_LOCK_LINES = frozenset(
    {
        "numpy==2.1.3",
        "scipy==1.15.3",
        "torch==2.10.0",
        "gpytorch==1.15.2",
        "pandas==2.2.3",
        "matplotlib==3.10.0",
        "ray==2.54.1",
        "botorch==0.17.2",
        "linear-operator==0.6.1",
        "zstandard==0.23.0",
    }
)
ANALYSIS_REGENERATION_RUNTIME_LINES = frozenset(
    {"python==3.13.5", "zstandard==0.23.0"}
)
COMPARATOR_EXTENSION_RUNTIME = {
    "gpytorch": "1.14",
    "numpy": "1.26.4",
    "python": "3.11.6",
    "ray": "2.10.0",
    "scipy": "1.16.3",
    "torch": "2.4.1",
    "zstandard": "0.19.0",
}


class IntegrityError(AssertionError):
    """Raised when the archive fails a deterministic integrity check."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"{path}: cannot read valid UTF-8 JSON") from exc


def _read_finite_json_bytes(payload: bytes, label: str) -> Any:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token}")

    try:
        return json.loads(payload, parse_constant=reject)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise IntegrityError(f"{label}: cannot read finite valid UTF-8 JSON") from exc


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise IntegrityError(f"{label}: expected a JSON object")
    return value


def _require_int(
    value: Any,
    label: str,
    *,
    lower: int | None = None,
    upper: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IntegrityError(f"{label}: expected an integer, found {value!r}")
    if lower is not None and value < lower:
        raise IntegrityError(f"{label}: {value} is below {lower}")
    if upper is not None and value > upper:
        raise IntegrityError(f"{label}: {value} is above {upper}")
    return value


def _finite(
    value: Any,
    label: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IntegrityError(f"{label}: expected a finite number, found {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise IntegrityError(f"{label}: expected a finite number, found {value!r}")
    if lower is not None and result < lower:
        raise IntegrityError(f"{label}: {result} is below {lower}")
    if upper is not None and result > upper:
        raise IntegrityError(f"{label}: {result} is above {upper}")
    return result


def _csv_float(value: Any, label: str) -> float:
    """Parse one finite numeric CSV cell without weakening JSON type checks."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"{label}: expected a finite CSV number, found {value!r}") from exc
    if not math.isfinite(result):
        raise IntegrityError(f"{label}: expected a finite CSV number, found {value!r}")
    return result


def _csv_int(value: Any, label: str, *, lower: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"{label}: expected an integer CSV cell, found {value!r}") from exc
    if str(result) != str(value):
        raise IntegrityError(f"{label}: expected an integer CSV cell, found {value!r}")
    if lower is not None and result < lower:
        raise IntegrityError(f"{label}: {result} is below {lower}")
    return result


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(SHA256_HEX)
    )


def _canonical_json_sha256(value: Any) -> str:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_compact_json_bytes(value: Any) -> bytes:
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


def _safe_archive_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise IntegrityError(f"{label}: expected a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise IntegrityError(f"{label}: unsafe archive path {value!r}")
    path = root / relative
    if not path.is_file():
        raise IntegrityError(f"{label}: archive file is missing: {value}")
    return path


def _isclose(left: Any, right: Any, label: str, *, abs_tol: float = 1e-12) -> None:
    observed = _csv_float(left, label)
    expected = _csv_float(right, label)
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=abs_tol):
        raise IntegrityError(
            f"{label}: {observed!r} does not equal the independently derived "
            f"value {expected!r}"
        )


def _bool_cell(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise IntegrityError(f"{label}: expected True or False, found {value!r}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise IntegrityError(f"{path}: cannot read a valid UTF-8 CSV") from exc
    if not rows:
        raise IntegrityError(f"{path}: analysis CSV is empty")
    return rows


def _manifest_record_items(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return the exact ten formal record commitments keyed by archive basename."""
    items: dict[str, Mapping[str, Any]] = {}
    for section_name in ("main_records", "supplemental_records"):
        section = _require_mapping(manifest.get(section_name), section_name)
        values = section.values()
        for index, value in enumerate(values):
            item = _require_mapping(value, f"{section_name}[{index}]")
            path_value = item.get("path")
            if not isinstance(path_value, str):
                raise IntegrityError(f"{section_name}[{index}]: missing record path")
            basename = Path(path_value).name
            if basename in items:
                raise IntegrityError(f"duplicate record basename in design manifest: {basename}")
            items[basename] = item
    for section_name in (
        "protocol_scaffold_sensitivity",
        "schedule_ratio_sensitivity",
    ):
        item = _require_mapping(manifest.get(section_name), section_name)
        path_value = item.get("path")
        if not isinstance(path_value, str):
            raise IntegrityError(f"{section_name}: missing record path")
        basename = Path(path_value).name
        if basename in items:
            raise IntegrityError(f"duplicate record basename in design manifest: {basename}")
        items[basename] = item
    if set(items) != set(FORMAL_RECORD_ROWS):
        raise IntegrityError(
            "formal projection/design-manifest record set mismatch: "
            f"missing={sorted(set(FORMAL_RECORD_ROWS) - set(items))}, "
            f"extra={sorted(set(items) - set(FORMAL_RECORD_ROWS))}"
        )
    return items


def locate_archive_root(explicit: Path | None = None) -> Path:
    """Resolve an archive root from a CLI argument or this script's ancestors."""
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not (root / "metadata" / "design_manifest.json").is_file():
            raise IntegrityError(
                f"{root}: not an archive root (metadata/design_manifest.json is missing)"
            )
        return root
    for parent in Path(__file__).resolve().parents:
        if (parent / "metadata" / "design_manifest.json").is_file() and (
            parent / "records"
        ).is_dir():
            return parent
    raise IntegrityError(
        "archive root not found; run this script from the reproduction archive or pass "
        "--archive-root /path/to/archive"
    )


def validate_checksum_manifest(root: Path) -> int:
    """Verify every archived payload file against ``SHA256SUMS``."""
    manifest_path = root / "SHA256SUMS"
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise IntegrityError(f"{manifest_path}: missing checksum manifest") from exc
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or not _valid_sha256(digest) or not relative:
            raise IntegrityError(
                f"SHA256SUMS line {line_number}: expected '<sha256>  <relative path>'"
            )
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise IntegrityError(f"SHA256SUMS line {line_number}: unsafe path {relative!r}")
        if relative in entries:
            raise IntegrityError(f"SHA256SUMS: duplicate path {relative!r}")
        entries[relative] = digest
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    expected = set(entries)
    if actual != expected:
        raise IntegrityError(
            "SHA256SUMS inventory mismatch: "
            f"missing={sorted(expected - actual)[:3]}, "
            f"unlisted={sorted(actual - expected)[:3]}"
        )
    for relative, expected_digest in entries.items():
        actual_digest = file_sha256(root / relative)
        if actual_digest != expected_digest:
            raise IntegrityError(f"{relative}: SHA-256 mismatch")
    return len(entries)


def validate_public_payload_paths(root: Path) -> None:
    """Reject cache files and absolute workstation paths in the public payload."""
    forbidden = (b"/" + b"Users/", b"file:" + b"///", b"/" + b"private/")
    # These source files intentionally contain adversarial/local-path literals:
    # the formal manifest quarantines a superseded hard-stop staging path, and
    # the tests assert that local paths are rejected from reader-facing data.
    # Their exact bytes are authenticated elsewhere; the exception does not
    # apply to result, metadata, documentation, or arbitrary source files.
    literal_test_sources = {
        "formal_rerun_manifest.json",
        "formal_rerun_preflight.py",
        "run_full_formal_rerun.py",
        "test_ckg_aggregate_oc_stability.py",
        "test_ckg_piecewise_exact_panel.py",
        "test_archive_validators.py",
        "test_formal_rerun_preflight.py",
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            raise IntegrityError(f"{relative}: cache/bytecode is not reproduction-archive payload")
        if any(part == "checkpoints" for part in relative.parts) or ".checkpoint." in path.name:
            raise IntegrityError(f"{relative}: resumable execution checkpoints are not public payload")
        payload = path.read_bytes()
        if path.name in literal_test_sources and (
            relative.parts[:2] == ("source", "package")
            or "scripts" in relative.parts
            or relative.parts[:1] == ("metadata",)
        ):
            continue
        for token in forbidden:
            if token in payload:
                raise IntegrityError(f"{relative}: contains a non-portable local path")


def _zstandard_module():
    try:
        import zstandard  # type: ignore
    except ModuleNotFoundError as exc:
        raise IntegrityError(
            "the lightweight validator requires only the `zstandard` package; "
            "install it with `python -m pip install zstandard`"
        ) from exc
    return zstandard


def materialize_record(compressed: Path, destination: Path) -> None:
    """Decompress one zstd record file without invoking research code."""
    zstandard = _zstandard_module()
    try:
        with compressed.open("rb") as source, destination.open("wb") as target:
            zstandard.ZstdDecompressor().copy_stream(source, target)
    except (OSError, zstandard.ZstdError) as exc:
        raise IntegrityError(f"{compressed}: zstd decompression failed") from exc


def _load_record_item(root: Path, label: str, item: Mapping[str, Any], work: Path):
    path_value = item.get("path")
    if not isinstance(path_value, str):
        raise IntegrityError(f"{label}: missing record path in design manifest")
    compressed = root / path_value
    if not compressed.is_file():
        raise IntegrityError(f"{path_value}: record file is missing")
    expected_compressed = item.get("compressed_sha256")
    if not _valid_sha256(expected_compressed):
        raise IntegrityError(f"{label}: invalid compressed_sha256 metadata")
    if file_sha256(compressed) != expected_compressed:
        raise IntegrityError(f"{path_value}: compressed SHA-256 mismatch")
    output = work / f"{label}.json"
    materialize_record(compressed, output)
    if output.stat().st_size != item.get("uncompressed_bytes"):
        raise IntegrityError(f"{path_value}: decompressed byte-size mismatch")
    expected_raw = item.get("uncompressed_sha256")
    if not _valid_sha256(expected_raw) or file_sha256(output) != expected_raw:
        raise IntegrityError(f"{path_value}: decompressed SHA-256 mismatch")
    rows = _read_json(output)
    if not isinstance(rows, list):
        raise IntegrityError(f"{path_value}: decompressed JSON must be a list")
    if len(rows) != item.get("rows"):
        raise IntegrityError(
            f"{path_value}: decompressed row count {len(rows)} != {item.get('rows')!r}"
        )
    return rows


def _validate_common_completed_row(row: Mapping[str, Any], label: str) -> None:
    for field in ("dose_units", "rpsel"):
        _finite(row.get(field), f"{label}.{field}", lower=0)
    for field in ("rec_true_eff", "rec_true_tox", "rec_d1", "rec_d2"):
        _finite(row.get(field), f"{label}.{field}")
    _require_int(row.get("rec_unsafe"), f"{label}.rec_unsafe", lower=0, upper=1)
    _require_int(row.get("toxic"), f"{label}.toxic", lower=0, upper=40)
    if row.get("recommendation_made", True) is not True:
        raise IntegrityError(f"{label}: completed record cannot deny its recommendation")


def _exact_cells(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    expected: set[tuple[Any, ...]],
    label: str,
) -> None:
    keys = []
    for index, row in enumerate(rows):
        try:
            keys.append(tuple(row[field] for field in fields))
        except KeyError as exc:
            raise IntegrityError(f"{label} row {index}: missing design field {exc.args[0]!r}")
    counts = Counter(keys)
    duplicates = [key for key, count in counts.items() if count > 1]
    actual = set(keys)
    if duplicates or actual != expected:
        raise IntegrityError(
            f"{label}: factorial mismatch; duplicate={duplicates[:1]}, "
            f"missing={sorted(expected - actual, key=repr)[:1]}, "
            f"extra={sorted(actual - expected, key=repr)[:1]}"
        )


def validate_main_records(rows: Sequence[Mapping[str, Any]], sim: str, seeds: int) -> None:
    expected = {
        (policy, seed, stratum, gamma)
        for policy in POLICIES
        for seed in range(seeds)
        for stratum in STRATA
        for gamma in (0.5, 0.6, 0.7, 0.8, 0.9)
    }
    _exact_cells(rows, ("policy", "seed", "stratum", "gamma"), expected, f"main/{sim}")
    for index, row in enumerate(rows):
        label = f"main/{sim} row {index}"
        if row.get("sim") != sim:
            raise IntegrityError(f"{label}.sim: expected {sim!r}")
        for field, expected_value in COMMON_FIXED_DESIGN.items():
            if row.get(field) != expected_value:
                raise IntegrityError(
                    f"{label}.{field}: {row.get(field)!r} != {expected_value!r}"
                )
        if row.get("kap", 1.0) != 1.0:
            raise IntegrityError(f"{label}.kap: expected 1.0")
        _validate_common_completed_row(row, label)


def _main_index(main_rows: Iterable[Mapping[str, Any]]):
    return {
        (
            row["policy"],
            row["seed"],
            row["sim"],
            row["stratum"],
            row["mode"],
            row["gamma"],
        ): row
        for row in main_rows
    }


def _compare_seeded_identity(
    rows: Sequence[Mapping[str, Any]],
    main: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    sims: set[str],
    gammas: set[float],
    policies: set[str],
    seeds: int,
    ignore_fields: set[str] | None = None,
) -> None:
    ignore = {"policy", "kap", *(ignore_fields or set())}
    selected: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        canonical = LEGACY_TO_CANONICAL_POLICY.get(row.get("policy"))
        if (
            canonical in policies
            and row.get("seed") in range(seeds)
            and row.get("sim") in sims
            and row.get("gamma") in gammas
            and row.get("mode") == "latent"
        ):
            key = (
                canonical,
                row["seed"],
                row["sim"],
                row["stratum"],
                row["mode"],
                row["gamma"],
            )
            selected[key] = row
    expected_keys = {
        key
        for key in main
        if key[0] in policies
        and key[1] < seeds
        and key[2] in sims
        and key[4] == "latent"
        and key[5] in gammas
    }
    if set(selected) != expected_keys:
        raise IntegrityError("seed-indexed supplemental/main identity keys do not match")
    for key in expected_keys:
        reference = main[key]
        candidate = selected[key]
        for field in set(reference) - ignore:
            if candidate.get(field) != reference.get(field):
                raise IntegrityError(
                    f"seed-indexed supplemental row drift at {key}, field={field!r}"
                )


def _validate_supplemental_row(row: Mapping[str, Any], label: str, *, trajectory=False):
    for field in ("budget", "warmup", "r_k", "grid_n", "empty_gate"):
        if row.get(field) != COMMON_FIXED_DESIGN[field]:
            raise IntegrityError(f"{label}.{field}: frozen-design mismatch")
    if row.get("noise") != "fixed":
        raise IntegrityError(f"{label}.noise: expected 'fixed'")
    _validate_common_completed_row(row, label)
    if not 0 <= _require_int(row.get("n_gate_pass"), f"{label}.n_gate_pass") <= 25:
        raise IntegrityError(f"{label}.n_gate_pass: outside [0, 25]")
    safe = _require_int(row.get("n_gate_pass_safe"), f"{label}.n_gate_pass_safe")
    if not 0 <= safe <= row["n_gate_pass"]:
        raise IntegrityError(f"{label}: invalid gate counts")
    _finite(row.get("obd_pf"), f"{label}.obd_pf", lower=0, upper=1)
    _finite(row.get("obd_sdg"), f"{label}.obd_sdg", lower=0)
    if trajectory:
        traj = _require_mapping(row.get("traj"), f"{label}.traj")
        expected_n = list(range(4, 40, 2))
        if traj.get("n") != expected_n:
            raise IntegrityError(f"{label}.traj.n: unexpected trajectory checkpoints")
        for field in ("du", "rpsel", "toxic"):
            values = traj.get(field)
            if not isinstance(values, list) or len(values) != len(expected_n):
                raise IntegrityError(f"{label}.traj.{field}: malformed trajectory vector")
    elif row.get("traj") is not None:
        raise IntegrityError(f"{label}.traj: unexpected trajectory payload")


def validate_supplemental_records(
    label: str,
    rows: Sequence[Mapping[str, Any]],
    main_rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = ("policy", "seed", "sim", "stratum", "mode", "gamma")
    main = _main_index(main_rows)
    if label == "gate_sixway":
        policies = ("cEI", "cKG1fix", "GBE", "straddle", "qBIG", "random")
        expected = {
            (policy, seed, sim, stratum, mode, gamma)
            for policy in policies
            for seed in range(100)
            for sim in ("osa", "gbump")
            for stratum in STRATA
            for mode in ("latent", "predictive")
            for gamma in (0.5, 0.6, 0.7, 0.8, 0.9)
        }
        _exact_cells(rows, fields, expected, label)
        for index, row in enumerate(rows):
            _validate_supplemental_row(row, f"{label} row {index}")
            if row.get("kap") != 1.0:
                raise IntegrityError(f"{label} row {index}.kap: expected 1.0")
        _compare_seeded_identity(
            rows,
            main,
            sims={"osa", "gbump"},
            gammas={0.5, 0.6, 0.7, 0.8, 0.9},
            policies=set(POLICIES),
            seeds=100,
        )
    elif label == "baselines_efftox_mariposa":
        policies = ("cEI", "cKG1fix", "GBE", "straddle", "qBIG")
        expected = {
            (policy, seed, sim, stratum, "latent", 0.7)
            for policy in policies
            for seed in range(100)
            for sim in ("efftox", "mariposa")
            for stratum in STRATA
        }
        _exact_cells(rows, fields, expected, label)
        for index, row in enumerate(rows):
            _validate_supplemental_row(row, f"{label} row {index}")
            if row.get("kap") != 1.0:
                raise IntegrityError(f"{label} row {index}.kap: expected 1.0")
        _compare_seeded_identity(
            rows,
            main,
            sims={"efftox", "mariposa"},
            gammas={0.7},
            policies=set(POLICIES),
            seeds=100,
        )
        mariposa: dict[tuple[Any, ...], dict[int, Mapping[str, Any]]] = {}
        for row in rows:
            if row["sim"] == "mariposa":
                key = (row["policy"], row["seed"], row["gamma"], row["mode"])
                mariposa.setdefault(key, {})[row["stratum"]] = row
        for key, pair in mariposa.items():
            if set(pair) != {0, 1}:
                raise IntegrityError(f"MARIPOSA pair missing a stratum at {key}")
            left = {field: value for field, value in pair[0].items() if field != "stratum"}
            right = {field: value for field, value in pair[1].items() if field != "stratum"}
            if left != right:
                raise IntegrityError(f"MARIPOSA duplicate-stratum identity drift at {key}")
    elif label == "kappa_sweep_fixed":
        kappa_fields = ("policy", "seed", "stratum", "gamma", "kap")
        expected = {
            (policy, seed, stratum, gamma, kap)
            for policy in ("cEI", "cKG1fix")
            for seed in range(100)
            for stratum in STRATA
            for gamma in (0.5, 0.7, 0.9)
            for kap in (0.25, 0.5, 1.0, 1.5, 2.0)
        }
        _exact_cells(rows, kappa_fields, expected, label)
        for index, row in enumerate(rows):
            _validate_supplemental_row(row, f"{label} row {index}")
            if row.get("sim") != "osa" or row.get("mode") != "latent":
                raise IntegrityError(f"{label} row {index}: surface/mode drift")
        _compare_seeded_identity(
            [row for row in rows if row["kap"] == 1.0],
            main,
            sims={"osa"},
            gammas={0.5, 0.7, 0.9},
            policies={"cEI", "cKG"},
            seeds=100,
        )
    elif label == "traj_osa":
        expected = {
            (policy, seed, "osa", stratum, "latent", 0.7)
            for policy in ("cEI", "cKG1fix", "GBE")
            for seed in range(80)
            for stratum in STRATA
        }
        _exact_cells(rows, fields, expected, label)
        for index, row in enumerate(rows):
            _validate_supplemental_row(row, f"{label} row {index}", trajectory=True)
        _compare_seeded_identity(
            rows,
            main,
            sims={"osa"},
            gammas={0.7},
            policies=set(POLICIES),
            seeds=80,
            ignore_fields={"traj"},
        )
    else:
        raise IntegrityError(f"unknown supplemental record set {label!r}")


def _point(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise IntegrityError(f"{label}: expected a two-coordinate list")
    return (_finite(value[0], f"{label}[0]", lower=0, upper=1),
            _finite(value[1], f"{label}[1]", lower=0, upper=1))


def _on_grid(point: tuple[float, float]) -> bool:
    return all(any(math.isclose(value, grid, abs_tol=1e-12) for grid in (0, .25, .5, .75, 1))
               for value in point)


def _eligible_points(
    history: Sequence[tuple[float, float]], cohort_q: int
) -> set[tuple[float, float]]:
    grid = {(d1 / 4, d2 / 4) for d1 in range(5) for d2 in range(5)}
    region = {point for point in grid if sum(point) <= 0.25 * cohort_q + 1e-12}
    if region == grid:
        return region
    unvisited = region - set(history)
    return unvisited or region


def validate_sensitivity_record(row: Mapping[str, Any], index: int = 0):
    """Validate one protocol-sensitivity record and return its design key/history."""
    label = f"protocol sensitivity row {index}"
    missing = REQUIRED_SENSITIVITY_FIELDS - set(row)
    if missing:
        raise IntegrityError(f"{label}: missing fields {sorted(missing)}")
    for field, expected in FROZEN_SENSITIVITY.items():
        if row[field] != expected:
            raise IntegrityError(f"{label}.{field}: {row[field]!r} != {expected!r}")
    scaffold, gamma, stratum, policy, seed = (
        row["protocol_scaffold"], row["gamma"], row["stratum"], row["policy"], row["seed"]
    )
    if scaffold not in SCAFFOLDS or gamma not in GAMMAS:
        raise IntegrityError(f"{label}: unknown scaffold or gamma")
    if stratum not in STRATA or policy not in POLICIES:
        raise IntegrityError(f"{label}: unknown stratum or policy")
    _require_int(stratum, f"{label}.stratum")
    _require_int(seed, f"{label}.seed", lower=0, upper=199)
    if type(row["recommendation_made"]) is not bool:
        raise IntegrityError(f"{label}.recommendation_made must be boolean")
    n_enrolled = _require_int(row["n_enrolled"], f"{label}.n_enrolled", lower=2, upper=40)
    if n_enrolled % 2:
        raise IntegrityError(f"{label}.n_enrolled must be even")
    raw_history = row["allocation_history"]
    if not isinstance(raw_history, list) or len(raw_history) != n_enrolled:
        raise IntegrityError(f"{label}.allocation_history must contain n_enrolled points")
    history = tuple(_point(value, f"{label}.allocation_history[{offset}]")
                    for offset, value in enumerate(raw_history))
    initialization_size = 4 if scaffold == "lhs_fixed" else 2
    if row["initialization_size"] != initialization_size:
        raise IntegrityError(f"{label}.initialization_size: expected {initialization_size}")
    if scaffold == "lhs_fixed":
        if n_enrolled != 40 or not row["recommendation_made"]:
            raise IntegrityError(f"{label}: lhs_fixed must complete at n=40")
        for warmup_index, point in enumerate(history[:4]):
            lower, upper = warmup_index / 4, (warmup_index + 1) / 4
            if not lower <= point[0] < upper:
                raise IntegrityError(
                    f"{label}: full-panel initialization d1 is outside its prescribed stratum"
                )
        adaptive_start = 4
    else:
        if history[:2] != ((0.0, 0.0), (0.0, 0.0)):
            raise IntegrityError(f"{label}: gradual access must initialize at the origin")
        adaptive_start = 2
    for offset in range(adaptive_start, n_enrolled, 2):
        if history[offset] != history[offset + 1]:
            raise IntegrityError(f"{label}: adaptive cohort at offset {offset} is not paired")
        if not _on_grid(history[offset]):
            raise IntegrityError(f"{label}: adaptive allocation is off the 5 x 5 grid")
        if scaffold == "start_low_expansion":
            cohort_q = offset // 2 + 1
            if history[offset] not in _eligible_points(history[:offset], cohort_q):
                raise IntegrityError(
                    f"{label}: cohort {cohort_q} violates gradual-access/no-repeat rules"
                )
    n_unique = _require_int(row["n_unique_doses"], f"{label}.n_unique_doses", lower=1)
    if n_unique != len(set(history)):
        raise IntegrityError(f"{label}.n_unique_doses disagrees with allocation history")
    for field in (
        "initialization_patients_above",
        "post_initialization_patients_above",
        "toxic",
    ):
        _require_int(row[field], f"{label}.{field}", lower=0, upper=n_enrolled)
    if row["initialization_patients_above"] + row["post_initialization_patients_above"] != row["toxic"]:
        raise IntegrityError(f"{label}: initialization and post-initialization exposure do not sum")
    gate_pass = _require_int(row["n_gate_pass"], f"{label}.n_gate_pass", lower=0, upper=25)
    gate_safe = _require_int(
        row["n_gate_pass_safe"], f"{label}.n_gate_pass_safe", lower=0, upper=25
    )
    if gate_safe > gate_pass:
        raise IntegrityError(f"{label}: safe gate count exceeds total gate count")
    _finite(row["obd_pf"], f"{label}.obd_pf", lower=0, upper=1)
    _finite(row["obd_sdg"], f"{label}.obd_sdg", lower=0)
    _finite(row["eff_noise_sd"], f"{label}.eff_noise_sd", lower=0)
    _finite(row["tox_noise_sd"], f"{label}.tox_noise_sd", lower=0)
    if row["traj"] is not None:
        raise IntegrityError(f"{label}.traj must be null in the frozen run")
    adaptive_allocations = (n_enrolled - initialization_size) // 2
    empty_events = _require_int(
        row["n_empty_gate_events"], f"{label}.n_empty_gate_events", lower=0
    )
    if row["recommendation_made"]:
        if n_enrolled != 40:
            raise IntegrityError(f"{label}: a completed trial must enroll 40")
        if row["stop_reason"] is not None or row["region_q_at_stop"] is not None:
            raise IntegrityError(f"{label}: completed trial has stop metadata")
        if empty_events > adaptive_allocations:
            raise IntegrityError(f"{label}: too many empty-gate events")
        _validate_common_completed_row(row, label)
        rec = (_finite(row["rec_d1"], f"{label}.rec_d1"),
               _finite(row["rec_d2"], f"{label}.rec_d2"))
        if not _on_grid(rec):
            raise IntegrityError(f"{label}: terminal recommendation is off grid")
        recs = row["recs"]
        if not isinstance(recs, list) or len(recs) != 6:
            raise IntegrityError(f"{label}.recs: expected six recommendation readouts")
        gates = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
        if tuple(item.get("grec") for item in recs if isinstance(item, dict)) != gates:
            raise IntegrityError(f"{label}.recs: unexpected gate sequence")
        for rec_index, item in enumerate(recs):
            rec_item = _require_mapping(item, f"{label}.recs[{rec_index}]")
            _finite(rec_item.get("du"), f"{label}.recs[{rec_index}].du", lower=0)
            _require_int(
                rec_item.get("unsafe"),
                f"{label}.recs[{rec_index}].unsafe",
                lower=0,
                upper=1,
            )
            _finite(rec_item.get("eff_reg"), f"{label}.recs[{rec_index}].eff_reg")
    else:
        if scaffold != "start_low_expansion" or n_enrolled >= 40:
            raise IntegrityError(f"{label}: only gradual access may stop before n=40")
        if row["stop_reason"] != "NO_FEASIBLE_DOSE":
            raise IntegrityError(f"{label}: stopped trial has unexpected internal stop code")
        expected_q = n_enrolled // 2 + 1
        if row["region_q_at_stop"] != expected_q:
            raise IntegrityError(f"{label}.region_q_at_stop: expected {expected_q}")
        if any(row[field] is not None for field in TERMINAL_FIELDS):
            raise IntegrityError(f"{label}: stopped trial contains terminal recommendation data")
        if row["recs"] != []:
            raise IntegrityError(f"{label}: stopped trial must have recs=[]")
        if empty_events < 3 or empty_events > adaptive_allocations + 1:
            raise IntegrityError(f"{label}: impossible empty-gate stop count")
    key = (scaffold, gamma, stratum, policy, seed)
    pair_key = (scaffold, gamma, stratum, seed)
    return key, pair_key, history[:initialization_size], not row["recommendation_made"]


def validate_sensitivity_records(rows: Sequence[Mapping[str, Any]]) -> int:
    if len(rows) != 4800:
        raise IntegrityError("protocol sensitivity must contain exactly 4,800 rows")
    keys: set[tuple[Any, ...]] = set()
    arm_counts: Counter[tuple[Any, ...]] = Counter()
    initializations: dict[tuple[Any, ...], tuple[tuple[float, float], ...]] = {}
    stopped = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise IntegrityError(f"protocol sensitivity row {index}: expected an object")
        key, pair_key, initialization, did_stop = validate_sensitivity_record(row, index)
        if key in keys:
            raise IntegrityError(f"duplicate protocol-sensitivity design cell: {key}")
        keys.add(key)
        arm_counts[key[:-1]] += 1
        previous = initializations.setdefault(pair_key, initialization)
        if previous != initialization:
            raise IntegrityError(
                f"paired-seed initialization differs across policies at {pair_key}"
            )
        stopped += int(did_stop)
    expected = {
        (scaffold, gamma, stratum, policy, seed)
        for scaffold in SCAFFOLDS
        for gamma in GAMMAS
        for stratum in STRATA
        for policy in POLICIES
        for seed in SEEDS
    }
    if keys != expected:
        raise IntegrityError(
            f"protocol-sensitivity factorial mismatch: {len(expected - keys)} missing, "
            f"{len(keys - expected)} extra"
        )
    if len(arm_counts) != 24 or set(arm_counts.values()) != {200}:
        raise IntegrityError("protocol sensitivity is not 24 arms x 200 seeds")
    return stopped


def validate_schedule_ratio_records(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate the 200 genuinely new post-hoc 2:1 schedule records."""
    if len(rows) != 200:
        raise IntegrityError("schedule-ratio sensitivity must contain exactly 200 rows")
    expected = {(seed, stratum) for seed in range(100) for stratum in STRATA}
    _exact_cells(rows, ("seed", "stratum"), expected, "schedule-ratio sensitivity")
    frozen = {
        "policy": "cEI-tMSE-2to1",
        "sim": "osa",
        "gamma": 0.7,
        "mode": "latent",
        "noise": "fixed",
        "kap": 1.0,
        "budget": 40,
        "warmup": 4,
        "r_k": 2,
        "grid_n": 5,
        "empty_gate": "pf",
        "protocol_scaffold": "lhs_fixed",
        "region_step": 0.25,
        "empty_gate_stop_after": 3,
        "exclude_repeats_during_expansion": True,
        "post_hoc_descriptive": True,
        "schedule_tuned": False,
        "schedule_ratio": "2:1 cEI:tMSE",
        "schedule_cycle": ["cEI", "cEI", "tMSE"],
        "schedule_phase_origin": "first adaptive cohort",
    }
    for index, row in enumerate(rows):
        label = f"schedule-ratio sensitivity row {index}"
        for field, expected_value in frozen.items():
            if row.get(field) != expected_value:
                raise IntegrityError(
                    f"{label}.{field}: {row.get(field)!r} != {expected_value!r}"
                )
        if row.get("recommendation_made") is not True or row.get("stop_reason") is not None:
            raise IntegrityError(f"{label}: every saved 2:1 trial must complete")
        if row.get("n_enrolled") != 40 or row.get("initialization_size") != 4:
            raise IntegrityError(f"{label}: wrong full-panel enrollment/init size")
        raw_history = row.get("allocation_history")
        if not isinstance(raw_history, list) or len(raw_history) != 40:
            raise IntegrityError(f"{label}: allocation_history must contain 40 points")
        history = tuple(
            _point(point, f"{label}.allocation_history[{offset}]")
            for offset, point in enumerate(raw_history)
        )
        for offset in range(4, 40, 2):
            if history[offset] != history[offset + 1] or not _on_grid(history[offset]):
                raise IntegrityError(f"{label}: malformed adaptive cohort at offset {offset}")
        if row.get("n_unique_doses") != len(set(history)):
            raise IntegrityError(f"{label}: n_unique_doses disagrees with allocation history")
        _validate_common_completed_row(row, label)


def validate_formal_projection_identity(
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the exact/stable-z record projection used for current claims.

    The public design manifest must bind the copied projection metadata by hash.
    That metadata in turn binds the formal-run manifest, every projected record,
    and the source files that implemented the formal run.  This check deliberately
    treats old Monte Carlo timing/stability artifacts as a separate historical
    archive component rather than evidence about the current cKG implementation.
    """
    declaration = _require_mapping(
        manifest.get("formal_projection"), "formal projection declaration"
    )
    metadata_relative = declaration.get(
        "metadata_path",
        declaration.get("projection_metadata_path", FORMAL_PROJECTION_METADATA_DEFAULT),
    )
    metadata_path = _safe_archive_path(
        root, metadata_relative, "formal projection metadata_path"
    )
    metadata_digest = file_sha256(metadata_path)
    declared_metadata_digest = declaration.get(
        "metadata_sha256", declaration.get("projection_metadata_sha256")
    )
    if not _valid_sha256(declared_metadata_digest) or (
        declared_metadata_digest != metadata_digest
    ):
        raise IntegrityError(
            "formal projection declaration does not bind the projection metadata SHA-256"
        )

    metadata = _require_mapping(_read_json(metadata_path), "formal projection metadata")
    if metadata.get("schema_version") != 1:
        raise IntegrityError("formal projection metadata has an unsupported schema")
    if metadata.get("status") != FORMAL_PROJECTION_STATUS:
        raise IntegrityError("formal projection metadata is not a complete authenticated projection")
    if not str(metadata.get("artifact_class", "")).startswith("formal_record_projection"):
        raise IntegrityError("formal projection metadata has an unexpected artifact class")
    if declaration.get("status") != FORMAL_PROJECTION_STATUS:
        raise IntegrityError("design manifest does not identify the completed formal projection")
    validation = _require_mapping(
        metadata.get("validation"), "formal projection validation flags"
    )
    for field in FORMAL_PROJECTION_VALIDATION_FLAGS:
        if validation.get(field) is not True:
            raise IntegrityError(f"formal projection validation flag failed: {field}")

    inputs = _require_mapping(metadata.get("inputs"), "formal projection inputs")
    commitments = _require_mapping(
        metadata.get("candidate_records_commitment"),
        "formal projection logical commitments",
    )
    outputs = _require_mapping(metadata.get("outputs"), "formal projection outputs")
    fingerprint = metadata.get("projection_fingerprint")
    if not _valid_sha256(fingerprint):
        raise IntegrityError("formal projection fingerprint is not a SHA-256 digest")
    expected_fingerprint = _canonical_json_sha256(
        {"inputs": inputs, "candidate_records": commitments}
    )
    if fingerprint != expected_fingerprint or declaration.get(
        "projection_fingerprint"
    ) != fingerprint:
        raise IntegrityError("formal projection fingerprint does not authenticate its inputs")

    formal_manifest = _require_mapping(inputs.get("manifest"), "formal run manifest input")
    formal_manifest_sha256 = formal_manifest.get("sha256")
    if (
        formal_manifest.get("basename") != "formal_rerun_manifest.json"
        or formal_manifest_sha256 != EXPECTED_FORMAL_MANIFEST_SHA256
        or declaration.get("formal_manifest_sha256") != formal_manifest_sha256
    ):
        raise IntegrityError("formal projection does not identify the frozen exact/stable-z run manifest")
    execution_sources = _require_mapping(
        inputs.get("formal_execution_source_sha256"),
        "formal projection execution-source hashes",
    )
    if execution_sources.get("paper/formal_rerun_manifest.json") != formal_manifest_sha256:
        raise IntegrityError("formal execution source hashes do not bind the run manifest")
    # Historical source path: preserve the key recorded in the original manifest.
    exact_digest = execution_sources.get("src/safedosebo/ckg_exact.py")
    if not _valid_sha256(exact_digest):
        raise IntegrityError("formal projection does not bind the exact cKG evaluator")
    source_root = root / "source" / "package"
    for relative, digest in execution_sources.items():
        if not _valid_sha256(digest):
            raise IntegrityError(f"formal execution source has invalid SHA-256: {relative}")
        source_path = _safe_archive_path(
            source_root, relative, f"formal execution source {relative}"
        )
        if file_sha256(source_path) != digest:
            raise IntegrityError(f"formal execution source-package hash drifted: {relative}")

    record_items = _manifest_record_items(manifest)
    expected_names = set(FORMAL_RECORD_ROWS)
    if set(commitments) != expected_names or set(outputs) != expected_names:
        raise IntegrityError("formal projection metadata does not commit to the exact ten records")
    total_rows = 0
    normalized_outputs: dict[str, dict[str, Any]] = {}
    for filename, expected_rows in FORMAL_RECORD_ROWS.items():
        item = record_items[filename]
        output = _require_mapping(outputs[filename], f"formal output {filename}")
        commitment = _require_mapping(
            commitments[filename], f"formal logical commitment {filename}"
        )
        normalized = {
            "rows": output.get("rows"),
            "compressed_bytes": output.get("compressed_bytes"),
            "compressed_sha256": output.get("compressed_sha256"),
            "uncompressed_bytes": output.get("uncompressed_bytes"),
            "uncompressed_sha256": output.get("uncompressed_sha256"),
        }
        if normalized["rows"] != expected_rows:
            raise IntegrityError(f"formal output {filename}: row count drifted")
        for field in ("compressed_bytes", "uncompressed_bytes"):
            _require_int(normalized[field], f"formal output {filename}.{field}", lower=1)
        for field in ("compressed_sha256", "uncompressed_sha256"):
            if not _valid_sha256(normalized[field]):
                raise IntegrityError(f"formal output {filename}: invalid {field}")
        expected_logical = {
            "rows": normalized["rows"],
            "uncompressed_bytes": normalized["uncompressed_bytes"],
            "uncompressed_sha256": normalized["uncompressed_sha256"],
        }
        if dict(commitment) != expected_logical:
            raise IntegrityError(f"formal output {filename}: logical commitment mismatch")
        for field in (
            "rows",
            "compressed_sha256",
            "uncompressed_bytes",
            "uncompressed_sha256",
        ):
            if item.get(field) != normalized[field]:
                raise IntegrityError(
                    f"formal output {filename}: design manifest differs at {field}"
                )
        record_path = _safe_archive_path(
            root, item.get("path"), f"formal output {filename}.path"
        )
        if record_path.name != filename or file_sha256(record_path) != normalized[
            "compressed_sha256"
        ]:
            raise IntegrityError(f"formal output {filename}: archived compressed bytes drifted")
        total_rows += expected_rows
        normalized_outputs[filename] = normalized

    counts = _require_mapping(metadata.get("counts"), "formal projection counts")
    count_files = _require_mapping(counts.get("files"), "formal projection counts.files")
    if set(count_files) != expected_names:
        raise IntegrityError("formal projection counts.files record set drifted")
    for filename, expected_rows in FORMAL_RECORD_ROWS.items():
        count_item = _require_mapping(
            count_files[filename], f"formal projection counts.files.{filename}"
        )
        if count_item.get("rows") != expected_rows:
            raise IntegrityError(f"formal projection counts.files.{filename}.rows drifted")
    if (
        total_rows != 55_480
        or counts.get("total_rows") != total_rows
        or counts.get("approved_replacement_rows") != total_rows
        or counts.get("unchanged_rows") != 0
    ):
        raise IntegrityError("formal projection total/replacement counts drifted")
    return {
        "projection_metadata_path": str(metadata_relative),
        "projection_metadata_sha256": metadata_digest,
        "projection_fingerprint": fingerprint,
        "formal_manifest_sha256": formal_manifest_sha256,
        "record_files": normalized_outputs,
    }


def _manifested_analysis_tree(
    root: Path,
    metadata_name: str,
    artifact_root: str,
    package_results_root: str,
) -> tuple[Mapping[str, Any], Path]:
    """Validate an analysis manifest, hashes, and its full source-package copies."""
    metadata = _require_mapping(
        _read_json(root / "metadata" / metadata_name), metadata_name
    )
    if metadata.get("artifact_root") != artifact_root:
        raise IntegrityError(f"{metadata_name}: artifact_root drifted")
    files = _require_mapping(metadata.get("files"), f"{metadata_name}.files")
    analysis_root = root / artifact_root
    actual = {
        str(path.relative_to(root))
        for path in analysis_root.rglob("*")
        if path.is_file()
    }
    if actual != set(files) or metadata.get("file_count") != len(files):
        raise IntegrityError(f"{metadata_name}: analysis file inventory mismatch")
    for relative, digest in files.items():
        archived = root / relative
        if not _valid_sha256(digest) or file_sha256(archived) != digest:
            raise IntegrityError(f"{relative}: analysis SHA-256 mismatch")
        source_relative = archived.relative_to(analysis_root)
        package_copy = root / "source" / "package" / package_results_root / source_relative
        if not package_copy.is_file() or file_sha256(package_copy) != digest:
            raise IntegrityError(f"{relative}: full source-package copy is missing or differs")
    return metadata, analysis_root


def _projection_commitment(
    reference: Any,
    identity: Mapping[str, Any],
    label: str,
    *,
    record: str | None = None,
) -> None:
    reference = _require_mapping(reference, label)
    for field in (
        "projection_metadata_sha256",
        "projection_fingerprint",
        "formal_manifest_sha256",
    ):
        if reference.get(field) != identity.get(field):
            raise IntegrityError(f"{label}: formal-projection identity drifted at {field}")
    if record is not None:
        if reference.get("record") != record:
            raise IntegrityError(f"{label}: selected formal record drifted")
        commitment = _require_mapping(
            reference.get("record_commitment"), f"{label}.record_commitment"
        )
        if dict(commitment) != identity["record_files"][record]:
            raise IntegrityError(f"{label}: formal record commitment drifted")


def _validate_max_t_specification(metadata: Mapping[str, Any], label: str) -> dict[str, float]:
    estimand = _require_mapping(metadata.get("estimand"), f"{label}.estimand")
    if "simultaneous across all 10" not in str(estimand.get("multiplicity", "")):
        raise IntegrityError(f"{label}: joint-family description drifted")
    max_t = _require_mapping(
        metadata.get("simultaneous_max_t"), f"{label}.simultaneous_max_t"
    )
    expected_specification = {
        "joint_family_size": 10,
        "n_seed_clusters": 200,
        "n_resamples": 500_000,
        "rng": "numpy.random.Generator(PCG64)",
        "rng_seed": 20_260_822,
        "batch_size": 1_000,
        "confidence_level": 0.95,
        "quantile_method": "higher",
        "studentization": "bootstrap-sample MCSE recomputed for each coordinate",
        "resampling_unit": "complete matched Monte Carlo replicate-set vector",
        "post_hoc_descriptive": True,
    }
    for field, expected in expected_specification.items():
        if max_t.get(field) != expected:
            raise IntegrityError(f"{label}: max-t reproducibility specification drifted at {field}")
    critical = _require_mapping(max_t.get("critical_values"), f"{label}.critical_values")
    expected_families = {
        "final_threshold_exceeding_recommendation",
        "above_boundary_simulated_assignments",
        "joint_10_contrasts",
    }
    if set(critical) != expected_families:
        raise IntegrityError(f"{label}: max-t family shape drifted")
    values = {
        family: _finite(value, f"{label}.critical_values.{family}", lower=0.0)
        for family, value in critical.items()
    }
    if values["joint_10_contrasts"] < max(
        values["final_threshold_exceeding_recommendation"],
        values["above_boundary_simulated_assignments"],
    ):
        raise IntegrityError(f"{label}: joint critical value is smaller than a subfamily value")
    return values


def _validate_gate_profile_rows(
    rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]
) -> None:
    """Validate family shape and every reported interval formula, not outcomes."""
    if len(rows) != 30:
        raise IntegrityError("main OSA gate-profile CSV must contain 30 contrast rows")
    expected_cells = {
        (str(tau), outcome, level, stratum)
        for tau in (0.5, 0.6, 0.7, 0.8, 0.9)
        for outcome in (
            "final_threshold_exceeding_recommendation",
            "above_boundary_simulated_assignments",
        )
        for level, stratum in (("pooled_strata", ""), ("stratum", "0"), ("stratum", "1"))
    }
    actual_cells = {
        (row["tau"], row["outcome_id"], row["analysis_level"], row["stratum"])
        for row in rows
    }
    if actual_cells != expected_cells:
        raise IntegrityError("main OSA gate-profile CSV factorial drifted")
    pooled = [row for row in rows if row["analysis_level"] == "pooled_strata"]
    if len(pooled) != 10:
        raise IntegrityError("main OSA gate-profile CSV must contain 10 pooled rows")
    critical_values = _validate_max_t_specification(metadata, "main OSA gate profile")
    for index, row in enumerate(rows):
        label = f"main OSA gate-profile row {index}"
        estimate = _csv_float(row.get("difference_mean_pp"), f"{label}.difference")
        mcse = _csv_float(row.get("difference_mcse_pp"), f"{label}.mcse")
        if mcse < 0.0:
            raise IntegrityError(f"{label}: MCSE must be nonnegative")
        left = _csv_float(row.get("cKG_mean_pct"), f"{label}.cKG mean")
        right = _csv_float(row.get("cEI_tMSE_mean_pct"), f"{label}.comparator mean")
        _isclose(estimate, left - right, f"{label}.contrast formula")
        _isclose(
            row.get("mc_interval_95_low_pp"), estimate - 1.96 * mcse,
            f"{label}.pointwise lower formula",
        )
        _isclose(
            row.get("mc_interval_95_high_pp"), estimate + 1.96 * mcse,
            f"{label}.pointwise upper formula",
        )
        if row["analysis_level"] == "pooled_strata":
            if row.get("eligible_pairs") != "400" or row.get("independent_seeds") != "200":
                raise IntegrityError(f"{label}: pooled denominator drifted")
            outcome_id = row["outcome_id"]
            curve = critical_values[outcome_id]
            joint = critical_values["joint_10_contrasts"]
            _isclose(row.get("curve_max_t_critical"), curve, f"{label}.curve critical")
            _isclose(row.get("joint10_max_t_critical"), joint, f"{label}.joint critical")
            _isclose(
                row.get("curve_simultaneous_95_low_pp"), estimate - curve * mcse,
                f"{label}.curve lower formula",
            )
            _isclose(
                row.get("curve_simultaneous_95_high_pp"), estimate + curve * mcse,
                f"{label}.curve upper formula",
            )
            _isclose(
                row.get("joint10_simultaneous_95_low_pp"), estimate - joint * mcse,
                f"{label}.joint lower formula",
            )
            _isclose(
                row.get("joint10_simultaneous_95_high_pp"), estimate + joint * mcse,
                f"{label}.joint upper formula",
            )
        else:
            if row.get("eligible_pairs") != "200" or row.get("independent_seeds") != "200":
                raise IntegrityError(f"{label}: stratum denominator drifted")
            for field in (
                "curve_max_t_critical",
                "curve_simultaneous_95_low_pp",
                "curve_simultaneous_95_high_pp",
                "joint10_max_t_critical",
                "joint10_simultaneous_95_low_pp",
                "joint10_simultaneous_95_high_pp",
            ):
                if row.get(field) not in (None, ""):
                    raise IntegrityError(f"{label}: non-plotted row carries {field}")


def _validate_main_osa_primary_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 3 or {row.get("policy") for row in rows} != set(POLICIES):
        raise IntegrityError("main OSA primary CSV must contain exactly the three policies")
    for index, row in enumerate(rows):
        label = f"main OSA primary row {index}"
        if row.get("chance_control") != "False" or row.get("n") != "200" or row.get("n_rows") != "400":
            raise IntegrityError(f"{label}: primary denominator/design drifted")
        patients = _csv_float(row.get("patients_above"), f"{label}.patients_above")
        percentage = _csv_float(
            row.get("pct_patients_above"), f"{label}.pct_patients_above"
        )
        _isclose(percentage, 100.0 * patients / 40.0, f"{label}.assignment percentage")
        for field in (
            "recommendation_pct",
            "pcs_within1",
            "rec_unsafe_pct",
            "rec_efficacy",
            "dose_units",
        ):
            _csv_float(row.get(field), f"{label}.{field}")


def validate_gate_profile_analysis(
    root: Path,
    manifest: Mapping[str, Any],
    projection_identity: Mapping[str, Any],
) -> None:
    """Validate the current primary table and five-threshold OSA profile."""
    gate_manifest, analysis_root = _manifested_analysis_tree(
        root,
        "main_osa_gate_profile_analysis_manifest.json",
        "analysis/main_osa",
        "results/main_osa_analysis",
    )
    required_artifacts = {
        "main_osa_primary.csv",
        "main_osa_primary.tex",
        "osa_gate_profile.eps",
        "osa_gate_profile.pdf",
        "osa_gate_profile.png",
        "osa_gate_profile_contrasts.csv",
    }
    required_files = required_artifacts | {
        name + ".metadata.json" for name in required_artifacts
    }
    actual_files = {path.name for path in analysis_root.iterdir() if path.is_file()}
    if (
        actual_files != required_files
        or gate_manifest.get("artifact_count") != len(required_artifacts)
        or gate_manifest.get("metadata_sidecar_count") != len(required_artifacts)
    ):
        raise IntegrityError("main OSA analysis artifact and sidecar inventory drifted")

    metadata_by_artifact = {}
    for artifact_name in sorted(required_artifacts):
        sidecar = analysis_root / f"{artifact_name}.metadata.json"
        metadata = _require_mapping(_read_json(sidecar), f"{sidecar.name} metadata")
        metadata_by_artifact[artifact_name] = metadata
        _projection_commitment(
            metadata.get("formal_projection"),
            projection_identity,
            f"{sidecar.name}.formal_projection",
            record="osa_main.json.zst",
        )
        if metadata.get("source_path") != "osa_main.json.zst":
            raise IntegrityError(f"{sidecar.name}: source_path drifted")
        expected = projection_identity["record_files"]["osa_main.json.zst"]
        if (
            metadata.get("source_sha256") != expected["compressed_sha256"]
            or metadata.get("source_uncompressed_sha256")
            != expected["uncompressed_sha256"]
        ):
            raise IntegrityError(f"{sidecar.name}: source commitment drifted")

    for artifact_name in (
        "osa_gate_profile.eps",
        "osa_gate_profile.pdf",
        "osa_gate_profile.png",
        "osa_gate_profile_contrasts.csv",
    ):
        metadata = metadata_by_artifact[artifact_name]
        if metadata.get("record_count") != 4_000 or metadata.get("seed_count") != 200:
            raise IntegrityError(f"{artifact_name}: gate-profile inventory drifted")
        if metadata.get("simultaneous_max_t") != metadata_by_artifact[
            "osa_gate_profile_contrasts.csv"
        ].get("simultaneous_max_t"):
            raise IntegrityError(f"{artifact_name}: max-t metadata differs across artifacts")
        _validate_max_t_specification(metadata, artifact_name)

    gate_rows = _read_csv(analysis_root / "osa_gate_profile_contrasts.csv")
    _validate_gate_profile_rows(
        gate_rows, metadata_by_artifact["osa_gate_profile_contrasts.csv"]
    )
    primary_rows = _read_csv(analysis_root / "main_osa_primary.csv")
    _validate_main_osa_primary_rows(primary_rows)
    for artifact_name in ("main_osa_primary.csv", "main_osa_primary.tex"):
        metadata = metadata_by_artifact[artifact_name]
        if metadata.get("record_count") != 1_200 or metadata.get("seed_count") != 200:
            raise IntegrityError(f"{artifact_name}: primary-table inventory drifted")

    figure = analysis_root / "osa_gate_profile.pdf"
    manuscript_figure = root / "source" / "manuscript" / "fig_osa_gate_profile.pdf"
    if file_sha256(figure) != file_sha256(manuscript_figure):
        raise IntegrityError("main OSA gate-profile PDF differs from the manuscript figure")
    generated_table = root / "source" / "manuscript" / "generated" / "main_osa_primary.tex"
    if file_sha256(analysis_root / "main_osa_primary.tex") != file_sha256(generated_table):
        raise IntegrityError("main OSA primary table differs from manuscript generated source")


def _cross_cell_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "testbed_id": row["testbed_id"],
        "testbed": row["testbed"],
        "tau": float(row["tau"]),
        "difference_mean_pp": float(row["difference_mean_pp"]),
        "pointwise_95_low_pp": float(row["pointwise_95_low_pp"]),
        "pointwise_95_high_pp": float(row["pointwise_95_high_pp"]),
    }


def _cross_sign_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    positive = sum(float(row["difference_mean_pp"]) > 0.0 for row in rows)
    negative = sum(float(row["difference_mean_pp"]) < 0.0 for row in rows)
    return {"positive": positive, "negative": negative, "zero": len(rows) - positive - negative}


def _cross_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes: dict[str, Any] = {}
    for outcome_id in ("terminal_recommendation", "assignment_percentage"):
        chosen = [row for row in rows if row["outcome_id"] == outcome_id]
        outcomes[outcome_id] = {
            "cells": len(chosen),
            "sign_counts": _cross_sign_counts(chosen),
            "pointwise_intervals_excluding_zero": sum(
                _bool_cell(row["interval_excludes_zero"], "cross exclusion")
                for row in chosen
            ),
            "pointwise_intervals_including_zero_cells": [
                _cross_cell_summary(row)
                for row in chosen
                if not _bool_cell(row["interval_excludes_zero"], "cross exclusion")
            ],
        }
    terminal = {
        (row["testbed_id"], float(row["tau"])): row
        for row in rows
        if row["outcome_id"] == "terminal_recommendation"
    }
    assignment = {
        (row["testbed_id"], float(row["tau"])): row
        for row in rows
        if row["outcome_id"] == "assignment_percentage"
    }
    if set(terminal) != set(assignment):
        raise IntegrityError("cross-testbed outcomes do not share the same cells")
    return {
        "outcomes": outcomes,
        "opposite_sign_point_estimate_cells": sum(
            float(terminal[key]["difference_mean_pp"])
            * float(assignment[key]["difference_mean_pp"])
            < 0.0
            for key in terminal
        ),
        "both_pointwise_intervals_excluding_zero": sum(
            _bool_cell(terminal[key]["interval_excludes_zero"], "terminal exclusion")
            and _bool_cell(assignment[key]["interval_excludes_zero"], "assignment exclusion")
            for key in terminal
        ),
    }


def _cross_minority_annotations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    annotations = []
    annotation_names = {
        "terminal_recommendation": "terminal-recommendation",
        "assignment_percentage": "assignment",
    }
    for outcome_id in ("terminal_recommendation", "assignment_percentage"):
        chosen = [row for row in rows if row["outcome_id"] == outcome_id]
        signs = _cross_sign_counts(chosen)
        if signs["positive"] == 1 and signs["negative"] > 1:
            direction = "positive"
        elif signs["negative"] == 1 and signs["positive"] > 1:
            direction = "negative"
        else:
            continue
        row = next(
            item
            for item in chosen
            if (float(item["difference_mean_pp"]) > 0.0) == (direction == "positive")
        )
        annotations.append({
            **_cross_cell_summary(row),
            "outcome_id": outcome_id,
            "direction": direction,
            "text": (
                f"Only {direction} {annotation_names[outcome_id]} estimate\n"
                f"{float(row['difference_mean_pp']):+.2f} "
                f"[{float(row['pointwise_95_low_pp']):.2f}, "
                f"{float(row['pointwise_95_high_pp']):.2f}]"
            ),
        })
    return annotations


def _validate_cross_rows_and_metadata(
    rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]
) -> None:
    expected_cells = {
        (testbed, str(tau), outcome)
        for testbed in ("osa", "gbump", "efftox", "mariposa")
        for tau in (0.5, 0.6, 0.7, 0.8, 0.9)
        for outcome in ("terminal_recommendation", "assignment_percentage")
    }
    actual_cells = {(row["testbed_id"], row["tau"], row["outcome_id"]) for row in rows}
    if len(rows) != 40 or actual_cells != expected_cells:
        raise IntegrityError("cross-testbed CSV factorial drifted")
    for index, row in enumerate(rows):
        label = f"cross-testbed row {index}"
        estimate = _csv_float(row.get("difference_mean_pp"), f"{label}.difference")
        mcse = _csv_float(row.get("difference_mcse_pp"), f"{label}.mcse")
        if mcse < 0.0:
            raise IntegrityError(f"{label}: MCSE must be nonnegative")
        left = _csv_float(row.get("cKG_mean_pct"), f"{label}.cKG mean")
        right = _csv_float(row.get("cEI_tMSE_mean_pct"), f"{label}.comparator mean")
        _isclose(estimate, left - right, f"{label}.contrast formula")
        low = estimate - 1.96 * mcse
        high = estimate + 1.96 * mcse
        _isclose(row.get("pointwise_95_low_pp"), low, f"{label}.lower formula")
        _isclose(row.get("pointwise_95_high_pp"), high, f"{label}.upper formula")
        if _bool_cell(row.get("interval_excludes_zero"), f"{label}.exclusion") != (
            low > 0.0 or high < 0.0
        ):
            raise IntegrityError(f"{label}: exclusion flag disagrees with interval bounds")
        n = _csv_int(row.get("independent_seeds"), f"{label}.independent_seeds", lower=2)
        if _csv_int(row.get("stratum_cells"), f"{label}.stratum_cells", lower=1) != 2 * n:
            raise IntegrityError(f"{label}: stratum-cell denominator drifted")

    summary = _cross_summary(rows)
    if metadata.get("result_summary") != summary:
        raise IntegrityError("cross-testbed metadata result_summary differs from CSV")
    terminal = summary["outcomes"]["terminal_recommendation"]
    assignment = summary["outcomes"]["assignment_percentage"]
    alias_expectations = {
        "positive_terminal_point_estimates": terminal["sign_counts"]["positive"],
        "negative_assignment_point_estimates": assignment["sign_counts"]["negative"],
        "terminal_pointwise_intervals_excluding_zero": terminal[
            "pointwise_intervals_excluding_zero"
        ],
        "assignment_pointwise_intervals_excluding_zero": assignment[
            "pointwise_intervals_excluding_zero"
        ],
        "opposite_direction_point_estimate_cells": summary[
            "opposite_sign_point_estimate_cells"
        ],
        "both_pointwise_intervals_excluding_zero": summary[
            "both_pointwise_intervals_excluding_zero"
        ],
        "terminal_pointwise_interval_exceptions": terminal[
            "pointwise_intervals_including_zero_cells"
        ],
        "assignment_pointwise_interval_exceptions": assignment[
            "pointwise_intervals_including_zero_cells"
        ],
        "pointwise_interval_exceptions": terminal[
            "pointwise_intervals_including_zero_cells"
        ],
        "minority_sign_annotations": _cross_minority_annotations(rows),
    }
    positive_assignment = [
        row for row in rows
        if row["outcome_id"] == "assignment_percentage"
        and float(row["difference_mean_pp"]) > 0.0
    ]
    alias_expectations["unique_positive_assignment_estimate"] = (
        _cross_cell_summary(positive_assignment[0])
        if len(positive_assignment) == 1
        else None
    )
    for field, expected in alias_expectations.items():
        if metadata.get(field) != expected:
            raise IntegrityError(f"cross-testbed metadata summary differs from CSV at {field}")
    scope = str(_require_mapping(metadata.get("estimand"), "cross estimand").get("scope", ""))
    interval = str(
        _require_mapping(metadata.get("estimand"), "cross estimand").get("interval", "")
    )
    if "no universal policy ordering" not in scope or "pointwise and unadjusted" not in interval:
        raise IntegrityError("cross-testbed descriptive scope drifted")


def validate_cross_testbed_analysis(
    root: Path,
    manifest: Mapping[str, Any],
    projection_identity: Mapping[str, Any],
) -> None:
    """Validate the descriptive profile without pinning its observed directions."""
    cross_manifest, analysis_root = _manifested_analysis_tree(
        root,
        "cross_testbed_analysis_manifest.json",
        "analysis/cross_testbed",
        "results/cross_testbed_analysis",
    )
    if (
        cross_manifest.get("artifact_count") != 4
        or cross_manifest.get("metadata_sidecar_count") != 4
        or cross_manifest.get("selected_record_count") != 12_000
        or cross_manifest.get("displayed_contrasts") != 40
    ):
        raise IntegrityError("cross-testbed manifest counts drifted")
    csv_path = analysis_root / "cross_testbed_terminal_contrasts.csv"
    csv_metadata = _require_mapping(
        _read_json(csv_path.with_suffix(csv_path.suffix + ".metadata.json")),
        "cross-testbed CSV metadata",
    )
    rows = _read_csv(csv_path)
    _validate_cross_rows_and_metadata(rows, csv_metadata)
    expected_input_names = {
        "osa": "osa_main.json.zst",
        "gbump": "gbump_main.json.zst",
        "efftox": "efftox_main.json.zst",
        "mariposa": "mariposa_main.json.zst",
    }
    for artifact in (
        "cross_testbed_terminal_contrasts.csv",
        "cross_testbed_terminal_contrasts.eps",
        "cross_testbed_terminal_contrasts.pdf",
        "cross_testbed_terminal_contrasts.png",
    ):
        path = analysis_root / artifact
        metadata = _require_mapping(
            _read_json(path.with_suffix(path.suffix + ".metadata.json")),
            f"{artifact} metadata",
        )
        if metadata.get("artifact_sha256") != file_sha256(path):
            raise IntegrityError(f"{artifact}: sidecar artifact hash drifted")
        _projection_commitment(
            metadata.get("projection"), projection_identity, f"{artifact}.projection"
        )
        if metadata.get("result_summary") != csv_metadata.get("result_summary"):
            raise IntegrityError(f"{artifact}: result summary differs across sidecars")
        inputs = _require_mapping(metadata.get("input_files"), f"{artifact}.input_files")
        if set(inputs) != set(expected_input_names):
            raise IntegrityError(f"{artifact}: formal input-file set drifted")
        for testbed, record in expected_input_names.items():
            item = _require_mapping(inputs[testbed], f"{artifact}.input_files.{testbed}")
            expected = projection_identity["record_files"][record]
            if item != {
                "source_path": record,
                "trial_rows": expected["rows"],
                "compressed_sha256": expected["compressed_sha256"],
                "uncompressed_sha256": expected["uncompressed_sha256"],
            }:
                raise IntegrityError(f"{artifact}: formal input commitment drifted for {testbed}")
    figure = analysis_root / "cross_testbed_terminal_contrasts.pdf"
    manuscript_figure = root / "source" / "manuscript" / "fig_cross_testbed_terminal.pdf"
    if file_sha256(figure) != file_sha256(manuscript_figure):
        raise IntegrityError("cross-testbed PDF differs from the manuscript figure")


SCHEDULE_POLICY_ORDER = (
    "cEI",
    "tMSE",
    "cEI-tMSE-1to1",
    "cEI-tMSE-2to1",
    "cKG",
)
SCHEDULE_REFERENCE_ALIASES = {
    "cEI": "cEI",
    "straddle": "tMSE",
    "GBE": "cEI-tMSE-1to1",
    "cKG1fix": "cKG",
}
SCHEDULE_CONTRASTS = (
    ("cKG", "cEI-tMSE-1to1"),
    ("cKG", "cEI-tMSE-2to1"),
    ("cKG", "tMSE"),
    ("cKG", "cEI"),
    ("cEI-tMSE-2to1", "cEI-tMSE-1to1"),
)
SCHEDULE_OUTCOMES = (
    "terminal_threshold_event",
    "above_boundary_assignment_count",
    "above_boundary_assignment_percentage",
)


def _mean_mcse(values: Sequence[float], label: str) -> tuple[float, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise IntegrityError(f"{label}: expected non-empty finite seed values")
    mean = math.fsum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    sum_squares = math.fsum((value - mean) ** 2 for value in values)
    mcse = math.sqrt(sum_squares / (len(values) - 1)) / math.sqrt(len(values))
    return mean, mcse


def _schedule_metric(row: Mapping[str, Any], outcome_id: str) -> float:
    if outcome_id == "terminal_threshold_event":
        return 100.0 * float(row["rec_unsafe"])
    if outcome_id == "above_boundary_assignment_count":
        return float(row["toxic"])
    if outcome_id == "above_boundary_assignment_percentage":
        return 100.0 * float(row["toxic"]) / float(row["budget"])
    raise IntegrityError(f"unknown schedule outcome {outcome_id!r}")


def _schedule_index(
    sixway_rows: Sequence[Mapping[str, Any]],
    schedule_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in sixway_rows:
        source_policy = row.get("policy")
        if not (
            row.get("sim") == "osa"
            and row.get("mode") == "latent"
            and row.get("gamma") == 0.7
            and source_policy in SCHEDULE_REFERENCE_ALIASES
        ):
            continue
        policy = SCHEDULE_REFERENCE_ALIASES[source_policy]
        key = (policy, int(row["seed"]), int(row["stratum"]))
        if key in indexed:
            raise IntegrityError(f"duplicate schedule reference cell: {key}")
        indexed[key] = row
    for row in schedule_rows:
        key = ("cEI-tMSE-2to1", int(row["seed"]), int(row["stratum"]))
        if key in indexed:
            raise IntegrityError(f"duplicate 2:1 schedule cell: {key}")
        indexed[key] = row
    expected = {
        (policy, seed, stratum)
        for policy in SCHEDULE_POLICY_ORDER
        for seed in range(100)
        for stratum in STRATA
    }
    if set(indexed) != expected:
        raise IntegrityError("authenticated inputs do not contain the exact schedule matrix")
    return indexed


def _derive_schedule_results(
    sixway_rows: Sequence[Mapping[str, Any]],
    schedule_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], tuple[float, float]], dict[tuple[str, str, str], tuple[float, float]]]:
    indexed = _schedule_index(sixway_rows, schedule_rows)
    summaries: dict[tuple[str, str], tuple[float, float]] = {}
    for policy in SCHEDULE_POLICY_ORDER:
        for outcome_id in SCHEDULE_OUTCOMES:
            seed_values = [
                math.fsum(
                    _schedule_metric(indexed[(policy, seed, stratum)], outcome_id)
                    for stratum in STRATA
                )
                / len(STRATA)
                for seed in range(100)
            ]
            summaries[(policy, outcome_id)] = _mean_mcse(
                seed_values, f"schedule summary {policy}/{outcome_id}"
            )
    contrasts: dict[tuple[str, str, str], tuple[float, float]] = {}
    for policy_a, policy_b in SCHEDULE_CONTRASTS:
        for outcome_id in SCHEDULE_OUTCOMES:
            seed_values = [
                math.fsum(
                    _schedule_metric(indexed[(policy_a, seed, stratum)], outcome_id)
                    - _schedule_metric(indexed[(policy_b, seed, stratum)], outcome_id)
                    for stratum in STRATA
                )
                / len(STRATA)
                for seed in range(100)
            ]
            contrasts[(policy_a, policy_b, outcome_id)] = _mean_mcse(
                seed_values, f"schedule contrast {policy_a}/{policy_b}/{outcome_id}"
            )
    return summaries, contrasts


def _validate_schedule_csvs(
    summary_rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
    derived_summaries: Mapping[tuple[str, str], tuple[float, float]],
    derived_contrasts: Mapping[tuple[str, str, str], tuple[float, float]],
) -> None:
    summary_index = {
        (row["policy"], row["outcome_id"]): row for row in summary_rows
    }
    if len(summary_rows) != 15 or set(summary_index) != set(derived_summaries):
        raise IntegrityError("schedule operating-characteristics CSV factorial drifted")
    for key, expected in derived_summaries.items():
        row = summary_index[key]
        mean, mcse = expected
        _isclose(row.get("mean"), mean, f"schedule summary {key}.mean")
        _isclose(row.get("mcse"), mcse, f"schedule summary {key}.mcse")
        _isclose(
            row.get("pointwise_95_low"), mean - 1.96 * mcse,
            f"schedule summary {key}.lower",
        )
        _isclose(
            row.get("pointwise_95_high"), mean + 1.96 * mcse,
            f"schedule summary {key}.upper",
        )
        if row.get("independent_seeds") != "100" or row.get("stratum_trials") != "200":
            raise IntegrityError(f"schedule summary {key}: denominator drifted")
    for policy in SCHEDULE_POLICY_ORDER:
        count = summary_index[(policy, "above_boundary_assignment_count")]
        percentage = summary_index[(policy, "above_boundary_assignment_percentage")]
        for field in ("mean", "mcse", "pointwise_95_low", "pointwise_95_high"):
            _isclose(
                percentage[field], 2.5 * float(count[field]),
                f"schedule summary {policy}: percentage/count {field}",
            )

    contrast_index = {
        (row["policy_a"], row["policy_b"], row["outcome_id"]): row
        for row in contrast_rows
    }
    if len(contrast_rows) != 15 or set(contrast_index) != set(derived_contrasts):
        raise IntegrityError("schedule paired-contrast CSV factorial drifted")
    for key, expected in derived_contrasts.items():
        row = contrast_index[key]
        mean, mcse = expected
        policy_a, policy_b, outcome_id = key
        _isclose(row.get("difference_mean"), mean, f"schedule contrast {key}.mean")
        _isclose(row.get("difference_mcse"), mcse, f"schedule contrast {key}.mcse")
        _isclose(
            row.get("pointwise_95_low"), mean - 1.96 * mcse,
            f"schedule contrast {key}.lower",
        )
        _isclose(
            row.get("pointwise_95_high"), mean + 1.96 * mcse,
            f"schedule contrast {key}.upper",
        )
        _isclose(
            row.get("policy_a_mean"), derived_summaries[(policy_a, outcome_id)][0],
            f"schedule contrast {key}.policy_a_mean",
        )
        _isclose(
            row.get("policy_b_mean"), derived_summaries[(policy_b, outcome_id)][0],
            f"schedule contrast {key}.policy_b_mean",
        )
        if row.get("independent_seeds") != "100" or row.get("eligible_stratum_pairs") != "200":
            raise IntegrityError(f"schedule contrast {key}: denominator drifted")
    for policy_a, policy_b in SCHEDULE_CONTRASTS:
        count = contrast_index[(policy_a, policy_b, "above_boundary_assignment_count")]
        percentage = contrast_index[
            (policy_a, policy_b, "above_boundary_assignment_percentage")
        ]
        for field in (
            "policy_a_mean",
            "policy_b_mean",
            "difference_mean",
            "difference_mcse",
            "pointwise_95_low",
            "pointwise_95_high",
        ):
            _isclose(
                percentage[field], 2.5 * float(count[field]),
                f"schedule contrast {policy_a}/{policy_b}: percentage/count {field}",
            )


def validate_schedule_ratio_analysis(
    root: Path,
    manifest: Mapping[str, Any],
    projection_identity: Mapping[str, Any],
    sixway_rows: Sequence[Mapping[str, Any]],
    schedule_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Recompute every schedule result from authenticated projected records."""
    schedule_manifest, analysis_root = _manifested_analysis_tree(
        root,
        "schedule_ratio_analysis_manifest.json",
        "analysis/schedule_ratio",
        "results/schedule_ratio_analysis",
    )
    required_files = {
        "schedule_ratio_analysis.metadata.json",
        "schedule_ratio_operating_characteristics.csv",
        "schedule_ratio_paired_contrasts.csv",
        "schedule_ratio_table.tex",
    }
    if (
        schedule_manifest.get("file_count") != len(required_files)
        or {path.name for path in analysis_root.iterdir() if path.is_file()} != required_files
    ):
        raise IntegrityError("schedule-ratio analysis must contain exactly four current files")
    schedule_item = _require_mapping(
        manifest.get("schedule_ratio_sensitivity"), "schedule-ratio record metadata"
    )
    raw_metadata_path = _safe_archive_path(
        root, schedule_item.get("record_metadata"), "schedule raw metadata path"
    )
    raw_metadata = _require_mapping(_read_json(raw_metadata_path), "schedule raw metadata")
    expected_raw = {
        "record_count": 200,
        "seed_count": 100,
        "seeds": list(range(100)),
        "new_policy": "cEI-tMSE-2to1",
        "schedule_cycle": ["cEI", "cEI", "tMSE"],
        "post_hoc_descriptive": True,
        "confirmatory": False,
        "ratio_tuned": False,
        "source_sha256": schedule_item.get("uncompressed_sha256"),
    }
    for field, expected_value in expected_raw.items():
        if raw_metadata.get(field) != expected_value:
            raise IntegrityError(f"schedule raw metadata drifted at {field!r}")
    analysis_metadata = _require_mapping(
        _read_json(analysis_root / "schedule_ratio_analysis.metadata.json"),
        "schedule analysis metadata",
    )
    counts = _require_mapping(analysis_metadata.get("record_counts"), "schedule counts")
    if counts != {
        "formal_sixway_reference_appearances": 800,
        "formal_2to1_appearances": 200,
        "combined_rows": 1000,
    }:
        raise IntegrityError("schedule analysis record counts drifted")
    if analysis_metadata.get("post_hoc_descriptive") is not True or analysis_metadata.get(
        "confirmatory"
    ) is not False or analysis_metadata.get("ratio_tuned") is not False:
        raise IntegrityError("schedule analysis status drifted")
    inputs = _require_mapping(analysis_metadata.get("inputs"), "schedule inputs")
    formal_input = _require_mapping(inputs.get("formal_projection"), "schedule formal input")
    _projection_commitment(formal_input, projection_identity, "schedule formal input")
    selected = _require_mapping(
        formal_input.get("selected_records"), "schedule selected formal records"
    )
    expected_selected = {
        name: projection_identity["record_files"][name]
        for name in (
            "gate_sixway_supplemental.json.zst",
            "schedule_ratio_sensitivity_2to1.json.zst",
        )
    }
    if selected != expected_selected:
        raise IntegrityError("schedule analysis selected-record commitments drifted")
    provenance = _require_mapping(analysis_metadata.get("provenance"), "schedule provenance")
    if provenance.get("all_analysis_rows_authenticated_by_formal_projection") is not True:
        raise IntegrityError("schedule analysis is not identified as formal-projection derived")

    outputs = _require_mapping(analysis_metadata.get("outputs"), "schedule outputs")
    for filename, expected_rows in (
        ("schedule_ratio_operating_characteristics.csv", 15),
        ("schedule_ratio_paired_contrasts.csv", 15),
        ("schedule_ratio_table.tex", 5),
    ):
        item = _require_mapping(outputs.get(filename), f"schedule output {filename}")
        if item.get("rows") != expected_rows or item.get("sha256") != file_sha256(
            analysis_root / filename
        ):
            raise IntegrityError(f"schedule output metadata drifted for {filename}")

    derived_summaries, derived_contrasts = _derive_schedule_results(
        sixway_rows, schedule_rows
    )
    summary_rows = _read_csv(
        analysis_root / "schedule_ratio_operating_characteristics.csv"
    )
    contrast_rows = _read_csv(analysis_root / "schedule_ratio_paired_contrasts.csv")
    _validate_schedule_csvs(
        summary_rows, contrast_rows, derived_summaries, derived_contrasts
    )
    if file_sha256(analysis_root / "schedule_ratio_table.tex") != file_sha256(
        root / "source" / "manuscript" / "generated" / "schedule_ratio_table.tex"
    ):
        raise IntegrityError("schedule-ratio table differs from manuscript generated source")


def validate_computational_diagnostics(root: Path) -> None:
    """Validate legacy Monte Carlo diagnostics as hash-bound historical material.

    These artifacts are retained to document the earlier numerical investigation.
    They are explicitly not accepted as evidence about the current exact cKG
    evaluator, so no old timing ratio, agreement percentage, or MC anchor value is
    an acceptance criterion for the current release.
    """
    compute_manifest, analysis_root = _manifested_analysis_tree(
        root,
        "computational_diagnostics_manifest.json",
        "analysis/computational_diagnostics",
        "results/computational_diagnostics",
    )
    if compute_manifest.get("artifact_count") != 6 or compute_manifest.get(
        "metadata_sidecar_count"
    ) != 6:
        raise IntegrityError("computational diagnostics artifact/sidecar counts drifted")
    scope = str(compute_manifest.get("claim_scope", "")).lower()
    if (
        compute_manifest.get("logical_trial_records") != 0
        or compute_manifest.get("current_manuscript_implementation") is not False
        or "historical" not in scope
        or "exact" not in scope
        or "not" not in scope
        or "current" not in scope
    ):
        raise IntegrityError(
            "legacy computational diagnostics must be marked historical and not the "
            "current exact cKG implementation"
        )
    for artifact in analysis_root.iterdir():
        if not artifact.is_file() or artifact.name.endswith(".metadata.json"):
            continue
        sidecar = artifact.with_suffix(artifact.suffix + ".metadata.json")
        metadata = _require_mapping(_read_json(sidecar), f"{sidecar.name} metadata")
        if metadata.get("artifact") != artifact.name or metadata.get(
            "artifact_sha256"
        ) != file_sha256(artifact):
            raise IntegrityError(f"{artifact.name}: computational metadata hash drifted")
        if metadata.get("status") != "post hoc and descriptive":
            raise IntegrityError(f"{artifact.name}: computational status drifted")
    baseline = _require_mapping(
        _read_json(analysis_root / "computational_diagnostics_raw.json"),
        "computational diagnostics raw",
    )
    for field in (
        "timing_trials",
        "stability_trials",
        "same_state_reference_trials",
        "same_state_decisions",
    ):
        value = baseline.get(field)
        if not isinstance(value, list) or not value or not all(
            isinstance(row, dict) for row in value
        ):
            raise IntegrityError(f"historical computational diagnostics {field} is malformed")
    anchor = _require_mapping(
        _read_json(analysis_root / "ckg_1024_anchor_raw.json"), "cKG-1024 anchor raw"
    )
    for field in (
        "anchor_full_path_trials",
        "reference_path_trials",
        "same_state_decisions",
    ):
        value = anchor.get(field)
        if not isinstance(value, list) or not value or not all(
            isinstance(row, dict) for row in value
        ):
            raise IntegrityError(f"historical cKG anchor {field} is malformed")
    for metadata_name in (
        "computational_diagnostics_raw.json.metadata.json",
        "ckg_1024_anchor_raw.json.metadata.json",
    ):
        metadata = _require_mapping(_read_json(analysis_root / metadata_name), metadata_name)
        source_hashes = _require_mapping(metadata.get("source_sha256"), f"{metadata_name}.source_sha256")
        for source_name, expected_digest in source_hashes.items():
            if not isinstance(source_name, str) or not _valid_sha256(expected_digest):
                raise IntegrityError(f"{metadata_name}: malformed historical source commitment")
    anchor_metadata = _require_mapping(
        _read_json(analysis_root / "ckg_1024_anchor_raw.json.metadata.json"), "anchor metadata"
    )
    reference = _require_mapping(
        _require_mapping(anchor_metadata.get("design"), "anchor metadata design").get("reference_source"),
        "anchor reference source",
    )
    if reference.get("artifact_sha256") != file_sha256(
        analysis_root / "computational_diagnostics_raw.json"
    ) or reference.get("metadata_sha256") != file_sha256(
        analysis_root / "computational_diagnostics_raw.json.metadata.json"
    ):
        raise IntegrityError("cKG-1024 parent-artifact link drifted")
    for summary_name in (
        "computational_diagnostics_summary.json",
        "ckg_1024_anchor_summary.json",
    ):
        summary = _require_mapping(_read_json(analysis_root / summary_name), summary_name)
        if summary.get("status") != "post hoc and descriptive" or summary.get(
            "schema_version"
        ) != 1:
            raise IntegrityError(f"{summary_name}: malformed historical summary envelope")


def _expected_comparator_extension_cells() -> set[tuple[Any, ...]]:
    expected: set[tuple[Any, ...]] = set()
    for policy in ("qBIG", "tmse"):
        for sim in ("osa", "gbump"):
            for gamma in (0.5, 0.6, 0.7, 0.8, 0.9):
                for stratum in STRATA:
                    for seed in range(100, 200):
                        expected.add((
                            "family_full_osa_gbump_boundary_completion",
                            policy, sim, "latent", "lhs_fixed", 1.0,
                            gamma, stratum, seed,
                        ))
        for sim in ("efftox", "mariposa"):
            for gamma in (0.5, 0.6, 0.8, 0.9):
                for stratum in STRATA:
                    for seed in range(100):
                        expected.add((
                            "family_full_efftox_mariposa_boundary_completion",
                            policy, sim, "latent", "lhs_fixed", 1.0,
                            gamma, stratum, seed,
                        ))
        for gamma in (0.7, 0.9):
            for stratum in STRATA:
                for seed in range(200):
                    expected.add((
                        "family_gradual_osa_boundary_completion",
                        policy, "osa", "latent", "start_low_expansion", 1.0,
                        gamma, stratum, seed,
                    ))
    return expected


def _validate_comparator_extension_payload(payload: Mapping[str, Any]) -> int:
    """Validate the exact separately executed 8,800-row extension envelope."""
    if payload.get("schema_version") != 1 or payload.get("status") != COMPARATOR_EXTENSION_STATUS:
        raise IntegrityError("comparator extension payload status/schema drifted")
    if payload.get("artifact_class") != COMPARATOR_EXTENSION_CLASS:
        raise IntegrityError("comparator extension artifact class drifted")
    projection = _require_mapping(
        payload.get("projection_contract"), "comparator extension projection contract"
    )
    if projection != {
        "analysis_scope": "separately authenticated exploratory extension",
        "extension_projected_into_formal_bundle": False,
        "immutable_formal_projection_changed": False,
        "immutable_formal_projection_rows": 55_480,
    }:
        raise IntegrityError("comparator extension was incorrectly merged into the formal projection")
    design = _require_mapping(payload.get("design"), "comparator extension design")
    if (
        design.get("expected_rows") != COMPARATOR_EXTENSION_ROWS
        or design.get("matrix_row_counts") != COMPARATOR_EXTENSION_MATRIX_COUNTS
        or design.get("canonical_sort_key") != list(COMPARATOR_EXTENSION_SORT_KEY)
    ):
        raise IntegrityError("comparator extension design/count declaration drifted")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != COMPARATOR_EXTENSION_ROWS:
        raise IntegrityError("comparator extension must contain exactly 8,800 rows")

    observed: list[tuple[Any, ...]] = []
    stopped = 0
    zero_audit_fields = (
        "eligible_gate_pass_set_difference_count",
        "gate_pass_set_difference_count",
        "operational_eligible_fallback_index_difference_count",
        "operational_full_fallback_index_difference_count",
    )
    for index, wrapper_value in enumerate(rows):
        wrapper = _require_mapping(wrapper_value, f"comparator extension row {index}")
        result = _require_mapping(wrapper.get("result"), f"comparator extension row {index}.result")
        key = (
            wrapper.get("matrix_id"), wrapper.get("policy"), result.get("sim"),
            result.get("mode"), result.get("protocol_scaffold"), result.get("kap"),
            result.get("gamma"), wrapper.get("stratum"), wrapper.get("seed"),
        )
        observed.append(key)
        if wrapper.get("run_class") != "current_harness_comparator":
            raise IntegrityError(f"comparator extension row {index}: run class drifted")
        if wrapper.get("harness_id") != "package-pf-stable-z-v2":
            raise IntegrityError(f"comparator extension row {index}: harness identity drifted")
        if wrapper.get("evaluator_id") != "registered-current-harness":
            raise IntegrityError(f"comparator extension row {index}: evaluator identity drifted")
        if wrapper.get("historical_wrapper_used") is not False:
            raise IntegrityError(f"comparator extension row {index}: historical wrapper was used")
        _finite(
            wrapper.get("elapsed_seconds_not_for_timing_inference"),
            f"comparator extension row {index}.elapsed", lower=0,
        )
        if result.get("policy") != wrapper.get("policy"):
            raise IntegrityError(f"comparator extension row {index}: policy differs across envelope")
        if result.get("seed") != wrapper.get("seed") or result.get("stratum") != wrapper.get("stratum"):
            raise IntegrityError(f"comparator extension row {index}: replicate identity drifted")
        common = {
            "mode": "latent", "noise": "fixed", "kap": 1.0, "budget": 40,
            "warmup": 4, "r_k": 2, "grid_n": 5, "empty_gate": "pf",
            "region_step": 0.25, "empty_gate_stop_after": 3,
            "exclude_repeats_during_expansion": True, "traj": None,
        }
        for field, expected in common.items():
            if result.get(field) != expected:
                raise IntegrityError(
                    f"comparator extension row {index}.{field}: {result.get(field)!r} != {expected!r}"
                )
        scaffold = result.get("protocol_scaffold")
        initialization_size = 4 if scaffold == "lhs_fixed" else 2
        if result.get("initialization_size") != initialization_size:
            raise IntegrityError(f"comparator extension row {index}: initialization size drifted")
        n_enrolled = _require_int(
            result.get("n_enrolled"), f"comparator extension row {index}.n_enrolled",
            lower=initialization_size, upper=40,
        )
        if n_enrolled % 2 or not isinstance(result.get("allocation_history"), list) or len(
            result["allocation_history"]
        ) != n_enrolled:
            raise IntegrityError(f"comparator extension row {index}: enrollment/history drifted")
        _require_int(result.get("toxic"), f"comparator extension row {index}.toxic", lower=0, upper=n_enrolled)
        if not isinstance(result.get("recommendation_made"), bool):
            raise IntegrityError(f"comparator extension row {index}: recommendation flag is not boolean")

        formal_key = [
            wrapper.get("matrix_id"), wrapper.get("policy"), wrapper.get("seed"),
            result.get("sim"), wrapper.get("stratum"), result.get("mode"),
            result.get("gamma"), result.get("kap"), scaffold, result.get("budget"),
            result.get("warmup"), result.get("r_k"), result.get("grid_n"),
            result.get("noise"), result.get("empty_gate"), result.get("region_step"),
            result.get("empty_gate_stop_after"),
            result.get("exclude_repeats_during_expansion"), False,
        ]
        if wrapper.get("formal_execution_key") != formal_key:
            raise IntegrityError(f"comparator extension row {index}: formal execution key drifted")

        audit = _require_mapping(
            wrapper.get("gate_numerics_audit"), f"comparator extension row {index}.audit"
        )
        if audit.get("terminal_state_checked") is not True:
            raise IntegrityError(f"comparator extension row {index}: terminal state was not audited")
        for field in zero_audit_fields:
            if audit.get(field) != 0:
                raise IntegrityError(f"comparator extension row {index}: stable-z audit differs at {field}")
        acquisition_states = _require_int(
            audit.get("acquisition_state_count"),
            f"comparator extension row {index}.acquisition_state_count", lower=0,
        )
        if audit.get("checked_state_count") != acquisition_states + 1:
            raise IntegrityError(f"comparator extension row {index}: terminal audit state is missing")

        if result["recommendation_made"]:
            if result.get("stop_reason") is not None:
                raise IntegrityError(f"comparator extension row {index}: completed row has stop reason")
            for field in COMPARATOR_TERMINAL_FIELDS:
                if field == "rec_unsafe":
                    _require_int(result.get(field), f"comparator extension row {index}.{field}", lower=0, upper=1)
                else:
                    _finite(result.get(field), f"comparator extension row {index}.{field}")
            if (
                audit.get("terminal_recommendation_status") != "checked"
                or audit.get("terminal_recommendation_index_difference_count") != 0
                or audit.get("terminal_recommendation_expected_index")
                != audit.get("terminal_recommendation_saved_index")
            ):
                raise IntegrityError(f"comparator extension row {index}: terminal recommendation audit failed")
        else:
            stopped += 1
            if scaffold != "start_low_expansion" or result.get("stop_reason") != "NO_FEASIBLE_DOSE":
                raise IntegrityError(f"comparator extension row {index}: invalid stopping envelope")
            if result.get("recs") != [] or any(result.get(field) is not None for field in COMPARATOR_TERMINAL_FIELDS):
                raise IntegrityError(f"comparator extension row {index}: stopped row contains terminal data")
            if (
                audit.get("terminal_recommendation_status") != "not_applicable_stopped"
                or any(
                    audit.get(field) is not None
                    for field in (
                        "terminal_recommendation_expected_index",
                        "terminal_recommendation_saved_index",
                        "terminal_recommendation_index_difference_count",
                    )
                )
            ):
                raise IntegrityError(f"comparator extension row {index}: stopped audit is not explicit")

    expected = _expected_comparator_extension_cells()
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise IntegrityError("comparator extension factorial, pairing, or replicate inventory drifted")
    if observed != sorted(observed):
        raise IntegrityError("comparator extension rows are not in canonical order")
    return stopped


def validate_comparator_family_extension(
    root: Path,
    manifest: Mapping[str, Any],
    projection_identity: Mapping[str, Any],
) -> int:
    declaration = _require_mapping(
        manifest.get("comparator_family_extension"), "comparator-family extension declaration"
    )
    if (
        declaration.get("status") != "SEPARATELY_AUTHENTICATED_EXPLORATORY_EXTENSION"
        or declaration.get("formal_projection_rows_unchanged") != 55_480
        or declaration.get("new_extension_executions") != COMPARATOR_EXTENSION_ROWS
        or declaration.get("selected_formal_rows_reused") != COMPARATOR_REUSED_ROWS
        or declaration.get("derived_family_rows") != COMPARATOR_FAMILY_ROWS
    ):
        raise IntegrityError("comparator-family accounting declaration drifted")
    raw_path = _safe_archive_path(root, declaration.get("path"), "comparator extension path")
    metadata_path = _safe_archive_path(
        root, declaration.get("record_metadata"), "comparator extension metadata path"
    )
    commit_path = _safe_archive_path(
        root, declaration.get("commit"), "comparator extension commit path"
    )
    metadata_raw = metadata_path.read_bytes()
    metadata = _require_mapping(_read_finite_json_bytes(metadata_raw, metadata_path.name), metadata_path.name)
    commit = _require_mapping(_read_json(commit_path), commit_path.name)
    if commit.get("status") != COMPARATOR_EXTENSION_COMMIT_STATUS:
        raise IntegrityError("comparator extension commit is not complete")
    expected_commit = {
        "artifact": raw_path.name,
        "artifact_sha256": file_sha256(raw_path),
        "metadata": metadata_path.name,
        "metadata_sha256": hashlib.sha256(metadata_raw).hexdigest(),
        "row_count": COMPARATOR_EXTENSION_ROWS,
        "immutable_formal_projection_changed": False,
    }
    for field, expected in expected_commit.items():
        if commit.get(field) != expected:
            raise IntegrityError(f"comparator extension commit mismatch at {field}")
    with tempfile.TemporaryDirectory(prefix="dose-combination-bo-extension-audit-") as temp:
        logical_path = Path(temp) / "extension.json"
        materialize_record(raw_path, logical_path)
        logical = logical_path.read_bytes()
    expected_metadata = {
        "status": COMPARATOR_EXTENSION_STATUS,
        "artifact_class": COMPARATOR_EXTENSION_CLASS,
        "artifact": raw_path.name,
        "artifact_sha256": file_sha256(raw_path),
        "artifact_bytes": raw_path.stat().st_size,
        "uncompressed_sha256": hashlib.sha256(logical).hexdigest(),
        "uncompressed_bytes": len(logical),
        "row_count": COMPARATOR_EXTENSION_ROWS,
        "immutable_formal_projection_changed": False,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise IntegrityError(f"comparator extension metadata mismatch at {field}")
    payload = _require_mapping(
        _read_finite_json_bytes(logical, raw_path.name), "comparator extension payload"
    )
    if logical != _canonical_pretty_json_bytes(payload):
        raise IntegrityError("comparator extension logical bytes are not canonical")
    if payload.get("bindings") != metadata.get("bindings"):
        raise IntegrityError("comparator extension bindings differ across envelopes")
    bindings = _require_mapping(payload.get("bindings"), "comparator extension bindings")
    if bindings.get("runtime_identity") != COMPARATOR_EXTENSION_RUNTIME:
        raise IntegrityError("comparator extension execution runtime identity drifted")
    if bindings.get("formal_manifest_sha256") != projection_identity.get("formal_manifest_sha256"):
        raise IntegrityError("comparator extension is not bound to the formal projection")
    source_hashes = _require_mapping(bindings.get("source_sha256"), "extension source hashes")
    for relative, digest in source_hashes.items():
        source = root / "source" / "package" / relative
        if not _valid_sha256(digest) or not source.is_file() or file_sha256(source) != digest:
            raise IntegrityError(f"comparator extension execution source differs: {relative}")
    spec = root / "source" / "package" / "paper" / "comparator_family_extension_prespec.md"
    completion_manifest = (
        root / "source" / "package" / "paper" / "comparator_family_completion_manifest.json"
    )
    if (
        not spec.is_file()
        or file_sha256(spec) != bindings.get("analysis_spec_sha256")
        or not completion_manifest.is_file()
        or file_sha256(completion_manifest) != bindings.get("completion_manifest_sha256")
    ):
        raise IntegrityError("comparator extension specification/manifest binding failed")
    return _validate_comparator_extension_payload(payload)


def validate_comparator_family_analysis(
    root: Path,
    projection_identity: Mapping[str, Any],
    extension_declaration: Mapping[str, Any],
) -> None:
    analysis_manifest, analysis_root = _manifested_analysis_tree(
        root,
        "comparator_family_analysis_manifest.json",
        "analysis/comparator_family",
        "results/comparator_family_analysis",
    )
    artifacts = [path for path in analysis_root.iterdir() if path.is_file() and not path.name.endswith(".metadata.json")]
    sidecars = list(analysis_root.glob("*.metadata.json"))
    if (
        analysis_manifest.get("artifact_count") != 24
        or analysis_manifest.get("metadata_sidecar_count") != 24
        or len(artifacts) != 24
        or len(sidecars) != 24
        or analysis_manifest.get("selected_formal_rows_reused") != COMPARATOR_REUSED_ROWS
        or analysis_manifest.get("new_extension_rows") != COMPARATOR_EXTENSION_ROWS
        or analysis_manifest.get("selected_family_rows") != COMPARATOR_FAMILY_ROWS
    ):
        raise IntegrityError("comparator-family analysis inventory/counts drifted")
    analyzer = root / "source" / "package" / "paper" / "analyze_comparator_family.py"
    if not analyzer.is_file():
        raise IntegrityError("comparator-family analyzer is absent from the source snapshot")
    selected_path = analysis_root / "comparator_family_selected_records.json.zst"
    selected_metadata = _require_mapping(
        _read_json(selected_path.with_suffix(selected_path.suffix + ".metadata.json")),
        "comparator-family selected-record metadata",
    )
    provenance = _require_mapping(selected_metadata.get("input_provenance"), "family provenance")
    formal = _require_mapping(provenance.get("formal_projection"), "family formal provenance")
    extension = _require_mapping(provenance.get("new_extension"), "family extension provenance")
    if (
        formal.get("selected_records") != COMPARATOR_REUSED_ROWS
        or formal.get("formal_manifest_sha256") != projection_identity.get("formal_manifest_sha256")
        or formal.get("projection_fingerprint") != projection_identity.get("projection_fingerprint")
        or formal.get("projection_metadata_sha256") != projection_identity.get("projection_metadata_sha256")
        or extension.get("row_count") != COMPARATOR_EXTENSION_ROWS
        or extension.get("artifact_sha256") != file_sha256(root / str(extension_declaration["path"]))
    ):
        raise IntegrityError("comparator-family input provenance/accounting drifted")
    commitment = _require_mapping(
        selected_metadata.get("selected_family_commitment"), "selected-family commitment"
    )
    if (
        commitment.get("rows") != COMPARATOR_FAMILY_ROWS
        or commitment.get("compressed_sha256") != file_sha256(selected_path)
        or commitment.get("compressed_bytes") != selected_path.stat().st_size
    ):
        raise IntegrityError("selected-family commitment drifted")
    with tempfile.TemporaryDirectory(prefix="dose-combination-bo-family-selected-") as temp:
        logical_path = Path(temp) / "selected.json"
        materialize_record(selected_path, logical_path)
        logical = logical_path.read_bytes()
    if (
        commitment.get("uncompressed_sha256") != hashlib.sha256(logical).hexdigest()
        or commitment.get("uncompressed_bytes") != len(logical)
    ):
        raise IntegrityError("selected-family logical commitment drifted")
    selected = _read_finite_json_bytes(logical, selected_path.name)
    if not isinstance(selected, list) or len(selected) != COMPARATOR_FAMILY_ROWS:
        raise IntegrityError("selected-family master must contain exactly 34,000 rows")
    if logical != _canonical_compact_json_bytes(selected):
        raise IntegrityError("selected-family master is not canonically serialized")
    origins = Counter(row.get("_source_origin") for row in selected if isinstance(row, dict))
    if origins != Counter({"reused_formal": COMPARATOR_REUSED_ROWS, "new_extension": COMPARATOR_EXTENSION_ROWS}):
        raise IntegrityError("selected-family reused/new accounting drifted")

    common_analysis: Mapping[str, Any] | None = None
    common_provenance: Mapping[str, Any] | None = None
    for artifact in artifacts:
        sidecar = artifact.with_suffix(artifact.suffix + ".metadata.json")
        metadata = _require_mapping(_read_json(sidecar), sidecar.name)
        if metadata.get("artifact") != artifact.name or metadata.get("artifact_sha256") != file_sha256(artifact):
            raise IntegrityError(f"{artifact.name}: family sidecar hash drifted")
        if metadata.get("analyzer") != "paper/analyze_comparator_family.py" or metadata.get(
            "analyzer_sha256"
        ) != file_sha256(analyzer):
            raise IntegrityError(f"{artifact.name}: family analyzer binding drifted")
        if metadata.get("selected_family_commitment") != commitment:
            raise IntegrityError(f"{artifact.name}: selected-family commitment differs across sidecars")
        if common_analysis is None:
            common_analysis = _require_mapping(metadata.get("analysis"), "family analysis scope")
            common_provenance = _require_mapping(metadata.get("input_provenance"), "family input provenance")
        elif metadata.get("analysis") != common_analysis or metadata.get("input_provenance") != common_provenance:
            raise IntegrityError(f"{artifact.name}: family sidecar scope/provenance differs")
    assert common_analysis is not None
    support = _require_mapping(common_analysis.get("support_summary"), "family support summary")
    if (
        common_analysis.get("principal_comparators") != ["tmse", "qBIG"]
        or common_analysis.get("secondary_comparators") != ["cEI", "cEI-tMSE"]
        or common_analysis.get("monte_carlo_symbol") != "M"
        or common_analysis.get("N_max") != 40
        or common_analysis.get("cell_vote_count_decision_rule") is not False
        or support.get("confirmatory_claim") is not False
        or support.get("binding_family_statement_permitted") is not False
        or support.get("interpretation_ladder_branch")
        != "boundary_comparators_must_remain_separate_or_contrast_specific"
    ):
        raise IntegrityError("comparator-family interpretation boundary drifted")


def _validate_qbig_audit_summary(audit: Mapping[str, Any]) -> None:
    if audit.get("schema_version") != 1 or audit.get("status") != QBIG_AUDIT_STATUS:
        raise IntegrityError("qBIG quadrature audit status/schema drifted")
    bounded = "decision-stable representative audit, not exact integration or universal guarantee"
    if audit.get("conclusion") != bounded or bounded not in str(audit.get("scope", "") + " " + bounded):
        raise IntegrityError("qBIG audit claim boundary drifted")
    summary = _require_mapping(audit.get("summary"), "qBIG audit summary")
    design = _require_mapping(summary.get("design"), "qBIG audit design")
    quadrature = _require_mapping(summary.get("quadrature"), "qBIG quadrature specification")
    states = audit.get("states")
    if not isinstance(states, list) or len(states) != 432:
        raise IntegrityError("qBIG audit must contain exactly 432 representative states")
    if (
        design.get("posterior_states") != 432
        or design.get("candidate_scores") != 10_800
        or design.get("candidates_per_state") != 25
        or design.get("new_monte_carlo_trials_generated") is not False
        or quadrature.get("frozen_method")
        != "9-node Gauss-Hermite expected finite-panel Bernoulli entropy reduction"
        or quadrature.get("terminology") != "entropy-based finite-panel SUR approximation"
        or quadrature.get("exact_integration_claim") is not False
    ):
        raise IntegrityError("qBIG quadrature design/terminology drifted")
    top = _require_mapping(summary.get("top_choice"), "qBIG top-choice summary")
    fallback = sum(bool(row.get("fallback_used")) for row in states if isinstance(row, dict))
    nonempty = len(states) - fallback
    operational_mismatch_fallback = sum(
        bool(row.get("operational_top_mismatch")) and bool(row.get("fallback_used"))
        for row in states if isinstance(row, dict)
    )
    operational_mismatch_nonempty = sum(
        bool(row.get("operational_top_mismatch")) and not bool(row.get("fallback_used"))
        for row in states if isinstance(row, dict)
    )
    raw_mismatch = sum(bool(row.get("raw_top_mismatch")) for row in states if isinstance(row, dict))
    if top != {
        **dict(top),
        "fallback_states": fallback,
        "nonempty_gate_states": nonempty,
        "operational_top_mismatches_fallback": operational_mismatch_fallback,
        "operational_top_mismatches_nonempty_gate": operational_mismatch_nonempty,
        "raw_top_mismatches": raw_mismatch,
    }:
        raise IntegrityError("qBIG top-choice summary is not derived from its state rows")
    if (
        top.get("archived_actions_checked") != 432
        or top.get("archived_action_reconstruction_failures") != 0
        or top.get("operational_top_mismatches_fallback") != 0
        or top.get("operational_top_mismatches_nonempty_gate") != 0
    ):
        raise IntegrityError("qBIG representative audit is not decision-stable")


def validate_qbig_quadrature_audit(root: Path, projection_identity: Mapping[str, Any]) -> None:
    audit_manifest, analysis_root = _manifested_analysis_tree(
        root,
        "qbig_quadrature_validation_manifest.json",
        "analysis/qbig_quadrature_validation",
        "results/qbig_quadrature_validation",
    )
    if audit_manifest.get("file_count") != 3 or {path.name for path in analysis_root.iterdir()} != {
        "qbig_quadrature_representative_audit.commit.json",
        "qbig_quadrature_representative_audit.json",
        "qbig_quadrature_representative_audit_states.csv",
    }:
        raise IntegrityError("qBIG quadrature audit file inventory drifted")
    commit = _require_mapping(
        _read_json(analysis_root / "qbig_quadrature_representative_audit.commit.json"),
        "qBIG quadrature commit",
    )
    if commit.get("status") != QBIG_AUDIT_COMMIT_STATUS:
        raise IntegrityError("qBIG quadrature audit commit is incomplete")
    artifacts = _require_mapping(commit.get("artifacts"), "qBIG committed artifacts")
    for name, item_value in artifacts.items():
        item = _require_mapping(item_value, f"qBIG committed artifact {name}")
        path = analysis_root / name
        if not path.is_file() or item.get("sha256") != file_sha256(path) or item.get("bytes") != path.stat().st_size:
            raise IntegrityError(f"qBIG committed artifact differs: {name}")
    audit = _require_mapping(
        _read_json(analysis_root / "qbig_quadrature_representative_audit.json"),
        "qBIG quadrature audit",
    )
    _validate_qbig_audit_summary(audit)
    rows = _read_csv(analysis_root / "qbig_quadrature_representative_audit_states.csv")
    if len(rows) != 432 or artifacts["qbig_quadrature_representative_audit_states.csv"].get(
        "rows_excluding_header"
    ) != 432:
        raise IntegrityError("qBIG state CSV row count drifted")
    states = audit["states"]
    state_keys = {
        (str(row["surface"]), str(row["gamma"]), str(row["stratum"]), str(row["seed"]), str(row["enrollment_n"]))
        for row in states
    }
    csv_keys = {
        (row["surface"], row["gamma"], row["stratum"], row["seed"], row["enrollment_n"])
        for row in rows
    }
    if state_keys != csv_keys or len(state_keys) != 432:
        raise IntegrityError("qBIG JSON/CSV representative-state inventory differs")
    provenance = _require_mapping(audit.get("provenance"), "qBIG audit provenance")
    if (
        provenance.get("formal_manifest_sha256") != projection_identity.get("formal_manifest_sha256")
        or provenance.get("projection_fingerprint") != projection_identity.get("projection_fingerprint")
        or provenance.get("local_formal_sources_verified") is not True
    ):
        raise IntegrityError("qBIG audit formal-projection provenance drifted")
    validator = root / "source" / "package" / "paper" / "validate_qbig_quadrature.py"
    if not validator.is_file() or provenance.get("validator_source_sha256") != file_sha256(validator):
        raise IntegrityError("qBIG audit validator source binding drifted")
    formal_raw = root / "analysis" / "formal_execution_provenance" / str(provenance.get("formal_raw"))
    if not formal_raw.is_file() or provenance.get("formal_raw_compressed_sha256") != file_sha256(formal_raw):
        raise IntegrityError("qBIG audit formal raw input binding drifted")


def validate_current_analysis_inventory(root: Path) -> None:
    inventory = _require_mapping(
        _read_json(root / "metadata" / "current_analysis_inventory.json"),
        "current analysis inventory",
    )
    trees = _require_mapping(inventory.get("trees"), "current analysis trees")
    inventoried: set[str] = set()
    for name, tree_value in trees.items():
        tree = _require_mapping(tree_value, f"analysis tree {name}")
        files = _require_mapping(tree.get("files"), f"analysis tree {name}.files")
        if tree.get("file_count") != len(files):
            raise IntegrityError(f"analysis tree {name}: declared file count drifted")
        for relative, digest in files.items():
            if relative in inventoried:
                raise IntegrityError(f"analysis inventory duplicates {relative}")
            path = root / relative
            if not _valid_sha256(digest) or not path.is_file() or file_sha256(path) != digest:
                raise IntegrityError(f"analysis inventory hash/path differs: {relative}")
            inventoried.add(relative)
    actual = {
        str(path.relative_to(root))
        for path in (root / "analysis").rglob("*")
        if path.is_file()
    }
    if inventoried != actual:
        raise IntegrityError("current analysis inventory does not cover the exact analysis tree")


def validate_source_snapshot(root: Path, manifest: Mapping[str, Any]) -> dict[str, int]:
    """Authenticate the exact source inventory and derive, rather than assume, counts."""
    source_manifest = _require_mapping(
        _read_json(root / "metadata" / "source_snapshot_manifest.json"),
        "source snapshot manifest",
    )
    source_files = _require_mapping(source_manifest.get("files"), "source snapshot files")
    actual_source = {
        str(path.relative_to(root))
        for source_root in (root / "source" / "package", root / "source" / "manuscript")
        for path in source_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if actual_source != set(source_files):
        raise IntegrityError("source snapshot manifest file inventory mismatch")
    if source_manifest.get("file_count") != len(source_files):
        raise IntegrityError("source snapshot manifest file_count differs from its inventory")
    for relative, digest in source_files.items():
        if not _valid_sha256(digest) or file_sha256(root / relative) != digest:
            raise IntegrityError(f"{relative}: source snapshot SHA-256 mismatch")
    source_declaration = _require_mapping(
        manifest.get("source_snapshot"), "source snapshot declaration"
    )
    package_count = sum(relative.startswith("source/package/") for relative in source_files)
    manuscript_count = sum(relative.startswith("source/manuscript/") for relative in source_files)
    if source_declaration.get("file_count") != len(source_files) or source_declaration.get(
        "package_file_count"
    ) != package_count or source_declaration.get("manuscript_file_count") != manuscript_count:
        raise IntegrityError("design-manifest source snapshot counts differ from the actual inventory")
    return {
        "file_count": len(source_files),
        "package_file_count": package_count,
        "manuscript_file_count": manuscript_count,
    }


def validate_metadata(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    inventory = _require_mapping(manifest.get("public_record_inventory"), "record inventory")
    expected_total = sum(
        _require_int(inventory.get(field), f"record inventory.{field}", lower=0)
        for field in (
            "main_records",
            "supplemental_records",
            "protocol_scaffold_sensitivity_records",
            "schedule_ratio_sensitivity_records",
        )
    )
    if expected_total != inventory.get("total_records") or expected_total != 55480:
        raise IntegrityError("public record inventory must sum to 55,480")
    main_design = _require_mapping(manifest.get("main_design"), "main design")
    if main_design.get("total_records") != inventory.get("main_records"):
        raise IntegrityError("main design and public inventory totals disagree")
    supplemental_design = _require_mapping(
        manifest.get("supplemental_design"), "supplemental design"
    )
    if supplemental_design.get("total_records") != inventory.get("supplemental_records"):
        raise IntegrityError("supplemental design and public inventory totals disagree")
    sensitivity = _require_mapping(
        manifest.get("protocol_scaffold_sensitivity"), "protocol sensitivity metadata"
    )
    if sensitivity.get("rows") != inventory.get("protocol_scaffold_sensitivity_records"):
        raise IntegrityError("sensitivity metadata and public inventory totals disagree")
    if sensitivity.get("factorial_arms") != 24 or sensitivity.get("seeds_per_arm") != 200:
        raise IntegrityError("sensitivity metadata does not describe 24 arms x 200 seeds")
    frozen = sensitivity.get("frozen_design")
    expected_frozen = {
        "protocol_scaffold": list(SCAFFOLDS),
        "gamma": list(GAMMAS),
        "stratum": list(STRATA),
        "policy": ["cEI", "cEI-tMSE", "cKG"],
        "seeds": "0..199",
        **FROZEN_SENSITIVITY,
    }
    if frozen != expected_frozen:
        raise IntegrityError("sensitivity frozen-design metadata drifted")
    metadata_path = sensitivity.get("record_metadata")
    if not isinstance(metadata_path, str):
        raise IntegrityError("sensitivity record-metadata path is missing")
    record_metadata = _require_mapping(
        _read_json(root / metadata_path), "sensitivity record metadata"
    )
    if record_metadata.get("record_count") != 4800:
        raise IntegrityError("sensitivity record metadata has the wrong record count")
    if record_metadata.get("artifact_type") != "raw_trial_records" or record_metadata.get(
        "schema_version"
    ) != 1:
        raise IntegrityError("sensitivity record metadata has an unsupported type/schema")
    if record_metadata.get("seed_count") != 200 or record_metadata.get("seeds") != list(SEEDS):
        raise IntegrityError("sensitivity record metadata has the wrong seed inventory")
    if record_metadata.get("source_sha256") != sensitivity.get("uncompressed_sha256"):
        raise IntegrityError("sensitivity record metadata has the wrong source SHA-256")
    if record_metadata.get("source_path") != Path(str(sensitivity.get("path"))).name:
        raise IntegrityError("sensitivity record metadata has a non-portable source path")
    expected_design = {
        "budget": [40],
        "empty_gate": ["pf"],
        "empty_gate_stop_after": [3],
        "exclude_repeats_during_expansion": [True],
        "gamma": [0.7, 0.9],
        "grid_n": [5],
        "kap": [1.0],
        "mode": ["latent"],
        "noise": ["fixed"],
        "policy": ["cEI", "cEI-tMSE", "cKG"],
        "protocol_scaffold": ["lhs_fixed", "start_low_expansion"],
        "r_k": [2],
        "region_step": [0.25],
        "sim": ["osa"],
        "stratum": [0, 1],
        "warmup": [4],
    }
    if record_metadata.get("design") != expected_design:
        raise IntegrityError("sensitivity record-metadata design drifted")
    schedule = _require_mapping(
        manifest.get("schedule_ratio_sensitivity"), "schedule-ratio sensitivity metadata"
    )
    if schedule.get("rows") != inventory.get("schedule_ratio_sensitivity_records"):
        raise IntegrityError("schedule metadata and public inventory totals disagree")
    if schedule.get("rows") != 200 or schedule.get("seeds_per_stratum") != 100:
        raise IntegrityError("schedule metadata does not describe 2 strata x 100 seeds")
    for field in ("compressed_sha256", "uncompressed_sha256"):
        if not _valid_sha256(schedule.get(field)):
            raise IntegrityError(f"schedule metadata has invalid {field}")
    for section_name in ("main_records", "supplemental_records"):
        section = _require_mapping(manifest.get(section_name), section_name)
        for label, value in section.items():
            item = _require_mapping(value, f"{section_name}.{label}")
            if not _valid_sha256(item.get("compressed_sha256")) or not _valid_sha256(
                item.get("uncompressed_sha256")
            ):
                raise IntegrityError(f"{section_name}.{label}: invalid SHA-256 metadata")

    status, doi = manifest.get("archive_status"), manifest.get("doi")
    if not isinstance(status, str) or not status:
        raise IntegrityError("archive_status metadata is missing")
    if not isinstance(doi, str) or not doi:
        raise IntegrityError("doi metadata is missing")
    if "STAGING" in status and doi != "DOI_PENDING":
        raise IntegrityError("an unpublished staging archive cannot claim a DOI")
    provenance = _require_mapping(
        manifest.get("environment_provenance"), "environment provenance"
    )
    for field, relative in provenance.items():
        if field in {"raw_full_freeze", "warning"}:
            continue
        if not isinstance(relative, str) or not (root / relative).is_file():
            raise IntegrityError(f"environment provenance path for {field!r} is missing")
    analysis_lock_path = _safe_archive_path(
        root,
        provenance.get("analysis_regeneration_lock"),
        "analysis-regeneration lock",
    )
    analysis_lock_lines = {
        line.strip()
        for line in analysis_lock_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing_analysis_pins = ANALYSIS_REGENERATION_LOCK_LINES - analysis_lock_lines
    if missing_analysis_pins:
        raise IntegrityError(
            "analysis-regeneration lock is missing exact pins: "
            + ", ".join(sorted(missing_analysis_pins))
        )
    runtime_path = _safe_archive_path(
        root,
        provenance.get("python_version_and_validation_runtime"),
        "analysis validation runtime",
    )
    runtime_lines = {
        line.strip()
        for line in runtime_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing_runtime_pins = ANALYSIS_REGENERATION_RUNTIME_LINES - runtime_lines
    if missing_runtime_pins:
        raise IntegrityError(
            "analysis validation runtime is missing exact pins: "
            + ", ".join(sorted(missing_runtime_pins))
        )

    validate_source_snapshot(root, manifest)

    projection_identity = validate_formal_projection_identity(root, manifest)
    stopped_extension_rows = validate_comparator_family_extension(
        root, manifest, projection_identity
    )
    if stopped_extension_rows < 1:
        raise IntegrityError("comparator-family extension unexpectedly contains no operational stops")
    extension_declaration = _require_mapping(
        manifest.get("comparator_family_extension"), "comparator-family extension declaration"
    )
    validate_comparator_family_analysis(root, projection_identity, extension_declaration)
    validate_qbig_quadrature_audit(root, projection_identity)
    projection_identity = dict(projection_identity)
    projection_identity["comparator_extension_feasibility_stops"] = stopped_extension_rows

    analysis_manifest = _require_mapping(
        _read_json(root / "metadata" / "protocol_scaffold_analysis_manifest.json"),
        "protocol analysis manifest",
    )
    analysis_files = _require_mapping(analysis_manifest.get("files"), "analysis files")
    actual_analysis = {
        str(path.relative_to(root))
        for path in (root / "analysis" / "protocol_scaffold").iterdir()
        if path.is_file()
    }
    if actual_analysis != set(analysis_files) or analysis_manifest.get("file_count") != len(
        analysis_files
    ):
        raise IntegrityError("protocol analysis manifest file inventory mismatch")
    if (
        analysis_manifest.get("source_record") != sensitivity.get("path")
        or analysis_manifest.get("source_uncompressed_sha256")
        != sensitivity.get("uncompressed_sha256")
    ):
        raise IntegrityError("protocol analysis manifest source commitment drifted")
    for relative, digest in analysis_files.items():
        if not _valid_sha256(digest) or file_sha256(root / relative) != digest:
            raise IntegrityError(f"{relative}: protocol analysis SHA-256 mismatch")
    analysis_root = root / "analysis" / "protocol_scaffold"
    artifacts = [
        path
        for path in analysis_root.iterdir()
        if path.is_file() and not path.name.endswith(".metadata.json")
    ]
    sidecars = list(analysis_root.glob("*.metadata.json"))
    if len(artifacts) != analysis_manifest.get("artifact_count") or len(
        sidecars
    ) != analysis_manifest.get("metadata_sidecar_count"):
        raise IntegrityError("protocol analysis artifact/sidecar counts disagree")
    protocol_selected_record_counts = {
        "protocol_scaffold_interactions.csv": 3_200,
        "protocol_scaffold_oc.csv": 4_800,
        "protocol_scaffold_oc_pooled.csv": 4_800,
        "protocol_scaffold_paired_grid.csv": 3_200,
        "protocol_scaffold_policy_card.html": 4_800,
        "protocol_scaffold_primary_secondary_contrasts.csv": 800,
        "protocol_scaffold_sensitivity.md": 4_800,
        "protocol_scaffold_sensitivity.eps": 3_200,
        "protocol_scaffold_sensitivity.pdf": 3_200,
        "protocol_scaffold_sensitivity.png": 3_200,
        "protocol_scaffold_supp_conditional.tex": 4_800,
        "protocol_scaffold_supp_contrasts.tex": 800,
        "protocol_scaffold_supp_interactions.tex": 3_200,
        "protocol_scaffold_supp_operations.tex": 4_800,
        "protocol_scaffold_supp_program.tex": 4_800,
        "protocol_scaffold_table_tau07.tex": 2_400,
        "protocol_scaffold_table_tau09.tex": 2_400,
    }
    if {artifact.name for artifact in artifacts} != set(
        protocol_selected_record_counts
    ):
        raise IntegrityError("protocol analysis artifact design inventory mismatch")
    for artifact in artifacts:
        sidecar = artifact.with_suffix(artifact.suffix + ".metadata.json")
        if not sidecar.is_file():
            raise IntegrityError(f"{artifact.name}: missing analysis metadata sidecar")
        metadata = _require_mapping(_read_json(sidecar), f"{sidecar.name} metadata")
        if metadata.get("source_sha256") != sensitivity.get("uncompressed_sha256"):
            raise IntegrityError(f"{sidecar.name}: wrong source SHA-256")
        if metadata.get("source_compressed_sha256") != sensitivity.get(
            "compressed_sha256"
        ):
            raise IntegrityError(f"{sidecar.name}: wrong compressed source SHA-256")
        if metadata.get("source_path") != Path(str(sensitivity.get("path"))).name:
            raise IntegrityError(f"{sidecar.name}: wrong portable source path")
        if metadata.get("schema_version") != 1:
            raise IntegrityError(f"{sidecar.name}: unsupported metadata schema")
        if metadata.get("seed_count") != 200 or metadata.get("seeds") != list(SEEDS):
            raise IntegrityError(f"{sidecar.name}: wrong seed inventory")
        if metadata.get("record_count") != protocol_selected_record_counts[
            artifact.name
        ]:
            raise IntegrityError(f"{sidecar.name}: wrong protocol record count")
        _projection_commitment(
            metadata.get("formal_projection"),
            projection_identity,
            f"{sidecar.name}.formal_projection",
        )

    validate_gate_profile_analysis(root, manifest, projection_identity)
    validate_cross_testbed_analysis(root, manifest, projection_identity)
    validate_computational_diagnostics(root)
    validate_current_analysis_inventory(root)

    script_provenance = _require_mapping(
        _read_json(root / "metadata" / "script_provenance.json"), "script provenance"
    )
    for group_name, group_value in script_provenance.items():
        if group_name == "note":
            continue
        group = _require_mapping(group_value, f"script provenance.{group_name}")
        for relative, item_value in group.items():
            item = _require_mapping(item_value, f"script provenance.{relative}")
            digest = item.get("sha256")
            if not _valid_sha256(digest) or file_sha256(root / relative) != digest:
                raise IntegrityError(f"{relative}: script-provenance SHA-256 mismatch")
    return projection_identity


def validate_archive(root: Path, *, check_sha256sums: bool = True) -> dict[str, int]:
    root = Path(root).resolve()
    manifest = _require_mapping(
        _read_json(root / "metadata" / "design_manifest.json"), "design manifest"
    )
    checksum_count = validate_checksum_manifest(root) if check_sha256sums else 0
    validate_public_payload_paths(root)
    projection_identity = validate_metadata(root, manifest)
    main_rows: list[Mapping[str, Any]] = []
    sixway_rows: list[Mapping[str, Any]] | None = None
    main_total = supplemental_total = sensitivity_total = schedule_total = stopped = 0
    with tempfile.TemporaryDirectory(prefix="dose-combination-bo-integrity-") as temp:
        work = Path(temp)
        main_section = _require_mapping(manifest.get("main_records"), "main records")
        seed_counts = {"osa": 200, "gbump": 200, "efftox": 100, "mariposa": 100}
        for sim in ("osa", "gbump", "efftox", "mariposa"):
            item = _require_mapping(main_section.get(sim), f"main records.{sim}")
            rows = _load_record_item(root, f"main_{sim}", item, work)
            validate_main_records(rows, sim, seed_counts[sim])
            main_rows.extend(rows)
            main_total += len(rows)
        supplemental_section = _require_mapping(
            manifest.get("supplemental_records"), "supplemental records"
        )
        for label in (
            "gate_sixway",
            "baselines_efftox_mariposa",
            "kappa_sweep_fixed",
            "traj_osa",
        ):
            item = _require_mapping(supplemental_section.get(label), f"supplemental.{label}")
            rows = _load_record_item(root, f"supplemental_{label}", item, work)
            validate_supplemental_records(label, rows, main_rows)
            if label == "gate_sixway":
                sixway_rows = rows
            supplemental_total += len(rows)
        sensitivity_item = _require_mapping(
            manifest.get("protocol_scaffold_sensitivity"), "protocol sensitivity"
        )
        sensitivity_rows = _load_record_item(
            root, "protocol_scaffold_sensitivity", sensitivity_item, work
        )
        stopped = validate_sensitivity_records(sensitivity_rows)
        sensitivity_total = len(sensitivity_rows)
        schedule_item = _require_mapping(
            manifest.get("schedule_ratio_sensitivity"), "schedule-ratio sensitivity"
        )
        schedule_rows = _load_record_item(
            root, "schedule_ratio_sensitivity", schedule_item, work
        )
        validate_schedule_ratio_records(schedule_rows)
        if sixway_rows is None:
            raise IntegrityError("schedule analysis requires the six-policy formal records")
        validate_schedule_ratio_analysis(
            root,
            manifest,
            projection_identity,
            sixway_rows,
            schedule_rows,
        )
        schedule_total = len(schedule_rows)
    total = main_total + supplemental_total + sensitivity_total + schedule_total
    inventory = manifest["public_record_inventory"]
    observed = (main_total, supplemental_total, sensitivity_total, schedule_total, total)
    expected = (
        inventory["main_records"],
        inventory["supplemental_records"],
        inventory["protocol_scaffold_sensitivity_records"],
        inventory["schedule_ratio_sensitivity_records"],
        inventory["total_records"],
    )
    if observed != expected:
        raise IntegrityError(f"record inventory mismatch: observed={observed}, expected={expected}")
    return {
        "checksum_files": checksum_count,
        "main_records": main_total,
        "supplemental_records": supplemental_total,
        "sensitivity_records": sensitivity_total,
        "schedule_ratio_records": schedule_total,
        "total_records": total,
        "feasibility_stops": stopped,
        "comparator_extension_records": COMPARATOR_EXTENSION_ROWS,
        "comparator_extension_feasibility_stops": projection_identity[
            "comparator_extension_feasibility_stops"
        ],
        "comparator_family_records": COMPARATOR_FAMILY_ROWS,
        "comparator_family_reused_records": COMPARATOR_REUSED_ROWS,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify archive hashes, record counts, frozen factorials, paired-seed "
            "initializations, cohort pairing, stop semantics, and metadata without "
            "loading torch, gpytorch, dose_combination_bo, or historical analysis modules."
        )
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        help="archive root; auto-detected when the script is inside the archive",
    )
    parser.add_argument(
        "--skip-sha256sums",
        action="store_true",
        help="skip the top-level SHA256SUMS inventory (record hashes are still checked)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = locate_archive_root(args.archive_root)
        counts = validate_archive(root, check_sha256sums=not args.skip_sha256sums)
    except (IntegrityError, OSError) as exc:
        print(f"Archive integrity validation FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "Archive integrity validation passed: "
        f"{counts['checksum_files']} checksummed files; "
        f"{counts['main_records']} main + {counts['supplemental_records']} supplemental + "
        f"{counts['sensitivity_records']} protocol sensitivity + "
        f"{counts['schedule_ratio_records']} schedule sensitivity = "
        f"{counts['total_records']} formal-projection logical rows; separately, "
        f"{counts['comparator_extension_records']} extension executions and "
        f"{counts['comparator_family_records']} selected family-analysis rows "
        f"({counts['comparator_family_reused_records']} reused + "
        f"{counts['comparator_extension_records']} new); "
        "saved computational executions are excluded from this inventory; "
        f"{counts['feasibility_stops']} formal-projection and "
        f"{counts['comparator_extension_feasibility_stops']} extension feasibility stops "
        "with no terminal recommendation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
