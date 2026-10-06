#!/usr/bin/env python3
"""Authenticate and analyze the frozen five-policy comparator-family extension.

This module is post-processing only.  It authenticates the immutable formal
projection and the separately committed 8,800-trial completion artifact, selects
the 25,200 prespecified reused trials, and constructs the exact 34,000-record
family analysis set. Its inferential unit is the matched Monte Carlo replicate
set (``M``). Full-grid simulations contain 40 response records, including four
numerical initialization records, and 36 post-initialization participant
assignments. The gradual design has a maximum enrollment of 40 participants.

The principal comparisons are cKG minus tMSE-only and cKG minus the
manuscript-defined ``finite-panel feasibility-entropy reduction`` comparator.
``qBIG`` remains the immutable internal policy identifier; reader-facing
artifacts label it ``Entropy reduction``.
cEI and the author-specified cEI--tMSE hybrid are retained only as secondary
comparators.  Aggregate decisions use the four frozen paired t intervals; no
cell vote count is computed or used as a decision rule.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats
import zstandard as zstd

import dose_combination_bo as sdb
from dose_combination_bo.surfaces import resolve_surface

try:
    from paper.analysis_record_bundle import ProjectionBundle, load_projection_bundle
except ModuleNotFoundError:  # Direct execution from paper/.
    from analysis_record_bundle import ProjectionBundle, load_projection_bundle


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "paper/comparator_family_extension_prespec.md"
MANIFEST_PATH = ROOT / "paper/comparator_family_completion_manifest.json"
ALLOCATION_CORRECTION_PATH = ROOT / "paper/allocation_estimand_correction.md"

SPEC_SHA256 = "117e8d1cd3221a54786075494c7b92503001dba90c72a22f4049d2fc807cccd8"
EXPECTED_REUSED = 25_200
EXPECTED_EXTENSION = 8_800
EXPECTED_FAMILY = 34_000
N_MAX = 40
FIGURE_DPI = 600
FIGURE_PAD_INCHES = 0.12
EXTENSION_MATRIX_COUNTS = {
    "family_full_osa_gbump_boundary_completion": 4_000,
    "family_full_efftox_mariposa_boundary_completion": 3_200,
    "family_gradual_osa_boundary_completion": 1_600,
}
EXTENSION_SORT_KEY = (
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

GATES = (0.5, 0.6, 0.7, 0.8, 0.9)
GRADUAL_GATES = (0.7, 0.9)
STRATA = (0, 1)
SURFACES = ("osa", "efftox", "mariposa", "gbump")
SURFACE_LABELS = {
    "osa": "OSA",
    "efftox": "Logistic",
    "mariposa": "MARIPOSA-motivated",
    "gbump": "Gaussian bump",
}
POLICIES = ("cKG", "tmse", "qBIG", "cEI", "cEI-tMSE")
BOUNDARY_COMPARATORS = ("tmse", "qBIG")
SECONDARY_COMPARATORS = ("cEI", "cEI-tMSE")
COMPARATORS = (*BOUNDARY_COMPARATORS, *SECONDARY_COMPARATORS)
POLICY_LABELS = {
    "cKG": "cKG",
    "tmse": "tMSE-only",
    "qBIG": "Entropy reduction",
    "cEI": "cEI",
    "cEI-tMSE": "cEI–tMSE hybrid",
}
POLICY_TEX = {
    "cKG": "cKG",
    "tmse": "tMSE-only",
    "qBIG": "Entropy reduction",
    "cEI": "cEI",
    "cEI-tMSE": "cEI--tMSE hybrid",
}
POLICY_DESCRIPTIONS = {
    "qBIG": "finite-panel feasibility-entropy reduction",
}

FORMAL_SOURCE_SPECS = (
    ("osa_main.json.zst", {"sim": {"osa"}, "policies": {"cEI", "cKG", "cEI-tMSE"}}),
    ("gbump_main.json.zst", {"sim": {"gbump"}, "policies": {"cEI", "cKG", "cEI-tMSE"}}),
    ("efftox_main.json.zst", {"sim": {"efftox"}, "policies": {"cEI", "cKG", "cEI-tMSE"}}),
    ("mariposa_main.json.zst", {"sim": {"mariposa"}, "policies": {"cEI", "cKG", "cEI-tMSE"}}),
    (
        "gate_sixway_supplemental.json.zst",
        {"sim": {"osa", "gbump"}, "policies": {"straddle", "qBIG"}},
    ),
    (
        "efftox_mariposa_baselines_supplemental.json.zst",
        {"sim": {"efftox", "mariposa"}, "policies": {"straddle", "qBIG"}},
    ),
    (
        "protocol_scaffold_sensitivity.json.zst",
        {
            "sim": {"osa"},
            "policies": {"cEI", "cKG", "cEI-tMSE"},
            "protocol": "start_low_expansion",
        },
    ),
)

EXTENSION_STATUS = "COMPLETE_COMPARATOR_FAMILY_RAW_EXTENSION"
EXTENSION_COMMIT_STATUS = "COMMITTED_COMPARATOR_FAMILY_RAW_EXTENSION"
EXTENSION_ARTIFACT_CLASS = (
    "authenticated_comparator_family_raw_extension_not_formal_projection"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ComparatorFamilyAnalysisError(RuntimeError):
    """Raised when authentication, selection, or an estimand invariant fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_pretty_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _canonical_compact_json(value: Any) -> bytes:
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


def _read_finite_json(raw: bytes, *, label: str) -> Any:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token}")

    try:
        return json.loads(raw, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ComparatorFamilyAnalysisError(f"{label} is not finite valid JSON") from exc


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ComparatorFamilyAnalysisError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_local_extension_bindings(bindings: Mapping[str, Any]) -> None:
    expected = {
        "completion_manifest_sha256": _sha256_file(MANIFEST_PATH),
        "analysis_spec_sha256": _sha256_file(SPEC_PATH),
    }
    if expected["analysis_spec_sha256"] != SPEC_SHA256:
        raise ComparatorFamilyAnalysisError("the frozen analysis specification changed")
    for field, value in expected.items():
        if bindings.get(field) != value:
            raise ComparatorFamilyAnalysisError(f"extension {field} binding changed")

    source_hashes = bindings.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ComparatorFamilyAnalysisError("extension source hash map is absent")
    for relative, expected_sha in sorted(source_hashes.items()):
        _require_sha(expected_sha, label=f"source_sha256.{relative}")
        source = ROOT / relative
        if not source.is_file() or _sha256_file(source) != expected_sha:
            raise ComparatorFamilyAnalysisError(
                f"extension-bound execution source is absent or changed: {relative}"
            )


def load_extension_records(
    raw_path: str | Path, *, verify_local_bindings: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate the committed completion artifact and return flattened results."""

    raw_path = Path(raw_path).resolve()
    metadata_path = Path(str(raw_path) + ".metadata.json")
    commit_path = Path(str(raw_path) + ".commit.json")
    if not all(path.is_file() for path in (raw_path, metadata_path, commit_path)):
        raise ComparatorFamilyAnalysisError(
            "extension input requires the raw zstd artifact, metadata, and commit marker"
        )

    compressed = raw_path.read_bytes()
    metadata_raw = metadata_path.read_bytes()
    metadata = _read_finite_json(metadata_raw, label=metadata_path.name)
    commit = _read_finite_json(commit_path.read_bytes(), label=commit_path.name)
    if not isinstance(metadata, dict) or not isinstance(commit, dict):
        raise ComparatorFamilyAnalysisError("extension metadata and commit must be objects")
    if commit.get("status") != EXTENSION_COMMIT_STATUS:
        raise ComparatorFamilyAnalysisError("extension commit status is not complete")
    expected_commit = {
        "artifact": raw_path.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256_bytes(metadata_raw),
        "row_count": EXPECTED_EXTENSION,
        "immutable_formal_projection_changed": False,
    }
    for field, expected in expected_commit.items():
        if commit.get(field) != expected:
            raise ComparatorFamilyAnalysisError(f"extension commit mismatch: {field}")

    try:
        uncompressed = zstd.ZstdDecompressor().decompress(compressed)
    except zstd.ZstdError as exc:
        raise ComparatorFamilyAnalysisError("extension raw input is not valid zstd") from exc
    payload = _read_finite_json(uncompressed, label=raw_path.name)
    if not isinstance(payload, dict):
        raise ComparatorFamilyAnalysisError("extension payload must be an object")
    expected_metadata = {
        "status": EXTENSION_STATUS,
        "artifact_class": EXTENSION_ARTIFACT_CLASS,
        "artifact": raw_path.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(uncompressed),
        "uncompressed_bytes": len(uncompressed),
        "row_count": EXPECTED_EXTENSION,
        "immutable_formal_projection_changed": False,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ComparatorFamilyAnalysisError(f"extension metadata mismatch: {field}")
    if uncompressed != _canonical_pretty_json(payload):
        raise ComparatorFamilyAnalysisError("extension logical bytes are not canonical")
    if payload.get("status") != EXTENSION_STATUS:
        raise ComparatorFamilyAnalysisError("extension payload status is not complete")
    if payload.get("artifact_class") != EXTENSION_ARTIFACT_CLASS:
        raise ComparatorFamilyAnalysisError("extension artifact class changed")
    if payload.get("schema_version") != 1 or metadata.get("schema_version") != 1 or commit.get(
        "schema_version"
    ) != 1:
        raise ComparatorFamilyAnalysisError("extension schema version changed")
    expected_design = {
        "expected_rows": EXPECTED_EXTENSION,
        "canonical_sort_key": list(EXTENSION_SORT_KEY),
        "matrix_row_counts": EXTENSION_MATRIX_COUNTS,
    }
    if payload.get("design") != expected_design:
        raise ComparatorFamilyAnalysisError("extension frozen design envelope changed")
    expected_projection_contract = {
        "immutable_formal_projection_rows": 55_480,
        "immutable_formal_projection_changed": False,
        "extension_projected_into_formal_bundle": False,
        "analysis_scope": "separately authenticated exploratory extension",
    }
    if payload.get("projection_contract") != expected_projection_contract:
        raise ComparatorFamilyAnalysisError("extension projection-separation contract changed")
    checkpoint_sha = payload.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_sha, dict)
        or len(checkpoint_sha) != 40
        or metadata.get("checkpoint_sha256") != checkpoint_sha
        or any(SHA256_RE.fullmatch(str(value)) is None for value in checkpoint_sha.values())
    ):
        raise ComparatorFamilyAnalysisError("extension checkpoint commitment set is invalid")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict) or metadata.get("bindings") != bindings:
        raise ComparatorFamilyAnalysisError("extension payload/metadata bindings differ")
    fingerprint_bindings = dict(bindings)
    launch = fingerprint_bindings.pop("launch_fingerprint", None)
    if _sha256_bytes(_canonical_pretty_json(fingerprint_bindings)) != launch:
        raise ComparatorFamilyAnalysisError("extension launch fingerprint is invalid")
    if commit.get("launch_fingerprint") != launch:
        raise ComparatorFamilyAnalysisError("extension commit launch fingerprint differs")
    if verify_local_bindings:
        _validate_local_extension_bindings(bindings)

    wrappers = payload.get("rows")
    if not isinstance(wrappers, list) or len(wrappers) != EXPECTED_EXTENSION:
        raise ComparatorFamilyAnalysisError("extension payload is not the 8,800-row master")
    rows: list[dict[str, Any]] = []
    for index, wrapper in enumerate(wrappers):
        if not isinstance(wrapper, dict) or not isinstance(wrapper.get("result"), dict):
            raise ComparatorFamilyAnalysisError(f"extension row {index} lacks result")
        if wrapper.get("run_class") != "current_harness_comparator":
            raise ComparatorFamilyAnalysisError(f"extension row {index} has wrong run class")
        result = dict(wrapper["result"])
        for field in ("policy", "seed", "stratum"):
            if result.get(field) != wrapper.get(field):
                raise ComparatorFamilyAnalysisError(
                    f"extension row {index} wrapper/result {field} differs"
                )
        if "kap" not in result:
            raise ComparatorFamilyAnalysisError(
                f"extension row {index} lacks required explicit kap"
            )
        try:
            extension_kap = float(result["kap"])
        except (TypeError, ValueError) as exc:
            raise ComparatorFamilyAnalysisError(
                f"extension row {index} has invalid explicit kap"
            ) from exc
        if (
            isinstance(result["kap"], bool)
            or not math.isfinite(extension_kap)
            or extension_kap != 1.0
        ):
            raise ComparatorFamilyAnalysisError(
                f"extension row {index} has explicit kap={result['kap']!r}; "
                "the frozen family requires 1.0"
            )
        audit = wrapper.get("gate_numerics_audit")
        if not isinstance(audit, dict):
            raise ComparatorFamilyAnalysisError(f"extension row {index} lacks gate audit")
        required_gate_zero = (
            "gate_pass_set_difference_count",
            "eligible_gate_pass_set_difference_count",
            "operational_full_fallback_index_difference_count",
            "operational_eligible_fallback_index_difference_count",
        )
        if any(audit.get(field) != 0 for field in required_gate_zero):
            raise ComparatorFamilyAnalysisError(
                f"extension row {index} did not pass the stable-z gate/fallback audit"
            )
        made = result.get("recommendation_made")
        stop_reason = result.get("stop_reason")
        terminal_status = audit.get("terminal_recommendation_status")
        terminal_difference = audit.get(
            "terminal_recommendation_index_difference_count"
        )
        terminal_expected = audit.get("terminal_recommendation_expected_index")
        terminal_saved = audit.get("terminal_recommendation_saved_index")
        terminal_state_checked = audit.get("terminal_state_checked")
        if type(made) is not bool:
            raise ComparatorFamilyAnalysisError(
                f"extension row {index} recommendation_made is not boolean"
            )
        if made:
            valid_terminal = (
                terminal_state_checked is True
                and stop_reason is None
                and terminal_status == "checked"
                and terminal_difference == 0
                and isinstance(terminal_expected, int)
                and not isinstance(terminal_expected, bool)
                and isinstance(terminal_saved, int)
                and not isinstance(terminal_saved, bool)
                and terminal_expected == terminal_saved
            )
        else:
            valid_terminal = (
                terminal_state_checked is True
                and stop_reason == "NO_FEASIBLE_DOSE"
                and terminal_status == "not_applicable_stopped"
                and terminal_difference is None
                and terminal_expected is None
                and terminal_saved is None
            )
        if not valid_terminal:
            raise ComparatorFamilyAnalysisError(
                f"extension row {index} has an inconsistent terminal-audit envelope"
            )
        result["_source_origin"] = "new_extension"
        result["_source_file"] = raw_path.name
        result["_matrix_id"] = wrapper.get("matrix_id")
        rows.append(result)

    _validate_extension_design(rows)
    provenance = {
        "artifact": raw_path.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256_bytes(metadata_raw),
        "commit": commit_path.name,
        "commit_sha256": _sha256_file(commit_path),
        "uncompressed_sha256": _sha256_bytes(uncompressed),
        "row_count": len(rows),
        "launch_fingerprint": launch,
        "bindings": bindings,
    }
    return rows, provenance


def _validate_extension_design(rows: Sequence[Mapping[str, Any]]) -> None:
    expected: set[tuple[Any, ...]] = set()
    for sim in ("osa", "gbump"):
        for policy in ("tmse", "qBIG"):
            for tau in GATES:
                for stratum in STRATA:
                    for seed in range(100, 200):
                        expected.add(("lhs_fixed", sim, tau, stratum, seed, policy))
    for sim in ("efftox", "mariposa"):
        for policy in ("tmse", "qBIG"):
            for tau in (0.5, 0.6, 0.8, 0.9):
                for stratum in STRATA:
                    for seed in range(100):
                        expected.add(("lhs_fixed", sim, tau, stratum, seed, policy))
    for policy in ("tmse", "qBIG"):
        for tau in GRADUAL_GATES:
            for stratum in STRATA:
                for seed in range(200):
                    expected.add(
                        ("start_low_expansion", "osa", tau, stratum, seed, policy)
                    )
    observed = {_design_key(row) for row in rows}
    if len(rows) != len(observed) or observed != expected:
        example = next(iter((expected - observed) or (observed - expected)), None)
        raise ComparatorFamilyAnalysisError(
            f"extension design is not the frozen 8,800-cell matrix; example={example!r}"
        )


def _normalize_policy(policy: Any) -> str:
    aliases = {
        "cKG-exact-formal": "cKG",
        "cKG1fix": "cKG",
        "straddle": "tmse",
        "GBE": "cEI-tMSE",
    }
    value = aliases.get(str(policy), str(policy))
    if value not in POLICIES:
        raise ComparatorFamilyAnalysisError(f"unplanned family policy {policy!r}")
    return value


def _protocol(row: Mapping[str, Any]) -> str:
    return str(row.get("protocol_scaffold", "lhs_fixed"))


def _design_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _protocol(row),
        str(row["sim"]),
        float(row["gamma"]),
        int(row["stratum"]),
        int(row["seed"]),
        _normalize_policy(row["policy"]),
    )


def _normalized_row(
    row: Mapping[str, Any], *, source_file: str, source_origin: str
) -> dict[str, Any]:
    item = dict(row)
    if source_origin == "reused_formal" and "kap" not in item:
        # The authenticated main formal records predate serialization of this
        # common fixed design field.  The formal manifest fixes kap=1.0, so the
        # omission is normalized only after bundle authentication and only for
        # reused-formal rows.  Extension rows never receive this default.
        item["kap"] = 1.0
    item["policy"] = _normalize_policy(item["policy"])
    item["protocol_scaffold"] = _protocol(item)
    item["_source_file"] = source_file
    item["_source_origin"] = source_origin
    return item


def select_formal_family_records(bundle: ProjectionBundle) -> list[dict[str, Any]]:
    """Select the frozen 25,200 reused records from authenticated formal inputs."""

    selected: list[dict[str, Any]] = []
    for filename, spec in FORMAL_SOURCE_SPECS:
        records = bundle.records.get(filename)
        if records is None:
            raise ComparatorFamilyAnalysisError(f"formal source is absent: {filename}")
        for row in records:
            if row.get("mode") != "latent":
                continue
            if row.get("sim") not in spec["sim"] or row.get("policy") not in spec["policies"]:
                continue
            protocol = _protocol(row)
            required_protocol = spec.get("protocol", "lhs_fixed")
            if protocol != required_protocol:
                continue
            if filename == "efftox_mariposa_baselines_supplemental.json.zst" and not math.isclose(
                float(row["gamma"]), 0.7
            ):
                continue
            if "kap" in row:
                try:
                    kap = float(row["kap"])
                except (TypeError, ValueError) as exc:
                    raise ComparatorFamilyAnalysisError(
                        f"target formal row in {filename} has invalid explicit kap"
                    ) from exc
                if not math.isfinite(kap) or kap != 1.0:
                    raise ComparatorFamilyAnalysisError(
                        f"target formal row in {filename} has explicit kap={row['kap']!r}; "
                        "the frozen family requires 1.0"
                    )
            selected.append(
                _normalized_row(row, source_file=filename, source_origin="reused_formal")
            )
    if len(selected) != EXPECTED_REUSED:
        raise ComparatorFamilyAnalysisError(
            f"formal selection has {len(selected):,} rows, expected {EXPECTED_REUSED:,}"
        )
    keys = [_design_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise ComparatorFamilyAnalysisError("formal family selection contains duplicates")
    return selected


def build_family_records(
    bundle: ProjectionBundle, extension_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    reused = select_formal_family_records(bundle)
    extension = [
        _normalized_row(
            row,
            source_file=str(row.get("_source_file", "comparator_family_completion_records.json.zst")),
            source_origin="new_extension",
        )
        for row in extension_rows
    ]
    if len(extension) != EXPECTED_EXTENSION:
        raise ComparatorFamilyAnalysisError("extension selection is not 8,800 records")
    records = reused + extension
    keys = [_design_key(row) for row in records]
    if len(records) != EXPECTED_FAMILY or len(keys) != len(set(keys)):
        raise ComparatorFamilyAnalysisError(
            "completed family is not 34,000 unique trial records"
        )
    _validate_completed_family(records)
    records.sort(key=_design_key)
    return records


def _validate_trial_fields(row: Mapping[str, Any], index: int) -> None:
    required = {
        "policy", "seed", "sim", "stratum", "gamma", "mode", "noise", "kap",
        "budget", "warmup", "r_k", "grid_n", "empty_gate", "toxic",
        "n_gate_pass", "n_gate_pass_safe", "obd_pf", "obd_sdg",
    }
    missing = required - set(row)
    if missing:
        raise ComparatorFamilyAnalysisError(f"family row {index} misses {sorted(missing)}")
    if row["mode"] != "latent" or row["noise"] != "fixed":
        raise ComparatorFamilyAnalysisError(f"family row {index} changes mode/noise")
    frozen = {"kap": 1.0, "budget": 40, "warmup": 4, "r_k": 2, "grid_n": 5, "empty_gate": "pf"}
    for field, expected in frozen.items():
        if row[field] != expected:
            raise ComparatorFamilyAnalysisError(
                f"family row {index} has {field}={row[field]!r}, expected {expected!r}"
            )
    toxic = int(row["toxic"])
    enrolled = int(row.get("n_enrolled", N_MAX))
    if toxic != row["toxic"] or not 0 <= toxic <= enrolled <= N_MAX:
        raise ComparatorFamilyAnalysisError(f"family row {index} has invalid enrollment/exposure")
    admitted = int(row["n_gate_pass"])
    safe = int(row["n_gate_pass_safe"])
    if not 0 <= safe <= admitted <= 25:
        raise ComparatorFamilyAnalysisError(f"family row {index} has invalid gate counts")
    made = row.get("recommendation_made", True)
    if type(made) is not bool:
        raise ComparatorFamilyAnalysisError(f"family row {index} recommendation flag is not bool")
    if made:
        for field in ("rec_unsafe", "rec_true_eff", "dose_units", "rec_d1", "rec_d2"):
            if row.get(field) is None:
                raise ComparatorFamilyAnalysisError(
                    f"family row {index} recommendation misses {field}"
                )
        if int(row["rec_unsafe"]) not in (0, 1):
            raise ComparatorFamilyAnalysisError(f"family row {index} has nonbinary rec_unsafe")
    elif row.get("stop_reason") != "NO_FEASIBLE_DOSE":
        raise ComparatorFamilyAnalysisError(
            f"family row {index} lacks the prespecified feasibility-stop reason"
        )


def _expected_family_keys() -> set[tuple[Any, ...]]:
    expected: set[tuple[Any, ...]] = set()
    for sim in ("osa", "gbump"):
        for tau in GATES:
            for stratum in STRATA:
                for seed in range(200):
                    for policy in POLICIES:
                        expected.add(("lhs_fixed", sim, tau, stratum, seed, policy))
    for sim in ("efftox", "mariposa"):
        for tau in GATES:
            for stratum in STRATA:
                for seed in range(100):
                    for policy in POLICIES:
                        expected.add(("lhs_fixed", sim, tau, stratum, seed, policy))
    for tau in GRADUAL_GATES:
        for stratum in STRATA:
            for seed in range(200):
                for policy in POLICIES:
                    expected.add(("start_low_expansion", "osa", tau, stratum, seed, policy))
    return expected


def _validate_completed_family(records: Sequence[Mapping[str, Any]]) -> None:
    for index, row in enumerate(records):
        _validate_trial_fields(row, index)
    observed = {_design_key(row) for row in records}
    expected = _expected_family_keys()
    if observed != expected or len(records) != len(expected):
        example = next(iter((expected - observed) or (observed - expected)), None)
        raise ComparatorFamilyAnalysisError(
            f"completed family factorial differs from the frozen design; example={example!r}"
        )
    origins = {origin: 0 for origin in ("reused_formal", "new_extension")}
    for row in records:
        origins[str(row["_source_origin"])] = origins.get(str(row["_source_origin"]), 0) + 1
    if origins != {"reused_formal": EXPECTED_REUSED, "new_extension": EXPECTED_EXTENSION}:
        raise ComparatorFamilyAnalysisError(f"family source counts changed: {origins}")


_TRUE_SAFE_CACHE: dict[tuple[str, int, int], int] = {}


def _true_safe_count(row: Mapping[str, Any]) -> int:
    key = (str(row["sim"]), int(row["stratum"]), int(row["grid_n"]))
    if key not in _TRUE_SAFE_CACHE:
        spec = resolve_surface(key[0], key[1])
        axis = np.linspace(0.0, 1.0, key[2])
        _TRUE_SAFE_CACHE[key] = sum(
            float(spec["tox"](float(d1), float(d2))) <= float(spec["gd"])
            for d1 in axis
            for d2 in axis
        )
    count = _TRUE_SAFE_CACHE[key]
    if count <= 0:
        raise ComparatorFamilyAnalysisError(f"surface {key[:2]} has no feasible panel dose")
    return count


def _made(row: Mapping[str, Any]) -> bool:
    return bool(row.get("recommendation_made", True))


_ANALYSIS_INITIALIZATION_ABOVE = "_analysis_initialization_patients_above"
_ANALYSIS_POST_INITIALIZATION_ABOVE = (
    "_analysis_post_initialization_patients_above"
)
_ANALYSIS_SURFACE_CACHE: dict[tuple[str, int], Mapping[str, Any]] = {}


def _enrollment(row: Mapping[str, Any]) -> int:
    value = int(row.get("n_enrolled", row.get("budget", N_MAX)))
    if value < 1:
        raise ComparatorFamilyAnalysisError("trial enrollment must be positive")
    return value


def _initialization_size(row: Mapping[str, Any]) -> int:
    value = int(row.get("initialization_size", row.get("warmup", 0)))
    enrolled = _enrollment(row)
    if not 0 <= value < enrolled:
        raise ComparatorFamilyAnalysisError(
            "initialization size must be smaller than trial enrollment"
        )
    return value


def _history_initialization_prefix(
    row: Mapping[str, Any], initialization_size: int
) -> tuple[tuple[float, float], ...] | None:
    history = row.get("allocation_history")
    if history is None:
        return None
    if not isinstance(history, Sequence) or len(history) < initialization_size:
        raise ComparatorFamilyAnalysisError(
            "allocation history is shorter than the initialization"
        )
    prefix: list[tuple[float, float]] = []
    for point in history[:initialization_size]:
        if not isinstance(point, Sequence) or len(point) != 2:
            raise ComparatorFamilyAnalysisError(
                "allocation history contains an invalid dose combination"
            )
        dose = (float(point[0]), float(point[1]))
        if not all(math.isfinite(value) for value in dose):
            raise ComparatorFamilyAnalysisError(
                "allocation history contains a non-finite dose combination"
            )
        prefix.append(dose)
    return tuple(prefix)


def _initialization_above_from_prefix(
    row: Mapping[str, Any], prefix: Sequence[tuple[float, float]]
) -> int:
    surface_key = (str(row["sim"]), int(row["stratum"]))
    if surface_key not in _ANALYSIS_SURFACE_CACHE:
        _ANALYSIS_SURFACE_CACHE[surface_key] = resolve_surface(*surface_key)
    surface = _ANALYSIS_SURFACE_CACHE[surface_key]
    threshold = float(surface["gd"])
    return int(
        sum(
            float(surface["tox"](float(dose[0]), float(dose[1]))) > threshold
            for dose in prefix
        )
    )


def _prepare_analysis_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach authenticated post-initialization counts for full-grid analysis.

    Older frozen rows often omit the count and their own allocation history. All
    full-grid policies and scenarios used the same seed-indexed four-point LHS
    initialization. We therefore reuse an observed prefix from any authenticated
    row with the same seed, after verifying that every available prefix agrees,
    and evaluate that prefix on the row's own toxicity surface. No dose history
    is reconstructed from a seed alone.
    """

    prefixes: dict[tuple[int, int], tuple[tuple[float, float], ...]] = {}
    for row in records:
        if _protocol(row) != "lhs_fixed":
            continue
        initialization_size = _initialization_size(row)
        prefix = _history_initialization_prefix(row, initialization_size)
        if prefix is None:
            continue
        key = (int(row["seed"]), initialization_size)
        previous = prefixes.setdefault(key, prefix)
        if previous != prefix:
            raise ComparatorFamilyAnalysisError(
                "authenticated full-grid initialization prefixes disagree for a seed"
            )

    prepared: list[dict[str, Any]] = []
    for source in records:
        row = dict(source)
        if _protocol(row) != "lhs_fixed":
            prepared.append(row)
            continue

        initialization_size = _initialization_size(row)
        enrolled = _enrollment(row)
        total_above = int(row["toxic"])
        prefix = _history_initialization_prefix(row, initialization_size)
        if prefix is None:
            prefix = prefixes.get((int(row["seed"]), initialization_size))

        candidates: list[int] = []
        if row.get("initialization_patients_above") is not None:
            candidates.append(int(row["initialization_patients_above"]))
        if row.get("post_initialization_patients_above") is not None:
            candidates.append(
                total_above - int(row["post_initialization_patients_above"])
            )
        if prefix is not None:
            candidates.append(_initialization_above_from_prefix(row, prefix))
        if not candidates:
            # Paired contrasts remain identifiable because the common
            # initialization cancels. Absolute full-grid allocation is not.
            prepared.append(row)
            continue
        if len(set(candidates)) != 1:
            raise ComparatorFamilyAnalysisError(
                "stored and allocation-history initialization counts disagree"
            )
        initialization_above = candidates[0]
        post_above = total_above - initialization_above
        if not 0 <= initialization_above <= initialization_size:
            raise ComparatorFamilyAnalysisError(
                "initialization above-threshold count is outside its valid range"
            )
        if not 0 <= post_above <= enrolled - initialization_size:
            raise ComparatorFamilyAnalysisError(
                "post-initialization above-threshold count is outside its valid range"
            )
        row[_ANALYSIS_INITIALIZATION_ABOVE] = initialization_above
        row[_ANALYSIS_POST_INITIALIZATION_ABOVE] = post_above
        prepared.append(row)
    return prepared


def _post_initialization_above(row: Mapping[str, Any]) -> int | None:
    value = row.get(_ANALYSIS_POST_INITIALIZATION_ABOVE)
    if value is None:
        value = row.get("post_initialization_patients_above")
    if value is None:
        initialization = row.get("initialization_patients_above")
        if initialization is not None:
            value = int(row["toxic"]) - int(initialization)
    if value is None:
        prefix = _history_initialization_prefix(row, _initialization_size(row))
        if prefix is not None:
            value = int(row["toxic"]) - _initialization_above_from_prefix(row, prefix)
    if value is None:
        return None
    result = int(value)
    if not 0 <= result <= _enrollment(row) - _initialization_size(row):
        raise ComparatorFamilyAnalysisError(
            "post-initialization above-threshold count is outside its valid range"
        )
    return result


def _metric(row: Mapping[str, Any], outcome: str) -> float | None:
    made = _made(row)
    if outcome == "terminal_recommendation_pct":
        return 100.0 * float(made and bool(row.get("rec_unsafe", 0)))
    if outcome == "assignment_pct_nmax":
        if _protocol(row) == "lhs_fixed":
            post_above = _post_initialization_above(row)
            if post_above is None:
                return None
            denominator = _enrollment(row) - _initialization_size(row)
            return 100.0 * float(post_above) / denominator
        # The gradual design starts with two real participants assigned at (0, 0),
        # so its existing initiated-trial N_max scale includes those assignments.
        return 100.0 * float(row["toxic"]) / N_MAX
    if outcome == "assignment_count":
        if _protocol(row) == "lhs_fixed":
            value = _post_initialization_above(row)
            return None if value is None else float(value)
        return float(row["toxic"])
    if outcome == "recommendation_pct":
        return 100.0 * float(made)
    if outcome == "stop_pct":
        return 100.0 * float(row.get("stop_reason") == "NO_FEASIBLE_DOSE")
    if outcome == "enrollment":
        return float(row.get("n_enrolled", N_MAX))
    if outcome == "recommended_efficacy":
        return float(row["rec_true_eff"]) if made else None
    if outcome == "dose_location_error":
        return float(row["dose_units"]) if made else None
    if outcome == "unsafe_given_recommendation_pct":
        return 100.0 * float(row["rec_unsafe"]) if made else None
    if outcome == "false_admissions":
        return float(row["n_gate_pass"]) - float(row["n_gate_pass_safe"])
    if outcome == "admitted_doses":
        return float(row["n_gate_pass"])
    if outcome == "feasible_recall_pct":
        return 100.0 * float(row["n_gate_pass_safe"]) / _true_safe_count(row)
    if outcome == "gate_precision_pct":
        return (
            100.0 * float(row["n_gate_pass_safe"]) / float(row["n_gate_pass"])
            if int(row["n_gate_pass"]) > 0
            else None
        )
    if outcome == "final_empty_gate_pct":
        return 100.0 * float(int(row["n_gate_pass"]) == 0)
    if outcome == "obd_feasibility_pct":
        value = row.get("obd_pf")
        return None if value is None else 100.0 * float(value)
    if outcome == "obd_posterior_sd":
        value = row.get("obd_sdg")
        return None if value is None else float(value)
    if outcome == "empty_gate_events":
        value = row.get("n_empty_gate_events")
        return None if value is None else float(value)
    raise KeyError(outcome)


def _paired_metric_difference(
    left: Mapping[str, Any], right: Mapping[str, Any], outcome: str
) -> float | None:
    """Return a paired difference, using cancellation when old rows lack counts."""

    if _protocol(left) != _protocol(right):
        raise ComparatorFamilyAnalysisError("paired rows use different protocols")
    if _protocol(left) == "lhs_fixed" and outcome in {
        "assignment_pct_nmax",
        "assignment_count",
    }:
        left_enrollment = _enrollment(left)
        right_enrollment = _enrollment(right)
        left_initialization = _initialization_size(left)
        right_initialization = _initialization_size(right)
        if (
            left_enrollment != right_enrollment
            or left_initialization != right_initialization
            or str(left["sim"]) != str(right["sim"])
            or int(left["stratum"]) != int(right["stratum"])
            or int(left["seed"]) != int(right["seed"])
        ):
            raise ComparatorFamilyAnalysisError(
                "full-grid allocation contrast is not paired on initialization"
            )
        left_post = _post_initialization_above(left)
        right_post = _post_initialization_above(right)
        if left_post is not None and right_post is not None:
            difference = float(left_post - right_post)
        else:
            # Under the validated design identity, the seed-indexed initialization
            # is common to both policies, hence it cancels exactly from total counts.
            difference = float(left["toxic"]) - float(right["toxic"])
        if outcome == "assignment_pct_nmax":
            difference *= 100.0 / (left_enrollment - left_initialization)
        return difference
    left_value = _metric(left, outcome)
    right_value = _metric(right, outcome)
    if left_value is None or right_value is None:
        return None
    return float(left_value) - float(right_value)


def _t_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ComparatorFamilyAnalysisError("paired t summary requires at least two finite values")
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / math.sqrt(array.size))
    critical = float(stats.t.ppf(0.975, df=array.size - 1))
    return {
        "estimate": mean,
        "mcse": se,
        "t_critical_95": critical,
        "ci95_low": mean - critical * se,
        "ci95_high": mean + critical * se,
        "monte_carlo_replicate_sets": int(array.size),
    }


def _index(records: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    output = {_design_key(row): row for row in records}
    if len(output) != len(records):
        raise ComparatorFamilyAnalysisError("analysis records contain duplicate design keys")
    return output


PRIMARY_OUTCOMES = (
    ("terminal_recommendation_pct", "True-boundary-exceeding final recommendation", "positive"),
    (
        "assignment_pct_nmax",
        "Post-initialization participant allocation above the true toxicity threshold",
        "negative",
    ),
)

AGGREGATE_DOMAINS = (
    ("osa_primary", "OSA primary", ("osa",), tuple(range(200))),
    (
        "binding_family",
        "Binding-boundary family",
        ("osa", "efftox", "mariposa"),
        tuple(range(100)),
    ),
    (
        "gaussian_nonbinding",
        "Multimodal nonbinding Gaussian-bump stress test",
        ("gbump",),
        tuple(range(200)),
    ),
)


def _paired_block_values(
    records: Sequence[Mapping[str, Any]],
    *,
    comparator: str,
    outcome: str,
    sims: Sequence[str],
    gates: Sequence[float],
    seeds: Sequence[int],
    protocol: str = "lhs_fixed",
    lookup: Mapping[tuple[Any, ...], Mapping[str, Any]] | None = None,
) -> tuple[list[float], int]:
    lookup = _index(records) if lookup is None else lookup
    values: list[float] = []
    eligible_pairs = 0
    for seed in seeds:
        differences: list[float] = []
        for sim in sims:
            for tau in gates:
                for stratum in STRATA:
                    key = (protocol, sim, float(tau), stratum, int(seed))
                    difference = _paired_metric_difference(
                        lookup[(*key, "cKG")], lookup[(*key, comparator)], outcome
                    )
                    if difference is None:
                        continue
                    differences.append(float(difference))
                    eligible_pairs += 1
        if differences:
            values.append(float(np.mean(differences)))
    return values, eligible_pairs


def aggregate_contrasts(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    full = [row for row in records if _protocol(row) == "lhs_fixed"]
    lookup = _index(full)
    for domain_id, domain_label, sims, seeds in AGGREGATE_DOMAINS:
        for comparator in BOUNDARY_COMPARATORS:
            for outcome, outcome_label, direction in PRIMARY_OUTCOMES:
                values, pairs = _paired_block_values(
                    full,
                    comparator=comparator,
                    outcome=outcome,
                    sims=sims,
                    gates=GATES,
                    seeds=seeds,
                    lookup=lookup,
                )
                summary = _t_summary(values)
                supported = (
                    summary["ci95_low"] > 0.0
                    if direction == "positive"
                    else summary["ci95_high"] < 0.0
                )
                output.append(
                    {
                        "domain_id": domain_id,
                        "domain": domain_label,
                        "surfaces": "+".join(sims),
                        "surface_weight": f"1/{len(sims)}",
                        "comparator": comparator,
                        "comparator_label": POLICY_LABELS[comparator],
                        "contrast": f"cKG minus {POLICY_LABELS[comparator]}",
                        "outcome_id": outcome,
                        "outcome": outcome_label,
                        "unit": "percentage points",
                        "prespecified_direction": direction,
                        **summary,
                        "eligible_trial_pairs": pairs,
                        "interval_supports_direction": bool(supported),
                    }
                )
    return output


def cellwise_primary_contrasts(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    full = [row for row in records if _protocol(row) == "lhs_fixed"]
    lookup = _index(full)
    seed_counts = {"osa": 200, "gbump": 200, "efftox": 100, "mariposa": 100}
    for sim in SURFACES:
        for tau in GATES:
            for comparator in COMPARATORS:
                for outcome, label, direction in PRIMARY_OUTCOMES:
                    values, pairs = _paired_block_values(
                        full,
                        comparator=comparator,
                        outcome=outcome,
                        sims=(sim,),
                        gates=(tau,),
                        seeds=tuple(range(seed_counts[sim])),
                        lookup=lookup,
                    )
                    summary = _t_summary(values)
                    output.append(
                        {
                            "surface_id": sim,
                            "surface": SURFACE_LABELS[sim],
                            "tau": tau,
                            "comparator": comparator,
                            "comparator_label": POLICY_LABELS[comparator],
                            "comparator_role": (
                                "principal_boundary"
                                if comparator in BOUNDARY_COMPARATORS
                                else "secondary"
                            ),
                            "contrast": f"cKG minus {POLICY_LABELS[comparator]}",
                            "outcome_id": outcome,
                            "outcome": label,
                            "unit": "percentage points",
                            "prespecified_direction": direction,
                            **summary,
                            "eligible_trial_pairs": pairs,
                            "pointwise_unadjusted": True,
                        }
                    )
    return output


def surface_aggregate_bivariate_contrasts(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Gate-integrate paired terminal/assignment vectors within each surface.

    The 95% ellipse is the standard Hotelling-T-squared confidence region for
    the two-dimensional matched-replicate mean.  It is a descriptive display,
    not the frozen four-interval decision rule.
    """

    full = [row for row in records if _protocol(row) == "lhs_fixed"]
    lookup = _index(full)
    M_by_surface = {"osa": 200, "gbump": 200, "efftox": 100, "mariposa": 100}
    output: list[dict[str, Any]] = []
    for sim in SURFACES:
        M = M_by_surface[sim]
        for comparator in BOUNDARY_COMPARATORS:
            vectors: list[tuple[float, float]] = []
            for seed in range(M):
                assignment_differences: list[float] = []
                terminal_differences: list[float] = []
                for tau in GATES:
                    for stratum in STRATA:
                        base = ("lhs_fixed", sim, tau, stratum, seed)
                        ckg = lookup[(*base, "cKG")]
                        other = lookup[(*base, comparator)]
                        assignment_differences.append(float(
                            _paired_metric_difference(
                                ckg, other, "assignment_pct_nmax"
                            )
                        ))
                        terminal_differences.append(
                            float(_metric(ckg, "terminal_recommendation_pct"))
                            - float(_metric(other, "terminal_recommendation_pct"))
                        )
                vectors.append(
                    (
                        float(np.mean(assignment_differences)),
                        float(np.mean(terminal_differences)),
                    )
                )
            array = np.asarray(vectors, dtype=float)
            if array.shape != (M, 2) or not np.isfinite(array).all():
                raise ComparatorFamilyAnalysisError("invalid surface bivariate vectors")
            mean = np.mean(array, axis=0)
            covariance = np.cov(array, rowvar=False, ddof=1)
            covariance_mean = covariance / M
            marginal_sd = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            correlation = (
                float(covariance[0, 1] / (marginal_sd[0] * marginal_sd[1]))
                if marginal_sd[0] > 0.0 and marginal_sd[1] > 0.0
                else None
            )
            marginal = [_t_summary(array[:, index]) for index in range(2)]
            p = 2
            critical = float(
                p * (M - 1) / (M - p) * stats.f.ppf(0.95, p, M - p)
            )
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_mean * critical)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            angle = float(
                np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            )
            output.append(
                {
                    "surface_id": sim,
                    "surface": SURFACE_LABELS[sim],
                    "comparator": comparator,
                    "comparator_label": POLICY_LABELS[comparator],
                    "assignment_estimate": float(mean[0]),
                    "assignment_mcse": marginal[0]["mcse"],
                    "assignment_ci95_low": marginal[0]["ci95_low"],
                    "assignment_ci95_high": marginal[0]["ci95_high"],
                    "terminal_estimate": float(mean[1]),
                    "terminal_mcse": marginal[1]["mcse"],
                    "terminal_ci95_low": marginal[1]["ci95_low"],
                    "terminal_ci95_high": marginal[1]["ci95_high"],
                    "paired_covariance_assignment_terminal": float(covariance[0, 1]),
                    "paired_correlation_assignment_terminal": correlation,
                    "hotelling_ellipse_confidence": 0.95,
                    "hotelling_ellipse_critical": critical,
                    "hotelling_ellipse_width": 2.0 * math.sqrt(float(eigenvalues[0])),
                    "hotelling_ellipse_height": 2.0 * math.sqrt(float(eigenvalues[1])),
                    "hotelling_ellipse_angle_degrees": angle,
                    "monte_carlo_replicate_sets": M,
                    "eligible_trial_pairs": M * len(GATES) * len(STRATA),
                    "display_role": "descriptive_not_decision_rule",
                }
            )
    if len(output) != 8:
        raise ComparatorFamilyAnalysisError("surface bivariate map must have eight points")
    return output


SECONDARY_OUTCOMES = (
    ("recommendation_pct", "Recommendation probability", "percentage points"),
    ("recommended_efficacy", "Recommended efficacy | recommend", "efficacy units"),
    ("dose_location_error", "Dose-location error | recommend", "dose-grid units"),
    ("false_admissions", "Final false admissions", "doses"),
    ("feasible_recall_pct", "Feasible-dose recall", "percentage points"),
    ("gate_precision_pct", "Admitted-dose precision | nonempty gate", "percentage points"),
    ("final_empty_gate_pct", "Final empty gate", "percentage points"),
    ("obd_feasibility_pct", "Feasibility probability at nearest OBD panel dose", "percentage points"),
    ("obd_posterior_sd", "Toxicity posterior SD at nearest OBD panel dose", "toxicity units"),
)

FULL_OC_OUTCOMES = (
    ("terminal_recommendation_pct", "True-boundary-exceeding final recommendation", "percentage points"),
    ("assignment_count", "Post-initialization participants assigned above the true toxicity threshold", "participants"),
    (
        "assignment_pct_nmax",
        "Post-initialization participant allocation above the true toxicity threshold",
        "percentage points",
    ),
    ("recommendation_pct", "Recommendation probability", "percentage points"),
    ("recommended_efficacy", "Recommended efficacy | recommend", "efficacy units"),
    ("dose_location_error", "Dose-location error | recommend", "dose-grid units"),
    ("admitted_doses", "Final admitted doses", "doses"),
    ("false_admissions", "Final false admissions", "doses"),
    ("feasible_recall_pct", "Feasible-dose recall", "percentage points"),
    ("gate_precision_pct", "Admitted-dose precision | nonempty gate", "percentage points"),
    ("final_empty_gate_pct", "Final empty gate", "percentage points"),
    ("obd_feasibility_pct", "Feasibility probability at nearest OBD panel dose", "percentage points"),
    ("obd_posterior_sd", "Toxicity posterior SD at nearest OBD panel dose", "toxicity units"),
)


def _full_policy_block_values(
    records: Sequence[Mapping[str, Any]],
    *,
    sim: str,
    tau: float,
    policy: str,
    outcome: str,
    M: int,
) -> tuple[list[float], int]:
    chosen = {
        (int(row["seed"]), int(row["stratum"])): row
        for row in records
        if _protocol(row) == "lhs_fixed"
        and row["sim"] == sim
        and row["policy"] == policy
        and float(row["gamma"]) == float(tau)
    }
    values: list[float] = []
    eligible = 0
    for seed in range(M):
        within: list[float] = []
        for stratum in STRATA:
            value = _metric(chosen[(seed, stratum)], outcome)
            if value is not None:
                within.append(float(value))
                eligible += 1
        if within:
            values.append(float(np.mean(within)))
    return values, eligible


def full_panel_operating_characteristics(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one absolute five-policy OC dossier row per surface/gate/policy."""

    records = _prepare_analysis_records(records)
    output: list[dict[str, Any]] = []
    M_by_surface = {"osa": 200, "gbump": 200, "efftox": 100, "mariposa": 100}
    for sim in SURFACES:
        M = M_by_surface[sim]
        for tau in GATES:
            for policy in POLICIES:
                row: dict[str, Any] = {
                    "surface_id": sim,
                    "surface": SURFACE_LABELS[sim],
                    "tau": tau,
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "policy_role": (
                        "target"
                        if policy == "cKG"
                        else (
                            "principal_boundary"
                            if policy in BOUNDARY_COMPARATORS
                            else "secondary"
                        )
                    ),
                    "monte_carlo_replicate_sets_planned": M,
                    "trial_rows": 2 * M,
                    "N_max": N_MAX,
                    "post_initialization_enrollment": N_MAX - 4,
                }
                for outcome, _label, _unit in FULL_OC_OUTCOMES:
                    values, eligible = _full_policy_block_values(
                        records,
                        sim=sim,
                        tau=tau,
                        policy=policy,
                        outcome=outcome,
                        M=M,
                    )
                    summary = _t_summary(values)
                    row[f"{outcome}_estimate"] = summary["estimate"]
                    row[f"{outcome}_mcse"] = summary["mcse"]
                    row[f"{outcome}_ci95_low"] = summary["ci95_low"]
                    row[f"{outcome}_ci95_high"] = summary["ci95_high"]
                    row[f"{outcome}_eligible_trials"] = eligible
                    row[f"{outcome}_eligible_replicate_sets"] = summary[
                        "monte_carlo_replicate_sets"
                    ]
                output.append(row)
    if len(output) != 100:
        raise ComparatorFamilyAnalysisError("full-panel OC dossier must have 100 rows")
    return output


SECONDARY_AGGREGATE_OUTCOMES = (
    ("recommendation_pct", "Recommendation probability", "percentage points"),
    ("recommended_efficacy", "Recommended efficacy | recommend", "efficacy units"),
    ("dose_location_error", "Dose-location error | recommend", "dose-grid units"),
    ("admitted_doses", "Final admitted doses", "doses"),
    ("false_admissions", "Final false admissions", "doses"),
    ("feasible_recall_pct", "Feasible-dose recall", "percentage points"),
    (
        "gate_precision_pct",
        "Admitted-dose precision | both final gates nonempty",
        "percentage points",
    ),
    ("final_empty_gate_pct", "Final empty gate", "percentage points"),
    (
        "obd_feasibility_pct",
        "Feasibility probability at nearest OBD panel dose",
        "percentage points",
    ),
    (
        "obd_posterior_sd",
        "Toxicity posterior SD at nearest OBD panel dose",
        "toxicity units",
    ),
)

SECONDARY_MATCHED_ELIGIBILITY_RULE = (
    "within each replicate and surface, equally average stratum-by-gate cells "
    "where both policies have the outcome; require at least one eligible cell "
    "in every specified surface; then equally average surfaces"
)


def _secondary_domain_values(
    *,
    comparator: str,
    outcome: str,
    sims: Sequence[str],
    seeds: Sequence[int],
    lookup: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> tuple[list[float], list[int]]:
    """Return replicate contrasts with fixed surface-equal eligible weighting."""

    values: list[float] = []
    pair_counts: list[int] = []
    for seed in seeds:
        surface_means: list[float] = []
        replicate_pairs = 0
        for sim in sims:
            differences: list[float] = []
            for tau in GATES:
                for stratum in STRATA:
                    key = ("lhs_fixed", sim, float(tau), stratum, int(seed))
                    left = _metric(lookup[(*key, "cKG")], outcome)
                    right = _metric(lookup[(*key, comparator)], outcome)
                    if left is None or right is None:
                        continue
                    differences.append(float(left) - float(right))
            if not differences:
                surface_means = []
                break
            surface_means.append(float(np.mean(differences)))
            replicate_pairs += len(differences)
        if len(surface_means) == len(sims):
            values.append(float(np.mean(surface_means)))
            pair_counts.append(replicate_pairs)
    return values, pair_counts


def secondary_aggregate_contrasts(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return prespecified-domain secondary aggregates for both comparators.

    These intervals are descriptive mechanism diagnostics, never inputs to the
    frozen primary interpretation ladder.  Conditional metrics use the declared
    matched-eligible rule while preserving equal surface weights.
    """

    full = [row for row in records if _protocol(row) == "lhs_fixed"]
    lookup = _index(full)
    output: list[dict[str, Any]] = []
    for domain_id, domain_label, sims, seeds in AGGREGATE_DOMAINS:
        planned_pairs = len(seeds) * len(sims) * len(GATES) * len(STRATA)
        for comparator in BOUNDARY_COMPARATORS:
            for outcome, outcome_label, unit in SECONDARY_AGGREGATE_OUTCOMES:
                values, pair_counts = _secondary_domain_values(
                    comparator=comparator,
                    outcome=outcome,
                    sims=sims,
                    seeds=seeds,
                    lookup=lookup,
                )
                summary = _t_summary(values)
                output.append(
                    {
                        "domain_id": domain_id,
                        "domain": domain_label,
                        "surfaces": "+".join(sims),
                        "surface_weight": f"1/{len(sims)}",
                        "comparator": comparator,
                        "comparator_label": POLICY_LABELS[comparator],
                        "contrast": f"cKG minus {POLICY_LABELS[comparator]}",
                        "outcome_id": outcome,
                        "outcome": outcome_label,
                        "unit": unit,
                        **summary,
                        "planned_replicate_sets": len(seeds),
                        "planned_trial_pairs": planned_pairs,
                        "eligible_trial_pairs": int(sum(pair_counts)),
                        "min_eligible_pairs_per_replicate": int(min(pair_counts)),
                        "max_eligible_pairs_per_replicate": int(max(pair_counts)),
                        "mean_eligible_pairs_per_replicate": float(
                            np.mean(pair_counts)
                        ),
                        "matched_eligibility_rule": SECONDARY_MATCHED_ELIGIBILITY_RULE,
                        "interval_role": "descriptive_secondary_not_decision_rule",
                    }
                )
    if len(output) != 60:
        raise ComparatorFamilyAnalysisError(
            "secondary aggregate dossier must contain exactly 60 rows"
        )
    return output


def secondary_cellwise_contrasts(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    full = [row for row in records if _protocol(row) == "lhs_fixed"]
    lookup = _index(full)
    seed_counts = {"osa": 200, "gbump": 200, "efftox": 100, "mariposa": 100}
    for sim in SURFACES:
        for tau in GATES:
            for comparator in COMPARATORS:
                for outcome, label, unit in SECONDARY_OUTCOMES:
                    values, pairs = _paired_block_values(
                        full,
                        comparator=comparator,
                        outcome=outcome,
                        sims=(sim,),
                        gates=(tau,),
                        seeds=tuple(range(seed_counts[sim])),
                        lookup=lookup,
                    )
                    summary = _t_summary(values)
                    output.append(
                        {
                            "surface_id": sim,
                            "surface": SURFACE_LABELS[sim],
                            "tau": tau,
                            "comparator": comparator,
                            "comparator_label": POLICY_LABELS[comparator],
                            "comparator_role": (
                                "principal_boundary"
                                if comparator in BOUNDARY_COMPARATORS
                                else "secondary"
                            ),
                            "contrast": f"cKG minus {POLICY_LABELS[comparator]}",
                            "outcome_id": outcome,
                            "outcome": label,
                            "unit": unit,
                            **summary,
                            "eligible_trial_pairs": pairs,
                            "pointwise_unadjusted": True,
                        }
                    )
    return output


GRADUAL_OUTCOMES = (
    ("terminal_recommendation_pct", "All-trial true-boundary-exceeding final recommendation", "percentage points"),
    ("recommendation_pct", "Recommendation probability", "percentage points"),
    ("stop_pct", "Feasibility-stop probability", "percentage points"),
    ("enrollment", "Observed enrollment", "patients"),
    ("assignment_count", "Above-boundary assignments per initiated trial", "assignments"),
    ("assignment_pct_nmax", "Above-boundary assignments / N_max", "percentage points"),
    ("recommended_efficacy", "Recommended efficacy | recommend", "efficacy units"),
    ("dose_location_error", "Dose-location error | recommend", "dose-grid units"),
    ("empty_gate_events", "Empty-gate/fallback events", "events"),
)


def _policy_block_values(
    records: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    outcome: str,
    tau: float,
) -> tuple[list[float], int]:
    chosen = {
        (int(row["seed"]), int(row["stratum"])): row
        for row in records
        if _protocol(row) == "start_low_expansion"
        and row["sim"] == "osa"
        and row["policy"] == policy
        and float(row["gamma"]) == float(tau)
    }
    values: list[float] = []
    eligible = 0
    for seed in range(200):
        within: list[float] = []
        for stratum in STRATA:
            value = _metric(chosen[(seed, stratum)], outcome)
            if value is not None:
                within.append(float(value))
                eligible += 1
        if within:
            values.append(float(np.mean(within)))
    return values, eligible


def _conditional_unsafe_ratio(
    records: Sequence[Mapping[str, Any]], *, policy: str, tau: float
) -> dict[str, Any]:
    """Return P(unsafe recommendation | recommendation) with cluster MCSE.

    The estimate is a ratio over the 2M stratum-specific trials, so multiplying
    it by the all-trial recommendation probability reproduces E(Y_T)
    algebraically.  Uncertainty uses the matched-replicate-set influence values
    ``(A_r - ratio B_r) / mean(B_r)`` rather than treating the stratum trials as
    independent.
    """

    chosen = {
        (int(row["seed"]), int(row["stratum"])): row
        for row in records
        if _protocol(row) == "start_low_expansion"
        and row["sim"] == "osa"
        and row["policy"] == policy
        and float(row["gamma"]) == float(tau)
    }
    unsafe_by_seed: list[float] = []
    rec_by_seed: list[float] = []
    for seed in range(200):
        seed_rows = [chosen[(seed, stratum)] for stratum in STRATA]
        rec = float(sum(_made(row) for row in seed_rows))
        unsafe = float(
            sum(_made(row) and bool(row.get("rec_unsafe", 0)) for row in seed_rows)
        )
        rec_by_seed.append(rec)
        unsafe_by_seed.append(unsafe)
    rec = np.asarray(rec_by_seed, dtype=float)
    unsafe = np.asarray(unsafe_by_seed, dtype=float)
    denominator = float(np.sum(rec))
    if denominator <= 0.0:
        raise ComparatorFamilyAnalysisError(
            "conditional unsafe probability is undefined without recommendations"
        )
    ratio = float(np.sum(unsafe) / denominator)
    mean_denominator = float(np.mean(rec))
    influence = (unsafe - ratio * rec) / mean_denominator
    mcse = float(np.std(influence, ddof=1) / math.sqrt(influence.size))
    critical = float(stats.t.ppf(0.975, df=influence.size - 1))
    return {
        "estimate": 100.0 * ratio,
        "cluster_influence_mcse": 100.0 * mcse,
        "ci95_low": 100.0 * (ratio - critical * mcse),
        "ci95_high": 100.0 * (ratio + critical * mcse),
        "recommendation_denominator": int(denominator),
        "initiated_trial_denominator": 2 * len(STRATA) * 100,
        "monte_carlo_replicate_sets": int(influence.size),
    }


def gradual_operating_characteristics(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for tau in GRADUAL_GATES:
        for policy in POLICIES:
            row: dict[str, Any] = {
                "tau": tau,
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "monte_carlo_replicate_sets_planned": 200,
                "initiated_trials": 400,
                "N_max": N_MAX,
            }
            for outcome, _label, _unit in GRADUAL_OUTCOMES:
                values, eligible = _policy_block_values(
                    records, policy=policy, outcome=outcome, tau=tau
                )
                summary = _t_summary(values)
                prefix = outcome
                row[f"{prefix}_estimate"] = summary["estimate"]
                row[f"{prefix}_mcse"] = summary["mcse"]
                row[f"{prefix}_ci95_low"] = summary["ci95_low"]
                row[f"{prefix}_ci95_high"] = summary["ci95_high"]
                row[f"{prefix}_eligible_trials"] = eligible
                row[f"{prefix}_eligible_replicate_sets"] = summary[
                    "monte_carlo_replicate_sets"
                ]
            unsafe = _conditional_unsafe_ratio(
                records,
                policy=policy,
                tau=tau,
            )
            row["unsafe_given_recommendation_pct_estimate"] = unsafe["estimate"]
            row["unsafe_given_recommendation_pct_cluster_influence_mcse"] = unsafe[
                "cluster_influence_mcse"
            ]
            row["unsafe_given_recommendation_pct_ci95_low"] = unsafe["ci95_low"]
            row["unsafe_given_recommendation_pct_ci95_high"] = unsafe["ci95_high"]
            row["unsafe_given_recommendation_pct_recommendation_denominator"] = unsafe[
                "recommendation_denominator"
            ]
            row["unsafe_given_recommendation_pct_replicate_sets"] = unsafe[
                "monte_carlo_replicate_sets"
            ]
            product = (
                row["recommendation_pct_estimate"]
                * row["unsafe_given_recommendation_pct_estimate"]
                / 100.0
            )
            row["terminal_factorization_product_pct"] = product
            row["terminal_factorization_minus_all_trial_pct"] = (
                product - row["terminal_recommendation_pct_estimate"]
            )
            if not math.isclose(
                row["terminal_factorization_minus_all_trial_pct"],
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ComparatorFamilyAnalysisError(
                    "P(recommend) x P(unsafe | recommend) does not reproduce E(Y_T)"
                )
            output.append(row)
    return output


def gradual_paired_contrasts(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    gradual = [row for row in records if _protocol(row) == "start_low_expansion"]
    lookup = _index(gradual)
    for tau in GRADUAL_GATES:
        for comparator in COMPARATORS:
            for outcome, label, unit in GRADUAL_OUTCOMES:
                values, pairs = _paired_block_values(
                    gradual,
                    comparator=comparator,
                    outcome=outcome,
                    sims=("osa",),
                    gates=(tau,),
                    seeds=tuple(range(200)),
                    protocol="start_low_expansion",
                    lookup=lookup,
                )
                output.append(
                    {
                        "tau": tau,
                        "comparator": comparator,
                        "comparator_label": POLICY_LABELS[comparator],
                        "comparator_role": (
                            "principal_boundary"
                            if comparator in BOUNDARY_COMPARATORS
                            else "secondary"
                        ),
                        "contrast": f"cKG minus {POLICY_LABELS[comparator]}",
                        "outcome_id": outcome,
                        "outcome": label,
                        "unit": unit,
                        **_t_summary(values),
                        "eligible_trial_pairs": pairs,
                        "pointwise_unadjusted": True,
                    }
                )
    return output


def _support_summary(aggregate_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def domain_support(domain: str) -> dict[str, Any]:
        rows = [row for row in aggregate_rows if row["domain_id"] == domain]
        if len(rows) != 4:
            raise ComparatorFamilyAnalysisError(f"{domain} does not have four frozen intervals")
        by_comparator = {
            comparator: all(
                bool(row["interval_supports_direction"])
                for row in rows
                if row["comparator"] == comparator
            )
            for comparator in BOUNDARY_COMPARATORS
        }
        return {
            "four_interval_conjunction": all(by_comparator.values()),
            "by_boundary_comparator": by_comparator,
            "decision_rule": (
                "all four prespecified paired 95% t intervals must lie strictly "
                "in their prespecified directions"
            ),
            "cell_vote_count_used": False,
        }

    osa = domain_support("osa_primary")
    binding = domain_support("binding_family")
    gaussian = domain_support("gaussian_nonbinding")
    family_permitted = bool(
        osa["four_interval_conjunction"] and binding["four_interval_conjunction"]
    )
    return {
        "osa_primary": osa,
        "binding_family": binding,
        "gaussian_nonbinding_stress_test": gaussian,
        "binding_family_statement_permitted": family_permitted,
        "interpretation_ladder_branch": (
            "binding_family_pattern_supported"
            if family_permitted
            else (
                "osa_only_objective_alignment_pattern"
                if osa["four_interval_conjunction"]
                else "boundary_comparators_must_remain_separate_or_contrast_specific"
            )
        ),
        "confirmatory_claim": False,
        "reason": (
            "The conjunction was frozen after partial archived pilot results were known; "
            "the family analysis is exploratory/descriptive."
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ComparatorFamilyAnalysisError(f"cannot write empty CSV {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_pretty_json(value))


def _write_selected_master(path: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    logical = _canonical_compact_json(list(records))
    compressed = zstd.ZstdCompressor(
        level=19,
        threads=0,
        write_checksum=True,
        write_content_size=True,
        write_dict_id=False,
    ).compress(logical)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return {
        "rows": len(records),
        "compressed_bytes": len(compressed),
        "compressed_sha256": _sha256_bytes(compressed),
        "uncompressed_bytes": len(logical),
        "uncompressed_sha256": _sha256_bytes(logical),
    }


def _fmt(value: float) -> str:
    """Format displayed estimates with conventional decimal half-up rounding."""

    rounded = Decimal(str(float(value))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    if rounded == 0:
        rounded = abs(rounded)
    return format(rounded, ".2f")


def _tex_aggregate_table(aggregate: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Scope & Comparator & Outcome & Estimate & MCSE & 95\% MC interval \\",
        r"\midrule",
    ]
    domain_tex = {
        "osa_primary": "OSA",
        "binding_family": "Equal-weight binding-surface aggregate",
        "gaussian_nonbinding": "Gaussian-bump stress test",
    }
    outcome_tex = {
        "terminal_recommendation_pct": (
            "True-boundary-exceeding recommendation (pp)"
        ),
        "assignment_pct_nmax": (
            r"$100N_{\rm above}^{\rm post}/(N_{\rm obs}-N_{\rm init})$ (pp)"
        ),
    }
    for row in aggregate:
        lines.append(
            f"{domain_tex[row['domain_id']]} & {POLICY_TEX[row['comparator']]} & "
            f"{outcome_tex[row['outcome_id']]} & {_fmt(row['estimate'])} & "
            f"{_fmt(row['mcse'])} & [{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}] "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _tex_gradual_oc(gradual: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{clrrrrrr}",
        r"\toprule",
        r"$\tau$ & Policy & Terminal exceed. (\%) & Recommend (\%) & Stop (\%) & $E(N_{\rm obs})$ & $E(C_A)$ & Assignment burden (\% of maximum enrollment) \\",
        r"\midrule",
    ]
    for row in gradual:
        lines.append(
            f"{float(row['tau']):.1f} & {POLICY_TEX[row['policy']]} & "
            f"{_fmt(row['terminal_recommendation_pct_estimate'])} & "
            f"{_fmt(row['recommendation_pct_estimate'])} & "
            f"{_fmt(row['stop_pct_estimate'])} & "
            f"{_fmt(row['enrollment_estimate'])} & "
            f"{_fmt(row['assignment_count_estimate'])} & "
            f"{_fmt(row['assignment_pct_nmax_estimate'])} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _tex_gradual_contrasts(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return compact principal-comparator gradual-access contrast panels."""

    outcome_tex = {
        "terminal_recommendation_pct": "Terminal boundary exceedance (pp)",
        "recommendation_pct": "Recommendation probability (pp)",
        "stop_pct": "Feasibility stop (pp)",
        "enrollment": "Enrollment (participants)",
        "assignment_count": r"$C_A$ (assignments)",
        "assignment_pct_nmax": r"$100C_A/N_{\max}$ (pp)",
        "recommended_efficacy": r"Efficacy$\mid$recommend",
        "dose_location_error": r"Location error$\mid$recommend",
        "empty_gate_events": "Empty-gate events",
    }
    lookup = {
        (float(row["tau"]), str(row["comparator"]), str(row["outcome_id"])): row
        for row in rows
        if row["comparator"] in BOUNDARY_COMPARATORS
    }
    expected = {
        (tau, comparator, outcome)
        for tau in GRADUAL_GATES
        for comparator in BOUNDARY_COMPARATORS
        for outcome, _label, _unit in GRADUAL_OUTCOMES
    }
    if set(lookup) != expected:
        raise ComparatorFamilyAnalysisError(
            "gradual contrast TeX lacks a principal comparator/gate/outcome cell"
        )
    lines: list[str] = []
    for comparator_index, comparator in enumerate(BOUNDARY_COMPARATORS):
        if comparator_index:
            lines.extend([r"\medskip", ""])
        lines.extend(
            [
                r"\begin{tabular}{@{}clrrr@{}}",
                rf"\multicolumn{{5}}{{l}}{{\textbf{{cKG minus {POLICY_TEX[comparator]}}}}} \\",
                r"\toprule",
                r"$\tau$ & Outcome & Estimate & 95\% MC interval & $n_{\rm pair}$ \\",
                r"\midrule",
            ]
        )
        for tau in GRADUAL_GATES:
            for outcome, _label, _unit in GRADUAL_OUTCOMES:
                row = lookup[(tau, comparator, outcome)]
                lines.append(
                    f"{tau:.1f} & {outcome_tex[outcome]} & {_fmt(row['estimate'])} & "
                    f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}] & "
                    f"{int(row['eligible_trial_pairs'])} " + r"\\"
                )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\smallskip", ""])
    return "\n".join(lines)


def _tex_tables(
    aggregate: Sequence[Mapping[str, Any]], gradual: Sequence[Mapping[str, Any]]
) -> str:
    """Return the two legacy-compatible tables as one deterministic fragment."""

    return _tex_aggregate_table(aggregate).rstrip() + "\n\n" + _tex_gradual_oc(
        gradual
    )


def _pm_tex(row: Mapping[str, Any], outcome: str) -> str:
    return (
        f"{_fmt(row[f'{outcome}_estimate'])} "
        + r"$\pm$"
        + f" {_fmt(row[f'{outcome}_mcse'])}"
    )


def _tex_full_panel_oc(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return surface-wise, publication-sized absolute-OC table panels.

    Panel A keeps terminal, assignment, and recommendation outcomes together and
    gives the all-trial and recommendation-conditional denominators.  Panel B
    contains the admitted-set and reference-dose posterior diagnostics, with
    separate nonempty-gate and available-OBD denominators.  Repeating the two
    compact panels by surface avoids the previous 14-column, 100-row table.
    """

    expected = {
        (sim, tau, policy)
        for sim in SURFACES
        for tau in GATES
        for policy in POLICIES
    }
    observed = {
        (str(row["surface_id"]), float(row["tau"]), str(row["policy"]))
        for row in rows
    }
    if len(rows) != 100 or observed != expected:
        raise ComparatorFamilyAnalysisError(
            "full-panel TeX requires exactly 100 surface/gate/policy rows"
        )

    by_key = {
        (str(row["surface_id"]), float(row["tau"]), str(row["policy"])): row
        for row in rows
    }
    lines: list[str] = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.0pt}",
    ]
    for surface_index, sim in enumerate(SURFACES):
        label = SURFACE_LABELS[sim]
        surface_rows = [
            by_key[(sim, tau, policy)] for tau in GATES for policy in POLICIES
        ]
        if surface_index:
            lines.extend([r"\bigskip", ""])
        lines.extend(
            [
                r"\begin{tabular}{@{}clrrrrrrr@{}}",
                rf"\multicolumn{{9}}{{l}}{{\textbf{{{label}: Panel A. Terminal, assignment, and recommendation outcomes}}}} \\",
                r"\toprule",
                r"$\tau$ & Policy & Terminal exceed. (\%) & Post-init. allocation (\%) & Recommend (\%) & Efficacy$\mid$rec. & Location$\mid$rec. & $n_{\rm all}$ & $n_{\rm rec}$ \\",
                r"\midrule",
            ]
        )
        for row in surface_rows:
            lines.append(
                f"{float(row['tau']):.1f} & {POLICY_TEX[row['policy']]} & "
                + " & ".join(
                    _pm_tex(row, outcome)
                    for outcome in (
                        "terminal_recommendation_pct",
                        "assignment_pct_nmax",
                        "recommendation_pct",
                        "recommended_efficacy",
                        "dose_location_error",
                    )
                )
                + f" & {int(row['trial_rows'])}"
                + f" & {int(row['recommended_efficacy_eligible_trials'])}"
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\smallskip", ""])

        lines.extend(
            [
                r"\begin{tabular}{@{}clrrrrrrrrr@{}}",
                rf"\multicolumn{{11}}{{l}}{{\textbf{{{label}: Panel B. Final-gate and reference-dose diagnostics}}}} \\",
                r"\toprule",
                r"$\tau$ & Policy & Admitted & False adm. & Recall (\%) & Precision (\%) & Empty (\%) & OBD feas. (\%) & OBD SD & $n_{\rm gate}$ & $n_{\rm OBD}$ \\",
                r"\midrule",
            ]
        )
        for row in surface_rows:
            obd_n = int(row["obd_posterior_sd_eligible_trials"])
            if int(row["obd_feasibility_pct_eligible_trials"]) != obd_n:
                raise ComparatorFamilyAnalysisError(
                    "OBD feasibility and posterior-SD eligible denominators differ"
                )
            lines.append(
                f"{float(row['tau']):.1f} & {POLICY_TEX[row['policy']]} & "
                + " & ".join(
                    _pm_tex(row, outcome)
                    for outcome in (
                        "admitted_doses",
                        "false_admissions",
                        "feasible_recall_pct",
                        "gate_precision_pct",
                        "final_empty_gate_pct",
                        "obd_feasibility_pct",
                        "obd_posterior_sd",
                    )
                )
                + f" & {int(row['gate_precision_pct_eligible_trials'])}"
                + f" & {obd_n}"
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    lines.append(r"\endgroup")
    return "\n".join(lines)


def _validate_no_partial_transparency(fig: Any) -> None:
    """Reject explicit partial artist alpha before writing PDF/EPS/PNG.

    Alpha 0 is permitted for an unfilled artist and alpha 1 is opaque.  Values
    strictly between them are intentionally forbidden because EPS cannot encode
    them consistently with PDF and PNG.
    """

    for artist in fig.findobj():
        if not artist.get_visible():
            continue
        getter = getattr(artist, "get_alpha", None)
        if getter is None:
            continue
        alpha = getter()
        if alpha is not None and 0.0 < float(alpha) < 1.0:
            raise ComparatorFamilyAnalysisError(
                f"partial transparency is not publication-safe: alpha={alpha}"
            )


def _save_figure(fig: Any, stem: Path, *, title: str, subject: str) -> tuple[Path, Path, Path]:
    _validate_no_partial_transparency(fig)
    pdf = stem.with_suffix(".pdf")
    eps = stem.with_suffix(".eps")
    png = stem.with_suffix(".png")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=FIGURE_PAD_INCHES,
        metadata={
            "Title": title,
            "Subject": subject,
            "Keywords": "dose optimization, comparator family, Monte Carlo",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    previous = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        fig.savefig(
            eps,
            format="eps",
            bbox_inches="tight",
            pad_inches=FIGURE_PAD_INCHES,
            metadata={"Creator": "dual-combo-BO comparator-family analyzer"},
        )
    finally:
        if previous is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = previous
    fig.savefig(
        png,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        pad_inches=FIGURE_PAD_INCHES,
        facecolor="white",
        metadata={"Title": title, "Description": subject},
    )
    return pdf, eps, png


def _plot_family_profile(
    rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    stem: Path,
) -> tuple[Path, Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = [
        row
        for row in rows
        if row["surface_id"] == "osa" and row["comparator"] in BOUNDARY_COMPARATORS
    ]
    styles = {
        "tmse": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
        "qBIG": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
    }
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=False)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.22, top=0.72, wspace=0.28)
    titles = {
        "terminal_recommendation_pct": "(a) Terminal recommendation",
        "assignment_pct_nmax": "(b) Trial assignments",
    }
    handles = []
    for axis, (outcome, _label, _direction) in zip(axes, PRIMARY_OUTCOMES):
        for comparator in BOUNDARY_COMPARATORS:
            values = [
                row
                for row in chosen
                if row["outcome_id"] == outcome and row["comparator"] == comparator
            ]
            values.sort(key=lambda row: float(row["tau"]))
            x = np.asarray([row["tau"] for row in values], dtype=float)
            y = np.asarray([row["estimate"] for row in values], dtype=float)
            low = np.asarray([row["ci95_low"] for row in values], dtype=float)
            high = np.asarray([row["ci95_high"] for row in values], dtype=float)
            style = styles[comparator]
            handle = axis.errorbar(
                x,
                y,
                yerr=np.vstack((y - low, high - y)),
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markerfacecolor="white",
                markeredgewidth=1.2,
                markersize=5.2,
                linewidth=1.6,
                elinewidth=1.0,
                capsize=2.8,
            )
            if axis is axes[0]:
                handles.append(handle)
            aggregate = [
                row
                for row in aggregate_rows
                if row["domain_id"] == "osa_primary"
                and row["outcome_id"] == outcome
                and row["comparator"] == comparator
            ]
            if len(aggregate) != 1:
                raise ComparatorFamilyAnalysisError(
                    "OSA profile requires one gate-integrated aggregate per curve"
                )
            item = aggregate[0]
            axis.errorbar(
                [0.99],
                [float(item["estimate"])],
                yerr=np.asarray(
                    [[float(item["estimate"]) - float(item["ci95_low"])],
                     [float(item["ci95_high"]) - float(item["estimate"])]]
                ),
                color=style["color"],
                # The gate-integrated mean is a separate estimand, not the
                # continuation of the threshold profile.  A common diamond
                # shape and an independent artist make that distinction
                # explicit while color continues to identify the comparator.
                marker="D",
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.6,
                linestyle="none",
                markersize=5.7,
                elinewidth=1.0,
                capsize=2.8,
                zorder=5,
            )
        axis.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        axis.axvline(0.945, color="#A0A0A0", linestyle="--", linewidth=0.8)
        axis.set_title(titles[outcome], loc="left", fontweight="semibold")
        axis.set_xticks(
            (*GATES, 0.99),
            [*(f"{tau:.1f}" for tau in GATES), "Overall"],
        )
        axis.set_xlim(0.475, 1.025)
        axis.set_xlabel(r"Toxicity-rule stringency $\tau$ (larger is stricter)")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("cKG minus comparator (pp)", labelpad=7.0)
    fig.legend(
        handles,
        [POLICY_LABELS[item] for item in BOUNDARY_COMPARATORS],
        loc="upper center",
        bbox_to_anchor=(0.68, 1.015),
        ncol=2,
        frameon=False,
        title="Comparator in cKG minus comparator",
        title_fontsize=8.8,
    )
    fig.suptitle(
        "OSA comparator-family profile",
        x=0.10,
        y=0.995,
        ha="left",
        fontsize=10.5,
        fontweight="semibold",
    )
    paths = _save_figure(
        fig,
        stem,
        title="OSA comparator-family profile",
        subject=(
            "M=200 matched replicate sets; pointwise paired 95% t intervals for "
            "cKG minus tMSE and entropy reduction "
            "terminal-recommendation and assignment contrasts"
        ),
    )
    plt.close(fig)
    return paths


def _plot_cellwise_quadrant(
    rows: Sequence[Mapping[str, Any]], stem: Path
) -> tuple[Path, Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    boundary = [row for row in rows if row["comparator"] in BOUNDARY_COMPARATORS]
    terminal = {
        (row["surface_id"], float(row["tau"]), row["comparator"]): row
        for row in boundary
        if row["outcome_id"] == "terminal_recommendation_pct"
    }
    assignments = {
        (row["surface_id"], float(row["tau"]), row["comparator"]): row
        for row in boundary
        if row["outcome_id"] == "assignment_pct_nmax"
    }
    if set(terminal) != set(assignments) or len(terminal) != 40:
        raise ComparatorFamilyAnalysisError("quadrant map does not have 40 paired cells")
    colors = {
        "osa": "#0072B2",
        "efftox": "#009E73",
        "mariposa": "#CC79A7",
        "gbump": "#D55E00",
    }
    markers = {0.5: "o", 0.6: "s", 0.7: "^", 0.8: "D", 0.9: "P"}
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.45), sharex=True, sharey=True)
    for axis, comparator in zip(axes, BOUNDARY_COMPARATORS):
        for key, yrow in terminal.items():
            sim, tau, policy = key
            if policy != comparator:
                continue
            xrow = assignments[key]
            x = float(xrow["estimate"])
            y = float(yrow["estimate"])
            axis.errorbar(
                x,
                y,
                xerr=np.asarray(
                    [[x - float(xrow["ci95_low"])], [float(xrow["ci95_high"]) - x]]
                ),
                yerr=np.asarray(
                    [[y - float(yrow["ci95_low"])], [float(yrow["ci95_high"]) - y]]
                ),
                color=colors[sim],
                marker=markers[tau],
                markerfacecolor="white" if sim == "gbump" else colors[sim],
                markeredgecolor=colors[sim],
                markeredgewidth=1.0,
                markersize=4.8,
                linestyle="none",
                elinewidth=0.65,
                capsize=1.6,
            )
        axis.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
        axis.axvline(0.0, color="black", linestyle=":", linewidth=0.9)
        axis.set_title(
            f"cKG minus {POLICY_LABELS[comparator]}",
            loc="left",
            fontweight="semibold",
        )
        axis.grid(color="#E3E3E3", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("cKG minus comparator: assignment contrast (pp)", y=0.16)
    fig.supylabel(
        "cKG minus comparator:\nterminal contrast (pp)",
        x=0.02,
    )
    surface_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=colors[sim], label=SURFACE_LABELS[sim])
        for sim in SURFACES
    ]
    gate_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[tau],
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="none",
            color="black",
            label=rf"$\tau={tau:.1f}$",
        )
        for tau in GATES
    ]
    fig.legend(
        surface_handles + gate_handles,
        [handle.get_label() for handle in surface_handles + gate_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=5,
        frameon=False,
        fontsize=8,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    fig.suptitle(
        "Cellwise terminal-versus-assignment contrasts",
        x=0.09,
        y=0.995,
        ha="left",
        fontsize=10.5,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.31, top=0.83, wspace=0.10)
    paths = _save_figure(
        fig,
        stem,
        title="Terminal-versus-assignment comparator-family quadrant map",
        subject=(
            "Cellwise point estimates and pointwise unadjusted paired 95% t intervals; "
            "the prespecified terminal-positive/assignment-negative direction is upper-left"
        ),
    )
    plt.close(fig)
    return paths


def _ellipse_axis_radii(
    width: float, height: float, angle_degrees: float
) -> tuple[float, float]:
    """Return the exact axis-aligned radii of a rotated ellipse."""

    a = float(width) / 2.0
    b = float(height) / 2.0
    theta = math.radians(float(angle_degrees))
    cosine = math.cos(theta)
    sine = math.sin(theta)
    x_radius = math.sqrt((a * cosine) ** 2 + (b * sine) ** 2)
    y_radius = math.sqrt((a * sine) ** 2 + (b * cosine) ** 2)
    return x_radius, y_radius


def _surface_quadrant_limits(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return padded limits containing every rotated Hotelling ellipse."""

    x_extent: list[float] = [0.0]
    y_extent: list[float] = [0.0]
    for row in rows:
        x = float(row["assignment_estimate"])
        y = float(row["terminal_estimate"])
        width = max(float(row["hotelling_ellipse_width"]), 1e-10)
        height = max(float(row["hotelling_ellipse_height"]), 1e-10)
        x_radius, y_radius = _ellipse_axis_radii(
            width, height, float(row["hotelling_ellipse_angle_degrees"])
        )
        x_extent.extend([x - x_radius, x + x_radius])
        y_extent.extend([y - y_radius, y + y_radius])
    x_span = max(max(x_extent) - min(x_extent), 1.0)
    y_span = max(max(y_extent) - min(y_extent), 1.0)
    return (
        (min(x_extent) - 0.10 * x_span, max(x_extent) + 0.10 * x_span),
        (min(y_extent) - 0.12 * y_span, max(y_extent) + 0.12 * y_span),
    )


def _plot_surface_quadrant(
    rows: Sequence[Mapping[str, Any]], stem: Path
) -> tuple[Path, Path, Path]:
    """Plot eight gate-integrated surface/comparator means with 95% ellipses."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    if len(rows) != 8:
        raise ComparatorFamilyAnalysisError("main quadrant requires eight surface points")
    colors = {
        "osa": "#0072B2",
        "efftox": "#009E73",
        "mariposa": "#CC79A7",
        "gbump": "#D55E00",
    }
    markers = {"osa": "o", "efftox": "s", "mariposa": "^", "gbump": "D"}
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), sharex=True, sharey=True)
    x_limits, y_limits = _surface_quadrant_limits(rows)
    for axis, comparator in zip(axes, BOUNDARY_COMPARATORS):
        current = [row for row in rows if row["comparator"] == comparator]
        if len(current) != 4:
            raise ComparatorFamilyAnalysisError(
                f"surface quadrant lacks four {comparator} points"
            )
        for row in current:
            sim = str(row["surface_id"])
            x = float(row["assignment_estimate"])
            y = float(row["terminal_estimate"])
            width = max(float(row["hotelling_ellipse_width"]), 1e-10)
            height = max(float(row["hotelling_ellipse_height"]), 1e-10)
            angle = float(row["hotelling_ellipse_angle_degrees"])
            ellipse = Ellipse(
                (x, y),
                width=width,
                height=height,
                angle=angle,
                edgecolor=colors[sim],
                fill=False,
                linewidth=1.2,
                zorder=2,
            )
            axis.add_patch(ellipse)
            axis.plot(
                x,
                y,
                marker=markers[sim],
                color=colors[sim],
                markerfacecolor="white" if sim == "gbump" else colors[sim],
                markeredgewidth=1.2,
                markersize=6.0,
                linestyle="none",
                zorder=4,
            )
        axis.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
        axis.axvline(0.0, color="black", linestyle=":", linewidth=0.9)
        axis.set_title(
            f"cKG minus {POLICY_LABELS[comparator]}",
            loc="left",
            fontweight="semibold",
        )
        axis.grid(color="#E3E3E3", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlim(*x_limits)
    axes[0].set_ylim(*y_limits)
    fig.supxlabel("cKG minus comparator: assignment contrast (pp)", y=0.10)
    fig.supylabel(
        "cKG minus comparator:\nterminal contrast (pp)",
        x=0.02,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker=markers[sim],
            color=colors[sim],
            markerfacecolor="white" if sim == "gbump" else colors[sim],
            markeredgecolor=colors[sim],
            markeredgewidth=1.2,
            linestyle="none",
            label=SURFACE_LABELS[sim],
        )
        for sim in SURFACES
    ]
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle(
        "Surface-level terminal-versus-assignment profile",
        x=0.09,
        y=0.995,
        ha="left",
        fontsize=10.5,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.25, top=0.83, wspace=0.10)
    paths = _save_figure(
        fig,
        stem,
        title="Surface-level terminal-versus-assignment comparator-family profile",
        subject=(
            "Eight gate-integrated paired surface/comparator means with descriptive "
            "95% Hotelling confidence ellipses; decision rules use the frozen univariate intervals"
        ),
    )
    plt.close(fig)
    return paths


def _input_provenance(
    bundle: ProjectionBundle, extension: Mapping[str, Any]
) -> dict[str, Any]:
    formal_files = {}
    for filename, _spec in FORMAL_SOURCE_SPECS:
        detail = bundle.provenance["record_files"][filename]
        formal_files[filename] = {
            "rows_in_authenticated_source": int(detail["rows"]),
            "compressed_sha256": detail["compressed_sha256"],
            "uncompressed_sha256": detail["uncompressed_sha256"],
        }
    return {
        "formal_projection": {
            "projection_fingerprint": bundle.provenance["projection_fingerprint"],
            "projection_metadata_sha256": bundle.provenance[
                "projection_metadata_sha256"
            ],
            "formal_manifest_sha256": bundle.provenance["formal_manifest_sha256"],
            "selected_source_files": formal_files,
            "selected_records": EXPECTED_REUSED,
        },
        "new_extension": dict(extension),
        "analysis_specification": {
            "path": "paper/comparator_family_extension_prespec.md",
            "sha256": _sha256_file(SPEC_PATH),
            "temporal_status": "frozen before generation of missing extension cells",
        },
    }


def _write_sidecar(
    artifact: Path,
    *,
    artifact_type: str,
    provenance: Mapping[str, Any],
    selected_commitment: Mapping[str, Any],
    support: Mapping[str, Any],
) -> Path:
    metadata = {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "artifact": artifact.name,
        "artifact_sha256": _sha256_file(artifact),
        "analyzer": "paper/analyze_comparator_family.py",
        "analyzer_sha256": _sha256_file(Path(__file__)),
        "input_provenance": provenance,
        "selected_family_commitment": dict(selected_commitment),
        "analysis": {
            "principal_comparators": list(BOUNDARY_COMPARATORS),
            "secondary_comparators": list(SECONDARY_COMPARATORS),
            "reader_facing_policy_labels": {
                policy: POLICY_LABELS[policy] for policy in POLICIES
            },
            "qBIG_reader_facing_description": POLICY_DESCRIPTIONS["qBIG"],
            "N_max": N_MAX,
            "monte_carlo_symbol": "M",
            "interval": "two-sided paired 95% t interval across matched replicate sets",
            "cellwise_intervals": "pointwise and unadjusted",
            "cell_vote_count_decision_rule": False,
            "support_summary": support,
        },
    }
    target = Path(str(artifact) + ".metadata.json")
    _write_json(target, metadata)
    return target


def analyze_records(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Create every deterministic analysis artifact from a validated family set."""

    _validate_completed_family(records)
    provenance = dict(provenance)
    provenance["post_hoc_allocation_estimand_correction"] = {
        "path": "paper/allocation_estimand_correction.md",
        "sha256": _sha256_file(ALLOCATION_CORRECTION_PATH),
        "status": "post_hoc_reporting_correction_raw_simulation_paths_unchanged",
    }
    analysis_records = _prepare_analysis_records(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = aggregate_contrasts(analysis_records)
    cellwise = cellwise_primary_contrasts(analysis_records)
    surface_bivariate = surface_aggregate_bivariate_contrasts(analysis_records)
    full_panel_oc = full_panel_operating_characteristics(analysis_records)
    secondary_aggregate = secondary_aggregate_contrasts(analysis_records)
    secondary = secondary_cellwise_contrasts(analysis_records)
    gradual_oc = gradual_operating_characteristics(analysis_records)
    gradual_contrasts = gradual_paired_contrasts(analysis_records)
    support = _support_summary(aggregate)

    selected_path = output_dir / "comparator_family_selected_records.json.zst"
    selected_commitment = _write_selected_master(selected_path, records)
    aggregate_path = output_dir / "comparator_family_aggregate_contrasts.csv"
    cellwise_path = output_dir / "comparator_family_cellwise_contrasts.csv"
    surface_bivariate_path = output_dir / "comparator_family_surface_bivariate.csv"
    full_panel_oc_path = output_dir / "comparator_family_full_panel_oc.csv"
    secondary_aggregate_path = output_dir / "comparator_family_secondary_aggregate.csv"
    secondary_path = output_dir / "comparator_family_secondary_diagnostics.csv"
    gradual_oc_path = output_dir / "comparator_family_gradual_oc.csv"
    gradual_contrast_path = output_dir / "comparator_family_gradual_contrasts.csv"
    tex_path = output_dir / "comparator_family_tables.tex"
    aggregate_tex_path = output_dir / "comparator_family_aggregate_table.tex"
    gradual_tex_path = output_dir / "comparator_family_gradual_oc.tex"
    gradual_contrast_tex_path = output_dir / "comparator_family_gradual_contrasts.tex"
    full_panel_tex_path = output_dir / "comparator_family_full_panel_oc.tex"
    summary_path = output_dir / "comparator_family_summary.json"
    _write_csv(aggregate_path, aggregate)
    _write_csv(cellwise_path, cellwise)
    _write_csv(surface_bivariate_path, surface_bivariate)
    _write_csv(full_panel_oc_path, full_panel_oc)
    _write_csv(secondary_aggregate_path, secondary_aggregate)
    _write_csv(secondary_path, secondary)
    _write_csv(gradual_oc_path, gradual_oc)
    _write_csv(gradual_contrast_path, gradual_contrasts)
    tex_path.write_text(_tex_tables(aggregate, gradual_oc), encoding="utf-8")
    aggregate_tex_path.write_text(_tex_aggregate_table(aggregate), encoding="utf-8")
    gradual_tex_path.write_text(_tex_gradual_oc(gradual_oc), encoding="utf-8")
    gradual_contrast_tex_path.write_text(
        _tex_gradual_contrasts(gradual_contrasts), encoding="utf-8"
    )
    full_panel_tex_path.write_text(
        _tex_full_panel_oc(full_panel_oc), encoding="utf-8"
    )

    summary = {
        "schema_version": 1,
        "status": "COMPLETE_COMPARATOR_FAMILY_ANALYSIS",
        "records": {
            "total": len(records),
            "reused_formal": sum(row["_source_origin"] == "reused_formal" for row in records),
            "new_extension": sum(row["_source_origin"] == "new_extension" for row in records),
            "selected_master": selected_commitment,
        },
        "policy_hierarchy": {
            "principal_boundary_comparators": list(BOUNDARY_COMPARATORS),
            "secondary_comparators": list(SECONDARY_COMPARATORS),
            "hybrid_represents_boundary_family": False,
            "reader_facing_labels": {
                policy: POLICY_LABELS[policy] for policy in POLICIES
            },
            "qBIG_reader_facing_description": POLICY_DESCRIPTIONS["qBIG"],
        },
        "estimands": {
            "OSA_M": 200,
            "binding_family_M": 100,
            "Gaussian_bump_M": 200,
            "gradual_OSA_M": 200,
            "N_max": N_MAX,
            "full_grid_allocation_estimand": (
                "100 * post-initialization above-threshold assignments / "
                "(N_obs - N_init)"
            ),
            "gradual_allocation_estimand": (
                "includes the two actual on-grid initialization participants"
            ),
            "aggregate_interval": "paired two-sided 95% t",
            "cellwise_interval": "paired pointwise unadjusted two-sided 95% t",
            "secondary_aggregate_interval": "paired descriptive two-sided 95% t",
            "secondary_matched_eligibility_rule": SECONDARY_MATCHED_ELIGIBILITY_RULE,
            "full_panel_absolute_oc_rows": len(full_panel_oc),
            "conditional_denominators": (
                "reported separately for recommendation-conditional and nonempty-gate outcomes"
            ),
        },
        "support": support,
        "input_provenance": provenance,
        "aggregate_contrasts": aggregate,
        "surface_bivariate_display": surface_bivariate,
        "secondary_aggregate_contrasts": secondary_aggregate,
    }
    _write_json(summary_path, summary)
    figure1 = _plot_family_profile(
        cellwise, aggregate, output_dir / "comparator_family_profile"
    )
    figure2 = _plot_surface_quadrant(
        surface_bivariate, output_dir / "comparator_family_quadrant"
    )
    cellwise_figure = _plot_cellwise_quadrant(
        cellwise, output_dir / "comparator_family_quadrant_cellwise"
    )

    artifacts: list[tuple[Path, str]] = [
        (selected_path, "selected_family_records"),
        (aggregate_path, "aggregate_paired_contrasts"),
        (cellwise_path, "cellwise_primary_contrasts"),
        (surface_bivariate_path, "surface_aggregate_bivariate_contrasts"),
        (full_panel_oc_path, "full_panel_five_policy_operating_characteristics"),
        (
            secondary_aggregate_path,
            "secondary_prespecified_domain_aggregate_contrasts",
        ),
        (secondary_path, "secondary_mechanism_diagnostics"),
        (gradual_oc_path, "gradual_access_operating_characteristics"),
        (gradual_contrast_path, "gradual_access_paired_contrasts"),
        (tex_path, "latex_tables"),
        (aggregate_tex_path, "aggregate_paired_contrasts_latex_table"),
        (gradual_tex_path, "gradual_access_operating_characteristics_latex_table"),
        (
            gradual_contrast_tex_path,
            "gradual_access_principal_paired_contrasts_latex_table",
        ),
        (full_panel_tex_path, "full_panel_five_policy_latex_table"),
        (summary_path, "analysis_summary"),
        *[(path, "figure_1_family_profile") for path in figure1],
        *[(path, "figure_2_terminal_assignment_quadrant") for path in figure2],
        *[(path, "si_cellwise_terminal_assignment_quadrant") for path in cellwise_figure],
    ]
    for artifact, artifact_type in artifacts:
        _write_sidecar(
            artifact,
            artifact_type=artifact_type,
            provenance=provenance,
            selected_commitment=selected_commitment,
            support=support,
        )
    return summary


def analyze(
    projection_dir: str | Path,
    extension_raw: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    bundle = load_projection_bundle(projection_dir)
    extension, extension_provenance = load_extension_records(extension_raw)
    if extension_provenance["bindings"].get("formal_manifest_sha256") != bundle.provenance.get(
        "formal_manifest_sha256"
    ):
        raise ComparatorFamilyAnalysisError(
            "extension and formal projection are not bound to the same formal manifest"
        )
    records = build_family_records(bundle, extension)
    provenance = _input_provenance(bundle, extension_provenance)
    return analyze_records(records, output_dir, provenance=provenance)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection-dir",
        required=True,
        help="authenticated formal_projection-<fingerprint> directory",
    )
    parser.add_argument(
        "--extension-raw",
        required=True,
        help="committed comparator_family_completion_records.json.zst",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    summary = analyze(args.projection_dir, args.extension_raw, args.out_dir)
    print(
        f"wrote comparator-family analysis: {summary['records']['total']:,} records; "
        f"interpretation={summary['support']['interpretation_ladder_branch']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
