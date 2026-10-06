#!/usr/bin/env python3
"""Authenticate and copy the frozen calibration publication into a manuscript.

This is deliberately an outcome-blind transport program.  It authenticates the
committed publication envelope, hashes every publication artifact and sidecar,
and copies only five frozen TeX fragments and five frozen PDF figures.  It does
not decode a scientific artifact, choose a result, generate prose, or edit a
manuscript source file.

The command-line interface authenticates only by default.  ``--execute`` is
required to create the fixed projection.  A projection is immutable: all
twelve managed destinations (ten artifacts plus metadata and commit) must be
absent, or all twelve must already be byte-identical to the newly authenticated
projection.  The commit marker is installed last.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from types import MappingProxyType
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_STAGING = ROOT / "results/budget_toxicity_calibration_staging"

TRUSTED_CONTRACT_SHA256 = (
    "a1904b435d4e42ef6df25094bc1f3ce35156ceaec5bf157ec24fb2ff2827b6a8"
)
TRUSTED_ADAPTER_SHA256 = (
    "e14c2464f586f3abb805bfa24bb47f2ecc61a93e7405e0cb51489d497c6aa919"
)
TRUSTED_COMMITMENTS = {
    "analyzer_sha256": "bcf3ec2964b88d61ac4becc8a9aa838831a3240d93a082abf42b5b328799ec2a",
    "manifest_sha256": "2e189ff6426bd3ad12fd29896fb6f76027a0d9e4424013d92edd0df3d143d3b6",
    "prespec_sha256": "1b1152d8f09462b96bf2f35252a9cd5703b331ab7a3546d4708b189f81e8c178",
}

CONTRACT_NAME = "budget_toxicity_calibration_publication_contract.json"
ADAPTER_NAME = "paper/generate_budget_toxicity_calibration_publication.py"
PUBLICATION_METADATA_NAME = "budget_toxicity_calibration_publication.metadata.json"
PUBLICATION_COMMIT_NAME = "budget_toxicity_calibration_publication.commit.json"
PROJECTION_METADATA_RELATIVE = Path(
    "generated/budget_toxicity_calibration_manuscript_projection.metadata.json"
)
PROJECTION_COMMIT_RELATIVE = Path(
    "generated/budget_toxicity_calibration_manuscript_projection.commit.json"
)

PUBLICATION_STATUS = "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_PUBLICATION"
PUBLICATION_COMMIT_STATUS = (
    "COMMITTED_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_PUBLICATION"
)
PUBLICATION_ARTIFACT_STATUS = (
    "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_PUBLICATION_ARTIFACT"
)
PUBLICATION_ARTIFACT_CLASS = (
    "outcome_independent_budget_toxicity_calibration_publication_projection"
)
ANALYSIS_STATUS = "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_ANALYSIS"
ANALYSIS_COMMIT_STATUS = (
    "COMMITTED_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_ANALYSIS"
)
BASELINE_STATUS = "PASS_AVAILABLE_FIELD_IDENTITY"
PROJECTION_STATUS = (
    "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_MANUSCRIPT_PROJECTION"
)
PROJECTION_COMMIT_STATUS = (
    "COMMITTED_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_MANUSCRIPT_PROJECTION"
)
PROJECTION_ARTIFACT_CLASS = (
    "outcome_blind_budget_toxicity_calibration_manuscript_projection"
)

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ELIGIBLE_FINAL_M = (200, 500, 1000)
INPUT_NAMES = (
    "budget_toxicity_calibration_analysis.metadata.json",
    "budget_toxicity_calibration_analysis.commit.json",
    "budget_toxicity_calibration_primary.json",
    "budget_toxicity_calibration_primary.json.metadata.json",
    "budget_toxicity_calibration_secondary.json",
    "budget_toxicity_calibration_secondary.json.metadata.json",
    "budget_toxicity_calibration_baseline_identity.json",
    "budget_toxicity_calibration_baseline_identity.json.metadata.json",
)

FIGURE_KEYS_TO_STEMS = {
    "principal_profile": "figure_budget_toxicity_calibration_principal_profile",
    "cei_reference": "figure_budget_toxicity_calibration_cei_reference",
    "absolute_gate_brier_profiles": (
        "figure_budget_toxicity_calibration_absolute_gate_brier_profiles"
    ),
    "calibration_penalties": (
        "figure_budget_toxicity_calibration_calibration_penalties"
    ),
    "attenuation_interactions": (
        "figure_budget_toxicity_calibration_attenuation_interactions"
    ),
}

EXPECTED_ARTIFACT_TYPES: dict[str, str] = {
    "budget_toxicity_calibration_all_118.csv": "all_118_primary_csv",
    "budget_toxicity_calibration_all_118.json": "all_118_primary_json",
    "budget_toxicity_calibration_plotted_secondary_72.json": (
        "plotted_secondary_absolute_72_json"
    ),
    "budget_toxicity_calibration_policy_vs_cei_did_36.json": (
        "complete_policy_vs_cEI_calibration_did_36_json"
    ),
    "budget_toxicity_calibration_policy_vs_cei_did_36.tex": (
        "complete_policy_vs_cEI_calibration_did_36_longtable"
    ),
    "budget_toxicity_calibration_family_A.tex": "complete_family_A_longtable",
    "budget_toxicity_calibration_family_B.tex": "complete_family_B_longtable",
    "budget_toxicity_calibration_family_C.tex": "complete_family_C_longtable",
    "budget_toxicity_calibration_result_facts.json": "frozen_result_facts_json",
    "budget_toxicity_calibration_result_facts.tex": "frozen_result_facts_tex",
}
for _figure_key, _figure_stem in FIGURE_KEYS_TO_STEMS.items():
    for _suffix in ("pdf", "png"):
        EXPECTED_ARTIFACT_TYPES[f"{_figure_stem}.{_suffix}"] = (
            f"publication_figure_{_figure_key}"
        )
if len(EXPECTED_ARTIFACT_TYPES) != 20:  # pragma: no cover - import-time invariant
    raise AssertionError("publication artifact contract is not exactly 20 files")

TEX_ARTIFACTS = (
    "budget_toxicity_calibration_family_A.tex",
    "budget_toxicity_calibration_family_B.tex",
    "budget_toxicity_calibration_family_C.tex",
    "budget_toxicity_calibration_policy_vs_cei_did_36.tex",
    "budget_toxicity_calibration_result_facts.tex",
)
PDF_ARTIFACTS = tuple(f"{stem}.pdf" for stem in FIGURE_KEYS_TO_STEMS.values())
DESTINATION_MAPPING: Mapping[str, Path] = MappingProxyType({
    **{name: Path("generated") / name for name in TEX_ARTIFACTS},
    **{name: Path(name) for name in PDF_ARTIFACTS},
})
if len(DESTINATION_MAPPING) != 10:  # pragma: no cover - import-time invariant
    raise AssertionError("manuscript projection contract is not exactly ten files")
if len({name.casefold() for name in EXPECTED_ARTIFACT_TYPES}) != 20:
    raise AssertionError("publication artifact names collide under case-folding")
if len(
    {destination.as_posix().casefold() for destination in DESTINATION_MAPPING.values()}
) != 10:
    raise AssertionError("manuscript destinations collide under case-folding")


class ManuscriptProjectionError(RuntimeError):
    """Raised when authentication or immutable projection fails closed."""


class PostCommitCleanupError(ManuscriptProjectionError):
    """A durable verified projection exists, but transaction residue remains."""


@dataclass(frozen=True)
class AuthenticatedPublication:
    root: Path
    artifact_hashes: Mapping[str, str]
    artifact_bytes: Mapping[str, int]
    sidecar_hashes: Mapping[str, str]
    metadata_sha256: str
    commit_sha256: str
    input_bundle_sha256: str
    output_bundle_sha256: str
    final_m: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _bundle_hash(mapping: Mapping[str, str]) -> str:
    return _sha256_bytes(_canonical_json_bytes(dict(sorted(mapping.items()))))


def _assert_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ManuscriptProjectionError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, label=f"{label}[{index}]")


def _decode_canonical_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token}")

    try:
        value = json.loads(raw, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ManuscriptProjectionError(f"{label} is not finite valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ManuscriptProjectionError(f"{label} is not a JSON object")
    _assert_finite(value, label=label)
    if raw != _canonical_json_bytes(value):
        raise ManuscriptProjectionError(f"{label} is not canonical JSON")
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: Iterable[str], *, label: str
) -> None:
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        raise ManuscriptProjectionError(
            f"{label} fields changed: missing={sorted(expected_set-observed)}, "
            f"extra={sorted(observed-expected_set)}"
        )


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ManuscriptProjectionError(f"{label} is not a lowercase SHA-256")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManuscriptProjectionError(f"{label} is not an integer >= {minimum}")
    return value


def _safe_basename(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ManuscriptProjectionError(f"{label} is not a safe basename")
    return value


def _safe_relative(value: Path, *, label: str) -> Path:
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", ".."} for part in value.parts)
        or any("/" in part or "\\" in part for part in value.parts)
    ):
        raise ManuscriptProjectionError(f"{label} is not a safe relative path")
    return value


def _absolute_without_symlink_resolution(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def _symlink_component(path: str | Path) -> Path | None:
    absolute = _absolute_without_symlink_resolution(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            observed = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ManuscriptProjectionError(
                f"cannot lstat path component: {current}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode):
            return current
    return None


def _guard_directory(path: str | Path, *, label: str) -> Path:
    lexical = _absolute_without_symlink_resolution(path)
    link = _symlink_component(lexical)
    if link is not None:
        raise ManuscriptProjectionError(f"{label} contains a symlink component: {link}")
    try:
        observed = os.lstat(lexical)
    except OSError as exc:
        raise ManuscriptProjectionError(f"{label} directory is absent") from exc
    if not stat.S_ISDIR(observed.st_mode):
        raise ManuscriptProjectionError(f"{label} is not a directory")
    return lexical.resolve(strict=True)


def _directory_open_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ManuscriptProjectionError(
            "platform lacks required no-follow directory primitives"
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _scan_flat_regular_files_nofollow(root: Path, *, label: str) -> tuple[str, ...]:
    """Return a flat directory inventory, rejecting links, dirs, specials, hardlinks."""

    try:
        root_lstat = os.lstat(root)
        root_fd = os.open(root, _directory_open_flags())
    except OSError as exc:
        raise ManuscriptProjectionError(f"cannot safely open {label}") from exc
    try:
        opened = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_lstat.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or (root_lstat.st_dev, root_lstat.st_ino)
            != (opened.st_dev, opened.st_ino)
        ):
            raise ManuscriptProjectionError(f"{label} root changed during scan")
        try:
            with os.scandir(root_fd) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise ManuscriptProjectionError(f"cannot scan {label}") from exc
        files: list[str] = []
        for entry in entries:
            name = _safe_basename(entry.name, label=f"{label} entry")
            try:
                observed = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ManuscriptProjectionError(
                    f"cannot lstat {label} entry: {name}"
                ) from exc
            if stat.S_ISLNK(observed.st_mode):
                raise ManuscriptProjectionError(f"{label} contains a symlink: {name}")
            if not stat.S_ISREG(observed.st_mode):
                raise ManuscriptProjectionError(
                    f"{label} contains a non-regular file: {name}"
                )
            if observed.st_nlink != 1:
                raise ManuscriptProjectionError(f"{label} contains a hardlink: {name}")
            files.append(name)
        return tuple(files)
    finally:
        os.close(root_fd)


def _read_regular_nofollow(root: Path, name: str, *, label: str) -> bytes:
    """Read one direct child, proving regular/single-link identity around the read."""

    safe_name = _safe_basename(name, label=label)
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        directory_fd = os.open(root, _directory_open_flags())
        before = os.stat(safe_name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ManuscriptProjectionError(f"{label} is not a regular file")
        if before.st_nlink != 1:
            raise ManuscriptProjectionError(f"{label} is a hardlink")
        file_fd = os.open(
            safe_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd
        )
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ManuscriptProjectionError(f"{label} changed before reading")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        if (
            after.st_nlink != 1
            or (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
        ):
            raise ManuscriptProjectionError(f"{label} changed while reading")
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            raise ManuscriptProjectionError(f"{label} byte count changed while reading")
        return raw
    except OSError as exc:
        raise ManuscriptProjectionError(f"cannot safely read {label}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _read_path_regular_nofollow(path: Path, *, label: str) -> bytes:
    link = _symlink_component(path)
    if link is not None:
        raise ManuscriptProjectionError(f"{label} contains a symlink component: {link}")
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ManuscriptProjectionError(f"{label} is absent") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ManuscriptProjectionError(f"{label} is not a regular file")
    if before.st_nlink != 1:
        raise ManuscriptProjectionError(f"{label} is a hardlink")
    try:
        file_fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ManuscriptProjectionError(f"cannot safely open {label}") from exc
    try:
        opened = os.fstat(file_fd)
        if (
            opened.st_nlink != 1
            or not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ManuscriptProjectionError(f"{label} changed before reading")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        raw = b"".join(chunks)
        if (
            after.st_nlink != 1
            or (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or len(raw) != after.st_size
        ):
            raise ManuscriptProjectionError(f"{label} changed while reading")
        return raw
    finally:
        os.close(file_fd)


def _validate_local_commitments() -> None:
    expected = {
        ROOT / ADAPTER_NAME: TRUSTED_ADAPTER_SHA256,
        ROOT / "paper" / CONTRACT_NAME: TRUSTED_CONTRACT_SHA256,
    }
    for path, digest in expected.items():
        raw = _read_path_regular_nofollow(path, label=f"trusted local source {path.name}")
        if _sha256_bytes(raw) != digest:
            raise ManuscriptProjectionError(
                f"trusted local source commitment changed: {path.name}"
            )


def _expected_publication_files() -> set[str]:
    artifacts = set(EXPECTED_ARTIFACT_TYPES)
    sidecars = {name + ".metadata.json" for name in artifacts}
    return artifacts | sidecars | {PUBLICATION_METADATA_NAME, PUBLICATION_COMMIT_NAME}


def _validate_inputs(metadata: Mapping[str, Any]) -> str:
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(INPUT_NAMES):
        raise ManuscriptProjectionError("publication input manifest changed")
    hashes: dict[str, str] = {}
    for name in INPUT_NAMES:
        entry = inputs[name]
        if not isinstance(entry, Mapping):
            raise ManuscriptProjectionError(f"publication input entry is malformed: {name}")
        _exact_fields(entry, {"sha256", "bytes"}, label=f"publication input {name}")
        hashes[name] = _sha(entry.get("sha256"), label=f"publication input {name} hash")
        _integer(entry.get("bytes"), label=f"publication input {name} bytes", minimum=1)
    computed = _bundle_hash(hashes)
    if metadata.get("input_bundle_sha256") != computed:
        raise ManuscriptProjectionError("publication input-bundle commitment differs")
    return computed


def _validate_authenticated_analysis(metadata: Mapping[str, Any]) -> int:
    analysis = metadata.get("authenticated_analysis")
    if not isinstance(analysis, Mapping):
        raise ManuscriptProjectionError("authenticated-analysis summary is absent")
    _exact_fields(analysis, {
        "status", "commit_status", "baseline_status", "final_M",
        "primary_estimand_count", "primary_family_counts",
        "secondary_absolute_row_count", "secondary_contrast_row_count",
        "secondary_absolute_profile_rows_used",
        "secondary_policy_vs_cEI_calibration_did_rows_published",
    }, label="authenticated-analysis summary")
    final_m = analysis.get("final_M")
    if (
        analysis.get("status") != ANALYSIS_STATUS
        or analysis.get("commit_status") != ANALYSIS_COMMIT_STATUS
        or analysis.get("baseline_status") != BASELINE_STATUS
        or final_m not in ELIGIBLE_FINAL_M
        or analysis.get("primary_estimand_count") != 118
        or analysis.get("primary_family_counts") != {"A": 54, "B": 48, "C": 16}
        or analysis.get("secondary_absolute_row_count") != 864
        or analysis.get("secondary_contrast_row_count") != 363
        or analysis.get("secondary_absolute_profile_rows_used") != 72
        or analysis.get(
            "secondary_policy_vs_cEI_calibration_did_rows_published"
        ) != 36
    ):
        raise ManuscriptProjectionError(
            "authenticated-analysis status/count/baseline summary changed"
        )
    return int(final_m)


def _validate_publication_rules(metadata: Mapping[str, Any]) -> None:
    expected = {
        "complete_primary_projection": True,
        "no_filtering_by_sign_interval_or_winner": True,
        "colors_encode_only_c_sigma": True,
        "finite_fixed_design_audit": True,
        "raw_estimates_mcse_and_intervals_retained": True,
        "primary_scientific_display_uses_guarded_simultaneous_bands": True,
        "secondary_scientific_display_uses_guarded_pointwise_intervals": True,
        "policy_vs_cEI_calibration_did_projection_complete": True,
        "secondary_cell_predicates_have_no_vote_or_family_theorem": True,
    }
    if metadata.get("publication_rules") != expected:
        raise ManuscriptProjectionError("publication rules changed")


def authenticate_publication(
    publication_dir: str | Path,
    *,
    trusted_publication_commit_sha256: str,
) -> AuthenticatedPublication:
    """Authenticate a 42-file publication against an external commit hash."""

    _validate_local_commitments()
    trusted_commit_sha = _sha(
        trusted_publication_commit_sha256,
        label="externally trusted publication commit SHA-256",
    )
    root = _guard_directory(publication_dir, label="publication directory")
    active = _absolute_without_symlink_resolution(ACTIVE_STAGING).resolve(strict=False)
    if _is_within(root, active):
        raise ManuscriptProjectionError(
            "publication directory is inside active calibration staging"
        )
    observed_files = set(
        _scan_flat_regular_files_nofollow(root, label="publication directory")
    )
    expected_files = _expected_publication_files()
    if len(observed_files) != 42 or observed_files != expected_files:
        raise ManuscriptProjectionError(
            "publication physical-file inventory changed: "
            f"count={len(observed_files)}, "
            f"missing={sorted(expected_files-observed_files)}, "
            f"extra={sorted(observed_files-expected_files)}"
        )

    metadata_raw = _read_regular_nofollow(
        root, PUBLICATION_METADATA_NAME, label="publication metadata"
    )
    commit_raw = _read_regular_nofollow(
        root, PUBLICATION_COMMIT_NAME, label="publication commit"
    )
    observed_commit_sha = _sha256_bytes(commit_raw)
    if observed_commit_sha != trusted_commit_sha:
        raise ManuscriptProjectionError(
            "publication commit differs from the externally trusted SHA-256"
        )
    metadata = _decode_canonical_json(metadata_raw, label="publication metadata")
    commit = _decode_canonical_json(commit_raw, label="publication commit")
    _exact_fields(metadata, {
        "schema_version", "status", "artifact_class", "adapter",
        "adapter_sha256", "contract", "contract_sha256",
        "trusted_commitments", "inputs", "input_bundle_sha256",
        "authenticated_analysis", "publication_rules", "artifacts",
        "output_bundle_sha256", "deterministic_timestamp_utc",
    }, label="publication metadata")
    _exact_fields(commit, {
        "schema_version", "status", "metadata", "metadata_sha256",
        "artifact_count", "adapter_sha256", "contract_sha256",
        "trusted_commitments", "input_bundle_sha256", "output_bundle_sha256",
        "baseline_identity_passed", "primary_estimand_count",
    }, label="publication commit")
    if (
        metadata.get("schema_version") != 3
        or metadata.get("status") != PUBLICATION_STATUS
        or metadata.get("artifact_class") != PUBLICATION_ARTIFACT_CLASS
        or metadata.get("adapter") != ADAPTER_NAME
        or metadata.get("adapter_sha256") != TRUSTED_ADAPTER_SHA256
        or metadata.get("contract") != CONTRACT_NAME
        or metadata.get("contract_sha256") != TRUSTED_CONTRACT_SHA256
        or metadata.get("trusted_commitments") != TRUSTED_COMMITMENTS
        or metadata.get("deterministic_timestamp_utc") != "2000-01-01T00:00:00Z"
    ):
        raise ManuscriptProjectionError("publication metadata envelope changed")
    if (
        commit.get("schema_version") != 3
        or commit.get("status") != PUBLICATION_COMMIT_STATUS
        or commit.get("metadata") != PUBLICATION_METADATA_NAME
        or commit.get("metadata_sha256") != _sha256_bytes(metadata_raw)
        or commit.get("artifact_count") != 20
        or commit.get("adapter_sha256") != TRUSTED_ADAPTER_SHA256
        or commit.get("contract_sha256") != TRUSTED_CONTRACT_SHA256
        or commit.get("trusted_commitments") != TRUSTED_COMMITMENTS
        or commit.get("baseline_identity_passed") is not True
        or commit.get("primary_estimand_count") != 118
    ):
        raise ManuscriptProjectionError("publication commit envelope changed")

    input_bundle = _validate_inputs(metadata)
    final_m = _validate_authenticated_analysis(metadata)
    _validate_publication_rules(metadata)
    if commit.get("input_bundle_sha256") != input_bundle:
        raise ManuscriptProjectionError("commit input-bundle commitment differs")

    artifact_manifest = metadata.get("artifacts")
    if (
        not isinstance(artifact_manifest, Mapping)
        or len(artifact_manifest) != 20
        or set(artifact_manifest) != set(EXPECTED_ARTIFACT_TYPES)
    ):
        raise ManuscriptProjectionError("publication artifact manifest changed")

    artifact_hashes: dict[str, str] = {}
    artifact_bytes: dict[str, int] = {}
    sidecar_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    for artifact_name, expected_type in sorted(EXPECTED_ARTIFACT_TYPES.items()):
        _safe_basename(artifact_name, label="publication artifact name")
        entry = artifact_manifest[artifact_name]
        if not isinstance(entry, Mapping):
            raise ManuscriptProjectionError(
                f"publication artifact entry is malformed: {artifact_name}"
            )
        _exact_fields(
            entry,
            {"sha256", "bytes", "metadata", "metadata_sha256", "artifact_type"},
            label=f"publication artifact entry {artifact_name}",
        )
        sidecar_name = artifact_name + ".metadata.json"
        if (
            entry.get("metadata") != sidecar_name
            or entry.get("artifact_type") != expected_type
        ):
            raise ManuscriptProjectionError(
                f"publication artifact manifest binding changed: {artifact_name}"
            )
        expected_artifact_sha = _sha(
            entry.get("sha256"), label=f"{artifact_name} manifest hash"
        )
        expected_sidecar_sha = _sha(
            entry.get("metadata_sha256"), label=f"{artifact_name} sidecar hash"
        )
        expected_artifact_bytes = _integer(
            entry.get("bytes"), label=f"{artifact_name} manifest bytes", minimum=1
        )
        artifact_raw = _read_regular_nofollow(
            root, artifact_name, label=f"publication artifact {artifact_name}"
        )
        if (
            _sha256_bytes(artifact_raw) != expected_artifact_sha
            or len(artifact_raw) != expected_artifact_bytes
        ):
            raise ManuscriptProjectionError(
                f"publication artifact hash/bytes differ: {artifact_name}"
            )
        sidecar_raw = _read_regular_nofollow(
            root, sidecar_name, label=f"publication sidecar {sidecar_name}"
        )
        if _sha256_bytes(sidecar_raw) != expected_sidecar_sha:
            raise ManuscriptProjectionError(
                f"publication sidecar hash differs: {sidecar_name}"
            )
        sidecar = _decode_canonical_json(
            sidecar_raw, label=f"publication sidecar {sidecar_name}"
        )
        _exact_fields(sidecar, {
            "schema_version", "status", "artifact_class", "artifact_type",
            "artifact", "artifact_sha256", "artifact_bytes", "adapter",
            "adapter_sha256", "contract", "contract_sha256",
            "trusted_commitments", "input_bundle_sha256", "description",
            "selection", "no_filtering_by_sign_interval_or_winner",
        }, label=f"publication sidecar {sidecar_name}")
        if (
            sidecar.get("schema_version") != 3
            or sidecar.get("status") != PUBLICATION_ARTIFACT_STATUS
            or sidecar.get("artifact_class") != PUBLICATION_ARTIFACT_CLASS
            or sidecar.get("artifact_type") != expected_type
            or sidecar.get("artifact") != artifact_name
            or sidecar.get("artifact_sha256") != expected_artifact_sha
            or sidecar.get("artifact_bytes") != expected_artifact_bytes
            or sidecar.get("adapter") != ADAPTER_NAME
            or sidecar.get("adapter_sha256") != TRUSTED_ADAPTER_SHA256
            or sidecar.get("contract") != CONTRACT_NAME
            or sidecar.get("contract_sha256") != TRUSTED_CONTRACT_SHA256
            or sidecar.get("trusted_commitments") != TRUSTED_COMMITMENTS
            or sidecar.get("input_bundle_sha256") != input_bundle
            or not isinstance(sidecar.get("description"), str)
            or not sidecar.get("description")
            or not isinstance(sidecar.get("selection"), Mapping)
            or sidecar.get("no_filtering_by_sign_interval_or_winner") is not True
        ):
            raise ManuscriptProjectionError(
                f"publication sidecar envelope differs: {sidecar_name}"
            )
        artifact_hashes[artifact_name] = expected_artifact_sha
        artifact_bytes[artifact_name] = expected_artifact_bytes
        sidecar_hashes[artifact_name] = expected_sidecar_sha
        output_hashes[artifact_name] = expected_artifact_sha
        output_hashes[sidecar_name] = expected_sidecar_sha

    output_bundle = _bundle_hash(output_hashes)
    if (
        metadata.get("output_bundle_sha256") != output_bundle
        or commit.get("output_bundle_sha256") != output_bundle
    ):
        raise ManuscriptProjectionError("publication output-bundle commitment differs")
    return AuthenticatedPublication(
        root=root,
        artifact_hashes=artifact_hashes,
        artifact_bytes=artifact_bytes,
        sidecar_hashes=sidecar_hashes,
        metadata_sha256=_sha256_bytes(metadata_raw),
        commit_sha256=observed_commit_sha,
        input_bundle_sha256=input_bundle,
        output_bundle_sha256=output_bundle,
        final_m=final_m,
    )


def _projection_payloads(
    publication: AuthenticatedPublication,
) -> tuple[dict[str, Any], dict[str, Any]]:
    files: dict[str, Any] = {}
    destination_hashes: dict[str, str] = {}
    for source_name, destination in sorted(
        DESTINATION_MAPPING.items(), key=lambda item: item[1].as_posix()
    ):
        destination_text = _safe_relative(
            destination, label="projection destination"
        ).as_posix()
        files[destination_text] = {
            "source_artifact": source_name,
            "source_artifact_sha256": publication.artifact_hashes[source_name],
            "source_artifact_bytes": publication.artifact_bytes[source_name],
            "source_sidecar": source_name + ".metadata.json",
            "source_sidecar_sha256": publication.sidecar_hashes[source_name],
            "copy_mode": "byte_for_byte_regular_file_copy",
        }
        destination_hashes[destination_text] = publication.artifact_hashes[source_name]
    projection_bundle = _bundle_hash(destination_hashes)
    projector_sha = _sha256_bytes(
        _read_path_regular_nofollow(Path(__file__), label="manuscript projector")
    )
    metadata = {
        "schema_version": 1,
        "status": PROJECTION_STATUS,
        "artifact_class": PROJECTION_ARTIFACT_CLASS,
        "projector": "paper/project_budget_toxicity_calibration_publication_to_manuscript.py",
        "projector_sha256": projector_sha,
        "source_publication": {
            "metadata": PUBLICATION_METADATA_NAME,
            "metadata_sha256": publication.metadata_sha256,
            "commit": PUBLICATION_COMMIT_NAME,
            "commit_sha256": publication.commit_sha256,
            "externally_trusted_commit_sha256": publication.commit_sha256,
            "contract_sha256": TRUSTED_CONTRACT_SHA256,
            "adapter_sha256": TRUSTED_ADAPTER_SHA256,
            "input_bundle_sha256": publication.input_bundle_sha256,
            "output_bundle_sha256": publication.output_bundle_sha256,
            "baseline_identity_passed": True,
            "primary_estimand_count": 118,
            "policy_vs_cEI_calibration_did_row_count": 36,
            "final_M": publication.final_m,
        },
        "projection_rules": {
            "scientific_artifacts_decoded": False,
            "conclusion_text_generated_or_modified": False,
            "selected_by_outcome_sign_interval_or_winner": False,
            "source_artifact_count_authenticated": 20,
            "source_physical_file_count_authenticated": 42,
            "tex_fragment_count": 5,
            "pdf_figure_count": 5,
            "destination_mapping_frozen": True,
            "copy_is_byte_for_byte": True,
        },
        "files": files,
        "projected_file_count": 10,
        "projection_bundle_sha256": projection_bundle,
    }
    commit = {
        "schema_version": 1,
        "status": PROJECTION_COMMIT_STATUS,
        "metadata": PROJECTION_METADATA_RELATIVE.as_posix(),
        "metadata_sha256": _sha256_bytes(_canonical_json_bytes(metadata)),
        "artifact_class": PROJECTION_ARTIFACT_CLASS,
        "projected_file_count": 10,
        "projection_bundle_sha256": projection_bundle,
        "source_publication_commit_sha256": publication.commit_sha256,
        "source_publication_output_bundle_sha256": publication.output_bundle_sha256,
    }
    return metadata, commit


def _managed_relatives() -> tuple[Path, ...]:
    return tuple(
        [
            destination
            for _, destination in sorted(
                DESTINATION_MAPPING.items(), key=lambda item: item[1].as_posix()
            )
        ]
        + [PROJECTION_METADATA_RELATIVE, PROJECTION_COMMIT_RELATIVE]
    )


@dataclass
class ManuscriptAnchors:
    """Open directory handles that pin the intended manuscript inodes."""

    root_path: Path
    root_fd: int
    generated_fd: int
    root_identity: tuple[int, int]
    generated_identity: tuple[int, int]


@dataclass
class StageAnchors:
    """Private staging directories addressed only below the pinned root fd."""

    name: str
    root_fd: int
    generated_fd: int
    root_identity: tuple[int, int]
    generated_identity: tuple[int, int]


def _stat_identity(observed: os.stat_result) -> tuple[int, int]:
    return observed.st_dev, observed.st_ino


def _open_manuscript_anchors(manuscript_source: str | Path) -> ManuscriptAnchors:
    root_path = _guard_directory(
        manuscript_source, label="manuscript source directory"
    )
    root_fd: int | None = None
    generated_fd: int | None = None
    try:
        root_before = os.lstat(root_path)
        root_fd = os.open(root_path, _directory_open_flags())
        root_opened = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_before.st_mode)
            or not stat.S_ISDIR(root_opened.st_mode)
            or _stat_identity(root_before) != _stat_identity(root_opened)
        ):
            raise ManuscriptProjectionError(
                "manuscript root changed while its directory handle was opened"
            )
        generated_before = os.stat(
            "generated", dir_fd=root_fd, follow_symlinks=False
        )
        if stat.S_ISLNK(generated_before.st_mode):
            raise ManuscriptProjectionError(
                "manuscript generated entry is a symlink"
            )
        if not stat.S_ISDIR(generated_before.st_mode):
            raise ManuscriptProjectionError(
                "manuscript generated entry is not a real directory"
            )
        generated_fd = os.open(
            "generated", _directory_open_flags(), dir_fd=root_fd
        )
        generated_opened = os.fstat(generated_fd)
        if (
            not stat.S_ISDIR(generated_opened.st_mode)
            or _stat_identity(generated_before) != _stat_identity(generated_opened)
        ):
            raise ManuscriptProjectionError(
                "manuscript generated directory changed while opening"
            )
        anchors = ManuscriptAnchors(
            root_path=root_path,
            root_fd=root_fd,
            generated_fd=generated_fd,
            root_identity=_stat_identity(root_opened),
            generated_identity=_stat_identity(generated_opened),
        )
        _revalidate_manuscript_anchors(anchors, label="initial manuscript anchor")
        return anchors
    except OSError as exc:
        if generated_fd is not None:
            os.close(generated_fd)
        if root_fd is not None:
            os.close(root_fd)
        raise ManuscriptProjectionError(
            "cannot safely anchor manuscript root/generated directories"
        ) from exc
    except Exception:
        if generated_fd is not None:
            os.close(generated_fd)
        if root_fd is not None:
            os.close(root_fd)
        raise


def _close_manuscript_anchors(anchors: ManuscriptAnchors) -> None:
    os.close(anchors.generated_fd)
    os.close(anchors.root_fd)


def _revalidate_manuscript_anchors(
    anchors: ManuscriptAnchors, *, label: str
) -> None:
    """Prove both lexical paths and held descriptors still name the same inodes."""

    link = _symlink_component(anchors.root_path)
    if link is not None:
        raise ManuscriptProjectionError(
            f"{label}: manuscript root acquired a symlink component: {link}"
        )
    try:
        root_opened = os.fstat(anchors.root_fd)
        generated_opened = os.fstat(anchors.generated_fd)
        root_lexical = os.lstat(anchors.root_path)
        fresh_root_fd = os.open(anchors.root_path, _directory_open_flags())
        try:
            root_fresh = os.fstat(fresh_root_fd)
        finally:
            os.close(fresh_root_fd)
        generated_entry = os.stat(
            "generated", dir_fd=anchors.root_fd, follow_symlinks=False
        )
        fresh_generated_fd = os.open(
            "generated", _directory_open_flags(), dir_fd=anchors.root_fd
        )
        try:
            generated_fresh = os.fstat(fresh_generated_fd)
        finally:
            os.close(fresh_generated_fd)
        # Re-read the lexical root after checking the generated child so a
        # rename/replacement between the two checks also fails closed.
        root_lexical_after = os.lstat(anchors.root_path)
    except OSError as exc:
        raise ManuscriptProjectionError(
            f"{label}: manuscript directory anchor cannot be revalidated"
        ) from exc
    for observed, expected, component in (
        (root_opened, anchors.root_identity, "held root"),
        (root_lexical, anchors.root_identity, "lexical root"),
        (root_fresh, anchors.root_identity, "fresh root"),
        (root_lexical_after, anchors.root_identity, "post-check lexical root"),
        (generated_opened, anchors.generated_identity, "held generated"),
        (generated_entry, anchors.generated_identity, "root/generated entry"),
        (generated_fresh, anchors.generated_identity, "fresh generated"),
    ):
        if not stat.S_ISDIR(observed.st_mode) or _stat_identity(observed) != expected:
            raise ManuscriptProjectionError(
                f"{label}: manuscript {component} inode changed"
            )


def _directory_for_managed_relative(
    anchors: ManuscriptAnchors, relative: Path
) -> tuple[int, str]:
    safe = _safe_relative(relative, label="managed target")
    if len(safe.parts) == 1:
        return anchors.root_fd, safe.name
    if len(safe.parts) == 2 and safe.parts[0] == "generated":
        return anchors.generated_fd, safe.parts[1]
    raise ManuscriptProjectionError(
        f"managed target is outside the frozen root/generated mapping: {safe}"
    )


def _directory_for_stage_relative(
    stage: StageAnchors, relative: Path
) -> tuple[int, str]:
    safe = _safe_relative(relative, label="staged target")
    if len(safe.parts) == 1:
        return stage.root_fd, safe.name
    if len(safe.parts) == 2 and safe.parts[0] == "generated":
        return stage.generated_fd, safe.parts[1]
    raise ManuscriptProjectionError(
        f"staged target is outside the frozen root/generated mapping: {safe}"
    )


def _read_regular_at(directory_fd: int, name: str, *, label: str) -> bytes:
    safe_name = _safe_basename(name, label=label)
    file_fd: int | None = None
    try:
        before = os.stat(safe_name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ManuscriptProjectionError(f"{label} is not a regular file")
        if before.st_nlink != 1:
            raise ManuscriptProjectionError(f"{label} is a hardlink")
        file_fd = os.open(
            safe_name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _stat_identity(opened) != _stat_identity(before)
        ):
            raise ManuscriptProjectionError(f"{label} changed before reading")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        raw = b"".join(chunks)
        if (
            after.st_nlink != 1
            or (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or len(raw) != after.st_size
        ):
            raise ManuscriptProjectionError(f"{label} changed while reading")
        return raw
    except OSError as exc:
        raise ManuscriptProjectionError(f"cannot safely read {label}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)


def _directory_entry_names_at(directory_fd: int, *, label: str) -> set[str]:
    """Return the physical DirEntry spellings from one pinned directory fd."""

    try:
        opened = os.fstat(directory_fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise ManuscriptProjectionError(f"{label} fd is not a directory")
        with os.scandir(directory_fd) as iterator:
            names = {entry.name for entry in iterator}
    except OSError as exc:
        raise ManuscriptProjectionError(f"cannot scan exact names in {label}") from exc
    return names


def _exact_managed_name_inventory(
    anchors: ManuscriptAnchors,
) -> tuple[set[str], set[str]]:
    return (
        _directory_entry_names_at(
            anchors.root_fd, label="manuscript root directory"
        ),
        _directory_entry_names_at(
            anchors.generated_fd, label="manuscript generated directory"
        ),
    )


def _transaction_residue_names(anchors: ManuscriptAnchors) -> tuple[str, ...]:
    root_names = _directory_entry_names_at(
        anchors.root_fd, label="manuscript root transaction-residue inventory"
    )
    lock_name = ".budget_toxicity_calibration_projection.install.lock"
    stage_prefix = ".budget_toxicity_calibration_projection.stage."
    return tuple(sorted(
        name
        for name in root_names
        if name == lock_name or name.startswith(stage_prefix)
    ))


def _target_state_anchored(
    anchors: ManuscriptAnchors,
) -> tuple[str, dict[Path, bytes]]:
    _revalidate_manuscript_anchors(anchors, label="target-state precheck")
    relatives = _managed_relatives()
    root_names, generated_names = _exact_managed_name_inventory(anchors)
    existing: list[Path] = []
    for relative in relatives:
        directory_fd, name = _directory_for_managed_relative(anchors, relative)
        physical_names = root_names if directory_fd == anchors.root_fd else generated_names
        case_variants = sorted(
            physical_name
            for physical_name in physical_names
            if physical_name != name and physical_name.casefold() == name.casefold()
        )
        if case_variants:
            raise ManuscriptProjectionError(
                "managed target has a case-folding filename collision: "
                f"expected {name!r}, observed {case_variants}"
            )
        if name not in physical_names:
            wrong_case = sorted(
                physical_name
                for physical_name in physical_names
                if physical_name.casefold() == name.casefold()
            )
            if wrong_case:
                raise ManuscriptProjectionError(
                    "managed target physical filename casing changed: "
                    f"expected {name!r}, observed {wrong_case}"
                )
            continue
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise ManuscriptProjectionError(
                f"managed target changed after exact-name inventory: {relative}"
            ) from exc
        except OSError as exc:
            raise ManuscriptProjectionError(
                f"cannot lstat managed target: {relative}"
            ) from exc
        existing.append(relative)
    if not existing:
        _revalidate_manuscript_anchors(anchors, label="absent target-state postcheck")
        return "absent", {}
    if len(existing) != len(relatives):
        raise ManuscriptProjectionError(
            "manuscript projection target is partial; immutable overwrite refused: "
            f"present={[path.as_posix() for path in existing]}"
        )
    payloads: dict[Path, bytes] = {}
    for relative in relatives:
        directory_fd, name = _directory_for_managed_relative(anchors, relative)
        payloads[relative] = _read_regular_at(
            directory_fd,
            name,
            label=f"existing managed target {relative.as_posix()}",
        )
    root_names_after, generated_names_after = _exact_managed_name_inventory(anchors)
    for relative in relatives:
        directory_fd, name = _directory_for_managed_relative(anchors, relative)
        physical_names = (
            root_names_after
            if directory_fd == anchors.root_fd
            else generated_names_after
        )
        case_variants = sorted(
            physical_name
            for physical_name in physical_names
            if physical_name != name and physical_name.casefold() == name.casefold()
        )
        if case_variants:
            raise ManuscriptProjectionError(
                "managed target acquired a case-folding filename collision: "
                f"expected {name!r}, observed {case_variants}"
            )
        if name not in physical_names:
            raise ManuscriptProjectionError(
                "managed target exact physical filename changed while reading: "
                f"{relative.as_posix()}"
            )
    _revalidate_manuscript_anchors(anchors, label="complete target-state postcheck")
    return "complete", payloads


def _create_stage(anchors: ManuscriptAnchors) -> StageAnchors:
    _revalidate_manuscript_anchors(anchors, label="stage-creation precheck")
    stage_name = ""
    for _ in range(100):
        candidate = (
            ".budget_toxicity_calibration_projection.stage."
            + secrets.token_hex(12)
        )
        try:
            os.mkdir(candidate, 0o700, dir_fd=anchors.root_fd)
            stage_name = candidate
            break
        except FileExistsError:
            continue
        except OSError as exc:
            raise ManuscriptProjectionError(
                "cannot create private manuscript projection stage"
            ) from exc
    if not stage_name:
        raise ManuscriptProjectionError("cannot allocate a unique projection stage")
    stage_fd: int | None = None
    generated_fd: int | None = None
    stage_identity: tuple[int, int] | None = None
    generated_identity: tuple[int, int] | None = None
    try:
        stage_entry = os.stat(
            stage_name, dir_fd=anchors.root_fd, follow_symlinks=False
        )
        stage_fd = os.open(
            stage_name, _directory_open_flags(), dir_fd=anchors.root_fd
        )
        stage_opened = os.fstat(stage_fd)
        if (
            not stat.S_ISDIR(stage_entry.st_mode)
            or _stat_identity(stage_entry) != _stat_identity(stage_opened)
        ):
            raise ManuscriptProjectionError("private projection stage inode changed")
        stage_identity = _stat_identity(stage_opened)
        os.mkdir("generated", 0o700, dir_fd=stage_fd)
        generated_entry = os.stat(
            "generated", dir_fd=stage_fd, follow_symlinks=False
        )
        generated_fd = os.open(
            "generated", _directory_open_flags(), dir_fd=stage_fd
        )
        generated_opened = os.fstat(generated_fd)
        if (
            not stat.S_ISDIR(generated_entry.st_mode)
            or _stat_identity(generated_entry) != _stat_identity(generated_opened)
        ):
            raise ManuscriptProjectionError(
                "private projection generated-stage inode changed"
            )
        generated_identity = _stat_identity(generated_opened)
        stage = StageAnchors(
            name=stage_name,
            root_fd=stage_fd,
            generated_fd=generated_fd,
            root_identity=_stat_identity(stage_opened),
            generated_identity=_stat_identity(generated_opened),
        )
        _revalidate_stage(anchors, stage, label="stage-creation postcheck")
        return stage
    except Exception:
        if generated_fd is not None:
            os.close(generated_fd)
        if stage_fd is not None:
            try:
                generated_current = os.stat(
                    "generated", dir_fd=stage_fd, follow_symlinks=False
                )
                if (
                    generated_identity is not None
                    and stat.S_ISDIR(generated_current.st_mode)
                    and _stat_identity(generated_current) == generated_identity
                ):
                    os.rmdir("generated", dir_fd=stage_fd)
            except OSError:
                pass
            os.close(stage_fd)
        try:
            stage_current = os.stat(
                stage_name, dir_fd=anchors.root_fd, follow_symlinks=False
            )
            if (
                stage_identity is not None
                and stat.S_ISDIR(stage_current.st_mode)
                and _stat_identity(stage_current) == stage_identity
            ):
                os.rmdir(stage_name, dir_fd=anchors.root_fd)
        except OSError:
            pass
        raise


def _revalidate_stage(
    anchors: ManuscriptAnchors, stage: StageAnchors, *, label: str
) -> None:
    _revalidate_manuscript_anchors(anchors, label=f"{label} manuscript")
    try:
        root_opened = os.fstat(stage.root_fd)
        root_entry = os.stat(
            stage.name, dir_fd=anchors.root_fd, follow_symlinks=False
        )
        generated_opened = os.fstat(stage.generated_fd)
        generated_entry = os.stat(
            "generated", dir_fd=stage.root_fd, follow_symlinks=False
        )
    except OSError as exc:
        raise ManuscriptProjectionError(f"{label}: stage cannot be revalidated") from exc
    for observed, expected, component in (
        (root_opened, stage.root_identity, "held stage root"),
        (root_entry, stage.root_identity, "root stage entry"),
        (generated_opened, stage.generated_identity, "held generated stage"),
        (generated_entry, stage.generated_identity, "stage generated entry"),
    ):
        if not stat.S_ISDIR(observed.st_mode) or _stat_identity(observed) != expected:
            raise ManuscriptProjectionError(f"{label}: {component} inode changed")


def _atomic_write_new_at(
    directory_fd: int, name: str, payload: bytes, *, label: str
) -> None:
    safe_name = _safe_basename(name, label=label)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        file_fd = os.open(safe_name, flags, 0o644, dir_fd=directory_fd)
    except OSError as exc:
        raise ManuscriptProjectionError(f"cannot create staged file: {label}") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(file_fd, view)
            if written <= 0:
                raise ManuscriptProjectionError(f"short write while staging: {label}")
            view = view[written:]
        os.fsync(file_fd)
    finally:
        os.close(file_fd)


def _fsync_fd(directory_fd: int, *, label: str) -> None:
    try:
        observed = os.fstat(directory_fd)
        if not stat.S_ISDIR(observed.st_mode):
            raise ManuscriptProjectionError(f"{label} fd is not a directory")
        os.fsync(directory_fd)
    except OSError as exc:
        raise ManuscriptProjectionError(f"cannot fsync {label}") from exc


def _rollback_identity_at(
    directory_fd: int, name: str, identity: tuple[int, int]
) -> None:
    try:
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except (FileNotFoundError, OSError):
        return
    if stat.S_ISREG(observed.st_mode) and _stat_identity(observed) == identity:
        try:
            os.unlink(name, dir_fd=directory_fd)
        except OSError:
            pass


def _install_new_file_exclusive_at(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> tuple[int, int]:
    """Publish one stage inode using only anchored *at operations."""

    source_name = _safe_basename(source_name, label="staged install source")
    destination_name = _safe_basename(
        destination_name, label="managed install destination"
    )
    staged_identity: tuple[int, int] | None = None
    linked = False
    try:
        staged = os.stat(
            source_name, dir_fd=source_directory_fd, follow_symlinks=False
        )
        if not stat.S_ISREG(staged.st_mode) or staged.st_nlink != 1:
            raise ManuscriptProjectionError(
                "staged projection file is not regular/single-link"
            )
        staged_identity = _stat_identity(staged)
        os.link(
            source_name,
            destination_name,
            src_dir_fd=source_directory_fd,
            dst_dir_fd=destination_directory_fd,
            follow_symlinks=False,
        )
        linked = True
        os.unlink(source_name, dir_fd=source_directory_fd)
        installed = os.stat(
            destination_name,
            dir_fd=destination_directory_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        if linked and staged_identity is not None:
            _rollback_identity_at(
                destination_directory_fd, destination_name, staged_identity
            )
        raise ManuscriptProjectionError(
            f"exclusive projection install failed: {destination_name}"
        ) from exc
    if not stat.S_ISREG(installed.st_mode) or installed.st_nlink != 1:
        if staged_identity is not None:
            _rollback_identity_at(
                destination_directory_fd, destination_name, staged_identity
            )
        raise ManuscriptProjectionError(
            f"installed projection file is not regular/single-link: {destination_name}"
        )
    return _stat_identity(installed)


def _remove_stage(
    anchors: ManuscriptAnchors,
    stage: StageAnchors,
    relatives: Iterable[Path],
) -> bool:
    """Remove only entries still below the pinned private stage inodes."""

    cleaned = True
    for relative in relatives:
        directory_fd, name = _directory_for_stage_relative(stage, relative)
        try:
            observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError:
            cleaned = False
            continue
        if stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except OSError:
                cleaned = False
        else:
            cleaned = False
    os.close(stage.generated_fd)
    try:
        generated_entry = os.stat(
            "generated", dir_fd=stage.root_fd, follow_symlinks=False
        )
        if (
            stat.S_ISDIR(generated_entry.st_mode)
            and _stat_identity(generated_entry) == stage.generated_identity
        ):
            os.rmdir("generated", dir_fd=stage.root_fd)
        else:
            cleaned = False
    except FileNotFoundError:
        pass
    except OSError:
        cleaned = False
    os.close(stage.root_fd)
    try:
        stage_entry = os.stat(
            stage.name, dir_fd=anchors.root_fd, follow_symlinks=False
        )
        if (
            stat.S_ISDIR(stage_entry.st_mode)
            and _stat_identity(stage_entry) == stage.root_identity
        ):
            os.rmdir(stage.name, dir_fd=anchors.root_fd)
        else:
            cleaned = False
    except FileNotFoundError:
        pass
    except OSError:
        cleaned = False
    try:
        os.stat(stage.name, dir_fd=anchors.root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return cleaned
    except OSError:
        return False
    return False


def _remove_lock(
    anchors: ManuscriptAnchors,
    lock_name: str,
    lock_identity: tuple[int, int] | None,
) -> bool:
    if lock_identity is None:
        return True
    try:
        observed = os.stat(
            lock_name, dir_fd=anchors.root_fd, follow_symlinks=False
        )
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if (
        not stat.S_ISDIR(observed.st_mode)
        or _stat_identity(observed) != lock_identity
    ):
        return False
    try:
        os.rmdir(lock_name, dir_fd=anchors.root_fd)
    except OSError:
        return False
    try:
        os.stat(lock_name, dir_fd=anchors.root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def project_publication_to_manuscript(
    *,
    publication_dir: str | Path,
    manuscript_source: str | Path,
    trusted_publication_commit_sha256: str,
) -> dict[str, Any]:
    """Authenticate, stage, and immutably install the fixed ten-file projection."""

    publication = authenticate_publication(
        publication_dir,
        trusted_publication_commit_sha256=trusted_publication_commit_sha256,
    )
    anchors = _open_manuscript_anchors(manuscript_source)
    try:
        manuscript_root = anchors.root_path
        if (
            _is_within(manuscript_root, publication.root)
            or _is_within(publication.root, manuscript_root)
        ):
            raise ManuscriptProjectionError(
                "publication and manuscript directories must not overlap"
            )
        metadata, commit = _projection_payloads(publication)
        expected_payloads: dict[Path, bytes] = {}
        for source_name, relative in DESTINATION_MAPPING.items():
            raw = _read_regular_nofollow(
                publication.root,
                source_name,
                label=f"selected publication artifact {source_name}",
            )
            if (
                _sha256_bytes(raw) != publication.artifact_hashes[source_name]
                or len(raw) != publication.artifact_bytes[source_name]
            ):
                raise ManuscriptProjectionError(
                    f"selected artifact changed after authentication: {source_name}"
                )
            expected_payloads[relative] = raw
        expected_payloads[PROJECTION_METADATA_RELATIVE] = _canonical_json_bytes(
            metadata
        )
        expected_payloads[PROJECTION_COMMIT_RELATIVE] = _canonical_json_bytes(commit)

        _revalidate_manuscript_anchors(
            anchors, label="initial transaction-residue precheck"
        )
        initial_residue = _transaction_residue_names(anchors)
        if initial_residue:
            raise ManuscriptProjectionError(
                "manuscript source has stale transaction residue before projection; "
                f"manual audit/cleanup required: {list(initial_residue)}"
            )
        _revalidate_manuscript_anchors(
            anchors, label="initial transaction-residue postcheck"
        )
        state, existing = _target_state_anchored(anchors)
        if state == "complete":
            differing = [
                relative.as_posix()
                for relative in _managed_relatives()
                if existing[relative] != expected_payloads[relative]
            ]
            if differing:
                raise ManuscriptProjectionError(
                    "manuscript projection target differs; immutable overwrite refused: "
                    f"{differing}"
                )
            residue = _transaction_residue_names(anchors)
            if residue:
                raise ManuscriptProjectionError(
                    "complete manuscript projection has stale transaction residue; "
                    f"manual audit/cleanup required: {list(residue)}"
                )
            _revalidate_manuscript_anchors(
                anchors, label="idempotent-return precheck"
            )
            return {
                "output_dir": manuscript_root,
                "metadata": manuscript_root / PROJECTION_METADATA_RELATIVE,
                "commit": manuscript_root / PROJECTION_COMMIT_RELATIVE,
                "projected_file_count": 10,
                "idempotent_existing": True,
            }

        stage = _create_stage(anchors)
        lock_name = ".budget_toxicity_calibration_projection.install.lock"
        lock_identity: tuple[int, int] | None = None
        installed: dict[Path, tuple[int, int]] = {}
        try:
            for relative, payload in expected_payloads.items():
                _revalidate_stage(
                    anchors, stage, label=f"stage-write {relative.as_posix()} precheck"
                )
                stage_directory_fd, stage_name = _directory_for_stage_relative(
                    stage, relative
                )
                _atomic_write_new_at(
                    stage_directory_fd,
                    stage_name,
                    payload,
                    label=relative.as_posix(),
                )
                staged = _read_regular_at(
                    stage_directory_fd,
                    stage_name,
                    label=f"staged projection {relative.as_posix()}",
                )
                if staged != payload:
                    raise ManuscriptProjectionError(
                        f"staged projection differs: {relative.as_posix()}"
                    )
                _revalidate_stage(
                    anchors, stage, label=f"stage-write {relative.as_posix()} postcheck"
                )
            try:
                os.mkdir(lock_name, 0o700, dir_fd=anchors.root_fd)
                lock_entry = os.stat(
                    lock_name,
                    dir_fd=anchors.root_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ManuscriptProjectionError(
                    "another or stale manuscript-projection install lock exists"
                ) from exc
            if not stat.S_ISDIR(lock_entry.st_mode):
                raise ManuscriptProjectionError(
                    "manuscript projection install lock is not a directory"
                )
            lock_identity = _stat_identity(lock_entry)
            _revalidate_stage(anchors, stage, label="post-lock anchor check")
            state_after_lock, _ = _target_state_anchored(anchors)
            if state_after_lock != "absent":
                raise ManuscriptProjectionError(
                    "manuscript projection target changed during staging"
                )

            precommit_order = [
                relative
                for relative in _managed_relatives()
                if relative != PROJECTION_COMMIT_RELATIVE
            ]
            for relative in precommit_order:
                _revalidate_stage(
                    anchors, stage, label=f"install {relative.as_posix()} precheck"
                )
                source_fd, source_name = _directory_for_stage_relative(stage, relative)
                destination_fd, destination_name = _directory_for_managed_relative(
                    anchors, relative
                )
                identity = _install_new_file_exclusive_at(
                    source_fd, source_name, destination_fd, destination_name
                )
                installed[relative] = identity
                _revalidate_stage(
                    anchors, stage, label=f"install {relative.as_posix()} postcheck"
                )

            # Durability barrier for all ten projected artifacts plus metadata.
            # Only after both destination directories are durable may the
            # commit marker become visible.
            _revalidate_stage(anchors, stage, label="precommit fsync precheck")
            _fsync_fd(anchors.generated_fd, label="manuscript generated directory")
            _fsync_fd(anchors.root_fd, label="manuscript root directory")
            _revalidate_stage(anchors, stage, label="precommit fsync postcheck")

            commit_relative = PROJECTION_COMMIT_RELATIVE
            source_fd, source_name = _directory_for_stage_relative(
                stage, commit_relative
            )
            destination_fd, destination_name = _directory_for_managed_relative(
                anchors, commit_relative
            )
            _revalidate_stage(anchors, stage, label="commit-install precheck")
            installed[commit_relative] = _install_new_file_exclusive_at(
                source_fd, source_name, destination_fd, destination_name
            )
            _revalidate_manuscript_anchors(
                anchors, label="commit-install postcheck"
            )
            _fsync_fd(
                anchors.generated_fd,
                label="manuscript generated directory after commit",
            )
            _revalidate_manuscript_anchors(
                anchors, label="commit-fsync postcheck"
            )

            final_state, final_payloads = _target_state_anchored(anchors)
            if final_state != "complete" or any(
                final_payloads[relative] != expected_payloads[relative]
                for relative in _managed_relatives()
            ):
                raise ManuscriptProjectionError(
                    "installed manuscript projection differs"
                )

            # Cleanup also occurs through pinned descriptors.  Both sides of
            # cleanup are identity checks, so a renamed/replaced lexical root
            # cannot be accepted merely because the writes themselves used an
            # old open directory descriptor.
            _revalidate_manuscript_anchors(
                anchors, label="successful-cleanup precheck"
            )
            if not _remove_lock(anchors, lock_name, lock_identity):
                lock_identity = None
                raise PostCommitCleanupError(
                    "post-commit install-lock cleanup failed; the durable "
                    "verified projection is retained and stale lock preserved "
                    "for audit"
                )
            lock_identity = None
            stage_to_remove = stage
            stage = None  # type: ignore[assignment]
            if not _remove_stage(anchors, stage_to_remove, expected_payloads):
                raise PostCommitCleanupError(
                    "post-commit stage cleanup failed; the durable verified "
                    "projection is retained and stage residue preserved for audit"
                )
            _revalidate_manuscript_anchors(
                anchors, label="successful-cleanup postcheck"
            )
        except Exception as exc:
            if not isinstance(exc, PostCommitCleanupError):
                for relative, identity in reversed(list(installed.items())):
                    destination_fd, destination_name = _directory_for_managed_relative(
                        anchors, relative
                    )
                    _rollback_identity_at(
                        destination_fd, destination_name, identity
                    )
                try:
                    _fsync_fd(
                        anchors.generated_fd,
                        label="generated directory after rollback",
                    )
                    _fsync_fd(
                        anchors.root_fd,
                        label="root directory after rollback",
                    )
                except ManuscriptProjectionError:
                    pass
            raise
        finally:
            cleanup_anchor_error: ManuscriptProjectionError | None = None
            try:
                _revalidate_manuscript_anchors(
                    anchors, label="transaction-cleanup precheck"
                )
            except ManuscriptProjectionError as exc:
                cleanup_anchor_error = exc
            if lock_identity is not None:
                if (
                    not _remove_lock(anchors, lock_name, lock_identity)
                    and cleanup_anchor_error is None
                ):
                    cleanup_anchor_error = ManuscriptProjectionError(
                        "manuscript projection install-lock cleanup failed; "
                        "stale lock preserved for audit"
                    )
            if stage is not None:
                stage_to_remove = stage
                stage = None  # type: ignore[assignment]
                if (
                    not _remove_stage(
                        anchors, stage_to_remove, expected_payloads
                    )
                    and cleanup_anchor_error is None
                ):
                    cleanup_anchor_error = ManuscriptProjectionError(
                        "manuscript projection stage cleanup failed; "
                        "stage residue preserved for audit"
                    )
            try:
                _revalidate_manuscript_anchors(
                    anchors, label="transaction-cleanup postcheck"
                )
            except ManuscriptProjectionError as exc:
                if cleanup_anchor_error is None:
                    cleanup_anchor_error = exc
            if cleanup_anchor_error is not None:
                raise cleanup_anchor_error

        _revalidate_manuscript_anchors(anchors, label="projection-return precheck")
        return {
            "output_dir": manuscript_root,
            "metadata": manuscript_root / PROJECTION_METADATA_RELATIVE,
            "commit": manuscript_root / PROJECTION_COMMIT_RELATIVE,
            "projected_file_count": 10,
            "idempotent_existing": False,
        }
    finally:
        _close_manuscript_anchors(anchors)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publication-dir", type=Path, required=True,
        help="flat committed 42-file publication directory",
    )
    parser.add_argument(
        "--trusted-publication-commit-sha256",
        required=True,
        help=(
            "externally frozen SHA-256 of the exact publication commit bytes; "
            "the publication directory cannot supply its own trust root"
        ),
    )
    parser.add_argument(
        "--manuscript-source", type=Path,
        help="manuscript source root containing a real generated/ directory",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="install the immutable ten-file projection (default: authenticate only)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.execute:
        if args.manuscript_source is None:
            raise SystemExit("--manuscript-source is required with --execute")
        result = project_publication_to_manuscript(
            publication_dir=args.publication_dir,
            manuscript_source=args.manuscript_source,
            trusted_publication_commit_sha256=(
                args.trusted_publication_commit_sha256
            ),
        )
        print(f"projection commit: {result['commit']}")
        print(f"idempotent existing: {result['idempotent_existing']}")
    else:
        publication = authenticate_publication(
            args.publication_dir,
            trusted_publication_commit_sha256=(
                args.trusted_publication_commit_sha256
            ),
        )
        print(f"authenticated publication: {publication.root}")
        print("artifact count: 20")
        print("physical file count: 42")


if __name__ == "__main__":
    main()
