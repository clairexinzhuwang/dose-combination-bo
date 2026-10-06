#!/usr/bin/env python3
"""Finalize a distinct M0500 raw-master index after a result-blind operator cap.

The frozen M0500 precision decision is never edited.  If it requests M1000,
that original ``next_M=1000`` remains in the capped master while a separate
operator-cap record sets the effective transition to null and enumerates every
unresolved guarded precision target.  If M0500 passes, this program refuses to
write and the original runner finalization path must be used.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence

import zstandard as zstd

try:
    from paper import create_budget_toxicity_calibration_operator_cap_amendment as amendment
    from paper import monitor_budget_toxicity_calibration_precision as monitor
    from paper import run_budget_toxicity_calibration_audit as runner
except ModuleNotFoundError:  # Direct execution from paper/.
    import create_budget_toxicity_calibration_operator_cap_amendment as amendment
    import monitor_budget_toxicity_calibration_precision as monitor
    import run_budget_toxicity_calibration_audit as runner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING = ROOT / "results/budget_toxicity_calibration_staging"
DEFAULT_AMENDMENT = amendment.DEFAULT_OUTPUT / amendment.ARTIFACT_NAME
ARTIFACT_NAME = (
    "budget_toxicity_calibration_raw_master_operator_capped_M0500.json.zst"
)
STATUS = "TERMINAL_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_OPERATOR_CAP"
COMMIT_STATUS = (
    "COMMITTED_TERMINAL_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_OPERATOR_CAP"
)
ARTIFACT_CLASS = "authenticated_operator_capped_index_over_frozen_seed_block_indexes"
STORAGE_MODEL = "authenticated_index_over_seed_block_indexes_plus_operator_cap"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MCSE_THRESHOLD_POINTS = 1.5

EXPECTED_SOURCE_HASHES = {
    "paper/analyze_budget_toxicity_calibration_audit.py": (
        "bcf3ec2964b88d61ac4becc8a9aa838831a3240d93a082abf42b5b328799ec2a"
    ),
    "paper/budget_toxicity_calibration_manifest.json": (
        "2e189ff6426bd3ad12fd29896fb6f76027a0d9e4424013d92edd0df3d143d3b6"
    ),
    "paper/budget_toxicity_calibration_prespec.md": (
        "1b1152d8f09462b96bf2f35252a9cd5703b331ab7a3546d4708b189f81e8c178"
    ),
    "paper/monitor_budget_toxicity_calibration_precision.py": (
        "3994535bc1e2c300388c2d161d326e66f6185452971075e8d7bc9caf76bb07e0"
    ),
    "paper/run_budget_toxicity_calibration_audit.py": (
        "c36420dc8a37c8a538ecba3b8d5d82753d428fb0eb4074dbe97baf83e474365d"
    ),
}

PAYLOAD_FIELDS = frozenset({
    "schema_version", "status", "artifact_class", "storage_model", "final_M",
    "cumulative_block_ids", "path_count", "snapshot_count", "bindings",
    "baseline_identity", "block_commits", "precision_decisions", "top_up_history",
    "operator_cap", "canonical_sort_key",
})
OPERATOR_CAP_FIELDS = frozenset({
    "cap_cumulative_M", "operator_cap_reached", "effective_next_M",
    "m1000_executed", "amendment", "original_final_precision_state",
    "precision_target_achieved", "terminated_with_unresolved_precision",
    "terminal_reason", "guarded_mcse_threshold_points",
    "unresolved_estimand_count", "unresolved_guarded_precision_targets",
    "original_precision_artifacts_immutable", "cap_finalizer_sha256",
})
FINAL_PRECISION_FIELDS = frozenset({
    "cumulative_M", "next_M", "all_estimands_pass_guarded_1_5pp",
    "top_up_limit_reached", "maximum_raw_mcse_points",
    "maximum_guarded_mcse_points",
})
AMENDMENT_INDEX_FIELDS = frozenset({
    "artifact", "artifact_sha256", "metadata", "metadata_sha256", "commit",
    "commit_sha256", "audit_id", "launch_fingerprint",
})
METADATA_FIELDS = frozenset({
    "schema_version", "status", "artifact_class", "storage_model", "artifact",
    "artifact_sha256", "artifact_bytes", "uncompressed_sha256",
    "uncompressed_bytes", "final_M", "path_count", "snapshot_count", "bindings",
    "operator_cap_summary", "amendment_bundle_sha256", "precision_bundle_sha256",
    "cap_finalizer_sha256", "compression", "environment", "generated_at_utc",
    "immutable",
})
COMMIT_FIELDS = frozenset({
    "schema_version", "status", "artifact", "artifact_sha256",
    "uncompressed_sha256", "metadata", "metadata_sha256", "final_M",
    "path_count", "snapshot_count", "launch_fingerprint",
    "amendment_bundle_sha256", "precision_bundle_sha256",
    "cap_finalizer_sha256", "immutable", "committed_at_utc",
})


class OperatorCappedMasterError(RuntimeError):
    """Raised when an operator-capped master is invalid or unsafe to write."""


class OriginalFinalizationRequired(OperatorCappedMasterError):
    """Raised when M0500 passed and the original frozen chain must be used."""


@dataclass(frozen=True)
class OperatorCappedMaster:
    payload: dict[str, Any]
    provenance: dict[str, Any]
    staging_root: Path
    block_payloads: tuple[dict[str, Any], ...]
    precision_payloads: tuple[dict[str, Any], ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _finite_json(raw: bytes, *, label: str) -> Any:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite token {token}")

    try:
        value = json.loads(raw, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise OperatorCappedMasterError(f"{label} is not finite valid JSON") from exc
    _assert_finite(value, label=label)
    return value


def _assert_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise OperatorCappedMasterError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, label=f"{label}[{index}]")


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise OperatorCappedMasterError(f"{label} is not a lowercase SHA-256")
    return value


def _regular_no_symlink(path: Path, *, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise OperatorCappedMasterError(f"{label} is absent: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise OperatorCappedMasterError(f"{label} is not a regular non-symlink file")
    return path.resolve(strict=True)


def _lexical_chain_no_symlink(path: Path, root: Path, *, label: str) -> Path:
    root_resolved = root.resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(os.path.abspath(candidate))
    candidate = Path(os.path.abspath(candidate))
    if not candidate.is_relative_to(root_resolved):
        raise OperatorCappedMasterError(f"{label} escapes its authenticated root")
    current = root_resolved
    for component in candidate.relative_to(root_resolved).parts:
        if component in ("", ".", ".."):
            raise OperatorCappedMasterError(f"{label} has a non-canonical component")
        current = current / component
        if current.is_symlink():
            raise OperatorCappedMasterError(f"{label} crosses a symlink")
    return candidate


def _assert_inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    root_resolved = root.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise OperatorCappedMasterError(f"{label} escapes its authenticated root")
    current = path
    while current.resolve(strict=False) != root_resolved:
        if current.exists() and stat.S_ISLNK(current.lstat().st_mode):
            raise OperatorCappedMasterError(f"{label} crosses a symlink")
        if current.parent == current:
            raise OperatorCappedMasterError(f"{label} has no authenticated ancestor")
        current = current.parent
    return resolved


def _utc_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise OperatorCappedMasterError(f"{label} must be an explicit UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise OperatorCappedMasterError(f"{label} is invalid") from exc
    if parsed.utcoffset() != timedelta(0):
        raise OperatorCappedMasterError(f"{label} is not UTC")
    return parsed


def _aware_utc_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise OperatorCappedMasterError(f"{label} is absent")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OperatorCappedMasterError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise OperatorCappedMasterError(f"{label} is not explicit UTC")
    return parsed


def _validate_environment(value: Any, *, generated_at: datetime) -> None:
    if not isinstance(value, Mapping):
        raise OperatorCappedMasterError("capped-master environment is absent")
    observed = dict(value)
    captured_at = _aware_utc_timestamp(
        observed.pop("captured_at_utc", None), label="environment.captured_at_utc"
    )
    expected = runner._privacy_safe_environment()
    expected.pop("captured_at_utc", None)
    if observed != expected:
        raise OperatorCappedMasterError("capped-master environment changed")
    if captured_at > generated_at or generated_at - captured_at > timedelta(minutes=5):
        raise OperatorCappedMasterError("capped-master environment timestamp is inconsistent")


def _atomic_write_new(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise OperatorCappedMasterError(f"immutable destination already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sidecars(path: Path) -> tuple[Path, Path]:
    return Path(str(path) + ".metadata.json"), Path(str(path) + ".commit.json")


def capped_master_paths(
    staging_root: str | Path = DEFAULT_STAGING,
) -> tuple[Path, Path, Path]:
    artifact_path = Path(staging_root) / "raw" / ARTIFACT_NAME
    metadata_path, commit_path = _sidecars(artifact_path)
    return artifact_path, metadata_path, commit_path


def _assert_runtime_and_sources() -> None:
    try:
        monitor._assert_exact_runtime()
    except BaseException as exc:
        raise OperatorCappedMasterError("exact frozen runtime check failed") from exc
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        path = _regular_no_symlink(ROOT / relative, label=f"frozen source {relative}")
        if _sha256_file(path) != expected:
            raise OperatorCappedMasterError(f"frozen source hash changed: {relative}")


def _assert_no_m1000(staging: Path) -> None:
    forbidden = sorted(path for path in staging.rglob("*M1000*") if path.exists())
    if forbidden:
        raise OperatorCappedMasterError("M1000 evidence exists despite the operator cap")


def _index_amendment(authenticated: Mapping[str, Any]) -> dict[str, Any]:
    paths = authenticated["paths"]
    hashes = authenticated["hashes"]
    root = ROOT.resolve(strict=True)
    index = {
        "artifact": Path(paths["artifact"]).relative_to(root).as_posix(),
        "artifact_sha256": hashes["artifact"],
        "metadata": Path(paths["metadata"]).relative_to(root).as_posix(),
        "metadata_sha256": hashes["metadata"],
        "commit": Path(paths["commit"]).relative_to(root).as_posix(),
        "commit_sha256": hashes["commit"],
        "audit_id": authenticated["payload"]["audit_id"],
        "launch_fingerprint": authenticated["payload"]["launch_fingerprint"],
    }
    if set(index) != AMENDMENT_INDEX_FIELDS:
        raise OperatorCappedMasterError("internal amendment index fields changed")
    return index


def validate_terminal_precision_state(
    m0200: Mapping[str, Any], m0500: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate only the blinded precision transition and build cap disclosure."""

    if m0200.get("cumulative_M") != 200 or m0200.get("next_M") != 500:
        raise OperatorCappedMasterError("M0200 decision does not justify M0500")
    if m0500.get("cumulative_M") != 500:
        raise OperatorCappedMasterError("terminal precision decision is not M0500")
    if m0500.get("top_up_limit_reached") is not False:
        raise OperatorCappedMasterError(
            "M0500 top_up_limit_reached must remain false under the original schema"
        )
    all_pass = m0500.get("all_estimands_pass_guarded_1_5pp")
    if type(all_pass) is not bool:
        raise OperatorCappedMasterError("M0500 all-pass flag is not boolean")
    expected_next = None if all_pass else 1000
    if m0500.get("next_M") != expected_next:
        raise OperatorCappedMasterError(
            "M0500 original next_M disagrees with its immutable precision flag"
        )
    for name in ("maximum_raw_mcse_points", "maximum_guarded_mcse_points"):
        value = m0500.get(name)
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise OperatorCappedMasterError(f"M0500 {name} is invalid")
    estimands = m0500.get("estimands")
    if not isinstance(estimands, list) or len(estimands) != 118:
        raise OperatorCappedMasterError("M0500 must retain all 118 blind estimands")
    unresolved: list[dict[str, Any]] = []
    identifiers: list[str] = []
    for row in estimands:
        if not isinstance(row, Mapping):
            raise OperatorCappedMasterError("M0500 blind estimand is not an object")
        identifier = row.get("estimand_id")
        if not isinstance(identifier, str) or not identifier:
            raise OperatorCappedMasterError("M0500 estimand identifier is invalid")
        identifiers.append(identifier)
        passed = row.get("passes_guarded_1_5pp")
        if type(passed) is not bool:
            raise OperatorCappedMasterError("M0500 estimand pass flag is not boolean")
        guarded = row.get("guarded_mcse_points")
        if type(guarded) is not float or not math.isfinite(guarded) or guarded < 0.0:
            raise OperatorCappedMasterError("M0500 guarded MCSE is invalid")
        if passed != (guarded <= MCSE_THRESHOLD_POINTS):
            raise OperatorCappedMasterError("M0500 threshold flag was altered")
        if not passed:
            unresolved.append(dict(row))
    if len(set(identifiers)) != 118:
        raise OperatorCappedMasterError("M0500 estimand identifiers are duplicated")
    if all_pass != (len(unresolved) == 0):
        raise OperatorCappedMasterError("M0500 all-pass flag disagrees with its rows")
    return {
        "original_final_precision_state": {
            "cumulative_M": 500,
            "next_M": expected_next,
            "all_estimands_pass_guarded_1_5pp": all_pass,
            "top_up_limit_reached": False,
            "maximum_raw_mcse_points": m0500["maximum_raw_mcse_points"],
            "maximum_guarded_mcse_points": m0500["maximum_guarded_mcse_points"],
        },
        "precision_target_achieved": all_pass,
        "terminated_with_unresolved_precision": not all_pass,
        "terminal_reason": (
            "PRECISION_TARGET_MET_AT_M0500"
            if all_pass
            else "OPERATOR_CAP_WITH_UNRESOLVED_PRECISION"
        ),
        "unresolved_estimand_count": len(unresolved),
        "unresolved_guarded_precision_targets": unresolved,
    }


def _precision_index(path: Path, payload: Mapping[str, Any], staging: Path) -> dict[str, Any]:
    metadata, commit = _sidecars(path)
    index = {
        "cumulative_M": int(payload["cumulative_M"]),
        "artifact": path.relative_to(staging).as_posix(),
        "artifact_sha256": _sha256_file(path),
        "metadata": metadata.relative_to(staging).as_posix(),
        "metadata_sha256": _sha256_file(metadata),
        "commit": commit.relative_to(staging).as_posix(),
        "commit_sha256": _sha256_file(commit),
        "next_M": payload["next_M"],
    }
    if set(index) != runner.PRECISION_INDEX_FIELDS:
        raise OperatorCappedMasterError("internal precision index fields changed")
    return index


def _authenticate_inputs(
    staging: Path, amendment_path: Path
) -> dict[str, Any]:
    _assert_runtime_and_sources()
    staging = _lexical_chain_no_symlink(staging, ROOT, label="staging root")
    staging = _assert_inside(staging, ROOT, label="staging root")
    if not staging.is_dir() or staging.is_symlink():
        raise OperatorCappedMasterError("staging root is not a regular directory")
    _assert_no_m1000(staging)
    manifest = runner._load_frozen_manifest()
    bindings = runner._build_bindings()
    if bindings.get("launch_fingerprint") != amendment.LAUNCH_FINGERPRINT:
        raise OperatorCappedMasterError("frozen launch fingerprint changed")
    authenticated_amendment = amendment.authenticate_amendment(
        amendment_path,
        root=ROOT,
        expected_launch_fingerprint=bindings["launch_fingerprint"],
        verify_bound_files=True,
    )
    amendment_index = _index_amendment(authenticated_amendment)
    cumulative_hashes: dict[str, str] = {
        "paper/budget_toxicity_calibration_manifest.json": bindings["manifest_sha256"],
        "paper/budget_toxicity_calibration_prespec.md": bindings["analysis_spec_sha256"],
    }
    hashes_by_m: dict[int, dict[str, str]] = {}
    block_commits: list[dict[str, Any]] = []
    block_payloads: list[dict[str, Any]] = []
    for block_id in ("M0200", "M0500_TOPUP"):
        groups = runner._group_jobs(runner._expand_jobs(manifest, block_id))
        artifact_path, _metadata, _commit, hashes, identity = (
            runner._validate_committed_block_envelope(
                staging, block_id=block_id, groups=groups, bindings=bindings
            )
        )
        block_payload, _block_hashes, _paths = monitor.authenticate_seed_block(
            artifact_path, verify_local_bindings=True
        )
        if block_payload["block"]["block_id"] != block_id:
            raise OperatorCappedMasterError("block order or identity changed")
        cumulative_hashes.update(hashes)
        cumulative_m = int(runner.BLOCK_BY_ID[block_id]["cumulative_M"])
        hashes_by_m[cumulative_m] = dict(sorted(cumulative_hashes.items()))
        block_commits.append(identity)
        block_payloads.append(dict(block_payload))
    baseline_identity, baseline_hashes = runner._load_baseline_identity(
        staging, expected_input_hashes=hashes_by_m[200]
    )
    for cumulative_m in (200, 500):
        hashes_by_m[cumulative_m] = dict(sorted({
            **hashes_by_m[cumulative_m], **baseline_hashes
        }.items()))
    expected_counts = {200: 80, 500: 155}
    if any(len(hashes_by_m[m]) != expected_counts[m] for m in (200, 500)):
        raise OperatorCappedMasterError("precision input-hash count changed")
    precision_payloads: list[dict[str, Any]] = []
    precision_indexes: list[dict[str, Any]] = []
    for cumulative_m in (200, 500):
        path = runner._precision_path(staging, cumulative_m)
        payload = monitor.load_blind_precision_decision(
            path,
            expected_m=cumulative_m,
            expected_input_hashes=hashes_by_m[cumulative_m],
        )
        precision_payloads.append(payload)
        precision_indexes.append(_precision_index(path, payload, staging))
    terminal_state = validate_terminal_precision_state(
        precision_payloads[0], precision_payloads[1]
    )
    top_up_history = [
        {
            "cumulative_M": index["cumulative_M"],
            "precision_artifact": index["artifact"],
            "precision_artifact_sha256": index["artifact_sha256"],
            "next_M": index["next_M"],
        }
        for index in precision_indexes
    ]
    runner._validate_master_indexes(
        staging,
        final_m=500,
        baseline_identity=baseline_identity,
        block_commits=block_commits,
        precision_decisions=precision_indexes,
        top_up_history=top_up_history,
    )
    return {
        "bindings": bindings,
        "baseline_identity": baseline_identity,
        "block_commits": block_commits,
        "block_payloads": block_payloads,
        "precision_payloads": precision_payloads,
        "precision_decisions": precision_indexes,
        "top_up_history": top_up_history,
        "amendment": amendment_index,
        "terminal_state": terminal_state,
    }


def _operator_cap_record(inputs: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        "cap_cumulative_M": 500,
        "operator_cap_reached": True,
        "effective_next_M": None,
        "m1000_executed": False,
        "amendment": dict(inputs["amendment"]),
        **dict(inputs["terminal_state"]),
        "guarded_mcse_threshold_points": MCSE_THRESHOLD_POINTS,
        "original_precision_artifacts_immutable": True,
        "cap_finalizer_sha256": _sha256_file(Path(__file__).resolve()),
    }
    validate_operator_cap_record(record)
    return record


def validate_operator_cap_record(record: Mapping[str, Any]) -> None:
    if set(record) != OPERATOR_CAP_FIELDS:
        raise OperatorCappedMasterError("operator-cap terminal fields changed")
    for field, expected in {
        "cap_cumulative_M": 500,
        "operator_cap_reached": True,
        "effective_next_M": None,
        "m1000_executed": False,
        "guarded_mcse_threshold_points": MCSE_THRESHOLD_POINTS,
        "original_precision_artifacts_immutable": True,
    }.items():
        if record.get(field) != expected:
            raise OperatorCappedMasterError(f"operator-cap terminal state changed: {field}")
    amendment_index = record.get("amendment")
    if not isinstance(amendment_index, Mapping) or set(amendment_index) != AMENDMENT_INDEX_FIELDS:
        raise OperatorCappedMasterError("operator-cap amendment index changed")
    for field in ("artifact_sha256", "metadata_sha256", "commit_sha256"):
        _require_sha(amendment_index.get(field), label=f"operator_cap.amendment.{field}")
    final = record.get("original_final_precision_state")
    if not isinstance(final, Mapping) or set(final) != FINAL_PRECISION_FIELDS:
        raise OperatorCappedMasterError("original terminal precision state changed")
    all_pass = final.get("all_estimands_pass_guarded_1_5pp")
    if type(all_pass) is not bool or final.get("cumulative_M") != 500:
        raise OperatorCappedMasterError("original terminal precision flags changed")
    if final.get("top_up_limit_reached") is not False:
        raise OperatorCappedMasterError("original M0500 top-up-limit flag was altered")
    if final.get("next_M") != (None if all_pass else 1000):
        raise OperatorCappedMasterError("original M0500 next_M was altered")
    unresolved = record.get("unresolved_guarded_precision_targets")
    if not isinstance(unresolved, list):
        raise OperatorCappedMasterError("unresolved precision list is absent")
    if type(record.get("unresolved_estimand_count")) is not int:
        raise OperatorCappedMasterError("unresolved precision count is invalid")
    if record["unresolved_estimand_count"] != len(unresolved):
        raise OperatorCappedMasterError("unresolved precision count/list disagree")
    if len({row.get("estimand_id") for row in unresolved if isinstance(row, Mapping)}) != len(unresolved):
        raise OperatorCappedMasterError("unresolved precision identifiers are duplicated")
    for row in unresolved:
        if not isinstance(row, Mapping) or row.get("passes_guarded_1_5pp") is not False:
            raise OperatorCappedMasterError("unresolved precision list contains a passing row")
        guarded = row.get("guarded_mcse_points")
        if type(guarded) is not float or not math.isfinite(guarded) or guarded <= MCSE_THRESHOLD_POINTS:
            raise OperatorCappedMasterError("unresolved precision threshold was altered")
    if record.get("precision_target_achieved") is not all_pass:
        raise OperatorCappedMasterError("precision-target flag differs from original decision")
    if record.get("terminated_with_unresolved_precision") is all_pass:
        raise OperatorCappedMasterError("unresolved termination flag differs from original decision")
    expected_reason = (
        "PRECISION_TARGET_MET_AT_M0500"
        if all_pass else "OPERATOR_CAP_WITH_UNRESOLVED_PRECISION"
    )
    if record.get("terminal_reason") != expected_reason:
        raise OperatorCappedMasterError("operator-cap terminal reason changed")
    if all_pass != (len(unresolved) == 0):
        raise OperatorCappedMasterError("operator-cap unresolved list disagrees with all-pass")
    _require_sha(record.get("cap_finalizer_sha256"), label="cap_finalizer_sha256")


def _build_payload(inputs: Mapping[str, Any]) -> dict[str, Any]:
    operator_cap = _operator_cap_record(inputs)
    payload = {
        "schema_version": 1,
        "status": STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "storage_model": STORAGE_MODEL,
        "final_M": 500,
        "cumulative_block_ids": ["M0200", "M0500_TOPUP"],
        "path_count": 24_000,
        "snapshot_count": 72_000,
        "bindings": dict(inputs["bindings"]),
        "baseline_identity": dict(inputs["baseline_identity"]),
        "block_commits": [dict(value) for value in inputs["block_commits"]],
        "precision_decisions": [
            dict(value) for value in inputs["precision_decisions"]
        ],
        "top_up_history": [dict(value) for value in inputs["top_up_history"]],
        "operator_cap": operator_cap,
        "canonical_sort_key": list(runner.CANONICAL_SORT_KEY),
    }
    validate_capped_payload(payload)
    return payload


def validate_capped_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != PAYLOAD_FIELDS:
        raise OperatorCappedMasterError("capped-master payload fields changed")
    for field, expected in {
        "schema_version": 1,
        "status": STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "storage_model": STORAGE_MODEL,
        "final_M": 500,
        "cumulative_block_ids": ["M0200", "M0500_TOPUP"],
        "path_count": 24_000,
        "snapshot_count": 72_000,
        "canonical_sort_key": list(runner.CANONICAL_SORT_KEY),
    }.items():
        if payload.get(field) != expected:
            raise OperatorCappedMasterError(f"capped-master payload changed: {field}")
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping) or bindings.get("launch_fingerprint") != amendment.LAUNCH_FINGERPRINT:
        raise OperatorCappedMasterError("capped-master launch bindings changed")
    if not isinstance(payload.get("baseline_identity"), Mapping):
        raise OperatorCappedMasterError("capped-master baseline identity is absent")
    blocks = payload.get("block_commits")
    if not isinstance(blocks, list) or [row.get("block_id") for row in blocks if isinstance(row, Mapping)] != [
        "M0200", "M0500_TOPUP"
    ]:
        raise OperatorCappedMasterError("capped-master block order changed")
    decisions = payload.get("precision_decisions")
    if not isinstance(decisions, list) or [row.get("cumulative_M") for row in decisions if isinstance(row, Mapping)] != [200, 500]:
        raise OperatorCappedMasterError("capped-master precision order changed")
    for row in decisions:
        if not isinstance(row, Mapping) or set(row) != runner.PRECISION_INDEX_FIELDS:
            raise OperatorCappedMasterError("capped-master precision index fields changed")
    if decisions[0]["next_M"] != 500:
        raise OperatorCappedMasterError("capped-master M0200 transition changed")
    cap = payload.get("operator_cap")
    if not isinstance(cap, Mapping):
        raise OperatorCappedMasterError("capped-master operator-cap record is absent")
    validate_operator_cap_record(cap)
    if decisions[1]["next_M"] != cap["original_final_precision_state"]["next_M"]:
        raise OperatorCappedMasterError("capped-master changed original M0500 next_M")
    history = payload.get("top_up_history")
    expected_history = [
        {
            "cumulative_M": row["cumulative_M"],
            "precision_artifact": row["artifact"],
            "precision_artifact_sha256": row["artifact_sha256"],
            "next_M": row["next_M"],
        }
        for row in decisions
    ]
    if history != expected_history:
        raise OperatorCappedMasterError("capped-master top-up history changed")


def finalize_operator_capped_master(
    *,
    staging_root: str | Path = DEFAULT_STAGING,
    amendment_path: str | Path = DEFAULT_AMENDMENT,
) -> Path:
    staging_input = _lexical_chain_no_symlink(
        Path(staging_root), ROOT, label="staging root"
    )
    staging = staging_input.resolve(strict=True)
    paths = capped_master_paths(staging)
    exists = tuple(path.exists() or path.is_symlink() for path in paths)
    if any(exists):
        if not all(exists):
            raise OperatorCappedMasterError("partial immutable capped-master envelope exists")
        load_operator_capped_master(
            paths[0], amendment_path=amendment_path, staging_root=staging
        )
        return paths[0]
    inputs = _authenticate_inputs(staging, Path(amendment_path))
    if inputs["terminal_state"]["precision_target_achieved"] is True:
        raise OriginalFinalizationRequired(
            "M0500 met the frozen precision target; use the original runner finalization"
        )
    payload = _build_payload(inputs)
    logical = _canonical_json_bytes(payload)
    compressed = zstd.ZstdCompressor(
        level=19, threads=0, write_checksum=True, write_content_size=True,
        write_dict_id=False,
    ).compress(logical)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    if paths[0].parent.is_symlink():
        raise OperatorCappedMasterError("capped-master output parent is a symlink")
    amendment_bundle_sha = _sha256_bytes(_canonical_json_bytes(payload["operator_cap"]["amendment"]))
    precision_bundle_sha = _sha256_bytes(_canonical_json_bytes(payload["precision_decisions"]))
    cap_summary = {
        "precision_target_achieved": False,
        "terminated_with_unresolved_precision": True,
        "terminal_reason": "OPERATOR_CAP_WITH_UNRESOLVED_PRECISION",
        "unresolved_estimand_count": payload["operator_cap"]["unresolved_estimand_count"],
        "original_next_M": 1000,
        "effective_next_M": None,
    }
    metadata = {
        "schema_version": 1,
        "status": STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "storage_model": STORAGE_MODEL,
        "artifact": paths[0].name,
        "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(logical),
        "uncompressed_bytes": len(logical),
        "final_M": 500,
        "path_count": 24_000,
        "snapshot_count": 72_000,
        "bindings": dict(payload["bindings"]),
        "operator_cap_summary": cap_summary,
        "amendment_bundle_sha256": amendment_bundle_sha,
        "precision_bundle_sha256": precision_bundle_sha,
        "cap_finalizer_sha256": payload["operator_cap"]["cap_finalizer_sha256"],
        "compression": dict(runner.COMPRESSION_CONTRACT),
        "environment": runner._privacy_safe_environment(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "immutable": True,
    }
    metadata_raw = _canonical_json_bytes(metadata)
    commit = {
        "schema_version": 1,
        "status": COMMIT_STATUS,
        "artifact": paths[0].name,
        "artifact_sha256": _sha256_bytes(compressed),
        "uncompressed_sha256": _sha256_bytes(logical),
        "metadata": paths[1].name,
        "metadata_sha256": _sha256_bytes(metadata_raw),
        "final_M": 500,
        "path_count": 24_000,
        "snapshot_count": 72_000,
        "launch_fingerprint": payload["bindings"]["launch_fingerprint"],
        "amendment_bundle_sha256": amendment_bundle_sha,
        "precision_bundle_sha256": precision_bundle_sha,
        "cap_finalizer_sha256": payload["operator_cap"]["cap_finalizer_sha256"],
        "immutable": True,
        "committed_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
    }
    _atomic_write_new(paths[0], compressed)
    _atomic_write_new(paths[1], metadata_raw)
    _atomic_write_new(paths[2], _canonical_json_bytes(commit))
    load_operator_capped_master(
        paths[0], amendment_path=amendment_path, staging_root=staging
    )
    return paths[0]


def load_operator_capped_master(
    raw_path: str | Path,
    *,
    amendment_path: str | Path = DEFAULT_AMENDMENT,
    staging_root: str | Path | None = None,
) -> OperatorCappedMaster:
    _assert_runtime_and_sources()
    raw_input = Path(raw_path)
    if staging_root is not None:
        staging_input = _lexical_chain_no_symlink(
            Path(staging_root), ROOT, label="staging root"
        )
    else:
        staging_input = raw_input.parent.parent
    staging = staging_input.resolve(strict=True)
    raw_input = _lexical_chain_no_symlink(
        raw_input, staging_input, label="capped-master artifact"
    )
    metadata_input, commit_input = _sidecars(raw_input)
    _lexical_chain_no_symlink(
        metadata_input, staging_input, label="capped-master metadata"
    )
    _lexical_chain_no_symlink(
        commit_input, staging_input, label="capped-master commit"
    )
    path = _regular_no_symlink(raw_input, label="capped-master artifact")
    metadata_path, commit_path = metadata_input, commit_input
    metadata_path = _regular_no_symlink(metadata_path, label="capped-master metadata")
    commit_path = _regular_no_symlink(commit_path, label="capped-master commit")
    _assert_inside(path, staging, label="capped-master artifact")
    compressed, metadata_raw, commit_raw = (
        path.read_bytes(), metadata_path.read_bytes(), commit_path.read_bytes()
    )
    metadata = _finite_json(metadata_raw, label=metadata_path.name)
    commit = _finite_json(commit_raw, label=commit_path.name)
    if not isinstance(metadata, dict) or not isinstance(commit, dict):
        raise OperatorCappedMasterError("capped-master sidecars must be objects")
    if metadata_raw != _canonical_json_bytes(metadata) or commit_raw != _canonical_json_bytes(commit):
        raise OperatorCappedMasterError("capped-master sidecar is not canonical JSON")
    try:
        logical = zstd.ZstdDecompressor().decompress(compressed)
    except zstd.ZstdError as exc:
        raise OperatorCappedMasterError("capped-master artifact is not valid zstd") from exc
    payload = _finite_json(logical, label=path.name)
    if not isinstance(payload, dict) or logical != _canonical_json_bytes(payload):
        raise OperatorCappedMasterError("capped-master payload is not canonical JSON")
    validate_capped_payload(payload)
    if set(metadata) != METADATA_FIELDS or set(commit) != COMMIT_FIELDS:
        raise OperatorCappedMasterError("capped-master sidecar fields changed")
    amendment_bundle_sha = _sha256_bytes(_canonical_json_bytes(payload["operator_cap"]["amendment"]))
    precision_bundle_sha = _sha256_bytes(_canonical_json_bytes(payload["precision_decisions"]))
    cap_summary = {
        "precision_target_achieved": payload["operator_cap"]["precision_target_achieved"],
        "terminated_with_unresolved_precision": payload["operator_cap"]["terminated_with_unresolved_precision"],
        "terminal_reason": payload["operator_cap"]["terminal_reason"],
        "unresolved_estimand_count": payload["operator_cap"]["unresolved_estimand_count"],
        "original_next_M": payload["operator_cap"]["original_final_precision_state"]["next_M"],
        "effective_next_M": None,
    }
    expected_metadata = {
        "schema_version": 1,
        "status": STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "storage_model": STORAGE_MODEL,
        "artifact": path.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(logical),
        "uncompressed_bytes": len(logical),
        "final_M": 500,
        "path_count": 24_000,
        "snapshot_count": 72_000,
        "bindings": dict(payload["bindings"]),
        "operator_cap_summary": cap_summary,
        "amendment_bundle_sha256": amendment_bundle_sha,
        "precision_bundle_sha256": precision_bundle_sha,
        "cap_finalizer_sha256": payload["operator_cap"]["cap_finalizer_sha256"],
        "compression": dict(runner.COMPRESSION_CONTRACT),
        "immutable": True,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise OperatorCappedMasterError(f"capped-master metadata mismatch: {field}")
    expected_commit = {
        "schema_version": 1,
        "status": COMMIT_STATUS,
        "artifact": path.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "uncompressed_sha256": _sha256_bytes(logical),
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256_bytes(metadata_raw),
        "final_M": 500,
        "path_count": 24_000,
        "snapshot_count": 72_000,
        "launch_fingerprint": payload["bindings"]["launch_fingerprint"],
        "amendment_bundle_sha256": amendment_bundle_sha,
        "precision_bundle_sha256": precision_bundle_sha,
        "cap_finalizer_sha256": payload["operator_cap"]["cap_finalizer_sha256"],
        "immutable": True,
    }
    for field, expected in expected_commit.items():
        if commit.get(field) != expected:
            raise OperatorCappedMasterError(f"capped-master commit mismatch: {field}")
    generated_at = _utc_timestamp(
        metadata.get("generated_at_utc"), label="metadata.generated_at_utc"
    )
    committed_at = _utc_timestamp(
        commit.get("committed_at_utc"), label="commit.committed_at_utc"
    )
    if committed_at < generated_at:
        raise OperatorCappedMasterError("capped master committed before generation")
    if committed_at > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise OperatorCappedMasterError("capped-master commit is implausibly in the future")
    _validate_environment(metadata.get("environment"), generated_at=generated_at)
    if payload["operator_cap"]["cap_finalizer_sha256"] != _sha256_file(Path(__file__).resolve()):
        raise OperatorCappedMasterError("cap-finalizer source changed")
    inputs = _authenticate_inputs(staging, Path(amendment_path))
    expected_payload = _build_payload(inputs)
    if payload != expected_payload:
        raise OperatorCappedMasterError("capped-master payload differs from authenticated inputs")
    for envelope_path in (path, metadata_path, commit_path):
        if envelope_path.stat().st_mode & 0o222:
            raise OperatorCappedMasterError("capped-master envelope is not filesystem read-only")
    provenance = {
        "artifact": path.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256_bytes(metadata_raw),
        "commit": commit_path.name,
        "commit_sha256": _sha256_bytes(commit_raw),
        "uncompressed_sha256": _sha256_bytes(logical),
        "final_M": 500,
        "path_count": 24_000,
        "snapshot_count": 72_000,
        "bindings": dict(payload["bindings"]),
        "operator_cap": dict(payload["operator_cap"]),
    }
    return OperatorCappedMaster(
        payload=dict(payload),
        provenance=provenance,
        staging_root=staging,
        block_payloads=tuple(dict(value) for value in inputs["block_payloads"]),
        precision_payloads=tuple(dict(value) for value in inputs["precision_payloads"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    args = parser.parse_args(argv)
    try:
        path = finalize_operator_capped_master(
            staging_root=args.staging_root,
            amendment_path=args.amendment,
        )
    except OriginalFinalizationRequired as exc:
        print(str(exc))
        return 2
    print(f"operator-capped raw master: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_NAME",
    "OperatorCappedMaster",
    "OperatorCappedMasterError",
    "OriginalFinalizationRequired",
    "capped_master_paths",
    "finalize_operator_capped_master",
    "load_operator_capped_master",
    "validate_capped_payload",
    "validate_operator_cap_record",
    "validate_terminal_precision_state",
]
