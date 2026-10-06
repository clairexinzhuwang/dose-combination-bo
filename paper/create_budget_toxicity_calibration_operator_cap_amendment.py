#!/usr/bin/env python3
"""Freeze and authenticate the result-blind M0500 operator-cap amendment.

This program deliberately reads no raw trial record and no M0500 precision
payload.  It hashes only already committed M0200 envelopes and frozen source
files, and it refuses to declare the cap once any M0500 precision sidecar or
M1000 execution artifact exists.
"""
from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING = ROOT / "results/budget_toxicity_calibration_staging"
DEFAULT_OUTPUT = (
    ROOT
    / "results/budget_toxicity_calibration_operator_cap_v2__launch_364a65e046c83acf__20260827"
)
SUPERSEDED_OUTPUT = (
    ROOT
    / "results/budget_toxicity_calibration_operator_cap__launch_364a65e046c83acf__20260827"
)
SCHEMA_PATH = ROOT / "paper/budget_toxicity_calibration_operator_cap_schema.json"
ARTIFACT_NAME = "budget_toxicity_calibration_operator_cap_M0500_amendment.json"
LAUNCH_FINGERPRINT = (
    "364a65e046c83acf5458885e8b07e2cb3de441c5ba01f0a1d02e6f59f6d3e0a5"
)
USER_DIRECTIVE = "ok最多就这个了不再追加。"
STATUS = "DECLARED_BUDGET_TOXICITY_CALIBRATION_OPERATOR_CAP"
METADATA_STATUS = "COMPLETE_BUDGET_TOXICITY_CALIBRATION_OPERATOR_CAP_AMENDMENT"
COMMIT_STATUS = "COMMITTED_BUDGET_TOXICITY_CALIBRATION_OPERATOR_CAP_AMENDMENT"
ARTIFACT_CLASS = "result_blind_operator_cap_amendment"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

FROZEN_SOURCE_HASHES = {
    "paper/analyze_budget_toxicity_calibration_audit.py": (
        "bcf3ec2964b88d61ac4becc8a9aa838831a3240d93a082abf42b5b328799ec2a"
    ),
    "paper/budget_toxicity_calibration_core.py": (
        "da51663bb6a6ae8e0bf1ceff4f6168d71d5ad71ceb3b193907c72c04ad0a7558"
    ),
    "paper/budget_toxicity_calibration_manifest.json": (
        "2e189ff6426bd3ad12fd29896fb6f76027a0d9e4424013d92edd0df3d143d3b6"
    ),
    "paper/budget_toxicity_calibration_prespec.md": (
        "1b1152d8f09462b96bf2f35252a9cd5703b331ab7a3546d4708b189f81e8c178"
    ),
    "paper/budget_toxicity_calibration_publication_contract.json": (
        "a1904b435d4e42ef6df25094bc1f3ce35156ceaec5bf157ec24fb2ff2827b6a8"
    ),
    "paper/generate_budget_toxicity_calibration_publication.py": (
        "e14c2464f586f3abb805bfa24bb47f2ecc61a93e7405e0cb51489d497c6aa919"
    ),
    "paper/monitor_budget_toxicity_calibration_precision.py": (
        "3994535bc1e2c300388c2d161d326e66f6185452971075e8d7bc9caf76bb07e0"
    ),
    "paper/run_budget_toxicity_calibration_audit.py": (
        "c36420dc8a37c8a538ecba3b8d5d82753d428fb0eb4074dbe97baf83e474365d"
    ),
}

PAYLOAD_FIELDS = frozenset({
    "schema_version", "status", "artifact_class", "audit_id",
    "launch_fingerprint", "declared_at_utc", "declaration_phase",
    "cap_cumulative_M", "prohibited_seed_blocks", "original_frozen_rule",
    "amended_terminal_rule", "result_blindness_assertion", "bindings",
})
METADATA_FIELDS = frozenset({
    "schema_version", "status", "artifact_class", "artifact",
    "artifact_sha256", "artifact_bytes", "audit_id", "launch_fingerprint",
    "cap_cumulative_M", "declared_at_utc", "bindings_sha256",
    "amendment_generator_sha256", "amendment_schema_sha256", "immutable",
})
COMMIT_FIELDS = frozenset({
    "schema_version", "status", "artifact", "artifact_sha256", "metadata",
    "metadata_sha256", "audit_id", "launch_fingerprint", "cap_cumulative_M",
    "immutable", "committed_at_utc",
})


class OperatorCapAmendmentError(RuntimeError):
    """Raised when a result-blind cap cannot be safely declared or verified."""


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
        raise OperatorCapAmendmentError(f"{label} is not finite valid JSON") from exc
    _assert_finite(value, label=label)
    return value


def _assert_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise OperatorCapAmendmentError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, label=f"{label}[{index}]")


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise OperatorCapAmendmentError(f"{label} is not a lowercase SHA-256")
    return value


def _assert_regular_no_symlink(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    try:
        mode = candidate.lstat().st_mode
    except FileNotFoundError as exc:
        raise OperatorCapAmendmentError(f"{label} is absent: {candidate}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise OperatorCapAmendmentError(f"{label} is not a regular non-symlink file")
    return candidate.resolve(strict=True)


def _assert_lexical_chain_no_symlink(path: Path, *, root: Path, label: str) -> None:
    """Reject lexical escape and every symlink component before resolution."""

    root_resolved = Path(root).resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root_resolved / candidate
    candidate = Path(os.path.abspath(candidate))
    if not candidate.is_relative_to(root_resolved):
        raise OperatorCapAmendmentError(f"{label} escapes its authenticated root")
    current = root_resolved
    for component in candidate.relative_to(root_resolved).parts:
        if component in ("", ".", ".."):
            raise OperatorCapAmendmentError(f"{label} has a non-canonical component")
        current = current / component
        if current.is_symlink():
            raise OperatorCapAmendmentError(f"{label} crosses a symlink: {current}")


def _assert_no_symlink_parents(path: Path, *, stop: Path) -> None:
    current = Path(path)
    stop_resolved = Path(stop).resolve(strict=True)
    while True:
        if current.exists() and stat.S_ISLNK(current.lstat().st_mode):
            raise OperatorCapAmendmentError(f"symlink path component is forbidden: {current}")
        if current.resolve(strict=False) == stop_resolved:
            return
        if current.parent == current:
            raise OperatorCapAmendmentError("path is outside its authenticated root")
        current = current.parent


def _relative_commitment(path: Path, *, root: Path, label: str) -> dict[str, Any]:
    _assert_lexical_chain_no_symlink(path, root=root, label=label)
    resolved = _assert_regular_no_symlink(path, label=label)
    root_resolved = root.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise OperatorCapAmendmentError(f"{label} escapes the repository root")
    _assert_no_symlink_parents(path.parent, stop=root_resolved)
    return {
        "path": resolved.relative_to(root_resolved).as_posix(),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _trio(artifact: Path, *, root: Path, label: str) -> dict[str, Any]:
    return {
        "artifact": _relative_commitment(artifact, root=root, label=f"{label} artifact"),
        "metadata": _relative_commitment(
            Path(str(artifact) + ".metadata.json"),
            root=root,
            label=f"{label} metadata",
        ),
        "commit": _relative_commitment(
            Path(str(artifact) + ".commit.json"),
            root=root,
            label=f"{label} commit",
        ),
    }


def _verify_frozen_sources(root: Path) -> None:
    for relative, expected in FROZEN_SOURCE_HASHES.items():
        _assert_lexical_chain_no_symlink(
            root / relative, root=root, label=f"frozen source {relative}"
        )
        source = _assert_regular_no_symlink(root / relative, label=f"frozen source {relative}")
        if _sha256_file(source) != expected:
            raise OperatorCapAmendmentError(f"frozen source hash changed: {relative}")


def _assert_predecision_absence(staging: Path) -> None:
    m0500 = staging / "precision/budget_toxicity_calibration_precision_M0500.json.zst"
    m0500_candidates = (
        m0500,
        Path(str(m0500) + ".metadata.json"),
        Path(str(m0500) + ".commit.json"),
    )
    if any(path.exists() for path in m0500_candidates):
        raise OperatorCapAmendmentError(
            "M0500 precision envelope already exists; result-blind declaration refused"
        )
    forbidden = sorted(path for path in staging.rglob("*M1000*") if path.exists())
    if forbidden:
        raise OperatorCapAmendmentError(
            "M1000 execution evidence already exists; operator-cap declaration refused"
        )


def _validate_trio_shape(
    value: Any,
    *,
    root: Path,
    label: str,
    expected_paths: Mapping[str, str] | None = None,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"artifact", "metadata", "commit"}:
        raise OperatorCapAmendmentError(f"{label} trio fields changed")
    for name in ("artifact", "metadata", "commit"):
        item = value[name]
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "bytes"}:
            raise OperatorCapAmendmentError(f"{label}.{name} commitment fields changed")
        relative = item["path"]
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise OperatorCapAmendmentError(f"{label}.{name} path is not canonical relative")
        normalized = Path(relative).as_posix()
        if relative != normalized or any(
            part in ("", ".", "..") for part in Path(relative).parts
        ):
            raise OperatorCapAmendmentError(f"{label}.{name} path is not canonical")
        if expected_paths is not None and relative != expected_paths[name]:
            raise OperatorCapAmendmentError(f"{label}.{name} path changed")
        path = (root / relative)
        _assert_lexical_chain_no_symlink(path, root=root, label=f"{label}.{name}")
        resolved = _assert_regular_no_symlink(path, label=f"{label}.{name}")
        if not resolved.is_relative_to(root.resolve(strict=True)):
            raise OperatorCapAmendmentError(f"{label}.{name} escapes the repository root")
        _require_sha(item["sha256"], label=f"{label}.{name}.sha256")
        if type(item["bytes"]) is not int or item["bytes"] <= 0:
            raise OperatorCapAmendmentError(f"{label}.{name}.bytes is invalid")
        if _sha256_file(resolved) != item["sha256"] or resolved.stat().st_size != item["bytes"]:
            raise OperatorCapAmendmentError(f"{label}.{name} file commitment changed")


def _utc_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise OperatorCapAmendmentError(f"{label} must be an explicit UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise OperatorCapAmendmentError(f"{label} is invalid") from exc
    if parsed.utcoffset() != timedelta(0):
        raise OperatorCapAmendmentError(f"{label} is not UTC")
    return parsed


def _expected_audit_id(bindings: Mapping[str, Any]) -> str:
    audit_seed = {
        "launch_fingerprint": LAUNCH_FINGERPRINT,
        "cap_cumulative_M": 500,
        "bindings": dict(bindings),
        "user_directive": USER_DIRECTIVE,
    }
    return "operator-cap-M0500-" + _sha256_bytes(
        _canonical_json_bytes(audit_seed)
    )[:16]


def _authenticate_m0200_chain(
    *,
    root: Path,
    staging: Path,
    baseline_trio: Mapping[str, Any],
    precision_trio: Mapping[str, Any],
) -> None:
    """Authenticate M0200 envelopes without reading any raw shard row."""

    try:
        from paper import analyze_budget_toxicity_calibration_audit as analyzer
        from paper import monitor_budget_toxicity_calibration_precision as precision_monitor
    except ModuleNotFoundError:
        import analyze_budget_toxicity_calibration_audit as analyzer
        import monitor_budget_toxicity_calibration_precision as precision_monitor
    block = staging / "raw/budget_toxicity_calibration_M0200_block_index.json.zst"
    block_payload, block_provenance, _paths = precision_monitor.authenticate_seed_block(
        block, verify_local_bindings=True
    )
    block_metadata = Path(str(block) + ".metadata.json")
    block_commit = Path(str(block) + ".commit.json")
    hashes = {
        "paper/budget_toxicity_calibration_manifest.json": FROZEN_SOURCE_HASHES[
            "paper/budget_toxicity_calibration_manifest.json"
        ],
        "paper/budget_toxicity_calibration_prespec.md": FROZEN_SOURCE_HASHES[
            "paper/budget_toxicity_calibration_prespec.md"
        ],
        block.relative_to(staging).as_posix(): str(block_provenance["artifact_sha256"]),
        block_metadata.relative_to(staging).as_posix(): str(block_provenance["metadata_sha256"]),
        block_commit.relative_to(staging).as_posix(): str(block_provenance["commit_sha256"]),
    }
    for shard in block_payload["shard_commits"]:
        for field, hash_field in (
            ("artifact", "artifact_sha256"),
            ("metadata", "metadata_sha256"),
            ("commit", "commit_sha256"),
        ):
            hashes[str(shard[field])] = str(shard[hash_field])
    if len(hashes) != 77:
        raise OperatorCapAmendmentError("M0200 baseline input commitment count changed")
    baseline_path = root / str(baseline_trio["artifact"]["path"])
    analyzer.load_baseline_identity_pass(
        baseline_path,
        expected_input_hashes=dict(sorted(hashes.items())),
        expected_m=200,
    )
    for name in ("artifact", "metadata", "commit"):
        bound_path = root / str(baseline_trio[name]["path"])
        hashes[bound_path.relative_to(staging).as_posix()] = str(
            baseline_trio[name]["sha256"]
        )
    if len(hashes) != 80:
        raise OperatorCapAmendmentError("M0200 precision input commitment count changed")
    precision_path = root / str(precision_trio["artifact"]["path"])
    precision_monitor.load_blind_precision_decision(
        precision_path,
        expected_m=200,
        expected_input_hashes=dict(sorted(hashes.items())),
    )


def validate_amendment_payload(
    payload: Mapping[str, Any], *, root: Path = ROOT, verify_files: bool = True
) -> None:
    if set(payload) != PAYLOAD_FIELDS:
        raise OperatorCapAmendmentError("operator-cap amendment fields changed")
    exact = {
        "schema_version": 1,
        "status": STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "launch_fingerprint": LAUNCH_FINGERPRINT,
        "declaration_phase": "M0500_TOPUP_IN_PROGRESS_BEFORE_M0500_PRECISION_MONITOR",
        "cap_cumulative_M": 500,
        "prohibited_seed_blocks": ["M1000_TOPUP"],
        "original_frozen_rule": {
            "initial_M": 200,
            "intermediate_M": 500,
            "maximum_M": 1000,
            "guarded_mcse_threshold_points": 1.5,
            "threshold_rule": "all_118_primary_estimands_guarded_mcse_at_most_1.5_points",
        },
        "amended_terminal_rule": {
            "stop_after_complete_balanced_M0500": True,
            "effective_next_M": None,
            "precision_threshold_unchanged": True,
            "original_precision_decision_must_remain_immutable": True,
            "if_original_next_M_1000": (
                "terminate_due_to_operator_cap_and_disclose_all_unresolved_guarded_precision_targets"
            ),
            "no_precision_pass_may_be_fabricated": True,
        },
        "result_blindness_assertion": {
            "m0500_scientific_effects_inspected": False,
            "m0500_precision_decision_available": False,
            "m0200_precision_decision_was_known": True,
            "decision_basis": "operator_compute_cap_not_M0500_result",
            "user_directive": USER_DIRECTIVE,
        },
    }
    for field, expected in exact.items():
        if payload.get(field) != expected:
            raise OperatorCapAmendmentError(f"operator-cap amendment changed: {field}")
    _require_sha(payload.get("launch_fingerprint"), label="launch_fingerprint")
    audit_id = payload.get("audit_id")
    if not isinstance(audit_id, str) or re.fullmatch(
        r"operator-cap-M0500-[0-9a-f]{16}", audit_id
    ) is None:
        raise OperatorCapAmendmentError("operator-cap audit_id is malformed")
    parsed = _utc_timestamp(payload.get("declared_at_utc"), label="declared_at_utc")
    if parsed > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise OperatorCapAmendmentError("declared_at_utc is implausibly in the future")
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "manifest_sha256", "prespec_sha256", "frozen_source_hashes",
        "amendment_generator_sha256", "amendment_schema_sha256",
        "baseline_identity_M0200", "precision_M0200", "user_directive_sha256",
        "superseded_amendment", "absence_evidence",
    }:
        raise OperatorCapAmendmentError("operator-cap binding fields changed")
    for field in (
        "manifest_sha256", "prespec_sha256", "amendment_generator_sha256",
        "amendment_schema_sha256", "user_directive_sha256",
    ):
        _require_sha(bindings.get(field), label=f"bindings.{field}")
    if bindings["manifest_sha256"] != FROZEN_SOURCE_HASHES[
        "paper/budget_toxicity_calibration_manifest.json"
    ] or bindings["prespec_sha256"] != FROZEN_SOURCE_HASHES[
        "paper/budget_toxicity_calibration_prespec.md"
    ]:
        raise OperatorCapAmendmentError("manifest/prespec binding changed")
    if bindings["frozen_source_hashes"] != FROZEN_SOURCE_HASHES:
        raise OperatorCapAmendmentError("frozen source-hash map changed")
    if bindings["user_directive_sha256"] != _sha256_bytes(USER_DIRECTIVE.encode("utf-8")):
        raise OperatorCapAmendmentError("user directive commitment changed")
    if bindings["absence_evidence"] != {
        "m0500_precision_envelope_absent": True,
        "m1000_artifacts_absent": True,
    }:
        raise OperatorCapAmendmentError("predecision absence evidence changed")
    superseded = bindings.get("superseded_amendment")
    if not isinstance(superseded, Mapping) or set(superseded) != {"reason", "trio"}:
        raise OperatorCapAmendmentError("superseded amendment binding changed")
    if superseded["reason"] != "superseded_due_to_authentication_hardening":
        raise OperatorCapAmendmentError("superseded amendment reason changed")
    if audit_id != _expected_audit_id(bindings):
        raise OperatorCapAmendmentError("operator-cap audit_id is not deterministic")
    if verify_files:
        _verify_frozen_sources(root)
        if bindings["amendment_generator_sha256"] != _sha256_file(Path(__file__).resolve()):
            raise OperatorCapAmendmentError("amendment generator source changed")
        if bindings["amendment_schema_sha256"] != _sha256_file(SCHEMA_PATH):
            raise OperatorCapAmendmentError("amendment schema source changed")
        for relative, expected in FROZEN_SOURCE_HASHES.items():
            if _sha256_file(root / relative) != expected:
                raise OperatorCapAmendmentError(f"frozen source changed: {relative}")
        _validate_trio_shape(
            bindings["baseline_identity_M0200"],
            root=root,
            label="baseline_identity_M0200",
            expected_paths={
                "artifact": "results/budget_toxicity_calibration_staging/baseline/budget_toxicity_calibration_baseline_identity_M0200.json.zst",
                "metadata": "results/budget_toxicity_calibration_staging/baseline/budget_toxicity_calibration_baseline_identity_M0200.json.zst.metadata.json",
                "commit": "results/budget_toxicity_calibration_staging/baseline/budget_toxicity_calibration_baseline_identity_M0200.json.zst.commit.json",
            },
        )
        _validate_trio_shape(
            bindings["precision_M0200"],
            root=root,
            label="precision_M0200",
            expected_paths={
                "artifact": "results/budget_toxicity_calibration_staging/precision/budget_toxicity_calibration_precision_M0200.json.zst",
                "metadata": "results/budget_toxicity_calibration_staging/precision/budget_toxicity_calibration_precision_M0200.json.zst.metadata.json",
                "commit": "results/budget_toxicity_calibration_staging/precision/budget_toxicity_calibration_precision_M0200.json.zst.commit.json",
            },
        )
        _validate_trio_shape(
            superseded["trio"],
            root=root,
            label="superseded_amendment",
            expected_paths={
                "artifact": "results/budget_toxicity_calibration_operator_cap__launch_364a65e046c83acf__20260827/budget_toxicity_calibration_operator_cap_M0500_amendment.json",
                "metadata": "results/budget_toxicity_calibration_operator_cap__launch_364a65e046c83acf__20260827/budget_toxicity_calibration_operator_cap_M0500_amendment.json.metadata.json",
                "commit": "results/budget_toxicity_calibration_operator_cap__launch_364a65e046c83acf__20260827/budget_toxicity_calibration_operator_cap_M0500_amendment.json.commit.json",
            },
        )
        _authenticate_m0200_chain(
            root=root,
            staging=root / "results/budget_toxicity_calibration_staging",
            baseline_trio=bindings["baseline_identity_M0200"],
            precision_trio=bindings["precision_M0200"],
        )


def _atomic_write_new(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise OperatorCapAmendmentError(f"immutable destination already exists: {path}")
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


def amendment_paths(output_dir: str | Path) -> tuple[Path, Path, Path]:
    artifact = Path(output_dir) / ARTIFACT_NAME
    return (
        artifact,
        Path(str(artifact) + ".metadata.json"),
        Path(str(artifact) + ".commit.json"),
    )


def authenticate_amendment(
    artifact_path: str | Path,
    *,
    root: Path = ROOT,
    expected_launch_fingerprint: str = LAUNCH_FINGERPRINT,
    verify_bound_files: bool = True,
) -> dict[str, Any]:
    _assert_lexical_chain_no_symlink(
        Path(artifact_path), root=root, label="amendment artifact"
    )
    _assert_lexical_chain_no_symlink(
        Path(str(artifact_path) + ".metadata.json"),
        root=root,
        label="amendment metadata",
    )
    _assert_lexical_chain_no_symlink(
        Path(str(artifact_path) + ".commit.json"),
        root=root,
        label="amendment commit",
    )
    artifact = _assert_regular_no_symlink(Path(artifact_path), label="amendment artifact")
    metadata_path = _assert_regular_no_symlink(
        Path(str(artifact_path) + ".metadata.json"), label="amendment metadata"
    )
    commit_path = _assert_regular_no_symlink(
        Path(str(artifact_path) + ".commit.json"), label="amendment commit"
    )
    artifact_raw, metadata_raw, commit_raw = (
        artifact.read_bytes(), metadata_path.read_bytes(), commit_path.read_bytes()
    )
    payload = _finite_json(artifact_raw, label=artifact.name)
    metadata = _finite_json(metadata_raw, label=metadata_path.name)
    commit = _finite_json(commit_raw, label=commit_path.name)
    if not all(isinstance(value, dict) for value in (payload, metadata, commit)):
        raise OperatorCapAmendmentError("amendment envelope members must be objects")
    if artifact_raw != _canonical_json_bytes(payload):
        raise OperatorCapAmendmentError("amendment artifact is not canonical JSON")
    if metadata_raw != _canonical_json_bytes(metadata) or commit_raw != _canonical_json_bytes(commit):
        raise OperatorCapAmendmentError("amendment sidecar is not canonical JSON")
    validate_amendment_payload(payload, root=root, verify_files=verify_bound_files)
    if payload["launch_fingerprint"] != expected_launch_fingerprint:
        raise OperatorCapAmendmentError("amendment launch fingerprint differs")
    if set(metadata) != METADATA_FIELDS or set(commit) != COMMIT_FIELDS:
        raise OperatorCapAmendmentError("amendment sidecar fields changed")
    expected_metadata = {
        "schema_version": 1,
        "status": METADATA_STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "artifact": artifact.name,
        "artifact_sha256": _sha256_bytes(artifact_raw),
        "artifact_bytes": len(artifact_raw),
        "audit_id": payload["audit_id"],
        "launch_fingerprint": expected_launch_fingerprint,
        "cap_cumulative_M": 500,
        "declared_at_utc": payload["declared_at_utc"],
        "bindings_sha256": _sha256_bytes(_canonical_json_bytes(payload["bindings"])),
        "amendment_generator_sha256": payload["bindings"]["amendment_generator_sha256"],
        "amendment_schema_sha256": payload["bindings"]["amendment_schema_sha256"],
        "immutable": True,
    }
    if metadata != expected_metadata:
        raise OperatorCapAmendmentError("amendment metadata differs from artifact")
    expected_commit = {
        "schema_version": 1,
        "status": COMMIT_STATUS,
        "artifact": artifact.name,
        "artifact_sha256": _sha256_bytes(artifact_raw),
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256_bytes(metadata_raw),
        "audit_id": payload["audit_id"],
        "launch_fingerprint": expected_launch_fingerprint,
        "cap_cumulative_M": 500,
        "immutable": True,
    }
    for field, expected in expected_commit.items():
        if commit.get(field) != expected:
            raise OperatorCapAmendmentError(f"amendment commit mismatch: {field}")
    declared_at = _utc_timestamp(payload["declared_at_utc"], label="declared_at_utc")
    committed_at = _utc_timestamp(
        commit.get("committed_at_utc"), label="committed_at_utc"
    )
    if committed_at < declared_at:
        raise OperatorCapAmendmentError("amendment was committed before it was declared")
    if committed_at > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise OperatorCapAmendmentError("committed_at_utc is implausibly in the future")
    for path in (artifact, metadata_path, commit_path):
        if path.stat().st_mode & 0o222:
            raise OperatorCapAmendmentError("amendment envelope is not filesystem read-only")
    return {
        "payload": dict(payload),
        "metadata": dict(metadata),
        "commit": dict(commit),
        "paths": {"artifact": artifact, "metadata": metadata_path, "commit": commit_path},
        "hashes": {
            "artifact": _sha256_bytes(artifact_raw),
            "metadata": _sha256_bytes(metadata_raw),
            "commit": _sha256_bytes(commit_raw),
        },
    }


def declare_operator_cap(
    *,
    staging_root: str | Path = DEFAULT_STAGING,
    output_dir: str | Path = DEFAULT_OUTPUT,
    root: Path = ROOT,
) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    _assert_lexical_chain_no_symlink(
        Path(staging_root), root=root, label="staging root"
    )
    staging = Path(staging_root).resolve(strict=True)
    if not staging.is_relative_to(root):
        raise OperatorCapAmendmentError("staging root escapes the repository root")
    output = Path(output_dir)
    _assert_lexical_chain_no_symlink(output, root=root, label="amendment output")
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise OperatorCapAmendmentError("amendment output is not a regular directory")
    paths = amendment_paths(output)
    exists = tuple(path.exists() or path.is_symlink() for path in paths)
    if any(exists):
        if not all(exists):
            raise OperatorCapAmendmentError("partial immutable amendment envelope exists")
        return authenticate_amendment(paths[0], root=root)
    _assert_predecision_absence(staging)
    _verify_frozen_sources(root)
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise OperatorCapAmendmentError("symlink amendment output is forbidden")
    _assert_no_symlink_parents(output, stop=root)
    baseline = staging / "baseline/budget_toxicity_calibration_baseline_identity_M0200.json.zst"
    precision = staging / "precision/budget_toxicity_calibration_precision_M0200.json.zst"
    bindings = {
        "manifest_sha256": FROZEN_SOURCE_HASHES[
            "paper/budget_toxicity_calibration_manifest.json"
        ],
        "prespec_sha256": FROZEN_SOURCE_HASHES[
            "paper/budget_toxicity_calibration_prespec.md"
        ],
        "frozen_source_hashes": dict(FROZEN_SOURCE_HASHES),
        "amendment_generator_sha256": _sha256_file(Path(__file__).resolve()),
        "amendment_schema_sha256": _sha256_file(SCHEMA_PATH),
        "baseline_identity_M0200": _trio(baseline, root=root, label="M0200 baseline"),
        "precision_M0200": _trio(precision, root=root, label="M0200 precision"),
        "superseded_amendment": {
            "reason": "superseded_due_to_authentication_hardening",
            "trio": _trio(
                SUPERSEDED_OUTPUT / ARTIFACT_NAME,
                root=root,
                label="superseded amendment",
            ),
        },
        "user_directive_sha256": _sha256_bytes(USER_DIRECTIVE.encode("utf-8")),
        "absence_evidence": {
            "m0500_precision_envelope_absent": True,
            "m1000_artifacts_absent": True,
        },
    }
    declared_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = {
        "schema_version": 1,
        "status": STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "audit_id": _expected_audit_id(bindings),
        "launch_fingerprint": LAUNCH_FINGERPRINT,
        "declared_at_utc": declared_at,
        "declaration_phase": "M0500_TOPUP_IN_PROGRESS_BEFORE_M0500_PRECISION_MONITOR",
        "cap_cumulative_M": 500,
        "prohibited_seed_blocks": ["M1000_TOPUP"],
        "original_frozen_rule": {
            "initial_M": 200,
            "intermediate_M": 500,
            "maximum_M": 1000,
            "guarded_mcse_threshold_points": 1.5,
            "threshold_rule": "all_118_primary_estimands_guarded_mcse_at_most_1.5_points",
        },
        "amended_terminal_rule": {
            "stop_after_complete_balanced_M0500": True,
            "effective_next_M": None,
            "precision_threshold_unchanged": True,
            "original_precision_decision_must_remain_immutable": True,
            "if_original_next_M_1000": (
                "terminate_due_to_operator_cap_and_disclose_all_unresolved_guarded_precision_targets"
            ),
            "no_precision_pass_may_be_fabricated": True,
        },
        "result_blindness_assertion": {
            "m0500_scientific_effects_inspected": False,
            "m0500_precision_decision_available": False,
            "m0200_precision_decision_was_known": True,
            "decision_basis": "operator_compute_cap_not_M0500_result",
            "user_directive": USER_DIRECTIVE,
        },
        "bindings": bindings,
    }
    validate_amendment_payload(payload, root=root, verify_files=True)
    artifact_raw = _canonical_json_bytes(payload)
    metadata = {
        "schema_version": 1,
        "status": METADATA_STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "artifact": paths[0].name,
        "artifact_sha256": _sha256_bytes(artifact_raw),
        "artifact_bytes": len(artifact_raw),
        "audit_id": payload["audit_id"],
        "launch_fingerprint": LAUNCH_FINGERPRINT,
        "cap_cumulative_M": 500,
        "declared_at_utc": declared_at,
        "bindings_sha256": _sha256_bytes(_canonical_json_bytes(bindings)),
        "amendment_generator_sha256": bindings["amendment_generator_sha256"],
        "amendment_schema_sha256": bindings["amendment_schema_sha256"],
        "immutable": True,
    }
    metadata_raw = _canonical_json_bytes(metadata)
    commit = {
        "schema_version": 1,
        "status": COMMIT_STATUS,
        "artifact": paths[0].name,
        "artifact_sha256": _sha256_bytes(artifact_raw),
        "metadata": paths[1].name,
        "metadata_sha256": _sha256_bytes(metadata_raw),
        "audit_id": payload["audit_id"],
        "launch_fingerprint": LAUNCH_FINGERPRINT,
        "cap_cumulative_M": 500,
        "immutable": True,
        "committed_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
    }
    _atomic_write_new(paths[0], artifact_raw)
    _atomic_write_new(paths[1], metadata_raw)
    _atomic_write_new(paths[2], _canonical_json_bytes(commit))
    return authenticate_amendment(paths[0], root=root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = declare_operator_cap(
        staging_root=args.staging_root,
        output_dir=args.output_dir,
    )
    for key in ("artifact", "metadata", "commit"):
        print(f"{key}: {result['paths'][key]}")
        print(f"{key}_sha256: {result['hashes'][key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_NAME",
    "FROZEN_SOURCE_HASHES",
    "LAUNCH_FINGERPRINT",
    "OperatorCapAmendmentError",
    "amendment_paths",
    "authenticate_amendment",
    "declare_operator_cap",
    "validate_amendment_payload",
]
