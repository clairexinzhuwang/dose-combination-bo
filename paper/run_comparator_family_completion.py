#!/usr/bin/env python3
"""Run the frozen 8,800-trial comparator-family completion with Ray.

The default invocation is a read-only preflight.  Execution and finalization
require ``--authorization-token COMPARATOR_FAMILY_EXTENSION_GO``.  This runner
does not edit or project into the immutable 55,480-row formal record bundle.
It writes a separately authenticated raw extension under its own staging root.

Each trial is delegated to :func:`paper.run_full_formal_rerun._run_job`, so the
same stable-z gate audit and independent terminal-recommendation replay are
mandatory.  Ray is not optional: a run starts a ten-CPU Ray runtime, assigns
one CPU to each trial, and keeps at most twenty object references in flight.
There is intentionally no serial or process-pool fallback.
"""
from __future__ import annotations

import os

for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ray
import zstandard as zstd

from paper import formal_rerun_preflight as formal_preflight
from paper import run_full_formal_rerun as formal_runner


AUTHORIZATION_TOKEN = "COMPARATOR_FAMILY_EXTENSION_GO"
RAY_NUM_CPUS = 10
WORKER_NUM_CPUS = 1
MAX_IN_FLIGHT = 20
CHECKPOINT_EVERY = 10
EXPECTED_EXECUTIONS = 8_800
EXPECTED_SHARDS = 40
IMMUTABLE_FORMAL_PROJECTION_ROWS = 55_480

FROZEN_ANALYSIS_SPEC_SHA256 = (
    "117e8d1cd3221a54786075494c7b92503001dba90c72a22f4049d2fc807cccd8"
)

MANIFEST_PATH = ROOT / "paper/comparator_family_completion_manifest.json"
ANALYSIS_SPEC_PATH = ROOT / "paper/comparator_family_extension_prespec.md"
FORMAL_MANIFEST_PATH = formal_preflight.MANIFEST_PATH
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/comparator_family_completion_staging"
).resolve()
ARTIFACT_CLASS = (
    "authenticated_comparator_family_raw_extension_not_formal_projection"
)
CHECKPOINT_STATUS = "PARTIAL_COMPARATOR_FAMILY_EXTENSION_NOT_FOR_ANALYSIS"
FINAL_STATUS = "COMPLETE_COMPARATOR_FAMILY_RAW_EXTENSION"
COMMIT_STATUS = "COMMITTED_COMPARATOR_FAMILY_RAW_EXTENSION"
FINAL_BASENAME = "comparator_family_completion_records.json.zst"
CANONICAL_SORT_KEY = formal_runner.CANONICAL_SORT_KEY


class ComparatorFamilyRunError(RuntimeError):
    """Raised when an extension design, provenance, or runtime invariant fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return formal_preflight.file_sha256(Path(path))


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_bytes_write(path: Path, value: bytes) -> None:
    """Durably replace one file without exposing partial bytes."""
    path = Path(path)
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
        try:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json_write(path: Path, value: Any) -> None:
    _atomic_bytes_write(Path(path), _canonical_json_bytes(value))


def _runtime_identity() -> dict[str, str]:
    return {
        **formal_runner._runtime_identity(),
        "ray": str(ray.__version__),
        "zstandard": str(zstd.__version__),
    }


def _formal_source_hashes() -> dict[str, str]:
    formal_manifest = formal_preflight.load_manifest(FORMAL_MANIFEST_PATH)
    relative_paths = formal_runner._required_source_files(formal_manifest)
    if len(relative_paths) != 14 or len(set(relative_paths)) != 14:
        raise ComparatorFamilyRunError(
            "formal execution-source contract is not exactly 14 unique files"
        )
    output: dict[str, str] = {}
    for relative in sorted(relative_paths):
        path = ROOT / relative
        if not path.is_file():
            raise ComparatorFamilyRunError(f"formal execution source is absent: {relative}")
        output[relative] = _sha256_file(path)
    return output


def _validate_execution_identity(manifest: Mapping[str, Any]) -> None:
    identity = manifest.get("execution_identity")
    if not isinstance(identity, Mapping):
        raise ComparatorFamilyRunError(
            "completion manifest lacks its frozen execution_identity"
        )
    runner_path = "paper/run_comparator_family_completion.py"
    if identity.get("runner_path") != runner_path:
        raise ComparatorFamilyRunError("completion runner path binding changed")
    observed_runner_sha256 = _sha256_file(ROOT / runner_path)
    if identity.get("runner_sha256") != observed_runner_sha256:
        raise ComparatorFamilyRunError(
            "completion runner source hash differs from the frozen manifest"
        )
    observed_formal = _formal_source_hashes()
    if identity.get("formal_source_sha256") != observed_formal:
        raise ComparatorFamilyRunError(
            "formal 14-file source hash map differs from the frozen manifest"
        )
    observed_runtime = _runtime_identity()
    if identity.get("runtime_identity") != observed_runtime:
        raise ComparatorFamilyRunError(
            "execution runtime differs from the frozen manifest: "
            f"observed={observed_runtime!r}, "
            f"required={identity.get('runtime_identity')!r}"
        )


def _privacy_safe_environment() -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "ray_contract": {
            "num_cpus": RAY_NUM_CPUS,
            "worker_num_cpus": WORKER_NUM_CPUS,
            "maximum_in_flight": MAX_IN_FLIGHT,
            "serial_fallback": False,
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "privacy_note": (
            "No username, home path, hostname, serial number, or persistent "
            "device identifier is recorded."
        ),
    }


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not Path(path).is_file():
        raise ComparatorFamilyRunError(f"frozen {label} is absent: {path}")
    observed = _sha256_file(Path(path))
    if observed != expected:
        raise ComparatorFamilyRunError(
            f"frozen {label} hash changed: {observed} != {expected}"
        )
    return observed


def _load_frozen_manifest(
    manifest_path: Path = MANIFEST_PATH,
    analysis_spec_path: Path = ANALYSIS_SPEC_PATH,
) -> dict[str, Any]:
    if not Path(manifest_path).is_file():
        raise ComparatorFamilyRunError(
            f"frozen comparator-family completion manifest is absent: {manifest_path}"
        )
    _require_hash(
        Path(analysis_spec_path),
        FROZEN_ANALYSIS_SPEC_SHA256,
        "comparator-family analysis specification",
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != (
        "FROZEN_BEFORE_MISSING_EXTENSION_EXECUTION"
    ):
        raise ComparatorFamilyRunError("completion manifest schema/status changed")
    if manifest.get("analysis_specification") != (
        "paper/comparator_family_extension_prespec.md"
    ):
        raise ComparatorFamilyRunError("completion manifest analysis-spec binding changed")
    requirements = manifest.get("execution_requirements")
    expected_requirements = {
        "ray_required": True,
        "ray_num_cpus": RAY_NUM_CPUS,
        "worker_num_cpus": WORKER_NUM_CPUS,
        "maximum_in_flight": MAX_IN_FLIGHT,
        "silent_serial_fallback_prohibited": True,
        "atomic_resume_checkpoints_required": True,
        "stable_z_gate_and_terminal_audit_required": True,
        "frozen_formal_sources_must_not_be_modified": True,
    }
    if requirements != expected_requirements:
        raise ComparatorFamilyRunError("completion manifest execution contract changed")
    counts = manifest.get("counts", {})
    if counts != {
        "authenticated_reused_records": 25_200,
        "new_extension_executions": EXPECTED_EXECUTIONS,
        "completed_family_records": 34_000,
        "immutable_formal_projection_records_unchanged": (
            IMMUTABLE_FORMAL_PROJECTION_ROWS
        ),
    }:
        raise ComparatorFamilyRunError("completion manifest record counts changed")
    _validate_execution_identity(manifest)
    return manifest


def _source_hashes() -> dict[str, str]:
    output = _formal_source_hashes()
    relative = "paper/run_comparator_family_completion.py"
    output[relative] = _sha256_file(ROOT / relative)
    return output


def _job_sort_key(job: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(job[field] for field in CANONICAL_SORT_KEY)


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    result = row.get("result")
    if not isinstance(result, Mapping):
        raise ComparatorFamilyRunError("extension row lacks a result mapping")
    values: dict[str, Any] = {
        "matrix_id": row.get("matrix_id"),
        "policy": row.get("policy"),
        "sim": result.get("sim"),
        "mode": result.get("mode"),
        "protocol_scaffold": result.get("protocol_scaffold"),
        "kap": result.get("kap"),
        "gamma": result.get("gamma"),
        "stratum": row.get("stratum"),
        "seed": row.get("seed"),
    }
    return tuple(values[field] for field in CANONICAL_SORT_KEY)


def _expand_jobs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    common = manifest.get("common_design")
    if not isinstance(common, Mapping):
        raise ComparatorFamilyRunError("completion manifest lacks common_design")
    expected_common = {
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
        "strata": [0, 1],
        "traj": False,
    }
    if dict(common) != expected_common:
        raise ComparatorFamilyRunError("completion manifest common design changed")
    matrices = manifest.get("new_execution_matrices")
    if not isinstance(matrices, list):
        raise ComparatorFamilyRunError("completion manifest lacks execution matrices")
    expected_matrix_counts = {
        "family_full_osa_gbump_boundary_completion": 4_000,
        "family_full_efftox_mariposa_boundary_completion": 3_200,
        "family_gradual_osa_boundary_completion": 1_600,
    }
    if {matrix.get("matrix_id") for matrix in matrices} != set(
        expected_matrix_counts
    ):
        raise ComparatorFamilyRunError("completion matrix identifiers changed")

    jobs: list[dict[str, Any]] = []
    common_job = {key: value for key, value in common.items() if key != "strata"}
    for matrix in matrices:
        matrix_id = str(matrix["matrix_id"])
        if matrix.get("run_class") != "current_harness_comparator":
            raise ComparatorFamilyRunError(f"{matrix_id}: run class changed")
        if matrix.get("policies") != ["tmse", "qBIG"]:
            raise ComparatorFamilyRunError(f"{matrix_id}: policy set/order changed")
        start = int(matrix["seed_start"])
        stop = int(matrix["seed_stop_exclusive"])
        matrix_jobs: list[dict[str, Any]] = []
        for policy in matrix["policies"]:
            for surface in matrix["surfaces"]:
                for gamma in matrix["gammas"]:
                    for stratum in common["strata"]:
                        for seed in range(start, stop):
                            matrix_jobs.append(
                                {
                                    **common_job,
                                    "matrix_id": matrix_id,
                                    "run_class": matrix["run_class"],
                                    "policy": str(policy),
                                    "seed": int(seed),
                                    "sim": str(surface),
                                    "stratum": int(stratum),
                                    "gamma": float(gamma),
                                    "protocol_scaffold": str(
                                        matrix["protocol_scaffold"]
                                    ),
                                    "harness_id": formal_runner.HARNESS_ID,
                                    "evaluator_id": formal_runner.REGISTERED_EVALUATOR_ID,
                                    "historical_wrapper_used": False,
                                }
                            )
        declared = int(matrix["expected_executions"])
        expected = expected_matrix_counts[matrix_id]
        if len(matrix_jobs) != declared or declared != expected:
            raise ComparatorFamilyRunError(
                f"{matrix_id}: Cartesian count {len(matrix_jobs)} != {declared} != {expected}"
            )
        jobs.extend(matrix_jobs)

    jobs.sort(key=_job_sort_key)
    execution_keys = [formal_runner._job_key(job) for job in jobs]
    design_keys = [formal_preflight.design_key(job) for job in jobs]
    if len(jobs) != EXPECTED_EXECUTIONS:
        raise ComparatorFamilyRunError(
            f"completion job count changed: {len(jobs)} != {EXPECTED_EXECUTIONS}"
        )
    if len(execution_keys) != len(set(execution_keys)):
        raise ComparatorFamilyRunError("completion jobs contain duplicate execution keys")
    if len(design_keys) != len(set(design_keys)):
        raise ComparatorFamilyRunError("completion jobs contain duplicate trial designs")
    _assert_no_formal_overlap(jobs)
    return jobs


def _assert_no_formal_overlap(jobs: Sequence[Mapping[str, Any]]) -> None:
    formal_manifest = formal_preflight.load_manifest(FORMAL_MANIFEST_PATH)
    decision, trajectory = formal_preflight.expand_execution_jobs(formal_manifest)
    existing = {
        formal_preflight.design_key(job) for job in (*decision, *trajectory)
    }
    overlap = {
        formal_preflight.design_key(job) for job in jobs
    } & existing
    if overlap:
        example = sorted(overlap, key=repr)[0]
        raise ComparatorFamilyRunError(
            f"completion design overlaps an existing formal execution: {example}"
        )


def _group_jobs(
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        shard = formal_preflight.shard_id(job)
        output.setdefault(shard, []).append(dict(job))
    output = {
        shard: sorted(shard_jobs, key=_job_sort_key)
        for shard, shard_jobs in sorted(output.items())
    }
    if len(output) != EXPECTED_SHARDS:
        raise ComparatorFamilyRunError(
            f"completion shard count changed: {len(output)} != {EXPECTED_SHARDS}"
        )
    return output


def _build_bindings(
    manifest_path: Path = MANIFEST_PATH,
    analysis_spec_path: Path = ANALYSIS_SPEC_PATH,
) -> dict[str, Any]:
    source_sha256 = _source_hashes()
    bindings: dict[str, Any] = {
        "completion_manifest_sha256": _sha256_file(Path(manifest_path)),
        "analysis_spec_sha256": _sha256_file(Path(analysis_spec_path)),
        "formal_manifest_sha256": _sha256_file(FORMAL_MANIFEST_PATH),
        "source_sha256": source_sha256,
        "runtime_identity": _runtime_identity(),
        "ray_contract": {
            "num_cpus": RAY_NUM_CPUS,
            "worker_num_cpus": WORKER_NUM_CPUS,
            "maximum_in_flight": MAX_IN_FLIGHT,
            "serial_fallback": False,
        },
    }
    bindings["launch_fingerprint"] = _sha256_bytes(
        _canonical_json_bytes(bindings)
    )
    return bindings


def _checkpoint_path(output_dir: Path, shard: str) -> Path:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in shard
    )
    if not safe or safe != shard:
        raise ComparatorFamilyRunError(f"unsafe extension shard id {shard!r}")
    return Path(output_dir) / "checkpoints" / f"{safe}.checkpoint.json"


def _validate_extension_row(
    row: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    formal_runner._validate_execution_row(row, job)
    if row.get("run_class") != "current_harness_comparator":
        raise ComparatorFamilyRunError("extension row has the wrong run class")
    if row.get("policy") not in {"tmse", "qBIG"}:
        raise ComparatorFamilyRunError("extension row has an unplanned policy")
    result = row.get("result")
    if not isinstance(result, Mapping) or result.get("traj") is not None:
        raise ComparatorFamilyRunError("extension row unexpectedly captures a trajectory")


def _canonical_rows(
    rows: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    *,
    complete: bool,
) -> list[dict[str, Any]]:
    job_by_key = {formal_runner._job_key(job): dict(job) for job in jobs}
    output: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        raw_key = row.get("formal_execution_key")
        if not isinstance(raw_key, list):
            raise ComparatorFamilyRunError("extension row lacks an execution key")
        key = tuple(raw_key)
        if key not in job_by_key:
            raise ComparatorFamilyRunError("extension row is outside its shard design")
        if key in output:
            raise ComparatorFamilyRunError("extension checkpoint contains duplicate rows")
        _validate_extension_row(row, job_by_key[key])
        output[key] = dict(row)
    if complete and set(output) != set(job_by_key):
        raise ComparatorFamilyRunError(
            f"extension coverage is incomplete: {len(output)}/{len(job_by_key)}"
        )
    return sorted(output.values(), key=_row_sort_key)


def _write_checkpoint(
    path: Path,
    *,
    shard: str,
    rows: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> None:
    canonical = _canonical_rows(rows, jobs, complete=False)
    payload = {
        "schema_version": 1,
        "status": CHECKPOINT_STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "shard_id": shard,
        "expected_row_count": len(jobs),
        "completed_row_count": len(canonical),
        "bindings": dict(bindings),
        "immutable_formal_projection_changed": False,
        "rows": canonical,
    }
    _atomic_json_write(Path(path), payload)


def _load_checkpoint(
    path: Path,
    *,
    shard: str,
    jobs: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComparatorFamilyRunError(
            f"extension checkpoint is unreadable: {path}"
        ) from error
    expected = {
        "schema_version": 1,
        "status": CHECKPOINT_STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "shard_id": shard,
        "expected_row_count": len(jobs),
        "bindings": dict(bindings),
        "immutable_formal_projection_changed": False,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ComparatorFamilyRunError(
                f"extension checkpoint {field} binding mismatch: {path}"
            )
    rows = payload.get("rows")
    if not isinstance(rows, list) or payload.get("completed_row_count") != len(rows):
        raise ComparatorFamilyRunError("extension checkpoint row count is malformed")
    canonical = _canonical_rows(rows, jobs, complete=False)
    if rows != canonical:
        raise ComparatorFamilyRunError("extension checkpoint rows are not canonical")
    return canonical


def _require_ray_capacity(resources: Mapping[str, Any]) -> None:
    cpu = float(resources.get("CPU", 0.0))
    if not math.isfinite(cpu) or cpu < float(RAY_NUM_CPUS):
        raise ComparatorFamilyRunError(
            f"Ray must expose at least {RAY_NUM_CPUS} CPUs; observed {cpu:g}"
        )


def _start_ray() -> None:
    if ray.is_initialized():
        raise ComparatorFamilyRunError(
            "a pre-existing Ray runtime is not accepted; this run must initialize "
            "its frozen ten-CPU contract"
        )
    try:
        ray.init(
            num_cpus=RAY_NUM_CPUS,
            include_dashboard=False,
            ignore_reinit_error=False,
            log_to_driver=False,
        )
        _require_ray_capacity(ray.cluster_resources())
        _require_ray_capacity(ray.available_resources())
    except BaseException:
        if ray.is_initialized():
            ray.shutdown()
        raise


def _ray_worker_run_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one audited formal-harness job inside a Ray worker process."""
    formal_runner._worker_initialize()
    try:
        return formal_runner._run_job(dict(job))
    finally:
        formal_runner._worker_cleanup()


def _run_jobs_ray(
    jobs: Sequence[Mapping[str, Any]],
    *,
    on_result: Callable[[dict[str, Any]], None],
) -> None:
    if not ray.is_initialized():
        raise ComparatorFamilyRunError("Ray is not initialized; serial fallback is prohibited")
    remote_job = ray.remote(
        num_cpus=WORKER_NUM_CPUS,
        max_retries=0,
    )(_ray_worker_run_job)
    iterator = iter(jobs)
    pending: dict[Any, tuple[Any, ...]] = {}

    def fill() -> None:
        while len(pending) < MAX_IN_FLIGHT:
            try:
                job = next(iterator)
            except StopIteration:
                return
            reference = remote_job.remote(dict(job))
            pending[reference] = formal_runner._job_key(job)

    fill()
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1)
        for reference in ready:
            key = pending.pop(reference)
            try:
                row = ray.get(reference)
            except BaseException as error:
                for remaining in pending:
                    ray.cancel(remaining, force=True)
                raise ComparatorFamilyRunError(
                    f"Ray comparator-family task failed: {key}"
                ) from error
            on_result(dict(row))
        fill()


def _compressed_payload_bytes(payload: Mapping[str, Any]) -> tuple[bytes, bytes]:
    uncompressed = _canonical_json_bytes(payload)
    compressor = zstd.ZstdCompressor(
        level=19,
        threads=0,
        write_checksum=True,
        write_content_size=True,
        write_dict_id=False,
    )
    return uncompressed, compressor.compress(uncompressed)


def _raw_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    artifact = Path(output_dir) / "raw" / FINAL_BASENAME
    metadata = Path(str(artifact) + ".metadata.json")
    commit = Path(str(artifact) + ".commit.json")
    return artifact, metadata, commit


def _validate_committed_raw(
    output_dir: Path,
    *,
    jobs: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> Path:
    artifact, metadata_path, commit_path = _raw_paths(output_dir)
    if not artifact.is_file() or not metadata_path.is_file() or not commit_path.is_file():
        raise ComparatorFamilyRunError("committed raw extension artifact is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    expected_commit = {
        "schema_version": 1,
        "status": COMMIT_STATUS,
        "artifact": artifact.name,
        "artifact_sha256": _sha256_file(artifact),
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256_file(metadata_path),
        "row_count": EXPECTED_EXECUTIONS,
        "launch_fingerprint": bindings["launch_fingerprint"],
        "immutable_formal_projection_changed": False,
    }
    for field, value in expected_commit.items():
        if commit.get(field) != value:
            raise ComparatorFamilyRunError(f"raw extension commit mismatch for {field}")
    compressed = artifact.read_bytes()
    try:
        uncompressed = zstd.ZstdDecompressor().decompress(compressed)
        payload = json.loads(uncompressed.decode("utf-8"))
    except BaseException as error:
        raise ComparatorFamilyRunError("raw extension zstd payload is unreadable") from error
    expected_metadata = {
        "schema_version": 1,
        "status": FINAL_STATUS,
        "artifact": artifact.name,
        "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(uncompressed),
        "uncompressed_bytes": len(uncompressed),
        "row_count": EXPECTED_EXECUTIONS,
        "bindings": dict(bindings),
        "immutable_formal_projection_changed": False,
    }
    for field, value in expected_metadata.items():
        if metadata.get(field) != value:
            raise ComparatorFamilyRunError(f"raw extension metadata mismatch for {field}")
    if payload.get("status") != FINAL_STATUS or payload.get("bindings") != dict(bindings):
        raise ComparatorFamilyRunError("raw extension payload provenance mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ComparatorFamilyRunError("raw extension payload lacks rows")
    canonical = _canonical_rows(rows, jobs, complete=True)
    if rows != canonical or len(rows) != EXPECTED_EXECUTIONS:
        raise ComparatorFamilyRunError("raw extension payload coverage/order mismatch")
    return artifact


def _finalize(
    output_dir: Path,
    *,
    jobs: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    bindings: Mapping[str, Any],
) -> Path:
    rows: list[dict[str, Any]] = []
    checkpoint_sha256: dict[str, str] = {}
    for shard, shard_jobs in groups.items():
        checkpoint = _checkpoint_path(output_dir, shard)
        shard_rows = _load_checkpoint(
            checkpoint,
            shard=shard,
            jobs=shard_jobs,
            bindings=bindings,
        )
        if len(shard_rows) != len(shard_jobs):
            raise ComparatorFamilyRunError(
                f"cannot finalize incomplete shard {shard}: "
                f"{len(shard_rows)}/{len(shard_jobs)}"
            )
        rows.extend(shard_rows)
        checkpoint_sha256[str(checkpoint.relative_to(output_dir))] = (
            _sha256_file(checkpoint)
        )
    canonical = _canonical_rows(rows, jobs, complete=True)
    if len(canonical) != EXPECTED_EXECUTIONS:
        raise ComparatorFamilyRunError("raw extension finalization count changed")

    payload = {
        "schema_version": 1,
        "status": FINAL_STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "design": {
            "expected_rows": EXPECTED_EXECUTIONS,
            "canonical_sort_key": list(CANONICAL_SORT_KEY),
            "matrix_row_counts": {
                "family_full_osa_gbump_boundary_completion": 4_000,
                "family_full_efftox_mariposa_boundary_completion": 3_200,
                "family_gradual_osa_boundary_completion": 1_600,
            },
        },
        "bindings": dict(bindings),
        "checkpoint_sha256": checkpoint_sha256,
        "projection_contract": {
            "immutable_formal_projection_rows": IMMUTABLE_FORMAL_PROJECTION_ROWS,
            "immutable_formal_projection_changed": False,
            "extension_projected_into_formal_bundle": False,
            "analysis_scope": "separately authenticated exploratory extension",
        },
        "rows": canonical,
    }
    uncompressed, compressed = _compressed_payload_bytes(payload)
    artifact, metadata_path, commit_path = _raw_paths(output_dir)
    _atomic_bytes_write(artifact, compressed)
    metadata = {
        "schema_version": 1,
        "status": FINAL_STATUS,
        "artifact": artifact.name,
        "artifact_class": ARTIFACT_CLASS,
        "artifact_sha256": _sha256_bytes(compressed),
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(uncompressed),
        "uncompressed_bytes": len(uncompressed),
        "compression": {
            "format": "zstandard",
            "level": 19,
            "threads": 0,
            "checksum": True,
            "content_size": True,
            "dict_id": False,
        },
        "row_count": EXPECTED_EXECUTIONS,
        "bindings": dict(bindings),
        "checkpoint_sha256": checkpoint_sha256,
        "immutable_formal_projection_changed": False,
        "environment": _privacy_safe_environment(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json_write(metadata_path, metadata)
    # This last replacement is the commit point.  Readers must require all
    # hashes in this marker before treating the raw extension as complete.
    _atomic_json_write(
        commit_path,
        {
            "schema_version": 1,
            "status": COMMIT_STATUS,
            "artifact": artifact.name,
            "artifact_sha256": _sha256_file(artifact),
            "metadata": metadata_path.name,
            "metadata_sha256": _sha256_file(metadata_path),
            "row_count": EXPECTED_EXECUTIONS,
            "launch_fingerprint": bindings["launch_fingerprint"],
            "immutable_formal_projection_changed": False,
            "committed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return _validate_committed_raw(
        output_dir,
        jobs=jobs,
        bindings=bindings,
    )


def _assert_separate_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    if any(part.startswith("formal_projection") for part in resolved.parts):
        raise ComparatorFamilyRunError(
            "comparator-family raw output cannot be written inside a formal projection"
        )
    if resolved == formal_runner.DEFAULT_STABLE_Z_STAGING:
        raise ComparatorFamilyRunError(
            "comparator-family raw output cannot reuse formal-rerun staging"
        )
    return resolved


def _authorization_is_valid(*, mutate: bool, token: str | None) -> bool:
    return not mutate or token == AUTHORIZATION_TOKEN


def _select_groups(
    groups: Mapping[str, list[dict[str, Any]]],
    requested: Sequence[str],
) -> list[tuple[str, list[dict[str, Any]]]]:
    if requested:
        unknown = set(requested) - set(groups)
        if unknown:
            raise ComparatorFamilyRunError(
                f"requested unknown extension shards: {sorted(unknown)}"
            )
        names = list(dict.fromkeys(requested))
    else:
        names = list(groups)
    return [(name, groups[name]) for name in names]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--analysis-spec", type=Path, default=ANALYSIS_SPEC_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shard-id", action="append", default=[])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--authorization-token")
    args = parser.parse_args(argv)

    mutating = bool(args.execute or args.finalize_only)
    if not _authorization_is_valid(mutate=mutating, token=args.authorization_token):
        parser.error(
            "execution/finalization requires --authorization-token "
            "COMPARATOR_FAMILY_EXTENSION_GO"
        )
    if args.finalize_only and args.shard_id:
        parser.error("--finalize-only does not accept --shard-id")

    manifest_path = args.manifest.resolve()
    analysis_spec_path = args.analysis_spec.resolve()
    manifest = _load_frozen_manifest(manifest_path, analysis_spec_path)
    jobs = _expand_jobs(manifest)
    groups = _group_jobs(jobs)
    selected = _select_groups(groups, args.shard_id)
    bindings = _build_bindings(manifest_path, analysis_spec_path)
    print(
        "[preflight] new_executions=8,800; shards=40; "
        f"selected_shards={len(selected)}; fingerprint="
        f"{bindings['launch_fingerprint']}",
        flush=True,
    )
    print(
        "[preflight] Ray=required; CPUs=10; worker_CPU=1; max_in_flight=20; "
        "serial_fallback=prohibited; immutable_projection_changed=false",
        flush=True,
    )
    if not mutating:
        print("[preflight] no checkpoint or raw artifact written", flush=True)
        return 0

    output_dir = _assert_separate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.execute:
        state: dict[str, dict[str, Any]] = {}
        pending_jobs: list[dict[str, Any]] = []
        for shard, shard_jobs in selected:
            path = _checkpoint_path(output_dir, shard)
            rows = _load_checkpoint(
                path,
                shard=shard,
                jobs=shard_jobs,
                bindings=bindings,
            )
            complete = {tuple(row["formal_execution_key"]) for row in rows}
            remaining = [
                job
                for job in shard_jobs
                if formal_runner._job_key(job) not in complete
            ]
            state[shard] = {
                "path": path,
                "jobs": shard_jobs,
                "rows": rows,
                "new_since_checkpoint": 0,
            }
            pending_jobs.extend(remaining)
            print(
                f"[resume] {shard}: {len(rows)}/{len(shard_jobs)}; "
                f"remaining={len(remaining)}",
                flush=True,
            )

        key_to_shard = {
            formal_runner._job_key(job): shard
            for shard, shard_jobs in selected
            for job in shard_jobs
        }

        def checkpoint_bucket(shard: str) -> None:
            bucket = state[shard]
            _write_checkpoint(
                bucket["path"],
                shard=shard,
                rows=bucket["rows"],
                jobs=bucket["jobs"],
                bindings=bindings,
            )
            bucket["new_since_checkpoint"] = 0

        def accept(row: dict[str, Any]) -> None:
            key = tuple(row["formal_execution_key"])
            if key not in key_to_shard:
                raise ComparatorFamilyRunError("Ray returned an unplanned execution key")
            shard = key_to_shard[key]
            bucket = state[shard]
            bucket["rows"].append(row)
            bucket["new_since_checkpoint"] += 1
            if bucket["new_since_checkpoint"] >= CHECKPOINT_EVERY:
                checkpoint_bucket(shard)

        if pending_jobs:
            _start_ray()
            try:
                _run_jobs_ray(pending_jobs, on_result=accept)
            finally:
                ray.shutdown()
        for shard, bucket in state.items():
            if bucket["new_since_checkpoint"] or not bucket["path"].exists():
                checkpoint_bucket(shard)
            if len(bucket["rows"]) != len(bucket["jobs"]):
                raise ComparatorFamilyRunError(
                    f"selected shard did not finish: {shard}"
                )

    finalized: Path | None = None
    if args.finalize_only or not args.shard_id:
        finalized = _finalize(
            output_dir,
            jobs=jobs,
            groups=groups,
            bindings=bindings,
        )
    if finalized is not None:
        print(f"[done] {finalized}: {_sha256_file(finalized)}", flush=True)
    elif args.shard_id:
        print("[done] selected shard checkpoints complete; no master implied", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
