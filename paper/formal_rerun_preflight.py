#!/usr/bin/env python3
"""Read-only formal-rerun preflight and atomic-checkpoint primitives.

This module does not call ``run_trial`` and cannot launch a simulation.  It makes
the approved design matrix, resume keys, gate-numerics invariant, archive reuse,
and deep-equality boundary executable before a separate runner is authorized.

The source-path keys and checks in the original run contract are historical.
They require the original study archive and are not proof that the renamed
current package produced the saved results. Expected hashes are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name("formal_rerun_manifest.json")
FORMER_CDF_HARD_STOP_STAGING = (ROOT / "results/formal_rerun_staging").resolve()
ARCHIVED_CDF_HARD_STOP_STAGING = Path(
    "/private/tmp/formal-rerun-cdf-hardstop-20260823"
).resolve()
QUARANTINED_FORMAL_STAGING_ROOTS = (
    FORMER_CDF_HARD_STOP_STAGING,
    ARCHIVED_CDF_HARD_STOP_STAGING,
)
HARNESS_ID = "package-pf-stable-z-v2"
EXACT_EVALUATOR_ID = "piecewise-exact-hard-gated-ckg"
REGISTERED_EVALUATOR_ID = "registered-current-harness"
STABLE_HARNESS_SOURCE_PATHS = (
    "src/safedosebo/acquisitions.py",
    "src/safedosebo/context.py",
    "src/safedosebo/gp.py",
    "src/safedosebo/protocol.py",
    "src/safedosebo/surfaces.py",
    "src/safedosebo/trial.py",
)
EXACT_VALIDATION_INPUT_SOURCES = {
    "primary": ROOT / "results/computational_diagnostics/ckg_1024_anchor_raw.json",
    "gradual": ROOT / "results/protocol_scaffold_sensitivity.json",
}


class PreflightError(RuntimeError):
    """Raised when a rerun invariant is not satisfied."""


def is_quarantined_formal_staging(path: Path) -> bool:
    """Return whether ``path`` is inside a provenance-only failed run tree."""
    resolved = Path(path).resolve()
    return any(
        resolved == root or resolved.is_relative_to(root)
        for root in QUARANTINED_FORMAL_STAGING_ROOTS
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_harness_source_sha256() -> dict[str, str]:
    """Return the complete source identity required by a schema-3 panel."""
    return {
        relative: file_sha256(ROOT / relative)
        for relative in STABLE_HARNESS_SOURCE_PATHS
    }


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise PreflightError("unsupported formal-rerun manifest schema")
    if value.get("status") != "PREFLIGHT_ONLY_NO_SIMULATION_AUTHORIZED":
        raise PreflightError("formal-rerun manifest does not retain its no-run status")
    return value


def _product_count(matrix: Mapping[str, Any]) -> int:
    per_sim = 0
    for sim in matrix["sim"]:
        per_sim += int(matrix["seed_count_by_sim"][sim])
    return (
        per_sim
        * len(matrix["policy"])
        * len(matrix["gamma"])
        * len(matrix["mode"])
        * len(matrix["kap"])
        * len(matrix["protocol_scaffold"])
        * 2
    )


def _job_sort_key(job: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        job["matrix_id"], job["policy"], job["sim"], job["mode"],
        job["protocol_scaffold"], float(job["kap"]), float(job["gamma"]),
        int(job["stratum"]), int(job["seed"]),
    )


def design_key(job: Mapping[str, Any]) -> tuple[Any, ...]:
    """Trial identity excluding matrix labels and trajectory logging."""
    return (
        job["policy"], int(job["seed"]), job["sim"], int(job["stratum"]),
        job["mode"], round(float(job["gamma"]), 12),
        round(float(job["kap"]), 12), job["protocol_scaffold"],
        int(job["budget"]), int(job["warmup"]), int(job["r_k"]),
        int(job["grid_n"]), job["noise"], job["empty_gate"],
        round(float(job["region_step"]), 12),
        int(job["empty_gate_stop_after"]),
        bool(job["exclude_repeats_during_expansion"]),
    )


def execution_key(job: Mapping[str, Any]) -> tuple[Any, ...]:
    return (job["matrix_id"], *design_key(job), bool(job.get("traj", False)))


def trajectory_fulfills_decision(job: Mapping[str, Any]) -> bool:
    """Whether a main decision identity is executed once with trajectory capture."""
    return bool(
        job["matrix_id"] == "main_latent_current_cei_controls"
        and job["policy"] in {"cEI", "cEI-tMSE"}
        and job["sim"] == "osa"
        and int(job["seed"]) < 80
        and math.isclose(float(job["gamma"]), 0.7)
        and job["mode"] == "latent"
        and math.isclose(float(job["kap"]), 1.0)
        and job["protocol_scaffold"] == "lhs_fixed"
    )


def shard_id(job: Mapping[str, Any]) -> str:
    fields = (
        job["matrix_id"], job["policy"], job["sim"], job["mode"],
        job["protocol_scaffold"], f"kap{float(job['kap']):g}",
        f"tau{float(job['gamma']):g}",
    )
    return "__".join(str(value).replace("/", "-") for value in fields)


def _expand_one(matrix: Mapping[str, Any], common: Mapping[str, Any]) -> list[dict[str, Any]]:
    if _product_count(matrix) != int(matrix["expected_executions"]):
        raise PreflightError(f"{matrix['matrix_id']}: declared count does not match Cartesian product")
    output: list[dict[str, Any]] = []
    for policy in matrix["policy"]:
        for sim in matrix["sim"]:
            for seed in range(int(matrix["seed_count_by_sim"][sim])):
                for stratum in common["strata"]:
                    for mode in matrix["mode"]:
                        for scaffold in matrix["protocol_scaffold"]:
                            for kap in matrix["kap"]:
                                for gamma in matrix["gamma"]:
                                    exact = matrix["run_class"] == "exact_ckg"
                                    capture = bool(
                                        matrix["matrix_id"] == "main_full_panel_exact_ckg"
                                        and sim == "osa"
                                        and math.isclose(float(gamma), 0.7)
                                        and seed < 80
                                    )
                                    output.append({
                                        **{key: value for key, value in common.items() if key != "strata"},
                                        "matrix_id": matrix["matrix_id"],
                                        "run_class": matrix["run_class"],
                                        "policy": policy,
                                        "seed": seed,
                                        "sim": sim,
                                        "stratum": int(stratum),
                                        "gamma": float(gamma),
                                        "mode": mode,
                                        "kap": float(kap),
                                        "protocol_scaffold": scaffold,
                                        "traj": bool(matrix.get("traj", False) or capture),
                                        "harness_id": HARNESS_ID,
                                        "evaluator_id": (
                                            EXACT_EVALUATOR_ID if exact else REGISTERED_EVALUATOR_ID
                                        ),
                                        "historical_wrapper_used": False,
                                    })
    output.sort(key=_job_sort_key)
    return output


def expand_jobs(manifest: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = load_manifest() if manifest is None else dict(manifest)
    common = manifest["common_design"]
    decision = [
        job
        for matrix in manifest["decision_matrices"]
        for job in _expand_one(matrix, common)
    ]
    trajectory = [
        job
        for matrix in manifest["trajectory_replay_matrices"]
        for job in _expand_one(matrix, common)
    ]
    decision.sort(key=_job_sort_key)
    trajectory.sort(key=_job_sort_key)
    decision_keys = [design_key(job) for job in decision]
    execution_keys = [execution_key(job) for job in (*decision, *trajectory)]
    if len(decision_keys) != len(set(decision_keys)):
        raise PreflightError("decision matrices contain duplicate trial identities")
    if len(execution_keys) != len(set(execution_keys)):
        raise PreflightError("formal execution matrices contain duplicate execution keys")
    totals = manifest["totals"]
    if len(decision) != int(totals["formal_decision_distinct_designs"]):
        raise PreflightError("formal decision count does not match manifest")
    if len(trajectory) != int(totals["trajectory_replay_executions"]):
        raise PreflightError("trajectory replay count does not match manifest")
    replay_fulfilled = [job for job in decision if trajectory_fulfills_decision(job)]
    replay_designs = {design_key(job) for job in trajectory}
    fulfilled_designs = {design_key(job) for job in replay_fulfilled}
    if (
        len(replay_fulfilled)
        != int(totals["decision_identities_fulfilled_by_trajectory_replay"])
        or replay_designs != fulfilled_designs
    ):
        raise PreflightError(
            "trajectory replays do not exactly cover their matching decision identities"
        )
    decision_executions = len(decision) - len(replay_fulfilled)
    if decision_executions != int(
        totals["decision_executions_excluding_trajectory_fulfilled_identities"]
    ):
        raise PreflightError("non-trajectory decision execution count does not match manifest")
    if decision_executions + len(trajectory) != int(totals["formal_execution_count"]):
        raise PreflightError("deduplicated formal execution total does not match manifest")
    exact = sum(job["run_class"] == "exact_ckg" for job in decision)
    comparator = sum(job["run_class"] == "current_harness_comparator" for job in decision)
    if exact != int(totals["exact_ckg_distinct_designs"]):
        raise PreflightError("exact-cKG count does not match manifest")
    if comparator != int(totals["current_harness_nonckg_distinct_designs"]):
        raise PreflightError("current-harness comparator count does not match manifest")
    return decision, trajectory


def expand_execution_jobs(
    manifest: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return actual executions after trajectory/decision identity deduplication."""
    decision, trajectory = expand_jobs(manifest)
    executed_decision = [
        job for job in decision if not trajectory_fulfills_decision(job)
    ]
    return executed_decision, trajectory


def primary_jobs(manifest: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    decision, _ = expand_jobs(manifest)
    selected = [
        job for job in decision
        if job["matrix_id"] == "main_full_panel_exact_ckg"
        and job["sim"] == "osa"
        and math.isclose(float(job["gamma"]), 0.7)
    ]
    if len(selected) != 400:
        raise PreflightError("primary exact-cKG gate must contain 400 cells")
    if sum(bool(job["traj"]) for job in selected) != 160:
        raise PreflightError("primary gate must capture 160 cKG trajectory payloads in place")
    return selected


def shard_summary(
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    decision, trajectory = expand_execution_jobs(manifest)

    def summarize(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for job in jobs:
            shard = shard_id(job)
            counts[shard] = counts.get(shard, 0) + 1
        histogram: dict[str, int] = {}
        for size in counts.values():
            histogram[str(size)] = histogram.get(str(size), 0) + 1
        return {
            "shard_count": len(counts),
            "size_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
            "execution_count": sum(counts.values()),
        }

    result = {"decision": summarize(decision), "trajectory": summarize(trajectory)}
    expected = (load_manifest() if manifest is None else manifest)["checkpoint_contract"][
        "expected_shards"
    ]
    if result != expected:
        raise PreflightError(f"shard plan differs from manifest: {result}")
    return result


def assert_gate_numerics_equivalent(
    standardized_feasibility: Sequence[float] | np.ndarray,
    tau: float,
    *,
    label: str = "gate state",
) -> None:
    """Validate a reconstructible stable-z gate state.

    Current assignment, exact cKG, and terminal recommendation all use the
    strict comparison ``z > ppf(tau)`` and the same ``argmax(z)`` fallback.
    The CDF is still evaluated here so prelaunch inputs exercise its numerical
    domain, but CDF rounding or saturation is diagnostic only and cannot make a
    valid stable-z state fail this static check.  Runtime Context identity is
    checked separately at every acquisition and terminal state.
    """
    z = np.asarray(standardized_feasibility, dtype=float).reshape(-1)
    if not len(z) or not np.isfinite(z).all():
        raise PreflightError(f"{label}: standardized feasibility must be finite and nonempty")
    if not np.isfinite(float(tau)) or not 0.0 < float(tau) < 1.0:
        raise PreflightError(f"{label}: tau must be strictly between zero and one")
    quantile = float(norm.ppf(float(tau)))
    if not np.isfinite(quantile):
        raise PreflightError(f"{label}: stable gate quantile is not finite")
    # Force both stable and legacy-reporting transforms to be numerically
    # reconstructible.  Their equality is deliberately not a decision gate.
    np.asarray(z > quantile, dtype=bool)
    probability = norm.cdf(z)
    if not np.isfinite(probability).all():
        raise PreflightError(f"{label}: CDF diagnostic is not finite")
    int(np.argmax(z))


def audit_gate_state(
    mu_g: Sequence[float] | np.ndarray,
    sd_g: Sequence[float] | np.ndarray,
    g_dagger: float,
    tau: float,
    *,
    label: str,
) -> None:
    mu = np.asarray(mu_g, dtype=float).reshape(-1)
    sd = np.asarray(sd_g, dtype=float).reshape(-1)
    if mu.shape != sd.shape or (sd <= 0.0).any():
        raise PreflightError(f"{label}: malformed gate mean/SD arrays")
    assert_gate_numerics_equivalent((float(g_dagger) - mu) / sd, tau, label=label)


_COMPARATOR_MATCH_FIELDS = (
    "gate_pass_set_matches",
    "eligible_gate_pass_set_matches",
    "operational_full_fallback_index_matches",
    "operational_eligible_fallback_index_matches",
    "eligible_gate_empty_matches",
)

_COMPARATOR_ZERO_SUMMARY_FIELDS = (
    "gate_pass_set_difference_state_count",
    "eligible_gate_pass_set_difference_state_count",
    "full_fallback_index_difference_state_count",
    "eligible_fallback_index_difference_state_count",
    "eligible_gate_empty_difference_state_count",
)

_COMPARATOR_DIAGNOSTIC_SUMMARY_FIELDS = (
    "inactive_full_fallback_rounding_difference_state_count",
    "inactive_eligible_fallback_rounding_difference_state_count",
    "operational_full_fallback_rounding_difference_state_count",
    "operational_eligible_fallback_rounding_difference_state_count",
    "cdf_gate_pass_set_rounding_difference_state_count",
    "cdf_eligible_gate_pass_set_rounding_difference_state_count",
    "cdf_saturated_zero_total",
    "cdf_saturated_one_total",
)

_COMPARATOR_DIAGNOSTIC_STATE_FIELDS = (
    "inactive_full_fallback_rounding_difference",
    "inactive_eligible_fallback_rounding_difference",
    "operational_full_fallback_rounding_difference",
    "operational_eligible_fallback_rounding_difference",
)


def audit_exact_validation_artifact(path: Path, *, require_sidecar: bool = True) -> dict[str, int]:
    """Require the complete schema-3 stable-z evaluator PASS before any OC launch."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_status = (
        "PASS_DETERMINISTIC_PIECEWISE_ANALYTIC_VALIDATION_"
        "EXACT_UP_TO_FLOATING_POINT_STABLE_Z_HARNESS_"
        "REGISTERED_CKG_EVALUATOR_UNCHANGED"
    )
    if payload.get("schema_version") != 3:
        raise PreflightError("exact validation panel is not the complete schema-3 artifact")
    if payload.get("summary", {}).get("status") != expected_status:
        raise PreflightError("exact validation panel does not carry the required PASS status")
    implementation = payload.get("implementation", {})
    if implementation.get("production_default_changed") is not False:
        raise PreflightError("exact validation panel changed the production cKG default")
    if implementation.get("production_harness_gate_semantics") != "stable_z_strict_ppf":
        raise PreflightError("exact validation panel did not validate the stable-z harness")
    source_checks = {
        ROOT / "src/safedosebo/ckg_exact.py": implementation.get("exact_module_sha256"),
        ROOT / "paper/run_ckg_piecewise_exact_validation.py": implementation.get(
            "generator_sha256"
        ),
    }
    for source, recorded in source_checks.items():
        if recorded != file_sha256(source):
            raise PreflightError(f"exact validation source hash mismatch: {source.name}")
    expected_stable_sources = stable_harness_source_sha256()
    recorded_stable_sources = implementation.get("stable_harness_source_sha256")
    if not isinstance(recorded_stable_sources, dict):
        raise PreflightError("exact validation panel lacks stable-harness source hashes")
    if set(recorded_stable_sources) != set(expected_stable_sources):
        raise PreflightError("exact validation stable-harness source key set mismatch")
    for relative, expected_sha256 in expected_stable_sources.items():
        if recorded_stable_sources.get(relative) != expected_sha256:
            raise PreflightError(
                f"exact validation stable-harness source hash mismatch: {relative}"
            )

    recorded_inputs = payload.get("sources")
    if not isinstance(recorded_inputs, dict) or set(recorded_inputs) != set(
        EXACT_VALIDATION_INPUT_SOURCES
    ):
        raise PreflightError("exact validation input-source key set mismatch")
    expected_input_hashes: dict[str, str] = {}
    for source_name, source_path in EXACT_VALIDATION_INPUT_SOURCES.items():
        metadata_path = Path(str(source_path) + ".metadata.json")
        if not source_path.is_file() or not metadata_path.is_file():
            raise PreflightError(
                f"exact validation input source or metadata is absent: {source_name}"
            )
        source_record = recorded_inputs.get(source_name)
        if not isinstance(source_record, dict):
            raise PreflightError(
                f"exact validation input source is malformed: {source_name}"
            )
        expected_source_sha256 = file_sha256(source_path)
        expected_metadata_sha256 = file_sha256(metadata_path)
        expected_input_hashes[source_path.name] = expected_source_sha256
        if source_record.get("path") != source_path.name:
            raise PreflightError(
                f"exact validation input source path mismatch: {source_name}"
            )
        if source_record.get("sha256") != expected_source_sha256:
            raise PreflightError(
                f"exact validation input source hash mismatch: {source_name}"
            )
        if source_record.get("metadata_sha256") != expected_metadata_sha256:
            raise PreflightError(
                f"exact validation input metadata hash mismatch: {source_name}"
            )

    primary_states = payload.get("primary_states")
    sentinel_states = payload.get("sentinel_states")
    if not isinstance(primary_states, list) or len(primary_states) != 720:
        raise PreflightError("exact validation panel must contain 720 primary states")
    if not isinstance(sentinel_states, list) or len(sentinel_states) != 28:
        raise PreflightError("exact validation panel must contain 28 sentinel states")
    summary = payload["summary"]
    primary = summary.get("primary", {})
    sentinels = summary.get("sentinels", {})
    required_primary = {
        "state_count": 720,
        "trajectory_count": 40,
        "steps_per_trajectory": 18,
        "source_512_decision_reconstruction_match_count": 720,
        "source_1024_decision_reconstruction_match_count": 720,
    }
    if any(primary.get(key) != value for key, value in required_primary.items()):
        raise PreflightError("exact validation primary reconstruction is incomplete")
    required_sentinels = {
        "state_count": 28,
        "four_testbed_five_gate_state_count": 20,
        "predictive_gate_state_count": 4,
        "gradual_access_state_count": 4,
    }
    if any(sentinels.get(key) != value for key, value in required_sentinels.items()):
        raise PreflightError("exact validation sentinel coverage is incomplete")
    for panel_name, panel_summary in (
        ("primary", primary.get("current_comparator_audit", {})),
        ("sentinels", sentinels.get("current_comparator_audit", {})),
    ):
        if any(panel_summary.get(field) != 0 for field in _COMPARATOR_ZERO_SUMMARY_FIELDS):
            raise PreflightError(f"{panel_name}: stable-z/current-Context summary differs")
        for field in _COMPARATOR_DIAGNOSTIC_SUMMARY_FIELDS:
            value = panel_summary.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PreflightError(f"{panel_name}: malformed CDF diagnostic {field}")
    for index, state in enumerate((*primary_states, *sentinel_states)):
        audit = state.get("current_comparator_audit", {})
        if not all(audit.get(field) is True for field in _COMPARATOR_MATCH_FIELDS):
            raise PreflightError(
                f"exact validation state {index} has a stable-z/current-Context difference"
            )
        if audit.get("current_definition") != "stable z>ppf(gamma), fallback argmax z":
            raise PreflightError(f"exact validation state {index} lacks stable-z semantics")
        if audit.get("cdf_diagnostic_definition") != (
            "cdf(z)>gamma, fallback argmax cdf(z)"
        ):
            raise PreflightError(f"exact validation state {index} lacks CDF diagnostics")
        if any(
            not isinstance(audit.get(field), bool)
            for field in _COMPARATOR_DIAGNOSTIC_STATE_FIELDS
        ):
            raise PreflightError(f"exact validation state {index} has malformed diagnostics")
        for field in (
            "cdf_gate_pass_set_difference_indices",
            "cdf_eligible_gate_pass_set_difference_indices",
        ):
            if not isinstance(audit.get(field), list):
                raise PreflightError(
                    f"exact validation state {index} has malformed {field}"
                )

    if require_sidecar:
        sidecar_path = Path(str(path) + ".metadata.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("schema_version") != 3:
            raise PreflightError("exact validation panel sidecar is not schema-3")
        if sidecar.get("artifact_sha256") != file_sha256(path):
            raise PreflightError("exact validation panel sidecar hash mismatch")
        if sidecar.get("production_default_changed") is not False:
            raise PreflightError("exact validation sidecar changed the production default")
        if sidecar.get("production_harness_gate_semantics") != "stable_z_strict_ppf":
            raise PreflightError("exact validation sidecar did not bind stable-z semantics")
        if sidecar.get("generator_sha256") != source_checks[
            ROOT / "paper/run_ckg_piecewise_exact_validation.py"
        ]:
            raise PreflightError("exact validation sidecar generator hash mismatch")
        if sidecar.get("exact_module_sha256") != source_checks[
            ROOT / "src/safedosebo/ckg_exact.py"
        ]:
            raise PreflightError("exact validation sidecar exact-module hash mismatch")
        if sidecar.get("stable_harness_source_sha256") != expected_stable_sources:
            raise PreflightError(
                "exact validation sidecar stable-harness source hashes mismatch"
            )
        if sidecar.get("source_sha256") != expected_input_hashes:
            raise PreflightError("exact validation sidecar input-source hashes mismatch")
    return {"primary_states": len(primary_states), "sentinel_states": len(sentinel_states)}


def _atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace one checkpoint without exposing a truncated JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, separators=(",", ":"), sort_keys=True, allow_nan=False)
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


def checkpoint_payload(
    *,
    manifest_sha256: str,
    source_sha256: Mapping[str, str],
    exact_validation_sha256: str,
    shard: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "PARTIAL_NOT_FOR_ANALYSIS",
        "manifest_sha256": manifest_sha256,
        "source_sha256": dict(source_sha256),
        "exact_validation_sha256": exact_validation_sha256,
        "shard_id": shard,
        "harness_id": HARNESS_ID,
        "rows": [dict(row) for row in rows],
    }


def write_checkpoint(
    path: Path,
    *,
    manifest_sha256: str,
    source_sha256: Mapping[str, str],
    exact_validation_sha256: str,
    shard: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    _atomic_write_json(
        path,
        checkpoint_payload(
            manifest_sha256=manifest_sha256,
            source_sha256=source_sha256,
            exact_validation_sha256=exact_validation_sha256,
            shard=shard,
            rows=rows,
        ),
    )


def validate_checkpoint(
    path: Path,
    *,
    manifest_sha256: str,
    source_sha256: Mapping[str, str],
    exact_validation_sha256: str,
    shard: str,
    expected_evaluator_by_key: Mapping[tuple[Any, ...], str],
) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise PreflightError("checkpoint schema mismatch")
    if payload.get("status") != "PARTIAL_NOT_FOR_ANALYSIS":
        raise PreflightError("checkpoint status is not partial")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise PreflightError("checkpoint manifest hash mismatch")
    if payload.get("source_sha256") != dict(source_sha256):
        raise PreflightError("checkpoint execution-source hash mismatch")
    if payload.get("exact_validation_sha256") != exact_validation_sha256:
        raise PreflightError("checkpoint exact-validation hash mismatch")
    if payload.get("shard_id") != shard or payload.get("harness_id") != HARNESS_ID:
        raise PreflightError("checkpoint shard/harness mismatch")
    expected = set(expected_evaluator_by_key)
    observed: set[tuple[Any, ...]] = set()
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise PreflightError("checkpoint rows must be a list")
    for index, row in enumerate(rows):
        if row.get("harness_id") != HARNESS_ID or row.get("historical_wrapper_used") is not False:
            raise PreflightError(f"checkpoint row {index} has prohibited harness provenance")
        raw_key = row.get("formal_execution_key")
        if not isinstance(raw_key, list):
            raise PreflightError(f"checkpoint row {index} lacks formal_execution_key")
        key = tuple(raw_key)
        if key not in expected or key in observed:
            raise PreflightError(f"checkpoint row {index} has unexpected/duplicate key")
        if row.get("evaluator_id") != expected_evaluator_by_key[key]:
            raise PreflightError(f"checkpoint row {index} has the wrong evaluator id")
        gate_audit = row.get("gate_numerics_audit")
        if not isinstance(gate_audit, dict):
            raise PreflightError(f"checkpoint row {index} lacks gate-numerics audit")
        if int(gate_audit.get("checked_state_count", 0)) < 1:
            raise PreflightError(f"checkpoint row {index} has no audited gate state")
        difference_fields = (
            "gate_pass_set_difference_count",
            "eligible_gate_pass_set_difference_count",
            "operational_full_fallback_index_difference_count",
            "operational_eligible_fallback_index_difference_count",
        )
        if any(gate_audit.get(field) != 0 for field in difference_fields):
            raise PreflightError(f"checkpoint row {index} has a gate-numerics difference")
        if gate_audit.get("terminal_state_checked") is not True:
            raise PreflightError(f"checkpoint row {index} lacks terminal gate audit")
        result = row.get("result")
        if not isinstance(result, dict):
            raise PreflightError(f"checkpoint row {index} lacks a result object")
        recommendation_made = result.get("recommendation_made")
        if recommendation_made is True:
            expected_index = gate_audit.get("terminal_recommendation_expected_index")
            saved_index = gate_audit.get("terminal_recommendation_saved_index")
            if (
                gate_audit.get("terminal_recommendation_status") != "checked"
                or gate_audit.get(
                    "terminal_recommendation_index_difference_count"
                ) != 0
                or not isinstance(expected_index, int)
                or isinstance(expected_index, bool)
                or not isinstance(saved_index, int)
                or isinstance(saved_index, bool)
                or expected_index != saved_index
            ):
                raise PreflightError(
                    f"checkpoint row {index} has a terminal recommendation difference"
                )
        elif recommendation_made is False:
            if (
                result.get("stop_reason") != "NO_FEASIBLE_DOSE"
                or gate_audit.get("terminal_recommendation_status")
                != "not_applicable_stopped"
                or gate_audit.get(
                    "terminal_recommendation_index_difference_count"
                ) is not None
                or gate_audit.get("terminal_recommendation_expected_index") is not None
                or gate_audit.get("terminal_recommendation_saved_index") is not None
            ):
                raise PreflightError(
                    f"checkpoint row {index} has a malformed stopped-recommendation audit"
                )
        else:
            raise PreflightError(
                f"checkpoint row {index} lacks recommendation status for terminal audit"
            )
        observed.add(key)
    return [dict(row) for row in rows]


def merge_rows_preserving_unapproved(
    original: Sequence[Mapping[str, Any]],
    replacements: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
    allowed_keys: Iterable[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    """Replace exactly the authorized keys while preserving every other row."""
    allowed = set(allowed_keys)
    if set(replacements) != allowed:
        missing = allowed - set(replacements)
        extra = set(replacements) - allowed
        raise PreflightError(f"replacement coverage mismatch: {len(missing)} missing, {len(extra)} extra")
    original_keys = [tuple(row[field] for field in key_fields) for row in original]
    if len(original_keys) != len(set(original_keys)):
        raise PreflightError("original rows contain duplicate keys")
    if not allowed <= set(original_keys):
        raise PreflightError("replacement key is absent from original rows")
    output: list[dict[str, Any]] = []
    for row, key in zip(original, original_keys):
        candidate = dict(replacements[key]) if key in allowed else dict(row)
        if key not in allowed and candidate != row:
            raise AssertionError("unapproved row changed during merge")
        output.append(candidate)
    if len(output) != len(original):
        raise AssertionError("merge changed the row count")
    return output


_RECORD_KEYS: dict[str, tuple[str, ...]] = {
    "osa_main.json.zst": ("policy", "seed", "sim", "stratum", "mode", "gamma"),
    "gbump_main.json.zst": ("policy", "seed", "sim", "stratum", "mode", "gamma"),
    "efftox_main.json.zst": ("policy", "seed", "sim", "stratum", "mode", "gamma"),
    "mariposa_main.json.zst": ("policy", "seed", "sim", "stratum", "mode", "gamma"),
    "gate_sixway_supplemental.json.zst": ("policy", "seed", "sim", "stratum", "mode", "gamma"),
    "efftox_mariposa_baselines_supplemental.json.zst": ("policy", "seed", "sim", "stratum", "mode", "gamma"),
    "kappa_sweep_fixed_supplemental.json.zst": ("policy", "seed", "stratum", "gamma", "kap"),
    "osa_trajectory_supplemental.json.zst": ("policy", "seed", "sim", "stratum", "mode", "gamma"),
    "protocol_scaffold_sensitivity.json.zst": ("protocol_scaffold", "gamma", "stratum", "policy", "seed"),
    "schedule_ratio_sensitivity_2to1.json.zst": ("policy", "seed", "stratum"),
}

_MAIN_RECORD_FILES = (
    "osa_main.json.zst",
    "gbump_main.json.zst",
    "efftox_main.json.zst",
    "mariposa_main.json.zst",
)

_CANONICAL_POLICY = {
    "cEI": "cEI",
    "cKG": "cKG",
    "cKG1fix": "cKG",
    "cEI-tMSE": "cEI-tMSE",
    "GBE": "cEI-tMSE",
    "tmse": "tmse",
    "straddle": "tmse",
    "qBIG": "qBIG",
    "random": "random",
}


def _canonical_policy(policy: Any) -> str:
    try:
        return _CANONICAL_POLICY[str(policy)]
    except KeyError as exc:
        raise PreflightError(f"unknown policy alias {policy!r}") from exc


def _main_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _canonical_policy(row["policy"]), int(row["seed"]), row["sim"],
        int(row["stratum"]), row["mode"], float(row["gamma"]),
    )


def _assert_main_projection_equal(
    reference: Mapping[str, Any],
    reused: Mapping[str, Any],
    *,
    target_name: str,
    ignored_reference_fields: Iterable[str] = (),
) -> None:
    """Prove a supplemental row is an exact projection of one main-run row.

    Supplemental schemas may add fields (for example ``kap`` or protocol
    diagnostics), but every retained main-schema field must be present and
    deeply equal.  Historical policy labels are canonicalized before comparison.
    """
    ignored = set(ignored_reference_fields)
    fields = set(reference) - ignored
    missing = fields - set(reused)
    if missing:
        raise PreflightError(
            f"{target_name}: reuse row lacks main fields {sorted(missing)} at "
            f"{_main_identity(reused)}"
        )
    for field in sorted(fields):
        expected = reference[field]
        observed = reused[field]
        if field == "policy":
            expected = _canonical_policy(expected)
            observed = _canonical_policy(observed)
        if observed != expected:
            raise PreflightError(
                f"{target_name}: reuse drift in {field!r} at {_main_identity(reused)}"
            )


def audit_frozen_reuse_identity(records: Path) -> dict[str, int]:
    """Deep-check every documented cross-file reuse in the frozen archive.

    This is a read-only audit.  It establishes that the logical duplicate rows
    can be generated from one formal execution without assuming equality from
    matching labels alone.
    """
    records = Path(records)
    main_rows = [
        row
        for filename in _MAIN_RECORD_FILES
        for row in _load_zstd_json(records / filename)
    ]
    main_index = {_main_identity(row): row for row in main_rows}
    if len(main_index) != len(main_rows):
        raise PreflightError("main archives contain duplicate canonical design identities")

    specifications: tuple[
        tuple[str, str, Any, frozenset[str], int, int], ...
    ] = (
        (
            "gate_latent_main_reuse",
            "gate_sixway_supplemental.json.zst",
            lambda row: row["mode"] == "latent"
            and row["policy"] in {"cEI", "cKG1fix", "GBE"},
            frozenset(),
            6_000,
            2_000,
        ),
        (
            "baseline_main_reuse",
            "efftox_mariposa_baselines_supplemental.json.zst",
            lambda row: row["policy"] in {"cEI", "cKG1fix", "GBE"},
            frozenset(),
            1_200,
            400,
        ),
        (
            "kappa_reference_main_reuse",
            "kappa_sweep_fixed_supplemental.json.zst",
            lambda row: math.isclose(float(row["kap"]), 1.0),
            frozenset(),
            1_200,
            600,
        ),
        (
            "trajectory_final_field_reuse",
            "osa_trajectory_supplemental.json.zst",
            lambda row: True,
            frozenset({"traj"}),
            480,
            160,
        ),
        (
            "protocol_lhs_main_reuse",
            "protocol_scaffold_sensitivity.json.zst",
            lambda row: row["protocol_scaffold"] == "lhs_fixed",
            frozenset(),
            2_400,
            800,
        ),
    )

    summary: dict[str, int] = {}
    total = ckg_total = 0
    for name, filename, selector, ignored, expected, expected_ckg in specifications:
        selected = [row for row in _load_zstd_json(records / filename) if selector(row)]
        identities = [_main_identity(row) for row in selected]
        if len(identities) != len(set(identities)):
            raise PreflightError(f"{name}: duplicate reuse identity")
        for row, identity in zip(selected, identities):
            try:
                reference = main_index[identity]
            except KeyError as exc:
                raise PreflightError(f"{name}: no matching main row at {identity}") from exc
            _assert_main_projection_equal(
                reference,
                row,
                target_name=name,
                ignored_reference_fields=ignored,
            )
        ckg_count = sum(_canonical_policy(row["policy"]) == "cKG" for row in selected)
        if len(selected) != expected or ckg_count != expected_ckg:
            raise PreflightError(
                f"{name}: expected {expected} rows/{expected_ckg} cKG, "
                f"observed {len(selected)}/{ckg_count}"
            )
        summary[name] = len(selected)
        total += len(selected)
        ckg_total += ckg_count

    if (total, ckg_total) != (11_280, 3_960):
        raise PreflightError(
            f"frozen reuse total mismatch: all={total}, cKG={ckg_total}"
        )
    summary["total_reuse_appearances"] = total
    summary["ckg_reuse_appearances"] = ckg_total
    return summary


def _replacement_allowed(filename: str, row: Mapping[str, Any]) -> bool:
    del row
    if filename not in _RECORD_KEYS:
        raise PreflightError(f"unknown record file {filename!r}")
    # Once the package fallback is standardized on z/ppf, every claim-supporting
    # policy row must come from the same verified harness.  No historical
    # CDF-fallback row is retained merely because its policy is not cKG.
    return True


def _load_zstd_json(path: Path) -> list[dict[str, Any]]:
    try:
        import zstandard as zstd
    except ImportError as exc:  # pragma: no cover - package environment has zstandard
        raise PreflightError("zstandard is required for archive deep-equality audit") from exc
    with Path(path).open("rb") as stream:
        value = json.loads(zstd.ZstdDecompressor().stream_reader(stream).read())
    if not isinstance(value, list):
        raise PreflightError(f"{path}: expected a JSON list")
    return value


def audit_archive_deep_equality(baseline_records: Path, candidate_records: Path) -> dict[str, int]:
    """Reject any candidate drift outside the explicitly approved row boundary."""
    unchanged = approved = total = 0
    for filename, key_fields in _RECORD_KEYS.items():
        baseline = _load_zstd_json(Path(baseline_records) / filename)
        candidate = _load_zstd_json(Path(candidate_records) / filename)
        baseline_index = {tuple(row[field] for field in key_fields): row for row in baseline}
        candidate_index = {tuple(row[field] for field in key_fields): row for row in candidate}
        if len(baseline_index) != len(baseline) or len(candidate_index) != len(candidate):
            raise PreflightError(f"{filename}: duplicate record key")
        if set(baseline_index) != set(candidate_index):
            raise PreflightError(f"{filename}: candidate changed the record key set")
        for key, original in baseline_index.items():
            revised = candidate_index[key]
            total += 1
            if _replacement_allowed(filename, original):
                approved += 1
            else:
                unchanged += 1
                if revised != original:
                    raise PreflightError(f"{filename}: unapproved row drift at {key}")
    if (total, approved, unchanged) != (55480, 55480, 0):
        raise PreflightError(
            f"archive row boundary mismatch: total={total}, approved={approved}, unchanged={unchanged}"
        )
    return {"total_rows": total, "approved_replacement_rows": approved, "unchanged_rows": unchanged}


def _read_gate_states(path: Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("states")
    if not isinstance(value, list):
        raise PreflightError("gate-state input must be a list or {'states': [...]} object")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--gate-states",
        type=Path,
        help="read-only JSON of reconstructible {mu_g, sd_g, g_dagger, tau, label} states",
    )
    parser.add_argument(
        "--exact-validation-panel",
        type=Path,
        help="schema-3 stable-z exact-evaluator PASS panel required before an OC launch",
    )
    parser.add_argument("--baseline-records", type=Path)
    parser.add_argument("--candidate-records", type=Path)
    parser.add_argument(
        "--frozen-records",
        type=Path,
        help="read-only record directory for documented cross-file reuse identities",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    decision, trajectory = expand_jobs(manifest)
    decision_executions, trajectory_executions = expand_execution_jobs(manifest)
    primary = primary_jobs(manifest)
    shards = shard_summary(manifest)
    print(f"manifest: {args.manifest} ({file_sha256(args.manifest)})")
    print(f"decision identities: {len(decision):,}")
    print(f"non-trajectory decision executions: {len(decision_executions):,}")
    print(f"trajectory replays: {len(trajectory_executions):,}")
    print(
        f"formal executions: {len(decision_executions) + len(trajectory_executions):,}"
    )
    print(f"primary gated cells: {len(primary):,}")
    print(
        "checkpoint shards: "
        f"{shards['decision']['shard_count']} decision + "
        f"{shards['trajectory']['shard_count']} trajectory"
    )
    print("simulation authorization: ABSENT (preflight only)")

    if args.gate_states:
        states = _read_gate_states(args.gate_states)
        for index, state in enumerate(states):
            audit_gate_state(
                state["mu_g"], state["sd_g"], state["g_dagger"], state["tau"],
                label=str(state.get("label", f"state {index}")),
            )
        print(f"gate-numerics states: {len(states):,} PASS")

    if args.exact_validation_panel:
        exact_panel = audit_exact_validation_artifact(args.exact_validation_panel)
        print(f"exact evaluator validation: {exact_panel} PASS")

    if args.frozen_records:
        reuse = audit_frozen_reuse_identity(args.frozen_records)
        print(f"frozen reuse identity: {reuse}")

    if bool(args.baseline_records) != bool(args.candidate_records):
        raise PreflightError("baseline and candidate record paths must be supplied together")
    if args.baseline_records:
        summary = audit_archive_deep_equality(args.baseline_records, args.candidate_records)
        print(f"archive deep equality: {summary}")


if __name__ == "__main__":
    main()
