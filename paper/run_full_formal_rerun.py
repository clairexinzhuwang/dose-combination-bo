#!/usr/bin/env python3
"""Staged runner for the locked 44,200-execution stable-z formal rerun.

The default invocation is a read-only provenance preflight.  Execution and
finalization additionally require ``--authorization-token FULL_FORMAL_RUN_GO``.
The completed, authenticated 400-cell primary gate is imported as already-run
decision identities; it is never silently recomputed.  A future authorized run
therefore launches 43,480 remaining non-trajectory decision trials and 320
trajectory replays.  Those replays also fulfill their 320 matching decision
identities, retaining 44,200 distinct decision identities in 44,200 executions.

Every trial context and the terminal context are audited for stable-z agreement
with the current package Context.  Any pass-set difference, or any fallback
index difference in a branch where that fallback is operational, invalidates
the row and stops execution.  CDF rounding is diagnostic only.  Checkpoints and
complete artifacts are written only to a new staging root using durable atomic
replacement.

The source-path keys and checks in the original run contract are historical.
They require the original study archive and are not proof that the renamed
current package produced the saved results. Expected hashes are unchanged.
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
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gpytorch
import numpy as np
import scipy
import torch
from scipy.stats import norm

from paper import analyze_primary_osa_exact_ckg_gate as primary_analyzer
from paper import formal_rerun_preflight as preflight
from paper import run_primary_osa_exact_ckg_gate as primary_runner
from dose_combination_bo.ckg_exact import ckg_scores_exact
from dose_combination_bo.context import Context as _BaseContext
from dose_combination_bo.gp import fit_gp, post_latent
from dose_combination_bo.protocol import initialization_size
from dose_combination_bo.registry import ACQUISITIONS
from dose_combination_bo.surfaces import resolve_surface
from dose_combination_bo import trial as trial_module


AUTHORIZATION_TOKEN = "FULL_FORMAL_RUN_GO"
EXACT_WORKER_POLICY = "__formal_full_exact_ckg"
SCHEDULE_WORKER_POLICY = "__formal_full_cei_tmse_2to1"
FORMAL_EXACT_POLICY = "cKG-exact-formal"
FORMAL_SCHEDULE_POLICY = "cEI-tMSE-2to1"
HARNESS_ID = preflight.HARNESS_ID
EXACT_EVALUATOR_ID = preflight.EXACT_EVALUATOR_ID
REGISTERED_EVALUATOR_ID = preflight.REGISTERED_EVALUATOR_ID
CHECKPOINT_EVERY = 12
MAX_WORKERS = 8
PRIMARY_CELL_COUNT = 400
DECISION_IDENTITY_COUNT = 44_200
DECISION_COUNT = 43_880
TRAJECTORY_COUNT = 320
FORMAL_EXECUTION_COUNT = 44_200
NEW_EXECUTION_COUNT_AFTER_PRIMARY = 43_800
ARTIFACT_CLASS = "formal_current_harness_staging_not_archive_records"
DECISION_STATUS = "COMPLETE_FORMAL_DECISION_MASTER"
TRAJECTORY_STATUS = "COMPLETE_FORMAL_TRAJECTORY_REPLAYS"
BUNDLE_STATUS = "COMPLETE_FORMAL_EXECUTION_BUNDLE"
CANONICAL_SORT_KEY = (
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

_STATIC_SOURCE_FILES = (
    "paper/run_full_formal_rerun.py",
    "paper/formal_rerun_preflight.py",
    "paper/formal_rerun_manifest.json",
    "paper/run_primary_osa_exact_ckg_gate.py",
    "paper/analyze_primary_osa_exact_ckg_gate.py",
)

SCHEDULE_POST_HOC_FIELDS: dict[str, Any] = {
    "post_hoc_descriptive": True,
    "schedule_tuned": False,
    "schedule_ratio": "2:1 cEI:tMSE",
    "schedule_cycle": ["cEI", "cEI", "tMSE"],
    "schedule_phase_origin": "first adaptive cohort",
    "schedule_review_status": (
        "fixed before this post-hoc review run after the original study results were known"
    ),
}
LEGACY_HARD_STOP_STAGING = preflight.FORMER_CDF_HARD_STOP_STAGING
ARCHIVED_HARD_STOP_STAGING = preflight.ARCHIVED_CDF_HARD_STOP_STAGING
DEFAULT_STABLE_Z_STAGING = (ROOT / "results/formal_rerun_stable_z_staging").resolve()


class FormalRunError(RuntimeError):
    """Raised when a formal-run identity or runtime invariant fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return preflight.file_sha256(path)


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _atomic_json_write(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json_bytes(value))
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


def _runtime_identity() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "gpytorch": gpytorch.__version__,
    }


def _privacy_safe_environment() -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "gpytorch": gpytorch.__version__,
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
            "No username, home path, hostname, serial number, or persistent device "
            "identifier is recorded."
        ),
    }


def _require_validation_runtime_match(panel_path: Path) -> dict[str, str]:
    panel = json.loads(Path(panel_path).read_text(encoding="utf-8"))
    recorded = panel.get("environment", {})
    current = _runtime_identity()
    mismatch = {
        key: {"validated": recorded.get(key), "current": value}
        for key, value in current.items()
        if recorded.get(key) != value
    }
    if mismatch:
        raise FormalRunError(
            f"formal runtime differs from exact-validation runtime: {mismatch}"
        )
    return current


def _required_source_files(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    execution = tuple(manifest["artifact_contract"]["execution_source_files"])
    expected_execution = (
        "src/safedosebo/acquisitions.py",
        "src/safedosebo/ckg_exact.py",
        "src/safedosebo/context.py",
        "src/safedosebo/gp.py",
        "src/safedosebo/metrics.py",
        "src/safedosebo/protocol.py",
        "src/safedosebo/registry.py",
        "src/safedosebo/surfaces.py",
        "src/safedosebo/trial.py",
    )
    if execution != expected_execution:
        raise FormalRunError("locked manifest execution-source list changed")
    return (*_STATIC_SOURCE_FILES, *execution)


def _source_hashes(
    manifest: Mapping[str, Any], manifest_path: Path
) -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in _required_source_files(manifest):
        source = ROOT / relative
        if relative == "paper/formal_rerun_manifest.json":
            source = Path(manifest_path)
        if not source.is_file():
            raise FormalRunError(f"execution source is absent: {relative}")
        output[relative] = _sha256_file(source)
    return output


def _authorization_is_valid(*, mutate: bool, token: str | None) -> bool:
    return not mutate or token == AUTHORIZATION_TOKEN


def _assert_not_legacy_staging(path: Path) -> Path:
    """Keep the failed CDF-era staging tree immutable and provenance-only."""
    resolved = Path(path).resolve()
    if preflight.is_quarantined_formal_staging(resolved):
        raise FormalRunError(
            "historical hard-stop staging is provenance-only and cannot be resumed"
        )
    return resolved


def _all_jobs(
    manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decision, trajectory = preflight.expand_execution_jobs(manifest)
    if (len(decision), len(trajectory), len(decision) + len(trajectory)) != (
        DECISION_COUNT,
        TRAJECTORY_COUNT,
        FORMAL_EXECUTION_COUNT,
    ):
        raise FormalRunError("locked manifest execution counts changed")
    if tuple(manifest["checkpoint_contract"]["canonical_sort_key"]) != CANONICAL_SORT_KEY:
        raise FormalRunError("locked manifest canonical sort key changed")
    failed = manifest["checkpoint_contract"].get("historical_failed_staging", {})
    if failed != {
        "path": "/private/tmp/formal-rerun-cdf-hardstop-20260823",
        "former_path": "results/formal_rerun_staging",
        "status": "PROVENANCE_ONLY_NEVER_RESUME",
        "read_only_inventory": {
            "checkpoint_payloads": 8,
            "commit_sidecars": 8,
            "temporary_files": 0,
            "decision_masters": 0,
            "trajectory_masters": 0,
        },
        "reason": (
            "Contains only valid committed CDF-era partial checkpoints preceding the "
            "operational fallback hard stop; the expanded stable-z run has a different "
            "manifest, harness id, source map, and output root."
        ),
    }:
        raise FormalRunError("historical hard-stop staging quarantine changed")
    return decision, trajectory


def _job_key(job: Mapping[str, Any]) -> tuple[Any, ...]:
    return preflight.execution_key(job)


def _job_map(jobs: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    output = {_job_key(job): dict(job) for job in jobs}
    if len(output) != len(jobs):
        raise FormalRunError("formal job list contains duplicate execution keys")
    return output


def _group_jobs(
    jobs: Sequence[Mapping[str, Any]], execution_kind: str
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        output.setdefault(preflight.shard_id(job), []).append(dict(job))
    expected = 185 if execution_kind == "decision" else 2
    if len(output) != expected:
        raise FormalRunError(
            f"{execution_kind} shard count changed: {len(output)} != {expected}"
        )
    return output


def _new_gate_audit() -> dict[str, Any]:
    audit = primary_runner._new_gate_audit()
    audit.setdefault("terminal_recommendation_status", "not_checked")
    audit.setdefault("terminal_recommendation_index_difference_count", None)
    audit.setdefault("terminal_recommendation_expected_index", None)
    audit.setdefault("terminal_recommendation_saved_index", None)
    return audit


_ACTIVE_GATE_AUDIT: dict[str, Any] | None = None


def _audited_trial_context(*args: Any, **kwargs: Any) -> _BaseContext:
    if _ACTIVE_GATE_AUDIT is None:
        raise FormalRunError("formal trial constructed a Context outside an active audit")
    ctx = _BaseContext(*args, **kwargs)
    primary_runner._audit_context_gate(
        ctx, _ACTIVE_GATE_AUDIT, terminal=False
    )
    return ctx


def _exact_scores(ctx: _BaseContext) -> np.ndarray:
    return np.asarray(ckg_scores_exact(ctx), dtype=float)


_exact_scores.acquisition_name = FORMAL_EXACT_POLICY  # type: ignore[attr-defined]


def _schedule_scores(ctx: _BaseContext) -> np.ndarray:
    """Process-local fixed [cEI, cEI, tMSE] adaptive-cohort cycle."""
    if int(ctx.step) % 3 in (0, 1):
        return np.asarray(ctx.restrict(ctx.cei_scores()), dtype=float)
    return np.asarray(ctx.restrict(ctx.tmse), dtype=float)


_schedule_scores.acquisition_name = FORMAL_SCHEDULE_POLICY  # type: ignore[attr-defined]


def _worker_initialize() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    existing = ACQUISITIONS.get(EXACT_WORKER_POLICY)
    if existing is not None and existing is not _exact_scores:
        raise FormalRunError("formal exact worker name is already registered")
    ACQUISITIONS[EXACT_WORKER_POLICY] = _exact_scores
    existing_schedule = ACQUISITIONS.get(SCHEDULE_WORKER_POLICY)
    if existing_schedule is not None and existing_schedule is not _schedule_scores:
        raise FormalRunError("formal schedule worker name is already registered")
    ACQUISITIONS[SCHEDULE_WORKER_POLICY] = _schedule_scores
    trial_module.Context = _audited_trial_context  # type: ignore[assignment]


def _worker_cleanup() -> None:
    """Restore process-local registrations for single-worker programmatic use."""
    global _ACTIVE_GATE_AUDIT
    if ACQUISITIONS.get(EXACT_WORKER_POLICY) is _exact_scores:
        ACQUISITIONS.pop(EXACT_WORKER_POLICY)
    if ACQUISITIONS.get(SCHEDULE_WORKER_POLICY) is _schedule_scores:
        ACQUISITIONS.pop(SCHEDULE_WORKER_POLICY)
    if trial_module.Context is _audited_trial_context:
        trial_module.Context = _BaseContext  # type: ignore[assignment]
    _ACTIVE_GATE_AUDIT = None


def _trial_policy(job: Mapping[str, Any]) -> str:
    if job["run_class"] == "exact_ckg":
        if job["policy"] != FORMAL_EXACT_POLICY:
            raise FormalRunError("an exact-cKG job carries the wrong formal policy")
        return EXACT_WORKER_POLICY
    if job["run_class"] not in {"current_harness_comparator", "trajectory_replay"}:
        raise FormalRunError(f"unknown formal run class {job['run_class']!r}")
    if job["policy"] == FORMAL_EXACT_POLICY:
        raise FormalRunError("comparator job attempted to use the exact evaluator")
    if job["policy"] == FORMAL_SCHEDULE_POLICY:
        if job["matrix_id"] != "schedule_ratio_2to1_current":
            raise FormalRunError("formal schedule policy is outside its locked matrix")
        return SCHEDULE_WORKER_POLICY
    return str(job["policy"])


def _trial_kwargs(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(job["seed"]),
        "z": int(job["stratum"]),
        "gamma": float(job["gamma"]),
        "sim": str(job["sim"]),
        "mode": str(job["mode"]),
        "noise": str(job["noise"]),
        "traj": bool(job["traj"]),
        "budget": int(job["budget"]),
        "warmup": int(job["warmup"]),
        "r_k": int(job["r_k"]),
        "grid_n": int(job["grid_n"]),
        "empty_gate": str(job["empty_gate"]),
        "kap": float(job["kap"]),
        "protocol_scaffold": str(job["protocol_scaffold"]),
        "region_step": float(job["region_step"]),
        "empty_gate_stop_after": int(job["empty_gate_stop_after"]),
        "exclude_repeats_during_expansion": bool(
            job["exclude_repeats_during_expansion"]
        ),
    }


def _replay_observations(
    job: Mapping[str, Any], result: Mapping[str, Any]
) -> tuple[np.ndarray, list[float], list[float], Mapping[str, Any]]:
    history = np.asarray(result.get("allocation_history"), dtype=float)
    n_enrolled = int(result.get("n_enrolled", -1))
    if history.shape != (n_enrolled, 2) or not 1 <= n_enrolled <= int(job["budget"]):
        raise FormalRunError("result has an invalid allocation history")
    surface = resolve_surface(str(job["sim"]), int(job["stratum"]))
    sf = float(job["kap"]) * float(surface["sf"])
    sg = float(job["kap"]) * float(surface["sg"])
    rng = np.random.default_rng(int(job["seed"]))
    efficacy: list[float] = []
    toxicity: list[float] = []
    scaffold = str(job["protocol_scaffold"])
    warmup = int(job["warmup"])
    r_k = int(job["r_k"])
    if scaffold == "lhs_fixed":
        cut = np.linspace(0.0, 1.0, warmup + 1)
    elif scaffold != "start_low_expansion":
        raise FormalRunError(f"unknown protocol scaffold {scaffold!r}")
    for index, (d1, d2) in enumerate(history):
        if scaffold == "lhs_fixed" and index < warmup:
            expected = np.asarray(
                [cut[index] + rng.uniform() * (cut[1] - cut[0]), rng.uniform()]
            )
            if not np.array_equal(expected, np.asarray([d1, d2])):
                raise FormalRunError("terminal replay cannot reproduce LHS initialization")
        elif scaffold == "start_low_expansion" and index < r_k:
            if not np.array_equal(np.asarray([d1, d2]), np.zeros(2)):
                raise FormalRunError("terminal replay cannot reproduce start-low initialization")
        efficacy.append(float(surface["eff"](d1, d2) + rng.normal(0.0, sf)))
        toxicity.append(float(surface["tox"](d1, d2) + rng.normal(0.0, sg)))
    return history, efficacy, toxicity, surface


def _terminal_context(job: Mapping[str, Any], result: Mapping[str, Any]) -> _BaseContext:
    if job["noise"] != "fixed":
        raise FormalRunError("locked formal manifest unexpectedly requested learned noise")
    history, efficacy, toxicity, surface = _replay_observations(job, result)
    sf = float(job["kap"]) * float(surface["sf"])
    sg = float(job["kap"]) * float(surface["sg"])
    observed = torch.tensor(history)
    me, le = fit_gp(observed, torch.tensor(efficacy), fixed_noise=sf**2)
    mt, lt = fit_gp(observed, torch.tensor(toxicity), fixed_noise=sg**2)
    return _BaseContext(
        Xset=torch.tensor(trial_module.grid(int(job["grid_n"]))),
        me=me,
        le=le,
        mt=mt,
        lt=lt,
        g_dagger=float(surface["gd"]),
        gamma=float(job["gamma"]),
        gate_mode=str(job["mode"]),
        r_k=int(job["r_k"]),
        empty_gate=str(job["empty_gate"]),
    )


def _audit_terminal_recommendation(
    terminal: _BaseContext,
    result: Mapping[str, Any],
    audit: dict[str, Any],
) -> None:
    """Independently reproduce and hard-check the stable-z terminal rule."""
    if audit.get("terminal_recommendation_status") not in (None, "not_checked"):
        raise FormalRunError("terminal recommendation audit was already completed")
    recommendation_made = result.get("recommendation_made")
    if recommendation_made is False:
        if result.get("stop_reason") != "NO_FEASIBLE_DOSE":
            raise FormalRunError(
                "non-recommending result lacks the prespecified stopping reason"
            )
        if result.get("rec_d1") is not None or result.get("rec_d2") is not None:
            raise FormalRunError("stopped result unexpectedly carries a recommendation")
        audit["terminal_recommendation_status"] = "not_applicable_stopped"
        audit["terminal_recommendation_index_difference_count"] = None
        audit["terminal_recommendation_expected_index"] = None
        audit["terminal_recommendation_saved_index"] = None
        return
    if recommendation_made is not True:
        raise FormalRunError("recommendation_made must be boolean before terminal audit")

    grid = np.asarray(terminal.Xset.detach().cpu().numpy(), dtype=float)
    if grid.ndim != 2 or grid.shape[1] != 2 or not np.isfinite(grid).all():
        raise FormalRunError("terminal recommendation audit has a malformed dose grid")
    mu_f_tensor, _ = post_latent(terminal.me, terminal.Xset)
    mu_f = np.asarray(mu_f_tensor.detach().cpu().numpy(), dtype=float).reshape(-1)
    z = (
        float(terminal.g_dagger) - np.asarray(terminal.mu_g, dtype=float).reshape(-1)
    ) / np.asarray(terminal.sd_g, dtype=float).reshape(-1)
    if (
        mu_f.shape != (grid.shape[0],)
        or z.shape != (grid.shape[0],)
        or not np.isfinite(mu_f).all()
        or not np.isfinite(z).all()
    ):
        raise FormalRunError("terminal recommendation audit has malformed posterior values")
    stable_gate = z > float(norm.ppf(float(terminal.gamma)))
    expected_index = int(
        np.where(stable_gate, mu_f, -np.inf).argmax()
        if stable_gate.any()
        else z.argmax()
    )

    saved_coordinates = np.asarray(
        [result.get("rec_d1"), result.get("rec_d2")], dtype=float
    )
    if not np.isfinite(saved_coordinates).all():
        raise FormalRunError("saved terminal recommendation is not finite")
    matches = np.flatnonzero(np.all(grid == saved_coordinates[None, :], axis=1))
    if len(matches) != 1:
        raise FormalRunError("saved terminal recommendation is not exactly on the dose grid")
    saved_index = int(matches[0])
    difference = int(expected_index != saved_index)
    audit["terminal_recommendation_status"] = "checked"
    audit["terminal_recommendation_index_difference_count"] = difference
    audit["terminal_recommendation_expected_index"] = expected_index
    audit["terminal_recommendation_saved_index"] = saved_index
    if difference:
        raise FormalRunError(
            "saved terminal recommendation differs from the independent stable-z rule"
        )


def _expected_trial_context_count(
    job: Mapping[str, Any], result: Mapping[str, Any]
) -> int:
    init_size = initialization_size(
        str(job["protocol_scaffold"]), int(job["warmup"]), int(job["r_k"])
    )
    n_enrolled = int(result["n_enrolled"])
    if n_enrolled < init_size or (n_enrolled - init_size) % int(job["r_k"]):
        raise FormalRunError("result enrollment is incompatible with the cohort scaffold")
    allocated = (n_enrolled - init_size) // int(job["r_k"])
    return allocated + int(result.get("stop_reason") is not None)


def _validate_result_design(
    result: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    expected = {
        "policy": job["policy"],
        "seed": int(job["seed"]),
        "sim": job["sim"],
        "stratum": int(job["stratum"]),
        "gamma": float(job["gamma"]),
        "mode": job["mode"],
        "budget": int(job["budget"]),
        "warmup": int(job["warmup"]),
        "r_k": int(job["r_k"]),
        "grid_n": int(job["grid_n"]),
        "noise": job["noise"],
        "kap": float(job["kap"]),
        "empty_gate": job["empty_gate"],
        "protocol_scaffold": job["protocol_scaffold"],
        "region_step": float(job["region_step"]),
        "empty_gate_stop_after": int(job["empty_gate_stop_after"]),
        "exclude_repeats_during_expansion": bool(
            job["exclude_repeats_during_expansion"]
        ),
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise FormalRunError(
                f"formal result design mismatch for {field}: "
                f"{result.get(field)!r} != {value!r}"
            )
    if job["policy"] == FORMAL_SCHEDULE_POLICY:
        for field, value in SCHEDULE_POST_HOC_FIELDS.items():
            if result.get(field) != value:
                raise FormalRunError(
                    f"formal schedule result lacks locked post-hoc field {field}"
                )
    trajectory = result.get("traj")
    if (trajectory is not None) != bool(job["traj"]):
        raise FormalRunError("formal result trajectory flag differs from its job")
    if result.get("protocol_scaffold") == "lhs_fixed":
        if (
            result.get("stop_reason") is not None
            or result.get("recommendation_made") is not True
            or int(result.get("n_enrolled", -1)) != int(job["budget"])
        ):
            raise FormalRunError("lhs-fixed formal result stopped or failed to recommend")
    n_enrolled = int(result.get("n_enrolled", -1))
    history = result.get("allocation_history")
    if not isinstance(history, list) or len(history) != n_enrolled:
        raise FormalRunError("formal result allocation history is incomplete")
    if not isinstance(result.get("toxic"), (int, np.integer)) or not (
        0 <= int(result["toxic"]) <= n_enrolled
    ):
        raise FormalRunError("formal result toxic count is malformed")
    grid_size = int(job["grid_n"]) ** 2
    for field in ("n_gate_pass", "n_gate_pass_safe"):
        value = result.get(field)
        if not isinstance(value, (int, np.integer)) or not 0 <= int(value) <= grid_size:
            raise FormalRunError(f"formal result {field} is malformed")
    recommendation_made = result.get("recommendation_made")
    if not isinstance(recommendation_made, (bool, np.bool_)):
        raise FormalRunError("formal result recommendation_made must be boolean")
    if recommendation_made:
        for field in (
            "rec_d1", "rec_d2", "rec_true_eff", "rec_true_tox",
            "dose_units", "rpsel",
        ):
            value = result.get(field)
            if not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(
                float(value)
            ):
                raise FormalRunError(f"formal result {field} must be finite")
        if result.get("rec_unsafe") not in (0, 1, False, True):
            raise FormalRunError("formal result rec_unsafe must be binary")
    else:
        for field in (
            "rec_d1",
            "rec_d2",
            "rec_true_eff",
            "rec_true_tox",
            "dose_units",
            "rpsel",
            "rec_unsafe",
        ):
            if result.get(field) is not None:
                raise FormalRunError(
                    f"formal result {field} must be null without a recommendation"
                )
    if trajectory is not None:
        if not isinstance(trajectory, dict):
            raise FormalRunError("formal trajectory payload is not an object")
        expected_states = _expected_trial_context_count(job, result)
        expected_n = list(
            range(
                initialization_size(
                    str(job["protocol_scaffold"]),
                    int(job["warmup"]),
                    int(job["r_k"]),
                ),
                initialization_size(
                    str(job["protocol_scaffold"]),
                    int(job["warmup"]),
                    int(job["r_k"]),
                )
                + expected_states * int(job["r_k"]),
                int(job["r_k"]),
            )
        )
        if trajectory.get("n") != expected_n:
            raise FormalRunError("formal trajectory cohort labels are incomplete")
        for field in ("du", "rpsel", "toxic"):
            values = trajectory.get(field)
            if not isinstance(values, list) or len(values) != expected_states:
                raise FormalRunError(f"formal trajectory field {field} is incomplete")


def _validate_gate_audit(
    audit: Mapping[str, Any], job: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    expected_trial = _expected_trial_context_count(job, result)
    if (
        audit.get("checked_state_count") != expected_trial + 1
        or audit.get("acquisition_state_count") != expected_trial
        or audit.get("terminal_state_checked") is not True
    ):
        raise FormalRunError(
            "formal row has incomplete gate audit: "
            f"expected trial/total={expected_trial}/{expected_trial + 1}, got {audit}"
        )
    zero_fields = (
        "gate_pass_set_difference_count",
        "eligible_gate_pass_set_difference_count",
        "operational_full_fallback_index_difference_count",
        "operational_eligible_fallback_index_difference_count",
    )
    if any(audit.get(field) != 0 for field in zero_fields):
        raise FormalRunError("formal row has a gate-numerics difference")
    for field in (
        "inactive_full_fallback_rounding_difference_count",
        "inactive_eligible_fallback_rounding_difference_count",
        "operational_full_fallback_rounding_difference_count",
        "operational_eligible_fallback_rounding_difference_count",
        "cdf_gate_pass_set_rounding_difference_count",
        "cdf_eligible_gate_pass_set_rounding_difference_count",
        "cdf_saturated_zero_total",
        "cdf_saturated_one_total",
    ):
        value = audit.get(field)
        if not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise FormalRunError(f"formal row has malformed gate diagnostic {field}")
    for field in ("minimum_abs_z_minus_q", "maximum_abs_standardized_feasibility"):
        value = audit.get(field)
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise FormalRunError(f"formal row has malformed gate diagnostic {field}")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise FormalRunError(f"formal row has nonfinite gate diagnostic {field}")
    if result.get("recommendation_made") is True:
        expected = audit.get("terminal_recommendation_expected_index")
        saved = audit.get("terminal_recommendation_saved_index")
        if (
            audit.get("terminal_recommendation_status") != "checked"
            or audit.get("terminal_recommendation_index_difference_count") != 0
            or not isinstance(expected, (int, np.integer))
            or isinstance(expected, (bool, np.bool_))
            or not isinstance(saved, (int, np.integer))
            or isinstance(saved, (bool, np.bool_))
            or int(expected) != int(saved)
        ):
            raise FormalRunError(
                "formal row lacks a matching terminal recommendation audit"
            )
    else:
        if (
            audit.get("terminal_recommendation_status")
            != "not_applicable_stopped"
            or audit.get("terminal_recommendation_index_difference_count") is not None
            or audit.get("terminal_recommendation_expected_index") is not None
            or audit.get("terminal_recommendation_saved_index") is not None
        ):
            raise FormalRunError(
                "stopped formal row lacks a not-applicable terminal recommendation audit"
            )


def _validate_execution_row(
    row: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    if row.get("formal_execution_key") != list(_job_key(job)):
        raise FormalRunError("formal row execution key differs from its job")
    expected_evaluator = (
        EXACT_EVALUATOR_ID
        if job["run_class"] == "exact_ckg"
        else REGISTERED_EVALUATOR_ID
    )
    expected_identity = {
        "matrix_id": job["matrix_id"],
        "run_class": job["run_class"],
        "policy": job["policy"],
        "seed": int(job["seed"]),
        "stratum": int(job["stratum"]),
        "harness_id": HARNESS_ID,
        "evaluator_id": expected_evaluator,
        "historical_wrapper_used": False,
    }
    for field, expected in expected_identity.items():
        if row.get(field) != expected:
            raise FormalRunError(f"formal row has invalid {field} provenance")
    elapsed = row.get("elapsed_seconds_not_for_timing_inference")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)):
        raise FormalRunError("formal row elapsed time is malformed")
    if float(elapsed) < 0.0:
        raise FormalRunError("formal row elapsed time is negative")
    result = row.get("result")
    audit = row.get("gate_numerics_audit")
    if not isinstance(result, dict) or not isinstance(audit, dict):
        raise FormalRunError("formal row lacks result or gate audit")
    _validate_result_design(result, job)
    _validate_gate_audit(audit, job, result)


def _run_job(job: Mapping[str, Any]) -> dict[str, Any]:
    global _ACTIVE_GATE_AUDIT
    if _ACTIVE_GATE_AUDIT is not None:
        raise FormalRunError("worker gate-audit state leaked from a previous job")
    audit = _new_gate_audit()
    _ACTIVE_GATE_AUDIT = audit
    started = time.perf_counter()
    try:
        result = trial_module.run_trial(_trial_policy(job), **_trial_kwargs(job))
        result = dict(result)
        result["policy"] = str(job["policy"])
        if job["policy"] == FORMAL_SCHEDULE_POLICY:
            result.update(SCHEDULE_POST_HOC_FIELDS)
        terminal = _terminal_context(job, result)
        primary_runner._audit_context_gate(terminal, audit, terminal=True)
        _audit_terminal_recommendation(terminal, result, audit)
    finally:
        _ACTIVE_GATE_AUDIT = None
    if int(result.get("n_gate_pass", -1)) != int(np.count_nonzero(terminal.gate_safe)):
        raise FormalRunError("terminal replay disagrees with saved gate-pass count")
    row = {
        "formal_execution_key": list(_job_key(job)),
        "matrix_id": job["matrix_id"],
        "run_class": job["run_class"],
        "policy": job["policy"],
        "seed": int(job["seed"]),
        "stratum": int(job["stratum"]),
        "harness_id": HARNESS_ID,
        "evaluator_id": job["evaluator_id"],
        "historical_wrapper_used": False,
        "gate_numerics_audit": audit,
        "elapsed_seconds_not_for_timing_inference": time.perf_counter() - started,
        "result": result,
    }
    _validate_execution_row(row, job)
    return row


def _primary_artifact_hashes(primary_raw: Path) -> dict[str, str]:
    metadata = Path(str(primary_raw) + ".metadata.json")
    if not primary_raw.is_file() or not metadata.is_file():
        raise FormalRunError("authenticated primary raw artifact or sidecar is absent")
    return {
        "primary_raw_sha256": _sha256_file(primary_raw),
        "primary_raw_metadata_sha256": _sha256_file(metadata),
    }


def _load_authenticated_primary_rows(
    *,
    primary_raw: Path,
    frozen_records: Path,
    exact_panel: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen_osa = Path(frozen_records) / "osa_main.json.zst"
    references, frozen_provenance = primary_analyzer._load_frozen_osa(frozen_osa)
    del references
    exact_sha256 = _sha256_file(exact_panel)
    lookup, authenticated = primary_analyzer._load_primary(
        primary_raw,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        exact_validation_sha256=exact_sha256,
        exact_validation_path=exact_panel,
        frozen_osa_provenance=frozen_provenance,
    )
    del lookup
    payload = json.loads(primary_raw.read_text(encoding="utf-8"))
    raw_by_key = {
        (int(row["seed"]), int(row["stratum"])): row for row in payload["rows"]
    }
    jobs = preflight.primary_jobs(manifest)
    rows: list[dict[str, Any]] = []
    for job in jobs:
        source = raw_by_key[(int(job["seed"]), int(job["stratum"]))]
        row = {
            "formal_execution_key": list(_job_key(job)),
            "matrix_id": job["matrix_id"],
            "run_class": job["run_class"],
            "policy": job["policy"],
            "seed": int(job["seed"]),
            "stratum": int(job["stratum"]),
            "harness_id": source["harness_id"],
            "evaluator_id": source["evaluator_id"],
            "historical_wrapper_used": source["historical_wrapper_used"],
            "gate_numerics_audit": source["gate_numerics_audit"],
            "elapsed_seconds_not_for_timing_inference": source[
                "elapsed_seconds_not_for_timing_inference"
            ],
            "result": source["result"],
        }
        _validate_execution_row(row, job)
        rows.append(row)
    if len(rows) != PRIMARY_CELL_COUNT:
        raise FormalRunError("authenticated primary import is not exactly 400 rows")
    return rows, {
        **_primary_artifact_hashes(primary_raw),
        "primary_validation": authenticated,
    }


def _checkpoint_path(output_dir: Path, execution_kind: str, shard: str) -> Path:
    safe = "".join(character if character.isalnum() or character in "-_." else "-" for character in shard)
    if not safe or safe != shard:
        raise FormalRunError(f"unsafe formal shard id {shard!r}")
    return output_dir / "checkpoints" / execution_kind / f"{safe}.checkpoint.json"


def _write_checkpoint(
    path: Path,
    *,
    execution_kind: str,
    shard: str,
    rows: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
    source_sha256: Mapping[str, str],
    exact_validation_sha256: str,
    exact_validation_metadata_sha256: str,
    primary_hashes: Mapping[str, str],
    runtime_identity: Mapping[str, str],
) -> None:
    payload = {
        "schema_version": 1,
        "status": "PARTIAL_NOT_FOR_ANALYSIS",
        "execution_kind": execution_kind,
        "manifest_sha256": manifest_sha256,
        "source_sha256": dict(source_sha256),
        "exact_validation_sha256": exact_validation_sha256,
        "exact_validation_metadata_sha256": exact_validation_metadata_sha256,
        **dict(primary_hashes),
        "runtime_identity": dict(runtime_identity),
        "shard_id": shard,
        "harness_id": HARNESS_ID,
        "completed_row_count": len(rows),
        "rows": [dict(row) for row in rows],
    }
    _atomic_json_write(path, payload)
    # The sidecar is replaced last and is the checkpoint commit marker.  A crash
    # between the two writes leaves a detectable hash mismatch, never a silently
    # accepted partial generation.
    _atomic_json_write(
        _metadata_path(path),
        {
            "schema_version": 1,
            "artifact": path.name,
            "artifact_sha256": _sha256_file(path),
            "artifact_bytes": path.stat().st_size,
            "status": payload["status"],
            "execution_kind": execution_kind,
            "shard_id": shard,
            "row_count": len(rows),
            "manifest_sha256": manifest_sha256,
            "source_sha256": dict(source_sha256),
            "exact_validation_sha256": exact_validation_sha256,
            "exact_validation_metadata_sha256": exact_validation_metadata_sha256,
            **dict(primary_hashes),
            "runtime_identity": dict(runtime_identity),
            "committed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _load_checkpoint(
    path: Path,
    *,
    execution_kind: str,
    shard: str,
    jobs: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
    source_sha256: Mapping[str, str],
    exact_validation_sha256: str,
    exact_validation_metadata_sha256: str,
    primary_hashes: Mapping[str, str],
    runtime_identity: Mapping[str, str],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    expected_evaluator = {
        _job_key(job): str(job["evaluator_id"]) for job in jobs
    }
    rows = preflight.validate_checkpoint(
        path,
        manifest_sha256=manifest_sha256,
        source_sha256=source_sha256,
        exact_validation_sha256=exact_validation_sha256,
        shard=shard,
        expected_evaluator_by_key=expected_evaluator,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    sidecar_path = _metadata_path(path)
    if not sidecar_path.is_file():
        raise FormalRunError(f"checkpoint commit marker is absent: {path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    required = {
        "execution_kind": execution_kind,
        "exact_validation_metadata_sha256": exact_validation_metadata_sha256,
        **dict(primary_hashes),
        "runtime_identity": dict(runtime_identity),
        "completed_row_count": len(rows),
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise FormalRunError(f"checkpoint {field} binding mismatch: {path}")
    required_sidecar = {
        "schema_version": 1,
        "artifact": path.name,
        "artifact_sha256": _sha256_file(path),
        "artifact_bytes": path.stat().st_size,
        "status": "PARTIAL_NOT_FOR_ANALYSIS",
        "execution_kind": execution_kind,
        "shard_id": shard,
        "row_count": len(rows),
        "manifest_sha256": manifest_sha256,
        "source_sha256": dict(source_sha256),
        "exact_validation_sha256": exact_validation_sha256,
        "exact_validation_metadata_sha256": exact_validation_metadata_sha256,
        **dict(primary_hashes),
        "runtime_identity": dict(runtime_identity),
    }
    for field, expected in required_sidecar.items():
        if sidecar.get(field) != expected:
            raise FormalRunError(f"checkpoint commit marker mismatch: {path} {field}")
    by_key = _job_map(jobs)
    for row in rows:
        _validate_execution_row(row, by_key[tuple(row["formal_execution_key"])])
    return rows


def _run_jobs_bounded(
    jobs: Sequence[Mapping[str, Any]], *, workers: int, on_result: Callable[[dict[str, Any]], None]
) -> None:
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_initialize,
    ) as pool:
        iterator = iter(jobs)
        pending: dict[concurrent.futures.Future, tuple[Any, ...]] = {}

        def fill() -> None:
            while len(pending) < 2 * workers:
                try:
                    job = next(iterator)
                except StopIteration:
                    return
                pending[pool.submit(_run_job, dict(job))] = _job_key(job)

        fill()
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                key = pending.pop(future)
                try:
                    row = future.result()
                except BaseException:
                    for remaining in pending:
                        remaining.cancel()
                    raise FormalRunError(f"formal task failed: {key}") from future.exception()
                on_result(row)
            fill()


def _allow_process_pool_when_sem_limit_query_is_blocked() -> bool:
    try:
        os.sysconf("SC_SEM_NSEMS_MAX")
        return False
    except PermissionError:
        original_sysconf = os.sysconf

        def guarded_sysconf(name: str | int) -> int:
            if name == "SC_SEM_NSEMS_MAX":
                return 256
            return int(original_sysconf(name))

        os.sysconf = guarded_sysconf  # type: ignore[assignment]
        return True


def _execution_provenance(
    *,
    manifest_sha256: str,
    exact_validation_sha256: str,
    exact_validation_metadata_sha256: str,
    primary_hashes: Mapping[str, str],
    runtime_identity: Mapping[str, str],
    include_primary: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "manifest_sha256": manifest_sha256,
        "exact_validation_panel_sha256": exact_validation_sha256,
        "exact_validation_metadata_sha256": exact_validation_metadata_sha256,
        "historical_wrappers_used": False,
        "production_ckg_default_changed": False,
        "authorization": AUTHORIZATION_TOKEN,
        "runtime_identity": dict(runtime_identity),
    }
    if include_primary:
        output.update(dict(primary_hashes))
    return output


def _metadata_path(path: Path) -> Path:
    return Path(str(path) + ".metadata.json")


def _finalize_kind(
    *,
    execution_kind: str,
    jobs: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    manifest_sha256: str,
    source_sha256: Mapping[str, str],
    exact_validation_sha256: str,
    exact_validation_metadata_sha256: str,
    primary_hashes: Mapping[str, str],
    runtime_identity: Mapping[str, str],
    environment: Mapping[str, Any],
) -> Path:
    rows: list[dict[str, Any]] = []
    checkpoint_sha256: dict[str, str] = {}
    checkpoint_metadata_sha256: dict[str, str] = {}
    for shard, shard_jobs in groups.items():
        path = _checkpoint_path(output_dir, execution_kind, shard)
        shard_rows = _load_checkpoint(
            path,
            execution_kind=execution_kind,
            shard=shard,
            jobs=shard_jobs,
            manifest_sha256=manifest_sha256,
            source_sha256=source_sha256,
            exact_validation_sha256=exact_validation_sha256,
            exact_validation_metadata_sha256=exact_validation_metadata_sha256,
            primary_hashes=primary_hashes,
            runtime_identity=runtime_identity,
        )
        if len(shard_rows) != len(shard_jobs):
            raise FormalRunError(
                f"cannot finalize incomplete {execution_kind} shard {shard}: "
                f"{len(shard_rows)}/{len(shard_jobs)}"
            )
        rows.extend(shard_rows)
        checkpoint_sha256[str(path.relative_to(output_dir))] = _sha256_file(path)
        checkpoint_metadata_sha256[
            str(_metadata_path(path).relative_to(output_dir))
        ] = _sha256_file(_metadata_path(path))
    by_key = {tuple(row["formal_execution_key"]): row for row in rows}
    if len(by_key) != len(rows) or set(by_key) != set(_job_map(jobs)):
        raise FormalRunError(f"{execution_kind} master coverage is incomplete or duplicated")
    canonical_rows = [by_key[_job_key(job)] for job in jobs]
    expected_rows = DECISION_COUNT if execution_kind == "decision" else TRAJECTORY_COUNT
    status = DECISION_STATUS if execution_kind == "decision" else TRAJECTORY_STATUS
    basename = (
        "formal_decision_master.json"
        if execution_kind == "decision"
        else "formal_trajectory_replays.json"
    )
    if len(canonical_rows) != expected_rows:
        raise FormalRunError(f"{execution_kind} master has the wrong row count")
    payload = {
        "schema_version": 1,
        "status": status,
        "artifact_class": ARTIFACT_CLASS,
        "design": {
            "execution_kind": execution_kind,
            "expected_rows": expected_rows,
            "canonical_sort_key": list(CANONICAL_SORT_KEY),
        },
        "provenance": _execution_provenance(
            manifest_sha256=manifest_sha256,
            exact_validation_sha256=exact_validation_sha256,
            exact_validation_metadata_sha256=exact_validation_metadata_sha256,
            primary_hashes=primary_hashes,
            runtime_identity=runtime_identity,
            include_primary=execution_kind == "decision",
        ),
        "source_sha256": dict(source_sha256),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_metadata_sha256": checkpoint_metadata_sha256,
        "environment": dict(environment),
        "rows": canonical_rows,
    }
    path = output_dir / basename
    _atomic_json_write(path, payload)
    sidecar = {
        "schema_version": 1,
        "artifact": path.name,
        "artifact_sha256": _sha256_file(path),
        "artifact_bytes": path.stat().st_size,
        "status": status,
        "row_count": expected_rows,
        "manifest_sha256": manifest_sha256,
        "exact_validation_panel_sha256": exact_validation_sha256,
        "exact_validation_metadata_sha256": exact_validation_metadata_sha256,
        "source_sha256": dict(source_sha256),
        "runtime_identity": dict(runtime_identity),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "publication_scope": "formal staging only; frozen archive unchanged",
    }
    if execution_kind == "decision":
        sidecar.update(dict(primary_hashes))
    _atomic_json_write(_metadata_path(path), sidecar)
    return path


def _validate_committed_artifact(path: Path, status: str, rows: int) -> dict[str, Any]:
    sidecar_path = _metadata_path(path)
    if not path.is_file() or not sidecar_path.is_file():
        raise FormalRunError(f"committed formal artifact is absent: {path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = {
        "artifact": path.name,
        "artifact_sha256": _sha256_file(path),
        "artifact_bytes": path.stat().st_size,
        "status": status,
        "row_count": rows,
    }
    for field, value in expected.items():
        if sidecar.get(field) != value:
            raise FormalRunError(f"formal artifact commit marker mismatch: {path} {field}")
    return sidecar


def _write_bundle_manifest(
    *,
    output_dir: Path,
    manifest_sha256: str,
    exact_validation_sha256: str,
    exact_validation_metadata_sha256: str,
    primary_hashes: Mapping[str, str],
    source_sha256: Mapping[str, str],
    runtime_identity: Mapping[str, str],
) -> Path:
    decision = output_dir / "formal_decision_master.json"
    trajectory = output_dir / "formal_trajectory_replays.json"
    decision_sidecar = _validate_committed_artifact(
        decision, DECISION_STATUS, DECISION_COUNT
    )
    trajectory_sidecar = _validate_committed_artifact(
        trajectory, TRAJECTORY_STATUS, TRAJECTORY_COUNT
    )
    payload = {
        "schema_version": 1,
        "status": BUNDLE_STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "counts": {
            "decision_identities": DECISION_IDENTITY_COUNT,
            "nontrajectory_decision_executions": DECISION_COUNT,
            "trajectory_replays": TRAJECTORY_COUNT,
            "formal_executions": FORMAL_EXECUTION_COUNT,
            "authenticated_primary_reuse": PRIMARY_CELL_COUNT,
            "new_executions_after_primary": NEW_EXECUTION_COUNT_AFTER_PRIMARY,
        },
        "artifacts": {
            "decision": {
                "basename": decision.name,
                "sha256": decision_sidecar["artifact_sha256"],
                "metadata_sha256": _sha256_file(_metadata_path(decision)),
            },
            "trajectory_replay": {
                "basename": trajectory.name,
                "sha256": trajectory_sidecar["artifact_sha256"],
                "metadata_sha256": _sha256_file(_metadata_path(trajectory)),
            },
        },
        "provenance": _execution_provenance(
            manifest_sha256=manifest_sha256,
            exact_validation_sha256=exact_validation_sha256,
            exact_validation_metadata_sha256=exact_validation_metadata_sha256,
            primary_hashes=primary_hashes,
            runtime_identity=runtime_identity,
            include_primary=True,
        ),
        "source_sha256": dict(source_sha256),
        "frozen_archive_changed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = output_dir / "formal_execution_manifest.json"
    _atomic_json_write(path, payload)
    _atomic_json_write(
        _metadata_path(path),
        {
            "schema_version": 1,
            "artifact": path.name,
            "artifact_sha256": _sha256_file(path),
            "artifact_bytes": path.stat().st_size,
            "status": BUNDLE_STATUS,
            "manifest_sha256": manifest_sha256,
            "source_sha256": dict(source_sha256),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return path


def _select_groups(
    decision_groups: Mapping[str, list[dict[str, Any]]],
    trajectory_groups: Mapping[str, list[dict[str, Any]]],
    *,
    kind: str,
    requested_shards: Sequence[str],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    available: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    if kind in {"all", "decision"}:
        available.update({key: ("decision", value) for key, value in decision_groups.items()})
    if kind in {"all", "trajectory_replay"}:
        for key, value in trajectory_groups.items():
            if key in available:
                raise FormalRunError(f"duplicate cross-kind shard id {key}")
            available[key] = ("trajectory_replay", value)
    if requested_shards:
        unknown = set(requested_shards) - set(available)
        if unknown:
            raise FormalRunError(f"requested unknown formal shards: {sorted(unknown)}")
        keys = list(dict.fromkeys(requested_shards))
    else:
        keys = list(available)
    return [(available[key][0], key, available[key][1]) for key in keys]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=preflight.MANIFEST_PATH)
    parser.add_argument("--exact-validation-panel", type=Path, required=True)
    parser.add_argument("--primary-raw", type=Path, required=True)
    parser.add_argument("--frozen-records", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_STABLE_Z_STAGING
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--kind",
        choices=("all", "decision", "trajectory_replay"),
        default="all",
    )
    parser.add_argument("--shard-id", action="append", default=[])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--authorization-token")
    args = parser.parse_args(argv)

    if not 1 <= int(args.workers) <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    mutating = bool(args.execute or args.finalize_only)
    if not _authorization_is_valid(mutate=mutating, token=args.authorization_token):
        parser.error("execution/finalization requires --authorization-token FULL_FORMAL_RUN_GO")
    if args.finalize_only and args.shard_id:
        parser.error("--finalize-only does not accept --shard-id")

    manifest_path = args.manifest.resolve()
    exact_panel = args.exact_validation_panel.resolve()
    primary_raw = args.primary_raw.resolve()
    frozen_records = args.frozen_records.resolve()
    manifest = preflight.load_manifest(manifest_path)
    decision, trajectory = _all_jobs(manifest)
    decision_groups = _group_jobs(decision, "decision")
    trajectory_groups = _group_jobs(trajectory, "trajectory_replay")
    selected = _select_groups(
        decision_groups,
        trajectory_groups,
        kind=str(args.kind),
        requested_shards=args.shard_id,
    )
    exact_summary = preflight.audit_exact_validation_artifact(exact_panel)
    runtime_identity = _require_validation_runtime_match(exact_panel)
    manifest_sha256 = _sha256_file(manifest_path)
    exact_sha256 = _sha256_file(exact_panel)
    exact_metadata_sha256 = _sha256_file(_metadata_path(exact_panel))
    source_sha256 = _source_hashes(manifest, manifest_path)
    primary_rows, primary_provenance = _load_authenticated_primary_rows(
        primary_raw=primary_raw,
        frozen_records=frozen_records,
        exact_panel=exact_panel,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    primary_hashes = {
        key: str(primary_provenance[key])
        for key in ("primary_raw_sha256", "primary_raw_metadata_sha256")
    }
    launch_fingerprint = _sha256_bytes(
        _canonical_json_bytes(
            {
                "manifest_sha256": manifest_sha256,
                "exact_validation_panel_sha256": exact_sha256,
                "exact_validation_metadata_sha256": exact_metadata_sha256,
                **primary_hashes,
                "source_sha256": source_sha256,
                "runtime_identity": runtime_identity,
            }
        )
    )
    print(
        "[preflight] decision_identities=44,200; decision_executions=43,880; "
        "trajectory=320; imported_primary=400; "
        f"future_new=43,800; exact={exact_summary}; fingerprint={launch_fingerprint}",
        flush=True,
    )
    print(
        f"[preflight] selected_shards={len(selected)}; authorization="
        f"{'EXERCISED' if mutating else 'ABSENT'}",
        flush=True,
    )
    if not mutating:
        print("[preflight] no checkpoint, trial, or staging artifact written", flush=True)
        return 0

    output_dir = _assert_not_legacy_staging(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_shard = preflight.shard_id(preflight.primary_jobs(manifest)[0])
    primary_jobs = decision_groups[primary_shard]
    if len(primary_jobs) != PRIMARY_CELL_COUNT:
        raise FormalRunError("primary import no longer maps to one 400-cell shard")
    primary_checkpoint = _checkpoint_path(output_dir, "decision", primary_shard)
    existing_primary = _load_checkpoint(
        primary_checkpoint,
        execution_kind="decision",
        shard=primary_shard,
        jobs=primary_jobs,
        manifest_sha256=manifest_sha256,
        source_sha256=source_sha256,
        exact_validation_sha256=exact_sha256,
        exact_validation_metadata_sha256=exact_metadata_sha256,
        primary_hashes=primary_hashes,
        runtime_identity=runtime_identity,
    )
    if existing_primary and existing_primary != primary_rows:
        raise FormalRunError("primary checkpoint differs from authenticated primary raw rows")
    if not existing_primary:
        _write_checkpoint(
            primary_checkpoint,
            execution_kind="decision",
            shard=primary_shard,
            rows=primary_rows,
            manifest_sha256=manifest_sha256,
            source_sha256=source_sha256,
            exact_validation_sha256=exact_sha256,
            exact_validation_metadata_sha256=exact_metadata_sha256,
            primary_hashes=primary_hashes,
            runtime_identity=runtime_identity,
        )

    if args.execute:
        state: dict[tuple[str, str], dict[str, Any]] = {}
        pending_jobs: list[dict[str, Any]] = []
        for execution_kind, shard, shard_jobs in selected:
            path = _checkpoint_path(output_dir, execution_kind, shard)
            rows = _load_checkpoint(
                path,
                execution_kind=execution_kind,
                shard=shard,
                jobs=shard_jobs,
                manifest_sha256=manifest_sha256,
                source_sha256=source_sha256,
                exact_validation_sha256=exact_sha256,
                exact_validation_metadata_sha256=exact_metadata_sha256,
                primary_hashes=primary_hashes,
                runtime_identity=runtime_identity,
            )
            complete = {tuple(row["formal_execution_key"]) for row in rows}
            remaining = [job for job in shard_jobs if _job_key(job) not in complete]
            state[(execution_kind, shard)] = {
                "path": path,
                "jobs": shard_jobs,
                "rows": rows,
                "new_since_checkpoint": 0,
            }
            pending_jobs.extend(remaining)
            print(
                f"[resume] {execution_kind}/{shard}: {len(rows)}/{len(shard_jobs)}; "
                f"remaining={len(remaining)}",
                flush=True,
            )

        job_to_shard = {
            _job_key(job): (execution_kind, shard)
            for execution_kind, shard, shard_jobs in selected
            for job in shard_jobs
        }

        def checkpoint_bucket(bucket: dict[str, Any], execution_kind: str, shard: str) -> None:
            _write_checkpoint(
                bucket["path"],
                execution_kind=execution_kind,
                shard=shard,
                rows=bucket["rows"],
                manifest_sha256=manifest_sha256,
                source_sha256=source_sha256,
                exact_validation_sha256=exact_sha256,
                exact_validation_metadata_sha256=exact_metadata_sha256,
                primary_hashes=primary_hashes,
                runtime_identity=runtime_identity,
            )
            bucket["new_since_checkpoint"] = 0

        def accept(row: dict[str, Any]) -> None:
            key = tuple(row["formal_execution_key"])
            execution_kind, shard = job_to_shard[key]
            bucket = state[(execution_kind, shard)]
            bucket["rows"].append(row)
            bucket["new_since_checkpoint"] += 1
            if bucket["new_since_checkpoint"] >= CHECKPOINT_EVERY:
                checkpoint_bucket(bucket, execution_kind, shard)

        if pending_jobs and int(args.workers) == 1:
            _worker_initialize()
            try:
                for job in pending_jobs:
                    accept(_run_job(job))
            finally:
                _worker_cleanup()
        elif pending_jobs:
            if _allow_process_pool_when_sem_limit_query_is_blocked():
                print(
                    "[environment] sandbox denied SC_SEM_NSEMS_MAX; using process-local 256",
                    flush=True,
                )
            _run_jobs_bounded(pending_jobs, workers=int(args.workers), on_result=accept)
        for (execution_kind, shard), bucket in state.items():
            if bucket["new_since_checkpoint"] or not bucket["path"].exists():
                checkpoint_bucket(bucket, execution_kind, shard)
            if len(bucket["rows"]) != len(bucket["jobs"]):
                raise FormalRunError(f"selected shard did not finish: {execution_kind}/{shard}")

    environment = _privacy_safe_environment()
    kinds_to_finalize = (
        ("decision", decision, decision_groups),
        ("trajectory_replay", trajectory, trajectory_groups),
    )
    if args.kind != "all":
        kinds_to_finalize = tuple(
            item for item in kinds_to_finalize if item[0] == args.kind
        )
    finalized: list[Path] = []
    if args.finalize_only or not args.shard_id:
        for execution_kind, jobs, groups in kinds_to_finalize:
            finalized.append(
                _finalize_kind(
                    execution_kind=execution_kind,
                    jobs=jobs,
                    groups=groups,
                    output_dir=output_dir,
                    manifest_sha256=manifest_sha256,
                    source_sha256=source_sha256,
                    exact_validation_sha256=exact_sha256,
                    exact_validation_metadata_sha256=exact_metadata_sha256,
                    primary_hashes=primary_hashes,
                    runtime_identity=runtime_identity,
                    environment=environment,
                )
            )
    if (output_dir / "formal_decision_master.json").is_file() and (
        output_dir / "formal_trajectory_replays.json"
    ).is_file():
        finalized.append(
            _write_bundle_manifest(
                output_dir=output_dir,
                manifest_sha256=manifest_sha256,
                exact_validation_sha256=exact_sha256,
                exact_validation_metadata_sha256=exact_metadata_sha256,
                primary_hashes=primary_hashes,
                source_sha256=source_sha256,
                runtime_identity=runtime_identity,
            )
        )
    for path in finalized:
        print(f"[done] {path}: {_sha256_file(path)}", flush=True)
    if args.shard_id:
        print("[done] selected shard checkpoints complete; no master implied", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
