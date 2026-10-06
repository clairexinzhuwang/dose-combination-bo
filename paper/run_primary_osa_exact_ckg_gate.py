#!/usr/bin/env python3
"""Staged runner for the authorized 400-cell exact-cKG primary gate.

The default invocation is read-only preflight.  Trial execution additionally
requires ``--execute --authorization-token PRIMARY_RUN_GO``.  This runner never
updates the frozen record archive: it writes hash-bound atomic checkpoints and a
new staging artifact only.

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
import hashlib
import json
import multiprocessing
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


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
import zstandard as zstd
from scipy.stats import norm

from paper import formal_rerun_preflight as preflight
from dose_combination_bo.ckg_exact import ckg_scores_exact
from dose_combination_bo.context import Context
from dose_combination_bo.gp import fit_gp
from dose_combination_bo.registry import ACQUISITIONS
from dose_combination_bo.surfaces import resolve_surface
from dose_combination_bo.trial import grid, run_trial


AUTHORIZATION_TOKEN = "PRIMARY_RUN_GO"
POLICY_NAME = "__formal_primary_ckg_exact"
FORMAL_POLICY = "cKG-exact-formal"
HARNESS_ID = preflight.HARNESS_ID
EVALUATOR_ID = preflight.EXACT_EVALUATOR_ID
TAU = 0.7
STRATA = (0, 1)
SEED_COUNT = 200
CHECKPOINT_EVERY = 12
MAX_WORKERS = 8
EXPECTED_OSA_COMPRESSED_SHA256 = (
    "204325858aba812f977757871d863b4babcad32f5b31fef266241bedf1b56afd"
)
EXPECTED_OSA_UNCOMPRESSED_SHA256 = (
    "62547e70a028fc4819b4acd57cec43c55d9e69195ecb0cac7eec2fb04138e9f8"
)
EXPECTED_OSA_ROWS = 6_000
REFERENCE_POLICIES = ("cEI", "cKG", "cEI-tMSE")
SOURCE_FILES = (
    "paper/run_primary_osa_exact_ckg_gate.py",
    "paper/formal_rerun_preflight.py",
    "paper/formal_rerun_manifest.json",
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


class GateNumericsMismatch(RuntimeError):
    """Raised before a row is checkpointable when gate semantics differ."""


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


def _source_hashes(manifest_path: Path = preflight.MANIFEST_PATH) -> dict[str, str]:
    values = {relative: _sha256_file(ROOT / relative) for relative in SOURCE_FILES}
    values["paper/formal_rerun_manifest.json"] = _sha256_file(manifest_path)
    return values


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


def _runtime_identity() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "gpytorch": gpytorch.__version__,
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
        raise RuntimeError(
            f"primary runtime differs from the exact-validation runtime: {mismatch}"
        )
    return current


def _load_frozen_osa(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix != ".zst":
        raise ValueError("the primary gate requires the frozen compressed osa_main.json.zst")
    compressed = path.read_bytes()
    compressed_sha = _sha256_bytes(compressed)
    if compressed_sha != EXPECTED_OSA_COMPRESSED_SHA256:
        raise ValueError("frozen OSA compressed hash mismatch")
    raw = zstd.ZstdDecompressor().decompress(compressed)
    if _sha256_bytes(raw) != EXPECTED_OSA_UNCOMPRESSED_SHA256:
        raise ValueError("frozen OSA uncompressed hash mismatch")
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != EXPECTED_OSA_ROWS:
        raise ValueError("frozen OSA archive must contain exactly 6000 rows")
    selected = [
        row
        for row in rows
        if row.get("policy") in REFERENCE_POLICIES
        and row.get("mode") == "latent"
        and row.get("gamma") == TAU
    ]
    keys = {
        (str(row["policy"]), int(row["seed"]), int(row["stratum"]))
        for row in selected
    }
    expected = {
        (policy, seed, stratum)
        for policy in REFERENCE_POLICIES
        for seed in range(SEED_COUNT)
        for stratum in STRATA
    }
    if keys != expected or len(selected) != len(expected):
        raise ValueError("frozen OSA primary-reference factorial is incomplete or duplicated")
    for row in selected:
        required = {
            "sim": "osa",
            "mode": "latent",
            "gamma": TAU,
            "noise": "fixed",
            "budget": 40,
            "warmup": 4,
            "r_k": 2,
            "grid_n": 5,
            "empty_gate": "pf",
        }
        if any(row.get(field) != value for field, value in required.items()):
            raise ValueError("frozen OSA reference row has incompatible design metadata")
    return {
        "path_basename": path.name,
        "compressed_sha256": compressed_sha,
        "uncompressed_sha256": EXPECTED_OSA_UNCOMPRESSED_SHA256,
        "archive_rows": len(rows),
        "selected_reference_rows": len(selected),
    }


def _all_tasks() -> list[tuple[int, int]]:
    return [(seed, stratum) for stratum in STRATA for seed in range(SEED_COUNT)]


def _task_key(task: tuple[int, int]) -> tuple[int, int]:
    return int(task[0]), int(task[1])


def _new_gate_audit() -> dict[str, Any]:
    return {
        "definition": (
            "stable z/ppf semantics versus the current Context; CDF rounding "
            "differences are diagnostic even when a fallback is operational"
        ),
        "checked_state_count": 0,
        "acquisition_state_count": 0,
        "terminal_state_checked": False,
        "terminal_recommendation_status": "not_checked",
        "terminal_recommendation_index_difference_count": None,
        "terminal_recommendation_expected_index": None,
        "terminal_recommendation_saved_index": None,
        "gate_pass_set_difference_count": 0,
        "eligible_gate_pass_set_difference_count": 0,
        "operational_full_fallback_index_difference_count": 0,
        "operational_eligible_fallback_index_difference_count": 0,
        "inactive_full_fallback_rounding_difference_count": 0,
        "inactive_eligible_fallback_rounding_difference_count": 0,
        "operational_full_fallback_rounding_difference_count": 0,
        "operational_eligible_fallback_rounding_difference_count": 0,
        "cdf_gate_pass_set_rounding_difference_count": 0,
        "cdf_eligible_gate_pass_set_rounding_difference_count": 0,
        "cdf_saturated_zero_total": 0,
        "cdf_saturated_one_total": 0,
        "minimum_abs_z_minus_q": None,
        "maximum_abs_standardized_feasibility": 0.0,
    }


def _audit_context_gate(ctx: Context, audit: dict[str, Any], *, terminal: bool) -> None:
    z = (float(ctx.g_dagger) - np.asarray(ctx.mu_g, float)) / np.asarray(ctx.sd_g, float)
    q = float(norm.ppf(float(ctx.gamma)))
    stable_gate = z > q
    current_gate = np.asarray(ctx.gate_safe, dtype=bool)
    cdf_gate = np.asarray(ctx.pf, dtype=float) > float(ctx.gamma)
    candidate = np.asarray(ctx.candidate_mask, dtype=bool)
    stable_safe = stable_gate & candidate
    current_safe = np.asarray(ctx.safe, dtype=bool)
    cdf_safe = cdf_gate & candidate
    stable_full_fallback = int(np.argmax(z))
    current_full_fallback = int(ctx.most_feasible_full_index)
    cdf_full_fallback = int(np.argmax(ctx.pf))
    stable_eligible_fallback = int(np.argmax(np.where(candidate, z, -np.inf)))
    current_eligible_fallback = int(ctx.most_feasible_index)
    cdf_eligible_fallback = int(np.argmax(np.where(candidate, ctx.pf, -np.inf)))
    full_operational = bool(not stable_gate.any())
    eligible_operational = bool(not stable_safe.any())

    gate_diff = not np.array_equal(stable_gate, current_gate)
    safe_diff = not np.array_equal(stable_safe, current_safe)
    full_diff = stable_full_fallback != current_full_fallback
    eligible_diff = stable_eligible_fallback != current_eligible_fallback
    cdf_full_diff = stable_full_fallback != cdf_full_fallback
    cdf_eligible_diff = stable_eligible_fallback != cdf_eligible_fallback
    audit["checked_state_count"] += 1
    audit["terminal_state_checked"] = bool(audit["terminal_state_checked"] or terminal)
    audit["acquisition_state_count"] += int(not terminal)
    audit["gate_pass_set_difference_count"] += int(gate_diff)
    audit["eligible_gate_pass_set_difference_count"] += int(safe_diff)
    audit["operational_full_fallback_index_difference_count"] += int(
        full_operational and full_diff
    )
    audit["operational_eligible_fallback_index_difference_count"] += int(
        eligible_operational and eligible_diff
    )
    audit["inactive_full_fallback_rounding_difference_count"] += int(
        not full_operational and cdf_full_diff
    )
    audit["inactive_eligible_fallback_rounding_difference_count"] += int(
        not eligible_operational and cdf_eligible_diff
    )
    audit["operational_full_fallback_rounding_difference_count"] += int(
        full_operational and cdf_full_diff
    )
    audit["operational_eligible_fallback_rounding_difference_count"] += int(
        eligible_operational and cdf_eligible_diff
    )
    audit["cdf_gate_pass_set_rounding_difference_count"] += int(
        not np.array_equal(stable_gate, cdf_gate)
    )
    audit["cdf_eligible_gate_pass_set_rounding_difference_count"] += int(
        not np.array_equal(stable_safe, cdf_safe)
    )
    audit["cdf_saturated_zero_total"] += int(np.count_nonzero(ctx.pf == 0.0))
    audit["cdf_saturated_one_total"] += int(np.count_nonzero(ctx.pf == 1.0))
    margin = float(np.min(np.abs(z - q)))
    current_minimum = audit["minimum_abs_z_minus_q"]
    audit["minimum_abs_z_minus_q"] = (
        margin if current_minimum is None else min(float(current_minimum), margin)
    )
    audit["maximum_abs_standardized_feasibility"] = max(
        float(audit["maximum_abs_standardized_feasibility"]),
        float(np.max(np.abs(z))),
    )

    if gate_diff or safe_diff or (full_operational and full_diff) or (
        eligible_operational and eligible_diff
    ):
        raise GateNumericsMismatch(
            "stable standardized and current Context gate semantics differ: "
            f"terminal={terminal}, gate_diff={gate_diff}, safe_diff={safe_diff}, "
            f"full_operational={full_operational}, full_diff={full_diff}, "
            f"eligible_operational={eligible_operational}, eligible_diff={eligible_diff}"
        )


def _audit_terminal_recommendation(
    ctx: Context, result: Mapping[str, Any], audit: dict[str, Any]
) -> None:
    """Independently reproduce the stable-z terminal recommendation."""
    z = (float(ctx.g_dagger) - np.asarray(ctx.mu_g, float)) / np.asarray(ctx.sd_g, float)
    q = float(norm.ppf(float(ctx.gamma)))
    gate = z > q
    mu_f, _ = ctx.latent_efficacy()
    mu_f = np.asarray(mu_f, float).reshape(-1)
    expected_index = int(
        np.argmax(np.where(gate, mu_f, -np.inf))
        if gate.any()
        else np.argmax(z)
    )
    saved_point = np.asarray(
        [result.get("rec_d1"), result.get("rec_d2")], dtype=float
    )
    grid_points = ctx.Xset.detach().cpu().numpy()
    saved_matches = np.flatnonzero(np.all(grid_points == saved_point, axis=1))
    if len(saved_matches) != 1:
        raise GateNumericsMismatch(
            "saved terminal recommendation is not exactly one grid point"
        )
    saved_index = int(saved_matches[0])
    difference = int(saved_index != expected_index)
    audit["terminal_recommendation_status"] = "checked"
    audit["terminal_recommendation_index_difference_count"] = difference
    audit["terminal_recommendation_expected_index"] = expected_index
    audit["terminal_recommendation_saved_index"] = saved_index
    if difference:
        raise GateNumericsMismatch(
            "saved terminal recommendation differs from independent stable-z replay: "
            f"expected_index={expected_index}, saved_index={saved_index}"
        )


_ACTIVE_GATE_AUDIT: dict[str, Any] | None = None


def _audited_exact_scores(ctx: Context) -> np.ndarray:
    if _ACTIVE_GATE_AUDIT is None:
        raise RuntimeError("exact-cKG primary acquisition called outside an audited task")
    _audit_context_gate(ctx, _ACTIVE_GATE_AUDIT, terminal=False)
    return np.asarray(ckg_scores_exact(ctx), dtype=float)


_audited_exact_scores.acquisition_name = FORMAL_POLICY  # type: ignore[attr-defined]


def _worker_initialize() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    existing = ACQUISITIONS.get(POLICY_NAME)
    if existing is not None and existing is not _audited_exact_scores:
        raise RuntimeError("formal exact policy name is already registered to another function")
    ACQUISITIONS[POLICY_NAME] = _audited_exact_scores


def _terminal_context(result: Mapping[str, Any]) -> Context:
    seed = int(result["seed"])
    stratum = int(result["stratum"])
    history = np.asarray(result["allocation_history"], dtype=float)
    if history.shape != (40, 2):
        raise RuntimeError("primary full-panel result lacks a 40-dose allocation history")
    surface = resolve_surface("osa", stratum)
    rng = np.random.default_rng(seed)
    cut = np.linspace(0.0, 1.0, 5)
    efficacy: list[float] = []
    toxicity: list[float] = []
    for index, (d1, d2) in enumerate(history):
        if index < 4:
            expected = np.asarray(
                [cut[index] + rng.uniform() * (cut[1] - cut[0]), rng.uniform()]
            )
            if not np.array_equal(expected, np.asarray([d1, d2])):
                raise RuntimeError("terminal replay cannot reproduce the LHS initialization")
        efficacy.append(surface["eff"](d1, d2) + rng.normal(0.0, surface["sf"]))
        toxicity.append(surface["tox"](d1, d2) + rng.normal(0.0, surface["sg"]))
    observed = torch.tensor(history)
    me, le = fit_gp(
        observed, torch.tensor(efficacy), fixed_noise=float(surface["sf"]) ** 2
    )
    mt, lt = fit_gp(
        observed, torch.tensor(toxicity), fixed_noise=float(surface["sg"]) ** 2
    )
    return Context(
        Xset=torch.tensor(grid(5)),
        me=me,
        le=le,
        mt=mt,
        lt=lt,
        g_dagger=float(surface["gd"]),
        gamma=TAU,
        gate_mode="latent",
        r_k=2,
        empty_gate="pf",
    )


def _trial_kwargs(seed: int, stratum: int) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "z": int(stratum),
        "gamma": TAU,
        "sim": "osa",
        "mode": "latent",
        "noise": "fixed",
        "traj": bool(seed < 80),
        "budget": 40,
        "warmup": 4,
        "r_k": 2,
        "grid_n": 5,
        "empty_gate": "pf",
        "kap": 1.0,
        "protocol_scaffold": "lhs_fixed",
        "region_step": 0.25,
        "empty_gate_stop_after": 3,
        "exclude_repeats_during_expansion": True,
    }


def _run_task(task: tuple[int, int]) -> dict[str, Any]:
    global _ACTIVE_GATE_AUDIT
    seed, stratum = _task_key(task)
    if _ACTIVE_GATE_AUDIT is not None:
        raise RuntimeError("worker gate-audit state leaked from a previous task")
    audit = _new_gate_audit()
    _ACTIVE_GATE_AUDIT = audit
    started = time.perf_counter()
    try:
        result = run_trial(POLICY_NAME, **_trial_kwargs(seed, stratum))
        terminal = _terminal_context(result)
        _audit_context_gate(terminal, audit, terminal=True)
        _audit_terminal_recommendation(terminal, result, audit)
    finally:
        _ACTIVE_GATE_AUDIT = None
    if audit["checked_state_count"] != 19 or audit["acquisition_state_count"] != 18:
        raise RuntimeError(f"primary row has incomplete gate audit: {audit}")
    if audit["terminal_state_checked"] is not True:
        raise RuntimeError("primary row lacks its terminal gate audit")
    if int(result.get("n_gate_pass", -1)) != int(np.count_nonzero(terminal.gate_safe)):
        raise RuntimeError("terminal replay disagrees with the saved gate-pass count")
    result = dict(result)
    result["policy"] = FORMAL_POLICY
    expected_trajectory = seed < 80
    if (result.get("traj") is not None) != expected_trajectory:
        raise RuntimeError("trajectory capture differs from the seeds 0..79 contract")
    if expected_trajectory and len(result["traj"].get("n", [])) != 18:
        raise RuntimeError("captured trajectory must contain 18 adaptive decisions")
    if (
        result.get("recommendation_made") is not True
        or result.get("stop_reason") is not None
        or int(result.get("n_enrolled", -1)) != 40
    ):
        raise RuntimeError("full-panel primary row unexpectedly stopped or omitted a recommendation")
    return {
        "policy": FORMAL_POLICY,
        "seed": seed,
        "stratum": stratum,
        "harness_id": HARNESS_ID,
        "evaluator_id": EVALUATOR_ID,
        "historical_wrapper_used": False,
        "gate_numerics_audit": audit,
        "elapsed_seconds_not_for_timing_inference": time.perf_counter() - started,
        "result": result,
    }


def _design() -> dict[str, Any]:
    return {
        "matrix_id": "primary_osa_tau07_exact_ckg",
        "policy": FORMAL_POLICY,
        "surface": "osa",
        "tau": TAU,
        "gate_mode": "latent",
        "protocol_scaffold": "lhs_fixed",
        "noise": "fixed",
        "kap": 1.0,
        "budget": 40,
        "warmup": 4,
        "cohort_size": 2,
        "grid_n": 5,
        "empty_gate": "pf",
        "seeds": list(range(SEED_COUNT)),
        "strata": list(STRATA),
        "expected_cells": 400,
        "trajectory_capture": "seeds 0..79 in both strata; 160 rows in place",
        "scope": (
            "Primary aggregate-OC gate only. This artifact does not authorize or "
            "contain the broader 44,200-execution formal rerun."
        ),
    }


def _fingerprint(
    *,
    manifest_sha256: str,
    exact_panel_sha256: str,
    frozen_osa: Mapping[str, Any],
    sources: Mapping[str, str],
    runtime_identity: Mapping[str, str],
    shard_count: int,
) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "design": _design(),
                "manifest_sha256": manifest_sha256,
                "exact_panel_sha256": exact_panel_sha256,
                "frozen_osa": dict(frozen_osa),
                "source_sha256": dict(sources),
                "runtime_identity": dict(runtime_identity),
                "shard_count": int(shard_count),
            }
        )
    )


def _validate_row(row: Mapping[str, Any], expected: set[tuple[int, int]]) -> tuple[int, int]:
    key = (int(row["seed"]), int(row["stratum"]))
    if key not in expected:
        raise ValueError(f"checkpoint contains a cell outside its shard: {key}")
    if (
        row.get("policy") != FORMAL_POLICY
        or row.get("harness_id") != HARNESS_ID
        or row.get("evaluator_id") != EVALUATOR_ID
        or row.get("historical_wrapper_used") is not False
    ):
        raise ValueError(f"checkpoint cell has invalid compute provenance: {key}")
    audit = row.get("gate_numerics_audit", {})
    zero_fields = (
        "gate_pass_set_difference_count",
        "eligible_gate_pass_set_difference_count",
        "operational_full_fallback_index_difference_count",
        "operational_eligible_fallback_index_difference_count",
    )
    if (
        audit.get("checked_state_count") != 19
        or audit.get("acquisition_state_count") != 18
        or audit.get("terminal_state_checked") is not True
        or any(audit.get(field) != 0 for field in zero_fields)
    ):
        raise ValueError(f"checkpoint cell has an incomplete/failed gate audit: {key}")
    result = row.get("result", {})
    if (
        result.get("policy") != FORMAL_POLICY
        or result.get("sim") != "osa"
        or result.get("mode") != "latent"
        or result.get("gamma") != TAU
        or result.get("empty_gate") != "pf"
    ):
        raise ValueError(f"checkpoint cell has incompatible trial metadata: {key}")
    if (result.get("traj") is not None) != (key[0] < 80):
        raise ValueError(f"checkpoint cell violates trajectory capture: {key}")
    return key


def _checkpoint_path(output_dir: Path, shard_count: int, shard_index: int) -> Path:
    if shard_count == 1:
        return output_dir / "primary_osa_exact_ckg_gate.checkpoint.json"
    return output_dir / (
        f"primary_osa_exact_ckg_gate.shard-{shard_index:02d}-"
        f"of-{shard_count:02d}.checkpoint.json"
    )


def _load_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    expected_tasks: Iterable[tuple[int, int]],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != (
        "PARTIAL_PRIMARY_400_NOT_FOR_ANALYSIS"
    ):
        raise ValueError("checkpoint schema/status mismatch")
    if payload.get("fingerprint") != fingerprint:
        raise ValueError("checkpoint fingerprint mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("checkpoint rows must be a list")
    expected = set(expected_tasks)
    keys = [_validate_row(row, expected) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("checkpoint contains duplicate cells")
    return [dict(row) for row in rows]


def _write_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    manifest_sha256: str,
    exact_panel_sha256: str,
    source_sha256: Mapping[str, str],
    runtime_identity: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    _atomic_json_write(
        path,
        {
            "schema_version": 1,
            "status": "PARTIAL_PRIMARY_400_NOT_FOR_ANALYSIS",
            "fingerprint": fingerprint,
            "manifest_sha256": manifest_sha256,
            "exact_validation_panel_sha256": exact_panel_sha256,
            "source_sha256": dict(source_sha256),
            "runtime_identity": dict(runtime_identity),
            "completed_cell_count": len(rows),
            "rows": sorted(
                (dict(row) for row in rows),
                key=lambda row: (int(row["stratum"]), int(row["seed"])),
            ),
        },
    )


def _run_tasks_bounded(
    tasks: Sequence[tuple[int, int]],
    *,
    workers: int,
    on_result: Any,
) -> None:
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_initialize,
    ) as pool:
        iterator = iter(tasks)
        pending: dict[concurrent.futures.Future, tuple[int, int]] = {}

        def fill() -> None:
            while len(pending) < 2 * workers:
                try:
                    task = next(iterator)
                except StopIteration:
                    return
                pending[pool.submit(_run_task, task)] = task

        fill()
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                task = pending.pop(future)
                try:
                    row = future.result()
                except BaseException:
                    for remaining in pending:
                        remaining.cancel()
                    raise RuntimeError(f"primary exact-cKG task failed: {task}") from future.exception()
                on_result(row)
            fill()


def _allow_process_pool_when_sem_limit_query_is_blocked() -> bool:
    """Use the conservative POSIX minimum only when macOS denies this query."""
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


def _authorization_is_valid(*, execute: bool, merge_only: bool, token: str | None) -> bool:
    return not (execute or merge_only) or token == AUTHORIZATION_TOKEN


def _finalize(
    output_dir: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
    exact_panel: Path,
    exact_panel_sha256: str,
    frozen_osa: Mapping[str, Any],
    sources: Mapping[str, str],
    runtime_identity: Mapping[str, str],
    fingerprint: str,
    environment: Mapping[str, Any],
) -> Path:
    expected = set(_all_tasks())
    keys = [_validate_row(row, expected) for row in rows]
    if len(keys) != 400 or set(keys) != expected:
        raise RuntimeError("final primary artifact requires exactly all 400 cells")
    if sum(row["result"].get("traj") is not None for row in rows) != 160:
        raise RuntimeError("final primary artifact requires exactly 160 trajectories")
    raw_path = output_dir / "primary_osa_exact_ckg_gate_raw.json"
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_PRIMARY_400_EXACT_CKG_GATE",
        "artifact_class": "formal_primary_gate_staging_not_archive_records",
        "design": _design(),
        "provenance": {
            "manifest_sha256": manifest_sha256,
            "exact_validation_panel": exact_panel.name,
            "exact_validation_panel_sha256": exact_panel_sha256,
            "exact_validation_metadata_sha256": _sha256_file(
                Path(str(exact_panel) + ".metadata.json")
            ),
            "frozen_osa": dict(frozen_osa),
            "checkpoint_fingerprint": fingerprint,
            "authorization": AUTHORIZATION_TOKEN,
            "historical_wrappers_used": False,
            "production_ckg_default_changed": False,
            "runtime_identity": dict(runtime_identity),
        },
        "source_sha256": dict(sources),
        "environment": dict(environment),
        "rows": sorted(
            (dict(row) for row in rows),
            key=lambda row: (int(row["stratum"]), int(row["seed"])),
        ),
    }
    _atomic_json_write(raw_path, payload)
    metadata_path = Path(str(raw_path) + ".metadata.json")
    _atomic_json_write(
        metadata_path,
        {
            "schema_version": 1,
            "artifact": raw_path.name,
            "artifact_sha256": _sha256_file(raw_path),
            "artifact_bytes": raw_path.stat().st_size,
            "status": payload["status"],
            "row_count": 400,
            "trajectory_row_count": 160,
            "manifest_sha256": manifest_sha256,
            "exact_validation_panel_sha256": exact_panel_sha256,
            "source_sha256": dict(sources),
            "runtime_identity": dict(runtime_identity),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "publication_scope": "staging primary gate only; frozen archive unchanged",
        },
    )
    return raw_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-osa", required=True, type=Path)
    parser.add_argument(
        "--exact-validation-panel",
        required=True,
        type=Path,
        help="complete schema-3 stable-z 720-state PASS panel",
    )
    parser.add_argument("--manifest", type=Path, default=preflight.MANIFEST_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/primary_osa_exact_ckg_gate",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--merge-only", action="store_true")
    parser.add_argument("--authorization-token")
    args = parser.parse_args(argv)

    if not 1 <= int(args.workers) <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    if not 1 <= int(args.shard_count) <= 16:
        parser.error("--shard-count must be between 1 and 16")
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        parser.error("--shard-index must be in [0, shard-count)")
    if args.merge_only and int(args.shard_count) < 2:
        parser.error("--merge-only requires --shard-count >= 2")
    if not _authorization_is_valid(
        execute=bool(args.execute),
        merge_only=bool(args.merge_only),
        token=args.authorization_token,
    ):
        parser.error("execution/final merge requires --authorization-token PRIMARY_RUN_GO")

    manifest = preflight.load_manifest(args.manifest.resolve())
    primary_jobs = preflight.primary_jobs(manifest)
    if len(primary_jobs) != 400:
        raise RuntimeError("formal manifest no longer defines the approved primary 400 cells")
    exact_summary = preflight.audit_exact_validation_artifact(
        args.exact_validation_panel.resolve()
    )
    runtime_identity = _require_validation_runtime_match(
        args.exact_validation_panel.resolve()
    )
    frozen_osa = _load_frozen_osa(args.frozen_osa.resolve())
    manifest_sha256 = _sha256_file(args.manifest.resolve())
    exact_panel_sha256 = _sha256_file(args.exact_validation_panel.resolve())
    sources = _source_hashes(args.manifest.resolve())
    fingerprint = _fingerprint(
        manifest_sha256=manifest_sha256,
        exact_panel_sha256=exact_panel_sha256,
        frozen_osa=frozen_osa,
        sources=sources,
        runtime_identity=runtime_identity,
        shard_count=int(args.shard_count),
    )
    print(
        f"[preflight] primary=400; trajectories=160; exact={exact_summary}; "
        f"fingerprint={fingerprint}",
        flush=True,
    )
    if not args.execute and not args.merge_only:
        print("[preflight] simulation authorization not exercised; no output written", flush=True)
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = _all_tasks()
    if args.merge_only:
        rows: list[dict[str, Any]] = []
        for shard_index in range(int(args.shard_count)):
            expected_tasks = {
                task
                for index, task in enumerate(all_tasks)
                if index % int(args.shard_count) == shard_index
            }
            rows.extend(
                _load_checkpoint(
                    _checkpoint_path(output_dir, int(args.shard_count), shard_index),
                    fingerprint=fingerprint,
                    expected_tasks=expected_tasks,
                )
            )
    else:
        expected_tasks = {
            task
            for index, task in enumerate(all_tasks)
            if index % int(args.shard_count) == int(args.shard_index)
        }
        checkpoint = _checkpoint_path(
            output_dir, int(args.shard_count), int(args.shard_index)
        )
        rows = _load_checkpoint(
            checkpoint, fingerprint=fingerprint, expected_tasks=expected_tasks
        )
        complete = {(int(row["seed"]), int(row["stratum"])) for row in rows}
        tasks = sorted(expected_tasks - complete)
        print(
            f"[execute] shard cells={len(expected_tasks)}; resumed={len(rows)}; "
            f"remaining={len(tasks)}; workers={args.workers}",
            flush=True,
        )
        new_since_checkpoint = 0

        def accept(row: dict[str, Any]) -> None:
            nonlocal new_since_checkpoint
            rows.append(row)
            new_since_checkpoint += 1
            if new_since_checkpoint >= CHECKPOINT_EVERY:
                _write_checkpoint(
                    checkpoint,
                    fingerprint=fingerprint,
                    manifest_sha256=manifest_sha256,
                    exact_panel_sha256=exact_panel_sha256,
                    source_sha256=sources,
                    runtime_identity=runtime_identity,
                    rows=rows,
                )
                new_since_checkpoint = 0

        if tasks and int(args.workers) == 1:
            _worker_initialize()
            for task in tasks:
                accept(_run_task(task))
        elif tasks:
            if _allow_process_pool_when_sem_limit_query_is_blocked():
                print(
                    "[environment] sandbox denied SC_SEM_NSEMS_MAX; using the "
                    "process-local conservative value 256",
                    flush=True,
                )
            _run_tasks_bounded(tasks, workers=int(args.workers), on_result=accept)
        if new_since_checkpoint or not checkpoint.exists():
            _write_checkpoint(
                checkpoint,
                fingerprint=fingerprint,
                manifest_sha256=manifest_sha256,
                exact_panel_sha256=exact_panel_sha256,
                source_sha256=sources,
                runtime_identity=runtime_identity,
                rows=rows,
            )
        if len(rows) != len(expected_tasks):
            raise RuntimeError("primary shard did not finish every expected cell")
        if int(args.shard_count) > 1:
            print(f"[done] complete shard checkpoint: {checkpoint}", flush=True)
            return 0

    raw_path = _finalize(
        output_dir,
        rows=rows,
        manifest_sha256=manifest_sha256,
        exact_panel=args.exact_validation_panel.resolve(),
        exact_panel_sha256=exact_panel_sha256,
        frozen_osa=frozen_osa,
        sources=sources,
        runtime_identity=runtime_identity,
        fingerprint=fingerprint,
        environment=_privacy_safe_environment(),
    )
    print(f"[done] primary staging artifact: {raw_path}", flush=True)
    print(f"[done] sha256={_sha256_file(raw_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
