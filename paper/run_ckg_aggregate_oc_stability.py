#!/usr/bin/env python3
"""Run the paired aggregate-OC cKG fantasy sensitivity.

This is an additive, post-hoc numerical diagnostic.  It does not change the
production cKG implementation, package defaults, public API, or frozen main
simulation records.  The primary OSA full-panel cell (tau=0.7, both strata) is
evaluated for

* the archived production cKG-512 policy (fantasy-bank seed 0),
* a custom prefix-preserving cKG-1024 anchor with bank seed 0, and
* the same cKG-1024 anchor with an independent fixed bank seed 1.

The 1024 construction preserves the two 512-channel production draw prefixes
before appending new draws.  Seed 1 is a second deterministic numerical arm; it
does not randomize fantasies within a trial.  Trial seeds still control the
warm-up and observation streams and are paired across every arm and comparator.

The runner validates the archived cKG-512 rows against the existing production-
equivalence diagnostic for seeds 0--19 before launching the long run.  Completed
trial rows are saved with an atomic checkpoint replacement, so an interrupted
run can resume without repeating completed cells.  Agreement or aggregate-OC
similarity is evidence about numerical stability only, not accuracy, convergence,
or a universally adequate fantasy count.
"""

from __future__ import annotations

import os

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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

from paper import run_ckg_1024_anchor as anchor
from paper import run_computational_diagnostics as diagnostic
from dose_combination_bo.registry import ACQUISITIONS
from dose_combination_bo.trial import run_trial


TAU = 0.7
STRATA = (0, 1)
DEFAULT_SEED_COUNT = 200
ARMS = ("cKG-1024-seed0", "cKG-1024-seed1")
FANTASY_SEED = {"cKG-1024-seed0": 0, "cKG-1024-seed1": 1}
POLICY_NAME = {
    "cKG-1024-seed0": "__diag_ckg_aggregate_1024_seed0",
    "cKG-1024-seed1": "__diag_ckg_aggregate_1024_seed1",
}
REFERENCE_POLICIES = ("cEI", "cKG", "cEI-tMSE")
EXPECTED_COMPRESSED_SHA256 = (
    "204325858aba812f977757871d863b4babcad32f5b31fef266241bedf1b56afd"
)
EXPECTED_UNCOMPRESSED_SHA256 = (
    "62547e70a028fc4819b4acd57cec43c55d9e69195ecb0cac7eec2fb04138e9f8"
)
EXPECTED_ARCHIVE_ROWS = 6000
REFERENCE_AUDIT_FIELDS = (
    "rec_d1",
    "rec_d2",
    "rec_unsafe",
    "rec_true_eff",
    "rec_true_tox",
    "dose_units",
    "rpsel",
    "toxic",
    "n_gate_pass",
    "n_gate_pass_safe",
    "obd_pf",
    "obd_sdg",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def atomic_json_write(path: Path, payload: Any) -> None:
    """Write JSON by fsyncing a sibling temporary file then replacing atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_record_bytes(path: Path) -> bytes:
    if path.suffix == ".zst":
        completed = subprocess.run(
            ["zstd", "-q", "-d", "-c", str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout
    return path.read_bytes()


def _validate_archive_row(row: dict[str, Any]) -> None:
    expected = {
        "sim": "osa",
        "gamma": TAU,
        "mode": "latent",
        "noise": "fixed",
        "budget": 40,
        "warmup": 4,
        "r_k": 2,
        "grid_n": 5,
        "empty_gate": "pf",
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"archive row has incompatible {key}: {row.get(key)!r}")
    if row.get("policy") not in REFERENCE_POLICIES:
        raise ValueError("unexpected reference policy")
    if int(row.get("stratum", -1)) not in STRATA:
        raise ValueError("unexpected OSA stratum")


def _compact_archive_row(row: dict[str, Any]) -> dict[str, Any]:
    # Full-panel lhs_fixed has no stopping rule.  Recommendation/stop fields are
    # encoded explicitly here as design facts, not estimated empirical findings.
    return {
        "policy": str(row["policy"]),
        "seed": int(row["seed"]),
        "stratum": int(row["stratum"]),
        "recommendation_made": True,
        "stop_reason": None,
        "n_enrolled": 40,
        "rec_d1": float(row["rec_d1"]),
        "rec_d2": float(row["rec_d2"]),
        "rec_unsafe": int(row["rec_unsafe"]),
        "rec_true_eff": float(row["rec_true_eff"]),
        "rec_true_tox": float(row["rec_true_tox"]),
        "dose_units": float(row["dose_units"]),
        "rpsel": float(row["rpsel"]),
        "above_boundary_assignments": int(row["toxic"]),
        "n_gate_pass": int(row["n_gate_pass"]),
        "n_gate_pass_safe": int(row["n_gate_pass_safe"]),
        "obd_pf": float(row["obd_pf"]),
        "obd_sdg": float(row["obd_sdg"]),
    }


def load_production_reference(
    record_path: Path, seed_count: int
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], dict[str, Any]], dict[str, Any]]:
    if not record_path.exists():
        raise FileNotFoundError(record_path)
    compressed_sha = sha256_file(record_path) if record_path.suffix == ".zst" else None
    if compressed_sha is not None and compressed_sha != EXPECTED_COMPRESSED_SHA256:
        raise ValueError("OSA compressed record hash does not match the frozen archive")
    raw_bytes = _read_record_bytes(record_path)
    if sha256_bytes(raw_bytes) != EXPECTED_UNCOMPRESSED_SHA256:
        raise ValueError("OSA decompressed record hash does not match the frozen archive")
    rows = json.loads(raw_bytes)
    if not isinstance(rows, list) or len(rows) != EXPECTED_ARCHIVE_ROWS:
        raise ValueError("OSA record does not contain the expected 6000 rows")
    selected = [
        row
        for row in rows
        if row.get("policy") in REFERENCE_POLICIES
        and row.get("gamma") == TAU
        and int(row.get("seed", -1)) < int(seed_count)
    ]
    expected_keys = {
        (policy, seed, stratum)
        for policy in REFERENCE_POLICIES
        for seed in range(seed_count)
        for stratum in STRATA
    }
    lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in selected:
        _validate_archive_row(row)
        key = (str(row["policy"]), int(row["seed"]), int(row["stratum"]))
        if key in lookup:
            raise ValueError(f"duplicate archive cell: {key}")
        lookup[key] = row
    if set(lookup) != expected_keys:
        missing = sorted(expected_keys - set(lookup))[:5]
        extra = sorted(set(lookup) - expected_keys)[:5]
        raise ValueError(f"reference factorial is incomplete; missing={missing}, extra={extra}")
    compact = [_compact_archive_row(lookup[key]) for key in sorted(lookup)]
    provenance = {
        "record_basename": record_path.name,
        "compressed_sha256": compressed_sha,
        "uncompressed_sha256": EXPECTED_UNCOMPRESSED_SHA256,
        "archive_record_count": EXPECTED_ARCHIVE_ROWS,
        "selected_reference_record_count": len(compact),
    }
    return compact, lookup, provenance


def validate_production_equivalence(
    archive_lookup: dict[tuple[str, int, int], dict[str, Any]],
    diagnostic_raw: Path,
) -> dict[str, Any]:
    production, _, source = anchor.load_reference(diagnostic_raw.resolve())
    mismatches: list[dict[str, Any]] = []
    for (seed, stratum), observed in production.items():
        archived = archive_lookup[("cKG", seed, stratum)]
        for field in REFERENCE_AUDIT_FIELDS:
            if archived[field] != observed[field]:
                mismatches.append(
                    {
                        "seed": seed,
                        "stratum": stratum,
                        "field": field,
                        "archive": archived[field],
                        "diagnostic": observed[field],
                    }
                )
    if mismatches:
        raise ValueError(f"archived cKG-512 rows differ from diagnostic: {mismatches[:3]}")
    active = anchor.ActiveRun()
    with anchor.installed_anchor_acquisitions(active):
        short_preflight = anchor.production_compatibility_preflight(active)
    return {
        "status": "pass",
        "short_current_code_preflight": short_preflight,
        "archived_seed_stratum_cells_checked": len(production),
        "audited_fields_per_cell": list(REFERENCE_AUDIT_FIELDS),
        "all_archived_fields_exact": True,
        "diagnostic_source": source,
        "interpretation": (
            "The archived 512 operating-characteristic fields for seeds 0--19 and "
            "both strata exactly match the production-compatible current diagnostic."
        ),
    }


def trial_kwargs(seed: int, stratum: int) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "z": int(stratum),
        "gamma": TAU,
        "sim": "osa",
        "mode": "latent",
        "noise": "fixed",
        "traj": False,
        "budget": 40,
        "warmup": 4,
        "r_k": 2,
        "grid_n": 5,
        "empty_gate": "pf",
        "protocol_scaffold": "lhs_fixed",
    }


def _scores(ctx: object, fantasy_seed: int) -> np.ndarray:
    scores: list[float] = []
    for index in range(ctx.Xset.shape[0]):
        _, value_1024 = anchor.ckg_one_step_gated_anchor_pair(
            ctx.Xset[index],
            ctx.Xset,
            ctx.me,
            ctx.le,
            ctx.mt,
            ctx.lt,
            ctx.g_dagger,
            ctx.gamma,
            r_k=ctx.r_k,
            seed=int(fantasy_seed),
            gate_mode=ctx.gate_mode,
        )
        scores.append(value_1024)
    return ctx.restrict(np.asarray(scores, dtype=float))


def _worker_initialize() -> None:
    diagnostic.configure_single_thread_runtime()

    def seed0(ctx: object) -> np.ndarray:
        return _scores(ctx, 0)

    def seed1(ctx: object) -> np.ndarray:
        return _scores(ctx, 1)

    seed0.acquisition_name = "post-hoc cKG-1024 seed 0"  # type: ignore[attr-defined]
    seed1.acquisition_name = "post-hoc cKG-1024 seed 1"  # type: ignore[attr-defined]
    ACQUISITIONS[POLICY_NAME["cKG-1024-seed0"]] = seed0
    ACQUISITIONS[POLICY_NAME["cKG-1024-seed1"]] = seed1


def _allow_process_pool_when_sem_limit_query_is_blocked() -> bool:
    """Work around a read-only macOS sandbox denial in ProcessPoolExecutor.

    CPython asks ``SC_SEM_NSEMS_MAX`` before creating a pool and treats a denied
    read exactly like an inadequate system limit.  The semaphore operations
    themselves remain available in this environment.  When only that query is
    denied, return the conservative POSIX minimum (256) for this process.  No
    package or operating-system setting is changed.
    """
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


def _compact_new_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendation_made": bool(result["recommendation_made"]),
        "stop_reason": result["stop_reason"],
        "n_enrolled": int(result["n_enrolled"]),
        "rec_d1": result["rec_d1"],
        "rec_d2": result["rec_d2"],
        "rec_unsafe": result["rec_unsafe"],
        "rec_true_eff": result["rec_true_eff"],
        "rec_true_tox": result["rec_true_tox"],
        "dose_units": result["dose_units"],
        "rpsel": result["rpsel"],
        "above_boundary_assignments": int(result["toxic"]),
        "n_gate_pass": int(result["n_gate_pass"]),
        "n_gate_pass_safe": int(result["n_gate_pass_safe"]),
        "obd_pf": float(result["obd_pf"]),
        "obd_sdg": float(result["obd_sdg"]),
        "allocation_history": result["allocation_history"],
    }


def _run_task(task: tuple[str, int, int]) -> dict[str, Any]:
    arm, seed, stratum = task
    started = time.perf_counter()
    result = run_trial(POLICY_NAME[arm], **trial_kwargs(seed, stratum))
    return {
        "arm": arm,
        "fantasy_count": 1024,
        "fantasy_seed": FANTASY_SEED[arm],
        "seed": int(seed),
        "stratum": int(stratum),
        "elapsed_seconds_not_for_timing_inference": time.perf_counter() - started,
        "result": _compact_new_result(result),
    }


def _design(seed_count: int, archive_provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "post hoc numerical-stability sensitivity",
        "surface": "osa",
        "protocol_scaffold": "lhs_fixed full panel",
        "gate_mode": "latent",
        "noise": "fixed",
        "tau": TAU,
        "strata": list(STRATA),
        "seeds": list(range(seed_count)),
        "independent_trial_seed_count": int(seed_count),
        "seed_stratum_cells_per_policy": int(seed_count * len(STRATA)),
        "budget": 40,
        "warmup": 4,
        "cohort_size": 2,
        "grid_n": 5,
        "empty_gate": "pf",
        "reference": "archived production cKG-512 with fantasy-bank seed 0",
        "new_arms": [
            {
                "arm": arm,
                "fantasy_count": 1024,
                "fantasy_seed": FANTASY_SEED[arm],
                "bank": (
                    "Draw Zf[0:512], Zg[0:512], then fresh Zf[512:1024], "
                    "Zg[512:1024]; compare channel-wise prefixes."
                ),
            }
            for arm in ARMS
        ],
        "pairing": (
            "Trial seed and stratum are identical across production cKG-512, both "
            "cKG-1024 arms, cEI, and the specified 1:1 cEI-tMSE reference."
        ),
        "full_panel_stopping_fact": (
            "lhs_fixed has no early-stopping branch; recommendation rate is 100% and "
            "stop rate is 0% by design, not an empirical numerical-stability finding."
        ),
        "interpretation": (
            "This post-hoc sensitivity assesses aggregate numerical stability, not "
            "decision accuracy, Monte Carlo convergence, or a universal fantasy count."
        ),
        "archive_provenance": archive_provenance,
    }


def privacy_safe_environment() -> dict[str, Any]:
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
            "dose-combination-bo": "source-tree",
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


def source_hashes() -> dict[str, str]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "paper/run_ckg_1024_anchor.py",
        ROOT / "paper/run_computational_diagnostics.py",
        ROOT / "src/dose_combination_bo/acquisitions.py",
        ROOT / "src/dose_combination_bo/context.py",
        ROOT / "src/dose_combination_bo/gp.py",
        ROOT / "src/dose_combination_bo/trial.py",
    )
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sources}


def _checkpoint_fingerprint(design: dict[str, Any], sources: dict[str, str]) -> str:
    payload = {"design": design, "source_sha256": sources, "arms": list(ARMS)}
    return sha256_bytes(canonical_json_bytes(payload))


def _load_checkpoint(path: Path, fingerprint: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        raise ValueError("checkpoint fingerprint differs from this design/source freeze")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("checkpoint rows must be a list")
    keys = [(row["arm"], int(row["seed"]), int(row["stratum"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("checkpoint contains duplicate cells")
    return rows


def _write_checkpoint(path: Path, fingerprint: str, rows: list[dict[str, Any]]) -> None:
    atomic_json_write(
        path,
        {
            "schema_version": 1,
            "status": "atomic resumable execution checkpoint",
            "fingerprint": fingerprint,
            "completed_cell_count": len(rows),
            "rows": sorted(rows, key=lambda row: (row["arm"], row["seed"], row["stratum"])),
        },
    )


def _metadata(
    artifact: Path,
    design: dict[str, Any],
    environment: dict[str, Any],
    sources: dict[str, str],
    record_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": artifact.name,
        "artifact_type": "ckg_aggregate_oc_stability_raw",
        "artifact_sha256": sha256_file(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post hoc numerical-stability sensitivity",
        "design": design,
        "environment": environment,
        "record_counts": record_counts,
        "source_sha256": sources,
        "interpretation": (
            "Numerical stability only; this artifact does not establish accuracy, "
            "convergence, or a universal fantasy count."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-record",
        type=Path,
        required=True,
        help="Frozen archive records/osa_main.json.zst (or exact decompressed JSON).",
    )
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge already-complete shard checkpoints and write the final raw artifact.",
    )
    parser.add_argument(
        "--diagnostic-raw",
        type=Path,
        default=ROOT / "results/computational_diagnostics/computational_diagnostics_raw.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/ckg_aggregate_oc_stability",
    )
    args = parser.parse_args(argv)
    if not 20 <= int(args.seed_count) <= DEFAULT_SEED_COUNT:
        parser.error("--seed-count must be between 20 and 200")
    if int(args.workers) < 1:
        parser.error("--workers must be positive")
    if int(args.shard_count) < 1:
        parser.error("--shard-count must be positive")
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        parser.error("--shard-index must be in [0, shard-count)")
    if args.merge_only and int(args.shard_count) < 2:
        parser.error("--merge-only requires --shard-count >= 2")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_rows, archive_lookup, archive_provenance = load_production_reference(
        args.production_record.resolve(), int(args.seed_count)
    )
    production_equivalence = validate_production_equivalence(
        archive_lookup, args.diagnostic_raw.resolve()
    )
    design = _design(int(args.seed_count), archive_provenance)
    sources = source_hashes()
    environment = privacy_safe_environment()
    fingerprint = _checkpoint_fingerprint(design, sources)
    if int(args.shard_count) == 1:
        checkpoint_path = output_dir / "ckg_aggregate_oc_stability.checkpoint.json"
    else:
        checkpoint_path = output_dir / (
            f"ckg_aggregate_oc_stability.shard-{int(args.shard_index):02d}-"
            f"of-{int(args.shard_count):02d}.checkpoint.json"
        )
    if args.merge_only:
        rows = []
        for shard_index in range(int(args.shard_count)):
            shard_path = output_dir / (
                f"ckg_aggregate_oc_stability.shard-{shard_index:02d}-"
                f"of-{int(args.shard_count):02d}.checkpoint.json"
            )
            rows.extend(_load_checkpoint(shard_path, fingerprint))
        keys = [(row["arm"], int(row["seed"]), int(row["stratum"])) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("merged shard checkpoints contain duplicate cells")
    else:
        rows = _load_checkpoint(checkpoint_path, fingerprint)
    completed = {(row["arm"], int(row["seed"]), int(row["stratum"])) for row in rows}
    all_tasks = [
        (arm, seed, stratum)
        for arm in ARMS
        for seed in range(int(args.seed_count))
        for stratum in STRATA
    ]
    if int(args.shard_count) > 1 and not args.merge_only:
        eligible_tasks = [
            task
            for index, task in enumerate(all_tasks)
            if index % int(args.shard_count) == int(args.shard_index)
        ]
    else:
        eligible_tasks = all_tasks
    tasks = [task for task in eligible_tasks if task not in completed]
    print(
        f"[design] {len(completed)} completed; {len(tasks)} remaining; "
        f"workers={args.workers}",
        flush=True,
    )
    started = time.perf_counter()
    if tasks and int(args.workers) == 1:
        _worker_initialize()
        total = len(tasks)
        for index, task in enumerate(tasks, start=1):
            row = _run_task(task)
            rows.append(row)
            _write_checkpoint(checkpoint_path, fingerprint, rows)
            if index == 1 or index % 10 == 0 or index == total:
                elapsed = time.perf_counter() - started
                print(
                    f"[progress] {index}/{total} new cells; elapsed={elapsed:.1f}s; "
                    f"latest={task}",
                    flush=True,
                )
    elif tasks:
        sem_query_workaround = _allow_process_pool_when_sem_limit_query_is_blocked()
        if sem_query_workaround:
            print(
                "[environment] SC_SEM_NSEMS_MAX read was sandbox-blocked; "
                "using the process-local conservative value 256",
                flush=True,
            )
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(args.workers),
            mp_context=context,
            initializer=_worker_initialize,
        ) as pool:
            futures = {pool.submit(_run_task, task): task for task in tasks}
            total = len(tasks)
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                task = futures[future]
                try:
                    row = future.result()
                except BaseException:
                    print(f"[failed] task={task}", flush=True)
                    raise
                rows.append(row)
                _write_checkpoint(checkpoint_path, fingerprint, rows)
                if index == 1 or index % 10 == 0 or index == total:
                    elapsed = time.perf_counter() - started
                    print(
                        f"[progress] {index}/{total} new cells; elapsed={elapsed:.1f}s; "
                        f"latest={task}",
                        flush=True,
                    )

    expected_new = len(ARMS) * int(args.seed_count) * len(STRATA)
    if int(args.shard_count) > 1 and not args.merge_only:
        expected_shard = len(eligible_tasks)
        if len(rows) != expected_shard:
            raise RuntimeError(f"completed {len(rows)} shard cells; expected {expected_shard}")
        print(f"[done] shard checkpoint={checkpoint_path}", flush=True)
        print(f"[done] shard cells={len(rows)}", flush=True)
        return 0
    if len(rows) != expected_new:
        raise RuntimeError(f"completed {len(rows)} new cells; expected {expected_new}")
    if any(not row["result"]["recommendation_made"] for row in rows):
        raise RuntimeError("full-panel execution unexpectedly omitted a recommendation")
    if any(row["result"]["stop_reason"] is not None for row in rows):
        raise RuntimeError("full-panel execution unexpectedly recorded a stop")
    if any(int(row["result"]["n_enrolled"]) != 40 for row in rows):
        raise RuntimeError("full-panel execution unexpectedly enrolled fewer than 40")

    elapsed = time.perf_counter() - started
    raw_path = output_dir / "ckg_aggregate_oc_stability_raw.json"
    raw = {
        "schema_version": 1,
        "status": "post hoc numerical-stability sensitivity",
        "design": design,
        "environment": environment,
        "production_equivalence": production_equivalence,
        "wall_clock_seconds_for_this_invocation_not_for_timing_inference": elapsed,
        "reference_rows": sorted(
            reference_rows, key=lambda row: (row["policy"], row["seed"], row["stratum"])
        ),
        "new_arm_rows": sorted(
            rows, key=lambda row: (row["arm"], row["seed"], row["stratum"])
        ),
        "source_sha256": sources,
    }
    atomic_json_write(raw_path, raw)
    counts = {
        "reference_policy_rows": len(reference_rows),
        "new_arm_rows": len(rows),
        "independent_trial_seeds": int(args.seed_count),
        "strata": len(STRATA),
        "production_equivalence_cells": production_equivalence[
            "archived_seed_stratum_cells_checked"
        ],
    }
    metadata_path = raw_path.with_suffix(raw_path.suffix + ".metadata.json")
    atomic_json_write(
        metadata_path,
        _metadata(raw_path, design, environment, sources, counts),
    )
    print(f"[done] raw={raw_path}", flush=True)
    print(f"[done] raw_sha256={sha256_file(raw_path)}", flush=True)
    print(f"[done] metadata={metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
