#!/usr/bin/env python3
"""Run the frozen 4,800-record-row OSA dose-access sensitivity.

The design constants in this file implement the frozen prespecification in
``docs/protocol_scaffold_sensitivity_prespec.md``. The only operational options
are the output location, worker count, and number of leading seeds (the latter is
for invariant smoke tests; manuscript results require exactly 200). The archived
analysis reused 2,400 corrected main-sweep rows and added 2,400 gradual-access rows;
a fresh invocation of this runner recomputes every requested factorial cell.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time

# Prevent each independent trial worker from starting its own BLAS thread pool.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from dose_combination_bo import run_trial  # noqa: E402
from dose_combination_bo.reporting import write_artifact_metadata  # noqa: E402


POLICIES = ("cEI", "cKG", "cEI-tMSE")
SCAFFOLDS = ("lhs_fixed", "start_low_expansion")
GAMMAS = (0.7, 0.9)
STRATA = (0, 1)

FROZEN = {
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


def _job_key(job):
    return (
        job["protocol_scaffold"], float(job["gamma"]), int(job["z"]),
        job["policy"], int(job["seed"]),
    )


def _jobs(seed_count):
    return [
        dict(
            policy=policy,
            seed=seed,
            z=stratum,
            gamma=gamma,
            protocol_scaffold=scaffold,
            **FROZEN,
        )
        for scaffold in SCAFFOLDS
        for gamma in GAMMAS
        for stratum in STRATA
        for policy in POLICIES
        for seed in range(seed_count)
    ]


def _run(job):
    return run_trial(**job)


def _record_key(record):
    return (
        record.get("protocol_scaffold", "lhs_fixed"), float(record["gamma"]),
        int(record["stratum"]), record["policy"], int(record["seed"]),
    )


def _atomic_json(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(rows, stream, separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_cached(record, expected_keys):
    key = _record_key(record)
    if key not in expected_keys:
        raise ValueError(f"checkpoint contains an unexpected design cell: {key}")
    for field, expected in FROZEN.items():
        actual = record.get(field, 1.0 if field == "kap" else None)
        if actual != expected:
            raise ValueError(
                f"checkpoint cell {key} has {field}={actual!r}; expected {expected!r}"
            )
    if record.get("recommendation_made") is None:
        raise ValueError("checkpoint predates stopping-aware trial records and cannot be resumed")
    return key


def run(output, *, seed_count=200, workers=None, checkpoint_every=12):
    if (
        not isinstance(seed_count, int)
        or isinstance(seed_count, bool)
        or not 1 <= seed_count <= 200
    ):
        raise ValueError("seed_count must be an integer from 1 to 200")
    jobs = _jobs(seed_count)
    expected = {_job_key(job) for job in jobs}
    if len(expected) != len(jobs):
        raise AssertionError("frozen job matrix contains duplicates")
    output = Path(output)
    checkpoint = output.with_suffix(output.suffix + ".partial")
    cached = []
    if checkpoint.exists() and checkpoint.stat().st_size:
        cached = json.loads(checkpoint.read_text(encoding="utf-8"))
    elif output.exists() and output.stat().st_size:
        cached = json.loads(output.read_text(encoding="utf-8"))
    seen = set()
    for record in cached:
        key = _validate_cached(record, expected)
        if key in seen:
            raise ValueError(f"checkpoint contains duplicate design cell: {key}")
        seen.add(key)
    pending = [job for job in jobs if _job_key(job) not in seen]
    rows = list(cached)
    workers = min(8, os.cpu_count() or 1) if workers is None else workers
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if (
        not isinstance(checkpoint_every, int)
        or isinstance(checkpoint_every, bool)
        or checkpoint_every < 1
    ):
        raise ValueError("checkpoint_every must be a positive integer")
    print(f"frozen matrix: {len(jobs)} record cells; {len(rows)} cached; {len(pending)} pending")
    print(f"workers: {workers}; checkpoint: {checkpoint}", flush=True)
    started = time.time()
    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            job_iter = iter(pending)
            futures = {}
            for job in job_iter:
                futures[pool.submit(_run, job)] = _job_key(job)
                if len(futures) >= 2 * workers:
                    break
            completed = 0
            while futures:
                future = next(as_completed(futures))
                futures.pop(future)
                try:
                    record = future.result()
                except BaseException:
                    rows.sort(key=_record_key)
                    _atomic_json(checkpoint, rows)
                    for outstanding in futures:
                        outstanding.cancel()
                    raise
                completed += 1
                key = _validate_cached(record, expected)
                if key in seen:
                    raise AssertionError(f"worker returned duplicate cell: {key}")
                seen.add(key)
                rows.append(record)
                if completed % checkpoint_every == 0 or completed == len(pending):
                    rows.sort(key=_record_key)
                    _atomic_json(checkpoint, rows)
                if completed % 25 == 0 or completed == len(pending):
                    elapsed = time.time() - started
                    rate = completed / elapsed
                    eta = (len(pending) - completed) / rate if rate else float("nan")
                    print(
                        f"{len(rows)}/{len(jobs)} complete; elapsed {elapsed / 60:.1f} min; "
                        f"ETA {eta / 60:.1f} min",
                        flush=True,
                    )
                try:
                    job = next(job_iter)
                except StopIteration:
                    pass
                else:
                    futures[pool.submit(_run, job)] = _job_key(job)
    if seen != expected:
        raise AssertionError(f"run ended with {len(expected - seen)} missing cells")
    rows.sort(key=_record_key)
    _atomic_json(output, rows)
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    write_artifact_metadata(
        metadata_path, rows, source_path=output, artifact_type="raw_trial_records"
    )
    if checkpoint.exists():
        checkpoint.unlink()
    print(f"complete: {output}")
    print(f"metadata: {metadata_path}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="final raw JSON path")
    parser.add_argument("--seeds", type=int, default=200,
                        help="leading seed count; manuscript analysis requires 200")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=12)
    args = parser.parse_args(argv)
    run(
        args.out,
        seed_count=args.seeds,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
