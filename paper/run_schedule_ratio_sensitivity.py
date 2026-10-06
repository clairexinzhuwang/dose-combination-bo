#!/usr/bin/env python3
"""Run the post-hoc 2:1 cEI--tMSE schedule sensitivity.

This additive runner creates only the genuinely new schedule arm requested during
post-hoc review.  It does not add the schedule to the package's built-in policy
registry or change any trial default.  The repeating adaptive-cohort cycle is
fixed as ``[cEI, cEI, tMSE]`` before this run; it starts at adaptive step zero.

The complete descriptive matrix is OSA, latent gate, full-panel access,
``tau=0.7``, two strata, and seed labels 0--99 (200 trial records).  Existing cEI,
cKG, pure-tMSE, and 1:1 cEI--tMSE arms are reused only by the companion analyzer.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

# One independent trial per worker; do not multiply BLAS thread pools.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import dose_combination_bo as sdb  # noqa: E402


POLICY = "cEI-tMSE-2to1"
SCHEDULE_CYCLE = ("cEI", "cEI", "tMSE")
STRATA = (0, 1)
MAX_SEEDS = 100
FROZEN = {
    "sim": "osa",
    "gamma": 0.7,
    "mode": "latent",
    "noise": "fixed",
    "kap": 1.0,
    "budget": 40,
    "warmup": 4,
    "r_k": 2,
    "grid_n": 5,
    "empty_gate": "pf",
    "protocol_scaffold": "lhs_fixed",
    "region_step": 0.25,
    "empty_gate_stop_after": 3,
    "exclude_repeats_during_expansion": True,
}
POST_HOC_FIELDS = {
    "post_hoc_descriptive": True,
    "schedule_tuned": False,
    "schedule_ratio": "2:1 cEI:tMSE",
    "schedule_cycle": list(SCHEDULE_CYCLE),
    "schedule_phase_origin": "first adaptive cohort",
    "schedule_review_status": (
        "fixed before this post-hoc review run after the original study results were known"
    ),
}
GENERATION_RUNNER_SHA256 = (
    "a729767a5d705c426bb9cbe42f93ff292d121273cdb85a1e59941a4247b65c59"
)
GENERATION_RUNNER_GZIP_SHA256 = (
    "2570b76275b9131577b35c0350bd4dcd7cda046646583c2db5dc18fd483b2539"
)


@sdb.acquisition(POLICY)
def _cei2_tmse1(ctx):
    """Two gated cEI cohorts followed by one gated tMSE cohort, repeated."""
    if ctx.step % len(SCHEDULE_CYCLE) in (0, 1):
        return ctx.restrict(ctx.cei_scores())
    return ctx.restrict(ctx.tmse)


def _jobs(seed_count):
    return [
        dict(policy=POLICY, seed=seed, z=stratum, **FROZEN)
        for stratum in STRATA
        for seed in range(seed_count)
    ]


def _job_key(job):
    return int(job["z"]), int(job["seed"])


def _record_key(record):
    return int(record["stratum"]), int(record["seed"])


def _run(job):
    record = sdb.run_trial(**job)
    record.update(POST_HOC_FIELDS)
    return record


def _atomic_json(path, payload, *, indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=indent,
                sort_keys=indent is not None,
                separators=(",", ":") if indent is None else None,
                allow_nan=False,
            )
            if indent is not None:
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_record(record, expected):
    key = _record_key(record)
    if key not in expected:
        raise ValueError(f"record contains an unexpected design cell: {key}")
    if record.get("policy") != POLICY:
        raise ValueError(f"record {key} has unexpected policy {record.get('policy')!r}")
    for field, value in FROZEN.items():
        record_field = "stratum" if field == "z" else field
        actual = record.get(record_field)
        if actual != value:
            raise ValueError(
                f"record {key} has {record_field}={actual!r}; expected {value!r}"
            )
    for field, value in POST_HOC_FIELDS.items():
        if record.get(field) != value:
            raise ValueError(f"record {key} is missing exact post-hoc field {field!r}")
    if record.get("recommendation_made") is not True:
        raise ValueError(f"full-panel record {key} did not make a recommendation")
    if record.get("n_enrolled") != FROZEN["budget"]:
        raise ValueError(f"full-panel record {key} did not reach the fixed budget")
    return key


def _metadata(rows, output, *, workers, pending_count, elapsed_seconds):
    metadata = sdb.artifact_metadata(
        rows, source_path=output, artifact_type="post_hoc_schedule_raw_trial_records"
    )
    metadata.update({
        "post_hoc_descriptive": True,
        "confirmatory": False,
        "ratio_tuned": False,
        "new_policy": POLICY,
        "schedule_cycle": list(SCHEDULE_CYCLE),
        "schedule_phase_origin": "first adaptive cohort",
        "adaptive_cohorts": 18,
        "cycle_counts_over_trial": {"cEI": 12, "tMSE": 6},
        "post_hoc_timing_status": (
            "The 2:1 cycle and specified post-hoc analysis targets were fixed before "
            "this post-hoc review run after the original study results were known; "
            "this is not a preregistered or confirmatory analysis."
        ),
        "outcome_payload_sha256": sdb.records_sha256([
            {key: value for key, value in row.items() if key not in POST_HOC_FIELDS}
            for row in rows
        ]),
        "provenance": {
            "record_origin": "genuinely new simulation rows",
            "existing_policy_rows_included": False,
            "companion_analyzer_reuses_frozen_sixway_rows": True,
            "generation_runner": (
                "schedule_ratio_analysis/provenance/"
                "run_schedule_ratio_sensitivity.generation.py.gz"
            ),
            "generation_runner_sha256": GENERATION_RUNNER_SHA256,
            "generation_runner_gzip_sha256": GENERATION_RUNNER_GZIP_SHA256,
            "generation_runner_hash_scope": (
                "generation_runner_sha256 is the SHA-256 of the exact decompressed "
                "generation source; generation_runner_gzip_sha256 hashes the stored "
                "gzip provenance artifact"
            ),
            "historical_generation_source_note": (
                "The compressed source is the exact code used for numerical generation "
                "and therefore predates the disclosure-only wording correction. It is "
                "retained solely as non-reader-facing provenance."
            ),
            "current_runner": Path(__file__).name,
            "current_runner_sha256": sdb.file_sha256(Path(__file__)),
            "core_api_or_default_policy_changed": False,
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": __import__("numpy").__version__,
            "scipy": __import__("scipy").__version__,
            "torch": __import__("torch").__version__,
            "gpytorch": __import__("gpytorch").__version__,
            "dose-combination-bo": sdb.__version__,
        },
        "execution": {
            "workers": int(workers),
            "new_cells_computed_this_invocation": int(pending_count),
            "elapsed_wall_seconds_this_invocation": float(elapsed_seconds),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
    })
    return metadata


def run(output, *, seed_count=MAX_SEEDS, workers=None, checkpoint_every=10):
    if (
        not isinstance(seed_count, int)
        or isinstance(seed_count, bool)
        or not 1 <= seed_count <= MAX_SEEDS
    ):
        raise ValueError(f"seed_count must be an integer from 1 to {MAX_SEEDS}")
    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if (
        not isinstance(checkpoint_every, int)
        or isinstance(checkpoint_every, bool)
        or checkpoint_every < 1
    ):
        raise ValueError("checkpoint_every must be a positive integer")

    jobs = _jobs(seed_count)
    expected = {_job_key(job) for job in jobs}
    if len(expected) != len(jobs):
        raise AssertionError("post-hoc job matrix contains duplicates")

    output = Path(output)
    checkpoint = output.with_suffix(output.suffix + ".partial")
    cached = []
    if checkpoint.exists() and checkpoint.stat().st_size:
        cached = json.loads(checkpoint.read_text(encoding="utf-8"))
    elif output.exists() and output.stat().st_size:
        cached = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(cached, list):
        raise ValueError("cached result file must contain a JSON list")

    seen = set()
    for record in cached:
        key = _validate_record(record, expected)
        if key in seen:
            raise ValueError(f"checkpoint contains duplicate design cell: {key}")
        seen.add(key)
    pending = [job for job in jobs if _job_key(job) not in seen]
    rows = list(cached)

    print(
        f"post-hoc 2:1 matrix: {len(jobs)} cells; {len(rows)} cached; "
        f"{len(pending)} pending",
        flush=True,
    )
    print(f"workers: {workers}; checkpoint: {checkpoint}", flush=True)
    started = time.perf_counter()
    if pending:
        completed = 0

        def accept(record):
            nonlocal completed
            key = _validate_record(record, expected)
            if key in seen:
                raise AssertionError(f"worker returned duplicate cell: {key}")
            seen.add(key)
            rows.append(record)
            completed += 1
            if completed % checkpoint_every == 0 or completed == len(pending):
                rows.sort(key=_record_key)
                _atomic_json(checkpoint, rows)
            if completed % 25 == 0 or completed == len(pending):
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed else float("inf")
                eta = (len(pending) - completed) / rate if rate else float("nan")
                print(
                    f"{len(rows)}/{len(jobs)} complete; elapsed "
                    f"{elapsed / 60:.1f} min; ETA {eta / 60:.1f} min",
                    flush=True,
                )

        futures = {}
        try:
            if workers == 1:
                # A true serial path is required in restricted environments where
                # constructing even a one-worker process pool is disallowed.
                for job in pending:
                    accept(_run(job))
            else:
                context = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                    futures = {pool.submit(_run, job): _job_key(job) for job in pending}
                    for future in as_completed(futures):
                        accept(future.result())
        except BaseException:
            rows.sort(key=_record_key)
            _atomic_json(checkpoint, rows)
            for future in futures:
                future.cancel()
            raise

    elapsed = time.perf_counter() - started
    if seen != expected:
        raise AssertionError(f"run ended with {len(expected - seen)} missing cells")
    rows.sort(key=_record_key)
    _atomic_json(output, rows)
    metadata = _metadata(
        rows,
        output,
        workers=workers,
        pending_count=len(pending),
        elapsed_seconds=elapsed,
    )
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    previous_metadata = None
    if metadata_path.exists():
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not pending and previous_metadata is not None:
        # Preserve the actual numerical-run timing when refreshing only disclosure
        # wording and hashes after all 200 outcome rows already exist.
        metadata["execution"] = previous_metadata["execution"]
        prior_rewrite = previous_metadata.get("metadata_only_rewrite", {})
        metadata["metadata_only_rewrite"] = {
            "reason": (
                "Replaced ambiguous review-timing wording after numerical completion; "
                "no outcome, allocation, recommendation, or design field changed."
            ),
            "outcomes_recomputed": False,
            "fields_changed": ["schedule_review_status", "post_hoc_timing_status"],
            "outcome_payload_sha256_before": prior_rewrite.get(
                "outcome_payload_sha256_before",
                previous_metadata.get(
                    "outcome_payload_sha256",
                    "40f08a2a1f4c8e013f59b0923eb75fcdd26baf3608c9eda895df2870a8d86822",
                ),
            ),
            "outcome_payload_sha256_after": metadata["outcome_payload_sha256"],
            "pre_rewrite_raw_sha256": prior_rewrite.get(
                "pre_rewrite_raw_sha256",
                previous_metadata.get(
                    "source_sha256",
                    "bcfaa149d3a86aff048745b81ac87f82874c3be30e6d3ce4df4b7b02f358595a",
                ),
            ),
        }
    _atomic_json(metadata_path, metadata, indent=2)
    if checkpoint.exists():
        checkpoint.unlink()
    print(f"complete: {output}", flush=True)
    print(f"raw SHA-256: {sdb.file_sha256(output)}", flush=True)
    print(f"metadata: {metadata_path}", flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="final raw JSON path")
    parser.add_argument(
        "--seeds",
        type=int,
        default=MAX_SEEDS,
        help="leading seed count; the complete descriptive analysis requires 100",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args(argv)
    run(
        args.out,
        seed_count=args.seeds,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
