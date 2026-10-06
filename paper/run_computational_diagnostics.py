#!/usr/bin/env python3
"""Post hoc computational-cost and cKG fantasy-count diagnostics.

This is deliberately additive: it does not modify ``run_trial`` or any built-in
acquisition.  It installs process-local diagnostic wrappers, runs one interleaved
CPU process, and restores the acquisition registry before exit.

Frozen design
-------------
* OSA, full-panel access, latent gate, fixed noise, tau=0.7, both strata.
* Timing: seeds 0--9 for cEI, tMSE, cEI--tMSE, and nested cKG with
  128/256/512 fantasies.
* Stability: seeds 0--19 for the three cKG fantasy counts.
* OPENBLAS/OMP/MKL threads are pinned to one before numerical libraries load.

The cKG variants draw the production-compatible 512-sample efficacy and toxicity
banks first, then take channel-wise prefixes.  The 512 variant is therefore
numerically identical to the package's current cKG implementation, while 128 and
256 are genuinely nested approximations rather than differently aligned RNG
streams.  All results are descriptive and post hoc.
"""

from __future__ import annotations

import os

# These must be set before importing numpy/scipy/torch.
for _thread_env in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gpytorch
import numpy as np
import scipy
import torch
from scipy.stats import norm

from dose_combination_bo.acquisitions import ckg_one_step_gated
from dose_combination_bo.gp import joint, post_latent
from dose_combination_bo.registry import ACQUISITIONS, get_acquisition
from dose_combination_bo.trial import run_trial


TAU = 0.7
TIMING_SEEDS = tuple(range(10))
STABILITY_SEEDS = tuple(range(20))
STRATA = (0, 1)
MAX_FANTASIES = 512
CKG_COUNTS = (128, 256, 512)
WARMUP = 4
COHORT_SIZE = 2
BUDGET = 40
GRID_N = 5

TIMING_VARIANTS = (
    "cEI",
    "tMSE",
    "cEI-tMSE",
    "cKG-128",
    "cKG-256",
    "cKG-512",
)
CKG_VARIANTS = ("cKG-128", "cKG-256", "cKG-512")
DIAGNOSTIC_POLICY_NAMES = {
    "cEI": "__diag_cEI",
    "tMSE": "__diag_tMSE",
    "cEI-tMSE": "__diag_cEI_tMSE",
    "cKG-128": "__diag_cKG_128",
    "cKG-256": "__diag_cKG_256",
    "cKG-512": "__diag_cKG_512",
    "same-state-512-path": "__diag_same_state_512_path",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fantasy_bank(seed: int = 0, max_fantasies: int = MAX_FANTASIES) -> tuple[np.ndarray, np.ndarray]:
    """Return production-ordered efficacy/toxicity standard-normal banks.

    Production draws all ``Zf`` values first and then all ``Zg`` values.  Taking
    prefixes only after both 512-value channels are built avoids the common error
    in which a smaller ``Zg`` draw starts inside the production ``Zf`` stream.
    """
    if max_fantasies < 1:
        raise ValueError("max_fantasies must be positive")
    rng = np.random.default_rng(int(seed))
    zf = rng.standard_normal((int(max_fantasies), 1))
    zg = rng.standard_normal((int(max_fantasies), 1))
    return zf, zg


def ckg_one_step_gated_nested(
    d: torch.Tensor,
    Xset: torch.Tensor,
    m_eff: object,
    l_eff: object,
    m_tox: object,
    l_tox: object,
    g_dagger: float,
    gamma: float,
    *,
    r_k: int = 1,
    nmc: int = 512,
    seed: int = 0,
    gate_mode: str = "latent",
) -> float:
    """Production cKG with nested, channel-wise prefixes of a 512 bank.

    At ``nmc=512`` the operations and RNG sequence match
    :func:`dose_combination_bo.acquisitions.ckg_one_step_gated` exactly.  A fresh full bank
    is generated for each candidate, as in production; this avoids understating
    the current 512 implementation's acquisition cost.
    """
    if int(nmc) not in CKG_COUNTS:
        raise ValueError(f"nmc must be one of {CKG_COUNTS}")
    mu_f, var_f, covf, s2f_full = joint(m_eff, l_eff, Xset, d)
    mu_g, var_g, covg, s2g_full = joint(m_tox, l_tox, Xset, d)
    nf = l_eff.noise.item()
    ng = l_tox.noise.item()
    s2f = (s2f_full - nf) + nf / r_k
    s2g = (s2g_full - ng) + ng / r_k
    mu_f = mu_f.cpu().numpy()
    mu_g = mu_g.cpu().numpy()
    wf = (covf / s2f).cpu().numpy()
    wg = (covg / s2g).cpu().numpy()
    sf = np.sqrt(s2f)
    sg = np.sqrt(s2g)
    obs_g = ng if gate_mode == "predictive" else 0.0
    resid_g = np.maximum((var_g - covg**2 / s2g).cpu().numpy(), 1e-12)
    sd_g_post = np.sqrt(resid_g + obs_g)
    _, vgc = post_latent(m_tox, Xset)
    sd_g_now = (vgc.clamp_min(1e-12).cpu().numpy() + obs_g) ** 0.5
    pf_now = norm.cdf((g_dagger - mu_g) / sd_g_now)
    gate_now = pf_now > gamma
    V_now = mu_f[gate_now].max() if gate_now.any() else mu_f[pf_now.argmax()]

    zf_bank, zg_bank = fantasy_bank(seed=seed, max_fantasies=MAX_FANTASIES)
    zf = zf_bank[: int(nmc)]
    zg = zg_bank[: int(nmc)]
    muf2 = mu_f[None, :] + wf[None, :] * sf * zf
    mug2 = mu_g[None, :] + wg[None, :] * sg * zg
    pf2 = norm.cdf((g_dagger - mug2) / sd_g_post[None, :])
    gate2 = pf2 > gamma
    vp = np.where(gate2, muf2, -np.inf).max(axis=1)
    bad = ~np.isfinite(vp)
    if bad.any():
        vp[bad] = muf2[bad][np.arange(int(bad.sum())), pf2[bad].argmax(axis=1)]
    return float(vp.mean() - V_now)


def nested_ckg_scores(ctx: object, nmc: int) -> np.ndarray:
    values = np.array(
        [
            ckg_one_step_gated_nested(
                ctx.Xset[i],
                ctx.Xset,
                ctx.me,
                ctx.le,
                ctx.mt,
                ctx.lt,
                ctx.g_dagger,
                ctx.gamma,
                r_k=ctx.r_k,
                nmc=nmc,
                seed=ctx.ckg_seed,
                gate_mode=ctx.gate_mode,
            )
            for i in range(ctx.Xset.shape[0])
        ]
    )
    return ctx.restrict(values)


def selected_index(ctx: object, values: np.ndarray) -> int:
    values = np.asarray(values, dtype=float)
    values = np.where(ctx.candidate_mask, values, -np.inf)
    return int(ctx.most_feasible_index if not np.isfinite(values).any() else values.argmax())


@dataclass
class ActiveRun:
    acquisition_calls: list[dict[str, Any]] | None = None
    same_state_calls: list[dict[str, Any]] | None = None
    seed: int | None = None
    stratum: int | None = None


@contextlib.contextmanager
def installed_diagnostic_acquisitions(active: ActiveRun):
    """Install process-local wrappers and restore the registry on exit."""
    originals = {name: ACQUISITIONS.get(name) for name in DIAGNOSTIC_POLICY_NAMES.values()}
    base = {
        "cEI": get_acquisition("cEI"),
        "tMSE": get_acquisition("tmse"),
        "cEI-tMSE": get_acquisition("cEI-tMSE"),
    }

    def timed(label: str, fn: Callable[[object], np.ndarray]) -> Callable[[object], np.ndarray]:
        def wrapper(ctx: object) -> np.ndarray:
            started = time.perf_counter_ns()
            values = np.asarray(fn(ctx), dtype=float)
            elapsed_ns = time.perf_counter_ns() - started
            if active.acquisition_calls is not None:
                active.acquisition_calls.append(
                    {
                        "variant": label,
                        "step": int(ctx.step),
                        "seconds": elapsed_ns / 1e9,
                        "selected_index": selected_index(ctx, values),
                    }
                )
            return values

        wrapper.acquisition_name = label  # type: ignore[attr-defined]
        return wrapper

    wrappers: dict[str, Callable[[object], np.ndarray]] = {
        DIAGNOSTIC_POLICY_NAMES["cEI"]: timed("cEI", base["cEI"]),
        DIAGNOSTIC_POLICY_NAMES["tMSE"]: timed("tMSE", base["tMSE"]),
        DIAGNOSTIC_POLICY_NAMES["cEI-tMSE"]: timed("cEI-tMSE", base["cEI-tMSE"]),
    }
    for nmc in CKG_COUNTS:
        label = f"cKG-{nmc}"
        wrappers[DIAGNOSTIC_POLICY_NAMES[label]] = timed(
            label, lambda ctx, count=nmc: nested_ckg_scores(ctx, count)
        )

    def same_state_512_path(ctx: object) -> np.ndarray:
        score_map = {count: nested_ckg_scores(ctx, count) for count in CKG_COUNTS}
        index_map = {count: selected_index(ctx, score_map[count]) for count in CKG_COUNTS}
        if active.same_state_calls is not None:
            active.same_state_calls.append(
                {
                    "seed": int(active.seed),
                    "stratum": int(active.stratum),
                    "step": int(ctx.step),
                    "selected_index": {str(k): int(v) for k, v in index_map.items()},
                    "selected_dose": {
                        str(k): [float(x) for x in ctx.Xset[v].detach().cpu().numpy()]
                        for k, v in index_map.items()
                    },
                    "gate_empty": bool(ctx.gate_empty),
                }
            )
        return score_map[512]

    same_state_512_path.acquisition_name = "same-state-512-path"  # type: ignore[attr-defined]
    wrappers[DIAGNOSTIC_POLICY_NAMES["same-state-512-path"]] = same_state_512_path

    try:
        ACQUISITIONS.update(wrappers)
        yield
    finally:
        for name, previous in originals.items():
            if previous is None:
                ACQUISITIONS.pop(name, None)
            else:
                ACQUISITIONS[name] = previous


def configure_single_thread_runtime() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Importing this module inside an already-active pytest process can make
        # the interop setting immutable.  The standalone diagnostic sets it first.
        pass


def _hardware_record() -> dict[str, Any]:
    """Return useful hardware fields without persistent device identifiers."""
    record: dict[str, Any] = {"logical_cpu_count": os.cpu_count()}
    if sys.platform == "darwin":
        try:
            cpu_model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if cpu_model:
                record["cpu_model"] = cpu_model
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            report = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            allowed = {
                "Model Name": "model_name",
                "Model Identifier": "model_identifier",
                "Model Number": "model_number",
                "Chip": "chip",
                "Total Number of Cores": "core_description",
                "Memory": "memory",
            }
            for line in report.splitlines():
                key, separator, value = line.strip().partition(":")
                if separator and key in allowed:
                    record[allowed[key]] = value.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        if "cpu_model" not in record and "chip" in record:
            record["cpu_model"] = record["chip"]
        return record
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                record["cpu_model"] = line.split(":", 1)[1].strip()
                break
    if "cpu_model" not in record:
        record["cpu_model"] = platform.processor() or "unknown"
    return record


def _numpy_configuration_record() -> dict[str, Any]:
    """Select reproducibility-relevant NumPy fields while omitting build paths."""
    config = getattr(np.__config__, "CONFIG", {})
    compilers = config.get("Compilers", {})
    compiler_record = {
        language: {
            key: details[key]
            for key in ("name", "version", "linker")
            if key in details
        }
        for language, details in compilers.items()
        if isinstance(details, dict)
    }
    dependencies = config.get("Build Dependencies", {})
    dependency_record = {}
    for library in ("blas", "lapack"):
        details = dependencies.get(library, {})
        if isinstance(details, dict):
            dependency_record[library] = {
                key: details[key]
                for key in ("name", "found", "version", "openblas configuration")
                if key in details
            }
    machine = config.get("Machine Information", {}).get("host", {})
    simd = config.get("SIMD Extensions", {})
    return {
        "compilers": compiler_record,
        "build_dependencies": dependency_record,
        "host_machine": {
            key: machine[key]
            for key in ("cpu", "family", "endian", "system")
            if key in machine
        },
        "simd_extensions": {
            key: simd[key] for key in ("baseline", "found") if key in simd
        },
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_record() -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware": _hardware_record(),
        "python": sys.version,
        "python_executable_name": Path(sys.executable).name,
        "versions": {
            "dose-combination-bo": _distribution_version("dose-combination-bo") or "source-tree",
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
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "numpy_configuration": _numpy_configuration_record(),
    }


def frozen_design() -> dict[str, Any]:
    return {
        "status": "post hoc and descriptive",
        "surface": "osa",
        "protocol_scaffold": "lhs_fixed",
        "gate_mode": "latent",
        "noise": "fixed",
        "tau": TAU,
        "strata": list(STRATA),
        "budget": BUDGET,
        "warmup": WARMUP,
        "cohort_size": COHORT_SIZE,
        "grid_n": GRID_N,
        "empty_gate": "pf",
        "timing_seeds": list(TIMING_SEEDS),
        "stability_seeds": list(STABILITY_SEEDS),
        "timing_variants": list(TIMING_VARIANTS),
        "ckg_fantasy_counts": list(CKG_COUNTS),
        "fantasy_nesting": (
            "Generate production-ordered Zf[0:512] and Zg[0:512] banks with seed 0; "
            "use channel-wise prefixes for 128 and 256."
        ),
        "timing_order": (
            "One process; seed then stratum; cyclically rotate the six variants within "
            "each seed-stratum cell."
        ),
        "acquisition_timing_boundary": (
            "Time only the registered acquisition callable; Context construction, GP fitting, "
            "allocation, final refits, and metrics are excluded."
        ),
        "total_timing_boundary": "Entire run_trial call, including final refits and metrics.",
        "stratum_clustering": (
            "Average the two strata within seed before timing summaries, ratios, and agreement summaries."
        ),
    }


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "recommendation_made",
        "n_enrolled",
        "allocation_history",
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
    return {key: result[key] for key in keys}


def run_cell(
    active: ActiveRun,
    variant: str,
    seed: int,
    stratum: int,
    *,
    phase: str,
    run_order: int,
    budget: int = BUDGET,
) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    active.acquisition_calls = calls
    active.seed = int(seed)
    active.stratum = int(stratum)
    started = time.perf_counter_ns()
    try:
        result = run_trial(
            DIAGNOSTIC_POLICY_NAMES[variant],
            seed=int(seed),
            z=int(stratum),
            gamma=TAU,
            sim="osa",
            mode="latent",
            noise="fixed",
            traj=False,
            budget=int(budget),
            warmup=WARMUP,
            r_k=COHORT_SIZE,
            grid_n=GRID_N,
            empty_gate="pf",
            protocol_scaffold="lhs_fixed",
        )
    finally:
        total_ns = time.perf_counter_ns() - started
        active.acquisition_calls = None
    return {
        "phase": phase,
        "variant": variant,
        "fantasy_count": int(variant.split("-")[-1]) if variant.startswith("cKG-") else None,
        "seed": int(seed),
        "stratum": int(stratum),
        "run_order": int(run_order),
        "total_trial_seconds": total_ns / 1e9,
        "acquisition_seconds": float(sum(row["seconds"] for row in calls)),
        "acquisition_call_count": len(calls),
        "acquisition_calls": calls,
        "result": _compact_result(result),
    }


def _same_result(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = (
        "allocation_history",
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
    return all(a[key] == b[key] for key in keys)


def production_compatibility_preflight(active: ActiveRun) -> dict[str, Any]:
    """Fail before the long run if nested cKG-512 changes a short trial path."""
    common = dict(
        seed=0,
        z=0,
        gamma=TAU,
        sim="osa",
        mode="latent",
        noise="fixed",
        budget=8,
        warmup=WARMUP,
        r_k=COHORT_SIZE,
        grid_n=GRID_N,
        empty_gate="pf",
        protocol_scaffold="lhs_fixed",
    )
    production = run_trial("cKG", **common)
    diagnostic = run_cell(
        active, "cKG-512", seed=0, stratum=0, phase="preflight", run_order=-1, budget=8
    )
    matched = _same_result(_compact_result(production), diagnostic["result"])
    if not matched:
        raise RuntimeError("nested cKG-512 failed the production compatibility preflight")
    return {
        "budget": 8,
        "seed": 0,
        "stratum": 0,
        "full_path_and_terminal_match": True,
        "note": "Production cKG and nested cKG-512 were identical on all audited fields.",
    }


def warm_numerical_stack(active: ActiveRun) -> None:
    """Untimed one-cohort warm-up for every diagnostic variant."""
    for variant in TIMING_VARIANTS:
        active.acquisition_calls = None
        run_trial(
            DIAGNOSTIC_POLICY_NAMES[variant],
            seed=10000,
            z=0,
            gamma=TAU,
            sim="osa",
            mode="latent",
            noise="fixed",
            budget=6,
            warmup=WARMUP,
            r_k=COHORT_SIZE,
            grid_n=GRID_N,
            empty_gate="pf",
            protocol_scaffold="lhs_fixed",
        )


def _rotated(items: tuple[str, ...], offset: int) -> tuple[str, ...]:
    k = int(offset) % len(items)
    return items[k:] + items[:k]


def run_timing(active: ActiveRun) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    order = 0
    total = len(TIMING_SEEDS) * len(STRATA) * len(TIMING_VARIANTS)
    for seed in TIMING_SEEDS:
        for stratum in STRATA:
            for variant in _rotated(TIMING_VARIANTS, 2 * seed + stratum):
                order += 1
                print(
                    f"[timing {order:03d}/{total}] seed={seed} z={stratum} {variant}",
                    flush=True,
                )
                rows.append(
                    run_cell(
                        active,
                        variant,
                        seed,
                        stratum,
                        phase="clean_timing",
                        run_order=order,
                    )
                )
    return rows


def run_stability_extras(
    active: ActiveRun, timing_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = [row for row in timing_rows if row["variant"] in CKG_VARIANTS]
    order = 0
    extra_seeds = tuple(seed for seed in STABILITY_SEEDS if seed not in TIMING_SEEDS)
    total = len(extra_seeds) * len(STRATA) * len(CKG_VARIANTS)
    for seed in extra_seeds:
        for stratum in STRATA:
            for variant in _rotated(CKG_VARIANTS, 2 * seed + stratum):
                order += 1
                print(
                    f"[stability {order:03d}/{total}] seed={seed} z={stratum} {variant}",
                    flush=True,
                )
                rows.append(
                    run_cell(
                        active,
                        variant,
                        seed,
                        stratum,
                        phase="stability_only",
                        run_order=order,
                    )
                )
    return rows


def run_same_state_reference_paths(
    active: ActiveRun, stability_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    reference_runs: list[dict[str, Any]] = []
    reference = {
        (row["seed"], row["stratum"]): row
        for row in stability_rows
        if row["variant"] == "cKG-512"
    }
    total = len(STABILITY_SEEDS) * len(STRATA)
    order = 0
    for seed in STABILITY_SEEDS:
        for stratum in STRATA:
            order += 1
            print(f"[same-state {order:03d}/{total}] seed={seed} z={stratum}", flush=True)
            active.seed = int(seed)
            active.stratum = int(stratum)
            local: list[dict[str, Any]] = []
            active.same_state_calls = local
            started = time.perf_counter_ns()
            try:
                result = run_trial(
                    DIAGNOSTIC_POLICY_NAMES["same-state-512-path"],
                    seed=seed,
                    z=stratum,
                    gamma=TAU,
                    sim="osa",
                    mode="latent",
                    noise="fixed",
                    budget=BUDGET,
                    warmup=WARMUP,
                    r_k=COHORT_SIZE,
                    grid_n=GRID_N,
                    empty_gate="pf",
                    protocol_scaffold="lhs_fixed",
                )
            finally:
                elapsed = (time.perf_counter_ns() - started) / 1e9
                active.same_state_calls = None
            compact = _compact_result(result)
            if not _same_result(reference[(seed, stratum)]["result"], compact):
                raise RuntimeError(
                    f"same-state 512 reference path drifted for seed={seed}, stratum={stratum}"
                )
            decisions.extend(local)
            reference_runs.append(
                {
                    "seed": seed,
                    "stratum": stratum,
                    "total_seconds_not_for_timing": elapsed,
                    "result": compact,
                }
            )
    return decisions, reference_runs


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary values must be nonempty and finite")
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25, method="linear")),
        "q3": float(np.quantile(array, 0.75, method="linear")),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_timing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row["phase"] != "clean_timing":
            continue
        by_variant_seed.setdefault((row["variant"], int(row["seed"])), []).append(row)
    seed_means: dict[str, dict[int, dict[str, float]]] = {}
    for (variant, seed), cells in by_variant_seed.items():
        if {int(cell["stratum"]) for cell in cells} != set(STRATA):
            raise ValueError(f"timing cells incomplete for {variant}, seed {seed}")
        seed_means.setdefault(variant, {})[seed] = {
            "total_trial_seconds": statistics.fmean(cell["total_trial_seconds"] for cell in cells),
            "acquisition_seconds": statistics.fmean(cell["acquisition_seconds"] for cell in cells),
            "acquisition_seconds_per_decision": statistics.fmean(
                cell["acquisition_seconds"] / cell["acquisition_call_count"] for cell in cells
            ),
        }
    if set(seed_means) != set(TIMING_VARIANTS):
        raise ValueError("timing variants are incomplete")
    if any(set(per_seed) != set(TIMING_SEEDS) for per_seed in seed_means.values()):
        raise ValueError("timing seed grid is incomplete")

    result: dict[str, Any] = {
        "unit": "seconds",
        "stratum_handling": "two strata averaged within seed before summaries",
        "variants": {},
    }
    baseline = seed_means["cEI"]
    for variant in TIMING_VARIANTS:
        per_seed = seed_means[variant]
        total = [per_seed[seed]["total_trial_seconds"] for seed in TIMING_SEEDS]
        acquisition = [per_seed[seed]["acquisition_seconds"] for seed in TIMING_SEEDS]
        per_decision = [
            per_seed[seed]["acquisition_seconds_per_decision"] for seed in TIMING_SEEDS
        ]
        total_ratio = [
            per_seed[seed]["total_trial_seconds"] / baseline[seed]["total_trial_seconds"]
            for seed in TIMING_SEEDS
        ]
        acquisition_ratio = [
            per_seed[seed]["acquisition_seconds"] / baseline[seed]["acquisition_seconds"]
            for seed in TIMING_SEEDS
        ]
        result["variants"][variant] = {
            "fantasy_count": int(variant.split("-")[-1]) if variant.startswith("cKG-") else None,
            "total_trial_seconds": _distribution(total),
            "acquisition_seconds_per_trial": _distribution(acquisition),
            "acquisition_seconds_per_decision": _distribution(per_decision),
            "paired_total_ratio_to_cEI": _distribution(total_ratio),
            "paired_acquisition_ratio_to_cEI": _distribution(acquisition_ratio),
        }
    return result


def _adaptive_cohort_path(result: dict[str, Any]) -> list[list[float]]:
    history = result["allocation_history"]
    return [history[index] for index in range(WARMUP, len(history), COHORT_SIZE)]


def _clustered_agreement(per_stratum: dict[tuple[int, int], float]) -> dict[str, Any]:
    per_seed = []
    for seed in STABILITY_SEEDS:
        values = [per_stratum[(seed, stratum)] for stratum in STRATA]
        per_seed.append(statistics.fmean(values))
    array = np.asarray(per_seed, dtype=float)
    return {
        "n_seeds": len(STABILITY_SEEDS),
        "n_seed_stratum_cells": len(per_stratum),
        "mean_pct": float(100 * np.mean(array)),
        "mcse_pct": float(100 * np.std(array, ddof=1) / math.sqrt(array.size)),
        "min_seed_average_pct": float(100 * np.min(array)),
        "max_seed_average_pct": float(100 * np.max(array)),
    }


def summarize_stability(
    rows: list[dict[str, Any]], same_state: list[dict[str, Any]]
) -> dict[str, Any]:
    lookup = {(row["variant"], row["seed"], row["stratum"]): row for row in rows}
    expected = {
        (variant, seed, stratum)
        for variant in CKG_VARIANTS
        for seed in STABILITY_SEEDS
        for stratum in STRATA
    }
    if set(lookup) != expected:
        raise ValueError("stability trial grid is incomplete")

    result: dict[str, Any] = {
        "reference": "cKG-512",
        "stratum_handling": "agreement computed by stratum, then the two strata averaged within seed",
        "comparisons": {},
    }
    for variant in ("cKG-128", "cKG-256"):
        exact_path: dict[tuple[int, int], float] = {}
        cohort_rate: dict[tuple[int, int], float] = {}
        terminal: dict[tuple[int, int], float] = {}
        unsafe: dict[tuple[int, int], float] = {}
        for seed in STABILITY_SEEDS:
            for stratum in STRATA:
                candidate = lookup[(variant, seed, stratum)]["result"]
                reference = lookup[("cKG-512", seed, stratum)]["result"]
                path_a = _adaptive_cohort_path(candidate)
                path_b = _adaptive_cohort_path(reference)
                if len(path_a) != len(path_b):
                    raise ValueError("adaptive paths have unequal lengths")
                matches = [a == b for a, b in zip(path_a, path_b)]
                key = (seed, stratum)
                exact_path[key] = float(all(matches))
                cohort_rate[key] = float(np.mean(matches))
                terminal[key] = float(
                    [candidate["rec_d1"], candidate["rec_d2"]]
                    == [reference["rec_d1"], reference["rec_d2"]]
                )
                unsafe[key] = float(candidate["rec_unsafe"] == reference["rec_unsafe"])
        result["comparisons"][f"{variant}_vs_cKG-512"] = {
            "full_adaptive_path_exact_agreement": _clustered_agreement(exact_path),
            "positionwise_adaptive_cohort_agreement": _clustered_agreement(cohort_rate),
            "terminal_recommendation_agreement": _clustered_agreement(terminal),
            "terminal_unsafe_status_agreement": _clustered_agreement(unsafe),
        }

    by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in same_state:
        by_cell.setdefault((int(row["seed"]), int(row["stratum"])), []).append(row)
    if set(by_cell) != {(seed, stratum) for seed in STABILITY_SEEDS for stratum in STRATA}:
        raise ValueError("same-state decision grid is incomplete")
    for variant, count in (("cKG-128", 128), ("cKG-256", 256)):
        rate: dict[tuple[int, int], float] = {}
        exact: dict[tuple[int, int], float] = {}
        for key, decisions in by_cell.items():
            decisions = sorted(decisions, key=lambda row: row["step"])
            matches = [
                row["selected_index"][str(count)] == row["selected_index"]["512"]
                for row in decisions
            ]
            rate[key] = float(np.mean(matches))
            exact[key] = float(all(matches))
        comparison = result["comparisons"][f"{variant}_vs_cKG-512"]
        comparison["same_state_selected_dose_agreement"] = _clustered_agreement(rate)
        comparison["same_state_all_decisions_exact_agreement"] = _clustered_agreement(exact)
    return result


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _metadata(
    artifact: Path,
    *,
    artifact_type: str,
    design: dict[str, Any],
    environment: dict[str, Any],
    counts: dict[str, int],
) -> dict[str, Any]:
    script = Path(__file__).resolve()
    source_files = [
        script,
        ROOT / "src/dose_combination_bo/acquisitions.py",
        ROOT / "src/dose_combination_bo/trial.py",
        ROOT / "src/dose_combination_bo/gp.py",
    ]
    return {
        "artifact": artifact.name,
        "artifact_type": artifact_type,
        "artifact_sha256": _sha256(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post hoc and descriptive",
        "design": design,
        "environment": environment,
        "record_counts": counts,
        "source_sha256": {path.name: _sha256(path) for path in source_files},
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Computational diagnostics",
        "",
        "These OSA diagnostics are **post hoc and descriptive**. They benchmark one captured",
        "CPU/software environment and do not establish portable wall-clock performance.",
        "",
        "## Timing",
        "",
        "The two strata were averaged within each seed. Values are medians [Q1, Q3] across",
        "10 seed averages; ratios are paired to cEI within seed.",
        "",
        "| Variant | Acquisition s/trial | Total s/trial | Acquisition ratio to cEI | Total ratio to cEI |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in TIMING_VARIANTS:
        row = summary["timing"]["variants"][variant]

        def fmt(item: dict[str, Any]) -> str:
            return f"{item['median']:.4f} [{item['q1']:.4f}, {item['q3']:.4f}]"

        lines.append(
            f"| {variant} | {fmt(row['acquisition_seconds_per_trial'])} | "
            f"{fmt(row['total_trial_seconds'])} | {fmt(row['paired_acquisition_ratio_to_cEI'])} | "
            f"{fmt(row['paired_total_ratio_to_cEI'])} |"
        )
    lines.extend(
        [
            "",
            "Acquisition-only time is the registered score-callable boundary. Total time is the",
            "entire trial, including GP fitting, allocation bookkeeping, final refits, and metrics.",
            "",
            "## Fantasy-count stability",
            "",
            "Agreement uses cKG-512 as the reference and averages the two strata within each of",
            "20 seeds before reporting means and Monte Carlo SEs.",
            "",
            "| Comparison | Same-state selected dose | Full path exact | Cohort-position agreement | Terminal recommendation | Unsafe status |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, row in summary["stability"]["comparisons"].items():

        def pct(metric: str) -> str:
            item = row[metric]
            return f"{item['mean_pct']:.1f}% +/- {item['mcse_pct']:.1f}%"

        lines.append(
            f"| {key.replace('_', ' ')} | {pct('same_state_selected_dose_agreement')} | "
            f"{pct('full_adaptive_path_exact_agreement')} | "
            f"{pct('positionwise_adaptive_cohort_agreement')} | "
            f"{pct('terminal_recommendation_agreement')} | "
            f"{pct('terminal_unsafe_status_agreement')} |"
        )
    lines.extend(
        [
            "",
            "Agreement measures numerical stability, not decision accuracy. Lower fantasy counts",
            "use channel-wise prefixes of the production-compatible 512-sample Zf/Zg banks.",
            "",
            "See `computational_diagnostics_raw.json` and its metadata sidecar for every timing",
            "call, allocation path, same-state choice, and the captured hardware/software record.",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    output_dir: Path,
    *,
    design: dict[str, Any],
    environment: dict[str, Any],
    preflight: dict[str, Any],
    timing_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    same_state: list[dict[str, Any]],
    same_state_runs: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "timing_trials": len(timing_rows),
        "stability_trials": len(stability_rows),
        "same_state_decisions": len(same_state),
        "same_state_reference_trials": len(same_state_runs),
    }
    raw = {
        "schema_version": 1,
        "status": "post hoc and descriptive",
        "design": design,
        "environment": environment,
        "production_compatibility_preflight": preflight,
        "wall_clock_seconds_for_complete_diagnostic": float(elapsed_seconds),
        "timing_trials": timing_rows,
        "stability_trials": stability_rows,
        "same_state_decisions": same_state,
        "same_state_reference_trials": same_state_runs,
    }
    summary = {
        "schema_version": 1,
        "status": "post hoc and descriptive",
        "design": design,
        "environment": environment,
        "production_compatibility_preflight": preflight,
        "timing": summarize_timing(timing_rows),
        "stability": summarize_stability(stability_rows, same_state),
    }

    raw_path = output_dir / "computational_diagnostics_raw.json"
    summary_path = output_dir / "computational_diagnostics_summary.json"
    markdown_path = output_dir / "computational_diagnostics_summary.md"
    _json_dump(raw_path, raw)
    _json_dump(summary_path, summary)
    markdown_path.write_text(render_summary_markdown(summary), encoding="utf-8")

    paths = {"raw": raw_path, "summary": summary_path, "markdown": markdown_path}
    for key, path in paths.items():
        metadata = _metadata(
            path,
            artifact_type=f"computational_diagnostics_{key}",
            design=design,
            environment=environment,
            counts=counts,
        )
        _json_dump(path.with_suffix(path.suffix + ".metadata.json"), metadata)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/computational_diagnostics",
        help="artifact directory (default: results/computational_diagnostics)",
    )
    args = parser.parse_args(argv)
    configure_single_thread_runtime()
    design = frozen_design()
    environment = environment_record()
    started = time.perf_counter()
    active = ActiveRun()
    with installed_diagnostic_acquisitions(active):
        print("[preflight] checking nested cKG-512 against production", flush=True)
        preflight = production_compatibility_preflight(active)
        print("[warm-up] one untimed cohort per variant", flush=True)
        warm_numerical_stack(active)
        timing_rows = run_timing(active)
        stability_rows = run_stability_extras(active, timing_rows)
        same_state, same_state_runs = run_same_state_reference_paths(active, stability_rows)
    elapsed = time.perf_counter() - started
    paths = write_artifacts(
        args.output_dir.resolve(),
        design=design,
        environment=environment,
        preflight=preflight,
        timing_rows=timing_rows,
        stability_rows=stability_rows,
        same_state=same_state,
        same_state_runs=same_state_runs,
        elapsed_seconds=elapsed,
    )
    print(f"[done] elapsed={elapsed:.1f}s", flush=True)
    for key, path in paths.items():
        print(f"[done] {key}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
