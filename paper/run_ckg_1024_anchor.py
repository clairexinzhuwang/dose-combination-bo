#!/usr/bin/env python3
"""Post hoc cKG-1024 numerical-stability anchor.

This additive diagnostic leaves the production cKG-512 implementation and all
package APIs/defaults unchanged.  It first draws the production-ordered
``Zf[0:512]`` and ``Zg[0:512]`` blocks, then appends fresh ``Zf[512:1024]`` and
``Zg[512:1024]`` draws.  Thus the two 512 channel prefixes remain exactly the
production streams while 1024 is a custom prefix-preserving numerical anchor,
not the package's conventional ``nmc=1024`` draw order.

The analysis is post hoc and descriptive.  Agreement measures numerical
stability relative to cKG-512, not decision accuracy or Monte Carlo convergence.
"""

from __future__ import annotations

import os

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
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from scipy.stats import norm

from paper import run_computational_diagnostics as base
from dose_combination_bo.acquisitions import ckg_one_step_gated
from dose_combination_bo.registry import ACQUISITIONS
from dose_combination_bo.trial import run_trial


BASE_FANTASIES = 512
ANCHOR_FANTASIES = 1024
EXTENSION_FANTASIES = ANCHOR_FANTASIES - BASE_FANTASIES
SEEDS = tuple(range(20))
STRATA = (0, 1)
REFERENCE_NAME = "__diag_ckg_512_1024_reference_path"
ANCHOR_NAME = "__diag_ckg_1024_anchor"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extended_fantasy_bank(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return nested 1024 banks whose 512 channel prefixes are production draws."""
    rng = np.random.default_rng(int(seed))
    zf_base = rng.standard_normal((BASE_FANTASIES, 1))
    zg_base = rng.standard_normal((BASE_FANTASIES, 1))
    zf_extension = rng.standard_normal((EXTENSION_FANTASIES, 1))
    zg_extension = rng.standard_normal((EXTENSION_FANTASIES, 1))
    return (
        np.concatenate((zf_base, zf_extension), axis=0),
        np.concatenate((zg_base, zg_extension), axis=0),
    )


def ckg_one_step_gated_anchor_pair(
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
    seed: int = 0,
    gate_mode: str = "latent",
) -> tuple[float, float]:
    """Return nested cKG values for 512 and 1024 from one posterior calculation."""
    mu_f, var_f, covf, s2f_full = base.joint(m_eff, l_eff, Xset, d)
    mu_g, var_g, covg, s2g_full = base.joint(m_tox, l_tox, Xset, d)
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
    _, vgc = base.post_latent(m_tox, Xset)
    sd_g_now = (vgc.clamp_min(1e-12).cpu().numpy() + obs_g) ** 0.5
    pf_now = norm.cdf((g_dagger - mu_g) / sd_g_now)
    gate_now = pf_now > gamma
    current_value = mu_f[gate_now].max() if gate_now.any() else mu_f[pf_now.argmax()]

    zf, zg = extended_fantasy_bank(seed=seed)
    updated_f = mu_f[None, :] + wf[None, :] * sf * zf
    updated_g = mu_g[None, :] + wg[None, :] * sg * zg
    updated_pf = norm.cdf((g_dagger - updated_g) / sd_g_post[None, :])
    updated_gate = updated_pf > gamma
    values = np.where(updated_gate, updated_f, -np.inf).max(axis=1)
    empty = ~np.isfinite(values)
    if empty.any():
        values[empty] = updated_f[empty][
            np.arange(int(empty.sum())), updated_pf[empty].argmax(axis=1)
        ]
    value_512 = float(values[:BASE_FANTASIES].mean() - current_value)
    value_1024 = float(values.mean() - current_value)
    return value_512, value_1024


def anchor_score_pair(ctx: object) -> dict[int, np.ndarray]:
    score_512: list[float] = []
    score_1024: list[float] = []
    for index in range(ctx.Xset.shape[0]):
        value_512, value_1024 = ckg_one_step_gated_anchor_pair(
            ctx.Xset[index],
            ctx.Xset,
            ctx.me,
            ctx.le,
            ctx.mt,
            ctx.lt,
            ctx.g_dagger,
            ctx.gamma,
            r_k=ctx.r_k,
            seed=ctx.ckg_seed,
            gate_mode=ctx.gate_mode,
        )
        score_512.append(value_512)
        score_1024.append(value_1024)
    return {
        512: ctx.restrict(np.asarray(score_512, dtype=float)),
        1024: ctx.restrict(np.asarray(score_1024, dtype=float)),
    }


@dataclass
class ActiveRun:
    decisions: list[dict[str, Any]] | None = None
    seed: int | None = None
    stratum: int | None = None


@contextlib.contextmanager
def installed_anchor_acquisitions(active: ActiveRun):
    originals = {
        REFERENCE_NAME: ACQUISITIONS.get(REFERENCE_NAME),
        ANCHOR_NAME: ACQUISITIONS.get(ANCHOR_NAME),
    }

    def anchor_1024(ctx: object) -> np.ndarray:
        return anchor_score_pair(ctx)[1024]

    def reference_path(ctx: object) -> np.ndarray:
        scores = anchor_score_pair(ctx)
        indices = {
            count: base.selected_index(ctx, values) for count, values in scores.items()
        }
        if active.decisions is not None:
            active.decisions.append(
                {
                    "seed": int(active.seed),
                    "stratum": int(active.stratum),
                    "step": int(ctx.step),
                    "selected_index": {str(k): int(v) for k, v in indices.items()},
                    "selected_dose": {
                        str(k): [float(x) for x in ctx.Xset[v].detach().cpu().numpy()]
                        for k, v in indices.items()
                    },
                    "gate_empty": bool(ctx.gate_empty),
                }
            )
        return scores[512]

    anchor_1024.acquisition_name = "cKG-1024-anchor"  # type: ignore[attr-defined]
    reference_path.acquisition_name = "cKG-512-reference-with-1024"  # type: ignore[attr-defined]
    try:
        ACQUISITIONS[ANCHOR_NAME] = anchor_1024
        ACQUISITIONS[REFERENCE_NAME] = reference_path
        yield
    finally:
        for name, previous in originals.items():
            if previous is None:
                ACQUISITIONS.pop(name, None)
            else:
                ACQUISITIONS[name] = previous


def trial_kwargs(seed: int, stratum: int, *, budget: int = 40) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "z": int(stratum),
        "gamma": 0.7,
        "sim": "osa",
        "mode": "latent",
        "noise": "fixed",
        "traj": False,
        "budget": int(budget),
        "warmup": 4,
        "r_k": 2,
        "grid_n": 5,
        "empty_gate": "pf",
        "protocol_scaffold": "lhs_fixed",
    }


def load_reference(
    path: Path,
) -> tuple[
    dict[tuple[int, int], dict[str, Any]],
    dict[tuple[int, int, int], dict[str, Any]],
    dict[str, Any],
]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing reference metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if _sha256(path) != metadata.get("artifact_sha256"):
        raise ValueError("reference raw artifact hash does not agree with its metadata")
    source_files = [
        ROOT / "paper/run_computational_diagnostics.py",
        ROOT / "src/dose_combination_bo/acquisitions.py",
        ROOT / "src/dose_combination_bo/trial.py",
        ROOT / "src/dose_combination_bo/gp.py",
    ]
    recorded_sources = metadata.get("source_sha256", {})
    for source in source_files:
        if recorded_sources.get(source.name) != _sha256(source):
            raise ValueError(f"reference source hash changed: {source.name}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    design = raw.get("design", {})
    expected_design = {
        "surface": "osa",
        "protocol_scaffold": "lhs_fixed",
        "gate_mode": "latent",
        "noise": "fixed",
        "tau": 0.7,
        "budget": 40,
        "warmup": 4,
        "cohort_size": 2,
        "grid_n": 5,
    }
    if any(design.get(key) != value for key, value in expected_design.items()):
        raise ValueError("reference raw artifact has an incompatible frozen design")
    if not raw.get("production_compatibility_preflight", {}).get(
        "full_path_and_terminal_match", False
    ):
        raise ValueError("reference raw artifact lacks the production-512 identity preflight")
    rows = raw.get("stability_trials", [])
    lookup = {
        (int(row["seed"]), int(row["stratum"])): row["result"]
        for row in rows
        if row["variant"] == "cKG-512"
    }
    expected = {(seed, stratum) for seed in SEEDS for stratum in STRATA}
    if set(lookup) != expected:
        raise ValueError("reference cKG-512 seed-stratum grid is incomplete")
    stored_decisions = {
        (int(row["seed"]), int(row["stratum"]), int(row["step"])): {
            "gate_empty": bool(row["gate_empty"]),
            "selected_index": int(row["selected_index"]["512"]),
            "selected_dose": [float(x) for x in row["selected_dose"]["512"]],
        }
        for row in raw.get("same_state_decisions", [])
    }
    expected_decisions = {
        (seed, stratum, step)
        for seed in SEEDS
        for stratum in STRATA
        for step in range(18)
    }
    if set(stored_decisions) != expected_decisions:
        raise ValueError("reference same-state decision grid is incomplete")
    return lookup, stored_decisions, {
        "artifact": path.name,
        "artifact_sha256": _sha256(path),
        "metadata_sha256": _sha256(metadata_path),
        "verified_source_sha256": {
            source.name: _sha256(source) for source in source_files
        },
    }


def production_compatibility_preflight(active: ActiveRun) -> dict[str, Any]:
    common = trial_kwargs(0, 0, budget=8)
    production = run_trial("cKG", **common)
    active.decisions = None
    reference = run_trial(REFERENCE_NAME, **common)
    if not base._same_result(base._compact_result(production), base._compact_result(reference)):
        raise RuntimeError("the nested 512 prefix changed the production cKG path")
    return {
        "budget": 8,
        "seed": 0,
        "stratum": 0,
        "identical_on_all_audited_fields": True,
        "note": "The nested 512 prefix and production cKG were identical on all audited fields.",
    }


def run_anchor_paths() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(SEEDS) * len(STRATA)
    order = 0
    for seed in SEEDS:
        for stratum in STRATA:
            order += 1
            print(f"[1024 path {order:02d}/{total}] seed={seed} z={stratum}", flush=True)
            started = time.perf_counter_ns()
            result = run_trial(ANCHOR_NAME, **trial_kwargs(seed, stratum))
            rows.append(
                {
                    "seed": seed,
                    "stratum": stratum,
                    "run_order": order,
                    "total_seconds_not_timing_comparable": (
                        time.perf_counter_ns() - started
                    )
                    / 1e9,
                    "result": base._compact_result(result),
                }
            )
    return rows


def run_same_state_reference_paths(
    active: ActiveRun,
    reference: dict[tuple[int, int], dict[str, Any]],
    stored_decisions: dict[tuple[int, int, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    total = len(SEEDS) * len(STRATA)
    order = 0
    for seed in SEEDS:
        for stratum in STRATA:
            order += 1
            print(f"[same-state {order:02d}/{total}] seed={seed} z={stratum}", flush=True)
            local: list[dict[str, Any]] = []
            active.decisions = local
            active.seed = seed
            active.stratum = stratum
            started = time.perf_counter_ns()
            try:
                result = run_trial(REFERENCE_NAME, **trial_kwargs(seed, stratum))
            finally:
                elapsed = (time.perf_counter_ns() - started) / 1e9
                active.decisions = None
            compact = base._compact_result(result)
            if not base._same_result(reference[(seed, stratum)], compact):
                raise RuntimeError(
                    f"nested 512 reference path changed for seed={seed}, stratum={stratum}"
                )
            if len(local) != 18:
                raise RuntimeError("same-state path did not contain 18 adaptive decisions")
            for row in local:
                key = (seed, stratum, int(row["step"]))
                stored = stored_decisions[key]
                current = {
                    "gate_empty": bool(row["gate_empty"]),
                    "selected_index": int(row["selected_index"]["512"]),
                    "selected_dose": [float(x) for x in row["selected_dose"]["512"]],
                }
                if current != stored:
                    raise RuntimeError(
                        f"nested 512 decision changed for seed={seed}, "
                        f"stratum={stratum}, step={row['step']}"
                    )
                administered = compact["allocation_history"][4 + 2 * int(row["step"])]
                if current["selected_dose"] != administered:
                    raise RuntimeError("logged 512 choice differs from administered cohort")
            decisions.extend(local)
            paths.append(
                {
                    "seed": seed,
                    "stratum": stratum,
                    "total_seconds_not_timing_comparable": elapsed,
                    "result": compact,
                }
            )
    return decisions, paths


def _adaptive_path(result: dict[str, Any]) -> list[list[float]]:
    return result["allocation_history"][4::2]


def _clustered(values: dict[tuple[int, int], float]) -> dict[str, Any]:
    seed_values = np.asarray(
        [statistics.fmean(values[(seed, stratum)] for stratum in STRATA) for seed in SEEDS],
        dtype=float,
    )
    return {
        "n_seeds": len(SEEDS),
        "n_seed_stratum_cells": len(values),
        "mean_pct": float(100 * seed_values.mean()),
        "mcse_pct": float(100 * seed_values.std(ddof=1) / math.sqrt(seed_values.size)),
        "min_seed_average_pct": float(100 * seed_values.min()),
        "max_seed_average_pct": float(100 * seed_values.max()),
    }


def summarize(
    anchor_rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    reference: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    anchors = {(row["seed"], row["stratum"]): row["result"] for row in anchor_rows}
    expected = {(seed, stratum) for seed in SEEDS for stratum in STRATA}
    if set(anchors) != expected:
        raise ValueError("cKG-1024 full-path grid is incomplete")
    exact_path: dict[tuple[int, int], float] = {}
    positionwise: dict[tuple[int, int], float] = {}
    terminal: dict[tuple[int, int], float] = {}
    unsafe: dict[tuple[int, int], float] = {}
    anchor_unsafe: dict[tuple[int, int], float] = {}
    reference_unsafe: dict[tuple[int, int], float] = {}
    anchor_unsafe_reference_safe: dict[tuple[int, int], float] = {}
    anchor_safe_reference_unsafe: dict[tuple[int, int], float] = {}
    for key in sorted(expected):
        candidate = anchors[key]
        baseline = reference[key]
        path_matches = [
            a == b for a, b in zip(_adaptive_path(candidate), _adaptive_path(baseline))
        ]
        if len(path_matches) != 18:
            raise ValueError("full adaptive paths must contain 18 cohorts")
        exact_path[key] = float(all(path_matches))
        positionwise[key] = float(np.mean(path_matches))
        terminal[key] = float(
            [candidate["rec_d1"], candidate["rec_d2"]]
            == [baseline["rec_d1"], baseline["rec_d2"]]
        )
        unsafe[key] = float(candidate["rec_unsafe"] == baseline["rec_unsafe"])
        anchor_unsafe[key] = float(bool(candidate["rec_unsafe"]))
        reference_unsafe[key] = float(bool(baseline["rec_unsafe"]))
        anchor_unsafe_reference_safe[key] = float(
            bool(candidate["rec_unsafe"]) and not bool(baseline["rec_unsafe"])
        )
        anchor_safe_reference_unsafe[key] = float(
            not bool(candidate["rec_unsafe"]) and bool(baseline["rec_unsafe"])
        )

    by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in decisions:
        by_cell.setdefault((int(row["seed"]), int(row["stratum"])), []).append(row)
    if set(by_cell) != expected:
        raise ValueError("same-state seed-stratum grid is incomplete")
    same_state: dict[tuple[int, int], float] = {}
    same_state_nonempty: dict[tuple[int, int], float] = {}
    same_state_exact: dict[tuple[int, int], float] = {}
    empty_gate_count = 0
    for key, rows in by_cell.items():
        rows = sorted(rows, key=lambda row: row["step"])
        if len(rows) != 18:
            raise ValueError("same-state paths must contain 18 decisions")
        indicators = [
            row["selected_index"]["1024"] == row["selected_index"]["512"]
            for row in rows
        ]
        nonempty_indicators = [
            indicator
            for indicator, row in zip(indicators, rows)
            if not bool(row["gate_empty"])
        ]
        if not nonempty_indicators:
            raise ValueError("a same-state path has no nonempty-gate decisions")
        empty_gate_count += sum(bool(row["gate_empty"]) for row in rows)
        same_state[key] = float(np.mean(indicators))
        same_state_nonempty[key] = float(np.mean(nonempty_indicators))
        same_state_exact[key] = float(all(indicators))
    return {
        "reference": "production-compatible cKG-512",
        "anchor": "post hoc cKG-1024 numerical anchor",
        "interpretation": "agreement measures numerical stability, not accuracy or convergence",
        "stratum_handling": "two strata averaged within seed before agreement summaries",
        "comparison": {
            "same_state_selected_dose_agreement": _clustered(same_state),
            "same_state_nonempty_gate_selected_dose_agreement": _clustered(
                same_state_nonempty
            ),
            "same_state_gate_empty_forced_fallback": {
                "count": int(empty_gate_count),
                "total_same_state_decisions": len(decisions),
                "pct": float(100 * empty_gate_count / len(decisions)),
                "note": (
                    "The common empty-gate fallback forces both counts to the same "
                    "most-feasible dose; nonempty-gate agreement removes these decisions."
                ),
            },
            "same_state_all_decisions_exact_agreement": _clustered(same_state_exact),
            "full_adaptive_path_exact_agreement": _clustered(exact_path),
            "positionwise_adaptive_cohort_agreement": _clustered(positionwise),
            "terminal_recommendation_agreement": _clustered(terminal),
            "terminal_unsafe_status_agreement": _clustered(unsafe),
            "anchor_terminal_unsafe_rate": _clustered(anchor_unsafe),
            "reference_terminal_unsafe_rate": _clustered(reference_unsafe),
            "anchor_unsafe_reference_safe_rate": _clustered(
                anchor_unsafe_reference_safe
            ),
            "anchor_safe_reference_unsafe_rate": _clustered(
                anchor_safe_reference_unsafe
            ),
        },
    }


def frozen_design(reference_source: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "post hoc and descriptive",
        "surface": "osa",
        "protocol_scaffold": "lhs_fixed",
        "gate_mode": "latent",
        "noise": "fixed",
        "tau": 0.7,
        "strata": list(STRATA),
        "seeds": list(SEEDS),
        "budget": 40,
        "warmup": 4,
        "cohort_size": 2,
        "grid_n": 5,
        "empty_gate": "pf",
        "reference_fantasies": BASE_FANTASIES,
        "anchor_fantasies": ANCHOR_FANTASIES,
        "bank_construction": (
            "Draw production Zf[0:512], then production Zg[0:512], then fresh "
            "Zf[512:1024], then fresh Zg[512:1024]; compare channel-wise prefixes."
        ),
        "reference_compatibility": (
            "The 512 channel prefixes and 512 reference paths remain production-compatible."
        ),
        "anchor_scope": (
            "cKG-1024 is a custom prefix-preserving post hoc numerical anchor, "
            "not a package default or the conventional nmc=1024 draw order."
        ),
        "fixed_fantasy_seed": 0,
        "fantasy_seed_note": (
            "The cKG fantasy bank is fixed at seed 0 for every trial; trial seeds vary "
            "data and paths, not the fantasy-bank realization."
        ),
        "interpretation": (
            "agreement is a nested-count stability sensitivity, not accuracy or convergence"
        ),
        "reference_source": reference_source,
    }


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def render_markdown(summary: dict[str, Any]) -> str:
    comparison = summary["stability"]["comparison"]

    def show(key: str) -> str:
        metric = comparison[key]
        return f"{metric['mean_pct']:.1f}% +/- {metric['mcse_pct']:.1f}%"

    return "\n".join(
        [
            "# cKG-1024 numerical-stability anchor",
            "",
            "This OSA analysis is **post hoc and descriptive**. The production-compatible",
            "cKG-512 streams are retained as channel-wise prefixes; cKG-1024 appends fresh",
            "efficacy and toxicity draws. It is a custom prefix-preserving anchor, not",
            "the package's conventional nmc=1024 draw order or a package default.",
            "",
            "The two strata were averaged within each of 20 trial seeds. Values are",
            "means +/- seed-clustered SE across those seed averages.",
            "",
            "| Comparison | Same-state selected dose | Full path exact | Cohort-position agreement | Terminal recommendation | Unsafe status |",
            "|---|---:|---:|---:|---:|---:|",
            (
                f"| cKG-1024 vs cKG-512 | {show('same_state_selected_dose_agreement')} | "
                f"{show('full_adaptive_path_exact_agreement')} | "
                f"{show('positionwise_adaptive_cohort_agreement')} | "
                f"{show('terminal_recommendation_agreement')} | "
                f"{show('terminal_unsafe_status_agreement')} |"
            ),
            "",
            (
                "Excluding decisions with an empty outer gate, same-state selected-dose "
                f"agreement was {show('same_state_nonempty_gate_selected_dose_agreement')}. "
                f"The forced fallback applied at {comparison['same_state_gate_empty_forced_fallback']['count']} "
                f"of {comparison['same_state_gate_empty_forced_fallback']['total_same_state_decisions']} states."
            ),
            "",
            (
                f"Unsafe recommendation rates were {show('anchor_terminal_unsafe_rate')} "
                f"for cKG-1024 and {show('reference_terminal_unsafe_rate')} for cKG-512. "
                f"Discordant directions were {show('anchor_unsafe_reference_safe_rate')} "
                "for 1024-unsafe/512-safe and "
                f"{show('anchor_safe_reference_unsafe_rate')} for 1024-safe/512-unsafe."
            ),
            "",
            "Agreement measures numerical stability relative to cKG-512, not decision",
            "accuracy or Monte Carlo convergence. The fantasy seed is fixed at 0 across",
            "trial seeds. No timing or GPU claim is made.",
            "",
        ]
    )


def write_artifacts(
    output_dir: Path,
    *,
    design: dict[str, Any],
    environment: dict[str, Any],
    preflight: dict[str, Any],
    anchor_rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    stability: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "anchor_full_path_trials": len(anchor_rows),
        "same_state_decisions": len(decisions),
        "reference_path_trials": len(reference_rows),
    }
    raw = {
        "schema_version": 1,
        "status": "post hoc and descriptive",
        "design": design,
        "environment": environment,
        "production_512_compatibility_preflight": preflight,
        "wall_clock_seconds_for_complete_anchor": float(elapsed_seconds),
        "anchor_full_path_trials": anchor_rows,
        "same_state_decisions": decisions,
        "reference_path_trials": reference_rows,
    }
    summary = {
        "schema_version": 1,
        "status": "post hoc and descriptive",
        "design": design,
        "environment": environment,
        "production_512_compatibility_preflight": preflight,
        "stability": stability,
    }
    raw_path = output_dir / "ckg_1024_anchor_raw.json"
    summary_path = output_dir / "ckg_1024_anchor_summary.json"
    markdown_path = output_dir / "ckg_1024_anchor_summary.md"
    _json_dump(raw_path, raw)
    _json_dump(summary_path, summary)
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")

    script = Path(__file__).resolve()
    source_files = [
        script,
        ROOT / "paper/run_computational_diagnostics.py",
        ROOT / "src/dose_combination_bo/acquisitions.py",
        ROOT / "src/dose_combination_bo/trial.py",
        ROOT / "src/dose_combination_bo/gp.py",
    ]
    paths = {"raw": raw_path, "summary": summary_path, "markdown": markdown_path}
    for key, path in paths.items():
        metadata = {
            "artifact": path.name,
            "artifact_type": f"ckg_1024_anchor_{key}",
            "artifact_sha256": _sha256(path),
            "artifact_bytes": path.stat().st_size,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "post hoc and descriptive",
            "design": design,
            "environment": environment,
            "record_counts": counts,
            "source_sha256": {item.name: _sha256(item) for item in source_files},
        }
        _json_dump(path.with_suffix(path.suffix + ".metadata.json"), metadata)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-raw",
        type=Path,
        default=ROOT
        / "results/computational_diagnostics/computational_diagnostics_raw.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/computational_diagnostics",
    )
    args = parser.parse_args(argv)
    base.configure_single_thread_runtime()
    reference, stored_decisions, reference_source = load_reference(
        args.reference_raw.resolve()
    )
    environment = base.environment_record()
    design = frozen_design(reference_source)
    started = time.perf_counter()
    active = ActiveRun()
    with installed_anchor_acquisitions(active):
        print("[preflight] checking nested 512 prefix against production", flush=True)
        preflight = production_compatibility_preflight(active)
        anchor_rows = run_anchor_paths()
        decisions, reference_rows = run_same_state_reference_paths(
            active, reference, stored_decisions
        )
    stability = summarize(anchor_rows, decisions, reference)
    elapsed = time.perf_counter() - started
    paths = write_artifacts(
        args.output_dir.resolve(),
        design=design,
        environment=environment,
        preflight=preflight,
        anchor_rows=anchor_rows,
        decisions=decisions,
        reference_rows=reference_rows,
        stability=stability,
        elapsed_seconds=elapsed,
    )
    print(f"[done] elapsed={elapsed:.1f}s", flush=True)
    for key, path in paths.items():
        print(f"[done] {key}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
