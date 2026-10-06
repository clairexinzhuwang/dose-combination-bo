#!/usr/bin/env python3
"""Validate the opt-in piecewise-analytic cKG evaluator on frozen posterior states.

The primary panel reconstructs all 20 seeds x 2 strata x 18 adaptive decisions
from the existing production-compatible cKG-512 reference paths in the tracked
cKG-1024 anchor. It saves exact, nested-512, and prefix-preserving-1024 complete
score vectors without running new trial decisions. A stratified 64-state subset
receives eight independently scrambled 2^17-point Sobol checks. Small additive
sentinels cover four testbeds x five gates, predictive gating, and gradual access.

This numerical runner does not register the exact evaluator, change the production
512-fantasy default, or run formal operating-characteristic trials.
"""
from __future__ import annotations

import os

for _thread_variable in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict, deque
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
from scipy.stats import norm, qmc

from paper import run_ckg_1024_anchor as anchor
from dose_combination_bo.ckg_exact import ckg_scores_exact
from dose_combination_bo.context import Context
from dose_combination_bo.gp import fit_gp, joint, post_latent
from dose_combination_bo.protocol import cohort_region_q, eligible_mask, region_mask
from dose_combination_bo.surfaces import resolve_surface
from dose_combination_bo.trial import grid


PRIMARY_SOURCE = ROOT / "results/computational_diagnostics/ckg_1024_anchor_raw.json"
GRADUAL_SOURCE = ROOT / "results/protocol_scaffold_sensitivity.json"
OUTPUT_DIR = ROOT / "results/ckg_piecewise_exact_validation"
EXACT_SOURCE = ROOT / "src/dose_combination_bo/ckg_exact.py"
STABLE_HARNESS_SOURCE_PATHS = (
    "src/dose_combination_bo/acquisitions.py",
    "src/dose_combination_bo/context.py",
    "src/dose_combination_bo/gp.py",
    "src/dose_combination_bo/protocol.py",
    "src/dose_combination_bo/surfaces.py",
    "src/dose_combination_bo/trial.py",
)

SEEDS = tuple(range(20))
STRATA = (0, 1)
STEPS = tuple(range(18))
TAU = 0.7
R_K = 2
GRID_N = 5
WARMUP = 4
GATES = (0.5, 0.6, 0.7, 0.8, 0.9)
SURFACES = ("osa", "gbump", "efftox", "mariposa")
SOBOL_POWER = 17
SOBOL_SCRAMBLES = tuple(range(8))
SOBOL_SUBSET_SIZE = 64
SENTINEL_SOBOL_SCRAMBLES = (0, 1)
INITIAL_SENTINEL_SEED = 271828
GRADUAL_SENTINELS = (
    (0, 0, 0.7, 0),
    (0, 0, 0.7, 3),
    (7, 1, 0.9, 3),
    (19, 1, 0.9, 7),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_harness_source_sha256() -> dict[str, str]:
    return {
        relative: _sha256(ROOT / relative)
        for relative in STABLE_HARNESS_SOURCE_PATHS
    }


def _write_json(path: Path, value: Any) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _verify_sidecar(path: Path) -> dict:
    metadata_path = Path(str(path) + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recorded_hash = metadata.get("artifact_sha256")
    if recorded_hash is None and metadata.get("source_path") == path.name:
        recorded_hash = metadata.get("source_sha256")
    if recorded_hash != _sha256(path):
        raise RuntimeError(f"source sidecar hash mismatch: {path.name}")
    return metadata


def _load_primary_source():
    metadata = _verify_sidecar(PRIMARY_SOURCE)
    payload = json.loads(PRIMARY_SOURCE.read_text(encoding="utf-8"))
    design = payload.get("design", {})
    expected_design = {
        "surface": "osa", "protocol_scaffold": "lhs_fixed",
        "gate_mode": "latent", "noise": "fixed", "tau": TAU,
        "budget": 40, "warmup": WARMUP, "cohort_size": R_K,
        "grid_n": GRID_N, "reference_fantasies": 512,
        "anchor_fantasies": 1024,
    }
    if any(design.get(key) != value for key, value in expected_design.items()):
        raise RuntimeError("primary source design is incompatible")
    trajectories = {
        (int(row["seed"]), int(row["stratum"])): row["result"]
        for row in payload["reference_path_trials"]
    }
    decisions = {
        (int(row["seed"]), int(row["stratum"]), int(row["step"])): row
        for row in payload["same_state_decisions"]
    }
    expected_trajectories = {(seed, z) for seed in SEEDS for z in STRATA}
    expected_decisions = {
        (seed, z, step) for seed in SEEDS for z in STRATA for step in STEPS
    }
    if set(trajectories) != expected_trajectories:
        raise RuntimeError("primary trajectory grid is incomplete")
    if set(decisions) != expected_decisions:
        raise RuntimeError("primary decision grid is incomplete")
    return payload, metadata, trajectories, decisions


def _replay_lhs_observations(seed: int, stratum: int, history: np.ndarray,
                             surface_name: str = "osa"):
    surface = resolve_surface(surface_name, stratum)
    rng = np.random.default_rng(seed)
    cut = np.linspace(0.0, 1.0, WARMUP + 1)
    yf, yg = [], []
    for index, (d1, d2) in enumerate(history):
        if index < WARMUP:
            expected = np.array([
                cut[index] + rng.uniform() * (cut[1] - cut[0]), rng.uniform(),
            ])
            if not np.array_equal(expected, np.asarray([d1, d2])):
                raise RuntimeError(
                    f"seed={seed} z={stratum}: LHS initialization cannot be replayed"
                )
        yf.append(surface["eff"](d1, d2) + rng.normal(0.0, surface["sf"]))
        yg.append(surface["tox"](d1, d2) + rng.normal(0.0, surface["sg"]))
    return np.asarray(yf), np.asarray(yg)


def _replay_gradual_observations(record: dict):
    surface = resolve_surface("osa", int(record["stratum"]))
    rng = np.random.default_rng(int(record["seed"]))
    history = np.asarray(record["allocation_history"], dtype=float)
    yf, yg = [], []
    for d1, d2 in history:
        yf.append(surface["eff"](d1, d2) + rng.normal(0.0, surface["sf"]))
        yg.append(surface["tox"](d1, d2) + rng.normal(0.0, surface["sg"]))
    return history, np.asarray(yf), np.asarray(yg)


def _fit_context(history: np.ndarray, yf: np.ndarray, yg: np.ndarray, prefix_n: int,
                 *, surface_name: str, stratum: int, gamma: float,
                 gate_mode: str = "latent", Xset: torch.Tensor | None = None,
                 candidate_mask: np.ndarray | None = None) -> Context:
    surface = resolve_surface(surface_name, stratum)
    observed = torch.tensor(history[:prefix_n])
    me, le = fit_gp(
        observed, torch.tensor(yf[:prefix_n]), fixed_noise=surface["sf"] ** 2
    )
    mt, lt = fit_gp(
        observed, torch.tensor(yg[:prefix_n]), fixed_noise=surface["sg"] ** 2
    )
    if Xset is None:
        Xset = torch.tensor(grid(GRID_N))
    return Context(
        Xset=Xset, me=me, le=le, mt=mt, lt=lt,
        g_dagger=surface["gd"], gamma=gamma, gate_mode=gate_mode,
        r_k=R_K, empty_gate="pf", candidate_mask=candidate_mask,
    )


def _raw_anchor_score_pair(ctx: Context):
    scores_512, scores_1024 = [], []
    for index in range(ctx.Xset.shape[0]):
        value_512, value_1024 = anchor.ckg_one_step_gated_anchor_pair(
            ctx.Xset[index], ctx.Xset, ctx.me, ctx.le, ctx.mt, ctx.lt,
            ctx.g_dagger, ctx.gamma, r_k=ctx.r_k, seed=ctx.ckg_seed,
            gate_mode=ctx.gate_mode,
        )
        scores_512.append(value_512)
        scores_1024.append(value_1024)
    return np.asarray(scores_512), np.asarray(scores_1024)


def _selected_index(raw_scores: np.ndarray, ctx: Context) -> int:
    restricted = ctx.restrict(raw_scores)
    restricted = np.where(ctx.candidate_mask, restricted, -np.inf)
    return int(
        ctx.most_feasible_index
        if not np.isfinite(restricted).any()
        else np.argmax(restricted)
    )


def _top_two_margin(raw_scores: np.ndarray, ctx: Context):
    candidates = np.flatnonzero(ctx.safe)
    if len(candidates) < 2:
        return None
    ordered = np.sort(raw_scores[candidates])[::-1]
    return float(ordered[0] - ordered[1])


def _current_comparator_audit(ctx: Context, state_id: str) -> dict:
    """Require the current Context to implement the stable z/ppf semantics.

    CDF comparisons are retained only as diagnostics for the historical
    implementation; CDF rounding is not allowed to select a current fallback.
    """
    z_now = (float(ctx.g_dagger) - ctx.mu_g) / ctx.sd_g
    q_gamma = float(norm.ppf(float(ctx.gamma)))
    stable_gate = z_now > q_gamma
    current_gate = np.asarray(ctx.gate_safe, dtype=bool)
    cdf_gate = np.asarray(ctx.pf, dtype=float) > float(ctx.gamma)
    candidate_mask = np.asarray(ctx.candidate_mask, dtype=bool)
    stable_safe = stable_gate & candidate_mask
    current_safe = np.asarray(ctx.safe, dtype=bool)
    cdf_safe = cdf_gate & candidate_mask

    stable_full_fallback = int(np.argmax(z_now))
    current_full_fallback = int(ctx.most_feasible_full_index)
    cdf_full_fallback = int(np.argmax(ctx.pf))
    stable_eligible_fallback = int(np.argmax(np.where(
        candidate_mask, z_now, -np.inf
    )))
    current_eligible_fallback = int(ctx.most_feasible_index)
    cdf_eligible_fallback = int(np.argmax(np.where(
        candidate_mask, ctx.pf, -np.inf
    )))
    full_fallback_matches = stable_full_fallback == current_full_fallback
    eligible_fallback_matches = (
        stable_eligible_fallback == current_eligible_fallback
    )
    full_fallback_operational = bool(not stable_gate.any())
    eligible_fallback_operational = bool(not stable_safe.any())
    gate_difference_indices = np.flatnonzero(stable_gate != current_gate).astype(int)
    safe_difference_indices = np.flatnonzero(stable_safe != current_safe).astype(int)
    cdf_gate_difference_indices = np.flatnonzero(stable_gate != cdf_gate).astype(int)
    cdf_safe_difference_indices = np.flatnonzero(stable_safe != cdf_safe).astype(int)
    cdf_full_fallback_differs = stable_full_fallback != cdf_full_fallback
    cdf_eligible_fallback_differs = (
        stable_eligible_fallback != cdf_eligible_fallback
    )
    audit = {
        "semantic_definition": "stable z>ppf(gamma), fallback argmax z",
        "current_definition": "stable z>ppf(gamma), fallback argmax z",
        "cdf_diagnostic_definition": "cdf(z)>gamma, fallback argmax cdf(z)",
        "gate_pass_set_matches": bool(not len(gate_difference_indices)),
        "gate_pass_set_difference_indices": gate_difference_indices.tolist(),
        "eligible_gate_pass_set_matches": bool(not len(safe_difference_indices)),
        "eligible_gate_pass_set_difference_indices": safe_difference_indices.tolist(),
        "cdf_gate_pass_set_difference_indices": cdf_gate_difference_indices.tolist(),
        "cdf_eligible_gate_pass_set_difference_indices": cdf_safe_difference_indices.tolist(),
        "stable_full_fallback_index": stable_full_fallback,
        "current_full_fallback_index": current_full_fallback,
        "cdf_full_fallback_index": cdf_full_fallback,
        "full_fallback_is_operational": full_fallback_operational,
        "full_fallback_index_matches": bool(full_fallback_matches),
        "operational_full_fallback_index_matches": bool(
            not full_fallback_operational or full_fallback_matches
        ),
        "inactive_full_fallback_rounding_difference": bool(
            not full_fallback_operational and cdf_full_fallback_differs
        ),
        "operational_full_fallback_rounding_difference": bool(
            full_fallback_operational and cdf_full_fallback_differs
        ),
        "stable_eligible_fallback_index": stable_eligible_fallback,
        "current_eligible_fallback_index": current_eligible_fallback,
        "cdf_eligible_fallback_index": cdf_eligible_fallback,
        "eligible_fallback_is_operational": eligible_fallback_operational,
        "eligible_fallback_index_matches": bool(eligible_fallback_matches),
        "operational_eligible_fallback_index_matches": bool(
            not eligible_fallback_operational or eligible_fallback_matches
        ),
        "inactive_eligible_fallback_rounding_difference": bool(
            not eligible_fallback_operational and cdf_eligible_fallback_differs
        ),
        "operational_eligible_fallback_rounding_difference": bool(
            eligible_fallback_operational and cdf_eligible_fallback_differs
        ),
        "stable_eligible_gate_empty": bool(not stable_safe.any()),
        "current_eligible_gate_empty": bool(ctx.gate_empty),
        "eligible_gate_empty_matches": bool(
            bool(not stable_safe.any()) == bool(ctx.gate_empty)
        ),
        "minimum_abs_z_minus_q": float(np.min(np.abs(z_now - q_gamma))),
        "maximum_abs_standardized_feasibility": float(np.max(np.abs(z_now))),
        "cdf_saturated_zero_count": int(np.count_nonzero(ctx.pf == 0.0)),
        "cdf_saturated_one_count": int(np.count_nonzero(ctx.pf == 1.0)),
    }
    required_matches = (
        "gate_pass_set_matches", "eligible_gate_pass_set_matches",
        "operational_full_fallback_index_matches",
        "operational_eligible_fallback_index_matches",
        "eligible_gate_empty_matches",
    )
    if not all(audit[field] for field in required_matches):
        raise RuntimeError(
            f"{state_id}: stable and current Context gate semantics differ: {audit}"
        )
    return audit


def _phase(step: int) -> str:
    return "early" if step <= 5 else ("middle" if step <= 11 else "late")


def _query_components(ctx: Context, query_index: int):
    d = ctx.Xset[query_index]
    mu_f_t, _var_f_t, covf_t, s2f_full = joint(ctx.me, ctx.le, ctx.Xset, d)
    mu_g_t, var_g_t, covg_t, s2g_full = joint(ctx.mt, ctx.lt, ctx.Xset, d)
    nf, ng = float(ctx.le.noise.item()), float(ctx.lt.noise.item())
    s2f = (float(s2f_full) - nf) + nf / ctx.r_k
    s2g = (float(s2g_full) - ng) + ng / ctx.r_k
    mu_f = mu_f_t.detach().cpu().numpy()
    mu_g = mu_g_t.detach().cpu().numpy()
    var_g = var_g_t.detach().cpu().numpy()
    covf = covf_t.detach().cpu().numpy()
    covg = covg_t.detach().cpu().numpy()
    b_f = covf / np.sqrt(s2f)
    b_g = covg / np.sqrt(s2g)
    observation_variance = ng if ctx.gate_mode == "predictive" else 0.0
    sd_post = np.sqrt(
        np.maximum(var_g - covg ** 2 / s2g, 1e-12) + observation_variance
    )
    _, var_now_t = post_latent(ctx.mt, ctx.Xset)
    sd_now = np.sqrt(
        np.maximum(var_now_t.detach().cpu().numpy(), 1e-12) + observation_variance
    )
    z_now = (ctx.g_dagger - mu_g) / sd_now
    gate_now = z_now > norm.ppf(ctx.gamma)
    current_value = (
        float(mu_f[gate_now].max()) if gate_now.any()
        else float(mu_f[int(np.argmax(z_now))])
    )
    return {
        "mu_f": mu_f, "b_f": b_f, "mu_g": mu_g, "b_g": b_g,
        "sd_post": sd_post, "g_dagger": float(ctx.g_dagger),
        "gamma": float(ctx.gamma), "current_value": current_value,
    }


def _sobol_banks(scrambles=SOBOL_SCRAMBLES):
    eps = np.finfo(float).eps
    return {
        seed: norm.ppf(np.clip(
            qmc.Sobol(2, scramble=True, seed=seed).random_base2(SOBOL_POWER),
            eps, 1.0 - eps,
        ))
        for seed in scrambles
    }


def _sobol_score(components: dict, normals: np.ndarray) -> float:
    mu_f, b_f = components["mu_f"], components["b_f"]
    mu_g, b_g = components["mu_g"], components["b_g"]
    updated_f = mu_f[None, :] + normals[:, 0, None] * b_f[None, :]
    updated_g = mu_g[None, :] + normals[:, 1, None] * b_g[None, :]
    standardised_feasibility = (
        (components["g_dagger"] - updated_g) / components["sd_post"][None, :]
    )
    gate = standardised_feasibility > norm.ppf(components["gamma"])
    values = np.where(gate, updated_f, -np.inf).max(axis=1)
    empty = ~np.isfinite(values)
    if empty.any():
        values[empty] = updated_f[empty][
            np.arange(int(empty.sum())),
            standardised_feasibility[empty].argmax(axis=1),
        ]
    return float(values.mean() - components["current_value"])


def _approximation_record(name: str, raw: np.ndarray, exact: np.ndarray,
                          ctx: Context, exact_choice: int) -> dict:
    choice = _selected_index(raw, ctx)
    error = raw - exact
    candidates = np.flatnonzero(ctx.safe)
    candidate_error = error[candidates] if len(candidates) else np.array([])
    return {
        "name": name, "raw_score_vector": raw.tolist(),
        "selected_index": choice,
        "selection_agrees_with_exact": bool(choice == exact_choice),
        "exact_regret_at_selected_query": float(exact[exact_choice] - exact[choice]),
        "max_abs_score_error_full_grid": float(np.max(np.abs(error))),
        "rmse_score_error_full_grid": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_score_error_candidate_set": (
            float(np.max(np.abs(candidate_error))) if len(candidate_error) else None
        ),
    }


def _evaluate_context(ctx: Context, *, state_id: str, state_metadata: dict,
                      expected_decision: dict | None = None):
    comparator_audit = _current_comparator_audit(ctx, state_id)
    exact_started = time.perf_counter()
    restricted_exact, diagnostics = ckg_scores_exact(
        ctx, return_diagnostics=True, include_regimes=False
    )
    exact_runtime = time.perf_counter() - exact_started
    exact_raw = np.asarray(diagnostics["raw_scores"], dtype=float)
    exact_choice = _selected_index(exact_raw, ctx)

    approximation_started = time.perf_counter()
    raw_512, raw_1024 = _raw_anchor_score_pair(ctx)
    approximation_runtime = time.perf_counter() - approximation_started
    approximation_rows = [
        _approximation_record(
            "legacy_mc_512_seed0", raw_512, exact_raw, ctx, exact_choice
        ),
        _approximation_record(
            "prefix_preserving_mc_1024_seed0", raw_1024, exact_raw, ctx, exact_choice
        ),
    ]

    if expected_decision is not None:
        if bool(expected_decision["gate_empty"]) != bool(ctx.gate_empty):
            raise RuntimeError(f"{state_id}: reconstructed gate-empty status changed")
        for count, approximation in zip((512, 1024), approximation_rows):
            expected_index = int(expected_decision["selected_index"][str(count)])
            if approximation["selected_index"] != expected_index:
                raise RuntimeError(
                    f"{state_id}: reconstructed {count} choice changed "
                    f"({approximation['selected_index']} != {expected_index})"
                )

    query_regimes = [{
        "query_index": index,
        "gate_threshold_count": detail["gate_threshold_count"],
        "gate_regime_count": detail["gate_regime_count"],
        "nonempty_gate_regime_count": detail["nonempty_gate_regime_count"],
        "empty_gate_regime_count": detail["empty_gate_regime_count"],
        "fallback_segment_count": detail["fallback_segment_count"],
        "integrated_segment_count": detail["integrated_segment_count"],
        "probability_mass": detail["probability_mass"],
    } for index, detail in enumerate(diagnostics["queries"])]
    record = {
        "state_id": state_id, **state_metadata,
        "accessible_grid_size": int(ctx.Xset.shape[0]),
        "accessible_grid": ctx.Xset.detach().cpu().numpy().tolist(),
        "candidate_mask_indices": np.flatnonzero(ctx.candidate_mask).astype(int).tolist(),
        "gate_passed_indices": np.flatnonzero(ctx.gate_safe).astype(int).tolist(),
        "gate_passed_and_eligible_indices": np.flatnonzero(ctx.safe).astype(int).tolist(),
        "gate_empty": bool(ctx.gate_empty),
        "most_feasible_index": int(ctx.most_feasible_index),
        "current_comparator_audit": comparator_audit,
        "exact": {
            "raw_score_vector": exact_raw.tolist(),
            "restricted_score_vector": [
                float(value) if np.isfinite(value) else None for value in restricted_exact
            ],
            "selected_index": exact_choice,
            "selected_dose": ctx.Xset[exact_choice].detach().cpu().numpy().tolist(),
            "margin_best_minus_runner_up": _top_two_margin(exact_raw, ctx),
            "runtime_seconds": exact_runtime,
            "query_regimes": query_regimes,
        },
        "approximations": approximation_rows,
        "approximation_pair_runtime_seconds": approximation_runtime,
    }
    return record, _query_components(ctx, exact_choice)


def _build_primary_panel(trajectories: dict, decisions: dict):
    states, components = [], {}
    total = len(SEEDS) * len(STRATA)
    completed = 0
    for seed in SEEDS:
        for stratum in STRATA:
            history = np.asarray(
                trajectories[(seed, stratum)]["allocation_history"], dtype=float
            )
            yf, yg = _replay_lhs_observations(seed, stratum, history)
            for step in STEPS:
                prefix_n = WARMUP + R_K * step
                ctx = _fit_context(
                    history, yf, yg, prefix_n,
                    surface_name="osa", stratum=stratum, gamma=TAU,
                )
                state_id = f"primary-osa-z{stratum}-seed{seed}-step{step:02d}"
                state, query_components = _evaluate_context(
                    ctx, state_id=state_id,
                    state_metadata={
                        "panel": "primary_720", "source_seed": seed,
                        "stratum": stratum, "step": step, "phase": _phase(step),
                        "prefix_n": prefix_n, "surface": "osa", "gamma": TAU,
                        "gate_mode": "latent", "protocol_scaffold": "lhs_fixed",
                    },
                    expected_decision=decisions[(seed, stratum, step)],
                )
                states.append(state)
                components[state_id] = query_components
            completed += 1
            print(
                f"[primary {completed:02d}/{total}] seed={seed} z={stratum}",
                flush=True,
            )
    return states, components


def _stable_state_order(state: dict) -> str:
    return hashlib.sha256(state["state_id"].encode("utf-8")).hexdigest()


def _select_sobol_subset(states: list[dict]) -> tuple[list[dict], float]:
    margins = [
        state["exact"]["margin_best_minus_runner_up"]
        for state in states
        if not state["gate_empty"]
        and state["exact"]["margin_best_minus_runner_up"] is not None
    ]
    median_margin = float(np.median(margins))
    groups: dict[tuple, deque] = defaultdict(deque)
    for state in states:
        margin = state["exact"]["margin_best_minus_runner_up"]
        if state["gate_empty"]:
            margin_band = "not_applicable_empty_gate"
        elif margin is None:
            margin_band = "single_candidate"
        elif margin <= median_margin:
            margin_band = "low_margin"
        else:
            margin_band = "high_margin"
        state["sobol_margin_band"] = margin_band
        key = (
            int(state["stratum"]), state["phase"],
            "empty" if state["gate_empty"] else "nonempty", margin_band,
        )
        groups[key].append(state)
    for key in groups:
        groups[key] = deque(sorted(groups[key], key=_stable_state_order))

    selected, keys = [], sorted(groups)
    while len(selected) < SOBOL_SUBSET_SIZE:
        made_progress = False
        for key in keys:
            if groups[key] and len(selected) < SOBOL_SUBSET_SIZE:
                selected.append(groups[key].popleft())
                made_progress = True
        if not made_progress:
            raise RuntimeError("not enough primary states for Sobol subset")
    return selected, median_margin


def _build_sobol_audit(states: list[dict], components: dict, banks: dict):
    selected, median_margin = _select_sobol_subset(states)
    audit_rows = []
    for order, state in enumerate(selected, 1):
        exact_score = float(
            state["exact"]["raw_score_vector"][state["exact"]["selected_index"]]
        )
        checks = []
        for scramble_seed in SOBOL_SCRAMBLES:
            estimate = _sobol_score(components[state["state_id"]], banks[scramble_seed])
            checks.append({
                "scramble_seed": scramble_seed, "estimate": estimate,
                "exact": exact_score, "error": float(estimate - exact_score),
                "absolute_error": float(abs(estimate - exact_score)),
            })
        audit_rows.append({
            "state_id": state["state_id"], "stratum": state["stratum"],
            "step": state["step"], "phase": state["phase"],
            "gate_empty": state["gate_empty"],
            "margin_band": state["sobol_margin_band"],
            "query_index": state["exact"]["selected_index"], "checks": checks,
        })
        print(f"[sobol {order:02d}/{len(selected)}] {state['state_id']}", flush=True)
    coverage = {
        "stratum": dict(Counter(str(row["stratum"]) for row in audit_rows)),
        "phase": dict(Counter(row["phase"] for row in audit_rows)),
        "gate_status": dict(Counter(
            "empty" if row["gate_empty"] else "nonempty" for row in audit_rows
        )),
        "margin_band": dict(Counter(row["margin_band"] for row in audit_rows)),
        "joint_strata": {
            "|".join(map(str, key)): value
            for key, value in Counter(
                (row["stratum"], row["phase"],
                 "empty" if row["gate_empty"] else "nonempty", row["margin_band"])
                for row in audit_rows
            ).items()
        },
    }
    return {
        "selection_rule": (
            "Deterministic round-robin over available stratum x early/middle/late "
            "x empty/nonempty x low/high/single margin categories; state order "
            "within category is SHA-256(state_id)."
        ),
        "nonempty_margin_median_used_for_stratification": median_margin,
        "state_count": len(audit_rows), "points_per_scramble": 2 ** SOBOL_POWER,
        "scramble_seeds": list(SOBOL_SCRAMBLES), "coverage": coverage,
        "states": audit_rows,
    }


def _initial_sentinel_data(surface_name: str, stratum: int = 0):
    surface = resolve_surface(surface_name, stratum)
    rng = np.random.default_rng(INITIAL_SENTINEL_SEED)
    cut = np.linspace(0.0, 1.0, WARMUP + 1)
    history, yf, yg = [], [], []
    for index in range(WARMUP):
        d1 = cut[index] + rng.uniform() * (cut[1] - cut[0])
        d2 = rng.uniform()
        history.append([d1, d2])
        yf.append(surface["eff"](d1, d2) + rng.normal(0.0, surface["sf"]))
        yg.append(surface["tox"](d1, d2) + rng.normal(0.0, surface["sg"]))
    return np.asarray(history), np.asarray(yf), np.asarray(yg)


def _build_testbed_gate_and_predictive_sentinels():
    rows, components = [], {}
    for surface_name in SURFACES:
        history, yf, yg = _initial_sentinel_data(surface_name)
        for gamma in GATES:
            ctx = _fit_context(
                history, yf, yg, WARMUP,
                surface_name=surface_name, stratum=0, gamma=gamma,
            )
            state_id = f"sentinel-{surface_name}-z0-tau{gamma:.1f}-latent"
            row, query_components = _evaluate_context(
                ctx, state_id=state_id,
                state_metadata={
                    "panel": "four_testbed_five_gate_sentinel",
                    "surface": surface_name, "stratum": 0, "gamma": gamma,
                    "gate_mode": "latent", "protocol_scaffold": "lhs_fixed",
                    "step": 0, "phase": "early", "prefix_n": WARMUP,
                    "construction": (
                        "Deterministic common-seed initial posterior; numerical "
                        "sentinel, not an archived trial record or OC replicate."
                    ),
                },
            )
            rows.append(row)
            components[state_id] = query_components
        ctx = _fit_context(
            history, yf, yg, WARMUP,
            surface_name=surface_name, stratum=0, gamma=0.7,
            gate_mode="predictive",
        )
        state_id = f"sentinel-{surface_name}-z0-tau0.7-predictive"
        row, query_components = _evaluate_context(
            ctx, state_id=state_id,
            state_metadata={
                "panel": "predictive_gate_sentinel", "surface": surface_name,
                "stratum": 0, "gamma": 0.7, "gate_mode": "predictive",
                "protocol_scaffold": "lhs_fixed", "step": 0,
                "phase": "early", "prefix_n": WARMUP,
                "construction": (
                    "Deterministic common-seed initial posterior; numerical sentinel, "
                    "not an archived trial record or OC replicate."
                ),
            },
        )
        rows.append(row)
        components[state_id] = query_components
    return rows, components


def _load_gradual_records():
    metadata = _verify_sidecar(GRADUAL_SOURCE)
    records = json.loads(GRADUAL_SOURCE.read_text(encoding="utf-8"))
    lookup = {
        (int(row["seed"]), int(row["stratum"]), float(row["gamma"])): row
        for row in records
        if row["policy"] == "cKG"
        and row["protocol_scaffold"] == "start_low_expansion"
    }
    required = {(seed, z, gamma) for seed, z, gamma, _ in GRADUAL_SENTINELS}
    if not required.issubset(lookup):
        raise RuntimeError("gradual sentinel source records are missing")
    return metadata, lookup


def _build_gradual_sentinels(lookup: dict):
    rows, components = [], {}
    full_grid = torch.tensor(grid(GRID_N))
    for seed, stratum, gamma, step in GRADUAL_SENTINELS:
        source_record = lookup[(seed, stratum, gamma)]
        history, yf, yg = _replay_gradual_observations(source_record)
        prefix_n = R_K + R_K * step
        if prefix_n >= len(history):
            raise RuntimeError("gradual sentinel has no recorded next cohort")
        region_q = cohort_region_q(prefix_n, R_K)
        in_region = region_mask(full_grid, region_q, 0.25)
        active_Xset = full_grid[torch.as_tensor(in_region, dtype=torch.bool)]
        eligible_full = eligible_mask(
            full_grid, history[:prefix_n], region_q, 0.25, True
        )
        candidate_mask = eligible_full[in_region]
        ctx = _fit_context(
            history, yf, yg, prefix_n,
            surface_name="osa", stratum=stratum, gamma=gamma,
            Xset=active_Xset, candidate_mask=candidate_mask,
        )
        state_id = (
            f"sentinel-gradual-osa-z{stratum}-seed{seed}-tau{gamma:.1f}-step{step:02d}"
        )
        row, query_components = _evaluate_context(
            ctx, state_id=state_id,
            state_metadata={
                "panel": "gradual_access_sentinel", "surface": "osa",
                "source_seed": seed, "stratum": stratum, "gamma": gamma,
                "gate_mode": "latent", "protocol_scaffold": "start_low_expansion",
                "step": step, "phase": _phase(step), "prefix_n": prefix_n,
                "region_q": region_q,
                "construction": (
                    "Reconstructed from an existing tracked gradual-access cKG record."
                ),
            },
        )
        recorded_next = history[prefix_n].tolist()
        selected_512 = next(
            item["selected_index"] for item in row["approximations"]
            if item["name"] == "legacy_mc_512_seed0"
        )
        selected_dose = active_Xset[selected_512].detach().cpu().numpy().tolist()
        if selected_dose != recorded_next:
            raise RuntimeError(
                f"{state_id}: reconstructed 512 choice differs from recorded next cohort"
            )
        row["recorded_next_cohort_dose"] = recorded_next
        rows.append(row)
        components[state_id] = query_components
    return rows, components


def _sentinel_sobol_checks(rows: list[dict], components: dict, banks: dict):
    output = []
    for row in rows:
        exact_score = float(
            row["exact"]["raw_score_vector"][row["exact"]["selected_index"]]
        )
        checks = []
        for seed in SENTINEL_SOBOL_SCRAMBLES:
            estimate = _sobol_score(components[row["state_id"]], banks[seed])
            checks.append({
                "scramble_seed": seed, "estimate": estimate, "exact": exact_score,
                "error": float(estimate - exact_score),
                "absolute_error": float(abs(estimate - exact_score)),
            })
        output.append({
            "state_id": row["state_id"],
            "query_index": row["exact"]["selected_index"], "checks": checks,
        })
    return output


def _approximation_summary(states: list[dict], name: str):
    rows = [
        next(item for item in state["approximations"] if item["name"] == name)
        for state in states
    ]
    regrets = np.asarray([item["exact_regret_at_selected_query"] for item in rows])
    return {
        "selection_agreement_count": int(sum(
            item["selection_agrees_with_exact"] for item in rows
        )),
        "selection_agreement_denominator": len(rows),
        "selection_agreement_nonempty_gate_count": int(sum(
            item["selection_agrees_with_exact"]
            for state, item in zip(states, rows) if not state["gate_empty"]
        )),
        "selection_agreement_nonempty_gate_denominator": int(sum(
            not state["gate_empty"] for state in states
        )),
        "max_exact_regret": float(regrets.max()),
        "median_exact_regret": float(np.median(regrets)),
        "p95_exact_regret": float(np.quantile(regrets, 0.95)),
        "max_abs_score_error_full_grid": float(max(
            item["max_abs_score_error_full_grid"] for item in rows
        )),
        "median_rmse_score_error_full_grid": float(np.median([
            item["rmse_score_error_full_grid"] for item in rows
        ])),
    }


def _comparator_summary(states: list[dict]) -> dict:
    audits = [state["current_comparator_audit"] for state in states]
    return {
        "state_count": len(audits),
        "gate_pass_set_difference_state_count": int(sum(
            not row["gate_pass_set_matches"] for row in audits
        )),
        "eligible_gate_pass_set_difference_state_count": int(sum(
            not row["eligible_gate_pass_set_matches"] for row in audits
        )),
        "full_fallback_index_difference_state_count": int(sum(
            not row["operational_full_fallback_index_matches"] for row in audits
        )),
        "eligible_fallback_index_difference_state_count": int(sum(
            not row["operational_eligible_fallback_index_matches"] for row in audits
        )),
        "inactive_full_fallback_rounding_difference_state_count": int(sum(
            row["inactive_full_fallback_rounding_difference"] for row in audits
        )),
        "inactive_eligible_fallback_rounding_difference_state_count": int(sum(
            row["inactive_eligible_fallback_rounding_difference"] for row in audits
        )),
        "operational_full_fallback_rounding_difference_state_count": int(sum(
            row["operational_full_fallback_rounding_difference"] for row in audits
        )),
        "operational_eligible_fallback_rounding_difference_state_count": int(sum(
            row["operational_eligible_fallback_rounding_difference"] for row in audits
        )),
        "cdf_gate_pass_set_rounding_difference_state_count": int(sum(
            bool(row["cdf_gate_pass_set_difference_indices"]) for row in audits
        )),
        "cdf_eligible_gate_pass_set_rounding_difference_state_count": int(sum(
            bool(row["cdf_eligible_gate_pass_set_difference_indices"]) for row in audits
        )),
        "eligible_gate_empty_difference_state_count": int(sum(
            not row["eligible_gate_empty_matches"] for row in audits
        )),
        "minimum_abs_z_minus_q": float(min(
            row["minimum_abs_z_minus_q"] for row in audits
        )),
        "maximum_abs_standardized_feasibility": float(max(
            row["maximum_abs_standardized_feasibility"] for row in audits
        )),
        "cdf_saturated_zero_total": int(sum(
            row["cdf_saturated_zero_count"] for row in audits
        )),
        "cdf_saturated_one_total": int(sum(
            row["cdf_saturated_one_count"] for row in audits
        )),
    }


def _summarize(primary_states: list[dict], sobol_audit: dict,
               sentinel_states: list[dict], sentinel_sobol: list[dict]):
    sobol_errors = [
        check["absolute_error"]
        for state in sobol_audit["states"] for check in state["checks"]
    ]
    sentinel_errors = [
        check["absolute_error"]
        for state in sentinel_sobol for check in state["checks"]
    ]
    margins = [
        state["exact"]["margin_best_minus_runner_up"]
        for state in primary_states
        if state["exact"]["margin_best_minus_runner_up"] is not None
    ]
    primary_comparator = _comparator_summary(primary_states)
    sentinel_comparator = _comparator_summary(sentinel_states)
    mismatch_keys = (
        "gate_pass_set_difference_state_count",
        "eligible_gate_pass_set_difference_state_count",
        "full_fallback_index_difference_state_count",
        "eligible_fallback_index_difference_state_count",
        "eligible_gate_empty_difference_state_count",
    )
    if any(primary_comparator[key] or sentinel_comparator[key]
           for key in mismatch_keys):
        raise RuntimeError("current standardized/CDF comparator audit did not pass")
    return {
        "status": (
            "PASS_DETERMINISTIC_PIECEWISE_ANALYTIC_VALIDATION_"
            "EXACT_UP_TO_FLOATING_POINT_STABLE_Z_HARNESS_"
            "REGISTERED_CKG_EVALUATOR_UNCHANGED"
        ),
        "primary": {
            "state_count": len(primary_states),
            "trajectory_count": len({
                (row["source_seed"], row["stratum"]) for row in primary_states
            }),
            "steps_per_trajectory": 18, "full_score_vector_length": 25,
            "total_exact_query_evaluations": sum(
                row["accessible_grid_size"] for row in primary_states
            ),
            "source_512_decision_reconstruction_match_count": 720,
            "source_1024_decision_reconstruction_match_count": 720,
            "empty_current_gate_state_count": int(sum(
                row["gate_empty"] for row in primary_states
            )),
            "exact_top_two_margin": {
                "non_null_count": len(margins), "min": float(min(margins)),
                "median": float(np.median(margins)),
                "p95": float(np.quantile(margins, 0.95)), "max": float(max(margins)),
            },
            "exact_runtime_seconds_total": float(sum(
                row["exact"]["runtime_seconds"] for row in primary_states
            )),
            "exact_runtime_seconds_median_per_full_vector": float(np.median([
                row["exact"]["runtime_seconds"] for row in primary_states
            ])),
            "approximation_pair_runtime_seconds_total": float(sum(
                row["approximation_pair_runtime_seconds"] for row in primary_states
            )),
            "exact_gate_regime_count_total": int(sum(
                query["gate_regime_count"]
                for row in primary_states for query in row["exact"]["query_regimes"]
            )),
            "exact_empty_gate_regime_count_total": int(sum(
                query["empty_gate_regime_count"]
                for row in primary_states for query in row["exact"]["query_regimes"]
            )),
            "exact_fallback_segment_count_total": int(sum(
                query["fallback_segment_count"]
                for row in primary_states for query in row["exact"]["query_regimes"]
            )),
            "current_comparator_audit": primary_comparator,
            "approximations": {
                name: _approximation_summary(primary_states, name)
                for name in (
                    "legacy_mc_512_seed0",
                    "prefix_preserving_mc_1024_seed0",
                )
            },
        },
        "sobol_audit": {
            "state_count": sobol_audit["state_count"],
            "check_count": len(sobol_errors),
            "points_per_check": sobol_audit["points_per_scramble"],
            "coverage": sobol_audit["coverage"],
            "max_abs_error": float(max(sobol_errors)),
            "median_abs_error": float(np.median(sobol_errors)),
            "p95_abs_error": float(np.quantile(sobol_errors, 0.95)),
        },
        "sentinels": {
            "state_count": len(sentinel_states),
            "four_testbed_five_gate_state_count": int(sum(
                row["panel"] == "four_testbed_five_gate_sentinel"
                for row in sentinel_states
            )),
            "predictive_gate_state_count": int(sum(
                row["panel"] == "predictive_gate_sentinel"
                for row in sentinel_states
            )),
            "gradual_access_state_count": int(sum(
                row["panel"] == "gradual_access_sentinel"
                for row in sentinel_states
            )),
            "sobol_check_count": len(sentinel_errors),
            "sobol_max_abs_error": float(max(sentinel_errors)),
            "current_comparator_audit": sentinel_comparator,
        },
        "interpretation": (
            "This validates a deterministic piecewise-analytic evaluator, exact up to "
            "floating-point arithmetic, on frozen/reconstructed posterior states and "
            "numerical sentinels. It does not run formal OC trials, prove a Monte Carlo "
            "convergence rate, or authorize changing the registered cKG default."
        ),
    }


def _render_markdown(summary: dict) -> str:
    primary = summary["primary"]
    rows = [
        "# Deterministic piecewise-analytic cKG validation panel", "",
        f"Status: `{summary['status']}`.", "",
        (
            f"The primary panel reconstructs {primary['state_count']} posterior states "
            f"({primary['trajectory_count']} existing trajectories x "
            f"{primary['steps_per_trajectory']} decisions) and saves all "
            f"{primary['total_exact_query_evaluations']:,} piecewise-analytic query scores "
            "(exact up to floating-point arithmetic) together "
            "with the nested 512- and prefix-preserving 1024-fantasy vectors."
        ),
        (
            f"Both archived decision grids were reconstructed exactly "
            f"({primary['source_512_decision_reconstruction_match_count']}/720 for 512; "
            f"{primary['source_1024_decision_reconstruction_match_count']}/720 for 1024)."
        ),
        (
            "The current Context implements the stable standardized gate and "
            "full/eligible fallback at every audited state; historical CDF "
            "rounding differences are retained as diagnostics "
            f"across all {primary['current_comparator_audit']['state_count']} primary states."
        ), "",
        "| Approximation | Exact selection agreement | Nonempty-gate agreement | Max exact regret | Max full-grid score error |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in primary["approximations"].items():
        rows.append(
            f"| {name} | {item['selection_agreement_count']}/{item['selection_agreement_denominator']} "
            f"| {item['selection_agreement_nonempty_gate_count']}/{item['selection_agreement_nonempty_gate_denominator']} "
            f"| {item['max_exact_regret']:.6g} | {item['max_abs_score_error_full_grid']:.6g} |"
        )
    sobol = summary["sobol_audit"]
    sentinels = summary["sentinels"]
    rows.extend([
        "",
        (
            f"The stratified Sobol audit contains {sobol['state_count']} primary states and "
            f"{sobol['check_count']} independently scrambled checks "
            f"({sobol['points_per_check']:,} points each); maximum absolute error was "
            f"{sobol['max_abs_error']:.6g}."
        ),
        (
            f"The additive sentinel panel contains {sentinels['four_testbed_five_gate_state_count']} "
            "four-testbed/five-gate states, "
            f"{sentinels['predictive_gate_state_count']} predictive-gate states, and "
            f"{sentinels['gradual_access_state_count']} reconstructed gradual-access states."
        ), "", summary["interpretation"], "",
    ])
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    primary_payload, _primary_metadata, trajectories, decisions = _load_primary_source()
    _gradual_metadata, gradual_lookup = _load_gradual_records()
    started = time.perf_counter()
    primary_states, primary_components = _build_primary_panel(trajectories, decisions)
    banks = _sobol_banks()
    sobol_audit = _build_sobol_audit(primary_states, primary_components, banks)
    initial_sentinels, initial_components = _build_testbed_gate_and_predictive_sentinels()
    gradual_sentinels, gradual_components = _build_gradual_sentinels(gradual_lookup)
    sentinel_states = initial_sentinels + gradual_sentinels
    sentinel_components = {**initial_components, **gradual_components}
    sentinel_sobol = _sentinel_sobol_checks(
        sentinel_states, sentinel_components, banks
    )
    summary = _summarize(primary_states, sobol_audit, sentinel_states, sentinel_sobol)
    elapsed = time.perf_counter() - started

    panel = {
        "schema_version": 3,
        "artifact_class": "post_hoc_fixed_posterior_numerical_validation",
        "numerical_claim": (
            "deterministic piecewise-analytic; exact up to floating-point arithmetic"
        ),
        "status": (
            "piecewise_analytic_evaluator_validation_only_"
            "stable_z_harness_registered_ckg_evaluator_unchanged"
        ),
        "design": {
            "primary": {
                "surface": "osa", "tau": TAU, "gate_mode": "latent",
                "protocol_scaffold": "lhs_fixed full panel", "seeds": list(SEEDS),
                "strata": list(STRATA), "steps": list(STEPS),
                "cohort_size": R_K, "grid_n": GRID_N, "state_count": 720,
            },
            "approximations": {
                "512": "production-compatible legacy fantasy bank, seed 0",
                "1024": (
                    "custom prefix-preserving extension of the 512 bank, seed 0; "
                    "not conventional nmc=1024 and not a package default"
                ),
            },
            "sobol": {
                "subset_size": SOBOL_SUBSET_SIZE, "power": SOBOL_POWER,
                "scramble_seeds": list(SOBOL_SCRAMBLES),
            },
            "sentinels": {
                "surfaces": list(SURFACES), "gates": list(GATES),
                "predictive_gate": "one tau=0.7 initial-posterior state per surface",
                "gradual_states": [list(row) for row in GRADUAL_SENTINELS],
            },
        },
        "sources": {
            "primary": {
                "path": PRIMARY_SOURCE.name, "sha256": _sha256(PRIMARY_SOURCE),
                "metadata_sha256": _sha256(Path(str(PRIMARY_SOURCE) + ".metadata.json")),
                "source_status": primary_payload["status"],
            },
            "gradual": {
                "path": GRADUAL_SOURCE.name, "sha256": _sha256(GRADUAL_SOURCE),
                "metadata_sha256": _sha256(Path(str(GRADUAL_SOURCE) + ".metadata.json")),
            },
        },
        "implementation": {
            "generator": Path(__file__).name,
            "generator_sha256": _sha256(Path(__file__)),
            "exact_module": str(EXACT_SOURCE.relative_to(ROOT)),
            "exact_module_sha256": _sha256(EXACT_SOURCE),
            "legacy_pair_runner": "paper/run_ckg_1024_anchor.py",
            "legacy_pair_runner_sha256": _sha256(ROOT / "paper/run_ckg_1024_anchor.py"),
            "stable_harness_source_sha256": _stable_harness_source_sha256(),
            "production_default_changed": False,
            "production_harness_gate_semantics": "stable_z_strict_ppf",
        },
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "gpytorch": gpytorch.__version__,
            "threading": "single-thread numerical libraries",
        },
        "wall_clock_seconds_not_portable_benchmark": elapsed,
        "primary_states": primary_states, "sobol_audit": sobol_audit,
        "sentinel_states": sentinel_states,
        "sentinel_sobol_checks": sentinel_sobol, "summary": summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_dir / "ckg_piecewise_exact_panel.json"
    summary_json_path = args.output_dir / "ckg_piecewise_exact_summary.json"
    summary_md_path = args.output_dir / "ckg_piecewise_exact_summary.md"
    _write_json(panel_path, panel)
    _write_json(summary_json_path, summary)
    _write_text_atomic(summary_md_path, _render_markdown(summary))
    common_metadata = {
        "schema_version": 3, "artifact_type": "numerical_validation",
        "source_sha256": {
            PRIMARY_SOURCE.name: _sha256(PRIMARY_SOURCE),
            GRADUAL_SOURCE.name: _sha256(GRADUAL_SOURCE),
        },
        "generator": Path(__file__).name,
        "generator_sha256": _sha256(Path(__file__)),
        "exact_module": str(EXACT_SOURCE.relative_to(ROOT)),
        "exact_module_sha256": _sha256(EXACT_SOURCE),
        "stable_harness_source_sha256": _stable_harness_source_sha256(),
        "primary_state_count": len(primary_states),
        "sentinel_state_count": len(sentinel_states),
        "production_default_changed": False,
        "production_harness_gate_semantics": "stable_z_strict_ppf",
    }
    for artifact in (panel_path, summary_json_path, summary_md_path):
        _write_json(Path(str(artifact) + ".metadata.json"), {
            **common_metadata, "artifact": artifact.name,
            "artifact_sha256": _sha256(artifact),
        })
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
