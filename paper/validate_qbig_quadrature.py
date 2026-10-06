#!/usr/bin/env python3
"""Read-only numerical audit of the frozen qBIG entropy quadrature.

The registered ``qBIG`` acquisition evaluates a finite-panel, entropy-based
stepwise-uncertainty-reduction score with nine-node Gauss--Hermite quadrature.
This script does not alter that acquisition and does not generate new Monte
Carlo trials.  Instead, it authenticates the formal execution master,
reconstructs selected archived qBIG posterior states from their saved allocation
histories and seeds, and compares the frozen score with a 128-node reference.
The 20 largest candidate-level discrepancies are checked again with adaptive
quadrature.

The fixed audit design contains 432 posterior states and 10,800 candidate
scores.  Its intended conclusion is deliberately bounded: a decision-stable
representative audit is neither exact integration nor a universal guarantee.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import warnings

import gpytorch
import numpy as np
import scipy
from scipy import integrate, special, stats
import torch
import zstandard as zstd

from dose_combination_bo import acquisitions as frozen_acquisitions
from dose_combination_bo.acquisitions import qbig_value
from dose_combination_bo.gp import fit_gp, joint, post_latent
from dose_combination_bo.surfaces import resolve_surface
from dose_combination_bo.trial import grid


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
STATUS = "COMPLETE_QBIG_QUADRATURE_REPRESENTATIVE_AUDIT"
CONCLUSION = (
    "decision-stable representative audit, not exact integration or universal guarantee"
)

EXPECTED_CONFIGS = (
    ("efftox", 0.7),
    ("gbump", 0.5),
    ("gbump", 0.6),
    ("gbump", 0.7),
    ("gbump", 0.8),
    ("gbump", 0.9),
    ("mariposa", 0.7),
    ("osa", 0.5),
    ("osa", 0.6),
    ("osa", 0.7),
    ("osa", 0.8),
    ("osa", 0.9),
)
AUDIT_SEEDS = (0, 37, 99)
AUDIT_STRATA = (0, 1)
AUDIT_ENROLLMENTS = (4, 10, 16, 22, 30, 38)
REFERENCE_NODES = 128
ADAPTIVE_WORST_COUNT = 20
EXPECTED_STATES = (
    len(EXPECTED_CONFIGS)
    * len(AUDIT_SEEDS)
    * len(AUDIT_STRATA)
    * len(AUDIT_ENROLLMENTS)
)
EXPECTED_CANDIDATES_PER_STATE = 25
EXPECTED_CANDIDATE_SCORES = EXPECTED_STATES * EXPECTED_CANDIDATES_PER_STATE

STATE_FIELDS = (
    "surface",
    "gamma",
    "stratum",
    "seed",
    "enrollment_n",
    "candidate_count",
    "gate_count",
    "fallback_used",
    "state_max_absolute_score_error",
    "state_mean_absolute_score_error",
    "state_normalized_infinity_error",
    "spearman_all_candidates",
    "kendall_all_candidates",
    "strict_rank_reversals_all_candidates",
    "strict_rank_comparisons_all_candidates",
    "spearman_gate_candidates",
    "kendall_gate_candidates",
    "strict_rank_reversals_gate_candidates",
    "strict_rank_comparisons_gate_candidates",
    "raw_top_frozen9_index",
    "raw_top_reference128_index",
    "raw_top_mismatch",
    "operational_top_frozen9_index",
    "operational_top_reference128_index",
    "operational_top_mismatch",
    "reference_regret_at_frozen9_choice",
    "reference_top_rank_under_frozen9",
    "frozen9_top_rank_under_reference",
    "archived_next_dose_index",
    "archived_action_reconstructed",
)


class QbigQuadratureAuditError(RuntimeError):
    """Raised when input authentication or replay invariants fail."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def read_finite_json(path: str | Path) -> Any:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token}")

    try:
        return json.loads(Path(path).read_bytes(), parse_constant=reject)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise QbigQuadratureAuditError(f"invalid finite JSON: {path}") from exc


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise QbigQuadratureAuditError(
            f"{label} mismatch: observed={observed!r}, expected={expected!r}"
        )


def authenticate_formal_master(
    raw_path: str | Path,
    metadata_path: str | Path,
    archive_manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate and return the formal execution wrappers used by the audit."""

    raw_path = Path(raw_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    archive_manifest_path = Path(archive_manifest_path).resolve()
    for path in (raw_path, metadata_path, archive_manifest_path):
        if not path.is_file():
            raise QbigQuadratureAuditError(f"required audit input is absent: {path}")

    metadata = read_finite_json(metadata_path)
    manifest = read_finite_json(archive_manifest_path)
    if not isinstance(metadata, dict) or not isinstance(manifest, dict):
        raise QbigQuadratureAuditError("formal metadata/manifest must be JSON objects")
    _require_equal(
        manifest.get("status"),
        "COMPLETE_SELF_CONTAINED_FORMAL_EXECUTION_PROVENANCE",
        "archive manifest status",
    )
    raw_entries = manifest.get("raw_artifacts")
    if not isinstance(raw_entries, dict) or raw_path.name not in raw_entries:
        raise QbigQuadratureAuditError("formal raw input is not in the archive manifest")
    entry = raw_entries[raw_path.name]
    if not isinstance(entry, dict):
        raise QbigQuadratureAuditError("formal raw manifest entry is malformed")

    compressed = raw_path.read_bytes()
    _require_equal(len(compressed), entry.get("compressed_bytes"), "compressed bytes")
    _require_equal(
        sha256_bytes(compressed), entry.get("compressed_sha256"), "compressed SHA-256"
    )
    _require_equal(
        sha256_file(metadata_path),
        entry.get("source_metadata_sha256"),
        "formal metadata SHA-256",
    )
    try:
        logical = zstd.ZstdDecompressor().decompress(compressed)
    except zstd.ZstdError as exc:
        raise QbigQuadratureAuditError("formal master is not valid zstd") from exc
    _require_equal(len(logical), entry.get("uncompressed_bytes"), "uncompressed bytes")
    _require_equal(
        sha256_bytes(logical), entry.get("uncompressed_sha256"), "uncompressed SHA-256"
    )
    _require_equal(
        sha256_bytes(logical), metadata.get("artifact_sha256"), "metadata logical SHA-256"
    )
    _require_equal(len(logical), metadata.get("artifact_bytes"), "metadata logical bytes")
    _require_equal(metadata.get("status"), "COMPLETE_FORMAL_DECISION_MASTER", "metadata status")

    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token}")

    try:
        payload = json.loads(logical, parse_constant=reject)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise QbigQuadratureAuditError("formal master logical JSON is invalid") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise QbigQuadratureAuditError("formal master payload is malformed")
    wrappers = payload["rows"]
    _require_equal(len(wrappers), metadata.get("row_count"), "formal row count")
    _require_equal(payload.get("source_sha256"), metadata.get("source_sha256"), "source map")

    source_hashes = metadata.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise QbigQuadratureAuditError("formal source hash map is absent")
    for relative, expected_sha in sorted(source_hashes.items()):
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected_sha:
            raise QbigQuadratureAuditError(
                f"formal execution source is absent or changed: {relative}"
            )

    provenance = {
        "formal_raw": raw_path.name,
        "formal_raw_compressed_bytes": len(compressed),
        "formal_raw_compressed_sha256": sha256_bytes(compressed),
        "formal_raw_uncompressed_bytes": len(logical),
        "formal_raw_uncompressed_sha256": sha256_bytes(logical),
        "formal_metadata": metadata_path.name,
        "formal_metadata_sha256": sha256_file(metadata_path),
        "formal_archive_manifest": archive_manifest_path.name,
        "formal_archive_manifest_sha256": sha256_file(archive_manifest_path),
        "formal_manifest_sha256": metadata.get("manifest_sha256"),
        "projection_fingerprint": manifest.get("projection_fingerprint"),
        "formal_rows": len(wrappers),
        "formal_source_sha256": source_hashes,
        "local_formal_sources_verified": True,
    }
    return wrappers, provenance


def select_audit_records(
    wrappers: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, float, int, int], dict[str, Any]]:
    """Select the exact archived qBIG trial records required by the fixed design."""

    selected: dict[tuple[str, float, int, int], dict[str, Any]] = {}
    wanted_configs = set(EXPECTED_CONFIGS)
    for wrapper in wrappers:
        if wrapper.get("policy") != "qBIG":
            continue
        result = wrapper.get("result")
        if not isinstance(result, dict):
            continue
        if result.get("mode") != "latent" or result.get("protocol_scaffold") != "lhs_fixed":
            continue
        config = (str(result.get("sim")), float(result.get("gamma")))
        if config not in wanted_configs:
            continue
        seed = int(result.get("seed"))
        stratum = int(result.get("stratum"))
        if seed not in AUDIT_SEEDS or stratum not in AUDIT_STRATA:
            continue
        key = (*config, stratum, seed)
        if key in selected:
            raise QbigQuadratureAuditError(f"duplicate selected formal qBIG row: {key}")
        for field, expected in {
            "policy": "qBIG",
            "noise": "fixed",
            "kap": 1.0,
            "budget": 40,
            "warmup": 4,
            "r_k": 2,
            "grid_n": 5,
            "empty_gate": "pf",
        }.items():
            _require_equal(result.get(field), expected, f"{key} {field}")
        if wrapper.get("seed") != seed or wrapper.get("stratum") != stratum:
            raise QbigQuadratureAuditError(f"wrapper/result identity differs for {key}")
        selected[key] = dict(result)

    expected_keys = {
        (sim, gamma, stratum, seed)
        for sim, gamma in EXPECTED_CONFIGS
        for stratum in AUDIT_STRATA
        for seed in AUDIT_SEEDS
    }
    if set(selected) != expected_keys:
        example = next(iter((expected_keys - set(selected)) or (set(selected) - expected_keys)))
        raise QbigQuadratureAuditError(
            f"fixed audit trial selection is incomplete or changed; example={example!r}"
        )
    return selected


def reconstruct_toxicity_observations(
    record: Mapping[str, Any],
) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray, float]:
    """Reconstruct archived observations without executing a new trial policy."""

    surface = resolve_surface(str(record["sim"]), int(record["stratum"]))
    history = np.asarray(record.get("allocation_history"), dtype=float)
    if history.shape != (40, 2):
        raise QbigQuadratureAuditError("archived qBIG allocation history is not 40 by 2")
    rng = np.random.default_rng(int(record["seed"]))
    sf = float(surface["sf"])
    sg = float(surface["sg"])
    warmup = int(record["warmup"])
    cuts = np.linspace(0.0, 1.0, warmup + 1)
    toxicity: list[float] = []
    for index, dose in enumerate(history):
        if index < warmup:
            generated = np.asarray(
                [
                    cuts[index] + rng.uniform() * (cuts[1] - cuts[0]),
                    rng.uniform(),
                ]
            )
            if not np.array_equal(generated, dose):
                raise QbigQuadratureAuditError(
                    f"warmup RNG replay differs at index {index}: {generated} versus {dose}"
                )
        # Consume both response-channel innovations in their archived order.
        _ = float(surface["eff"](*dose) + rng.normal(0.0, sf))
        toxicity.append(float(surface["tox"](*dose) + rng.normal(0.0, sg)))
    for start in range(warmup, len(history), int(record["r_k"])):
        cohort = history[start : start + int(record["r_k"])]
        if cohort.shape[0] == int(record["r_k"]) and not np.all(cohort == cohort[0]):
            raise QbigQuadratureAuditError("archived adaptive cohort contains different doses")
    return surface, history, np.asarray(toxicity), sg


def bernoulli_entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    return -(
        probability * np.log(probability)
        + (1.0 - probability) * np.log(1.0 - probability)
    )


def score_parameters(
    model: Any,
    likelihood: Any,
    panel: torch.Tensor,
    query: torch.Tensor,
    threshold: float,
    cohort_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return current entropy and normal-update parameters for one query dose."""

    mean, variance, covariance, query_variance_with_noise = joint(
        model, likelihood, panel, query
    )
    noise = float(likelihood.noise.item())
    fantasy_variance = (
        float(query_variance_with_noise) - noise + noise / int(cohort_size)
    )
    mean_np = mean.cpu().numpy()
    variance_np = variance.cpu().numpy()
    covariance_np = covariance.cpu().numpy()
    current_sd = np.sqrt(variance_np)
    residual = np.maximum(
        variance_np - covariance_np**2 / fantasy_variance,
        1e-12,
    )
    posterior_sd = np.sqrt(residual)
    current_entropy = bernoulli_entropy(
        stats.norm.cdf((float(threshold) - mean_np) / current_sd)
    )
    zeta = (float(threshold) - mean_np) / posterior_sd
    update_scale = (
        np.abs(covariance_np) / np.sqrt(fantasy_variance) / posterior_sd
    )
    return current_entropy, zeta, update_scale


def quadrature_score(
    current_entropy: np.ndarray,
    zeta: np.ndarray,
    update_scale: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    expected_entropy = np.sum(
        np.asarray(weights)[:, None]
        * bernoulli_entropy(
            stats.norm.cdf(
                np.asarray(zeta)[None, :]
                + np.asarray(update_scale)[None, :] * np.asarray(nodes)[:, None]
            )
        ),
        axis=0,
    )
    return float(np.sum(current_entropy - expected_entropy))


def adaptive_score(
    current_entropy: np.ndarray,
    zeta: np.ndarray,
    update_scale: np.ndarray,
) -> tuple[float, float]:
    """Adaptive reference over [-10,10]; omitted normal tail is negligible here."""

    def scalar_entropy(probability: float) -> float:
        probability = min(max(float(probability), 1e-12), 1.0 - 1e-12)
        return -(
            probability * math.log(probability)
            + (1.0 - probability) * math.log(1.0 - probability)
        )

    expected = []
    error_bounds = []
    for center, scale in zip(zeta, update_scale):
        value, error = integrate.quad(
            lambda x: scalar_entropy(stats.norm.cdf(center + scale * x))
            * stats.norm.pdf(x),
            -10.0,
            10.0,
            epsabs=2e-13,
            epsrel=2e-13,
            limit=300,
        )
        expected.append(value)
        error_bounds.append(error)
    return float(np.sum(current_entropy) - np.sum(expected)), float(sum(error_bounds))


def _finite_correlation(left: np.ndarray, right: np.ndarray, method: str) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.spearmanr(left, right) if method == "spearman" else stats.kendalltau(left, right)
    value = float(result.statistic)
    return value if np.isfinite(value) else None


def strict_rank_reversals(
    left: np.ndarray, right: np.ndarray, indices: Iterable[int] | None = None
) -> tuple[int, int]:
    """Count strict pairwise order reversals, excluding ties in either score."""

    chosen = list(range(len(left))) if indices is None else list(indices)
    reversals = 0
    comparisons = 0
    for offset, first in enumerate(chosen):
        for second in chosen[offset + 1 :]:
            delta_left = float(left[first] - left[second])
            delta_right = float(right[first] - right[second])
            if delta_left == 0.0 or delta_right == 0.0:
                continue
            comparisons += 1
            reversals += int(delta_left * delta_right < 0.0)
    return reversals, comparisons


def descending_min_rank(values: np.ndarray, index: int) -> int:
    return int(stats.rankdata(-np.asarray(values), method="min")[int(index)])


def quantile_summary(values: Iterable[float | int | None]) -> dict[str, float] | None:
    array = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=float,
    )
    if array.size == 0:
        return None
    probabilities = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
    labels = ("min", "median", "p90", "p95", "p99", "max")
    return {
        label: float(np.quantile(array, probability))
        for label, probability in zip(labels, probabilities)
    }


def _panel_index(panel: np.ndarray, dose: np.ndarray) -> int:
    distances = np.linalg.norm(panel - np.asarray(dose, dtype=float), axis=1)
    index = int(np.argmin(distances))
    if not np.array_equal(panel[index], np.asarray(dose, dtype=float)):
        raise QbigQuadratureAuditError(f"archived dose is not on the panel: {dose}")
    return index


def run_audit(
    wrappers: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the fixed read-only posterior-state audit and return a finite payload."""

    if len(frozen_acquisitions._QX) != 9 or len(frozen_acquisitions._QW) != 9:
        raise QbigQuadratureAuditError("frozen qBIG no longer uses nine quadrature nodes")
    selected = select_audit_records(wrappers)
    reference_nodes, reference_weights = special.roots_hermitenorm(REFERENCE_NODES)
    reference_weights = reference_weights / np.sqrt(2.0 * np.pi)

    state_rows: list[dict[str, Any]] = []
    candidate_errors: list[dict[str, Any]] = []
    all_candidate_absolute_errors: list[float] = []
    for simulation, gamma in EXPECTED_CONFIGS:
        for stratum in AUDIT_STRATA:
            for seed in AUDIT_SEEDS:
                record = selected[(simulation, gamma, stratum, seed)]
                surface, history, toxicity, toxicity_sd = reconstruct_toxicity_observations(record)
                panel = torch.tensor(grid(int(record["grid_n"])))
                panel_np = panel.cpu().numpy()
                threshold = float(surface["gd"])
                gate_quantile = float(stats.norm.ppf(float(gamma)))
                for enrollment in AUDIT_ENROLLMENTS:
                    model, likelihood = fit_gp(
                        torch.tensor(history[:enrollment]),
                        torch.tensor(toxicity[:enrollment]),
                        fixed_noise=toxicity_sd**2,
                    )
                    frozen_scores: list[float] = []
                    reference_scores: list[float] = []
                    parameters: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
                    for query in panel:
                        frozen_score = qbig_value(
                            query,
                            panel,
                            model,
                            likelihood,
                            threshold,
                            "latent",
                            int(record["r_k"]),
                        )
                        parameter = score_parameters(
                            model,
                            likelihood,
                            panel,
                            query,
                            threshold,
                            int(record["r_k"]),
                        )
                        reference_score = quadrature_score(
                            *parameter, reference_nodes, reference_weights
                        )
                        frozen_scores.append(frozen_score)
                        reference_scores.append(reference_score)
                        parameters.append(parameter)
                    frozen_array = np.asarray(frozen_scores)
                    reference_array = np.asarray(reference_scores)
                    absolute_error = np.abs(frozen_array - reference_array)
                    all_candidate_absolute_errors.extend(absolute_error.tolist())

                    posterior_mean, posterior_variance = post_latent(model, panel)
                    standardized_feasibility = (
                        threshold - posterior_mean.cpu().numpy()
                    ) / np.sqrt(posterior_variance.cpu().numpy())
                    gate = standardized_feasibility > gate_quantile
                    gate_indices = np.flatnonzero(gate)
                    fallback = gate_indices.size == 0

                    raw_frozen_top = int(np.argmax(frozen_array))
                    raw_reference_top = int(np.argmax(reference_array))
                    if fallback:
                        operational_frozen_top = int(np.argmax(standardized_feasibility))
                        operational_reference_top = operational_frozen_top
                    else:
                        operational_frozen_top = int(
                            np.argmax(np.where(gate, frozen_array, -np.inf))
                        )
                        operational_reference_top = int(
                            np.argmax(np.where(gate, reference_array, -np.inf))
                        )
                    archived_next = _panel_index(panel_np, history[enrollment])
                    action_matches = operational_frozen_top == archived_next
                    if not action_matches:
                        raise QbigQuadratureAuditError(
                            "frozen score failed to reconstruct archived qBIG action: "
                            f"{simulation}, gamma={gamma}, z={stratum}, seed={seed}, "
                            f"n={enrollment}, computed={operational_frozen_top}, "
                            f"archived={archived_next}"
                        )

                    all_reversals, all_comparisons = strict_rank_reversals(
                        frozen_array, reference_array
                    )
                    gate_reversals, gate_comparisons = strict_rank_reversals(
                        frozen_array, reference_array, gate_indices
                    )
                    scale = max(float(np.max(np.abs(reference_array))), 1e-12)
                    state = {
                        "surface": simulation,
                        "gamma": float(gamma),
                        "stratum": int(stratum),
                        "seed": int(seed),
                        "enrollment_n": int(enrollment),
                        "candidate_count": int(panel.shape[0]),
                        "gate_count": int(gate_indices.size),
                        "fallback_used": bool(fallback),
                        "state_max_absolute_score_error": float(np.max(absolute_error)),
                        "state_mean_absolute_score_error": float(np.mean(absolute_error)),
                        "state_normalized_infinity_error": float(
                            np.max(absolute_error) / scale
                        ),
                        "spearman_all_candidates": _finite_correlation(
                            frozen_array, reference_array, "spearman"
                        ),
                        "kendall_all_candidates": _finite_correlation(
                            frozen_array, reference_array, "kendall"
                        ),
                        "strict_rank_reversals_all_candidates": int(all_reversals),
                        "strict_rank_comparisons_all_candidates": int(all_comparisons),
                        "spearman_gate_candidates": _finite_correlation(
                            frozen_array[gate_indices],
                            reference_array[gate_indices],
                            "spearman",
                        ),
                        "kendall_gate_candidates": _finite_correlation(
                            frozen_array[gate_indices],
                            reference_array[gate_indices],
                            "kendall",
                        ),
                        "strict_rank_reversals_gate_candidates": int(gate_reversals),
                        "strict_rank_comparisons_gate_candidates": int(gate_comparisons),
                        "raw_top_frozen9_index": raw_frozen_top,
                        "raw_top_reference128_index": raw_reference_top,
                        "raw_top_mismatch": bool(raw_frozen_top != raw_reference_top),
                        "operational_top_frozen9_index": operational_frozen_top,
                        "operational_top_reference128_index": operational_reference_top,
                        "operational_top_mismatch": bool(
                            operational_frozen_top != operational_reference_top
                        ),
                        "reference_regret_at_frozen9_choice": (
                            0.0
                            if fallback
                            else float(
                                reference_array[operational_reference_top]
                                - reference_array[operational_frozen_top]
                            )
                        ),
                        "reference_top_rank_under_frozen9": descending_min_rank(
                            frozen_array, raw_reference_top
                        ),
                        "frozen9_top_rank_under_reference": descending_min_rank(
                            reference_array, raw_frozen_top
                        ),
                        "archived_next_dose_index": archived_next,
                        "archived_action_reconstructed": True,
                    }
                    state_rows.append(state)
                    for candidate_index, candidate_error in enumerate(absolute_error):
                        current_entropy, zeta, update_scale = parameters[candidate_index]
                        candidate_errors.append(
                            {
                                "absolute_error": float(candidate_error),
                                "surface": simulation,
                                "gamma": float(gamma),
                                "stratum": int(stratum),
                                "seed": int(seed),
                                "enrollment_n": int(enrollment),
                                "candidate_index": int(candidate_index),
                                "dose_d1": float(panel_np[candidate_index, 0]),
                                "dose_d2": float(panel_np[candidate_index, 1]),
                                "frozen9_score": float(frozen_array[candidate_index]),
                                "reference128_score": float(
                                    reference_array[candidate_index]
                                ),
                                "current_entropy": current_entropy,
                                "zeta": zeta,
                                "update_scale": update_scale,
                            }
                        )

    if len(state_rows) != EXPECTED_STATES:
        raise QbigQuadratureAuditError(
            f"audit reconstructed {len(state_rows)} states, expected {EXPECTED_STATES}"
        )
    if len(candidate_errors) != EXPECTED_CANDIDATE_SCORES:
        raise QbigQuadratureAuditError(
            "audit candidate-score count changed: "
            f"{len(candidate_errors)} versus {EXPECTED_CANDIDATE_SCORES}"
        )

    candidate_errors.sort(
        key=lambda row: (
            -float(row["absolute_error"]),
            str(row["surface"]),
            float(row["gamma"]),
            int(row["stratum"]),
            int(row["seed"]),
            int(row["enrollment_n"]),
            int(row["candidate_index"]),
        )
    )
    adaptive_checks: list[dict[str, Any]] = []
    for candidate in candidate_errors[:ADAPTIVE_WORST_COUNT]:
        adaptive, error_bound = adaptive_score(
            candidate["current_entropy"],
            candidate["zeta"],
            candidate["update_scale"],
        )
        adaptive_checks.append(
            {
                key: candidate[key]
                for key in (
                    "surface",
                    "gamma",
                    "stratum",
                    "seed",
                    "enrollment_n",
                    "candidate_index",
                    "dose_d1",
                    "dose_d2",
                    "frozen9_score",
                    "reference128_score",
                )
            }
            | {
                "adaptive_score": adaptive,
                "absolute_frozen9_minus_reference128": float(
                    candidate["absolute_error"]
                ),
                "absolute_reference128_minus_adaptive": abs(
                    float(candidate["reference128_score"]) - adaptive
                ),
                "absolute_frozen9_minus_adaptive": abs(
                    float(candidate["frozen9_score"]) - adaptive
                ),
                "adaptive_reported_error_bound_sum": error_bound,
            }
        )

    state_rows.sort(
        key=lambda row: (
            str(row["surface"]),
            float(row["gamma"]),
            int(row["stratum"]),
            int(row["seed"]),
            int(row["enrollment_n"]),
        )
    )
    nonempty = [row for row in state_rows if not row["fallback_used"]]
    fallback = [row for row in state_rows if row["fallback_used"]]
    by_cell: list[dict[str, Any]] = []
    for simulation, gamma in EXPECTED_CONFIGS:
        cells = [
            row
            for row in state_rows
            if row["surface"] == simulation and row["gamma"] == gamma
        ]
        by_cell.append(
            {
                "surface": simulation,
                "gamma": gamma,
                "states": len(cells),
                "nonempty_gate_states": sum(not row["fallback_used"] for row in cells),
                "fallback_states": sum(row["fallback_used"] for row in cells),
                "maximum_absolute_score_error": max(
                    row["state_max_absolute_score_error"] for row in cells
                ),
                "raw_top_mismatches": sum(row["raw_top_mismatch"] for row in cells),
                "operational_top_mismatches": sum(
                    row["operational_top_mismatch"] for row in cells
                ),
                "archived_action_reconstruction_failures": sum(
                    not row["archived_action_reconstructed"] for row in cells
                ),
            }
        )

    summary = {
        "design": {
            "surface_gate_configurations": [
                {"surface": simulation, "gamma": gamma}
                for simulation, gamma in EXPECTED_CONFIGS
            ],
            "strata": list(AUDIT_STRATA),
            "archived_seeds": list(AUDIT_SEEDS),
            "enrollment_states": list(AUDIT_ENROLLMENTS),
            "posterior_states": len(state_rows),
            "candidate_scores": len(candidate_errors),
            "candidates_per_state": EXPECTED_CANDIDATES_PER_STATE,
            "new_monte_carlo_trials_generated": False,
            "archived_posterior_state_reconstruction": True,
        },
        "quadrature": {
            "frozen_method": "9-node Gauss-Hermite expected finite-panel Bernoulli entropy reduction",
            "reference_method": "128-node scipy.special.roots_hermitenorm",
            "adaptive_method": "scipy.integrate.quad over [-10,10] for the 20 largest candidate-level discrepancies",
            "adaptive_worst_candidate_checks": len(adaptive_checks),
            "terminology": "entropy-based finite-panel SUR approximation",
            "exact_integration_claim": False,
        },
        "score_error": {
            "candidate_absolute_error": quantile_summary(all_candidate_absolute_errors),
            "state_max_absolute_error": quantile_summary(
                row["state_max_absolute_score_error"] for row in state_rows
            ),
            "state_mean_absolute_error": quantile_summary(
                row["state_mean_absolute_score_error"] for row in state_rows
            ),
            "state_normalized_infinity_error": quantile_summary(
                row["state_normalized_infinity_error"] for row in state_rows
            ),
            "maximum_absolute_reference128_minus_adaptive": max(
                row["absolute_reference128_minus_adaptive"]
                for row in adaptive_checks
            ),
            "maximum_absolute_frozen9_minus_adaptive": max(
                row["absolute_frozen9_minus_adaptive"] for row in adaptive_checks
            ),
        },
        "rank_discrepancy": {
            "spearman_all_candidates": quantile_summary(
                row["spearman_all_candidates"] for row in state_rows
            ),
            "kendall_all_candidates": quantile_summary(
                row["kendall_all_candidates"] for row in state_rows
            ),
            "undefined_all_candidate_correlations": sum(
                row["spearman_all_candidates"] is None for row in state_rows
            ),
            "total_strict_rank_reversals_all_candidates": sum(
                row["strict_rank_reversals_all_candidates"] for row in state_rows
            ),
            "total_strict_rank_comparisons_all_candidates": sum(
                row["strict_rank_comparisons_all_candidates"] for row in state_rows
            ),
            "total_strict_rank_reversals_gate_candidates": sum(
                row["strict_rank_reversals_gate_candidates"] for row in nonempty
            ),
            "total_strict_rank_comparisons_gate_candidates": sum(
                row["strict_rank_comparisons_gate_candidates"] for row in nonempty
            ),
        },
        "top_choice": {
            "raw_score_states": len(state_rows),
            "raw_top_mismatches": sum(row["raw_top_mismatch"] for row in state_rows),
            "nonempty_gate_states": len(nonempty),
            "operational_top_mismatches_nonempty_gate": sum(
                row["operational_top_mismatch"] for row in nonempty
            ),
            "fallback_states": len(fallback),
            "operational_top_mismatches_fallback": sum(
                row["operational_top_mismatch"] for row in fallback
            ),
            "maximum_reference_regret_at_frozen9_choice": max(
                row["reference_regret_at_frozen9_choice"] for row in state_rows
            ),
            "archived_actions_checked": len(state_rows),
            "archived_action_reconstruction_failures": sum(
                not row["archived_action_reconstructed"] for row in state_rows
            ),
        },
        "by_surface_gate": by_cell,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "conclusion": CONCLUSION,
        "scope": (
            "Read-only replay of a fixed representative subset of authenticated archived "
            "qBIG posterior states; no claim covers every archived or future state."
        ),
        "provenance": dict(provenance)
        | {
            "validator_source": "paper/validate_qbig_quadrature.py",
            "validator_source_sha256": sha256_file(Path(__file__)),
            "runtime_identity": {
                "python": ".".join(map(str, __import__("sys").version_info[:3])),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
                "gpytorch": gpytorch.__version__,
                "zstandard": zstd.__version__,
            },
        },
        "summary": summary,
        "adaptive_worst_candidate_checks": adaptive_checks,
        "states": state_rows,
    }
    # The canonical serializer is also the final finite-value guard.
    canonical_json_bytes(payload)
    return payload


def write_artifacts(payload: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "qbig_quadrature_representative_audit.json"
    csv_path = output_dir / "qbig_quadrature_representative_audit_states.csv"
    commit_path = output_dir / "qbig_quadrature_representative_audit.commit.json"
    json_path.write_bytes(canonical_json_bytes(payload))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(STATE_FIELDS),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(payload["states"])
    commit = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMMITTED_QBIG_QUADRATURE_REPRESENTATIVE_AUDIT",
        "conclusion": CONCLUSION,
        "validator_source": "paper/validate_qbig_quadrature.py",
        "validator_source_sha256": sha256_file(Path(__file__)),
        "artifacts": {
            json_path.name: {
                "bytes": json_path.stat().st_size,
                "sha256": sha256_file(json_path),
            },
            csv_path.name: {
                "bytes": csv_path.stat().st_size,
                "sha256": sha256_file(csv_path),
                "rows_excluding_header": len(payload["states"]),
            },
        },
    }
    commit_path.write_bytes(canonical_json_bytes(commit))
    return commit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-raw", required=True)
    parser.add_argument("--formal-metadata", required=True)
    parser.add_argument("--archive-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    torch.set_default_dtype(torch.double)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    wrappers, provenance = authenticate_formal_master(
        args.formal_raw, args.formal_metadata, args.archive_manifest
    )
    payload = run_audit(wrappers, provenance)
    commit = write_artifacts(payload, args.output_dir)
    print(
        "qBIG quadrature audit complete: "
        f"{payload['summary']['design']['posterior_states']} states, "
        f"{payload['summary']['design']['candidate_scores']} candidate scores, "
        f"commit={sha256_bytes(canonical_json_bytes(commit))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
