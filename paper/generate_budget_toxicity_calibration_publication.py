#!/usr/bin/env python3
"""Build the outcome-independent publication projection for the calibration audit.

This adapter reads only eight authenticated, committed analyzer files.  It does
not read raw shards, checkpoints, precision-monitor artifacts, or trial
histories.  All 118 locked primary estimands are projected without filtering by
estimate, sign, interval, or apparent winner.  Raw uncertainty fields remain
visible, while every scientific classification, predicate, table, and figure
uses the prespecified numerically guarded interval.
The complete frozen 36-row policy-vs-cEI calibration difference-in-differences
projection is also published without sign-based selection; its statements are
cell-specific and never aggregated by a vote or promoted to a family theorem.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "paper/budget_toxicity_calibration_publication_contract.json"
DEFAULT_OUTPUT = ROOT / "results/budget_toxicity_calibration_publication"
ACTIVE_STAGING = ROOT / "results/budget_toxicity_calibration_staging"
TRUSTED_CONTRACT_SHA256 = (
    "a1904b435d4e42ef6df25094bc1f3ce35156ceaec5bf157ec24fb2ff2827b6a8"
)
TRUSTED_COMMITMENTS = {
    "analyzer_sha256": "bcf3ec2964b88d61ac4becc8a9aa838831a3240d93a082abf42b5b328799ec2a",
    "prespec_sha256": "1b1152d8f09462b96bf2f35252a9cd5703b331ab7a3546d4708b189f81e8c178",
    "manifest_sha256": "2e189ff6426bd3ad12fd29896fb6f76027a0d9e4424013d92edd0df3d143d3b6",
}

INPUT_NAMES = (
    "budget_toxicity_calibration_analysis.metadata.json",
    "budget_toxicity_calibration_analysis.commit.json",
    "budget_toxicity_calibration_primary.json",
    "budget_toxicity_calibration_primary.json.metadata.json",
    "budget_toxicity_calibration_secondary.json",
    "budget_toxicity_calibration_secondary.json.metadata.json",
    "budget_toxicity_calibration_baseline_identity.json",
    "budget_toxicity_calibration_baseline_identity.json.metadata.json",
)
PRIMARY_NAME = "budget_toxicity_calibration_primary.json"
SECONDARY_NAME = "budget_toxicity_calibration_secondary.json"
BASELINE_NAME = "budget_toxicity_calibration_baseline_identity.json"
ANALYSIS_METADATA_NAME = "budget_toxicity_calibration_analysis.metadata.json"
ANALYSIS_COMMIT_NAME = "budget_toxicity_calibration_analysis.commit.json"

ANALYSIS_STATUS = "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_ANALYSIS"
ANALYSIS_COMMIT_STATUS = "COMMITTED_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_ANALYSIS"
ANALYZER_ARTIFACT_STATUS = (
    "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_ANALYSIS_ARTIFACT"
)
BASELINE_STATUS = "PASS_AVAILABLE_FIELD_IDENTITY"
PUBLICATION_STATUS = "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_PUBLICATION"
PUBLICATION_COMMIT_STATUS = "COMMITTED_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_PUBLICATION"
PUBLICATION_ARTIFACT_STATUS = (
    "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_PUBLICATION_ARTIFACT"
)
PUBLICATION_ARTIFACT_CLASS = (
    "outcome_independent_budget_toxicity_calibration_publication_projection"
)

POLICIES = ("cKG-exact-formal", "tmse", "qBIG", "cEI")
COMPARATORS = ("tmse", "qBIG", "cEI")
PRINCIPAL_COMPARATORS = ("tmse", "qBIG")
DID_POLICIES = ("cKG-exact-formal", "tmse", "qBIG")
FACTORS = (0.5, 1.0, 2.0)
ERRONEOUS_FACTORS = (0.5, 2.0)
BUDGETS = (20, 40, 80)
ELIGIBLE_M = (200, 500, 1000)

TERMINAL = "terminal_true_boundary_exceedance_pct"
ASSIGNMENT = "above_boundary_assignment_pct"
EXCLUSION = "terminal_target_exclusion_pct"
BRIER = "panel_brier_score_pct"
PRIMARY_COUNTS = {"A": 54, "B": 48, "C": 16}
SECONDARY_COUNTS = {
    "absolute": 864,
    "contrast_and_conditional": 363,
    "policy_vs_cei_calibration_did": 36,
}
MAX_T_SEEDS = {"A": 202_608_261, "B": 202_608_262, "C": 202_608_263}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SAFE_STEM_RE = re.compile(r"[A-Za-z0-9_-]+\Z")

NUMERICAL_ETA = 2.0 ** -26
PANEL_BRIER_SNAPSHOT_BOUND_POINTS = 200.0 * NUMERICAL_ETA
FAMILY_B_BRIER_BOUND_POINTS = 400.0 * NUMERICAL_ETA
FAMILY_C_BRIER_BOUND_POINTS = 800.0 * NUMERICAL_ETA
GUARD_CLASS_EXACT_DISCRETE = "exact_discrete_primary_contribution"
GUARD_CLASS_FAMILY_B_BRIER = "family_B_panel_brier_difference"
GUARD_CLASS_FAMILY_C_BRIER = "family_C_panel_brier_difference_in_differences"

SECONDARY_ANALYSIS_ABSOLUTE_PROFILE = "absolute_equal_gate_stratum_profile"
SECONDARY_ANALYSIS_BUDGET_CONTRAST = "absolute_budget_contrast_N80_minus_N20"
SECONDARY_ANALYSIS_CALIBRATION_DID = (
    "policy_minus_cEI_calibration_difference_in_differences"
)
SECONDARY_GUARD_CLASS_NON_BRIER_ZERO = "secondary_non_brier_zero_guard"
SECONDARY_GUARD_CLASS_ABSOLUTE_BRIER = "secondary_absolute_panel_brier_snapshot"
SECONDARY_GUARD_CLASS_BUDGET_BRIER = (
    "secondary_within_policy_N80_minus_N20_panel_brier_contrast"
)
SECONDARY_GUARD_CLASS_CALIBRATION_DID_BRIER = (
    "secondary_policy_vs_cEI_panel_brier_calibration_did"
)

ABSOLUTE_METRICS = (
    TERMINAL,
    "above_boundary_assignment_count",
    ASSIGNMENT,
    "post_initialization_above_boundary_assignment_pct",
    EXCLUSION,
    "target_ever_admitted_pct",
    "target_finally_admitted_pct",
    "target_never_admitted_pct",
    "target_admitted_then_excluded_pct",
    "target_exclusion_occupancy_pct",
    "target_terminal_exclusion_run_fraction_pct",
    "target_recovered_after_preterminal_exclusion_pct",
    "target_admitted_but_not_selected_pct",
    BRIER,
    "latent_95_interval_coverage_pct",
    "target_latent_95_interval_coverage_pct",
    "mean_latent_95_interval_width",
    "panel_correct_selection_pct",
    "recommendation_grid_distance",
    "true_recommended_efficacy",
    "final_gate_size",
    "final_empty_gate_pct",
    "false_admissions",
    "feasible_dose_recall_pct",
)

ANALYZER_POLICY_LABELS = {
    "cKG-exact-formal": "cKG",
    "tmse": "tMSE-only",
    "qBIG": "Entropy reduction",
    "cEI": "cEI",
}
EXPECTED_READER_LABELS = {
    "audit_scope": "Finite fixed-design audit",
    "metric_terminal": "True-boundary-exceeding terminal recommendation (%)",
    "metric_assignment": "Above-boundary assignments / N (%)",
    "metric_exclusion": "True panel OBD excluded from terminal gate (%)",
    "metric_brier": "Panel feasibility Brier score ×100",
    "c_sigma_axis": "Assumed / true toxicity observation-noise SD, c_sigma",
    "x_budget": "Per-stratum simulated enrollment N",
    "difference_axis": "Difference (percentage points)",
}
METRIC_LABEL_KEYS = {
    TERMINAL: "metric_terminal",
    ASSIGNMENT: "metric_assignment",
    EXCLUSION: "metric_exclusion",
    BRIER: "metric_brier",
}


class PublicationAdapterError(RuntimeError):
    """Raised when authentication or the frozen publication contract fails."""


@dataclass(frozen=True)
class EstimandSpec:
    family: str
    estimand_id: str
    policy: str
    comparator: str
    metric: str
    budget_n: int | None
    budget_contrast: str
    factor: float
    reference_factor: float | None


@dataclass(frozen=True)
class AuthenticatedBundle:
    analysis_dir: Path
    input_paths: Mapping[str, Path]
    input_hashes: Mapping[str, str]
    input_bytes: Mapping[str, int]
    analysis_metadata: Mapping[str, Any]
    analysis_commit: Mapping[str, Any]
    primary: Mapping[str, Any]
    secondary: Mapping[str, Any]
    baseline: Mapping[str, Any]
    final_m: int
    primary_rows: tuple[Mapping[str, Any], ...]
    absolute_rows: tuple[Mapping[str, Any], ...]
    secondary_contrast_rows: tuple[Mapping[str, Any], ...]
    policy_cei_did_rows: tuple[Mapping[str, Any], ...]
    primary_multiplicity: Mapping[str, Any]
    secondary_guard_metadata: Mapping[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _analyzer_canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _assert_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PublicationAdapterError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, label=f"{label}[{index}]")


def _decode_json(raw: bytes, *, label: str) -> Any:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token}")

    try:
        value = json.loads(raw, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PublicationAdapterError(f"{label} is not finite valid JSON") from exc
    _assert_finite(value, label=label)
    return value


def _absolute_without_symlink_resolution(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def _symlink_component(path: Path) -> Path | None:
    """Return the first existing symlink component without following it."""

    absolute = _absolute_without_symlink_resolution(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            return current
    return None


def _guard_untrusted_input_path(path: str | Path, *, label: str) -> Path:
    """Reject active-staging and symlink paths before reading any bytes."""

    lexical = _absolute_without_symlink_resolution(path)
    active_lexical = _absolute_without_symlink_resolution(ACTIVE_STAGING)
    if _is_within(lexical, active_lexical):
        raise PublicationAdapterError(
            f"{label} is under active budget/toxicity-calibration staging"
        )
    symlink = _symlink_component(lexical)
    if symlink is not None:
        raise PublicationAdapterError(
            f"{label} contains a symlink component: {symlink}"
        )
    resolved = lexical.resolve(strict=False)
    active_resolved = active_lexical.resolve(strict=False)
    if _is_within(resolved, active_resolved):
        raise PublicationAdapterError(
            f"resolved {label} is under active budget/toxicity-calibration staging"
        )
    return resolved


def _stage_child(stage: Path, filename: str, *, label: str) -> Path:
    """Resolve a single basename and prove containment before any write/save."""

    if (
        not isinstance(filename, str) or not filename
        or Path(filename).name != filename or filename in {".", ".."}
        or "/" in filename or "\\" in filename
    ):
        raise PublicationAdapterError(f"{label} is not a safe basename")
    stage_resolved = stage.resolve(strict=False)
    candidate = stage / filename
    resolved = candidate.resolve(strict=False)
    if resolved.parent != stage_resolved or candidate.is_symlink():
        raise PublicationAdapterError(f"{label} escapes the publication stage")
    return candidate


def _read_analyzer_json(path: Path) -> tuple[Any, bytes]:
    if not path.is_file():
        raise PublicationAdapterError(f"required analyzer input is absent: {path.name}")
    raw = path.read_bytes()
    value = _decode_json(raw, label=path.name)
    if raw != _analyzer_canonical_json_bytes(value):
        raise PublicationAdapterError(f"analyzer input is not canonical JSON: {path.name}")
    return value, raw


def _read_contract(path: Path) -> tuple[dict[str, Any], bytes]:
    default_path = _guard_untrusted_input_path(
        DEFAULT_CONTRACT, label="default publication contract"
    )
    requested_path = _guard_untrusted_input_path(path, label="publication contract")
    if not default_path.is_file() or not requested_path.is_file():
        raise PublicationAdapterError(f"publication contract is absent: {requested_path}")
    default_raw = default_path.read_bytes()
    default_value = _decode_json(default_raw, label=default_path.name)
    if default_raw != _canonical_json_bytes(default_value):
        raise PublicationAdapterError("default publication contract is not canonical JSON")
    if _sha256_bytes(default_raw) != TRUSTED_CONTRACT_SHA256:
        raise PublicationAdapterError("default publication contract full-field commitment changed")
    raw = default_raw if requested_path == default_path else requested_path.read_bytes()
    value = _decode_json(raw, label=requested_path.name)
    if raw != _canonical_json_bytes(value):
        raise PublicationAdapterError("publication contract is not canonical JSON")
    if raw != default_raw:
        raise PublicationAdapterError(
            "custom publication contract must be byte-identical to DEFAULT_CONTRACT"
        )
    if not isinstance(value, dict):
        raise PublicationAdapterError("publication contract must be a JSON object")
    _validate_contract(value)
    return value, raw


def _exact_fields(value: Mapping[str, Any], expected: Iterable[str], *, label: str) -> None:
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        raise PublicationAdapterError(
            f"{label} fields changed: missing={sorted(expected_set-observed)}, "
            f"extra={sorted(observed-expected_set)}"
        )


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PublicationAdapterError(f"{label} is not a lowercase SHA-256")
    return value


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise PublicationAdapterError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PublicationAdapterError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise PublicationAdapterError(f"{label} must be a finite number")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise PublicationAdapterError(f"{label} must be an exact integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PublicationAdapterError(f"{label} must be an exact integer") from exc
    if result != value:
        raise PublicationAdapterError(f"{label} must be an exact integer")
    return result


def _primary_numerical_guard_metadata() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "eta": NUMERICAL_ETA,
        "eta_formula": "2^-26 = sqrt(binary64_epsilon)",
        "panel_brier_snapshot_bound_points": PANEL_BRIER_SNAPSHOT_BOUND_POINTS,
        "panel_brier_snapshot_bound_formula": "200 * eta",
        "family_B_brier_replicate_bound_points": FAMILY_B_BRIER_BOUND_POINTS,
        "family_B_brier_replicate_bound_formula": "400 * eta",
        "family_C_brier_interaction_bound_points": FAMILY_C_BRIER_BOUND_POINTS,
        "family_C_brier_interaction_bound_formula": "800 * eta",
        "mcse_guard_formula": "replicate_bound_points / sqrt(cumulative_M - 1)",
        "guarded_mcse_formula": "raw_mcse_points + mcse_guard_points",
        "threshold_points": 1.5,
        "threshold_rule": (
            "all guarded_mcse_points <= threshold_points; top up the whole "
            "factorial from M=200 to 500 or from M=500 to 1000; at M=1000 "
            "stop and retain any rows over threshold"
        ),
        "whole_factorial_top_up": [
            {"from_M": 200, "to_M": 500},
            {"from_M": 500, "to_M": 1000},
        ],
        "maximum_M": 1000,
        "row_guard_classes": [
            {
                "row_guard_class": GUARD_CLASS_EXACT_DISCRETE,
                "applies_to": "all locked non-Brier primary rows",
                "replicate_bound_points": 0.0,
            },
            {
                "row_guard_class": GUARD_CLASS_FAMILY_B_BRIER,
                "applies_to": "Family B panel-Brier differences",
                "replicate_bound_points": FAMILY_B_BRIER_BOUND_POINTS,
            },
            {
                "row_guard_class": GUARD_CLASS_FAMILY_C_BRIER,
                "applies_to": "Family C panel-Brier differences-in-differences",
                "replicate_bound_points": FAMILY_C_BRIER_BOUND_POINTS,
            },
        ],
    }


def _secondary_numerical_guard_metadata() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "eta": NUMERICAL_ETA,
        "eta_expression": "2^-26",
        "unit": "percentage points for panel_brier_score_pct; zero otherwise",
        "row_guard_classes": {
            SECONDARY_GUARD_CLASS_NON_BRIER_ZERO: {
                "applies_to": "every non-Brier secondary row",
                "replicate_numerical_bound_points": 0.0,
                "bound_expression": "0",
            },
            SECONDARY_GUARD_CLASS_ABSOLUTE_BRIER: {
                "applies_to": SECONDARY_ANALYSIS_ABSOLUTE_PROFILE,
                "replicate_numerical_bound_points": PANEL_BRIER_SNAPSHOT_BOUND_POINTS,
                "bound_expression": "delta0 = 200*eta",
            },
            SECONDARY_GUARD_CLASS_BUDGET_BRIER: {
                "applies_to": SECONDARY_ANALYSIS_BUDGET_CONTRAST,
                "replicate_numerical_bound_points": 2.0
                * PANEL_BRIER_SNAPSHOT_BOUND_POINTS,
                "bound_expression": "2*delta0 = 400*eta",
            },
            SECONDARY_GUARD_CLASS_CALIBRATION_DID_BRIER: {
                "applies_to": SECONDARY_ANALYSIS_CALIBRATION_DID,
                "replicate_numerical_bound_points": 4.0
                * PANEL_BRIER_SNAPSHOT_BOUND_POINTS,
                "bound_expression": "4*delta0 = 800*eta",
            },
        },
        "mcse_numerical_guard_formula": "a = b/sqrt(M-1)",
        "pointwise_numerical_halfwidth_guard_formula": "h = b + t_crit*a",
        "guarded_pointwise_interval_formula": (
            "guarded_low = raw_ci95_low - h; guarded_high = raw_ci95_high + h"
        ),
        "raw_pointwise_interval_fields": ["ci95_low", "ci95_high"],
        "guarded_pointwise_interval_fields": [
            "guarded_ci95_low",
            "guarded_ci95_high",
        ],
        "guarded_scientific_sign_rule": (
            "positive iff guarded_low > 0; negative iff guarded_high < 0; "
            "otherwise unresolved, including an endpoint exactly equal to zero"
        ),
    }


def _strict_interval_sign(low: float, high: float) -> str:
    if low > 0.0:
        return "positive"
    if high < 0.0:
        return "negative"
    return "unresolved"


def _primary_guard(spec: EstimandSpec, final_m: int) -> tuple[str, float, float]:
    if spec.family == "B" and spec.metric == BRIER:
        guard_class, bound = GUARD_CLASS_FAMILY_B_BRIER, FAMILY_B_BRIER_BOUND_POINTS
    elif spec.family == "C" and spec.metric == BRIER:
        guard_class, bound = GUARD_CLASS_FAMILY_C_BRIER, FAMILY_C_BRIER_BOUND_POINTS
    else:
        guard_class, bound = GUARD_CLASS_EXACT_DISCRETE, 0.0
    return guard_class, bound, bound / math.sqrt(final_m - 1)


def _secondary_guard(
    analysis_type: str, metric: str, final_m: int
) -> tuple[str, float, float]:
    if metric != BRIER:
        guard_class, bound = SECONDARY_GUARD_CLASS_NON_BRIER_ZERO, 0.0
    else:
        choices = {
            SECONDARY_ANALYSIS_ABSOLUTE_PROFILE: (
                SECONDARY_GUARD_CLASS_ABSOLUTE_BRIER,
                PANEL_BRIER_SNAPSHOT_BOUND_POINTS,
            ),
            SECONDARY_ANALYSIS_BUDGET_CONTRAST: (
                SECONDARY_GUARD_CLASS_BUDGET_BRIER,
                2.0 * PANEL_BRIER_SNAPSHOT_BOUND_POINTS,
            ),
            SECONDARY_ANALYSIS_CALIBRATION_DID: (
                SECONDARY_GUARD_CLASS_CALIBRATION_DID_BRIER,
                4.0 * PANEL_BRIER_SNAPSHOT_BOUND_POINTS,
            ),
        }
        if analysis_type not in choices:
            raise PublicationAdapterError(
                "secondary Brier row has no frozen numerical guard"
            )
        guard_class, bound = choices[analysis_type]
    return guard_class, bound, bound / math.sqrt(final_m - 1)


def _factor_token(value: float) -> str:
    return {0.5: "0p5", 1.0: "1", 2.0: "2"}[float(value)]


def expected_estimands() -> list[EstimandSpec]:
    output: list[EstimandSpec] = []
    for comparator in COMPARATORS:
        for metric in (TERMINAL, ASSIGNMENT):
            for budget in BUDGETS:
                for factor in FACTORS:
                    output.append(EstimandSpec(
                        family="A",
                        estimand_id=(
                            f"A|cKG-minus-{comparator}|{metric}|N={budget}|"
                            f"c={_factor_token(factor)}"
                        ),
                        policy="cKG-exact-formal",
                        comparator=comparator,
                        metric=metric,
                        budget_n=budget,
                        budget_contrast="",
                        factor=factor,
                        reference_factor=None,
                    ))
    for policy in POLICIES:
        for budget in BUDGETS:
            for factor in ERRONEOUS_FACTORS:
                for metric in (EXCLUSION, BRIER):
                    output.append(EstimandSpec(
                        family="B",
                        estimand_id=(
                            f"B|{policy}|{metric}|N={budget}|"
                            f"c={_factor_token(factor)}-minus-1"
                        ),
                        policy=policy,
                        comparator="",
                        metric=metric,
                        budget_n=budget,
                        budget_contrast="",
                        factor=factor,
                        reference_factor=1.0,
                    ))
    for policy in POLICIES:
        for factor in ERRONEOUS_FACTORS:
            for metric in (EXCLUSION, BRIER):
                output.append(EstimandSpec(
                    family="C",
                    estimand_id=(
                        f"C|{policy}|{metric}|N80-minus-N20|"
                        f"c={_factor_token(factor)}-minus-1"
                    ),
                    policy=policy,
                    comparator="",
                    metric=metric,
                    budget_n=None,
                    budget_contrast="N=80 minus N=20",
                    factor=factor,
                    reference_factor=1.0,
                ))
    if len(output) != 118 or len({item.estimand_id for item in output}) != 118:
        raise AssertionError("internal publication estimand grid is not 118 unique rows")
    return output


def _validate_contract(contract: Mapping[str, Any]) -> None:
    _exact_fields(contract, {
        "schema_version", "status", "artifact_class", "authentication",
        "captions", "colors_by_c_sigma", "estimands", "figures",
        "interpretation", "labels", "mcse_target_points", "outputs",
        "policy_labels", "secondary_projection",
    }, label="publication contract")
    if (
        contract.get("schema_version") != 3
        or contract.get("status") != "FROZEN_OUTCOME_INDEPENDENT_PUBLICATION_CONTRACT"
        or contract.get("artifact_class") != PUBLICATION_ARTIFACT_CLASS
    ):
        raise PublicationAdapterError("publication contract envelope changed")
    authentication = contract.get("authentication")
    if not isinstance(authentication, Mapping):
        raise PublicationAdapterError("contract authentication section is absent")
    if authentication.get("allowed_input_files") != list(INPUT_NAMES):
        raise PublicationAdapterError("contract allowed-input list changed")
    if authentication.get("analyzer_payload_schema_versions") != {
        "primary": 2, "secondary": 2,
    }:
        raise PublicationAdapterError("contract analyzer payload schemas changed")
    if authentication.get("binding_state") != (
        "BOUND_FINAL_THIRD_AMENDMENT_MANIFEST_SHA256"
    ):
        raise PublicationAdapterError("contract manifest-binding state changed")
    if authentication.get("primary_family_counts") != PRIMARY_COUNTS:
        raise PublicationAdapterError("contract primary-family counts changed")
    if authentication.get("primary_total") != 118:
        raise PublicationAdapterError("contract primary total changed")
    if authentication.get("secondary_counts") != {
        "absolute": 864,
        "contrast_and_conditional": 363,
        "plotted_absolute": 72,
        "published_policy_vs_cei_calibration_did": 36,
    }:
        raise PublicationAdapterError("contract secondary counts changed")
    if authentication.get("eligible_common_m") != list(ELIGIBLE_M):
        raise PublicationAdapterError("contract eligible-M set changed")
    if authentication.get("trusted_commitments") != TRUSTED_COMMITMENTS:
        raise PublicationAdapterError("contract trusted source commitments changed")
    if contract.get("labels") != EXPECTED_READER_LABELS:
        raise PublicationAdapterError("reader-facing labels changed")
    if contract.get("policy_labels") != {
        "cKG-exact-formal": "cKG", "tmse": "tMSE-only",
        "qBIG": "Entropy reduction", "cEI": "cEI",
    }:
        raise PublicationAdapterError("reader-facing policy labels changed")
    if contract.get("colors_by_c_sigma") != {
        "0.5": "#0072B2", "1.0": "#666666", "2.0": "#D55E00"
    }:
        raise PublicationAdapterError("factor-only color mapping changed")
    if _number(contract.get("mcse_target_points"), label="mcse target") != 1.5:
        raise PublicationAdapterError("MCSE target changed")
    figures = contract.get("figures")
    if not isinstance(figures, Mapping) or figures.get("formats") != ["png", "pdf"]:
        raise PublicationAdapterError("publication figure-format contract changed")
    expected_figure_counts = {
        "principal_profile": (36, 0),
        "cei_reference": (18, 0),
        "calibration_penalties": (48, 0),
        "attenuation_interactions": (16, 0),
        "absolute_gate_brier_profiles": (0, 72),
    }
    observed_stems: list[str] = []
    for name, (primary_rows, secondary_rows) in expected_figure_counts.items():
        spec = figures.get(name)
        if not isinstance(spec, Mapping):
            raise PublicationAdapterError(f"figure contract absent: {name}")
        if (
            spec.get("primary_rows") != primary_rows
            or spec.get("secondary_absolute_rows") != secondary_rows
            or not isinstance(spec.get("stem"), str)
        ):
            raise PublicationAdapterError(f"figure contract changed: {name}")
        stem = str(spec["stem"])
        if (
            SAFE_STEM_RE.fullmatch(stem) is None
            or Path(stem).name != stem or stem in {".", ".."}
        ):
            raise PublicationAdapterError(f"figure stem is not basename-safe: {name}")
        observed_stems.append(stem)
    if len(set(observed_stems)) != len(observed_stems):
        raise PublicationAdapterError("publication figure stems are not unique")
    secondary_projection = contract.get("secondary_projection")
    if secondary_projection != {
        "analysis_type": SECONDARY_ANALYSIS_CALIBRATION_DID,
        "budgets": list(BUDGETS),
        "c_sigma": list(ERRONEOUS_FACTORS),
        "complete_without_sign_filtering": True,
        "inference_scope": (
            "cell-specific guarded pointwise statements only; no majority vote "
            "or family theorem"
        ),
        "metrics": [EXCLUSION, BRIER],
        "order": ["policy", "c_sigma", "budget", "metric"],
        "policies": list(DID_POLICIES),
        "row_count": 36,
    }:
        raise PublicationAdapterError("secondary DiD projection contract changed")
    outputs = contract.get("outputs")
    if not isinstance(outputs, Mapping) or len(set(outputs.values())) != len(outputs):
        raise PublicationAdapterError("publication output names are malformed")
    for value in outputs.values():
        if not isinstance(value, str) or Path(value).name != value or not value:
            raise PublicationAdapterError("publication output name escapes its directory")
    estimands = contract.get("estimands")
    if not isinstance(estimands, Mapping) or any(
        estimands.get(key) is not True
        for key in (
            "no_cell_vote", "no_filtering_by_sign_interval_or_winner",
            "unresolved_is_not_sameness",
        )
    ):
        raise PublicationAdapterError("publication no-filter/no-vote contract changed")
    # Canonical-object hashing is the exhaustive full-field validation.  The
    # structural checks above provide specific diagnostics; this commitment
    # rejects any changed, added, removed, or resealed field anywhere below.
    if _sha256_bytes(_canonical_json_bytes(dict(contract))) != TRUSTED_CONTRACT_SHA256:
        raise PublicationAdapterError("publication contract full-field commitment changed")


def _validate_baseline(baseline: Mapping[str, Any]) -> None:
    _exact_fields(baseline, {
        "status", "common_field_cells", "locked_common_fields_per_cell",
        "common_field_comparisons", "allocation_history_cells",
        "historical_cells_without_allocation_history",
        "allocation_history_exception", "construction_prefix_identity",
        "streaming_and_crn_audit", "oracle_provenance",
    }, label="baseline PASS JSON")
    if baseline.get("status") != BASELINE_STATUS:
        raise PublicationAdapterError("baseline identity is not PASS")
    expected_counts = {
        "common_field_cells": 3200,
        "allocation_history_cells": 3040,
        "historical_cells_without_allocation_history": 160,
    }
    for field, expected in expected_counts.items():
        if baseline.get(field) != expected:
            raise PublicationAdapterError(f"baseline PASS count changed: {field}")
    if _integer(
        baseline.get("locked_common_fields_per_cell"),
        label="baseline locked-common-field count",
    ) <= 0 or _integer(
        baseline.get("common_field_comparisons"),
        label="baseline comparison count",
    ) <= 0:
        raise PublicationAdapterError("baseline PASS comparison counts are invalid")
    if baseline.get("allocation_history_exception") != {
        "policy": "cEI", "gamma": 0.7, "seed_start": 0,
        "seed_stop_exclusive": 80, "strata": [0, 1],
    }:
        raise PublicationAdapterError("baseline allocation-history exception changed")
    if baseline.get("construction_prefix_identity") != (
        "validated_by_frozen_core_test_contract"
    ):
        raise PublicationAdapterError("baseline construction-prefix identity changed")
    if not isinstance(baseline.get("streaming_and_crn_audit"), Mapping):
        raise PublicationAdapterError("baseline streaming/CRN audit is absent")
    if not isinstance(baseline.get("oracle_provenance"), Mapping):
        raise PublicationAdapterError("baseline oracle provenance is absent")


def _validate_artifact_entry(entry: Mapping[str, Any], *, label: str) -> None:
    _exact_fields(
        entry, {"sha256", "bytes", "metadata", "metadata_sha256"}, label=label
    )
    _sha(entry.get("sha256"), label=f"{label}.sha256")
    _sha(entry.get("metadata_sha256"), label=f"{label}.metadata_sha256")
    if _integer(entry.get("bytes"), label=f"{label}.bytes") < 0:
        raise PublicationAdapterError(f"{label}.bytes is negative")
    if not isinstance(entry.get("metadata"), str):
        raise PublicationAdapterError(f"{label}.metadata is not a filename")


def _validate_sidecar(
    sidecar: Mapping[str, Any],
    *,
    artifact_path: Path,
    artifact_raw: bytes,
    artifact_type: str,
    analysis_metadata: Mapping[str, Any],
    expected_absolute_count: int,
    expected_contrast_count: int,
) -> None:
    _exact_fields(sidecar, {
        "schema_version", "status", "artifact_type", "artifact",
        "artifact_sha256", "artifact_bytes", "analyzer", "analyzer_sha256",
        "prespec_sha256", "manifest_sha256", "raw_provenance",
        "baseline_identity", "analysis",
    }, label=f"{artifact_path.name} sidecar")
    if (
        sidecar.get("schema_version") != 1
        or sidecar.get("status") != ANALYZER_ARTIFACT_STATUS
        or sidecar.get("artifact_type") != artifact_type
        or sidecar.get("artifact") != artifact_path.name
        or sidecar.get("artifact_sha256") != _sha256_bytes(artifact_raw)
        or sidecar.get("artifact_bytes") != len(artifact_raw)
        or sidecar.get("analyzer")
        != "paper/analyze_budget_toxicity_calibration_audit.py"
    ):
        raise PublicationAdapterError(f"{artifact_path.name} sidecar envelope differs")
    for field in ("analyzer_sha256", "prespec_sha256", "manifest_sha256"):
        _sha(sidecar.get(field), label=f"{artifact_path.name}.{field}")
        if sidecar.get(field) != analysis_metadata.get(field):
            raise PublicationAdapterError(f"{artifact_path.name} sidecar {field} differs")
    if sidecar.get("raw_provenance") != analysis_metadata.get("raw_provenance"):
        raise PublicationAdapterError(f"{artifact_path.name} raw provenance differs")
    if sidecar.get("baseline_identity") != analysis_metadata.get("baseline_identity"):
        raise PublicationAdapterError(f"{artifact_path.name} baseline binding differs")
    details = sidecar.get("analysis")
    if not isinstance(details, Mapping):
        raise PublicationAdapterError(f"{artifact_path.name} analysis sidecar is absent")
    _exact_fields(details, {
        "primary_estimand_count", "secondary_absolute_row_count",
        "secondary_contrast_row_count", "max_t_draws", "max_t_random_seeds",
        "numerical_equivalence_guard", "secondary_numerical_equivalence_guard",
        "scientific_sign_source", "secondary_scientific_sign_source",
        "cell_vote_used",
    }, label=f"{artifact_path.name} sidecar analysis")
    if (
        details.get("primary_estimand_count") != 118
        or details.get("secondary_absolute_row_count") != expected_absolute_count
        or details.get("secondary_contrast_row_count") != expected_contrast_count
        or details.get("max_t_draws") != 100_000
        or details.get("max_t_random_seeds") != {
            "A": MAX_T_SEEDS["A"], "B": MAX_T_SEEDS["B"], "C": MAX_T_SEEDS["C"]
        }
        or details.get("numerical_equivalence_guard")
        != _primary_numerical_guard_metadata()
        or details.get("secondary_numerical_equivalence_guard")
        != _secondary_numerical_guard_metadata()
        or details.get("scientific_sign_source")
        != "guarded simultaneous 95% band only"
        or details.get("secondary_scientific_sign_source")
        != "guarded pointwise 95% interval only"
        or details.get("cell_vote_used") is not False
    ):
        raise PublicationAdapterError(f"{artifact_path.name} analysis sidecar differs")


def _validate_primary(
    payload: Mapping[str, Any], *, final_m: int
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    _exact_fields(
        payload, {"schema_version", "analysis", "rows", "multiplicity"},
        label="primary JSON",
    )
    if (
        payload.get("schema_version") != 2
        or payload.get("analysis") != "locked_primary_118_estimands"
    ):
        raise PublicationAdapterError("primary JSON envelope changed")
    rows = payload.get("rows")
    multiplicity = payload.get("multiplicity")
    if not isinstance(rows, list) or not isinstance(multiplicity, Mapping):
        raise PublicationAdapterError("primary rows or multiplicity are absent")
    specs = expected_estimands()
    expected_ids = [item.estimand_id for item in specs]
    observed_ids = [row.get("estimand_id") for row in rows if isinstance(row, Mapping)]
    if len(rows) != 118 or observed_ids != expected_ids:
        raise PublicationAdapterError(
            "primary rows are not the exact ordered 118-estimand grid"
        )
    _exact_fields(multiplicity, {
        "family_bands", "pointwise_method", "aggregation",
        "numerical_equivalence_guard",
        "simultaneous_numerical_halfwidth_guard_formula",
        "scientific_sign_source", "cell_vote_used",
    }, label="primary multiplicity")
    if (
        multiplicity.get("pointwise_method") != "paired seed-level t interval"
        or multiplicity.get("aggregation")
        != "equal weight over two strata and two gates within seed"
        or multiplicity.get("numerical_equivalence_guard")
        != _primary_numerical_guard_metadata()
        or multiplicity.get("simultaneous_numerical_halfwidth_guard_formula")
        != (
            "h = replicate_numerical_bound_points + "
            "family_max_t_critical_95 * mcse_numerical_guard_points"
        )
        or multiplicity.get("scientific_sign_source")
        != "guarded simultaneous 95% band only"
        or multiplicity.get("cell_vote_used") is not False
    ):
        raise PublicationAdapterError("primary multiplicity method changed")
    family_bands = multiplicity.get("family_bands")
    if not isinstance(family_bands, Mapping) or set(family_bands) != set(PRIMARY_COUNTS):
        raise PublicationAdapterError("primary family-band metadata changed")
    family_critical: dict[str, float] = {}
    for family, count in PRIMARY_COUNTS.items():
        details = family_bands[family]
        if not isinstance(details, Mapping):
            raise PublicationAdapterError(f"Family {family} max-t metadata is absent")
        _exact_fields(details, {
            "family", "draws", "random_seed", "multiplier", "statistic",
            "critical_statistic", "quantile", "quantile_method",
            "critical_value", "zero_variance_rule", "estimand_count",
            "monte_carlo_replicates", "numerical_equivalence_guard",
            "simultaneous_numerical_halfwidth_guard_formula",
            "guarded_simultaneous_band_formula",
            "guarded_scientific_sign_rule", "row_guard_class_counts",
        }, label=f"Family {family} max-t metadata")
        critical = _number(details.get("critical_value"), label=f"Family {family} critical")
        expected_guard_counts = {
            "A": {GUARD_CLASS_EXACT_DISCRETE: 54},
            "B": {
                GUARD_CLASS_EXACT_DISCRETE: 24,
                GUARD_CLASS_FAMILY_B_BRIER: 24,
            },
            "C": {
                GUARD_CLASS_EXACT_DISCRETE: 8,
                GUARD_CLASS_FAMILY_C_BRIER: 8,
            },
        }[family]
        if (
            details.get("family") != family
            or details.get("draws") != 100_000
            or details.get("random_seed") != MAX_T_SEEDS[family]
            or details.get("multiplier") != "Rademacher"
            or details.get("statistic")
            != "sum_i xi_i*(Y_i-Ybar)/sqrt(sum_i (Y_i-Ybar)^2)"
            or details.get("critical_statistic") != "max absolute coordinate T"
            or details.get("quantile") != 0.95
            or details.get("quantile_method") != "higher"
            or details.get("zero_variance_rule")
            != (
                "T=0 and zero-width raw interval; a nonzero numerical bound "
                "may still widen the guarded band"
            )
            or details.get("numerical_equivalence_guard")
            != _primary_numerical_guard_metadata()
            or details.get("simultaneous_numerical_halfwidth_guard_formula")
            != (
                "h = replicate_numerical_bound_points + "
                "family_max_t_critical_95 * mcse_numerical_guard_points"
            )
            or details.get("guarded_simultaneous_band_formula")
            != "guarded_low = raw_low - h; guarded_high = raw_high + h"
            or details.get("guarded_scientific_sign_rule")
            != (
                "positive iff guarded_low > 0; negative iff guarded_high < 0; "
                "otherwise unresolved, including an endpoint exactly equal to zero"
            )
            or details.get("row_guard_class_counts") != expected_guard_counts
            or details.get("estimand_count") != count
            or details.get("monte_carlo_replicates") != final_m
            or critical < 0.0
        ):
            raise PublicationAdapterError(f"Family {family} max-t metadata differs")
        family_critical[family] = critical

    row_fields = {
        "family", "estimand_id", "estimate_points", "mcse_points",
        "pointwise_t_critical_95", "pointwise_95_low_points",
        "pointwise_95_high_points", "family_max_t_critical_95",
        "simultaneous_95_low_points", "simultaneous_95_high_points",
        "row_guard_class", "replicate_numerical_bound_points",
        "mcse_numerical_guard_points",
        "simultaneous_numerical_halfwidth_guard_points",
        "guarded_simultaneous_95_low_points",
        "guarded_simultaneous_95_high_points",
        "guarded_simultaneous_95_strictly_positive",
        "guarded_simultaneous_95_strictly_negative",
        "guarded_simultaneous_95_excludes_zero", "guarded_scientific_sign",
        "zero_variance", "monte_carlo_replicates",
    }
    point_criticals: list[float] = []
    observed_counts = {family: 0 for family in PRIMARY_COUNTS}
    validated: list[Mapping[str, Any]] = []
    for index, (row, spec) in enumerate(zip(rows, specs, strict=True)):
        if not isinstance(row, Mapping):
            raise PublicationAdapterError(f"primary row {index} is not an object")
        _exact_fields(row, row_fields, label=f"primary row {index}")
        if row.get("family") != spec.family or row.get("estimand_id") != spec.estimand_id:
            raise PublicationAdapterError(f"primary row {index} identity differs")
        observed_counts[spec.family] += 1
        estimate = _number(row.get("estimate_points"), label=f"row {index} estimate")
        mcse = _number(row.get("mcse_points"), label=f"row {index} MCSE")
        point_critical = _number(
            row.get("pointwise_t_critical_95"), label=f"row {index} point critical"
        )
        family_value = _number(
            row.get("family_max_t_critical_95"), label=f"row {index} family critical"
        )
        point_low = _number(
            row.get("pointwise_95_low_points"), label=f"row {index} point low"
        )
        point_high = _number(
            row.get("pointwise_95_high_points"), label=f"row {index} point high"
        )
        simultaneous_low = _number(
            row.get("simultaneous_95_low_points"), label=f"row {index} simultaneous low"
        )
        simultaneous_high = _number(
            row.get("simultaneous_95_high_points"), label=f"row {index} simultaneous high"
        )
        guard_class, replicate_bound, mcse_guard = _primary_guard(spec, final_m)
        observed_replicate_bound = _number(
            row.get("replicate_numerical_bound_points"),
            label=f"row {index} replicate numerical bound",
        )
        observed_mcse_guard = _number(
            row.get("mcse_numerical_guard_points"),
            label=f"row {index} MCSE numerical guard",
        )
        guard_halfwidth = replicate_bound + family_value * mcse_guard
        observed_guard_halfwidth = _number(
            row.get("simultaneous_numerical_halfwidth_guard_points"),
            label=f"row {index} simultaneous numerical halfwidth guard",
        )
        guarded_low = _number(
            row.get("guarded_simultaneous_95_low_points"),
            label=f"row {index} guarded simultaneous low",
        )
        guarded_high = _number(
            row.get("guarded_simultaneous_95_high_points"),
            label=f"row {index} guarded simultaneous high",
        )
        if mcse < 0.0 or point_critical < 0.0 or family_value < 0.0:
            raise PublicationAdapterError(f"primary row {index} has a negative uncertainty value")
        if family_value != family_critical[spec.family]:
            raise PublicationAdapterError(f"primary row {index} max-t critical mismatch")
        canonical_point_low = estimate - point_critical * mcse
        canonical_point_high = estimate + point_critical * mcse
        canonical_simultaneous_low = estimate - family_value * mcse
        canonical_simultaneous_high = estimate + family_value * mcse
        canonical_guarded_low = canonical_simultaneous_low - guard_halfwidth
        canonical_guarded_high = canonical_simultaneous_high + guard_halfwidth
        expected_values = (
            (point_low, canonical_point_low, "pointwise low"),
            (point_high, canonical_point_high, "pointwise high"),
            (simultaneous_low, canonical_simultaneous_low, "simultaneous low"),
            (simultaneous_high, canonical_simultaneous_high, "simultaneous high"),
            (observed_replicate_bound, replicate_bound, "replicate numerical bound"),
            (observed_mcse_guard, mcse_guard, "MCSE numerical guard"),
            (observed_guard_halfwidth, guard_halfwidth, "simultaneous numerical halfwidth guard"),
            (guarded_low, canonical_guarded_low, "guarded simultaneous low"),
            (guarded_high, canonical_guarded_high, "guarded simultaneous high"),
        )
        for observed, expected, label in expected_values:
            if observed != expected:
                raise PublicationAdapterError(
                    f"primary row {index} {label} is not the exact canonical recomputation"
                )
        if row.get("row_guard_class") != guard_class:
            raise PublicationAdapterError(f"primary row {index} guard class differs")
        sign = _strict_interval_sign(
            canonical_guarded_low, canonical_guarded_high
        )
        guarded_flags = {
            "guarded_simultaneous_95_strictly_positive": sign == "positive",
            "guarded_simultaneous_95_strictly_negative": sign == "negative",
            "guarded_simultaneous_95_excludes_zero": sign != "unresolved",
        }
        if any(
            type(row.get(field)) is not bool or row.get(field) is not expected
            for field, expected in guarded_flags.items()
        ) or row.get("guarded_scientific_sign") != sign:
            raise PublicationAdapterError(
                f"primary row {index} guarded scientific sign differs"
            )
        zero_variance = row.get("zero_variance")
        if type(zero_variance) is not bool or zero_variance != (mcse == 0.0):
            raise PublicationAdapterError(f"primary row {index} zero-variance flag differs")
        if row.get("monte_carlo_replicates") != final_m:
            raise PublicationAdapterError(f"primary row {index} does not use common final M")
        point_criticals.append(point_critical)
        canonical_row = dict(row)
        canonical_row.update({
            "pointwise_95_low_points": canonical_point_low,
            "pointwise_95_high_points": canonical_point_high,
            "simultaneous_95_low_points": canonical_simultaneous_low,
            "simultaneous_95_high_points": canonical_simultaneous_high,
            "replicate_numerical_bound_points": replicate_bound,
            "mcse_numerical_guard_points": mcse_guard,
            "simultaneous_numerical_halfwidth_guard_points": guard_halfwidth,
            "guarded_simultaneous_95_low_points": canonical_guarded_low,
            "guarded_simultaneous_95_high_points": canonical_guarded_high,
            "guarded_simultaneous_95_strictly_positive": sign == "positive",
            "guarded_simultaneous_95_strictly_negative": sign == "negative",
            "guarded_simultaneous_95_excludes_zero": sign != "unresolved",
            "guarded_scientific_sign": sign,
        })
        validated.append(canonical_row)
    if observed_counts != PRIMARY_COUNTS:
        raise PublicationAdapterError("primary 54/48/16 family split changed")
    if any(value != point_criticals[0] for value in point_criticals[1:]):
        raise PublicationAdapterError("pointwise t critical is not common across final-M rows")
    return tuple(validated), dict(multiplicity)


def _validate_secondary_guarded_row(
    row: Mapping[str, Any], *, final_m: int, label: str
) -> dict[str, Any]:
    required = {
        "analysis_type", "metric", "estimate", "mcse",
        "pointwise_t_critical_95", "ci95_low", "ci95_high",
        "monte_carlo_replicates", "row_guard_class",
        "replicate_numerical_bound_points", "mcse_numerical_guard_points",
        "pointwise_numerical_halfwidth_guard_points", "guarded_ci95_low",
        "guarded_ci95_high", "guarded_ci95_strictly_positive",
        "guarded_ci95_strictly_negative", "guarded_ci95_excludes_zero",
        "guarded_pointwise_scientific_sign",
    }
    if not required.issubset(row):
        raise PublicationAdapterError(
            f"{label} lacks guarded pointwise interval fields"
        )
    if row.get("monte_carlo_replicates") != final_m:
        raise PublicationAdapterError(f"{label} does not use common final M")
    analysis_type = str(row.get("analysis_type"))
    metric = str(row.get("metric"))
    critical = _number(
        row.get("pointwise_t_critical_95"), label=f"{label} point critical"
    )
    if critical < 0.0:
        raise PublicationAdapterError(f"{label} has a negative point critical")
    guard_class, bound, mcse_guard = _secondary_guard(
        analysis_type, metric, final_m
    )
    halfwidth_guard = bound + critical * mcse_guard
    expected_numeric = (
        (row.get("replicate_numerical_bound_points"), bound, "replicate bound"),
        (row.get("mcse_numerical_guard_points"), mcse_guard, "MCSE guard"),
        (
            row.get("pointwise_numerical_halfwidth_guard_points"),
            halfwidth_guard,
            "pointwise halfwidth guard",
        ),
    )
    for observed, expected, field in expected_numeric:
        if _number(observed, label=f"{label} {field}") != expected:
            raise PublicationAdapterError(
                f"{label} {field} is not the exact canonical recomputation"
            )
    if row.get("row_guard_class") != guard_class:
        raise PublicationAdapterError(f"{label} guard class differs")

    raw_values = tuple(row.get(field) for field in ("estimate", "mcse", "ci95_low", "ci95_high"))
    if all(value is None for value in raw_values):
        if bound != 0.0:
            raise PublicationAdapterError(f"{label} guarded Brier interval is unavailable")
        if row.get("guarded_ci95_low") is not None or row.get("guarded_ci95_high") is not None:
            raise PublicationAdapterError(f"{label} guarded unavailable endpoints differ")
        canonical_raw_low = None
        canonical_raw_high = None
        canonical_guarded_low = None
        canonical_guarded_high = None
        sign = "unavailable"
    elif any(value is None for value in raw_values):
        raise PublicationAdapterError(f"{label} raw interval is partially unavailable")
    else:
        estimate = _number(row.get("estimate"), label=f"{label} estimate")
        mcse = _number(row.get("mcse"), label=f"{label} MCSE")
        raw_low = _number(row.get("ci95_low"), label=f"{label} raw CI low")
        raw_high = _number(row.get("ci95_high"), label=f"{label} raw CI high")
        guarded_low = _number(
            row.get("guarded_ci95_low"), label=f"{label} guarded CI low"
        )
        guarded_high = _number(
            row.get("guarded_ci95_high"), label=f"{label} guarded CI high"
        )
        if mcse < 0.0 or raw_low > raw_high:
            raise PublicationAdapterError(f"{label} raw interval differs")
        canonical_raw_low = estimate - critical * mcse
        canonical_raw_high = estimate + critical * mcse
        canonical_guarded_low = canonical_raw_low - halfwidth_guard
        canonical_guarded_high = canonical_raw_high + halfwidth_guard
        expected_intervals = (
            (raw_low, canonical_raw_low, "raw CI low"),
            (raw_high, canonical_raw_high, "raw CI high"),
            (guarded_low, canonical_guarded_low, "guarded CI low"),
            (guarded_high, canonical_guarded_high, "guarded CI high"),
        )
        for observed, expected, field in expected_intervals:
            if observed != expected:
                raise PublicationAdapterError(
                    f"{label} {field} is not the exact canonical recomputation"
                )
        sign = _strict_interval_sign(
            canonical_guarded_low, canonical_guarded_high
        )
    expected_flags = {
        "guarded_ci95_strictly_positive": sign == "positive",
        "guarded_ci95_strictly_negative": sign == "negative",
        "guarded_ci95_excludes_zero": sign in {"positive", "negative"},
    }
    if any(
        type(row.get(field)) is not bool or row.get(field) is not expected
        for field, expected in expected_flags.items()
    ) or row.get("guarded_pointwise_scientific_sign") != sign:
        raise PublicationAdapterError(f"{label} guarded scientific sign differs")
    canonical_row = dict(row)
    canonical_row.update({
        "ci95_low": canonical_raw_low,
        "ci95_high": canonical_raw_high,
        "replicate_numerical_bound_points": bound,
        "mcse_numerical_guard_points": mcse_guard,
        "pointwise_numerical_halfwidth_guard_points": halfwidth_guard,
        "guarded_ci95_low": canonical_guarded_low,
        "guarded_ci95_high": canonical_guarded_high,
        "guarded_ci95_strictly_positive": sign == "positive",
        "guarded_ci95_strictly_negative": sign == "negative",
        "guarded_ci95_excludes_zero": sign in {"positive", "negative"},
        "guarded_pointwise_scientific_sign": sign,
    })
    return canonical_row


def _validate_secondary(
    payload: Mapping[str, Any], *, final_m: int
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    Mapping[str, Any],
]:
    _exact_fields(payload, {
        "schema_version", "analysis", "absolute_rows",
        "contrast_and_conditional_rows", "numerical_equivalence_guard",
        "scientific_sign_source", "cell_vote_used",
    }, label="secondary JSON")
    guard_metadata = payload.get("numerical_equivalence_guard")
    if (
        payload.get("schema_version") != 2
        or payload.get("analysis") != "prespecified_secondary_diagnostics"
        or guard_metadata != _secondary_numerical_guard_metadata()
        or payload.get("scientific_sign_source")
        != "guarded pointwise 95% interval only"
        or payload.get("cell_vote_used") is not False
    ):
        raise PublicationAdapterError("secondary JSON envelope changed")
    absolute = payload.get("absolute_rows")
    contrasts = payload.get("contrast_and_conditional_rows")
    if not isinstance(absolute, list) or not isinstance(contrasts, list):
        raise PublicationAdapterError("secondary rows are absent")
    if len(absolute) != SECONDARY_COUNTS["absolute"]:
        raise PublicationAdapterError("secondary absolute row count changed")
    if len(contrasts) != SECONDARY_COUNTS["contrast_and_conditional"]:
        raise PublicationAdapterError("secondary contrast row count changed")

    validated_absolute: list[dict[str, Any]] = []
    selected: dict[tuple[str, float, int, str], Mapping[str, Any]] = {}
    selected_fields = {
        "analysis_type", "policy", "policy_label",
        "assumed_toxicity_noise_sd_factor", "budget", "metric", "estimate",
        "mcse", "pointwise_t_critical_95", "ci95_low", "ci95_high",
        "monte_carlo_replicates", "row_guard_class",
        "replicate_numerical_bound_points", "mcse_numerical_guard_points",
        "pointwise_numerical_halfwidth_guard_points", "guarded_ci95_low",
        "guarded_ci95_high", "guarded_ci95_strictly_positive",
        "guarded_ci95_strictly_negative", "guarded_ci95_excludes_zero",
        "guarded_pointwise_scientific_sign",
    }
    absolute_identity: set[tuple[str, float, int, str]] = set()
    for index, raw_row in enumerate(absolute):
        if not isinstance(raw_row, Mapping):
            raise PublicationAdapterError(f"secondary absolute row {index} is not an object")
        row = _validate_secondary_guarded_row(
            raw_row, final_m=final_m, label=f"secondary absolute row {index}"
        )
        policy = row.get("policy")
        factor = _number(
            row.get("assumed_toxicity_noise_sd_factor"),
            label=f"secondary absolute row {index} factor",
        )
        budget = _integer(
            row.get("budget"), label=f"secondary absolute row {index} budget"
        )
        metric = str(row.get("metric"))
        identity = (str(policy), factor, budget, metric)
        if (
            row.get("analysis_type") != SECONDARY_ANALYSIS_ABSOLUTE_PROFILE
            or policy not in POLICIES or factor not in FACTORS
            or budget not in BUDGETS or metric not in ABSOLUTE_METRICS
            or row.get("policy_label") != ANALYZER_POLICY_LABELS[policy]
            or identity in absolute_identity
        ):
            raise PublicationAdapterError(
                f"secondary absolute row {index} identity differs"
            )
        absolute_identity.add(identity)
        validated_absolute.append(row)
        if metric in (EXCLUSION, BRIER):
            _exact_fields(row, selected_fields, label=f"selected secondary row {index}")
            selected[identity] = row
    expected_absolute_identity = {
        (policy, factor, budget, metric)
        for policy in POLICIES for factor in FACTORS for budget in BUDGETS
        for metric in ABSOLUTE_METRICS
    }
    if absolute_identity != expected_absolute_identity:
        raise PublicationAdapterError("secondary absolute grid changed")
    expected_selected = {
        (policy, factor, budget, metric)
        for policy in POLICIES for factor in FACTORS for budget in BUDGETS
        for metric in (EXCLUSION, BRIER)
    }
    if set(selected) != expected_selected:
        raise PublicationAdapterError("absolute gate/Brier profile is not all 72 cells")

    validated_contrasts: list[Mapping[str, Any]] = []
    did_rows: dict[tuple[str, float, int, str], Mapping[str, Any]] = {}
    for index, raw_row in enumerate(contrasts):
        if not isinstance(raw_row, Mapping):
            raise PublicationAdapterError(f"secondary contrast row {index} is not an object")
        row = _validate_secondary_guarded_row(
            raw_row, final_m=final_m, label=f"secondary contrast row {index}"
        )
        validated_contrasts.append(row)
        if row.get("analysis_type") == SECONDARY_ANALYSIS_CALIBRATION_DID:
            _exact_fields(
                row, selected_fields,
                label=f"policy-vs-cEI calibration DiD row {index}",
            )
            policy = str(row.get("policy"))
            factor = _number(
                row.get("assumed_toxicity_noise_sd_factor"),
                label=f"policy-vs-cEI calibration DiD row {index} factor",
            )
            budget = _integer(
                row.get("budget"),
                label=f"policy-vs-cEI calibration DiD row {index} budget",
            )
            metric = str(row.get("metric"))
            identity = (policy, factor, budget, metric)
            if (
                policy not in DID_POLICIES
                or factor not in ERRONEOUS_FACTORS
                or budget not in BUDGETS
                or metric not in (EXCLUSION, BRIER)
                or row.get("policy_label") != ANALYZER_POLICY_LABELS[policy]
                or identity in did_rows
            ):
                raise PublicationAdapterError(
                    f"policy-vs-cEI calibration DiD row {index} identity differs"
                )
            did_rows[identity] = row
    expected_did_identity = {
        (policy, factor, budget, metric)
        for policy in DID_POLICIES for factor in ERRONEOUS_FACTORS
        for budget in BUDGETS for metric in (EXCLUSION, BRIER)
    }
    if set(did_rows) != expected_did_identity:
        raise PublicationAdapterError(
            "policy-vs-cEI calibration DiD projection is not all 36 cells"
        )
    ordered_selected = tuple(selected[key] for key in sorted(
        selected,
        key=lambda key: (
            POLICIES.index(key[0]), (EXCLUSION, BRIER).index(key[3]),
            FACTORS.index(key[1]), BUDGETS.index(key[2]),
        ),
    ))
    ordered_did = tuple(did_rows[key] for key in sorted(
        did_rows,
        key=lambda key: (
            DID_POLICIES.index(key[0]), ERRONEOUS_FACTORS.index(key[1]),
            BUDGETS.index(key[2]), (EXCLUSION, BRIER).index(key[3]),
        ),
    ))
    return (
        ordered_selected, tuple(validated_contrasts), ordered_did,
        dict(guard_metadata),
    )


def authenticate_analyzer_bundle(analysis_dir: str | Path) -> AuthenticatedBundle:
    """Authenticate only the eight analyzer outputs allowed by the contract."""

    root = _guard_untrusted_input_path(analysis_dir, label="analysis bundle")
    if not root.is_dir():
        raise PublicationAdapterError(f"analysis directory is absent: {root}")
    paths: dict[str, Path] = {}
    for name in INPUT_NAMES:
        candidate = _guard_untrusted_input_path(
            root / name, label=f"allowed analyzer input {name}"
        )
        if candidate.parent != root:
            raise PublicationAdapterError(
                f"allowed analyzer input escapes its resolved bundle: {name}"
            )
        paths[name] = candidate
    decoded: dict[str, Any] = {}
    raw: dict[str, bytes] = {}
    for name in INPUT_NAMES:
        decoded[name], raw[name] = _read_analyzer_json(paths[name])
    if any(not isinstance(decoded[name], Mapping) for name in INPUT_NAMES):
        raise PublicationAdapterError("every allowed analyzer input must be a JSON object")

    analysis_metadata = decoded[ANALYSIS_METADATA_NAME]
    analysis_commit = decoded[ANALYSIS_COMMIT_NAME]
    _exact_fields(analysis_metadata, {
        "schema_version", "status", "artifact_class", "analyzer_sha256",
        "prespec_sha256", "manifest_sha256", "raw_provenance",
        "baseline_identity", "primary_estimand_count",
        "numerical_equivalence_guard", "secondary_numerical_equivalence_guard",
        "scientific_sign_source", "secondary_scientific_sign_source",
        "secondary_absolute_row_count", "secondary_contrast_row_count",
        "artifacts",
    }, label="analysis metadata")
    _exact_fields(analysis_commit, {
        "schema_version", "status", "metadata", "metadata_sha256",
        "artifact_count", "baseline_identity_passed",
    }, label="analysis commit")
    if (
        analysis_metadata.get("schema_version") != 1
        or analysis_metadata.get("status") != ANALYSIS_STATUS
        or analysis_metadata.get("artifact_class")
        != "authenticated_postprocessing_analysis_not_raw_trials"
        or analysis_metadata.get("primary_estimand_count") != 118
        or analysis_metadata.get("numerical_equivalence_guard")
        != _primary_numerical_guard_metadata()
        or analysis_metadata.get("secondary_numerical_equivalence_guard")
        != _secondary_numerical_guard_metadata()
        or analysis_metadata.get("scientific_sign_source")
        != "guarded simultaneous 95% band only"
        or analysis_metadata.get("secondary_scientific_sign_source")
        != "guarded pointwise 95% interval only"
    ):
        raise PublicationAdapterError("analysis metadata envelope changed")
    for field, expected in TRUSTED_COMMITMENTS.items():
        _sha(analysis_metadata.get(field), label=f"analysis metadata {field}")
        if analysis_metadata.get(field) != expected:
            raise PublicationAdapterError(
                f"analysis metadata {field} differs from the frozen trusted commitment"
            )
    artifacts = analysis_metadata.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PublicationAdapterError("analysis artifact manifest is absent")
    for name, entry in artifacts.items():
        if not isinstance(name, str) or not isinstance(entry, Mapping):
            raise PublicationAdapterError("analysis artifact manifest is malformed")
        _validate_artifact_entry(entry, label=f"analysis artifact {name}")
    if (
        analysis_commit.get("schema_version") != 1
        or analysis_commit.get("status") != ANALYSIS_COMMIT_STATUS
        or analysis_commit.get("metadata") != ANALYSIS_METADATA_NAME
        or analysis_commit.get("metadata_sha256")
        != _sha256_bytes(raw[ANALYSIS_METADATA_NAME])
        or analysis_commit.get("artifact_count") != len(artifacts)
        or analysis_commit.get("baseline_identity_passed") is not True
    ):
        raise PublicationAdapterError("final analyzer commit does not authenticate metadata")

    baseline = decoded[BASELINE_NAME]
    if not isinstance(baseline, Mapping):
        raise PublicationAdapterError("baseline PASS JSON is not an object")
    _validate_baseline(baseline)
    if analysis_metadata.get("baseline_identity") != baseline:
        raise PublicationAdapterError("analysis metadata does not bind the baseline PASS JSON")
    raw_provenance = analysis_metadata.get("raw_provenance")
    if not isinstance(raw_provenance, Mapping):
        raise PublicationAdapterError("analysis raw provenance is absent")
    final_m = _integer(raw_provenance.get("final_M"), label="analysis final_M")
    if final_m not in ELIGIBLE_M:
        raise PublicationAdapterError("analysis final_M is outside 200/500/1000")

    absolute_count = _integer(
        analysis_metadata.get("secondary_absolute_row_count"),
        label="secondary absolute row count",
    )
    contrast_count = _integer(
        analysis_metadata.get("secondary_contrast_row_count"),
        label="secondary contrast row count",
    )
    if absolute_count != SECONDARY_COUNTS["absolute"]:
        raise PublicationAdapterError("analysis secondary absolute count changed")
    if contrast_count != SECONDARY_COUNTS["contrast_and_conditional"]:
        raise PublicationAdapterError("analysis secondary contrast count changed")
    selected_types = {
        PRIMARY_NAME: "primary_contrasts_json",
        SECONDARY_NAME: "secondary_diagnostics_json",
        BASELINE_NAME: "baseline_identity_json",
    }
    for artifact_name, artifact_type in selected_types.items():
        sidecar_name = artifact_name + ".metadata.json"
        entry = artifacts.get(artifact_name)
        if not isinstance(entry, Mapping):
            raise PublicationAdapterError(f"analysis manifest lacks {artifact_name}")
        if (
            entry.get("sha256") != _sha256_bytes(raw[artifact_name])
            or entry.get("bytes") != len(raw[artifact_name])
            or entry.get("metadata") != sidecar_name
            or entry.get("metadata_sha256") != _sha256_bytes(raw[sidecar_name])
        ):
            raise PublicationAdapterError(f"analysis manifest does not bind {artifact_name}")
        _validate_sidecar(
            decoded[sidecar_name],
            artifact_path=paths[artifact_name], artifact_raw=raw[artifact_name],
            artifact_type=artifact_type, analysis_metadata=analysis_metadata,
            expected_absolute_count=absolute_count,
            expected_contrast_count=contrast_count,
        )

    primary = decoded[PRIMARY_NAME]
    secondary = decoded[SECONDARY_NAME]
    if not isinstance(primary, Mapping) or not isinstance(secondary, Mapping):
        raise PublicationAdapterError("primary/secondary JSON must be objects")
    primary_rows, multiplicity = _validate_primary(primary, final_m=final_m)
    (
        absolute_rows, secondary_contrast_rows, policy_cei_did_rows,
        secondary_guard_metadata,
    ) = _validate_secondary(secondary, final_m=final_m)
    if len(secondary.get("absolute_rows", [])) != absolute_count:
        raise PublicationAdapterError("secondary absolute count differs from analysis metadata")
    if len(secondary.get("contrast_and_conditional_rows", [])) != contrast_count:
        raise PublicationAdapterError("secondary contrast count differs from analysis metadata")

    return AuthenticatedBundle(
        analysis_dir=root,
        input_paths=paths,
        input_hashes={name: _sha256_bytes(raw[name]) for name in INPUT_NAMES},
        input_bytes={name: len(raw[name]) for name in INPUT_NAMES},
        analysis_metadata=dict(analysis_metadata),
        analysis_commit=dict(analysis_commit),
        primary=dict(primary), secondary=dict(secondary), baseline=dict(baseline),
        final_m=final_m, primary_rows=primary_rows,
        absolute_rows=absolute_rows,
        secondary_contrast_rows=secondary_contrast_rows,
        policy_cei_did_rows=policy_cei_did_rows,
        primary_multiplicity=multiplicity,
        secondary_guard_metadata=secondary_guard_metadata,
    )


def _guarded_band_classification(row: Mapping[str, Any]) -> str:
    low = _number(
        row["guarded_simultaneous_95_low_points"],
        label="guarded simultaneous low",
    )
    high = _number(
        row["guarded_simultaneous_95_high_points"],
        label="guarded simultaneous high",
    )
    return _strict_interval_sign(low, high)


def _contrast_label(spec: EstimandSpec, policy_labels: Mapping[str, str]) -> str:
    if spec.family == "A":
        return f"cKG minus {policy_labels[spec.comparator]}"
    if spec.family == "B":
        return (
            f"{policy_labels[spec.policy]}: c_sigma={spec.factor:g} minus c_sigma=1"
        )
    return (
        f"{policy_labels[spec.policy]}: (c_sigma={spec.factor:g} minus 1) at N=80 "
        "minus the same contrast at N=20"
    )


def enrich_primary_rows(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    specs = expected_estimands()
    policy_labels = contract["policy_labels"]
    labels = contract["labels"]
    output: list[dict[str, Any]] = []
    for order, (row, spec) in enumerate(zip(rows, specs, strict=True), start=1):
        output.append({
            "row_order": order,
            "family": spec.family,
            "estimand_id": spec.estimand_id,
            "contrast": _contrast_label(spec, policy_labels),
            "policy": spec.policy,
            "policy_label": policy_labels[spec.policy],
            "comparator": spec.comparator,
            "comparator_label": (
                policy_labels[spec.comparator] if spec.comparator else ""
            ),
            "metric": spec.metric,
            "metric_label": labels[METRIC_LABEL_KEYS[spec.metric]],
            "budget_n": spec.budget_n if spec.budget_n is not None else "",
            "budget_contrast": spec.budget_contrast,
            "assumed_toxicity_noise_sd_factor": spec.factor,
            "reference_toxicity_noise_sd_factor": (
                spec.reference_factor if spec.reference_factor is not None else ""
            ),
            "estimate_points": row["estimate_points"],
            "mcse_points": row["mcse_points"],
            "pointwise_t_critical_95": row["pointwise_t_critical_95"],
            "pointwise_95_low_points": row["pointwise_95_low_points"],
            "pointwise_95_high_points": row["pointwise_95_high_points"],
            "family_max_t_critical_95": row["family_max_t_critical_95"],
            "simultaneous_95_low_points": row["simultaneous_95_low_points"],
            "simultaneous_95_high_points": row["simultaneous_95_high_points"],
            "row_guard_class": row["row_guard_class"],
            "replicate_numerical_bound_points": row[
                "replicate_numerical_bound_points"
            ],
            "mcse_numerical_guard_points": row[
                "mcse_numerical_guard_points"
            ],
            "simultaneous_numerical_halfwidth_guard_points": row[
                "simultaneous_numerical_halfwidth_guard_points"
            ],
            "guarded_simultaneous_95_low_points": row[
                "guarded_simultaneous_95_low_points"
            ],
            "guarded_simultaneous_95_high_points": row[
                "guarded_simultaneous_95_high_points"
            ],
            "guarded_simultaneous_95_strictly_positive": row[
                "guarded_simultaneous_95_strictly_positive"
            ],
            "guarded_simultaneous_95_strictly_negative": row[
                "guarded_simultaneous_95_strictly_negative"
            ],
            "guarded_simultaneous_95_excludes_zero": row[
                "guarded_simultaneous_95_excludes_zero"
            ],
            "guarded_scientific_sign": row["guarded_scientific_sign"],
            "guarded_simultaneous_band_classification": (
                _guarded_band_classification(row)
            ),
            "zero_variance": row["zero_variance"],
            "monte_carlo_replicates": row["monte_carlo_replicates"],
        })
    if len(output) != 118 or {row["family"] for row in output} != {"A", "B", "C"}:
        raise AssertionError("complete primary publication projection failed")
    return output


def _did_estimand_id(row: Mapping[str, Any]) -> str:
    return (
        f"D|{row['policy']}-minus-cEI|{row['metric']}|N={int(row['budget'])}|"
        f"c={_factor_token(float(row['assumed_toxicity_noise_sd_factor']))}-minus-1"
    )


def enrich_policy_cei_did_rows(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if len(rows) != SECONDARY_COUNTS["policy_vs_cei_calibration_did"]:
        raise PublicationAdapterError(
            "policy-vs-cEI calibration DiD publication requires all 36 rows"
        )
    output: list[dict[str, Any]] = []
    for order, row in enumerate(rows, start=1):
        policy = str(row["policy"])
        metric = str(row["metric"])
        output.append({
            "row_order": order,
            "estimand_id": _did_estimand_id(row),
            "analysis_type": row["analysis_type"],
            "policy": policy,
            "policy_label": contract["policy_labels"][policy],
            "reference_policy": "cEI",
            "reference_policy_label": contract["policy_labels"]["cEI"],
            "assumed_toxicity_noise_sd_factor": row[
                "assumed_toxicity_noise_sd_factor"
            ],
            "c_sigma": row["assumed_toxicity_noise_sd_factor"],
            "reference_toxicity_noise_sd_factor": 1.0,
            "budget": row["budget"],
            "metric": metric,
            "metric_label": contract["labels"][METRIC_LABEL_KEYS[metric]],
            "estimate": row["estimate"],
            "mcse": row["mcse"],
            "pointwise_t_critical_95": row["pointwise_t_critical_95"],
            "ci95_low": row["ci95_low"],
            "ci95_high": row["ci95_high"],
            "row_guard_class": row["row_guard_class"],
            "replicate_numerical_bound_points": row[
                "replicate_numerical_bound_points"
            ],
            "mcse_numerical_guard_points": row[
                "mcse_numerical_guard_points"
            ],
            "pointwise_numerical_halfwidth_guard_points": row[
                "pointwise_numerical_halfwidth_guard_points"
            ],
            "guarded_ci95_low": row["guarded_ci95_low"],
            "guarded_ci95_high": row["guarded_ci95_high"],
            "guarded_pointwise_scientific_sign": row[
                "guarded_pointwise_scientific_sign"
            ],
            "monte_carlo_replicates": row["monte_carlo_replicates"],
        })
    ids = [row["estimand_id"] for row in output]
    if len(set(ids)) != SECONDARY_COUNTS["policy_vs_cei_calibration_did"]:
        raise PublicationAdapterError(
            "policy-vs-cEI calibration DiD estimand identifiers are not unique"
        )
    return output


def _scope_state(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classes = [
        str(row["guarded_simultaneous_band_classification"]) for row in rows
    ]
    if classes and all(value == "positive" for value in classes):
        state = "positive"
    elif classes and all(value == "negative" for value in classes):
        state = "negative"
    elif classes and all(value == "unresolved" for value in classes):
        state = "unresolved"
    else:
        state = "mixed"
    return {
        "state": state,
        "resolved_same_direction": state in {"positive", "negative"},
        "positive_count": classes.count("positive"),
        "negative_count": classes.count("negative"),
        "unresolved_count": classes.count("unresolved"),
        "estimand_count": len(rows),
        "estimand_ids": [str(row["estimand_id"]) for row in rows],
    }


def _discordance_predicate(
    rows: Sequence[Mapping[str, Any]],
    *, comparator: str,
    budgets: set[int],
    factors: set[float],
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["family"] == "A" and row["comparator"] == comparator
        and int(row["budget_n"]) in budgets
        and float(row["assumed_toxicity_noise_sd_factor"]) in factors
    ]
    terminal = _scope_state([row for row in selected if row["metric"] == TERMINAL])
    assignment = _scope_state([row for row in selected if row["metric"] == ASSIGNMENT])
    opposite = (
        terminal["state"] in {"positive", "negative"}
        and assignment["state"] in {"positive", "negative"}
        and terminal["state"] != assignment["state"]
    )
    return {
        "supported": bool(opposite),
        "opposite_resolved_directions": bool(opposite),
        "terminal": terminal,
        "assignment": assignment,
        "budgets": sorted(budgets),
        "c_sigma": sorted(factors),
    }


def _acquisition_predicates(
    rows: Sequence[Mapping[str, Any]], comparator: str
) -> dict[str, Any]:
    return {
        "calibration_robust_at_N40": _discordance_predicate(
            rows, comparator=comparator, budgets={40}, factors=set(FACTORS)
        ),
        "budget_robust_within_c_sigma": {
            f"{factor:g}": _discordance_predicate(
                rows, comparator=comparator, budgets=set(BUDGETS), factors={factor}
            )
            for factor in FACTORS
        },
        "robust_over_evaluated_3x3_grid": _discordance_predicate(
            rows, comparator=comparator, budgets=set(BUDGETS), factors=set(FACTORS)
        ),
    }


def _shared_gate_history_cells(
    primary_rows: Sequence[Mapping[str, Any]],
    did_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    primary_by_id = {str(row["estimand_id"]): row for row in primary_rows}
    did_by_cell = {
        (
            str(row["policy"]),
            float(row["assumed_toxicity_noise_sd_factor"]),
            int(row["budget"]),
            str(row["metric"]),
        ): row
        for row in did_rows
    }
    output: list[dict[str, Any]] = []
    for factor in ERRONEOUS_FACTORS:
        for budget in BUDGETS:
            for metric in (EXCLUSION, BRIER):
                penalty_rows: list[dict[str, Any]] = []
                for policy in POLICIES:
                    estimand_id = (
                        f"B|{policy}|{metric}|N={budget}|"
                        f"c={_factor_token(factor)}-minus-1"
                    )
                    row = primary_by_id[estimand_id]
                    penalty_rows.append({
                        "policy": policy,
                        "policy_label": contract["policy_labels"][policy],
                        "estimand_id": estimand_id,
                        "guarded_classification": row[
                            "guarded_simultaneous_band_classification"
                        ],
                    })
                differential_rows: list[dict[str, Any]] = []
                for policy in DID_POLICIES:
                    row = did_by_cell[(policy, factor, budget, metric)]
                    differential_rows.append({
                        "policy": policy,
                        "policy_label": contract["policy_labels"][policy],
                        "estimand_id": row["estimand_id"],
                        "guarded_classification": row[
                            "guarded_pointwise_scientific_sign"
                        ],
                    })
                all_penalties_positive = all(
                    row["guarded_classification"] == "positive"
                    for row in penalty_rows
                )
                all_differentials_unresolved = all(
                    row["guarded_classification"] == "unresolved"
                    for row in differential_rows
                )
                any_differential_resolved = any(
                    row["guarded_classification"] in {"positive", "negative"}
                    for row in differential_rows
                )
                compatible = (
                    all_penalties_positive and all_differentials_unresolved
                )
                if any_differential_resolved:
                    reader_code = "policy_history_modification_resolved"
                elif compatible:
                    reader_code = "shared_gate_fragility_compatible"
                elif not all_penalties_positive:
                    reader_code = "all_policy_penalties_not_guarded_positive"
                else:
                    reader_code = "cell_specific_differential_unresolved"
                output.append({
                    "c_sigma": factor,
                    "budget": budget,
                    "metric": metric,
                    "metric_label": contract["labels"][METRIC_LABEL_KEYS[metric]],
                    "policy_penalties": penalty_rows,
                    "policy_vs_cEI_differentials": differential_rows,
                    "all_four_policy_penalties_guarded_positive": (
                        all_penalties_positive
                    ),
                    "all_three_policy_vs_cEI_differentials_guarded_unresolved": (
                        all_differentials_unresolved
                    ),
                    "any_policy_vs_cEI_differential_guarded_resolved": (
                        any_differential_resolved
                    ),
                    "shared_gate_fragility_compatible": compatible,
                    "reader_code": reader_code,
                    "inference_scope": (
                        "this factor-budget-metric cell only; no majority vote "
                        "or family theorem"
                    ),
                })
    if len(output) != 12:
        raise AssertionError("shared-gate history projection is not 12 cells")
    return output


def build_result_facts(
    bundle: AuthenticatedBundle,
    rows: Sequence[Mapping[str, Any]],
    did_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    family_bands = bundle.primary_multiplicity["family_bands"]
    point_critical = float(rows[0]["pointwise_t_critical_95"])
    class_counts = {
        state: sum(
            row["guarded_simultaneous_band_classification"] == state
            for row in rows
        )
        for state in ("positive", "negative", "unresolved")
    }
    scientific = {
        comparator: _acquisition_predicates(rows, comparator)
        for comparator in PRINCIPAL_COMPARATORS
    }
    cei_reference = _acquisition_predicates(rows, "cEI")
    joint = {
        "calibration_robust_at_N40": all(
            scientific[item]["calibration_robust_at_N40"]["supported"]
            for item in PRINCIPAL_COMPARATORS
        ),
        "budget_robust_within_c_sigma": {
            f"{factor:g}": all(
                scientific[item]["budget_robust_within_c_sigma"][f"{factor:g}"]["supported"]
                for item in PRINCIPAL_COMPARATORS
            )
            for factor in FACTORS
        },
        "robust_over_evaluated_3x3_grid": all(
            scientific[item]["robust_over_evaluated_3x3_grid"]["supported"]
            for item in PRINCIPAL_COMPARATORS
        ),
        "rule": "tMSE-only and Entropy reduction must each meet the same frozen scope",
    }

    by_id = {str(row["estimand_id"]): row for row in rows}
    attenuation: list[dict[str, Any]] = []
    for policy in POLICIES:
        for factor in ERRONEOUS_FACTORS:
            for metric in (EXCLUSION, BRIER):
                b_id = (
                    f"B|{policy}|{metric}|N=20|c={_factor_token(factor)}-minus-1"
                )
                c_id = (
                    f"C|{policy}|{metric}|N80-minus-N20|"
                    f"c={_factor_token(factor)}-minus-1"
                )
                b_class = str(
                    by_id[b_id]["guarded_simultaneous_band_classification"]
                )
                c_class = str(
                    by_id[c_id]["guarded_simultaneous_band_classification"]
                )
                penalty = b_class == "positive"
                negative_interaction = c_class == "negative"
                attenuation_supported = penalty and negative_interaction
                if attenuation_supported:
                    code = "finite_range_attenuation_supported"
                elif not penalty:
                    code = "N20_penalty_not_guarded_positive"
                elif c_class == "positive":
                    code = "penalty_increased_over_N20_to_N80"
                else:
                    code = "guarded_attenuation_interaction_unresolved"
                attenuation.append({
                    "policy": policy,
                    "policy_label": contract["policy_labels"][policy],
                    "metric": metric,
                    "metric_label": contract["labels"][METRIC_LABEL_KEYS[metric]],
                    "c_sigma": factor,
                    "family_B_N20_estimand_id": b_id,
                    "family_B_N20_classification": b_class,
                    "family_C_interaction_estimand_id": c_id,
                    "family_C_interaction_classification": c_class,
                    "N20_penalty_guarded_positive": penalty,
                    "family_C_interaction_guarded_negative": (
                        negative_interaction
                    ),
                    "finite_range_attenuation_supported": attenuation_supported,
                    "reader_code": code,
                })

    mcse_target = float(contract["mcse_target_points"])
    raw_mcse_values = [float(row["mcse_points"]) for row in rows]
    guarded_mcse_values = [
        float(row["mcse_points"]) + float(row["mcse_numerical_guard_points"])
        for row in rows
    ]
    secondary_class_counts = {
        state: sum(
            row["guarded_pointwise_scientific_sign"] == state
            for row in bundle.absolute_rows
        )
        for state in ("positive", "negative", "unresolved", "unavailable")
    }
    shared_gate_cells = _shared_gate_history_cells(rows, did_rows, contract)
    return {
        "schema_version": 3,
        "status": "COMPLETE_BUDGET_TOXICITY_CALIBRATION_RESULT_FACTS",
        "complete_unfiltered_primary_projection": True,
        "baseline_identity": {
            "status": bundle.baseline["status"],
            "passed": bundle.baseline["status"] == BASELINE_STATUS,
            "common_field_cells": bundle.baseline["common_field_cells"],
            "allocation_history_cells": bundle.baseline["allocation_history_cells"],
            "historical_cells_without_allocation_history": bundle.baseline[
                "historical_cells_without_allocation_history"
            ],
        },
        "final_M": bundle.final_m,
        "finite_snapshot_budgets": list(BUDGETS),
        "primary_estimand_count": len(rows),
        "primary_family_counts": dict(PRIMARY_COUNTS),
        "published_policy_vs_cEI_calibration_did_count": len(did_rows),
        "mcse": {
            "target_points": mcse_target,
            "scientific_target_source": "guarded MCSE",
            "raw_meeting_target_count": sum(
                value <= mcse_target for value in raw_mcse_values
            ),
            "raw_not_meeting_target_count": sum(
                value > mcse_target for value in raw_mcse_values
            ),
            "raw_maximum_points": max(raw_mcse_values),
            "guarded_meeting_target_count": sum(
                value <= mcse_target for value in guarded_mcse_values
            ),
            "guarded_not_meeting_target_count": sum(
                value > mcse_target for value in guarded_mcse_values
            ),
            "guarded_maximum_points": max(guarded_mcse_values),
            "all_guarded_meet_target": all(
                value <= mcse_target for value in guarded_mcse_values
            ),
        },
        "numerical_guards": {
            "primary": _primary_numerical_guard_metadata(),
            "secondary": dict(bundle.secondary_guard_metadata),
            "primary_scientific_interval_source": (
                "guarded simultaneous 95% band only"
            ),
            "secondary_scientific_interval_source": (
                "guarded pointwise 95% interval only"
            ),
        },
        "multiplicity": {
            "pointwise_t_critical_95": point_critical,
            "family_max_t_critical_95": {
                family: float(family_bands[family]["critical_value"])
                for family in ("A", "B", "C")
            },
            "family_max_t_draws": {
                family: int(family_bands[family]["draws"])
                for family in ("A", "B", "C")
            },
            "guarded_simultaneous_band_classification_counts": class_counts,
            "plotted_secondary_guarded_pointwise_classification_counts": (
                secondary_class_counts
            ),
        },
        "interpretation_predicates": {
            "scientific_acquisition_comparators": scientific,
            "cEI_shared_gate_reference": cei_reference,
            "tmse_only_and_entropy_same_scope": joint,
            "calibration_penalty_attenuation": attenuation,
            "finite_range_attenuation_supported_count": sum(
                item["finite_range_attenuation_supported"]
                for item in attenuation
            ),
            "policy_vs_cEI_calibration_differential_cells": [
                {
                    "estimand_id": row["estimand_id"],
                    "policy": row["policy"],
                    "policy_label": row["policy_label"],
                    "c_sigma": row["assumed_toxicity_noise_sd_factor"],
                    "budget": row["budget"],
                    "metric": row["metric"],
                    "metric_label": row["metric_label"],
                    "guarded_classification": row[
                        "guarded_pointwise_scientific_sign"
                    ],
                    "guarded_differential_resolved": row[
                        "guarded_pointwise_scientific_sign"
                    ] in {"positive", "negative"},
                    "history_modification_direction": {
                        "positive": "policy_more_sensitive_than_cEI",
                        "negative": "policy_less_sensitive_than_cEI",
                        "unresolved": "unresolved",
                    }[row["guarded_pointwise_scientific_sign"]],
                }
                for row in did_rows
            ],
            "shared_gate_history_cells": shared_gate_cells,
        },
        "interpretation_rules": dict(contract["interpretation"]),
        "reader_labels": dict(contract["labels"]),
        "limitations": list(contract["interpretation"]["prohibited_inferences"]),
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if len(rows) != 118:
        raise PublicationAdapterError("publication CSV requires all 118 primary rows")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise PublicationAdapterError("publication CSV row schemas differ")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "×": r"$\times$",
    }
    return "".join(replacements.get(char, char) for char in text)


def _fmt(value: Any) -> str:
    return f"{float(value):.2f}"


def _longtable_bytes(
    rows: Sequence[Mapping[str, Any]], *, family: str, caption: str, label: str
) -> bytes:
    if len(rows) != PRIMARY_COUNTS[family]:
        raise PublicationAdapterError(f"Family {family} table is incomplete")
    if family == "A":
        header = (
            r"Comparator & Outcome & $c_\sigma$ & $N$ & Estimate & Raw MCSE & "
            r"Raw simultaneous 95\% band & Guarded simultaneous 95\% band & "
            r"Guarded classification \\"
        )
        columns = "p{1.8cm}p{3.0cm}rrrrp{2.6cm}p{2.6cm}p{1.6cm}"
    elif family == "B":
        header = (
            r"Policy & Outcome & $c_\sigma$ vs 1 & $N$ & Estimate & Raw MCSE & "
            r"Raw simultaneous 95\% band & Guarded simultaneous 95\% band & "
            r"Guarded classification \\"
        )
        columns = "p{1.8cm}p{3.0cm}rrrrp{2.6cm}p{2.6cm}p{1.6cm}"
    else:
        header = (
            r"Policy & Outcome & $c_\sigma$ vs 1 & Budget contrast & Estimate & Raw MCSE & "
            r"Raw simultaneous 95\% band & Guarded simultaneous 95\% band & "
            r"Guarded classification \\"
        )
        columns = "p{1.8cm}p{3.0cm}rp{2.0cm}rrp{2.6cm}p{2.6cm}p{1.6cm}"
    lines = [
        f"% Complete frozen Family {family}: {len(rows)} data rows; no sign-based filtering.",
        f"\\begin{{longtable}}{{{columns}}}",
        f"\\caption{{{_tex_escape(caption)}}}\\label{{{label}}}\\\\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
    ]
    for row in rows:
        raw_band = (
            f"[{_fmt(row['simultaneous_95_low_points'])}, "
            f"{_fmt(row['simultaneous_95_high_points'])}]"
        )
        guarded_band = (
            f"[{_fmt(row['guarded_simultaneous_95_low_points'])}, "
            f"{_fmt(row['guarded_simultaneous_95_high_points'])}]"
        )
        if family == "A":
            values = (
                row["comparator_label"], row["metric_label"],
                f"{float(row['assumed_toxicity_noise_sd_factor']):g}",
                row["budget_n"], _fmt(row["estimate_points"]),
                _fmt(row["mcse_points"]), raw_band, guarded_band,
                row["guarded_simultaneous_band_classification"],
            )
        elif family == "B":
            values = (
                row["policy_label"], row["metric_label"],
                f"{float(row['assumed_toxicity_noise_sd_factor']):g}",
                row["budget_n"], _fmt(row["estimate_points"]),
                _fmt(row["mcse_points"]), raw_band, guarded_band,
                row["guarded_simultaneous_band_classification"],
            )
        else:
            values = (
                row["policy_label"], row["metric_label"],
                f"{float(row['assumed_toxicity_noise_sd_factor']):g}",
                row["budget_contrast"], _fmt(row["estimate_points"]),
                _fmt(row["mcse_points"]), raw_band, guarded_band,
                row["guarded_simultaneous_band_classification"],
            )
        lines.append(" & ".join(_tex_escape(value) for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _did_longtable_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if len(rows) != SECONDARY_COUNTS["policy_vs_cei_calibration_did"]:
        raise PublicationAdapterError(
            "policy-vs-cEI calibration DiD table requires all 36 rows"
        )
    lines = [
        "% Complete frozen policy-vs-cEI calibration DiD projection: 36 rows; no sign-based filtering.",
        r"\begin{longtable}{p{1.9cm}rp{3.1cm}rrrrp{2.7cm}p{2.7cm}p{1.5cm}}",
        (
            r"\caption{All 36 policy-vs-cEI calibration difference-in-differences "
            r"with raw and numerically guarded pointwise intervals in the finite "
            r"fixed-design audit.}\label{tab:budget-toxicity-calibration-policy-cei-did}\\"
        ),
        r"\toprule",
        (
            r"Policy & $c_\sigma$ vs 1 & Outcome & $N$ & Estimate & Raw MCSE & "
            r"Raw pointwise 95\% interval & Guarded pointwise 95\% interval & "
            r"Guarded classification \\"
        ),
        r"\midrule", r"\endfirsthead", r"\toprule",
        (
            r"Policy & $c_\sigma$ vs 1 & Outcome & $N$ & Estimate & Raw MCSE & "
            r"Raw pointwise 95\% interval & Guarded pointwise 95\% interval & "
            r"Guarded classification \\"
        ),
        r"\midrule", r"\endhead",
    ]
    for row in rows:
        raw_interval = (
            f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}]"
        )
        guarded_interval = (
            f"[{_fmt(row['guarded_ci95_low'])}, "
            f"{_fmt(row['guarded_ci95_high'])}]"
        )
        values = (
            row["policy_label"],
            f"{float(row['assumed_toxicity_noise_sd_factor']):g}",
            row["metric_label"], row["budget"], _fmt(row["estimate"]),
            _fmt(row["mcse"]), raw_interval, guarded_interval,
            row["guarded_pointwise_scientific_sign"],
        )
        lines.append(" & ".join(_tex_escape(value) for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _bool_token(value: bool) -> str:
    return "true" if value else "false"


def _facts_tex_bytes(facts: Mapping[str, Any]) -> bytes:
    multiplicity = facts["multiplicity"]
    predicates = facts["interpretation_predicates"]
    lines = [
        "% Deterministic facts generated from all 118 primary estimands and the complete 36-row policy-vs-cEI DiD projection.",
        (
            "% true/false values use only the prespecified numerically guarded "
            "intervals (primary simultaneous bands and, where applicable, "
            "secondary pointwise intervals)."
        ),
        rf"\providecommand{{\CalBaselineIdentityStatus}}{{{_tex_escape(facts['baseline_identity']['status'])}}}",
        rf"\providecommand{{\CalFinalM}}{{{facts['final_M']}}}",
        rf"\providecommand{{\CalPrimaryEstimandCount}}{{{facts['primary_estimand_count']}}}",
        rf"\providecommand{{\CalMCSETargetPoints}}{{{_fmt(facts['mcse']['target_points'])}}}",
        rf"\providecommand{{\CalRawMCSEMeetingTargetCount}}{{{facts['mcse']['raw_meeting_target_count']}}}",
        rf"\providecommand{{\CalRawMaximumMCSEPoints}}{{{_fmt(facts['mcse']['raw_maximum_points'])}}}",
        rf"\providecommand{{\CalGuardedMCSEMeetingTargetCount}}{{{facts['mcse']['guarded_meeting_target_count']}}}",
        rf"\providecommand{{\CalGuardedMCSENotMeetingTargetCount}}{{{facts['mcse']['guarded_not_meeting_target_count']}}}",
        rf"\providecommand{{\CalGuardedMaximumMCSEPoints}}{{{_fmt(facts['mcse']['guarded_maximum_points'])}}}",
        rf"\providecommand{{\CalPointwiseCritical}}{{{float(multiplicity['pointwise_t_critical_95']):.6f}}}",
    ]
    for family in ("A", "B", "C"):
        lines.append(
            rf"\providecommand{{\CalFamily{family}MaxTCritical}}{{{float(multiplicity['family_max_t_critical_95'][family]):.6f}}}"
        )
    tokens = {"tmse": "TMSE", "qBIG": "Entropy", "cEI": "CEI"}
    for comparator, values in {
        **predicates["scientific_acquisition_comparators"],
        "cEI": predicates["cEI_shared_gate_reference"],
    }.items():
        token = tokens[comparator]
        lines.append(
            rf"\providecommand{{\Cal{token}CalibrationRobustNForty}}{{{_bool_token(values['calibration_robust_at_N40']['supported'])}}}"
        )
        calibration_scope = values["calibration_robust_at_N40"]
        lines.append(
            rf"\providecommand{{\Cal{token}CalibrationNFortyTerminalState}}{{{calibration_scope['terminal']['state']}}}"
        )
        lines.append(
            rf"\providecommand{{\Cal{token}CalibrationNFortyAssignmentState}}{{{calibration_scope['assignment']['state']}}}"
        )
        lines.append(
            rf"\providecommand{{\Cal{token}RobustFullGrid}}{{{_bool_token(values['robust_over_evaluated_3x3_grid']['supported'])}}}"
        )
        full_scope = values["robust_over_evaluated_3x3_grid"]
        lines.append(
            rf"\providecommand{{\Cal{token}FullGridTerminalState}}{{{full_scope['terminal']['state']}}}"
        )
        lines.append(
            rf"\providecommand{{\Cal{token}FullGridAssignmentState}}{{{full_scope['assignment']['state']}}}"
        )
        for factor, factor_token in ((0.5, "Low"), (1.0, "Correct"), (2.0, "High")):
            factor_scope = values["budget_robust_within_c_sigma"][f"{factor:g}"]
            supported = factor_scope["supported"]
            lines.append(
                rf"\providecommand{{\Cal{token}BudgetRobust{factor_token}}}{{{_bool_token(supported)}}}"
            )
            lines.append(
                rf"\providecommand{{\Cal{token}Budget{factor_token}TerminalState}}{{{factor_scope['terminal']['state']}}}"
            )
            lines.append(
                rf"\providecommand{{\Cal{token}Budget{factor_token}AssignmentState}}{{{factor_scope['assignment']['state']}}}"
            )
    joint = predicates["tmse_only_and_entropy_same_scope"]
    lines.extend([
        rf"\providecommand{{\CalPrincipalBothCalibrationRobustNForty}}{{{_bool_token(joint['calibration_robust_at_N40'])}}}",
        rf"\providecommand{{\CalPrincipalBothRobustFullGrid}}{{{_bool_token(joint['robust_over_evaluated_3x3_grid'])}}}",
        rf"\providecommand{{\CalFiniteRangeAttenuationSupportedCount}}{{{predicates['finite_range_attenuation_supported_count']}}}",
    ])
    policy_tokens = {
        "cKG-exact-formal": "CKG", "tmse": "TMSE",
        "qBIG": "Entropy", "cEI": "CEI",
    }
    metric_tokens = {EXCLUSION: "Exclusion", BRIER: "Brier"}
    factor_tokens = {0.5: "Low", 2.0: "High"}
    for item in predicates["calibration_penalty_attenuation"]:
        macro = (
            "\\CalAttenuation" + policy_tokens[item["policy"]]
            + metric_tokens[item["metric"]] + factor_tokens[float(item["c_sigma"])]
        )
        lines.append(
            rf"\providecommand{{{macro}}}{{{_bool_token(item['finite_range_attenuation_supported'])}}}"
        )
    budget_tokens = {20: "NTwenty", 40: "NForty", 80: "NEighty"}
    for item in predicates["policy_vs_cEI_calibration_differential_cells"]:
        macro = (
            "\\CalDiD" + policy_tokens[item["policy"]]
            + metric_tokens[item["metric"]]
            + factor_tokens[float(item["c_sigma"])]
            + budget_tokens[int(item["budget"])] + "State"
        )
        lines.append(
            rf"\providecommand{{{macro}}}{{{item['guarded_classification']}}}"
        )
    for cell in predicates["shared_gate_history_cells"]:
        suffix = (
            metric_tokens[cell["metric"]]
            + factor_tokens[float(cell["c_sigma"])]
            + budget_tokens[int(cell["budget"])]
        )
        lines.append(
            rf"\providecommand{{\CalSharedGateCompatible{suffix}}}{{{_bool_token(cell['shared_gate_fragility_compatible'])}}}"
        )
        lines.append(
            rf"\providecommand{{\CalPolicyHistoryModified{suffix}}}{{{_bool_token(cell['any_policy_vs_cEI_differential_guarded_resolved'])}}}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _plot_setup() -> tuple[Any, Any]:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "path.simplify": False,
        "savefig.transparent": False,
    })
    import matplotlib.pyplot as plt
    return matplotlib, plt


def _draw_line(
    ax: Any,
    points: Sequence[Mapping[str, Any]],
    *,
    factor: float,
    color: str,
    absolute: bool = False,
) -> None:
    if absolute:
        x = [int(row["budget"]) for row in points]
        y = [float(row["estimate"]) for row in points]
        low = [float(row["guarded_ci95_low"]) for row in points]
        high = [float(row["guarded_ci95_high"]) for row in points]
    else:
        x = [int(row["budget_n"]) for row in points]
        y = [float(row["estimate_points"]) for row in points]
        low = [
            float(row["guarded_simultaneous_95_low_points"]) for row in points
        ]
        high = [
            float(row["guarded_simultaneous_95_high_points"]) for row in points
        ]
    lower = [max(0.0, center - endpoint) for center, endpoint in zip(y, low)]
    upper = [max(0.0, endpoint - center) for center, endpoint in zip(y, high)]
    ax.errorbar(
        x, y, yerr=[lower, upper], color=color, marker="o", markersize=4,
        linewidth=1.35, elinewidth=0.85, capsize=2.2, capthick=0.85,
        label=f"{factor:g}", zorder=2,
    )


def _save_figure(
    fig: Any,
    *,
    output_dir: Path,
    stem: str,
    caption: str,
    formats: Sequence[str],
) -> list[Path]:
    outputs: list[Path] = []
    fixed_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
    for suffix in formats:
        path = _stage_child(
            output_dir, f"{stem}.{suffix}", label=f"{stem} {suffix} figure"
        )
        if suffix == "png":
            metadata = {
                "Title": caption,
                "Author": "dose-combination-bo calibration publication adapter",
                "Software": "Matplotlib",
            }
            fig.savefig(
                path, format="png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none", transparent=False,
                metadata=metadata,
            )
        elif suffix == "pdf":
            metadata = {
                "Title": caption,
                "Author": "dose-combination-bo calibration publication adapter",
                "Creator": "dose-combination-bo calibration publication adapter",
                "Producer": "Matplotlib",
                "CreationDate": fixed_date,
                "ModDate": fixed_date,
            }
            fig.savefig(
                path, format="pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none", transparent=False,
                metadata=metadata,
            )
        else:
            raise PublicationAdapterError(f"unsupported figure format: {suffix}")
        outputs.append(path)
    return outputs


def make_publication_figures(
    primary_rows: Sequence[Mapping[str, Any]],
    absolute_rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    contract: Mapping[str, Any],
) -> dict[Path, dict[str, Any]]:
    previous_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "946684800"
    _matplotlib, plt = _plot_setup()
    colors = {
        float(key): value for key, value in contract["colors_by_c_sigma"].items()
    }
    labels = contract["labels"]
    figures = contract["figures"]
    captions = contract["captions"]
    policy_labels = contract["policy_labels"]
    grid_color = "#D9D9D9"
    outputs: dict[Path, dict[str, Any]] = {}

    def finish(
        fig: Any, *, figure_key: str, selected_primary: int,
        selected_secondary: int,
    ) -> None:
        spec = figures[figure_key]
        if (
            selected_primary != spec["primary_rows"]
            or selected_secondary != spec["secondary_absolute_rows"]
        ):
            raise PublicationAdapterError(f"{figure_key} selection count changed")
        fig.suptitle(labels["audit_scope"], fontsize=9.5, fontweight="normal")
        paths = _save_figure(
            fig, output_dir=output_dir, stem=spec["stem"],
            caption=captions[figure_key], formats=figures["formats"],
        )
        for path in paths:
            outputs[path] = {
                "artifact_type": f"publication_figure_{figure_key}",
                "description": captions[figure_key],
                "selection": {
                    "primary_rows": selected_primary,
                    "secondary_absolute_rows": selected_secondary,
                    "complete_without_sign_filtering": True,
                },
            }
        plt.close(fig)

    principal = [
        row for row in primary_rows
        if row["family"] == "A" and row["comparator"] in PRINCIPAL_COMPARATORS
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex=True)
    for column, comparator in enumerate(PRINCIPAL_COMPARATORS):
        for row_index, metric in enumerate((TERMINAL, ASSIGNMENT)):
            ax = axes[row_index, column]
            for factor in FACTORS:
                points = sorted([
                    row for row in principal
                    if row["comparator"] == comparator and row["metric"] == metric
                    and float(row["assumed_toxicity_noise_sd_factor"]) == factor
                ], key=lambda row: int(row["budget_n"]))
                _draw_line(ax, points, factor=factor, color=colors[factor])
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(f"cKG minus {policy_labels[comparator]}", fontweight="normal")
            if column == 0:
                ax.set_ylabel(labels[METRIC_LABEL_KEYS[metric]] + "\n" + labels["difference_axis"])
            if row_index == 1:
                ax.set_xlabel(labels["x_budget"])
            ax.set_xticks(BUDGETS)
            ax.grid(color=grid_color, linewidth=0.6)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="lower center", ncol=3, frameon=False,
        title=labels["c_sigma_axis"],
    )
    fig.subplots_adjust(bottom=0.17, hspace=0.34, wspace=0.28)
    finish(fig, figure_key="principal_profile", selected_primary=len(principal), selected_secondary=0)

    cei = [
        row for row in primary_rows
        if row["family"] == "A" and row["comparator"] == "cEI"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    for column, metric in enumerate((TERMINAL, ASSIGNMENT)):
        ax = axes[column]
        for factor in FACTORS:
            points = sorted([
                row for row in cei if row["metric"] == metric
                and float(row["assumed_toxicity_noise_sd_factor"]) == factor
            ], key=lambda row: int(row["budget_n"]))
            _draw_line(ax, points, factor=factor, color=colors[factor])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("cKG minus cEI (shared-gate reference)", fontweight="normal")
        ax.set_ylabel(labels[METRIC_LABEL_KEYS[metric]] + "\n" + labels["difference_axis"])
        ax.set_xlabel(labels["x_budget"])
        ax.set_xticks(BUDGETS)
        ax.grid(color=grid_color, linewidth=0.6)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="lower center", ncol=3, frameon=False,
        title=labels["c_sigma_axis"],
    )
    fig.subplots_adjust(bottom=0.28, wspace=0.30)
    finish(fig, figure_key="cei_reference", selected_primary=len(cei), selected_secondary=0)

    family_b = [row for row in primary_rows if row["family"] == "B"]
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.2), sharex=True)
    for column, policy in enumerate(POLICIES):
        for row_index, metric in enumerate((EXCLUSION, BRIER)):
            ax = axes[row_index, column]
            for factor in ERRONEOUS_FACTORS:
                points = sorted([
                    row for row in family_b
                    if row["policy"] == policy and row["metric"] == metric
                    and float(row["assumed_toxicity_noise_sd_factor"]) == factor
                ], key=lambda row: int(row["budget_n"]))
                _draw_line(ax, points, factor=factor, color=colors[factor])
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(policy_labels[policy], fontweight="normal")
            if column == 0:
                ax.set_ylabel(labels[METRIC_LABEL_KEYS[metric]] + "\n" + labels["difference_axis"])
            if row_index == 1:
                ax.set_xlabel(labels["x_budget"])
            ax.set_xticks(BUDGETS)
            ax.grid(color=grid_color, linewidth=0.6)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="lower center", ncol=2, frameon=False,
        title=labels["c_sigma_axis"],
    )
    fig.subplots_adjust(bottom=0.17, hspace=0.34, wspace=0.34)
    finish(fig, figure_key="calibration_penalties", selected_primary=len(family_b), selected_secondary=0)

    family_c = [row for row in primary_rows if row["family"] == "C"]
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.2), sharex=True)
    for column, policy in enumerate(POLICIES):
        for row_index, metric in enumerate((EXCLUSION, BRIER)):
            ax = axes[row_index, column]
            selected = [
                row for row in family_c
                if row["policy"] == policy and row["metric"] == metric
            ]
            for item in selected:
                factor = float(item["assumed_toxicity_noise_sd_factor"])
                y = float(item["estimate_points"])
                low = float(item["guarded_simultaneous_95_low_points"])
                high = float(item["guarded_simultaneous_95_high_points"])
                ax.errorbar(
                    [factor], [y], yerr=[[max(0.0, y-low)], [max(0.0, high-y)]],
                    color=colors[factor], marker="o", markersize=4,
                    linestyle="none", elinewidth=0.9, capsize=2.2,
                    label=f"{factor:g}", zorder=2,
                )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(policy_labels[policy], fontweight="normal")
            if column == 0:
                ax.set_ylabel(labels[METRIC_LABEL_KEYS[metric]] + "\nN=80 penalty minus N=20 penalty (points)")
            if row_index == 1:
                ax.set_xlabel(labels["c_sigma_axis"])
            ax.set_xticks(ERRONEOUS_FACTORS)
            ax.grid(color=grid_color, linewidth=0.6)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(bottom=0.15, hspace=0.34, wspace=0.34)
    finish(fig, figure_key="attenuation_interactions", selected_primary=len(family_c), selected_secondary=0)

    absolute = list(absolute_rows)
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.2), sharex=True)
    for column, policy in enumerate(POLICIES):
        for row_index, metric in enumerate((EXCLUSION, BRIER)):
            ax = axes[row_index, column]
            for factor in FACTORS:
                points = sorted([
                    row for row in absolute
                    if row["policy"] == policy and row["metric"] == metric
                    and float(row["assumed_toxicity_noise_sd_factor"]) == factor
                ], key=lambda row: int(row["budget"]))
                _draw_line(ax, points, factor=factor, color=colors[factor], absolute=True)
            ax.set_title(policy_labels[policy], fontweight="normal")
            if column == 0:
                ax.set_ylabel(labels[METRIC_LABEL_KEYS[metric]])
            if row_index == 1:
                ax.set_xlabel(labels["x_budget"])
            ax.set_xticks(BUDGETS)
            ax.grid(color=grid_color, linewidth=0.6)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="lower center", ncol=3, frameon=False,
        title=labels["c_sigma_axis"],
    )
    fig.subplots_adjust(bottom=0.17, hspace=0.34, wspace=0.34)
    finish(
        fig, figure_key="absolute_gate_brier_profiles",
        selected_primary=0, selected_secondary=len(absolute),
    )
    if previous_epoch is None:
        os.environ.pop("SOURCE_DATE_EPOCH", None)
    else:
        os.environ["SOURCE_DATE_EPOCH"] = previous_epoch
    return outputs


def _artifact_sidecar(
    artifact: Path,
    *,
    stage: Path,
    artifact_type: str,
    description: str,
    selection: Mapping[str, Any],
    adapter_sha256: str,
    contract_name: str,
    contract_sha256: str,
    input_bundle_sha256: str,
) -> Path:
    metadata = {
        "schema_version": 3,
        "status": PUBLICATION_ARTIFACT_STATUS,
        "artifact_class": PUBLICATION_ARTIFACT_CLASS,
        "artifact_type": artifact_type,
        "artifact": artifact.name,
        "artifact_sha256": _sha256_file(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "adapter": "paper/generate_budget_toxicity_calibration_publication.py",
        "adapter_sha256": adapter_sha256,
        "contract": contract_name,
        "contract_sha256": contract_sha256,
        "trusted_commitments": dict(TRUSTED_COMMITMENTS),
        "input_bundle_sha256": input_bundle_sha256,
        "description": description,
        "selection": dict(selection),
        "no_filtering_by_sign_interval_or_winner": True,
    }
    if artifact.parent.resolve(strict=False) != stage.resolve(strict=False):
        raise PublicationAdapterError("publication artifact is outside its stage")
    path = _stage_child(
        stage, artifact.name + ".metadata.json",
        label=f"{artifact.name} metadata sidecar",
    )
    _atomic_write(path, _canonical_json_bytes(metadata))
    return path


def _bundle_hash(mapping: Mapping[str, str]) -> str:
    return _sha256_bytes(_canonical_json_bytes(dict(sorted(mapping.items()))))


def _directory_open_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise PublicationAdapterError(
            "platform lacks the no-follow directory primitives required to fail closed"
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _scan_regular_tree_nofollow(root: Path, *, label: str) -> tuple[Path, ...]:
    """No-follow lstat every descendant; allow only dirs and regular files."""

    lexical_root = _absolute_without_symlink_resolution(root)
    try:
        root_lstat = os.lstat(lexical_root)
    except OSError as exc:
        raise PublicationAdapterError(f"cannot lstat {label}") from exc
    if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
        raise PublicationAdapterError(f"{label} root is not a regular directory")
    try:
        root_fd = os.open(lexical_root, _directory_open_flags())
    except OSError as exc:
        raise PublicationAdapterError(f"cannot safely open {label}") from exc
    files: list[Path] = []

    def walk(directory_fd: int, relative_directory: Path) -> None:
        try:
            directory_stat = os.fstat(directory_fd)
            if not stat.S_ISDIR(directory_stat.st_mode):
                raise PublicationAdapterError(f"{label} contains a non-directory branch")
            with os.scandir(directory_fd) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            raise PublicationAdapterError(f"cannot scan {label} without following links") from exc
        for entry in entries:
            name = entry.name
            if (
                not name or Path(name).name != name or name in {".", ".."}
                or "/" in name or "\\" in name
            ):
                raise PublicationAdapterError(f"{label} contains an unsafe entry name")
            relative = relative_directory / name
            candidate = _absolute_without_symlink_resolution(lexical_root / relative)
            expected_parent = _absolute_without_symlink_resolution(
                lexical_root / relative_directory
            )
            if candidate.parent != expected_parent or not _is_within(candidate, lexical_root):
                raise PublicationAdapterError(
                    f"{label} descendant is not directly contained under its destination"
                )
            try:
                # DirEntry.stat(..., follow_symlinks=False) is the directory-fd
                # equivalent of lstat and never resolves the candidate target.
                observed = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise PublicationAdapterError(f"cannot lstat {label} descendant") from exc
            mode = observed.st_mode
            if stat.S_ISLNK(mode):
                raise PublicationAdapterError(f"{label} contains a symlink: {relative}")
            if stat.S_ISDIR(mode):
                try:
                    child_fd = os.open(
                        name, _directory_open_flags(), dir_fd=directory_fd
                    )
                except OSError as exc:
                    raise PublicationAdapterError(
                        f"cannot safely open {label} directory: {relative}"
                    ) from exc
                try:
                    opened = os.fstat(child_fd)
                    if (
                        not stat.S_ISDIR(opened.st_mode)
                        or (opened.st_dev, opened.st_ino)
                        != (observed.st_dev, observed.st_ino)
                    ):
                        raise PublicationAdapterError(
                            f"{label} directory changed during no-follow scan: {relative}"
                        )
                    walk(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(mode):
                files.append(relative)
            else:
                raise PublicationAdapterError(
                    f"{label} contains a non-regular file: {relative}"
                )

    try:
        opened_root = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(opened_root.st_mode)
            or (opened_root.st_dev, opened_root.st_ino)
            != (root_lstat.st_dev, root_lstat.st_ino)
        ):
            raise PublicationAdapterError(f"{label} root changed during no-follow scan")
        walk(root_fd, Path())
    finally:
        os.close(root_fd)
    return tuple(sorted(files))


def _sha256_regular_nofollow(root: Path, relative: Path) -> str:
    """Hash one direct-contained regular file through no-follow directory FDs."""

    if (
        relative.is_absolute() or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise PublicationAdapterError("tree-hash candidate is not a safe relative file")
    lexical_root = _absolute_without_symlink_resolution(root)
    candidate = _absolute_without_symlink_resolution(lexical_root / relative)
    if not _is_within(candidate, lexical_root) or candidate == lexical_root:
        raise PublicationAdapterError("tree-hash candidate escapes its destination")
    directory_fds: list[int] = []
    file_fd: int | None = None
    try:
        current_fd = os.open(lexical_root, _directory_open_flags())
        directory_fds.append(current_fd)
        for part in relative.parts[:-1]:
            current_fd = os.open(
                part, _directory_open_flags(), dir_fd=current_fd
            )
            directory_fds.append(current_fd)
        filename = relative.parts[-1]
        before = os.stat(filename, dir_fd=current_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise PublicationAdapterError("tree-hash candidate is not a regular file")
        file_fd = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=current_fd,
        )
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise PublicationAdapterError("tree-hash candidate changed before hashing")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise PublicationAdapterError("no-follow tree hash failed closed") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


def _same_tree(left: Path, right: Path) -> bool:
    """Compare two regular trees without following any link; fail closed."""

    try:
        left_files = _scan_regular_tree_nofollow(left, label="candidate publication tree")
        right_files = _scan_regular_tree_nofollow(right, label="existing publication tree")
        if left_files != right_files:
            return False
        return all(
            _sha256_regular_nofollow(left, relative)
            == _sha256_regular_nofollow(right, relative)
            for relative in left_files
        )
    except (OSError, PublicationAdapterError):
        return False


def _build_publication_tree(
    stage: Path,
    *,
    bundle: AuthenticatedBundle,
    contract: Mapping[str, Any],
    contract_path: Path,
    contract_raw: bytes,
) -> dict[str, Any]:
    adapter_path = Path(__file__).resolve()
    adapter_sha = _sha256_file(adapter_path)
    contract_sha = _sha256_bytes(contract_raw)
    input_bundle_sha = _bundle_hash(bundle.input_hashes)
    outputs = contract["outputs"]
    enriched = enrich_primary_rows(bundle.primary_rows, contract)
    enriched_did = enrich_policy_cei_did_rows(
        bundle.policy_cei_did_rows, contract
    )
    facts = build_result_facts(bundle, enriched, enriched_did, contract)

    artifacts: dict[Path, dict[str, Any]] = {}
    all_csv = _stage_child(
        stage, outputs["all_primary_csv"], label="all-primary CSV"
    )
    all_json = _stage_child(
        stage, outputs["all_primary_json"], label="all-primary JSON"
    )
    _atomic_write(all_csv, _csv_bytes(enriched))
    _atomic_write(all_json, _canonical_json_bytes({
        "schema_version": 3,
        "status": "COMPLETE_UNFILTERED_BUDGET_TOXICITY_CALIBRATION_PRIMARY_118",
        "complete_without_sign_interval_or_winner_filtering": True,
        "row_count": 118,
        "family_counts": dict(PRIMARY_COUNTS),
        "rows": enriched,
    }))
    artifacts[all_csv] = {
        "artifact_type": "all_118_primary_csv",
        "description": "Complete 118-row primary publication table in CSV form.",
        "selection": {"primary_rows": 118, "family_counts": dict(PRIMARY_COUNTS)},
    }
    artifacts[all_json] = {
        "artifact_type": "all_118_primary_json",
        "description": "Complete 118-row primary publication table in JSON form.",
        "selection": {"primary_rows": 118, "family_counts": dict(PRIMARY_COUNTS)},
    }

    plotted_secondary_json = _stage_child(
        stage,
        outputs["plotted_secondary_json"],
        label="plotted-secondary JSON",
    )
    _atomic_write(plotted_secondary_json, _canonical_json_bytes({
        "schema_version": 3,
        "status": "COMPLETE_GUARDED_PLOTTED_SECONDARY_72",
        "interval_source": "guarded pointwise 95% interval only",
        "raw_fields_retained": [
            "estimate", "mcse", "ci95_low", "ci95_high",
        ],
        "guarded_fields_used_for_scientific_display": [
            "guarded_ci95_low", "guarded_ci95_high",
            "guarded_pointwise_scientific_sign",
        ],
        "row_count": len(bundle.absolute_rows),
        "rows": list(bundle.absolute_rows),
    }))
    artifacts[plotted_secondary_json] = {
        "artifact_type": "plotted_secondary_absolute_72_json",
        "description": (
            "All 72 plotted absolute gate/Brier rows with raw estimates, raw "
            "MCSEs, raw intervals, numerical guards, and guarded intervals."
        ),
        "selection": {
            "secondary_absolute_rows": 72,
            "complete_without_sign_filtering": True,
        },
    }

    did_json = _stage_child(
        stage, outputs["policy_cei_did_json"],
        label="policy-vs-cEI calibration DiD JSON",
    )
    did_tex = _stage_child(
        stage, outputs["policy_cei_did_tex"],
        label="policy-vs-cEI calibration DiD longtable",
    )
    _atomic_write(did_json, _canonical_json_bytes({
        "schema_version": 3,
        "status": "COMPLETE_GUARDED_POLICY_VS_CEI_CALIBRATION_DID_36",
        "analysis_type": SECONDARY_ANALYSIS_CALIBRATION_DID,
        "complete_without_sign_filtering": True,
        "inference_scope": (
            "cell-specific guarded pointwise statements only; no majority vote "
            "or family theorem"
        ),
        "row_count": len(enriched_did),
        "rows": enriched_did,
    }))
    _atomic_write(did_tex, _did_longtable_bytes(enriched_did))
    artifacts[did_json] = {
        "artifact_type": "complete_policy_vs_cEI_calibration_did_36_json",
        "description": (
            "All 36 frozen policy-vs-cEI calibration difference-in-differences "
            "with raw and guarded pointwise intervals."
        ),
        "selection": {
            "secondary_policy_vs_cEI_calibration_did_rows": 36,
            "complete_without_sign_filtering": True,
        },
    }
    artifacts[did_tex] = {
        "artifact_type": "complete_policy_vs_cEI_calibration_did_36_longtable",
        "description": (
            "Reader-facing complete 36-row policy-vs-cEI calibration "
            "difference-in-differences table."
        ),
        "selection": {
            "secondary_policy_vs_cEI_calibration_did_rows": 36,
            "complete_without_sign_filtering": True,
        },
    }

    table_specs = {
        "A": (
            outputs["family_A_tex"],
            "All 54 cKG-minus-comparator policy contrasts; cEI is shown only as a shared-gate reference.",
            "tab:budget-toxicity-calibration-family-a",
        ),
        "B": (
            outputs["family_B_tex"],
            "All 48 erroneous-minus-correct assumed toxicity-noise calibration contrasts.",
            "tab:budget-toxicity-calibration-family-b",
        ),
        "C": (
            outputs["family_C_tex"],
            "All 16 N=80-minus-N=20 calibration-penalty interactions used by the frozen finite-range attenuation predicate in the finite fixed-design audit.",
            "tab:budget-toxicity-calibration-family-c",
        ),
    }
    for family, (filename, caption, latex_label) in table_specs.items():
        table_path = _stage_child(
            stage, filename, label=f"Family {family} longtable"
        )
        family_rows = [row for row in enriched if row["family"] == family]
        _atomic_write(
            table_path,
            _longtable_bytes(
                family_rows, family=family, caption=caption, label=latex_label
            ),
        )
        artifacts[table_path] = {
            "artifact_type": f"complete_family_{family}_longtable",
            "description": caption,
            "selection": {"family": family, "primary_rows": len(family_rows)},
        }

    facts_json = _stage_child(
        stage, outputs["result_facts_json"], label="result-facts JSON"
    )
    facts_tex = _stage_child(
        stage, outputs["result_facts_tex"], label="result-facts TeX"
    )
    _atomic_write(facts_json, _canonical_json_bytes(facts))
    _atomic_write(facts_tex, _facts_tex_bytes(facts))
    artifacts[facts_json] = {
        "artifact_type": "frozen_result_facts_json",
        "description": "Authenticated counts, critical values, and numerically guarded interpretation predicates.",
        "selection": {"primary_rows": 118, "predicate_rule_frozen_before_outcomes": True},
    }
    artifacts[facts_tex] = {
        "artifact_type": "frozen_result_facts_tex",
        "description": "TeX macros for authenticated counts, critical values, and numerically guarded interpretation predicates.",
        "selection": {"primary_rows": 118, "predicate_rule_frozen_before_outcomes": True},
    }

    artifacts.update(make_publication_figures(
        enriched, bundle.absolute_rows, output_dir=stage, contract=contract
    ))
    sidecars: dict[Path, Path] = {}
    for artifact, details in sorted(artifacts.items(), key=lambda item: item[0].name):
        sidecars[artifact] = _artifact_sidecar(
            artifact, stage=stage,
            artifact_type=details["artifact_type"],
            description=details["description"],
            selection=details["selection"],
            adapter_sha256=adapter_sha, contract_name=contract_path.name,
            contract_sha256=contract_sha,
            input_bundle_sha256=input_bundle_sha,
        )

    artifact_manifest = {
        artifact.name: {
            "sha256": _sha256_file(artifact),
            "bytes": artifact.stat().st_size,
            "metadata": sidecars[artifact].name,
            "metadata_sha256": _sha256_file(sidecars[artifact]),
            "artifact_type": artifacts[artifact]["artifact_type"],
        }
        for artifact in sorted(artifacts, key=lambda path: path.name)
    }
    output_hashes: dict[str, str] = {}
    for artifact in sorted(artifacts, key=lambda path: path.name):
        output_hashes[artifact.name] = _sha256_file(artifact)
        output_hashes[sidecars[artifact].name] = _sha256_file(sidecars[artifact])
    output_bundle_sha = _bundle_hash(output_hashes)
    metadata_path = _stage_child(
        stage, outputs["publication_metadata"], label="publication metadata"
    )
    metadata = {
        "schema_version": 3,
        "status": PUBLICATION_STATUS,
        "artifact_class": PUBLICATION_ARTIFACT_CLASS,
        "adapter": "paper/generate_budget_toxicity_calibration_publication.py",
        "adapter_sha256": adapter_sha,
        "contract": contract_path.name,
        "contract_sha256": contract_sha,
        "trusted_commitments": dict(TRUSTED_COMMITMENTS),
        "inputs": {
            name: {"sha256": bundle.input_hashes[name], "bytes": bundle.input_bytes[name]}
            for name in INPUT_NAMES
        },
        "input_bundle_sha256": input_bundle_sha,
        "authenticated_analysis": {
            "status": bundle.analysis_metadata["status"],
            "commit_status": bundle.analysis_commit["status"],
            "baseline_status": bundle.baseline["status"],
            "final_M": bundle.final_m,
            "primary_estimand_count": 118,
            "primary_family_counts": dict(PRIMARY_COUNTS),
            "secondary_absolute_row_count": SECONDARY_COUNTS["absolute"],
            "secondary_contrast_row_count": SECONDARY_COUNTS[
                "contrast_and_conditional"
            ],
            "secondary_absolute_profile_rows_used": 72,
            "secondary_policy_vs_cEI_calibration_did_rows_published": 36,
        },
        "publication_rules": {
            "complete_primary_projection": True,
            "no_filtering_by_sign_interval_or_winner": True,
            "colors_encode_only_c_sigma": True,
            "finite_fixed_design_audit": True,
            "raw_estimates_mcse_and_intervals_retained": True,
            "primary_scientific_display_uses_guarded_simultaneous_bands": True,
            "secondary_scientific_display_uses_guarded_pointwise_intervals": True,
            "policy_vs_cEI_calibration_did_projection_complete": True,
            "secondary_cell_predicates_have_no_vote_or_family_theorem": True,
        },
        "artifacts": artifact_manifest,
        "output_bundle_sha256": output_bundle_sha,
        "deterministic_timestamp_utc": "2000-01-01T00:00:00Z",
    }
    _atomic_write(metadata_path, _canonical_json_bytes(metadata))
    commit_path = _stage_child(
        stage, outputs["publication_commit"], label="publication commit"
    )
    commit = {
        "schema_version": 3,
        "status": PUBLICATION_COMMIT_STATUS,
        "metadata": metadata_path.name,
        "metadata_sha256": _sha256_file(metadata_path),
        "artifact_count": len(artifact_manifest),
        "adapter_sha256": adapter_sha,
        "contract_sha256": contract_sha,
        "trusted_commitments": dict(TRUSTED_COMMITMENTS),
        "input_bundle_sha256": input_bundle_sha,
        "output_bundle_sha256": output_bundle_sha,
        "baseline_identity_passed": True,
        "primary_estimand_count": 118,
    }
    _atomic_write(commit_path, _canonical_json_bytes(commit))
    return {
        "metadata": metadata_path,
        "commit": commit_path,
        "artifacts": artifact_manifest,
        "facts": facts,
    }


def generate_publication(
    *,
    analysis_dir: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT,
    contract_path: str | Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Authenticate final analyzer outputs and atomically publish the projection."""

    contract_candidate = _absolute_without_symlink_resolution(contract_path)
    contract, contract_raw = _read_contract(contract_candidate)
    contract_resolved = _guard_untrusted_input_path(
        contract_candidate, label="publication contract"
    )
    bundle = authenticate_analyzer_bundle(analysis_dir)
    destination = _guard_untrusted_input_path(
        output_dir, label="publication destination"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing_files = _scan_regular_tree_nofollow(
            destination, label="existing publication destination"
        )
        expected_commit = Path(contract["outputs"]["publication_commit"])
        if expected_commit not in existing_files:
            raise PublicationAdapterError(
                "publication destination already exists with different or partial content; "
                "immutable overwrite refused"
            )
    stage = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.build.", dir=destination.parent
    ))
    try:
        result = _build_publication_tree(
            stage, bundle=bundle, contract=contract,
            contract_path=contract_resolved, contract_raw=contract_raw,
        )
        if destination.exists():
            if not _same_tree(stage, destination):
                raise PublicationAdapterError(
                    "publication destination already exists with different or partial content; "
                    "immutable overwrite refused"
                )
            return {
                **result,
                "metadata": destination / Path(result["metadata"]).name,
                "commit": destination / Path(result["commit"]).name,
                "output_dir": destination,
                "idempotent_existing": True,
            }
        os.replace(stage, destination)
        stage = destination
        return {
            **result,
            "metadata": destination / Path(result["metadata"]).name,
            "commit": destination / Path(result["commit"]).name,
            "output_dir": destination,
            "idempotent_existing": False,
        }
    finally:
        if stage.exists() and stage != destination:
            shutil.rmtree(stage)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir", required=True, type=Path,
        help="directory containing the eight committed analyzer inputs",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = generate_publication(
        analysis_dir=args.analysis_dir,
        output_dir=args.output_dir,
        contract_path=args.contract,
    )
    print(f"publication metadata: {result['metadata']}")
    print(f"publication commit: {result['commit']}")
    print("complete primary rows: 118 (54/48/16; no sign-based filtering)")


if __name__ == "__main__":
    main()


__all__ = [
    "AuthenticatedBundle",
    "PublicationAdapterError",
    "authenticate_analyzer_bundle",
    "build_result_facts",
    "enrich_policy_cei_did_rows",
    "enrich_primary_rows",
    "expected_estimands",
    "generate_publication",
    "make_publication_figures",
]
