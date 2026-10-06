#!/usr/bin/env python3
"""Analyze paired aggregate operating characteristics for cKG fantasy anchors.

The analysis clusters the two OSA strata within trial seed.  Pointwise 95%
Monte Carlo intervals are paired t intervals over seed-level averages.  These
intervals are descriptive and unadjusted.  In particular, an interval containing
zero is not evidence of equivalence because no clinical or numerical equivalence
margin was prespecified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from scipy.stats import t as student_t


PRODUCTION = "cKG-512-seed0"
NEW_ARMS = ("cKG-1024-seed0", "cKG-1024-seed1")
POLICIES = ("cEI", PRODUCTION, "cKG-1024-seed0", "cKG-1024-seed1", "cEI-tMSE")
STRATA = (0, 1)
EXPECTED_FULL_SEEDS = 200
ARTIFACT_CLASS = "computational_diagnostic"
GENERATION_RUNNER_SHA256 = (
    "9ff2820d4a46cda16b24d6e2c04d15bcb1b46807cf7f94e583d26a3160021c52"
)
PRE_MIGRATION_RAW_SHA256 = (
    "e44179341cf762500a132154720dcd83e9ed174d7ab391109ee9a892e4f8fc14"
)
MIGRATION_SOURCE = "paper/migrate_ckg_aggregate_oc_stability_metadata.py"
INVENTORY_NOTE = (
    "The 1,200 reused reference rows and 800 new computational-diagnostic "
    "executions are excluded from the logical trial-record inventory; the archived "
    "logical total remains 55,480."
)
FANTASY_BANK_CONDITIONALITY = (
    "Paired t Monte Carlo intervals use trial seeds as the only random unit and are "
    "conditional on each evaluated fixed fantasy bank. Fantasy-bank seed 1 is a "
    "second deterministic numerical arm; two fixed banks do not estimate a "
    "between-bank seed distribution."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _outcome_payload_sha256(raw: dict[str, Any]) -> str:
    payload = {
        "new_arm_rows": raw["new_arm_rows"],
        "production_equivalence": raw["production_equivalence"],
        "reference_rows": raw["reference_rows"],
    }
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_raw(
    path: Path, *, allow_unclassified_independent_reproduction: bool = False
) -> dict[str, Any]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing raw metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_sha256") != _sha256(path):
        raise ValueError("raw artifact hash differs from its metadata")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported raw schema")
    if "metadata_schema_migration" not in raw:
        if not allow_unclassified_independent_reproduction:
            raise ValueError(
                "raw artifact lacks the frozen metadata-only migration; use "
                "--allow-unclassified-independent-reproduction only for a newly "
                "executed runner output"
            )
        if raw.get("source_sha256", {}).get(
            "paper/run_ckg_aggregate_oc_stability.py"
        ) != GENERATION_RUNNER_SHA256 or metadata.get("source_sha256", {}).get(
            "paper/run_ckg_aggregate_oc_stability.py"
        ) != GENERATION_RUNNER_SHA256:
            raise ValueError("independent reproduction does not preserve runner provenance")
        if raw.get("production_equivalence", {}).get("status") != "pass" or not raw.get(
            "production_equivalence", {}
        ).get("all_archived_fields_exact"):
            raise ValueError("independent reproduction lacks production equivalence")
        return raw
    if raw.get("artifact_class") != ARTIFACT_CLASS:
        raise ValueError("raw artifact is not classified as a computational diagnostic")
    if raw.get("logical_trial_records") != 0:
        raise ValueError("computational diagnostics must contribute zero logical trial records")
    if raw.get("inventory_note") != INVENTORY_NOTE:
        raise ValueError("raw artifact inventory note is missing or changed")
    if metadata.get("artifact_class") != ARTIFACT_CLASS:
        raise ValueError("raw sidecar is not classified as a computational diagnostic")
    if metadata.get("logical_trial_records") != 0:
        raise ValueError("raw sidecar must contribute zero logical trial records")
    if metadata.get("inventory_note") != INVENTORY_NOTE:
        raise ValueError("raw sidecar inventory note is missing or changed")
    migration = raw.get("metadata_schema_migration")
    if not isinstance(migration, dict) or metadata.get(
        "metadata_schema_migration"
    ) != migration:
        raise ValueError("raw and sidecar do not share the metadata migration record")
    if migration.get("pre_migration_raw_sha256") != PRE_MIGRATION_RAW_SHA256:
        raise ValueError("metadata migration does not name the frozen pre-migration raw")
    if migration.get("generation_runner_sha256") != GENERATION_RUNNER_SHA256:
        raise ValueError("metadata migration does not preserve execution provenance")
    if migration.get("outcomes_recomputed") is not False:
        raise ValueError("metadata migration must not recompute outcomes")
    outcome_hash = _outcome_payload_sha256(raw)
    if migration.get("outcome_payload_sha256_before") != outcome_hash or migration.get(
        "outcome_payload_sha256_after"
    ) != outcome_hash:
        raise ValueError("metadata migration outcome-payload hashes are not exact")
    if raw.get("source_sha256", {}).get(
        "paper/run_ckg_aggregate_oc_stability.py"
    ) != GENERATION_RUNNER_SHA256:
        raise ValueError("raw source provenance does not preserve the execution runner")
    migration_path = ROOT / MIGRATION_SOURCE
    if migration.get("migration_source") != MIGRATION_SOURCE or migration.get(
        "migration_source_sha256"
    ) != _sha256(migration_path):
        raise ValueError("metadata migration source hash differs from the frozen script")
    if raw.get("design", {}).get(
        "fantasy_bank_conditionality"
    ) != FANTASY_BANK_CONDITIONALITY:
        raise ValueError("raw artifact lacks the fixed-bank conditionality statement")
    if raw.get("production_equivalence", {}).get("status") != "pass":
        raise ValueError("raw artifact lacks a passing production-equivalence audit")
    if not raw.get("production_equivalence", {}).get("all_archived_fields_exact"):
        raise ValueError("archived production fields were not exact")
    return raw


def _validate_existing_seed0_anchor(
    by_policy: dict[str, dict[tuple[int, int], dict[str, Any]]], anchor_raw: Path
) -> dict[str, Any]:
    """Require exact reproduction of the previously saved 20-seed 1024 anchor."""
    metadata_path = anchor_raw.with_suffix(anchor_raw.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_sha256") != _sha256(anchor_raw):
        raise ValueError("existing 1024-anchor raw artifact differs from its metadata")
    prior = json.loads(anchor_raw.read_text(encoding="utf-8"))
    old = {
        (int(row["seed"]), int(row["stratum"])): row["result"]
        for row in prior.get("anchor_full_path_trials", [])
    }
    expected = {(seed, stratum) for seed in range(20) for stratum in STRATA}
    if set(old) != expected:
        raise ValueError("existing 1024 anchor lacks the frozen first-20 factorial")
    fields = (
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
        "n_gate_pass",
        "n_gate_pass_safe",
        "obd_pf",
        "obd_sdg",
    )
    current = by_policy["cKG-1024-seed0"]
    for key in sorted(expected):
        for field in fields:
            if current[key][field] != old[key][field]:
                raise ValueError(
                    f"new seed-0 anchor differs from the saved anchor at {key} {field}"
                )
        if current[key]["above_boundary_assignments"] != old[key]["toxic"]:
            raise ValueError(f"new seed-0 anchor toxicity count differs at {key}")
    return {
        "status": "pass",
        "existing_anchor_basename": anchor_raw.name,
        "existing_anchor_sha256": _sha256(anchor_raw),
        "existing_anchor_metadata_sha256": _sha256(metadata_path),
        "seed_stratum_cells_checked": len(expected),
        "all_path_and_terminal_fields_exact": True,
        "audited_fields": list(fields) + ["above_boundary_assignments versus toxic"],
    }


def _flatten_rows(raw: dict[str, Any]) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
    design = raw.get("design", {})
    seeds = tuple(int(seed) for seed in design.get("seeds", []))
    if not seeds or seeds != tuple(range(len(seeds))):
        raise ValueError("trial seeds must be the consecutive frozen labels 0,...,n-1")
    expected = {(seed, stratum) for seed in seeds for stratum in STRATA}
    by_policy: dict[str, dict[tuple[int, int], dict[str, Any]]] = {
        policy: {} for policy in POLICIES
    }
    for row in raw.get("reference_rows", []):
        source_policy = str(row["policy"])
        policy = PRODUCTION if source_policy == "cKG" else source_policy
        if policy not in by_policy:
            raise ValueError(f"unexpected reference policy {source_policy!r}")
        key = (int(row["seed"]), int(row["stratum"]))
        if key in by_policy[policy]:
            raise ValueError(f"duplicate reference row {policy} {key}")
        by_policy[policy][key] = dict(row)
    for row in raw.get("new_arm_rows", []):
        policy = str(row["arm"])
        if policy not in NEW_ARMS:
            raise ValueError(f"unexpected numerical arm {policy!r}")
        key = (int(row["seed"]), int(row["stratum"]))
        if key in by_policy[policy]:
            raise ValueError(f"duplicate numerical-arm row {policy} {key}")
        compact = dict(row["result"])
        compact.update({"policy": policy, "seed": key[0], "stratum": key[1]})
        by_policy[policy][key] = compact
    for policy, lookup in by_policy.items():
        if set(lookup) != expected:
            raise ValueError(f"{policy} does not contain the complete seed-stratum grid")
        for key, row in lookup.items():
            if not bool(row["recommendation_made"]):
                raise ValueError(f"full-panel {policy} unexpectedly lacks recommendation at {key}")
            if row.get("stop_reason") is not None:
                raise ValueError(f"full-panel {policy} unexpectedly stopped at {key}")
            if int(row["n_enrolled"]) != 40:
                raise ValueError(f"full-panel {policy} did not enroll 40 at {key}")
            if not 0 <= int(row["above_boundary_assignments"]) <= 40:
                raise ValueError("above-boundary assignment count is outside [0,40]")
            if policy in NEW_ARMS:
                history = row.get("allocation_history")
                if not isinstance(history, list) or len(history) != 40:
                    raise ValueError("new numerical-arm row lacks a 40-assignment path")
                for cohort_start in range(4, 40, 2):
                    if history[cohort_start] != history[cohort_start + 1]:
                        raise ValueError("adaptive cohort members do not share a dose")
    if len(seeds) == EXPECTED_FULL_SEEDS:
        expected_counts = {"reference_rows": 1200, "new_arm_rows": 800}
        observed_counts = {
            "reference_rows": len(raw.get("reference_rows", [])),
            "new_arm_rows": len(raw.get("new_arm_rows", [])),
        }
        if observed_counts != expected_counts:
            raise ValueError(f"full study record counts differ: {observed_counts}")
    return by_policy


def _recommended_efficacy(row: dict[str, Any]) -> float | None:
    return float(row["rec_true_eff"]) if row["recommendation_made"] else None


def _unsafe_recommendation_all_trial(row: dict[str, Any]) -> float:
    return 100.0 * float(bool(row["recommendation_made"]) and bool(row["rec_unsafe"]))


def _above_boundary_assignment_pct(row: dict[str, Any]) -> float:
    return 100.0 * float(row["above_boundary_assignments"]) / float(row["n_enrolled"])


def _dose_location_error(row: dict[str, Any]) -> float | None:
    return float(row["dose_units"]) if row["recommendation_made"] else None


def _recommendation_rate(row: dict[str, Any]) -> float:
    return 100.0 * float(bool(row["recommendation_made"]))


def _stop_rate(row: dict[str, Any]) -> float:
    return 100.0 * float(not bool(row["recommendation_made"]))


OUTCOMES: dict[str, dict[str, Any]] = {
    "recommended_efficacy": {
        "label": "Recommended true efficacy",
        "unit": "OSA efficacy units",
        "denominator": "trials producing a recommendation",
        "fn": _recommended_efficacy,
    },
    "threshold_exceeding_final_recommendation": {
        "label": "All-trial threshold-exceeding final recommendation",
        "unit": "percentage points",
        "denominator": "all simulated trials",
        "fn": _unsafe_recommendation_all_trial,
    },
    "above_boundary_assignments": {
        "label": "Above-boundary simulated assignments",
        "unit": "percentage points of 40 assignments",
        "denominator": "all simulated assignments",
        "fn": _above_boundary_assignment_pct,
    },
    "dose_location_error": {
        "label": "Dose-location error",
        "unit": "panel dose units",
        "denominator": "trials producing a recommendation",
        "fn": _dose_location_error,
    },
    "recommendation_rate": {
        "label": "Recommendation rate",
        "unit": "percentage points",
        "denominator": "all simulated trials",
        "fn": _recommendation_rate,
    },
    "stop_rate": {
        "label": "Stop rate",
        "unit": "percentage points",
        "denominator": "all simulated trials",
        "fn": _stop_rate,
    },
}


def _interval(seed_values: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(list(seed_values), dtype=float)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("seed-level values must contain at least two finite entries")
    mean = float(values.mean())
    mcse = float(values.std(ddof=1) / math.sqrt(values.size))
    critical = float(student_t.ppf(0.975, values.size - 1))
    return {
        "independent_seeds": int(values.size),
        "mean": mean,
        "mcse": mcse,
        "ci95_low": mean - critical * mcse,
        "ci95_high": mean + critical * mcse,
        "critical_t": critical,
    }


def _policy_seed_values(
    lookup: dict[tuple[int, int], dict[str, Any]],
    outcome: str,
    seeds: Iterable[int],
) -> tuple[list[float], int]:
    fn: Callable[[dict[str, Any]], float | None] = OUTCOMES[outcome]["fn"]
    values: list[float] = []
    eligible_rows = 0
    for seed in seeds:
        stratum_values = [fn(lookup[(seed, stratum)]) for stratum in STRATA]
        observed = [float(value) for value in stratum_values if value is not None]
        eligible_rows += len(observed)
        if not observed:
            continue
        values.append(float(statistics.fmean(observed)))
    return values, eligible_rows


def _paired_seed_values(
    left: dict[tuple[int, int], dict[str, Any]],
    right: dict[tuple[int, int], dict[str, Any]],
    outcome: str,
    seeds: Iterable[int],
) -> tuple[list[float], int]:
    fn: Callable[[dict[str, Any]], float | None] = OUTCOMES[outcome]["fn"]
    values: list[float] = []
    eligible_pairs = 0
    for seed in seeds:
        within: list[float] = []
        for stratum in STRATA:
            a = fn(left[(seed, stratum)])
            b = fn(right[(seed, stratum)])
            if a is None or b is None:
                continue
            within.append(float(a) - float(b))
        eligible_pairs += len(within)
        if within:
            values.append(float(statistics.fmean(within)))
    return values, eligible_pairs


def _oc_rows(
    by_policy: dict[str, dict[tuple[int, int], dict[str, Any]]], seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for outcome, spec in OUTCOMES.items():
            values, eligible = _policy_seed_values(by_policy[policy], outcome, seeds)
            estimate = _interval(values)
            row = {
                "policy": policy,
                "outcome_id": outcome,
                "outcome": spec["label"],
                "unit": spec["unit"],
                "denominator_definition": spec["denominator"],
                "eligible_seed_stratum_rows": eligible,
                **estimate,
            }
            if outcome == "above_boundary_assignments":
                row["mean_assignments_of_40"] = estimate["mean"] * 40.0 / 100.0
            else:
                row["mean_assignments_of_40"] = None
            rows.append(row)
    return rows


def _contrast_specs() -> list[tuple[str, str, str]]:
    specs = [
        ("1024-seed0 minus production-512", "cKG-1024-seed0", PRODUCTION),
        ("1024-seed1 minus production-512", "cKG-1024-seed1", PRODUCTION),
        ("1024-seed1 minus 1024-seed0", "cKG-1024-seed1", "cKG-1024-seed0"),
    ]
    for ckg in (PRODUCTION,) + NEW_ARMS:
        specs.append((f"{ckg} minus cEI", ckg, "cEI"))
        specs.append((f"{ckg} minus specified 1:1 cEI-tMSE", ckg, "cEI-tMSE"))
    return specs


def _contrast_rows(
    by_policy: dict[str, dict[tuple[int, int], dict[str, Any]]], seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    if len(seeds) < 20:
        raise ValueError("at least 20 seeds are required for the aggregate-OC analysis")
    first20 = tuple(seed for seed in seeds if seed < 20)
    rows: list[dict[str, Any]] = []
    for contrast, left, right in _contrast_specs():
        for outcome, spec in OUTCOMES.items():
            contrast_denominator = (
                "common-recommendation seed-stratum pairs"
                if outcome in {"recommended_efficacy", "dose_location_error"}
                else spec["denominator"]
            )
            values, eligible = _paired_seed_values(
                by_policy[left], by_policy[right], outcome, seeds
            )
            first_values, first_eligible = _paired_seed_values(
                by_policy[left], by_policy[right], outcome, first20
            )
            full = _interval(values)
            n20 = _interval(first_values)
            row: dict[str, Any] = {
                "contrast": contrast,
                "left_policy": left,
                "right_policy": right,
                "outcome_id": outcome,
                "outcome": spec["label"],
                "unit": spec["unit"],
                "denominator_definition": contrast_denominator,
                "eligible_seed_stratum_pairs": eligible,
                **full,
                "first20_eligible_seed_stratum_pairs": first_eligible,
                "first20_independent_seeds": n20["independent_seeds"],
                "first20_mean": n20["mean"],
                "first20_mcse": n20["mcse"],
                "first20_ci95_low": n20["ci95_low"],
                "first20_ci95_high": n20["ci95_high"],
                "first20_ci_width_over_full_ci_width": (
                    (n20["ci95_high"] - n20["ci95_low"])
                    / (full["ci95_high"] - full["ci95_low"])
                    if full["ci95_high"] > full["ci95_low"]
                    else None
                ),
            }
            if outcome == "above_boundary_assignments":
                row["mean_difference_in_assignments_of_40"] = full["mean"] * 0.4
            else:
                row["mean_difference_in_assignments_of_40"] = None
            rows.append(row)
    return rows


def _find_contrast(
    rows: list[dict[str, Any]], left: str, right: str, outcome: str
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["left_policy"] == left
        and row["right_policy"] == right
        and row["outcome_id"] == outcome
    ]
    if len(matches) != 1:
        raise ValueError("contrast lookup is not unique")
    return matches[0]


def _direction_checks(contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requested = tuple(
        (f"{OUTCOMES[outcome]['label']} versus {label}", comparator, outcome)
        for label, comparator in (
            ("cEI", "cEI"),
            ("specified 1:1 cEI-tMSE", "cEI-tMSE"),
        )
        for outcome in (
            "recommended_efficacy",
            "threshold_exceeding_final_recommendation",
            "above_boundary_assignments",
            "dose_location_error",
        )
    )
    checks: list[dict[str, Any]] = []
    for label, comparator, outcome in requested:
        production_row = _find_contrast(contrasts, PRODUCTION, comparator, outcome)
        if production_row["mean"] == 0:
            raise ValueError("a production comparator contrast has exactly zero mean")
        expected_sign = 1 if production_row["mean"] > 0 else -1
        production_interval_supports = (
            production_row["ci95_low"] > 0
            if expected_sign > 0
            else production_row["ci95_high"] < 0
        )
        for ckg in (PRODUCTION,) + NEW_ARMS:
            row = _find_contrast(contrasts, ckg, comparator, outcome)
            sign_preserved = row["mean"] * expected_sign > 0
            current_interval_direction = (
                "positive"
                if row["ci95_low"] > 0
                else "negative"
                if row["ci95_high"] < 0
                else "includes_zero"
            )
            interval_supports = (
                row["ci95_low"] > 0 if expected_sign > 0 else row["ci95_high"] < 0
            )
            checks.append(
                {
                    "claim": label,
                    "ckg_policy": ckg,
                    "comparator": comparator,
                    "outcome_id": outcome,
                    "expected_sign": "positive" if expected_sign > 0 else "negative",
                    "production_baseline_estimate": production_row["mean"],
                    "production_baseline_ci95_low": production_row["ci95_low"],
                    "production_baseline_ci95_high": production_row["ci95_high"],
                    "interval_excludes_zero_in_production_baseline": bool(
                        production_interval_supports
                    ),
                    "estimate": row["mean"],
                    "mcse": row["mcse"],
                    "ci95_low": row["ci95_low"],
                    "ci95_high": row["ci95_high"],
                    "point_estimate_direction_preserved": bool(sign_preserved),
                    "current_interval_contains_zero": bool(
                        row["ci95_low"] <= 0 <= row["ci95_high"]
                    ),
                    "current_interval_direction": current_interval_direction,
                    "pointwise_interval_supports_expected_direction": bool(interval_supports),
                    "new_nominal_separation_when_production_interval_included_zero": bool(
                        ckg != PRODUCTION
                        and not production_interval_supports
                        and current_interval_direction != "includes_zero"
                    ),
                }
            )
    return checks


def _direct_stability(contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left in NEW_ARMS:
        for outcome in OUTCOMES:
            row = _find_contrast(contrasts, left, PRODUCTION, outcome)
            rows.append(
                {
                    "arm": left,
                    "outcome_id": outcome,
                    "estimate": row["mean"],
                    "mcse": row["mcse"],
                    "ci95_low": row["ci95_low"],
                    "ci95_high": row["ci95_high"],
                    "pointwise_interval_contains_zero": bool(
                        row["ci95_low"] <= 0 <= row["ci95_high"]
                    ),
                    "first20_estimate": row["first20_mean"],
                    "first20_mcse": row["first20_mcse"],
                    "first20_ci95_low": row["first20_ci95_low"],
                    "first20_ci95_high": row["first20_ci95_high"],
                    "first20_ci_width_over_full_ci_width": row[
                        "first20_ci_width_over_full_ci_width"
                    ],
                }
            )
    return rows


def _between_bank_stability(contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        row = _find_contrast(
            contrasts, "cKG-1024-seed1", "cKG-1024-seed0", outcome
        )
        rows.append(
            {
                "contrast": "cKG-1024-seed1 minus cKG-1024-seed0",
                "outcome_id": outcome,
                "estimate": row["mean"],
                "mcse": row["mcse"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "pointwise_interval_contains_zero": bool(
                    row["ci95_low"] <= 0 <= row["ci95_high"]
                ),
                "first20_estimate": row["first20_mean"],
                "first20_mcse": row["first20_mcse"],
                "first20_ci95_low": row["first20_ci95_low"],
                "first20_ci95_high": row["first20_ci95_high"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fmt(row: dict[str, Any], digits: int = 3) -> str:
    return (
        f"{row['mean']:.{digits}f} "
        f"[{row['ci95_low']:.{digits}f}, {row['ci95_high']:.{digits}f}]"
    )


def _render_markdown(summary: dict[str, Any], contrast_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Paired aggregate-OC cKG fantasy sensitivity",
        "",
        "This analysis is **post hoc** and concerns numerical stability, not accuracy,",
        "Monte Carlo convergence, or a universally adequate fantasy count. The two OSA",
        "strata are averaged within each of 200 paired trial seeds. Intervals are",
        "pointwise 95% paired t Monte Carlo intervals and are not multiplicity-adjusted.",
        "The 1,200 reused reference rows and 800 new executions are computational",
        "diagnostics and contribute zero rows to the 55,480 logical trial-record inventory.",
        "Trial seeds are the only random unit. Results are conditional on the two",
        "evaluated fixed fantasy banks; two banks do not estimate a between-bank",
        "seed distribution.",
        "",
        "## Direct numerical contrasts",
        "",
        "| 1024 arm minus production cKG-512 | Recommended efficacy | Threshold-exceeding final recommendation (pp) | Above-boundary assignments (pp of 40) | Dose-location error |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in NEW_ARMS:
        cells = []
        for outcome in (
            "recommended_efficacy",
            "threshold_exceeding_final_recommendation",
            "above_boundary_assignments",
            "dose_location_error",
        ):
            cells.append(_fmt(_find_contrast(contrast_rows, arm, PRODUCTION, outcome)))
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    bank_cells = []
    for outcome in (
        "recommended_efficacy",
        "threshold_exceeding_final_recommendation",
        "above_boundary_assignments",
        "dose_location_error",
    ):
        bank_cells.append(
            _fmt(
                _find_contrast(
                    contrast_rows,
                    "cKG-1024-seed1",
                    "cKG-1024-seed0",
                    outcome,
                )
            )
        )
    lines.extend(
        [
            "",
            "At the same 1,024-fantasy count, seed1 minus seed0 was: "
            + "; ".join(bank_cells)
            + " (efficacy; terminal pp; assignment pp; dose error).",
        ]
    )
    lines.extend(
        [
            "",
            "Recommendation rate was 100% and stop rate was 0% for every arm because",
            "the full-panel `lhs_fixed` design has no early-stopping branch. These are",
            "design facts, not empirical evidence of fantasy-count stability.",
            "",
            "## Interpretation",
            "",
            "The boundary-related operating characteristics were not uniformly stable.",
            "For seed0, doubling from 512 to 1,024 changed threshold-exceeding final",
            "recommendations by -7.25 percentage points and above-boundary assignments",
            "by +1.99 percentage points, with both pointwise intervals excluding zero.",
            "The two 1,024 banks also differed on both outcomes. These evaluated runs",
            "therefore do not support describing the aggregate OCs as stable or converged.",
            "",
            "Every comparator direction whose production interval excluded zero remained",
            "interval-supported under both 1,024 arms. Among production-unresolved cells,",
            "seed0 produced a new nominal positive assignment separation versus cEI; the",
            "other evaluated intervals remained unresolved. This is directional",
            "preservation evidence, not evidence that the absolute OCs are stable.",
            "",
            "## Primary comparator-direction audit",
            "",
            "| Contrast audited | cKG implementation | Estimate [95% interval] | Production interval | Current interval status |",
            "|---|---|---:|:---:|:---:|",
        ]
    )
    for check in summary["primary_comparator_direction_checks"]:
        interval = (
            f"{check['estimate']:.3f} [{check['ci95_low']:.3f}, "
            f"{check['ci95_high']:.3f}]"
        )
        if check["interval_excludes_zero_in_production_baseline"]:
            current_status = (
                f"supported {check['expected_sign']} direction"
                if check["pointwise_interval_supports_expected_direction"]
                else "unresolved"
                if check["current_interval_contains_zero"]
                else f"opposite {check['current_interval_direction']} direction"
            )
        else:
            current_status = (
                f"new nominal {check['current_interval_direction']} separation"
                if check[
                    "new_nominal_separation_when_production_interval_included_zero"
                ]
                else "unresolved"
            )
        lines.append(
            f"| {check['claim']} | {check['ckg_policy']} | {interval} | "
            f"{'excluded zero' if check['interval_excludes_zero_in_production_baseline'] else 'included zero (unresolved)'} | "
            f"{current_status} |"
        )
    ratios = [
        row["first20_ci_width_over_full_ci_width"]
        for row in summary["direct_stability_contrasts"]
        if row["first20_ci_width_over_full_ci_width"] is not None
    ]
    lines.extend(
        [
            "",
            "## Why the original 20 seeds were not enough",
            "",
            (
                "Across non-degenerate direct contrasts, first-20 pointwise intervals were "
                f"a median {statistics.median(ratios):.2f} times as wide as the 200-seed "
                "intervals. The first-20 subset is retained for transparency, but it was "
                "not a sufficiently precise basis for the requested aggregate-OC check."
            ),
            "",
            "No equivalence margin was prespecified. Therefore, an interval containing zero",
            "must not be translated into formal equivalence. The defensible interpretation",
            "depends on the direct effect-size bounds and whether the manuscript's named",
            "comparator directions remain intact.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_metadata(
    artifact: Path,
    raw_path: Path,
    seed_count: int,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_class": ARTIFACT_CLASS,
        "logical_trial_records": 0,
        "inventory_note": INVENTORY_NOTE,
        "artifact": artifact.name,
        "artifact_sha256": _sha256(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post hoc numerical-stability sensitivity",
        "input": raw_path.name,
        "input_sha256": _sha256(raw_path),
        "independent_trial_seeds": seed_count,
        "seed_stratum_cells_per_policy": seed_count * 2,
        "interval": "pointwise unadjusted paired-seed t Monte Carlo interval",
        "clustering": "two OSA strata averaged within trial seed",
        "fantasy_bank_conditionality": FANTASY_BANK_CONDITIONALITY,
        "source_sha256": source_hashes,
        "interpretation": (
            "Numerical stability only; absence of a statistically detectable difference "
            "is not formal equivalence and does not establish accuracy or convergence."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw",
        nargs="?",
        type=Path,
        default=ROOT
        / "results/ckg_aggregate_oc_stability/ckg_aggregate_oc_stability_raw.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/ckg_aggregate_oc_stability",
    )
    parser.add_argument(
        "--existing-anchor-raw",
        type=Path,
        default=ROOT / "results/computational_diagnostics/ckg_1024_anchor_raw.json",
    )
    parser.add_argument(
        "--allow-unclassified-independent-reproduction",
        action="store_true",
        help=(
            "Analyze a newly executed runner output that predates the one-use frozen "
            "artifact metadata migration; generation provenance is still required."
        ),
    )
    args = parser.parse_args(argv)
    raw_path = args.raw.resolve()
    raw = _load_raw(
        raw_path,
        allow_unclassified_independent_reproduction=(
            args.allow_unclassified_independent_reproduction
        ),
    )
    by_policy = _flatten_rows(raw)
    existing_anchor_validation = _validate_existing_seed0_anchor(
        by_policy, args.existing_anchor_raw.resolve()
    )
    seeds = tuple(int(seed) for seed in raw["design"]["seeds"])
    if len(seeds) < 20:
        raise ValueError("the final aggregate-OC analysis requires at least 20 seeds")
    oc_rows = _oc_rows(by_policy, seeds)
    contrast_rows = _contrast_rows(by_policy, seeds)
    direction_checks = _direction_checks(contrast_rows)
    direct = _direct_stability(contrast_rows)
    between_bank = _between_bank_stability(contrast_rows)
    nondegenerate_ratios = [
        row["first20_ci_width_over_full_ci_width"]
        for row in direct
        if row["first20_ci_width_over_full_ci_width"] is not None
    ]
    production_supported_new_arm_checks = [
        row
        for row in direction_checks
        if row["ckg_policy"] in NEW_ARMS
        and row["interval_excludes_zero_in_production_baseline"]
    ]
    baseline_unresolved_new_arm_checks = [
        {
            "claim": row["claim"],
            "ckg_policy": row["ckg_policy"],
            "current_interval_direction": row["current_interval_direction"],
            "new_nominal_separation_when_production_interval_included_zero": row[
                "new_nominal_separation_when_production_interval_included_zero"
            ],
        }
        for row in direction_checks
        if row["ckg_policy"] in NEW_ARMS
        and not row["interval_excludes_zero_in_production_baseline"]
    ]
    summary = {
        "schema_version": 1,
        "artifact_class": ARTIFACT_CLASS,
        "logical_trial_records": 0,
        "inventory_note": INVENTORY_NOTE,
        "status": "post hoc numerical-stability sensitivity",
        "design": raw["design"],
        "input_metadata_schema_migration": raw.get(
            "metadata_schema_migration",
            {
                "status": "not_applied_to_independent_reproduction",
                "generation_runner_sha256": GENERATION_RUNNER_SHA256,
            },
        ),
        "production_equivalence": raw["production_equivalence"],
        "existing_seed0_1024_anchor_validation": existing_anchor_validation,
        "interval": {
            "level": 0.95,
            "method": "paired seed-clustered t Monte Carlo interval",
            "multiplicity": "pointwise and unadjusted",
            "clustering": "average the two OSA strata within trial seed",
            "random_unit": "trial seed",
            "fantasy_bank_conditionality": FANTASY_BANK_CONDITIONALITY,
        },
        "denominators": {
            "independent_trial_seeds": len(seeds),
            "seed_stratum_cells_per_policy": len(seeds) * 2,
            "simulated_assignments_per_policy": len(seeds) * 2 * 40,
            "recommendation_conditional_rows_per_policy": len(seeds) * 2,
            "common_recommendation_seed_stratum_pairs_per_pairwise_contrast": len(seeds)
            * 2,
            "all_trial_rows_per_policy": len(seeds) * 2,
        },
        "full_panel_recommendation_stop_fact": (
            "Recommendation rate is 100% and stop rate is 0% by lhs_fixed design; "
            "these are not empirical numerical-stability results."
        ),
        "direct_stability_contrasts": direct,
        "between_1024_fixed_bank_contrasts": between_bank,
        "primary_comparator_direction_checks": direction_checks,
        "interpretation": {
            "aggregate_oc_stability_supported": False,
            "aggregate_oc_stability_conclusion": (
                "Boundary-related operating characteristics were not uniformly stable "
                "across the evaluated fixed fantasy banks; do not claim aggregate-OC "
                "stability or convergence."
            ),
            "production_interval_supported_direction_cells_checked_across_new_arms": len(
                production_supported_new_arm_checks
            ),
            "all_production_interval_supported_directions_retained_by_both_1024_arms": all(
                row["pointwise_interval_supports_expected_direction"]
                for row in production_supported_new_arm_checks
            ),
            "production_interval_unresolved_cells_across_new_arms": (
                baseline_unresolved_new_arm_checks
            ),
            "fantasy_bank_scope": FANTASY_BANK_CONDITIONALITY,
        },
        "first20_precision_audit": {
            "seed_labels": list(range(20)),
            "full_seed_count": len(seeds),
            "median_first20_ci_width_over_full_ci_width": float(
                statistics.median(nondegenerate_ratios)
            ),
            "verdict": (
                "The first 20 seeds were not a sufficiently precise basis for the "
                "requested aggregate-OC sensitivity; use the full paired-seed analysis."
            ),
        },
        "claim_guardrail": (
            "No equivalence margin was prespecified. Report effect sizes and interval "
            "bounds; do not equate a zero-containing interval with formal equivalence. "
            "For efficacy and dose-location contrasts the denominator is common-"
            "recommendation seed-stratum pairs. A new interval excluding zero where "
            "the production interval included zero is only a new nominal separation, not "
            "equivalence or preservation of a supported conclusion. This analysis "
            "addresses numerical stability, not accuracy or convergence."
            " " + FANTASY_BANK_CONDITIONALITY
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    oc_path = output_dir / "ckg_aggregate_oc_stability_policy_oc.csv"
    contrast_path = output_dir / "ckg_aggregate_oc_stability_paired_contrasts.csv"
    summary_path = output_dir / "ckg_aggregate_oc_stability_summary.json"
    markdown_path = output_dir / "ckg_aggregate_oc_stability_summary.md"
    _write_csv(oc_path, oc_rows)
    _write_csv(contrast_path, contrast_rows)
    _atomic_json(summary_path, summary)
    _atomic_text(markdown_path, _render_markdown(summary, contrast_rows))

    source_paths = (
        Path(__file__).resolve(),
        ROOT / "paper/run_ckg_aggregate_oc_stability.py",
        ROOT / "paper/migrate_ckg_aggregate_oc_stability_metadata.py",
        ROOT / "paper/run_ckg_1024_anchor.py",
        ROOT / "src/dose_combination_bo/trial.py",
        ROOT / "src/dose_combination_bo/acquisitions.py",
    )
    source_hashes = {
        path.relative_to(ROOT).as_posix(): _sha256(path) for path in source_paths
    }
    for artifact in (oc_path, contrast_path, summary_path, markdown_path):
        _atomic_json(
            artifact.with_suffix(artifact.suffix + ".metadata.json"),
            _artifact_metadata(artifact, raw_path, len(seeds), source_hashes),
        )
    print(f"[done] policy OC: {oc_path}")
    print(f"[done] paired contrasts: {contrast_path}")
    print(f"[done] summary: {summary_path}")
    print(f"[done] markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
