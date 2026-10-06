#!/usr/bin/env python3
"""Project the post-initialization Family-A allocation estimand.

The frozen calibration analysis prespecified above-boundary assignments divided
by the total analysis-record count ``N`` as a Family-A primary outcome.  The raw
snapshots also stored the supporting participant-allocation outcome used here:
it excludes the four fixed initialization observations and divides by ``N - 4``.
Changing the hash-bound analyzer would invalidate the raw-master, precision,
publication, and final-seal provenance chains.  This adapter therefore leaves
every frozen input untouched and creates a clearly labeled post-hoc supporting
projection.

For each paired policy contrast, the four initialization observations are shared
and cancel.  Consequently, the post-initialization contrast, its Monte Carlo
standard error, and its interval endpoints equal the frozen values multiplied
by ``N / (N - 4)``.  The point estimate is independently recovered from the
authenticated post-initialization absolute profiles already present in the
analysis bundle.  The adapter verifies the two calculations agree before it
writes anything.  No trial simulation is rerun.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

try:  # Support both ``python paper/script.py`` and package imports in tests.
    from paper import generate_budget_toxicity_calibration_publication as publication
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI smoke test
    import generate_budget_toxicity_calibration_publication as publication


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = ROOT / "results/budget_toxicity_calibration_analysis"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/budget_toxicity_calibration_family_A_post_initialization"
)

INITIALIZATION_SIZE = 4
SOURCE_ASSIGNMENT = publication.ASSIGNMENT
POST_INITIALIZATION_ASSIGNMENT = (
    "post_initialization_above_boundary_assignment_pct"
)
POST_INITIALIZATION_LABEL = (
    "Post-initialization above-boundary allocation (%)"
)

TRUSTED_FROZEN_ADAPTER_SHA256 = (
    "e14c2464f586f3abb805bfa24bb47f2ecc61a93e7405e0cb51489d497c6aa919"
)
TRUSTED_FROZEN_CONTRACT_SHA256 = publication.TRUSTED_CONTRACT_SHA256

JSON_NAME = "budget_toxicity_calibration_family_A_post_initialization.json"
CSV_NAME = "budget_toxicity_calibration_family_A_post_initialization.csv"
TEX_NAME = "budget_toxicity_calibration_family_A_post_initialization.tex"
METADATA_NAME = (
    "budget_toxicity_calibration_family_A_post_initialization.metadata.json"
)
COMMIT_NAME = (
    "budget_toxicity_calibration_family_A_post_initialization.commit.json"
)

STATUS = "POST_HOC_SUPPORTING_FAMILY_A_POST_INITIALIZATION_PROJECTION"
COMMIT_STATUS = (
    "COMMITTED_POST_HOC_SUPPORTING_FAMILY_A_POST_INITIALIZATION_PROJECTION"
)


class PostInitializationProjectionError(RuntimeError):
    """Raised when authentication or the correction identity fails closed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
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


def _atomic_write(path: Path, value: bytes) -> None:
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


def _strict_interval_sign(low: float, high: float) -> str:
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise PostInitializationProjectionError("invalid corrected interval")
    if low > 0.0:
        return "positive"
    if high < 0.0:
        return "negative"
    return "unresolved"


def _authenticate_frozen_dependencies() -> tuple[Mapping[str, Any], bytes]:
    adapter_path = Path(publication.__file__).resolve()
    if _sha256_file(adapter_path) != TRUSTED_FROZEN_ADAPTER_SHA256:
        raise PostInitializationProjectionError(
            "the frozen publication adapter differs from its trusted source hash"
        )
    try:
        contract, raw = publication._read_contract(publication.DEFAULT_CONTRACT)
    except publication.PublicationAdapterError as exc:
        raise PostInitializationProjectionError(str(exc)) from exc
    if _sha256_bytes(raw) != TRUSTED_FROZEN_CONTRACT_SHA256:
        raise PostInitializationProjectionError(
            "the frozen publication contract differs from its trusted hash"
        )
    return contract, raw


def _absolute_profile_map(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> dict[tuple[str, float, int], float]:
    selected: dict[tuple[str, float, int], float] = {}
    for row in rows:
        if (
            row.get("analysis_type")
            == publication.SECONDARY_ANALYSIS_ABSOLUTE_PROFILE
            and row.get("metric") == metric
        ):
            key = (
                str(row["policy"]),
                float(row["assumed_toxicity_noise_sd_factor"]),
                int(row["budget"]),
            )
            if key in selected:
                raise PostInitializationProjectionError(
                    f"duplicate absolute profile for {metric}: {key!r}"
                )
            selected[key] = float(row["estimate"])
    expected = {
        (policy, factor, budget)
        for policy in publication.POLICIES
        for factor in publication.FACTORS
        for budget in publication.BUDGETS
    }
    if set(selected) != expected:
        raise PostInitializationProjectionError(
            f"authenticated analysis lacks the complete {metric} absolute grid"
        )
    return selected


def _corrected_allocation_row(
    source: Mapping[str, Any], *, direct_estimate: float
) -> dict[str, Any]:
    budget = int(source["budget_n"])
    if budget not in publication.BUDGETS or budget <= INITIALIZATION_SIZE:
        raise PostInitializationProjectionError(
            f"unsupported Family-A allocation budget: {budget}"
        )
    if (
        source.get("family") != "A"
        or source.get("metric") != SOURCE_ASSIGNMENT
        or source.get("row_guard_class")
        != publication.GUARD_CLASS_EXACT_DISCRETE
    ):
        raise PostInitializationProjectionError(
            "allocation correction received a non-Family-A source row"
        )

    scale = budget / (budget - INITIALIZATION_SIZE)
    scaled_source_estimate = float(source["estimate_points"]) * scale
    if not math.isclose(
        direct_estimate, scaled_source_estimate, rel_tol=0.0, abs_tol=1e-11
    ):
        raise PostInitializationProjectionError(
            "stored post-initialization estimate and paired rescaling disagree"
        )

    mcse = float(source["mcse_points"]) * scale
    point_critical = float(source["pointwise_t_critical_95"])
    family_critical = float(source["family_max_t_critical_95"])
    replicate_bound = float(source["replicate_numerical_bound_points"]) * scale
    mcse_guard = float(source["mcse_numerical_guard_points"]) * scale
    guard_halfwidth = replicate_bound + family_critical * mcse_guard
    point_low = direct_estimate - point_critical * mcse
    point_high = direct_estimate + point_critical * mcse
    simultaneous_low = direct_estimate - family_critical * mcse
    simultaneous_high = direct_estimate + family_critical * mcse
    guarded_low = simultaneous_low - guard_halfwidth
    guarded_high = simultaneous_high + guard_halfwidth
    sign = _strict_interval_sign(guarded_low, guarded_high)
    if sign != source.get("guarded_scientific_sign"):
        raise PostInitializationProjectionError(
            "positive rescaling unexpectedly changed scientific classification"
        )

    source_id = str(source["estimand_id"])
    if source_id.count(SOURCE_ASSIGNMENT) != 1:
        raise PostInitializationProjectionError(
            "Family-A allocation estimand identifier is not canonical"
        )
    corrected = dict(source)
    corrected.update({
        "estimand_id": source_id.replace(
            SOURCE_ASSIGNMENT, POST_INITIALIZATION_ASSIGNMENT
        ),
        "metric": POST_INITIALIZATION_ASSIGNMENT,
        "metric_label": POST_INITIALIZATION_LABEL,
        "estimate_points": direct_estimate,
        "mcse_points": mcse,
        "pointwise_95_low_points": point_low,
        "pointwise_95_high_points": point_high,
        "simultaneous_95_low_points": simultaneous_low,
        "simultaneous_95_high_points": simultaneous_high,
        "replicate_numerical_bound_points": replicate_bound,
        "mcse_numerical_guard_points": mcse_guard,
        "simultaneous_numerical_halfwidth_guard_points": guard_halfwidth,
        "guarded_simultaneous_95_low_points": guarded_low,
        "guarded_simultaneous_95_high_points": guarded_high,
        "guarded_simultaneous_95_strictly_positive": sign == "positive",
        "guarded_simultaneous_95_strictly_negative": sign == "negative",
        "guarded_simultaneous_95_excludes_zero": sign != "unresolved",
        "guarded_scientific_sign": sign,
        "guarded_simultaneous_band_classification": sign,
        "zero_variance": mcse == 0.0,
        "source_estimand_id": source_id,
        "source_metric": SOURCE_ASSIGNMENT,
        "projection_role": "post_hoc_supporting_allocation",
        "transformation_scale": scale,
        "allocation_denominator": budget - INITIALIZATION_SIZE,
    })
    return corrected


def project_family_a_rows(
    primary_rows: Sequence[Mapping[str, Any]],
    absolute_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the 54-row Family-A supporting projection and its audit."""

    family_a = [dict(row) for row in primary_rows if row.get("family") == "A"]
    if len(family_a) != publication.PRIMARY_COUNTS["A"]:
        raise PostInitializationProjectionError(
            "source projection does not contain all 54 Family-A rows"
        )
    post_absolute = _absolute_profile_map(
        absolute_rows, POST_INITIALIZATION_ASSIGNMENT
    )
    source_absolute = _absolute_profile_map(absolute_rows, SOURCE_ASSIGNMENT)

    projected: list[dict[str, Any]] = []
    scaling_errors: list[float] = []
    source_errors: list[float] = []
    corrected_estimates: list[float] = []
    corrected_mcses: list[float] = []
    corrected_count = 0
    for source in family_a:
        metric = str(source.get("metric"))
        if metric == publication.TERMINAL:
            unchanged = dict(source)
            unchanged.update({
                "source_estimand_id": str(source["estimand_id"]),
                "source_metric": publication.TERMINAL,
                "projection_role": "prespecified_terminal_outcome_unchanged",
                "transformation_scale": 1.0,
                "allocation_denominator": "",
            })
            projected.append(unchanged)
            continue
        if metric != SOURCE_ASSIGNMENT:
            raise PostInitializationProjectionError(
                f"unexpected Family-A outcome: {metric}"
            )

        comparator = str(source["comparator"])
        factor = float(source["assumed_toxicity_noise_sd_factor"])
        budget = int(source["budget_n"])
        ckg_key = ("cKG-exact-formal", factor, budget)
        comparator_key = (comparator, factor, budget)
        direct = post_absolute[ckg_key] - post_absolute[comparator_key]
        source_direct = source_absolute[ckg_key] - source_absolute[comparator_key]
        source_error = abs(source_direct - float(source["estimate_points"]))
        source_errors.append(source_error)
        if source_error > 1e-11:
            raise PostInitializationProjectionError(
                "frozen Family-A estimate disagrees with its absolute profiles"
            )
        scale = budget / (budget - INITIALIZATION_SIZE)
        scaling_errors.append(
            abs(direct - float(source["estimate_points"]) * scale)
        )
        corrected = _corrected_allocation_row(
            source, direct_estimate=direct
        )
        projected.append(corrected)
        corrected_estimates.append(float(corrected["estimate_points"]))
        corrected_mcses.append(float(corrected["mcse_points"]))
        corrected_count += 1

    if corrected_count != 27 or len(projected) != 54:
        raise PostInitializationProjectionError(
            "post-initialization projection did not produce 27 allocation rows"
        )
    ids = [str(row["estimand_id"]) for row in projected]
    if len(ids) != len(set(ids)):
        raise PostInitializationProjectionError(
            "post-initialization Family-A identifiers are not unique"
        )
    audit = {
        "family_A_rows": len(projected),
        "corrected_allocation_rows": corrected_count,
        "unchanged_terminal_rows": len(projected) - corrected_count,
        "initialization_observations_excluded": INITIALIZATION_SIZE,
        "budgets": list(publication.BUDGETS),
        "allocation_denominators": [
            budget - INITIALIZATION_SIZE for budget in publication.BUDGETS
        ],
        "transformation_scales": [
            budget / (budget - INITIALIZATION_SIZE)
            for budget in publication.BUDGETS
        ],
        "maximum_source_absolute_identity_error_points": max(source_errors),
        "maximum_post_initialization_scaling_identity_error_points": max(
            scaling_errors
        ),
        "corrected_allocation_estimate_min_points": min(corrected_estimates),
        "corrected_allocation_estimate_max_points": max(corrected_estimates),
        "corrected_allocation_maximum_mcse_points": max(corrected_mcses),
        "scientific_classifications_changed": 0,
    }
    return projected, audit


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise PostInitializationProjectionError("cannot serialize empty rows")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise PostInitializationProjectionError("projection row schemas differ")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _tex_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    caption = (
        "Post-hoc supporting Family A projection. Allocation rows exclude the "
        "four fixed initialization observations and use the N minus 4 "
        "participant allocations; terminal "
        "recommendation rows are unchanged. This table does not replace the "
        "frozen prespecified primary analysis."
    )
    raw = publication._longtable_bytes(
        rows,
        family="A",
        caption=caption,
        label="tab:budget-toxicity-calibration-family-a-post-initialization",
    )
    frozen_comment = (
        b"% Complete frozen Family A: 54 data rows; no sign-based filtering.\n"
    )
    if not raw.startswith(frozen_comment):
        raise PostInitializationProjectionError(
            "frozen Family-A table serializer changed unexpectedly"
        )
    return (
        b"% Post-hoc supporting Family A projection: 54 unfiltered rows; "
        b"27 allocation rows use N-4.\n"
        + raw[len(frozen_comment):]
    )


def _build_tree(
    stage: Path,
    *,
    bundle: publication.AuthenticatedBundle,
    rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    contract_raw: bytes,
) -> dict[str, Path]:
    json_path = stage / JSON_NAME
    csv_path = stage / CSV_NAME
    tex_path = stage / TEX_NAME
    payload = {
        "schema_version": 1,
        "status": STATUS,
        "methodological_status": "post_hoc_supporting_estimand",
        "frozen_primary_analysis_replaced": False,
        "trial_simulations_rerun": False,
        "complete_without_sign_or_result_filtering": True,
        "row_count": len(rows),
        "corrected_allocation_row_count": audit["corrected_allocation_rows"],
        "rows": list(rows),
    }
    _atomic_write(json_path, _canonical_json_bytes(payload))
    _atomic_write(csv_path, _csv_bytes(rows))
    _atomic_write(tex_path, _tex_bytes(rows))
    artifacts = {
        path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in (json_path, csv_path, tex_path)
    }
    metadata_path = stage / METADATA_NAME
    metadata = {
        "schema_version": 1,
        "status": STATUS,
        "artifact_class": "post_hoc_supporting_estimand_projection",
        "projector": Path(__file__).relative_to(ROOT).as_posix(),
        "projector_sha256": _sha256_file(Path(__file__).resolve()),
        "frozen_adapter_sha256": TRUSTED_FROZEN_ADAPTER_SHA256,
        "frozen_contract_sha256": _sha256_bytes(contract_raw),
        "source_analysis": {
            "status": bundle.analysis_metadata["status"],
            "commit_status": bundle.analysis_commit["status"],
            "input_hashes": dict(bundle.input_hashes),
            "input_bundle_sha256": publication._bundle_hash(bundle.input_hashes),
            "final_M": bundle.final_m,
            "raw_provenance": bundle.analysis_metadata["raw_provenance"],
        },
        "correction": {
            "methodological_status": "post_hoc_supporting_estimand",
            "scope": "Family-A allocation contrasts only",
            "source_metric": SOURCE_ASSIGNMENT,
            "source_denominator": "N analysis records",
            "target_metric": POST_INITIALIZATION_ASSIGNMENT,
            "target_denominator": "N-4 post-initialization participant allocations",
            "initialization_observations_excluded": INITIALIZATION_SIZE,
            "stored_raw_snapshot_field": (
                "post_initialization_above_boundary_pct"
            ),
            "stored_analyzer_field": POST_INITIALIZATION_ASSIGNMENT,
            "paired_contrast_identity": (
                "Delta_post = Delta_all * N / (N - 4) because the four "
                "initialization observations are common within each paired "
                "policy comparison"
            ),
            "point_estimate_source": (
                "difference of authenticated post-initialization absolute "
                "profiles"
            ),
            "uncertainty_source": (
                "exact positive rescaling of the paired frozen replicate "
                "contrast"
            ),
            "studentized_max_t_critical_changed": False,
            "frozen_inputs_modified": False,
            "trial_simulations_rerun": False,
        },
        "audit": dict(audit),
        "artifacts": artifacts,
    }
    _atomic_write(metadata_path, _canonical_json_bytes(metadata))
    commit_path = stage / COMMIT_NAME
    commit = {
        "schema_version": 1,
        "status": COMMIT_STATUS,
        "metadata": METADATA_NAME,
        "metadata_sha256": _sha256_file(metadata_path),
        "artifact_count": len(artifacts),
        "source_input_bundle_sha256": publication._bundle_hash(
            bundle.input_hashes
        ),
    }
    _atomic_write(commit_path, _canonical_json_bytes(commit))
    return {
        "json": json_path,
        "csv": csv_path,
        "tex": tex_path,
        "metadata": metadata_path,
        "commit": commit_path,
    }


def generate_projection(
    *, analysis_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Authenticate the frozen analysis and write an immutable projection."""

    contract, contract_raw = _authenticate_frozen_dependencies()
    try:
        bundle = publication.authenticate_analyzer_bundle(analysis_dir)
        enriched = publication.enrich_primary_rows(bundle.primary_rows, contract)
        absolute_rows = bundle.secondary["absolute_rows"]
        if not isinstance(absolute_rows, list):
            raise PostInitializationProjectionError(
                "authenticated secondary absolute rows are absent"
            )
        rows, audit = project_family_a_rows(enriched, absolute_rows)
        destination = publication._guard_untrusted_input_path(
            output_dir, label="post-initialization projection destination"
        )
    except publication.PublicationAdapterError as exc:
        raise PostInitializationProjectionError(str(exc)) from exc
    if destination == bundle.analysis_dir or destination.is_relative_to(
        bundle.analysis_dir
    ) or bundle.analysis_dir.is_relative_to(destination):
        raise PostInitializationProjectionError(
            "projection output and authenticated analysis must not overlap"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.build.", dir=destination.parent
    ))
    try:
        staged = _build_tree(
            stage,
            bundle=bundle,
            rows=rows,
            audit=audit,
            contract_raw=contract_raw,
        )
        if destination.exists():
            if not publication._same_tree(stage, destination):
                raise PostInitializationProjectionError(
                    "projection destination already exists with different or "
                    "partial content; immutable overwrite refused"
                )
            result_paths = {
                key: destination / path.name for key, path in staged.items()
            }
            return {
                **result_paths,
                "output_dir": destination,
                "audit": audit,
                "idempotent_existing": True,
            }
        os.replace(stage, destination)
        stage = destination
        result_paths = {
            key: destination / path.name for key, path in staged.items()
        }
        return {
            **result_paths,
            "output_dir": destination,
            "audit": audit,
            "idempotent_existing": False,
        }
    finally:
        if stage.exists() and stage != destination:
            shutil.rmtree(stage)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR,
        help="directory containing the eight committed frozen analyzer inputs",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="new immutable directory for the post-hoc supporting projection",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = generate_projection(
        analysis_dir=args.analysis_dir, output_dir=args.output_dir
    )
    audit = result["audit"]
    print(f"projection metadata: {result['metadata']}")
    print(f"projection commit: {result['commit']}")
    print(
        "post-initialization allocation estimate range: "
        f"{audit['corrected_allocation_estimate_min_points']:.6g} to "
        f"{audit['corrected_allocation_estimate_max_points']:.6g} "
        "percentage points"
    )
    print("trial simulations rerun: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMMIT_STATUS",
    "INITIALIZATION_SIZE",
    "POST_INITIALIZATION_ASSIGNMENT",
    "POST_INITIALIZATION_LABEL",
    "PostInitializationProjectionError",
    "generate_projection",
    "project_family_a_rows",
]
