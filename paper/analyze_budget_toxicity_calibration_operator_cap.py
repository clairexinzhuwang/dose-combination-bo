#!/usr/bin/env python3
"""Analyze an authenticated M0500 operator-capped calibration master.

This is an authentication/orchestration wrapper only.  All scientific replay,
estimands, confidence bands, figures, and standard analysis envelopes are
produced by the unchanged frozen analyzer module.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Sequence

try:
    from paper import analyze_budget_toxicity_calibration_audit as analyzer
    from paper import finalize_budget_toxicity_calibration_operator_capped_master as capped
except ModuleNotFoundError:  # Direct execution from paper/.
    import analyze_budget_toxicity_calibration_audit as analyzer
    import finalize_budget_toxicity_calibration_operator_capped_master as capped


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ANALYZER_SHA256 = (
    "bcf3ec2964b88d61ac4becc8a9aa838831a3240d93a082abf42b5b328799ec2a"
)
EXPECTED_MANIFEST_SHA256 = (
    "2e189ff6426bd3ad12fd29896fb6f76027a0d9e4424013d92edd0df3d143d3b6"
)
EXPECTED_PRESPEC_SHA256 = (
    "1b1152d8f09462b96bf2f35252a9cd5703b331ab7a3546d4708b189f81e8c178"
)


class OperatorCapAnalysisError(RuntimeError):
    """Raised when capped analysis authentication or isolation fails."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_analyzer() -> None:
    expected = {
        Path(analyzer.__file__).resolve(): EXPECTED_ANALYZER_SHA256,
        analyzer.MANIFEST_PATH.resolve(): EXPECTED_MANIFEST_SHA256,
        analyzer.PRESPEC_PATH.resolve(): EXPECTED_PRESPEC_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != digest:
            raise OperatorCapAnalysisError(f"frozen analysis dependency changed: {path}")
    try:
        analyzer._assert_exact_runtime()
    except BaseException as exc:
        raise OperatorCapAnalysisError("exact frozen analysis runtime check failed") from exc


def _require_fresh_output(path: Path) -> Path:
    destination = Path(path)
    root = ROOT.resolve(strict=True)
    lexical = (
        destination
        if destination.is_absolute()
        else Path(os.path.abspath(destination))
    )
    if not lexical.is_relative_to(root):
        raise OperatorCapAnalysisError("operator-cap analysis output escapes the repository")
    protected_staging = (
        root / "results/budget_toxicity_calibration_staging"
    ).resolve(strict=False)
    if lexical == protected_staging or lexical.is_relative_to(protected_staging):
        raise OperatorCapAnalysisError(
            "operator-cap analysis output must remain outside raw experiment staging"
        )
    current = root
    for component in lexical.relative_to(root).parts:
        current = current / component
        if current.is_symlink():
            raise OperatorCapAnalysisError(
                "operator-cap analysis output crosses a symlink"
            )
    if destination.exists() or destination.is_symlink():
        raise OperatorCapAnalysisError(
            "operator-cap analysis output must be a new, absent directory"
        )
    resolved_parent = lexical.parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(root):
        raise OperatorCapAnalysisError("operator-cap analysis output escapes the repository")
    return lexical


def _analysis_provenance(
    master: capped.OperatorCappedMaster,
) -> dict[str, Any]:
    provenance = dict(master.provenance)
    cap_record = dict(master.payload["operator_cap"])
    cap_record["authentication"] = {
        "cap_aware_analyzer_sha256": _sha256_file(Path(__file__).resolve()),
        "cap_finalizer_sha256": _sha256_file(Path(capped.__file__).resolve()),
        "frozen_analyzer_sha256": _sha256_file(Path(analyzer.__file__).resolve()),
        "amendment_artifact_sha256": cap_record["amendment"]["artifact_sha256"],
        "amendment_metadata_sha256": cap_record["amendment"]["metadata_sha256"],
        "amendment_commit_sha256": cap_record["amendment"]["commit_sha256"],
        "launch_fingerprint": master.payload["bindings"]["launch_fingerprint"],
    }
    provenance["operator_cap"] = cap_record
    provenance["precision_decisions"] = [
        {
            "cumulative_M": decision["cumulative_M"],
            "maximum_raw_mcse_points": decision["maximum_raw_mcse_points"],
            "maximum_guarded_mcse_points": decision[
                "maximum_guarded_mcse_points"
            ],
            "all_estimands_pass_guarded_1_5pp": decision[
                "all_estimands_pass_guarded_1_5pp"
            ],
            "top_up_limit_reached": decision["top_up_limit_reached"],
            "numerical_equivalence_guard": decision[
                "numerical_equivalence_guard"
            ],
            "next_M": decision["next_M"],
            "input_hashes": decision["input_hashes"],
        }
        for decision in master.precision_payloads
    ]
    return provenance


def run_operator_cap_analysis(
    *,
    raw_master_path: str | Path,
    amendment_path: str | Path,
    staging_root: str | Path,
    selected_oracle_path: str | Path,
    formal_oracle_path: str | Path,
    formal_oracle_metadata_path: str | Path,
    extension_oracle_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    _verify_frozen_analyzer()
    output = _require_fresh_output(Path(output_dir))
    cap_master = capped.load_operator_capped_master(
        raw_master_path,
        amendment_path=amendment_path,
        staging_root=staging_root,
    )
    if cap_master.payload["operator_cap"]["precision_target_achieved"] is not False:
        raise OperatorCapAnalysisError(
            "passing M0500 decisions must use the original analyzer chain"
        )
    if cap_master.payload["operator_cap"]["terminated_with_unresolved_precision"] is not True:
        raise OperatorCapAnalysisError("capped analysis lacks unresolved-precision disclosure")
    master = analyzer.RawMaster(
        payload={"final_M": 500},
        provenance=dict(cap_master.provenance),
        staging_root=cap_master.staging_root,
        block_payloads=cap_master.block_payloads,
    )
    selected_rows, selected_provenance = analyzer.load_selected_common_oracle(
        selected_oracle_path
    )
    formal_rows, formal_provenance = analyzer.load_formal_allocation_oracle(
        formal_oracle_path, formal_oracle_metadata_path
    )
    extension_rows, extension_provenance = analyzer.load_extension_allocation_oracle(
        extension_oracle_path
    )
    logical_rows, baseline_envelopes, streaming_audit = (
        analyzer.stream_master_analysis_state(master)
    )
    baseline = analyzer.baseline_identity_audit(
        baseline_envelopes, selected_rows, [*formal_rows, *extension_rows]
    )
    baseline["streaming_and_crn_audit"] = streaming_audit
    baseline["oracle_provenance"] = {
        "selected_family": selected_provenance,
        "formal_decision_master": formal_provenance,
        "comparator_extension": extension_provenance,
    }
    primary_rows, primary_details = analyzer.primary_analysis(logical_rows)
    absolute_rows, secondary_contrasts = analyzer.secondary_analysis(logical_rows)
    provenance = _analysis_provenance(cap_master)
    provenance["streaming_and_crn_audit"] = streaming_audit
    return analyzer.write_analysis_artifacts(
        output,
        primary_rows=primary_rows,
        primary_details=primary_details,
        absolute_rows=absolute_rows,
        secondary_contrasts=secondary_contrasts,
        provenance=provenance,
        baseline=baseline,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-master", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--staging-root", required=True, type=Path)
    parser.add_argument("--baseline-selected", required=True, type=Path)
    parser.add_argument("--baseline-formal-raw", required=True, type=Path)
    parser.add_argument("--baseline-formal-metadata", required=True, type=Path)
    parser.add_argument("--baseline-extension", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_operator_cap_analysis(
        raw_master_path=args.raw_master,
        amendment_path=args.amendment,
        staging_root=args.staging_root,
        selected_oracle_path=args.baseline_selected,
        formal_oracle_path=args.baseline_formal_raw,
        formal_oracle_metadata_path=args.baseline_formal_metadata,
        extension_oracle_path=args.baseline_extension,
        output_dir=args.output_dir,
    )
    print(f"analysis metadata: {result['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OperatorCapAnalysisError",
    "run_operator_cap_analysis",
]
