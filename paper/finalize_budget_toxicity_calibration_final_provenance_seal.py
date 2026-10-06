#!/usr/bin/env python3
"""Create an outcome-blind, hash-bound final calibration provenance seal.

The seal authenticates envelopes and operational precision state only.  It
never decodes an analyzer scientific artifact or a publication scientific
artifact, never classifies an effect, and never selects a sign or winner.

Two mutually exclusive terminal branches are accepted:

* the unchanged frozen chain when the original M0500 precision decision passed;
* the separately amended operator-cap chain when the original M0500 decision
  requested M1000 but the result-blind M0500 operator cap terminated execution.

Every upstream commit SHA-256 must be supplied from an external trusted record.
The exact pre-outcome v2 amendment trio and every production source dependency
are pinned below.  A completed seal is itself trusted only by an externally
recorded seal-commit SHA-256.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import secrets
import stat
from typing import Any, Iterable, Mapping, Sequence

import zstandard as zstd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_RELATIVE = Path(
    "paper/budget_toxicity_calibration_final_provenance_seal_schema.json"
)
FINALIZER_RELATIVE = Path(
    "paper/finalize_budget_toxicity_calibration_final_provenance_seal.py"
)
SCHEMA_SHA256 = "59e382276864571d51e75fb07d9c6d7b841694302d7f27ab6dcb5d695e03b25f"

DEFAULT_OUTPUT = ROOT / "results/budget_toxicity_calibration_final_provenance_seal"
ARTIFACT_NAME = "budget_toxicity_calibration_final_provenance_seal.json"
METADATA_NAME = ARTIFACT_NAME + ".metadata.json"
COMMIT_NAME = ARTIFACT_NAME + ".commit.json"

STATUS = (
    "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_FINAL_PROVENANCE_SEAL"
)
COMMIT_STATUS = (
    "COMMITTED_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_FINAL_PROVENANCE_SEAL"
)
ARTIFACT_CLASS = "outcome_blind_hash_bound_calibration_final_provenance_seal"
PASS_BRANCH = "ORIGINAL_M0500_PRECISION_PASS"
CAPPED_BRANCH = "OPERATOR_CAPPED_M0500_UNRESOLVED"

ORIGINAL_MASTER_STATUS = "COMPLETE_BUDGET_TOXICITY_CALIBRATION_RAW_MASTER"
ORIGINAL_MASTER_COMMIT_STATUS = (
    "COMMITTED_BUDGET_TOXICITY_CALIBRATION_RAW_MASTER"
)
ORIGINAL_MASTER_CLASS = "authenticated_budget_toxicity_calibration_raw_master"
ORIGINAL_MASTER_STORAGE = "authenticated_index_over_seed_block_indexes"
CAPPED_MASTER_STATUS = (
    "TERMINAL_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_OPERATOR_CAP"
)
CAPPED_MASTER_COMMIT_STATUS = (
    "COMMITTED_TERMINAL_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_OPERATOR_CAP"
)
CAPPED_MASTER_CLASS = (
    "authenticated_operator_capped_index_over_frozen_seed_block_indexes"
)
CAPPED_MASTER_STORAGE = (
    "authenticated_index_over_seed_block_indexes_plus_operator_cap"
)
ANALYSIS_STATUS = "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_ANALYSIS"
ANALYSIS_COMMIT_STATUS = (
    "COMMITTED_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_ANALYSIS"
)
PUBLICATION_STATUS = (
    "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_PUBLICATION"
)
PUBLICATION_COMMIT_STATUS = (
    "COMMITTED_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_PUBLICATION"
)
PUBLICATION_ARTIFACT_STATUS = (
    "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_PUBLICATION_ARTIFACT"
)
PROJECTION_STATUS = (
    "COMPLETE_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_MANUSCRIPT_PROJECTION"
)
PROJECTION_COMMIT_STATUS = (
    "COMMITTED_AUTHENTICATED_BUDGET_TOXICITY_CALIBRATION_MANUSCRIPT_PROJECTION"
)

ANALYSIS_METADATA_NAME = "budget_toxicity_calibration_analysis.metadata.json"
ANALYSIS_COMMIT_NAME = "budget_toxicity_calibration_analysis.commit.json"
PUBLICATION_METADATA_NAME = "budget_toxicity_calibration_publication.metadata.json"
PUBLICATION_COMMIT_NAME = "budget_toxicity_calibration_publication.commit.json"
PROJECTION_METADATA_NAME = (
    "budget_toxicity_calibration_manuscript_projection.metadata.json"
)
PROJECTION_COMMIT_NAME = (
    "budget_toxicity_calibration_manuscript_projection.commit.json"
)

ANALYSIS_INPUT_NAMES = (
    ANALYSIS_METADATA_NAME,
    ANALYSIS_COMMIT_NAME,
    "budget_toxicity_calibration_primary.json",
    "budget_toxicity_calibration_primary.json.metadata.json",
    "budget_toxicity_calibration_secondary.json",
    "budget_toxicity_calibration_secondary.json.metadata.json",
    "budget_toxicity_calibration_baseline_identity.json",
    "budget_toxicity_calibration_baseline_identity.json.metadata.json",
)

PRODUCTION_ANALYSIS_ARTIFACTS = (
    "budget_toxicity_calibration_primary.csv",
    "budget_toxicity_calibration_primary.tex",
    "budget_toxicity_calibration_primary.json",
    "budget_toxicity_calibration_secondary.csv",
    "budget_toxicity_calibration_secondary.tex",
    "budget_toxicity_calibration_secondary.json",
    "budget_toxicity_calibration_baseline_identity.json",
    "figure_calibration_acquisition_tradeoff.eps",
    "figure_calibration_acquisition_tradeoff.pdf",
    "figure_calibration_acquisition_tradeoff.png",
    "figure_calibration_penalties.eps",
    "figure_calibration_penalties.pdf",
    "figure_calibration_penalties.png",
    "figure_gate_failure_recovery_profiles.eps",
    "figure_gate_failure_recovery_profiles.pdf",
    "figure_gate_failure_recovery_profiles.png",
)

PRODUCTION_PUBLICATION_ARTIFACTS = (
    "budget_toxicity_calibration_all_118.csv",
    "budget_toxicity_calibration_all_118.json",
    "budget_toxicity_calibration_plotted_secondary_72.json",
    "budget_toxicity_calibration_policy_vs_cei_did_36.json",
    "budget_toxicity_calibration_policy_vs_cei_did_36.tex",
    "budget_toxicity_calibration_family_A.tex",
    "budget_toxicity_calibration_family_B.tex",
    "budget_toxicity_calibration_family_C.tex",
    "budget_toxicity_calibration_result_facts.json",
    "budget_toxicity_calibration_result_facts.tex",
    "figure_budget_toxicity_calibration_principal_profile.pdf",
    "figure_budget_toxicity_calibration_principal_profile.png",
    "figure_budget_toxicity_calibration_cei_reference.pdf",
    "figure_budget_toxicity_calibration_cei_reference.png",
    "figure_budget_toxicity_calibration_absolute_gate_brier_profiles.pdf",
    "figure_budget_toxicity_calibration_absolute_gate_brier_profiles.png",
    "figure_budget_toxicity_calibration_calibration_penalties.pdf",
    "figure_budget_toxicity_calibration_calibration_penalties.png",
    "figure_budget_toxicity_calibration_attenuation_interactions.pdf",
    "figure_budget_toxicity_calibration_attenuation_interactions.png",
)


def _production_publication_sidecar_contracts() -> dict[str, dict[str, Any]]:
    family_counts = {"A": 54, "B": 48, "C": 16}
    output: dict[str, dict[str, Any]] = {
        "budget_toxicity_calibration_all_118.csv": {
            "artifact_type": "all_118_primary_csv",
            "description": "Complete 118-row primary publication table in CSV form.",
            "selection": {"primary_rows": 118, "family_counts": family_counts},
        },
        "budget_toxicity_calibration_all_118.json": {
            "artifact_type": "all_118_primary_json",
            "description": "Complete 118-row primary publication table in JSON form.",
            "selection": {"primary_rows": 118, "family_counts": family_counts},
        },
        "budget_toxicity_calibration_plotted_secondary_72.json": {
            "artifact_type": "plotted_secondary_absolute_72_json",
            "description": (
                "All 72 plotted absolute gate/Brier rows with raw estimates, raw "
                "MCSEs, raw intervals, numerical guards, and guarded intervals."
            ),
            "selection": {
                "secondary_absolute_rows": 72,
                "complete_without_sign_filtering": True,
            },
        },
        "budget_toxicity_calibration_policy_vs_cei_did_36.json": {
            "artifact_type": "complete_policy_vs_cEI_calibration_did_36_json",
            "description": (
                "All 36 frozen policy-vs-cEI calibration difference-in-differences "
                "with raw and guarded pointwise intervals."
            ),
            "selection": {
                "secondary_policy_vs_cEI_calibration_did_rows": 36,
                "complete_without_sign_filtering": True,
            },
        },
        "budget_toxicity_calibration_policy_vs_cei_did_36.tex": {
            "artifact_type": (
                "complete_policy_vs_cEI_calibration_did_36_longtable"
            ),
            "description": (
                "Reader-facing complete 36-row policy-vs-cEI calibration "
                "difference-in-differences table."
            ),
            "selection": {
                "secondary_policy_vs_cEI_calibration_did_rows": 36,
                "complete_without_sign_filtering": True,
            },
        },
        "budget_toxicity_calibration_family_A.tex": {
            "artifact_type": "complete_family_A_longtable",
            "description": (
                "All 54 cKG-minus-comparator policy contrasts; cEI is shown only "
                "as a shared-gate reference."
            ),
            "selection": {"family": "A", "primary_rows": 54},
        },
        "budget_toxicity_calibration_family_B.tex": {
            "artifact_type": "complete_family_B_longtable",
            "description": (
                "All 48 erroneous-minus-correct assumed toxicity-noise "
                "calibration contrasts."
            ),
            "selection": {"family": "B", "primary_rows": 48},
        },
        "budget_toxicity_calibration_family_C.tex": {
            "artifact_type": "complete_family_C_longtable",
            "description": (
                "All 16 N=80-minus-N=20 calibration-penalty interactions used by "
                "the frozen finite-range attenuation predicate in the finite "
                "fixed-design audit."
            ),
            "selection": {"family": "C", "primary_rows": 16},
        },
        "budget_toxicity_calibration_result_facts.json": {
            "artifact_type": "frozen_result_facts_json",
            "description": (
                "Authenticated counts, critical values, and numerically guarded "
                "interpretation predicates."
            ),
            "selection": {
                "primary_rows": 118,
                "predicate_rule_frozen_before_outcomes": True,
            },
        },
        "budget_toxicity_calibration_result_facts.tex": {
            "artifact_type": "frozen_result_facts_tex",
            "description": (
                "TeX macros for authenticated counts, critical values, and "
                "numerically guarded interpretation predicates."
            ),
            "selection": {
                "primary_rows": 118,
                "predicate_rule_frozen_before_outcomes": True,
            },
        },
    }
    figure_specs = {
        "principal_profile": (
            "figure_budget_toxicity_calibration_principal_profile",
            36,
            0,
            "cKG-minus-tMSE-only and cKG-minus-Entropy contrasts with numerically "
            "guarded familywise simultaneous 95% max-t bands in the finite "
            "fixed-design audit. Lines connect the evaluated N=20, 40, and 80 "
            "snapshots.",
        ),
        "cei_reference": (
            "figure_budget_toxicity_calibration_cei_reference",
            18,
            0,
            "Shared-gate cEI reference contrasts with numerically guarded familywise "
            "simultaneous 95% max-t bands in the finite fixed-design audit. This "
            "reference is not a four-policy ranking.",
        ),
        "absolute_gate_brier_profiles": (
            "figure_budget_toxicity_calibration_absolute_gate_brier_profiles",
            0,
            72,
            "Raw absolute estimates with numerically guarded pointwise 95% Monte "
            "Carlo intervals in the finite fixed-design audit. Lines connect the "
            "evaluated N=20, 40, and 80 snapshots.",
        ),
        "calibration_penalties": (
            "figure_budget_toxicity_calibration_calibration_penalties",
            48,
            0,
            "Erroneous-minus-correct assumed toxicity-noise contrasts with "
            "numerically guarded familywise simultaneous 95% max-t bands in the "
            "finite fixed-design audit. Lines connect the evaluated N=20, 40, and "
            "80 snapshots.",
        ),
        "attenuation_interactions": (
            "figure_budget_toxicity_calibration_attenuation_interactions",
            16,
            0,
            "Calibration-penalty interactions between the evaluated N=20 and N=80 "
            "snapshots, with numerically guarded familywise simultaneous 95% max-t "
            "bands. A guarded-negative interaction supports finite-range "
            "attenuation only when its matched N=20 penalty is guarded-positive.",
        ),
    }
    for key, (stem, primary_rows, secondary_rows, description) in figure_specs.items():
        for suffix in ("pdf", "png"):
            output[f"{stem}.{suffix}"] = {
                "artifact_type": f"publication_figure_{key}",
                "description": description,
                "selection": {
                    "primary_rows": primary_rows,
                    "secondary_absolute_rows": secondary_rows,
                    "complete_without_sign_filtering": True,
                },
            }
    if set(output) != set(PRODUCTION_PUBLICATION_ARTIFACTS):
        raise AssertionError("publication sidecar contract coverage changed")
    return output

PRODUCTION_PROJECTION_MAPPING = (
    (
        "budget_toxicity_calibration_family_A.tex",
        "generated/budget_toxicity_calibration_family_A.tex",
    ),
    (
        "budget_toxicity_calibration_family_B.tex",
        "generated/budget_toxicity_calibration_family_B.tex",
    ),
    (
        "budget_toxicity_calibration_family_C.tex",
        "generated/budget_toxicity_calibration_family_C.tex",
    ),
    (
        "budget_toxicity_calibration_policy_vs_cei_did_36.tex",
        "generated/budget_toxicity_calibration_policy_vs_cei_did_36.tex",
    ),
    (
        "budget_toxicity_calibration_result_facts.tex",
        "generated/budget_toxicity_calibration_result_facts.tex",
    ),
    (
        "figure_budget_toxicity_calibration_principal_profile.pdf",
        "figure_budget_toxicity_calibration_principal_profile.pdf",
    ),
    (
        "figure_budget_toxicity_calibration_cei_reference.pdf",
        "figure_budget_toxicity_calibration_cei_reference.pdf",
    ),
    (
        "figure_budget_toxicity_calibration_absolute_gate_brier_profiles.pdf",
        "figure_budget_toxicity_calibration_absolute_gate_brier_profiles.pdf",
    ),
    (
        "figure_budget_toxicity_calibration_calibration_penalties.pdf",
        "figure_budget_toxicity_calibration_calibration_penalties.pdf",
    ),
    (
        "figure_budget_toxicity_calibration_attenuation_interactions.pdf",
        "figure_budget_toxicity_calibration_attenuation_interactions.pdf",
    ),
)

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SAFE_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+\Z")
PYTHON_VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)?\Z")
MAX_INPUT_FILE_BYTES = 256 * 1024 * 1024
MAX_MASTER_LOGICAL_BYTES = 64 * 1024 * 1024
MAX_BLOCK_INDEX_LOGICAL_BYTES = 16 * 1024 * 1024
MAX_BASELINE_LOGICAL_BYTES = 4 * 1024 * 1024
MAX_PRECISION_LOGICAL_BYTES = 16 * 1024 * 1024

PRIVACY_NOTE = (
    "No username, home path, hostname, serial number, or persistent "
    "device identifier is recorded."
)
THREAD_ENVIRONMENT_NAMES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
FROZEN_RUNTIME_IDENTITY = {
    "python": "3.11.6",
    "numpy": "1.26.4",
    "scipy": "1.16.3",
    "torch": "2.4.1",
    "gpytorch": "1.14",
    "matplotlib": "3.10.7",
    "ray": "2.10.0",
    "zstandard": "0.19.0",
}
RAY_CONTRACT = {
    "num_cpus": 10,
    "worker_num_cpus": 1,
    "maximum_in_flight": 20,
    "serial_fallback": False,
}

EXECUTION_BOUND_SOURCE_NAMES = (
    "paper/analyze_budget_toxicity_calibration_audit.py",
    "paper/budget_toxicity_calibration_core.py",
    "paper/monitor_budget_toxicity_calibration_precision.py",
    "paper/run_budget_toxicity_calibration_audit.py",
    "paper/analyze_primary_osa_exact_ckg_gate.py",
    "paper/formal_rerun_manifest.json",
    "paper/formal_rerun_preflight.py",
    "paper/run_full_formal_rerun.py",
    "paper/run_primary_osa_exact_ckg_gate.py",
    "src/safedosebo/acquisitions.py",
    "src/safedosebo/ckg_exact.py",
    "src/safedosebo/context.py",
    "src/safedosebo/gp.py",
    "src/safedosebo/metrics.py",
    "src/safedosebo/protocol.py",
    "src/safedosebo/registry.py",
    "src/safedosebo/surfaces.py",
    "src/safedosebo/trial.py",
)
IMPLEMENTATION_BOUND_SOURCE_NAMES = EXECUTION_BOUND_SOURCE_NAMES[:4]
FROZEN_FORMAL_BOUND_SOURCE_NAMES = EXECUTION_BOUND_SOURCE_NAMES[4:]


@dataclass(frozen=True)
class TrustedProfile:
    source_hashes: Mapping[str, str]
    amendment_relative: str
    amendment_hashes: Mapping[str, str]
    launch_fingerprint: str
    analysis_artifacts: tuple[str, ...]
    publication_artifacts: tuple[str, ...]
    publication_sidecar_contracts: Mapping[str, Mapping[str, Any]]
    projection_mapping: tuple[tuple[str, str], ...]


PRODUCTION_PROFILE = TrustedProfile(
    source_hashes={
        "paper/budget_toxicity_calibration_core.py": (
            "da51663bb6a6ae8e0bf1ceff4f6168d71d5ad71ceb3b193907c72c04ad0a7558"
        ),
        "paper/budget_toxicity_calibration_manifest.json": (
            "2e189ff6426bd3ad12fd29896fb6f76027a0d9e4424013d92edd0df3d143d3b6"
        ),
        "paper/budget_toxicity_calibration_prespec.md": (
            "1b1152d8f09462b96bf2f35252a9cd5703b331ab7a3546d4708b189f81e8c178"
        ),
        "paper/run_budget_toxicity_calibration_audit.py": (
            "c36420dc8a37c8a538ecba3b8d5d82753d428fb0eb4074dbe97baf83e474365d"
        ),
        "paper/monitor_budget_toxicity_calibration_precision.py": (
            "3994535bc1e2c300388c2d161d326e66f6185452971075e8d7bc9caf76bb07e0"
        ),
        "paper/analyze_budget_toxicity_calibration_audit.py": (
            "bcf3ec2964b88d61ac4becc8a9aa838831a3240d93a082abf42b5b328799ec2a"
        ),
        "paper/analyze_comparator_family.py": (
            "c1e2280d20c08a658a9ae71437e8de15e3e702fccb5deb65d488d5856cd6019c"
        ),
        "paper/analyze_primary_osa_exact_ckg_gate.py": (
            "579d4a390c051316b31cbd8775a655b274c4f0952a85090493fdc5c6a75dbbb2"
        ),
        "paper/formal_rerun_manifest.json": (
            "8fde3c59e83da0f8bce76f4d7d3caf6a7aa4e03c2fbb3d71f633d6563f207b24"
        ),
        "paper/formal_rerun_preflight.py": (
            "a326911a293600e2951f380bb8297b08ec1fda7440b417c8d9d0dda9b7de208b"
        ),
        "paper/run_full_formal_rerun.py": (
            "e3037131097e7be5672495604574ba20fbaff34014c27c978811e06c3b7c3d8f"
        ),
        "paper/run_primary_osa_exact_ckg_gate.py": (
            "063a8aa444be800d1be442dce0b13fba84bd96e493bf2022ca72e785461847a9"
        ),
        "paper/create_budget_toxicity_calibration_operator_cap_amendment.py": (
            "aafe5eaa13222ba2c108cf2dc656fb324ef9d63ea1f21665d98814997b3e0e7a"
        ),
        "paper/budget_toxicity_calibration_operator_cap_schema.json": (
            "c5aba84194d7600261222ce5950a096b8980a85e5c2af9cc9743614ccf803034"
        ),
        "paper/finalize_budget_toxicity_calibration_operator_capped_master.py": (
            "6e8f0644f7d91b38056731816e1ac4fe1c1d1623135dbea11298c3721dfe7a5f"
        ),
        "paper/analyze_budget_toxicity_calibration_operator_cap.py": (
            "d8bb96a3a8d42574047717b076bf0d6d1a91b143bff01970811fdf8beb5b4031"
        ),
        "paper/budget_toxicity_calibration_publication_contract.json": (
            "a1904b435d4e42ef6df25094bc1f3ce35156ceaec5bf157ec24fb2ff2827b6a8"
        ),
        "paper/generate_budget_toxicity_calibration_publication.py": (
            "e14c2464f586f3abb805bfa24bb47f2ecc61a93e7405e0cb51489d497c6aa919"
        ),
        "paper/project_budget_toxicity_calibration_publication_to_manuscript.py": (
            "12db5f3ba56ffc0e97263998a676f1310f6b7204a6ad6b9286ef6ec5fdd5cb12"
        ),
        "src/safedosebo/acquisitions.py": (
            "1e06ad27fd355fb8106d3a07d44ad446edbddd3e3d1f4e36766d2fc68d0c5756"
        ),
        "src/safedosebo/ckg_exact.py": (
            "b0689cdfb1bc95f0bb6da12745962fb51f42cfc5e8fb15f39b07de702527484d"
        ),
        "src/safedosebo/context.py": (
            "61085802c1d2d80ade1622aa062b0a10a1f567120404227d2e09d1622dfe65b1"
        ),
        "src/safedosebo/gp.py": (
            "d4ce5cc9b1d2aabe4fe1b5480bcb0c244782c7a914bff075ec43c6fd0917ed72"
        ),
        "src/safedosebo/metrics.py": (
            "45a62ea39dc36af6ee7a5e17d547e6bbb2ad40499d028195c3f4fe3cf0d35fb7"
        ),
        "src/safedosebo/protocol.py": (
            "ff0a55b3e12bceb5a556121f721b44a9e186b0dc823d895cf98cda9681f3e8b7"
        ),
        "src/safedosebo/registry.py": (
            "3d097a6b33a27f192d0dd7a90b4e4a89b34f04821c8b42c2681fc101460e5241"
        ),
        "src/safedosebo/surfaces.py": (
            "48d9a238e17c919c09fc7f577e4cd3c85e21023b60a0df7ee1d99d23a3419899"
        ),
        "src/safedosebo/trial.py": (
            "61929fa04d30b27155c2aa99992bed30ad7ccb072b43e0c46a1e282dfd81a3ce"
        ),
    },
    amendment_relative=(
        "results/budget_toxicity_calibration_operator_cap_v2__launch_"
        "364a65e046c83acf__20260827/"
        "budget_toxicity_calibration_operator_cap_M0500_amendment.json"
    ),
    amendment_hashes={
        "artifact": "3097380d0f3fe661807e162c99a1b83583d1c2ae2ca1dcc31e400e1c47455485",
        "metadata": "d56c68e4d36516502abfb07d80f93037525a7fca91ad3a62a4a084a62021319e",
        "commit": "1e0db4c035327553b3e1fe97822acb2c01fe10b21be9743a243a56dbc6c710d6",
    },
    launch_fingerprint=(
        "364a65e046c83acf5458885e8b07e2cb3de441c5ba01f0a1d02e6f59f6d3e0a5"
    ),
    analysis_artifacts=PRODUCTION_ANALYSIS_ARTIFACTS,
    publication_artifacts=PRODUCTION_PUBLICATION_ARTIFACTS,
    publication_sidecar_contracts=_production_publication_sidecar_contracts(),
    projection_mapping=PRODUCTION_PROJECTION_MAPPING,
)


class FinalProvenanceSealError(RuntimeError):
    """Raised when a provenance input or final seal fails closed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _strict_json_equal(observed: Any, expected: Any) -> bool:
    """JSON equality with exact scalar types (notably bool is never an int)."""

    if isinstance(observed, Mapping) or isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
            return False
        if set(observed) != set(expected):
            return False
        return all(
            _strict_json_equal(observed[key], expected[key]) for key in expected
        )
    if isinstance(observed, list) or isinstance(expected, list):
        if not isinstance(observed, list) or not isinstance(expected, list):
            return False
        return len(observed) == len(expected) and all(
            _strict_json_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return type(observed) is type(expected) and observed == expected


def _assert_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FinalProvenanceSealError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, label=f"{label}[{index}]")


def _decode_canonical_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite token {token}")

    try:
        value = json.loads(raw, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FinalProvenanceSealError(f"{label} is not finite valid JSON") from exc
    if not isinstance(value, dict):
        raise FinalProvenanceSealError(f"{label} is not a JSON object")
    _assert_finite(value, label=label)
    if raw != _canonical_json_bytes(value):
        raise FinalProvenanceSealError(f"{label} is not canonical JSON")
    return value


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise FinalProvenanceSealError(f"{label} is not a lowercase SHA-256")
    return value


def _exact_fields(value: Mapping[str, Any], expected: Iterable[str], *, label: str) -> None:
    observed = set(value)
    required = set(expected)
    if observed != required:
        raise FinalProvenanceSealError(
            f"{label} fields changed: missing={sorted(required-observed)}, "
            f"extra={sorted(observed-required)}"
        )


def _safe_relative(value: str | Path, *, label: str) -> Path:
    try:
        text = os.fspath(value)
    except TypeError as exc:
        raise FinalProvenanceSealError(f"{label} is absent") from exc
    if not isinstance(text, str) or not text:
        raise FinalProvenanceSealError(f"{label} is absent")
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or text != pure.as_posix()
        or any(part in ("", ".", "..") for part in pure.parts)
        or "\\" in text
    ):
        raise FinalProvenanceSealError(f"{label} is not a canonical relative path")
    return Path(*pure.parts)


def _absolute_lexical(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def _guard_root(root: str | Path) -> Path:
    candidate = Path(os.path.abspath(os.fspath(root)))
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current = current / component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise FinalProvenanceSealError(
                f"repository root component is absent: {current}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise FinalProvenanceSealError(
                f"repository root crosses a symlink: {current}"
            )
    try:
        mode = candidate.lstat().st_mode
    except FileNotFoundError as exc:
        raise FinalProvenanceSealError(f"repository root is absent: {candidate}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise FinalProvenanceSealError("repository root must be a non-symlink directory")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate:
        raise FinalProvenanceSealError("repository root is not a canonical physical path")
    return resolved


def _guard_path(
    path: str | Path,
    *,
    root: Path,
    label: str,
    kind: str,
    must_exist: bool = True,
) -> Path:
    root_resolved = _guard_root(root)
    candidate = _absolute_lexical(path, root=root_resolved)
    if not candidate.is_relative_to(root_resolved):
        raise FinalProvenanceSealError(f"{label} escapes the repository root")
    current = root_resolved
    parts = candidate.relative_to(root_resolved).parts
    for index, component in enumerate(parts):
        if component in ("", ".", ".."):
            raise FinalProvenanceSealError(f"{label} has a non-canonical component")
        current = current / component
        exists = current.exists() or current.is_symlink()
        if exists:
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise FinalProvenanceSealError(f"{label} crosses a symlink")
            if index < len(parts) - 1 and not stat.S_ISDIR(mode):
                raise FinalProvenanceSealError(f"{label} crosses a non-directory")
    if not must_exist:
        return candidate
    try:
        mode = candidate.lstat().st_mode
    except FileNotFoundError as exc:
        raise FinalProvenanceSealError(f"{label} is absent: {candidate}") from exc
    expected = stat.S_ISREG(mode) if kind == "file" else stat.S_ISDIR(mode)
    if not expected:
        raise FinalProvenanceSealError(f"{label} is not a regular {kind}")
    return candidate


def _open_rooted_directory(path: Path, *, root: Path, label: str) -> int:
    """Open every directory component with O_NOFOLLOW and return a pinned fd."""

    root_resolved = _guard_root(root)
    candidate = _guard_path(path, root=root_resolved, label=label, kind="dir")
    relative = candidate.relative_to(root_resolved)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(root_resolved, flags)
    try:
        for component in relative.parts:
            next_descriptor = os.open(
                component, flags, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular(
    path: str | Path,
    *,
    root: Path,
    label: str,
    maximum_bytes: int = MAX_INPUT_FILE_BYTES,
) -> bytes:
    candidate = _guard_path(path, root=root, label=label, kind="file")
    if not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY")):
        raise FinalProvenanceSealError("platform lacks O_NOFOLLOW")
    parent_descriptor = _open_rooted_directory(
        candidate.parent, root=root, label=f"{label} parent"
    )
    try:
        descriptor = os.open(
            candidate.name, os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    finally:
        os.close(parent_descriptor)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise FinalProvenanceSealError(f"{label} changed during authentication")
        if (
            isinstance(maximum_bytes, bool)
            or not isinstance(maximum_bytes, int)
            or maximum_bytes < 0
            or observed.st_size > maximum_bytes
        ):
            raise FinalProvenanceSealError(
                f"{label} exceeds its authenticated byte bound"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            remaining = maximum_bytes - total
            chunk = os.read(descriptor, min(1024 * 1024, remaining + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise FinalProvenanceSealError(
                    f"{label} exceeds its authenticated byte bound while reading"
                )
            chunks.append(chunk)
        final = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            any(getattr(final, field) != getattr(observed, field) for field in stable_fields)
            or total != observed.st_size
        ):
            raise FinalProvenanceSealError(
                f"{label} changed during bounded authentication"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _relative_path(path: Path, *, root: Path, label: str) -> str:
    candidate = _guard_path(path, root=root, label=label, kind="file")
    return candidate.relative_to(_guard_root(root)).as_posix()


def _file_commitment(path: Path, *, root: Path, label: str) -> dict[str, Any]:
    raw = _read_regular(path, root=root, label=label)
    return {
        "path": _relative_path(path, root=root, label=label),
        "sha256": _sha256_bytes(raw),
        "bytes": len(raw),
    }


def _flat_inventory(directory: Path, *, root: Path, label: str) -> set[str]:
    folder = _guard_path(directory, root=root, label=label, kind="dir")
    output: set[str] = set()
    descriptor = _open_rooted_directory(folder, root=root, label=label)
    try:
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise FinalProvenanceSealError(
                        f"{label} contains a non-regular or symlink entry: {entry.name}"
                    )
                if SAFE_NAME_RE.fullmatch(entry.name) is None:
                    raise FinalProvenanceSealError(
                        f"{label} contains a non-canonical filename: {entry.name}"
                    )
                output.add(entry.name)
    finally:
        os.close(descriptor)
    return output


def _assert_exact_inventory(
    directory: Path,
    expected: Iterable[str],
    *,
    root: Path,
    label: str,
) -> None:
    observed = _flat_inventory(directory, root=root, label=label)
    required = set(expected)
    if observed != required:
        raise FinalProvenanceSealError(
            f"{label} inventory changed: missing={sorted(required-observed)}, "
            f"extra={sorted(observed-required)}"
        )


def _assert_master_raw_inventory(
    directory: Path,
    expected_files: Iterable[str],
    *,
    root: Path,
) -> None:
    """Validate the raw top level while leaving outcome-bearing shards opaque."""

    folder = _guard_path(
        directory, root=root, label="master raw directory", kind="dir"
    )
    expected = {name: "file" for name in expected_files}
    expected["shards"] = "directory"
    observed: dict[str, str] = {}
    descriptor = _open_rooted_directory(
        folder, root=root, label="master raw directory"
    )
    try:
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if entry.is_symlink() or SAFE_NAME_RE.fullmatch(entry.name) is None:
                    raise FinalProvenanceSealError(
                        f"master raw directory contains an unsafe entry: {entry.name}"
                    )
                if entry.is_file(follow_symlinks=False):
                    kind = "file"
                elif entry.is_dir(follow_symlinks=False):
                    kind = "directory"
                else:
                    raise FinalProvenanceSealError(
                        f"master raw directory contains a non-regular entry: {entry.name}"
                    )
                observed[entry.name] = kind
    finally:
        os.close(descriptor)
    if observed != expected:
        raise FinalProvenanceSealError(
            "master raw directory inventory changed: "
            f"expected={sorted(expected.items())}, observed={sorted(observed.items())}"
        )
    _guard_path(
        folder / "shards", root=root, label="opaque raw shard directory", kind="dir"
    )


def _timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise FinalProvenanceSealError(f"{label} is absent")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FinalProvenanceSealError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise FinalProvenanceSealError(f"{label} is not explicit UTC")
    return parsed


def _assert_time_order(
    generated: Any,
    committed: Any,
    *,
    label: str,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    first = _timestamp(generated, label=f"{label} generated timestamp")
    second = _timestamp(committed, label=f"{label} committed timestamp")
    current = datetime.now(timezone.utc) if now is None else now
    if second < first:
        raise FinalProvenanceSealError(f"{label} was committed before generation")
    if second - first > timedelta(minutes=5):
        raise FinalProvenanceSealError(f"{label} commit delay exceeds five minutes")
    if second > current + timedelta(minutes=5):
        raise FinalProvenanceSealError(f"{label} timestamp is implausibly in the future")
    return first, second


def _runtime_environment(*, captured_at: datetime | None = None) -> dict[str, Any]:
    moment = datetime.now(timezone.utc) if captured_at is None else captured_at
    return {
        "captured_at_utc": moment.isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in THREAD_ENVIRONMENT_NAMES
        },
        "privacy_note": PRIVACY_NOTE,
    }


def _validate_recorded_host_fields(
    value: Mapping[str, Any], *, label: str, require_single_thread: bool
) -> None:
    for field in ("platform", "machine"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise FinalProvenanceSealError(f"{label} {field} is malformed")
    if (
        not isinstance(value.get("python"), str)
        or PYTHON_VERSION_RE.fullmatch(value["python"]) is None
    ):
        raise FinalProvenanceSealError(f"{label} Python version is malformed")
    cpu_count = value.get("logical_cpu_count")
    if (
        isinstance(cpu_count, bool)
        or not isinstance(cpu_count, int)
        or cpu_count <= 0
    ):
        raise FinalProvenanceSealError(f"{label} logical CPU count is malformed")
    thread_environment = value.get("thread_environment")
    if (
        not isinstance(thread_environment, Mapping)
        or set(thread_environment) != set(THREAD_ENVIRONMENT_NAMES)
    ):
        raise FinalProvenanceSealError(f"{label} thread environment changed")
    for name in THREAD_ENVIRONMENT_NAMES:
        setting = thread_environment[name]
        if require_single_thread:
            valid = setting == "1"
        else:
            valid = setting is None or isinstance(setting, str)
        if not valid:
            raise FinalProvenanceSealError(
                f"{label} thread environment value is malformed: {name}"
            )
    if value.get("privacy_note") != PRIVACY_NOTE:
        raise FinalProvenanceSealError(f"{label} privacy note changed")


def _validate_seal_environment(value: Any, *, generated_at: datetime) -> None:
    if not isinstance(value, Mapping):
        raise FinalProvenanceSealError("seal environment is absent")
    _exact_fields(value, {
        "captured_at_utc", "platform", "machine", "python",
        "logical_cpu_count", "thread_environment", "privacy_note",
    }, label="seal environment")
    captured = _timestamp(
        value.get("captured_at_utc"), label="environment.captured_at_utc"
    )
    _validate_recorded_host_fields(
        value, label="seal environment", require_single_thread=False
    )
    if captured > generated_at or generated_at - captured > timedelta(minutes=5):
        raise FinalProvenanceSealError("seal environment timestamp is inconsistent")


def _validate_upstream_environment(value: Any, *, generated_at: datetime) -> None:
    if not isinstance(value, Mapping):
        raise FinalProvenanceSealError("master environment is absent")
    required = {
        "captured_at_utc", "platform", "machine", "python",
        "logical_cpu_count", "ray_contract", "thread_environment", "privacy_note",
    }
    _exact_fields(value, required, label="master environment")
    captured = _timestamp(value.get("captured_at_utc"), label="master environment timestamp")
    if captured > generated_at or generated_at - captured > timedelta(minutes=5):
        raise FinalProvenanceSealError("master environment timestamp is inconsistent")
    if not _strict_json_equal(value.get("ray_contract"), RAY_CONTRACT):
        raise FinalProvenanceSealError("master Ray environment contract changed")
    _validate_recorded_host_fields(
        value, label="master environment", require_single_thread=True
    )


def _bundle_hash(mapping: Mapping[str, str]) -> str:
    return _sha256_bytes(_canonical_json_bytes(dict(sorted(mapping.items()))))


def _commitments_for_flat_directory(
    directory: Path,
    names: Iterable[str],
    *,
    root: Path,
    label: str,
) -> dict[str, dict[str, Any]]:
    return {
        name: _file_commitment(
            directory / name, root=root, label=f"{label} {name}"
        )
        for name in sorted(names)
    }


def _hash_map(files: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {name: str(entry["sha256"]) for name, entry in files.items()}


def _verify_schema_and_finalizer(root: Path) -> tuple[dict[str, Any], str]:
    schema_raw = _read_regular(
        root / SCHEMA_RELATIVE, root=root, label="final-seal schema"
    )
    if _sha256_bytes(schema_raw) != SCHEMA_SHA256:
        raise FinalProvenanceSealError("final-seal schema hash changed")
    schema = _decode_canonical_json(schema_raw, label="final-seal schema")
    if (
        schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        or schema.get("$id")
        != (
            "https://safedosebo.local/schemas/"
            "budget_toxicity_calibration_final_provenance_seal.schema.json"
        )
        or schema.get("title")
        != "SafeDoseBO budget-toxicity calibration final provenance seal"
        or schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or not isinstance(schema.get("$defs"), Mapping)
        or not isinstance(schema.get("properties"), Mapping)
        or not isinstance(schema.get("required"), list)
    ):
        raise FinalProvenanceSealError("final-seal schema identity changed")
    finalizer_raw = _read_regular(
        root / FINALIZER_RELATIVE, root=root, label="final-seal finalizer"
    )
    return schema, _sha256_bytes(finalizer_raw)


def _verify_sources(
    root: Path, profile: TrustedProfile
) -> tuple[dict[str, dict[str, Any]], str]:
    commitments: dict[str, dict[str, Any]] = {}
    for relative, expected in sorted(profile.source_hashes.items()):
        _require_sha(expected, label=f"trusted source hash {relative}")
        safe = _safe_relative(relative, label=f"trusted source path {relative}")
        entry = _file_commitment(
            root / safe, root=root, label=f"frozen source {relative}"
        )
        if entry["sha256"] != expected:
            raise FinalProvenanceSealError(f"frozen source hash changed: {relative}")
        commitments[relative] = entry
    return commitments, _bundle_hash(_hash_map(commitments))


def _authenticate_amendment(
    root: Path, profile: TrustedProfile
) -> dict[str, Any]:
    relative = _safe_relative(profile.amendment_relative, label="amendment v2 path")
    artifact = _guard_path(
        root / relative, root=root, label="amendment v2 artifact", kind="file"
    )
    metadata = Path(str(artifact) + ".metadata.json")
    commit = Path(str(artifact) + ".commit.json")
    expected_names = {artifact.name, metadata.name, commit.name}
    _assert_exact_inventory(
        artifact.parent, expected_names, root=root, label="amendment v2 directory"
    )
    raw = {
        "artifact": _read_regular(artifact, root=root, label="amendment artifact"),
        "metadata": _read_regular(metadata, root=root, label="amendment metadata"),
        "commit": _read_regular(commit, root=root, label="amendment commit"),
    }
    for kind, value in raw.items():
        expected = _require_sha(
            profile.amendment_hashes.get(kind), label=f"trusted amendment {kind} hash"
        )
        if _sha256_bytes(value) != expected:
            raise FinalProvenanceSealError(f"exact amendment v2 {kind} hash changed")
    payload = _decode_canonical_json(raw["artifact"], label="amendment artifact")
    sidecar = _decode_canonical_json(raw["metadata"], label="amendment metadata")
    marker = _decode_canonical_json(raw["commit"], label="amendment commit")
    if (
        not _strict_json_equal(payload.get("schema_version"), 1)
        or payload.get("status") != "DECLARED_BUDGET_TOXICITY_CALIBRATION_OPERATOR_CAP"
        or payload.get("artifact_class") != "result_blind_operator_cap_amendment"
        or payload.get("launch_fingerprint") != profile.launch_fingerprint
        or not _strict_json_equal(payload.get("cap_cumulative_M"), 500)
        or payload.get("declaration_phase")
        != "M0500_TOPUP_IN_PROGRESS_BEFORE_M0500_PRECISION_MONITOR"
    ):
        raise FinalProvenanceSealError("amendment v2 payload identity changed")
    if (
        sidecar.get("artifact") != artifact.name
        or sidecar.get("artifact_sha256") != _sha256_bytes(raw["artifact"])
        or sidecar.get("artifact_bytes") != len(raw["artifact"])
        or sidecar.get("audit_id") != payload.get("audit_id")
        or sidecar.get("launch_fingerprint") != profile.launch_fingerprint
        or not _strict_json_equal(sidecar.get("cap_cumulative_M"), 500)
        or sidecar.get("immutable") is not True
    ):
        raise FinalProvenanceSealError("amendment v2 metadata binding changed")
    if (
        marker.get("artifact") != artifact.name
        or marker.get("artifact_sha256") != _sha256_bytes(raw["artifact"])
        or marker.get("metadata") != metadata.name
        or marker.get("metadata_sha256") != _sha256_bytes(raw["metadata"])
        or marker.get("audit_id") != payload.get("audit_id")
        or marker.get("launch_fingerprint") != profile.launch_fingerprint
        or not _strict_json_equal(marker.get("cap_cumulative_M"), 500)
        or marker.get("immutable") is not True
    ):
        raise FinalProvenanceSealError("amendment v2 commit binding changed")
    declared, committed = _assert_time_order(
        payload.get("declared_at_utc"), marker.get("committed_at_utc"),
        label="amendment v2",
    )
    if sidecar.get("declared_at_utc") != payload.get("declared_at_utc"):
        raise FinalProvenanceSealError("amendment v2 declaration timestamp differs")
    if declared > committed:
        raise FinalProvenanceSealError("amendment v2 timestamp order changed")
    for path in (artifact, metadata, commit):
        if path.stat().st_mode & 0o222:
            raise FinalProvenanceSealError("amendment v2 trio is not filesystem read-only")
    return {
        "envelope": {
            "artifact": _file_commitment(artifact, root=root, label="amendment artifact"),
            "metadata": _file_commitment(metadata, root=root, label="amendment metadata"),
            "commit": _file_commitment(commit, root=root, label="amendment commit"),
        },
        "audit_id": payload["audit_id"],
        "launch_fingerprint": profile.launch_fingerprint,
        "declared_at_utc": payload["declared_at_utc"],
        "committed_at_utc": marker["committed_at_utc"],
        "cap_cumulative_M": 500,
    }


ORIGINAL_MASTER_METADATA_FIELDS = {
    "schema_version", "status", "artifact_class", "storage_model", "artifact",
    "artifact_sha256", "artifact_bytes", "uncompressed_sha256",
    "uncompressed_bytes", "final_M", "path_count", "snapshot_count",
    "bindings", "compression", "environment", "generated_at_utc",
}

MASTER_BINDING_FIELDS = {
    "manifest_sha256", "analysis_spec_sha256", "design_sha256",
    "implementation_source_sha256", "frozen_formal_source_sha256",
    "source_sha256", "runtime_identity", "ray_contract", "launch_fingerprint",
}
BLOCK_INDEX_FIELDS = {
    "block_id", "artifact", "artifact_sha256", "uncompressed_sha256",
    "uncompressed_bytes", "metadata", "metadata_sha256", "commit",
    "commit_sha256", "path_count", "snapshot_count", "shard_count",
}
BLOCK_METADATA_FIELDS = {
    "schema_version", "status", "artifact_class", "storage_model", "artifact",
    "artifact_sha256", "artifact_bytes", "uncompressed_sha256",
    "uncompressed_bytes", "block_id", "seed_start", "seed_stop_exclusive",
    "path_count", "snapshot_count", "shard_count", "bindings", "compression",
    "environment", "generated_at_utc",
}
BLOCK_COMMIT_FIELDS = {
    "schema_version", "status", "artifact", "artifact_sha256",
    "uncompressed_sha256", "metadata", "metadata_sha256", "block_id",
    "path_count", "snapshot_count", "shard_count", "launch_fingerprint",
    "immutable", "committed_at_utc",
}
BLOCK_PAYLOAD_FIELDS = {
    "schema_version", "status", "artifact_class", "storage_model", "block",
    "bindings", "canonical_sort_key", "shard_commits", "path_count",
    "snapshot_count",
}
SHARD_INDEX_FIELDS = {
    "shard_id", "block_id", "artifact", "artifact_sha256",
    "uncompressed_sha256", "uncompressed_bytes", "metadata",
    "metadata_sha256", "commit", "commit_sha256", "path_count",
    "snapshot_count", "execution_keys_sha256", "first_execution_key",
    "last_execution_key",
}
PRECISION_INDEX_FIELDS = {
    "cumulative_M", "artifact", "artifact_sha256", "metadata",
    "metadata_sha256", "commit", "commit_sha256", "next_M",
}
PRECISION_PAYLOAD_FIELDS = {
    "schema_version", "status", "artifact_class", "cumulative_M",
    "numerical_equivalence_guard", "estimands", "maximum_raw_mcse_points",
    "maximum_guarded_mcse_points", "all_estimands_pass_guarded_1_5pp",
    "top_up_limit_reached", "next_M", "input_hashes",
}
PRECISION_ROW_FIELDS = {
    "estimand_id", "cumulative_M", "row_guard_class",
    "replicate_bound_points", "raw_mcse_points", "mcse_guard_points",
    "guarded_mcse_points", "passes_guarded_1_5pp",
}
PRECISION_METADATA_FIELDS = {
    "schema_version", "status", "artifact_class", "artifact",
    "artifact_sha256", "artifact_bytes", "uncompressed_sha256",
    "uncompressed_bytes", "cumulative_M", "estimand_count",
    "numerical_equivalence_guard", "maximum_raw_mcse_points",
    "maximum_guarded_mcse_points", "all_estimands_pass_guarded_1_5pp",
    "top_up_limit_reached", "next_M", "input_hashes",
}
PRECISION_COMMIT_FIELDS = {
    "schema_version", "status", "artifact", "artifact_sha256", "metadata",
    "metadata_sha256", "cumulative_M", "estimand_count",
    "maximum_guarded_mcse_points", "all_estimands_pass_guarded_1_5pp",
    "top_up_limit_reached", "next_M",
}
BASELINE_INDEX_FIELDS = {
    "status", "cumulative_M", "pass", "artifact", "artifact_sha256",
    "metadata", "metadata_sha256", "commit", "commit_sha256",
    "checked_common_cells", "checked_allocation_histories",
    "unavailable_allocation_histories",
}
BASELINE_PAYLOAD_FIELDS = {
    "schema_version", "status", "artifact_class", "cumulative_M",
    "input_hashes", "oracle_hashes", "checked_common_cells",
    "checked_allocation_histories", "unavailable_allocation_histories",
    "exception_definition", "pass",
}
BASELINE_METADATA_FIELDS = {
    "schema_version", "status", "artifact_class", "artifact",
    "artifact_sha256", "artifact_bytes", "uncompressed_sha256",
    "uncompressed_bytes", "input_hashes", "oracle_hashes", "cumulative_M",
    "pass", "checked_common_cells", "checked_allocation_histories",
    "unavailable_allocation_histories",
}
BASELINE_COMMIT_FIELDS = {
    "schema_version", "status", "artifact", "artifact_sha256", "metadata",
    "metadata_sha256", "cumulative_M", "pass", "checked_common_cells",
    "checked_allocation_histories", "unavailable_allocation_histories",
}
BLOCK_STATUS = "COMPLETE_IMMUTABLE_BUDGET_TOXICITY_CALIBRATION_SEED_BLOCK"
BLOCK_COMMIT_STATUS = (
    "COMMITTED_IMMUTABLE_BUDGET_TOXICITY_CALIBRATION_SEED_BLOCK"
)
BLOCK_ARTIFACT_CLASS = "authenticated_seed_block_index_over_immutable_shards"
BLOCK_STORAGE_MODEL = "authenticated_index_over_immutable_shards"
PRECISION_STATUS = (
    "COMPLETE_BLINDED_BUDGET_TOXICITY_CALIBRATION_PRECISION_DECISION"
)
PRECISION_COMMIT_STATUS = (
    "COMMITTED_BLINDED_BUDGET_TOXICITY_CALIBRATION_PRECISION_DECISION"
)
PRECISION_ARTIFACT_CLASS = "blind_precision_decision_no_effect_estimates"
BASELINE_STATUS = "PASS_BUDGET_TOXICITY_CALIBRATION_BASELINE_IDENTITY"
BASELINE_COMMIT_STATUS = (
    "COMMITTED_BUDGET_TOXICITY_CALIBRATION_BASELINE_IDENTITY"
)
BASELINE_ARTIFACT_CLASS = "authenticated_baseline_identity_pass_only"
BASELINE_RELATIVE = (
    "baseline/budget_toxicity_calibration_baseline_identity_M0200.json.zst"
)
BASELINE_ORACLE_HASHES = {
    "comparator_extension.artifact_sha256": (
        "866c6ab2af18aa54187001c9286c9b20b1ebb6258756895d5610a3f1ce9bbf5c"
    ),
    "comparator_extension.commit_sha256": (
        "ebc9ec4d6327f00069999431d424364413c2018e5546815db924a75a6aa8fb2a"
    ),
    "comparator_extension.metadata_sha256": (
        "88e22c060471d95d659a0d81b6eaff42d2dd9a35d5a03f93464fb6d601ba8f35"
    ),
    "formal_decision_master.artifact_sha256": (
        "fc60ac3250421c69a52fa9b932581e25c8c54d6f1bdfed3e1c6c7b35b02b3fe9"
    ),
    "formal_decision_master.metadata_sha256": (
        "7daab718463d1f6f0107cb0c7d6d33e6b0ee4300c0eb2fdc957c10b11f3f528e"
    ),
    "selected_comparator_family.artifact_sha256": (
        "b5ebd7768efee30fdc3b0a252e6c7c702215c444adf3a9326b5423c068b6283e"
    ),
    "selected_comparator_family.metadata_sha256": (
        "75dce1739ae521ece6a48a02a38458d52e9d4f6341721fa2daf7cb17e82f0150"
    ),
}
COMPARATOR_EXTENSION_COMPLETION_MANIFEST_SHA256 = (
    "40ed347851d1866d7860ccc40751985f2fbb0a3b45ef063d1af1488c14834c69"
)
COMPARATOR_EXTENSION_SPEC_SHA256 = (
    "117e8d1cd3221a54786075494c7b92503001dba90c72a22f4049d2fc807cccd8"
)
COMPARATOR_EXTENSION_RUNNER_SHA256 = (
    "5047ea4b4379545cbd881578b5c8374adda58e43c07fd7a43e8941010e9ff5f3"
)
COMPARATOR_EXTENSION_RUNTIME_IDENTITY = {
    "python": "3.11.6",
    "numpy": "1.26.4",
    "scipy": "1.16.3",
    "torch": "2.4.1",
    "gpytorch": "1.14",
    "ray": "2.10.0",
    "zstandard": "0.19.0",
}
COMPARATOR_EXTENSION_RAY_CONTRACT = {
    "num_cpus": 10,
    "worker_num_cpus": 1,
    "maximum_in_flight": 20,
    "serial_fallback": False,
}
PRECISION_POLICIES = ("cKG-exact-formal", "tmse", "qBIG", "cEI")
PRECISION_COMPARATORS = ("tmse", "qBIG", "cEI")
PRECISION_FACTORS = (0.5, 1.0, 2.0)
PRECISION_GATES = (0.7, 0.9)
PRECISION_STRATA = (0, 1)
PRECISION_ERRONEOUS_FACTORS = (0.5, 2.0)
PRECISION_BUDGETS = (20, 40, 80)
PRECISION_OUTCOME_TERMINAL = "terminal_true_boundary_exceedance_pct"
PRECISION_OUTCOME_ASSIGNMENT = "above_boundary_assignment_pct"
PRECISION_OUTCOME_EXCLUSION = "terminal_target_exclusion_pct"
PRECISION_OUTCOME_BRIER = "panel_brier_score_pct"
NUMERICAL_EQUIVALENCE_ETA = 2.0 ** -26
PANEL_BRIER_SNAPSHOT_BOUND_POINTS = 200.0 * NUMERICAL_EQUIVALENCE_ETA
FAMILY_B_BRIER_REPLICATE_BOUND_POINTS = 400.0 * NUMERICAL_EQUIVALENCE_ETA
FAMILY_C_BRIER_INTERACTION_BOUND_POINTS = 800.0 * NUMERICAL_EQUIVALENCE_ETA
GUARD_CLASS_EXACT_DISCRETE = "exact_discrete_primary_contribution"
GUARD_CLASS_FAMILY_B_BRIER = "family_B_panel_brier_difference"
GUARD_CLASS_FAMILY_C_BRIER = "family_C_panel_brier_difference_in_differences"
COMPRESSION_CONTRACT = {
    "format": "zstandard", "level": 19, "threads": 0, "checksum": True,
    "content_size": True, "dict_id": False,
}
CANONICAL_SORT_KEY = [
    "block_id", "policy", "assumed_tox_noise_sd_factor", "gamma", "stratum",
    "seed",
]
BLOCK_DESCRIPTORS = {
    "M0200": {
        "block_id": "M0200", "seed_start": 0, "seed_stop_exclusive": 200,
        "cumulative_M": 200, "path_count": 9_600, "snapshot_count": 28_800,
        "shard_count": 24,
    },
    "M0500_TOPUP": {
        "block_id": "M0500_TOPUP", "seed_start": 200,
        "seed_stop_exclusive": 500, "cumulative_M": 500,
        "path_count": 14_400, "snapshot_count": 43_200, "shard_count": 24,
    },
}
OPERATOR_CAP_FIELDS = {
    "cap_cumulative_M", "operator_cap_reached", "effective_next_M",
    "m1000_executed", "amendment", "original_final_precision_state",
    "precision_target_achieved", "terminated_with_unresolved_precision",
    "terminal_reason", "guarded_mcse_threshold_points",
    "unresolved_estimand_count", "unresolved_guarded_precision_targets",
    "original_precision_artifacts_immutable", "cap_finalizer_sha256",
}
FINAL_PRECISION_FIELDS = {
    "cumulative_M", "next_M", "all_estimands_pass_guarded_1_5pp",
    "top_up_limit_reached", "maximum_raw_mcse_points",
    "maximum_guarded_mcse_points",
}
AMENDMENT_INDEX_FIELDS = {
    "artifact", "artifact_sha256", "metadata", "metadata_sha256", "commit",
    "commit_sha256", "audit_id", "launch_fingerprint",
}
ORIGINAL_MASTER_COMMIT_FIELDS = {
    "schema_version", "status", "artifact", "artifact_sha256",
    "uncompressed_sha256", "metadata", "metadata_sha256", "final_M",
    "path_count", "snapshot_count", "launch_fingerprint", "committed_at_utc",
}
COMMON_MASTER_PAYLOAD_FIELDS = {
    "schema_version", "status", "artifact_class", "storage_model", "final_M",
    "cumulative_block_ids", "path_count", "snapshot_count", "bindings",
    "baseline_identity", "block_commits", "precision_decisions",
    "top_up_history", "canonical_sort_key",
}
CAPPED_MASTER_METADATA_FIELDS = ORIGINAL_MASTER_METADATA_FIELDS | {
    "operator_cap_summary", "amendment_bundle_sha256", "precision_bundle_sha256",
    "cap_finalizer_sha256", "immutable",
}
CAPPED_MASTER_COMMIT_FIELDS = ORIGINAL_MASTER_COMMIT_FIELDS | {
    "amendment_bundle_sha256", "precision_bundle_sha256",
    "cap_finalizer_sha256", "immutable",
}


def _positive_integer(value: Any, *, label: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FinalProvenanceSealError(f"{label} is not a positive integer")
    if maximum is not None and value > maximum:
        raise FinalProvenanceSealError(f"{label} exceeds its authenticated bound")
    return value


def _finite_float(value: Any, *, label: str, nonnegative: bool = True) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise FinalProvenanceSealError(f"{label} is not a finite float")
    if nonnegative and value < 0.0:
        raise FinalProvenanceSealError(f"{label} is negative")
    return value


def _bounded_zstd_payload(
    compressed: bytes,
    *,
    expected_uncompressed_sha256: Any,
    expected_uncompressed_bytes: Any,
    maximum_uncompressed_bytes: int,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    logical_sha = _require_sha(
        expected_uncompressed_sha256, label=f"{label} logical SHA-256"
    )
    logical_bytes = _positive_integer(
        expected_uncompressed_bytes,
        label=f"{label} logical byte count",
        maximum=maximum_uncompressed_bytes,
    )
    try:
        frame_size = zstd.frame_content_size(compressed)
    except zstd.ZstdError as exc:
        raise FinalProvenanceSealError(f"{label} is not a valid zstd frame") from exc
    if frame_size != logical_bytes:
        raise FinalProvenanceSealError(f"{label} zstd content-size commitment changed")
    try:
        logical = zstd.ZstdDecompressor().decompress(
            compressed, max_output_size=logical_bytes
        )
    except zstd.ZstdError as exc:
        raise FinalProvenanceSealError(f"{label} bounded decompression failed") from exc
    if len(logical) != logical_bytes or _sha256_bytes(logical) != logical_sha:
        raise FinalProvenanceSealError(f"{label} logical hash/size binding changed")
    return _decode_canonical_json(logical, label=label), logical


def _validate_master_bindings(
    value: Any, *, profile: TrustedProfile, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalProvenanceSealError(f"{label} bindings are absent")
    bindings = dict(value)
    _exact_fields(bindings, MASTER_BINDING_FIELDS, label=f"{label} bindings")
    expected_implementation = {
        name: profile.source_hashes[name]
        for name in IMPLEMENTATION_BOUND_SOURCE_NAMES
    }
    expected_formal = {
        name: profile.source_hashes[name]
        for name in FROZEN_FORMAL_BOUND_SOURCE_NAMES
    }
    expected_sources = {**expected_formal, **expected_implementation}
    if (
        bindings.get("manifest_sha256")
        != profile.source_hashes["paper/budget_toxicity_calibration_manifest.json"]
        or bindings.get("analysis_spec_sha256")
        != profile.source_hashes["paper/budget_toxicity_calibration_prespec.md"]
        or not _strict_json_equal(
            bindings.get("implementation_source_sha256"), expected_implementation
        )
        or not _strict_json_equal(
            bindings.get("frozen_formal_source_sha256"), expected_formal
        )
        or not _strict_json_equal(bindings.get("source_sha256"), expected_sources)
        or not _strict_json_equal(
            bindings.get("runtime_identity"), FROZEN_RUNTIME_IDENTITY
        )
        or not _strict_json_equal(bindings.get("ray_contract"), RAY_CONTRACT)
    ):
        raise FinalProvenanceSealError(f"{label} frozen execution binding changed")
    _require_sha(bindings.get("design_sha256"), label=f"{label} design SHA-256")
    fingerprint_input = dict(bindings)
    fingerprint = fingerprint_input.pop("launch_fingerprint", None)
    if (
        fingerprint != profile.launch_fingerprint
        or _sha256_bytes(_canonical_json_bytes(fingerprint_input)) != fingerprint
    ):
        raise FinalProvenanceSealError(f"{label} launch fingerprint changed")
    return bindings


def _indexed_path(
    staging_root: Path,
    value: Any,
    *,
    root: Path,
    expected: str,
    label: str,
) -> Path:
    if value != expected:
        raise FinalProvenanceSealError(f"{label} indexed path changed")
    relative = _safe_relative(str(value), label=f"{label} indexed path")
    return _guard_path(
        staging_root / relative, root=root, label=label, kind="file"
    )


def _frozen_shard_descriptors(block_id: str) -> list[dict[str, Any]]:
    """Mirror the frozen runner/monitor shard design without opening shard rows."""

    descriptor = BLOCK_DESCRIPTORS[block_id]
    start = int(descriptor["seed_start"])
    stop = int(descriptor["seed_stop_exclusive"])
    output: list[dict[str, Any]] = []
    for policy in PRECISION_POLICIES:
        for factor in PRECISION_FACTORS:
            for gamma in PRECISION_GATES:
                execution_keys = [
                    [block_id, policy, factor, gamma, stratum, seed]
                    for stratum in PRECISION_STRATA
                    for seed in range(start, stop)
                ]
                shard_id = (
                    f"{block_id}__{policy}__csigma{factor:g}__tau{gamma:g}"
                )
                output.append(
                    {
                        "shard_id": shard_id,
                        "block_id": block_id,
                        "path_count": len(execution_keys),
                        "snapshot_count": 3 * len(execution_keys),
                        "execution_keys_sha256": _sha256_bytes(
                            _canonical_json_bytes(execution_keys)
                        ),
                        "first_execution_key": execution_keys[0],
                        "last_execution_key": execution_keys[-1],
                    }
                )
    return sorted(output, key=lambda row: str(row["shard_id"]))


def _authenticate_block_indexes(
    rows: Any,
    *,
    staging_root: Path,
    root: Path,
    bindings: Mapping[str, Any],
    master_generated_at: datetime,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or any(not isinstance(row, Mapping) for row in rows)
        or [row.get("block_id") for row in rows] != ["M0200", "M0500_TOPUP"]
    ):
        raise FinalProvenanceSealError("master block-index coverage changed")
    commitments: dict[str, dict[str, Any]] = {}
    input_hashes_by_block: dict[str, dict[str, str]] = {}
    for row in rows:
        block_id = str(row["block_id"])
        descriptor = BLOCK_DESCRIPTORS[block_id]
        _exact_fields(row, BLOCK_INDEX_FIELDS, label=f"master block index {block_id}")
        base = f"raw/budget_toxicity_calibration_{block_id}_block_index.json.zst"
        block_expected_paths = {
            "artifact": base,
            "metadata": base + ".metadata.json",
            "commit": base + ".commit.json",
        }
        paths = {
            field: _indexed_path(
                staging_root, row.get(field), root=root, expected=expected,
                label=f"indexed block {block_id} {field}",
            )
            for field, expected in block_expected_paths.items()
        }
        metadata_raw = _read_regular(
            paths["metadata"], root=root, label=f"block {block_id} metadata"
        )
        commit_raw = _read_regular(
            paths["commit"], root=root, label=f"block {block_id} commit"
        )
        compressed = _read_regular(
            paths["artifact"], root=root, label=f"block {block_id} artifact"
        )
        actual_hashes = {
            "artifact_sha256": _sha256_bytes(compressed),
            "metadata_sha256": _sha256_bytes(metadata_raw),
            "commit_sha256": _sha256_bytes(commit_raw),
        }
        for field, expected in actual_hashes.items():
            _require_sha(row.get(field), label=f"block {block_id} {field}")
            if not _strict_json_equal(row.get(field), expected):
                raise FinalProvenanceSealError(
                    f"indexed block {block_id} file hash changed: {field}"
                )
        metadata = _decode_canonical_json(
            metadata_raw, label=f"block {block_id} metadata"
        )
        commit = _decode_canonical_json(commit_raw, label=f"block {block_id} commit")
        _exact_fields(metadata, BLOCK_METADATA_FIELDS, label=f"block {block_id} metadata")
        _exact_fields(commit, BLOCK_COMMIT_FIELDS, label=f"block {block_id} commit")
        for count_field in ("path_count", "snapshot_count", "shard_count"):
            if not _strict_json_equal(row.get(count_field), descriptor[count_field]):
                raise FinalProvenanceSealError(
                    f"block {block_id} index count changed: {count_field}"
                )
        if row.get("uncompressed_sha256") != metadata.get("uncompressed_sha256"):
            raise FinalProvenanceSealError(f"block {block_id} logical hash index changed")
        if row.get("uncompressed_bytes") != metadata.get("uncompressed_bytes"):
            raise FinalProvenanceSealError(f"block {block_id} logical size index changed")
        expected_metadata = {
            "schema_version": 1,
            "status": BLOCK_STATUS,
            "artifact_class": BLOCK_ARTIFACT_CLASS,
            "storage_model": BLOCK_STORAGE_MODEL,
            "artifact": paths["artifact"].name,
            "artifact_sha256": actual_hashes["artifact_sha256"],
            "artifact_bytes": len(compressed),
            "block_id": block_id,
            "seed_start": descriptor["seed_start"],
            "seed_stop_exclusive": descriptor["seed_stop_exclusive"],
            "path_count": descriptor["path_count"],
            "snapshot_count": descriptor["snapshot_count"],
            "shard_count": descriptor["shard_count"],
            "bindings": dict(bindings),
            "compression": COMPRESSION_CONTRACT,
        }
        for field, expected in expected_metadata.items():
            if not _strict_json_equal(metadata.get(field), expected):
                raise FinalProvenanceSealError(
                    f"block {block_id} metadata changed: {field}"
                )
        expected_commit = {
            "schema_version": 1,
            "status": BLOCK_COMMIT_STATUS,
            "artifact": paths["artifact"].name,
            "artifact_sha256": actual_hashes["artifact_sha256"],
            "uncompressed_sha256": metadata.get("uncompressed_sha256"),
            "metadata": paths["metadata"].name,
            "metadata_sha256": actual_hashes["metadata_sha256"],
            "block_id": block_id,
            "path_count": descriptor["path_count"],
            "snapshot_count": descriptor["snapshot_count"],
            "shard_count": descriptor["shard_count"],
            "launch_fingerprint": bindings["launch_fingerprint"],
            "immutable": True,
        }
        for field, expected in expected_commit.items():
            if not _strict_json_equal(commit.get(field), expected):
                raise FinalProvenanceSealError(
                    f"block {block_id} commit changed: {field}"
                )
        generated, committed = _assert_time_order(
            metadata.get("generated_at_utc"), commit.get("committed_at_utc"),
            label=f"block {block_id}",
        )
        if committed > master_generated_at:
            raise FinalProvenanceSealError(
                f"block {block_id} was committed after the master was generated"
            )
        _validate_upstream_environment(metadata.get("environment"), generated_at=generated)
        payload, _logical = _bounded_zstd_payload(
            compressed,
            expected_uncompressed_sha256=metadata.get("uncompressed_sha256"),
            expected_uncompressed_bytes=metadata.get("uncompressed_bytes"),
            maximum_uncompressed_bytes=MAX_BLOCK_INDEX_LOGICAL_BYTES,
            label=f"block {block_id} logical index",
        )
        _exact_fields(payload, BLOCK_PAYLOAD_FIELDS, label=f"block {block_id} payload")
        expected_payload = {
            "schema_version": 1,
            "status": BLOCK_STATUS,
            "artifact_class": BLOCK_ARTIFACT_CLASS,
            "storage_model": BLOCK_STORAGE_MODEL,
            "block": descriptor,
            "bindings": dict(bindings),
            "canonical_sort_key": CANONICAL_SORT_KEY,
            "path_count": descriptor["path_count"],
            "snapshot_count": descriptor["snapshot_count"],
        }
        for field, expected in expected_payload.items():
            if not _strict_json_equal(payload.get(field), expected):
                raise FinalProvenanceSealError(
                    f"block {block_id} logical index changed: {field}"
                )
        shards = payload.get("shard_commits")
        if (
            not isinstance(shards, list)
            or len(shards) != 24
            or any(not isinstance(shard, Mapping) for shard in shards)
        ):
            raise FinalProvenanceSealError(f"block {block_id} shard coverage changed")
        expected_shards = _frozen_shard_descriptors(block_id)
        block_input_hashes: dict[str, str] = {}
        for shard, expected_shard in zip(shards, expected_shards, strict=True):
            _exact_fields(shard, SHARD_INDEX_FIELDS, label=f"block {block_id} shard index")
            for field, expected in expected_shard.items():
                if not _strict_json_equal(shard.get(field), expected):
                    raise FinalProvenanceSealError(
                        f"block {block_id} frozen shard design changed: {field}"
                    )
            identifier = str(expected_shard["shard_id"])
            shard_expected_paths = {
                "artifact": f"raw/shards/{identifier}.paths.json.zst",
                "metadata": (
                    f"raw/shards/{identifier}.paths.json.zst.metadata.json"
                ),
                "commit": f"raw/shards/{identifier}.paths.json.zst.commit.json",
            }
            for field, expected in shard_expected_paths.items():
                if not _strict_json_equal(shard.get(field), expected):
                    raise FinalProvenanceSealError(
                        f"block {block_id} frozen shard path changed: {field}"
                    )
            for field in (
                "artifact_sha256", "uncompressed_sha256", "metadata_sha256",
                "commit_sha256", "execution_keys_sha256",
            ):
                _require_sha(shard.get(field), label=f"block {block_id} shard {field}")
            for field, hash_field in (
                ("artifact", "artifact_sha256"),
                ("metadata", "metadata_sha256"),
                ("commit", "commit_sha256"),
            ):
                _safe_relative(shard[field], label=f"block {block_id} shard {field}")
                block_input_hashes[str(shard[field])] = str(shard[hash_field])
            _positive_integer(shard.get("uncompressed_bytes"), label="shard logical bytes")
            _positive_integer(shard.get("path_count"), label="shard path count")
            _positive_integer(shard.get("snapshot_count"), label="shard snapshot count")
        if (
            sum(int(shard["path_count"]) for shard in shards) != descriptor["path_count"]
            or sum(int(shard["snapshot_count"]) for shard in shards)
            != descriptor["snapshot_count"]
        ):
            raise FinalProvenanceSealError(f"block {block_id} shard counts changed")
        for field, indexed in block_expected_paths.items():
            commitment = _file_commitment(
                paths[field], root=root, label=f"indexed block {block_id} {field}"
            )
            commitments[indexed] = commitment
            block_input_hashes[indexed] = commitment["sha256"]
        if len(block_input_hashes) != 75:
            raise FinalProvenanceSealError(
                f"block {block_id} exact input commitment count changed"
            )
        input_hashes_by_block[block_id] = dict(sorted(block_input_hashes.items()))
    return dict(sorted(commitments.items())), input_hashes_by_block


def _raw_design_input_hashes(
    *,
    profile: TrustedProfile,
    block_input_hashes: Mapping[str, Mapping[str, str]],
    block_ids: Sequence[str],
) -> dict[str, str]:
    output = {
        "paper/budget_toxicity_calibration_manifest.json": profile.source_hashes[
            "paper/budget_toxicity_calibration_manifest.json"
        ],
        "paper/budget_toxicity_calibration_prespec.md": profile.source_hashes[
            "paper/budget_toxicity_calibration_prespec.md"
        ],
    }
    for block_id in block_ids:
        block_hashes = block_input_hashes.get(block_id)
        if not isinstance(block_hashes, Mapping) or len(block_hashes) != 75:
            raise FinalProvenanceSealError(
                f"authenticated raw input map changed for block {block_id}"
            )
        for name, digest in block_hashes.items():
            if name in output:
                raise FinalProvenanceSealError(
                    f"authenticated raw input path is duplicated: {name}"
                )
            output[str(name)] = str(digest)
    expected_count = 2 + 75 * len(block_ids)
    if len(output) != expected_count:
        raise FinalProvenanceSealError("authenticated raw input coverage changed")
    return dict(sorted(output.items()))


def _authenticate_baseline_index(
    value: Any,
    *,
    staging_root: Path,
    root: Path,
    profile: TrustedProfile,
    block_input_hashes: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Authenticate the frozen PASS-only baseline trio and its exact M0200 inputs."""

    if not isinstance(value, Mapping):
        raise FinalProvenanceSealError("master baseline identity index is absent")
    _exact_fields(value, BASELINE_INDEX_FIELDS, label="master baseline identity index")
    expected_shared = {
        "status": BASELINE_STATUS,
        "cumulative_M": 200,
        "pass": True,
        "checked_common_cells": 3_200,
        "checked_allocation_histories": 3_040,
        "unavailable_allocation_histories": 160,
    }
    for field, expected in expected_shared.items():
        if not _strict_json_equal(value.get(field), expected):
            raise FinalProvenanceSealError(
                f"master baseline identity changed: {field}"
            )
    expected_paths = {
        "artifact": BASELINE_RELATIVE,
        "metadata": BASELINE_RELATIVE + ".metadata.json",
        "commit": BASELINE_RELATIVE + ".commit.json",
    }
    paths = {
        field: _indexed_path(
            staging_root,
            value.get(field),
            root=root,
            expected=expected,
            label=f"indexed baseline {field}",
        )
        for field, expected in expected_paths.items()
    }
    _assert_exact_inventory(
        paths["artifact"].parent,
        {path.name for path in paths.values()},
        root=root,
        label="baseline directory",
    )
    compressed = _read_regular(paths["artifact"], root=root, label="baseline artifact")
    metadata_raw = _read_regular(
        paths["metadata"], root=root, label="baseline metadata"
    )
    commit_raw = _read_regular(paths["commit"], root=root, label="baseline commit")
    actual_hashes = {
        "artifact_sha256": _sha256_bytes(compressed),
        "metadata_sha256": _sha256_bytes(metadata_raw),
        "commit_sha256": _sha256_bytes(commit_raw),
    }
    for field, actual in actual_hashes.items():
        _require_sha(value.get(field), label=f"baseline index {field}")
        if value.get(field) != actual:
            raise FinalProvenanceSealError(
                f"baseline index file hash changed: {field}"
            )
    metadata = _decode_canonical_json(metadata_raw, label="baseline metadata")
    commit = _decode_canonical_json(commit_raw, label="baseline commit")
    _exact_fields(metadata, BASELINE_METADATA_FIELDS, label="baseline metadata")
    _exact_fields(commit, BASELINE_COMMIT_FIELDS, label="baseline commit")
    if (
        metadata.get("artifact_sha256") != actual_hashes["artifact_sha256"]
        or metadata.get("artifact_bytes") != len(compressed)
        or commit.get("artifact_sha256") != actual_hashes["artifact_sha256"]
        or commit.get("metadata_sha256") != actual_hashes["metadata_sha256"]
    ):
        raise FinalProvenanceSealError("baseline pre-decompression binding changed")
    payload, logical = _bounded_zstd_payload(
        compressed,
        expected_uncompressed_sha256=metadata.get("uncompressed_sha256"),
        expected_uncompressed_bytes=metadata.get("uncompressed_bytes"),
        maximum_uncompressed_bytes=MAX_BASELINE_LOGICAL_BYTES,
        label="baseline PASS-only payload",
    )
    _exact_fields(payload, BASELINE_PAYLOAD_FIELDS, label="baseline PASS-only payload")
    raw_m0200_hashes = _raw_design_input_hashes(
        profile=profile,
        block_input_hashes=block_input_hashes,
        block_ids=("M0200",),
    )
    if len(raw_m0200_hashes) != 77:
        raise FinalProvenanceSealError("baseline M0200 input coverage changed")
    expected_exception = {
        "policy": "cEI",
        "gamma": 0.7,
        "seed_start": 0,
        "seed_stop_exclusive": 80,
        "strata": [0, 1],
        "count": 160,
    }
    expected_payload = {
        "schema_version": 1,
        "status": BASELINE_STATUS,
        "artifact_class": BASELINE_ARTIFACT_CLASS,
        "cumulative_M": 200,
        "input_hashes": raw_m0200_hashes,
        "oracle_hashes": BASELINE_ORACLE_HASHES,
        "checked_common_cells": 3_200,
        "checked_allocation_histories": 3_040,
        "unavailable_allocation_histories": 160,
        "exception_definition": expected_exception,
        "pass": True,
    }
    if not _strict_json_equal(payload, expected_payload):
        raise FinalProvenanceSealError("baseline PASS-only payload changed")
    expected_metadata = {
        "schema_version": 1,
        "status": BASELINE_STATUS,
        "artifact_class": BASELINE_ARTIFACT_CLASS,
        "artifact": paths["artifact"].name,
        "artifact_sha256": actual_hashes["artifact_sha256"],
        "artifact_bytes": len(compressed),
        "uncompressed_sha256": _sha256_bytes(logical),
        "uncompressed_bytes": len(logical),
        "input_hashes": raw_m0200_hashes,
        "oracle_hashes": BASELINE_ORACLE_HASHES,
        **expected_shared,
    }
    if not _strict_json_equal(metadata, expected_metadata):
        raise FinalProvenanceSealError("baseline PASS-only metadata changed")
    expected_commit = {
        "schema_version": 1,
        "status": BASELINE_COMMIT_STATUS,
        "artifact": paths["artifact"].name,
        "artifact_sha256": actual_hashes["artifact_sha256"],
        "metadata": paths["metadata"].name,
        "metadata_sha256": actual_hashes["metadata_sha256"],
        **{field: expected_shared[field] for field in (
            "cumulative_M", "pass", "checked_common_cells",
            "checked_allocation_histories", "unavailable_allocation_histories",
        )},
    }
    if not _strict_json_equal(commit, expected_commit):
        raise FinalProvenanceSealError("baseline PASS-only commit changed")
    commitments = {
        expected_paths[field]: _file_commitment(
            paths[field], root=root, label=f"indexed baseline {field}"
        )
        for field in ("artifact", "metadata", "commit")
    }
    hashes = {
        expected_paths["artifact"]: actual_hashes["artifact_sha256"],
        expected_paths["metadata"]: actual_hashes["metadata_sha256"],
        expected_paths["commit"]: actual_hashes["commit_sha256"],
    }
    return dict(sorted(commitments.items())), dict(sorted(hashes.items()))


def _precision_factor_label(value: float) -> str:
    return {0.5: "0p5", 1.0: "1", 2.0: "2"}[float(value)]


def _frozen_primary_estimand_ids() -> list[str]:
    output: list[str] = []
    for comparator in PRECISION_COMPARATORS:
        for outcome in (PRECISION_OUTCOME_TERMINAL, PRECISION_OUTCOME_ASSIGNMENT):
            for budget in PRECISION_BUDGETS:
                for factor in PRECISION_FACTORS:
                    output.append(
                        f"A|cKG-minus-{comparator}|{outcome}|N={budget}|"
                        f"c={_precision_factor_label(factor)}"
                    )
    for policy in PRECISION_POLICIES:
        for budget in PRECISION_BUDGETS:
            for factor in PRECISION_ERRONEOUS_FACTORS:
                for outcome in (PRECISION_OUTCOME_EXCLUSION, PRECISION_OUTCOME_BRIER):
                    output.append(
                        f"B|{policy}|{outcome}|N={budget}|"
                        f"c={_precision_factor_label(factor)}-minus-1"
                    )
    for policy in PRECISION_POLICIES:
        for factor in PRECISION_ERRONEOUS_FACTORS:
            for outcome in (PRECISION_OUTCOME_EXCLUSION, PRECISION_OUTCOME_BRIER):
                output.append(
                    f"C|{policy}|{outcome}|N80-minus-N20|"
                    f"c={_precision_factor_label(factor)}-minus-1"
                )
    if len(output) != 118 or len(set(output)) != 118:
        raise FinalProvenanceSealError("internal frozen precision family changed")
    return output


def _frozen_numerical_guard_metadata() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "eta": NUMERICAL_EQUIVALENCE_ETA,
        "eta_formula": "2^-26 = sqrt(binary64_epsilon)",
        "panel_brier_snapshot_bound_points": PANEL_BRIER_SNAPSHOT_BOUND_POINTS,
        "panel_brier_snapshot_bound_formula": "200 * eta",
        "family_B_brier_replicate_bound_points": (
            FAMILY_B_BRIER_REPLICATE_BOUND_POINTS
        ),
        "family_B_brier_replicate_bound_formula": "400 * eta",
        "family_C_brier_interaction_bound_points": (
            FAMILY_C_BRIER_INTERACTION_BOUND_POINTS
        ),
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
                "replicate_bound_points": FAMILY_B_BRIER_REPLICATE_BOUND_POINTS,
            },
            {
                "row_guard_class": GUARD_CLASS_FAMILY_C_BRIER,
                "applies_to": "Family C panel-Brier differences-in-differences",
                "replicate_bound_points": FAMILY_C_BRIER_INTERACTION_BOUND_POINTS,
            },
        ],
    }


def _frozen_row_guard(estimand_id: str, cumulative_m: int) -> dict[str, Any]:
    if (
        estimand_id.startswith("B|")
        and f"|{PRECISION_OUTCOME_BRIER}|" in estimand_id
    ):
        guard_class = GUARD_CLASS_FAMILY_B_BRIER
        replicate_bound = FAMILY_B_BRIER_REPLICATE_BOUND_POINTS
    elif (
        estimand_id.startswith("C|")
        and f"|{PRECISION_OUTCOME_BRIER}|" in estimand_id
    ):
        guard_class = GUARD_CLASS_FAMILY_C_BRIER
        replicate_bound = FAMILY_C_BRIER_INTERACTION_BOUND_POINTS
    else:
        guard_class = GUARD_CLASS_EXACT_DISCRETE
        replicate_bound = 0.0
    return {
        "row_guard_class": guard_class,
        "replicate_bound_points": replicate_bound,
        "mcse_guard_points": replicate_bound / math.sqrt(cumulative_m - 1),
    }


def _validate_precision_payload(
    payload: Mapping[str, Any], *, cumulative_m: int
) -> list[dict[str, Any]]:
    _exact_fields(payload, PRECISION_PAYLOAD_FIELDS, label=f"M{cumulative_m} precision")
    if (
        not _strict_json_equal(payload.get("schema_version"), 2)
        or payload.get("status") != PRECISION_STATUS
        or payload.get("artifact_class") != PRECISION_ARTIFACT_CLASS
        or not _strict_json_equal(payload.get("cumulative_M"), cumulative_m)
        or not _strict_json_equal(
            payload.get("numerical_equivalence_guard"),
            _frozen_numerical_guard_metadata(),
        )
    ):
        raise FinalProvenanceSealError(f"M{cumulative_m} blind precision identity changed")
    estimands = payload.get("estimands")
    if (
        not isinstance(estimands, list)
        or len(estimands) != 118
        or any(not isinstance(row, Mapping) for row in estimands)
    ):
        raise FinalProvenanceSealError(f"M{cumulative_m} precision family changed")
    expected_identifiers = _frozen_primary_estimand_ids()
    observed_identifiers = [row.get("estimand_id") for row in estimands]
    if observed_identifiers != expected_identifiers:
        raise FinalProvenanceSealError(
            f"M{cumulative_m} frozen precision estimand order/family changed"
        )
    failures: list[dict[str, Any]] = []
    for row, identifier in zip(estimands, expected_identifiers, strict=True):
        _exact_fields(row, PRECISION_ROW_FIELDS, label=f"M{cumulative_m} precision row")
        if not _strict_json_equal(row.get("cumulative_M"), cumulative_m):
            raise FinalProvenanceSealError("blind precision row M changed")
        expected_guard = _frozen_row_guard(identifier, cumulative_m)
        for field, expected in expected_guard.items():
            if not _strict_json_equal(row.get(field), expected):
                raise FinalProvenanceSealError(
                    f"blind precision frozen row guard changed: {field}"
                )
        raw_mcse = _finite_float(row.get("raw_mcse_points"), label="raw MCSE")
        guard = float(expected_guard["mcse_guard_points"])
        guarded = _finite_float(row.get("guarded_mcse_points"), label="guarded MCSE")
        passed = row.get("passes_guarded_1_5pp")
        if (
            type(passed) is not bool
            or guarded != raw_mcse + guard
            or passed is not (guarded <= 1.5)
        ):
            raise FinalProvenanceSealError("blind precision guarded decision changed")
        if not passed:
            failures.append(dict(row))
    maximum_raw = _finite_float(
        payload.get("maximum_raw_mcse_points"), label="maximum raw MCSE"
    )
    maximum_guarded = _finite_float(
        payload.get("maximum_guarded_mcse_points"), label="maximum guarded MCSE"
    )
    all_pass = payload.get("all_estimands_pass_guarded_1_5pp")
    if (
        maximum_raw != max(float(row["raw_mcse_points"]) for row in estimands)
        or maximum_guarded
        != max(float(row["guarded_mcse_points"]) for row in estimands)
        or type(all_pass) is not bool
        or all_pass is not (len(failures) == 0)
        or payload.get("top_up_limit_reached") is not False
    ):
        raise FinalProvenanceSealError("blind precision aggregate decision changed")
    expected_next = (
        500 if cumulative_m == 200 and not all_pass
        else 1000 if cumulative_m == 500 and not all_pass
        else None
    )
    if not _strict_json_equal(payload.get("next_M"), expected_next):
        raise FinalProvenanceSealError("blind precision next-M rule changed")
    if cumulative_m == 200 and (all_pass is not False or expected_next != 500):
        raise FinalProvenanceSealError("M0200 does not justify the authenticated M0500")
    input_hashes = payload.get("input_hashes")
    expected_count = 80 if cumulative_m == 200 else 155
    if not isinstance(input_hashes, Mapping) or len(input_hashes) != expected_count:
        raise FinalProvenanceSealError(
            f"M{cumulative_m} precision input-hash coverage changed"
        )
    for name, digest in input_hashes.items():
        if not isinstance(name, str):
            raise FinalProvenanceSealError("precision input-hash name is malformed")
        _safe_relative(name, label="precision input-hash path")
        _require_sha(digest, label=f"precision input hash {name}")
    return failures


def _authenticate_precision_indexes(
    rows: Any,
    *,
    staging_root: Path,
    root: Path,
    profile: TrustedProfile,
    block_input_hashes: Mapping[str, Mapping[str, str]],
    baseline_input_hashes: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or any(not isinstance(row, Mapping) for row in rows)
        or [row.get("cumulative_M") for row in rows] != [200, 500]
    ):
        raise FinalProvenanceSealError("master precision-index coverage changed")
    precision_directory = _guard_path(
        staging_root / "precision", root=root, label="precision directory", kind="dir"
    )
    expected_inventory = {
        suffix
        for cumulative_m in (200, 500)
        for suffix in (
            f"budget_toxicity_calibration_precision_M{cumulative_m:04d}.json.zst",
            f"budget_toxicity_calibration_precision_M{cumulative_m:04d}.json.zst.metadata.json",
            f"budget_toxicity_calibration_precision_M{cumulative_m:04d}.json.zst.commit.json",
        )
    }
    _assert_exact_inventory(
        precision_directory, expected_inventory, root=root, label="precision directory"
    )
    payloads: list[dict[str, Any]] = []
    commitments: dict[str, dict[str, Any]] = {}
    for row in rows:
        cumulative_m = int(row["cumulative_M"])
        _exact_fields(row, PRECISION_INDEX_FIELDS, label=f"M{cumulative_m} precision index")
        base = f"precision/budget_toxicity_calibration_precision_M{cumulative_m:04d}.json.zst"
        expected_paths = {
            "artifact": base,
            "metadata": base + ".metadata.json",
            "commit": base + ".commit.json",
        }
        paths = {
            field: _indexed_path(
                staging_root, row.get(field), root=root, expected=expected,
                label=f"indexed M{cumulative_m} precision {field}",
            )
            for field, expected in expected_paths.items()
        }
        metadata_raw = _read_regular(
            paths["metadata"], root=root, label=f"M{cumulative_m} precision metadata"
        )
        commit_raw = _read_regular(
            paths["commit"], root=root, label=f"M{cumulative_m} precision commit"
        )
        compressed = _read_regular(
            paths["artifact"], root=root, label=f"M{cumulative_m} precision artifact"
        )
        actual_hashes = {
            "artifact_sha256": _sha256_bytes(compressed),
            "metadata_sha256": _sha256_bytes(metadata_raw),
            "commit_sha256": _sha256_bytes(commit_raw),
        }
        for field, actual in actual_hashes.items():
            _require_sha(row.get(field), label=f"M{cumulative_m} {field}")
            if row.get(field) != actual:
                raise FinalProvenanceSealError(
                    f"indexed M{cumulative_m} precision hash changed: {field}"
                )
        metadata = _decode_canonical_json(
            metadata_raw, label=f"M{cumulative_m} precision metadata"
        )
        commit = _decode_canonical_json(
            commit_raw, label=f"M{cumulative_m} precision commit"
        )
        _exact_fields(metadata, PRECISION_METADATA_FIELDS, label="precision metadata")
        _exact_fields(commit, PRECISION_COMMIT_FIELDS, label="precision commit")
        if (
            not _strict_json_equal(metadata.get("schema_version"), 2)
            or metadata.get("status") != PRECISION_STATUS
            or metadata.get("artifact_class") != PRECISION_ARTIFACT_CLASS
            or metadata.get("artifact") != paths["artifact"].name
            or metadata.get("artifact_sha256") != actual_hashes["artifact_sha256"]
            or metadata.get("artifact_bytes") != len(compressed)
            or not _strict_json_equal(metadata.get("cumulative_M"), cumulative_m)
            or not _strict_json_equal(metadata.get("estimand_count"), 118)
        ):
            raise FinalProvenanceSealError(f"M{cumulative_m} precision metadata changed")
        if (
            not _strict_json_equal(commit.get("schema_version"), 2)
            or commit.get("status") != PRECISION_COMMIT_STATUS
            or commit.get("artifact") != paths["artifact"].name
            or commit.get("artifact_sha256") != actual_hashes["artifact_sha256"]
            or commit.get("metadata") != paths["metadata"].name
            or commit.get("metadata_sha256") != actual_hashes["metadata_sha256"]
        ):
            raise FinalProvenanceSealError(f"M{cumulative_m} precision commit changed")
        payload, logical = _bounded_zstd_payload(
            compressed,
            expected_uncompressed_sha256=metadata.get("uncompressed_sha256"),
            expected_uncompressed_bytes=metadata.get("uncompressed_bytes"),
            maximum_uncompressed_bytes=MAX_PRECISION_LOGICAL_BYTES,
            label=f"M{cumulative_m} blind precision decision",
        )
        failures = _validate_precision_payload(payload, cumulative_m=cumulative_m)
        del failures
        expected_metadata_fields = {
            "numerical_equivalence_guard": payload["numerical_equivalence_guard"],
            "maximum_raw_mcse_points": payload["maximum_raw_mcse_points"],
            "maximum_guarded_mcse_points": payload["maximum_guarded_mcse_points"],
            "all_estimands_pass_guarded_1_5pp": payload[
                "all_estimands_pass_guarded_1_5pp"
            ],
            "top_up_limit_reached": payload["top_up_limit_reached"],
            "next_M": payload["next_M"],
            "input_hashes": payload["input_hashes"],
            "uncompressed_sha256": _sha256_bytes(logical),
            "uncompressed_bytes": len(logical),
        }
        for field, expected in expected_metadata_fields.items():
            if not _strict_json_equal(metadata.get(field), expected):
                raise FinalProvenanceSealError(
                    f"M{cumulative_m} precision metadata differs from payload: {field}"
                )
        for field in (
            "cumulative_M", "estimand_count", "maximum_guarded_mcse_points",
            "all_estimands_pass_guarded_1_5pp", "top_up_limit_reached", "next_M",
        ):
            if not _strict_json_equal(commit.get(field), metadata.get(field)):
                raise FinalProvenanceSealError(
                    f"M{cumulative_m} precision commit differs: {field}"
                )
        if row.get("next_M") != payload["next_M"]:
            raise FinalProvenanceSealError(
                f"M{cumulative_m} master index contradicts the precision decision"
            )
        required_blocks = ("M0200",) if cumulative_m == 200 else (
            "M0200", "M0500_TOPUP"
        )
        expected_input_hashes = _raw_design_input_hashes(
            profile=profile,
            block_input_hashes=block_input_hashes,
            block_ids=required_blocks,
        )
        for name, digest in baseline_input_hashes.items():
            if name in expected_input_hashes:
                raise FinalProvenanceSealError(
                    f"precision input path is duplicated by baseline: {name}"
                )
            expected_input_hashes[str(name)] = str(digest)
        expected_input_hashes = dict(sorted(expected_input_hashes.items()))
        expected_count = 80 if cumulative_m == 200 else 155
        if (
            len(expected_input_hashes) != expected_count
            or payload["input_hashes"] != expected_input_hashes
        ):
            raise FinalProvenanceSealError(
                f"M{cumulative_m} precision input hash map changed"
            )
        for field, indexed in expected_paths.items():
            commitments[indexed] = _file_commitment(
                paths[field], root=root, label=f"indexed M{cumulative_m} precision {field}"
            )
        payloads.append(payload)
    return payloads, dict(sorted(commitments.items()))


def _authenticate_master(
    path: str | Path,
    *,
    root: Path,
    profile: TrustedProfile,
    amendment: Mapping[str, Any],
    trusted_commit_sha256: str,
) -> dict[str, Any]:
    trusted_commit = _require_sha(
        trusted_commit_sha256, label="externally trusted master commit hash"
    )
    artifact = _guard_path(path, root=root, label="master artifact", kind="file")
    if artifact.parent.name != "raw":
        raise FinalProvenanceSealError("master artifact is not in the staging raw directory")
    staging_root = _guard_path(
        artifact.parent.parent, root=root, label="master staging root", kind="dir"
    )
    metadata_path = Path(str(artifact) + ".metadata.json")
    commit_path = Path(str(artifact) + ".commit.json")
    metadata_raw = _read_regular(metadata_path, root=root, label="master metadata")
    commit_raw = _read_regular(commit_path, root=root, label="master commit")
    if _sha256_bytes(commit_raw) != trusted_commit:
        raise FinalProvenanceSealError(
            "master commit differs from externally trusted SHA-256"
        )
    metadata = _decode_canonical_json(metadata_raw, label="master metadata")
    commit = _decode_canonical_json(commit_raw, label="master commit")
    status = metadata.get("status")
    if status == ORIGINAL_MASTER_STATUS:
        branch = PASS_BRANCH
        expected_name = "budget_toxicity_calibration_raw_master.json.zst"
        expected_metadata_fields = ORIGINAL_MASTER_METADATA_FIELDS
        expected_commit_fields = ORIGINAL_MASTER_COMMIT_FIELDS
        expected_payload_fields = COMMON_MASTER_PAYLOAD_FIELDS
        expected_commit_status = ORIGINAL_MASTER_COMMIT_STATUS
        expected_class = ORIGINAL_MASTER_CLASS
        expected_storage = ORIGINAL_MASTER_STORAGE
    elif status == CAPPED_MASTER_STATUS:
        branch = CAPPED_BRANCH
        expected_name = (
            "budget_toxicity_calibration_raw_master_operator_capped_M0500.json.zst"
        )
        expected_metadata_fields = CAPPED_MASTER_METADATA_FIELDS
        expected_commit_fields = CAPPED_MASTER_COMMIT_FIELDS
        expected_payload_fields = COMMON_MASTER_PAYLOAD_FIELDS | {"operator_cap"}
        expected_commit_status = CAPPED_MASTER_COMMIT_STATUS
        expected_class = CAPPED_MASTER_CLASS
        expected_storage = CAPPED_MASTER_STORAGE
    else:
        raise FinalProvenanceSealError("master branch/status is not eligible")
    if artifact.name != expected_name:
        raise FinalProvenanceSealError("master artifact name changed")
    _exact_fields(metadata, expected_metadata_fields, label="master metadata")
    _exact_fields(commit, expected_commit_fields, label="master commit")
    compressed = _read_regular(artifact, root=root, label="master artifact")
    compressed_sha = _sha256_bytes(compressed)
    if (
        metadata.get("artifact") != artifact.name
        or metadata.get("artifact_sha256") != compressed_sha
        or metadata.get("artifact_bytes") != len(compressed)
        or not _strict_json_equal(metadata.get("compression"), COMPRESSION_CONTRACT)
        or not _strict_json_equal(commit.get("schema_version"), 1)
        or commit.get("status") != expected_commit_status
        or commit.get("artifact") != artifact.name
        or commit.get("artifact_sha256") != compressed_sha
        or commit.get("uncompressed_sha256") != metadata.get("uncompressed_sha256")
        or commit.get("metadata") != metadata_path.name
        or commit.get("metadata_sha256") != _sha256_bytes(metadata_raw)
    ):
        raise FinalProvenanceSealError("master pre-decompression envelope binding changed")
    payload, logical = _bounded_zstd_payload(
        compressed,
        expected_uncompressed_sha256=metadata.get("uncompressed_sha256"),
        expected_uncompressed_bytes=metadata.get("uncompressed_bytes"),
        maximum_uncompressed_bytes=MAX_MASTER_LOGICAL_BYTES,
        label="master logical index",
    )
    _exact_fields(payload, expected_payload_fields, label="master payload")
    if "rows" in payload:
        raise FinalProvenanceSealError("master index unexpectedly contains raw rows")
    for envelope, label in ((payload, "payload"), (metadata, "metadata")):
        if (
            not _strict_json_equal(envelope.get("schema_version"), 1)
            or envelope.get("status") != status
            or envelope.get("artifact_class") != expected_class
            or envelope.get("storage_model") != expected_storage
        ):
            raise FinalProvenanceSealError(f"master {label} identity changed")
    for envelope, label in ((payload, "payload"), (metadata, "metadata"), (commit, "commit")):
        if (
            not _strict_json_equal(envelope.get("final_M"), 500)
            or not _strict_json_equal(envelope.get("path_count"), 24_000)
            or not _strict_json_equal(envelope.get("snapshot_count"), 72_000)
        ):
            raise FinalProvenanceSealError(f"master {label} M/count identity changed")
    if (
        not _strict_json_equal(
            payload.get("cumulative_block_ids"), ["M0200", "M0500_TOPUP"]
        )
        or not _strict_json_equal(
            payload.get("canonical_sort_key"), CANONICAL_SORT_KEY
        )
    ):
        raise FinalProvenanceSealError("master index identity changed")
    bindings = _validate_master_bindings(
        payload.get("bindings"), profile=profile, label="master"
    )
    if (
        not _strict_json_equal(metadata.get("bindings"), bindings)
        or commit.get("launch_fingerprint") != profile.launch_fingerprint
    ):
        raise FinalProvenanceSealError("master envelope bindings differ")
    generated, _committed = _assert_time_order(
        metadata.get("generated_at_utc"), commit.get("committed_at_utc"),
        label="master envelope",
    )
    _validate_upstream_environment(metadata.get("environment"), generated_at=generated)
    if _timestamp(
        amendment.get("committed_at_utc"), label="amendment committed timestamp"
    ) > generated:
        raise FinalProvenanceSealError("amendment v2 was committed after the master")
    indexed_blocks, block_input_hashes = _authenticate_block_indexes(
        payload.get("block_commits"), staging_root=staging_root, root=root,
        bindings=bindings, master_generated_at=generated,
    )
    indexed_baseline, baseline_input_hashes = _authenticate_baseline_index(
        payload.get("baseline_identity"),
        staging_root=staging_root,
        root=root,
        profile=profile,
        block_input_hashes=block_input_hashes,
    )
    precision_payloads, indexed_precision = _authenticate_precision_indexes(
        payload.get("precision_decisions"), staging_root=staging_root, root=root,
        profile=profile,
        block_input_hashes=block_input_hashes,
        baseline_input_hashes=baseline_input_hashes,
    )
    expected_raw_names = {
        artifact.name, metadata_path.name, commit_path.name,
        *[Path(name).name for name in indexed_blocks],
    }
    _assert_master_raw_inventory(
        artifact.parent, expected_raw_names, root=root
    )
    decisions = payload["precision_decisions"]
    expected_history = [
        {
            "cumulative_M": row["cumulative_M"],
            "precision_artifact": row["artifact"],
            "precision_artifact_sha256": row["artifact_sha256"],
            "next_M": row["next_M"],
        }
        for row in decisions
    ]
    if not _strict_json_equal(payload.get("top_up_history"), expected_history):
        raise FinalProvenanceSealError("master precision history changed")
    final_precision = precision_payloads[-1]
    original_all_pass = final_precision["all_estimands_pass_guarded_1_5pp"]
    original_next = final_precision["next_M"]
    failures = [
        dict(row) for row in final_precision["estimands"]
        if row["passes_guarded_1_5pp"] is False
    ]
    cap_record: Mapping[str, Any] | None = None
    if branch == PASS_BRANCH:
        if original_all_pass is not True or original_next is not None or "operator_cap" in payload:
            raise FinalProvenanceSealError(
                "ordinary master contradicts the authenticated M0500 precision decision"
            )
        precision_state = {
            "original_frozen_decision": {
                "cumulative_M": 500,
                "all_estimands_pass_guarded_1_5pp": True,
                "next_M": None,
            },
            "effective_terminal_state": {
                "operator_cap_applied": False,
                "effective_next_M": None,
                "precision_target_achieved": True,
                "terminated_with_unresolved_precision": False,
                "unresolved_estimand_count": 0,
            },
        }
    else:
        if original_all_pass is not False or original_next != 1000:
            raise FinalProvenanceSealError(
                "capped master contradicts the authenticated M0500 precision decision"
            )
        cap_record = payload.get("operator_cap")
        if not isinstance(cap_record, Mapping):
            raise FinalProvenanceSealError("capped master lacks operator-cap record")
        _exact_fields(cap_record, OPERATOR_CAP_FIELDS, label="operator-cap record")
        amendment_index = cap_record.get("amendment")
        original = cap_record.get("original_final_precision_state")
        if not isinstance(amendment_index, Mapping) or not isinstance(original, Mapping):
            raise FinalProvenanceSealError("capped master terminal state is malformed")
        _exact_fields(amendment_index, AMENDMENT_INDEX_FIELDS, label="operator-cap amendment")
        _exact_fields(original, FINAL_PRECISION_FIELDS, label="original M0500 precision")
        expected_amendment = {
            "artifact": amendment["envelope"]["artifact"]["path"],
            "artifact_sha256": amendment["envelope"]["artifact"]["sha256"],
            "metadata": amendment["envelope"]["metadata"]["path"],
            "metadata_sha256": amendment["envelope"]["metadata"]["sha256"],
            "commit": amendment["envelope"]["commit"]["path"],
            "commit_sha256": amendment["envelope"]["commit"]["sha256"],
            "audit_id": amendment["audit_id"],
            "launch_fingerprint": profile.launch_fingerprint,
        }
        expected_original = {
            "cumulative_M": 500,
            "next_M": 1000,
            "all_estimands_pass_guarded_1_5pp": False,
            "top_up_limit_reached": False,
            "maximum_raw_mcse_points": final_precision["maximum_raw_mcse_points"],
            "maximum_guarded_mcse_points": final_precision[
                "maximum_guarded_mcse_points"
            ],
        }
        if (
            not _strict_json_equal(amendment_index, expected_amendment)
            or not _strict_json_equal(original, expected_original)
        ):
            raise FinalProvenanceSealError("capped master amendment/precision binding changed")
        expected_cap_values = {
            "cap_cumulative_M": 500,
            "operator_cap_reached": True,
            "effective_next_M": None,
            "m1000_executed": False,
            "precision_target_achieved": False,
            "terminated_with_unresolved_precision": True,
            "terminal_reason": "OPERATOR_CAP_WITH_UNRESOLVED_PRECISION",
            "guarded_mcse_threshold_points": 1.5,
            "unresolved_estimand_count": len(failures),
            "unresolved_guarded_precision_targets": failures,
            "original_precision_artifacts_immutable": True,
            "cap_finalizer_sha256": profile.source_hashes[
                "paper/finalize_budget_toxicity_calibration_operator_capped_master.py"
            ],
        }
        for field, expected in expected_cap_values.items():
            if not _strict_json_equal(cap_record.get(field), expected):
                raise FinalProvenanceSealError(
                    f"capped master branch/effective state changed: {field}"
                )
        if not 1 <= len(failures) <= 118:
            raise FinalProvenanceSealError("capped master has no unresolved precision target")
        amendment_bundle = _sha256_bytes(_canonical_json_bytes(dict(amendment_index)))
        precision_bundle = _sha256_bytes(_canonical_json_bytes(decisions))
        summary = {
            "precision_target_achieved": False,
            "terminated_with_unresolved_precision": True,
            "terminal_reason": "OPERATOR_CAP_WITH_UNRESOLVED_PRECISION",
            "unresolved_estimand_count": len(failures),
            "original_next_M": 1000,
            "effective_next_M": None,
        }
        for envelope in (metadata, commit):
            _require_sha(
                envelope.get("amendment_bundle_sha256"),
                label="capped master amendment bundle",
            )
            _require_sha(
                envelope.get("precision_bundle_sha256"),
                label="capped master precision bundle",
            )
            if (
                envelope.get("amendment_bundle_sha256") != amendment_bundle
                or envelope.get("precision_bundle_sha256") != precision_bundle
                or envelope.get("cap_finalizer_sha256")
                != expected_cap_values["cap_finalizer_sha256"]
                or envelope.get("immutable") is not True
            ):
                raise FinalProvenanceSealError("capped master wrapper commitment changed")
        if not _strict_json_equal(metadata.get("operator_cap_summary"), summary):
            raise FinalProvenanceSealError("capped master metadata summary changed")
        precision_state = {
            "original_frozen_decision": {
                "cumulative_M": 500,
                "all_estimands_pass_guarded_1_5pp": False,
                "next_M": 1000,
            },
            "effective_terminal_state": {
                "operator_cap_applied": True,
                "effective_next_M": None,
                "precision_target_achieved": False,
                "terminated_with_unresolved_precision": True,
                "unresolved_estimand_count": len(failures),
            },
        }
    envelope = {
        "artifact": _file_commitment(artifact, root=root, label="master artifact"),
        "metadata": _file_commitment(metadata_path, root=root, label="master metadata"),
        "commit": _file_commitment(commit_path, root=root, label="master commit"),
    }
    return {
        "branch": branch,
        "precision_state": precision_state,
        "precision_payloads": precision_payloads,
        "cap_record": dict(cap_record) if isinstance(cap_record, Mapping) else None,
        "payload": payload,
        "metadata": metadata,
        "commit": commit,
        "envelope": {
            "envelope": envelope,
            "indexed_block_files": indexed_blocks,
            "indexed_baseline_files": indexed_baseline,
            "indexed_precision_files": indexed_precision,
            "trusted_commit_sha256": trusted_commit,
            "uncompressed_sha256": _sha256_bytes(logical),
            "uncompressed_bytes": len(logical),
            "status": status,
            "commit_status": expected_commit_status,
            "artifact_class": expected_class,
            "storage_model": expected_storage,
        },
    }


ANALYSIS_METADATA_FIELDS = {
    "schema_version", "status", "artifact_class", "analyzer_sha256",
    "prespec_sha256", "manifest_sha256", "raw_provenance",
    "baseline_identity", "primary_estimand_count", "numerical_equivalence_guard",
    "secondary_numerical_equivalence_guard", "scientific_sign_source",
    "secondary_scientific_sign_source", "secondary_absolute_row_count",
    "secondary_contrast_row_count", "artifacts",
}
ANALYSIS_COMMIT_FIELDS = {
    "schema_version", "status", "metadata", "metadata_sha256",
    "artifact_count", "baseline_identity_passed",
}
ANALYSIS_PRECISION_FIELDS = {
    "cumulative_M", "maximum_raw_mcse_points", "maximum_guarded_mcse_points",
    "all_estimands_pass_guarded_1_5pp", "top_up_limit_reached",
    "numerical_equivalence_guard", "next_M", "input_hashes",
}


def _frozen_secondary_numerical_guard_metadata() -> dict[str, Any]:
    """Pure mirror of the frozen analyzer's secondary guard contract."""

    return {
        "schema_version": 1,
        "eta": NUMERICAL_EQUIVALENCE_ETA,
        "eta_expression": "2^-26",
        "unit": "percentage points for panel_brier_score_pct; zero otherwise",
        "row_guard_classes": {
            "secondary_non_brier_zero_guard": {
                "applies_to": "every non-Brier secondary row",
                "replicate_numerical_bound_points": 0.0,
                "bound_expression": "0",
            },
            "secondary_absolute_panel_brier_snapshot": {
                "applies_to": "absolute_equal_gate_stratum_profile",
                "replicate_numerical_bound_points": (
                    PANEL_BRIER_SNAPSHOT_BOUND_POINTS
                ),
                "bound_expression": "delta0 = 200*eta",
            },
            "secondary_within_policy_N80_minus_N20_panel_brier_contrast": {
                "applies_to": "absolute_budget_contrast_N80_minus_N20",
                "replicate_numerical_bound_points": (
                    2.0 * PANEL_BRIER_SNAPSHOT_BOUND_POINTS
                ),
                "bound_expression": "2*delta0 = 400*eta",
            },
            "secondary_policy_vs_cEI_panel_brier_calibration_did": {
                "applies_to": (
                    "policy_minus_cEI_calibration_difference_in_differences"
                ),
                "replicate_numerical_bound_points": (
                    4.0 * PANEL_BRIER_SNAPSHOT_BOUND_POINTS
                ),
                "bound_expression": "4*delta0 = 800*eta",
            },
        },
        "mcse_numerical_guard_formula": "a = b/sqrt(M-1)",
        "pointwise_numerical_halfwidth_guard_formula": "h = b + t_crit*a",
        "guarded_pointwise_interval_formula": (
            "guarded_low = raw_ci95_low - h; "
            "guarded_high = raw_ci95_high + h"
        ),
        "raw_pointwise_interval_fields": ["ci95_low", "ci95_high"],
        "guarded_pointwise_interval_fields": [
            "guarded_ci95_low", "guarded_ci95_high",
        ],
        "guarded_scientific_sign_rule": (
            "positive iff guarded_low > 0; negative iff guarded_high < 0; "
            "otherwise unresolved, including an endpoint exactly equal to zero"
        ),
    }


def _frozen_analysis_baseline_identity() -> dict[str, Any]:
    """Pure PASS/count-only core returned by the frozen baseline audit."""

    return {
        "status": "PASS_AVAILABLE_FIELD_IDENTITY",
        "common_field_cells": 3_200,
        # BASELINE_COMMON_FIELDS is the 42-field union over the heterogeneous
        # historical serializers.  The frozen analyzer increments this count
        # only for fields actually serialized in each of the 3,200 cells, so
        # the production total is not the Cartesian product 3,200 * 42.
        "locked_common_fields_per_cell": 42,
        "common_field_comparisons": 100_800,
        "allocation_history_cells": 3_040,
        "historical_cells_without_allocation_history": 160,
        "allocation_history_exception": {
            "policy": "cEI",
            "gamma": 0.7,
            "seed_start": 0,
            "seed_stop_exclusive": 80,
            "strata": [0, 1],
        },
        "construction_prefix_identity": (
            "validated_by_frozen_core_test_contract"
        ),
    }


def _frozen_analysis_streaming_audit() -> dict[str, Any]:
    """Pure M0500 streaming/CRN record emitted by either frozen analyzer."""

    return {
        "streaming_storage": "one_authenticated_shard_at_a_time",
        "raw_records_accumulated": False,
        "physical_paths_validated": 24_000,
        "logical_snapshots_recomputed": 72_000,
        "common_random_number_seeds": 500,
        "innovation_hash_differences_across_factorial": 0,
        "truth_eff_noise_sd": 7.68,
        "truth_tox_noise_sd": 1.29,
    }


def _frozen_comparator_extension_bindings(
    profile: TrustedProfile,
) -> dict[str, Any]:
    """Pure source/runtime binding emitted by the frozen extension runner."""

    source_hashes = {
        name: profile.source_hashes[name]
        for name in FROZEN_FORMAL_BOUND_SOURCE_NAMES
    }
    source_hashes["paper/run_comparator_family_completion.py"] = (
        COMPARATOR_EXTENSION_RUNNER_SHA256
    )
    bindings: dict[str, Any] = {
        "completion_manifest_sha256": (
            COMPARATOR_EXTENSION_COMPLETION_MANIFEST_SHA256
        ),
        "analysis_spec_sha256": COMPARATOR_EXTENSION_SPEC_SHA256,
        "formal_manifest_sha256": profile.source_hashes[
            "paper/formal_rerun_manifest.json"
        ],
        "source_sha256": dict(sorted(source_hashes.items())),
        "runtime_identity": dict(COMPARATOR_EXTENSION_RUNTIME_IDENTITY),
        "ray_contract": dict(COMPARATOR_EXTENSION_RAY_CONTRACT),
    }
    bindings["launch_fingerprint"] = _sha256_bytes(
        _canonical_json_bytes(bindings)
    )
    return bindings


def _validate_analysis_baseline_identity(
    value: Any, *, profile: TrustedProfile
) -> None:
    """Validate the exact outcome-free production baseline metadata shape."""

    if not isinstance(value, Mapping):
        raise FinalProvenanceSealError("analysis baseline identity is absent")
    core = _frozen_analysis_baseline_identity()
    expected_fields = set(core) | {
        "streaming_and_crn_audit", "oracle_provenance"
    }
    _exact_fields(value, expected_fields, label="analysis baseline identity")
    for field, expected in core.items():
        if not _strict_json_equal(value.get(field), expected):
            raise FinalProvenanceSealError(
                f"analysis baseline PASS core changed: {field}"
            )

    streaming = value.get("streaming_and_crn_audit")
    if not _strict_json_equal(streaming, _frozen_analysis_streaming_audit()):
        raise FinalProvenanceSealError(
            "analysis baseline streaming/CRN audit changed"
        )

    provenance = value.get("oracle_provenance")
    if not isinstance(provenance, Mapping):
        raise FinalProvenanceSealError(
            "analysis baseline oracle provenance is absent"
        )
    _exact_fields(
        provenance,
        {"selected_family", "formal_decision_master", "comparator_extension"},
        label="analysis baseline oracle provenance",
    )

    selected = provenance.get("selected_family")
    expected_selected = {
        "artifact_sha256": BASELINE_ORACLE_HASHES[
            "selected_comparator_family.artifact_sha256"
        ],
        "metadata_sha256": BASELINE_ORACLE_HASHES[
            "selected_comparator_family.metadata_sha256"
        ],
    }
    if not _strict_json_equal(selected, expected_selected):
        raise FinalProvenanceSealError(
            "analysis selected-family oracle provenance changed"
        )

    formal = provenance.get("formal_decision_master")
    if not isinstance(formal, Mapping):
        raise FinalProvenanceSealError(
            "analysis formal-master oracle provenance is absent"
        )
    _exact_fields(
        formal,
        {"compressed_sha256", "uncompressed_sha256", "metadata_sha256"},
        label="analysis formal-master oracle provenance",
    )
    if (
        formal.get("compressed_sha256")
        != BASELINE_ORACLE_HASHES[
            "formal_decision_master.artifact_sha256"
        ]
        or formal.get("metadata_sha256")
        != BASELINE_ORACLE_HASHES[
            "formal_decision_master.metadata_sha256"
        ]
    ):
        raise FinalProvenanceSealError(
            "analysis formal-master oracle commitments changed"
        )
    _require_sha(
        formal.get("uncompressed_sha256"),
        label="analysis formal-master logical hash",
    )

    extension = provenance.get("comparator_extension")
    if not isinstance(extension, Mapping):
        raise FinalProvenanceSealError(
            "analysis comparator-extension oracle provenance is absent"
        )
    extension_fields = {
        "artifact", "artifact_sha256", "metadata", "metadata_sha256",
        "commit", "commit_sha256", "uncompressed_sha256", "row_count",
        "launch_fingerprint", "bindings",
    }
    _exact_fields(
        extension, extension_fields,
        label="analysis comparator-extension oracle provenance",
    )
    expected_extension_scalars = {
        "artifact": "comparator_family_completion_records.json.zst",
        "artifact_sha256": BASELINE_ORACLE_HASHES[
            "comparator_extension.artifact_sha256"
        ],
        "metadata": (
            "comparator_family_completion_records.json.zst.metadata.json"
        ),
        "metadata_sha256": BASELINE_ORACLE_HASHES[
            "comparator_extension.metadata_sha256"
        ],
        "commit": "comparator_family_completion_records.json.zst.commit.json",
        "commit_sha256": BASELINE_ORACLE_HASHES[
            "comparator_extension.commit_sha256"
        ],
        "row_count": 8_800,
    }
    for field, expected in expected_extension_scalars.items():
        if not _strict_json_equal(extension.get(field), expected):
            raise FinalProvenanceSealError(
                f"analysis comparator-extension commitment changed: {field}"
            )
    _require_sha(
        extension.get("uncompressed_sha256"),
        label="analysis comparator-extension logical hash",
    )
    expected_bindings = _frozen_comparator_extension_bindings(profile)
    if not _strict_json_equal(extension.get("bindings"), expected_bindings):
        raise FinalProvenanceSealError(
            "analysis comparator-extension frozen bindings changed"
        )
    if extension.get("launch_fingerprint") != expected_bindings[
        "launch_fingerprint"
    ]:
        raise FinalProvenanceSealError(
            "analysis comparator-extension launch fingerprint changed"
        )


def _authenticate_analysis(
    directory: str | Path,
    *,
    root: Path,
    profile: TrustedProfile,
    master: Mapping[str, Any],
    trusted_commit_sha256: str,
) -> dict[str, Any]:
    folder = _guard_path(directory, root=root, label="analysis directory", kind="dir")
    trusted_commit = _require_sha(
        trusted_commit_sha256, label="externally trusted analysis commit hash"
    )
    expected_artifacts = set(profile.analysis_artifacts)
    expected_files = (
        expected_artifacts
        | {name + ".metadata.json" for name in expected_artifacts}
        | {ANALYSIS_METADATA_NAME, ANALYSIS_COMMIT_NAME}
    )
    _assert_exact_inventory(folder, expected_files, root=root, label="analysis directory")
    raw = {
        name: _read_regular(folder / name, root=root, label=f"analysis file {name}")
        for name in sorted(expected_files)
    }
    if _sha256_bytes(raw[ANALYSIS_COMMIT_NAME]) != trusted_commit:
        raise FinalProvenanceSealError(
            "analysis commit differs from externally trusted SHA-256"
        )
    metadata = _decode_canonical_json(
        raw[ANALYSIS_METADATA_NAME], label="analysis metadata"
    )
    commit = _decode_canonical_json(raw[ANALYSIS_COMMIT_NAME], label="analysis commit")
    _exact_fields(metadata, ANALYSIS_METADATA_FIELDS, label="analysis metadata")
    _exact_fields(commit, ANALYSIS_COMMIT_FIELDS, label="analysis commit")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise FinalProvenanceSealError("analysis artifact manifest/inventory changed")
    if (
        not _strict_json_equal(metadata.get("schema_version"), 1)
        or metadata.get("status") != ANALYSIS_STATUS
        or metadata.get("artifact_class")
        != "authenticated_postprocessing_analysis_not_raw_trials"
        or metadata.get("analyzer_sha256")
        != profile.source_hashes["paper/analyze_budget_toxicity_calibration_audit.py"]
        or metadata.get("prespec_sha256")
        != profile.source_hashes["paper/budget_toxicity_calibration_prespec.md"]
        or metadata.get("manifest_sha256")
        != profile.source_hashes["paper/budget_toxicity_calibration_manifest.json"]
        or not _strict_json_equal(metadata.get("primary_estimand_count"), 118)
    ):
        raise FinalProvenanceSealError("analysis metadata identity changed")
    _validate_analysis_baseline_identity(
        metadata.get("baseline_identity"), profile=profile
    )
    expected_analysis_semantics = {
        "numerical_equivalence_guard": _frozen_numerical_guard_metadata(),
        "secondary_numerical_equivalence_guard": (
            _frozen_secondary_numerical_guard_metadata()
        ),
        "scientific_sign_source": "guarded simultaneous 95% band only",
        "secondary_scientific_sign_source": (
            "guarded pointwise 95% interval only"
        ),
        "secondary_absolute_row_count": 864,
        "secondary_contrast_row_count": 363,
    }
    for field, expected in expected_analysis_semantics.items():
        if not _strict_json_equal(metadata.get(field), expected):
            raise FinalProvenanceSealError(
                f"analysis frozen semantic contract changed: {field}"
            )
    if (
        not _strict_json_equal(commit.get("schema_version"), 1)
        or commit.get("status") != ANALYSIS_COMMIT_STATUS
        or commit.get("metadata") != ANALYSIS_METADATA_NAME
        or commit.get("metadata_sha256") != _sha256_bytes(raw[ANALYSIS_METADATA_NAME])
        or not _strict_json_equal(
            commit.get("artifact_count"), len(expected_artifacts)
        )
        or commit.get("baseline_identity_passed") is not True
    ):
        raise FinalProvenanceSealError("analysis commit binding changed")
    for name, entry in artifacts.items():
        if not isinstance(entry, Mapping):
            raise FinalProvenanceSealError(f"analysis manifest entry is malformed: {name}")
        _exact_fields(
            entry, {"sha256", "bytes", "metadata", "metadata_sha256"},
            label=f"analysis manifest {name}",
        )
        sidecar = name + ".metadata.json"
        if (
            entry.get("sha256") != _sha256_bytes(raw[name])
            or not _strict_json_equal(entry.get("bytes"), len(raw[name]))
            or entry.get("metadata") != sidecar
            or entry.get("metadata_sha256") != _sha256_bytes(raw[sidecar])
        ):
            raise FinalProvenanceSealError(f"analysis artifact binding changed: {name}")
    provenance = metadata.get("raw_provenance")
    if not isinstance(provenance, Mapping):
        raise FinalProvenanceSealError("analysis raw provenance is absent")
    expected_provenance_fields = {
        "artifact", "artifact_sha256", "metadata", "metadata_sha256",
        "commit", "commit_sha256", "uncompressed_sha256", "final_M",
        "path_count", "snapshot_count", "bindings",
        "streaming_and_crn_audit", "precision_decisions",
    }
    if master["branch"] == CAPPED_BRANCH:
        expected_provenance_fields.add("operator_cap")
    _exact_fields(
        provenance, expected_provenance_fields,
        label="analysis raw provenance",
    )
    if not _strict_json_equal(
        provenance.get("streaming_and_crn_audit"),
        _frozen_analysis_streaming_audit(),
    ):
        raise FinalProvenanceSealError(
            "analysis raw streaming/CRN audit changed"
        )
    master_envelope = master["envelope"]["envelope"]
    expected_master_provenance = {
        "artifact": Path(master_envelope["artifact"]["path"]).name,
        "artifact_sha256": master_envelope["artifact"]["sha256"],
        "metadata": Path(master_envelope["metadata"]["path"]).name,
        "metadata_sha256": master_envelope["metadata"]["sha256"],
        "commit": Path(master_envelope["commit"]["path"]).name,
        "commit_sha256": master_envelope["commit"]["sha256"],
        "uncompressed_sha256": master["envelope"]["uncompressed_sha256"],
        "final_M": 500,
        "path_count": 24_000,
        "snapshot_count": 72_000,
        "bindings": master["payload"]["bindings"],
    }
    for field, expected in expected_master_provenance.items():
        if not _strict_json_equal(provenance.get(field), expected):
            raise FinalProvenanceSealError(
                f"analysis does not bind the authenticated master: {field}"
            )
    analysis_decisions = provenance.get("precision_decisions")
    if (
        not isinstance(analysis_decisions, list)
        or len(analysis_decisions) != 2
        or not all(isinstance(value, Mapping) for value in analysis_decisions)
        or not _strict_json_equal(
            [value.get("cumulative_M") for value in analysis_decisions],
            [200, 500],
        )
    ):
        raise FinalProvenanceSealError("analysis precision-decision coverage changed")
    expected_analysis_decisions = []
    for payload in master["precision_payloads"]:
        expected_analysis_decisions.append({
            "cumulative_M": payload["cumulative_M"],
            "maximum_raw_mcse_points": payload["maximum_raw_mcse_points"],
            "maximum_guarded_mcse_points": payload["maximum_guarded_mcse_points"],
            "all_estimands_pass_guarded_1_5pp": payload[
                "all_estimands_pass_guarded_1_5pp"
            ],
            "top_up_limit_reached": payload["top_up_limit_reached"],
            "numerical_equivalence_guard": payload["numerical_equivalence_guard"],
            "next_M": payload["next_M"],
            "input_hashes": payload["input_hashes"],
        })
    for index, decision in enumerate(analysis_decisions):
        _exact_fields(
            decision, ANALYSIS_PRECISION_FIELDS,
            label=f"analysis precision decision {index}",
        )
    if not _strict_json_equal(analysis_decisions, expected_analysis_decisions):
        raise FinalProvenanceSealError(
            "analysis does not preserve the authenticated blind precision decisions"
        )
    if master["branch"] == PASS_BRANCH:
        if "operator_cap" in provenance:
            raise FinalProvenanceSealError("ordinary analysis contains capped-branch provenance")
    else:
        observed_cap = provenance.get("operator_cap")
        if not isinstance(observed_cap, Mapping):
            raise FinalProvenanceSealError("capped analysis lacks cap provenance")
        observed_cap_without_auth = dict(observed_cap)
        authentication = observed_cap_without_auth.pop("authentication", None)
        if not _strict_json_equal(observed_cap_without_auth, master["cap_record"]):
            raise FinalProvenanceSealError("capped analysis cap record differs from master")
        if not isinstance(authentication, Mapping) or not _strict_json_equal(authentication, {
            "cap_aware_analyzer_sha256": profile.source_hashes[
                "paper/analyze_budget_toxicity_calibration_operator_cap.py"
            ],
            "cap_finalizer_sha256": profile.source_hashes[
                "paper/finalize_budget_toxicity_calibration_operator_capped_master.py"
            ],
            "frozen_analyzer_sha256": profile.source_hashes[
                "paper/analyze_budget_toxicity_calibration_audit.py"
            ],
            "amendment_artifact_sha256": profile.amendment_hashes["artifact"],
            "amendment_metadata_sha256": profile.amendment_hashes["metadata"],
            "amendment_commit_sha256": profile.amendment_hashes["commit"],
            "launch_fingerprint": profile.launch_fingerprint,
        }):
            raise FinalProvenanceSealError("capped analysis wrapper authentication changed")
    commitments = _commitments_for_flat_directory(
        folder, expected_files, root=root, label="analysis"
    )
    return {
        "metadata": metadata,
        "commit": commit,
        "raw": raw,
        "envelope": {
            "files": commitments,
            "trusted_commit_sha256": trusted_commit,
            "file_count": len(commitments),
            "artifact_count": len(expected_artifacts),
            "file_bundle_sha256": _bundle_hash(_hash_map(commitments)),
            "status": ANALYSIS_STATUS,
            "commit_status": ANALYSIS_COMMIT_STATUS,
            "raw_master_commit_sha256": master_envelope["commit"]["sha256"],
        },
    }


PUBLICATION_METADATA_FIELDS = {
    "schema_version", "status", "artifact_class", "adapter", "adapter_sha256",
    "contract", "contract_sha256", "trusted_commitments", "inputs",
    "input_bundle_sha256", "authenticated_analysis", "publication_rules",
    "artifacts", "output_bundle_sha256", "deterministic_timestamp_utc",
}
PUBLICATION_COMMIT_FIELDS = {
    "schema_version", "status", "metadata", "metadata_sha256", "artifact_count",
    "adapter_sha256", "contract_sha256", "trusted_commitments",
    "input_bundle_sha256", "output_bundle_sha256", "baseline_identity_passed",
    "primary_estimand_count",
}
PUBLICATION_SIDECAR_FIELDS = {
    "schema_version", "status", "artifact_class", "artifact_type", "artifact",
    "artifact_sha256", "artifact_bytes", "adapter", "adapter_sha256",
    "contract", "contract_sha256", "trusted_commitments",
    "input_bundle_sha256", "description", "selection",
    "no_filtering_by_sign_interval_or_winner",
}
PUBLICATION_ANALYSIS_SUMMARY_FIELDS = {
    "status", "commit_status", "baseline_status", "final_M",
    "primary_estimand_count", "primary_family_counts",
    "secondary_absolute_row_count", "secondary_contrast_row_count",
    "secondary_absolute_profile_rows_used",
    "secondary_policy_vs_cEI_calibration_did_rows_published",
}


def _authenticate_publication(
    directory: str | Path,
    *,
    root: Path,
    profile: TrustedProfile,
    analysis: Mapping[str, Any],
    trusted_commit_sha256: str,
) -> dict[str, Any]:
    folder = _guard_path(directory, root=root, label="publication directory", kind="dir")
    trusted_commit = _require_sha(
        trusted_commit_sha256, label="externally trusted publication commit hash"
    )
    expected_artifacts = set(profile.publication_artifacts)
    sidecar_contracts = profile.publication_sidecar_contracts
    if (
        not isinstance(sidecar_contracts, Mapping)
        or set(sidecar_contracts) != expected_artifacts
        or any(not isinstance(value, Mapping) for value in sidecar_contracts.values())
    ):
        raise FinalProvenanceSealError(
            "trusted publication sidecar contract coverage changed"
        )
    expected_files = (
        expected_artifacts
        | {name + ".metadata.json" for name in expected_artifacts}
        | {PUBLICATION_METADATA_NAME, PUBLICATION_COMMIT_NAME}
    )
    _assert_exact_inventory(folder, expected_files, root=root, label="publication directory")
    raw = {
        name: _read_regular(folder / name, root=root, label=f"publication file {name}")
        for name in sorted(expected_files)
    }
    if _sha256_bytes(raw[PUBLICATION_COMMIT_NAME]) != trusted_commit:
        raise FinalProvenanceSealError(
            "publication commit differs from externally trusted SHA-256"
        )
    metadata = _decode_canonical_json(
        raw[PUBLICATION_METADATA_NAME], label="publication metadata"
    )
    commit = _decode_canonical_json(
        raw[PUBLICATION_COMMIT_NAME], label="publication commit"
    )
    _exact_fields(metadata, PUBLICATION_METADATA_FIELDS, label="publication metadata")
    _exact_fields(commit, PUBLICATION_COMMIT_FIELDS, label="publication commit")
    adapter_sha = profile.source_hashes[
        "paper/generate_budget_toxicity_calibration_publication.py"
    ]
    contract_sha = profile.source_hashes[
        "paper/budget_toxicity_calibration_publication_contract.json"
    ]
    expected_trusted_commitments = {
        "analyzer_sha256": profile.source_hashes[
            "paper/analyze_budget_toxicity_calibration_audit.py"
        ],
        "manifest_sha256": profile.source_hashes[
            "paper/budget_toxicity_calibration_manifest.json"
        ],
        "prespec_sha256": profile.source_hashes[
            "paper/budget_toxicity_calibration_prespec.md"
        ],
    }
    expected_publication_rules = {
        "complete_primary_projection": True,
        "no_filtering_by_sign_interval_or_winner": True,
        "colors_encode_only_c_sigma": True,
        "finite_fixed_design_audit": True,
        "raw_estimates_mcse_and_intervals_retained": True,
        "primary_scientific_display_uses_guarded_simultaneous_bands": True,
        "secondary_scientific_display_uses_guarded_pointwise_intervals": True,
        "policy_vs_cEI_calibration_did_projection_complete": True,
        "secondary_cell_predicates_have_no_vote_or_family_theorem": True,
    }
    if (
        not _strict_json_equal(metadata.get("schema_version"), 3)
        or metadata.get("status") != PUBLICATION_STATUS
        or metadata.get("artifact_class")
        != "outcome_independent_budget_toxicity_calibration_publication_projection"
        or metadata.get("adapter")
        != "paper/generate_budget_toxicity_calibration_publication.py"
        or metadata.get("adapter_sha256") != adapter_sha
        or metadata.get("contract")
        != "budget_toxicity_calibration_publication_contract.json"
        or metadata.get("contract_sha256") != contract_sha
        or not _strict_json_equal(
            metadata.get("trusted_commitments"), expected_trusted_commitments
        )
        or not _strict_json_equal(
            metadata.get("publication_rules"), expected_publication_rules
        )
        or metadata.get("deterministic_timestamp_utc") != "2000-01-01T00:00:00Z"
    ):
        raise FinalProvenanceSealError("publication metadata identity changed")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise FinalProvenanceSealError("publication artifact manifest/inventory changed")
    if (
        not _strict_json_equal(commit.get("schema_version"), 3)
        or commit.get("status") != PUBLICATION_COMMIT_STATUS
        or commit.get("metadata") != PUBLICATION_METADATA_NAME
        or commit.get("metadata_sha256") != _sha256_bytes(raw[PUBLICATION_METADATA_NAME])
        or not _strict_json_equal(
            commit.get("artifact_count"), len(expected_artifacts)
        )
        or commit.get("adapter_sha256") != adapter_sha
        or commit.get("contract_sha256") != contract_sha
        or not _strict_json_equal(
            commit.get("trusted_commitments"), expected_trusted_commitments
        )
        or commit.get("baseline_identity_passed") is not True
        or not _strict_json_equal(commit.get("primary_estimand_count"), 118)
    ):
        raise FinalProvenanceSealError("publication commit binding changed")
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(ANALYSIS_INPUT_NAMES):
        raise FinalProvenanceSealError("publication analysis-input manifest changed")
    analysis_raw = analysis["raw"]
    input_hashes: dict[str, str] = {}
    for name in ANALYSIS_INPUT_NAMES:
        entry = inputs.get(name)
        if not isinstance(entry, Mapping):
            raise FinalProvenanceSealError(f"publication input is malformed: {name}")
        _exact_fields(entry, {"sha256", "bytes"}, label=f"publication input {name}")
        observed = analysis_raw[name]
        if (
            entry.get("sha256") != _sha256_bytes(observed)
            or not _strict_json_equal(entry.get("bytes"), len(observed))
        ):
            raise FinalProvenanceSealError(
                f"publication input differs from authenticated analysis: {name}"
            )
        input_hashes[name] = entry["sha256"]
    input_bundle = _bundle_hash(input_hashes)
    if (
        metadata.get("input_bundle_sha256") != input_bundle
        or commit.get("input_bundle_sha256") != input_bundle
    ):
        raise FinalProvenanceSealError("publication input-bundle binding changed")
    analysis_summary = metadata.get("authenticated_analysis")
    if (
        not isinstance(analysis_summary, Mapping)
        or set(analysis_summary) != PUBLICATION_ANALYSIS_SUMMARY_FIELDS
        or analysis_summary.get("status") != ANALYSIS_STATUS
        or analysis_summary.get("commit_status") != ANALYSIS_COMMIT_STATUS
        or analysis_summary.get("baseline_status") != "PASS_AVAILABLE_FIELD_IDENTITY"
        or not _strict_json_equal(analysis_summary.get("final_M"), 500)
        or not _strict_json_equal(
            analysis_summary.get("primary_estimand_count"), 118
        )
        or not _strict_json_equal(
            analysis_summary.get("primary_family_counts"),
            {"A": 54, "B": 48, "C": 16},
        )
        or not _strict_json_equal(
            analysis_summary.get("secondary_absolute_row_count"), 864
        )
        or not _strict_json_equal(
            analysis_summary.get("secondary_contrast_row_count"), 363
        )
        or not _strict_json_equal(
            analysis_summary.get("secondary_absolute_profile_rows_used"), 72
        )
        or not _strict_json_equal(
            analysis_summary.get(
                "secondary_policy_vs_cEI_calibration_did_rows_published"
            ),
            36,
        )
    ):
        raise FinalProvenanceSealError("publication analysis summary changed")
    output_hashes: dict[str, str] = {}
    for name, entry in artifacts.items():
        if not isinstance(entry, Mapping):
            raise FinalProvenanceSealError(f"publication manifest entry malformed: {name}")
        _exact_fields(
            entry, {"sha256", "bytes", "metadata", "metadata_sha256", "artifact_type"},
            label=f"publication manifest {name}",
        )
        sidecar_name = name + ".metadata.json"
        contract = sidecar_contracts[name]
        _exact_fields(
            contract, {"artifact_type", "description", "selection"},
            label=f"trusted publication sidecar contract {name}",
        )
        if (
            entry.get("sha256") != _sha256_bytes(raw[name])
            or not _strict_json_equal(entry.get("bytes"), len(raw[name]))
            or entry.get("metadata") != sidecar_name
            or entry.get("metadata_sha256") != _sha256_bytes(raw[sidecar_name])
            or not _strict_json_equal(
                entry.get("artifact_type"), contract.get("artifact_type")
            )
        ):
            raise FinalProvenanceSealError(f"publication artifact binding changed: {name}")
        sidecar = _decode_canonical_json(
            raw[sidecar_name], label=f"publication sidecar {sidecar_name}"
        )
        _exact_fields(
            sidecar, PUBLICATION_SIDECAR_FIELDS,
            label=f"publication sidecar {sidecar_name}",
        )
        if (
            not _strict_json_equal(sidecar.get("schema_version"), 3)
            or sidecar.get("status") != PUBLICATION_ARTIFACT_STATUS
            or sidecar.get("artifact_class")
            != "outcome_independent_budget_toxicity_calibration_publication_projection"
            or not _strict_json_equal(
                sidecar.get("artifact_type"), contract.get("artifact_type")
            )
            or sidecar.get("artifact") != name
            or sidecar.get("artifact_sha256") != entry["sha256"]
            or sidecar.get("artifact_bytes") != entry["bytes"]
            or sidecar.get("adapter")
            != "paper/generate_budget_toxicity_calibration_publication.py"
            or sidecar.get("adapter_sha256") != adapter_sha
            or sidecar.get("contract")
            != "budget_toxicity_calibration_publication_contract.json"
            or sidecar.get("contract_sha256") != contract_sha
            or not _strict_json_equal(
                sidecar.get("trusted_commitments"), expected_trusted_commitments
            )
            or sidecar.get("input_bundle_sha256") != input_bundle
            or not _strict_json_equal(
                sidecar.get("description"), contract.get("description")
            )
            or not _strict_json_equal(
                sidecar.get("selection"), contract.get("selection")
            )
            or sidecar.get("no_filtering_by_sign_interval_or_winner") is not True
        ):
            raise FinalProvenanceSealError(f"publication sidecar binding changed: {name}")
        output_hashes[name] = entry["sha256"]
        output_hashes[sidecar_name] = entry["metadata_sha256"]
    output_bundle = _bundle_hash(output_hashes)
    if (
        metadata.get("output_bundle_sha256") != output_bundle
        or commit.get("output_bundle_sha256") != output_bundle
    ):
        raise FinalProvenanceSealError("publication output-bundle binding changed")
    commitments = _commitments_for_flat_directory(
        folder, expected_files, root=root, label="publication"
    )
    analysis_commit_sha = _sha256_bytes(analysis_raw[ANALYSIS_COMMIT_NAME])
    return {
        "metadata": metadata,
        "commit": commit,
        "raw": raw,
        "envelope": {
            "files": commitments,
            "trusted_commit_sha256": trusted_commit,
            "file_count": len(commitments),
            "artifact_count": len(expected_artifacts),
            "file_bundle_sha256": _bundle_hash(_hash_map(commitments)),
            "input_bundle_sha256": input_bundle,
            "output_bundle_sha256": output_bundle,
            "status": PUBLICATION_STATUS,
            "commit_status": PUBLICATION_COMMIT_STATUS,
            "analysis_commit_sha256": analysis_commit_sha,
        },
    }


def _authenticate_projection(
    commit_path: str | Path | None,
    *,
    root: Path,
    projection_root: str | Path | None,
    profile: TrustedProfile,
    publication: Mapping[str, Any],
    trusted_commit_sha256: str | None,
) -> dict[str, Any] | None:
    supplied = (commit_path is not None, projection_root is not None, trusted_commit_sha256 is not None)
    if not any(supplied):
        return None
    if not all(supplied):
        raise FinalProvenanceSealError(
            "projection root, commit path, and externally trusted hash must be supplied together"
        )
    manuscript_root = _guard_root(projection_root)
    trusted_commit = _require_sha(
        trusted_commit_sha256, label="externally trusted projection commit hash"
    )
    commit_file = _guard_path(
        commit_path, root=manuscript_root, label="projection commit", kind="file"
    )
    if commit_file != manuscript_root / "generated" / PROJECTION_COMMIT_NAME:
        raise FinalProvenanceSealError("projection commit path/name changed")
    metadata_file = commit_file.parent / PROJECTION_METADATA_NAME
    commit_raw = _read_regular(
        commit_file, root=manuscript_root, label="projection commit"
    )
    metadata_raw = _read_regular(
        metadata_file, root=manuscript_root, label="projection metadata"
    )
    if _sha256_bytes(commit_raw) != trusted_commit:
        raise FinalProvenanceSealError(
            "projection commit differs from externally trusted SHA-256"
        )
    metadata = _decode_canonical_json(metadata_raw, label="projection metadata")
    commit = _decode_canonical_json(commit_raw, label="projection commit")
    _exact_fields(metadata, {
        "schema_version", "status", "artifact_class", "projector",
        "projector_sha256", "source_publication", "projection_rules", "files",
        "projected_file_count", "projection_bundle_sha256",
    }, label="projection metadata")
    _exact_fields(commit, {
        "schema_version", "status", "metadata", "metadata_sha256",
        "artifact_class", "projected_file_count", "projection_bundle_sha256",
        "source_publication_commit_sha256",
        "source_publication_output_bundle_sha256",
    }, label="projection commit")
    mapping = dict(profile.projection_mapping)
    files = metadata.get("files")
    if not isinstance(files, Mapping) or set(files) != set(mapping.values()):
        raise FinalProvenanceSealError("projection destination inventory changed")
    publication_commit_sha = publication["envelope"]["trusted_commit_sha256"]
    publication_output_bundle = publication["envelope"]["output_bundle_sha256"]
    source_publication = metadata.get("source_publication")
    if not isinstance(source_publication, Mapping):
        raise FinalProvenanceSealError("projection source-publication record is absent")
    _exact_fields(
        source_publication,
        {
            "metadata", "metadata_sha256", "commit", "commit_sha256",
            "externally_trusted_commit_sha256", "contract_sha256",
            "adapter_sha256", "input_bundle_sha256", "output_bundle_sha256",
            "baseline_identity_passed", "primary_estimand_count",
            "policy_vs_cEI_calibration_did_row_count", "final_M",
        },
        label="projection source publication",
    )
    publication_files = publication["envelope"]["files"]
    expected_source_publication = {
        "metadata": PUBLICATION_METADATA_NAME,
        "metadata_sha256": publication_files[PUBLICATION_METADATA_NAME]["sha256"],
        "commit": PUBLICATION_COMMIT_NAME,
        "commit_sha256": publication_commit_sha,
        "externally_trusted_commit_sha256": publication_commit_sha,
        "contract_sha256": profile.source_hashes[
            "paper/budget_toxicity_calibration_publication_contract.json"
        ],
        "adapter_sha256": profile.source_hashes[
            "paper/generate_budget_toxicity_calibration_publication.py"
        ],
        "input_bundle_sha256": publication["envelope"]["input_bundle_sha256"],
        "output_bundle_sha256": publication_output_bundle,
        "baseline_identity_passed": True,
        "primary_estimand_count": 118,
        "policy_vs_cEI_calibration_did_row_count": 36,
        "final_M": 500,
    }
    expected_projection_rules = {
        "scientific_artifacts_decoded": False,
        "conclusion_text_generated_or_modified": False,
        "selected_by_outcome_sign_interval_or_winner": False,
        "source_artifact_count_authenticated": len(profile.publication_artifacts),
        "source_physical_file_count_authenticated": publication["envelope"]["file_count"],
        "tex_fragment_count": sum(
            destination.endswith(".tex") for destination in mapping.values()
        ),
        "pdf_figure_count": sum(
            destination.endswith(".pdf") for destination in mapping.values()
        ),
        "destination_mapping_frozen": True,
        "copy_is_byte_for_byte": True,
    }
    if (
        not _strict_json_equal(metadata.get("schema_version"), 1)
        or metadata.get("status") != PROJECTION_STATUS
        or metadata.get("artifact_class")
        != "outcome_blind_budget_toxicity_calibration_manuscript_projection"
        or metadata.get("projector")
        != "paper/project_budget_toxicity_calibration_publication_to_manuscript.py"
        or metadata.get("projector_sha256")
        != profile.source_hashes[
            "paper/project_budget_toxicity_calibration_publication_to_manuscript.py"
        ]
        or not _strict_json_equal(
            metadata.get("projected_file_count"), len(mapping)
        )
        or not _strict_json_equal(source_publication, expected_source_publication)
        or not _strict_json_equal(
            metadata.get("projection_rules"), expected_projection_rules
        )
    ):
        raise FinalProvenanceSealError("projection metadata/source binding changed")
    if (
        not _strict_json_equal(commit.get("schema_version"), 1)
        or commit.get("status") != PROJECTION_COMMIT_STATUS
        or commit.get("metadata") != f"generated/{PROJECTION_METADATA_NAME}"
        or commit.get("metadata_sha256") != _sha256_bytes(metadata_raw)
        or commit.get("artifact_class")
        != "outcome_blind_budget_toxicity_calibration_manuscript_projection"
        or not _strict_json_equal(commit.get("projected_file_count"), len(mapping))
        or commit.get("source_publication_commit_sha256") != publication_commit_sha
        or commit.get("source_publication_output_bundle_sha256")
        != publication_output_bundle
    ):
        raise FinalProvenanceSealError("projection commit binding changed")
    destination_hashes: dict[str, str] = {}
    commitments: dict[str, dict[str, Any]] = {}
    for source_name, destination in sorted(mapping.items(), key=lambda row: row[1]):
        safe_destination = _safe_relative(destination, label="projection destination")
        entry = files.get(destination)
        if not isinstance(entry, Mapping):
            raise FinalProvenanceSealError(f"projection entry malformed: {destination}")
        _exact_fields(
            entry,
            {
                "source_artifact", "source_artifact_sha256",
                "source_artifact_bytes", "source_sidecar",
                "source_sidecar_sha256", "copy_mode",
            },
            label=f"projection entry {destination}",
        )
        source_entry = publication_files.get(source_name)
        source_sidecar = publication_files.get(source_name + ".metadata.json")
        if not isinstance(source_entry, Mapping) or not isinstance(source_sidecar, Mapping):
            raise FinalProvenanceSealError(f"projection source is absent: {source_name}")
        if (
            entry.get("source_artifact") != source_name
            or entry.get("source_artifact_sha256") != source_entry["sha256"]
            or not _strict_json_equal(
                entry.get("source_artifact_bytes"), source_entry["bytes"]
            )
            or entry.get("source_sidecar") != source_name + ".metadata.json"
            or entry.get("source_sidecar_sha256") != source_sidecar["sha256"]
            or entry.get("copy_mode") != "byte_for_byte_regular_file_copy"
        ):
            raise FinalProvenanceSealError(f"projection source mapping changed: {destination}")
        projected = manuscript_root / safe_destination
        projected_entry = _file_commitment(
            projected, root=manuscript_root, label=f"projected file {destination}"
        )
        if (
            projected_entry["sha256"] != source_entry["sha256"]
            or projected_entry["bytes"] != source_entry["bytes"]
        ):
            raise FinalProvenanceSealError(f"projected file is not byte-identical: {destination}")
        commitments[destination] = projected_entry
        destination_hashes[destination] = projected_entry["sha256"]
    projection_bundle = _bundle_hash(destination_hashes)
    if (
        metadata.get("projection_bundle_sha256") != projection_bundle
        or commit.get("projection_bundle_sha256") != projection_bundle
    ):
        raise FinalProvenanceSealError("projection bundle commitment changed")
    commitments[f"generated/{PROJECTION_METADATA_NAME}"] = _file_commitment(
        metadata_file, root=manuscript_root, label="projection metadata"
    )
    commitments[f"generated/{PROJECTION_COMMIT_NAME}"] = _file_commitment(
        commit_file, root=manuscript_root, label="projection commit"
    )
    return {
        "path_root_role": "explicit_manuscript_source_root",
        "files": dict(sorted(commitments.items())),
        "trusted_commit_sha256": trusted_commit,
        "file_count": len(commitments),
        "file_bundle_sha256": _bundle_hash(_hash_map(commitments)),
        "projection_bundle_sha256": projection_bundle,
        "status": PROJECTION_STATUS,
        "commit_status": PROJECTION_COMMIT_STATUS,
        "source_publication_commit_sha256": publication_commit_sha,
    }


def _validate_file_commitment(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalProvenanceSealError(f"{label} is not a file commitment")
    _exact_fields(value, {"path", "sha256", "bytes"}, label=label)
    if not isinstance(value.get("path"), str):
        raise FinalProvenanceSealError(f"{label} path is malformed")
    _safe_relative(value["path"], label=f"{label} path")
    _require_sha(value.get("sha256"), label=f"{label} SHA-256")
    _positive_integer(value.get("bytes"), label=f"{label} bytes")
    return dict(value)


def _validate_file_map(
    value: Any, *, label: str, exact_count: int | None = None
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise FinalProvenanceSealError(f"{label} is not a nonempty file map")
    if exact_count is not None and len(value) != exact_count:
        raise FinalProvenanceSealError(f"{label} file count changed")
    output: dict[str, dict[str, Any]] = {}
    for name, entry in value.items():
        if not isinstance(name, str):
            raise FinalProvenanceSealError(f"{label} key is malformed")
        _safe_relative(name, label=f"{label} key")
        output[name] = _validate_file_commitment(entry, label=f"{label} {name}")
    return output


def _validate_trio(value: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise FinalProvenanceSealError(f"{label} trio is absent")
    _exact_fields(value, {"artifact", "metadata", "commit"}, label=f"{label} trio")
    return {
        name: _validate_file_commitment(value[name], label=f"{label} {name}")
        for name in ("artifact", "metadata", "commit")
    }


def _validate_seal_payload(
    value: Any, *, profile: TrustedProfile
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalProvenanceSealError("final provenance payload is not an object")
    payload = dict(value)
    expected_top = {
        "schema_version", "status", "artifact_class", "branch", "scope",
        "audit_identity", "precision_state", "amendment_v2",
        "source_commitments", "source_bundle_sha256", "master_envelope",
        "analysis_envelope", "publication_envelope", "projection_envelope",
        "upstream_commit_bundle_sha256",
    }
    _exact_fields(payload, expected_top, label="final provenance payload")
    branch = payload.get("branch")
    if (
        not _strict_json_equal(payload.get("schema_version"), 1)
        or payload.get("status") != STATUS
        or payload.get("artifact_class") != ARTIFACT_CLASS
        or branch not in (PASS_BRANCH, CAPPED_BRANCH)
    ):
        raise FinalProvenanceSealError("final provenance payload identity changed")
    expected_scope = {
        "scientific_artifacts_decoded": False,
        "effect_values_interpreted": False,
        "effect_directions_interpreted": False,
        "winner_or_sign_selected": False,
        "operational_precision_state_authenticated": True,
        "cryptographic_provenance_only": True,
    }
    if not _strict_json_equal(payload.get("scope"), expected_scope):
        raise FinalProvenanceSealError("final provenance outcome-blind scope changed")
    if not _strict_json_equal(payload.get("audit_identity"), {
        "launch_fingerprint": profile.launch_fingerprint, "final_M": 500,
    }):
        raise FinalProvenanceSealError("final provenance audit identity changed")
    state = payload.get("precision_state")
    if not isinstance(state, Mapping):
        raise FinalProvenanceSealError("final precision state is absent")
    _exact_fields(
        state, {"original_frozen_decision", "effective_terminal_state"},
        label="final precision state",
    )
    original = state.get("original_frozen_decision")
    effective = state.get("effective_terminal_state")
    if not isinstance(original, Mapping) or not isinstance(effective, Mapping):
        raise FinalProvenanceSealError("final precision branch state is malformed")
    _exact_fields(
        original, {"cumulative_M", "all_estimands_pass_guarded_1_5pp", "next_M"},
        label="original frozen precision state",
    )
    _exact_fields(
        effective,
        {
            "operator_cap_applied", "effective_next_M",
            "precision_target_achieved", "terminated_with_unresolved_precision",
            "unresolved_estimand_count",
        },
        label="effective precision state",
    )
    unresolved = effective.get("unresolved_estimand_count")
    if isinstance(unresolved, bool) or not isinstance(unresolved, int) or not 0 <= unresolved <= 118:
        raise FinalProvenanceSealError("effective unresolved count is malformed")
    expected_state = (
        (
            {"cumulative_M": 500, "all_estimands_pass_guarded_1_5pp": True, "next_M": None},
            {
                "operator_cap_applied": False, "effective_next_M": None,
                "precision_target_achieved": True,
                "terminated_with_unresolved_precision": False,
                "unresolved_estimand_count": 0,
            },
        )
        if branch == PASS_BRANCH
        else (
            {"cumulative_M": 500, "all_estimands_pass_guarded_1_5pp": False, "next_M": 1000},
            {
                "operator_cap_applied": True, "effective_next_M": None,
                "precision_target_achieved": False,
                "terminated_with_unresolved_precision": True,
                "unresolved_estimand_count": unresolved,
            },
        )
    )
    if (
        not _strict_json_equal(original, expected_state[0])
        or not _strict_json_equal(effective, expected_state[1])
    ):
        raise FinalProvenanceSealError("final precision branch state is confused")
    if branch == CAPPED_BRANCH and not 1 <= unresolved <= 118:
        raise FinalProvenanceSealError("capped precision state is not unresolved")

    amendment = payload.get("amendment_v2")
    if not isinstance(amendment, Mapping):
        raise FinalProvenanceSealError("amendment v2 commitment is absent")
    _exact_fields(
        amendment,
        {
            "envelope", "audit_id", "launch_fingerprint", "declared_at_utc",
            "committed_at_utc", "cap_cumulative_M",
        },
        label="amendment v2 commitment",
    )
    amendment_trio = _validate_trio(amendment.get("envelope"), label="amendment v2")
    if (
        not isinstance(amendment.get("audit_id"), str)
        or re.fullmatch(r"operator-cap-M0500-[0-9a-f]{16}", amendment["audit_id"])
        is None
        or amendment.get("launch_fingerprint") != profile.launch_fingerprint
        or not _strict_json_equal(amendment.get("cap_cumulative_M"), 500)
    ):
        raise FinalProvenanceSealError("amendment v2 sealed identity changed")
    declared, committed = _assert_time_order(
        amendment.get("declared_at_utc"), amendment.get("committed_at_utc"),
        label="sealed amendment v2",
    )
    if declared > committed:
        raise FinalProvenanceSealError("sealed amendment v2 timestamp order changed")
    expected_amendment_paths = {
        "artifact": profile.amendment_relative,
        "metadata": profile.amendment_relative + ".metadata.json",
        "commit": profile.amendment_relative + ".commit.json",
    }
    for name, entry in amendment_trio.items():
        if (
            entry["path"] != expected_amendment_paths[name]
            or entry["sha256"] != profile.amendment_hashes[name]
        ):
            raise FinalProvenanceSealError("sealed amendment v2 trio changed")

    sources = _validate_file_map(
        payload.get("source_commitments"), label="source commitments",
        exact_count=len(profile.source_hashes),
    )
    if set(sources) != set(profile.source_hashes):
        raise FinalProvenanceSealError("source commitment paths changed")
    for name, expected in profile.source_hashes.items():
        if sources[name]["path"] != name or sources[name]["sha256"] != expected:
            raise FinalProvenanceSealError(f"source commitment changed: {name}")
    source_bundle = _bundle_hash(_hash_map(sources))
    if payload.get("source_bundle_sha256") != source_bundle:
        raise FinalProvenanceSealError("source bundle commitment changed")

    master = payload.get("master_envelope")
    if not isinstance(master, Mapping):
        raise FinalProvenanceSealError("master envelope is absent")
    _exact_fields(
        master,
        {
            "envelope", "indexed_block_files", "indexed_baseline_files",
            "indexed_precision_files",
            "trusted_commit_sha256", "uncompressed_sha256", "uncompressed_bytes",
            "status", "commit_status", "artifact_class", "storage_model",
        },
        label="sealed master envelope",
    )
    master_trio = _validate_trio(master.get("envelope"), label="sealed master")
    block_files = _validate_file_map(
        master.get("indexed_block_files"), label="sealed indexed blocks", exact_count=6
    )
    baseline_files = _validate_file_map(
        master.get("indexed_baseline_files"),
        label="sealed indexed baseline",
        exact_count=3,
    )
    precision_files = _validate_file_map(
        master.get("indexed_precision_files"), label="sealed indexed precision", exact_count=6
    )
    expected_block_keys = {
        suffix
        for block_id in ("M0200", "M0500_TOPUP")
        for suffix in (
            f"raw/budget_toxicity_calibration_{block_id}_block_index.json.zst",
            f"raw/budget_toxicity_calibration_{block_id}_block_index.json.zst.metadata.json",
            f"raw/budget_toxicity_calibration_{block_id}_block_index.json.zst.commit.json",
        )
    }
    expected_precision_keys = {
        suffix
        for cumulative_m in (200, 500)
        for suffix in (
            f"precision/budget_toxicity_calibration_precision_M{cumulative_m:04d}.json.zst",
            f"precision/budget_toxicity_calibration_precision_M{cumulative_m:04d}.json.zst.metadata.json",
            f"precision/budget_toxicity_calibration_precision_M{cumulative_m:04d}.json.zst.commit.json",
        )
    }
    expected_baseline_keys = {
        BASELINE_RELATIVE,
        BASELINE_RELATIVE + ".metadata.json",
        BASELINE_RELATIVE + ".commit.json",
    }
    _require_sha(master.get("uncompressed_sha256"), label="sealed master logical SHA")
    if (
        set(block_files) != expected_block_keys
        or set(baseline_files) != expected_baseline_keys
        or set(precision_files) != expected_precision_keys
        or master.get("trusted_commit_sha256") != master_trio["commit"]["sha256"]
    ):
        raise FinalProvenanceSealError("sealed master trust root changed")
    _positive_integer(master.get("uncompressed_bytes"), label="sealed master logical bytes")
    expected_master_identity = (
        (
            ORIGINAL_MASTER_STATUS, ORIGINAL_MASTER_COMMIT_STATUS,
            ORIGINAL_MASTER_CLASS, ORIGINAL_MASTER_STORAGE,
        )
        if branch == PASS_BRANCH
        else (
            CAPPED_MASTER_STATUS, CAPPED_MASTER_COMMIT_STATUS,
            CAPPED_MASTER_CLASS, CAPPED_MASTER_STORAGE,
        )
    )
    if tuple(master.get(name) for name in (
        "status", "commit_status", "artifact_class", "storage_model"
    )) != expected_master_identity:
        raise FinalProvenanceSealError("sealed master branch identity changed")
    del block_files, baseline_files, precision_files

    analysis = payload.get("analysis_envelope")
    if not isinstance(analysis, Mapping):
        raise FinalProvenanceSealError("analysis envelope is absent")
    _exact_fields(
        analysis,
        {
            "files", "trusted_commit_sha256", "file_count", "artifact_count",
            "file_bundle_sha256", "status", "commit_status",
            "raw_master_commit_sha256",
        },
        label="sealed analysis envelope",
    )
    analysis_files = _validate_file_map(analysis.get("files"), label="analysis files")
    expected_analysis_files = (
        set(profile.analysis_artifacts)
        | {name + ".metadata.json" for name in profile.analysis_artifacts}
        | {ANALYSIS_METADATA_NAME, ANALYSIS_COMMIT_NAME}
    )
    if (
        set(analysis_files) != expected_analysis_files
        or not _strict_json_equal(
            analysis.get("file_count"), len(analysis_files)
        )
        or not _strict_json_equal(
            analysis.get("artifact_count"), len(profile.analysis_artifacts)
        )
        or analysis.get("file_bundle_sha256") != _bundle_hash(_hash_map(analysis_files))
        or analysis.get("trusted_commit_sha256")
        != analysis_files[ANALYSIS_COMMIT_NAME]["sha256"]
        or analysis.get("raw_master_commit_sha256") != master["trusted_commit_sha256"]
        or analysis.get("status") != ANALYSIS_STATUS
        or analysis.get("commit_status") != ANALYSIS_COMMIT_STATUS
    ):
        raise FinalProvenanceSealError("sealed analysis envelope changed")

    publication = payload.get("publication_envelope")
    if not isinstance(publication, Mapping):
        raise FinalProvenanceSealError("publication envelope is absent")
    _exact_fields(
        publication,
        {
            "files", "trusted_commit_sha256", "file_count", "artifact_count",
            "file_bundle_sha256", "input_bundle_sha256", "output_bundle_sha256",
            "status", "commit_status", "analysis_commit_sha256",
        },
        label="sealed publication envelope",
    )
    publication_files = _validate_file_map(
        publication.get("files"), label="publication files"
    )
    expected_publication_files = (
        set(profile.publication_artifacts)
        | {name + ".metadata.json" for name in profile.publication_artifacts}
        | {PUBLICATION_METADATA_NAME, PUBLICATION_COMMIT_NAME}
    )
    for field in ("input_bundle_sha256", "output_bundle_sha256"):
        _require_sha(publication.get(field), label=f"sealed publication {field}")
    if (
        set(publication_files) != expected_publication_files
        or not _strict_json_equal(
            publication.get("file_count"), len(publication_files)
        )
        or not _strict_json_equal(
            publication.get("artifact_count"), len(profile.publication_artifacts)
        )
        or publication.get("file_bundle_sha256")
        != _bundle_hash(_hash_map(publication_files))
        or publication.get("trusted_commit_sha256")
        != publication_files[PUBLICATION_COMMIT_NAME]["sha256"]
        or publication.get("analysis_commit_sha256")
        != analysis["trusted_commit_sha256"]
        or publication.get("status") != PUBLICATION_STATUS
        or publication.get("commit_status") != PUBLICATION_COMMIT_STATUS
    ):
        raise FinalProvenanceSealError("sealed publication envelope changed")

    projection = payload.get("projection_envelope")
    if projection is not None:
        if not isinstance(projection, Mapping):
            raise FinalProvenanceSealError("projection envelope is malformed")
        _exact_fields(
            projection,
            {
                "path_root_role", "files", "trusted_commit_sha256", "file_count",
                "file_bundle_sha256", "projection_bundle_sha256", "status",
                "commit_status", "source_publication_commit_sha256",
            },
            label="sealed projection envelope",
        )
        projection_files = _validate_file_map(
            projection.get("files"), label="projection files",
            exact_count=len(profile.projection_mapping) + 2,
        )
        commit_key = f"generated/{PROJECTION_COMMIT_NAME}"
        expected_projection_files = {
            destination for _source, destination in profile.projection_mapping
        } | {
            f"generated/{PROJECTION_METADATA_NAME}", commit_key,
        }
        if (
            set(projection_files) != expected_projection_files
            or projection.get("path_root_role") != "explicit_manuscript_source_root"
            or not _strict_json_equal(
                projection.get("file_count"), len(projection_files)
            )
            or projection.get("file_bundle_sha256")
            != _bundle_hash(_hash_map(projection_files))
            or projection.get("trusted_commit_sha256")
            != projection_files[commit_key]["sha256"]
            or projection.get("source_publication_commit_sha256")
            != publication["trusted_commit_sha256"]
            or projection.get("status") != PROJECTION_STATUS
            or projection.get("commit_status") != PROJECTION_COMMIT_STATUS
        ):
            raise FinalProvenanceSealError("sealed projection envelope changed")
        _require_sha(
            projection.get("projection_bundle_sha256"),
            label="sealed projection bundle",
        )
    commit_hashes = {
        "amendment_v2": amendment_trio["commit"]["sha256"],
        "master": master["trusted_commit_sha256"],
        "analysis": analysis["trusted_commit_sha256"],
        "publication": publication["trusted_commit_sha256"],
    }
    if projection is not None:
        commit_hashes["projection"] = projection["trusted_commit_sha256"]
    if payload.get("upstream_commit_bundle_sha256") != _bundle_hash(commit_hashes):
        raise FinalProvenanceSealError("upstream commit bundle changed")
    return payload


def build_provenance_payload(
    *,
    master_path: str | Path,
    trusted_master_commit_sha256: str,
    analysis_dir: str | Path,
    trusted_analysis_commit_sha256: str,
    publication_dir: str | Path,
    trusted_publication_commit_sha256: str,
    projection_root: str | Path | None = None,
    projection_commit_path: str | Path | None = None,
    trusted_projection_commit_sha256: str | None = None,
    root: str | Path = ROOT,
    profile: TrustedProfile = PRODUCTION_PROFILE,
) -> dict[str, Any]:
    """Authenticate the chain and return an outcome-blind deterministic payload."""

    repo = _guard_root(root)
    _schema, _finalizer_sha = _verify_schema_and_finalizer(repo)
    sources, source_bundle = _verify_sources(repo, profile)
    amendment = _authenticate_amendment(repo, profile)
    master = _authenticate_master(
        master_path, root=repo, profile=profile, amendment=amendment,
        trusted_commit_sha256=trusted_master_commit_sha256,
    )
    analysis = _authenticate_analysis(
        analysis_dir, root=repo, profile=profile, master=master,
        trusted_commit_sha256=trusted_analysis_commit_sha256,
    )
    publication = _authenticate_publication(
        publication_dir, root=repo, profile=profile, analysis=analysis,
        trusted_commit_sha256=trusted_publication_commit_sha256,
    )
    projection = _authenticate_projection(
        projection_commit_path, root=repo, projection_root=projection_root,
        profile=profile, publication=publication,
        trusted_commit_sha256=trusted_projection_commit_sha256,
    )
    commit_hashes = {
        "amendment_v2": amendment["envelope"]["commit"]["sha256"],
        "master": master["envelope"]["trusted_commit_sha256"],
        "analysis": analysis["envelope"]["trusted_commit_sha256"],
        "publication": publication["envelope"]["trusted_commit_sha256"],
    }
    if projection is not None:
        commit_hashes["projection"] = projection["trusted_commit_sha256"]
    payload = {
        "schema_version": 1,
        "status": STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "branch": master["branch"],
        "scope": {
            "scientific_artifacts_decoded": False,
            "effect_values_interpreted": False,
            "effect_directions_interpreted": False,
            "winner_or_sign_selected": False,
            "operational_precision_state_authenticated": True,
            "cryptographic_provenance_only": True,
        },
        "audit_identity": {
            "launch_fingerprint": profile.launch_fingerprint,
            "final_M": 500,
        },
        "precision_state": master["precision_state"],
        "amendment_v2": amendment,
        "source_commitments": sources,
        "source_bundle_sha256": source_bundle,
        "master_envelope": master["envelope"],
        "analysis_envelope": analysis["envelope"],
        "publication_envelope": publication["envelope"],
        "projection_envelope": projection,
        "upstream_commit_bundle_sha256": _bundle_hash(commit_hashes),
    }
    return _validate_seal_payload(payload, profile=profile)


METADATA_FIELDS = {
    "schema_version", "status", "artifact_class", "artifact", "artifact_sha256",
    "artifact_bytes", "branch", "schema", "schema_sha256", "finalizer",
    "finalizer_sha256", "source_bundle_sha256", "upstream_commit_bundle_sha256",
    "master_commit_sha256", "analysis_commit_sha256", "publication_commit_sha256",
    "projection_commit_sha256", "environment", "generated_at_utc", "immutable",
}
COMMIT_FIELDS = {
    "schema_version", "status", "artifact", "artifact_sha256", "metadata",
    "metadata_sha256", "branch", "schema_sha256", "finalizer_sha256",
    "upstream_commit_bundle_sha256", "immutable", "committed_at_utc",
}


def _open_or_create_rooted_directory(
    path: Path, *, root: Path, label: str
) -> int:
    """Traverse/create a directory beneath root using only pinned dirfds."""

    required = (os.open, os.mkdir)
    if any(function not in os.supports_dir_fd for function in required):
        raise FinalProvenanceSealError("platform lacks required dirfd creation support")
    repo = _guard_root(root)
    candidate = _guard_path(
        path, root=repo, label=label, kind="dir", must_exist=False
    )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(repo, flags)
    try:
        for component in candidate.relative_to(repo).parts:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                    os.fsync(descriptor)
                    next_descriptor = os.open(component, flags, dir_fd=descriptor)
                except OSError as exc:
                    raise FinalProvenanceSealError(
                        f"{label} could not be created without following links"
                    ) from exc
            except OSError as exc:
                raise FinalProvenanceSealError(
                    f"{label} could not be opened without following links"
                ) from exc
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _create_pinned_output_directory(
    path: Path, *, root: Path
) -> tuple[int, int, os.stat_result]:
    """Create a new output directory and retain its parent and directory fds."""

    if any(
        function not in os.supports_dir_fd
        for function in (os.mkdir, os.open, os.stat, os.rmdir)
    ):
        raise FinalProvenanceSealError("platform lacks required output dirfd support")
    parent_descriptor = _open_or_create_rooted_directory(
        path.parent, root=root, label="seal output parent"
    )
    directory_descriptor: int | None = None
    created_identity: os.stat_result | None = None
    try:
        try:
            os.mkdir(path.name, mode=0o755, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise FinalProvenanceSealError(
                "seal output directory must be new and absent"
            ) from exc
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory_descriptor = os.open(
            path.name, flags, dir_fd=parent_descriptor
        )
        identity = os.fstat(directory_descriptor)
        created_identity = identity
        indexed = os.stat(
            path.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(identity.st_mode)
            or (identity.st_dev, identity.st_ino) != (indexed.st_dev, indexed.st_ino)
        ):
            raise FinalProvenanceSealError(
                "new seal output directory identity changed during creation"
            )
        os.fsync(parent_descriptor)
        return parent_descriptor, directory_descriptor, identity
    except BaseException:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        try:
            indexed = os.stat(
                path.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            if created_identity is not None and (
                indexed.st_dev, indexed.st_ino
            ) == (created_identity.st_dev, created_identity.st_ino):
                os.rmdir(path.name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
        except OSError:
            pass
        os.close(parent_descriptor)
        raise


def _atomic_write_new_at(
    directory_descriptor: int, name: str, value: bytes
) -> None:
    """Atomically publish one immutable file entirely through a pinned dirfd."""

    if SAFE_NAME_RE.fullmatch(name) is None:
        raise FinalProvenanceSealError("immutable destination name is not canonical")
    if any(
        function not in os.supports_dir_fd
        for function in (os.open, os.link, os.unlink, os.stat)
    ) or os.link not in os.supports_follow_symlinks:
        raise FinalProvenanceSealError("platform lacks required atomic dirfd support")
    temporary_name = f".{name}.{secrets.token_hex(12)}.tmp"
    temporary_descriptor: int | None = None
    linked = False
    complete = False
    try:
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_descriptor,
        )
        offset = 0
        while offset < len(value):
            written = os.write(temporary_descriptor, value[offset:])
            if written <= 0:
                raise FinalProvenanceSealError("immutable file write made no progress")
            offset += written
        os.fsync(temporary_descriptor)
        os.fchmod(temporary_descriptor, 0o444)
        os.fsync(temporary_descriptor)
        temporary_identity = os.fstat(temporary_descriptor)
        if (
            not stat.S_ISREG(temporary_identity.st_mode)
            or temporary_identity.st_size != len(value)
        ):
            raise FinalProvenanceSealError("immutable temporary file identity changed")
        os.link(
            temporary_name,
            name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        linked = True
        published = os.stat(
            name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(published.st_mode)
            or (published.st_dev, published.st_ino)
            != (temporary_identity.st_dev, temporary_identity.st_ino)
            or published.st_size != len(value)
            or published.st_mode & 0o222
        ):
            raise FinalProvenanceSealError("immutable published file identity changed")
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        temporary_name = ""
        os.fsync(directory_descriptor)
        complete = True
    except FileExistsError as exc:
        raise FinalProvenanceSealError(
            f"immutable destination already exists: {name}"
        ) from exc
    except OSError as exc:
        raise FinalProvenanceSealError(
            f"immutable dirfd publication failed: {name}"
        ) from exc
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
        if linked and not complete:
            # A failure after link must not leave a seemingly complete
            # immutable destination.
            try:
                os.unlink(name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass


def _assert_exact_inventory_at(
    directory_descriptor: int, expected: Iterable[str], *, label: str
) -> None:
    observed: set[str] = set()
    try:
        names = os.listdir(directory_descriptor)
    except OSError as exc:
        raise FinalProvenanceSealError(f"cannot scan pinned {label}") from exc
    for name in names:
        if SAFE_NAME_RE.fullmatch(name) is None:
            raise FinalProvenanceSealError(
                f"{label} contains a non-canonical filename: {name}"
            )
        try:
            identity = os.stat(
                name, dir_fd=directory_descriptor, follow_symlinks=False
            )
        except OSError as exc:
            raise FinalProvenanceSealError(
                f"{label} changed during pinned inventory"
            ) from exc
        if not stat.S_ISREG(identity.st_mode):
            raise FinalProvenanceSealError(
                f"{label} contains a non-regular or symlink entry: {name}"
            )
        observed.add(name)
    required = set(expected)
    if observed != required:
        raise FinalProvenanceSealError(
            f"{label} inventory changed: missing={sorted(required-observed)}, "
            f"extra={sorted(observed-required)}"
        )


def _assert_exact_immutable_files_at(
    directory_descriptor: int,
    expected: Mapping[str, bytes],
    *,
    label: str,
    path: Path | None = None,
    root: Path | None = None,
    parent_descriptor: int | None = None,
) -> None:
    """Jointly attest the pinned path, directory, and every immutable file."""

    joint_values = (path, root, parent_descriptor)
    if any(value is not None for value in joint_values) and not all(
        value is not None for value in joint_values
    ):
        raise FinalProvenanceSealError(
            "joint final attestation requires path, root, and parent dirfd"
        )
    parent_initial = (
        None if parent_descriptor is None else os.fstat(parent_descriptor)
    )
    directory_initial = os.fstat(directory_descriptor)
    _assert_exact_inventory_at(directory_descriptor, expected, label=label)
    opened: dict[str, tuple[int, os.stat_result]] = {}
    try:
        for name in sorted(expected):
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise FinalProvenanceSealError(
                    f"{label} immutable file cannot be pinned: {name}"
                ) from exc
            identity = os.fstat(descriptor)
            if (
                not stat.S_ISREG(identity.st_mode)
                or identity.st_mode & 0o777 != 0o444
                or identity.st_nlink != 1
                or identity.st_size != len(expected[name])
            ):
                os.close(descriptor)
                raise FinalProvenanceSealError(
                    f"{label} immutable file identity changed: {name}"
                )
            opened[name] = (descriptor, identity)

        for name in sorted(expected):
            descriptor, initial = opened[name]
            chunks: list[bytes] = []
            total = 0
            maximum = len(expected[name])
            while True:
                remaining = maximum - total
                chunk = os.read(
                    descriptor, min(1024 * 1024, remaining + 1)
                )
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum:
                    raise FinalProvenanceSealError(
                        f"{label} immutable file grew while reading: {name}"
                    )
                chunks.append(chunk)
            final = os.fstat(descriptor)
            stable_fields = (
                "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                "st_mtime_ns", "st_ctime_ns",
            )
            observed_raw = b"".join(chunks)
            if (
                total != len(expected[name])
                or observed_raw != expected[name]
                or _sha256_bytes(observed_raw)
                != _sha256_bytes(expected[name])
                or any(
                    getattr(final, field) != getattr(initial, field)
                    for field in stable_fields
                )
            ):
                raise FinalProvenanceSealError(
                    f"{label} immutable file content changed: {name}"
                )

        if path is not None and root is not None and parent_descriptor is not None:
            _assert_pinned_directory_path(
                path,
                root=root,
                parent_descriptor=parent_descriptor,
                directory_descriptor=directory_descriptor,
            )
            _assert_exact_inventory_at(
                directory_descriptor, expected, label=label
            )

        # Re-resolve every published name only after all bytes have been read.
        # The name must still point to the exact open inode; a post-inventory
        # unlink/replacement is therefore rejected before success.
        for name in sorted(expected):
            descriptor, initial = opened[name]
            indexed = os.stat(
                name, dir_fd=directory_descriptor, follow_symlinks=False
            )
            final = os.fstat(descriptor)
            if (
                (indexed.st_dev, indexed.st_ino)
                != (initial.st_dev, initial.st_ino)
                or (final.st_dev, final.st_ino)
                != (initial.st_dev, initial.st_ino)
                or final.st_mode & 0o777 != 0o444
                or final.st_nlink != 1
                or final.st_size != len(expected[name])
            ):
                raise FinalProvenanceSealError(
                    f"{label} immutable filename binding changed: {name}"
                )

        if path is not None and root is not None and parent_descriptor is not None:
            # The second lexical resolution is bracketed by open-file and
            # directory snapshots, making it the joint path/content point.
            _assert_pinned_directory_path(
                path,
                root=root,
                parent_descriptor=parent_descriptor,
                directory_descriptor=directory_descriptor,
            )

        # A directory-entry replacement after an earlier per-name check still
        # changes the pinned directory timestamps and is rejected here.
        for name in sorted(expected):
            descriptor, initial = opened[name]
            final = os.fstat(descriptor)
            stable_file_fields = (
                "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                "st_mtime_ns", "st_ctime_ns",
            )
            if any(
                getattr(final, field) != getattr(initial, field)
                for field in stable_file_fields
            ):
                raise FinalProvenanceSealError(
                    f"{label} immutable open-file identity changed: {name}"
                )
        directory_final = os.fstat(directory_descriptor)
        stable_directory_fields = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
            "st_mtime_ns", "st_ctime_ns",
        )
        if any(
            getattr(directory_final, field)
            != getattr(directory_initial, field)
            for field in stable_directory_fields
        ):
            raise FinalProvenanceSealError(
                f"{label} directory entries changed during final attestation"
            )
        if parent_initial is not None and parent_descriptor is not None:
            parent_final = os.fstat(parent_descriptor)
            if any(
                getattr(parent_final, field)
                != getattr(parent_initial, field)
                for field in stable_directory_fields
            ):
                raise FinalProvenanceSealError(
                    f"{label} parent directory changed during final attestation"
                )
    except OSError as exc:
        raise FinalProvenanceSealError(
            f"{label} changed during final immutable-file attestation"
        ) from exc
    finally:
        for descriptor, _identity in opened.values():
            os.close(descriptor)


def _assert_pinned_directory_path(
    path: Path,
    *,
    root: Path,
    parent_descriptor: int,
    directory_descriptor: int,
) -> None:
    """Require the lexical parent and destination to retain both pinned inodes."""

    pinned_parent = os.fstat(parent_descriptor)
    pinned = os.fstat(directory_descriptor)
    reopened_parent = _open_rooted_directory(
        path.parent, root=root, label="completed seal parent directory"
    )
    try:
        observed_parent = os.fstat(reopened_parent)
        if (pinned_parent.st_dev, pinned_parent.st_ino) != (
            observed_parent.st_dev, observed_parent.st_ino
        ):
            raise FinalProvenanceSealError(
                "completed seal parent path no longer names the pinned inode"
            )
    finally:
        os.close(reopened_parent)
    reopened = _open_rooted_directory(
        path, root=root, label="completed seal directory"
    )
    try:
        observed = os.fstat(reopened)
        if (pinned.st_dev, pinned.st_ino) != (observed.st_dev, observed.st_ino):
            raise FinalProvenanceSealError(
                "completed seal path no longer names the pinned output directory"
            )
    finally:
        os.close(reopened)


def _cleanup_pinned_output(
    parent_descriptor: int,
    directory_descriptor: int,
    directory_name: str,
    directory_identity: os.stat_result,
) -> None:
    """Remove every seal entry through its fd, even if its path was swapped."""

    for name in (COMMIT_NAME, METADATA_NAME, ARTIFACT_NAME):
        try:
            identity = os.stat(
                name, dir_fd=directory_descriptor, follow_symlinks=False
            )
            if stat.S_ISREG(identity.st_mode):
                os.unlink(name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        except OSError:
            pass
    try:
        os.fsync(directory_descriptor)
    except OSError:
        pass
    try:
        indexed = os.stat(
            directory_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (indexed.st_dev, indexed.st_ino) == (
            directory_identity.st_dev,
            directory_identity.st_ino,
        ):
            os.rmdir(directory_name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
    except OSError:
        pass


def finalize_provenance_seal(
    *,
    master_path: str | Path,
    trusted_master_commit_sha256: str,
    analysis_dir: str | Path,
    trusted_analysis_commit_sha256: str,
    publication_dir: str | Path,
    trusted_publication_commit_sha256: str,
    projection_root: str | Path | None = None,
    projection_commit_path: str | Path | None = None,
    trusted_projection_commit_sha256: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT,
    root: str | Path = ROOT,
    profile: TrustedProfile = PRODUCTION_PROFILE,
) -> dict[str, Path]:
    """Create a new immutable seal trio; existing destinations are refused."""

    repo = _guard_root(root)
    destination = _guard_path(
        output_dir, root=repo, label="seal output directory", kind="dir",
        must_exist=False,
    )
    protected = {
        _guard_path(Path(master_path).parent, root=repo, label="master directory", kind="dir"),
        _guard_path(analysis_dir, root=repo, label="analysis directory", kind="dir"),
        _guard_path(publication_dir, root=repo, label="publication directory", kind="dir"),
    }
    active_staging = (repo / "results/budget_toxicity_calibration_staging").resolve(strict=False)
    if (
        destination == active_staging
        or destination.is_relative_to(active_staging)
        or destination in protected
        or any(destination.is_relative_to(path) for path in protected)
    ):
        raise FinalProvenanceSealError("seal output overlaps a protected input/staging path")
    if destination.exists() or destination.is_symlink():
        raise FinalProvenanceSealError("seal output directory must be new and absent")
    payload = build_provenance_payload(
        master_path=master_path,
        trusted_master_commit_sha256=trusted_master_commit_sha256,
        analysis_dir=analysis_dir,
        trusted_analysis_commit_sha256=trusted_analysis_commit_sha256,
        publication_dir=publication_dir,
        trusted_publication_commit_sha256=trusted_publication_commit_sha256,
        projection_root=projection_root,
        projection_commit_path=projection_commit_path,
        trusted_projection_commit_sha256=trusted_projection_commit_sha256,
        root=repo,
        profile=profile,
    )
    schema, finalizer_sha = _verify_schema_and_finalizer(repo)
    del schema
    artifact = destination / ARTIFACT_NAME
    metadata_path = destination / METADATA_NAME
    commit_path = destination / COMMIT_NAME
    artifact_raw = _canonical_json_bytes(payload)
    generated_at = datetime.now(timezone.utc)
    projection = payload["projection_envelope"]
    metadata = {
        "schema_version": 1,
        "status": STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "artifact": ARTIFACT_NAME,
        "artifact_sha256": _sha256_bytes(artifact_raw),
        "artifact_bytes": len(artifact_raw),
        "branch": payload["branch"],
        "schema": SCHEMA_RELATIVE.as_posix(),
        "schema_sha256": SCHEMA_SHA256,
        "finalizer": FINALIZER_RELATIVE.as_posix(),
        "finalizer_sha256": finalizer_sha,
        "source_bundle_sha256": payload["source_bundle_sha256"],
        "upstream_commit_bundle_sha256": payload["upstream_commit_bundle_sha256"],
        "master_commit_sha256": payload["master_envelope"]["trusted_commit_sha256"],
        "analysis_commit_sha256": payload["analysis_envelope"]["trusted_commit_sha256"],
        "publication_commit_sha256": payload["publication_envelope"]["trusted_commit_sha256"],
        "projection_commit_sha256": (
            None if projection is None else projection["trusted_commit_sha256"]
        ),
        "environment": _runtime_environment(captured_at=generated_at),
        "generated_at_utc": generated_at.isoformat(),
        "immutable": True,
    }
    metadata_raw = _canonical_json_bytes(metadata)
    committed_at = datetime.now(timezone.utc)
    commit = {
        "schema_version": 1,
        "status": COMMIT_STATUS,
        "artifact": ARTIFACT_NAME,
        "artifact_sha256": _sha256_bytes(artifact_raw),
        "metadata": METADATA_NAME,
        "metadata_sha256": _sha256_bytes(metadata_raw),
        "branch": payload["branch"],
        "schema_sha256": SCHEMA_SHA256,
        "finalizer_sha256": finalizer_sha,
        "upstream_commit_bundle_sha256": payload["upstream_commit_bundle_sha256"],
        "immutable": True,
        "committed_at_utc": committed_at.isoformat(),
    }
    commit_raw = _canonical_json_bytes(commit)
    immutable_files = {
        ARTIFACT_NAME: artifact_raw,
        METADATA_NAME: metadata_raw,
        COMMIT_NAME: commit_raw,
    }
    parent_descriptor: int | None = None
    directory_descriptor: int | None = None
    directory_identity: os.stat_result | None = None
    try:
        parent_descriptor, directory_descriptor, directory_identity = (
            _create_pinned_output_directory(destination, root=repo)
        )
        _atomic_write_new_at(directory_descriptor, ARTIFACT_NAME, artifact_raw)
        _atomic_write_new_at(directory_descriptor, METADATA_NAME, metadata_raw)
        _atomic_write_new_at(directory_descriptor, COMMIT_NAME, commit_raw)
        _assert_exact_immutable_files_at(
            directory_descriptor, immutable_files,
            label="completed seal directory",
        )
        os.fsync(directory_descriptor)
        _assert_exact_immutable_files_at(
            directory_descriptor, immutable_files,
            label="completed seal directory at success",
            path=destination,
            root=repo,
            parent_descriptor=parent_descriptor,
        )
        # The joint path/dirfd/file re-read above is the success linearization
        # point; no path-based or content operation follows before fd close.
    except BaseException:
        if (
            parent_descriptor is not None
            and directory_descriptor is not None
            and directory_identity is not None
        ):
            _cleanup_pinned_output(
                parent_descriptor,
                directory_descriptor,
                destination.name,
                directory_identity,
            )
        raise
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    return {"artifact": artifact, "metadata": metadata_path, "commit": commit_path}


def authenticate_provenance_seal(
    artifact_path: str | Path,
    *,
    trusted_seal_commit_sha256: str,
    root: str | Path = ROOT,
    projection_root: str | Path | None = None,
    profile: TrustedProfile = PRODUCTION_PROFILE,
) -> dict[str, Any]:
    """Authenticate a seal by an external commit hash and replay its chain."""

    repo = _guard_root(root)
    trusted_commit = _require_sha(
        trusted_seal_commit_sha256, label="externally trusted seal commit hash"
    )
    artifact = _guard_path(
        artifact_path, root=repo, label="final seal artifact", kind="file"
    )
    if artifact.name != ARTIFACT_NAME:
        raise FinalProvenanceSealError("final seal artifact name changed")
    metadata_path = artifact.parent / METADATA_NAME
    commit_path = artifact.parent / COMMIT_NAME
    _assert_exact_inventory(
        artifact.parent, {ARTIFACT_NAME, METADATA_NAME, COMMIT_NAME},
        root=repo, label="final seal directory",
    )
    artifact_raw = _read_regular(artifact, root=repo, label="final seal artifact")
    metadata_raw = _read_regular(metadata_path, root=repo, label="final seal metadata")
    commit_raw = _read_regular(commit_path, root=repo, label="final seal commit")
    if _sha256_bytes(commit_raw) != trusted_commit:
        raise FinalProvenanceSealError(
            "seal commit differs from externally trusted SHA-256"
        )
    payload = _decode_canonical_json(artifact_raw, label="final seal artifact")
    metadata = _decode_canonical_json(metadata_raw, label="final seal metadata")
    commit = _decode_canonical_json(commit_raw, label="final seal commit")
    _exact_fields(metadata, METADATA_FIELDS, label="final seal metadata")
    _exact_fields(commit, COMMIT_FIELDS, label="final seal commit")
    _schema, finalizer_sha = _verify_schema_and_finalizer(repo)
    _validate_seal_payload(payload, profile=profile)
    expected_metadata = {
        "schema_version": 1,
        "status": STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "artifact": ARTIFACT_NAME,
        "artifact_sha256": _sha256_bytes(artifact_raw),
        "artifact_bytes": len(artifact_raw),
        "branch": payload["branch"],
        "schema": SCHEMA_RELATIVE.as_posix(),
        "schema_sha256": SCHEMA_SHA256,
        "finalizer": FINALIZER_RELATIVE.as_posix(),
        "finalizer_sha256": finalizer_sha,
        "source_bundle_sha256": payload["source_bundle_sha256"],
        "upstream_commit_bundle_sha256": payload["upstream_commit_bundle_sha256"],
        "master_commit_sha256": payload["master_envelope"]["trusted_commit_sha256"],
        "analysis_commit_sha256": payload["analysis_envelope"]["trusted_commit_sha256"],
        "publication_commit_sha256": payload["publication_envelope"]["trusted_commit_sha256"],
        "projection_commit_sha256": (
            None if payload["projection_envelope"] is None
            else payload["projection_envelope"]["trusted_commit_sha256"]
        ),
        "immutable": True,
    }
    for field, expected in expected_metadata.items():
        if not _strict_json_equal(metadata.get(field), expected):
            raise FinalProvenanceSealError(f"final seal metadata mismatch: {field}")
    expected_commit = {
        "schema_version": 1,
        "status": COMMIT_STATUS,
        "artifact": ARTIFACT_NAME,
        "artifact_sha256": _sha256_bytes(artifact_raw),
        "metadata": METADATA_NAME,
        "metadata_sha256": _sha256_bytes(metadata_raw),
        "branch": payload["branch"],
        "schema_sha256": SCHEMA_SHA256,
        "finalizer_sha256": finalizer_sha,
        "upstream_commit_bundle_sha256": payload["upstream_commit_bundle_sha256"],
        "immutable": True,
    }
    for field, expected in expected_commit.items():
        if not _strict_json_equal(commit.get(field), expected):
            raise FinalProvenanceSealError(f"final seal commit mismatch: {field}")
    generated, _committed = _assert_time_order(
        metadata.get("generated_at_utc"), commit.get("committed_at_utc"),
        label="final seal",
    )
    _validate_seal_environment(metadata.get("environment"), generated_at=generated)
    for path in (artifact, metadata_path, commit_path):
        if path.stat().st_mode & 0o222:
            raise FinalProvenanceSealError("final seal trio is not filesystem read-only")
    # Replay every upstream authentication from the immutable relative paths.
    master_path = repo / payload["master_envelope"]["envelope"]["artifact"]["path"]
    analysis_metadata = repo / payload["analysis_envelope"]["files"][ANALYSIS_METADATA_NAME]["path"]
    publication_metadata = repo / payload["publication_envelope"]["files"][PUBLICATION_METADATA_NAME]["path"]
    projection = payload["projection_envelope"]
    projection_path = None
    projection_hash = None
    if projection is not None:
        if projection_root is None:
            raise FinalProvenanceSealError(
                "sealed projection requires its explicit trusted manuscript root"
            )
        manuscript_root = _guard_root(projection_root)
        projection_path = manuscript_root / projection["files"][
            f"generated/{PROJECTION_COMMIT_NAME}"
        ]["path"]
        projection_hash = projection["trusted_commit_sha256"]
    rebuilt = build_provenance_payload(
        master_path=master_path,
        trusted_master_commit_sha256=payload["master_envelope"]["trusted_commit_sha256"],
        analysis_dir=analysis_metadata.parent,
        trusted_analysis_commit_sha256=payload["analysis_envelope"]["trusted_commit_sha256"],
        publication_dir=publication_metadata.parent,
        trusted_publication_commit_sha256=payload["publication_envelope"]["trusted_commit_sha256"],
        projection_root=projection_root,
        projection_commit_path=projection_path,
        trusted_projection_commit_sha256=projection_hash,
        root=repo,
        profile=profile,
    )
    if payload != rebuilt:
        raise FinalProvenanceSealError("final seal differs from replayed authenticated chain")
    return {
        "payload": payload,
        "metadata": metadata,
        "commit": commit,
        "hashes": {
            "artifact": _sha256_bytes(artifact_raw),
            "metadata": _sha256_bytes(metadata_raw),
            "commit": _sha256_bytes(commit_raw),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--trusted-master-commit-sha256", required=True)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--trusted-analysis-commit-sha256", required=True)
    parser.add_argument("--publication-dir", required=True, type=Path)
    parser.add_argument("--trusted-publication-commit-sha256", required=True)
    parser.add_argument("--projection-root", type=Path)
    parser.add_argument("--projection-commit", type=Path)
    parser.add_argument("--trusted-projection-commit-sha256")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--execute", action="store_true",
        help="write the new immutable trio; without this flag, authenticate only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "master_path": args.master,
        "trusted_master_commit_sha256": args.trusted_master_commit_sha256,
        "analysis_dir": args.analysis_dir,
        "trusted_analysis_commit_sha256": args.trusted_analysis_commit_sha256,
        "publication_dir": args.publication_dir,
        "trusted_publication_commit_sha256": args.trusted_publication_commit_sha256,
        "projection_root": args.projection_root,
        "projection_commit_path": args.projection_commit,
        "trusted_projection_commit_sha256": args.trusted_projection_commit_sha256,
    }
    if not args.execute:
        payload = build_provenance_payload(**common)
        print(f"authenticated branch: {payload['branch']}")
        print(f"upstream commit bundle: {payload['upstream_commit_bundle_sha256']}")
        print("no files written; pass --execute to create the immutable seal")
        return 0
    paths = finalize_provenance_seal(output_dir=args.output_dir, **common)
    print(f"final provenance seal: {paths['artifact']}")
    print(f"final provenance seal commit: {paths['commit']}")
    print(f"record externally trusted seal commit SHA-256: {_sha256_bytes(paths['commit'].read_bytes())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_NAME",
    "CAPPED_BRANCH",
    "COMMIT_NAME",
    "FinalProvenanceSealError",
    "METADATA_NAME",
    "PASS_BRANCH",
    "PRODUCTION_PROFILE",
    "TrustedProfile",
    "authenticate_provenance_seal",
    "build_provenance_payload",
    "finalize_provenance_seal",
]
