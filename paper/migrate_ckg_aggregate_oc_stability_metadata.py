#!/usr/bin/env python3
"""Apply the metadata-only classification migration to the aggregate-OC raw file.

The numerical execution completed before the packaging classification fields
were added.  This one-use migration verifies the exact pre-migration raw and
sidecar hashes, preserves the execution-time source hashes, adds only
classification/interpretation metadata, and proves that the canonical outcome
payload is unchanged before and after the migration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRE_MIGRATION_RAW_SHA256 = (
    "e44179341cf762500a132154720dcd83e9ed174d7ab391109ee9a892e4f8fc14"
)
PRE_MIGRATION_RAW_BYTES = 3_248_457
PRE_MIGRATION_SIDECAR_SHA256 = (
    "391f15d717d22ab62a23257368e2c37b918249497e2fcadcc002ad30ca2c6b1d"
)
GENERATION_RUNNER_SHA256 = (
    "9ff2820d4a46cda16b24d6e2c04d15bcb1b46807cf7f94e583d26a3160021c52"
)
ARTIFACT_CLASS = "computational_diagnostic"
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


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _outcome_payload_sha256(raw: dict[str, Any]) -> str:
    """Hash all archived/new result rows plus the production-equivalence evidence."""
    payload = {
        "new_arm_rows": raw["new_arm_rows"],
        "production_equivalence": raw["production_equivalence"],
        "reference_rows": raw["reference_rows"],
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def migrate(raw_path: Path) -> dict[str, str | int | bool]:
    raw_path = raw_path.resolve()
    sidecar_path = raw_path.with_suffix(raw_path.suffix + ".metadata.json")
    runner_path = ROOT / "paper/run_ckg_aggregate_oc_stability.py"

    observed_raw_sha = _sha256_file(raw_path)
    observed_sidecar_sha = _sha256_file(sidecar_path)
    observed_runner_sha = _sha256_file(runner_path)
    if observed_raw_sha != PRE_MIGRATION_RAW_SHA256:
        raise ValueError(f"unexpected pre-migration raw hash: {observed_raw_sha}")
    if raw_path.stat().st_size != PRE_MIGRATION_RAW_BYTES:
        raise ValueError("unexpected pre-migration raw byte count")
    if observed_sidecar_sha != PRE_MIGRATION_SIDECAR_SHA256:
        raise ValueError(f"unexpected pre-migration sidecar hash: {observed_sidecar_sha}")
    if observed_runner_sha != GENERATION_RUNNER_SHA256:
        raise ValueError("execution-time runner bytes are not preserved")

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("artifact_sha256") != observed_raw_sha:
        raise ValueError("pre-migration sidecar does not authenticate the raw artifact")
    if raw.get("source_sha256", {}).get(
        "paper/run_ckg_aggregate_oc_stability.py"
    ) != GENERATION_RUNNER_SHA256:
        raise ValueError("raw generation provenance does not name the execution runner")
    if sidecar.get("source_sha256", {}).get(
        "paper/run_ckg_aggregate_oc_stability.py"
    ) != GENERATION_RUNNER_SHA256:
        raise ValueError("sidecar generation provenance does not name the execution runner")
    if len(raw.get("reference_rows", [])) != 1_200:
        raise ValueError("expected 1,200 reused reference rows")
    if len(raw.get("new_arm_rows", [])) != 800:
        raise ValueError("expected 800 new diagnostic executions")

    outcome_before = _outcome_payload_sha256(raw)
    migration_source = Path(__file__).resolve()
    migration = {
        "fields_added": [
            "artifact_class",
            "logical_trial_records",
            "inventory_note",
            "design.fantasy_bank_conditionality",
            "fantasy_bank_conditionality (sidecar only)",
            "metadata_schema_migration",
        ],
        "generation_runner_sha256": GENERATION_RUNNER_SHA256,
        "migration_kind": "metadata_only_no_numerical_rows_changed",
        "migration_source": "paper/migrate_ckg_aggregate_oc_stability_metadata.py",
        "migration_source_sha256": _sha256_file(migration_source),
        "outcome_payload_hash_definition": (
            "SHA-256 of canonical indented/sorted JSON containing new_arm_rows, "
            "production_equivalence, and reference_rows"
        ),
        "outcome_payload_sha256_after": outcome_before,
        "outcome_payload_sha256_before": outcome_before,
        "outcomes_recomputed": False,
        "pre_migration_raw_bytes": PRE_MIGRATION_RAW_BYTES,
        "pre_migration_raw_sha256": PRE_MIGRATION_RAW_SHA256,
        "pre_migration_sidecar_sha256": PRE_MIGRATION_SIDECAR_SHA256,
    }

    raw["artifact_class"] = ARTIFACT_CLASS
    raw["logical_trial_records"] = 0
    raw["inventory_note"] = INVENTORY_NOTE
    raw["design"]["fantasy_bank_conditionality"] = FANTASY_BANK_CONDITIONALITY
    raw["metadata_schema_migration"] = migration
    outcome_after = _outcome_payload_sha256(raw)
    if outcome_after != outcome_before:
        raise RuntimeError("outcome payload changed during metadata-only migration")
    _atomic_json(raw_path, raw)

    sidecar["artifact_class"] = ARTIFACT_CLASS
    sidecar["logical_trial_records"] = 0
    sidecar["inventory_note"] = INVENTORY_NOTE
    sidecar["design"]["fantasy_bank_conditionality"] = FANTASY_BANK_CONDITIONALITY
    sidecar["fantasy_bank_conditionality"] = FANTASY_BANK_CONDITIONALITY
    sidecar["metadata_schema_migration"] = migration
    sidecar["artifact_sha256"] = _sha256_file(raw_path)
    sidecar["artifact_bytes"] = raw_path.stat().st_size
    _atomic_json(sidecar_path, sidecar)

    return {
        "artifact_bytes": raw_path.stat().st_size,
        "artifact_sha256": _sha256_file(raw_path),
        "generation_runner_sha256": GENERATION_RUNNER_SHA256,
        "outcome_payload_sha256": outcome_before,
        "outcomes_recomputed": False,
        "sidecar_sha256": _sha256_file(sidecar_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=(
            ROOT
            / "results/ckg_aggregate_oc_stability/ckg_aggregate_oc_stability_raw.json"
        ),
    )
    args = parser.parse_args(argv)
    print(json.dumps(migrate(args.raw), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
