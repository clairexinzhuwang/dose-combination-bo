#!/usr/bin/env python3
"""Regenerate the compact main-paper OSA table from a formal projection.

The ``cEI-tMSE`` machine ID denotes the study-defined reference schedule that
alternates cEI and tMSE across successive adaptive cohorts. Compact outputs use
the reader-facing alias ``cEI–tMSE`` rather than presenting it as a named method.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import warnings

import dose_combination_bo as sdb

from paper.analysis_record_bundle import load_projection_record

POLICIES = ("cEI", "cKG", "cEI-tMSE")
POLICY_LABELS = {
    "cEI": "cEI",
    "cKG": "cKG",
    "cEI-tMSE": "cEI--tMSE",
}
GATE_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)
PRIMARY_TAU = 0.7
OSA_RECORD_NAME = "osa_main.json.zst"


def _pm(row, field, digits):
    return f"{row[field]:.{digits}f} $\\pm$ {row[f'{field}_mcse']:.{digits}f}"


def _validate_osa_main_records(records):
    expected = {
        (policy, seed, stratum, tau)
        for policy in POLICIES
        for seed in range(200)
        for stratum in (0, 1)
        for tau in GATE_LEVELS
    }
    observed = set()
    for index, row in enumerate(records):
        if row.get("sim") != "osa":
            raise ValueError(f"record {index} is not an OSA trial")
        try:
            key = (
                row["policy"],
                int(row["seed"]),
                int(row["stratum"]),
                float(row["gamma"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"record {index} has an invalid OSA design identity") from exc
        if key in observed:
            raise ValueError(f"duplicate OSA main design cell: {key}")
        observed.add(key)
    if observed != expected:
        raise ValueError("OSA main records do not contain the complete declared factorial")


def _write_metadata(path, selected, artifact_type, provenance):
    metadata = sdb.write_artifact_metadata(
        path, selected, artifact_type=artifact_type
    )
    commitment = provenance["selected_record_commitment"]
    metadata.update({
        "source_path": OSA_RECORD_NAME,
        "source_sha256": commitment["compressed_sha256"],
        "source_uncompressed_sha256": commitment["uncompressed_sha256"],
        "formal_projection": {
            "projection_metadata_path": provenance["projection_metadata_path"],
            "projection_metadata_sha256": provenance["projection_metadata_sha256"],
            "projection_fingerprint": provenance["projection_fingerprint"],
            "formal_manifest_sha256": provenance["formal_manifest_sha256"],
            "record": OSA_RECORD_NAME,
            "record_commitment": dict(commitment),
        },
        "estimand": {
            "sim": "osa",
            "posterior_feasibility_threshold_tau": PRIMARY_TAU,
            "policies": list(POLICIES),
            "reader_facing_policy_alias": {"cEI-tMSE": "cEI–tMSE"},
            "full_panel_reference_schedule": {
                "scope": "full-panel access only",
                "maximum_assignments": 40,
                "initialization_assignments": 4,
                "cohort_size": 2,
                "scheduled_adaptive_cohort_positions": 18,
                "cEI_scheduled_positions": 9,
                "tMSE_scheduled_positions": 9,
                "empty_gate_override": (
                    "An empty eligible gate can invoke the common "
                    "greatest-feasibility fallback instead of the scheduled acquisition."
                ),
            },
        },
    })
    Path(path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def analyze(projection_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, provenance = load_projection_record(projection_dir, OSA_RECORD_NAME)
    _validate_osa_main_records(records)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="no 'random' policy rows were supplied.*", category=UserWarning
        )
        table = sdb.oc_table(records, sim="osa", gamma=PRIMARY_TAU, by=("policy",))
    rows = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    rows.sort(key=lambda row: POLICIES.index(row["policy"]))

    csv_path = output_dir / "main_osa_primary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tex_path = output_dir / "main_osa_primary.tex"
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Terminal recommendation} & \multicolumn{1}{c}{Trial operation} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}",
        r"Policy & \shortstack{Recommended\\efficacy} & \shortstack{Dose-location\\error} & \shortstack{Within 1 grid unit\\of continuous OBD (\%)} & \shortstack{True-boundary-exceeding\\final recommendation (\%)} & \shortstack{Above-boundary\\simulated assignments (\%)} \\",
        r"\midrule",
    ]
    for row in rows:
        values = (
            _pm(row, "rec_efficacy", 3),
            _pm(row, "dose_units", 2),
            _pm(row, "pcs_within1", 1),
            _pm(row, "rec_unsafe_pct", 1),
            _pm(row, "pct_patients_above", 1),
        )
        lines.append(f"{POLICY_LABELS[row['policy']]} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    selected = [row for row in records if float(row["gamma"]) == PRIMARY_TAU]
    if len(selected) != 1_200:
        raise ValueError("the tau=0.7 primary OSA estimand must contain 1,200 trials")
    for artifact, kind in ((csv_path, "table_data"), (tex_path, "latex_table")):
        _write_metadata(
            artifact.with_suffix(artifact.suffix + ".metadata.json"), selected,
            kind, provenance,
        )
    print(f"wrote {csv_path}")
    print(f"wrote {tex_path}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection", help="authenticated formal_projection-* directory")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    analyze(args.projection, args.out_dir)


if __name__ == "__main__":
    main()
