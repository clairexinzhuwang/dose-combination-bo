#!/usr/bin/env python3
"""Analyze the exploratory OSA cEI-heavy schedule-composition sensitivity.

The analyzer selects the 800 corresponding ``tau=0.7`` OSA/latent/full-panel
appearances of the study-defined reference schedule alternating cEI and tMSE
across successive adaptive cohorts and combines them with
the 200 fixed cEI--cEI--tMSE-cycle appearances.  Both inputs must come from one
authenticated formal projection.  It reports descriptive matched-replicate
contrasts only.  No ratio is tuned, no p-value is used, and the pointwise Monte Carlo
intervals are unadjusted.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

import dose_combination_bo as sdb

# Executing ``python paper/analyze_....py`` puts ``paper/``, not the package root,
# on sys.path. Add the root only so this additive analyzer can reuse the exact
# run-design constants without installing ``paper`` as a distribution package.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from paper.analysis_record_bundle import load_projection_bundle
from paper.run_schedule_ratio_sensitivity import (
    FROZEN,
    MAX_SEEDS,
    POLICY as NEW_POLICY,
    POST_HOC_FIELDS,
    SCHEDULE_CYCLE,
    STRATA,
)

SIXWAY_RECORD_NAME = "gate_sixway_supplemental.json.zst"
SCHEDULE_RECORD_NAME = "schedule_ratio_sensitivity_2to1.json.zst"

REFERENCE_ALIASES = {
    "cEI": "cEI",
    "cKG1fix": "cKG",
    "GBE": "cEI-tMSE-1to1",
    "straddle": "tMSE",
}
POLICY_ORDER = (
    "cEI",
    "tMSE",
    "cEI-tMSE-1to1",
    NEW_POLICY,
    "cKG",
)
TABLE_POLICY_ORDER = (
    "cEI",
    "cKG",
    "tMSE",
    "cEI-tMSE-1to1",
    NEW_POLICY,
)
POLICY_LABELS = {
    "cEI": "pure cEI",
    "tMSE": "pure tMSE",
    "cEI-tMSE-1to1": "cEI–tMSE",
    NEW_POLICY: "cEI-heavy (12 cEI, 6 tMSE cohorts)",
    "cKG": "cKG",
}
POLICY_CYCLES = {
    "cEI": ("cEI",),
    "tMSE": ("tMSE",),
    "cEI-tMSE-1to1": ("cEI", "tMSE"),
    NEW_POLICY: SCHEDULE_CYCLE,
    "cKG": ("cKG",),
}
CONTRASTS = (
    ("cKG", "cEI-tMSE-1to1", "existing main comparator"),
    ("cKG", NEW_POLICY, "new schedule-composition check"),
    ("cKG", "tMSE", "pure-tMSE component reference"),
    ("cKG", "cEI", "pure-cEI component reference"),
    (NEW_POLICY, "cEI-tMSE-1to1", "schedule-composition check"),
)
OUTCOMES = (
    (
        "terminal_threshold_event",
        "True-boundary-exceeding final recommendation",
        "_terminal_threshold_pct",
        "percentage points",
    ),
    (
        "above_boundary_assignment_count",
        "Above-boundary simulated assignments (all 40 assignments)",
        "_above_boundary_assignment_count",
        "assignments per 40-assignment simulated trial",
    ),
    (
        "above_boundary_assignment_percentage",
        "Above-boundary simulated assignments (all 40 assignments)",
        "_above_boundary_assignment_pct",
        "percentage points",
    ),
)


def _rows(table):
    return table.to_dict("records") if hasattr(table, "to_dict") else list(table)


def _validate_corresponding_design(row, *, label):
    legacy_defaults = {
        "protocol_scaffold": "lhs_fixed",
        "region_step": 0.25,
        "empty_gate_stop_after": 3,
        "exclude_repeats_during_expansion": True,
    }
    for field, value in FROZEN.items():
        if row.get(field, legacy_defaults.get(field)) != value:
            raise ValueError(
                f"{label} has {field}={row.get(field)!r}; expected {value!r}"
            )
    seed, stratum = row.get("seed"), row.get("stratum")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed not in range(MAX_SEEDS):
        raise ValueError(f"{label} has an invalid seed")
    if stratum not in STRATA:
        raise ValueError(f"{label} has an invalid stratum")
    terminal = row.get("rec_unsafe")
    assignments = row.get("toxic")
    if terminal not in (0, 1, False, True):
        raise ValueError(f"{label} has a non-binary terminal threshold event")
    if not isinstance(assignments, (int, float)) or not 0 <= assignments <= 40:
        raise ValueError(f"{label} has an invalid above-boundary assignment count")


def _select_reference(records):
    selected = []
    seen = set()
    for index, source in enumerate(records):
        if not (
            source.get("sim") == "osa"
            and source.get("mode") == "latent"
            and source.get("gamma") == 0.7
            and source.get("policy") in REFERENCE_ALIASES
        ):
            continue
        _validate_corresponding_design(source, label=f"six-policy row {index}")
        policy = REFERENCE_ALIASES[source["policy"]]
        key = (policy, int(source["seed"]), int(source["stratum"]))
        if key in seen:
            raise ValueError(f"duplicate six-policy reference cell: {key}")
        seen.add(key)
        row = dict(source)
        row["policy"] = policy
        row["policy_source"] = "authenticated formal six-policy projection"
        row["source_policy_code"] = source["policy"]
        selected.append(row)
    expected = {
        (policy, seed, stratum)
        for policy in REFERENCE_ALIASES.values()
        for seed in range(MAX_SEEDS)
        for stratum in STRATA
    }
    if seen != expected:
        raise ValueError(
            "six-policy source lacks the exact 800 corresponding reference cells"
        )
    return selected


def _validate_new(records, *, require_full=True):
    if not isinstance(records, list) or not records:
        raise ValueError("new cEI-heavy source must be a non-empty JSON list")
    seen = set()
    for index, row in enumerate(records):
        _validate_corresponding_design(row, label=f"new row {index}")
        if row.get("policy") != NEW_POLICY:
            raise ValueError(f"new row {index} is not the fixed cEI-heavy policy")
        for field, value in POST_HOC_FIELDS.items():
            if row.get(field) != value:
                raise ValueError(
                    f"new row {index} lacks exact post-hoc metadata {field!r}"
                )
        if row.get("recommendation_made") is not True or row.get("n_enrolled") != 40:
            raise ValueError(f"new full-panel row {index} has incomplete terminal semantics")
        key = (int(row["seed"]), int(row["stratum"]))
        if key in seen:
            raise ValueError(f"duplicate new cEI-heavy cell: {key}")
        seen.add(key)
    if require_full:
        expected = {(seed, stratum) for seed in range(MAX_SEEDS) for stratum in STRATA}
        if seen != expected:
            raise ValueError("new cEI-heavy source must contain the exact 200-cell matrix")
    return sorted(seen)


def _combine(reference, new_records):
    combined = [dict(row) for row in reference]
    for source in new_records:
        row = dict(source)
        row["policy_source"] = "authenticated formal cEI-heavy schedule projection"
        row["source_policy_code"] = NEW_POLICY
        combined.append(row)
    expected = {
        (policy, seed, stratum)
        for policy in POLICY_ORDER
        for seed in range(MAX_SEEDS)
        for stratum in STRATA
    }
    observed = set()
    for index, row in enumerate(combined):
        key = (row["policy"], int(row["seed"]), int(row["stratum"]))
        if key in observed:
            raise ValueError(f"duplicate combined schedule cell: {key}")
        observed.add(key)
        count = float(row["toxic"])
        budget = float(row["budget"])
        row["_terminal_threshold_pct"] = 100.0 * float(row["rec_unsafe"])
        row["_above_boundary_assignment_count"] = count
        row["_above_boundary_assignment_pct"] = 100.0 * count / budget
    if observed != expected:
        raise ValueError("combined schedule analysis is not the exact 1,000-cell matrix")
    return combined


def _summary_rows(records):
    output = []
    for outcome_id, outcome_label, metric, unit in OUTCOMES:
        summaries = {
            row["policy"]: row
            for row in _rows(sdb.summarize(records, metric=metric, by=("policy",)))
        }
        for policy in POLICY_ORDER:
            summary = summaries[policy]
            mean = float(summary[f"{metric}_mean"])
            mcse = float(summary[f"{metric}_se"])
            output.append({
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "schedule_cycle": " -> ".join(POLICY_CYCLES[policy]),
                "policy_source": (
                    "formal cEI-heavy schedule projection"
                    if policy == NEW_POLICY
                    else "formal six-policy projection"
                ),
                "outcome_id": outcome_id,
                "outcome": outcome_label,
                "unit": unit,
                "mean": mean,
                "mcse": mcse,
                "pointwise_95_low": mean - 1.96 * mcse,
                "pointwise_95_high": mean + 1.96 * mcse,
                "stratum_trials": 200,
                "independent_seeds": int(summary["n"]),
                "tau": 0.7,
                "post_hoc_descriptive": True,
            })
    return output


def _contrast_rows(records, summaries):
    mean_lookup = {
        (row["policy"], row["outcome_id"]): float(row["mean"])
        for row in summaries
    }
    output = []
    for policy_a, policy_b, role in CONTRASTS:
        for outcome_id, outcome_label, metric, unit in OUTCOMES:
            contrast = _rows(sdb.paired_contrast(
                records,
                policy_a,
                policy_b,
                metric=metric,
                by=("sim", "gamma"),
            ))[0]
            mean = float(contrast["difference_mean"])
            mcse = float(contrast["difference_se"])
            output.append({
                "contrast": f"{POLICY_LABELS[policy_a]} minus {POLICY_LABELS[policy_b]}",
                "contrast_role": role,
                "policy_a": policy_a,
                "policy_b": policy_b,
                "outcome_id": outcome_id,
                "outcome": outcome_label,
                "unit": unit,
                "policy_a_mean": mean_lookup[(policy_a, outcome_id)],
                "policy_b_mean": mean_lookup[(policy_b, outcome_id)],
                "difference_mean": mean,
                "difference_mcse": mcse,
                "pointwise_95_low": mean - 1.96 * mcse,
                "pointwise_95_high": mean + 1.96 * mcse,
                "eligible_stratum_pairs": int(contrast["n_pairs"]),
                "independent_seeds": int(contrast["n"]),
                "tau": 0.7,
                "post_hoc_descriptive": True,
            })
    return output


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=path.name + ".", suffix=".tmp",
        dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty analysis CSV")
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, stream.getvalue())


def _write_json(path, payload):
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _latex_table(summary_rows):
    lookup = {
        (row["policy"], row["outcome_id"]): row
        for row in summary_rows
    }
    tex_labels = {
        "cEI": "pure cEI",
        "cKG": "cKG",
        "tMSE": "pure tMSE",
        "cEI-tMSE-1to1": "cEI--tMSE",
        NEW_POLICY: "cEI--cEI--tMSE cycle",
    }
    lines = [
        "% Generated by analyze_schedule_ratio_sensitivity.py; do not hand-edit.",
        "\\begin{table}[H]",
        "\\centering\\small",
        "\\begin{tabular}{@{}lcc@{}}",
        "\\toprule",
        "Policy & \\shortstack{True-boundary-exceeding\\\\final recommendation (\\%)} & "
        "\\shortstack{Above-boundary simulated\\\\assignments (count/40)} \\\\",
        "\\midrule",
    ]
    for policy in TABLE_POLICY_ORDER:
        terminal = lookup[(policy, "terminal_threshold_event")]
        assignments = lookup[(policy, "above_boundary_assignment_count")]
        lines.append(
            f"{tex_labels[policy]} & "
            f"${float(terminal['mean']):.1f}{{\\pm}}{float(terminal['mcse']):.1f}$ & "
            f"${float(assignments['mean']):.2f}{{\\pm}}{float(assignments['mcse']):.2f}$ "
            "\\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Exploratory post hoc schedule-composition sensitivity for the full-panel "
        "OSA latent-gate cell at $\\tau=0.7$. Entries are means $\\pm$ Monte Carlo SE "
        "computed across 100 independent matched replicate sets after averaging the "
        "two strata within each set (200 simulated trials per policy). The "
        "cEI--tMSE is the study-defined reference schedule. Under this full-panel "
        "design, $N_{\\max}=40$, four initialization assignments, and cohorts of two "
        "give 18 scheduled adaptive-cohort positions: nine cEI and nine tMSE. "
        "An empty eligible gate can instead invoke the common greatest-feasibility "
        "fallback. The "
        "cEI--cEI--tMSE cycle gives 12 cEI and "
        "six tMSE adaptive cohorts. It was specified after the original study results "
        "were examined and was not tuned. Assignment counts include all 40 "
        "simulated assignments. This analysis is descriptive and nonconfirmatory.}",
        "\\label{tab:schedule_ratio_posthoc}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def analyze(projection_dir, out_dir):
    projection_dir = Path(projection_dir)
    out_dir = Path(out_dir)
    bundle = load_projection_bundle(projection_dir)
    sixway_records = bundle.records[SIXWAY_RECORD_NAME]
    new_records = bundle.records[SCHEDULE_RECORD_NAME]
    if len(sixway_records) != 24_000:
        raise ValueError("six-policy formal projection must contain exactly 24,000 rows")
    _validate_new(new_records, require_full=True)
    reference = _select_reference(sixway_records)
    combined = _combine(reference, new_records)
    summaries = _summary_rows(combined)
    contrasts = _contrast_rows(combined, summaries)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "schedule_ratio_operating_characteristics.csv"
    contrast_path = out_dir / "schedule_ratio_paired_contrasts.csv"
    table_path = out_dir / "schedule_ratio_table.tex"
    _write_csv(summary_path, summaries)
    _write_csv(contrast_path, contrasts)
    _atomic_text(table_path, _latex_table(summaries))

    manifest = {
        "schema_version": 2,
        "artifact_type": "post_hoc_schedule_ratio_analysis",
        "post_hoc_descriptive": True,
        "confirmatory": False,
        "ratio_tuned": False,
        "post_hoc_timing_status": (
            "The cEI-heavy [cEI,cEI,tMSE] cycle and post-hoc contrasts were specified "
            "after the original study results were examined."
        ),
        "design": {
            **FROZEN,
            "strata": list(STRATA),
            "seeds": list(range(MAX_SEEDS)),
            "policies": list(POLICY_ORDER),
            "new_schedule_cycle": list(SCHEDULE_CYCLE),
        },
        "full_panel_reference_schedule": {
            "policy": "cEI-tMSE-1to1",
            "reader_facing_alias": "cEI–tMSE",
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
            "scope": "full-panel access only",
        },
        "record_counts": {
            "formal_sixway_reference_appearances": len(reference),
            "formal_2to1_appearances": len(new_records),
            "combined_rows": len(combined),
        },
        "uncertainty": {
            "unit": "independent matched replicate set after averaging its two strata",
            "interval": "mean plus/minus 1.96 paired Monte Carlo SE",
            "multiplicity_adjustment": "none; descriptive pointwise intervals",
            "p_values": "not computed",
        },
        "outcome_note": (
            "The assignment count includes all 40 simulated assignments. Its "
            "percentage is exactly 100*count/40 and is reported only to supply both "
            "requested scales, not as an independent outcome."
        ),
        "contrasts": [
            {"policy_a": a, "policy_b": b, "role": role}
            for a, b, role in CONTRASTS
        ],
        "inputs": {
            "formal_projection": {
                "projection_metadata_path": bundle.provenance[
                    "projection_metadata_path"
                ],
                "projection_metadata_sha256": bundle.provenance[
                    "projection_metadata_sha256"
                ],
                "projection_fingerprint": bundle.provenance[
                    "projection_fingerprint"
                ],
                "formal_manifest_sha256": bundle.provenance[
                    "formal_manifest_sha256"
                ],
                "selected_records": {
                    filename: bundle.provenance["record_files"][filename]
                    for filename in (SIXWAY_RECORD_NAME, SCHEDULE_RECORD_NAME)
                },
            },
        },
        "provenance": {
            "all_analysis_rows_authenticated_by_formal_projection": True,
            "row_authentication": (
                "All 1,000 analysis appearances were loaded from one authenticated "
                "formal projection; 800 are selected six-policy reference appearances "
                "and 200 are fixed cEI-heavy schedule appearances."
            ),
            "reference_selection": (
                "OSA latent-gate tau=0.7 full-panel cells for cEI, cKG1fix, GBE, "
                "and straddle, mapped to the analysis policy labels"
            ),
            "analyzer": Path(__file__).name,
            "current_analysis_source_sha256": sdb.file_sha256(Path(__file__)),
            "core_api_or_default_policy_changed": False,
        },
        "outputs": {
            summary_path.name: {
                "rows": len(summaries),
                "sha256": sdb.file_sha256(summary_path),
            },
            contrast_path.name: {
                "rows": len(contrasts),
                "sha256": sdb.file_sha256(contrast_path),
            },
            table_path.name: {
                "rows": len(TABLE_POLICY_ORDER),
                "sha256": sdb.file_sha256(table_path),
            },
        },
    }
    manifest_path = out_dir / "schedule_ratio_analysis.metadata.json"
    _write_json(manifest_path, manifest)
    print(f"operating characteristics: {summary_path}")
    print(f"paired contrasts: {contrast_path}")
    print(f"TeX table: {table_path}")
    print(f"analysis metadata: {manifest_path}")
    return summaries, contrasts, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection-dir",
        required=True,
        help="committed authenticated formal-projection directory",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    analyze(args.projection_dir, args.out_dir)


if __name__ == "__main__":
    main()
