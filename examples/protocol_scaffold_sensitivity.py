"""Small, resume-safe example of the prespecified dose-availability comparison.

The default four seeds are for learning the workflow, not for inference. Use the
original runner in the separate research archive at
``study/paper/run_protocol_scaffold_sensitivity.py`` for the complete historical
200-seed, 4,800-record-row analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import dose_combination_bo as bo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--out-dir", default="protocol_scaffold_example")
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for scaffold in ("lhs_fixed", "start_low_expansion"):
        path = out_dir / f"{scaffold}.json"
        records.extend(bo.sweep(
            ["cEI", "cKG", "cEI-tMSE"],
            surfaces=["osa"],
            seeds=args.seeds,
            gammas=[0.7, 0.9],
            strata=[0, 1],
            protocol_scaffold=scaffold,
            region_step=0.25,
            empty_gate_stop_after=3,
            exclude_repeats_during_expansion=True,
            out=str(path),
            parallel=args.parallel,
        ))
    combined = out_dir / "protocol_scaffold_sensitivity.json"
    combined.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    card = out_dir / "policy_card.html"
    bo.policy_card_from_file(combined, card)
    table = bo.oc_table(
        records,
        by=("policy", "protocol_scaffold", "gamma", "stratum"),
    )
    columns = [
        "policy", "protocol_scaffold", "gamma", "stratum",
        "recommendation_pct", "no_feasible_dose_pct", "rec_efficacy",
        "rec_unsafe_pct", "dose_units", "expected_n",
    ]
    if hasattr(table, "to_string"):
        print(table[columns].to_string(index=False))
    else:
        print(json.dumps(
            [{column: row[column] for column in columns} for row in table],
            indent=2,
        ))
    print(f"\nRaw records: {combined}\nPolicy card: {card}")


if __name__ == "__main__":
    main()
