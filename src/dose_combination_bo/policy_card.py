"""Self-contained HTML summaries of saved trial simulations."""
from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
import warnings

from .oc import oc_table
from .reporting import artifact_metadata, write_artifact_metadata


_CARD_GROUPS = (
    "policy", "sim", "protocol_scaffold", "gamma", "stratum", "mode",
    "noise", "kap", "budget", "r_k", "grid_n", "warmup", "empty_gate",
    "region_step", "empty_gate_stop_after", "exclude_repeats_during_expansion",
)

_COLUMNS = (
    ("policy_label", "Policy", None),
    ("sim", "Surface", None),
    ("protocol_scaffold", "Dose-availability design", None),
    ("gamma", "tau", 2),
    ("stratum", "Stratum", 0),
    ("rec_efficacy_n_rows", "Trials with a final selection, n", 0),
    ("recommendation_pct", "Trials with a final selection, %", 1),
    ("no_feasible_dose_pct", "Stopped without a final selection, %", 1),
    (
        "rec_efficacy",
        "True mean efficacy at final selection, among trials with a selection",
        3,
    ),
    (
        "rec_unsafe_pct",
        "Final selection above toxicity limit, all initiated trials, %",
        1,
    ),
    (
        "unsafe_given_recommendation_pct",
        "Final selection above toxicity limit, among trials with a selection, %",
        1,
    ),
    ("dose_units", "Distance to legacy continuous-domain reference, among selections (dose units)", 2),
    (
        "pcs_within1",
        "Selection within one grid unit of legacy reference, all initiated trials, %",
        1,
    ),
    (
        "initialization_patients_above",
        "Mean no. of above-threshold initialization records or participants (design dependent)",
        2,
    ),
    (
        "post_initialization_patients_above",
        "Mean no. of participants assigned to above-threshold combinations after initialization",
        2,
    ),
    ("expected_n", "Mean response records or enrollment (design dependent)", 1),
    (
        "n_unique_doses",
        "Mean no. of distinct response locations or assigned combinations (design dependent)",
        1,
    ),
    ("n_empty_gate_events", "Mean no. of empty qualifying-set occasions", 2),
)


def _records(value):
    if hasattr(value, "to_dict"):
        value = value.to_dict("records")
    else:
        value = list(value)
    if not value:
        raise ValueError("records must contain at least one trial")
    return value


def _fmt(value, digits=None):
    if value is None:
        return "--"
    if isinstance(value, float) and not math.isfinite(value):
        return "--"
    if digits is None:
        return escape(str(value))
    return f"{float(value):.{digits}f}"


def _estimate(row, metric, digits):
    if metric == "protocol_scaffold":
        return escape(_protocol_label(row.get(metric)))
    if metric in {"sim", "empty_gate", "noise"}:
        return escape(_reader_label(metric, row.get(metric)))
    value = row.get(metric)
    if digits is None or metric not in {
        "recommendation_pct", "no_feasible_dose_pct", "rec_efficacy",
        "rec_unsafe_pct", "unsafe_given_recommendation_pct", "dose_units",
        "pcs_within1", "initialization_patients_above",
        "post_initialization_patients_above", "expected_n", "n_unique_doses",
        "n_empty_gate_events",
    }:
        return _fmt(value, digits)
    mcse = row.get(f"{metric}_mcse")
    if value is None or mcse is None:
        return "--"
    try:
        if not math.isfinite(float(value)) or not math.isfinite(float(mcse)):
            return "--"
    except (TypeError, ValueError):
        return "--"
    return f"{float(value):.{digits}f} &plusmn; {float(mcse):.{digits}f}"


def _levels(metadata, field):
    values = metadata["design"][field]
    return ", ".join(escape(_reader_label(field, value)) for value in values)


def _reader_label(field, value):
    labels = {
        "sim": {"osa": "OSA"},
        "empty_gate": {
            "pf": (
                "assign the protocol-eligible combination with the largest standardized "
                "toxicity margin"
            ),
            "ungated": (
                "compatibility option that lets the acquisition rank without the "
                "toxicity rule"
            ),
        },
        "noise": {
            "fixed": "same as the data-generating outcome variance",
            "learned": "outcome variance estimated from simulated data",
            "resample": "outcome variance resampled",
        },
        "exclude_repeats_during_expansion": {True: "yes", False: "no"},
    }
    return labels.get(field, {}).get(value, str(value))


def _protocol_label(value):
    return {
        "lhs_fixed": "All candidate combinations available",
        "start_low_expansion": "Gradual expansion",
    }.get(value, str(value))


def _protocol_levels(metadata):
    values = metadata["design"]["protocol_scaffold"]
    return ", ".join(escape(_protocol_label(value)) for value in values)


def _initialization_summary(metadata):
    labels = []
    scaffolds = metadata["design"]["protocol_scaffold"]
    if "lhs_fixed" in scaffolds:
        warmups = ", ".join(str(value) for value in metadata["design"]["warmup"])
        labels.append(
            f"All candidate combinations available: {warmups} off-grid, simulator-generated "
            "initialization records; these are neither participant data nor assignments "
            "to administrable doses and are excluded from participant-allocation summaries"
        )
    if "start_low_expansion" in scaffolds:
        cohort_sizes = ", ".join(str(value) for value in metadata["design"]["r_k"])
        labels.append(
            f"Gradual expansion: starting cohort of {cohort_sizes} actual participants "
            "at the lowest grid combination (0, 0)"
        )
    return escape("; ".join(labels))


def build_policy_card(
    records,
    *,
    source_path=None,
    title="Simulation summary by acquisition function",
):
    """Return a self-contained HTML summary of simulated trials.

    Metrics labelled "among trials with a selection" exclude trials stopped
    without a final selection. Other displayed probabilities include all initiated
    trials. Monte Carlo standard errors are computed across independent simulation
    replicates.
    """
    records = _records(records)
    metadata = artifact_metadata(records, source_path=source_path, artifact_type="policy_card")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="no 'random' policy rows were supplied.*", category=UserWarning
        )
        table = oc_table(records, by=_CARD_GROUPS)
    rows = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    headers = "".join(f"<th>{escape(label)}</th>" for _, label, _ in _COLUMNS)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{_estimate(row, field, digits)}</td>"
            for field, _, digits in _COLUMNS
        )
        body.append(f"<tr>{cells}</tr>")
    conditions = (
        ("Surfaces", _levels(metadata, "sim")),
        ("Dose-availability designs", _protocol_levels(metadata)),
        (
            "Required probability that mean toxicity does not exceed its limit, "
            "tau (larger is stricter)",
            _levels(metadata, "gamma"),
        ),
        ("Maximum response records or enrollment (design dependent)", _levels(metadata, "budget")),
        ("Cohort size", _levels(metadata, "r_k")),
        (
            "Off-grid simulator-generated initialization records when all combinations are available",
            _levels(metadata, "warmup"),
        ),
        ("Initialization", _initialization_summary(metadata)),
        (
            "Candidate dose-combination grid",
            ", ".join(f"{v}x{v}" for v in metadata["design"]["grid_n"]),
        ),
        (
            "Assignment rule when no protocol-eligible combination meets the toxicity rule",
            _levels(metadata, "empty_gate"),
        ),
        ("Outcome variance in the analysis", _levels(metadata, "noise")),
        ("Expansion step per cohort (d1 + d2)", _levels(metadata, "region_step")),
        (
            "Gradual expansion only: stop after this many consecutive empty qualifying-set occasions",
            _levels(metadata, "empty_gate_stop_after"),
        ),
        (
            "Avoid revisiting a combination during expansion",
            _levels(metadata, "exclude_repeats_during_expansion"),
        ),
        (
            "Simulation replicates and trials",
            f"{metadata['seed_count']} independent simulation replicates; "
            f"{metadata['record_count']} simulated-trial records",
        ),
    )
    condition_html = "".join(
        f"<div><dt>{escape(label)}</dt><dd>{value}</dd></div>"
        for label, value in conditions
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title>
<style>
:root{{--ink:#17212b;--muted:#5d6b78;--line:#d7dee5;--wash:#f3f7fa;--accent:#176b87}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);font:14px/1.42 system-ui,-apple-system,sans-serif;background:#fff}}
main{{max-width:1500px;margin:0 auto;padding:28px}} h1{{font-size:27px;margin:0 0 4px}} h2{{font-size:17px;margin:24px 0 10px}}
.sub{{color:var(--muted);margin:0 0 20px}} dl{{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:8px;margin:0}}
dl div{{background:var(--wash);border:1px solid var(--line);padding:9px 11px;border-radius:6px}} dt{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}} dd{{margin:3px 0 0}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:6px}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:7px 8px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}} th{{position:sticky;top:0;background:#eaf2f6;color:#173b49;font-size:11px}} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}
.note,.warning{{padding:11px 13px;border-radius:6px;margin-top:12px}} .note{{background:var(--wash)}} .warning{{border-left:4px solid #bc6b15;background:#fff7ed}}
code{{font-size:11px;word-break:break-all}} footer{{margin-top:18px;color:var(--muted);font-size:12px}}
@media(max-width:850px){{dl{{grid-template-columns:1fr 1fr}} main{{padding:16px}}}}
@media print{{main{{padding:0}} .table-wrap{{overflow:visible}} th{{position:static}}}}
</style></head><body><main>
<h1>{escape(title)}</h1><p class="sub">Operating characteristics from simulated trials</p>
<h2>Design conditions</h2><dl>{condition_html}</dl>
<h2>Operating characteristics</h2>
<div class="table-wrap"><table><thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table></div>
<p class="note"><strong>Uncertainty and denominators.</strong> Each estimate is mean &plusmn; Monte Carlo SE computed across independent simulation replicates. True mean efficacy at final selection, reference-point distance, and selection-conditional percentages exclude trials stopped without a selection. Stopping and above-limit selection outcomes use all initiated trials. The stored continuous-domain point is a legacy location reference, not the finite-grid trial target. When all combinations are available, initialization uses off-grid simulator-generated records rather than participant data; use post-initialization fields for participant allocation. Under gradual expansion, the starting cohort contains actual participants.</p>
<p class="warning"><strong>Scope.</strong> This summary compares acquisition rules under specified simulated dose-availability designs. It is not a clinical-trial design or a substitute for protocol-specific simulation. The posterior-probability cutoff tau is a simulation setting, not a clinically calibrated safety probability. If the final qualifying set is empty, the simulator returns a largest-margin fallback selection; this selection does not meet the model-based toxicity criterion. More generally, any final selection may be untried because the simulator imposes no minimum-exposure rule. An above-limit combination has true mean toxicity above the prespecified simulation limit; allocation and final-selection outcomes do not count observed toxicity events. The simulations do not model delayed outcomes, pharmacokinetics, permanent dose elimination, or acquisition-value stopping.</p>
<footer>Generated by the accompanying reproducibility software.</footer>
</main></body></html>"""


def policy_card_from_file(input_path, output_path):
    """Load JSON records, write the HTML card, and write its metadata sidecar."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open(encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("results JSON must contain a list of per-trial records")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_policy_card(records, source_path=input_path), encoding="utf-8"
    )
    sidecar = output_path.with_suffix(output_path.suffix + ".metadata.json")
    write_artifact_metadata(
        sidecar, records, source_path=input_path, artifact_type="policy_card"
    )
    return output_path, sidecar


__all__ = ["build_policy_card", "policy_card_from_file"]
