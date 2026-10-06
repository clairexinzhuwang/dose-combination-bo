"""Self-contained, one-page audit cards for saved trial records."""
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
    ("protocol_scaffold", "Access protocol", None),
    ("gamma", "tau", 2),
    ("stratum", "Stratum", 0),
    ("rec_efficacy_n_rows", "Recommendations, n", 0),
    ("recommendation_pct", "Recommend, %", 1),
    ("no_feasible_dose_pct", "Feasibility stop, %", 1),
    ("rec_efficacy", "Efficacy | recommend", 3),
    ("rec_unsafe_pct", "True-boundary-exceeding final recommendation, all trials, %", 1),
    ("unsafe_given_recommendation_pct", "True-boundary-exceeding | recommend, %", 1),
    ("dose_units", "Dose-location error (dose units) | recommend", 2),
    ("pcs_within1", "Within one grid unit, all trials, %", 1),
    ("initialization_patients_above", "Initialization assignments > latent boundary", 2),
    ("post_initialization_patients_above", "Post-initialization assignments > latent boundary", 2),
    ("expected_n", "Mean enrollment", 1),
    ("n_unique_doses", "Unique doses", 1),
    ("n_empty_gate_events", "Empty-gate events", 2),
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
            "pf": "posterior-feasibility gate",
            "ungated": "ungated fallback",
        },
        "noise": {
            "fixed": "fixed observation noise",
            "resample": "resampled observation noise",
        },
    }
    return labels.get(field, {}).get(value, str(value))


def _protocol_label(value):
    return {
        "lhs_fixed": "Full-panel access",
        "start_low_expansion": "Gradual panel access",
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
            f"Full-panel access: {warmups} continuous simulated assignments; "
            "first coordinate stratified, second uniform"
        )
    if "start_low_expansion" in scaffolds:
        cohort_sizes = ", ".join(str(value) for value in metadata["design"]["r_k"])
        labels.append(f"Gradual panel access: {cohort_sizes} simulated assignments at (0,0)")
    return escape("; ".join(labels))


def build_policy_card(records, *, source_path=None, title="SafeDoseBO operating-characteristic card"):
    """Return a self-contained HTML audit card.

    Estimates after ``| recommend`` exclude trials stopped without a recommendation.
    Every other displayed probability keeps all simulated trials in its denominator.
    Values are shown with Monte Carlo standard errors computed across independent
    replicate sets.
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
        ("Access protocols", _protocol_levels(metadata)),
        ("Gate threshold tau", _levels(metadata, "gamma")),
        ("Maximum sample size", _levels(metadata, "budget")),
        ("Cohort size", _levels(metadata, "r_k")),
        ("Full-panel initialization size", _levels(metadata, "warmup")),
        ("Initialization", _initialization_summary(metadata)),
        ("Grid", ", ".join(f"{v}x{v}" for v in metadata["design"]["grid_n"])),
        ("Gate / noise", f"{_levels(metadata, 'empty_gate')} / {_levels(metadata, 'noise')}"),
        ("Gradual-access region step", _levels(metadata, "region_step")),
        ("Consecutive empty gates before stop", _levels(metadata, "empty_gate_stop_after")),
        (
            "Exclude repeats during expansion",
            _levels(metadata, "exclude_repeats_during_expansion"),
        ),
        (
            "Independent replicate sets / trials",
            f"{metadata['seed_count']} replicate sets; "
            f"{metadata['record_count']} archived simulated-trial records",
        ),
        ("Source SHA-256", f"<code>{metadata['source_sha256']}</code>"),
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
<h1>{escape(title)}</h1><p class="sub">One-page operating-characteristic audit</p>
<h2>Design conditions</h2><dl>{condition_html}</dl>
<h2>Operating characteristics</h2>
<div class="table-wrap"><table><thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table></div>
<p class="note"><strong>Uncertainty and denominators.</strong> Each estimate is mean &plusmn; Monte Carlo SE computed across independent replicate sets. Recommended efficacy, dose-location error, and true-boundary-exceeding status among recommendations are conditional on a recommendation. Feasibility-stop, true-boundary-exceeding final-recommendation, and within-one-grid-unit percentages use all simulated trials.</p>
<p class="warning"><strong>Scope warning.</strong> This is a simulation comparison of acquisition-rule operating characteristics under specified dose-access settings, not a deployable clinical-trial design or a substitute for protocol-specific simulation. These simulation protocols do not model delayed outcomes, pharmacokinetics, permanent dose elimination, or acquisition-value stopping.</p>
<footer>Generated by SafeDoseBO. Verify the source hash and design conditions before comparing cards.</footer>
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
