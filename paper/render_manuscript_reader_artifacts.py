#!/usr/bin/env python3
"""Render manuscript-ready artifacts from authenticated result artifacts.

The scientific result tree remains immutable. This presentation layer reads the
corrected comparator-family outputs, the frozen calibration-publication bundle,
and the authenticated post-hoc projection that excludes four simulator-generated
initialization records from Family-A participant allocation. It then writes a
separate candidate manuscript tree. The frozen calibration analyzer and its
publication bundle therefore remain byte-for-byte unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

from paper import analyze_comparator_family as comparator
from paper import generate_budget_toxicity_calibration_publication as calibration
from paper import (
    project_budget_toxicity_calibration_family_a_post_initialization as postinit,
)


ROOT = Path(__file__).resolve().parents[1]
COMPARATOR_RESULTS = ROOT / "results" / "comparator_family_analysis"
CALIBRATION_RESULTS = (
    ROOT / "results" / "budget_toxicity_calibration_publication"
)
POSTINIT_RESULTS = (
    ROOT / "results" / "budget_toxicity_calibration_family_A_post_initialization"
)
CALIBRATION_CONTRACT = (
    ROOT / "paper" / "budget_toxicity_calibration_publication_contract.json"
)
ADDITIONAL_EMBEDDED_CALIBRATION_TMSE_PDFS = (
    "figure_budget_toxicity_calibration_absolute_gate_brier_profiles.pdf",
    "figure_budget_toxicity_calibration_calibration_penalties.pdf",
)

CALIBRATION_TABLE_INPUTS = {
    "family_a": ("budget_toxicity_calibration_family_A.tex", 54, 18),
    "family_b": ("budget_toxicity_calibration_family_B.tex", 48, 12),
    "family_c": ("budget_toxicity_calibration_family_C.tex", 16, 4),
    "policy_vs_cei_did": (
        "budget_toxicity_calibration_policy_vs_cei_did_36.tex",
        36,
        12,
    ),
}

COMPARATOR_INPUTS = {
    "aggregate": (
        "comparator_family_aggregate_contrasts.csv",
        12,
    ),
    "cellwise": (
        "comparator_family_cellwise_contrasts.csv",
        160,
    ),
    "surface": (
        "comparator_family_surface_bivariate.csv",
        8,
    ),
    "secondary_diagnostics": (
        "comparator_family_secondary_diagnostics.csv",
        720,
    ),
    "full_panel": (
        "comparator_family_full_panel_oc.csv",
        100,
    ),
    "gradual": (
        "comparator_family_gradual_contrasts.csv",
        72,
    ),
}

POSTINIT_FAMILY_A_LABEL_REPLACEMENTS = {
    "True-boundary-exceeding terminal recommendation (\\%)": (
        "Probability of an above-threshold final selection (\\%)"
    ),
    "Post-initialization above-boundary allocation (\\%)": (
        r"Post-initialization participant allocation (\%)"
    ),
}

COMPARATOR_FIGURE_REPLACEMENTS = {
    "tMSE-only": "tMSE",
    " (shared-gate reference)": "",
    "entropy reduction": "entropy rule",
    "Toxicity-rule stringency $\\tau$ (larger is stricter)": (
        "Model-based toxicity-rule stringency, $\\tau$"
    ),
    "OSA comparator-family profile": "Primary OSA-derived simulation",
    "Comparator in cKG minus comparator": "Comparator",
    "(a) Terminal recommendation": "(a) Final selection\nabove toxicity limit",
    "(b) Trial assignments": (
        "(b) Post-initialization allocation\nabove toxicity limit"
    ),
    "cKG minus comparator (pp)": (
        "Difference: cKG \N{MINUS SIGN} comparator\n(percentage points)"
    ),
    "cKG minus tMSE": "cKG \N{MINUS SIGN} tMSE",
    "cKG minus Entropy rule": "cKG \N{MINUS SIGN} entropy rule",
    "cKG minus comparator: assignment contrast (pp)": (
        "Allocation above toxicity limit\n"
        "(cKG \N{MINUS SIGN} comparator; percentage points)"
    ),
    "cKG minus comparator:\nterminal contrast (pp)": (
        "Final selection above toxicity limit\n"
        "(cKG \N{MINUS SIGN} comparator; percentage points)"
    ),
    "Cellwise terminal-versus-assignment contrasts": (
        "Differences by simulation setting and toxicity-rule stringency"
    ),
    "Surface-level terminal-versus-assignment profile": (
        "Surface-level operating-characteristic contrasts"
    ),
    "Terminal-versus-assignment comparator-family quadrant map": (
        "Allocation and final-selection differences by simulation setting"
    ),
    "Surface-level terminal-versus-assignment comparator-family profile": (
        "Operating-characteristic contrasts by response surface"
    ),
    "terminal-recommendation and assignment contrasts": (
        "final-selection and participant-allocation contrasts"
    ),
    "terminal-positive/assignment-negative": (
        "selection-positive/allocation-negative"
    ),
    "Cellwise point estimates": (
        "Point estimates for each simulation setting and toxicity-rule stringency"
    ),
}

FULL_PANEL_LABEL_REPLACEMENTS = {
    "Terminal, assignment, and recommendation outcomes": (
        "Above-threshold final-selection probability and participant-allocation outcomes"
    ),
    "Terminal exceed. (\\%)": "Above-threshold final-selection probability (\\%)",
    r"Post-init. allocation (\%)": (
        r"Post-initialization participant allocation (\%)"
    ),
}

GRADUAL_LABEL_REPLACEMENTS = {
    "Terminal boundary exceedance (pp)": (
        "Above-threshold final selection (pp)"
    ),
    r"$C_A$ (assignments)": r"$N_{\rm above}$ (participants)",
    r"$100C_A/N_{\max}$": r"$100N_{\rm above}/N_{\max}$",
}

NUMERIC_TOKEN = re.compile(
    r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


class ReaderArtifactError(RuntimeError):
    """Raised when the presentation layer cannot prove label-only derivation."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_sidecar(path: Path) -> None:
    sidecar = Path(str(path) + ".metadata.json")
    if not sidecar.is_file():
        raise ReaderArtifactError(f"missing metadata sidecar for {path}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    artifact_name = payload.get("artifact", payload.get("artifact_basename"))
    if artifact_name != path.name:
        raise ReaderArtifactError(f"sidecar artifact name mismatch for {path}")
    if payload.get("artifact_sha256") != _sha256(path):
        raise ReaderArtifactError(f"sidecar artifact hash mismatch for {path}")


def _read_csv(path: Path, expected_rows: int) -> list[dict[str, str]]:
    _verify_sidecar(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_rows:
        raise ReaderArtifactError(
            f"{path.name} has {len(rows)} rows; expected {expected_rows}"
        )
    return rows


def _read_json_rows(path: Path, *, key: str, expected_rows: int) -> list[dict[str, Any]]:
    _verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get(key)
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ReaderArtifactError(
            f"{path.name}:{key} does not contain {expected_rows} rows"
        )
    return [dict(row) for row in rows]


def _numeric_tokens(text: str) -> tuple[str, ...]:
    return tuple(NUMERIC_TOKEN.findall(text))


def _replace_exact(
    text: str, replacements: Mapping[str, str], *, expected_count: int
) -> str:
    output = text
    for old, new in replacements.items():
        count = output.count(old)
        if count != expected_count:
            raise ReaderArtifactError(
                f"reader label {old!r} occurs {count} times; "
                f"expected {expected_count}"
            )
        output = output.replace(old, new)
    return output


def _reader_figure_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    output = value
    for old, new in COMPARATOR_FIGURE_REPLACEMENTS.items():
        output = output.replace(old, new)
    if output == "OSA":
        return "OSA-derived"
    return output


@contextmanager
def _comparator_reader_figure_context() -> Any:
    """Scope exact label substitutions to manuscript figure rendering."""

    import matplotlib.axes
    import matplotlib.figure

    axes_class = matplotlib.axes.Axes
    figure_class = matplotlib.figure.Figure
    original_set_title = axes_class.set_title
    original_set_xlabel = axes_class.set_xlabel
    original_set_ylabel = axes_class.set_ylabel
    original_suptitle = figure_class.suptitle
    original_supxlabel = figure_class.supxlabel
    original_supylabel = figure_class.supylabel
    original_legend = figure_class.legend
    original_save_figure = comparator._save_figure
    original_tmse_label = comparator.POLICY_LABELS["tmse"]
    original_entropy_label = comparator.POLICY_LABELS["qBIG"]

    def set_title(axis: Any, label: Any, *args: Any, **kwargs: Any) -> Any:
        return original_set_title(
            axis, _reader_figure_text(label), *args, **kwargs
        )

    def set_ylabel(axis: Any, ylabel: Any, *args: Any, **kwargs: Any) -> Any:
        return original_set_ylabel(
            axis, _reader_figure_text(ylabel), *args, **kwargs
        )

    def set_xlabel(axis: Any, xlabel: Any, *args: Any, **kwargs: Any) -> Any:
        return original_set_xlabel(
            axis, _reader_figure_text(xlabel), *args, **kwargs
        )

    def suptitle(figure: Any, text: Any, *args: Any, **kwargs: Any) -> Any:
        return original_suptitle(
            figure, _reader_figure_text(text), *args, **kwargs
        )

    def supxlabel(figure: Any, text: Any, *args: Any, **kwargs: Any) -> Any:
        return original_supxlabel(
            figure, _reader_figure_text(text), *args, **kwargs
        )

    def supylabel(figure: Any, text: Any, *args: Any, **kwargs: Any) -> Any:
        return original_supylabel(
            figure, _reader_figure_text(text), *args, **kwargs
        )

    def legend(figure: Any, *args: Any, **kwargs: Any) -> Any:
        reader_args = list(args)
        if len(reader_args) >= 2 and isinstance(reader_args[1], Sequence):
            reader_args[1] = [
                _reader_figure_text(label) for label in reader_args[1]
            ]
        if "title" in kwargs:
            kwargs["title"] = _reader_figure_text(kwargs["title"])
        return original_legend(figure, *reader_args, **kwargs)

    def save_figure(
        figure: Any,
        stem: Path,
        *,
        title: str,
        subject: str,
    ) -> tuple[Path, Path, Path]:
        return original_save_figure(
            figure,
            stem,
            title=_reader_figure_text(title),
            subject=_reader_figure_text(subject),
        )

    axes_class.set_title = set_title
    axes_class.set_xlabel = set_xlabel
    axes_class.set_ylabel = set_ylabel
    figure_class.suptitle = suptitle
    figure_class.supxlabel = supxlabel
    figure_class.supylabel = supylabel
    figure_class.legend = legend
    comparator._save_figure = save_figure
    comparator.POLICY_LABELS["tmse"] = "tMSE"
    comparator.POLICY_LABELS["qBIG"] = "Entropy rule"
    try:
        yield
    finally:
        comparator.POLICY_LABELS["qBIG"] = original_entropy_label
        comparator.POLICY_LABELS["tmse"] = original_tmse_label
        comparator._save_figure = original_save_figure
        figure_class.legend = original_legend
        figure_class.supylabel = original_supylabel
        figure_class.supxlabel = original_supxlabel
        figure_class.suptitle = original_suptitle
        axes_class.set_ylabel = original_set_ylabel
        axes_class.set_xlabel = original_set_xlabel
        axes_class.set_title = original_set_title


@contextmanager
def _calibration_reader_figure_context() -> Any:
    """Replace legacy labels and branding in reader-facing calibration figures."""

    import matplotlib.axes

    axes_class = matplotlib.axes.Axes
    original_set_title = axes_class.set_title
    original_save_figure = calibration._save_figure

    def set_title(axis: Any, label: Any, *args: Any, **kwargs: Any) -> Any:
        return original_set_title(
            axis, _reader_figure_text(label), *args, **kwargs
        )

    def save_figure(
        figure: Any,
        *,
        output_dir: Path,
        stem: str,
        caption: str,
        formats: Sequence[str],
    ) -> list[Path]:
        outputs: list[Path] = []
        fixed_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
        for suffix in formats:
            path = calibration._stage_child(
                output_dir,
                f"{stem}.{suffix}",
                label=f"{stem} {suffix} figure",
            )
            if suffix == "png":
                metadata = {
                    "Title": caption,
                    "Author": "dual-combo-BO",
                    "Software": "Matplotlib",
                }
                figure.savefig(
                    path,
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                    transparent=False,
                    metadata=metadata,
                )
            elif suffix == "pdf":
                metadata = {
                    "Title": caption,
                    "Author": "dual-combo-BO",
                    "Creator": "dual-combo-BO",
                    "Producer": "Matplotlib",
                    "CreationDate": fixed_date,
                    "ModDate": fixed_date,
                }
                figure.savefig(
                    path,
                    format="pdf",
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                    transparent=False,
                    metadata=metadata,
                )
            else:
                raise ReaderArtifactError(
                    f"unsupported reader-figure format: {suffix}"
                )
            outputs.append(path)
        return outputs

    axes_class.set_title = set_title
    calibration._save_figure = save_figure
    try:
        yield
    finally:
        calibration._save_figure = original_save_figure
        axes_class.set_title = original_set_title


def _write_label_only_tex(
    target: Path,
    *,
    legacy: str,
    reader: str,
    expected_rows: int,
    row_prefixes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if _numeric_tokens(legacy) != _numeric_tokens(reader):
        raise ReaderArtifactError(
            f"numeric token stream changed in {target.name}"
        )
    if row_prefixes is None:
        legacy_rows = expected_rows
        reader_rows = expected_rows
    else:
        legacy_rows = sum(
            line.startswith(row_prefixes) and line.rstrip().endswith(r"\\")
            for line in legacy.splitlines()
        )
        reader_rows = sum(
            line.startswith(row_prefixes) and line.rstrip().endswith(r"\\")
            for line in reader.splitlines()
        )
    if legacy_rows != expected_rows or reader_rows != expected_rows:
        raise ReaderArtifactError(
            f"row count changed in {target.name}: "
            f"{legacy_rows} -> {reader_rows}, expected {expected_rows}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(reader, encoding="utf-8")
    return {
        "rows": expected_rows,
        "numeric_tokens": len(_numeric_tokens(reader)),
        "sha256": _sha256(target),
    }


def _render_comparator(output_dir: Path) -> dict[str, Any]:
    loaded: dict[str, list[dict[str, str]]] = {}
    input_hashes: dict[str, str] = {}
    for key, (name, expected_rows) in COMPARATOR_INPUTS.items():
        path = COMPARATOR_RESULTS / name
        loaded[key] = _read_csv(path, expected_rows)
        input_hashes[name] = _sha256(path)

    with _comparator_reader_figure_context():
        _plot_cei_efficacy_profile(
            loaded["secondary_diagnostics"],
            output_dir / "fig_cei_efficacy_profile",
        )
        comparator._plot_family_profile(
            loaded["cellwise"],
            loaded["aggregate"],
            output_dir / "fig_comparator_family_profile",
        )
        comparator._plot_surface_quadrant(
            loaded["surface"],
            output_dir / "fig_comparator_family_quadrant",
        )
        comparator._plot_cellwise_quadrant(
            loaded["cellwise"],
            output_dir / "fig_comparator_family_quadrant_cellwise",
        )
    shutil.copyfile(
        output_dir / "fig_cei_efficacy_profile.eps",
        output_dir / "Figure_1.eps",
    )
    shutil.copyfile(
        output_dir / "fig_comparator_family_profile.eps",
        output_dir / "Figure_2.eps",
    )

    generated = output_dir / "generated"
    legacy_aggregate = comparator._tex_aggregate_table(loaded["aggregate"])
    reader_aggregate = _replace_exact(
        legacy_aggregate,
        {
            "Equal-weight binding-surface aggregate": (
                "OSA, Logistic, and MARIPOSA-motivated summary"
            ),
            "Gaussian-bump stress test": (
                "Additional multimodal Gaussian-bump scenario"
            ),
        },
        expected_count=4,
    )
    reader_aggregate = _replace_exact(
        reader_aggregate,
        {
            "tMSE-only": "tMSE",
            "True-boundary-exceeding recommendation (pp)": (
                "Final selection above toxicity limit (pp)"
            ),
            r"$100N_{\rm above}^{\rm post}/(N_{\rm obs}-N_{\rm init})$ (pp)": (
                r"Post-initialization participant allocation ($100\times$ proportion; pp)"
            ),
        },
        expected_count=6,
    )
    aggregate_audit = _write_label_only_tex(
        generated / "comparator_family_aggregate_table.tex",
        legacy=legacy_aggregate,
        reader=reader_aggregate,
        expected_rows=12,
    )
    legacy_full = comparator._tex_full_panel_oc(loaded["full_panel"])
    reader_full = _replace_exact(
        legacy_full,
        FULL_PANEL_LABEL_REPLACEMENTS,
        expected_count=4,
    )
    reader_full = _replace_exact(
        reader_full,
        {"tMSE-only": "tMSE"},
        expected_count=40,
    )
    if reader_full.count("cEI--tMSE hybrid") != 40:
        raise ReaderArtifactError(
            "full-panel cEI--tMSE hybrid label count changed"
        )
    full_audit = _write_label_only_tex(
        generated / "comparator_family_full_panel_oc.tex",
        legacy=legacy_full,
        reader=reader_full,
        expected_rows=100,
    )
    legacy_gradual = comparator._tex_gradual_contrasts(loaded["gradual"])
    reader_gradual = _replace_exact(
        legacy_gradual,
        GRADUAL_LABEL_REPLACEMENTS,
        expected_count=4,
    )
    reader_gradual = _replace_exact(
        reader_gradual,
        {"tMSE-only": "tMSE"},
        expected_count=1,
    )
    gradual_audit = _write_label_only_tex(
        generated / "comparator_family_gradual_contrasts.tex",
        legacy=legacy_gradual,
        reader=reader_gradual,
        expected_rows=72,
    )
    return {
        "inputs": input_hashes,
        "aggregate_table": aggregate_audit,
        "full_panel_table": full_audit,
        "gradual_contrasts_table": gradual_audit,
    }


def _plot_cei_efficacy_profile(
    rows: Sequence[Mapping[str, Any]],
    stem: Path,
) -> tuple[Path, Path, Path]:
    """Plot the primary cKG--cEI final-selection efficacy comparison."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = [
        row
        for row in rows
        if row["comparator"] == "cEI"
        and row["outcome_id"] == "recommended_efficacy"
    ]
    if len(chosen) != 20:
        raise ReaderArtifactError(
            "cKG--cEI efficacy profile requires 20 surface-by-tau contrasts"
        )

    surface_order = ("osa", "efftox", "mariposa", "gbump")
    titles = {
        "osa": "(a) OSA-derived",
        "efftox": "(b) Logistic",
        "mariposa": "(c) MARIPOSA-motivated",
        "gbump": "(d) Gaussian bump",
    }
    plt.rcParams.update(
        {
            "font.size": 9.3,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.3,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.45), sharex=True)
    fig.subplots_adjust(
        left=0.13,
        right=0.985,
        bottom=0.15,
        top=0.96,
        hspace=0.34,
        wspace=0.28,
    )
    for axis, surface_id in zip(axes.flat, surface_order):
        points = sorted(
            [row for row in chosen if row["surface_id"] == surface_id],
            key=lambda row: float(row["tau"]),
        )
        if len(points) != 5:
            raise ReaderArtifactError(
                f"cKG--cEI efficacy profile is incomplete for {surface_id}"
            )
        x = [float(row["tau"]) for row in points]
        y = [float(row["estimate"]) for row in points]
        low = [float(row["ci95_low"]) for row in points]
        high = [float(row["ci95_high"]) for row in points]
        axis.errorbar(
            x,
            y,
            yerr=(
                [estimate - bound for estimate, bound in zip(y, low)],
                [bound - estimate for estimate, bound in zip(y, high)],
            ),
            color="#0072B2",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.1,
            markersize=4.8,
            linewidth=1.45,
            elinewidth=0.9,
            capsize=2.5,
        )
        axis.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
        axis.set_title(titles[surface_id], loc="left", fontweight="semibold")
        axis.set_xticks(comparator.GATES)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.supxlabel(
        r"Stringency of the model-based toxicity rule, $\tau$ "
        r"(higher is stricter)",
        y=0.035,
    )
    fig.supylabel(
        "Final-selection efficacy difference\n"
        "(cKG \N{MINUS SIGN} cEI; setting-specific efficacy units)",
        x=0.015,
    )
    paths = comparator._save_figure(
        fig,
        stem,
        title="Final-selection efficacy: cKG versus cEI",
        subject=(
            "Surface-specific paired differences in true mean efficacy at the "
            "final selected combination across five toxicity-rule stringency settings"
        ),
    )
    plt.close(fig)
    return paths


def _reader_postinit_family_a(source: str) -> str:
    """Apply journal-facing labels to the authenticated corrected table."""

    reader = source
    for old, new in POSTINIT_FAMILY_A_LABEL_REPLACEMENTS.items():
        count = reader.count(old)
        if count != 27:
            raise ReaderArtifactError(
                f"Family A label {old!r} occurs {count} times; expected 27"
            )
        reader = reader.replace(old, new)
    source_caption = (
        "Post-hoc supporting Family A projection. Allocation rows exclude the "
        "four fixed initialization observations and use the N minus 4 "
        "participant allocations; terminal recommendation rows are unchanged. "
        "This table does not replace the frozen prespecified primary analysis."
    )
    reader_caption = (
        "Complete cKG-minus-comparator sensitivity results. Final-selection "
        "rows are unchanged from the frozen analysis. Allocation rows exclude "
        "the 4 simulator-generated initialization records and use the "
        "post-initialization participant count as the denominator."
    )
    if reader.count(source_caption) != 1:
        raise ReaderArtifactError("Family A caption is not unique")
    reader = reader.replace(source_caption, reader_caption)
    reader = _replace_exact(
        reader,
        {r"$N$": "Total records"},
        expected_count=2,
    )
    reader = _replace_exact(
        reader,
        {"tMSE-only": "tMSE"},
        expected_count=18,
    )
    return reader


def _reader_calibration_table(
    legacy: str,
    *,
    expected_tmse_rows: int,
    expected_metric_rows: int,
) -> str:
    reader = _replace_exact(
        legacy,
        {"tMSE-only": "tMSE"},
        expected_count=expected_tmse_rows,
    )
    reader = _replace_exact(
        reader,
        {
            "True panel OBD excluded from terminal gate (\\%)": (
                "True grid OBD excluded from final gate (\\%)"
            ),
            "Panel feasibility Brier score $\\times$100": (
                "Candidate-grid feasibility Brier score $\\times$100"
            ),
        },
        expected_count=expected_metric_rows,
    )
    return reader


def _render_calibration(output_dir: Path) -> dict[str, Any]:
    primary_path = CALIBRATION_RESULTS / "budget_toxicity_calibration_all_118.json"
    secondary_path = (
        CALIBRATION_RESULTS
        / "budget_toxicity_calibration_plotted_secondary_72.json"
    )
    table_paths = {
        key: CALIBRATION_RESULTS / name
        for key, (name, _rows, _tmse_rows) in CALIBRATION_TABLE_INPUTS.items()
    }
    primary = _read_json_rows(primary_path, key="rows", expected_rows=118)
    secondary = _read_json_rows(secondary_path, key="rows", expected_rows=72)
    for table_path in table_paths.values():
        _verify_sidecar(table_path)
    contract = json.loads(CALIBRATION_CONTRACT.read_text(encoding="utf-8"))
    projection = postinit.generate_projection(
        analysis_dir=postinit.DEFAULT_ANALYSIS_DIR,
        output_dir=POSTINIT_RESULTS,
    )
    postinit_payload = json.loads(
        Path(projection["json"]).read_text(encoding="utf-8")
    )
    corrected_family_a = [dict(row) for row in postinit_payload["rows"]]
    if len(corrected_family_a) != 54:
        raise ReaderArtifactError(
            "post-initialization Family A projection does not contain 54 rows"
        )
    # The frozen plotting routine dispatches on its historical metric identifier.
    # Only that in-memory identifier is restored; the estimates, MCSEs, and
    # guarded intervals remain those of the authenticated corrected projection.
    corrected_for_plot: list[dict[str, Any]] = []
    for source_row in corrected_family_a:
        row = dict(source_row)
        if row["metric"] == postinit.POST_INITIALIZATION_ASSIGNMENT:
            row["metric"] = calibration.ASSIGNMENT
        corrected_for_plot.append(row)
    primary_for_plot = corrected_for_plot + [
        dict(row) for row in primary if row["family"] != "A"
    ]
    if len(primary_for_plot) != 118:
        raise ReaderArtifactError("corrected calibration figure input is incomplete")
    presentation_contract = copy.deepcopy(contract)
    for key, caption in presentation_contract["captions"].items():
        presentation_contract["captions"][key] = (
            str(caption)
            .replace(
                "terminal recommendation",
                "final selection",
            )
            .replace("assignments", "participant allocation")
            .replace("tMSE-only", "tMSE")
            .replace("numerically guarded ", "")
            .replace("bands", "intervals")
        )
    presentation_contract["policy_labels"]["tmse"] = "tMSE"
    presentation_contract["policy_labels"]["qBIG"] = "entropy rule"
    presentation_contract["labels"].update(
        {
            "audit_scope": (
                "Sensitivity to participant count and assumed toxicity-outcome variability"
            ),
            "metric_terminal": (
                "Final selected combination above\ntoxicity limit (%)"
            ),
            "metric_assignment": (
                "Post-initialization allocation\nabove toxicity limit (%)"
            ),
            "c_sigma_axis": (
                "Toxicity-outcome SD ratio (assumed/data-generating), $c_\\sigma$"
            ),
            "metric_exclusion": (
                "Efficacy-maximizing acceptable combination\n"
                "excluded by final toxicity rule (%)"
            ),
            "metric_brier": (
                "Toxicity-classification\nBrier score × 100"
            ),
            "x_budget": "Total response records",
        }
    )
    presentation_contract["captions"]["principal_profile"] = (
        "cKG-minus-tMSE and cKG-minus-entropy-rule differences with "
        "familywise simultaneous 95% max-t Monte Carlo intervals. "
        "Allocation excludes four simulator-generated initialization records; 20, 40, "
        "and 80 total records correspond to 16, 36, and 76 participants."
    )
    presentation_contract["captions"]["cei_reference"] = (
        "cKG-minus-cEI differences with familywise simultaneous 95% max-t "
        "Monte Carlo intervals. Allocation excludes four numerical "
        "initialization records; 20, 40, and 80 total records correspond to "
        "16, 36, and 76 participants."
    )
    with _calibration_reader_figure_context():
        calibration.make_publication_figures(
            primary_for_plot,
            secondary,
            output_dir=output_dir,
            contract=presentation_contract,
        )
        _matplotlib, plt = calibration._plot_setup()
        factors = tuple(calibration.FACTORS)
        colors = {
            float(key): value
            for key, value in presentation_contract["colors_by_c_sigma"].items()
        }
        labels = presentation_contract["labels"]
        policy_labels = presentation_contract["policy_labels"]
        comparators = ("tmse", "qBIG", "cEI")
        metrics = (calibration.TERMINAL, calibration.ASSIGNMENT)
        family_a = [row for row in primary_for_plot if row["family"] == "A"]
        if len(family_a) != 54:
            raise ReaderArtifactError("combined sensitivity figure requires 54 rows")
        fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.3), sharex=True)
        for column, comparator_name in enumerate(comparators):
            for row_index, metric in enumerate(metrics):
                axis = axes[row_index, column]
                for factor in factors:
                    points = sorted(
                        [
                            row
                            for row in family_a
                            if row["comparator"] == comparator_name
                            and row["metric"] == metric
                            and float(row["assumed_toxicity_noise_sd_factor"])
                            == factor
                        ],
                        key=lambda row: int(row["budget_n"]),
                    )
                    calibration._draw_line(
                        axis,
                        points,
                        factor=factor,
                        color=colors[factor],
                    )
                axis.axhline(0.0, color="black", linewidth=0.8)
                if row_index == 0:
                    axis.set_title(
                        f"cKG \N{MINUS SIGN} {policy_labels[comparator_name]}",
                        fontweight="normal",
                    )
                if column == 0:
                    if metric == calibration.TERMINAL:
                        axis.set_ylabel(
                            "Above-limit final selection\n"
                            "(cKG \N{MINUS SIGN} comparator; pp)"
                        )
                    else:
                        axis.set_ylabel(
                            "Above-limit allocation\n"
                            "(cKG \N{MINUS SIGN} comparator; pp)"
                        )
                if row_index == 1:
                    axis.set_xlabel("Post-initialization participants")
                axis.set_xticks(calibration.BUDGETS, (16, 36, 76))
                axis.grid(color="#D9D9D9", linewidth=0.6)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            title=labels["c_sigma_axis"],
        )
        fig.subplots_adjust(bottom=0.18, hspace=0.26, wspace=0.27)
        combined_caption = (
            "Sensitivity to participant count and assumed toxicity-outcome variability"
        )
        calibration._save_figure(
            fig,
            output_dir=output_dir,
            stem="figure_budget_toxicity_calibration_all_comparators",
            caption=combined_caption,
            formats=presentation_contract["figures"]["formats"],
        )
        plt.close(fig)
    tmse_figure_hashes: dict[str, str] = {}
    for name in ADDITIONAL_EMBEDDED_CALIBRATION_TMSE_PDFS:
        rendered_path = output_dir / name
        if not rendered_path.is_file():
            raise ReaderArtifactError(
                f"embedded calibration figure was not rendered: {name}"
            )
        tmse_figure_hashes[name] = _sha256(rendered_path)

    family_a_path = Path(projection["tex"])
    source_family_a = family_a_path.read_text(encoding="utf-8")
    reader_family_a = _reader_postinit_family_a(source_family_a)
    family_a_audit = _write_label_only_tex(
        output_dir / "generated" / "budget_toxicity_calibration_family_A.tex",
        legacy=source_family_a,
        reader=reader_family_a,
        expected_rows=54,
        row_prefixes=(
            "tMSE-only &",
            "tMSE &",
            "cKG &",
            "Entropy reduction &",
            "cEI &",
        ),
    )
    additional_table_audits: dict[str, dict[str, Any]] = {}
    for key in ("family_b", "family_c", "policy_vs_cei_did"):
        name, expected_rows, expected_tmse_rows = CALIBRATION_TABLE_INPUTS[key]
        table_path = table_paths[key]
        legacy = table_path.read_text(encoding="utf-8")
        reader = _reader_calibration_table(
            legacy,
            expected_tmse_rows=expected_tmse_rows,
            expected_metric_rows=expected_rows // 2,
        )
        additional_table_audits[key] = _write_label_only_tex(
            output_dir / "generated" / name,
            legacy=legacy,
            reader=reader,
            expected_rows=expected_rows,
            row_prefixes=(
                "tMSE-only &",
                "tMSE &",
                "cKG &",
                "Entropy reduction &",
                "cEI &",
            ),
        )
    return {
        "inputs": {
            primary_path.name: _sha256(primary_path),
            secondary_path.name: _sha256(secondary_path),
            Path(projection["json"]).name: _sha256(Path(projection["json"])),
            Path(projection["tex"]).name: _sha256(Path(projection["tex"])),
            Path(projection["metadata"]).name: _sha256(
                Path(projection["metadata"])
            ),
            Path(projection["commit"]).name: _sha256(
                Path(projection["commit"])
            ),
            **{
                table_path.name: _sha256(table_path)
                for table_path in table_paths.values()
            },
            CALIBRATION_CONTRACT.name: _sha256(CALIBRATION_CONTRACT),
        },
        "family_a_table": family_a_audit,
        "additional_tables": additional_table_audits,
        "additional_embedded_tmse_figures": tmse_figure_hashes,
    }


def render(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    comparator_report = _render_comparator(output_dir)
    calibration_report = _render_calibration(output_dir)
    output_paths = sorted(
        path for path in output_dir.rglob("*") if path.is_file()
    )
    return {
        "status": "READER_LABEL_PRESENTATION_ARTIFACTS_RENDERED",
        "output_dir": str(output_dir),
        "comparator": comparator_report,
        "calibration": calibration_report,
        "outputs": {
            path.relative_to(output_dir).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in output_paths
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = render(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
