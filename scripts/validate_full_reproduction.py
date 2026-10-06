#!/usr/bin/env python3
"""Run full-stack archive checks after lightweight integrity.

Unlike ``validate_integrity.py``, this entry point intentionally imports the
historical analysis modules and requires NumPy, torch, gpytorch, and the package
test dependencies.  It recomputes truth-surface record semantics and then runs
the archived package tests.  It does not recreate the 55,480 archived record rows,
which include documented reuse of seed-indexed cells across supporting files.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence


sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import validate_integrity as integrity  # noqa: E402


FULL_STACK_MODULES = (
    "numpy",
    "scipy",
    "torch",
    "gpytorch",
    "pytest",
)
ANALYSIS_REGENERATION_RUNTIME = {
    "python": "3.13.5",
    "numpy": "2.1.3",
    "scipy": "1.15.3",
    "torch": "2.10.0",
    "gpytorch": "1.15.2",
    "pandas": "2.2.3",
    "matplotlib": "3.10.0",
    "ray": "2.54.1",
    "botorch": "0.17.2",
    "linear-operator": "0.6.1",
    "zstandard": "0.23.0",
}


def _require_full_stack() -> None:
    missing = []
    for name in FULL_STACK_MODULES:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            missing.append(name)
    if missing:
        raise integrity.IntegrityError(
            "full reproduction requires the archived scientific/test environment; "
            f"missing modules: {', '.join(missing)}"
        )


def _require_analysis_regeneration_runtime() -> None:
    observed = {"python": platform.python_version()}
    for distribution in ANALYSIS_REGENERATION_RUNTIME:
        if distribution == "python":
            continue
        try:
            observed[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise integrity.IntegrityError(
                f"analysis regeneration requires {distribution}"
            ) from exc
    drift = {
        name: {"expected": expected, "observed": observed.get(name)}
        for name, expected in ANALYSIS_REGENERATION_RUNTIME.items()
        if observed.get(name) != expected
    }
    if drift:
        raise integrity.IntegrityError(
            "byte-identical comparator-family regeneration requires the exact "
            f"analysis lock; runtime drift={json.dumps(drift, sort_keys=True)}"
        )


def _materialize_all(root: Path, work: Path, manifest):
    paths = {}
    for sim, item in manifest["main_records"].items():
        target = work / item["source_name"]
        integrity.materialize_record(root / item["path"], target)
        paths[sim] = target
    for label, item in manifest["supplemental_records"].items():
        target = work / item["source_name"]
        integrity.materialize_record(root / item["path"], target)
        paths[label] = target
    item = manifest["protocol_scaffold_sensitivity"]
    target = work / item["source_name"]
    integrity.materialize_record(root / item["path"], target)
    paths["protocol_scaffold_sensitivity"] = target
    item = manifest["schedule_ratio_sensitivity"]
    target = work / item["source_name"]
    integrity.materialize_record(root / item["path"], target)
    paths["schedule_ratio_sensitivity"] = target
    return paths


def validate_truth_surface_semantics(root: Path) -> None:
    """Run the archived GP-dependent historical record audits in a temp tree."""
    reused = root / "scripts" / "reused"
    required = (reused / "build_reference.py", reused / "audit_supplemental.py")
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise integrity.IntegrityError(
            f"full-stack historical auditors are missing: {', '.join(missing)}"
        )
    sys.path.insert(0, str(reused))
    build_reference = importlib.import_module("build_reference")
    audit_supplemental = importlib.import_module("audit_supplemental")
    manifest = json.loads(
        (root / "metadata" / "design_manifest.json").read_text(encoding="utf-8")
    )
    with tempfile.TemporaryDirectory(prefix="dose-combination-bo-full-record-audit-") as temp:
        work = Path(temp)
        paths = _materialize_all(root, work, manifest)
        for sim in ("osa", "gbump", "efftox", "mariposa"):
            rows = json.loads(paths[sim].read_text(encoding="utf-8"))
            build_reference.validate_records(
                rows, manifest["main_records"][sim]["source_name"], sim
            )
        audit_supplemental.HERE = work
        audit_supplemental.main()


def run_package_tests(root: Path, pytest_args: Sequence[str]) -> None:
    package = root / "source" / "package"
    tests = package / "tests"
    if not tests.is_dir():
        raise integrity.IntegrityError("source/package/tests is missing from the archive")
    environment = dict(os.environ)
    source = str(package / "src")
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not previous else os.pathsep.join((source, previous))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        *pytest_args,
        str(tests),
    ]
    subprocess.run(command, cwd=package, env=environment, check=True)


def regenerate_comparator_family_analysis(root: Path) -> None:
    """Regenerate the 34,000-row family analysis without rerunning simulations."""
    package = root / "source" / "package"
    analyzer = package / "paper" / "analyze_comparator_family.py"
    extension = (
        root
        / "records"
        / "comparator_family_extension"
        / "comparator_family_completion_records.json.zst"
    )
    expected = root / "analysis" / "comparator_family"
    projection_metadata = root / "metadata" / "formal_projection.metadata.json"
    if not all(path.is_file() for path in (analyzer, extension, projection_metadata)):
        raise integrity.IntegrityError(
            "comparator-family analyzer, extension record, or formal projection metadata is missing"
        )
    fingerprint = json.loads(projection_metadata.read_text(encoding="utf-8")).get(
        "projection_fingerprint"
    )
    if not isinstance(fingerprint, str):
        raise integrity.IntegrityError("formal projection fingerprint is missing")
    with tempfile.TemporaryDirectory(prefix="dose-combination-bo-family-regeneration-") as temp:
        work = Path(temp)
        projection = work / f"formal_projection-{fingerprint}"
        projection_records = projection / "records"
        projection_records.mkdir(parents=True)
        shutil.copyfile(projection_metadata, projection / "formal_projection.metadata.json")
        for filename in integrity.FORMAL_RECORD_ROWS:
            shutil.copyfile(root / "records" / filename, projection_records / filename)
        regenerated = work / "analysis"
        environment = dict(os.environ)
        source = str(package / "src")
        previous = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source if not previous else os.pathsep.join((source, previous))
        )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(
            [
                sys.executable,
                str(analyzer),
                "--projection-dir",
                str(projection),
                "--extension-raw",
                str(extension),
                "--out-dir",
                str(regenerated),
            ],
            cwd=package,
            env=environment,
            check=True,
        )
        expected_files = {
            path.name: integrity.file_sha256(path)
            for path in expected.iterdir()
            if path.is_file()
        }
        regenerated_files = {
            path.name: integrity.file_sha256(path)
            for path in regenerated.iterdir()
            if path.is_file()
        }
        if regenerated_files != expected_files:
            missing = sorted(set(expected_files) - set(regenerated_files))
            extra = sorted(set(regenerated_files) - set(expected_files))
            changed = sorted(
                name
                for name in set(expected_files) & set(regenerated_files)
                if expected_files[name] != regenerated_files[name]
            )
            raise integrity.IntegrityError(
                "comparator-family regeneration differs from archive: "
                f"missing={missing[:3]}, extra={extra[:3]}, changed={changed[:3]}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run lightweight archive integrity first, then GP-dependent truth-surface "
            "record audits and the archived package test suite. This validates the "
            "saved records and software; it does not rerun the research-scale sweeps."
        )
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        help="archive root; auto-detected when the script is inside the archive",
    )
    parser.add_argument(
        "--skip-sha256sums",
        action="store_true",
        help="skip the top-level SHA256SUMS inventory during the integrity phase",
    )
    parser.add_argument(
        "--skip-record-semantics",
        action="store_true",
        help="skip GP-dependent truth-surface checks of the saved records",
    )
    parser.add_argument(
        "--skip-package-tests",
        action="store_true",
        help="skip the archived package pytest suite",
    )
    parser.add_argument(
        "--skip-family-analysis-regeneration",
        action="store_true",
        help=(
            "skip deterministic regeneration of the 34,000-row comparator-family "
            "analysis from 25,200 formal rows plus the committed 8,800-row extension"
        ),
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="argument forwarded to pytest; repeat for multiple arguments",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = integrity.locate_archive_root(args.archive_root)
        counts = integrity.validate_archive(
            root, check_sha256sums=not args.skip_sha256sums
        )
        if (
            not args.skip_record_semantics
            or not args.skip_package_tests
            or not args.skip_family_analysis_regeneration
        ):
            _require_full_stack()
        if not args.skip_family_analysis_regeneration:
            # Fail before the slower record audits and package tests when the
            # interpreter cannot reproduce the committed derived bytes.
            _require_analysis_regeneration_runtime()
        if not args.skip_record_semantics:
            validate_truth_surface_semantics(root)
        if not args.skip_package_tests:
            run_package_tests(root, args.pytest_arg)
        if not args.skip_family_analysis_regeneration:
            regenerate_comparator_family_analysis(root)
    except (
        integrity.IntegrityError,
        ImportError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Full reproduction validation FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "Full reproduction validation passed: lightweight integrity; "
        f"{counts['total_records']} formal-projection rows plus "
        f"{counts['comparator_extension_records']} separate extension executions; "
        + ("truth-surface semantics; " if not args.skip_record_semantics else "")
        + ("archived package tests; " if not args.skip_package_tests else "")
        + (
            "comparator-family analysis byte-identical regeneration."
            if not args.skip_family_analysis_regeneration
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
