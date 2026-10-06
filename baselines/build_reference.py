"""Build ``baselines/reference_numbers.json`` from a formal projection.

Authenticates all ten projected record files, then pools the four main records
over strata and gates and records, per (surface,
acquisition), the mean +/- seed-clustered Monte Carlo SE of the headline operating
characteristics. These are the numbers a new acquisition compares itself against.

Usage:
    python baselines/build_reference.py FORMAL_PROJECTION_DIR
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(HERE).parent
sys.path.insert(0, str(ROOT / "paper"))

from analysis_record_bundle import (  # noqa: E402
    ProjectionBundleError,
    load_projection_bundle,
)

# Formal-projection record file -> surface label it reports.
FILES = {
    "osa_main.json.zst": "osa",
    "efftox_main.json.zst": "efftox",
    "mariposa_main.json.zst": "mariposa",
    "gbump_main.json.zst": "gbump",
}
DISPLAY = {}
METRICS = ["dose_units", "rec_true_eff", "rpsel", "rec_unsafe", "toxic"]
BY_GAMMA_METRICS = ["dose_units", "rec_true_eff", "rec_unsafe"]
POLICIES = ("cEI", "cEI-tMSE", "cKG")
GAMMAS = (0.5, 0.6, 0.7, 0.8, 0.9)
STRATA = (0, 1)
N_SEEDS = {"osa": 200, "gbump": 200, "efftox": 100, "mariposa": 100}
DESIGN = {
    "mode": "latent",
    "noise": "fixed",
    "budget": 40,
    "r_k": 2,
    "grid_n": 5,
    "warmup": 4,
    "empty_gate": "pf",
}
REQUIRED_FIELDS = {
    "policy", "seed", "sim", "stratum", "gamma", *DESIGN, *METRICS,
}


def pool(records, metric):
    by_seed = {}
    for r in records:
        by_seed.setdefault(r["seed"], []).append(float(r[metric]))
    a = np.asarray([np.mean(v) for _, v in sorted(by_seed.items())], float)
    return dict(mean=float(a.mean()),
                se=float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
                n_seeds=len(a))


def validate_records(records, fname, sim):
    """Fail before summarizing unless ``records`` are the exact corrected main design."""
    if not isinstance(records, list) or not records:
        raise ValueError(f"{fname} must contain a non-empty JSON list")

    expected_seeds = set(range(N_SEEDS[sim]))
    expected_cells = {
        (policy, seed, stratum, gamma)
        for policy in POLICIES
        for seed in expected_seeds
        for stratum in STRATA
        for gamma in GAMMAS
    }
    seen_cells = set()
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"{fname} row {index} is not a JSON object")
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(f"{fname} row {index} is missing required fields {missing}")
        if row["sim"] != sim:
            raise ValueError(f"{fname} row {index} has sim={row['sim']!r}; expected {sim!r}")
        if row["policy"] not in POLICIES:
            raise ValueError(
                f"{fname} row {index} has policy={row['policy']!r}; expected exactly {POLICIES}"
            )
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
            raise ValueError(f"{fname} row {index} has a non-integer seed")
        if row["seed"] not in expected_seeds:
            raise ValueError(
                f"{fname} row {index} has seed={row['seed']!r}; "
                f"expected integers 0 through {N_SEEDS[sim] - 1}"
            )
        if row["stratum"] not in STRATA:
            raise ValueError(
                f"{fname} row {index} has stratum={row['stratum']!r}; expected {STRATA}"
            )
        try:
            gamma = float(row["gamma"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{fname} row {index} has a non-numeric gamma") from exc
        if gamma not in GAMMAS:
            raise ValueError(
                f"{fname} row {index} has gamma={row['gamma']!r}; expected {GAMMAS}"
            )
        for field, expected in DESIGN.items():
            if row[field] != expected:
                raise ValueError(
                    f"{fname} row {index} has {field}={row[field]!r}; expected {expected!r}"
                )
        try:
            kap = float(row.get("kap", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{fname} row {index} has a non-numeric kap") from exc
        if not np.isfinite(kap) or kap != 1.0:
            raise ValueError(f"{fname} row {index} has kap={kap!r}; expected calibrated kap=1.0")
        try:
            values = {metric: float(row[metric]) for metric in METRICS}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{fname} row {index} has a non-numeric reference metric") from exc
        if not np.isfinite(list(values.values())).all():
            raise ValueError(f"{fname} row {index} has a non-finite reference metric")
        if values["dose_units"] < 0 or values["rpsel"] < 0:
            raise ValueError(f"{fname} row {index} has a negative error metric")
        if values["rec_unsafe"] not in (0.0, 1.0):
            raise ValueError(f"{fname} row {index} has non-binary rec_unsafe")
        if not 0 <= values["toxic"] <= DESIGN["budget"]:
            raise ValueError(f"{fname} row {index} has toxic outside [0, budget]")

        cell = (row["policy"], row["seed"], row["stratum"], gamma)
        if cell in seen_cells:
            raise ValueError(f"{fname} contains duplicate design cell {cell}")
        seen_cells.add(cell)

    missing_cells = expected_cells - seen_cells
    unexpected_cells = seen_cells - expected_cells
    if missing_cells or unexpected_cells:
        detail = []
        if missing_cells:
            detail.append(f"missing expected cell {sorted(missing_cells)[0]}")
        if unexpected_cells:
            detail.append(f"unexpected cell {sorted(unexpected_cells)[0]}")
        raise ValueError(f"{fname} has an incomplete or invalid design: {'; '.join(detail)}")


def build_reference(src):
    bundle = load_projection_bundle(src)
    out = {"_projection": {
               "fingerprint": bundle.provenance["projection_fingerprint"],
               "metadata_sha256": bundle.provenance["projection_metadata_sha256"],
               "formal_manifest_sha256": bundle.provenance["formal_manifest_sha256"],
           },
           "_source_files": {},
           "_note": (
               "authenticated exact/stable-z formal projection; cEI-tMSE is the "
               "specified untuned 1:1 alternation; surfaces pool over strata/gates, "
               "by_gamma pools over strata; seed-clustered MCSE uses ddof=1"
           ),
           "surfaces": {},
           "by_gamma": {}}
    for fname, sim in FILES.items():
        commitment = bundle.provenance["record_files"][fname]
        out["_source_files"][fname] = {
            "surface": sim,
            "rows": commitment["rows"],
            "compressed_sha256": commitment["compressed_sha256"],
            "uncompressed_sha256": commitment["uncompressed_sha256"],
        }
        recs = bundle.records[fname]
        validate_records(recs, fname, sim)
        by_pol = {}
        for r in recs:
            by_pol.setdefault(r["policy"], []).append(r)
        out["surfaces"][sim] = {}
        out["by_gamma"][sim] = {}
        for pol, rs in sorted(by_pol.items()):
            name = DISPLAY.get(pol, pol)
            out["surfaces"][sim][name] = {m: pool(rs, m) for m in METRICS}
            out["by_gamma"][sim][name] = {}
            for gamma in sorted({float(r["gamma"]) for r in rs}):
                cell = [r for r in rs if np.isclose(float(r["gamma"]), gamma)]
                out["by_gamma"][sim][name][f"{gamma:.1f}"] = {
                    m: pool(cell, m) for m in BY_GAMMA_METRICS
                }
        print(f"  {sim}: {len(recs)} records, policies={sorted(DISPLAY.get(p, p) for p in by_pol)}")
    return out


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python baselines/build_reference.py FORMAL_PROJECTION_DIR")
    out = build_reference(Path(sys.argv[1]))
    dest = os.path.join(HERE, "reference_numbers.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
