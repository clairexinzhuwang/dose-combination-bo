#!/usr/bin/env python3
"""Bounded CPU timing of an installed dose-combination-bo wheel; no adaptive trials.

Run outside the repository with an isolated installed-wheel interpreter:
    python benchmark_installed.py --output OUTPUT --wheel PATH_TO_WHEEL
Optional --repo PATH records (without requiring) current-source byte agreement.
The fixed plan and generated observations are saved before any model is fitted.
"""
import os

# Must precede NumPy/PyTorch imports. No global environment is changed.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import csv
import hashlib
import importlib.metadata as im
import json
import platform
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
import zipfile


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def timeout(_signum, _frame):
    raise TimeoutError("Prespecified 300-second wall-clock benchmark limit reached")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "benchmark_results.json"
    if result_path.exists():
        raise FileExistsError("Choose a new output directory; existing results are not overwritten")
    plan = {
        "purpose": "Computational benchmark only; not an operating-characteristic experiment",
        "surface": "osa", "stratum": 0, "seed": 20261001,
        "record_counts": [4, 20, 40], "grid_n": 5,
        "available_candidates": 25, "gate_tau": 0.7, "cohort_size": 2,
        "rules": ["cKG", "tmse", "entropy", "cEI"],
        "observations": "Nested fixed nonadaptive on-grid observations; independent Gaussian outcome noise; individual records",
        "dose_order": "First 40 indices from two independently permuted 25-point grids, using seed20261001",
        "gp": "Independent efficacy/toxicity exact GPs; constant mean; Matern5/2 ARD; known individual noise variances; 120 Adam steps, learning rate0.1",
        "fitting": "One unreported-timing warm-up pair at n4; one measured pair for each n; 8 individual fits total (6 measured,2 warm-up)",
        "scoring": "One warm-up and five measured repetitions per rule/dataset; fresh Context per call; no score reuse; same fitted models for all four rules within each dataset",
        "primary_timing": "Context construction plus registered full-vector scorer; includes gate and tMSE preparation; excludes fitting and file I/O",
        "gp_prediction_cache": "Fitted-GP prediction caches are warmed by the per-rule warm-up, retained across scoring calls; score/Context objects are fresh",
        "execution": "Serial CPU, one PyTorch intra/inter-op thread; numerical-library thread limits1",
        "maximum_wall_seconds": 300,
        "scope_limit": "Three prespecified data sizes and one stratum; timings are not full-trial runtimes and not a representative distribution over adaptive trial states",
    }
    write_json(out / "benchmark_plan.json", plan)
    result = {"status": "RUNNING", "plan": plan, "fits": [], "warmup_fits": [],
              "timings": [], "score_vectors": {}, "posteriors": {}}
    started = time.perf_counter()
    process_started = time.process_time()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    try:
        import numpy as np
        import torch
        import gpytorch
        import scipy
        import dose_combination_bo as sdb
        from dose_combination_bo.gp import fit_gp, post_latent

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(plan["seed"])
        package = Path(sdb.__file__).resolve().parent
        if "site-packages" not in package.parts:
            raise RuntimeError(f"Expected installed wheel, found {package}")
        if sdb.__version__ != "0.2.1rc5":
            raise RuntimeError(f"Expected 0.2.1rc5, got {sdb.__version__}")
        wheel = args.wheel.resolve()
        distribution = im.distribution("dose-combination-bo")
        direct_url = json.loads(distribution.read_text("direct_url.json") or "null")
        wheel_sha256 = digest(wheel.read_bytes())
        with zipfile.ZipFile(wheel) as z:
            module_names = sorted(n for n in z.namelist() if n.startswith("dose_combination_bo/") and n.endswith(".py"))
            mismatches = [n for n in module_names if (package.parent / n).read_bytes() != z.read(n)]
            metadata_agreement = {}
            for basename in ("METADATA", "WHEEL", "entry_points.txt"):
                member = next(n for n in z.namelist() if n.endswith(".dist-info/" + basename))
                metadata_agreement[basename] = z.read(member).decode() == distribution.read_text(basename)
        if mismatches:
            raise RuntimeError(f"Installed modules differ from supplied wheel: {mismatches}")
        if not all(metadata_agreement.values()):
            raise RuntimeError(f"Installed distribution metadata differs from wheel: {metadata_agreement}")
        if direct_url["archive_info"]["hashes"]["sha256"] != wheel_sha256:
            raise RuntimeError("Installed wheel origin hash differs from supplied wheel")
        repo_differences = []
        if args.repo:
            repo_differences = [n for n in module_names if
                not (args.repo / "src" / n).exists() or
                (args.repo / "src" / n).read_bytes() != (package.parent / n).read_bytes()]
        hardware = {}
        try:
            raw = subprocess.check_output(["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"], text=True, timeout=15)
            h = json.loads(raw)["SPHardwareDataType"][0]
            # Deliberately exclude device serial and unique identifiers.
            hardware = {k: h.get(k) for k in ("chip_type", "machine_model", "machine_name", "number_processors", "physical_memory")}
        except Exception as exc:
            hardware = {"unavailable_reason": str(exc)}
        result["environment"] = {
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "architecture": platform.machine(),
            "hardware": hardware, "dose-combination-bo": sdb.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "gpytorch": gpytorch.__version__,
            "package_path": str(package), "wheel_filename": wheel.name,
            "wheel_sha256": wheel_sha256,
            "installed_wheel_modules_identical": len(module_names),
            "installed_metadata_agreement": metadata_agreement,
            "installed_origin_hash_agreement": True,
            "repo_module_differences": repo_differences,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "thread_environment": {name: os.environ[name] for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "distribution_direct_url": direct_url,
        }
        rng = np.random.default_rng(plan["seed"])
        grid = np.array([(a, b) for a in np.linspace(0, 1, 5) for b in np.linspace(0, 1, 5)])
        indices = np.concatenate([rng.permutation(25), rng.permutation(25)])[:40]
        x = grid[indices]
        spec = sdb.get_surface("osa")(0)
        true_f = np.array([spec["eff"](*d) for d in x])
        true_g = np.array([spec["tox"](*d) for d in x])
        yf = true_f + spec["sf"] * rng.standard_normal(40)
        yg = true_g + spec["sg"] * rng.standard_normal(40)
        observations = {"grid": grid.tolist(), "dose_indices": indices.tolist(),
                        "X": x.tolist(), "efficacy": yf.tolist(), "toxicity": yg.tolist(),
                        "true_efficacy": true_f.tolist(), "true_toxicity": true_g.tolist(),
                        "sf": spec["sf"], "sg": spec["sg"], "g_dagger": spec["gd"]}
        write_json(out / "fixed_observations.json", observations)
        result["fixed_observations_sha256"] = digest((out / "fixed_observations.json").read_bytes())
        Xset = torch.tensor(grid)

        def pair_fit(n, warmup=False):
            Xt = torch.tensor(x[:n])
            models = []
            for outcome, y, noise in (("efficacy", yf, spec["sf"] ** 2),
                                      ("toxicity", yg, spec["sg"] ** 2)):
                torch.manual_seed(plan["seed"])
                before = time.perf_counter()
                fitted = fit_gp(Xt, torch.tensor(y[:n]), iters=120, lr=0.1, fixed_noise=noise)
                elapsed = time.perf_counter() - before
                models.extend(fitted)
                rec = {"n_records": n, "outcome": outcome, "seconds": elapsed,
                       "optimizer_steps": 120, "known_noise_variance": noise}
                result["warmup_fits" if warmup else "fits"].append(rec)
            return models

        pair_fit(4, warmup=True)
        for n in plan["record_counts"]:
            me, le, mt, lt = pair_fit(n)
            for outcome, m in (("efficacy", me), ("toxicity", mt)):
                mean, variance = post_latent(m, Xset)
                result["posteriors"][f"{n}_{outcome}"] = {
                    "mean": mean.detach().numpy().tolist(), "variance": variance.detach().numpy().tolist(),
                    "parameters": {name: par.detach().numpy().tolist() for name, par in m.named_parameters()}}

            def score(rule, repetition, warmup):
                # No Context, gate, tMSE vector or acquisition score is reused.
                before = time.perf_counter()
                ctx = sdb.Context(Xset, me, le, mt, lt, spec["gd"], 0.7,
                                  gate_mode="latent", r_k=2, empty_gate="pf")
                prepared = time.perf_counter()
                values = np.asarray(sdb.get_acquisition(rule)(ctx), dtype=float)
                after = time.perf_counter()
                if values.shape != (25,) or np.isnan(values).any() or np.isposinf(values).any():
                    raise AssertionError(f"Invalid score vector{n}:{rule}")
                if not np.isfinite(values[ctx.safe]).all():
                    raise AssertionError("A gate-admissible candidate has a nonfinite score")
                key = f"{n}_{rule}"
                encoded = [float(v) if np.isfinite(v) else "-inf" for v in values]
                if key in result["score_vectors"]:
                    if encoded != result["score_vectors"][key]["scores"]:
                        raise AssertionError(f"Deterministic scores changed at {key}")
                else:
                    result["score_vectors"][key] = {
                        "scores": encoded, "gate_admissible": np.flatnonzero(ctx.safe).tolist(),
                        "available_count": 25, "admissible_count": int(ctx.safe.sum()),
                        "gate_empty": bool(ctx.gate_empty)}
                result["timings"].append({
                    "n_records": n, "rule": rule, "repetition": repetition,
                    "warmup": warmup, "available_candidates": 25,
                    "admissible_candidates": int(ctx.safe.sum()),
                    "context_seconds": prepared-before,
                    "callable_seconds": after-prepared,
                    "total_score_seconds": after-before})

            for rule in plan["rules"]:
                score(rule, 0, True)
            for rep in range(1, 6):
                rules = plan["rules"]
                rotated = rules[(rep-1) % 4:] + rules[:(rep-1) % 4]
                for rule in rotated:
                    score(rule, rep, False)
            write_json(result_path, result)
            print(f"Completed n={n}:2GP fits,4rules,1warm-up+5timed repeats each", flush=True)

        rows = []
        for n in plan["record_counts"]:
            fit_sum = sum(r["seconds"] for r in result["fits"] if r["n_records"] == n)
            for rule in plan["rules"]:
                r = [r for r in result["timings"] if r["n_records"] == n and r["rule"] == rule and not r["warmup"]]
                times = [v["total_score_seconds"] for v in r]
                rows.append({"n_records": n, "rule": rule, "fit_pair_seconds": fit_sum,
                             "available_candidates": 25, "admissible_candidates": r[0]["admissible_candidates"],
                             "repetitions": len(times), "median_seconds": statistics.median(times),
                             "min_seconds": min(times), "max_seconds": max(times),
                             "median_context_seconds": statistics.median(v["context_seconds"] for v in r),
                             "median_callable_seconds": statistics.median(v["callable_seconds"] for v in r)})
        with (out / "timing_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        result["summary"] = rows
        result["status"] = "PASS"
    except BaseException as exc:
        result["status"] = "TIME_LIMIT" if isinstance(exc, TimeoutError) else "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        signal.alarm(0)
        result["elapsed_wall_seconds"] = time.perf_counter() - started
        result["elapsed_process_cpu_seconds"] = time.process_time() - process_started
        result["script_sha256"] = digest(Path(__file__).read_bytes())
        write_json(result_path, result)
    print(json.dumps({"status": result["status"], "wall_seconds": result["elapsed_wall_seconds"],
                      "individual_GP_fits": len(result["fits"]),
                      "warmup_GP_fits": len(result["warmup_fits"]),
                      "scoring_calls": len(result["timings"])}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
