"""Command-line interface for simulations, discovery, and HTML summaries."""
import argparse
import importlib
from importlib.metadata import PackageNotFoundError, version

from .registry import (
    ACQUISITIONS,
    SURFACES,
    list_acquisitions,
    list_surfaces,
)
from .sweep import sweep, leaderboard
from .policy_card import policy_card_from_file


_MAIN_ACQUISITIONS = ("cKG", "tmse", "entropy", "cEI")
_CLI_METRICS = (
    "grid_dose_units",
    "dose_units",
    "rpsel",
    "rec_unsafe",
    "toxic",
    "rec_true_eff",
    "rec_true_tox",
)
_LOWER_IS_BETTER = {
    "grid_dose_units", "dose_units", "rpsel", "rec_unsafe", "toxic", "rec_true_tox",
}
_RECOMMENDATION_CONDITIONAL = {
    "grid_dose_units", "dose_units", "rpsel", "rec_true_eff", "rec_true_tox",
}


def _release_version() -> str:
    try:
        return version("dose-combination-bo")
    except PackageNotFoundError:
        return "0.2.1rc5"


def _csv(text: str, cast=str):
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("provide at least one comma-separated value")
    try:
        return [cast(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _thresholds(text: str) -> list[float]:
    values = _csv(text, float)
    if any(not 0 < value < 1 for value in values):
        raise argparse.ArgumentTypeError("every threshold must lie strictly between 0 and 1")
    return values


def _strata(text: str) -> list[int]:
    values = _csv(text, int)
    if any(value not in {0, 1} for value in values):
        raise argparse.ArgumentTypeError("strata must be 0, 1, or both")
    return values


def _modes(text: str) -> list[str]:
    values = _csv(text)
    if any(value not in {"latent", "predictive"} for value in values):
        raise argparse.ArgumentTypeError("modes must be latent, predictive, or both")
    return values


def _aliases(registry, attribute: str) -> list[str]:
    return sorted(
        key
        for key, fn in registry.items()
        if key != getattr(fn, attribute, key)
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="dose-combination-bo",
        description="Simulate and compare acquisition functions for dual-agent dose-combination finding",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {_release_version()}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run simulations and show a leaderboard")
    r.add_argument(
        "--acquisitions",
        type=_csv,
        default=list(_MAIN_ACQUISITIONS),
        help=(
            "comma-separated rules (default: cKG,tmse,entropy,cEI); "
            "run 'dose-combination-bo list --all' for historical aliases"
        ),
    )
    r.add_argument(
        "--surfaces", type=_csv, default=["osa"],
        help="comma-separated response surfaces (default: osa)",
    )
    r.add_argument(
        "--seeds", type=_positive_int, default=5,
        help="number of independent simulation replicates (seeds start at 0; default: 5)",
    )
    r.add_argument(
        "--gammas", type=_thresholds, default=[0.7],
        help=(
            "comma-separated required posterior probabilities that mean toxicity "
            "does not exceed its limit, tau; larger values are more stringent "
            "simulation settings, not clinically calibrated safety probabilities "
            "(default: 0.7)"
        ),
    )
    r.add_argument(
        "--modes", type=_modes, default=["latent"],
        help=(
            "whether the toxicity rule concerns uncertainty about mean toxicity "
            "(latent) or a future outcome (predictive); comma-separated (default: latent)"
        ),
    )
    r.add_argument(
        "--strata", type=_strata, default=[0],
        help="comma-separated stratum identifiers (default: 0)",
    )
    r.add_argument("--noise", default="fixed", choices=["fixed", "learned"],
                   help="use the data-generating outcome variance or estimate it from simulated data")
    r.add_argument("--kap", "--noise-multiplier", dest="kap", type=float, default=1.0,
                   help="positive multiplier on the efficacy and toxicity outcome SDs (default: 1)")
    r.add_argument("--allow-ungated-diagnostic", action="store_true",
                   help="explicitly enable historical ungated diagnostic rules; not a clinical safety mode")
    r.add_argument("--empty-gate", default="pf", choices=["pf", "ungated"],
                   help="assignment behavior when no protocol-eligible combination meets "
                        "the model-based toxicity rule: pf assigns the eligible combination "
                        "with the largest standardized toxicity margin (default); ungated lets "
                        "the acquisition rank without the rule and is retained only for "
                        "compatibility. Scheduled final selection always uses the "
                        "largest-margin fallback")
    r.add_argument(
        "--budget", type=_positive_int, default=12,
        help=(
            "maximum model-fitting response records when all combinations are available "
            "(including off-grid simulator-generated initialization records), or maximum "
            "enrollment under gradual expansion (default: 12)"
        ),
    )
    r.add_argument("--protocol-scaffold", default="lhs_fixed",
                   choices=["lhs_fixed", "start_low_expansion"],
                   help="all candidate combinations available or gradual expansion from the lowest combination")
    r.add_argument("--region-step", type=float, default=0.25,
                   help="increment in d1+d2 for the gradually expanding candidate region (default: 0.25)")
    r.add_argument("--empty-gate-stop-after", type=int, default=3,
                   help="under gradual expansion only, number of consecutive occasions on "
                        "which no available combination meets the toxicity rule before "
                        "stopping (default: 3)")
    r.add_argument("--exclude-repeats-during-expansion", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="avoid revisiting a combination while an untried combination remains "
                        "in the expanding dose region")
    r.add_argument(
        "--metric", default="grid_dose_units", choices=_CLI_METRICS,
        help=("leaderboard outcome (default: grid_dose_units, distance to the finite-grid target); the compatibility field "
              "'rec_unsafe' indicates, across initiated trials, whether the final "
              "selected combination's true mean toxicity exceeds the prespecified limit, with "
              "a stopped trial contributing zero; the compatibility field 'toxic' counts "
              "response records assigned to combinations whose true mean toxicity exceeds "
              "that limit. In the all-available design, 'toxic' includes off-grid, "
              "simulator-generated initialization records and is not a participant-allocation outcome; under "
              "gradual expansion all records are participant assignments. Neither field "
              "counts observed toxicity events. Final-selection rec_* fields include the "
              "scheduled empty-set fallback; a fallback does not meet the model-based "
              "toxicity criterion, and any "
              "final selection may be untried because no minimum-exposure rule is imposed. "
              "The dose_units field is a legacy distance to a continuous-domain reference, "
              "not the finite-grid trial target"),
    )
    r.add_argument("--serial", action="store_true", help="disable Ray parallelism")
    r.add_argument(
        "--out", default=None,
        help="JSON checkpoint; a matching existing run resumes from completed trials",
    )
    r.add_argument("--plugin", action="append", default=[], metavar="MODULE",
                   help="import a module before running so its @acquisition / @surface "
                        "registrations are visible (repeatable)")

    ls = sub.add_parser("list", help="list registered acquisitions and surfaces")
    ls.add_argument(
        "--all", action="store_true",
        help="also show additional research rules and compatibility aliases",
    )
    ls.add_argument("--plugin", action="append", default=[], metavar="MODULE",
                    help="import a module first so its registrations are listed")

    pc = sub.add_parser("policy-card", help="create a self-contained HTML simulation summary")
    pc.add_argument("results", help="JSON list of per-trial records")
    pc.add_argument("--out", required=True, help="output HTML path")

    a = ap.parse_args(argv)
    for module in getattr(a, "plugin", []):
        importlib.import_module(module)
    if a.cmd == "list":
        registered = list_acquisitions()
        main = [name for name in _MAIN_ACQUISITIONS if name in registered]
        additional = [name for name in registered if name not in main]
        print("main acquisitions:      ", ", ".join(main))
        if a.all and additional:
            print("additional acquisitions:", ", ".join(additional))
            print("acquisition aliases:    ", ", ".join(_aliases(ACQUISITIONS, "acquisition_name")))
        print("surfaces:               ", ", ".join(list_surfaces()))
        if a.all:
            print("surface aliases:         ", ", ".join(_aliases(SURFACES, "surface_name")))
        print("parameterized surface:   synth:k=..;a=..;t=..")
        return
    if a.cmd == "policy-card":
        output, sidecar = policy_card_from_file(a.results, a.out)
        print(f"wrote {output}")
        print(f"wrote {sidecar}")
        return

    n_trials = (
        len(a.acquisitions) * len(a.surfaces) * a.seeds * len(a.gammas)
        * len(a.modes) * len(a.strata)
    )
    print(
        f"Running {n_trials} trials: {len(a.acquisitions)} acquisitions x "
        f"{len(a.surfaces)} surfaces x {len(a.strata)} strata x "
        f"{len(a.gammas)} tau values x {len(a.modes)} toxicity-rule modes x "
        f"{a.seeds} replicates."
    )
    res = sweep(
        acquisitions=a.acquisitions,
        surfaces=a.surfaces,
        seeds=a.seeds,
        gammas=a.gammas,
        modes=a.modes,
        strata=a.strata,
        noise=a.noise,
        kap=a.kap,
        empty_gate=a.empty_gate,
        allow_ungated_diagnostic=a.allow_ungated_diagnostic,
        protocol_scaffold=a.protocol_scaffold,
        region_step=a.region_step,
        empty_gate_stop_after=a.empty_gate_stop_after,
        exclude_repeats_during_expansion=a.exclude_repeats_during_expansion,
        budget=a.budget,
        parallel=not a.serial,
        out=a.out,
    )
    print()
    lower = a.metric in _LOWER_IS_BETTER
    conditional_metric = a.metric in _RECOMMENDATION_CONDITIONAL
    modes = a.modes
    for sim in a.surfaces:
        for mode in modes:
            # leaderboard refuses to pool different modes, so ask for one at a time.
            rows = [r for r in res if r.get("mode") == mode] if len(modes) > 1 else res
            if len(modes) > 1:
                print(f"[mode={mode}]")
            leaderboard(
                rows,
                metric=a.metric,
                sim=sim,
                lower_is_better=lower,
                recommendation_conditional=conditional_metric,
            )
            print()


if __name__ == "__main__":
    main()
