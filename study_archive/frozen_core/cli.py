"""Command-line interface for simulation, discovery, and policy cards."""
import argparse
import importlib
import json

from .registry import list_acquisitions, list_surfaces
from .sweep import sweep, leaderboard
from .policy_card import policy_card_from_file


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="safedosebo",
        description="Posterior-feasibility-gated dose-optimization simulation benchmark",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a sweep and print a leaderboard")
    r.add_argument(
        "--acquisitions",
        default="cKG,tmse,entropy,cEI",
        help=(
            "comma-separated acquisition names (three principal policies plus "
            "the cEI reference: cKG,tmse,entropy,cEI); cKG is deterministic exact, qBIG is the "
            "entropy provenance alias, and cEI-tMSE is a secondary hybrid"
        ),
    )
    r.add_argument("--surfaces", default="osa", help="comma-separated surface names")
    r.add_argument("--seeds", type=int, default=50)
    r.add_argument("--gammas", default="0.5,0.7,0.9")
    r.add_argument("--modes", default="latent")
    r.add_argument("--noise", default="fixed", choices=["fixed", "learned"],
                   help="noise-fitting mode: pin the data-generating variance or estimate it")
    r.add_argument("--kap", "--noise-multiplier", dest="kap", type=float, default=1.0,
                   help="positive multiplier on both surface observation-noise SDs (default: 1)")
    r.add_argument("--empty-gate", default="pf", choices=["pf", "ungated"],
                   help="empty-gate rule: pf selects the most-feasible dose (default); "
                        "ungated is retained only for historical identity tests")
    r.add_argument("--budget", type=int, default=40)
    r.add_argument("--protocol-scaffold", default="lhs_fixed",
                   choices=["lhs_fixed", "start_low_expansion"],
                   help="fixed full-panel initialization or the prespecified start-low expansion")
    r.add_argument("--region-step", type=float, default=0.25,
                   help="start-low d1+d2 expansion per cohort (default: 0.25)")
    r.add_argument("--empty-gate-stop-after", type=int, default=3,
                   help="consecutive eligible empty gates before stopping (default: 3)")
    r.add_argument("--exclude-repeats-during-expansion", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="exclude visited doses while an unvisited dose remains in a partial region")
    r.add_argument("--metric", default="dose_units")
    r.add_argument("--serial", action="store_true", help="disable Ray parallelism")
    r.add_argument("--out", default=None, help="checkpoint/read results JSON")
    r.add_argument("--plugin", action="append", default=[], metavar="MODULE",
                   help="import a module before running so its @acquisition / @surface "
                        "registrations are visible (repeatable)")

    ls = sub.add_parser("list", help="list registered acquisitions and surfaces")
    ls.add_argument("--plugin", action="append", default=[], metavar="MODULE",
                    help="import a module first so its registrations are listed")

    pc = sub.add_parser("policy-card", help="create a self-contained HTML audit card")
    pc.add_argument("results", help="JSON list of per-trial records")
    pc.add_argument("--out", required=True, help="output HTML path")

    a = ap.parse_args(argv)
    for module in getattr(a, "plugin", []):
        importlib.import_module(module)
    if a.cmd == "list":
        print("acquisitions:", ", ".join(list_acquisitions()))
        print("surfaces:    ", ", ".join(list_surfaces()))
        print("surfaces:     also 'synth:k=..;a=..;t=..' (parameterized family)")
        return
    if a.cmd == "policy-card":
        output, sidecar = policy_card_from_file(a.results, a.out)
        print(f"wrote {output}")
        print(f"wrote {sidecar}")
        return

    res = sweep(
        acquisitions=a.acquisitions.split(","),
        surfaces=a.surfaces.split(","),
        seeds=a.seeds,
        gammas=[float(x) for x in a.gammas.split(",")],
        modes=a.modes.split(","),
        noise=a.noise,
        kap=a.kap,
        empty_gate=a.empty_gate,
        protocol_scaffold=a.protocol_scaffold,
        region_step=a.region_step,
        empty_gate_stop_after=a.empty_gate_stop_after,
        exclude_repeats_during_expansion=a.exclude_repeats_during_expansion,
        budget=a.budget,
        parallel=not a.serial,
        out=a.out,
    )
    print()
    lower = a.metric in ("dose_units", "rpsel", "toxic", "rec_unsafe")
    conditional_metric = a.metric in {"dose_units", "rpsel", "rec_true_eff", "rec_true_tox"}
    modes = a.modes.split(",")
    for sim in a.surfaces.split(","):
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
