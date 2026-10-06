"""safedosebo: a benchmark for acquisition choice under posterior-feasibility gates.

Compare the principal acquisition functions (cKG, tmse, entropy) and a cEI reference inside one fixed
Bayesian-optimization simulation framework. Within an access protocol, the surrogate,
posterior-feasibility gate, panel, enrollment limit, and recommendation rule are held
common so the acquisition choice can be compared under specified conditions. Add your
own acquisition or surface with one decorated function and compare it with the included
reference baselines.

Quickstart
----------
    import safedosebo as sdb

    # one trial
    r = sdb.run_trial("cKG", seed=0, z=0, gamma=0.7, sim="osa")
    print(r["dose_units"], r["rec_unsafe"])

    # a sweep + leaderboard
    res = sdb.sweep(["cKG", "tmse", "entropy", "cEI"], surfaces=["osa"], seeds=50)
    sdb.leaderboard(res, metric="dose_units", sim="osa")

    # your own acquisition
    @sdb.acquisition("my_rule")
    def my_rule(ctx):
        mu_e, _ = ctx.latent_efficacy()
        return ctx.restrict(mu_e + 2.0 * ctx.sd_g)
    sdb.sweep(["cKG", "my_rule"], surfaces=["osa"], seeds=50)
"""
from .registry import (
    acquisition, surface, get_acquisition, get_surface,
    list_acquisitions, list_surfaces, ACQUISITIONS, SURFACES,
)
from .context import Context
from .trial import run_trial, grid
from .sweep import sweep, summarize, paired_contrast, scaffold_interaction, leaderboard
from .metrics import recommend_obd, dose_units, n_toxic
from .oc import oc_table, selection_table, oc_latex
from .protocol import ProtocolScaffold, PROTOCOL_SCAFFOLDS
from .reporting import artifact_metadata, write_artifact_metadata, records_sha256, file_sha256
from .policy_card import build_policy_card, policy_card_from_file
from .ckg_exact import (
    ckg_one_step_gated_exact,
    ckg_scores_exact,
    expected_terminal_value_exact,
)
from . import acquisitions as builtin_acquisitions   # noqa: F401  (populates ACQUISITIONS)
from . import public_ckg as public_ckg_bindings       # noqa: F401  (release API bindings)
from . import public_entropy as public_entropy_bindings  # noqa: F401  (release API bindings)
# Use the release resolver at package level so the deprecated SUR alias warns.
get_acquisition = public_entropy_bindings.get_acquisition
from . import surfaces as builtin_surfaces           # noqa: F401  (populates SURFACES)

__version__ = "0.2.0"

__all__ = [
    "acquisition", "surface", "get_acquisition", "get_surface",
    "list_acquisitions", "list_surfaces", "ACQUISITIONS", "SURFACES",
    "Context", "run_trial", "grid", "sweep", "summarize", "paired_contrast",
    "scaffold_interaction", "leaderboard", "ProtocolScaffold", "PROTOCOL_SCAFFOLDS",
    "recommend_obd", "dose_units", "n_toxic", "oc_table", "selection_table", "oc_latex",
    "artifact_metadata", "write_artifact_metadata", "records_sha256", "file_sha256",
    "build_policy_card", "policy_card_from_file",
    "ckg_one_step_gated_exact", "ckg_scores_exact", "expected_terminal_value_exact",
    "__version__",
]
