"""Research software for comparing acquisitions in dual-agent dose-combination finding.

The simulator includes cKG, tMSE, an author-specified toxicity-classification
entropy rule, and cEI. Within each dose-availability design, the rules use the
same GP response models, model-based toxicity rule, candidate combination set,
response-record or enrollment limit, and final-selection rule. Users can
add an acquisition or response surface with one decorated function and compare
it with the included methods.

Quickstart
----------
    import dose_combination_bo as sdb

    # one trial
    r = sdb.run_trial("cKG", seed=0, z=0, gamma=0.7, sim="osa")
    print(r["grid_dose_units"], r["rec_true_eff"])

    # a sweep + leaderboard
    res = sdb.sweep(["cKG", "tmse", "entropy", "cEI"], surfaces=["osa"], seeds=50)
    sdb.leaderboard(res, metric="grid_dose_units", sim="osa")

    # your own acquisition
    @sdb.acquisition("my_rule")
    def my_rule(ctx):
        mu_e, _ = ctx.latent_efficacy()
        return mu_e + 2.0 * ctx.sd_g
    sdb.sweep(["cKG", "my_rule"], surfaces=["osa"], seeds=50)
"""
from .registry import (
    surface, get_acquisition, get_surface,
    list_acquisitions, list_surfaces, ACQUISITIONS, SURFACES,
)
from .context import Context
from .trial import run_trial, grid
from .sweep import sweep, summarize, paired_contrast, scaffold_interaction, leaderboard
from .metrics import recommend_obd, dose_units, n_toxic

# Historical execution-source bytes are preserved separately in ../study.
# The current package is versioned independently and exposes accurate help.
recommend_obd.__doc__ = """Return the operational final selection.

Select the combination with the largest posterior mean efficacy among those
meeting the model-based toxicity rule. If none meet it, return the largest-margin
fallback; that fallback does not meet the model-based toxicity criterion. No
minimum-exposure requirement is imposed. The function name is retained for
compatibility.
"""
dose_units.__doc__ = """Return distance from a final selection to a supplied reference.

The built-in reference is a legacy continuous-domain diagnostic, not the finite-grid
trial target. Distances are reported in grid units; lower is better.
"""
n_toxic.__doc__ = """Count supplied toxicity-history values above the limit.

This helper does not distinguish simulator-generated initialization records from
participant assignments and does not count observed toxicity events.
"""
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
from .public_registry import acquisition, install_reader_facing_surface_names

install_reader_facing_surface_names()

from .version import VERSION as __version__
from .decision import Selection, Allocation, select_from_posterior, choose_allocation

__all__ = [
    "acquisition", "surface", "get_acquisition", "get_surface",
    "list_acquisitions", "list_surfaces", "ACQUISITIONS", "SURFACES",
    "Context", "run_trial", "grid", "sweep", "summarize", "paired_contrast",
    "scaffold_interaction", "leaderboard", "ProtocolScaffold", "PROTOCOL_SCAFFOLDS",
    "recommend_obd", "dose_units", "n_toxic", "oc_table", "selection_table", "oc_latex",
    "artifact_metadata", "write_artifact_metadata", "records_sha256", "file_sha256",
    "build_policy_card", "policy_card_from_file",
    "ckg_one_step_gated_exact", "ckg_scores_exact", "expected_terminal_value_exact",
    "Selection", "Allocation", "select_from_posterior", "choose_allocation",
    "__version__",
]
