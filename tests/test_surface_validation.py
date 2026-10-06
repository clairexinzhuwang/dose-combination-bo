"""A user-registered surface with a mis-declared OBD must fail loudly.

Selection error and efficacy regret are both measured relative to ``dopt``. A
declared OBD that is feasible but suboptimal is therefore undetectable downstream
-- every number simply comes out wrong. This is the check that catches it at
registration, and these tests pin both that it fires and that it does not fire on
the built-ins, whose OBDs are finite grid searches and so are approximate by
construction.
"""
import numpy as np
import pytest

import dose_combination_bo as sdb
from dose_combination_bo.surfaces import _OPT_REL_TOL, _OPT_SCAN_N, _builtin_spec, resolve_surface

BUILT_INS = ["osa", "mariposa", "efftox", "gbump", "synth:k=1;a=1.2;t=0.6"]


@pytest.mark.parametrize("sim", BUILT_INS)
@pytest.mark.parametrize("z", [0, 1])
def test_built_in_surfaces_validate(sim, z):
    assert resolve_surface(sim, z) is not None


@pytest.mark.parametrize("sim", ["osa", "mariposa", "efftox", "gbump"])
def test_built_in_dopt_shortfall_stays_well_inside_the_bar(sim):
    """Their OBDs are grid searches; quantify the gap rather than assume it is zero."""
    spec = _builtin_spec(sim, 0)
    lin = np.linspace(0.0, 1.0, _OPT_SCAN_N)
    d1, d2 = np.meshgrid(lin, lin, indexing="ij")
    eff = np.vectorize(lambda a, b: float(spec["eff"](a, b)))(d1, d2)
    tox = np.vectorize(lambda a, b: float(spec["tox"](a, b)))(d1, d2)
    feasible = tox <= float(spec["gd"]) + 1e-10
    span = float(eff[feasible].max() - eff[feasible].min())
    shortfall = float(eff[feasible].max()) - float(spec["fopt"])
    # Discretization, not error: an order of magnitude inside the rejection bar.
    assert shortfall / span < _OPT_REL_TOL / 5


def test_feasible_but_suboptimal_dopt_is_rejected():
    @sdb.surface("_test_bad_obd")
    def _bad(z):
        return dict(gd=1.0, dopt=(0.0, 0.0), fopt=0.0,
                    eff=lambda a, b: a + b, tox=lambda a, b: 0.5, sf=1.0, sg=1.0)

    with pytest.raises(ValueError, match="feasible but materially suboptimal"):
        resolve_surface("_test_bad_obd", 0)


def test_infeasible_dopt_is_rejected():
    @sdb.surface("_test_infeasible_obd")
    def _bad(z):
        return dict(gd=0.1, dopt=(1.0, 1.0), fopt=2.0,
                    eff=lambda a, b: a + b, tox=lambda a, b: a + b, sf=1.0, sg=1.0)

    with pytest.raises(ValueError, match="infeasible"):
        resolve_surface("_test_infeasible_obd", 0)


def test_fopt_inconsistent_with_dopt_is_rejected():
    @sdb.surface("_test_bad_fopt")
    def _bad(z):
        return dict(gd=10.0, dopt=(1.0, 1.0), fopt=99.0,
                    eff=lambda a, b: a + b, tox=lambda a, b: 0.0, sf=1.0, sg=1.0)

    with pytest.raises(ValueError, match="fopt"):
        resolve_surface("_test_bad_fopt", 0)


def test_a_correctly_declared_user_surface_passes():
    @sdb.surface("_test_good_obd")
    def _good(z):
        return dict(gd=1.0, dopt=(1.0, 1.0), fopt=2.0,
                    eff=lambda a, b: a + b, tox=lambda a, b: 0.5, sf=1.0, sg=1.0)

    assert resolve_surface("_test_good_obd", 0)["fopt"] == 2.0
