"""Integration checks requiring the REAL installed GP stack; no mock providers.

Import errors are errors and must never be converted to skips.
"""
import json
import numpy as np
import pytest
import torch
import dose_combination_bo as sdb
from dose_combination_bo.gp import fit_gp, post_latent, post_predictive
from dose_combination_bo.version import VERSION, implementation_fingerprint

@pytest.mark.parametrize("policy", ["cKG", "cEI", "tmse", "entropy"])
@pytest.mark.parametrize("sim", ["osa", "efftox", "mariposa", "gbump"])
def test_actual_gp_trial(policy, sim):
    r=sdb.run_trial(policy, seed=0, z=0, gamma=.7, sim=sim, budget=6)
    assert r["n_enrolled"] == 6
    assert len(r["allocation_history"]) == 6
    assert r["software_version"] == VERSION == sdb.__version__
    assert r["core_source_sha256"] == implementation_fingerprint()
    assert r["selection_status"] in ("qualified_selection", "fallback_selection")
    assert r["rec_passed_criterion"] == (r["selection_status"] == "qualified_selection")
    assert np.isfinite(r["grid_dose_units"])
    assert r["grid_dose_units"] >= 0
    json.dumps(r, allow_nan=False)

def test_actual_gp_latent_predictive_variance():
    X=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]],dtype=torch.double)
    y=torch.tensor([0.,.8,.9,1.1],dtype=torch.double)
    m,l=fit_gp(X,y,fixed_noise=.04)
    ml,vl=post_latent(m,X);mp,vp=post_predictive(m,l,X)
    torch.testing.assert_close(ml,mp)
    torch.testing.assert_close(vp-vl,torch.full_like(vl,.04),rtol=1e-6,atol=1e-8)
    assert torch.isfinite(ml).all() and torch.isfinite(vl).all()

def test_actual_gp_resume(tmp_path):
    path=tmp_path/'run.json'
    args=dict(acquisitions=['cEI'],surfaces=['osa'],seeds=[0],gammas=[.7001],
              strata=[0],budget=6,parallel=False,verbose=False,out=str(path))
    a=sdb.sweep(**args); b=sdb.sweep(**args)
    assert a == b
    with pytest.raises(ValueError,match='design does not match'):
        sdb.sweep(**{**args,'gammas':[.7002]})

def test_ungated_diagnostic_requires_opt_in():
    with pytest.raises(ValueError,match='allow_ungated_diagnostic'):
        sdb.run_trial('ustrad',seed=0,z=0,gamma=.7,sim='osa',budget=6)

@pytest.mark.parametrize('policy', ['cKG', 'cEI', 'tmse', 'entropy'])
def test_actual_gp_stops_without_selection(policy):
    # Inadmissible starting doses exercise stopping with real
    # fitted posteriors. This is a boundary test, not an OC study or clinical design.
    @sdb.surface('acceptance_all_above')
    def all_above(z):
        return dict(gd=-5., dopt=(1.,1.), fopt=2.,
                    eff=lambda a,b:a+b, tox=lambda a,b:10.-20.*a*b,
                    sf=.05, sg=.05)
    r=sdb.run_trial(policy, seed=71, z=0, gamma=.9,
                    sim='acceptance_all_above', budget=40,
                    protocol_scaffold='start_low_expansion', empty_gate_stop_after=3)
    assert r['selection_status']=='stopped_no_selection'
    assert r['n_enrolled']==6
    assert r['recommendation_made'] is False
    assert r['rec_d1'] is None and r['rec_d2'] is None
    assert len(r['allocation_history'])==6
