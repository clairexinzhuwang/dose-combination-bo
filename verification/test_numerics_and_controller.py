"""Independent algebra checks and explicitly fixture-backed controller tests."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm, qmc
from fixture_loader import load, attach_controller_fixture

P=load();OLD=load('review_sdb_old',old=True)
D=P.decision; E=P.ckg_exact

@pytest.mark.parametrize('case',range(40))
def test_exact_integral_against_brute_piecewise_quadrature(case):
    rng=np.random.default_rng(91000+case); n=1+case%9
    a,b,c,d=rng.normal(size=(4,n));sd=np.exp(rng.normal(0,.4,n));tau=[.5,.7,.9,.95][case%4]
    if case%5==0: d[0]=0
    if case%7==0: b[:]=0
    intercept=-c/sd;slope=-d/sd;threshold=norm.ppf(tau)
    cuts=[-12.,12.]
    for i in range(n):
        if slope[i]:cuts.append((threshold-intercept[i])/slope[i])
        for j in range(i):
            if slope[i]!=slope[j]:cuts.append((intercept[j]-intercept[i])/(slope[i]-slope[j]))
    cuts=sorted(set(float(x) for x in cuts if -12<=x<=12))
    val=0.
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        mid=(lo+hi)/2; gate=intercept+slope*mid>threshold
        if gate.any():
            aa=a[gate];bb=b[gate];fc=[-12.,12.]
            for i in range(len(aa)):
                for j in range(i):
                    if bb[i]!=bb[j]:fc.append((aa[j]-aa[i])/(bb[i]-bb[j]))
            fc=sorted(set(float(x) for x in fc if -12<=x<=12))
            ev=sum(quad(lambda t: float(np.max(aa+bb*t))*norm.pdf(t),l,h,
                        epsabs=1e-11,epsrel=1e-11)[0] for l,h in zip(fc[:-1],fc[1:]))
        else:ev=a[np.argmax(intercept+slope*mid)]
        val+=ev*quad(norm.pdf,lo,hi,epsabs=1e-12)[0]
    actual,diag=E.expected_terminal_value_exact(a,b,c,d,sd,0,tau,return_diagnostics=True)
    assert actual==pytest.approx(val,abs=2e-9)
    assert diag['probability_mass']==pytest.approx(1,abs=2e-14)

@pytest.mark.parametrize('seed',range(10))
def test_positive_scale_and_offset_invariance(seed):
    rng=np.random.default_rng(seed);a,b,c,d=rng.normal(size=(4,6));sd=np.ones(6)
    f=lambda a,b:E.expected_terminal_value_exact(a,b,c,d,sd,0,.7)
    v=f(a,b)
    assert f(a+125,b)==pytest.approx(v+125,abs=2e-12)
    assert f(3*a,3*b)==pytest.approx(3*v,abs=2e-12)

@pytest.mark.parametrize('seed',range(6))
def test_sobol_defining_terminal_rule(seed):
    rng=np.random.default_rng(305+seed);a,b,c,d=rng.normal(size=(4,5));sd=np.ones(5)
    exact=E.expected_terminal_value_exact(a,b,c,d,sd,0,.7)
    est=[]
    for scramble in range(4):
        z=norm.ppf(qmc.Sobol(2,scramble=True,seed=1000+seed*10+scramble).random_base2(16))
        eff=a+z[:,0,None]*b; zz=(-c-z[:,1,None]*d)/sd;gate=zz>norm.ppf(.7)
        idx=np.argmax(np.where(gate,eff,-np.inf),axis=1)
        fallback=~gate.any(axis=1);idx[fallback]=np.argmax(zz[fallback],axis=1)
        est.append(eff[np.arange(len(z)),idx].mean())
    se=np.std(est,ddof=1)/2
    assert abs(np.mean(est)-exact)<max(8*se,8e-4)

def test_two_line_closed_form_and_ties():
    assert E._affine_max_expectation([0,0],[1,-1])==pytest.approx(math.sqrt(2/math.pi),abs=1e-14)
    assert E._affine_max_expectation([3,3],[2,2])==3
    assert E.expected_terminal_value_exact([3],[15],[50],[.4],[1],0,.9)==pytest.approx(3)

@pytest.mark.parametrize('lo,hi',[(8,9),(-9,-8),(-1,1),(-np.inf,0),(0,np.inf),(15,16)])
def test_normal_interval_tails(lo,hi):
    expected=norm.sf(lo)-norm.sf(hi) if lo>=0 else norm.cdf(hi)-norm.cdf(lo)
    assert E._normal_interval_probability(lo,hi)==pytest.approx(expected,rel=1e-12,abs=1e-300)

def test_strict_gate_and_fallback_status():
    x=D.select_from_posterior([50,1],[0,-1],[1,1],0,.5)
    assert (x.index,x.n_passing,x.status)==(1,1,'qualified_selection')
    x=D.select_from_posterior([50,1],[1,2],[1,1],0,.5)
    assert (x.index,x.status)==(0,'fallback_selection')
    x=D.select_from_posterior([50,1],[1,2],[1,1],0,.5,on_empty='none')
    assert x.index is None and x.status=='no_qualified_selection'

@pytest.mark.parametrize('bad',[0,1,-.1,1.2,float('nan'),float('inf')])
def test_invalid_gamma(bad):
    with pytest.raises(ValueError):D.select_from_posterior([0],[0],[1],0,bad)

@pytest.mark.parametrize('bad',[[1],[1,2,3],[[1,2]],[float('nan'),2],[float('inf'),2]])
def test_invalid_scores(bad):
    with pytest.raises(ValueError):D.choose_allocation(bad,np.array([True,False]),np.ones(2,bool),[1,0])

def test_gate_cannot_be_bypassed_and_negative_kg_not_clipped():
    x=D.choose_allocation([-2,500,-1],np.array([1,0,1],bool),np.ones(3,bool),[1,-1,2])
    assert x.index==2 and x.passed_criterion and not x.used_fallback
    with pytest.raises(ValueError):D.choose_allocation([-np.inf]*2,np.array([1,0],bool),np.ones(2,bool),[1,-1])

def test_fallback_uses_standardized_feasibility_not_saturated_cdf():
    x=D.choose_allocation([900,0],np.zeros(2,bool),np.ones(2,bool),[-101,-100])
    assert x.index==1 and x.used_fallback

@pytest.mark.parametrize('bad',[[1,0],[np.nan,0],[True],np.ones((2,1),bool)])
def test_bad_mask_rejected(bad):
    with pytest.raises(ValueError):D.choose_allocation([0,1],bad,np.ones(2,bool),[1,-1])

def test_context_no_scalar_broadcast(monkeypatch):
    attach_controller_fixture(P,monkeypatch,'allpass')
    lik=SimpleNamespace(noise=torch.tensor(.04));m=SimpleNamespace(kind=1)
    ctx=P.context.Context(torch.tensor([[0.,0.],[1.,1.]]),m,lik,m,lik,0,.5)
    with pytest.raises(ValueError):ctx.restrict(np.array([3.]))
    with pytest.raises(ValueError):ctx.restrict(np.array([-np.inf,-np.inf]))
    assert np.array_equal(ctx.restrict(np.array([[2.],[3.]])),[2,3])

def test_custom_plugin_cannot_bypass_gate(monkeypatch):
    attach_controller_fixture(P,monkeypatch)
    P.registry.acquisition('hostile')(lambda ctx:ctx.Xset.sum(axis=1).numpy())
    row=P.trial.run_trial('hostile',1,0,.5,'osa',budget=8)
    assert row['allocation_history'][4:]==[[0.,0.]]*4
    assert all(r['passed_criterion'] for r in row['allocation_decisions'])

def test_same_plugin_exposes_baseline_bug(monkeypatch):
    attach_controller_fixture(OLD,monkeypatch)
    OLD.registry.acquisition('hostile')(lambda ctx:ctx.Xset.sum(axis=1).numpy())
    row=OLD.trial.run_trial('hostile',1,0,.5,'osa',budget=6)
    assert row['allocation_history'][4:]==[[1.,1.]]*2

@pytest.mark.parametrize('bad',[np.full(25,-np.inf),np.zeros(1),np.full(25,np.nan),np.full(25,np.inf)])
def test_controller_fails_closed_on_bad_plugin(monkeypatch,bad):
    attach_controller_fixture(P,monkeypatch,'allpass')
    P.registry.acquisition('invalid')(lambda ctx:bad)
    with pytest.raises(ValueError):P.trial.run_trial('invalid',1,0,.5,'osa',budget=6)

def test_gradual_initialization_cannot_select_unopened_dose(monkeypatch):
    attach_controller_fixture(P,monkeypatch,'allpass')
    row=P.trial.run_trial('tmse',1,0,.5,'osa',budget=2,protocol_scaffold='start_low_expansion')
    assert row['terminal_candidate_count']==1
    assert [row['rec_d1'],row['rec_d2']]==[0,0]
    assert row['rec_direct_observations']==2

def test_baseline_selects_unopened_dose(monkeypatch):
    attach_controller_fixture(OLD,monkeypatch,'allpass')
    row=OLD.trial.run_trial('tmse',1,0,.5,'osa',budget=2,protocol_scaffold='start_low_expansion')
    assert [row['rec_d1'],row['rec_d2']]==[1,1]

def test_gradual_three_empty_events_stop_and_no_selection(monkeypatch):
    attach_controller_fixture(P,monkeypatch,'empty')
    row=P.trial.run_trial('tmse',1,0,.5,'osa',budget=40,protocol_scaffold='start_low_expansion')
    assert row['n_empty_gate_events']==3 and row['n_enrolled']==6
    assert row['selection_status']=='stopped_no_selection'
    assert row['rec_d1'] is None and not row['recommendation_made']
    assert row['rec_passed_criterion'] is None
    assert all(r['used_fallback'] for r in row['allocation_decisions'])

@pytest.mark.parametrize('policy',['tmse','cEI','random'])
@pytest.mark.parametrize('mode',['allpass','mixed','empty'])
def test_unchanged_builtin_paths_under_same_posterior_fixture(monkeypatch,policy,mode):
    attach_controller_fixture(P,monkeypatch,mode);attach_controller_fixture(OLD,monkeypatch,mode)
    new=P.trial.run_trial(policy,23,0,.7,'osa',budget=10)
    old=OLD.trial.run_trial(policy,23,0,.7,'osa',budget=10)
    for field in ('allocation_history','rec_d1','rec_d2','rec_unsafe','toxic','rec_true_eff'):
        assert new[field]==old[field]

def test_ungated_diagnostic_requires_opt_in(monkeypatch):
    attach_controller_fixture(P,monkeypatch)
    with pytest.raises(ValueError,match='explicit|intentionally'):P.trial.run_trial('utmse',1,0,.5,'osa',budget=6)
    row=P.trial.run_trial('utmse',1,0,.5,'osa',budget=6,allow_ungated_diagnostic=True)
    assert row['allocation_decisions'][0]['diagnostic_ungated']

def test_public_aliases_are_exact_not_legacy():
    assert P.registry.get_acquisition('cKG') is P.registry.get_acquisition('cKG-exact')
    assert P.registry.get_acquisition('cKG') is not P.registry.get_acquisition('cKG1fix')
    assert P.registry.get_acquisition('cKG1fix') is P.registry.get_acquisition('cKG-legacy512')

@pytest.mark.parametrize('seed',range(12))
@pytest.mark.parametrize('m',[1,2,5])
def test_cohort_mean_conditioning_matches_individual_replicates(seed,m):
    # An independent Gaussian-conditioning derivation, no GP fit involved.
    rng=np.random.default_rng(921+seed);A=rng.normal(size=(5,5));C=A@A.T+.2*np.eye(5)
    noise=.4;cross=C[:-1,-1];vq=C[-1,-1]
    yi_cov=np.full((m,m),vq)+noise*np.eye(m)
    ci=np.repeat(cross[:,None],m,axis=1)
    individual_residual=C[:-1,:-1]-ci@np.linalg.solve(yi_cov,ci.T)
    mean_residual=C[:-1,:-1]-np.outer(cross,cross)/(vq+noise/m)
    np.testing.assert_allclose(individual_residual,mean_residual,atol=1e-12)
    errors=rng.normal(size=m)
    individual_update=ci@np.linalg.solve(yi_cov,errors)
    mean_update=cross/(vq+noise/m)*errors.mean()
    np.testing.assert_allclose(individual_update,mean_update,atol=1e-12)

def test_signed_ckg_matches_joint_posterior_formula(monkeypatch):
    # Two coherent query/grid covariance columns. A toxic reclassification can
    # legitimately lower the rule's terminal value even with a valid covariance.
    X=torch.tensor([[0.,0.],[1.,1.]])
    ef=SimpleNamespace(kind=0);tg=SimpleNamespace(kind=1)
    lik=SimpleNamespace(noise=torch.tensor(.1))
    af=torch.tensor([2.,0.]);cg=torch.tensor([-.2,-1.])
    vg=torch.tensor([.2,.2]); cf=torch.tensor([.1,.02]); ct=torch.tensor([.2,0.])
    def joint(model,lk,x,q):
        return (af if model.kind==0 else cg),vg,(cf if model.kind==0 else ct),.3
    monkeypatch.setattr(E,'joint',joint)
    monkeypatch.setattr(E,'post_latent',lambda model,x:(cg,vg))
    score=E.ckg_one_step_gated_exact(X[0],X,ef,lik,tg,lik,0,.5,r_k=2)
    den=.2+.1/2;future_sd=np.sqrt(vg.numpy()-ct.numpy()**2/den)
    expected=E.expected_terminal_value_exact(af.numpy()-2,cf.numpy()/np.sqrt(den),cg.numpy(),ct.numpy()/np.sqrt(den),future_sd,0,.5)
    assert score==pytest.approx(expected,abs=1e-12)
    assert score<0

def test_resume_keys_do_not_round_distinct_cutoffs():
    import importlib
    sweep=importlib.import_module('review_sdb.sweep')
    args=(40,2,5,'fixed',4,'pf',1.,'lhs_fixed',.25,3,True)
    assert sweep._design_key(('cKG',0,'osa',0,'latent',.7001),*args)!=sweep._design_key(('cKG',0,'osa',0,'latent',.7002),*args)

def test_resume_matches_new_record_and_rejects_old_record(monkeypatch,tmp_path):
    import importlib,json
    sweep=importlib.import_module('review_sdb.sweep')
    attach_controller_fixture(P,monkeypatch,'allpass')
    path=tmp_path/'results.json'
    rows=sweep.sweep(['tmse'],['osa'],seeds=1,gammas=[.7001,.7002],strata=[0],budget=6,parallel=False,out=path,verbose=False)
    assert len(rows)==2
    assert sweep.sweep(['tmse'],['osa'],seeds=1,gammas=[.7001,.7002],strata=[0],budget=6,parallel=False,out=path,verbose=False)==rows
    del rows[0]['core_source_sha256'];path.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='metadata'):sweep.sweep(['tmse'],['osa'],seeds=1,gammas=[.7001,.7002],strata=[0],budget=6,parallel=False,out=path,verbose=False)

@pytest.mark.parametrize('seed_arg',[False,True,0,-1])
def test_bad_sweep_seed_count_rejected(seed_arg):
    import importlib
    sweep=importlib.import_module('review_sdb.sweep')
    with pytest.raises(ValueError):sweep.sweep(['tmse'],['osa'],seeds=seed_arg,parallel=False)

def test_reported_grid_metric_is_not_silently_legacy_target(monkeypatch):
    attach_controller_fixture(P,monkeypatch,'allpass')
    # The fixture's historical reference is (0,0), while its true best acceptable
    # point under the actual grid is also (0,0); both fields remain explicit.
    row=P.trial.run_trial('tmse',1,0,.5,'osa',budget=4)
    assert 'grid_dose_units' in row and row['grid_true_acceptable_count']==1
    assert row['historical_reference_dopt']==[0.,0.]
    assert row['grid_dose_units']==pytest.approx(row['dose_units'])


def test_callback_cannot_change_controller_masks_in_place(monkeypatch):
    attach_controller_fixture(P,monkeypatch)
    def modifies_context(ctx):
        ctx.gate_safe[:]=True
        ctx.candidate_mask[:]=True
        ctx.standardized_feasibility[:]=999
        return ctx.Xset.sum(axis=1).numpy()
    P.registry.acquisition('mutating_context_regression')(modifies_context)
    row=P.trial.run_trial('mutating_context_regression',1,0,.5,'osa',budget=6)
    assert row['allocation_history'][4:]==[[0.,0.]]*2
    assert row['allocation_decisions'][0]['passed_criterion']


def test_latest_public_decorator_is_retained_and_works_with_controller(monkeypatch):
    import importlib
    # Isolated namespace avoids changing earlier raw-registry regression checks.
    Q=load('review_public_wrapper');attach_controller_fixture(Q,monkeypatch)
    public=importlib.import_module('review_public_wrapper.public_registry')
    assert Q.registry.acquisition is public.acquisition
    @public.acquisition('latest_wrapper_column')
    def rule(ctx):
        return ctx.Xset.sum(axis=1).numpy()[:,None]
    row=Q.trial.run_trial('latest_wrapper_column',1,0,.5,'osa',budget=6)
    assert row['allocation_history'][4:]==[[0.,0.]]*2
    assert hasattr(rule,'raw_acquisition')
