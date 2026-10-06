"""Load real decision/controller modules without package __init__.

Only GPyTorch-backed posterior providers are substituted with deterministic test
fixtures. No test using this loader validates GP fitting, dependency integration,
or live-trial operation. All pure integration and decision algorithms are the
actual source files, not copies. The separate full-suite attempt has no stubs.
"""
import ast
import importlib
import sys
import types
from pathlib import Path
import numpy as np
import torch
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[1]

def load(name='review_sdb', old=False):
    if (BASE/'pyproject.toml').is_file():
        root = BASE/'study_archive/frozen_core' if old else BASE/'src/dose_combination_bo'
    else:
        root = BASE/'audit/frozen_aug30_source/safedosebo' if old else BASE/'software/src/dose_combination_bo'
    pkg = types.ModuleType(name); pkg.__path__=[str(root)];sys.modules[name]=pkg
    gp=types.ModuleType(name+'.gp')
    def unavailable(*args,**kwargs):
        raise RuntimeError('GP fitting/joint posterior unavailable: use an explicit test fixture')
    for attr in ('fit_gp','post_latent','post_predictive','joint','kg_lines'):
        setattr(gp,attr,unavailable)
    # This scalar helper is copied as an AST node from the actual source, not a
    # fitted GP implementation. Its source origin is recorded in the loader.
    node=next(n for n in ast.parse((root/'gp.py').read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='kg_lines')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(root/'gp.py'),'exec'),
         dict(np=np,norm=norm), env:={})
    gp.kg_lines=env['kg_lines'];sys.modules[name+'.gp']=gp
    modules={key:importlib.import_module(name+'.'+key) for key in
             ('registry','acquisitions','protocol','metrics','context','ckg_exact','trial')}
    if not old:
        modules['decision']=importlib.import_module(name+'.decision')
    # Import the actual registry binding (default exact cKG).
    importlib.import_module(name+'.public_ckg')
    return types.SimpleNamespace(**modules)


def attach_controller_fixture(modules, monkeypatch, mode='mixed'):
    """Explicit artificial means/variances; fit_gp is NOT called in this check."""
    calls={'n':0}
    def fit(X,y,**kwargs):
        kind=calls['n']%2; calls['n']+=1
        return types.SimpleNamespace(kind=kind),types.SimpleNamespace(noise=torch.tensor(.04))
    def posterior(m,X):
        xx=X.detach().cpu().numpy(); total=xx.sum(axis=1)
        if m.kind==0: mu=1+total
        elif mode=='empty': mu=1+total
        elif mode=='allpass': mu=-2+0*total
        else: mu=2*total-.3
        return torch.tensor(mu,dtype=torch.double),torch.full((len(X),),.04,dtype=torch.double)
    def predictive(m,l,X):
        mu,v=posterior(m,X);return mu,v+l.noise
    def surface(sim,z):
        return {'gd':0.,'dopt':np.array([0.,0.]),'fopt':1.,
                'eff':lambda a,b:1+a+b,'tox':lambda a,b:2*(a+b)-.3,
                'sf':.2,'sg':.2}
    monkeypatch.setattr(modules.trial,'fit_gp',fit)
    monkeypatch.setattr(modules.trial,'resolve_surface',surface)
    for m in (modules.trial,modules.context,modules.metrics,modules.acquisitions,modules.ckg_exact):
        for field,fn in [('post_latent',posterior),('post_predictive',predictive)]:
            if hasattr(m,field):monkeypatch.setattr(m,field,fn)
    return calls
