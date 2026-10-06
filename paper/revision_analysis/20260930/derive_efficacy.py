"""Read archived records/summaries; derive efficacy displays, never run trials."""
from pathlib import Path
import argparse, csv, hashlib, json, math
import numpy as np
from scipy.stats import t
import zstandard
def repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent/'results/comparator_family_analysis/comparator_family_selected_records.json.zst').is_file():
            return parent
    return None
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, default=repo_root(), help='Repository root; inferred from script parents when archived under paper/revision_analysis.')
parser.add_argument('--output-dir', type=Path, required=True, help='Directory for derived CSVs and provenance; historical inputs are read-only.')
args=parser.parse_args()
if args.repo is None:
    parser.error('Could not infer repository root; pass --repo /path/to/dual-combo-BO.')
ROOT=args.repo.resolve()
OUT=args.output_dir.resolve()
if OUT == ROOT/'results' or (ROOT/'results') in OUT.parents:
    parser.error('Choose an output directory outside the historical results/ tree.')
OUT.mkdir(parents=True,exist_ok=True)
F=ROOT/'results/comparator_family_analysis'
S=ROOT/'results/budget_toxicity_calibration_analysis'

def csvrows(p):return list(csv.DictReader(p.open()))
def dump(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summary(values):
    v=np.array(values,dtype=float);m=len(v);est=v.mean();se=v.std(ddof=1)/math.sqrt(m);crit=t.ppf(.975,m-1)
    return dict(estimate=float(est),mcse=float(se),ci95_low=float(est-crit*se),ci95_high=float(est+crit*se),replicate_sets=m)
def ratio_summary(numerators,denominators):
    a=np.array(numerators,dtype=float);b=np.array(denominators,dtype=float);m=len(a)
    est=a.sum()/b.sum();influence=(a-est*b)/b.mean();se=influence.std(ddof=1)/math.sqrt(m);crit=t.ppf(.975,m-1)
    return dict(estimate=float(est),mcse=float(se),ci95_low=float(est-crit*se),ci95_high=float(est+crit*se),replicate_sets=m)
def auth(p):
    d=json.loads(Path(str(p)+'.metadata.json').read_text());h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==d['artifact_sha256'];return h
raw=(F/'comparator_family_selected_records.json.zst').read_bytes()
records=json.loads(zstandard.ZstdDecompressor().decompress(raw))
meta=json.loads((F/'comparator_family_surface_bivariate.csv.metadata.json').read_text())
assert len(records)==34000
assert hashlib.sha256(raw).hexdigest()==meta['selected_family_commitment']['compressed_sha256']
lookup={(r.get('protocol_scaffold','lhs_fixed'),r['sim'],float(r['gamma']),r['stratum'],r['seed'],r['policy']):r for r in records}
assert len(lookup)==len(records)
full=[]
for sim,m in [('osa',200),('efftox',100),('mariposa',100),('gbump',200)]:
    for comp in ['tmse','qBIG','cEI']:
        values=[]
        for seed in range(m):
            diffs=[]
            for tau in [.5,.6,.7,.8,.9]:
                for z in [0,1]:
                    l,r=[lookup['lhs_fixed',sim,tau,z,seed,p] for p in ['cKG',comp]]
                    assert l.get('recommendation_made',True) and r.get('recommendation_made',True)
                    diffs.append(float(l['rec_true_eff'])-float(r['rec_true_eff']))
            values.append(np.mean(diffs))
        full.append(dict(surface=sim,comparator=comp,**summary(values),paired_trials=m*10,interval='pointwise_unadjusted_t',weighting='two_strata_five_thresholds_equal_within_replicate'))
dump('full_grid_efficacy_contrasts.csv',full)
archabs={(float(r['tau']),r['policy']):r for r in csvrows(F/'comparator_family_gradual_oc.csv')}
archpair={(float(r['tau']),r['comparator']):r for r in csvrows(F/'comparator_family_gradual_contrasts.csv') if r['outcome_id']=='recommended_efficacy'}
gradabs=[];gradpair=[]
for tau in [.7,.9]:
    for policy in ['cKG','tmse','qBIG','cEI']:
        block=[];a=[];b=[]
        for seed in range(200):
            eligible=[lookup['start_low_expansion','osa',tau,z,seed,policy] for z in [0,1]]
            v=[float(r['rec_true_eff']) for r in eligible if r.get('recommendation_made',True)]
            a.append(sum(v));b.append(len(v))
            if v:block.append(np.mean(v))
        equal=summary(block);ratio=ratio_summary(a,b);old=archabs[tau,policy]
        for k in ['estimate','mcse','ci95_low','ci95_high']:
            assert abs(equal[k]-float(old['recommended_efficacy_'+k]))<1e-11
        assert sum(b)==int(old['recommended_efficacy_eligible_trials'])
        for method,v in [('selection_only_trial_ratio',ratio),('archived_equal_eligible_replicate',equal)]:
            gradabs.append(dict(tau=tau,policy=policy,method=method,**v,selected_trials=sum(b),initiated_trials=400,eligible_replicate_sets=len(block),interval='pointwise_unadjusted_t_cluster_influence' if method=='selection_only_trial_ratio' else 'pointwise_unadjusted_t'))
    for comp in ['tmse','qBIG','cEI']:
        block=[];a=[];b=[]
        for seed in range(200):
            v=[]
            for z in [0,1]:
                l,r=[lookup['start_low_expansion','osa',tau,z,seed,p] for p in ['cKG',comp]]
                if l.get('recommendation_made',True) and r.get('recommendation_made',True):v.append(float(l['rec_true_eff'])-float(r['rec_true_eff']))
            a.append(sum(v));b.append(len(v))
            if v:block.append(np.mean(v))
        equal=summary(block);ratio=ratio_summary(a,b);old=archpair[tau,comp]
        for k in ['estimate','mcse','ci95_low','ci95_high']:assert abs(equal[k]-float(old[k]))<1e-11
        assert sum(b)==int(old['eligible_trial_pairs'])
        for method,v in [('both_selecting_trial_pair_ratio',ratio),('archived_equal_eligible_replicate',equal)]:
            gradpair.append(dict(tau=tau,comparator=comp,method=method,**v,both_selecting_trial_pairs=sum(b),initiated_trial_pairs=400,eligible_replicate_sets=len(block),interval='pointwise_unadjusted_t_cluster_influence' if method=='both_selecting_trial_pair_ratio' else 'pointwise_unadjusted_t'))
dump('gradual_efficacy_absolute.csv',gradabs);dump('gradual_efficacy_paired.csv',gradpair)
source=S/'budget_toxicity_calibration_secondary.csv';auth(source)
json_source=S/'budget_toxicity_calibration_secondary.json';auth(json_source)
json_rows=json.loads(json_source.read_text())['absolute_rows']
json_idx={(r['policy'],r['assumed_toxicity_noise_sd_factor'],r['budget']):r for r in json_rows if r['metric']=='true_recommended_efficacy'}
rows=[r for r in csvrows(source) if r['section']=='absolute' and r['metric']=='true_recommended_efficacy']
assert len(rows)==36
for r in rows:
    assert int(r['monte_carlo_replicates'])==500
    paired_json=json_idx[r['policy'],float(r['assumed_toxicity_noise_sd_factor']),int(r['budget'])]
    for field in ['estimate','mcse','ci95_low','ci95_high']:assert float(r[field])==paired_json[field]
    assert abs((float(r['ci95_high'])-float(r['estimate']))-float(r['mcse'])*t.ppf(.975,499))<1e-11
nested=[dict(policy=r['policy'],sd_factor=float(r['assumed_toxicity_noise_sd_factor']),snapshot=int(r['budget']),estimate=float(r['estimate']),mcse=float(r['mcse']),ci95_low=float(r['ci95_low']),ci95_high=float(r['ci95_high']),replicate_sets=500,interval='archived_pointwise_unadjusted_t',weighting='two_strata_two_thresholds_equal_within_replicate') for r in rows]
dump('nested_efficacy_absolute.csv',nested)
idx={(r['policy'],r['sd_factor'],r['snapshot']):r for r in nested}
nd=[]
for factor in [.5,1.,2.]:
    for n in [20,40,80]:
        for comp in ['tmse','qBIG','cEI']:
            nd.append(dict(sd_factor=factor,snapshot=n,comparator=comp,estimate=idx['cKG-exact-formal',factor,n]['estimate']-idx[comp,factor,n]['estimate'],interval='not_reconstructed_without_paired_raw_snapshots'))
dump('nested_efficacy_mean_differences.csv',nd)
notes={'source_selected_records':str(F/'comparator_family_selected_records.json.zst'),'selected_record_count':len(records),'all_gradual_archived_summaries_reproduced_to_absolute_tolerance':1e-11,'nested_source':str(source),'nested_source_hash_verified':True,'nested_raw_limitation':'Nested raw snapshots are not inputs to this script. Absolute means/MCSEs/intervals are read from the authenticated archived summary CSV and checked against independently authenticated JSON. Paired-difference covariance cannot be recovered from marginal summaries alone; no paired nested efficacy intervals are reconstructed.','family_54_toxicity_contrasts':'Unchanged. All newly reported efficacy intervals are descriptive, pointwise, unadjusted, separate from the 54-toxicity max-t family.'}
(OUT/'provenance.json').write_text(json.dumps(notes,indent=2)+'\n')
for label,rr in [('FULL',full),('GRAD ABS',[r for r in gradabs if r['method']=='selection_only_trial_ratio']),('GRAD PAIR',[r for r in gradpair if r['method']=='both_selecting_trial_pair_ratio'])]:
    print(label)
    for r in rr:print({k:v for k,v in r.items() if k not in ['interval','weighting','method']})
print('NESTED DIFFERENCES')
for r in nd: print(r)
print('Outputs',OUT)
