from common import *
from original_resampling import limits
import numpy as np,pandas as pd,ast
checks=[];access=[];result=[]
R=pd.read_csv(ROOT/'data/scored_records.csv',dtype={'county_fips':str},float_precision='round_trip')
arch=pd.read_csv(ROOT/'expected/archived_QA1_ranges.csv',float_precision='round_trip')
policies=[f'TV2_L{l}_G{g}_R{r}' for l in [1,2] for g in [0,1] for r in [0,1]]+['B2_L1_equalblock_selected_PB3','ridge','weighted_logit','depth2_tree','G0_MATCHED_L2_R1']
states=('IA','IL','IN','KS','NE','TX');regions=('1','2','5','6','9')
def check(label,v,e,tol=1e-10):
    err=float(np.max(np.abs(np.asarray(v,dtype=float)-np.asarray(e,dtype=float))))
    checks.append({'check':label,'pass':bool(err<=tol),'max_abs_error':err,'tolerance':tol})
for cond,support,axis,seed in [('held','development','state',20260914),('external','external','ERS_region',20260917)]:
    rows=pd.read_csv(ROOT/f'data/S5_{cond}_quantiles.csv',dtype={'county_fips':str},float_precision='round_trip').to_dict('records')
    f=R[(R.crop=='corn')&(R.condition==cond)].merge(pd.DataFrame(rows),on=['county_fips','year'],validate='one_to_one')
    check(cond+' exact scored key count',len(f),2637 if cond=='held' else 1173,0)
    f['ERS_region']=f.ERS_region.fillna(0).astype(int).astype(str)
    for pol in policies+['ALWAYS_A1']:
        e=f.yield_mg_ha.to_numpy()[:,None]-f[[f'{pol}_{q}' for q in [10,50,90]]].to_numpy();tau=np.array([.1,.5,.9])
        f[pol]=np.maximum(tau*e,(tau-1)*e).mean(axis=1)
    blocks=f.groupby([axis,'year'])[policies+['ALWAYS_A1']].mean()
    for pol in policies:
        delta=blocks[pol]-blocks.ALWAYS_A1;lo,hi,n=limits(delta,axis,seed,5000)
        a=arch[(arch.support==support)&(arch.policy==pol)&(arch.comparator=='ALWAYS_A1')&(arch.version=='WEIGHT_CORRECTED')].iloc[0]
        for field,value in [('paired_delta_PB3',delta.mean()),('descriptive_2p5',lo),('descriptive_97p5',hi),('valid_replicates',n)]:check(cond+'/'+pol+'/'+field,value,a[field])
        result.append({'support':support,'policy':pol,'selection_gain':-delta.mean(),'gain_lo':-hi,'gain_hi':-lo,'seed':seed,'draws':n})
outcsv(TEST/'S5_recomputed_variants.csv',result);outcsv(TEST/'S5_checks.csv',checks)
dump('s5_recompute_summary.json',{'checks':len(checks),'passed':sum(r['pass'] for r in checks),'failed':[r for r in checks if not r['pass']],'policies':len(policies),'sources':len(access),'script_sha256':sha(__file__)})
print(json.dumps({'checks':len(checks),'failed':[r for r in checks if not r['pass']]}))

assert all(c['pass'] for c in checks)
