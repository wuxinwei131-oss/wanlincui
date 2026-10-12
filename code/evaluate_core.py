"""Recompute frozen prediction losses and published diagnostics without fitting models."""
from common import *
import numpy as np
import pandas as pd
import math, platform, importlib.metadata
E=ROOT/'expected'; checks=[]; reads=[]
def read(name):
    p=(ROOT/'data' if name=='scored_records.csv' else E)/name
    # Inspect year projection before outcome fields. The retained analysis tables contain only 2019-2024.
    columns=pd.read_csv(p,nrows=0).columns
    if 'year' in columns:
        y=pd.read_csv(p,usecols=['year'])['year']
        assert y.between(2019,2024).all(), 'Forbidden or unexpected outcome year'
    reads.append({'path':str(p),'sha256':sha(p),'role':'retained analysis evidence'})
    return pd.read_csv(p,dtype={'county_fips':str},float_precision='round_trip')
def check(name,actual,expected,tol=1e-10):
    if isinstance(actual,(str,bool,np.bool_)):
        ok=actual==expected;err=0 if ok else None
    elif np.isscalar(actual) and np.isscalar(expected) and pd.isna(actual) and pd.isna(expected):
        ok=True;err=0
    else:
        err=float(np.max(np.abs(np.asarray(actual,dtype=float)-np.asarray(expected,dtype=float))))
        ok=bool(np.isfinite(err) and err<=tol)
    checks.append({'check':name,'pass':bool(ok),'max_abs_error':err,'tolerance':tol})
def wm(v,w):return float(np.dot(np.asarray(v,dtype=float),np.asarray(w,dtype=float))/np.sum(w)) if np.sum(w)>0 else np.nan
def equalweights(f):return 1/(f.block.nunique()*f.groupby('block').block.transform('size'))
R=read('scored_records.csv')
check('records',len(R),10924,0)
check('unique condition-specific keys',not R.duplicated(['crop','condition','county_fips','year']).any(),True)
check('FIPS strings five digits',R.county_fips.str.fullmatch(r'\d{5}').all(),True)
check('threshold action positive only',np.array_equal(np.where(R.benefit_score>0,'A1','A0'),R.action),True)
for q,tau in [(10,.1),(50,.5),(90,.9)]:
    check(f'complete-vector selection {q}',R[f'selector_q{q}'],np.where(R.action=='A1',R[f'A1_q{q}'],R[f'A0_q{q}']))
    for m in ['A0','A1','selector']:
        err=R.yield_mg_ha-R[f'{m}_q{q}']
        loss=np.maximum(tau*err,(tau-1)*err)
        check(f'{m} pinball {q}',loss,R[f'{m}_pin{q}'])
        R[f'{m}_pin{q}']=loss
for m in ['A0','A1','selector']:
    loss=R[[f'{m}_pin{q}' for q in [10,50,90]]].mean(axis=1)
    check(m+' recomputed PB3 versus retained',loss,R[f'{m}_PB3'])
    check(m+' recomputed PB3 versus frozen original',loss,R[f'frozen_{m}_PB3'],1e-8)
    R[f'{m}_PB3']=loss
R['Bnew']=R.A0_PB3-R.A1_PB3;R['Vnew']=R.A1_PB3-R.selector_PB3
for c in ['B','V']:check(c+' row difference',R[c+'new'],R[c]);R[c]=R[c+'new']
R['O']=np.minimum(R.A0_PB3,R.A1_PB3)
R['avoided']=np.where(R.action=='A0',(-R.B).clip(lower=0),0)
R['added']=np.where(R.action=='A0',R.B.clip(lower=0),0)
R['missed']=np.where(R.action=='A1',(-R.B).clip(lower=0),0)
expected_counts={('corn','source'):(2257,30),('corn','held'):(2637,36),('corn','external'):(1173,30),('soybean','source'):(1702,36),('soybean','held'):(1702,36),('soybean','external'):(1453,42)}
summary=read('Figure_2_summary.csv'); decom=read('all_six_loss_decompositions.csv')
summaries=[]
for key,g in R.groupby(['crop','condition']):
    tag='/'.join(key);n,b=expected_counts[key]
    check(tag+' n',len(g),n,0);check(tag+' blocks',g.block.nunique(),b,0)
    check(tag+' weight sum',g.weight.sum(),1);check(tag+' equal block weights',g.weight,equalweights(g))
    check(tag+' end year',g.year.max(),2023 if key==('corn','source') else 2024,0)
    d={'crop':key[0],'condition':key[1],'n':len(g),'blocks':g.block.nunique()}
    for metric,col in [('R0','A0_PB3'),('R1','A1_PB3'),('Rpi','selector_PB3'),('RO','O'),('B','B'),('V','V'),('avoided','avoided'),('added','added'),('missed','missed')]:d[metric]=wm(g[col],g.weight)
    d['H']=d['R1']-d['RO'];d['G']=d['Rpi']-d['RO']
    for source,df in [('S4',summary),('S8',decom)]:
        row=df[(df.crop==key[0])&(df.condition==key[1])].iloc[0]
        for m in ['R0','R1','Rpi','RO','B','V','avoided','added','missed','H','G']:check(tag+' '+source+' '+m,d[m],row[m])
        for m in ['B','V']:check(tag+' '+source+' '+m+' sign',np.sign(d[m]),np.sign(row[m]),0)
    summaries.append(d)
outcsv(TEST/'six_condition_metrics.csv',summaries)
flow=read('S1_flow.csv');annual=read('S1_annual.csv')
for row in flow.itertuples():
    tag=f'S1 {row.crop}/{row.condition}';g=R[(R.crop==row.crop)&(R.condition==row.condition)]
    a=annual[(annual.crop==row.crop)&(annual.condition==row.condition)]
    check(tag+' scored',len(g),row.scored,0)
    check(tag+' prefilter accounting',row.predictor_eligible-row.prefilter,row.forecast,0)
    check(tag+' postforecast accounting',row.forecast-row.post_unscored,row.scored,0)
    check(tag+' yearly forecast sum',a.forecast.sum(),row.forecast,0)
    for ar in a.itertuples():check(tag+' '+str(ar.year)+' scored',len(g[g.year==ar.year]),ar.scored,0)
cuts=read('group_thresholds.csv');groups=read('support_groups.csv')
for row in groups.to_dict('records'):
    g=R[(R.crop==row['crop'])&(R.condition==row['condition'])];c=cuts[(cuts.crop==row['crop'])&(cuts.variable==row['variable'])].iloc[0]
    chosen=np.where(g[row['variable']]<=c.cut_1,'Low',np.where(g[row['variable']]<=c.cut_2,'Middle','High'))
    sub=g.loc[chosen==row['group']];tag='S6 '+'/'.join(str(row[k]) for k in ['crop','condition','variable','group'])
    for field,value in [('n',len(sub)),('counties',sub.county_fips.nunique()),('blocks',sub.block.nunique()),('regions',sub.geo.nunique()),('source_weight_mass',sub.weight.sum())]:check(tag+' '+field,value,row[field])
    for m in ['B','V']:
        check(tag+' '+m,wm(sub[m],sub.weight),row[m]);check(tag+' '+m+' equal block',sub.groupby('block')[m].mean().mean(),row[m+'_equalblock'])
matched=read('Supplementary_Table_S7_matched_absolute_losses.csv')
contrasts=read('Supplementary_Table_S7_paired_contrasts_and_ranges.csv')
for crop in ['corn','soybean']:
    a=R[(R.crop==crop)&(R.condition=='source')].set_index(['county_fips','year'])
    b=R[(R.crop==crop)&(R.condition=='held')].set_index(['county_fips','year'])
    keys=a.index.intersection(b.index);a=a.loc[keys];b=b.loc[keys]
    check(crop+' paired labels',a.yield_mg_ha,b.yield_mg_ha)
    # Corn held-state includes 2024, source ends in 2023; common-sample weights renormalize.
    check(crop+' paired normalized original weights',a.weight/a.weight.sum(),b.weight/b.weight.sum())
    for cond,g in [('source',a),('held',b)]:
        for m in ['A0','A1','selector']:
            row=matched[(matched.crop==crop)&(matched.condition==cond)&(matched.method==m)].iloc[0]
            check(f'S7 {crop}/{cond}/{m}',wm(g[m+'_PB3'],g.weight),row.PB3);check(f'S7 {crop}/{cond}/{m} n',len(g),row.n,0)
    for m in ['B','V']:
        row=contrasts[(contrasts.crop==crop)&(contrasts.metric==m)].iloc[0]
        check('S7 paired contrast '+crop+'/'+m,wm(b[m]-a[m],a.weight),row.difference)
deletions=read('Supplementary_Table_S7_all_deletions.csv')
for i,row in deletions.iterrows():
    g=R[(R.crop==row.crop)&(R.condition==row.condition)]
    sub=g[g[row.axis].astype(str)!=str(row.omitted)]
    # Original Figure 4 deletion_diagnostics lines 157-158 rebuild county means
    # inside each surviving block; state removal can alter a region-year block.
    full=wm(g[row.metric],g.weight);remaining=float(sub.groupby('block')[row.metric].mean().mean())
    for field,value in [('full_gain',full),('remaining_gain',remaining),('influence',remaining-full),('n_remaining',len(sub)),('blocks_remaining',sub.block.nunique())]:check(f'S7 deletion {i} '+field,value,row[field])
    check(f'S7 deletion {i} remaining sign',np.sign(remaining),np.sign(row.remaining_gain),0)
freq=read('all_six_action_frequencies_and_severity.csv')
cats=np.select([(R.action=='A0')&(R.B<0),(R.action=='A0')&(R.B>0),(R.action=='A1')&(R.B>0),(R.action=='A1')&(R.B<0)],['Beneficial fallback','Harmful fallback','Correct retention','Missed opportunity'],default='Tie')
check('S8 categories',np.array_equal(cats,R.category),True)
for i,row in freq.iterrows():
    g=R[(R.crop==row.crop)&(R.condition==row.condition)&(R.category==row.category)]
    for f,v in [('n',len(g)),('weight_share',g.weight.sum()),('mean_absolute_gap_within_category',wm(g.B.abs(),g.weight)),('full_sample_absolute_gap_contribution',float(np.dot(g.B.abs(),g.weight)))]:check(f'S8 frequency {i} '+f,v,row[f])
reliability=read('reliability_full_S10.csv')
for i,row in reliability.iterrows():
    g=R[(R.crop==row.crop)&(R.condition==row.condition)];m=row.method;y=g.yield_mg_ha;qs=g[[m+'_q'+str(q) for q in [10,50,90]]].to_numpy();low,mid,high=qs.T;w=g.weight
    values={'n':len(g),'blocks':g.block.nunique(),'PB3':wm(g[m+'_PB3'],w),'median_MAE':wm(abs(y-mid),w),'lower_tail':wm(y<low,w),'upper_tail':wm(y>high,w),'coverage':wm((low<=y)&(y<=high),w),'interval_width':wm(high-low,w),'adjacent_crossing_n':int(((low>mid)|(mid>high)).sum()),'adjacent_crossing_weight':wm((low>mid)|(mid>high),w),'inverted_n':int((low>high).sum()),'zero_width_n':int((low==high).sum()),'negative_any_n':int((qs<0).any(axis=1).sum()),'negative_any_weight':wm((qs<0).any(axis=1),w),'negative_entries_n':int((qs<0).sum())}
    for q,tau in [(10,.1),(50,.5),(90,.9)]:
        values['below_q'+str(q)]=wm(y<=g[m+'_q'+str(q)],w);values['ties_q'+str(q)+'_n']=int((y==g[m+'_q'+str(q)]).sum());values['deviation_q'+str(q)+'_pp']=100*(values['below_q'+str(q)]-tau)
    for f,v in values.items():check(f'S10 reliability {i} '+f,v,row[f])
gains=read('quantile_gain_and_PB3_contribution.csv')
for i,row in gains.iterrows():
    g=R[(R.crop==row.crop)&(R.condition==row.condition)];q=str(round(row.tau*100))
    gain=wm(g['A1_pin'+q]-g['selector_pin'+q],g.weight)
    check(f'S10 quantile gain {i}',gain,row.gain);check(f'S10 PB3 contribution {i}',gain/3,row.PB3_contribution)
def event(g,m):return np.where(g.yield_mg_ha<g[m+'_q10'],'Below',np.where(g.yield_mg_ha>g[m+'_q90'],'Above','Within'))
trans=read('paired_interval_membership_transitions.csv')
for i,row in trans.iterrows():
    g=R[(R.crop==row.crop)&(R.condition==row.condition)];a=event(g,row.left_method);b=event(g,row.right_method)
    selected=(a==row.left_event)&(b==row.right_event)
    check(f'S10 transition {i} n',selected.sum(),row.n,0);check(f'S10 transition {i} mass',g.loc[selected,'weight'].sum(),row.weight)
exchanges=read('coverage_exchanges.csv')
for i,row in exchanges.iterrows():
    g=R[(R.crop==row.crop)&(R.condition==row.condition)];a=event(g,row.left_method);b=event(g,row.right_method)
    new=g.loc[(a!='Within')&(b=='Within'),'weight'].sum();lost=g.loc[(a=='Within')&(b!='Within'),'weight'].sum()
    for f,v in [('newly_covered_weight',new),('newly_uncovered_weight',lost),('net_coverage_change_pp',100*(new-lost)),('different_state_weight',g.loc[a!=b,'weight'].sum())]:check(f'S10 exchange {i} '+f,v,row[f])
P=json.loads(text(ROOT/'config/reporting_values.json'))
for row in P['tables']['Table S9']['sections'][0]['rows']:
    crop,cond,z,*expected=row
    if z!='Z2':continue
    g=R[(R.crop==crop.lower())&(R.condition=={'Temporal':'source','Held-state':'held','External':'external'}[cond])]
    x=g.benefit_score;y=g.B;w=g.weight;mx=wm(x,w);my=wm(y,w)
    slope=wm((x-mx)*(y-my),w)/wm((x-mx)**2,w);intercept=my-slope*mx
    nonzero=g.B!=0;agreement=wm((g.action[nonzero]=='A1')==(g.B[nonzero]>0),w[nonzero])*100
    vals=[wm(g.selector_PB3,w),wm(abs(x-y),w),agreement,intercept,slope]
    for name,v,e,tol in zip(['PB3','score MAE','agreement','intercept','slope'],vals,expected,[.5e-6,.5e-6,.005,.5e-6,.5e-6]):check(f'S9 main {crop}/{cond}/{name}',v,e,tol+1e-12)
outcsv(TEST/'frozen_numeric_checks.csv',checks)

dump('independent_recompute_summary.json',{'checks':len(checks),'passed':sum(x['pass'] for x in checks),'failed':[x for x in checks if not x['pass']],'environment':{'python':platform.python_version(),**{x:importlib.metadata.version(x) for x in ['numpy','pandas']}},'script_sha256':sha(__file__),'no_training':True,'no_network':True})
print(json.dumps({'checks':len(checks),'failure_count':sum(not x['pass'] for x in checks),'failures':[x for x in checks if not x['pass']][:20]},indent=2))
print(pd.DataFrame(summaries)[['crop','condition','n','blocks','R0','R1','Rpi','B','V']].to_string(index=False))

assert all(c['pass'] for c in checks)
