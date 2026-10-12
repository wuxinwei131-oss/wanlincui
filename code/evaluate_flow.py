from common import *
import numpy as np,pandas as pd
R=pd.read_csv(ROOT/'data/scored_records.csv',dtype={'county_fips':str},float_precision='round_trip')
f=pd.read_csv(ROOT/'data/forecast_keys.csv',dtype={'county_fips':str})
s=pd.read_csv(ROOT/'data/predictor_support.csv',dtype={'GEOID':str},float_precision='round_trip')
assert f.year.between(2019,2024).all() and s.year.between(2019,2024).all()
f=f.merge(R[['crop','condition','county_fips','year']].assign(scored=True),how='left',on=['crop','condition','county_fips','year'],validate='one_to_one');f.scored=f.scored.fillna(False).astype(bool)
flow=[];annual=[]
for crop,cond in R.groupby(['crop','condition']).groups:
    g=f[(f.crop==crop)&(f.condition==cond)];e=s[(s.crop==crop)&(s.domain==('external' if cond=='external' else 'development'))]
    if crop=='corn' and cond=='source':e=e[e.year<=2023]
    flow.append(dict(crop=crop,condition=cond,predictor_eligible=len(e),forecast=len(g),scored=int(g.scored.sum()),prefilter=len(e)-len(g),post_unscored=int((~g.scored).sum())))
    for y,h in g.groupby('year'):annual.append(dict(crop=crop,condition=cond,year=y,forecast=len(h),scored=int(h.scored.sum()),counties=h.county_fips.nunique()))
for name,records,keys in [('S1_flow.csv',flow,['crop','condition']),('S1_annual.csv',annual,['crop','condition','year'])]:
    a=pd.DataFrame(records);b=pd.read_csv(ROOT/'expected'/name);j=a.merge(b,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    for c in a.columns:
        if c not in keys:assert np.array_equal(j[c+'_new'],j[c+'_old']),c
    a.to_csv(TEST/name,index=False)
cuts=pd.read_csv(ROOT/'expected/group_thresholds.csv');computed=[]
for r in cuts.itertuples():
    g=s[(s.crop==r.crop)&(s.domain=='development')].copy()
    g['w']=1/(g.block_id.nunique()*g.groupby('block_id').block_id.transform('size'))
    ordered=g.sort_values([r.variable,'GEOID','year']);cdf=ordered.w.cumsum()/ordered.w.sum()
    lo,hi=[float(ordered[r.variable].iloc[np.searchsorted(cdf,q,side='left')]) for q in [1/3,2/3]]
    assert abs(lo-r.cut_1)<=1e-9 and abs(hi-r.cut_2)<=1e-9
    assert len(g)==r.reference_n
    computed.append(dict(crop=r.crop,variable=r.variable,cut_1=lo,cut_2=hi,reference_n=len(g)))
pd.DataFrame(computed).to_csv(TEST/'S6_thresholds_recomputed.csv',index=False,float_format='%.17g')
print('S1 forecast flow and S6 predictor-based thresholds passed')
