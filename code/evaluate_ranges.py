from common import *
from original_resampling import resample,limits
import ast,numpy as np,pandas as pd
checks=[];sources=[]
def check(name,v,e,tol=1e-10):
    err=float(np.max(abs(np.asarray(v,dtype=float)-np.asarray(e,dtype=float))))
    checks.append({'check':name,'pass':bool(err<=tol),'max_abs_error':err,'tolerance':tol})
R=pd.read_csv(ROOT/'data/scored_records.csv',dtype={'county_fips':str},float_precision='round_trip');S=pd.read_csv(ROOT/'expected/Figure_2_summary.csv',float_precision='round_trip')
for row in S.itertuples():
    f=R[(R.crop==row.crop)&(R.condition==row.condition)]
    b=f.groupby(['geo','year'],as_index=False)[['B','V']].mean()
    for metric in ['B','V']:
        tag=f'S4 {row.crop}/{row.condition}/{metric}'
        if row.crop=='corn' and row.condition!='source':
            axis='state' if row.condition=='held' else 'ERS_region'
            tmp=f.copy();tmp.ERS_region=tmp.ERS_region.fillna(0).astype(int).astype(str)
            delta=tmp.groupby([axis,'year'])[metric].mean();lo,hi,n=limits(delta,axis,20260914 if row.condition=='held' else 20260917,5000)
        else:
            lo,hi,n,seed,draws=resample(b,metric,row.crop)
            drawfile=ROOT/f'data/draws/Figure_2_{row.crop}_{row.condition}_{metric}_draws.csv'
            if drawfile.exists():
                d=pd.read_csv(drawfile,float_precision='round_trip');check(tag+' retained draw sequence',draws,d.iloc[:,-1].to_numpy(),1e-12);sources.append({'path':str(drawfile),'sha256':sha(drawfile),'role':'retained S4 descriptive draw vector'})
        check(tag+' lower',lo,getattr(row,metric+'lo'));check(tag+' upper',hi,getattr(row,metric+'hi'));check(tag+' count',n,getattr(row,metric+'_range_n'),0)
contrasts=pd.read_csv(ROOT/'expected/Supplementary_Table_S7_paired_contrasts_and_ranges.csv',float_precision='round_trip')
for crop in ['corn','soybean']:
    a=R[(R.crop==crop)&(R.condition=='source')];b=R[(R.crop==crop)&(R.condition=='held')]
    m=a.merge(b,on=['county_fips','year','state'],suffixes=('_s','_h'),validate='one_to_one');m['geo']=m.state
    for metric in ['B','V']:
        m[metric]=m[metric+'_h']-m[metric+'_s'];blocks=m.groupby(['geo','year'],as_index=False)[metric].mean();lo,hi,n,seed,_=resample(blocks,metric,crop)
        row=contrasts[(contrasts.crop==crop)&(contrasts.metric==metric)].iloc[0]
        for field,value in [('lo',lo),('hi',hi),('draws',n),('seed',seed)]:check('S7 '+crop+'/'+metric+'/'+field,value,row[field])
outcsv(TEST/'range_checks.csv',checks)
assert all(c['pass'] for c in checks)
print('S4/S7 ranges',len(checks),'passed')
