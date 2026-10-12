"""Read original 2019-2024 alternative forecasts and re-evaluate; no fitting."""
from common import *
from original_resampling import paired_two_way,two_way_descriptive
import numpy as np,pandas as pd,ast

R=pd.read_csv(ROOT/'data/scored_records.csv',dtype={'county_fips':str},float_precision='round_trip')
P=json.loads(text(ROOT/'config/reporting_values.json')); checks=[];access=[];outputs=[];frames={}
def check(label,v,e,tol=1e-10):
    err=float(np.max(np.abs(np.asarray(v,dtype=float)-np.asarray(e,dtype=float))))
    checks.append({'check':label,'pass':bool(err<=tol),'max_abs_error':err,'tolerance':tol})
def wm(v,w):return float(np.average(v,weights=w))
for crop,zs,states in [('corn',['Z0','Z1','Z3'],['IA','IL','IN','KS','NE','TX']),('soybean',['Z1'],['AR','IA','KS','MI','NC','ND'])]:
    ref=R[(R.crop==crop)&(R.condition=='held')]
    for z in zs:
        raw=pd.read_csv(ROOT/f'data/S9_{crop}_{z}.csv',dtype={'county_fips':str},float_precision='round_trip')
        assert raw.year.between(2019,2024).all()
        if crop=='soybean':
            raw_z2=pd.read_csv(ROOT/'data/S9_soybean_Z2.csv',dtype={'county_fips':str},float_precision='round_trip')
            both=raw.merge(raw_z2,on=['state','county_fips','year'],suffixes=('_Z1','_Z2'),validate='one_to_one')
            for (st,yr),part in both.groupby(['state','year']):
                for aq in [f'{a}_q{q}_mg_ha' for a in ['A0','A1'] for q in [10,50,90]]:check(f'Soy raw exact endpoints {st}/{yr}/{aq}',part[aq+'_Z1'],part[aq+'_Z2'],0)
        f=ref.merge(raw,on=['state','county_fips','year'],suffixes=('_Z2',''),validate='one_to_one')
        check(crop+'/'+z+' scored keys',len(f),len(ref),0)
        check(crop+'/'+z+' threshold',(f.action==np.where(f.benefit_score>0,'A1','A0')).all(),True,0)
        selected=np.where(f.action=='A1',f.A1_PB3,f.A0_PB3);w=f.weight;x=f.benefit_score;y=f.B;xm=wm(x,w);ym=wm(y,w)
        slope=wm((x-xm)*(y-ym),w)/wm((x-xm)**2,w);intercept=ym-slope*xm
        values=[wm(selected,w),wm(abs(x-y),w),100*wm((f.action=='A1')==(f.B>0),w),intercept,slope]
        row=next(r for r in P['tables']['Table S9']['sections'][0]['rows'] if r[:3]==[crop.capitalize(),'Held-state',z])
        for name,v,e,tol in zip(['PB3','score_MAE','agreement_percent','intercept','slope'],values,row[3:],[.5e-6,.5e-6,.005,.5e-6,.5e-6]):check(crop+'/'+z+'/'+name,v,e,tol+1e-12)
        outputs.append(dict(zip(['crop','condition','Z','PB3','score_MAE','agreement_percent','intercept','slope'],[crop,'held',z]+values)))
        f['selected_PB3']=selected;frames[(crop,z)]=f
        if crop=='soybean':
            for a in ['A0','A1']:
                for q in [10,50,90]:check('Soy Z1 endpoint identity '+a+str(q),f[f'{a}_q{q}'],f[f'{a}_q{q}_mg_ha'])
arch=pd.read_csv(ROOT/'expected/QA2_PAIRED_TWO_WAY_RESAMPLING.csv',float_precision='round_trip')
blocks=[]
for z in ['Z0','Z1','Z2','Z3']:
    f=frames[('corn',z)] if z!='Z2' else R[(R.crop=='corn')&(R.condition=='held')].assign(selected_PB3=lambda x:x.selector_PB3)
    b=f.groupby(['state','year'],as_index=False).selected_PB3.mean().rename(columns={'selected_PB3':'PB3'});b['policy']=z;b['condition']='UNSEEN_GEOGRAPHY_FUTURE_YEAR';blocks.append(b)
blocks=pd.concat(blocks,ignore_index=True);ranges=[]
for left,right in [('Z1','Z0'),('Z2','Z1'),('Z3','Z2')]:
    res=paired_two_way(blocks,'UNSEEN_GEOGRAPHY_FUTURE_YEAR',left,right)
    row=arch[(arch.left.replace({'TV2_L2_G0_R1':'Z2'})==left)&(arch.right.replace({'TV2_L2_G0_R1':'Z2'})==right)&(arch.metric=='PB3')].iloc[0]
    for field in ['observed_left_minus_right','two_way_2p5','two_way_97p5']:check('S9 corn '+left+'-'+right+' '+field,res[field],row[field])
    ranges.append({'crop':'corn','left':left,'right':right,'point':res['observed_left_minus_right'],'lo':res['two_way_2p5'],'hi':res['two_way_97p5'],'draws':10000,'seed':20260913,'source':'QA2 raw forecasts; exact original algorithm replay'})
f=frames[('soybean','Z1')];f['difference']=f.selector_PB3-f.selected_PB3
b=f.groupby(['state','year'],as_index=False).difference.mean().rename(columns={'state':'block'})
res=two_way_descriptive(b,'difference')
archsoy=json.loads(text(ROOT/'expected/soybean_S9_original_archive.json'))
for field in ['point','lower_2p5','upper_97p5']:check('S9 soybean '+field,res[field],archsoy[field],1e-10)
ranges.append({'crop':'soybean','left':'Z2','right':'Z1','point':res['point'],'lo':res['lower_2p5'],'hi':res['upper_97p5'],'draws':2000,'seed':20260920,'source':'TV5 original frozen forecasts and archived RQ4 full precision'})
outcsv(TEST/'S9_recovered_diagnostics.csv',outputs);outcsv(TEST/'S9_recovered_ranges.csv',ranges);outcsv(TEST/'S9_recovery_checks.csv',checks)
dump('s9_recovery_summary.json',{'checks':len(checks),'passed':sum(r['pass'] for r in checks),'failed':[r for r in checks if not r['pass']],'sources':len(access),'status':'ORIGINAL_FORECASTS_RECOVERED_WITH_RETAINED_CSV_PRECISION_WARNING','summary_only':False,'script_sha256':sha(__file__),'note':'Six raw-Parquet versus retained-CSV checks retain their 1e-10 threshold and failures; direct raw Z1 versus raw Z2 endpoint identity is separately tested at exact zero tolerance. No source or tolerance changed.'})
print(json.dumps({'checks':len(checks),'failures':[r for r in checks if not r['pass']],'recovered_ranges':ranges},indent=2),flush=True)

assert all(c['pass'] or c['check'].startswith('Soy Z1 endpoint identity ') for c in checks)
assert sum(not c['pass'] for c in checks)==6
