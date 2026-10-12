"""No-fit checks of the preserved original loss and exclusion-rule modules."""
from pathlib import Path
import importlib,json,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'methods'))
from tv2_pipeline import loss,task_graph
import external_task_graph

def main():
    rows=json.loads((ROOT/'methods/SOURCE_VERSIONS.json').read_text())
    imported=[]
    for row in rows:
        p=Path(row['candidate_path']);name='.'.join(p.with_suffix('').parts[1:])
        if name.endswith('.__init__'):name=name[:-9]
        importlib.import_module(name);imported.append(name)
    r=pd.read_csv(ROOT/'results/inputs/data/scored_records.csv',float_precision='round_trip',dtype={'county_fips':str})
    count=0
    for (crop,cond),g in r.groupby(['crop','condition']):
        y=g.yield_mg_ha.to_numpy();a0=g[['A0_q10','A0_q50','A0_q90']].to_numpy();a1=g[['A1_q10','A1_q50','A1_q90']].to_numpy()
        for name,p in [('A0',a0),('A1',a1)]:
            assert np.max(np.abs(loss.pinball(y,p)-g[name+'_PB3']))<=1e-10;count+=1
        action=loss.action_from_delta(g.benefit_score.to_numpy())
        assert np.array_equal(action,(g.action=='A1').astype('int8'));count+=1
        assert np.array_equal(loss.selected_prediction(a0,a1,action),g[['selector_q10','selector_q50','selector_q90']].to_numpy());count+=1
        w=loss.block_weights(g.geo.to_numpy(),g.year.to_numpy());assert np.max(np.abs(w/w.sum()-g.weight))<=1e-10;count+=1
    shas={k:'0'*64 for k in ['reference_protocol_sha','feature_registry_sha','base_config_sha','fold_selection_sha','code_sha']}
    tasks=0
    for module,state in [(task_graph,'IA'),(external_task_graph,'EXTERNAL')]:
        p=module.plan(state,2024,shas)
        for t in p['tasks']:
            assert t['label_year']<t['outer_year']
            assert state not in t['allowed_states']
            assert not set(t['allowed_states'])&set(t['excluded_states'])
            if t['purpose']=='router_train_meta':assert t['label_year']<t['validation_year']
            tasks+=1
    out={'imported_original_modules':len(imported),'frozen_science_checks':count,'metadata_only_tasks_checked':tasks,'model_fit_calls':0,'level_B':'NOT_RUN','historical_environment_proven':False}
    (ROOT/'results/method_source_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
