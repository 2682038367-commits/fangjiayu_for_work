from pathlib import Path
import hashlib,json,datetime
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).resolve().parent
OUT=ROOT/'results/extensions'/RUN.name
manifest=json.loads((RUN/'manifest.json').read_text())
for p,h in manifest['hashes'].items():
 assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
train=np.load(RUN/'train.npz'); val=np.load(RUN/'validation.npz'); plan=np.load(RUN/'plan.npz'); proto=np.load(RUN/'prototypes.npz')
assert not set(train['engine_id']) & set(val['engine_id'])
assert np.array_equal(np.unique(train['engine_id']),proto['fit_engine_ids'])
segments=[]; pairs=[]
for seed in [23,13]:
 audits=[json.loads((OUT/f'seed{seed}'/g/'training.json').read_text()) for g in ['A','B']]
 for key in ['shared_initial_hash','order_hash','rng_hash','updates']:
  assert audits[0][key]==audits[1][key],(seed,key)
 assert audits[0]['updates']==3080
 for epoch in [64,70]:
  samples=[np.load(OUT/f'seed{seed}'/g/f'samples{epoch}.npz') for g in ['A','B']]
  sa=[json.loads((OUT/f'seed{seed}'/g/f'sampling{epoch}.json').read_text()) for g in ['A','B']]
  assert sa[0]['rng_hash']==sa[1]['rng_hash']
  assert np.array_equal(samples[0]['label'],samples[1]['label']) and np.array_equal(samples[0]['label'],plan['target_rul'])
  assert samples[0]['data'].shape==samples[1]['data'].shape==(11056,48,14)
  assert all(np.isfinite(z['data']).all() for z in samples)
  for es in [3,13,23,33,43]:
   ms=[]
   for g in ['A','B']:
    d=OUT/f'seed{seed}'/g/f'epoch{epoch}_eval{es}'
    pred=np.load(d/'predictions.npz');m=json.loads((d/'metrics.json').read_text());ms.append(m)
    assert np.array_equal(pred['label'],val['label'].reshape(-1))
    assert np.array_equal(pred['engine_id'],val['engine_id'])
    sq=(pred['prediction'].astype('float64')-pred['label'])**2
    score=np.sqrt(np.mean([sq[pred['engine_id']==i].mean() for i in np.unique(pred['engine_id'])]))
    assert abs(score-m['rmse'])<1e-5,(d,score,m['rmse'])
    table=pd.read_csv(d/'segments.csv');table['generator_seed']=seed;table['epoch']=epoch;table['group']=g;table['predictor_seed']=es;segments.append(table)
   assert ms[0]['updates']==ms[1]['updates']==1085
   assert ms[0]['order_hash']==ms[1]['order_hash']
   pairs.append(dict(generator_seed=seed,epoch=epoch,predictor_seed=es,A=ms[0]['rmse'],B=ms[1]['rmse'],delta=ms[1]['rmse']-ms[0]['rmse']))
pd.DataFrame(pairs).to_csv(OUT/'paired_recomputed.csv',index=False)
s=pd.concat(segments,ignore_index=True)
a=s.groupby(['generator_seed','epoch','group','bin'],sort=False).mse_engine.mean().pow(.5).unstack('group').reset_index();a['B-A']=a.B-a.A
a.to_csv(OUT/'segment_summary.csv',index=False)
status=RUN/'execution_status.json'
end=datetime.datetime.fromtimestamp(status.stat().st_mtime,datetime.timezone(datetime.timedelta(hours=8)))
start=datetime.datetime.fromisoformat(json.loads((RUN/'background_launch.json').read_text())['launched_at'])
audit=dict(passed=True,source_and_input_hashes_verified=True,train_validation_disjoint=True,train_only_prototype_engine_ids=True,generators=4,sample_sets=8,predictors=40,generator_updates_per_model=3080,predictor_updates_per_model=1085,shared_initialization_order_rng_equal=True,sampling_rng_and_labels_equal=True,predictor_order_equal=True,real_rmse_recomputed=True,finished_at=end.isoformat(),elapsed_seconds=(end-start).total_seconds())
(RUN/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
print(a[a['bin']=='all'].to_string(index=False))
