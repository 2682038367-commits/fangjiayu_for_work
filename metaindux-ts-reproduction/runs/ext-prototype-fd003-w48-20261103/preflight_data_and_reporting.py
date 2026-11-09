from pathlib import Path
import sys,json,tempfile,shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
ROOT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'upstream'));sys.path.insert(0,str(ROOT/'scripts'))
from data.CMAPSSDataset import columns,feature_columns
from prototype_guidance import features,BINS
from summarize_prototype_fd003 import summarize_segments
import summarize_prototype_fd003 as reporting
train=np.load(RUN/'train.npz');val=np.load(RUN/'validation.npz');stats=np.load(RUN/'prototypes.npz');norm=np.load(RUN/'normalization.npz')
f=pd.read_csv(ROOT/'upstream/data/train_FD003.txt',sep=r'\s+',header=None,names=columns)
scaler=MinMaxScaler().fit(f.loc[f.id.isin(norm['fit_engine_ids']),feature_columns])
assert np.array_equal(scaler.data_min_,norm['data_min']) and np.array_equal(scaler.data_max_,norm['data_max'])
assert not set(norm['fit_engine_ids']) & set(val['engine_id'])
z=features(train['data']);assert np.allclose(z.mean(0,dtype='float64'),stats['feature_mean'])
z=(z-stats['feature_mean'])/stats['feature_scale']
ids=np.argmin(((z[:,None,:]-stats['centers'][None,:,:])**2).sum(2),axis=1)
assert np.array_equal(ids,stats['prototype_id'])
y=train['label'].reshape(-1)
for i,(_,lo,hi) in enumerate(BINS):
 mask=(y>=lo)&(y<=hi);counts=np.bincount(ids[mask],minlength=4)/mask.sum()
 assert np.allclose(counts,stats['frequency'][i])
# Exercise new overall+target reporting against prior completed FD001 result,
# copying only metric tables into a temporary directory; never alter its files.
parent=ROOT/'results/extensions/ext-prototype-fd001-w48-20261103'
with tempfile.TemporaryDirectory() as tmp:
 reporting.OUT=Path(tmp)
 for p in parent.glob('seed*/*/epoch*_eval*/segments.csv'):
  dst=reporting.OUT/p.relative_to(parent);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
 shutil.copy2(parent/'selected.json',reporting.OUT/'selected.json')
 summarize_segments()
 got=pd.read_csv(reporting.OUT/'segment_summary.csv');old=pd.read_csv(parent/'segment_summary.csv')
 for row in got.itertuples():
  prior=old[(old.generator_seed==row.generator_seed)&(old.epoch==row.epoch)&(old.bin==row.bin)].iloc[0]
  assert np.allclose([row.A,row.B],[prior.A,prior.B])
 choices=json.loads((reporting.OUT/'selected_comparison.json').read_text())
 assert len(choices)==2 and not all(r['both_improve'] for r in choices)
 assert choices[0]['A_epoch']==64 and choices[0]['B_epoch']==70
(RUN/'data_reporting_checks.json').write_text(json.dumps(dict(passed=True,minmax_refit_train_only=True,prototype_assignment_and_frequency_recomputed=True,reporting_matches_completed_parent_metrics=True,selected_stage_uses_overall_selected_epochs=True,validation_windows=len(val['data']),target_windows=int(((val['label']>=31)&(val['label']<=60)).sum())),indent=2)+'\n')
print('DATA AND REPORTING CHECKS PASSED')
