from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
OUT=ROOT/'results/diagnostics/fd003_prototype_probe_20261103'
m=json.loads((RUN/'manifest.json').read_text())
for section in ['input_hashes','output_hashes']:
 for rel,h in m[section].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
assert m['ridge_fits_completed']==2
align=json.loads((RUN/'alignment_audit.json').read_text())
assert align['additional_ridge_fits']==0
assert hashlib.sha256((ROOT/'analysis/diagnostics/audit_fd003_prototype_alignment.py').read_bytes()).hexdigest()==align['source_sha256']
matched=pd.read_csv(OUT/'aligned_matched_windows.csv')
assert (matched.rul_a==matched.rul_b).all() and (matched.engine_a!=matched.engine_b).all()
assert (matched.rul_a>=31).all() and (matched.rul_a<=60).all()
for (a,b),t in matched.groupby(['prototype_a','prototype_b']):
 assert set(t[t.split=='train'].rul_a)==set(t[t.split=='validation'].rul_a)
 for split,d in t.groupby('split'):
  assert d.index_a.is_unique and d.index_b.is_unique
  assert d.groupby('rul_a').size().min()>=3
pred=np.load(OUT/'probe_predictions.npz');table=pd.read_csv(OUT/'probe_summary.csv')
config=json.loads((ROOT/'configs/diagnostics/fd003_prototype_probe.json').read_text())
for name,lo,hi in config['bins']:
 take=(pred['label']>=lo)&(pred['label']<=hi);ids=np.unique(pred['engine_id'][take]);ms={}
 for group in ['A','B']:
  ms[group]=np.array([np.mean((pred[group][take&(pred['engine_id']==e)]-pred['label'][take&(pred['engine_id']==e)])**2) for e in ids])
 r=table[table.bin==name].iloc[0]
 assert np.allclose([np.sqrt(ms['A'].mean()),np.sqrt(ms['B'].mean())],[r.A_RMSE,r.B_RMSE],atol=1e-10)
 assert int((ms['B']<ms['A']).sum())==r.improved_engines
v=dict(passed=True,all_input_output_hashes_verified=True,two_fits_only=True,no_additional_fit_in_alignment=True,exact_RUL_different_engine_no_replacement_matching=True,same_RUL_support_across_train_validation=True,all_probe_RMSE_and_improved_engine_counts_recomputed=True)
(RUN/'final_verification.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(v,indent=2))
