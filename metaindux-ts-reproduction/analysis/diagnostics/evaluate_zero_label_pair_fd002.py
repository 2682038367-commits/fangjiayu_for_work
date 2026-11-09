"""Evaluate the preregistered FD002 seed3 whole-batch zeroing intervention."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/diagnostics/zero_label_pair_fd002_seed3'
SEEDS=[3,13,23,33,43]
BINS=[('0',0,0),('1',1,1),('2-5',2,5),('6-30',6,30),('31-60',31,60),
      ('61-90',61,90),('91-124',91,124),('125',125,125),('all',0,125)]


def main():
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'upstream'))
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    labels=None
    for arm in ['control','no_batch_zero']:
        path=ROOT/f'outputs/FD002_w48_seed3_zero_label_{arm}.npz'
        with np.load(path) as z:
            x,y=z['data'],z['label'].ravel().astype(np.float64)
        assert x.shape==(41539,48,14) and np.isfinite(x).all()
        if labels is None:labels=y.copy()
        else:assert np.array_equal(labels,y)
        predictions=[]
        for e in SEEDS:
            model=torch.load(ROOT/f'checkpoints/evaluators/real_data_baseline_w48/FD002_eval{e}.pth',
                             map_location='cpu',weights_only=False).eval()
            with torch.no_grad():
                p=np.concatenate([model(torch.from_numpy(x[i:i+256])).numpy().ravel()
                                  for i in range(0,len(x),256)]).astype(np.float64)
            assert np.isfinite(p).all()
            predictions.append(p)
            for name,lo,hi in BINS:
                mask=(y>=lo)&(y<=hi);err=p[mask]-y[mask]
                rows.append(dict(arm=arm,evaluator_seed=e,bin=name,n_windows=int(mask.sum()),
                                 label_mean=float(y[mask].mean()),prediction_mean=float(p[mask].mean()),
                                 bias=float(err.mean()),mse=float(np.mean(err*err)),
                                 rmse=float(np.sqrt(np.mean(err*err))),mae=float(np.mean(abs(err)))))
        np.savez_compressed(OUT/f'{arm}_predictions.npz',label=y,prediction=np.array(predictions),
                            evaluator_seeds=np.array(SEEDS))
    with (OUT/'per_model_bins.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries=[]
    for name,lo,hi in BINS:
        d={'bin':name}
        for arm in ['control','no_batch_zero']:
            selected=[r for r in rows if r['arm']==arm and r['bin']==name]
            d[arm+'_prediction_mean']=float(np.mean([r['prediction_mean'] for r in selected]))
            d[arm+'_bias']=float(np.mean([r['bias'] for r in selected]))
            d[arm+'_rmse']=float(np.sqrt(np.mean([r['mse'] for r in selected])))
        d['rmse_ratio_fix_over_control']=d['no_batch_zero_rmse']/d['control_rmse']
        summaries.append(d)
    with (OUT/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    z=summaries[0]
    verdict=dict(primary='RUL0 consistency judged by same five real-trained predictors',
                 baseline_anomaly_reproduced=z['control_prediction_mean']>30,
                 mean_prediction_reduced_by_half=z['no_batch_zero_prediction_mean']<=.5*z['control_prediction_mean'],
                 rmse_reduced_by_half=z['no_batch_zero_rmse']<=.5*z['control_rmse'],
                 limitation='One generation seed; diagnostic intervention, not strict paper result or physical RUL ground truth.',
                 comparison=summaries)
    verdict['supports_zero_label_collision']=all(verdict[k] for k in ['baseline_anomaly_reproduced','mean_prediction_reduced_by_half','rmse_reduced_by_half'])
    (OUT/'verdict.json').write_text(json.dumps(verdict,indent=2))
    print(json.dumps(verdict,indent=2),flush=True)


if __name__=='__main__':main()
