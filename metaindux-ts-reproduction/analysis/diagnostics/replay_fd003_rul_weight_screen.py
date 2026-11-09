"""Numerical replay audit only: do not modify fits or selection inputs."""
from pathlib import Path
import argparse
import hashlib
import os
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/ext-fd003-rul-weight-screen-20261027'
OUT=ROOT/'results/exploratory/fd003_rul_weight_screen_20261027'


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--device',choices=['cpu','cuda'],required=True);args=cli.parse_args()
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    sys.path.insert(0,str(ROOT/'upstream'))
    from models.LSTM import LSTM
    from types import SimpleNamespace
    with np.load(RUN/'real_validation.npz') as z:x=z['data'];y=z['label'].astype(float);ids=z['engine_id']
    tx=torch.from_numpy(x).to(args.device);rows=[]
    (OUT/'replay').mkdir(exist_ok=True)
    for r in pd.read_csv(OUT/'training_audit.csv').itertuples():
        assert hashlib.sha256((ROOT/r.checkpoint).read_bytes()).hexdigest()==r.checkpoint_sha256
        ck=torch.load(ROOT/r.checkpoint,map_location='cpu',weights_only=True)
        model=LSTM(SimpleNamespace(**ck['architecture'])).to(args.device).eval();model.load_state_dict(ck['state_dict'])
        with torch.inference_mode():
            p=np.concatenate([model(tx[start:start+512]).cpu().numpy().ravel() for start in range(0,len(tx),512)]).astype(float)
        name=f'gen{r.generation_seed}_weight{r.weight}_init3'
        with np.load(OUT/'predictions'/f'{name}.npz') as z:saved=z['real_validation']
        difference=abs(p-saved);maxdelta=float(difference.max())
        if args.device=='cuda':assert maxdelta<.001,(name,maxdelta)
        np.save(OUT/'replay'/f'{args.device}_{name}.npy',p)
        for stage,lo,hi in [('all',0,125),('31-60',31,60),('0-30',0,30),('61-90',61,90),('91-124',91,124),('125',125,125)]:
            mask=(y>=lo)&(y<=hi);means=[];original=[]
            for engine in np.unique(ids[mask]):
                take=mask&(ids==engine);means.append(np.mean((p[take]-y[take])**2));original.append(np.mean((saved[take]-y[take])**2))
            rows.append(dict(device=args.device,generation_seed=r.generation_seed,weight=r.weight,bin=stage,
                max_prediction_difference=maxdelta,cpu_absolute_1e3_gate_pass=maxdelta<.001,
                rmse_engine=float(np.sqrt(np.mean(means))),mse_engine=float(np.mean(means)),
                saved_rmse_engine=float(np.sqrt(np.mean(original))),
                rmse_difference=float(np.sqrt(np.mean(means))-np.sqrt(np.mean(original))),
                cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32))
    result=pd.DataFrame(rows);result.to_csv(OUT/f'{args.device}_numerical_replay.csv',index=False)
    print(result[result.bin=='all'].round(7).to_string(index=False))


if __name__=='__main__':main()
