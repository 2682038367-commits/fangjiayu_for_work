"""Expand the existing inverse-condition audit to every generated RUL 0/1 window."""
from pathlib import Path
import csv
import hashlib
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis/diagnostics'))
from real_predictor_condition_audit import infer, SEEDS


def main():
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'upstream'))
    run=ROOT/'runs/diagnostic-real-predictor-condition-w48-20261023'
    out=ROOT/'results/diagnostics/real_predictor_condition_w48'
    now=lambda:datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    m=dict(status='running',scope_class='diagnostic',role='supporting',
           parent_run=run.name,started_at=now(),no_training=True,
           purpose='all-window coverage check, not an independent causal experiment',
           command=f'{sys.executable} analysis/diagnostics/check_all_zero_rul.py',
           input_hashes={})
    def bind(p):m['input_hashes'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    bind(Path(__file__))
    bind(ROOT/'upstream/TrainCondition.py')
    bind(ROOT/'upstream/DiffusionFreeGuidence/Unet1D_fre.py')
    rows=[]
    for ds in ['FD001','FD002','FD003','FD004']:
        models=[]
        for e in SEEDS:
            p=ROOT/f'checkpoints/evaluators/real_data_baseline_w48/{ds}_eval{e}.pth'
            bind(p)
            models.append(torch.load(p,map_location='cpu',weights_only=False).eval())
        for g in SEEDS:
            p=ROOT/f'outputs/{ds}_w48_seed{g}.npz'
            bind(p)
            with np.load(p) as z:
                x=z['data'];y=z['label'].ravel()
            for label in [0,1]:
                xx=x[y==label]
                assert len(xx)>0
                for e,model in zip(SEEDS,models):
                    pred=infer(model,xx)
                    rows.append(dict(dataset=ds,generation_seed=g,evaluator_seed=e,rul=label,
                                     n_windows=len(xx),prediction_mean=float(pred.mean()),
                                     prediction_median=float(np.median(pred)),
                                     mse=float(np.mean((pred-label)**2)),
                                     mae=float(np.mean(abs(pred-label))),
                                     bias=float(np.mean(pred-label))))
        print(ds,'all RUL0/1 checked',flush=True)
    p=out/'all_generated_rul0_rul1.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    m.update(status='complete',finished_at=now(),output_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (run/'zero_rul_fullcheck_manifest.json').write_text(json.dumps(m,indent=2))


if __name__=='__main__':main()
