"""Audit and visualize the completed 20-run downstream diagnostic."""
from pathlib import Path
import hashlib
import json
import os

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=ROOT/'results/diagnostics/zero_label_downstream_20261026'
    run=ROOT/'runs/diagnostic-zero-label-downstream-20261026'
    mp=run/'run-manifest.json';m=json.loads(mp.read_text())
    if m['status']!='complete':raise ValueError('training is not complete')
    raw=pd.read_csv(out/'raw.csv');seg=pd.read_csv(out/'per_model_segments.csv')
    pred=pd.read_csv(out/'predictions.csv');comp=pd.read_csv(out/'comparison.csv')
    assert len(raw)==20 and len(seg)==120 and not raw.duplicated(['dataset','arm','evaluator_seed']).any()
    assert set(raw.evaluator_seed)=={3,13,23,33,43}
    for p,h in m['inputs_sha256'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    for r in raw.itertuples():
        p=ROOT/r.checkpoint;assert p.is_file()
        a=pred[(pred.dataset==r.dataset)&(pred.arm==r.arm)&(pred.evaluator_seed==r.evaluator_seed)]
        expected=100 if r.dataset=='FD001' else 259
        assert len(a)==expected and a.engine_id.nunique()==expected
        assert abs(np.sqrt(np.mean(a.error**2))-r.rmse)<.001
        for row in seg[(seg.dataset==r.dataset)&(seg.arm==r.arm)&(seg.evaluator_seed==r.evaluator_seed)].itertuples():
            if row.bin=='all':sub=a
            elif row.bin=='125':sub=a[a.rul==125]
            else:
                lo,hi=map(int,row.bin.split('-'));sub=a[(a.rul>=lo)&(a.rul<=hi)]
            assert len(sub)==row.n_engines and abs(np.sqrt(np.mean(sub.error**2))-row.rmse)<1e-8
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ds,ax in zip(['FD001','FD002'],axes):
        pivot=raw[raw.dataset==ds].pivot(index='evaluator_seed',columns='arm',values='rmse')
        for seed,row in pivot.iterrows():ax.plot([0,1],[row.control,row.no_batch_zero],marker='o',label=str(seed))
        ax.set_xticks([0,1],['Original generator','No batch label zeroing'])
        ax.set_title(ds+' | same generation seed3');ax.set_ylabel('Real-test RMSE')
        ax.grid(alpha=.2);ax.legend(title='Evaluator seed',fontsize=7)
    fig.suptitle('Actual downstream utility: paired evaluator seeds, original real test sets')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'overall_paired.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    bins=['0-30','31-60','61-90','91-124','125']
    for ds,ax in zip(['FD001','FD002'],axes):
        c=comp[comp.dataset==ds].set_index('bin').reindex(bins)
        ax.plot(range(5),c.control_mean_rmse,marker='o',label='Original generator')
        ax.plot(range(5),c.fix_mean_rmse,marker='o',label='No batch label zeroing')
        ax.set_xticks(range(5),bins);ax.set_title(ds);ax.set_ylabel('Mean evaluator RMSE on real test windows')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Downstream errors by true capped RUL | one fixed generation seed per arm')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'segmented_comparison.{ext}',dpi=180)
    plt.close(fig)
    m['posthoc_audit']='20/20 paired evaluations; raw and all segmented errors reproduced from saved per-engine predictions; original input/source hashes unchanged'
    m['analysis_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    m['checkpoint_sha256']={r.checkpoint:hashlib.sha256((ROOT/r.checkpoint).read_bytes()).hexdigest() for r in raw.itertuples()}
    m['outputs_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    mp.write_text(json.dumps(m,indent=2))
    print(comp.round(4).to_string(index=False))


if __name__=='__main__':main()
