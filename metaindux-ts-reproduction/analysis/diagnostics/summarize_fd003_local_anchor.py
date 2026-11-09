"""Audit and visualize the fixed-budget FD003 local anchor pilot."""
from pathlib import Path
import os
import json
import hashlib
import re

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=ROOT/'results/diagnostics/fd003_local_anchor_20261026'
    run=ROOT/'runs/ext-local-anchor-fd003-seed3-20261026'
    mp=run/'run-manifest.json';m=json.loads(mp.read_text())
    assert m['status']=='complete'
    raw=pd.read_csv(out/'raw.csv');seg=pd.read_csv(out/'per_model_segments.csv')
    pred=pd.read_csv(out/'predictions.csv');comp=pd.read_csv(out/'comparison.csv')
    assert len(raw)==10 and len(seg)==60 and len(pred)==1000
    assert not raw.duplicated(['arm','evaluator_seed']).any()
    for name,h in m['inputs_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    with np.load(out/'replacement_indices.npz') as z:
        assert len(z['target'])==209 and len(set(z['donor']))==209
        assert np.array_equal(z['target_rul'],z['donor_rul'])
        assert np.isin(z['donor'],z['train_indices']).all()
        assert np.isin(z['target'],z['train_indices']).all()
        assert not np.intersect1d(z['donor'],z['dev_indices']).size
    console=(ROOT/'logs/fd003_local_anchor_20261026_console.log').read_text(errors='replace')
    for r in raw.itertuples():
        name=f'FD003_{r.arm}_eval{r.evaluator_seed}'
        match=re.search(r'START '+name+r'\s.*?DONE '+name+r'\s',console,re.S)
        assert match and len(re.findall(r'Epoch \[\d+/35\]',match.group()))==35,name
        p=pred[(pred.arm==r.arm)&(pred.evaluator_seed==r.evaluator_seed)]
        assert len(p)==100 and p.engine_id.nunique()==100
        assert abs(np.sqrt(np.mean(p.error**2))-r.rmse)<.001
        for s in seg[(seg.arm==r.arm)&(seg.evaluator_seed==r.evaluator_seed)].itertuples():
            if s.bin=='all':ss=p
            elif s.bin=='125':ss=p[p.rul==125]
            else:
                lo,hi=map(int,s.bin.split('-'));ss=p[(p.rul>=lo)&(p.rul<=hi)]
            assert len(ss)==s.n_engines and abs(np.sqrt(np.mean(ss.error**2))-s.rmse)<1e-8
    raw.groupby('arm').agg(mean_rmse=('rmse','mean'),evaluator_sd=('rmse','std')).to_csv(out/'summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for b,ax in zip(['31-60','all'],axes):
        g=seg[seg.bin==b].pivot(index='evaluator_seed',columns='arm',values='rmse')
        for seed,row in g.iterrows():ax.plot([0,1],[row.control,row.anchor10],marker='o',label=str(seed))
        ax.set_xticks([0,1],['Synthetic only','10% local real anchors'])
        ax.set_title('FD003 | RUL '+b);ax.set_ylabel('Real-test RMSE');ax.grid(alpha=.2)
        ax.legend(title='Evaluator seed',fontsize=7)
    fig.suptitle('Fixed35epochs, same sample count/RUL distribution/dev | generation seed3')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'paired_rmse.{ext}',dpi=180)
    plt.close(fig)
    bins=['0-30','31-60','61-90','91-124','125'];g=comp.set_index('bin').reindex(bins)
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    ax.plot(range(5),g.control_mean_rmse,marker='o',label='Synthetic only')
    ax.plot(range(5),g.anchor_mean_rmse,marker='o',label='10% local real anchors')
    ax.set_xticks(range(5),bins);ax.set_ylabel('Mean evaluator RMSE on real test windows')
    ax.set_title('FD003 | local31–60 replacement, unchanged other training samples')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'segmented_rmse.{ext}',dpi=180)
    plt.close(fig)
    m['audit']='10/10 models trained35epochs(1925 updates);209 exact-RUL replacements from fixedrealtrain;dev excluded;all per-engine/segment metrics reproduced;input/sourcehashes match'
    m['checkpoint_sha256']={r.checkpoint:hashlib.sha256((ROOT/r.checkpoint).read_bytes()).hexdigest() for r in raw.itertuples()}
    m['analysis_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    m['outputs_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    mp.write_text(json.dumps(m,indent=2))
    print(comp.round(4).to_string(index=False))


if __name__=='__main__':main()
