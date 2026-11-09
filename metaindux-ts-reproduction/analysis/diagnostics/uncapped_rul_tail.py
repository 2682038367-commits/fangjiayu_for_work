"""Recover raw window-end RUL before the 125 cap; describe its upper tail."""
from pathlib import Path
import os
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'results/diagnostics/uncapped_rul_tail_w48'
BINS=[(126,150),(151,175),(176,200),(201,250),(251,300),(301,np.inf)]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows,summary,exact=[],[],[]
    audit=dict(date='2026-10-23',scope='supporting descriptive artifact analysis',window=48,
               unit='real training window-end RUL before cap; stride1, engine1000 excluded',
               inputs_sha256={},no_training=True)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
    for ds,ax in zip(['FD001','FD002','FD003','FD004'],axes.flat):
        path=ROOT/f'upstream/data/train_{ds}.txt'
        audit['inputs_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        frame=pd.read_csv(path,sep=r'\s+',header=None)
        labels=[]
        for engine,g in frame.groupby(0,sort=True):
            if engine!=1000 and len(g)>=48:
                labels.extend((g[1].max()-g[1].iloc[47:]).to_numpy())
        y=np.array(labels,dtype=int)
        tail=y[y>125]
        capped=int((y>=125).sum())
        # Verify the reconstructed window count and capped-label count against prior audit.
        prior=pd.read_csv(ROOT/'results/zeroshot/rul_distribution/summary.csv')
        prior=prior[(prior.dataset==ds)&(prior.source=='real_train_windows')].iloc[0]
        assert len(y)==prior['count'] and capped==prior['rul_eq125']
        summary.append(dict(dataset=ds,total_windows=len(y),raw_rul_eq125=int((y==125).sum()),
                            raw_rul_gt125=len(tail),raw_gt125_percent_all_windows=float(np.mean(y>125)*100),
                            raw_tail_min=int(tail.min()),raw_tail_max=int(tail.max()),
                            raw_tail_mean=float(tail.mean()),raw_tail_median=float(np.median(tail)),
                            raw_tail_q25=float(np.percentile(tail,25)),raw_tail_q75=float(np.percentile(tail,75))))
        values=[]
        for lo,hi in BINS:
            count=int(np.sum((tail>=lo)&(tail<=hi)))
            pct=count/len(tail)*100
            label=f'{lo}–{int(hi)}' if np.isfinite(hi) else '>300'
            rows.append(dict(dataset=ds,raw_rul_bin=label,count=count,percent_within_gt125=pct))
            values.append(pct)
        unique,counts=np.unique(tail,return_counts=True)
        for label,count in zip(unique,counts):
            exact.append(dict(dataset=ds,raw_rul=int(label),count=int(count),percent_within_gt125=float(count/len(tail)*100)))
        ax.bar(range(len(BINS)),values,color='#427da5')
        ax.set_xticks(range(len(BINS)),['126–150','151–175','176–200','201–250','251–300','>300'],rotation=25)
        ax.set_title(f'{ds}: {len(tail):,} windows; raw RUL {tail.min()}–{tail.max()}')
        ax.set_ylabel('Percent of windows with raw RUL >125')
        ax.grid(axis='y',alpha=.2)
    for filename,data in [('bins.csv',rows),('summary.csv',summary),('exact_raw_rul_counts.csv',exact)]:
        pd.DataFrame(data).to_csv(OUT/filename,index=False)
    fig.suptitle('What is merged into label 125? Raw RUL >125 in real training windows (w48)')
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(OUT/f'uncapped_rul_distribution.{suffix}',dpi=180)
    plt.close(fig)
    audit['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    audit['outputs_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='audit.json'}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':main()
