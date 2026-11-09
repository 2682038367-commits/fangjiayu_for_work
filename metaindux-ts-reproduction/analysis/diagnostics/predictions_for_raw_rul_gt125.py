"""Distribution of final test predictions where uncapped ground truth exceeds125."""
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


def main():
    out=ROOT/'results/diagnostics/predictions_raw_rul_gt125_w48'
    out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/diagnostics/rul_segment_errors_w48/predictions.csv'
    r=pd.read_csv(source)
    audit=dict(date='2026-10-23',scope='supporting descriptive audit',no_training=True,
               unit='model × independent test engine, no ensemble averaging',
               model_counts='FD002 headline7 generation×5 evaluators; other datasets5×5; real baseline5',
               inputs_sha256={str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest()})
    allrows,stats=[],[]
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    edges=np.arange(0,161,5)
    for ds,ax in zip(['FD001','FD002','FD003','FD004'],axes.flat):
        truthpath=ROOT/f'upstream/data/RUL_{ds}.txt'
        audit['inputs_sha256'][str(truthpath.relative_to(ROOT))]=hashlib.sha256(truthpath.read_bytes()).hexdigest()
        truth=pd.read_csv(truthpath,sep=r'\s+',header=None)[0]
        mapping=dict(zip(truth.index+1,truth))
        a=r[r.dataset==ds].copy()
        a['raw_true_rul']=a.engine_id.map(mapping)
        a=a[a.raw_true_rul>125]
        assert (a.rul==125).all()
        allrows.append(a)
        for label,group in a.groupby('source'):
            p=group.prediction.to_numpy()
            stats.append(dict(dataset=ds,source=label,n_engines=group.engine_id.nunique(),
                n_model_predictions=len(p),mean=float(p.mean()),median=float(np.median(p)),
                q10=float(np.percentile(p,10)),q25=float(np.percentile(p,25)),
                q75=float(np.percentile(p,75)),q90=float(np.percentile(p,90)),
                minimum=float(p.min()),maximum=float(p.max()),
                percent_below100=float(np.mean(p<100)*100),percent_below125=float(np.mean(p<125)*100),
                rmse_vs_capped125=float(np.sqrt(np.mean((p-125)**2)))))
            counts,_=np.histogram(p,bins=edges)
            ax.stairs(counts/len(p)*100,edges,label='Synthetic-trained' if label=='generated_headline' else 'Real-trained',linewidth=1.7)
        ax.axvline(125,color='gray',linestyle='--',label='Evaluation target =125')
        ax.set_title(f'{ds}: {a.engine_id.nunique()} test engines with raw RUL >125')
        ax.set_ylabel('Predictions per5-cycle bin (%)')
        ax.set_xlabel('Final predicted RUL (not clipped)')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    full=pd.concat(allrows)
    full.to_csv(out/'all_model_predictions.csv',index=False)
    full.groupby(['dataset','source','engine_id']).agg(raw_true_rul=('raw_true_rul','first'),
        mean_prediction=('prediction','mean'),min_prediction=('prediction','min'),
        max_prediction=('prediction','max')).to_csv(out/'per_engine_summary.csv')
    pd.DataFrame(stats).to_csv(out/'distribution_summary.csv',index=False)
    fig.suptitle('Final full-data w48 predictions | select by raw ground truth >125\n'
                 'Repeated models share test engines; histogram is descriptive, not independent replicates')
    fig.tight_layout(rect=(0,0,1,.94))
    for suffix in ['png','pdf']:fig.savefig(out/f'prediction_distribution.{suffix}',dpi=180)
    plt.close(fig)
    audit['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(stats).round(2).to_string(index=False))


if __name__=='__main__':main()
