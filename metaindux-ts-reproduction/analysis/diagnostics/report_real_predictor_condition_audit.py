"""Summarize inverse condition checks using saved real-trained predictors."""
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

OUT=ROOT/'results/diagnostics/real_predictor_condition_w48'
RUN=ROOT/'runs/diagnostic-real-predictor-condition-w48-20261023'
BIN_NAMES=['0-30','31-60','61-90','91-124','125']


def main():
    r=pd.read_csv(OUT/'per_model_bins.csv')
    summary=r.groupby(['dataset','reference','source','bin']).agg(
        n_windows=('n_windows','first'), n_models=('mse','size'),
        label_mean=('label_mean','mean'), prediction_mean=('prediction_mean','mean'),
        mse=('mse','mean'), mae=('mae','mean'), bias=('bias','mean'),
        mean_model_rmse=('rmse','mean')).reset_index()
    summary['pooled_rmse']=np.sqrt(summary.mse)
    summary.to_csv(OUT/'summary.csv',index=False)
    # Keep seed-level results: no independence assumptions for overlapping windows.
    r[r.source=='generated'].groupby(['dataset','reference','generation_seed','bin']).agg(
        mse=('mse','mean'),bias=('bias','mean'),mae=('mae','mean')).to_csv(OUT/'generation_seed_summary.csv')
    contrasts=[]
    for (ds,ref,binname),g in summary.groupby(['dataset','reference','bin']):
        a=g[g.source=='generated'].iloc[0]
        b=g[g.source=='real'].iloc[0]
        contrasts.append(dict(dataset=ds,reference=ref,bin=binname,
                              n_windows=int(a.n_windows),condition_mean=float(a.label_mean),
                              generated_prediction_mean=float(a.prediction_mean),real_prediction_mean=float(b.prediction_mean),
                              generated_rmse=float(a.pooled_rmse),real_rmse=float(b.pooled_rmse),
                              generated_bias=float(a.bias),real_bias=float(b.bias),
                              extra_bias=float(a.bias-b.bias),mse_difference=float(a.mse-b.mse)))
    pd.DataFrame(contrasts).to_csv(OUT/'generated_vs_real.csv',index=False)
    low=[]
    for ds in ['FD001','FD002','FD003','FD004']:
        with np.load(OUT/f'{ds}_predictions.npz') as z:
            y=z['real_dev_labels'];g=z['generated_dev_predictions'];rr=z['real_dev_predictions']
            for lo,hi in [(0,0),(1,5),(6,30)]:
                mask=(y>=lo)&(y<=hi)
                if mask.any():
                    for source,p in [('real',rr[:,mask]),('generated',g[:,:,mask])]:
                        err=p-y[mask]
                        low.append(dict(dataset=ds,bin=f'{lo}-{hi}',source=source,n_windows=int(mask.sum()),
                                        pooled_rmse=float(np.sqrt(np.mean(err*err))),mae=float(np.mean(abs(err))),
                                        bias=float(err.mean())))
    pd.DataFrame(low).to_csv(OUT/'low_rul_detail.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ds,ax in zip(['FD001','FD002','FD003','FD004'],axes.flat):
        for source,label in [('real','Real test windows'),('generated','Generated (same exact RUL distribution)')]:
            g=summary[(summary.dataset==ds)&(summary.reference=='test_matched')&(summary.source==source)].set_index('bin')
            ax.plot(range(5),g.reindex(BIN_NAMES).pooled_rmse,marker='o',label=label)
        ax.set_title(ds)
        ax.set_xticks(range(5),BIN_NAMES)
        ax.set_ylabel('RUL prediction RMSE (pooled squared errors)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Real-trained predictors: generated-label consistency vs independent real test errors')
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(OUT/f'inverse_condition_rmse.{suffix}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(12,13))
    for i,ds in enumerate(['FD001','FD002','FD003','FD004']):
        for j,ref in enumerate(['test_matched','dev_matched']):
            ax=axes[i,j]
            for source,label in [('real','Real reference'),('generated','Generated')]:
                g=summary[(summary.dataset==ds)&(summary.reference==ref)&(summary.source==source)].set_index('bin').reindex(BIN_NAMES)
                ax.plot(range(5),g.prediction_mean,marker='o',label=label)
            ax.plot(range(5),g.label_mean,color='gray',linestyle='--',label='Condition label mean')
            ax.set_title(ds+' | '+ref)
            ax.set_xticks(range(5),BIN_NAMES)
            ax.set_ylabel('Mean predicted RUL')
            ax.grid(alpha=.2)
            if i==0:ax.legend(fontsize=8)
    fig.suptitle('Inverse condition check | predictions are not clipped\nDev reference excluded from optimizer but used for checkpoint selection')
    fig.tight_layout(rect=(0,0,1,.95))
    for suffix in ['png','pdf']:fig.savefig(OUT/f'predicted_vs_condition.{suffix}',dpi=170)
    plt.close(fig)
    zero_path=OUT/'all_generated_rul0_rul1.csv'
    if zero_path.exists():
        zero=pd.read_csv(zero_path)
        zsum=zero.groupby(['dataset','rul']).agg(
            n_windows_per_generation=('n_windows','first'),prediction_mean=('prediction_mean','mean'),
            mse=('mse','mean'),bias=('bias','mean')).reset_index()
        zsum['pooled_rmse']=np.sqrt(zsum.mse)
        zsum.to_csv(OUT/'all_rul0_rul1_summary.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        datasets=['FD001','FD002','FD003','FD004']
        for label,ax in zip([0,1],axes):
            z=zsum[zsum.rul==label].set_index('dataset').reindex(datasets)
            ax.bar(datasets,z.prediction_mean,color='#cb703f')
            ax.axhline(label,color='gray',linestyle='--',label=f'Condition RUL={label}')
            ax.set_title(f'All generated RUL={label} windows')
            ax.set_ylabel('Mean prediction of real-trained predictors')
            ax.legend()
        fig.suptitle('Boundary check: same generator and predictor sets; no retraining')
        fig.tight_layout()
        for suffix in ['png','pdf']:fig.savefig(OUT/f'rul0_boundary.{suffix}',dpi=170)
        plt.close(fig)
    path=RUN/'run-manifest.json'
    manifest=json.loads(path.read_text())
    manifest['report_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['output_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
    path.write_text(json.dumps(manifest,indent=2))
    print(summary[summary.bin=='all'].round(3).to_string(index=False))


if __name__=='__main__':
    main()
