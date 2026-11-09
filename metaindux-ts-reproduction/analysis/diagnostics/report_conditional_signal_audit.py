"""Aggregate descriptive signal diagnostics and render reproducible figures."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = ROOT/'results/diagnostics/conditional_signal_w48'
RUN = ROOT/'runs/diagnostic-conditional-signal-w48-20261022'
BIN_NAMES = ['0-30', '31-60', '61-90', '91-124', '125']


def main():
    r = pd.read_csv(OUT/'metrics_by_seed_bin.csv')
    c = pd.read_csv(OUT/'conditional_separation.csv')
    sensors = pd.read_csv(OUT/'sensor_metrics.csv')
    numeric = [x for x in r.columns if x not in ['dataset','seed','bin','comparison']]
    r.groupby(['dataset','bin','comparison'])[numeric].agg(['mean','std']).to_csv(OUT/'summary_by_bin.csv')
    seedmean = r.groupby(['dataset','comparison','seed'])[numeric].mean().reset_index()
    seedmean.to_csv(OUT/'seed_means_equal_bin_weight.csv', index=False)
    summary = []
    for ds in ['FD001','FD002','FD003','FD004']:
        d = {'dataset':ds}
        for comp in ['generated','real_B']:
            rr = seedmean[(seedmean.dataset==ds)&(seedmean.comparison==comp)]
            cc = c[(c.dataset==ds)&(c.comparison==comp)]
            d[comp] = {metric:dict(mean=float(rr[metric].mean()), seed_min=float(rr[metric].min()),
                                 seed_max=float(rr[metric].max())) for metric in numeric}
            d[comp]['conditional_distance_ratio'] = float(cc.other_distance.sum()/cc.real_distance.sum())
            d[comp]['conditional_direction_cosine'] = float(np.average(cc.direction_cosine,weights=cc.real_distance))
        summary.append(d)
    (OUT/'overview.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig, axes = plt.subplots(4,3,figsize=(14,13))
    for i, ds in enumerate(['FD001','FD002','FD003','FD004']):
        for j,(metric,title) in enumerate([('mean_wasserstein','Window-mean distribution distance'),
                                          ('diff_rms_ratio','First-difference RMS ratio (other / real)'),
                                          ('diff_corr_mae','Change correlation error (91 channel pairs)')]):
            ax = axes[i,j]
            for comp,label in [('generated','Generated vs real A'),('real_B','Real B vs real A (different engines)')]:
                vals=r[(r.dataset==ds)&(r.comparison==comp)].groupby('bin')[metric]
                mean=vals.mean().reindex(BIN_NAMES).to_numpy()
                low=vals.min().reindex(BIN_NAMES).to_numpy()
                high=vals.max().reindex(BIN_NAMES).to_numpy()
                ax.plot(range(5),mean,marker='o',label=label)
                ax.fill_between(range(5),low,high,alpha=.12)
            if metric.endswith('ratio'):
                ax.axhline(1,color='gray',linestyle='--',linewidth=.8)
            ax.set_title(ds+' | '+title,fontsize=9)
            ax.set_xticks(range(5),BIN_NAMES,rotation=20)
            ax.grid(alpha=.2)
            if i==0 and j==0:
                ax.legend(fontsize=7)
    fig.suptitle('Full-data w48 | exact-RUL matched samples | 5 seeds\n'
                 'Lines: mean; bands: min-max across seed/split pairs (not confidence intervals)',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95))
    for suffix in ['png','pdf']:
        fig.savefig(OUT/f'signal_diagnostics.{suffix}',dpi=170)
    plt.close(fig)
    fig, axes=plt.subplots(4,3,figsize=(13,13))
    selected={}
    for i,ds in enumerate(['FD001','FD002','FD003','FD004']):
        subset=sensors[(sensors.dataset==ds)&(sensors.comparison=='generated')]
        real=subset.groupby(['sensor','bin']).centroid_real.mean().unstack().reindex(columns=BIN_NAMES)
        top=real.std(axis=1).nlargest(3).index.tolist()
        selected[ds]=top
        for j,sensor in enumerate(top):
            ax=axes[i,j]
            anchor=real.loc[sensor,'125']
            for comp,column,label in [('generated','centroid_real','Real A'),('real_B','centroid_other','Real B'),('generated','centroid_other','Generated')]:
                group=sensors[(sensors.dataset==ds)&(sensors.sensor==sensor)&(sensors.comparison==comp)].groupby('bin')[column]
                vals=group.mean().reindex(BIN_NAMES).to_numpy()-anchor
                ax.plot(range(5),vals,marker='o',label=label)
            ax.set_title(f'{ds} | {sensor}')
            ax.set_ylabel('Mean level / training sensor SD\n(offset by real RUL=125 mean)',fontsize=8)
            ax.set_xticks(range(5),BIN_NAMES,rotation=20)
            ax.grid(alpha=.2)
            if i==0 and j==0:ax.legend(fontsize=8)
    fig.suptitle('Conditional sensor means | Top 3 sensors by REAL between-bin variability\n'
                 'Selection uses real data only; all 14 sensors remain in the CSV metrics',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95))
    for suffix in ['png','pdf']:
        fig.savefig(OUT/f'conditional_sensor_means.{suffix}',dpi=170)
    plt.close(fig)
    (OUT/'plotted_sensors.json').write_text(json.dumps(selected,indent=2))
    path=RUN/'run-manifest.json'
    manifest=json.loads(path.read_text())
    manifest['report_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['output_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
    path.write_text(json.dumps(manifest,indent=2))
    print('Summaries and figures written.')


if __name__=='__main__':
    main()
