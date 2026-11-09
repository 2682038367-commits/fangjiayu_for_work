"""Enrich coverage cases with raw operating settings and render descriptive plots."""
from pathlib import Path
import os
import json
import hashlib

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'results/diagnostics/test_window_coverage_w48'
RUN=ROOT/'runs/diagnostic-test-window-coverage-20261026'


def profile(g):
    s=g.iloc[-48:,2:5].copy()
    if len(s)<48:s=pd.concat([s.iloc[[0]]]*(48-len(s))+[s])
    s.columns=['setting1','setting2','setting3']
    s=s.round({'setting1':0,'setting2':2,'setting3':0})+0.0
    signatures=s.apply(lambda r:f'{r.setting1:.0f}/{r.setting2:.2f}/{r.setting3:.0f}',axis=1)
    counts=signatures.value_counts(normalize=True)
    return dict(dominant_setting=counts.index[0],dominant_fraction=float(counts.iloc[0]),
                n_regimes=len(counts),setting_profile=json.dumps(counts.to_dict()))


def main():
    e=pd.read_csv(OUT/'engine_summary.csv');d=pd.read_csv(OUT/'distances_by_engine_seed.csv')
    profiles=[]
    for ds in ['FD003','FD004']:
        test=pd.read_csv(ROOT/f'upstream/data/test_{ds}.txt',sep=r'\s+',header=None)
        for engine in e[e.dataset==ds].engine_id.unique():
            profiles.append(dict(dataset=ds,engine_id=int(engine),**profile(test[test[0]==engine])))
    profiles=pd.DataFrame(profiles);profiles.to_csv(OUT/'raw_operating_profiles.csv',index=False)
    enriched=e.merge(profiles,on=['dataset','engine_id'],validate='many_to_one')
    enriched.to_csv(OUT/'engine_summary_with_settings.csv',index=False)
    top=enriched[enriched.representation=='level'].sort_values(['dataset','stage','synthetic_rmse'],ascending=[True,True,False]).groupby(['dataset','stage']).head(8)
    top.to_csv(OUT/'highest_error_cases.csv',index=False)
    group=enriched[enriched.representation=='level'].groupby(['dataset','stage','dominant_setting']).agg(
        n_engines=('engine_id','size'),mean_synthetic_rmse=('synthetic_rmse','mean'),mean_real_rmse=('real_rmse','mean'),
        mean_gap=('distance_gap','mean'),median_gap=('distance_gap','median'))
    group.to_csv(OUT/'operating_group_summary.csv')
    associations=[]
    for (ds,st,rep,seed),g in d.groupby(['dataset','stage','representation','seed']):
        associations.append(dict(dataset=ds,stage=st,representation=rep,seed=seed,
            rho_gap_error=float(spearmanr(g.distance_gap,g.synthetic_rmse).statistic),
            rho_gap_excess_mse=float(spearmanr(g.distance_gap,g.excess_mse).statistic)))
    pd.DataFrame(associations).to_csv(OUT/'seed_correlations.csv',index=False)
    d['min_gap']=d.min_generated-d.min_real
    sensitivity=[]
    for (ds,st,rep),g in d.groupby(['dataset','stage','representation']):
        z=g.groupby('engine_id')[['min_gap','excess_mse']].mean()
        sensitivity.append(dict(dataset=ds,stage=st,representation=rep,
            rho_min_gap_excess_mse=float(spearmanr(z.min_gap,z.excess_mse).statistic)))
    pd.DataFrame(sensitivity).to_csv(OUT/'one_neighbor_sensitivity.csv',index=False)
    residual=pd.read_csv(OUT/'level_residuals.csv')
    residual.groupby(['dataset','stage','sensor']).agg(
        real_mean_residual=('real_residual_sd','mean'),gen_mean_residual=('generated_residual_sd','mean'),
        gen_abs_residual=('generated_residual_sd',lambda x:float(np.mean(abs(x)))),
        real_abs_residual=('real_residual_sd',lambda x:float(np.mean(abs(x))))).to_csv(OUT/'sensor_residual_summary.csv')
    targets=[('FD003','31-60'),('FD004','91-124'),('FD004','125')]
    fig,axes=plt.subplots(3,3,figsize=(13,11))
    for i,(ds,st) in enumerate(targets):
        for j,rep in enumerate(['level','dynamics','coupling_raw']):
            g=e[(e.dataset==ds)&(e.stage==st)&(e.representation==rep)]
            ax=axes[i,j]
            ax.scatter(g.distance_gap,g.synthetic_rmse,alpha=.75,s=24)
            rho=spearmanr(g.distance_gap,g.synthetic_rmse).statistic
            ax.axvline(0,color='gray',linestyle='--',linewidth=.8)
            ax.set_title(f'{ds}, RUL {st} | {rep} | rho={rho:.2f}',fontsize=10)
            ax.set_xlabel('5-neighbor distance: generated minus real')
            ax.set_ylabel('Synthetic-trained test RMSE per engine')
            ax.grid(alpha=.2)
    fig.suptitle('Coverage vs prediction error | each dot is one independent test engine\n'
                 'Distances and errors averaged over 5 generation seeds; exploratory associations, no causal claim')
    fig.tight_layout(rect=(0,0,1,.94))
    for suffix in ['png','pdf']:fig.savefig(OUT/f'distance_vs_error.{suffix}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    sr=pd.read_csv(OUT/'sensor_residual_summary.csv')
    for ax,(ds,st) in zip(axes,targets):
        g=sr[(sr.dataset==ds)&(sr.stage==st)]
        ax.bar(np.arange(len(g))-.2,g.real_mean_residual,width=.4,label='Real-neighbor residual')
        ax.bar(np.arange(len(g))+.2,g.gen_mean_residual,width=.4,label='Generated-neighbor residual')
        ax.axhline(0,color='gray',linewidth=.8)
        ax.set_xticks(range(len(g)),g.sensor,rotation=70)
        ax.set_title(f'{ds}, RUL {st}')
        ax.set_ylabel('Test minus nearest-neighbor mean (real SD units)')
        ax.legend(fontsize=7)
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(OUT/f'neighbor_level_residuals.{suffix}',dpi=170)
    plt.close(fig)
    # Visualization only: PCA fitted on real stage references, never on error labels.
    import sys
    sys.path.insert(0,str(ROOT/'upstream'))
    from data.CMAPSSDataset import CMAPSSDataset
    cwd=Path.cwd();os.chdir(ROOT/'upstream')
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    try:
        for ds in ['FD003','FD004']:
            data=CMAPSSDataset(ds,48,[1000])
            x=data.get_feature_slice(data.get_train_data()).numpy()
            y=data.get_label_slice(data.get_train_data()).numpy().ravel()
            tx,ty=data.get_last_data_slice(data.get_test_data());tx=tx.numpy();ty=ty.numpy().ravel()
            ids=np.array(sorted(set(data.get_test_data().id)-{1000}));ids=ids[ids<=data.engine_size]
            f=np.concatenate([x.mean(1),x[:,-1,:]],axis=1).astype(float)
            tf=np.concatenate([tx.mean(1),tx[:,-1,:]],axis=1).astype(float)
            scale=np.maximum(f.std(0),1e-6)
            generated=[]
            for seed in [3,13,23,33,43]:
                with np.load(ROOT/f'outputs/{ds}_w48_seed{seed}.npz') as z:gx=z['data']
                generated.append(np.concatenate([gx.mean(1),gx[:,-1,:]],axis=1).astype(float))
            for panel,(target_ds,st) in enumerate(targets):
                if target_ds!=ds:continue
                lo,hi=(125,125) if st=='125' else tuple(map(int,st.split('-')))
                mask=(y>=lo)&(y<=hi);tm=(ty>=lo)&(ty<=hi)
                ref=f[mask]/scale;center=ref.mean(0)
                val,vec=np.linalg.eigh(np.cov(ref,rowvar=False));basis=vec[:,-2:][:,::-1]
                rng=np.random.RandomState(20261026)
                rr=(ref-center)@basis
                gg=np.concatenate([((g[mask]/scale-center)@basis)[rng.choice(mask.sum(),min(1200,mask.sum()),replace=False)] for g in generated])
                ax=axes[panel]
                ri=rng.choice(len(rr),min(6000,len(rr)),replace=False)
                ax.scatter(rr[ri,0],rr[ri,1],s=3,alpha=.12,label='Real training')
                ax.scatter(gg[:,0],gg[:,1],s=3,alpha=.1,label='Generated (5 seeds)')
                point=(tf[tm]/scale-center)@basis
                errors=e[(e.dataset==ds)&(e.stage==st)&(e.representation=='level')].set_index('engine_id').reindex(ids[tm])
                artist=ax.scatter(point[:,0],point[:,1],c=errors.synthetic_rmse,cmap='viridis',s=35,edgecolors='black',linewidths=.4)
                fig.colorbar(artist,ax=ax,label='Test-engine prediction RMSE')
                ax.set_title(f'{ds}, RUL {st}\nReal PCA top2: {val[-2:].sum()/val.sum()*100:.1f}% variance')
                ax.set_xlabel('PC1');ax.set_ylabel('PC2');ax.legend(fontsize=7)
        fig.suptitle('Sensor-level support projection | PCA from real training descriptors only\nProjection illustrates overlap; nearest-neighbor analysis uses all 28 dimensions')
        fig.tight_layout(rect=(0,0,1,.9))
        for suffix in ['png','pdf']:fig.savefig(OUT/f'level_support_pca.{suffix}',dpi=170)
        plt.close(fig)
    finally:os.chdir(cwd)
    mp=RUN/'run-manifest.json';m=json.loads(mp.read_text())
    m['report_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    m['outputs_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
    mp.write_text(json.dumps(m,indent=2))
    print(group.round(3).to_string())


if __name__=='__main__':main()
