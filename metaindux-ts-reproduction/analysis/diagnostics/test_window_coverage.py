"""Local support diagnostics using saved full-data w48 samples and predictions."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import sys
import json
import hashlib

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/diagnostics/test_window_coverage_w48'
RUN=ROOT/'runs/diagnostic-test-window-coverage-20261026'
SEEDS=[3,13,23,33,43]
TRI=np.triu_indices(14,1)


def describe(x):
    out={k:[] for k in ['level','dynamics','coupling']}
    t=np.arange(48,dtype=float)-23.5
    for start in range(0,len(x),512):
        a=x[start:start+512].astype(np.float64)
        mean=a.mean(1);center=a-mean[:,None,:]
        std=np.sqrt(np.mean(center**2,axis=1))
        drms=np.sqrt(np.mean(np.diff(a,axis=1)**2,axis=1))
        slope=np.einsum('ntc,t->nc',a,t)/np.sum(t*t)*47
        cov=np.einsum('ntc,ntd->ncd',center,center)
        den=np.sqrt(np.einsum('ntc,ntc->nc',center,center))
        den=den[:,:,None]*den[:,None,:]
        corr=np.divide(cov,den,out=np.zeros_like(cov),where=den>1e-12)
        out['level'].append(np.concatenate([mean,a[:,-1,:]],axis=1))
        out['dynamics'].append(np.concatenate([std,drms,slope],axis=1))
        out['coupling'].append(corr[:,TRI[0],TRI[1]])
    result={k:np.concatenate(v) for k,v in out.items()}
    result['coupling_raw']=result['coupling'].copy()
    return result


def nearest(q,reference,indices,k=5):
    distance=np.sqrt(np.mean((reference[indices]-q)**2,axis=1))
    selected=np.argpartition(distance,min(k,len(distance))-1)[:min(k,len(distance))]
    selected=selected[np.argsort(distance[selected])]
    return float(distance[selected].mean()),float(distance[selected[0]]),indices[selected]


def stage(ds,y):
    return '31-60' if ds=='FD003' else ('125' if y==125 else '91-124')


def main():
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'upstream'))
    from data.CMAPSSDataset import CMAPSSDataset,feature_columns
    now=lambda:datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    manifest=dict(run_id=RUN.name,status='running',scope_class='diagnostic',role='supporting',
                  scope_contract='scope-20261026-test-window-coverage',started_at=now(),no_training=True,
                  command=f'{sys.executable} analysis/diagnostics/test_window_coverage.py',
                  source_snapshot='SOURCE_SNAPSHOT.md',seeds=SEEDS,k=5,rul_caliper=5,
                  features={'level':'14 means +14 terminal levels','dynamics':'14 temporalSD+14 differenceRMS+14 total linear slopes','coupling':'91 within-window demeaned channel correlations'},
                  scale='per-descriptor coordinate standard deviation fitted on ALL real training windows only; coupling_raw additionally uses unscaled correlation coefficients',
                  sensitivity_note='Unscaled coupling added after initial standardized pass showed sensitivity to near-constant correlation coordinates; original pass archived under initial_standardized_pass',
                  inputs_sha256={},caveat='Distances locate mismatches, cannot causally separate missing support from displaced support.')
    def bind(p):manifest['inputs_sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    bind(Path(__file__));bind(ROOT/'upstream/data/CMAPSSDataset.py')
    prediction_path=ROOT/'results/diagnostics/rul_segment_errors_w48/predictions.csv'
    bind(prediction_path);pred=pd.read_csv(prediction_path)
    mp=RUN/'run-manifest.json';mp.write_text(json.dumps(manifest,indent=2))
    rows,cases,residuals=[],[],[]
    cwd=Path.cwd()
    try:
        os.chdir(ROOT/'upstream')
        for ds in ['FD003','FD004']:
            d=CMAPSSDataset(ds,48,[1000]);frame=d.get_train_data()
            real=d.get_feature_slice(frame).numpy();y=d.get_label_slice(frame).numpy().ravel()
            tx,ty=d.get_last_data_slice(d.get_test_data());tx=tx.numpy();ty=ty.numpy().ravel()
            allids=sorted(set(d.get_test_data().id)-{1000});allids=[i for i in allids if i<=d.engine_size]
            mask=(ty>=31)&(ty<=60) if ds=='FD003' else ty>=91
            tx=tx[mask];ty=ty[mask];testids=np.array(allids)[mask]
            trainids=np.concatenate([np.repeat(i,max(len(g)-47,0)) for i,g in frame.groupby('id',sort=True) if i!=1000])
            assert len(trainids)==len(y)
            for kind in ['train','test','RUL']:bind(ROOT/f'upstream/data/{kind}_{ds}.txt')
            rf=describe(real);tf=describe(tx)
            means={k:v.mean(0) for k,v in rf.items()};scales={k:np.maximum(v.std(0),1e-6) for k,v in rf.items()}
            means['coupling_raw']=np.zeros(91);scales['coupling_raw']=np.ones(91)
            rz={k:(v-means[k])/scales[k] for k,v in rf.items()};tz={k:(v-means[k])/scales[k] for k,v in tf.items()}
            pools=[];real_near={};broad_near={}
            for j,(engine,label) in enumerate(zip(testids,ty)):
                st=stage(ds,label)
                pool=np.flatnonzero(y==125) if label==125 else np.flatnonzero((abs(y-label)<=5)&(y>=int(st.split('-')[0]))&(y<=int(st.split('-')[-1])))
                assert len(pool)>=5;pools.append(pool)
                broad=np.flatnonzero(y==125) if label==125 else np.flatnonzero((y>=int(st.split('-')[0]))&(y<=int(st.split('-')[-1])))
                for rep in rf:
                    real_near[j,rep]=nearest(tz[rep][j],rz[rep],pool)
                    broad_near[j,rep]=nearest(tz[rep][j],rz[rep],broad)
                unit=d.get_test_data();unit=unit[unit.id==engine].tail(48)
                # Rounded settings only describe the observed operating profile; not a new normalizer.
                settings=unit[['setting1','setting2','setting3']].copy()
                # These columns are already MinMax-scaled by the author loader.
                signature=settings.round(2).astype(str).agg('/'.join,axis=1).value_counts(normalize=True)
                cases.append(dict(dataset=ds,engine_id=int(engine),rul=float(label),stage=st,
                                  setting_profile_scaled_json=json.dumps(signature.to_dict()),
                                  dominant_scaled_setting=signature.index[0],dominant_fraction=float(signature.iloc[0]),
                                  n_setting_patterns=len(signature),candidate_count=len(pool)))
            for seed in SEEDS:
                path=ROOT/f'outputs/{ds}_w48_seed{seed}.npz';bind(path)
                with np.load(path) as z:
                    generated=z['data'];assert np.array_equal(y,z['label'].ravel()) and generated.shape==real.shape
                gf=describe(generated);gz={k:(v-means[k])/scales[k] for k,v in gf.items()}
                for j,(engine,label) in enumerate(zip(testids,ty)):
                    pp=pred[(pred.dataset==ds)&(pred.engine_id==engine)]
                    synth=pp[(pp.source=='generated_headline')&(pp.generation_seed==seed)]
                    baseline=pp[pp.source=='real_baseline']
                    assert len(synth)==5 and len(baseline)==5
                    smse=float(np.mean(synth.error**2));rmse=float(np.mean(baseline.error**2))
                    st=stage(ds,label)
                    broad=np.flatnonzero(y==125) if label==125 else np.flatnonzero((y>=int(st.split('-')[0]))&(y<=int(st.split('-')[-1])))
                    for rep in rf:
                        gd,gmin,gi=nearest(tz[rep][j],gz[rep],pools[j]);rd,rmin,ri=real_near[j,rep]
                        bg,_,_=nearest(tz[rep][j],gz[rep],broad);br,_,_=broad_near[j,rep]
                        rows.append(dict(dataset=ds,stage=st,engine_id=int(engine),rul=float(label),seed=seed,
                                         representation=rep,n_reference=len(pools[j]),d_real=rd,d_generated=gd,
                                         distance_gap=gd-rd,distance_ratio=gd/max(rd,1e-12),
                                         min_real=rmin,min_generated=gmin,broad_gap=bg-br,
                                         nearest_real_indices=json.dumps(ri.tolist()),nearest_generated_indices=json.dumps(gi.tolist()),
                                         nearest_real_engine=int(trainids[ri[0]]),
                                         synthetic_rmse=smse**.5,real_rmse=rmse**.5,excess_mse=smse-rmse))
                        if rep=='level':
                            for c,sensor in enumerate(feature_columns):
                                residuals.append(dict(dataset=ds,stage=st,engine_id=int(engine),seed=seed,sensor=sensor,
                                    real_residual_sd=float((tf['level'][j,c]-rf['level'][ri,c].mean())/scales['level'][c]),
                                    generated_residual_sd=float((tf['level'][j,c]-gf['level'][gi,c].mean())/scales['level'][c])))
                del generated,gf,gz
                print(ds,'seed',seed,'done',flush=True)
            del real,rf,rz
        distances=pd.DataFrame(rows);distances.to_csv(OUT/'distances_by_engine_seed.csv',index=False)
        pd.DataFrame(cases).to_csv(OUT/'operating_profiles.csv',index=False)
        pd.DataFrame(residuals).to_csv(OUT/'level_residuals.csv',index=False)
        engine=distances.groupby(['dataset','stage','engine_id','rul','representation']).agg(
            d_real=('d_real','first'),d_generated=('d_generated','mean'),distance_gap=('distance_gap','mean'),
            ratio=('distance_ratio','mean'),broad_gap=('broad_gap','mean'),
            synthetic_mse=('synthetic_rmse',lambda v:float(np.mean(v*v))),real_rmse=('real_rmse','first'),
            excess_mse=('excess_mse','mean')).reset_index()
        engine['synthetic_rmse']=np.sqrt(engine.synthetic_mse)
        engine.to_csv(OUT/'engine_summary.csv',index=False)
        corrs=[]
        for (ds,st,rep),g in engine.groupby(['dataset','stage','representation']):
            z=dict(dataset=ds,stage=st,representation=rep,n_engines=len(g),
                   median_d_real=float(g.d_real.median()),median_d_generated=float(g.d_generated.median()),
                   median_gap=float(g.distance_gap.median()),percent_gen_farther=float(np.mean(g.distance_gap>0)*100))
            for xx,yy in [('distance_gap','synthetic_rmse'),('distance_gap','excess_mse'),('d_real','synthetic_rmse'),('d_generated','synthetic_rmse'),('broad_gap','excess_mse')]:
                z[f'rho_{xx}_vs_{yy}']=float(spearmanr(g[xx],g[yy]).statistic)
            corrs.append(z)
        pd.DataFrame(corrs).to_csv(OUT/'stage_correlations.csv',index=False)
        manifest.update(status='complete',n_engine_seed_representation_rows=len(distances),
                        distinct_test_engines=engine[['dataset','engine_id']].drop_duplicates().shape[0],
                        outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()})
    except BaseException as e:
        manifest.update(status='failed',error=repr(e));raise
    finally:
        os.chdir(cwd);manifest['finished_at']=now();mp.write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
