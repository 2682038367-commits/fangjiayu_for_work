"""Check whether the actual209 anchors improved local support, without training."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import sys
import json
import hashlib

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'results/diagnostics/fd003_anchor_support_20261026'
RUN=ROOT/'runs/diagnostic-fd003-anchor-support-20261026'


def main():
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'upstream'));sys.path.insert(0,str(ROOT/'analysis/diagnostics'))
    from data.CMAPSSDataset import CMAPSSDataset
    from test_window_coverage import describe,nearest
    source=ROOT/'results/diagnostics/fd003_local_anchor_20261026'
    now=lambda:datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    m=dict(run_id=RUN.name,status='running',scope_class='diagnostic',role='supporting followup ofEXT-001',
           started_at=now(),no_training=True,inputs_sha256={},k=5,rul_caliper=5,
           primary='actual predictor training pool only; before/after209 replacements; same real-fitted descriptor scales',
           caveat='Descriptive test-engine diagnosis, not an additional intervention or independent causal test.')
    def bind(p):m['inputs_sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [Path(__file__),ROOT/'analysis/diagnostics/test_window_coverage.py',ROOT/'upstream/data/CMAPSSDataset.py',
              source/'replacement_indices.npz',source/'predictions.csv',ROOT/'outputs/FD003_w48_seed3.npz',
              ROOT/'results/diagnostics/test_window_coverage_w48/engine_summary.csv']+[ROOT/f'upstream/data/{kind}_FD003.txt' for kind in ['train','test','RUL']]:bind(p)
    mp=RUN/'run-manifest.json';mp.write_text(json.dumps(m,indent=2))
    cwd=Path.cwd()
    try:
        os.chdir(ROOT/'upstream')
        data=CMAPSSDataset('FD003',48,[1000]);real=data.get_feature_slice(data.get_train_data()).numpy()
        y=data.get_label_slice(data.get_train_data()).numpy().ravel()
        tx,ty=data.get_last_data_slice(data.get_test_data());tx=tx.numpy();ty=ty.numpy().ravel()
        testids=np.array(sorted(set(data.get_test_data().id)-{1000}));testids=testids[testids<=data.engine_size]
        mask=(ty>=31)&(ty<=60);tx=tx[mask];ty=ty[mask];testids=testids[mask]
        with np.load(ROOT/'outputs/FD003_w48_seed3.npz') as z:
            generated=z['data'];assert np.array_equal(z['label'].ravel(),y)
        with np.load(source/'replacement_indices.npz') as z:
            target=z['target'];donor=z['donor'];ti=z['train_indices'];vi=z['dev_indices']
        assert len(target)==209 and np.array_equal(y[target],y[donor])
        assert np.isin(target,ti).all() and np.isin(donor,ti).all()
        rf=describe(real);gf=describe(generated);tf=describe(tx)
        pred=pd.read_csv(source/'predictions.csv')
        old=pd.read_csv(ROOT/'results/diagnostics/test_window_coverage_w48/engine_summary.csv')
        old=old[(old.dataset=='FD003')&(old.representation=='level')]
        high_ids=set(old.nlargest(5,'distance_gap').engine_id)
        rows=[]
        for rep in ['level','dynamics','coupling_raw']:
            mu=rf[rep].mean(0);scale=np.maximum(rf[rep].std(0),1e-6)
            if rep=='coupling_raw':mu=np.zeros(91);scale=np.ones(91)
            r=(rf[rep]-mu)/scale;before=(gf[rep]-mu)/scale;t=(tf[rep]-mu)/scale
            after=before.copy();after[target]=r[donor]
            assert np.array_equal(before[vi],after[vi])
            for j,(engine,label) in enumerate(zip(testids,ty)):
                pool=ti[(y[ti]>=31)&(y[ti]<=60)&(abs(y[ti]-label)<=5)]
                dr,rmin,ri=nearest(t[j],r,pool)
                db,bmin,bi=nearest(t[j],before,pool)
                da,amin,ai=nearest(t[j],after,pool)
                eligible=np.flatnonzero((y[target]>=31)&(y[target]<=60)&(abs(y[target]-label)<=5))
                distance_to_donors=np.sqrt(np.mean((r[donor[eligible]]-t[j])**2,axis=1))
                pa=pred[(pred.engine_id==engine)&(pred.arm=='anchor10')];pc=pred[(pred.engine_id==engine)&(pred.arm=='control')]
                assert len(pa)==len(pc)==5
                cmse=float(np.mean(pc.error**2));amse=float(np.mean(pa.error**2))
                rows.append(dict(engine_id=int(engine),rul=float(label),representation=rep,
                    previously_high_gap=int(engine in high_ids),n_candidates=len(pool),n_anchor_candidates=len(eligible),
                    d_real_train=dr,d_before=db,d_after=da,distance_improvement=db-da,
                    percent_distance_reduction=(db-da)/db*100,min_before=bmin,min_after=amin,
                    closest_anchor_distance=float(distance_to_donors.min()),
                    anchors_in_top5_after=int(np.isin(ai,target).sum()),
                    nearest_before=json.dumps(bi.tolist()),nearest_after=json.dumps(ai.tolist()),
                    control_rmse=cmse**.5,anchor_rmse=amse**.5,
                    rmse_improvement=cmse**.5-amse**.5,mse_improvement=cmse-amse))
        table=pd.DataFrame(rows);table.to_csv(OUT/'per_engine_support.csv',index=False)
        summaries=[]
        for rep,g in table.groupby('representation'):
            summaries.append(dict(representation=rep,n_engines=len(g),n_distance_improved=int((g.distance_improvement>1e-10).sum()),
                n_distance_worsened=int((g.distance_improvement< -1e-10).sum()),
                n_anchors_in_top5=int((g.anchors_in_top5_after>0).sum()),
                median_percent_distance_reduction=float(g.percent_distance_reduction.median()),
                mean_anchors_in_top5=float(g.anchors_in_top5_after.mean()),
                rho_distance_reduction_vs_mse_reduction=float(spearmanr(g.distance_improvement,g.mse_improvement).statistic),
                rho_distance_reduction_vs_rmse_reduction=float(spearmanr(g.distance_improvement,g.rmse_improvement).statistic)))
        pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
        table[(table.representation=='level')&table.previously_high_gap.astype(bool)].to_csv(OUT/'previous_high_gap_cases.csv',index=False)
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for rep,ax in zip(['level','dynamics','coupling_raw'],axes):
            g=table[table.representation==rep]
            ax.scatter(g.distance_improvement,g.mse_improvement,c=g.previously_high_gap,cmap='coolwarm',s=38)
            for row in g[g.previously_high_gap==1].itertuples():ax.annotate(str(row.engine_id),(row.distance_improvement,row.mse_improvement),fontsize=8)
            ax.axvline(0,color='gray',linestyle='--');ax.axhline(0,color='gray',linestyle='--')
            ax.set_title(rep);ax.set_xlabel('Local5-NN distance reduction');ax.set_ylabel('TestMSE reduction (control minus anchor)')
            ax.grid(alpha=.2)
        fig.suptitle('Did the actual209 anchors reach previous gaps? | red: previous top5 level gaps\nOne test engine per dot; positive x/y means improvement, no causal association claim')
        fig.tight_layout(rect=(0,0,1,.88))
        for ext in ['png','pdf']:fig.savefig(OUT/f'distance_change_vs_error_change.{ext}',dpi=180)
        plt.close(fig)
        m.update(status='complete',summary=summaries,previous_high_gap_engines=sorted(map(int,high_ids)),
                 outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()})
    except BaseException as exc:m.update(status='failed',error=repr(exc));raise
    finally:
        os.chdir(cwd);m['finished_at']=now();mp.write_text(json.dumps(m,indent=2))
    print(pd.DataFrame(summaries).round(3).to_string(index=False))
    print(table[(table.representation=='level')&table.previously_high_gap.astype(bool)].round(3).to_string(index=False))


if __name__=='__main__':main()
