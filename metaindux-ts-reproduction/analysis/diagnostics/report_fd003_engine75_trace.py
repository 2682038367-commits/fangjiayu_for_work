"""Postprocess frozen75 trace: reconstruct metrics, fixed example, and figures."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=ROOT/'results/diagnostics/fd003_engine75_trace_20261027'
    mp=ROOT/'runs/diagnostic-fd003-engine75-trace-20261027/run-manifest.json'
    m=json.loads(mp.read_text());assert m['status']=='complete'
    for name,digest in m['inputs_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in m['outputs_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in m['selection_sha256'].items():assert sha(out/name)==digest,name
    r=pd.read_csv(out/'real_query_predictions.csv');g=pd.read_csv(out/'generated_query_predictions.csv')
    selected=pd.read_csv(out/'frozen_generated_selections.csv');q=pd.read_csv(out/'frozen_real_queries.csv')
    with np.load(ROOT/'results/fd003_w48_public_code_split.npz') as z:ti=z['predictive_train'];vi=z['predictive_dev']
    for pool,indices in [('train',ti),('dev',vi)]:assert np.isin(selected[selected.pool==pool].window,indices).all()
    assert np.isin(q[q.role=='real_train_neighbor'].window,ti).all()
    assert np.isin(q[q.role=='real_dev_neighbor'].window,vi).all()
    assert (abs(selected.rul-selected.query_rul)<=5).all()
    for table in [r,g]:assert np.allclose(table.error,table.prediction-table.rul) and np.isfinite(table.prediction).all()
    for _,v in g.groupby(['selection_id','model_family']):assert len(v)==5 and v.evaluator_seed.nunique()==5
    support=pd.read_csv(out/'support_by_query_seed.csv')
    assert (support.n_real_candidates==support.n_generated_candidates).all()
    assert np.allclose(support.ratio,support.d_generated5/support.d_real5)
    # Fixed generationseed3 example (5models in each family), selected by distance.
    chain=[]
    closest=selected[(selected.query_id=='train2_w368')&(selected.generation_seed==3)&(selected.pool=='train')&(selected.selection_rep=='joint')&(selected['rank']==1)]
    assert len(closest)==1
    generated_window=int(closest.window.iloc[0]);expected_selection=int(closest.index[0])
    for sample,label in [('test75','Real test75'),('train2_w368','Nearest realtrain2')]:
        for family in ['real','synthetic']:
            a=r[(r.sample_key==sample)&(r.model_family==family)]
            if family=='synthetic':a=a[a.generation_seed==3]
            assert len(a)==5
            chain.append(dict(sample=label,sample_key=sample,rul=float(a.rul.iloc[0]),model_family=family,
                n_models=5,mean_prediction=float(a.prediction.mean()),rmse=float(np.sqrt(np.mean(a.error**2)))))
    for family in ['real','synthetic']:
        a=g[(g.selection_id==expected_selection)&(g.model_family==family)];assert len(a)==5
        chain.append(dict(sample=f'Nearestgen3_w{generated_window}',sample_key=a.sample_key.iloc[0],rul=float(a.rul.iloc[0]),
            model_family=family,n_models=5,mean_prediction=float(a.prediction.mean()),rmse=float(np.sqrt(np.mean(a.error**2)))))
    pd.DataFrame(chain).to_csv(out/'same_count_example_chain.csv',index=False)
    # Secondary posthoc local-fit comparison: same model, label histogram and split.
    refpath=ROOT/'results/diagnostics/fd003_predictor_fit_20261026/per_model_exact_rul.csv'
    ref=pd.read_csv(refpath);comparisons=[]
    for pool,domain in [('train','fit_train'),('dev','synthetic_dev')]:
        a=g[(g.query_id=='test75')&(g.pool==pool)&(g.selection_rep=='joint')&(g.model_family=='synthetic')]
        expected=[]
        for row in a.itertuples():
            v=ref[(ref.source=='headline')&(ref.generation_seed==row.generation_seed)&(ref.evaluator_seed==row.evaluator_seed)&(ref.domain==domain)&(ref.rul==row.rul)]
            assert len(v)==1;expected.append(float(v.mse.iloc[0]))
        comparisons.append(dict(pool=pool,matched_global_own_rmse=float(np.sqrt(np.mean(expected))),
            local_selected_rmse=float(np.sqrt(np.mean(a.error**2)))))
    pd.DataFrame(comparisons).to_csv(out/'local_vs_global_generated_fit.csv',index=False)
    # Recompute summaries from individual predictions, not plotting values.
    batch=pd.read_csv(out/'real_batch_summary.csv')
    for row in batch.itertuples():
        a=r[(r.batch==row.batch)&(r.model_family==row.model_family)]
        assert len(a)==row.n_predictions and abs(np.sqrt(np.mean(a.error**2))-row.rmse)<1e-9
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    groups=['real_test','real_train_neighbor','real_dev_neighbor'];labels=['Realtest75','5realtrain neighbors','5realdev neighbors']
    x=np.arange(3)
    for family,delta,color,name in [('real',-.17,'#377eb8','Real-trained'),('synthetic',.17,'#e41a1c','Synthetic-trained')]:
        values=[float(batch[(batch.batch==b)&(batch.model_family==family)].rmse.iloc[0]) for b in groups]
        bars=axes[0].bar(x+delta,values,width=.34,color=color,label=name)
        axes[0].bar_label(bars,fmt='%.2f',fontsize=8)
    axes[0].set_xticks(x,labels,rotation=12);axes[0].set_ylabel('RMSE: same windows and labels per group');axes[0].set_ylim(0,52);axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)
    reps=['level','dynamics','joint','coupling_raw','centered_shape']
    v=support[(support.query_id=='test75')&(support.caliper==5)]
    means=[]
    for i,rep in enumerate(reps):
        z=v[v.representation==rep].ratio.to_numpy();means.append(z.mean())
        axes[1].scatter(np.full(len(z),i),z,color='gray',s=23,alpha=.6)
    axes[1].plot(range(5),means,'o-',color='#984ea3',label='Mean across5generationseeds')
    axes[1].axhline(1,color='black',ls='--',lw=1)
    axes[1].set_xticks(range(5),['Level','Dynamics','Joint','Coupling','Centeredshape'],rotation=12)
    axes[1].set_ylabel('Generated5NN distance / real5NN distance');axes[1].legend(fontsize=7);axes[1].grid(alpha=.2)
    fig.suptitle('FD003 test75 neighborhood | predictions worsen across real neighbors; support depends on representation')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'neighborhood_prediction_and_support.{ext}',dpi=180)
    plt.close(fig)
    audit=dict(status='passed',source_input_hashes_verified=True,selections_unchanged=True,
        n_selected_generated_rows=len(selected),n_unique_generated_samples=selected[['generation_seed','window']].drop_duplicates().shape[0],
        n_real_prediction_rows=len(r),n_generated_prediction_rows=len(g),exact_same_labels_in_family_comparisons=True,
        real_generated_candidate_counts_matched=True,train_dev_membership_checked=True,
        per_batch_rmse_independently_recomputed=True,fixed_same_count_chain_generation_seed=3,no_training=True,
        max_real_predictor_replay_difference=m['max_replay_difference'])
    (out/'closeout_audit.json').write_text(json.dumps(audit,indent=2))
    m['postprocess_inputs_sha256']={str(refpath.relative_to(ROOT)):sha(refpath),str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    m['audit']=audit;m['outputs_sha256']={str(p.relative_to(ROOT)):sha(p) for p in out.iterdir() if p.is_file()}
    mp.write_text(json.dumps(m,indent=2))
    print(pd.DataFrame(chain).round(4).to_string(index=False));print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
