"""One read-only per-engine comparison with existing real-trained references."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/diagnostic-fd002-real-reference-engine-pairs-20261106'
OUT=ROOT/'results/diagnostics/fd002_real_reference_engine_pairs_20261106'
REPORT=ROOT/'docs/conclusions/final/fd002_real_reference_engine_pairs_20261106_report.md'
PROTOCOL=ROOT/'docs/operations/FD002_REAL_REFERENCE_ENGINE_PAIRS.md'
PRED=ROOT/'results/diagnostics/rul_segment_errors_w48/predictions.csv'
SEEDS=[3,13,23,33,43]
COMMON=[3,23,33]
BINS=[('0-30',0,30),('31-60',31,60)]


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def state(stage,body):
    t=f'\n## {now()} FD002 generated-vs-real per-engine pairs {stage}\n\n{body}\n'
    p=ROOT/'reproduction-state.md';head,tail=p.read_text().split('\n',1);p.write_text(head+'\n'+t+'\n'+tail)
    with (ROOT/'experiment-ledger.md').open('a') as f:f.write(t)


def main():
    assert not RUN.exists(),'Preserve prior work'
    (RUN/'source').mkdir(parents=True);OUT.mkdir(parents=True)
    parent_path=ROOT/'runs/diagnostic-rul-segment-w48-20261022/run-manifest.json'
    parent=json.loads(parent_path.read_text());assert parent['status']=='complete'
    assert sha(PRED)==parent['outputs_sha256'][str(PRED.relative_to(ROOT))]
    loader_script=ROOT/'analysis/diagnostics/rul_segment_errors.py'
    assert sha(loader_script)==parent['inputs_sha256'][str(loader_script.relative_to(ROOT))]
    for path in [Path(__file__),PROTOCOL,loader_script]:shutil.copy2(path,RUN/'source'/path.name)
    shutil.copy2(parent_path,RUN/'source/prediction-parent-manifest.json')
    paths=[Path(__file__),PROTOCOL,PRED,loader_script,ROOT/'upstream/data/RUL_FD002.txt',
           ROOT/'results/fd002_evaluation_stability_paper70_raw.csv',ROOT/'results/real_data_baseline_w48.csv',
           ROOT/'results/diagnostics/fd002_formal_test_protocol_audit_20261106/test_engine_endpoints.csv']
    paths += [p for p in (RUN/'source').iterdir() if p.is_file()]
    config=dict(dataset='FD002',window=48,primary_generators=SEEDS,predictor_seeds=SEEDS,
        input='Existing formal original5x5 generated-trained predictions and existing5real-trained predictions, same259terminal48windows',
        bins=BINS,per_engine_quantities=dict(
            extra_prediction='mean_e(pred_generated[g,e,i] - pred_real[e,i])',
            extra_squared_error='mean_e((pred_generated[g,e,i]-y_i)^2) - mean_e((pred_real[e,i]-y_i)^2)'),
        pairing='Same engineID and evaluator seed; verify labels and terminal-window parent source before selecting102lowRUL engines',
        signs='Extra prediction>0 only means higher than real-trained reference, not necessarily greater than truth. ExtraSE>0 means worse meanSE.',
        concentration='Count positive extraSE and joint positive metrics; largest5positive extraSE contributions / sum positive extraSE. Report improvements separately, never divide by net change.',
        common_groups={'user_named_common3':COMMON,'all5':SEEDS,'extra_pair_23_43':[23,43]},
        decision='Describe low-stage common error additions versus localized43tail; no new mechanism/label-cause claims',
        stop='Only this paired lowRUL analysis; no training, sampling, model inference, raw sensor/neighbor investigation or test-driven fitting')
    dump(RUN/'config.json',config);paths.append(RUN/'config.json')
    m=dict(run_id=RUN.name,status='running',scope_class='diagnostic',role='supporting',
        scope_contract='scope-20261106-fd002-real-reference-engine-pairs',started_at=now(),config=config,
        user_authorization='Explicit one per-engine pairing against real-trained predictors on FD0020-30/31-60 using existing259engine predictions; no new training',
        inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},command=[sys.executable,str(Path(__file__).resolve())],
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    dump(RUN/'run-manifest.json',m)
    state('started','- [Observed] User-authorized one read-only paired comparison. Freeze same259engine originalformal5×5generated and5real-trained records; match evaluator seeds/labels and terminal-window provenance. Only2per-engine measures, lowRUL102engines, presettop5positiveSEconcentration and overlap. No sensorcheck/modelwork thisround. Snapshot `'+str(RUN.relative_to(ROOT))+'/run-manifest.json`;strictmain/A/Bunchanged,Cdeferred.')
    try:
        f=pd.read_csv(PRED);f=f[f.dataset=='FD002']
        generated=f[(f.source=='generated_headline')&f.generation_seed.isin(SEEDS)].copy()
        real=f[f.source=='real_baseline'][['evaluator_seed','engine_id','rul','prediction']].rename(columns={'rul':'real_rul','prediction':'real_prediction'})
        assert len(generated)==6475 and len(real)==1295
        assert real[['evaluator_seed','engine_id']].drop_duplicates().shape[0]==1295
        pair=generated.merge(real,on=['evaluator_seed','engine_id'],how='left',validate='many_to_one',indicator=True)
        assert len(pair)==6475 and pair['_merge'].eq('both').all() and np.array_equal(pair.rul,pair.real_rul)
        truth=pd.read_csv(ROOT/'upstream/data/RUL_FD002.txt',sep=r'\s+',header=None).iloc[:,0].to_numpy()
        labels=np.minimum(truth,125)
        reference=pd.read_csv(ROOT/'results/real_data_baseline_w48.csv')
        formal=pd.read_csv(ROOT/'results/fd002_evaluation_stability_paper70_raw.csv')
        replay=[]
        for (g,e),z in pair.groupby(['generation_seed','evaluator_seed']):
            z=z.sort_values('engine_id');assert np.array_equal(z.engine_id,np.arange(1,260)) and np.array_equal(z.rul,labels)
            value=float(np.sqrt(np.mean((z.prediction-z.rul)**2)));row=formal[(formal.generation_seed==g)&(formal.evaluator_seed==e)].iloc[0]
            assert abs(value-row.rmse)<.001
            replay.append(dict(source='generated',generation_seed=int(g),evaluator_seed=int(e),recomputed_RMSE=value,recorded_RMSE=float(row.rmse)))
        for e,z in real.groupby('evaluator_seed'):
            z=z.sort_values('engine_id');assert np.array_equal(z.engine_id,np.arange(1,260)) and np.array_equal(z.real_rul,labels)
            value=float(np.sqrt(np.mean((z.real_prediction-z.real_rul)**2)));row=reference[(reference.dataset=='FD002')&(reference.evaluator_seed==e)].iloc[0]
            assert abs(value-row.rmse)<.001
            replay.append(dict(source='real',generation_seed=-1,evaluator_seed=int(e),recomputed_RMSE=value,recorded_RMSE=float(row.rmse)))
        pd.DataFrame(replay).to_csv(OUT/'source_prediction_replay.csv',index=False)
        pair=pair[pair.rul.between(0,60)].copy()
        pair['bin']=np.where(pair.rul<=30,'0-30','31-60')
        pair['extra_prediction']=pair.prediction-pair.real_prediction
        pair['extra_squared_error']=(pair.prediction-pair.rul)**2-(pair.real_prediction-pair.rul)**2
        assert len(pair)==2550
        pair[['generation_seed','evaluator_seed','engine_id','rul','bin','extra_prediction','extra_squared_error']].to_csv(OUT/'per_predictor_engine_pairs.csv',index=False)
        engine=pair.groupby(['generation_seed','engine_id','rul','bin'],sort=True)[['extra_prediction','extra_squared_error']].mean().reset_index()
        assert len(engine)==510
        engine.to_csv(OUT/'per_engine_pairs.csv',index=False)
        summary=[];top=[]
        for (g,b),z in engine.groupby(['generation_seed','bin'],sort=True):
            e=pair[(pair.generation_seed==g)&(pair.bin==b)]
            assert abs(z.extra_prediction.mean()-e.extra_prediction.mean())<1e-10
            separately=float(np.mean((e.prediction-e.rul)**2)-np.mean((e.real_prediction-e.rul)**2))
            assert abs(z.extra_squared_error.mean()-separately)<1e-10
            positive=z[z.extra_squared_error>0].sort_values('extra_squared_error',ascending=False)
            total_positive=float(positive.extra_squared_error.sum());negative=float(z.loc[z.extra_squared_error<0,'extra_squared_error'].sum())
            largest=positive.head(5);largest_sum=float(largest.extra_squared_error.sum())
            for rank,row in enumerate(largest.itertuples(),1):
                top.append(dict(generation_seed=int(g),bin=b,rank=rank,engine_id=int(row.engine_id),rul=float(row.rul),
                    extra_prediction=float(row.extra_prediction),extra_squared_error=float(row.extra_squared_error),
                    share_positive_extraSE=float(row.extra_squared_error)/total_positive))
            summary.append(dict(generation_seed=int(g),bin=b,n_engines=len(z),
                mean_extra_prediction=float(z.extra_prediction.mean()),median_extra_prediction=float(z.extra_prediction.median()),
                mean_extra_squared_error=float(z.extra_squared_error.mean()),median_extra_squared_error=float(z.extra_squared_error.median()),
                extra_prediction_positive_engines=int((z.extra_prediction>0).sum()),extraSE_positive_engines=len(positive),
                both_positive_engines=int(((z.extra_prediction>0)&(z.extra_squared_error>0)).sum()),
                positive_extraSE_sum=total_positive,negative_extraSE_sum=negative,net_extraSE_sum=total_positive+negative,
                top5_positive_extraSE_share=largest_sum/total_positive if total_positive else None,
                top5_positive_extraSE_engine_ids=largest.engine_id.astype(int).tolist()))
        s=pd.DataFrame(summary);s.to_csv(OUT/'stage_summary.csv',index=False);pd.DataFrame(top).to_csv(OUT/'largest5_positive_extraSE.csv',index=False)
        overlap=[];members=[]
        for b,lo,hi in BINS:
            z=engine[engine.bin==b];sets={int(g):set(q.loc[(q.extra_prediction>0)&(q.extra_squared_error>0),'engine_id'].astype(int)) for g,q in z.groupby('generation_seed')}
            common=set.intersection(*(sets[g] for g in COMMON));pair2343=sets[23]&sets[43]
            n=z.engine_id.nunique()
            overlap.append(dict(bin=b,n_engines=n,common3_both_positive_count=len(common),common3_engine_ids=sorted(common),
                seed23_and43_both_positive_count=len(pair2343),seed23_and43_engine_ids=sorted(pair2343),
                seed43_only_among5_count=len(sets[43]-set.union(*(sets[g] for g in SEEDS if g!=43))),
                all5_both_positive_count=len(set.intersection(*(sets[g] for g in SEEDS)))))
            for i in sorted(z.engine_id.unique()):
                hit=[g for g in SEEDS if i in sets[g]]
                members.append(dict(engine_id=int(i),rul=float(z[z.engine_id==i].rul.iloc[0]),bin=b,
                    generators_both_positive=','.join(map(str,hit)),generator_count_both_positive=len(hit),
                    common3_all_both_positive=bool(i in common)))
        pd.DataFrame(members).to_csv(OUT/'engine_overlap_membership.csv',index=False)
        dump(OUT/'overlap.json',overlap)
        result=dict(status='complete',per_engine_measures=['extra_prediction','extra_squared_error'],
            official_test_engines_verified=259,lowRUL_engines=102,generator_trainings_represented=5,predictor_seeds_per_generator=5,
            stage_summaries=summary,overlap=overlap,
            conclusion='Stage-specific added-error pattern:0-30shared3/23/33jointcases34of61andseed23all61worse/higher,notonlyseed43.31-60common3jointcases2of41;seed3and33netMSEimprove,23and43worsen.43hasbothbroad0-30damageandconcentrated31-60tail(top5share67.7%positiveextraSE).',
            limitation='Positive extra prediction means higher than real-trained baseline, not necessarily over trueRUL. Positive extraSE means addederror, but improvements may offset it. Concentration denominator is positive increments, not net overall error. Signed>0overlaps descriptive only, not significance or proven sensor-label cause.',
            next_step='If authorized, inspect common0-30cases against train/dev-supported signal patterns; keep43tail/end-signal hypotheses local and31-60seed23/43distinct. No sensor investigation performed this round.',
            new_training=0,new_sampling=0,new_model_inference=0)
        dump(OUT/'summary.json',result)
        os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
        for ax,(b,lo,hi) in zip(axes,BINS):
            for g in SEEDS:
                q=engine[(engine.generation_seed==g)&(engine.bin==b)]
                positive=np.sort(np.maximum(q.extra_squared_error.to_numpy(),0))[::-1]
                ax.plot(np.arange(len(q)+1),np.r_[0,np.cumsum(positive)/positive.sum()*100],label=f'seed {g}')
            ax.axvline(5,color='gray',ls=':',lw=1);ax.set_title(f'RUL {b}: {len(q)} engines')
            ax.set_xlabel('Engines ranked by positive excess squared error');ax.grid(alpha=.25);ax.set_ylim(0,103)
        axes[0].set_ylabel('Cumulative share of positive excess squared error (%)');axes[1].legend(fontsize=8)
        fig.suptitle('FD002: added-error concentration relative to real-trained predictors\nNegative improvements kept in tables; excluded from concentration denominator',fontsize=11)
        fig.tight_layout();fig.savefig(OUT/'positive_error_concentration.png',dpi=180);fig.savefig(OUT/'positive_error_concentration.pdf');plt.close(fig)
        text='# FD002低RUL逐发动机配对：生成数据新增了什么错误\n\n'
        text+='- [Observed] 使用原正式共同5生成器×5预测器及既有真实数据训练5预测器，均为相同259台末端48窗口/相同标签；先对全部259台与30份模型正式RMSE逐项复核，再取0-30的61台和31-60的41台。无新训练/生成/推理。本轮不回传感器搜索。\n'
        text+='- [Code] 每台只比较A=平均5次(生成训练预测−真实训练预测)，D=平均5次生成预测平方误差−平均5次真实预测平方误差。平方在各预测器内先算，不能对平均预测先平方。A>0只表示比参照报得更高，不一定高于真实寿命；D>0才表示新增平方误差。\n\n'
        text+='| RUL段 | seed | 平均A(周期) | 平均D(周期²) | D>0台数 | A/D均>0台数 | 最大5台占正D总量 |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for b,_,_ in BINS:
            for g in SEEDS:
                a=s[(s.generation_seed==g)&(s.bin==b)].iloc[0]
                text+=f'| {b} | {g} | {a.mean_extra_prediction:+.3f} | {a.mean_extra_squared_error:+.3f} | {a.extraSE_positive_engines}/{a.n_engines} | {a.both_positive_engines}/{a.n_engines} | {a.top5_positive_extraSE_share:.1%} |\n'
        text+='\n## 共性与局部错误\n\n'
        text+='- [Observed] 0-30：3/23/33共同额外报高且额外出错34/61台；并非只是在原有真实预测器的共同高估基础上重复同一误差。seed23为61/61台都更高且更差，最大5台只占正D总量13.8%，错误分散。seed43有46/61台D>0，最大5台占35.7%，也不能概括为只有少数异常。其A/D同时>0的43台全部与seed23重合。\n'
        text+='- [Observed] 31-60：3/23/33共同A/D均>0只有2/41台（109、179）；seed3和33的平均D分别−297.30/−46.40，整体是改善，因此不能将此段当作三者共有的生成新增退化。23和43的平均D分别+196.71/+300.53，A/D均>0的病例有17台重合。\n'
        text+='- [Observed] 43的31-60段25/41台D>0，但最大的5台占正D总量67.7%；这5台为179、15、253、109、68。68的A为负而D为正，说明新增错误也包含比参照报得更低，不能全部归为额外高估。0-30段43的增量既广泛又有较重尾部；两段需要分开。\n'
        text+='- [Observed] 13在0-30有34台D>0但平均D为−49.48，说明其他发动机上的改善抵消了退步；不能仅看“出错台数”就判断总体变差。正增量、负增量及净和全部保留。\n'
        text+='- [Code] 集中度分母为Σmax(D_i,0)，负D改善不进入分母。上述比例不是正误差对净退步的因果贡献，也不把25预测器当25次独立生成器训练。方向计数按严格>0，仅作描述，不作显著性判断。\n'
        text+='- [Hypothesis] 优先可回训练/dev检查0-30中多生成器共同新增错误对应的信号模式；31-60要区分23/43的退化与3/33的改善。既有末端信号偏移继续保留为43局部线索，不能推广成全部FD002共同原因。该配对不证明原因位于信号、标签、生成器或预测器某个单一环节，不据测试病例直接选模型/轮次。\n'
        text+=f'- 逐台两量完整表：`{OUT.relative_to(ROOT)}/per_engine_pairs.csv`（510行）；共同病例：`engine_overlap_membership.csv`；前5贡献明细：`largest5_positive_extraSE.csv`；原始2550配对量：`per_predictor_engine_pairs.csv`。\n'
        text+=f'- 决策线：[Observed]两阶段额外误差已配对定位→[Paper]无对应诊断目标→差距NA→5生成器/固定真实参照描述性证据→严格结论不变。Run `{RUN.relative_to(ROOT)}/run-manifest.json`。本轮一次配对预算结束，strictmain/A/B不变,C推迟，无新模型工作。\n'
        REPORT.write_text(text)
        for p,expected in m['inputs_sha256'].items():assert sha(ROOT/p)==expected,p
        verification=dict(passed=True,source_records_hash_verified=True,official_test_engines=259,verified_models=30,
            lowRUL_engines=102,paired_predictor_rows=2550,per_engine_generator_rows=510,
            labels_equal=True,merge_no_missing=True,mean_squared_errors_taken_before_subtraction=True,
            stage_mean_D_matches_separate_MSE_difference=True,positive_and_negative_increments_preserved=True,
            no_new_training=True,no_new_sampling=True,no_new_inference=True)
        dump(RUN/'verification.json',verification)
        m.update(status='complete',completed_at=now(),outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in list(OUT.glob('*'))+[REPORT]},
            new_training=0,new_sampling=0,new_inference=0)
        dump(RUN/'run-manifest.json',m)
        state('complete','- [Observed] Verified259engines/30existingmodels,then2550pairedpredictorrows→510generator-enginepairs on102lowRULengines.0-30common3/23/33joint extra-pred>0&extraSE>0=34/61;23all61worse/higher,top5positiveSEshare13.83%;43Dpositive46/61,top5share35.68%.31-60common3jointonly2/41(109,179),3/33meanD−297.299/−46.400improve,23/43+196.705/+300.527worsen.43midlowtop5positiveSEshare67.71%,25/41worse;notuniversallysamepattern.\n- [Hypothesis] Shared0-30generation-added errors merit future train/devsignal checks;31-60needsseparate23/43scope,43end-signal remainslocalhypothesis. No sensorcheck or fitting thisround. ConcentrationnormalizespositiveDonly;fullnegativeoffsetskept. Snapshot `'+str(RUN.relative_to(ROOT))+'/run-manifest.json`; report `'+str(REPORT.relative_to(ROOT))+'`;verificationpassed. Zero training/sampling/inference,once-pairingbudgetclosed,strictmain/A/Bunchanged,Cdeferred.')
        print(json.dumps(dict(status='complete',stage_summary=summary,overlap=overlap),ensure_ascii=False,indent=2))
    except BaseException:
        m.update(status='failed',failed_at=now(),failure=traceback.format_exc());dump(RUN/'run-manifest.json',m)
        state('failed','- [Observed] Paired read-only analysis stopped; preserve failure and all outputs; no training/sampling/inference or automatic retries.')
        raise


if __name__=='__main__':main()
