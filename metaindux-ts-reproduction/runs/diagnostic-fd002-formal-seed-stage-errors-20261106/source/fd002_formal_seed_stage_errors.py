"""Reuse saved FD002 formal-test predictions for stage/bias and seed43 localization."""
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
RUN=ROOT/'runs/diagnostic-fd002-formal-seed-stage-errors-20261106'
OUT=ROOT/'results/diagnostics/fd002_formal_seed_stage_errors_20261106'
REPORT=ROOT/'docs/conclusions/final/fd002_formal_seed_stage_errors_20261106_report.md'
PROTOCOL=ROOT/'docs/operations/FD002_FORMAL_SEED_STAGE_ERRORS.md'
PRED=ROOT/'results/diagnostics/rul_segment_errors_w48/predictions.csv'
RAW=ROOT/'results/fd002_evaluation_stability_paper70_raw.csv'
SIGNAL=ROOT/'results/diagnostics/conditional_signal_w48/metrics_by_seed_bin.csv'
READOUT=ROOT/'results/diagnostics/real_predictor_condition_w48/generation_seed_summary.csv'
SEEDS=[3,13,23,33,43]
COMMON=[3,23,33]
BINS=[('0-30',0,30),('31-60',31,60),('61-90',61,90),('91-124',91,124),('125',125,125)]


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def state(stage,body):
    t=f'\n## {now()} FD002 formal per-seed RUL-stage errors {stage}\n\n{body}\n'
    p=ROOT/'reproduction-state.md';head,tail=p.read_text().split('\n',1);p.write_text(head+'\n'+t+'\n'+tail)
    with (ROOT/'experiment-ledger.md').open('a') as f:f.write(t)


def model_metrics(f,g,e,source):
    rows=[]
    for name,lo,hi in [('all',0,125),*BINS]:
        z=f[f.rul.between(lo,hi)];err=(z.prediction-z.rul).to_numpy()
        rows.append(dict(source=source,generation_seed=g,evaluator_seed=e,bin=name,n_engines=len(z),
            mse=float(np.mean(err**2)),rmse=float(np.sqrt(np.mean(err**2))),
            signed_error=float(np.mean(err)),mae=float(np.mean(abs(err)))))
    return rows


def main():
    assert not RUN.exists(),'No overwrite'
    (RUN/'source').mkdir(parents=True);OUT.mkdir(parents=True)
    prior=ROOT/'runs/diagnostic-rul-segment-w48-20261022/run-manifest.json'
    parent=json.loads(prior.read_text());assert parent['status']=='complete'
    assert sha(PRED)==parent['outputs_sha256'][str(PRED.relative_to(ROOT))]
    for p in [Path(__file__),PROTOCOL]:shutil.copy2(p,RUN/'source'/p.name)
    shutil.copy2(prior,RUN/'source/prediction-parent-manifest.json')
    paths=[Path(__file__),PROTOCOL,PRED,RAW,SIGNAL,READOUT,ROOT/'upstream/data/RUL_FD002.txt',
        ROOT/'docs/conclusions/final/conditional_signal_diagnostic_w48.md',
        ROOT/'docs/conclusions/final/real_predictor_condition_audit_w48.md',
        ROOT/'results/diagnostics/fd002_formal_generator_seeds_20261106/summary.json']
    paths += [p for p in (RUN/'source').iterdir() if p.is_file()]
    config=dict(dataset='FD002',window=48,generation_seeds=SEEDS,evaluator_seeds=SEEDS,bins=BINS,
        test_engines=259,test_sample='Same one terminal window per official test engine; capped125 labels, unclipped output',
        RMSE='For each predictor sqrt(mean squared error in bin), then arithmetic mean5predictor RMSEs; same hierarchy as26.7918',
        signed_error='Mean(prediction minus capped trueRUL), averaged over5predictors and same bin engines',
        auxiliary_MSE='Mean per-predictor MSE, used for additive contributions; pooledRMSE kept separate from main meanRMSE',
        common_seed_group=COMMON,seed43_reference='Mean per-engine/per-bin MSE of user-named3/23/33, not best-seed selection',
        seed43_contribution='n_bin/259*(MSE43_bin - mean(MSE3_bin,MSE23_bin,MSE33_bin)); sums to overall excessMSE, not additive RMSE difference',
        controls='Existing5real-trained predictor records auxiliary; existing RUL-matched training/dev signal/readout artifacts only, no new inference',
        stop='Descriptive tables, engine-direction overlaps, MSE arithmetic, prior signal evidence cross-reference; no training/sampling/inference, no sensor/neighbor search or test-driven fitting')
    dump(RUN/'config.json',config);paths.append(RUN/'config.json')
    manifest=dict(run_id=RUN.name,status='running',scope_class='diagnostic',role='supporting',
        scope_contract='scope-20261106-fd002-formal-seed-stage-errors',started_at=now(),config=config,
        user_authorization='Compare existing5x5per-engine formal-test errors by trueRUL, common3/23/33directions and seed43stage localization; reuse prior train/dev checks where shared pattern present',
        inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        command=[sys.executable,str(Path(__file__).resolve())],
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    dump(RUN/'run-manifest.json',manifest)
    state('started','- [Observed] Freeze original6475common5generator×5predictor×259engine records. PresetRUL0-30/31-60/61-90/91-124/125separate;formalmean-per-predictorRMSE, signederror and matched-engine seed43excessMSE vs3/23/33. Existingrealbaseline/train-devdiagnosticrecords auxiliary only; no new fits/inference or test-driven model selection. Snapshot `'+str(RUN.relative_to(ROOT))+'/run-manifest.json`;strictmain/A/Bunchanged,Cdeferred.')
    try:
        allf=pd.read_csv(PRED);f=allf[(allf.dataset=='FD002')&(allf.source=='generated_headline')&allf.generation_seed.isin(SEEDS)].copy()
        assert len(f)==6475 and len(f[['generation_seed','evaluator_seed','engine_id']].drop_duplicates())==6475
        truth=pd.read_csv(ROOT/'upstream/data/RUL_FD002.txt',sep=r'\s+',header=None).iloc[:,0].to_numpy()
        labels=np.minimum(truth,125);engines=np.arange(1,260)
        raw=pd.read_csv(RAW);rows=[];replay=[]
        for (g,e),z in f.groupby(['generation_seed','evaluator_seed']):
            z=z.sort_values('engine_id');assert np.array_equal(z.engine_id,engines) and np.array_equal(z.rul,labels)
            values=model_metrics(z,int(g),int(e),'generated');rows.extend(values)
            ref=raw[(raw.generation_seed==g)&(raw.evaluator_seed==e)];assert len(ref)==1
            difference=abs(values[0]['rmse']-float(ref.iloc[0].rmse));assert difference<.001
            replay.append(dict(generation_seed=int(g),evaluator_seed=int(e),saved_RMSE=values[0]['rmse'],formal_RMSE=float(ref.iloc[0].rmse),abs_difference=difference))
        baseline=allf[(allf.dataset=='FD002')&(allf.source=='real_baseline')]
        assert len(baseline)==1295
        for e,z in baseline.groupby('evaluator_seed'):rows.extend(model_metrics(z,-1,int(e),'real_baseline'))
        permodel=pd.DataFrame(rows);permodel.to_csv(OUT/'per_predictor_by_stage.csv',index=False)
        pd.DataFrame(replay).to_csv(OUT/'formal_replay.csv',index=False)
        summary=[]
        for (source,g,b),z in permodel.groupby(['source','generation_seed','bin'],sort=False):
            assert z.evaluator_seed.tolist()==SEEDS
            summary.append(dict(source=source,generation_seed=int(g),bin=b,n_engines=int(z.n_engines.iloc[0]),
                mean_predictor_RMSE=float(z.rmse.mean()),RMSE_SD=float(z.rmse.std(ddof=1)),
                predictor_RMSE_min=float(z.rmse.min()),predictor_RMSE_max=float(z.rmse.max()),
                mean_MSE=float(z.mse.mean()),pooled_RMSE=float(np.sqrt(z.mse.mean())),
                signed_error=float(z.signed_error.mean()),predictor_bias_min=float(z.signed_error.min()),predictor_bias_max=float(z.signed_error.max()),
                positive_bias_predictors=int((z.signed_error>0).sum()),negative_bias_predictors=int((z.signed_error<0).sum())))
        s=pd.DataFrame(summary);s.to_csv(OUT/'per_generator_by_stage.csv',index=False)
        main=s[s.source=='generated']
        f['error']=f.prediction-f.rul
        byengine=f.groupby(['generation_seed','engine_id'],sort=True).agg(rul=('rul','first'),
            mean_signed_error=('error','mean'),mean_MSE=('error',lambda x:float(np.mean(x*x)))).reset_index()
        byengine.to_csv(OUT/'per_engine_by_generator.csv',index=False)
        common=[]
        for name,lo,hi in BINS:
            z=permodel[(permodel.source=='generated')&permodel.generation_seed.isin(COMMON)&(permodel.bin==name)]
            assert len(z)==15
            engine=byengine[byengine.rul.between(lo,hi)].pivot(index='engine_id',columns='generation_seed',values='mean_signed_error')[COMMON]
            strict=f[f.rul.between(lo,hi)&f.generation_seed.isin(COMMON)].pivot(index='engine_id',columns=['generation_seed','evaluator_seed'],values='error')
            common.append(dict(bin=name,n_engines=len(engine),positive_bias_models=int((z.signed_error>0).sum()),negative_bias_models=int((z.signed_error<0).sum()),
                engines_common3_mean_errors_positive=int((engine>0).all(axis=1).sum()),engines_common3_mean_errors_negative=int((engine<0).all(axis=1).sum()),
                engines_all15_predictions_positive=int((strict>0).all(axis=1).sum()),engines_all15_predictions_negative=int((strict<0).all(axis=1).sum())))
        pd.DataFrame(common).to_csv(OUT/'common3_direction_agreement.csv',index=False)
        contributions=[]
        for name,lo,hi in BINS:
            z=main[main.bin==name];a=z[z.generation_seed==43].iloc[0];ref=z[z.generation_seed.isin(COMMON)].mean_MSE.mean()
            n=int(a.n_engines);delta=float(a.mean_MSE-ref)
            contributions.append(dict(bin=name,n_engines=n,test_fraction=n/259,reference_mean_MSE=float(ref),seed43_MSE=float(a.mean_MSE),
                MSE_difference=delta,contribution_to_overall_excess_MSE=n/259*delta,
                seed43_worse_than_reference=bool(delta>0)))
        overall=main[main.bin=='all'];total=float(overall[overall.generation_seed==43].mean_MSE.iloc[0]-overall[overall.generation_seed.isin(COMMON)].mean_MSE.mean())
        assert abs(sum(x['contribution_to_overall_excess_MSE'] for x in contributions)-total)<1e-10
        for x in contributions:x['share_of_net_excess_MSE']=x['contribution_to_overall_excess_MSE']/total
        pd.DataFrame(contributions).to_csv(OUT/'seed43_MSE_contributions.csv',index=False)
        low_share=sum(x['share_of_net_excess_MSE'] for x in contributions if x['bin'] in ['0-30','31-60'])
        signal=pd.read_csv(SIGNAL);signal=signal[(signal.dataset=='FD002')&signal.seed.isin(SEEDS)]
        signal.to_csv(OUT/'reused_signal_diagnostics.csv',index=False)
        readout=pd.read_csv(READOUT);readout=readout[(readout.dataset=='FD002')&(readout.reference=='dev_matched')&readout.generation_seed.isin(SEEDS)].copy()
        readout['condition_readout_RMSE']=np.sqrt(readout.mse);readout.to_csv(OUT/'reused_condition_readout.csv',index=False)
        direction_by_bin={x['bin']:x for x in common}
        assert all(direction_by_bin[b]['positive_bias_models']==15 for b in ['0-30','31-60'])
        assert all(direction_by_bin[b]['negative_bias_models']==15 for b in ['91-124','125'])
        # Existing readouts do not support a common uniform condition shift across3/23/33.
        readout_pattern={}
        for name in ['0-30','31-60']:
            z=readout[(readout.bin==name)&readout.generation_seed.isin(COMMON)]
            readout_pattern[name]={str(int(row.generation_seed)):float(row.bias) for row in z.itertuples()}
        result=dict(status='complete',test_engines=259,generator_trainings_represented=5,predictors_per_generator=5,
            bins=[dict(bin=name,n_engines=int(np.sum((labels>=lo)&(labels<=hi)))) for name,lo,hi in BINS],
            per_generator=summary,common3=common,seed43_reference_generators=COMMON,
            seed43_total_excess_MSE=total,seed43_contributions=contributions,seed43_low0_60_share=low_share,
            prior_condition_readout_bias=readout_pattern,
            interpretation='Common3/23/33 low0-60overestimation and high91+underestimation, all15stage averages agree. Seed43 worst at0-30/31-60, best61-90; low0-60accounts73.28%of extraMSE vs named3seed reference, capped12528.03%, middle stage improves.',
            limits='Real-trained baseline also shows same broad error direction. Existing train/dev condition-readout shifts are not uniform across3/23/33; cannot infer common generator signal-label mismatch from test bias alone. No matched operating/fault controls in prior signal summaries. SameRUL group does not imply identical engine physics or whole-pipeline independent validation.',
            new_training=0,new_sampling=0,new_model_inference=0)
        dump(OUT/'summary.json',result)
        os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(10,8),sharex=True)
        xx=np.arange(len(BINS))
        for g in SEEDS:
            z=main[main.generation_seed==g].set_index('bin').loc[[b[0] for b in BINS]]
            axes[0].plot(xx,z.mean_predictor_RMSE,marker='o',label=f'generator seed {g}')
            axes[1].plot(xx,z.signed_error,marker='o',label=f'generator seed {g}')
        z=s[s.source=='real_baseline'].set_index('bin').loc[[b[0] for b in BINS]]
        axes[0].plot(xx,z.mean_predictor_RMSE,color='black',ls='--',label='real-trained reference')
        axes[1].plot(xx,z.signed_error,color='black',ls='--',label='real-trained reference')
        axes[1].axhline(0,color='gray',lw=1)
        axes[0].set_ylabel('Mean of five predictor RMSEs (cycles)')
        axes[1].set_ylabel('Mean(prediction - capped true RUL)')
        axes[1].set_xlabel('True RUL group: one terminal window per engine')
        axes[1].set_xticks(xx,[f'{name if name!="125" else "125 (capped)"}\nn={int(np.sum((labels>=lo)&(labels<=hi)))}' for name,lo,hi in BINS])
        for ax in axes:ax.grid(alpha=.25)
        axes[0].legend(ncol=3,fontsize=8)
        fig.suptitle('FD002 formal test: stage error and direction by generator seed\n259 engines; saved predictions only; no new training/inference')
        fig.tight_layout();fig.savefig(OUT/'stage_errors.png',dpi=180);fig.savefig(OUT/'stage_errors.pdf');plt.close(fig)
        text='# FD002正式测试：各生成器种子的分段误差与方向\n\n'
        text+='- [Observed] 复用共同5生成种子×5预测器×259台发动机的6475条已有预测，所有标签与官方末端标签cap125逐项一致；25个总体RMSE复算匹配原记录（既有CPU回放容差.001）。无新训练、采样、模型推理。\n'
        text+='- [Code] 每格RMSE先在该段发动机上逐预测器计算，再平均5预测器，与26.7918的正式汇总层级一致；有向误差=预测−截断真实RUL，正数高估，负数低估。辅助MSE/pooledRMSE另存，不混用；125单列包含未截断RUL>125的57台，不能视为真实寿命恰125。\n\n'
        text+='| RUL段/发动机数 | seed3 RMSE/偏差 | seed13 RMSE/偏差 | seed23 RMSE/偏差 | seed33 RMSE/偏差 | seed43 RMSE/偏差 |\n|---|---:|---:|---:|---:|---:|\n'
        for name,lo,hi in BINS:
            n=int(np.sum((labels>=lo)&(labels<=hi)));line=f'| {name}/{n} '
            for g in SEEDS:
                row=main[(main.generation_seed==g)&(main.bin==name)].iloc[0];line+=f'| {row.mean_predictor_RMSE:.2f} / {row.signed_error:+.2f} '
            text+=line+'|\n'
        text+='\n## 共同方向与相同发动机\n\n'
        text+='- [Observed] seed3/23/33各自5个预测器在0-30、31-60均高估；在91-124、125均低估，四段均15/15方向一致。61-90不一致：seed3低估、23/33高估。\n'
        for row in common:text+=f'- [Observed] {row["bin"]}: 同一台发动机上，三生成器各自平均5预测器后，全部高估{row["engines_common3_mean_errors_positive"]}/{row["n_engines"]}台，全部低估{row["engines_common3_mean_errors_negative"]}/{row["n_engines"]}台；更严格15次单独预测全部高估/低估为{row["engines_all15_predictions_positive"]}/{row["engines_all15_predictions_negative"]}台。\n'
        text+='\n## seed43额外误差定位\n\n'
        text+='- [Observed] 43在0-30/31-60两段RMSE为五组最差，61-90为五组最好；91-124与125也偏高，但都不是每次都比其他所有生成器差。总体偏差约−.034，是低段高估和高段低估抵消，不能当作预测准确。\n'
        text+='- [Code] 以用户点名3/23/33的平均MSE作参考，每段贡献=n段/259×(MSE43段−参考MSE段)，五段精确相加为总体额外MSE；不是把RMSE差直接相加，也不是因果归因。\n\n'
        text+='| RUL段 | 相对参考MSE差 | 加权总体额外MSE贡献 | 占净额外MSE |\n|---|---:|---:|---:|\n'
        for row in contributions:text+=f'| {row["bin"]} | {row["MSE_difference"]:+.5f} | {row["contribution_to_overall_excess_MSE"]:+.5f} | {row["share_of_net_excess_MSE"]:+.2%} |\n'
        text+=f'\n- [Observed] 0-60合计贡献{low_share:.2%}；125贡献28.03%；61-90改善抵消7.43%。因此43既不是所有阶段统一变差，也不是单段独占；优先范围是低RUL两段，并独立保留125组。\n'
        text+='\n## 按共同偏差回看已有训练/验证诊断\n\n'
        text+='- [Observed] 既有真实训练预测器在同一真实测试集也呈低段高估、高段低估：0-30/31-60偏差+14.49/+11.95，91-124/125偏差−12.34/−20.49。因此同方向偏差不是生成模型独有证据。\n'
        text+=f'- [Observed] 复用旧的内部dev整数RUL匹配条件读出，真实训练预测器读生成数据的偏差（读出−条件）为：{readout_pattern}。三个生成器并非同方向统一偏移；不能从真实测试偏差直接推断共同标签错配。读出只是代理，不是生成样本真实寿命，且此dev_matched不是本轮重新划分或新推理。\n'
        text+='- [Observed] 旧同RUL信号诊断在3/23/33/43的0-30、31-60都存在生成/真实水平差；43窗口末端分布距离尤其大：0-30约.2438对真实A/B参照.0429，31-60约.2557对.0459。其窗口均值距离却不总是更大，不能只看窗口均值。原始CSV完整复用至本目录。\n'
        text+='- [Hypothesis] 共同行为支持继续核对相同寿命段的训练分布/可辨识性与共有处理，但现有条件读出不支持一个统一方向的生成信号标签平移解释。43应重点查低寿命的末端信号和另列125，而非直接假定与3/23/33同因。旧信号诊断没有匹配工况/故障类型，不能把距离或读出偏差当因果证据。\n'
        text+='- [Code] 本轮只定位正式测试误差并复用既有训练/dev证据；不按测试结果选择轮次/阈值，不改模型，不启动邻域或传感器搜索。5训练种子不是25独立生成器实验。\n'
        text+=f'- 决策线：[Observed] 已有26.7918正式整体均值分段定位→[Paper]无对应分段目标→差距NA→跨生成器5组/预测器5次描述性一致→严格结论不变。Run `{RUN.relative_to(ROOT)}/run-manifest.json`；表与图 `{OUT.relative_to(ROOT)}/`。Strictmain/A/B不变,C推迟,预算关闭。\n'
        REPORT.write_text(text)
        for p,expected in manifest['inputs_sha256'].items():assert sha(ROOT/p)==expected,p
        verify=dict(passed=True,source_prediction_hash_bound=True,unique_records=6475,engines_per_model=259,
            models=25,truth_labels_match=True,replayed_formal_RMSEs=25,max_formal_replay_abs_difference=max(r['abs_difference'] for r in replay),
            bin_counts_sum_to259=True,seed43_MSE_contributions_sum=True,new_training=0,new_sampling=0,new_inference=0)
        dump(RUN/'verification.json',verify)
        manifest.update(status='complete',completed_at=now(),outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in list(OUT.glob('*'))+[REPORT]},
            new_training=0,new_sampling=0,new_inference=0)
        dump(RUN/'run-manifest.json',manifest)
        state('complete','- [Observed]25savedformalpredictors/259engines verified;5RULbins counts61/41/47/53/57. Common3/23/33all15predictors overestimate0-60andunderestimate91+,61-90mixed. Sameenginecommonmeanerrors:0-30allover52/61,31-60allover24/41,91-124allunder36/53,125allunder51/57. Seed43worst0-30/31-60,best61-90;relative3/23/33reference,low0-60accounts73.2805%ofnetextraMSE,capped12528.0310%,61-90offset−7.4304%. Overall43bias−.034is cancellation.\n- [Observed] Existingrealbaseline has samebroaddirections;priorRUL-matcheddevreadoutshiftsfor3/23/33mixedratherthanuniform. Oldsignaldiagnostic43end-level distanceslowRULmuchlargerthanrealA/Breference,meansnotuniformlylarge. [Hypothesis] Sharedtestbiasalonecannotprovecommonsignal/labelmismatch;43localizedlow-RULplus125patternrequiresdistinctscope,notallstagefailure. No newtrain/devinferenceorsensorsearch.\n- Snapshot `'+str(RUN.relative_to(ROOT))+'/run-manifest.json`; report `'+str(REPORT.relative_to(ROOT))+'`;verificationpassed,plot/tables saved. Diagnostic/supporting,strictmain/A/Bunchanged,Cdeferred;noofficialtesttuning,epochsearchpaused,budgetclosed.')
        print(json.dumps(dict(status='complete',counts=result['bins'],seed43_low_share=low_share,contributions=contributions,common3=common),ensure_ascii=False,indent=2))
    except BaseException:
        manifest.update(status='failed',failed_at=now(),failure=traceback.format_exc());dump(RUN/'run-manifest.json',manifest)
        state('failed','- [Observed] Read-only stage analysis stopped; preserve failure/outputs; no model training/sampling/inference and no automatic retries.')
        raise


if __name__=='__main__':main()
