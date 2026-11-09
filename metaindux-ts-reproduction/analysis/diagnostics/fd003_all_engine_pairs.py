"""ALL100FD003 matched test-window MSE differences, from frozen predictions."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import platform
import sys
import traceback

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'results/diagnostics/fd003_all_engine_pairs_20261027'
RUN=ROOT/'runs/diagnostic-fd003-all-engine-pairs-20261027'
REPORT=ROOT/'docs/conclusions/final/fd003_all_engine_pairs_20261027_report.md'
PROTOCOL=ROOT/'docs/operations/FD003_ALL_ENGINE_PAIRED_MSE.md'
SEEDS=[3,13,23,33,43]
BINS=[('0-30',0,30),('31-60',31,60),('61-90',61,90),('91-124',91,124),('125',125,125)]


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(v):
    d=v.delta.to_numpy()
    return dict(n_engines=len(v),n_synthetic_worse=int((d>1e-10).sum()),
        n_synthetic_better=int((d< -1e-10).sum()),n_equal=int((abs(d)<=1e-10).sum()),
        mean_delta=float(d.mean()),median_delta=float(np.median(d)),
        q25_delta=float(np.quantile(d,.25)),q75_delta=float(np.quantile(d,.75)),
        real_mse=float(v.real_mse.mean()),synthetic_mse=float(v.synthetic_mse.mean()),
        real_pooled_rmse=float(np.sqrt(v.real_mse.mean())),synthetic_pooled_rmse=float(np.sqrt(v.synthetic_mse.mean())))


def main():
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    mp=RUN/'run-manifest.json'
    if mp.exists():raise RuntimeError('Preserve existing run')
    prediction_path=ROOT/'results/diagnostics/rul_segment_errors_w48/predictions.csv'
    prior_path=ROOT/'runs/diagnostic-fd003-predictor-fit-20261026/run-manifest.json'
    paths=[Path(__file__),PROTOCOL,prediction_path,prior_path,
        ROOT/'results/diagnostics/rul_segment_errors_w48/summary.csv',
        ROOT/'results/fd003_w48_public_code_evaluation_raw.csv',ROOT/'results/real_data_baseline_w48.csv',
        ROOT/'upstream/data/test_FD003.txt',ROOT/'upstream/data/RUL_FD003.txt']
    m=dict(run_id=RUN.name,scope_class='diagnostic',role='supporting',status='running',
        scope_contract='scope-20261027-fd003-all-engine-pairs',started_at=now(),no_training=True,no_new_inference=True,
        command=[sys.executable,str(Path(__file__).relative_to(ROOT))],
        user_authorization='User requests all100engine paired mean-squared-error differences before model changes',
        source_snapshot='SOURCE_SNAPSHOT.md',inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        seeds=SEEDS,bootstrap=dict(seed=20261027,n_resamples=20000,unit='paired testengine',conditional_on='fixed fitted model families'),
        environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()),
        definition='delta_i=mean25syntheticmodelSE_i-mean5realmodelSE_i; identical realtest lastwindow/label')
    mp.write_text(json.dumps(m,indent=2))
    try:
        prior=json.loads(prior_path.read_text());assert prior['status']=='complete'
        key=str(prediction_path.relative_to(ROOT));assert sha(prediction_path)==prior['inputs_sha256'][key]
        raw=pd.read_csv(prediction_path);raw=raw[raw.dataset=='FD003'].copy()
        assert len(raw)==3000 and not raw.duplicated(['source','generation_seed','evaluator_seed','engine_id']).any()
        synthetic=raw[raw.source=='generated_headline'].copy();real=raw[raw.source=='real_baseline'].copy()
        assert len(synthetic)==2500 and len(real)==500
        assert set(synthetic.engine_id)==set(real.engine_id)==set(range(1,101))
        assert set(synthetic.generation_seed)==set(synthetic.evaluator_seed)==set(real.evaluator_seed)==set(SEEDS)
        assert set(real.generation_seed)=={-1}
        assert raw.groupby('engine_id').rul.nunique().eq(1).all()
        assert np.isfinite(raw.prediction).all() and np.allclose(raw.error,raw.prediction-raw.rul)
        source_truth=pd.read_csv(ROOT/'upstream/data/RUL_FD003.txt',sep=r'\s+',header=None).iloc[:,0].to_numpy()
        source_records=pd.read_csv(ROOT/'upstream/data/test_FD003.txt',sep=r'\s+',header=None,usecols=[0,1])
        assert len(source_truth)==100
        metadata=[]
        for engine,unit in source_records.groupby(0,sort=True):
            truth=float(source_truth[int(engine)-1]);cycle=unit[1].to_numpy()
            metadata.append(dict(engine_id=int(engine),rul=min(truth,125),raw_terminal_rul=truth,
                window_start_cycle=int(cycle[-min(48,len(cycle))]),window_end_cycle=int(cycle[-1]),
                padding_rows=max(48-len(cycle),0)))
        metadata=pd.DataFrame(metadata)
        assert np.array_equal(raw.groupby('engine_id').rul.first().sort_index().to_numpy(),metadata.rul.to_numpy())
        # Pair by engine AND evaluatorseed. Each realfit is reused, never refit.
        pair=synthetic[['engine_id','rul','generation_seed','evaluator_seed','prediction','error']].rename(
            columns={'prediction':'synthetic_prediction','error':'synthetic_error'}).merge(
            real[['engine_id','rul','evaluator_seed','prediction','error']].rename(
                columns={'prediction':'real_prediction','error':'real_error'}),
            on=['engine_id','rul','evaluator_seed'],validate='many_to_one')
        assert len(pair)==2500 and pair.groupby('engine_id').size().eq(25).all()
        pair['synthetic_se']=pair.synthetic_error**2;pair['real_se']=pair.real_error**2
        pair['delta']=pair.synthetic_se-pair.real_se
        pair.to_csv(OUT/'paired_model_errors.csv',index=False)
        bygen=pair.groupby(['engine_id','rul','generation_seed']).agg(
            synthetic_mse=('synthetic_se','mean'),real_mse=('real_se','mean'),delta=('delta','mean')).reset_index()
        bygen.to_csv(OUT/'per_engine_generation_seed.csv',index=False)
        byeval=pair.groupby(['engine_id','rul','evaluator_seed']).agg(
            synthetic_mse=('synthetic_se','mean'),real_mse=('real_se','mean'),delta=('delta','mean')).reset_index()
        byeval.to_csv(OUT/'per_engine_evaluator_seed.csv',index=False)
        engine=pair.groupby('engine_id').agg(synthetic_mse=('synthetic_se','mean'),real_mse=('real_se','mean'),
            delta=('delta','mean'),synthetic_mean_prediction=('synthetic_prediction','mean'),
            real_mean_prediction=('real_prediction','mean')).reset_index().merge(metadata,on='engine_id',validate='one_to_one')
        engine['n_synthetic_models']=25;engine['n_real_models']=5
        engine['synthetic_rms_error']=np.sqrt(engine.synthetic_mse);engine['real_rms_error']=np.sqrt(engine.real_mse)
        stability=bygen.groupby('engine_id').agg(n_generation_seeds_worse=('delta',lambda v:int((v>1e-10).sum())),
            delta_gen_min=('delta','min'),delta_gen_max=('delta','max')).reset_index()
        engine=engine.merge(stability,on='engine_id',validate='one_to_one')
        # Evaluation seeds are also shared across the two families.
        eval_stability=byeval.groupby('engine_id').agg(n_evaluator_seeds_worse=('delta',lambda v:int((v>1e-10).sum())),
            delta_eval_min=('delta','min'),delta_eval_max=('delta','max')).reset_index()
        engine=engine.merge(eval_stability,on='engine_id',validate='one_to_one').sort_values('engine_id')
        assert np.allclose(engine.delta,engine.synthetic_mse-engine.real_mse)
        engine.to_csv(OUT/'all_100_engine_pairs.csv',index=False)
        total=stats(engine);rng=np.random.RandomState(20261027)
        indices=rng.randint(100,size=(20000,100));draw=engine.delta.to_numpy()[indices]
        total.update(mean_delta_engine_bootstrap_ci95=np.quantile(draw.mean(1),[.025,.975]).tolist(),
            median_delta_engine_bootstrap_ci95=np.quantile(np.median(draw,axis=1),[.025,.975]).tolist(),
            n_worse_all5generationseeds=int((engine.n_generation_seeds_worse==5).sum()),
            n_better_all5generationseeds=int((engine.n_generation_seeds_worse==0).sum()),
            n_generationseed_mixed=int(engine.n_generation_seeds_worse.between(1,4).sum()),
            n_padded_windows=int((engine.padding_rows>0).sum()))
        segments=[]
        for name,lo,hi in BINS:
            v=engine[engine.rul.between(lo,hi)];s=stats(v)
            segments.append(dict(bin=name,**s,contribution_to_overall_mean_delta=float(v.delta.sum()/100),
                fraction_of_net_total_delta=float(v.delta.sum()/engine.delta.sum())))
        pd.DataFrame(segments).to_csv(OUT/'rul_segment_summary.csv',index=False)
        generation=[]
        for g,v in bygen.groupby('generation_seed'):
            generation.append(dict(generation_seed=int(g),**stats(v)))
        pd.DataFrame(generation).to_csv(OUT/'generation_seed_summary.csv',index=False)
        positives=engine[engine.delta>0].sort_values('delta',ascending=False)
        net=float(engine.delta.sum());gross=float(positives.delta.sum())
        concentration=[]
        for k in [1,5,10]:
            chosen=positives.head(k);left=engine[~engine.engine_id.isin(chosen.engine_id)]
            concentration.append(dict(top_k=k,engines=','.join(map(str,chosen.engine_id)),
                sum_delta=float(chosen.delta.sum()),share_gross_positive=float(chosen.delta.sum()/gross),
                share_net_total=float(chosen.delta.sum()/net),remaining_mean_delta=float(left.delta.mean()),
                remaining_synthetic_pooled_rmse=float(np.sqrt(left.synthetic_mse.mean())),
                remaining_real_pooled_rmse=float(np.sqrt(left.real_mse.mean()))))
        pd.DataFrame(concentration).to_csv(OUT/'concentration.csv',index=False)
        no75=stats(engine[engine.engine_id!=75]);total['without75']=no75
        total['engine75_delta']=float(engine[engine.engine_id==75].delta.iloc[0])
        total['engine75_share_net_total']=total['engine75_delta']/net
        # Compare the30source checkpoints' global errors to original records.
        formal=pd.read_csv(ROOT/'results/fd003_w48_public_code_evaluation_raw.csv')
        baseline=pd.read_csv(ROOT/'results/real_data_baseline_w48.csv');baseline=baseline[baseline.dataset=='FD003']
        checks=[]
        for (g,e),v in synthetic.groupby(['generation_seed','evaluator_seed']):
            a=formal[(formal.generation_seed==g)&(formal.evaluator_seed==e)];assert len(a)==1
            replay=float(np.sqrt(np.mean(v.error**2)));delta=abs(replay-float(a.rmse.iloc[0]));assert delta<.001
            checks.append(dict(source='synthetic',generation_seed=int(g),evaluator_seed=int(e),max_rmse_difference=delta))
        for e,v in real.groupby('evaluator_seed'):
            a=baseline[baseline.evaluator_seed==e];assert len(a)==1
            replay=float(np.sqrt(np.mean(v.error**2)));delta=abs(replay-float(a.rmse.iloc[0]));assert delta<.001
            checks.append(dict(source='real',generation_seed=-1,evaluator_seed=int(e),max_rmse_difference=delta))
        pd.DataFrame(checks).to_csv(OUT/'source_metric_checks.csv',index=False)
        old_summary=pd.read_csv(ROOT/'results/diagnostics/rul_segment_errors_w48/summary.csv')
        for source,value in [('generated_common5',total['synthetic_pooled_rmse']),('real_baseline',total['real_pooled_rmse'])]:
            a=old_summary[(old_summary.dataset=='FD003')&(old_summary.source==source)&(old_summary.bin=='all')]
            assert len(a)==1 and abs(float(a.pooled_rmse.iloc[0])-value)<1e-8
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        ax=axes[0,0];ax.bar(engine.engine_id,engine.delta,color=np.where(engine.delta>0,'#d95f02','#377eb8'))
        ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Testengine ID (ALL100)',ylabel='Delta MSE (cycles²)',title='Same-window paired effects; positive = synthetic worse')
        for i in set(positives.head(5).engine_id.tolist()+[75]):
            v=engine[engine.engine_id==i].iloc[0];ax.annotate(str(i),(i,v.delta),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
        ax=axes[0,1];ax.scatter(engine.real_rms_error,engine.synthetic_rms_error,c=engine.rul,cmap='viridis',s=30)
        high=max(engine.real_rms_error.max(),engine.synthetic_rms_error.max())*1.04
        ax.plot([0,high],[0,high],'--',color='black',lw=.8);ax.set(xlabel='Real-trained per-engine RMSerror',ylabel='Synthetic-trained per-engine RMSerror',title='Same realtest window per point')
        for i in set(positives.head(5).engine_id.tolist()+[75]):
            v=engine[engine.engine_id==i].iloc[0];ax.annotate(str(i),(v.real_rms_error,v.synthetic_rms_error),fontsize=8)
        ax=axes[1,0];order=np.sort(engine.delta);ax.step(order,np.arange(1,101)/100,where='post')
        ax.axvline(0,color='black',ls='--',lw=.8);ax.axhline(.5,color='gray',ls=':',lw=.8)
        ax.set(xlabel='Delta MSE (cycles²)',ylabel='Fraction of testengines',title=f'Empirical paired distribution | median {total["median_delta"]:.2f}')
        ax=axes[1,1];curve=positives.delta.cumsum()/gross*100
        ax.plot(np.arange(1,len(curve)+1),curve,marker='.',ms=3)
        for k in [5,10]:ax.scatter(k,float(curve.iloc[k-1]),color='#d95f02');ax.annotate(f'top{k}: {curve.iloc[k-1]:.1f}%',(k,curve.iloc[k-1]),xytext=(5,-13),textcoords='offset points',fontsize=8)
        ax.set(xlabel='Number of engines, ranked by positive Delta',ylabel='Cumulative share of gross positive Delta (%)',title='Concentration of worse-engine losses (before offsets)')
        for ax in axes.flat:ax.grid(alpha=.15)
        fig.suptitle('FD003 ALL100paired realtest engines | fixed25synthetic fits versus5real fits\nDelta is mean squared-error difference, not ensemble error; no model changes')
        fig.tight_layout(rect=(0,0,1,.94))
        for ext in ['png','pdf']:fig.savefig(OUT/f'all_engine_paired_mse.{ext}',dpi=180)
        plt.close(fig)
        (OUT/'summary.json').write_text(json.dumps(total,indent=2))
        # Complete100rows in the report, not only selected failures.
        all_rows='\n'.join(f'| {r.engine_id} | {int(r.rul)} | {r.real_mse:.3f} | {r.synthetic_mse:.3f} | {r.delta:+.3f} | {r.n_generation_seeds_worse}/5 |' for r in engine.itertuples())
        segment_rows='\n'.join(f'| {r["bin"]} | {r["n_engines"]} | {r["n_synthetic_worse"]}/{r["n_synthetic_better"]} | {r["real_pooled_rmse"]:.3f} | {r["synthetic_pooled_rmse"]:.3f} | {r["mean_delta"]:+.3f} | {100*r["fraction_of_net_total_delta"]:+.1f}% |' for r in segments)
        generation_rows='\n'.join(f'| {r["generation_seed"]} | {r["n_synthetic_worse"]}/{r["n_synthetic_better"]} | {r["mean_delta"]:+.3f} | {r["median_delta"]:+.3f} | {r["synthetic_pooled_rmse"]:.3f} |' for r in generation)
        concentration_rows='\n'.join(f'| {r["top_k"]} | {r["engines"]} | {100*r["share_gross_positive"]:.1f}% | {100*r["share_net_total"]:.1f}% | {r["remaining_mean_delta"]:+.3f} | {r["remaining_real_pooled_rmse"]:.3f}/{r["remaining_synthetic_pooled_rmse"]:.3f} |' for r in concentration)
        ci=total['mean_delta_engine_bootstrap_ci95'];mi=total['median_delta_engine_bootstrap_ci95']
        text=f'''# FD003全部100台测试发动机的配对MSE结果

Run ID：`{RUN.name}`；scope：diagnostic/supporting；2026-10-27。用户明确要求先看全部100台配对差值，再决定是否修改模型。本轮仅聚合既有预测，无新推理或训练。

## 定义与完整性

**[Observed]** 每台都使用同一个真实测试末端48步窗口及同一个RUL标签，两边分别计算模型平方误差均值。`delta=synthetic_mse-real_mse`，正值表示生成数据训练模型较差，负值表示较好；单位为寿命周期平方。不是轨迹RMSE，也不是平均预测的误差。全部100台、ID1–100均纳入，无剔除病例。

**[Code]** 主值为25生成模型（5生成种子×5评估种子）与5真实模型的平均平方误差之差。另按同一个评估种子配对：每个生成种子5模型对同一5真实模型，逐台差值先平均评估种子再平均生成种子，代数上与主定义一致。共享真实模型不当作25个独立真实重复。

**[Observed]** 标签与原始truth逐项核对，原始RUL>=125仍合并为125，输出未裁剪；原文件周期元数据保存在完整CSV。少于48真实周期需重复首行补齐，本批有{total['n_padded_windows']}台，两类模型使用同一补齐结果。3000原始预测记录及30模型总体RMSE均核对通过。

## 总体结果

**[Observed]** 生成较差{total['n_synthetic_worse']}台，较好{total['n_synthetic_better']}台，相等{total['n_equal']}台。delta均值{total['mean_delta']:+.3f}，中位数{total['median_delta']:+.3f}，四分位范围[{total['q25_delta']:.3f}, {total['q75_delta']:.3f}]。两边合并平方误差后开方，真实RMSE{total['real_pooled_rmse']:.3f}、生成{total['synthetic_pooled_rmse']:.3f}；这不同于逐模型RMSE算术平均的历史headline数值。

**[Observed]** 固定已训练模型，对100台发动机做配对bootstrap（20000次，seed20261027），平均delta95%描述区间[{ci[0]:.3f}, {ci[1]:.3f}]；中位数区间[{mi[0]:.3f}, {mi[1]:.3f}]。区间只描述该经验测试发动机集合、固定模型下的抽样变化，不包含完整生成/训练算法不确定性，不把2500模型配对行视为2500台独立发动机。

## 逐RUL段

| 真RUL段 | 发动机数 | 生成较差/较好台数 | 真实pooledRMSE | 生成pooledRMSE | 平均delta | 占总净差值 |
|---|---:|---:|---:|---:|---:|---:|
{segment_rows}

**[Observed]** 每段的全局均值贡献为该段delta总和除以100，完整值保存在`rul_segment_summary.csv`，各段贡献相加等于总体平均delta。净差值份额允许负值，不能把它当作发动机占比。

## 生成种子敏感性：每组均5模型对5模型

| 生成种子 | 生成较差/较好台数 | 平均delta | 中位数delta | 生成pooledRMSE |
|---|---:|---:|---:|---:|
{generation_rows}

**[Observed]** 逐台5生成种子全部较差{total['n_worse_all5generationseeds']}台，全部较好{total['n_better_all5generationseeds']}台，方向随生成种子变化{total['n_generationseed_mixed']}台。按评估种子的稳定性也已逐台保存，两个种子维度不可混为同一个变化来源。

## 差距是否由少数发动机拉高

| 最大正delta前k台 | 发动机ID | 占全部正差值 | 占总净差值 | 其余发动机平均delta | 其余真实/生成pooledRMSE |
|---|---|---:|---:|---:|---:|
{concentration_rows}

**[Observed]** 正差值总和表示所有变差发动机造成的平方误差增加；总净差值还扣除了变好发动机的收益，因而前k占净差值可能超过100%。这里的“其余”只为敏感性解释，不建议删除测试病例，不改变正式评价集。

**[Observed]** 75号delta={total['engine75_delta']:.3f}，占总体净差值{100*total['engine75_share_net_total']:.2f}%。仅不计75号，剩余99台平均delta={no75['mean_delta']:+.3f}，真实/生成pooledRMSE={no75['real_pooled_rmse']:.3f}/{no75['synthetic_pooled_rmse']:.3f}。这不是重定义正式结果。

## 决策边界与停止

**[Hypothesis]** 全100台的符号分布、中位数与误差集中程度应共同决定下一步优先查局部区域还是整体训练分布，不能由75号一例决定。delta本身仍不能唯一判定覆盖、信号偏差、模型容量或优化原因；真实数据训练预测器也不是误差下界。

**[Observed]** 用户要求的100台配对表已完成。当前不修改模型，不启动原型、局部权重、锚点或新训练，也不根据测试delta挑选训练数据。任何后续方案需在训练/验证数据上定义规则并冻结独立验证。历史strictmain/A/B不变，C延后，零标签路线仍暂停。

**[Observed]** 五段式：本轮delta全表与汇总已测→论文无逐发动机配对诊断目标→论文差距不适用→5生成种子及固定模型发动机bootstrap单独报告，不能唯一因果归因→不改变既有严格复现结论。Scope `scope-20261027-fd003-all-engine-pairs`。

## 全部100台明细（按发动机ID）

**[Observed]** MSE单位均为周期平方；最后一列为5个生成种子中delta>0的个数。CSV还保留两边RMSerror、平均预测、原始RUL、窗口周期、补齐数和评估种子稳定性。

| 测试发动机 | cappedRUL | 真实MSE | 生成MSE | delta | 生成种子较差数 |
|---|---:|---:|---:|---:|---:|
{all_rows}

## 产物

- `results/diagnostics/fd003_all_engine_pairs_20261027/all_100_engine_pairs.csv`：完整100台主结果。
- 同目录`paired_model_errors.csv`：2500行同评估种子配对平方误差；`per_engine_generation_seed.csv`与`per_engine_evaluator_seed.csv`：各500行种子分解。
- 同目录`summary.json`、`rul_segment_summary.csv`、`generation_seed_summary.csv`、`concentration.csv`：汇总、分段、种子和集中程度。
- 同目录`all_engine_paired_mse.png/pdf`：全部ID差值、两边RMSerror、经验分布和正差值集中图。
- 同目录`source_metric_checks.csv`：30个既有模型的原总体RMSE复算核对。
- `runs/{RUN.name}/run-manifest.json`：执行前协议、输入/源码及输出哈希；协议`docs/operations/FD003_ALL_ENGINE_PAIRED_MSE.md`。
'''
        REPORT.write_text(text)
        m.update(status='complete',n_engines=100,n_source_predictions=len(raw),n_paired_rows=len(pair),
            n_generationseed_rows=len(bygen),n_evaluatorseed_rows=len(byeval),summary=total,
            report=str(REPORT.relative_to(ROOT)),report_sha256=sha(REPORT),
            max_original_model_rmse_difference=max(c['max_rmse_difference'] for c in checks),
            outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file()})
        print(json.dumps(total,indent=2));print(pd.DataFrame(segments).round(4).to_string(index=False));print(pd.DataFrame(concentration).round(4).to_string(index=False))
    except BaseException:
        m.update(status='failed',error=traceback.format_exc());raise
    finally:
        m['finished_at']=now();mp.write_text(json.dumps(m,indent=2))


if __name__=='__main__':main()
