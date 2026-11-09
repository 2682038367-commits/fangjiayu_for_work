"""Audit EXT-002 final checkpoints and select weights using realinternalval only."""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN=ROOT/'runs/ext-fd003-rul-weight-screen-20261027'
OUT=ROOT/'results/exploratory/fd003_rul_weight_screen_20261027'
REPORT=ROOT/'docs/conclusions/final/fd003_rul_weight_screen_20261027_report.md'
BINS=['all','0-30','31-60','61-90','91-124','125']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    mp=RUN/'run-manifest.json';m=json.loads(mp.read_text());assert m['status']=='complete' and m['completed_models']==15
    for field in ['inputs_sha256','outputs_sha256','checkpoint_sha256']:
        for name,digest in m[field].items():assert sha(ROOT/name)==digest,name
    c=m['config'];segments=pd.read_csv(OUT/'segments.csv');engines=pd.read_csv(OUT/'validation_engines.csv')
    training=pd.read_csv(OUT/'training_audit.csv');history=pd.read_csv(OUT/'history.csv')
    assert len(training)==15 and len(history)==525 and len(segments)==270
    assert not training.duplicated(['generation_seed','weight']).any()
    assert training.updates.eq(1925).all() and training.sample_visits.eq(490490).all() and training.epochs.eq(35).all()
    assert training.initial_sha256.nunique()==training.consumed_order_sha256.nunique()==1
    assert set(training.consumed_order_sha256)=={c['expected_consumed_order_sha256']}
    for _,v in history.groupby(['generation_seed','weight']):
        assert v.epoch.tolist()==list(range(1,36)) and v.updates_so_far.tolist()==[55*i for i in range(1,36)]
        assert np.allclose(v.learning_rate,[.002]*15+[.0002]*10+[.00002]*10)
    with np.load(RUN/'real_validation.npz') as z:
        vx=z['data'];vy=z['label'].astype(float);ids=z['engine_id']
    assert len(np.unique(ids))==30 and len(vx)==5134
    sys.path.insert(0,str(ROOT/'upstream'))
    from models.LSTM import LSTM
    from types import SimpleNamespace
    torch.set_num_threads(2)
    # Separate CPU and original-CUDA replays were frozen after the first strict
    # CPU absolute1e-3 check failed. Preserve that failure, do not loosen it.
    cpu_replay=pd.read_csv(OUT/'cpu_numerical_replay.csv')
    cuda_replay=pd.read_csv(OUT/'cuda_numerical_replay.csv')
    assert len(cpu_replay)==len(cuda_replay)==90
    assert cuda_replay.max_prediction_difference.max()<.001
    checks=[]
    for row in training.itertuples():
        ck=torch.load(ROOT/row.checkpoint,map_location='cpu',weights_only=True)
        assert ck['epoch']==35 and ck['updates']==1925 and ck['initial_sha256']==row.initial_sha256
        assert ck['consumed_order_sha256']==row.consumed_order_sha256
        name=f'gen{row.generation_seed}_weight{row.weight}_init3'
        p=np.load(OUT/'replay'/f'cpu_{name}.npy')
        gpu=np.load(OUT/'replay'/f'cuda_{name}.npy')
        with np.load(OUT/'predictions'/f'{name}.npz') as z:saved=z['real_validation'];sp=z['synthetic']
        delta=float(np.max(abs(p-saved)))
        gpu_delta=float(np.max(abs(gpu-saved)));assert gpu_delta<.001,(name,gpu_delta)
        per=engines[(engines.generation_seed==row.generation_seed)&(engines.weight==row.weight)]
        for r in per.itertuples():
            mask=ids==r.engine_id
            if r.bin!='all':
                lo,hi=(125,125) if r.bin=='125' else map(int,r.bin.split('-'))
                mask&=(vy>=lo)&(vy<=hi)
            assert mask.sum()==r.n_windows
            assert abs(np.mean((saved[mask]-vy[mask])**2)-r.mse)<1e-8
        checks.append(dict(generation_seed=row.generation_seed,weight=row.weight,cpu_replay_max_difference=delta,
            cpu_absolute_1e3_gate_pass=delta<.001,cuda_replay_max_difference=gpu_delta))
    pd.DataFrame(checks).to_csv(OUT/'checkpoint_replay_checks.csv',index=False)
    # Aggregate generatorseeds in MSE space, preserving equalengine weights.
    summaries=[]
    for (domain,weight,stage),v in segments.groupby(['domain','weight','bin']):
        assert len(v)==5
        summaries.append(dict(domain=domain,weight=int(weight),bin=stage,n_generation_seeds=5,
            n_windows=int(v.n_windows.iloc[0]),n_engines=int(v.n_engines.iloc[0]),
            mse_window=float(v.mse_window.mean()),rmse_window=float(np.sqrt(v.mse_window.mean())),
            mse_engine=float(v.mse_engine.mean()),rmse_engine=float(np.sqrt(v.mse_engine.mean()))))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'summary.csv',index=False)
    val=summary[summary.domain=='real_validation'];pairs=[]
    detailed=segments[segments.domain=='real_validation']
    for weight in [2,4]:
        for seed in c['generation_seeds']:
            base=detailed[(detailed.generation_seed==seed)&(detailed.weight==1)].set_index('bin')
            candidate=detailed[(detailed.generation_seed==seed)&(detailed.weight==weight)].set_index('bin')
            for stage in BINS:
                a=base.loc[stage];b=candidate.loc[stage]
                pairs.append(dict(generation_seed=seed,weight=weight,bin=stage,
                    baseline_rmse_engine=float(a.rmse_engine),candidate_rmse_engine=float(b.rmse_engine),
                    delta_mse_engine=float(b.mse_engine-a.mse_engine),delta_rmse_engine=float(b.rmse_engine-a.rmse_engine),
                    percent_rmse_change=float((b.rmse_engine/a.rmse_engine-1)*100)))
    pair=pd.DataFrame(pairs);pair.to_csv(OUT/'paired_generation_deltas.csv',index=False)
    baseline=val[val.weight==1].set_index('bin');choices=[]
    for weight in [2,4]:
        candidate=val[val.weight==weight].set_index('bin')
        improvement=int(((pair.weight==weight)&(pair.bin=='31-60')&(pair.delta_mse_engine<0)).sum())
        others=[b for b in BINS if b not in ['all','31-60']]
        changes={b:float(candidate.loc[b,'rmse_engine']/baseline.loc[b,'rmse_engine']-1) for b in others}
        overall=bool(candidate.loc['all','mse_engine']<baseline.loc['all','mse_engine'])
        target=bool(candidate.loc['31-60','mse_engine']<baseline.loc['31-60','mse_engine'])
        majority=improvement>=c['selection']['min_target_improved_generation_seeds']
        guard=all(x<=c['selection']['max_other_bin_rmse_regression_fraction'] for x in changes.values())
        choices.append(dict(weight=weight,overall_improves=overall,target_improves=target,
            target_improved_generation_seeds=improvement,majority_target_pass=majority,other_bins_guard_pass=guard,
            max_other_bin_rmse_change_percent=max(changes.values())*100,
            overall_mse_engine=float(candidate.loc['all','mse_engine']),eligible=bool(overall and target and majority and guard)))
    choice=pd.DataFrame(choices);eligible=choice[choice.eligible].sort_values(['overall_mse_engine','weight'])
    selected=int(eligible.weight.iloc[0]) if len(eligible) else 1
    choice['selected']=choice.weight==selected;choice.to_csv(OUT/'weight_selection.csv',index=False)
    cpu_scores=cpu_replay.groupby(['weight','bin']).mse_engine.mean().unstack('bin')
    cpu_candidates=[]
    for weight in [2,4]:
        target=cpu_replay[cpu_replay.bin=='31-60'].pivot(index='generation_seed',columns='weight',values='mse_engine')
        nbetter=int((target[weight]<target[1]).sum())
        eligible_cpu=bool(cpu_scores.loc[weight,'all']<cpu_scores.loc[1,'all']
            and cpu_scores.loc[weight,'31-60']<cpu_scores.loc[1,'31-60'] and nbetter>=3
            and all(np.sqrt(cpu_scores.loc[weight,b]/cpu_scores.loc[1,b])-1<=.05 for b in ['0-30','61-90','91-124','125']))
        cpu_candidates.append(dict(weight=weight,eligible=eligible_cpu,overall_mse=float(cpu_scores.loc[weight,'all'])))
    cpu_eligible=sorted([v for v in cpu_candidates if v['eligible']],key=lambda v:(v['overall_mse'],v['weight']))
    cpu_selected=cpu_eligible[0]['weight'] if cpu_eligible else 1
    assert cpu_selected==selected
    assert all(v['eligible']==bool(choice[choice.weight==v['weight']].eligible.iloc[0]) for v in cpu_candidates)
    numerical=dict(initial_cpu_absolute_1e3_gate_pass=False,cpu_absolute_gate_failed_models=int((~pd.DataFrame(checks).cpu_absolute_1e3_gate_pass).sum()),
        cuda_strict_replay_pass=True,max_cuda_prediction_difference=float(cuda_replay.max_prediction_difference.max()),
        max_cpu_prediction_difference=float(cpu_replay.max_prediction_difference.max()),
        max_cpu_realvalidation_rmse_difference=float(cpu_replay.rmse_difference.abs().max()),
        selected_weight_original_gpu=selected,selected_weight_cpu=cpu_selected,all_candidate_eligibility_identical=True,
        explanation='Device/backend numerical-path differences observed;original CUDA predictions reproduced exactly;CPU threshold failure retained,not relabeled as strictCPU pass')
    (OUT/'numerical_selection_sensitivity.json').write_text(json.dumps(numerical,indent=2))
    selection=dict(selected_weight=selected,screen_only=True,global_default_changed=False,
        official_test_accessed=False,generator_seen_validation_engines=True,predictor_initialization_seeds=[3],
        rule=c['selection'],candidates=choices,
        decision='Conditional exploratory candidate only;independent fullpipeline/init validation unmeasured' if selected!=1 else 'No elevated weight passed frozen rule;retain1without expanding search')
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    # Visualize ALL five generationseeds and all life stages.
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for weight,color in [(1,'#377eb8'),(2,'#ff7f00'),(4,'#984ea3')]:
        v=val[val.weight==weight].set_index('bin').reindex(BINS[1:])
        axes[0].plot(range(5),v.rmse_engine,marker='o',color=color,label=f'Weight{weight}')
    axes[0].set_xticks(range(5),BINS[1:]);axes[0].set(ylabel='Realvalidation engine-balanced RMSE',xlabel='True cappedRUL',title='All5generationseeds pooled in MSE')
    axes[0].legend();axes[0].grid(alpha=.2)
    for weight,color in [(2,'#ff7f00'),(4,'#984ea3')]:
        for stage,marker,offset in [('all','o',-.12),('31-60','s',.12)]:
            v=pair[(pair.weight==weight)&(pair.bin==stage)].set_index('generation_seed').reindex(c['generation_seeds'])
            axes[1].plot(np.arange(5)+offset,v.percent_rmse_change,marker=marker,color=color,ls='-' if stage=='all' else '--',label=f'Weight{weight} | {stage}')
    axes[1].axhline(0,color='black',ls=':',lw=1);axes[1].set_xticks(range(5),c['generation_seeds'])
    axes[1].set(ylabel='Paired realvalidation RMSE change (%)',xlabel='Generationseed',title='Negative = improvement versusweight1')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.suptitle('Exploratory FD00331–60lossweight screen | samefinal1925updates,init and order\nRealvalidation30engines;fixedgenerator DID see these engines;officialtest unused')
    fig.tight_layout(rect=(0,0,1,.88))
    for ext in ['png','pdf']:fig.savefig(OUT/f'validation_tradeoffs.{ext}',dpi=180)
    plt.close(fig)
    # Meaningful loss boundary check, independent small arrays, no modeltraining.
    truth=torch.tensor([30.,31.,60.,61.]);target=torch.tensor([0.,0.,0.,0.],requires_grad=True)
    tree=ast.parse((ROOT/'scripts/run_fd003_rul_weight_screen.py').read_text())
    expressions=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='w' for t in n.targets)
        and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='where']
    assert len(expressions)==1
    expression=compile(ast.Expression(expressions[0]),'<frozen-training-weight-expression>','eval')
    for weight in [2,4]:
        normalizer=c['normalizers'][str(weight)]
        observed_weight=eval(expression,{'torch':torch,'float':float},{'batch_y':truth,'weight':weight})
        assert torch.equal(observed_weight,torch.tensor([1.,weight,weight,1.]))
        loss=(observed_weight*(target-truth).square()).mean()/normalizer
        grad=torch.autograd.grad(loss,target)[0]
        assert torch.allclose(grad,2*(target-truth)/4*torch.tensor([1.,weight,weight,1.])/normalizer)
    audit=dict(status='passed_cuda_replay_with_cpu_numerical_note',n_models=15,n_epochs_each=35,n_updates_each=1925,sample_visits_each=490490,
        same_initial_parameter_hash=True,same_consumed_sample_order_hash=True,source_data_weights_hashes_verified=True,
        real_validation_engines=30,validation_windows=5134,n_per_engine_rows_recomputed=len(engines),
        cpu_replay_all15_models=True,cpu_absolute_1e3_gate_pass=False,
        max_cpu_gpu_prediction_difference=max(r['cpu_replay_max_difference'] for r in checks),
        cuda_replay_all15_models=True,max_cuda_replay_difference=max(r['cuda_replay_max_difference'] for r in checks),
        cpu_gpu_selection_identical=True,
        inclusive31_60_loss_boundary_checked=True,no_official_test_access=True,generator_exposure_disclosed=True)
    (OUT/'closeout_audit.json').write_text(json.dumps(audit,indent=2))
    mainrows='\n'.join('| '+stage+' | '+' | '.join(f'{val[(val.weight==w)&(val.bin==stage)].rmse_engine.iloc[0]:.3f}' for w in [1,2,4])+' |' for stage in BINS)
    pairrows='\n'.join(f'| {r.generation_seed} | {r.weight} | {r.bin} | {r.baseline_rmse_engine:.3f} | {r.candidate_rmse_engine:.3f} | {r.percent_rmse_change:+.2f}% |' for r in pair[pair.bin.isin(['all','31-60'])].itertuples())
    choicerows='\n'.join(f'| {r.weight} | {r.overall_improves} | {r.target_improves} | {r.target_improved_generation_seeds}/5 | {r.max_other_bin_rmse_change_percent:+.2f}% | {r.eligible} |' for r in choice.itertuples())
    report=f'''# FD003固定生成器的31–60损失加权筛查（探索性EXT-002）

Run ID：`{RUN.name}`；scope：extension:EXT-002/supporting；日期2026-10-27。用户授权固定生成器与生成样本，仅提高31–60在预测器损失中的权重，并明确要求内部发动机验证选择及探索性标注。

## 当前结论

**[Observed]** 15/15训练完成：5生成种子×权重1/2/4，初始化seed3、最终35epoch、1925更新、490490样本访问、实际样本顺序完全一致。冻结内部验证规则选出的权重为**{selected}**。{'这只保留原1倍控制；2与4均未通过全部筛查条件，不追加权重搜索。' if selected==1 else '这是条件性的探索候选，不修改全局默认，也不表示独立生成流程验证通过。'}

**[Observed]** 4倍目标段RMSE25.324→24.177（−4.53%，4/5生成种子改善），但91–124段14.646→16.218（+10.73%）、125段16.909→18.675（+10.45%），整体19.067→19.032（仅−0.18%，只有2/5生成种子整体改善）。2倍目标段25.324→25.361（+0.15%），整体19.067→19.515（+2.35%）。区域间代价不支持采用本次2/4倍加权。

**[Hypothesis]** 提高31–60相对权重确实能改变该段拟合和迁移误差，但当前筛查没有获得兼顾其他区域的收益，不能由此认定权重不足是唯一原因。保留1倍、停止本权重网格；不因为4倍在目标段改善就忽略高RUL退化或按官方测试结果继续挑权重。

**[Observed]** 31–60范围受到既往官方测试结果启发，本实验是探索性。用户已确认保留原生成器，因此原生成器与MinMax已见过所有100台真实训练发动机；本次30台真实验证发动机只是下游选择集，不是整条流程独立验证。原官方测试文件、标签及预测结果本轮未加载，不报告新测试结果。

## 配置与公平对照

**[Code]** 原LSTM14→28循环隐藏、48步展开→64ReLU→1，Adam.002、原15/25零基epoch十倍衰减、batch256。原生成数据20,020窗口，固定synthetictrain14,014/dev6,006不变；未增删、替换、重采样或修改标签。31–60占train14.97%。损失为`mean(weight*SE)/full_train_mean_weight`，使全train平均权重为1；1/2/4对应归一化常数1/1.149707/1.449122。

**[Observed]** 每组取最终第35轮，确保实际检查点更新数也完全相同。新的1倍对照在同一协议下重跑；历史headline使用bestdev/提前停止，不能直接拿历史数值与本实验相减。syntheticdev每轮只记诊断，不用于选轮；真实内部验证不参与梯度、earlystop或epoch选择。

**[Observed]** 原真实train内部以seed20261027固定30发动机，5134个48步窗口，包含31–60的900窗口。125段仅20台发动机，其余各段30台。完整发动机列表见配置；剩余70台不用于本次权重选择。生成窗口没有可靠发动机归属，未按数组位置伪造发动机隔离。

## 真实内部验证整体与分段

**[Observed]** 每个发动机先计算本段窗口MSE，在发动机间等权平均，再在5个生成种子间平均MSE后开方。窗口重叠不作为独立重复。此处是原训练发动机全寿命窗口验证，与官方每台末端窗口测试RMSE口径不同，不能直接比较数值。

| RUL段 | 权重1 | 权重2 | 权重4 |
|---|---:|---:|---:|
{mainrows}

**[Observed]** 原生窗口加权结果和synthetictrain/dev分段指标保存在`summary.csv`，每生成种子/权重的原始指标在`segments.csv`。每台发动机逐段MSE保存在`validation_engines.csv`。

## 每个生成种子的配对变化（负值改善）

| 生成种子 | 权重 | 段 | 1倍RMSE | 加权RMSE | 相对变化 |
|---|---:|---|---:|---:|---:|
{pairrows}

## 事先冻结的选择规则

**[Code]** 候选需同时满足：整体发动机等权RMSE下降、31–60下降、目标段至少3/5生成种子改善、其他每段平均RMSE退化不超过5%。5%仅为事前实用筛查容忍值，不是显著性阈值。通过者选整体MSE最低者，精确同分取较小权重；没有通过则保留1。不得按各生成种子各自挑最好的权重。

| 权重 | 整体改善 | 31–60改善 | 改善生成种子 | 其他段最大变化 | 通过全部条件 |
|---|---|---|---|---:|---|
{choicerows}

**[Observed]** 选择结果仅从上述内部验证指标计算。真实验证集同时用于挑选权重，因此选中候选的验证收益有选择乐观性；单一初始化、单一30发动机划分不构成稳定采用证据。固定生成器见过验证发动机的限制没有被权重对照消除。

## 审计与停止

**[Observed]** 首次CPU独立复算的绝对逐预测1e-3阈值未通过，保留此记录；全15模型CPU/GPU最大逐预测差{audit['max_cpu_gpu_prediction_difference']:.5f}周期。随后使用原CUDA后端复查15个最终权重，保存预测逐值一致（最大差{audit['max_cuda_replay_difference']:.3g}）。CPU各验证段RMSE与原GPU值最大差{numerical['max_cpu_realvalidation_rmse_difference']:.5g}，全部候选的资格与选中权重一致。这里只确认设备/后端数值路径有差异，不唯一归因于某个库或精度开关，没有放宽CPU阈值假称通过，也没有改变训练数据、模型或主GPU指标。

**[Observed]** 所有逐发动机段MSE从保存预测独立复算。15模型初始化和消费的样本顺序SHA均相同，每个模型35轮1925更新，源码/数据/split/计划/权重哈希核对通过。31与60包含、30与61不加权的损失边界已核对。

**[Hypothesis]** 本实验只回答在固定生成分布、一个初始化与本验证集下，加大31–60相对损失能否带来收益及区域代价；不能把结果唯一归因于生成覆盖、信号偏移或优化器，也不能形成原型机制结论。若后续报告官方测试，必须连同“31–60由官方测试分析启发、权重为探索性选择、生成器已见验证发动机”一并注明，不能称为未触碰测试的确认性实验。

**[Observed]** 15次预算用完即停，无额外权重、种子、生成器、架构、原型、真实锚点或测试实验。全局默认不变，零标签路线仍暂停，EXT-001保持冻结。Strictmain/A/B历史结果不变，C延后。

**[Observed]** 五段式：本筛查内部验证结果已测→论文无该权重实验→论文数值差距不适用→有5生成种子配对、无初始化/完整流程独立验证→不改变严格复现结论。Scope `scope-20261027-fd003-rul-weight-screen`。

## 产物

- `results/exploratory/fd003_rul_weight_screen_20261027/summary.csv`、`segments.csv`：整体、各段、各数据域结果。
- 同目录`paired_generation_deltas.csv`、`weight_selection.csv`、`selection.json`：5生成种子配对与冻结选择审计。
- 同目录`validation_engines.csv`、`predictions/*.npz`：逐发动机、逐窗口输出。
- 同目录`training_audit.csv`、`history.csv`、`checkpoint_replay_checks.csv`、`closeout_audit.json`：初始化/顺序/步数、525轮训练记录和CPU复算。
- 同目录`cpu_numerical_replay.csv`、`cuda_numerical_replay.csv`、`replay/*.npy`、`numerical_selection_sensitivity.json`：保留CPU阈值失败、严格CUDA重放与选择稳定性证据。
- 同目录`validation_tradeoffs.png/pdf`：所有寿命段及5生成种子收益/代价。
- `runs/{RUN.name}/config.json`、`frozen_plan.npz`、`real_validation.npz`、`run-manifest.json`：执行前冻结配置、发动机划分和样本顺序及哈希。
- `checkpoints/evaluators/fd003_rul_weight_screen_20261027/`：15个最终权重；协议`docs/operations/FD003_RUL_WEIGHT_SCREEN.md`。
'''
    REPORT.write_text(report)
    m.update(selection=selection,analysis_script_sha256=sha(Path(__file__)),numerical_replay_script_sha256=sha(ROOT/'analysis/diagnostics/replay_fd003_rul_weight_screen.py'),
        report=str(REPORT.relative_to(ROOT)),report_sha256=sha(REPORT),closeout_audit=audit,numerical_audit=numerical,
        outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file()})
    mp.write_text(json.dumps(m,indent=2))
    print(val[['weight','bin','rmse_engine','rmse_window']].round(4).to_string(index=False))
    print(choice.to_string(index=False));print('SELECTED',selected);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
