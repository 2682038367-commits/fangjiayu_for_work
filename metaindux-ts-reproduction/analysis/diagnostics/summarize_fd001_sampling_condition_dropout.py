"""Predeclared screening gate, paired analysis and automatic diagnostic closeout."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/diagnostic-fd001-sampling-condition-dropout-20261030'
OUT=ROOT/'results/diagnostics/fd001_sampling_condition_dropout_20261030'
BASE=ROOT/'results/diagnostics/fd001_checkpoint_resampling_20261029'
REPORT=ROOT/'docs/conclusions/final/fd001_sampling_condition_dropout_20261030_report.md'
EVALUATORS=[3,13,23,33,43]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def comparisons(cells):
    ds_rows,condition_rows,raw_ds,raw_condition=[],[],[],[]
    for checkpoint,seed in cells:
        control=pd.read_csv(BASE/f'evaluation_checkpoint{checkpoint}_sample{seed}.csv').sort_values('evaluator_seed')
        intervention=pd.read_csv(OUT/f'evaluation_checkpoint{checkpoint}_sample{seed}.csv').sort_values('evaluator_seed')
        for arm,data in [('control',control),('retain_condition',intervention)]:
            assert len(data)==5 and list(data.evaluator_seed)==EVALUATORS
            score=data.discriminative_score
            assert np.isfinite(score).all() and score.between(0,.5).all()
            assert np.max(abs(score-abs(.5-(data.fake_accuracy+data.real_accuracy)/2)))<1e-12
            raw_ds.append(data.assign(arm=arm))
        difference=intervention.discriminative_score.to_numpy()-control.discriminative_score.to_numpy()
        change=float(difference.mean())
        ds_rows.append(dict(checkpoint_seed=checkpoint,sampling_seed=seed,
            control_mean=float(control.discriminative_score.mean()),intervention_mean=float(intervention.discriminative_score.mean()),
            control_evaluator_sd=float(control.discriminative_score.std(ddof=1)),
            intervention_evaluator_sd=float(intervention.discriminative_score.std(ddof=1)),
            mean_change=change,n_paired_changes_same_direction=int((difference*np.sign(change)>0).sum()),
            control_failed=bool(control.discriminative_score.mean()>.3),intervention_failed=bool(intervention.discriminative_score.mean()>.3)))
        data=pd.read_csv(OUT/f'condition_checkpoint{checkpoint}_sample{seed}.csv')
        raw_condition.append(data)
        for (reference,bin_name),group in data.groupby(['reference','bin'],sort=True):
            control=group[group.arm=='control'].sort_values('evaluator_seed')
            intervention=group[group.arm=='retain_condition'].sort_values('evaluator_seed')
            assert list(control.evaluator_seed)==list(intervention.evaluator_seed)==EVALUATORS
            assert np.array_equal(control.n_windows.to_numpy(),intervention.n_windows.to_numpy())
            assert np.array_equal(control.condition_mean.to_numpy(),intervention.condition_mean.to_numpy())
            before=float(np.sqrt(control.mse.mean()));after=float(np.sqrt(intervention.mse.mean()))
            delta=intervention.mse.to_numpy()-control.mse.to_numpy()
            condition_rows.append(dict(checkpoint_seed=checkpoint,sampling_seed=seed,reference=reference,bin=bin_name,
                n_windows=int(control.n_windows.iloc[0]),control_rmse=before,intervention_rmse=after,
                rmse_change=after-before,relative_rmse_change=(after-before)/before,
                control_bias=float(control.bias.mean()),intervention_bias=float(intervention.bias.mean()),
                control_prediction_mean=float(control.prediction_mean.mean()),
                intervention_prediction_mean=float(intervention.prediction_mean.mean()),
                n_paired_changes_same_direction=int((delta*np.sign(after-before)>0).sum())))
    return pd.DataFrame(ds_rows),pd.DataFrame(condition_rows),pd.concat(raw_ds,ignore_index=True),pd.concat(raw_condition,ignore_index=True)


def decision(ds,condition,config):
    gate=config['clear_change'];rows=[]
    for checkpoint in [13,23]:
        d=ds[(ds.checkpoint_seed==checkpoint)&(ds.sampling_seed==103)].iloc[0]
        c=condition[(condition.checkpoint_seed==checkpoint)&(condition.sampling_seed==103)&
                    (condition.reference==gate['inverse_reference'])&(condition.bin=='all')].iloc[0]
        ds_gate=abs(d.mean_change)>=gate['ds_absolute_mean_change'] and d.n_paired_changes_same_direction>=gate['paired_same_direction_at_least']
        inverse_gate=(abs(c.rmse_change)>=gate['inverse_absolute_rmse_change'] and
                      abs(c.relative_rmse_change)>=gate['inverse_relative_rmse_change'] and
                      c.n_paired_changes_same_direction>=gate['paired_same_direction_at_least'])
        rows.append(dict(checkpoint_seed=checkpoint,ds_change=float(d.mean_change),
                         ds_same_direction_count=int(d.n_paired_changes_same_direction),ds_gate=bool(ds_gate),
                         inverse_reference=gate['inverse_reference'],inverse_rmse_change=float(c.rmse_change),
                         inverse_relative_rmse_change=float(c.relative_rmse_change),
                         inverse_same_direction_count=int(c.n_paired_changes_same_direction),inverse_gate=bool(inverse_gate)))
    return dict(confirmation_triggered=any(row['ds_gate'] or row['inverse_gate'] for row in rows),
                gates=config['clear_change'],checkpoint_decisions=rows,
                interpretation='Exploratory practical effect gate, either direction; not statistical significance or adoption rule')


def audit(cells,raw_condition):
    checks=[]
    with np.load(RUN/'frozen_inputs.npz') as z:y=z['label'].ravel()
    for checkpoint,seed in cells:
        record=json.loads((OUT/f'sample_checkpoint{checkpoint}_seed{seed}.json').read_text())
        assert record['noise_audit']['status']=='passed'
        assert record['noise_audit']['normal_tensors_exactly_matched']==63000
        assert record['noise_audit']['conditional_uniform_draws_exactly_matched']==63000
        assert record['noise_audit']['unconditional_calls_retained']==63000
        assert record['noise_audit']['conditional_dropped_windows']==0
        assert record['prefix_replay']['passed']
        path=ROOT/record['artifact_path'];assert sha(path)==record['artifact_sha256']
        with np.load(path) as z:
            assert z['data'].shape==(15931,48,14) and np.isfinite(z['data']).all()
            assert np.array_equal(z['label'].ravel(),y)
            assert hashlib.sha256(z['data'].tobytes()).hexdigest()==record['data_sha256']
        for arm in ['control','retain_condition']:
            with np.load(OUT/f'condition_predictions_checkpoint{checkpoint}_sample{seed}_{arm}.npz') as z:
                assert z['prediction'].shape==(5,15931) and np.isfinite(z['prediction']).all()
                assert np.array_equal(z['label'],y) and list(z['evaluator_seeds'])==EVALUATORS
                rows=raw_condition[(raw_condition.checkpoint_seed==checkpoint)&(raw_condition.sampling_seed==seed)&(raw_condition.arm==arm)]
                bounds={'all':(0,125),'0':(0,0),'1':(1,1),'2-5':(2,5),'6-30':(6,30),
                        '0-30':(0,30),'31-60':(31,60),'61-90':(61,90),'91-124':(91,124),'125':(125,125)}
                for row in rows.itertuples():
                    lo,hi=bounds[row.bin]
                    indices=z[row.reference]
                    indices=indices[(y[indices]>=lo)&(y[indices]<=hi)]
                    error=z['prediction'][EVALUATORS.index(row.evaluator_seed),indices]-y[indices]
                    assert len(indices)==row.n_windows
                    assert abs(float(np.mean(error**2))-row.mse)<1e-8
        checks.append(dict(checkpoint_seed=checkpoint,sampling_seed=seed,
                           prefix_bit_identical=record['prefix_replay']['bit_identical'],
                           prefix_max_abs_difference=record['prefix_replay']['max_absolute_difference'],
                           matched_noise_tensors=63000,noise_audit='passed',proxy_mse_recomputed=True))
    pd.DataFrame(checks).to_csv(OUT/'closeout_checks.csv',index=False)
    return checks


def main(screen=False):
    manifest=json.loads((RUN/'run-manifest.json').read_text())
    for name,expected in manifest['inputs_sha256'].items():assert sha(ROOT/name)==expected,name
    config=json.loads((RUN/'config.json').read_text())
    cells=[[13,103],[23,103]] if screen else manifest['completed_cells']
    ds,condition,raw_ds,raw_condition=comparisons(cells)
    selected=decision(ds,condition,config)
    if screen:
        assert not (OUT/'screening_decision.json').exists()
        selected['created_at']=now()
        (OUT/'screening_decision.json').write_text(json.dumps(selected,indent=2)+'\n')
        ds.to_csv(OUT/'screening_ds_comparison.csv',index=False)
        condition.to_csv(OUT/'screening_condition_comparison.csv',index=False)
        print(json.dumps(selected,indent=2));return
    assert manifest['status']=='computed_pending_summary'
    assert sha(OUT/'screening_decision.json')==manifest['screening_decision_sha256']
    previous=json.loads((OUT/'screening_decision.json').read_text())
    assert {k:v for k,v in previous.items() if k!='created_at'}==selected
    assert selected['confirmation_triggered']==manifest['confirmation_triggered']
    assert len(cells)==(6 if selected['confirmation_triggered'] else 2)
    assert len(raw_ds)==len(cells)*10 and manifest['completed_discriminator_evaluations']==len(cells)*5
    checks=audit(cells,raw_condition)
    ds.to_csv(OUT/'ds_comparison.csv',index=False)
    condition.to_csv(OUT/'condition_comparison.csv',index=False)
    raw_ds.to_csv(OUT/'raw_ds_comparisons.csv',index=False)
    raw_condition.to_csv(OUT/'raw_condition_metrics.csv',index=False)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for col,checkpoint in enumerate([13,23]):
        scores=ds[ds.checkpoint_seed==checkpoint].sort_values('sampling_seed')
        axes[0,col].plot(scores.sampling_seed,scores.control_mean,'o-',label='Legacy sampling')
        axes[0,col].plot(scores.sampling_seed,scores.intervention_mean,'s-',label='Always retain condition')
        axes[0,col].axhline(.3,color='gray',linestyle='--')
        axes[0,col].set(title=f'Fixed checkpoint {checkpoint}: discriminator',ylabel='DS (lower is better)',ylim=(0,.52))
        proxy=condition[(condition.checkpoint_seed==checkpoint)&(condition.reference=='dev_matched')&(condition.bin=='all')].sort_values('sampling_seed')
        axes[1,col].plot(proxy.sampling_seed,proxy.control_rmse,'o-',label='Legacy sampling')
        axes[1,col].plot(proxy.sampling_seed,proxy.intervention_rmse,'s-',label='Always retain condition')
        axes[1,col].set(title='Frozen real-predictor inverse readout, 3000 matched windows',ylabel='Condition-readout RMSE (cycles)',xlabel='Sampling seed')
        for row in [0,1]:axes[row,col].legend();axes[row,col].set_xticks(scores.sampling_seed)
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(OUT/f'sampling_dropout_comparison.{suffix}',dpi=180)
    plt.close(fig)
    lines=['# FD001仅关闭采样条件随机丢弃（2026-10-30）','',
           '[Observed] 固定seed13第64轮、seed23第15轮权重；只让采样CFG条件分支始终保留指定RUL嵌入，显式无条件分支用途与CFG3保持不变。训练行为、权重、标签数量/顺序、原DDPM1000及判别配置均未改。', '',
           '[Code] 条件掩码仍调用原随机函数以消耗相同均匀随机数，再将掩码强制为全保留；无条件掩码仍全丢弃。避免删除随机抽样导致后续扩散噪声错位。', '',
           f'[Observed] 103筛查是否触发113/123确认：**{selected["confirmation_triggered"]}**；实际完成{len(cells)}批实验采样/{len(cells)*5}次新判别评价，对照复用既有对应数据及{len(cells)*5}次判别评价。筛查阈值在结果前冻结，详见screening_decision.json；改善或恶化均触发，不是显著性或采用规则。', '',
           '| 固定检查点 | 采样种子 | 原采样DS | 保留条件DS | 变化（实验−对照） | 同向次数 |',
           '|---|---|---:|---:|---:|---:|']
    for row in ds.itertuples():
        lines.append(f'| {row.checkpoint_seed} | {row.sampling_seed} | {row.control_mean:.6f} | {row.intervention_mean:.6f} | {row.mean_change:+.6f} | {row.n_paired_changes_same_direction}/5 |')
    lines+=['','## 真实预测器读取寿命条件的辅助检查','',
            '[Observed] 使用原5个真实数据训练预测器，固定同一checkpoint对应的历史寿命匹配索引：test_matched100及dev_matched3000；另保留所有15931窗口与RUL0/1/2–5/6–30及各阶段。对照和实验各5个模型的平方误差先平均再开根号，输出不截断。完整预测、偏差及分段见NPZ/CSV。','',
            '| 检查点 | 采样种子 | 匹配窗口集合 | 对照条件RMSE | 实验条件RMSE | 变化 |',
            '|---|---|---|---:|---:|---:|']
    for row in condition[(condition.bin=='all')&condition.reference.isin(['test_matched','dev_matched'])].itertuples():
        lines.append(f'| {row.checkpoint_seed} | {row.sampling_seed} | {row.reference} | {row.control_rmse:.6f} | {row.intervention_rmse:.6f} | {row.rmse_change:+.6f} |')
    lines+=['','## 配对与边界','',
            f'[Observed] {len(checks)}次控制首256窗口重放全部在maxabs1e-6内；其中{sum(r["prefix_bit_identical"] for r in checks)}次逐元素完全相同。每批63000个初始/逐步标准正态张量及63000次原条件均匀随机抽样均与独立legacy-schedule CUDA生成器逐次完全相同，前后状态匹配；每批显式无条件调用63000次保持不变。条件分支实际丢弃窗口数为0。所有权重/源代码/输入哈希未变，标签逐元素一致。','',
            '[Observed] 旧控制数据没有保存完整噪声带。本次与它的联系是冻结源代码/权重/RNG起点、首分块重放和原调用顺序的噪声流重建；没有把旧噪声带冒充已记录。新增干预噪声与参考流的逐张量相等核验明确执行。','',
            '[Hypothesis] 此处仅识别在选定固定权重上改变采样条件保留的影响，不能解释此前所有检查点差异或推广到FD002–4；共享评价种子/相同匹配窗口相关，条件读出预测器也并非寿命真值。条件指标改善不能替代生成数据训练预测器在真实测试集上的下游评价。','',
            '[Observed] 先103筛查后有条件确认属于探索性分析；不得把触发后的选择性确认当作无选择的总体稳定性估计。没有重新训练生成器、真实预测器或下游预测器，没有修改严格复现默认采样行为。','',
            '[Observed] 当前诊断结果 → 论文无同设计目标 → 论文数值差距不适用 → 配对5评价种子及按门槛最多3采样种子，非正式显著性 → 严格复现结论不变。预算完成后停止，不自动追加实验，训练数值置零方向仍暂停、EXT-002仍保留weight1。','',
            f'[Observed] Run `{RUN.name}`；快照 `runs/{RUN.name}/run-manifest.json`；协议 `docs/operations/FD001_SAMPLING_CONDITION_DROPOUT.md`；全部输出 `results/diagnostics/fd001_sampling_condition_dropout_20261030/`。']
    REPORT.write_text('\n'.join(lines)+'\n')
    audit_result=dict(status='passed',new_batches=len(cells),new_discriminator_fits=len(cells)*5,
                      cached_control_batches=len(cells),noise_alignment_passed=True,
                      prefix_replays_passed=True,proxy_metrics_independently_recomputed=True,
                      screening_gate_recomputed=True,completed_at=now())
    (OUT/'closeout_audit.json').write_text(json.dumps(audit_result,indent=2)+'\n')
    manifest.update(status='complete',completed_at=audit_result['completed_at'],audit=audit_result,
                    outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [*OUT.rglob('*'),REPORT] if p.is_file()})
    (RUN/'run-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary='; '.join(f'ckpt{int(c)} meanDSdelta{group.mean_change.mean():+.6f}' for c,group in ds.groupby('checkpoint_seed'))
    with (ROOT/'experiment-ledger.md').open('a') as handle:
        handle.write(f'\n## `{RUN.name}` completed\n\n- [Observed] Complete {now()},new samples={len(cells)},new DSfits={len(cells)*5};controls cached,weights/training unchanged. Screen confirmation={selected["confirmation_triggered"]};{summary}. See report for pairedDS and inverseproxytest/dev/fullRUL results.\n- [Observed] Allnormal tensors/legacyuniform draws exactlymatched independentreferencegenerator;prefixreplayspassed,fullmetric/source/hash/label audits passed. Oldcontrol noise tape notstored;link is frozen provenance+prefix+legacy RNGschedule. No downstreamutilityclaim,strictresults/defaults unchanged;stopbudget,zero-labelpaused,EXT-002weight1.\n- Report `{REPORT.relative_to(ROOT)}`;snapshot `runs/{RUN.name}/run-manifest.json`;outputs `{OUT.relative_to(ROOT)}`.\n')
    state=ROOT/'reproduction-state.md';text=state.read_text()
    entry=f'## Latest checkpoint: 2026-10-30 sampling condition-dropout diagnostic complete\n\n- [Observed] `{RUN.name}` diagnostic/supporting complete{len(cells)}newsamples/{len(cells)*5}newDSfits;cachedmatchedcontrols.103gate triggeredconfirmation={selected["confirmation_triggered"]};{summary}. Fixedweights,onlyeffectiveconditionalmaskretentionchanged;unconditionalCFGpreserved.\n- [Observed] Perdrawexactnoise/referenceRNGchecks andprefixreplay/labels/source/metric audits passed. Oldcontrolnoise tapenotstored,provenancelimitexplicit. Inverseproxyreadoutnotphysicaltruth/downstreamutility. Budgetclosed,noautomaticfollowups;strictdefaultsunchanged,zero-labelpaused,EXT-002weight1.\n- Report `{REPORT.relative_to(ROOT)}`;snapshot `runs/{RUN.name}/run-manifest.json`;outputs `{OUT.relative_to(ROOT)}`.\n\n'
    state.write_text(text.replace('# Persistent reproduction state\n\n','# Persistent reproduction state\n\n'+entry,1))
    print(ds.to_string(index=False));print(json.dumps(selected,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--screen',action='store_true')
    main(parser.parse_args().screen)
