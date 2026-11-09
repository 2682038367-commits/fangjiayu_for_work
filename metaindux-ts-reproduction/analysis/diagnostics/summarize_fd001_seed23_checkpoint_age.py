"""Functional rerun reproduction gate and conditional15/64/70 quality comparison."""
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
RUN=ROOT/'runs/diagnostic-fd001-seed23-checkpoint-age-20261030'
OUT=ROOT/'results/diagnostics/fd001_seed23_checkpoint_age_20261030'
ANCHOR=ROOT/'results/diagnostics/fd001_sampling_condition_dropout_paired_20261030'
REPORT=ROOT/'docs/conclusions/final/fd001_seed23_checkpoint_age_20261030_report.md'
EVALUATORS=[3,13,23,33,43]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def original_anchor():
    ds=pd.read_csv(ANCHOR/'retain_condition/evaluation_checkpoint23_sample103.csv').sort_values('evaluator_seed')
    metrics=pd.read_csv(ANCHOR/'condition_checkpoint23_sample103.csv')
    metrics=metrics[metrics.arm=='retain_condition'].copy()
    return ds,metrics


def epoch15_gate(config):
    old_ds,old_inv=original_anchor()
    new_ds=pd.read_csv(OUT/'epoch15/retain_condition/evaluation_checkpoint23_sample103.csv').sort_values('evaluator_seed')
    new_inv=pd.read_csv(OUT/'epoch15/condition_metrics.csv')
    old_proxy=old_inv[(old_inv.reference=='dev_matched')&(old_inv.bin=='all')]
    new_proxy=new_inv[(new_inv.reference=='dev_matched')&(new_inv.bin=='all')]
    assert list(old_ds.evaluator_seed)==list(new_ds.evaluator_seed)==EVALUATORS
    old_score,new_score=float(old_ds.discriminative_score.mean()),float(new_ds.discriminative_score.mean())
    old_rmse,new_rmse=float(np.sqrt(old_proxy.mse.mean())),float(np.sqrt(new_proxy.mse.mean()))
    rules=config['reproduction']
    result=dict(original_mean_ds=old_score,rerun_mean_ds=new_score,mean_ds_change=new_score-old_score,
                ds_within_tolerance=abs(new_score-old_score)<=rules['max_abs_mean_ds_difference'],
                failure_category_unchanged=bool((old_score>.3)==(new_score>.3)),
                original_dev_inverse_rmse=old_rmse,rerun_dev_inverse_rmse=new_rmse,
                inverse_relative_change=(new_rmse-old_rmse)/old_rmse,
                inverse_within_tolerance=abs(new_rmse-old_rmse)/old_rmse<=rules['max_relative_dev_inverse_rmse_difference'],
                rules=rules,limitation='Practical functional reproduction only; no equivalence test or proof of missing historical late weights')
    result['passed']=all(result[k] for k in ['ds_within_tolerance','failure_category_unchanged','inverse_within_tolerance'])
    return result


def main(reproduction=False):
    manifest=json.loads((RUN/'run-manifest.json').read_text())
    for name,h in manifest['inputs_sha256'].items():assert sha(ROOT/name)==h,name
    config=json.loads((RUN/'config.json').read_text())
    train=json.loads((OUT/'training_audit.json').read_text())
    assert train['actual_optimizer_updates']==4410 and train['actual_completed_epochs']==70
    history=pd.read_csv(OUT/'training_loss.csv')
    old_history=pd.read_csv(ROOT/'logs/FD001_w48_seed23_training_loss.csv')
    assert np.array_equal(history.epoch,np.arange(70))
    assert np.array_equal(history.learning_rate,old_history.learning_rate)
    assert all(s['snapshot_preserved_rng'] for s in train['snapshots'])
    with np.load(OUT/'training_epoch_orders.npz') as z:
        orders=z['orders']
        assert orders.shape==(70,15931)
        assert np.array_equal(np.sort(orders,axis=1),np.broadcast_to(np.arange(15931),orders.shape))
    assert hashlib.sha256(orders.tobytes()).hexdigest()==train['data_order_sha256']
    gate=epoch15_gate(config)
    if reproduction:
        assert not (OUT/'epoch15_reproduction.json').exists()
        gate['created_at']=now()
        (OUT/'epoch15_reproduction.json').write_text(json.dumps(gate,indent=2)+'\n')
        print(json.dumps(gate,indent=2));return
    assert manifest['status']=='computed_pending_summary'
    assert sha(OUT/'epoch15_reproduction.json')==manifest['reproduction_decision_sha256']
    old_gate=json.loads((OUT/'epoch15_reproduction.json').read_text())
    assert {k:v for k,v in old_gate.items() if k!='created_at'}==gate
    epochs=manifest['completed_epochs_evaluated']
    assert epochs==([15,64,70] if gate['passed'] else [15])
    assert manifest['training_runs_completed']==1 and manifest['completed_sampling']==len(epochs)
    assert manifest['completed_discriminator_evaluations']==5*len(epochs)
    old_ds,old_proxy=original_anchor()
    ds_frames=[old_ds.assign(model='original_epoch15',epoch_one_based=15)]
    proxy_frames=[old_proxy.assign(model='original_epoch15',epoch_one_based=15)]
    ds_rows=[];audits=[]
    new15=pd.read_csv(OUT/'epoch15/retain_condition/evaluation_checkpoint23_sample103.csv').sort_values('evaluator_seed')
    with np.load(RUN/'frozen_inputs.npz') as z:labels=z['label'].ravel()
    anchor_noise=json.loads((ANCHOR/'retain_condition/sample_checkpoint23_seed103.json').read_text())['noise_audit']
    for epoch in epochs:
        ds=pd.read_csv(OUT/f'epoch{epoch}/retain_condition/evaluation_checkpoint23_sample103.csv').sort_values('evaluator_seed')
        assert list(ds.evaluator_seed)==EVALUATORS
        assert np.max(abs(ds.discriminative_score-abs(.5-(ds.fake_accuracy+ds.real_accuracy)/2)))<1e-12
        difference=ds.discriminative_score.to_numpy()-new15.discriminative_score.to_numpy()
        ds_frames.append(ds.assign(model=f'rerun_epoch{epoch}',epoch_one_based=epoch))
        ds_rows.append(dict(epoch_one_based=epoch,mean_ds=float(ds.discriminative_score.mean()),
            evaluator_sd=float(ds.discriminative_score.std(ddof=1)),ds_min=float(ds.discriminative_score.min()),
            ds_max=float(ds.discriminative_score.max()),change_from_rerun15=float(difference.mean()),
            n_paired_ds_decreases=int((difference<0).sum()),failed=bool(ds.discriminative_score.mean()>.3)))
        proxy=pd.read_csv(OUT/f'epoch{epoch}/condition_metrics.csv')
        proxy_frames.append(proxy.assign(model=f'rerun_epoch{epoch}'))
        record=json.loads((OUT/f'epoch{epoch}/retain_condition/sample_checkpoint23_seed103.json').read_text())
        for key in ['legacy_schedule_fingerprint','start_cuda_rng_sha256','final_cuda_rng_sha256','normal_scalar_count']:
            assert record['noise_audit'][key]==anchor_noise[key]
        assert record['noise_audit']['conditional_dropped_windows']==0
        assert record['noise_audit']['normal_tensors_exactly_matched']==63000
        assert sha(ROOT/record['artifact_path'])==record['artifact_sha256']
        with np.load(ROOT/record['artifact_path']) as z:
            assert z['data'].shape==(15931,48,14) and np.isfinite(z['data']).all()
            assert np.array_equal(z['label'].ravel(),labels)
        with np.load(OUT/f'epoch{epoch}/condition_predictions.npz') as z:
            assert z['prediction'].shape==(5,15931) and np.isfinite(z['prediction']).all()
            assert np.array_equal(z['label'],labels)
            for ref in ['all_windows','dev_matched','test_matched']:
                rows=proxy[(proxy.reference==ref)&(proxy.bin=='all')].sort_values('evaluator_seed')
                actual=np.mean((z['prediction'][:,z[ref]]-labels[z[ref]][None,:])**2,axis=1)
                assert np.max(abs(actual-rows.mse.to_numpy()))<1e-8
        audits.append(dict(epoch=epoch,noise_anchor_match=True,data_and_labels_passed=True,score_identity_passed=True,proxy_mse_recomputed=True))
    raw_ds=pd.concat(ds_frames,ignore_index=True)
    raw_proxy=pd.concat(proxy_frames,ignore_index=True)
    raw_ds.to_csv(OUT/'raw_discriminator_comparison.csv',index=False)
    raw_proxy.to_csv(OUT/'raw_condition_comparison.csv',index=False)
    scores=pd.DataFrame(ds_rows);scores.to_csv(OUT/'checkpoint_quality.csv',index=False)
    inverse=[]
    for (model,epoch,reference,bin_name),group in raw_proxy.groupby(['model','epoch_one_based','reference','bin'],sort=True):
        inverse.append(dict(model=model,epoch_one_based=int(epoch),reference=reference,bin=bin_name,
             n_windows=int(group.n_windows.iloc[0]),pooled_rmse=float(np.sqrt(group.mse.mean())),
             bias=float(group.bias.mean()),prediction_mean=float(group.prediction_mean.mean())))
    inverse=pd.DataFrame(inverse);inverse.to_csv(OUT/'condition_comparison.csv',index=False)
    interpretation=dict(functional_epoch15_reproduction=gate['passed'],historical_late_weights_observed=False,
        rerun_best_epoch_one_based=train['best_epoch_zero_based']+1,original_best_epoch_one_based=15,
        rerun_selection_also_chose_epoch15=train['best_epoch_zero_based']==14,
        original_late_model_quality='unobserved; original snapshots missing')
    if gate['passed']:
        rules=config['clear_later_gain']
        interpretation['late_epochs_with_clear_ds_gain']=[int(r.epoch_one_based) for r in scores.itertuples()
            if r.epoch_one_based!=15 and -r.change_from_rerun15>=rules['min_mean_ds_reduction']
            and r.n_paired_ds_decreases>=rules['min_paired_ds_decreases']]
        values=scores.sort_values('epoch_one_based').mean_ds.to_numpy()
        interpretation['ds_monotonically_improved']=bool(np.all(np.diff(values)<0))
        interpretation['final_still_above_ds_point3']=bool(values[-1]>.3)
        interpretation['rerun_rule_directly_discarded_evaluated_better_model']=bool(
            interpretation['rerun_selection_also_chose_epoch15'] and interpretation['late_epochs_with_clear_ds_gain'])
    else:
        interpretation['stopped_before_late_evaluation']=True
    (OUT/'interpretation.json').write_text(json.dumps(interpretation,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    axes[0].plot(history.epoch+1,history.diffusion_loss,label='Rerun seed23')
    axes[0].plot(old_history.epoch+1,old_history.diffusion_loss,alpha=.6,label='Original seed23')
    for e in [15,64,70]:axes[0].axvline(e,color='gray',alpha=.25)
    axes[0].set(xlabel='Training epoch (one-based)',ylabel='Mean training diffusion loss',title='Original settings;70 epochs');axes[0].legend()
    axes[1].errorbar(scores.epoch_one_based,scores.mean_ds,yerr=scores.evaluator_sd,marker='o',capsize=3,label='Rerun: mean ± evaluator SD')
    axes[1].axhline(float(old_ds.discriminative_score.mean()),linestyle=':',color='gray',label='Original epoch15')
    axes[1].axhline(.3,linestyle='--',color='red',alpha=.6)
    axes[1].set(xlabel='Retained checkpoint epoch',ylabel='Discriminative score (lower is better)',ylim=(0,.52),title='Same103noise and always-retained conditions')
    axes[1].set_xticks(epochs);axes[1].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(OUT/f'checkpoint_age_quality.{suffix}',dpi=180)
    plt.close(fig)
    lines=['# FD001 seed23第15/64/70轮模型比较（2026-10-30）','',
       '[Observed] 原训练后期权重未找到，因此**无法直接回溯原来后55轮是否更好**。本次按原设置仅重跑seed23一次70轮，额外保存第15、64、70轮；下列结果属于同设置重跑的轨迹证据，不冒充原始历史模型。','',
       f'[Observed] 第15轮功能性重现是否通过：**{gate["passed"]}**。原平均DS={gate["original_mean_ds"]:.6f}，重跑={gate["rerun_mean_ds"]:.6f}；原dev匹配条件RMSE={gate["original_dev_inverse_rmse"]:.6f}，重跑={gate["rerun_dev_inverse_rmse"]:.6f}。实用容差提前固定为DS绝对差<=.03、>.3类别相同、条件RMSE相对差<=10%；不代表统计等效或逐位训练重现。','',
       f'[Observed] 原规则选中第15轮；本次重跑规则实际选中第{train["best_epoch_zero_based"]+1}轮。后期是否明显更好必须结合这个差异解释。只有重跑规则同样选15且后期明显更好，才能直接说本次重跑的best-loss规则遗漏了已评价的更好模型；若重跑已选其他轮次，不得据此宣称规则在本次必然丢弃后期。','',
       '| 重跑权重轮次 | 平均DS | 5次评价SD | 相对重跑15变化 | 5次中改善数 | 均值>.3 |',
       '|---|---:|---:|---:|---:|---|']
    for r in scores.itertuples():lines.append(f'| {r.epoch_one_based} | {r.mean_ds:.6f} | {r.evaluator_sd:.6f} | {r.change_from_rerun15:+.6f} | {r.n_paired_ds_decreases}/5 | {r.failed} |')
    lines+=['','## 真实预测器条件读出（辅助）','','| 模型 | 匹配集合 | 窗口数 | 条件RMSE | 偏差 |','|---|---|---:|---:|---:|']
    for r in inverse[(inverse.bin=='all')&inverse.reference.isin(['test_matched','dev_matched'])].itertuples():
        lines.append(f'| {r.model} | {r.reference} | {r.n_windows} | {r.pooled_rmse:.6f} | {r.bias:+.6f} |')
    lines+=['','## 审计与解释边界','',
       '[Observed] 原训练函数未改。实际记录70轮/4410次Adam更新，三个保存点为945/4032/4410；逐参数optimizer计数器核对通过，完整数据顺序和原标签置零随机抽样保留。额外序列化前后TorchCPU/CUDA/NumPy RNG状态相同；学习率70轮序列与原记录逐元素一致。原训练条件丢弃.2、数值整批标签置零.1、网络/阈值/优化器/梯度裁剪/训练时长均保留。','',
       '[Observed] 重跑15与原15的权重比较：'+json.dumps(train['original_epoch15_weight_comparison'])+'。保留原cuDNN benchmark与非确定性设置，没有为追求逐位重现而修改算法；原训练数据顺序未保存，不能由同seed单独证明原始逐步路径相同。','',
       '[Observed] 采样全部使用关闭条件丢弃版本，仍消耗原掩码随机数，保留CFG无条件分支。sampling103、15931条件顺序/DDPM1000/CFG3/batch256一致；每批63000初始/逐步正常噪声张量与参考流逐张量核对，各阶段完整RNG指纹与原15保留条件锚点一致。5个判别器种子与固定划分、5个真实预测器及100/3000寿命匹配窗口均未变。','',
       '[Hypothesis] 功能重现通过时，后期DS降低至少.03且>=4/5配对评价改善，可支持这条重跑轨迹的后期模型更好；若后期仍>.3但持续改善，只能为单独训练时长试验提供动机。若后期无改善/更差，第15轮选择不足以解释这条轨迹的差距。两份后期权重不足以证明所有中间轮次或最终收敛趋势。','',
       '[Hypothesis] 条件读出并非真实寿命真值，不能替代真实下游评价；DS与条件误差可能存在取舍。仅一次seed23训练/一次采样种子/固定评价器，只能作诊断，不宣称总体机制或论文结果已修复。原后期权重缺失这一限制不会因功能重现通过而消失。','',
       '[Observed] 当前为检查点年龄诊断 → 论文无同设计目标 → 论文数值差距不适用 → 一次同设训练、5评价种子，未新增种子搜索 → 严格复现结论不变。一次70轮训练预算已用，采样按第15轮重现门槛停止或最多3批/15新DSfits；不自动延长训练、改学习率/阈值/模型或继续其他机制试验。数值标签修复方向仍暂停，EXT-002仍保留weight1。','',
       f'[Observed] Run `{RUN.name}`；快照 `runs/{RUN.name}/run-manifest.json`；协议 `docs/operations/FD001_SEED23_CHECKPOINT_AGE.md`；全部输出 `results/diagnostics/fd001_seed23_checkpoint_age_20261030/`。']
    REPORT.write_text('\n'.join(lines)+'\n')
    audit=dict(status='passed',training_runs=1,training_updates=4410,evaluated_epochs=epochs,
               noise_anchor_matches=True,raw_scores_and_proxy_errors_recomputed=True,gate_recomputed=True,completed_at=now())
    (OUT/'closeout_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    manifest.update(status='complete',completed_at=audit['completed_at'],audit=audit,
       outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [*OUT.rglob('*'),REPORT] if p.is_file()})
    (RUN/'run-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary='; '.join(f'epoch{r.epoch_one_based}DS={r.mean_ds:.6f}' for r in scores.itertuples())
    with (ROOT/'experiment-ledger.md').open('a') as f:
        f.write(f'\n## `{RUN.name}` completed\n\n- [Observed] Complete{now()},oneoriginalsettingseed23rerun70epochs/4410actualupdates,extra15/64/70weightsretained;functional15reproduction={gate["passed"]},evaluatedepochs={epochs};{summary}. Originalhistorical15anchorDS{gate["original_mean_ds"]:.6f};rerunbestselectedepoch{train["best_epoch_zero_based"]+1}.\n- [Observed] Originaltrainingbehavior/LRorderunchanged,snapshotRNGinvariance/actualoptimizercounts/permutationrecorded,allnoise-anchor/source/data/metricsauditspassed. No originalhistorical lateweights;onlyreruntrajectoryevidence,notdirecthistoricalrecovery. Strictstatusunchanged,budgetclosed,nofurthertraining/changes;zero-labelfixpaused,EXT-002weight1.\n- Report `{REPORT.relative_to(ROOT)}`;snapshot `runs/{RUN.name}/run-manifest.json`;outputs `{OUT.relative_to(ROOT)}`.\n')
    state=ROOT/'reproduction-state.md';text=state.read_text()
    entry=f'## Latest checkpoint: 2026-10-30 seed23 checkpoint-age diagnostic complete\n\n- [Observed] `{RUN.name}` completeone70epoch/4410updateseed23rerun;15/64/70weightsretained,15functionalrepro={gate["passed"]},evaluated={epochs};{summary}. Rerunbestepoch{train["best_epoch_zero_based"]+1};originalselected15. Originalhistorical lateweightsmissing,neverclaimtheirqualitydirectlyobserved.\n- [Observed] Originalrecipeunchanged,snapshotRNGinvariance/actualstepcounts/noise-anchoridentity/source/metric audits passed. Noextraseeds/trainingafterbudget;strictresultsunchanged,zero-labelfixpaused,EXT-002weight1.\n- Report `{REPORT.relative_to(ROOT)}`;snapshot `runs/{RUN.name}/run-manifest.json`;outputs `{OUT.relative_to(ROOT)}`.\n\n'
    state.write_text(text.replace('# Persistent reproduction state\n\n','# Persistent reproduction state\n\n'+entry,1))
    print(scores.to_string(index=False));print(json.dumps(interpretation,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reproduction',action='store_true')
    main(parser.parse_args().reproduction)
