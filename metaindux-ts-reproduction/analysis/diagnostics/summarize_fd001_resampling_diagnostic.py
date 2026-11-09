"""Audit and summarize the predeclared 2 x 3 x 5 FD001 resampling diagnostic."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/diagnostic-fd001-checkpoint-resampling-20261029'
OUT = ROOT / 'results/diagnostics/fd001_checkpoint_resampling_20261029'
REPORT = ROOT / 'docs/conclusions/final/fd001_checkpoint_resampling_20261029_report.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((RUN / 'run-manifest.json').read_text())
    assert manifest['status'] == 'computed_pending_summary'
    for name, expected in manifest['inputs_sha256'].items():
        assert sha(ROOT / name) == expected, name
    config = json.loads((RUN / 'config.json').read_text())
    with np.load(RUN / 'frozen_inputs.npz') as archive:
        labels = archive['label'].copy()
        real_shape = archive['real'].shape
    audit = json.loads((RUN / 'checkpoint_audit.json').read_text())
    historical = {row['checkpoint_seed']: row['original_mean_ds'] for row in audit['checkpoints']}
    sample_rows, frames, rng = [], [], {}
    for sampling in config['sampling_seeds']:
        for checkpoint in config['checkpoint_training_seeds']:
            record = json.loads((OUT / f'sample_checkpoint{checkpoint}_seed{sampling}.json').read_text())
            artifact = ROOT / record['artifact_path']
            assert sha(artifact) == record['artifact_sha256']
            with np.load(artifact) as archive:
                assert archive['data'].shape == real_shape
                assert np.isfinite(archive['data']).all()
                assert np.array_equal(archive['label'], labels)
                assert hashlib.sha256(archive['data'].tobytes()).hexdigest() == record['data_sha256']
            rng[checkpoint, sampling] = record['pre_model_construction_rng_sha256']
            rows = pd.read_csv(OUT / f'evaluation_checkpoint{checkpoint}_sample{sampling}.csv')
            assert len(rows) == 5 and sorted(rows.evaluator_seed.tolist()) == config['discriminator_seeds']
            assert rows.checkpoint_seed.eq(checkpoint).all() and rows.sampling_seed.eq(sampling).all()
            assert np.isfinite(rows[['discriminative_score', 'fake_accuracy', 'real_accuracy']]).all().all()
            assert rows.discriminative_score.between(0, .5).all()
            identity = abs(.5 - (rows.fake_accuracy + rows.real_accuracy) / 2)
            assert np.max(abs(identity - rows.discriminative_score)) < 1e-12
            scores = rows.discriminative_score
            sample_rows.append(dict(checkpoint_seed=checkpoint, sampling_seed=sampling,
                                    ds_mean=float(scores.mean()), evaluator_sd=float(scores.std(ddof=1)),
                                    ds_min=float(scores.min()), ds_max=float(scores.max()),
                                    evaluator_range=float(scores.max()-scores.min()),
                                    failed=bool(scores.mean() > .3),
                                    n_evaluators_above_threshold=int((scores > .3).sum()),
                                    original_mean_ds=historical[checkpoint],
                                    new_minus_original_mean=float(scores.mean()-historical[checkpoint]),
                                    sampling_seconds=record['elapsed_seconds'],
                                    data_sha256=record['data_sha256']))
            frames.append(rows)
    for sampling in config['sampling_seeds']:
        assert rng[13, sampling] == rng[23, sampling], 'Matched sampling RNG state differs'
    raw = pd.concat(frames, ignore_index=True)
    assert len(raw) == 30 and not raw.duplicated(['checkpoint_seed', 'sampling_seed', 'evaluator_seed']).any()
    batches = pd.DataFrame(sample_rows).sort_values(['checkpoint_seed', 'sampling_seed'])
    assert batches.data_sha256.nunique() == 6
    checkpoint_rows = []
    for checkpoint, group in batches.groupby('checkpoint_seed'):
        checkpoint_rows.append(dict(checkpoint_seed=int(checkpoint), original_mean_ds=historical[checkpoint],
                                    resampling_mean_ds=float(group.ds_mean.mean()),
                                    sampling_mean_sd=float(group.ds_mean.std(ddof=1)),
                                    sampling_mean_min=float(group.ds_mean.min()), sampling_mean_max=float(group.ds_mean.max()),
                                    sampling_mean_range=float(group.ds_mean.max()-group.ds_mean.min()),
                                    mean_evaluator_sd=float(group.evaluator_sd.mean()),
                                    max_evaluator_range=float(group.evaluator_range.max()),
                                    failed_batches=int(group.failed.sum()), sampled_batches=len(group)))
    checkpoints = pd.DataFrame(checkpoint_rows)
    paired = raw.pivot(index=['sampling_seed', 'evaluator_seed'], columns='checkpoint_seed', values='discriminative_score').reset_index()
    paired.columns.name = None
    paired = paired.rename(columns={13: 'historical_better_checkpoint13_ds', 23: 'historical_worse_checkpoint23_ds'})
    paired['worse_minus_better_ds'] = paired.historical_worse_checkpoint23_ds - paired.historical_better_checkpoint13_ds
    paired_means = paired.groupby('sampling_seed').worse_minus_better_ds.agg(['mean', 'std', 'min', 'max']).reset_index()
    decision = dict(checkpoint13_all3_at_or_below_threshold=bool((batches[batches.checkpoint_seed.eq(13)].ds_mean <= .3).all()),
                    checkpoint23_all3_above_threshold=bool((batches[batches.checkpoint_seed.eq(23)].ds_mean > .3).all()),
                    all3_paired_mean_contrasts_positive=bool((paired_means['mean'] > 0).all()),
                    mean_paired_checkpoint_difference=float(paired.worse_minus_better_ds.mean()),
                    minimum_paired_sampling_mean_difference=float(paired_means['mean'].min()),
                    maximum_paired_sampling_mean_difference=float(paired_means['mean'].max()),
                    note='Descriptive exploratory comparison of two historical extremes, three matched sampling seeds; no causal attribution or population variance estimate.')
    decision['predeclared_checkpoint_pattern_observed'] = all(decision[k] for k in [
        'checkpoint13_all3_at_or_below_threshold', 'checkpoint23_all3_above_threshold', 'all3_paired_mean_contrasts_positive'])
    raw.to_csv(OUT / 'raw_discriminator_evaluations.csv', index=False)
    batches.to_csv(OUT / 'per_batch_summary.csv', index=False)
    checkpoints.to_csv(OUT / 'per_checkpoint_summary.csv', index=False)
    paired.to_csv(OUT / 'paired_evaluator_differences.csv', index=False)
    paired_means.to_csv(OUT / 'paired_sampling_mean_differences.csv', index=False)
    (OUT / 'interpretation.json').write_text(json.dumps(decision, indent=2) + '\n')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for index, row in enumerate(batches.itertuples()):
        vals = raw[(raw.checkpoint_seed == row.checkpoint_seed) & (raw.sampling_seed == row.sampling_seed)].sort_values('evaluator_seed').discriminative_score
        color = '#2476ad' if row.checkpoint_seed == 13 else '#d57528'
        ax.scatter(index + np.linspace(-.11, .11, 5), vals, color=color, s=35, alpha=.8)
        ax.errorbar(index, row.ds_mean, yerr=row.evaluator_sd, fmt='D', color=color, capsize=4)
        ax.scatter(index, row.original_mean_ds, marker='_', s=500, color='#777777', linewidth=2)
    ax.axhline(.3, color='#b22b35', linestyle='--', label='User failure threshold 0.3')
    ax.set_xticks(range(6), [f'ckpt{r.checkpoint_seed}\nsample{r.sampling_seed}' for r in batches.itertuples()])
    ax.set_ylim(0, .52)
    ax.set_ylabel('Discriminative score (lower is better)')
    ax.set_title('FD001: fixed checkpoints, 3 sampling seeds, 5 discriminator seeds')
    ax.text(.01, .98, 'Dots: evaluator seeds; diamond: mean ± SD; gray line: historical mean',
            transform=ax.transAxes, ha='left', va='top', fontsize=9)
    ax.legend(loc='lower right')
    fig.tight_layout()
    for suffix in ['png', 'pdf']:
        fig.savefig(OUT / f'discriminative_resampling.{suffix}', dpi=180)
    plt.close(fig)
    if decision['predeclared_checkpoint_pattern_observed']:
        interpretation = '[Hypothesis] 预先约定的检查点差异模式成立：较好检查点3批均未超过0.3，较差检查点3批均超过0.3，且3组配对均保持原排序。下一步优先检查生成器训练、存储频率参数及最佳训练损失检查点选择。仍不能把差异归因到其中某一个因素。'
    else:
        interpretation = '[Observed] 预先约定的“较好3批均好、较差3批均差且配对排序一致”模式未完整成立。应结合下表采样均值范围与每批评价波动判断，不能自动将波动归因于训练或采样中的单一环节。'
    lines = ['# FD001固定检查点重采样诊断（2026-10-29）', '', interpretation, '',
             f'[Observed] 两个检查点来自独立70轮训练，{audit["different_tensor_count"]}个状态张量不同；历史seed13平均DS={historical[13]:.6f}、seed23={historical[23]:.6f}。分别固定权重，使用相同采样种子103/113/123。每批15931×48×14，标签及顺序、DDPM1000步/CFG3/batch256、原固定80/20判别划分、5个判别器种子均一致。没有重新训练生成器或训练下游预测器。', '',
             '| 检查点训练种子 | 新采样种子 | 5次判别均值 | 判别SD | 判别最小–最大 | 均值>0.3 |',
             '|---|---|---:|---:|---:|---|']
    for row in batches.itertuples():
        lines.append(f'| {row.checkpoint_seed} | {row.sampling_seed} | {row.ds_mean:.6f} | {row.evaluator_sd:.6f} | {row.ds_min:.6f}–{row.ds_max:.6f} | {"是" if row.failed else "否"} |')
    lines += ['', '## 不同波动来源的描述性比较', '',
              '| 检查点 | 三批总均值 | 三批均值SD | 三批均值范围 | 每批评价SD的平均 | 最大批内评价范围 | 失败批数 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in checkpoints.itertuples():
        lines.append(f'| {row.checkpoint_seed} | {row.resampling_mean_ds:.6f} | {row.sampling_mean_sd:.6f} | {row.sampling_mean_min:.6f}–{row.sampling_mean_max:.6f} | {row.mean_evaluator_sd:.6f} | {row.max_evaluator_range:.6f} | {row.failed_batches}/3 |')
    lines += ['', '[Observed] 同一采样种子下，seed23检查点减seed13检查点的平均DS差值：' +
              '；'.join(f'{int(r.sampling_seed)}: {r.mean:+.6f}' for r in paired_means.itertuples()) + '。', '',
              '[Hypothesis] 三批均值SD包含有限5个评价种子带来的波动；使用相同评价种子使配对结果相关。以上是原始尺度比较，不是独立方差分量估计或显著性检验。阈值跨越不必然代表大幅变化。两检查点按历史极值选择，三次采样为小样本探索；不能推广成所有训练种子的失败概率。', '',
              '[Code] 判别器未改变：原TensorFlow compat-v1 GRU7、3000次Adam更新、batch128、固定原划分；DS=abs(0.5-accuracy)。详情见冻结协议和源代码哈希。', '',
              '[Observed] 六份生成输出均有限、不同，标签逐元素一致；两检查点在匹配新种子下的采样模型构造前CPU/CUDA/NumPy RNG哈希一致。原检查点/代码/数据哈希未变；30行评价完整且无重复，DS由真假准确率还原。检查点权重SHA、分批进度、环境、原始评价、配对差值与图均保留。没有额外种子、重采样、评价调参或下游测试。', '',
              '[Observed] 五段比较：当前为重采样诊断结果 → 论文没有相应诊断目标 → 数值论文差距不适用 → 3个采样种子×5个共享评价种子，描述性比较且不作总体推断 → 严格复现结论不变。完成6次采样和30次判别评价后停止，零标签路线继续暂停，EXT-002保留weight1。', '',
              f'[Observed] Run `{RUN.name}`；快照 `runs/{RUN.name}/run-manifest.json`；冻结协议 `docs/operations/FD001_CHECKPOINT_RESAMPLING_DIAGNOSTIC.md`；全部输出 `results/diagnostics/fd001_checkpoint_resampling_20261029/`。']
    REPORT.write_text('\n'.join(lines) + '\n')
    audit_result = dict(status='passed', generated_batches=6, discriminator_evaluations=30,
                        matched_rng_states=True, labels_identical=True, checkpoints_unchanged=True,
                        source_input_hashes_unchanged=True, finite_data=True, distinct_samples=True,
                        raw_score_identity=True, completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    (OUT / 'closeout_audit.json').write_text(json.dumps(audit_result, indent=2) + '\n')
    manifest.update(status='complete', completed_at=audit_result['completed_at'], audit=audit_result,
                    outputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in [*sorted(OUT.rglob('*')), REPORT] if p.is_file()})
    (RUN / 'run-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    summary_text = '; '.join(f'checkpoint{r.checkpoint_seed}:meanDS{r.resampling_mean_ds:.6f},failed{r.failed_batches}/3,samplingSD{r.sampling_mean_sd:.6f},meanEvaluatorSD{r.mean_evaluator_sd:.6f}' for r in checkpoints.itertuples())
    with (ROOT / 'experiment-ledger.md').open('a') as handle:
        handle.write(f'\n## `{RUN.name}` completed\n\n- [Observed] Diagnostic/supporting completed6samples/30discriminatorfits at {audit_result["completed_at"]}; no generator/downstream training. {summary_text}. Predeclared persistent checkpoint pattern={decision["predeclared_checkpoint_pattern_observed"]}.\n- [Hypothesis] Consult paired3sampling contrasts andwithinbatch5evaluator spread before assigning next code audit. Two selected historical extremes,three samples,shared evaluator seeds are exploratory; no unique causal component or formal variance inference.\n- [Observed] Original checkpoints/source/data hashes unchanged;matched sampling RNGs,label/count/split identity,finite distinct6outputs,complete30seed grid andscore identities passed. Frozenbudget exhausted,noextra sampling/evaluation/training. Paper comparison notapplicable;strictscope unchanged,zero-labelpaused,EXT-002closed/weight1.\n- Report `{REPORT.relative_to(ROOT)}`;snapshot `runs/{RUN.name}/run-manifest.json`;outputs `{OUT.relative_to(ROOT)}`.\n')
    state_path = ROOT / 'reproduction-state.md'
    state = state_path.read_text()
    checkpoint_text = f'## Latest checkpoint: 2026-10-29 FD001 fixed-checkpoint resampling complete\n\n- [Observed] `{RUN.name}` diagnostic/supporting complete6/6sampling,30/30discriminatorfits. {summary_text}. Persistent checkpoint pattern={decision["predeclared_checkpoint_pattern_observed"]}; twohistoricalextremes/threefreshsamplingseeds/shared5evals,exploratory.\n- [Observed] Exact original15931labels/windows/split/DDPM1000/CFG3/batch256,unchangedgeneratorweights,matchedRNGhashes;finite distinctdata/30rows/scoreidentities/inputhash audits passed. No generator/downstreamtraining. Budgetclosed,noautomaticfollowups;strictresultsunchanged,zero-labelpaused,EXT-002weight1.\n- Report `{REPORT.relative_to(ROOT)}`;snapshot `runs/{RUN.name}/run-manifest.json`;outputs `{OUT.relative_to(ROOT)}`.\n\n'
    state_path.write_text(state.replace('# Persistent reproduction state\n\n', '# Persistent reproduction state\n\n' + checkpoint_text, 1))
    print(checkpoints.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
