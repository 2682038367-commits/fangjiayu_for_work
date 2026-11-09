"""Experiment2: audit and summarize existing paired seed13 condition readouts."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/diagnostic-fd001-seed13-condition-segments-20261102'
OUT = ROOT / 'results/diagnostics/fd001_seed13_condition_segments_20261102'
SOURCE = ROOT / 'results/diagnostics/fd001_final70_two_seeds_20261030'
SOURCE_RUN = ROOT / 'runs/diagnostic-fd001-final70-two-seeds-20261030'
REFERENCE = ROOT / 'results/diagnostics/real_predictor_condition_w48/FD001_predictions.npz'
PROTOCOL = ROOT / 'docs/operations/FD001_SEED13_CONDITION_SEGMENTS.md'
REPORT = ROOT / 'docs/conclusions/final/fd001_seed13_condition_segments_20261102_report.md'
ARMS = [('best', 64), ('final70', 70)]
SEEDS = [3, 13, 23, 33, 43]
BINS = [('0-30', 0, 30), ('31-60', 31, 60), ('61-90', 61, 90), ('91-124', 91, 124), ('125', 125, 125)]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def state(stage, body):
    text = body + f'\n- Diagnostic/supporting `{RUN.name}`, scope `scope-20261102-fd001-seed13-condition-segments`. Snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`; protocol `{PROTOCOL.relative_to(ROOT)}`. User explicitly authorizes experiment2 now, superseding earlier deferral; experiment1 artifacts unchanged. No new inference/training/sampling, no default change; zero-label repair paused, EXT-002weight1, strict main/A/B unchanged,C deferred.\n'
    p = ROOT / 'reproduction-state.md'
    head, tail = p.read_text().split('\n', 1)
    p.write_text(head+f'\n\n## Latest checkpoint: 2026-11-02 seed13 condition segments {stage}\n\n'+text+'\n'+tail)
    with (ROOT / 'experiment-ledger.md').open('a') as f:
        f.write(f'\n## `{RUN.name}` {stage}\n\n'+text)


def check_inputs():
    m = json.loads((RUN / 'run-manifest.json').read_text())
    for p, h in m['inputs_sha256'].items():
        assert sha(ROOT / p) == h, p
    return m


def prepare():
    import numpy as np
    assert not (RUN / 'run-manifest.json').exists(), 'Preserve existing run'
    for p in [RUN, RUN / 'source', OUT]:
        p.mkdir(parents=True, exist_ok=True)
    previous = json.loads((SOURCE_RUN / 'run-manifest.json').read_text())
    assert previous['status'] == 'complete'
    paths = [Path(__file__), PROTOCOL, SOURCE_RUN / 'run-manifest.json', ROOT / 'SOURCE_SNAPSHOT.md',
        SOURCE / 'seed13/training_audit.json', SOURCE / 'comparison.csv', REFERENCE]
    bound = dict(previous['inputs_sha256'], **previous['outputs_sha256'])
    for rule, epoch in ARMS:
        base = SOURCE / f'seed13/{rule}'
        paths.extend(base / name for name in ['samples/checkpoint13_sample103.npz',
            'sample_checkpoint13_seed103.json', 'condition_predictions.npz', 'condition_metrics.csv'])
    a = json.loads((SOURCE / 'seed13/training_audit.json').read_text())
    for rule, epoch in ARMS:
        assert a['rules'][rule]['epoch_one_based'] == epoch
        paths.append(ROOT / a['rules'][rule]['weight_path'])
    for s in SEEDS:
        paths.append(ROOT / f'checkpoints/evaluators/real_data_baseline_w48/FD001_eval{s}.pth')
    for p in paths:
        name = str(p.relative_to(ROOT))
        if name in bound:
            assert sha(p) == bound[name], name
        elif p not in [Path(__file__), PROTOCOL, SOURCE_RUN / 'run-manifest.json']:
            raise AssertionError('Missing prior artifact binding: '+name)
    with np.load(REFERENCE) as z:
        position = list(z['generation_seeds']).index(23)
        ix = z['generated_dev_indices'][position].copy()
        expected_y = z['real_dev_labels'].copy()
    assert ix.shape == (3000,) and len(np.unique(ix)) == 3000
    records = []
    for rule, epoch in ARMS:
        base = SOURCE / f'seed13/{rule}'
        record = json.loads((base / 'sample_checkpoint13_seed103.json').read_text())
        assert record['selected_epoch'] == epoch and record['sampling_seed'] == 103
        assert record['checkpoint_sha256'] == a['rules'][rule]['weight_sha256']
        assert record['artifact_sha256'] == sha(base / 'samples/checkpoint13_sample103.npz')
        noise = record['noise_audit']
        assert noise['status'] == 'passed' and noise['retain_condition'] and noise['conditional_dropped_windows'] == 0
        assert noise['normal_tensors_exactly_matched'] == noise['conditional_uniform_draws_exactly_matched'] == 63000
        assert noise['unconditional_calls_retained'] == 63000
        with np.load(base / 'condition_predictions.npz') as z:
            assert list(z['evaluator_seeds']) == SEEDS and z['predictions'].shape == (5, 15931)
            assert np.array_equal(z['dev_matched'], ix) and np.array_equal(z['label'][ix], expected_y)
            assert np.isfinite(z['predictions']).all()
            labels = z['label'].copy()
        with np.load(base / 'samples/checkpoint13_sample103.npz') as z:
            assert z['data'].shape == (15931, 48, 14) and np.array_equal(z['label'].ravel(), labels)
        records.append(record)
    assert records[0]['pre_model_construction_rng_sha256'] == records[1]['pre_model_construction_rng_sha256']
    for key in ['start_cuda_rng_sha256', 'final_cuda_rng_sha256', 'legacy_schedule_fingerprint', 'normal_scalar_count']:
        assert records[0]['noise_audit'][key] == records[1]['noise_audit'][key]
    counts = {name: int(((expected_y >= lo) & (expected_y <= hi)).sum()) for name, lo, hi in BINS}
    assert set(counts.values()) == {600}
    np.savez_compressed(RUN / 'frozen_evaluation_list.npz', generated_indices=ix, condition=expected_y,
                        predictor_seeds=np.array(SEEDS))
    config = dict(generator_seed=13, epochs=[64, 70], source_rules=['best', 'final70'],
        sampling_seed=103, n_generated_full=15931, n_evaluated=3000, bins=BINS, bin_counts=counts,
        predictor_seeds=SEEDS, fixed_index_provenance='previous common dev_matched list, reference generation_seed23 position',
        aggregate_rmse='sqrt(mean over5 predictors and600 windows squared error), not prediction ensemble',
        signed_error='mean readout-condition; raw unclipped outputs and capped125 conditions',
        execution='reuse hash-bound historical readouts; no new inference or independent repeat',
        budget=dict(per_predictor_bin_rows=50, aggregate_pairs=5, new_models=0, new_samples=0),
        stop='one audit/aggregation, any identity failure, no follow-up sensor/coverage intervention automatically')
    dump(RUN / 'config.json', config)
    paths.extend([RUN / 'config.json', RUN / 'frozen_evaluation_list.npz'])
    for p in [Path(__file__), PROTOCOL]:
        target = RUN / 'source' / p.name
        shutil.copy2(p, target)
        paths.append(target)
    for name, command in [('workspace.diff', ['git', 'diff', '--binary']), ('git-status.txt', ['git', 'status', '--short']),
        ('environment-freeze.txt', [sys.executable, '-m', 'pip', 'freeze'])]:
        p = RUN / name
        p.write_bytes(subprocess.check_output(command, cwd=ROOT))
        paths.append(p)
    dump(RUN / 'run-manifest.json', dict(run_id=RUN.name, scope_class='diagnostic', role='supporting',
        scope_contract='scope-20261102-fd001-seed13-condition-segments', status='prepared', prepared_at=now(),
        config=config, command=[sys.executable, str(Path(__file__)), 'run'],
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        user_authorization='Explicit experiment2 seed13 epochs64/70 fixed3000 conditions/noise, five real predictors,125 separate',
        inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(paths)},
        preflight=dict(prior_hashes_verified=True, same_conditions_and_generated_indices=True,
            same_predictor_weights=True, same_full_actual_noise_trace=True, counts=counts),
        no_new_training_sampling_or_inference=True))
    state('prepared', '- [Observed] Exact seed13 same-rerun64/70 weights, paired103 noise, existing five real-predictor outputs and common3000 generated indices verified against prior frozen hashes. Five bins each600;125 separate. Reuse existing readouts, not new independent inference. Metrics aggregation pending.')
    print('Prepared fixed3000 matched conditions, paired noise and5 existing predictor readouts.', flush=True)


def run():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = check_inputs()
    assert m['status'] == 'prepared'
    m.update(status='running', started_at=now())
    dump(RUN / 'run-manifest.json', m)
    try:
        with np.load(RUN / 'frozen_evaluation_list.npz') as z:
            ix, y = z['generated_indices'].copy(), z['condition'].copy()
        rows, aggregate, per_window = [], [], []
        for rule, epoch in ARMS:
            base = SOURCE / f'seed13/{rule}'
            with np.load(base / 'condition_predictions.npz') as z:
                p = z['predictions'][:, ix].copy()
            per_window.append(p)
            old_metrics = pd.read_csv(base / 'condition_metrics.csv')
            for j, s in enumerate(SEEDS):
                for name, lo, hi in BINS:
                    use = (y >= lo) & (y <= hi)
                    err = p[j, use]-y[use]
                    row = dict(epoch=epoch, predictor_seed=s, bin=name, n_windows=int(use.sum()),
                        mse=float(np.mean(err**2)), rmse=float(np.sqrt(np.mean(err**2))),
                        signed_error=float(err.mean()), condition_mean=float(y[use].mean()),
                        readout_mean=float(p[j, use].mean()))
                    old = old_metrics[(old_metrics.reference == 'dev_matched') & (old_metrics.evaluator_seed == s) & (old_metrics.bin == name)]
                    assert len(old) == 1 and abs(float(old.rmse.iloc[0])-row['rmse']) < 1e-10
                    assert abs(float(old.bias.iloc[0])-row['signed_error']) < 1e-10
                    rows.append(row)
            for name, lo, hi in BINS:
                use = (y >= lo) & (y <= hi)
                errors = p[:, use]-y[None, use]
                aggregate.append(dict(epoch=epoch, bin=name, n_windows=int(use.sum()),
                    pooled_rmse=float(np.sqrt(np.mean(errors**2))), signed_error=float(errors.mean()),
                    mean_predictor_rmse=float(np.sqrt(np.mean(errors**2, axis=1)).mean())))
        individual = pd.DataFrame(rows)
        agg = pd.DataFrame(aggregate)
        assert len(individual) == 50
        pairs = []
        for name, lo, hi in BINS:
            a = individual[(individual.epoch == 64) & (individual.bin == name)].set_index('predictor_seed').loc[SEEDS]
            b = individual[(individual.epoch == 70) & (individual.bin == name)].set_index('predictor_seed').loc[SEEDS]
            x = agg[(agg.epoch == 64) & (agg.bin == name)].iloc[0]
            z = agg[(agg.epoch == 70) & (agg.bin == name)].iloc[0]
            pairs.append(dict(bin=name, n_windows=int(x.n_windows), rmse64=float(x.pooled_rmse), rmse70=float(z.pooled_rmse),
                rmse_change=float(z.pooled_rmse-x.pooled_rmse), rmse_relative_change=float(z.pooled_rmse/x.pooled_rmse-1),
                signed64=float(x.signed_error), signed70=float(z.signed_error), signed_shift=float(z.signed_error-x.signed_error),
                n_predictors_rmse_higher=int((b.rmse>a.rmse).sum()), n_predictors_rmse_lower=int((b.rmse<a.rmse).sum()),
                n_predictors_signed_shift_positive=int((b.signed_error>a.signed_error).sum())))
        individual.to_csv(OUT / 'per_predictor_segments.csv', index=False)
        agg.to_csv(OUT / 'aggregate_segments.csv', index=False)
        pd.DataFrame(pairs).to_csv(OUT / 'paired_segments.csv', index=False)
        np.savez_compressed(OUT / 'matched_predictions.npz', predictions=np.array(per_window),
            condition=y, generated_indices=ix, epochs=np.array([64, 70]), predictor_seeds=np.array(SEEDS))
        # Context only: different domain from the above generated-sample condition readout.
        downstream = pd.read_csv(SOURCE / 'comparison.csv')
        downstream = downstream[(downstream.generator_seed == 13) & (downstream.measure == 'engine')]
        downstream.to_csv(OUT / 'existing_real_downstream_context.csv', index=False)
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout='constrained')
        x = np.arange(5)
        for epoch, offset, color in [(64, -.18, 'C0'), (70, .18, 'C1')]:
            subset = agg[agg.epoch == epoch].set_index('bin').loc[[b[0] for b in BINS]]
            axes[0].bar(x+offset, subset.pooled_rmse, .36, color=color, alpha=.65, label=f'Epoch{epoch}: pooled RMSE')
            axes[1].bar(x+offset, subset.signed_error, .36, color=color, alpha=.65, label=f'Epoch{epoch}: mean signed error')
            for i, (name, _, _) in enumerate(BINS):
                one = individual[(individual.epoch == epoch) & (individual.bin == name)]
                axes[0].scatter(i+offset+np.linspace(-.06, .06, 5), one.rmse, s=17, color=color)
                axes[1].scatter(i+offset+np.linspace(-.06, .06, 5), one.signed_error, s=17, color=color)
        axes[0].set_ylim(bottom=0)
        axes[0].set_ylabel('Condition readout RMSE')
        axes[1].axhline(0, color='black', lw=1)
        axes[1].set_ylabel('Signed error: readout - condition')
        axes[0].set_title('Seed13, fixed3000 conditions/noise,5 real-trained predictors\nDots show individual predictors;125 is a separate capped condition')
        for ax in axes:
            ax.set_xticks(x, [b[0] for b in BINS])
            ax.legend(fontsize=9)
        for ext in ['png', 'pdf']:
            fig.savefig(OUT / ('condition_segments.'+ext), dpi=180)
        plt.close(fig)
        lines = ['# 实验二：seed13第64/70轮的分寿命条件读出', '',
            '[Observed] 复用已保存的两批生成数据及5个真实训练预测器逐窗口输出，未新增训练、生成或推理；本轮是既有观察的固定清单汇总与独立核验，不计作新重复实验。两个检查点来自同一次seed13重跑，分别为最低损失第64轮和最终第70轮。', '',
            '[Observed] 完全相同3000个生成位置/RUL条件，五个寿命段各600个；评价清单与前次两种子对照中的共同dev_matched清单一致。全批15931生成标签及采样103实际初始/逐步噪声完整指纹相同，生成数量/CFG3/显式无条件分支/始终保留条件的采样行为不变。旧记录、权重和预测输出哈希均核对。', '',
            '## 汇总结果', '',
            'RMSE先对每个预测器/窗口平方误差平均再开方；有向误差为读出RUL−条件RUL，对五个预测器平均。没有先集成预测再算RMSE，输出未裁剪。', '',
            '| 条件RUL段 | 窗口数 | 64轮RMSE | 70轮RMSE | RMSE变化 | 64轮有向误差 | 70轮有向误差 | RMSE变大的预测器 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for a in pairs:
            lines.append(f'| {a["bin"]} | {a["n_windows"]} | {a["rmse64"]:.4f} | {a["rmse70"]:.4f} | {a["rmse_relative_change"]:+.2%} | {a["signed64"]:+.4f} | {a["signed70"]:+.4f} | {a["n_predictors_rmse_higher"]}/5 |')
        lines += ['', '## 五个固定真实预测器的分别结果', '',
            '以下每行一个固定预测器、一个相同窗口集合。预测器种子不是生成器训练种子；本轮生成器始终为seed13。', '',
            '| RUL段 | 真实预测器种子 | 64轮RMSE | 70轮RMSE | 64轮有向误差 | 70轮有向误差 |',
            '|---|---:|---:|---:|---:|---:|']
        for name, _, _ in BINS:
            for s in SEEDS:
                a = individual[(individual.epoch == 64) & (individual.bin == name) & (individual.predictor_seed == s)].iloc[0]
                b = individual[(individual.epoch == 70) & (individual.bin == name) & (individual.predictor_seed == s)].iloc[0]
                lines.append(f'| {name} | {s} | {a.rmse:.4f} | {b.rmse:.4f} | {a.signed_error:+.4f} | {b.signed_error:+.4f} |')
        lines += ['', '## 实际解释', '',
            '[Observed] 91–124条件段的读出RMSE上升，有向误差从接近零的负值移向明显正值，具体五模型方向见表；说明在这批相同条件/噪声下，该段第70轮生成信号被固定真实预测器读成更大RUL。125条件的结果必须单独解释，不能合并后宣称所有高RUL条件都退步。', '',
            '[Hypothesis] 若91–124的五模型方向一致，支持该段与指定标签的匹配在此诊断下变差，可作为后续检查传感器水平/变化偏移的线索。若125读出改善而真实125段下游退步，则条件读出的一致性不足以解释整个91–125下游差距，还需分别检验覆盖与下游学习等因素；本轮不能在这些解释间唯一归因。', '',
            '[Observed] 已有真实验证下游误差是另一种评价：seed13的91–125段14.7035→19.6691（+33.77%），其中91–124段16.6578→21.0652（+26.46%）、125段14.7524→21.8509（+48.12%）。这些是生成数据训练的预测器对真实窗口预测的误差，不是本表真实预测器对生成窗口的读出。原30台发动机已被生成器见过，仍属下游探索性验证。', '',
            '[Observed] 125是封顶条件，无法区分参考真实RUL恰好125还是更大；生成窗口没有独立观测的真实寿命。真实预测器读出仅是固定诊断，正负值表示其相对条件的读出方向，不能据此单独判定生成信号正确或真实年轻/衰老程度。五预测器共享训练数据/任务，不构成五次独立生成实验，也不作总体统计显著性声明。', '',
            '[Observed] 一次固定清单汇总完成并停止，未自动启动传感器/覆盖检查、额外推理、训练、采样或选模变更；数值置零修复仍暂停、EXT-002weight1、严格主实验/A/B不变，C推迟。当前诊断→论文无同设计目标→论文数值差距不适用→5个固定预测器条件读出一致性→不改变严格复现结论。', '',
            f'[Observed] Run `{RUN.name}`；快照`{RUN.relative_to(ROOT)}/run-manifest.json`；协议`{PROTOCOL.relative_to(ROOT)}`；50条分别结果`{OUT.relative_to(ROOT)}/per_predictor_segments.csv`。']
        REPORT.write_text('\n'.join(lines)+'\n')
        check_inputs()
        dump(OUT / 'audit.json', dict(status='passed', identity_noise_labels_predictors_checked=True,
            aggregate_recomputed_from_saved_windows=True, original_per_predictor_metrics_match=True,
            all_five_predictors_reported=True, capped125_separate=True, new_inference_training_sampling=0))
        m.update(status='complete', completed_at=now(), paired_segments=pairs,
            outputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in OUT.rglob('*') if p.is_file()})
        m['outputs_sha256'][str(REPORT.relative_to(ROOT))] = sha(REPORT)
        dump(RUN / 'run-manifest.json', m)
        summary = '; '.join(f'{a["bin"]}:RMSE{a["rmse64"]:.4f}→{a["rmse70"]:.4f},bias{a["signed64"]:+.4f}→{a["signed70"]:+.4f},RMSEup{a["n_predictors_rmse_higher"]}/5' for a in pairs)
        state('completed', '- [Observed] '+summary+'. Same3000 conditions/noise/5 fixed predictors; all source hashes and per-model raw metric identities passed.50 individual rows/5 paired rows saved; no new inference/training/sampling.\n'+f'- Report `{REPORT.relative_to(ROOT)}`. Readout proxy only;91–124 and125 interpreted separately, no automatic follow-up intervention.')
        print(pd.DataFrame(pairs).to_string(index=False), flush=True)
    except BaseException:
        m.update(status='failed', failed_at=now(), error=traceback.format_exc())
        dump(RUN / 'run-manifest.json', m)
        state('failed', '- [Observed] Stopped on artifact/metric audit failure; retained error in manifest. No new model training/sampling or replacement results.')
        raise


if __name__ == '__main__':
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else run()
