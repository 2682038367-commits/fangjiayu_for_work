"""Audit and summarize the frozen two-seed final70 rule comparison."""
from pathlib import Path
import csv
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('two_seed_protocol', ROOT / 'scripts/run_fd001_final70_two_seeds.py')
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)
REPORT = ROOT / 'docs/conclusions/final/fd001_final70_two_seeds_20261030_report.md'


def canonical(seed, rule):
    p = job.folder(seed, rule) / 'alias.json'
    if p.exists():
        d = json.loads(p.read_text())
        assert d['canonical_rule'] == 'best' and rule == 'final70'
        assert job.training_audit(seed)['best_and_final_tensors_equal']
        return 'best'
    return rule


def main():
    job.environment()
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = job.check_inputs()
    assert m['status'] == 'computed_pending_summary'
    alias_count = len(m['aliases'])
    assert m['training_runs_completed'] == 2
    assert m['completed_sampling'] == 4-alias_count
    assert m['completed_discriminator_evaluations'] == m['completed_predictor_fits'] == (4-alias_count)*5
    assert len(m['completed_cells']) == 4
    with np.load(job.RUN / 'real_validation.npz') as z:
        y, engines = z['label'].astype(float), z['engine_id'].copy()
    with np.load(job.RUN / 'frozen_plan.npz') as z:
        ti, di = z['train_indices'], z['dev_indices']
    with np.load(job.RUN / 'frozen_inputs.npz') as z:
        all_labels = z['label'].ravel().astype(float)
    anchor = json.loads(job.ANCHOR.read_text())
    all_rows, comparisons, gen_metrics, training, pair_checks, engine_rows = [], [], [], [], [], []
    predictor_checkpoints = {}
    for seed in job.SEEDS:
        audit = job.training_audit(seed)
        assert audit['actual_optimizer_updates'] == 4410 and audit['completed_epochs'] == 70
        assert audit['lr_sequence_matches_original'] and audit['final_serialization_rng_unchanged']
        history = np.loadtxt(job.OUT / f'seed{seed}/training_loss.csv', delimiter=',', skiprows=1)
        eligible = history[history[:, 0] > 5]
        best_epoch = int(eligible[np.argmin(eligible[:, 1]), 0])+1
        assert best_epoch == audit['rerun_best_epoch_one_based'] == audit['rules']['best']['epoch_one_based']
        assert audit['rules']['final70']['epoch_one_based'] == 70
        with np.load(job.OUT / f'seed{seed}/training_epoch_orders.npz') as z:
            assert z['orders'].shape == (70, 15931)
            assert np.array_equal(np.sort(z['orders'], axis=1), np.tile(np.arange(15931), (70, 1)))
        assert len(pd.read_csv(job.OUT / f'seed{seed}/training_batch_zero_draws.csv')) == 4410
        for rule in job.RULES:
            info = audit['rules'][rule]
            state = torch.load(ROOT / info['weight_path'], map_location='cpu', weights_only=True)
            assert job.tensor_hash(state) == info['tensors_sha256']
            assert job.sha(ROOT / info['weight_path']) == info['weight_sha256']
            actual_rule = canonical(seed, rule)
            target = job.folder(seed, actual_rule)
            record = json.loads((target / f'sample_checkpoint{seed}_seed103.json').read_text())
            noise = record['noise_audit']
            assert noise['normal_tensors_exactly_matched'] == noise['conditional_uniform_draws_exactly_matched'] == noise['unconditional_calls_retained'] == 63000
            assert noise['conditional_dropped_windows'] == 0 and noise['retain_condition']
            for k in ['start_cuda_rng_sha256', 'final_cuda_rng_sha256', 'legacy_schedule_fingerprint', 'normal_scalar_count']:
                assert noise[k] == anchor['noise_audit'][k]
            assert record['checkpoint_sha256'] == audit['rules'][actual_rule]['weight_sha256']
            artifact = target / f'samples/checkpoint{seed}_sample103.npz'
            assert job.sha(artifact) == record['artifact_sha256']
            with np.load(artifact) as z:
                assert z['data'].shape == (15931, 48, 14) and np.isfinite(z['data']).all()
                assert np.array_equal(z['label'].ravel(), all_labels)
            ds = pd.read_csv(target / f'evaluation_checkpoint{seed}_sample103.csv')
            assert set(ds.evaluator_seed) == set(job.EVALUATORS) and len(ds) == 5
            recomputed = abs(.5 - (ds.fake_accuracy+ds.real_accuracy)/2)
            assert np.max(abs(recomputed-ds.discriminative_score)) < 1e-12
            with np.load(target / 'condition_predictions.npz') as z:
                cp, cy, ix = z['predictions'], z['label'], z['dev_matched']
            assert cp.shape == (5, 15931) and np.array_equal(cy, all_labels)
            proxy = float(np.sqrt(np.mean((cp[:, ix]-cy[ix])**2)))
            cond = pd.read_csv(target / 'condition_metrics.csv')
            check = cond[(cond.reference == 'dev_matched') & (cond.bin == 'all')]
            assert abs(proxy - np.sqrt(check.mse.mean())) < 1e-10
            gen_metrics.append(dict(generator_seed=seed, rule=rule, actual_rule=actual_rule,
                epoch=info['epoch_one_based'], mean_ds=float(ds.discriminative_score.mean()),
                ds_sd=float(ds.discriminative_score.std(ddof=1)), dev_condition_rmse=proxy))
            segments = pd.read_csv(target / 'per_model_segments.csv')
            actual_epoch = audit['rules'][actual_rule]['epoch_one_based']
            for s in job.EVALUATORS:
                name = f'epoch{actual_epoch}_eval{s}'
                a = json.loads((target / 'audits' / f'{name}.json').read_text())
                assert job.sha(ROOT / a['checkpoint']) == a['checkpoint_sha256']
                predictor_checkpoints[a['checkpoint']] = a['checkpoint_sha256']
                training.append(dict(generator_seed=seed, rule=rule, actual_rule=actual_rule,
                    evaluator_seed=s, epochs=a['completed_epochs'], updates=a['actual_updates'],
                    selected_epoch=a['selected_epoch_one_based'], selected_update=a['selected_update']))
                with np.load(target / 'predictions' / f'{name}.npz') as z:
                    p, sp = z['real_validation'], z['synthetic']
                assert len(p) == len(y) == 4875 and np.isfinite(p).all() and np.isfinite(sp).all()
                for domain, indices in [('synthetic_train', ti), ('synthetic_dev', di)]:
                    stored = segments[(segments.evaluator_seed == s) & (segments.domain == domain) & (segments.bin == 'all')]
                    assert abs(float(stored.mse_window.iloc[0])-float(np.mean((sp[indices]-all_labels[indices])**2))) < 1e-9
                for b, lo, hi in m['config']['bins']:
                    mask = (y >= lo) & (y <= hi)
                    e = p[mask]-y[mask]
                    per = []
                    for eng in np.unique(engines[mask]):
                        take = mask & (engines == eng)
                        mse = float(np.mean((p[take]-y[take])**2))
                        per.append(mse)
                        engine_rows.append(dict(generator_seed=seed, rule=rule, evaluator_seed=s, bin=b,
                            engine_id=int(eng), n_windows=int(take.sum()), mse=mse))
                    re = float(np.sqrt(np.mean(per)))
                    rw = float(np.sqrt(np.mean(e**2)))
                    stored = segments[(segments.evaluator_seed == s) & (segments.domain == 'real_validation') & (segments.bin == b)]
                    assert len(stored) == 1 and abs(re-float(stored.rmse_engine.iloc[0])) < 1e-9
                    assert abs(rw-float(stored.rmse_window.iloc[0])) < 1e-9
                    all_rows.append(dict(generator_seed=seed, rule=rule, evaluator_seed=s, bin=b,
                        n_windows=int(mask.sum()), n_engines=len(per), mse_engine=float(np.mean(per)),
                        rmse_engine=re, mse_window=float(np.mean(e**2)), rmse_window=rw))
        # Check paired initial tensors and consumed order prefixes after both logical rules are resolved.
        for s in job.EVALUATORS:
            audits, orders = [], []
            for rule in job.RULES:
                actual = canonical(seed, rule)
                ep = audit['rules'][actual]['epoch_one_based']
                target = job.folder(seed, actual)
                audits.append(json.loads((target / 'audits' / f'epoch{ep}_eval{s}.json').read_text()))
                with np.load(target / 'audits' / f'epoch{ep}_eval{s}_orders.npz') as z:
                    orders.append(z['orders'])
            assert audits[0]['initialization_sha256'] == audits[1]['initialization_sha256']
            n = min(map(len, orders))
            assert np.array_equal(orders[0][:n], orders[1][:n])
            pair_checks.append(dict(generator_seed=seed, evaluator_seed=s, same_initial_tensors=True,
                same_consumed_order_prefix=True, compared_epochs=n, best_updates=audits[0]['actual_updates'],
                final_updates=audits[1]['actual_updates'], alias=canonical(seed, 'final70') == 'best'))
    all_rows = pd.DataFrame(all_rows)
    engines_df = pd.DataFrame(engine_rows)
    for seed in job.SEEDS:
        for b, _, _ in m['config']['bins']:
            group = all_rows[(all_rows.generator_seed == seed) & (all_rows.bin == b)]
            for measure in ['engine', 'window']:
                pivot = group.pivot(index='evaluator_seed', columns='rule', values='rmse_'+measure).reindex(job.EVALUATORS)
                delta = pivot.final70-pivot.best
                before = float(np.sqrt(group[group.rule == 'best']['mse_'+measure].mean()))
                after = float(np.sqrt(group[group.rule == 'final70']['mse_'+measure].mean()))
                comparisons.append(dict(generator_seed=seed, bin=b, measure=measure, best_rmse=before,
                    final70_rmse=after, change=after-before, relative_change=after/before-1,
                    paired_mean_change=float(delta.mean()), paired_change_sd=float(delta.std(ddof=1)),
                    paired_improving=int((delta<0).sum()), n_windows=int(group.n_windows.iloc[0]), n_engines=int(group.n_engines.iloc[0])))
    comp = pd.DataFrame(comparisons)
    primary = comp[(comp.bin == 'all') & (comp.measure == 'engine')].set_index('generator_seed')
    flags = dict(seed13_no_material_overall_degradation=bool(primary.loc[13, 'relative_change'] <= .05),
        seed3_clear_downstream_improvement=bool(primary.loc[3, 'relative_change'] <= -.05 and primary.loc[3, 'paired_improving'] >= 4),
        not_a_significance_test=True, no_default_rule_automatically_adopted=True)
    all_rows.to_csv(job.OUT / 'raw_real_validation_metrics.csv', index=False)
    engines_df.to_csv(job.OUT / 'per_engine_comparison_inputs.csv', index=False)
    comp.to_csv(job.OUT / 'comparison.csv', index=False)
    pd.DataFrame(gen_metrics).to_csv(job.OUT / 'generation_quality.csv', index=False)
    pd.DataFrame(training).to_csv(job.OUT / 'predictor_training.csv', index=False)
    job.write_csv(job.OUT / 'predictor_pairing_audit.csv', pair_checks)
    job.dump(job.OUT / 'interpretation.json', dict(primary=primary.reset_index().to_dict('records'), flags=flags,
        training_seeds_chosen_by_historic_quality=True, generator_saw_validation_engines=True))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained', sharey=True)
    plot_bins = ['all', '0-30', '31-60', '61-90', '91-125']
    x = np.arange(len(plot_bins))
    for ax, seed in zip(axes, job.SEEDS):
        values = comp[(comp.generator_seed == seed) & (comp.measure == 'engine')].set_index('bin').loc[plot_bins]
        ax.bar(x-.18, values.best_rmse, .36, label='Best training loss')
        ax.bar(x+.18, values.final70_rmse, .36, label='Final epoch70')
        ax.set_xticks(x, plot_bins)
        ax.set_title(f'Generator seed{seed}: same training, fixed rule contrast')
        ax.set_ylim(bottom=0)
        ax.legend()
    axes[0].set_ylabel('Real-validation engine-balanced pooled RMSE')
    for ext in ['png', 'pdf']:
        fig.savefig(job.OUT / ('final70_downstream.'+ext), dpi=180)
    plt.close(fig)
    lines = ['# FD001固定最终70轮规则：两个额外生成训练种子', '',
        '[Observed] 本轮仅重跑seed13与seed3，各按原参数训练70轮，比较各自同次训练内最低训练损失权重和最终70轮权重；不逐种子搜索轮次。历史选中seed13/64轮、seed3/43轮。原后期权重缺失，新轨迹不是原历史模型的直接恢复。', '',
        '[Observed] seed13由用户指定；seed3按原DS排除已研究seed23后取最差者，历史DS分别0.212708和0.359052。种子选择事先固定，未按重跑结果换人。', '',
        f'[Observed] 实际新增训练{m["training_runs_completed"]}次、采样{m["completed_sampling"]}批、DS拟合{m["completed_discriminator_evaluations"]}次、下游预测器拟合{m["completed_predictor_fits"]}次；同权重别名数{alias_count}，如有别名明确共享计算，不计为新增独立结果。', '',
        '[Code] 原生成器训练设置、数值整批标签置零.1/训练条件丢弃.2保留。采样103/全部15931寿命标签/DDPM1000/CFG3/批256与此前一致，采样条件分支始终保留标签，显式无条件CFG分支保留；所有实际初始/逐步噪声匹配原完整指纹。', '',
        '[Code] 真实验证集合直接复用前一轮30台/4875窗口及身份、截断125标签与11151/4780生成train/dev划分，原预测器配置/最高35轮/原生成dev选模与早停、五个预测器种子不变。真实验证不参与预测器选模。', '',
        '## 主要结果', '',
        'RMSE：每台发动机先平均窗口平方误差，再平均发动机及5个预测器，最后开方；不是预测集成。', '',
        '| 生成训练种子 | 重跑最低损失轮次 | 最低损失权重 RMSE | 最终70轮 RMSE | 相对变化 | 配对改善数 |',
        '|---|---:|---:|---:|---:|---:|']
    for seed in job.SEEDS:
        a = primary.loc[seed]
        lines.append(f'| {seed} | {job.training_audit(seed)["rerun_best_epoch_one_based"]} | {a.best_rmse:.4f} | {a.final70_rmse:.4f} | {a.relative_change:+.2%} | {int(a.paired_improving)}/5 |')
    lines += ['', '## 分段取舍', '', '| 生成种子 | 真实RUL段 | 最低损失权重 | 最终70轮 | 相对变化 | 配对改善数 |', '|---|---|---:|---:|---:|---:|']
    for a in comp[(comp.measure == 'engine') & (comp.bin != 'all')].itertuples():
        lines.append(f'| {a.generator_seed} | {a.bin} | {a.best_rmse:.4f} | {a.final70_rmse:.4f} | {a.relative_change:+.2%} | {a.paired_improving}/5 |')
    lines += ['', '## 生成评价（辅助）', '', '| 生成种子 | 规则 | 实际轮次 | 平均DS | 匹配3000窗口条件读出RMSE |', '|---|---|---:|---:|---:|']
    for a in gen_metrics:
        lines.append(f'| {a["generator_seed"]} | {a["rule"]} | {a["epoch"]} | {a["mean_ds"]:.6f} | {a["dev_condition_rmse"]:.4f} |')
    lines += ['', '## 结论、审计与限制', '',
        f'[Observed] 预先固定的seed13总体不劣化超过5%标志={flags["seed13_no_material_overall_degradation"]}；seed3总体至少改善5%且>=4/5配对改善标志={flags["seed3_clear_downstream_improvement"]}。这是实用探索性判断，不是显著性检验；不能据总体标志掩盖分段退步。', '',
        '[Observed] 两次训练各70轮/4410更新、70份数据顺序/4410批置零抽样及原学习率序列已核对；最佳权重确为原规则最小值，最终权重在4410次更新后保存，保存随机状态不变。所有生成标签/噪声、DS分数算术、真实预测器条件读出、保存预测的真实分段RMSE独立重算通过。同种子预测器初始参数和消耗数据顺序公共前缀一致，实际训练轮次/更新/选模轮次见predictor_training.csv。', '',
        '[Hypothesis] 若好种子总体不受损且差种子重复改善，支持最终70轮是值得扩大验证的统一候选；若好种子退步或差种子不改善，应如实报告其适用边界。只能比较已观测的两次重跑，不证明任意种子最终权重都好，也不唯一解释所有历史RMSE差距。', '',
        '[Observed] 一个固定采样种子、历史质量挑选的两个训练种子，以及此前已分析过的验证集合，不能形成无偏总体规律。生成器和MinMax已学习过这30台验证发动机，本轮是下游探索性验证而非整条流程独立验证；未评价原真实测试窗口。条件读出不替代真实下游收益，未据测试结果逐种子挑轮次。', '',
        '[Observed] 当前诊断结果→论文无同设计指标→论文数值差距不适用→五个预测器种子只描述条件波动→严格复现结论不变。预算结束，未自动采用最终70轮默认值、未新增种子/轮次/采样种子/真实测试评价。数值置零修复仍暂停，EXT-002保留weight1，主实验/A/B原状态、C推迟。', '',
        f'[Observed] Run `{job.RUN.name}`；快照`{job.RUN.relative_to(ROOT)}/run-manifest.json`；协议`{job.PROTOCOL.relative_to(ROOT)}`；产物`{job.OUT.relative_to(ROOT)}`。']
    REPORT.write_text('\n'.join(lines)+'\n')
    job.dump(job.OUT / 'closeout_audit.json', dict(status='passed', audited_training_runs=2,
        all_source_data_weight_hashes_unchanged=True, actual_noise_anchor_matches=True,
        original_best_loss_rule_verified=True, original_source_functions_unchanged=True,
        saved_predictions_metrics_recomputed=True, predictor_initial_order_prefix_matches=True,
        aliases_explicit=alias_count, checked_at=job.now()))
    m.update(status='complete', completed_at=job.now(), flags=flags,
        primary_comparisons=primary.reset_index().to_dict('records'), predictor_checkpoints_sha256=predictor_checkpoints,
        outputs_sha256={str(p.relative_to(ROOT)): job.sha(p) for p in job.OUT.rglob('*') if p.is_file()})
    m['outputs_sha256'][str(REPORT.relative_to(ROOT))] = job.sha(REPORT)
    job.dump(job.RUN / 'run-manifest.json', m)
    text = '; '.join(f'seed{s}:best{job.training_audit(s)["rerun_best_epoch_one_based"]} RMSE{primary.loc[s,"best_rmse"]:.6f}→final70 {primary.loc[s,"final70_rmse"]:.6f} ({primary.loc[s,"relative_change"]:+.2%}),{int(primary.loc[s,"paired_improving"])}/5paired improve' for s in job.SEEDS)
    job.update_state('completed', '- [Observed] '+text+'. Fixed within-rerun rule contrast, all noise/source/predictor/order/saved-metric audits passed.\n'
        f'- [Observed] Actual2training/{m["completed_sampling"]}samples/{m["completed_discriminator_evaluations"]}DSfits/{m["completed_predictor_fits"]}predictorfits; alias{alias_count}. Budget closed; all segments/flags in report, no automatic rule/default adoption or further runs.\n'
        f'- Report `{REPORT.relative_to(ROOT)}`. Current diagnostic→no paper target→paper gap not applicable→conditional five-predictor variation→strict conclusions unchanged.')
    print(primary.to_string(), flush=True)
    print(json.dumps(flags, indent=2), flush=True)


if __name__ == '__main__':
    main()
