"""Summarize existing formal runs using user-defined mean DS > 0.3."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'diagnostic-discriminative-seed-failures-20261028'
OUT = ROOT / 'results/diagnostics/discriminative_seed_failures_20261028'
RUN = ROOT / 'runs' / RUN_ID
REPORT = ROOT / 'docs/conclusions/final/discriminative_seed_failures_20261028_report.md'
THRESHOLD = 0.3
SEEDS = [3, 13, 23, 33, 43]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    sources = []
    for dataset in ['FD001', 'FD002', 'FD003', 'FD004']:
        name = dataset.lower()
        full = (f'results/{name}_w48_public_code_manifest.json' if dataset != 'FD002'
                else 'results/frequency_threshold_validation/fd002_public_arm_seven_seeds_manifest.json')
        sources.extend([
            ('full_w48', dataset, full),
            ('full_w24', dataset, f'results/w24/{name}_w24_public_code_manifest.json'),
            ('zero_w48', dataset, f'results/zeroshot/{name}_table_v_evaluation_manifest.json'),
        ])
        if dataset != 'FD004':
            sources.append(('few_w48', dataset, f'results/fewshot/{name}_table_vi_evaluation_manifest.json'))
    inputs = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    rows, raw_frames, audits = [], [], []
    snapshot = dict(run_id=RUN_ID, scope_class='diagnostic', role='supporting',
                    status='running', started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    command=f'.venv/bin/python analysis/diagnostics/{Path(__file__).name}',
                    definition='User-defined strict mean discriminative_score > 0.3 across five evaluator seeds; no rounding before comparison.',
                    changed_factor='none; aggregate existing formal evaluation CSVs only',
                    excluded='experimental variants, repaired generators, duplicate evaluations, unscored artifacts',
                    python=sys.version, platform=platform.platform())
    (RUN / 'run-manifest.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    for setting, dataset, manifest_path in sources:
        manifest = json.loads((ROOT / manifest_path).read_text())
        assert manifest['status'] == 'complete', manifest_path
        cfg = manifest['config']
        assert cfg['experiment']['dataset'] == dataset
        generation_seeds = cfg['experiment']['generation_seeds']
        evaluator_seeds = cfg['experiment']['evaluator_seeds']
        assert sorted(evaluator_seeds) == SEEDS
        raw_path = cfg['outputs']['raw_csv']
        data = pd.read_csv(ROOT / raw_path)
        inputs[manifest_path] = sha(ROOT / manifest_path)
        inputs[raw_path] = sha(ROOT / raw_path)
        assert not data.duplicated(['generation_seed', 'evaluator_seed']).any()
        assert sorted(data.generation_seed.unique()) == sorted(generation_seeds)
        assert len(data) == len(generation_seeds) * len(evaluator_seeds)
        assert np.isfinite(data.discriminative_score).all()
        assert data.discriminative_score.between(0, .5).all()
        reconstructed = abs(.5 - (data.fake_accuracy + data.real_accuracy) / 2)
        difference = float(abs(reconstructed - data.discriminative_score).max())
        assert difference < 1e-12, (raw_path, difference)
        for seed, group in data.groupby('generation_seed', sort=True):
            assert sorted(group.evaluator_seed.tolist()) == SEEDS
            scores = group.discriminative_score
            rows.append(dict(setting=setting, dataset=dataset, generation_seed=int(seed),
                             evaluation_count=len(group), ds_mean=float(scores.mean()),
                             ds_sd=float(scores.std(ddof=1)), ds_min=float(scores.min()),
                             ds_max=float(scores.max()), failed=bool(scores.mean() > THRESHOLD),
                             individual_evaluations_above_threshold=int((scores > THRESHOLD).sum()),
                             source_csv=raw_path))
        raw_frames.append(data.assign(setting=setting, dataset=dataset, source_csv=raw_path))
        audits.append(dict(setting=setting, dataset=dataset, generation_count=len(generation_seeds),
                           evaluation_count=len(data), score_identity_max_difference=difference))
    per_seed = pd.DataFrame(rows)
    summaries = []
    for (setting, dataset), group in per_seed.groupby(['setting', 'dataset'], sort=True):
        failed = group[group.failed]
        summaries.append(dict(setting=setting, dataset=dataset, failed_seeds=len(failed),
                              evaluated_seeds=len(group), failure_ratio=len(failed) / len(group),
                              failed_seed_ids=','.join(map(str, failed.generation_seed))))
    summary = pd.DataFrame(summaries)
    totals = per_seed.groupby('dataset').failed.agg(failed_runs='sum', evaluated_runs='count').reset_index()
    totals['failure_ratio'] = totals.failed_runs / totals.evaluated_runs
    per_seed.to_csv(OUT / 'per_seed_scores.csv', index=False)
    summary.to_csv(OUT / 'failure_ratios_by_setting.csv', index=False)
    totals.to_csv(OUT / 'inventory_totals_by_fd.csv', index=False)
    pd.concat(raw_frames, ignore_index=True).to_csv(OUT / 'source_evaluations.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'audit_checks.csv', index=False)
    lines = ['# 生成种子判别分数失败比例（2026-10-28）', '',
             '[Observed] 按用户指定，单个生成种子的5个判别评估种子的 discriminative_score 取算术平均，严格大于0.3记为失败；不先四舍五入。分母为已完成正式评估的生成种子数，不是判别器次数或生成窗口条数。', '',
             '[Code] 判别分数为 abs(0.5 - accuracy)，见 `upstream/measure_score/Utils/discriminative_metric.py:196`。0.3是本次用户指定的统计阈值，不作为论文规定阈值。', '',
             '[Observed] 覆盖原版正式全数据w48、全数据w24、少样本w48、零样本w48；FD002全数据w48含额外53/63两个生成种子。各配置分别报告，不混入生成器修复版、阈值变体或下游加权实验。FD004少样本没有已完成的正式判别评估，记为缺失，不计为成功或失败。', '',
             '| 配置 | FD | 失败 / 已评估种子 | 比例 | 失败生成种子 |',
             '|---|---|---:|---:|---|']
    for row in summary.itertuples():
        lines.append(f'| {row.setting} | {row.dataset} | {row.failed_seeds}/{row.evaluated_seeds} | {row.failure_ratio:.2%} | {row.failed_seed_ids or "无"} |')
    lines += ['', '## 逐生成种子分数', '',
              '| 配置 | FD | 生成种子 | 平均判别分数 | 最小–最大 | 5次中超过0.3的次数 | 按均值失败 |',
              '|---|---|---:|---:|---:|---:|---|']
    for row in per_seed.sort_values(['setting', 'dataset', 'generation_seed']).itertuples():
        lines.append(f'| {row.setting} | {row.dataset} | {row.generation_seed} | {row.ds_mean:.6f} | {row.ds_min:.6f}–{row.ds_max:.6f} | {row.individual_evaluations_above_threshold}/5 | {"是" if row.failed else "否"} |')
    lines += ['', '## 全部配置库存合计', '',
              '[Observed] 以下仅为现有已评估生成运行的库存比例：不同窗口长度、数据量和训练任务混合，不能视作某个固定配置的失败概率。同一个数字种子在不同配置下代表不同生成运行。', '',
              '| FD | 失败运行 / 已评估运行 | 比例 |', '|---|---:|---:|']
    for row in totals.itertuples():
        lines.append(f'| {row.dataset} | {row.failed_runs}/{row.evaluated_runs} | {row.failure_ratio:.2%} |')
    lines += ['', '[Observed] 阈值作用于一个生成种子对应整批窗口的判别分数，不意味着这批数据中每个窗口均无效；也不等价于下游RUL预测RMSE高。', '',
              f'[Observed] 共{len(sources)}份正式评估配置，{len(per_seed)}个生成运行，{sum(a["evaluation_count"] for a in audits)}次判别评估；评估种子笛卡尔积、重复、有限值、范围与真假准确率还原分数均通过检查。未重新训练或重新生成。', '',
              f'[Observed] Run `{RUN_ID}`；快照 `runs/{RUN_ID}/run-manifest.json`；原始来源逐行保留在 `results/diagnostics/discriminative_seed_failures_20261028/source_evaluations.csv`，其余明细同目录。无对应论文目标，本次诊断不改变严格复现或EXT-002结论。']
    REPORT.write_text('\n'.join(lines) + '\n')
    snapshot.update(status='complete', completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    inputs_sha256=inputs, threshold=THRESHOLD,
                    evaluator_seeds=SEEDS, setting_count=len(sources), generation_run_count=len(per_seed),
                    outputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in [*sorted(OUT.glob('*.csv')), REPORT]})
    (RUN / 'run-manifest.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    print(summary.to_string(index=False))
    print(totals.to_string(index=False))


if __name__ == '__main__':
    main()
