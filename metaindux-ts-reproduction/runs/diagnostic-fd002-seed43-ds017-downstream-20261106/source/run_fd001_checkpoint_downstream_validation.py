"""Original predictor training, fixed real validation: checkpoint15/70, auxiliary64."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/diagnostic-fd001-checkpoint-downstream-validation-20261030'
OUT = ROOT / 'results/diagnostics/fd001_checkpoint_downstream_validation_20261030'
AGE = ROOT / 'results/diagnostics/fd001_seed23_checkpoint_age_20261030'
AGE_RUN = ROOT / 'runs/diagnostic-fd001-seed23-checkpoint-age-20261030'
PROTOCOL = ROOT / 'docs/operations/FD001_CHECKPOINT_DOWNSTREAM_VALIDATION.md'
REPORT = ROOT / 'docs/conclusions/final/fd001_checkpoint_downstream_validation_20261030_report.md'
CHECKPOINTS = ROOT / 'checkpoints/evaluators/fd001_checkpoint_downstream_validation_20261030'
EPOCHS = [15, 70, 64]
SEEDS = [3, 13, 23, 33, 43]
BINS = [('all', 0, 125), ('0-30', 0, 30), ('31-60', 31, 60),
        ('61-90', 61, 90), ('91-125', 91, 125), ('91-124', 91, 124), ('125', 125, 125)]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def environment():
    for k, v in dict(WANDB_MODE='disabled', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                     OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
                     MPLCONFIGDIR=str(ROOT / '.cache/matplotlib')).items():
        os.environ.setdefault(k, v)


def sample_path(epoch):
    return AGE / f'epoch{epoch}/retain_condition/samples/checkpoint23_sample103.npz'


def check_inputs():
    m = json.loads((RUN / 'run-manifest.json').read_text())
    for name, expected in m['inputs_sha256'].items():
        assert sha(ROOT / name) == expected, name
    return m


def update_state(title, body):
    p = ROOT / 'reproduction-state.md'
    text = p.read_text()
    head, rest = text.split('\n', 1)
    p.write_text(head + '\n\n## Latest checkpoint: ' + title + '\n\n' + body + '\n' + rest)
    with (ROOT / 'experiment-ledger.md').open('a') as f:
        f.write('\n## ' + title + '\n\n' + body + '\n')


def prepare():
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from sklearn.preprocessing import MinMaxScaler
    assert not (RUN / 'run-manifest.json').exists(), 'Existing snapshot must be preserved'
    for p in [RUN, OUT, OUT / 'logs', OUT / 'predictions', OUT / 'audits', CHECKPOINTS, RUN / 'source']:
        p.mkdir(parents=True, exist_ok=True)
    previous = json.loads((AGE_RUN / 'run-manifest.json').read_text())
    assert previous['status'] == 'complete' and previous['epoch15_reproduction_passed']
    with np.load(AGE_RUN / 'frozen_inputs.npz') as z:
        frozen_x, frozen_y = z['real'].copy(), z['label'].ravel().copy()
    columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    sensors = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    frame = pd.read_csv(ROOT / 'upstream/data/train_FD001.txt', sep=r'\s+', header=None, names=columns)
    scaled = MinMaxScaler().fit_transform(frame[sensors])
    xs, ys, ids, cycles = [], [], [], []
    for engine, g in frame.groupby('id', sort=True):
        v = scaled[(frame.id == engine).to_numpy()]
        end = g.cycle.to_numpy()[47:]
        xs.append(np.lib.stride_tricks.sliding_window_view(v, 48, axis=0).transpose(0, 2, 1))
        ys.extend(g.cycle.max() - end)
        ids.extend([int(engine)] * len(end))
        cycles.extend(end)
    real = np.concatenate(xs).astype(np.float32)
    uncapped = np.asarray(ys, dtype=np.int64)
    y = np.minimum(uncapped, 125).astype(np.float32)
    ids, cycles = np.asarray(ids), np.asarray(cycles)
    assert np.array_equal(real, frozen_x) and np.array_equal(y, frozen_y)
    engines = np.unique(ids)
    assert len(engines) == 100 and real.shape == (15931, 48, 14)
    selected = np.sort(np.random.RandomState(20261030).permutation(engines)[:30])
    vi = np.flatnonzero(np.isin(ids, selected))
    with np.load(ROOT / 'results/fd001_w48_public_code_split.npz') as z:
        ti, di = z['predictive_train'].copy(), z['predictive_dev'].copy()
    assert len(ti) == 11151 and len(di) == 4780
    assert np.array_equal(np.sort(np.concatenate([ti, di])), np.arange(len(y)))
    np.savez_compressed(RUN / 'frozen_plan.npz', train_indices=ti, dev_indices=di,
                        real_indices=vi, validation_engines=selected,
                        other_engines=np.setdiff1d(engines, selected))
    np.savez_compressed(RUN / 'real_validation.npz', data=real[vi], label=y[vi],
                        uncapped_rul=uncapped[vi], engine_id=ids[vi], end_cycle=cycles[vi],
                        original_window_index=vi)
    config = yaml.safe_load((ROOT / 'configs/evaluation_fd001_w48_public_code.yaml').read_text())
    sys.path.insert(0, str(ROOT / 'upstream'))
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        from args import args
    finally:
        sys.argv = argv
    resolved = vars(args).copy()
    resolved.update(dataset='FD001', window_size=48, input_size=14, output_size=1, device='cuda',
                    model_name='DiffUnet_fre', eva_epoch=int(config['evaluation']['evaluator_epochs']))
    resolved = {k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
                for k, v in resolved.items()}
    c = dict(epoch_order=EPOCHS, main_comparison=[15, 70], auxiliary=64, evaluator_seeds=SEEDS,
             generator_seed=23, sampling_seed=103, no_new_generation=True,
             predictor_args=resolved, synthetic_train_n=len(ti), synthetic_dev_n=len(di),
             max_epochs=35, max_updates_per_model=35 * math.ceil(len(ti) / 256),
             checkpoint_rule='ORIGINAL best synthetic-dev RMSE, zero-based epoch>5; patience10',
             same_rule_not_forced_same_actual_updates=True, validation_split_seed=20261030,
             validation_engines=selected.tolist(), validation_windows=len(vi),
             bins=BINS, primary='sqrt(mean_over_predictors(mean_over_engines(mean_window_squared_error)))',
             clear_gain=dict(min_relative_overall_reduction=.05, min_paired_improvements=4),
             budget=15, no_official_test_access=True,
             limitation='Generator and training MinMax saw validation engines; downstream exploratory validation only')
    dump(RUN / 'config.json', c)
    paths = [Path(__file__), PROTOCOL, ROOT / 'SOURCE_SNAPSHOT.md',
             ROOT / 'upstream/eva_regressor.py', ROOT / 'upstream/models/LSTM.py',
             ROOT / 'upstream/utils.py', ROOT / 'upstream/args.py',
             ROOT / 'upstream/data/train_FD001.txt', ROOT / 'results/fd001_w48_public_code_split.npz',
             ROOT / 'configs/evaluation_fd001_w48_public_code.yaml',
             AGE_RUN / 'run-manifest.json', AGE_RUN / 'frozen_inputs.npz',
             RUN / 'config.json', RUN / 'frozen_plan.npz', RUN / 'real_validation.npz']
    fingerprints = []
    for epoch in EPOCHS:
        p = sample_path(epoch)
        assert sha(p) == previous['outputs_sha256'][str(p.relative_to(ROOT))]
        with np.load(p) as z:
            assert z['data'].shape == real.shape and np.isfinite(z['data']).all()
            assert np.array_equal(z['label'].ravel(), y)
        record = AGE / f'epoch{epoch}/retain_condition/sample_checkpoint23_seed103.json'
        match = AGE / f'epoch{epoch}/noise_anchor_match.json'
        for q in [record, match]:
            assert sha(q) == previous['outputs_sha256'][str(q.relative_to(ROOT))]
        fingerprints.append(json.loads(match.read_text()))
        paths.extend([p, record, match])
    dump(RUN / 'data_preflight.json', dict(raw_reconstruction_bit_identical=True,
         shape=list(real.shape), validation_windows=len(vi), bins={name: int(((y[vi] >= lo) & (y[vi] <= hi)).sum())
         for name, lo, hi in BINS}, prior_noise_matches=fingerprints))
    for p in [Path(__file__), PROTOCOL]:
        q = RUN / 'source' / p.name
        q.write_bytes(p.read_bytes())
        paths.append(q)
    (RUN / 'workspace.diff').write_bytes(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT))
    (RUN / 'git-status.txt').write_bytes(subprocess.check_output(['git', 'status', '--short'], cwd=ROOT))
    (RUN / 'environment-freeze.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    paths += [RUN / 'workspace.diff', RUN / 'git-status.txt', RUN / 'environment-freeze.txt', RUN / 'data_preflight.json']
    dump(RUN / 'run-manifest.json', dict(run_id=RUN.name, scope_class='diagnostic', role='supporting',
         scope_contract='scope-20261030-fd001-checkpoint-downstream-validation', status='prepared',
         prepared_at=now(), changed_factor='generator checkpoint epoch15 versus70; auxiliary64',
         user_authorization='Explicit fixed-data five-seed downstream validation, epoch64 auxiliary',
         command=[sys.executable, '-u', str(Path(__file__)), 'run'], config=c,
         inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
         budget=15, completed_models=0, no_official_test_access=True,
         environment_prepare=dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__),
                                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())))
    update_state('2026-10-30 FD001 checkpoint downstream validation prepared',
        '- User authorizes fixed generated datasets from rerun epochs15/70 (main) and64 (auxiliary), five paired predictor seeds each. Diagnostic/supporting; strict main/A/B unchanged, C deferred, numeric-zeroing repair paused, EXT-002 closed/weight1.\n'
        '- [Observed] Existing full15931-window sampling103 datasets/labels/noise records verified; raw training reconstruction equals frozen tensor. Freeze30 real training engines before fitting, all their windows; generator/MinMax already saw them, so downstream exploratory validation only. No official test access.\n'
        '- [Code] Original predictor architecture/train function/config/synthetic split/35-epoch maximum/best synthetic-dev checkpoint/early stopping retained. Audit identical initialization and consumed train-order prefixes; actual update counts may differ under the same original stopping rule. Budget15 fits, no generator training/sampling or automatic further seeds.\n'
        f'- Snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`; protocol `{PROTOCOL.relative_to(ROOT)}`; output `{OUT.relative_to(ROOT)}`. Metrics not measured yet; paper gap not applicable.')
    print(json.dumps(dict(status='prepared', validation_engines=selected.tolist(), validation_windows=len(vi)), indent=2))


def model_hash(model):
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def predict(model, x, batch=2048):
    import numpy as np
    import torch
    model.eval()
    with torch.inference_mode():
        out = np.concatenate([model(x[i:i + batch]).cpu().numpy().ravel() for i in range(0, len(x), batch)])
    assert np.isfinite(out).all()
    return out.astype(float)


def metrics(prediction, y, engine_ids, meta, domain):
    import numpy as np
    rows, perengine = [], []
    for name, lo, hi in BINS:
        mask = (y >= lo) & (y <= hi)
        assert mask.any()
        e = prediction[mask] - y[mask]
        r = dict(**meta, domain=domain, bin=name, n_windows=int(mask.sum()),
                 mse_window=float(np.mean(e ** 2)), rmse_window=float(np.sqrt(np.mean(e ** 2))),
                 mae_window=float(np.mean(abs(e))), bias_window=float(np.mean(e)))
        if engine_ids is not None:
            mse = []
            for engine in np.unique(engine_ids[mask]):
                take = mask & (engine_ids == engine)
                errors = prediction[take] - y[take]
                value = float(np.mean(errors ** 2))
                mse.append(value)
                perengine.append(dict(**meta, domain=domain, bin=name, engine_id=int(engine),
                                      n_windows=int(take.sum()), mse=value, rmse=float(np.sqrt(value)),
                                      bias=float(errors.mean())))
            r.update(n_engines=len(mse), mse_engine=float(np.mean(mse)), rmse_engine=float(np.sqrt(np.mean(mse))))
        rows.append(r)
    return rows, perengine


def fit(eva, args, generated, real_x, real_y, ti, di, epoch, seed):
    """Passive observers around the unchanged public training entry point."""
    import numpy as np
    import torch
    from torch.utils.data import RandomSampler
    checkpoint = CHECKPOINTS / f'epoch{epoch}_eval{seed}.pth'
    args.save_path = str(checkpoint)
    audit = dict(epoch=epoch, evaluator_seed=seed, actual_updates=0, completed_epochs=0,
                 selected_update=None, initialization_sha256=None)
    orders, history, saves = [], [], []
    original_train, original_eval = eva.train, eva.evaluate
    original_step, original_iter, original_save = torch.optim.Adam.step, RandomSampler.__iter__, torch.save
    state = {}

    def observed_train(config, model, train_iter, dev_iter, test_iter=None):
        audit['initialization_sha256'] = model_hash(model)
        state.update(sampler=train_iter.sampler, dev=dev_iter, real=test_iter)
        return original_train(config, model, train_iter, dev_iter, test_iter)

    def observed_step(optimizer, *a, **kw):
        value = original_step(optimizer, *a, **kw)
        audit['actual_updates'] += 1
        counters = [int(s['step'].item()) for s in optimizer.state.values() if 'step' in s]
        assert counters and set(counters) == {audit['actual_updates']}
        return value

    def observed_iter(sampler):
        collect = sampler is state.get('sampler')
        indices = []
        for index in original_iter(sampler):
            if collect:
                indices.append(index)
            yield index
        if collect:
            assert np.array_equal(np.sort(indices), np.arange(len(ti)))
            orders.append(np.asarray(indices, dtype=np.int32))
            audit['completed_epochs'] = len(orders)

    def observed_eval(model, loader):
        value = original_eval(model, loader)
        if loader is state.get('dev'):
            history.append(dict(update=audit['actual_updates'], synthetic_dev_rmse=float(value[0])))
        return value

    def observed_save(obj, path, *a, **kw):
        value = original_save(obj, path, *a, **kw)
        if str(path) == str(checkpoint):
            audit['selected_update'] = audit['actual_updates']
            saves.append(dict(update=audit['actual_updates'], synthetic_dev_rmse=history[-1]['synthetic_dev_rmse']))
        return value

    eva.train, eva.evaluate = observed_train, observed_eval
    torch.optim.Adam.step, RandomSampler.__iter__, torch.save = observed_step, observed_iter, observed_save
    started = time.time()
    try:
        reported = eva.predictive_score_metrics(args, {'data': real_x, 'label': real_y}, generated,
                                                eval_seed=seed, split_indices=(ti, di))
    finally:
        eva.train, eva.evaluate = original_train, original_eval
        torch.optim.Adam.step, RandomSampler.__iter__, torch.save = original_step, original_iter, original_save
    assert checkpoint.exists() and audit['selected_update'] is not None
    assert len(orders) <= 35 and audit['actual_updates'] == len(orders) * math.ceil(len(ti) / 256)
    order_array = np.stack(orders)
    np.savez_compressed(OUT / 'audits' / f'epoch{epoch}_eval{seed}_orders.npz', orders=order_array)
    model = torch.load(checkpoint, map_location='cuda', weights_only=False).eval()
    vp = predict(model, real_x.to('cuda'))
    sp = predict(model, torch.as_tensor(generated['data'], device='cuda'))
    y = real_y.cpu().numpy().ravel().astype(float)
    replay = float(np.sqrt(np.mean((vp - y) ** 2)))
    assert abs(replay - float(reported[0])) < 1e-5
    selected = [x for x in history if x['update'] > 6 * math.ceil(len(ti) / 256)]
    assert abs(min(x['synthetic_dev_rmse'] for x in selected) - saves[-1]['synthetic_dev_rmse']) < 1e-8
    audit.update(elapsed_seconds=time.time() - started, checkpoint=str(checkpoint.relative_to(ROOT)),
                 checkpoint_sha256=sha(checkpoint), replay_rmse_window=replay,
                 selected_epoch_one_based=math.ceil(audit['selected_update'] / math.ceil(len(ti) / 256)),
                 consumed_order_sha256=hashlib.sha256(ti[order_array].astype('<i8').tobytes()).hexdigest(),
                 history=history, checkpoint_saves=saves)
    dump(OUT / 'audits' / f'epoch{epoch}_eval{seed}.json', audit)
    np.savez_compressed(OUT / 'predictions' / f'epoch{epoch}_eval{seed}.npz', real_validation=vp, synthetic=sp)
    del model
    torch.cuda.empty_cache()
    return audit, vp, sp


def summarize(audits, rows, perengine):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seg = pd.DataFrame(rows)
    real = seg[seg.domain == 'real_validation']
    comparisons = []
    for name, _, _ in BINS:
        group = real[real.bin == name]
        for measure in ['engine', 'window']:
            pivot = group.pivot(index='evaluator_seed', columns='epoch', values=f'rmse_{measure}').reindex(SEEDS)
            for later in [70, 64]:
                delta = pivot[later] - pivot[15]
                old = float(np.sqrt(group[group.epoch == 15][f'mse_{measure}'].mean()))
                new = float(np.sqrt(group[group.epoch == later][f'mse_{measure}'].mean()))
                comparisons.append(dict(bin=name, measure=measure, contrast=f'{later}-15',
                    baseline_pooled_rmse=old, later_pooled_rmse=new, pooled_change=new-old,
                    relative_change=new/old-1, paired_mean_rmse_change=float(delta.mean()),
                    paired_change_sd=float(delta.std(ddof=1)), n_improving=int((delta < 0).sum()),
                    n_windows=int(group.n_windows.iloc[0]), n_engines=int(group.n_engines.iloc[0])))
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(OUT / 'comparison.csv', index=False)
    summary = real.groupby(['epoch', 'bin']).agg(mean_rmse_engine=('rmse_engine', 'mean'),
        evaluator_sd_engine=('rmse_engine', 'std'), mean_rmse_window=('rmse_window', 'mean'),
        pooled_mse_engine=('mse_engine', 'mean'), pooled_mse_window=('mse_window', 'mean')).reset_index()
    summary['pooled_rmse_engine'] = np.sqrt(summary.pop('pooled_mse_engine'))
    summary['pooled_rmse_window'] = np.sqrt(summary.pop('pooled_mse_window'))
    summary.to_csv(OUT / 'summary.csv', index=False)
    primary = next(r for r in comparisons if r['bin'] == 'all' and r['measure'] == 'engine' and r['contrast'] == '70-15')
    clear = bool(primary['relative_change'] <= -.05 and primary['n_improving'] >= 4)
    dump(OUT / 'interpretation.json', dict(primary=primary, clear_downstream_gain=clear,
        threshold='at least5% pooled engine-balanced RMSE decrease and4/5 paired improvements',
        not_statistical_significance=True, generator_seed_count=1, sampling_seed_count=1,
        no_default_checkpoint_rule_selected=True, generator_saw_validation_engines=True))
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    plot_bins = ['all', '0-30', '31-60', '61-90', '91-125']
    x = np.arange(len(plot_bins))
    for offset, epoch in enumerate([15, 64, 70]):
        values = summary[summary.epoch == epoch].set_index('bin').loc[plot_bins]
        ax.bar(x + (offset-1)*.24, values.pooled_rmse_engine, width=.24, label=f'Epoch {epoch}')
    ax.set_xticks(x, plot_bins)
    ax.set_ylabel('Real-validation engine-balanced pooled RMSE')
    ax.set_ylim(bottom=0)
    ax.set_title('FD001: fixed generated samples, five paired predictor seeds')
    ax.legend()
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'downstream_validation.{ext}', dpi=180)
    plt.close(fig)
    lines = ['# FD001检查点选择：真实验证集下游对照', '',
        '[Observed] 固定同一次seed23生成器重跑的第15/70轮，64轮仅为辅助；复用原sampling103三批数据，无生成器重训/重新采样。15个原配置预测器全部完成。', '',
        '[Observed] 验证集从真实训练100台发动机中按固定随机种子20261030选30台，使用全部窗口；生成器和MinMax已经见过这些发动机。因此仅是下游探索性验证，不是整条流程的独立验证。未读取原真实测试数据或使用测试结果挑轮次。', '',
        '[Code] 使用未修改的eva_regressor训练函数、原11151/4780生成train/dev划分、Adam.002/批256/最高35轮/原学习率衰减、原生成dev选模和早停。真实验证只供原test_iter参数报告，不参与训练、选模、早停。不同组实际训练长度可因原早停规则而不同，详见审计。', '',
        '## 主要对照：70相对15', '',
        'RMSE口径：每台发动机先平均窗口平方误差，再对发动机及5个预测器平均，最后开方。不是先平均预测值的集成误差。', '',
        '| 真实RUL段 | 第15轮 RMSE | 第70轮 RMSE | 差值70−15 | 相对变化 | 配对改善数 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in comparisons:
        if r['measure'] == 'engine' and r['contrast'] == '70-15':
            lines.append(f"| {r['bin']} | {r['baseline_pooled_rmse']:.4f} | {r['later_pooled_rmse']:.4f} | {r['pooled_change']:+.4f} | {r['relative_change']:+.2%} | {r['n_improving']}/5 |")
    lines += ['', f'[Observed] 预先固定的实用明显改善标志（总体至少改善5%、至少4/5种子改善）={clear}；这不是统计显著性检验。所有分段退步仍完整报告。', '',
              '## 第64轮辅助对照', '', '| RUL段 | 第64轮 RMSE | 相对第15轮 | 配对改善数 |', '|---|---:|---:|---:|']
    for r in comparisons:
        if r['measure'] == 'engine' and r['contrast'] == '64-15':
            lines.append(f"| {r['bin']} | {r['later_pooled_rmse']:.4f} | {r['relative_change']:+.2%} | {r['n_improving']}/5 |")
    lines += ['', '## 训练审计', '', '| 生成轮次 | 预测器种子 | 实际训练轮数 | 更新次数 | 选中轮次 | 选中更新 |', '|---|---:|---:|---:|---:|---:|']
    for a in audits:
        lines.append(f"| {a['epoch']} | {a['evaluator_seed']} | {a['completed_epochs']} | {a['actual_updates']} | {a['selected_epoch_one_based']} | {a['selected_update']} |")
    lines += ['', '[Observed] 所有配对预测器初始参数完全相同、已消耗训练索引顺序的公共前缀完全相同；更新计数与Adam状态核对，保存模型在同一真实窗口上回放及RMSE重算通过。原始预测、每台发动机误差、生成训练/验证误差、常规窗口加权RMSE和各评价种子结果均保存。原文件、源代码和默认设置未改。', '',
        '[Hypothesis] 若70下游改善，支持选模是这条重跑轨迹下游差距的贡献因素；若没有改善，DS/条件读出改善尚不足以建立下游收益。单个生成训练种子、一个采样种子、5个预测器种子不能代表生成训练种子的总体规律；64辅助结果不改变预先指定的70对15主比较，也不自动确定新的选模规则。原始历史后期权重仍缺失。', '',
        '[Observed] 当前结果→论文无同设计验证指标→论文数值差距不适用→只描述条件于一批生成数据的5预测器波动→不改变严格复现结论。主实验/A/B保持原状态，C推迟，数值标签置零修复仍暂停，EXT-002保留weight1。15次预算结束，无后续自动实验。', '',
        f'[Observed] Run `{RUN.name}`；配置/快照 `{RUN.relative_to(ROOT)}/run-manifest.json`；协议 `{PROTOCOL.relative_to(ROOT)}`；结果 `{OUT.relative_to(ROOT)}`。']
    REPORT.write_text('\n'.join(lines) + '\n')
    return primary, clear


def run():
    import numpy as np
    import pandas as pd
    import torch
    from types import SimpleNamespace
    m = check_inputs()
    assert m['status'] == 'prepared', 'No silent resume or automatic retry'
    assert torch.cuda.is_available(), 'CUDA unavailable'
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT / 'upstream'))
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        import eva_regressor as eva
        import wandb
    finally:
        sys.argv = argv
    args = SimpleNamespace(**m['config']['predictor_args'])
    with np.load(RUN / 'frozen_plan.npz') as z:
        ti, di = z['train_indices'], z['dev_indices']
    with np.load(RUN / 'real_validation.npz') as z:
        vx = torch.from_numpy(z['data'].copy())
        y = z['label'].copy().astype(float)
        vy = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        ids = z['engine_id'].copy()
    audits, rows, perengine = [], [], []
    m.update(status='running', started_at=now(), gpu=torch.cuda.get_device_name(0),
             torch=str(torch.__version__), torch_cuda=torch.version.cuda)
    dump(RUN / 'run-manifest.json', m)
    wandb.init(project='fd001-checkpoint-downstream-diagnostic', mode='disabled')
    try:
        for epoch in EPOCHS:
            with np.load(sample_path(epoch)) as z:
                generated = dict(data=z['data'].copy(), label=z['label'].copy())
            sy = generated['label'].ravel().astype(float)
            for seed in SEEDS:
                name = f'epoch{epoch}_eval{seed}'
                m.update(current_epoch=epoch, current_evaluator_seed=seed)
                dump(RUN / 'run-manifest.json', m)
                print('START', name, now(), flush=True)
                log = OUT / 'logs' / f'{name}.log'
                import contextlib
                with log.open('w', buffering=1) as handle, contextlib.redirect_stdout(handle):
                    a, vp, sp = fit(eva, args, generated, vx, vy, ti, di, epoch, seed)
                meta = dict(epoch=epoch, evaluator_seed=seed)
                for domain, p, yy, engines in [('real_validation', vp, y, ids),
                    ('synthetic_train', sp[ti], sy[ti], None), ('synthetic_dev', sp[di], sy[di], None)]:
                    r, pe = metrics(p, yy, engines, meta, domain)
                    rows.extend(r)
                    perengine.extend(pe)
                for previous in [x for x in audits if x['evaluator_seed'] == seed]:
                    assert previous['initialization_sha256'] == a['initialization_sha256']
                    with np.load(OUT / 'audits' / f"epoch{previous['epoch']}_eval{seed}_orders.npz") as z:
                        old_order = z['orders']
                    with np.load(OUT / 'audits' / f'{name}_orders.npz') as z:
                        new_order = z['orders']
                    n = min(len(old_order), len(new_order))
                    assert np.array_equal(old_order[:n], new_order[:n]), 'Unmatched training order'
                audits.append(a)
                pd.DataFrame(rows).to_csv(OUT / 'per_model_segments.csv', index=False)
                pd.DataFrame(perengine).to_csv(OUT / 'per_engine.csv', index=False)
                pd.DataFrame([{k: v for k, v in x.items() if k not in ['history', 'checkpoint_saves']} for x in audits]).to_csv(OUT / 'training_audit.csv', index=False)
                m['completed_models'] = len(audits)
                dump(RUN / 'run-manifest.json', m)
                print('DONE', name, 'window_RMSE', a['replay_rmse_window'], 'seconds', a['elapsed_seconds'], flush=True)
        assert len(audits) == 15
        # Independent replay from persisted arrays, rather than trusting online summaries.
        replay_rows = []
        for a in audits:
            with np.load(OUT / 'predictions' / f"epoch{a['epoch']}_eval{a['evaluator_seed']}.npz") as z:
                p = z['real_validation']
            replay, _ = metrics(p, y, ids, dict(epoch=a['epoch'], evaluator_seed=a['evaluator_seed']), 'real_validation')
            replay_rows.extend(replay)
        original = pd.DataFrame(rows).query("domain == 'real_validation'").reset_index(drop=True)
        replay = pd.DataFrame(replay_rows)
        assert np.array_equal(original['rmse_window'], replay['rmse_window'])
        assert np.array_equal(original['rmse_engine'], replay['rmse_engine'])
        check_inputs()
        primary, clear = summarize(audits, rows, perengine)
        dump(OUT / 'closeout_audit.json', dict(status='passed', completed_models=15,
             inputs_unchanged=True, initialization_and_order_prefix_pairs_match=True,
             actual_optimizer_counters_checked=True, saved_prediction_metrics_recomputed=True,
             generator_saw_validation_engines=True, no_official_test_access=True))
        m.update(status='complete', finished_at=now(), primary_comparison=primary, clear_downstream_gain=clear,
                 checkpoint_sha256={a['checkpoint']: a['checkpoint_sha256'] for a in audits},
                 outputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in OUT.rglob('*') if p.is_file()})
        m['outputs_sha256'][str(REPORT.relative_to(ROOT))] = sha(REPORT)
        update_state('2026-10-30 FD001 checkpoint downstream validation completed',
            f'- [Observed] Run `{RUN.name}` complete15/15 original-rule predictors. Main70-vs15 real-validation engine-balanced pooled RMSE {primary["baseline_pooled_rmse"]:.6f} → {primary["later_pooled_rmse"]:.6f} ({primary["relative_change"]:+.2%}); paired improving {primary["n_improving"]}/5; clear practical gain={clear}.64 auxiliary and all segments reported.\n'
            '- [Observed] Fixed30 training-engine validation, generator/MinMax exposure disclosed; downstream exploration only, no official test access. Original initialization/order prefixes/optimizer count/source and saved-prediction metric audits passed; original best synthetic-dev predictor selection and early stopping retained. Budget15 exhausted, no automatic next seeds/default changes; strict status unchanged, zero-label repair paused, EXT-002weight1.\n'
            f'- Report `{REPORT.relative_to(ROOT)}`; snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`; results `{OUT.relative_to(ROOT)}`. Current diagnostic→no matching paper metric→paper gap not applicable→conditional five-predictor variation only→strict conclusions unchanged.')
        print(json.dumps(dict(primary=primary, clear_downstream_gain=clear), indent=2), flush=True)
    except BaseException:
        m.update(status='failed', finished_at=now(), error=traceback.format_exc())
        update_state('2026-10-30 FD001 checkpoint downstream validation failed',
                     f'- [Observed] `{RUN.name}` failed after {len(audits)}/15 models. Error retained in manifest/log; stop without automatic retries or rule changes. Strict results unchanged.')
        raise
    finally:
        wandb.finish()
        dump(RUN / 'run-manifest.json', m)


if __name__ == '__main__':
    environment()
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    arguments = parser.parse_args()
    prepare() if arguments.action == 'prepare' else run()
