"""Cached downstream errors, engine-equal signed direction and additive MSE."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/diagnostic-fd001-seed13-real-high-rul-direction-20261102'
OUT = ROOT / 'results/diagnostics/fd001_seed13_real_high_rul_direction_20261102'
SOURCE = ROOT / 'results/diagnostics/fd001_final70_two_seeds_20261030'
SOURCE_RUN = ROOT / 'runs/diagnostic-fd001-final70-two-seeds-20261030'
PROTOCOL = ROOT / 'docs/operations/FD001_SEED13_REAL_HIGH_RUL_DIRECTION.md'
REPORT = ROOT / 'docs/conclusions/final/fd001_seed13_real_high_rul_direction_20261102_report.md'
SEEDS = [3, 13, 23, 33, 43]
ARMS = [('best', 64), ('final70', 70)]
LEAVES = [('0-30', 0, 30), ('31-60', 31, 60), ('61-90', 61, 90), ('91-124', 91, 124), ('125', 125, 125)]
BINS = [('all', 0, 125), ('91-125', 91, 125), *LEAVES]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p, obj):
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    temp.replace(p)


def state(stage, body):
    p = ROOT / 'reproduction-state.md'
    head, tail = p.read_text().split('\n', 1)
    text = body+f'\n- Diagnostic/supporting `{RUN.name}`,scope `scope-20261102-fd001-seed13-real-high-rul-direction`; snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`; protocol `{PROTOCOL.relative_to(ROOT)}`. Cached real-downstream records only, no inference/training/sampling; numeric-zero repair paused, EXT-002weight1, strict main/A/B unchanged,C deferred.\n'
    p.write_text(head+f'\n\n## Latest checkpoint: 2026-11-02 seed13 real high-RUL direction {stage}\n\n'+text+'\n'+tail)
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
    old = json.loads((SOURCE_RUN / 'run-manifest.json').read_text())
    assert old['status'] == 'complete'
    bound = {**old['inputs_sha256'], **old['outputs_sha256'], **old['predictor_checkpoints_sha256']}
    paths = [Path(__file__), PROTOCOL, SOURCE_RUN / 'run-manifest.json', SOURCE_RUN / 'real_validation.npz',
        SOURCE_RUN / 'frozen_plan.npz', SOURCE / 'comparison.csv', SOURCE / 'seed13/training_audit.json']
    for rule, epoch in ARMS:
        base = SOURCE / f'seed13/{rule}'
        paths.extend([base / 'per_model_segments.csv', base / 'per_engine.csv'])
        for s in SEEDS:
            paths.extend([base / 'predictions' / f'epoch{epoch}_eval{s}.npz',
                          base / 'audits' / f'epoch{epoch}_eval{s}.json'])
            a = json.loads(paths[-1].read_text())
            paths.append(ROOT / a['checkpoint'])
    for p in paths:
        name = str(p.relative_to(ROOT))
        if name in bound:
            assert sha(p) == bound[name], name
        else:
            assert p in [Path(__file__), PROTOCOL, SOURCE_RUN / 'run-manifest.json'], name
    with np.load(SOURCE_RUN / 'real_validation.npz') as z:
        assert len(z['label']) == 4875 and len(np.unique(z['engine_id'])) == 30
        assert np.array_equal(np.minimum(z['uncapped_rul'], 125), z['label'])
    c = dict(generator_seed=13, generator_epochs=[64, 70], predictor_seeds=SEEDS, bins=BINS,
        leaf_bins=LEAVES, validation_n=4875, total_engines=30,
        signed_error='mean over engines of within-engine mean(pred-truth), then mean5 predictors',
        mse='mean over5 predictors and eligible engines of within-engine MSE; RMSE=sqrt(MSE)',
        contributions='parent per-window weight1/(eligible parent engines * engine parent window count)',
        parents=['91-125', 'all'], primary_bins=['91-124', '125'],
        budget='one cached10-prediction audit, no new inference/training/generation',
        no_automatic_followup=True, validation_generator_exposure=True)
    dump(RUN / 'config.json', c)
    paths.append(RUN / 'config.json')
    for p in [Path(__file__), PROTOCOL]:
        target = RUN / 'source' / p.name
        shutil.copy2(p, target)
        paths.append(target)
    for name, cmd in [('workspace.diff', ['git', 'diff', '--binary']), ('git-status.txt', ['git', 'status', '--short']),
        ('environment-freeze.txt', [sys.executable, '-m', 'pip', 'freeze'])]:
        p = RUN / name
        p.write_bytes(subprocess.check_output(cmd, cwd=ROOT))
        paths.append(p)
    dump(RUN / 'run-manifest.json', dict(run_id=RUN.name, scope_class='diagnostic', role='supporting',
        scope_contract='scope-20261102-fd001-seed13-real-high-rul-direction', status='prepared', prepared_at=now(), config=c,
        user_authorization='Explicit single cached real91-125 split91-124/125 with engine-equal signed errors and MSE change',
        command=[sys.executable, str(Path(__file__)), 'run'],
        inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(paths)}))
    state('prepared', '- [Observed] Verified same10 synthetic-trained predictors and exact4875 real validation windows from completed seed13 same-rerun64/70 comparison. Freeze engine-equal MSE/signed error and additive parent weights before arithmetic; both high-parent and ALL net changes will be reconciled. No new inference or training.')
    print('Prepared exact real-validation records and engine-equal/additive definitions.')


def run():
    import numpy as np
    import pandas as pd
    m = check_inputs()
    assert m['status'] == 'prepared'
    m.update(status='running', started_at=now())
    dump(RUN / 'run-manifest.json', m)
    try:
        with np.load(SOURCE_RUN / 'real_validation.npz') as z:
            y, ids = z['label'].astype(float), z['engine_id'].copy()
            cycles, global_ix = z['end_cycle'].copy(), z['original_window_index'].copy()
        pred = []
        for rule, epoch in ARMS:
            one = []
            for s in SEEDS:
                with np.load(SOURCE / f'seed13/{rule}/predictions/epoch{epoch}_eval{s}.npz') as z:
                    p = z['real_validation'].copy()
                    assert p.shape == (4875,) and np.isfinite(p).all()
                    one.append(p)
            pred.append(np.array(one))
        pred = np.array(pred)
        error = pred-y[None, None, :]
        masks, weights, individual, engine_rows, aggregates = {}, {}, [], [], []
        for name, lo, hi in BINS:
            mask = (y >= lo) & (y <= hi)
            engines = np.unique(ids[mask])
            w = np.zeros(len(y), dtype=float)
            for engine in engines:
                take = mask & (ids == engine)
                w[take] = 1/(len(engines)*take.sum())
                for ai, (_, epoch) in enumerate(ARMS):
                    for j, s in enumerate(SEEDS):
                        e = error[ai, j, take]
                        engine_rows.append(dict(epoch=epoch, predictor_seed=s, bin=name, engine_id=int(engine),
                            n_windows=int(take.sum()), mse=float(np.mean(e**2)), signed_error=float(e.mean())))
            assert abs(w.sum()-1) < 1e-12
            masks[name], weights[name] = mask, w
            for ai, (rule, epoch) in enumerate(ARMS):
                old = pd.read_csv(SOURCE / f'seed13/{rule}/per_model_segments.csv')
                mse = (error[ai]**2*w[None, :]).sum(axis=1)
                bias = (error[ai]*w[None, :]).sum(axis=1)
                for j, s in enumerate(SEEDS):
                    one = old[(old.domain == 'real_validation') & (old.evaluator_seed == s) & (old.bin == name)]
                    assert len(one) == 1 and abs(float(one.mse_engine.iloc[0])-mse[j]) < 1e-8
                    individual.append(dict(epoch=epoch, predictor_seed=s, bin=name, n_windows=int(mask.sum()),
                        n_engines=len(engines), mse=float(mse[j]), rmse=float(np.sqrt(mse[j])), signed_error=float(bias[j])))
                aggregates.append(dict(epoch=epoch, bin=name, n_windows=int(mask.sum()), n_engines=len(engines),
                    mse=float(mse.mean()), rmse=float(np.sqrt(mse.mean())), signed_error=float(bias.mean())))
        individual = pd.DataFrame(individual)
        agg = pd.DataFrame(aggregates)
        paired = []
        for name, _, _ in BINS:
            a = agg[(agg.epoch == 64) & (agg.bin == name)].iloc[0]
            b = agg[(agg.epoch == 70) & (agg.bin == name)].iloc[0]
            x = individual[(individual.epoch == 64) & (individual.bin == name)].set_index('predictor_seed').loc[SEEDS]
            z = individual[(individual.epoch == 70) & (individual.bin == name)].set_index('predictor_seed').loc[SEEDS]
            paired.append(dict(bin=name, n_windows=int(a.n_windows), n_engines=int(a.n_engines),
                rmse64=float(a.rmse), rmse70=float(b.rmse), signed64=float(a.signed_error), signed70=float(b.signed_error),
                signed_shift=float(b.signed_error-a.signed_error), mse64=float(a.mse), mse70=float(b.mse),
                mse_change=float(b.mse-a.mse), n_predictors_mse_higher=int((z.mse>x.mse).sum()),
                n_predictors_signed_more_negative=int((z.signed_error<x.signed_error).sum())))
        delta_sq = error[1]**2-error[0]**2
        contribution = []
        for parent in ['91-125', 'all']:
            children = LEAVES[-2:] if parent == '91-125' else LEAVES
            total = next(x['mse_change'] for x in paired if x['bin'] == parent)
            values = []
            for name, _, _ in children:
                v = float((delta_sq*weights[parent][None, :]*masks[name][None, :]).sum(axis=1).mean())
                values.append(v)
                contribution.append(dict(parent=parent, child_bin=name, mse_change_contribution=v,
                    parent_mse_change=total, share_of_parent_net_change=v/total))
            assert abs(sum(values)-total) < 1e-9, (parent, values, total)
        agg.to_csv(OUT / 'aggregate_metrics.csv', index=False)
        individual.to_csv(OUT / 'per_predictor_metrics.csv', index=False)
        pd.DataFrame(engine_rows).to_csv(OUT / 'per_engine_metrics.csv', index=False)
        pd.DataFrame(paired).to_csv(OUT / 'paired_metrics.csv', index=False)
        pd.DataFrame(contribution).to_csv(OUT / 'mse_contributions.csv', index=False)
        np.savez_compressed(OUT / 'fixed_predictions_and_weights.npz', predictions=pred, label=y, engine_id=ids,
            end_cycle=cycles, original_window_index=global_ix, predictor_seeds=np.array(SEEDS), epochs=np.array([64, 70]),
            **{'weight_'+k:v for k,v in weights.items()})
        lines = ['# seed13真实高RUL下游误差：方向及MSE贡献', '',
            '[Observed] 本轮只复用既有seed13第64/70轮生成数据训练的10个预测器，在完全相同4875真实验证窗口上的保存预测；无新增训练、生成或推理。输入文件与原运行哈希匹配。', '',
            '[Code] 各RUL段先对每台发动机内部窗口计算均方误差和有向误差，再对该段有窗口的发动机等权、对5个预测器等权；RMSE为所得MSE开方。有向误差为预测−封顶真实标签，未使用旧表窗口等权bias，未先集成预测。', '',
            '## 用户指定的两个寿命段', '',
            '| 真实RUL段 | 发动机/窗口数 | RMSE：64→70轮 | 平均预测−真实：64→70轮 | MSE：64→70轮 | MSE变化 |',
            '|---|---:|---:|---:|---:|---:|']
        for a in paired:
            if a['bin'] in ['91-124', '125']:
                lines.append(f'| {a["bin"]} | {a["n_engines"]}/{a["n_windows"]} | {a["rmse64"]:.4f}→{a["rmse70"]:.4f} | {a["signed64"]:+.4f}→{a["signed70"]:+.4f} | {a["mse64"]:.4f}→{a["mse70"]:.4f} | {a["mse_change"]:+.4f} |')
        lines += ['', '## 五个预测器分别结果', '', '| RUL段 | 预测器种子 | RMSE64→70 | 有向误差64→70 | MSE变化 |', '|---|---:|---:|---:|---:|']
        for name in ['91-124', '125']:
            for s in SEEDS:
                a = individual[(individual.epoch == 64) & (individual.bin == name) & (individual.predictor_seed == s)].iloc[0]
                b = individual[(individual.epoch == 70) & (individual.bin == name) & (individual.predictor_seed == s)].iloc[0]
                lines.append(f'| {name} | {s} | {a.rmse:.4f}→{b.rmse:.4f} | {a.signed_error:+.4f}→{b.signed_error:+.4f} | {b.mse-a.mse:+.4f} |')
        lines += ['', '## 精确可加的MSE贡献', '',
            '[Code] 两段的独立MSE重新给各自发动机等权，因此不能直接相加。贡献保留父段每台发动机的原权重，以及该发动机在子段的窗口比例：C_b(P)=五预测器平均Σ子段窗口[1/(父段发动机数×该发动机父段窗口数)]×(70轮平方误差−64轮平方误差)。父段不同，贡献不同。', '',
            '| 父集合 | 子段 | 对父MSE变化的贡献 | 占父MSE净变化 |', '|---|---|---:|---:|']
        for a in contribution:
            lines.append(f'| {a["parent"]} | {a["child_bin"]} | {a["mse_change_contribution"]:+.4f} | {a["share_of_parent_net_change"]:+.2%} |')
        for parent in ['91-125', 'all']:
            a = next(x for x in paired if x['bin'] == parent)
            lines.append(f'\n[Observed] 父集合{parent}：RMSE{a["rmse64"]:.4f}→{a["rmse70"]:.4f}；MSE{a["mse64"]:.4f}→{a["mse70"]:.4f}，净变化{a["mse_change"]:+.4f}。对应子段贡献精确加和通过。')
        lines += ['', '[Code] 对整体净退化贡献可超过100%，因为其他寿命段的改善抵消了一部分退步；这是同权重下的误差算术分解，不是因果归因。', '',
            '## 解释与边界', '',
            '[Hypothesis] 若91–124均方误差增加且有向误差更负，与前次生成信号被真实预测器读成更年轻、其指定标签相对偏低、下游对真实窗口低估这一方向一致。它仍不能证明因果链或每个样本均统一偏移。若没有更负，则该统一低估解释不足。具体五预测器方向由表中数值与配对CSV判定。', '',
            '[Hypothesis] 125须单独判断贡献：其条件读出改善与真实下游退步可以同时发生；其他段标签错配的间接影响、125覆盖或下游学习等仍需后续区分。本轮只做已授权检查，不自动启动传感器或干预实验。', '',
            '[Observed] 真值125为封顶标签，误差不是相对未截断RUL；原30台验证发动机已被生成器/MinMax见过，仍属下游探索。五预测器不是五次生成器训练重复；结果来自同一生成器和固定采样。当前诊断→论文无相同设计指标→论文数值差距不适用→配对五预测器的条件方向→严格复现结论不变。无网络/选模默认变化，数值置零修复暂停、EXT-002weight1、主实验/A/B不变，C推迟。', '',
            f'[Observed] Run `{RUN.name}`；快照`{RUN.relative_to(ROOT)}/run-manifest.json`；协议`{PROTOCOL.relative_to(ROOT)}`；结果`{OUT.relative_to(ROOT)}`。']
        REPORT.write_text('\n'.join(lines)+'\n')
        check_inputs()
        dump(OUT / 'audit.json', dict(status='passed', source_hashes_verified=True, same_real_windows_labels=True,
            source_engine_equal_mse_reproduced=True, bias_engine_equal_not_window_equal=True,
            mse_parent_contribution_closure=True, cached_predictors=10, new_training_sampling_inference=0))
        m.update(status='complete', completed_at=now(), paired_metrics=paired, contributions=contribution,
            outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file()})
        m['outputs_sha256'][str(REPORT.relative_to(ROOT))] = sha(REPORT)
        dump(RUN / 'run-manifest.json', m)
        text = '; '.join(f'{a["bin"]}:RMSE{a["rmse64"]:.4f}→{a["rmse70"]:.4f},signed{a["signed64"]:+.4f}→{a["signed70"]:+.4f},deltaMSE{a["mse_change"]:+.4f},more-negative{a["n_predictors_signed_more_negative"]}/5' for a in paired if a['bin'] in ['91-124','125'])
        state('completed', '- [Observed] '+text+'. Exact same4875 windows/30 engines,10 cached predictions; original engine-equal MSE replay and both parent contribution closure passed.\n'+f'- Report `{REPORT.relative_to(ROOT)}`. No new inference/training/sampling or automatic follow-up.')
        print(pd.DataFrame(paired).to_string(index=False))
        print(pd.DataFrame(contribution).to_string(index=False))
    except BaseException:
        m.update(status='failed', failed_at=now(), error=traceback.format_exc())
        dump(RUN / 'run-manifest.json', m)
        state('failed', '- [Observed] Cached-record audit failed; preserve error/artifacts, no automatic replacement or model runs.')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else run()
