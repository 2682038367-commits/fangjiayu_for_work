"""Bounded seed43 full-training reproducibility diagnostic; no default changes."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/diagnostic-fd002-fixed-conv-full70-20261105'
OUT = ROOT / 'results/diagnostics/fd002_fixed_conv_full70_20261105'
WEIGHTS = ROOT / 'checkpoints/diagnostics/fd002_fixed_conv_full70_20261105'
PARENT = ROOT / 'runs/diagnostic-fd002-final70-two-seeds-20261102'
OLD = ROOT / 'results/diagnostics/fd002_final70_two_seeds_20261102'
PROTOCOL = ROOT / 'docs/operations/FD002_FIXED_CONV_FULL70.md'


def helper():
    spec = importlib.util.spec_from_file_location('frozen_fd002_helper', RUN / 'source/run_fd002_final70_two_seeds.py')
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    h.RUN = RUN
    h.environment()
    return h


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def stamp(stage, body):
    text = f'\n## {helper().now()} FD002 fixed-convolution full70 {stage}\n\n{body}\n'
    p = ROOT / 'reproduction-state.md'
    head, tail = p.read_text().split('\n', 1)
    p.write_text(head + '\n' + text + '\n' + tail)
    with (ROOT / 'experiment-ledger.md').open('a') as f:
        f.write(text)


def flags(torch):
    return dict(benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic,
                global_strict=torch.are_deterministic_algorithms_enabled(),
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32)


def prepare():
    import torch
    assert not RUN.exists(), 'No overwrite'
    for p in [RUN / 'source', OUT / 'logs', WEIGHTS]:
        p.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / 'scripts/run_fd002_final70_two_seeds.py', PROTOCOL]
    for p in sources:
        shutil.copy2(p, RUN / 'source' / p.name)
    shutil.copy2(PARENT / 'run-manifest.json', RUN / 'source/parent-run-manifest.json')
    os.link(PARENT / 'frozen_inputs.npz', RUN / 'frozen_inputs.npz')
    shutil.copy2(OLD / 'noise_anchor.json', OUT / 'noise_anchor.json')
    h = helper()
    parent = json.loads((PARENT / 'run-manifest.json').read_text())
    # Keep the source closure and datasets frozen from the existing audited run.
    inherited = {k: v for k, v in parent['inputs_sha256'].items()
                 if not k.startswith(str(PARENT.relative_to(ROOT)) + '/')}
    paths = list((RUN / 'source').iterdir()) + [RUN / 'frozen_inputs.npz', OUT / 'noise_anchor.json']
    for rule in ['original', 'phase']:
        paths += [OLD / f'seed43/{rule}/evaluation_checkpoint43_sample103.csv',
                  OLD / f'seed43/{rule}/sample_checkpoint43_seed103.json']
    for rep in ['repeat1', 'repeat2']:
        (OUT / rep / 'seed43').mkdir(parents=True)
        (WEIGHTS / rep / 'seed43').mkdir(parents=True)
    for p in ['samples', 'logs', 'audits']:
        (OUT / 'repeat1/seed43/phase' / p).mkdir(parents=True)
    shutil.copy2(OUT / 'noise_anchor.json', OUT / 'repeat1/noise_anchor.json')
    inputs = dict(inherited)
    inputs.update({str(p.relative_to(ROOT)): h.sha(p) for p in paths})
    for name, expected in inputs.items():
        assert h.sha(ROOT / name) == expected, name
    config = dict(original_best_epochs={'43': 67}, generator_seed=43, training_repetitions=2,
        epochs=70, updates_per_run=11410, compare_epoch=67,
        changed_factor='Training cuDNN convolution path selection only: benchmark=False, deterministic=True',
        global_strict=False, sampling_mode='unchanged native benchmark=True/deterministic=False',
        sampling_seed=103, sample_count=41539, evaluator_seeds=[3, 13, 23, 33, 43],
        primary_gate='All 11410 recorded losses, all70 epoch parameter/optimizer/RNG hashes, orders, draws, learning-rate histories, best/67/70 named tensors exactly equal',
        stop='If full-training exact-repeat gate fails, stop before sampling; no automatic reruns or other seeds',
        evaluation_budget='One epoch67 dataset and five discriminator fits only after repeat gate passes; no downstream fits',
        limitation='Historical trajectories unavailable. This cannot uniquely attribute the full historical DS gap; unchanged native sampler can retain kernel-path variability.')
    save(RUN / 'run-manifest.json', dict(run_id=RUN.name, status='prepared', scope_class='diagnostic', role='supporting',
        scope_contract='scope-20261105-fd002-fixed-conv-full70', prepared_at=h.now(), config=config,
        command=[sys.executable, '-u', str(Path(__file__).resolve()), 'run'],
        user_authorization='Explicit 验证一下 after actual CUDA kernel-path comparison; bounded full-training repeatability and matched epoch67 discrimination diagnostic',
        inputs_sha256=inputs, git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        torch=str(torch.__version__), python=sys.version, completed_training_runs=0, completed_sampling=0,
        completed_discriminator_fits=0))
    for filename, command in [('environment-freeze.txt', [sys.executable, '-m', 'pip', 'freeze']),
                              ('workspace.diff', ['git', 'diff', '--binary']), ('git-status.txt', ['git', 'status', '--short'])]:
        (RUN / filename).write_bytes(subprocess.check_output(command, cwd=ROOT))
    stamp('prepared', '- [Observed] Two independent seed43 trainings frozen at70epochs/11410updates each. Only training convolution selection changes. Reuse original data/orders/labels/hyperparameters. Save best/67/70; generate once and evaluate five times only if exact-repeat gate passes. No downstream or extra seed training.\n- Snapshot `' + str(RUN.relative_to(ROOT)) + '/run-manifest.json`; protocol `' + str(PROTOCOL.relative_to(ROOT)) + '`. Diagnostic/supporting; strict main/A/B unchanged,C deferred; FD003/4 and125 analyses remain paused. Metrics pending→no paper target→gapNA→repeatability unknown→strict conclusions unchanged.')


def train(rep):
    import numpy as np
    import torch
    h = helper()
    h.OUT, h.WEIGHTS = OUT / rep, WEIGHTS / rep
    sys.path.insert(0, str(ROOT / 'upstream'))
    import utils
    original_seed = utils.torch_seed
    def seed(seed_value):
        original_seed(seed_value)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        assert not torch.are_deterministic_algorithms_enabled()
    utils.torch_seed = seed
    original_step = torch.optim.Adam.step
    rows, epochs = [], []
    def digest_state(value):
        digest = hashlib.sha256()
        def visit(v):
            if torch.is_tensor(v):
                digest.update(str((v.dtype, tuple(v.shape))).encode())
                digest.update(v.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(v, dict):
                for k in sorted(v, key=str):
                    digest.update(str(k).encode()); visit(v[k])
            elif isinstance(v, (list, tuple)):
                for x in v: visit(x)
            else:
                digest.update(repr(v).encode())
        visit(value)
        return digest.hexdigest()
    def step(opt, *a, **kw):
        actual = flags(torch)
        assert actual['benchmark'] is False and actual['deterministic'] is True and actual['global_strict'] is False
        frame = sys._getframe(1)
        while Path(frame.f_code.co_filename).name != 'TrainCondition.py':
            frame = frame.f_back
            assert frame is not None
        loss = float(frame.f_locals['loss'].detach().item())
        model = frame.f_locals['net_model']
        result = original_step(opt, *a, **kw)
        rows.append(dict(update=len(rows)+1, loss=loss))
        if len(rows) % 163 == 0:
            epochs.append(dict(epoch=len(rows)//163, updates=len(rows), model_sha256=h.tensor_hash(model.state_dict()),
                optimizer_sha256=digest_state(opt.state_dict()),
                cpu_rng_sha256=digest_state(torch.get_rng_state()), cuda_rng_sha256=digest_state(torch.cuda.get_rng_state()),
                numpy_rng_sha256=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(), flags=actual))
            save(h.OUT / 'seed43/epoch_state_hashes.json', epochs)
            print('FIXED_TRAIN', rep, 'epoch', len(rows)//163, 'of70', flush=True)
        return result
    torch.optim.Adam.step = step
    try:
        h.train(43)
    finally:
        torch.optim.Adam.step = original_step
        utils.torch_seed = original_seed
        if rows:
            h.write_csv(h.OUT / 'seed43/update_losses.csv', rows)
    assert len(rows) == 11410 and len(epochs) == 70
    auditpath = h.OUT / 'seed43/training_audit.json'
    audit = json.loads(auditpath.read_text())
    audit.update(training_convolution_flags=flags(torch),
                 caveat='Fixed convolution selection; global strict mode remains disabled. Full-repeat claim depends on independent repeat comparison.')
    save(auditpath, audit)


def compare():
    import numpy as np
    import torch
    h = helper()
    h.check_inputs()
    paths = [OUT / rep / 'seed43' for rep in ['repeat1', 'repeat2']]
    audits = [json.loads((p / 'training_audit.json').read_text()) for p in paths]
    checks = {}
    for name in ['epoch_state_hashes.json', 'update_losses.csv', 'training_batch_zero_draws.csv', 'training_loss.csv']:
        checks[name] = paths[0].joinpath(name).read_bytes() == paths[1].joinpath(name).read_bytes()
    with np.load(paths[0] / 'training_epoch_orders.npz') as a, np.load(paths[1] / 'training_epoch_orders.npz') as b:
        checks['all70_training_orders'] = np.array_equal(a['orders'], b['orders'])
    checks['initial_parameters'] = audits[0]['initial_tensor_sha256'] == audits[1]['initial_tensor_sha256']
    checks['best_selection'] = audits[0]['rerun_best_epoch_one_based'] == audits[1]['rerun_best_epoch_one_based']
    for rule in ['best', 'phase', 'final70']:
        states = [torch.load(ROOT / a['rules'][rule]['weight_path'], map_location='cpu', weights_only=True) for a in audits]
        checks[rule + '_all_named_tensors'] = states[0].keys() == states[1].keys() and all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    e = [json.loads((p / 'epoch_state_hashes.json').read_text()) for p in paths]
    first = next((i+1 for i in range(70) if e[0][i] != e[1][i]), None)
    result = dict(passed=all(checks.values()), checks=checks, first_different_epoch=first,
                  observed_epochs=70, observed_updates_each=11410,
                  limitation='Two current processes on this environment only; not a cross-device/global determinism guarantee or proof of unique historical attribution.')
    save(OUT / 'full_training_repeatability.json', result)
    return result


def report(status):
    import pandas as pd
    h = helper()
    gate = json.loads((OUT / 'full_training_repeatability.json').read_text())
    rows = []
    for label, path in [('historic_epoch67', OLD / 'seed43/original'), ('native_rerun_epoch67', OLD / 'seed43/phase'),
                        ('fixed_repeat1_epoch67', OUT / 'repeat1/seed43/phase')]:
        p = path / 'evaluation_checkpoint43_sample103.csv'
        if not p.exists(): continue
        f = pd.read_csv(p).sort_values('evaluator_seed')
        assert f.evaluator_seed.tolist() == [3,13,23,33,43]
        rows.append(dict(checkpoint=label, mean_DS=float(f.discriminative_score.mean()),
                         scores=f.discriminative_score.tolist(), source=str(p.relative_to(ROOT))))
    save(OUT / 'summary.json', dict(status=status, repeatability=gate, discriminative_comparison=rows))
    p = ROOT / 'docs/conclusions/final/fd002_fixed_conv_full70_20261105_report.md'
    text = '# FD002 seed43固定卷积路径完整训练验证\n\n'
    text += f'- [Observed] 完整70轮重复检查通过：{gate["passed"]}。每次11410次更新；逐更新损失、逐轮完整参数/优化器状态/RNG、训练顺序、学习率及best/67/70权重对比见结果。\n'
    text += '- [Code] 只把训练卷积设置改为benchmark=False、deterministic=True；全局strict=False，保留原median、数据、网络、阈值、条件丢弃、优化器及70轮预算。采样设置与已有对照相同。\n'
    for r in rows:
        text += f'- [Observed] {r["checkpoint"]}: DS={r["mean_DS"]:.9f}；五次={r["scores"]}。来源 `{r["source"]}`。\n'
    text += '- [Hypothesis] 完整训练可重复若成立，支持固定卷积路径消除了本环境这两次训练的已测分叉；不能唯一解释历史判别差距，也不能保证分数必然改善。原始历史训练中间状态未保存；沿用的native采样仍可能存在数值路径差异。没有新增下游验证。\n'
    text += '- 诊断结果→无论文对应目标→差距NA→仅两次本环境重复→严格复现结论不变。预算到此停止，不自动改默认配置、扩种子或运行下游。\n'
    text += f'- Run `{RUN.relative_to(ROOT)}/run-manifest.json`；结果 `{OUT.relative_to(ROOT)}/summary.json`。\n'
    p.write_text(text)
    stamp(status, f'- [Observed] Full-training repeatability gate={gate["passed"]}; two70epoch runs complete. Report `{p.relative_to(ROOT)}`; summary `{OUT.relative_to(ROOT)}/summary.json`. Fixed diagnostic budget closed; no automatic additional training. Strict main/A/B unchanged,C deferred; other datasets/125 analyses paused.')


def run():
    h = helper()
    m = h.check_inputs()
    assert m['status'] == 'prepared'
    m.update(status='running', started_at=h.now())
    save(RUN / 'run-manifest.json', m)
    try:
        import torch
        assert torch.cuda.is_available()
        save(RUN / 'gpu-environment.json', dict(torch=str(torch.__version__), cudnn=torch.backends.cudnn.version(),
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
            nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True, errors='replace')))
        for rep in ['repeat1', 'repeat2']:
            m['current_stage'] = 'train_' + rep
            save(RUN / 'run-manifest.json', m)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'train', rep]
            with (OUT / 'logs' / f'train_{rep}.log').open('w') as f:
                subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=f, stderr=subprocess.STDOUT, check=True)
            m['completed_training_runs'] += 1
            save(RUN / 'run-manifest.json', m)
        gate = compare()
        if gate['passed']:
            for action in ['sample', 'evaluate']:
                m['current_stage'] = action + '_epoch67'
                save(RUN / 'run-manifest.json', m)
                with (OUT / 'logs' / f'{action}_epoch67.log').open('w') as f:
                    subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), action], cwd=ROOT,
                                   env=os.environ.copy(), stdout=f, stderr=subprocess.STDOUT, check=True)
                m['completed_sampling' if action == 'sample' else 'completed_discriminator_fits'] = 1 if action == 'sample' else 5
            m['status'] = 'complete'
        else:
            m['status'] = 'complete_repeatability_failed_sampling_stopped'
        h.check_inputs()
        report(m['status'])
        m.update(completed_at=h.now(), current_stage='finished')
        m['outputs_sha256'] = {str(p.relative_to(ROOT)): h.sha(p) for p in OUT.rglob('*') if p.is_file()}
        m['checkpoint_sha256'] = {str(p.relative_to(ROOT)): h.sha(p) for p in WEIGHTS.rglob('*.pth')}
        save(RUN / 'run-manifest.json', m)
    except BaseException:
        m.update(status='failed', failed_at=h.now(), failure=traceback.format_exc())
        save(RUN / 'run-manifest.json', m)
        stamp('failed', '- [Observed] Stopped without automatic retry; see run manifest and per-stage logs. Existing artifacts preserved.')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'train', 'sample', 'evaluate'])
    parser.add_argument('repeat', nargs='?', choices=['repeat1', 'repeat2'])
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'run': run()
    elif args.action == 'train':
        assert args.repeat
        train(args.repeat)
    else:
        import torch
        h = helper()
        h.OUT, h.WEIGHTS = OUT / 'repeat1', WEIGHTS / 'repeat1'
        # Fresh subprocess: preserve the historical native sampler, changing training only.
        torch.backends.cudnn.deterministic = False
        assert not torch.are_deterministic_algorithms_enabled()
        (h.sample if args.action == 'sample' else h.evaluate)(43, 'phase')
