"""Forward-only fixed exam for seven existing FD001 generator checkpoints."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/diagnostic-fd001-fixed-denoising-exam-20261102'
OUT = ROOT / 'results/diagnostics/fd001_fixed_denoising_exam_20261102'
PROTOCOL = ROOT / 'docs/operations/FD001_FIXED_DENOISING_EXAM.md'
REPORT = ROOT / 'docs/conclusions/final/fd001_fixed_denoising_exam_20261102_report.md'
AGE = ROOT / 'results/diagnostics/fd001_seed23_checkpoint_age_20261030'
TWO = ROOT / 'results/diagnostics/fd001_final70_two_seeds_20261030'
PRIOR = ROOT / 'runs/diagnostic-fd001-checkpoint-downstream-validation-20261030'
DOMAINS = ['train', 'internal_validation']
QUESTIONS = [1103, 1113, 1123]
N = 4096
BATCH = 256


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def state_hash(state):
    h = hashlib.sha256()
    for k, t in sorted(state.items()):
        h.update(k.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def environment():
    for k, v in dict(WANDB_MODE='disabled', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MPLCONFIGDIR=str(ROOT / '.cache/matplotlib')).items():
        os.environ.setdefault(k, v)


def check_inputs():
    m = json.loads((RUN / 'run-manifest.json').read_text())
    for name, expected in m['inputs_sha256'].items():
        assert sha(ROOT / name) == expected, name
    return m


def checkpoint_inventory():
    import numpy as np
    import pandas as pd
    rows = []
    summary = pd.read_csv(ROOT / 'results/diagnostics/fd001_checkpoint_downstream_validation_20261030/summary.csv')
    comparison = pd.read_csv(TWO / 'comparison.csv')
    for seed, epochs in [(23, [15, 64, 70]), (13, [64, 70]), (3, [43, 70])]:
        if seed == 23:
            history_path = AGE / 'training_loss.csv'
        else:
            history_path = TWO / f'seed{seed}/training_loss.csv'
        history = np.loadtxt(history_path, delimiter=',', skiprows=1)
        for epoch in epochs:
            if seed == 23:
                snapshot = AGE / f'training_snapshot_epoch{epoch}.json'
                d = json.loads(snapshot.read_text())
                weight, expected = ROOT / d['weight_path'], d['weight_sha256']
                downstream = float(summary[(summary.epoch == epoch) & (summary.bin == 'all')].pooled_rmse_engine.iloc[0])
                origin = ROOT / 'runs/diagnostic-fd001-seed23-checkpoint-age-20261030/run-manifest.json'
                baseline_epoch = 15
            else:
                snapshot = TWO / f'seed{seed}/training_audit.json'
                a = json.loads(snapshot.read_text())
                rule = 'final70' if epoch == 70 else 'best'
                d = a['rules'][rule]
                weight, expected = ROOT / d['weight_path'], d['weight_sha256']
                subset = comparison[(comparison.generator_seed == seed) & (comparison.bin == 'all') & (comparison.measure == 'engine')]
                downstream = float(subset['final70_rmse' if rule == 'final70' else 'best_rmse'].iloc[0])
                origin = ROOT / 'runs/diagnostic-fd001-final70-two-seeds-20261030/run-manifest.json'
                baseline_epoch = a['rules']['best']['epoch_one_based']
            assert sha(weight) == expected
            assert int(history[epoch-1, 0]) == epoch-1
            rows.append(dict(checkpoint=f'seed{seed}_epoch{epoch}', generator_seed=seed, epoch=epoch,
                baseline_epoch=baseline_epoch, weight_path=str(weight.relative_to(ROOT)), weight_sha256=expected,
                source_training_history=str(history_path.relative_to(ROOT)), source_record=str(snapshot.relative_to(ROOT)),
                source_manifest=str(origin.relative_to(ROOT)), recorded_training_loss=float(history[epoch-1, 1]),
                downstream_rmse=downstream))
    assert len(rows) == 7
    return rows


def update_state(stage, body):
    p = ROOT / 'reproduction-state.md'
    head, tail = p.read_text().split('\n', 1)
    message = body + f'\n- Diagnostic/supporting `{RUN.name}`,scope `scope-20261102-fd001-fixed-denoising-exam`. Snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`; protocol `{PROTOCOL.relative_to(ROOT)}`. Experiment2 explicitly deferred; no generator training/sampling/downstream fits or default changes; numeric-zero repair paused, EXT-002weight1, strict main/A/B unchanged,C deferred.\n'
    p.write_text(head+f'\n\n## Latest checkpoint: 2026-11-02 fixed denoising exam {stage}\n\n'+message+'\n'+tail)
    with (ROOT / 'experiment-ledger.md').open('a') as f:
        f.write(f'\n## `{RUN.name}` {stage}\n\n'+message)


def prepare():
    import numpy as np
    import torch
    assert not (RUN / 'run-manifest.json').exists(), 'Do not overwrite an existing experiment'
    for p in [RUN, RUN / 'questions', RUN / 'source', OUT, OUT / 'per_window']:
        p.mkdir(parents=True, exist_ok=True)
    checkpoints = checkpoint_inventory()
    source_data = ROOT / 'runs/diagnostic-fd001-seed23-checkpoint-age-20261030/frozen_inputs.npz'
    old = json.loads((PRIOR / 'run-manifest.json').read_text())
    for name in ['real_validation.npz', 'frozen_plan.npz']:
        assert sha(PRIOR/name) == old['inputs_sha256'][str((PRIOR/name).relative_to(ROOT))]
    shutil.copy2(source_data, RUN / 'frozen_inputs.npz')
    shutil.copy2(PRIOR / 'frozen_plan.npz', RUN / 'previous_frozen_plan.npz')
    with np.load(source_data) as z:
        x, y = z['real'].copy(), z['label'].copy()
    with np.load(PRIOR / 'frozen_plan.npz') as z:
        vi = z['real_indices'].copy()
        validation_engines = z['validation_engines'].copy()
    with np.load(PRIOR / 'real_validation.npz') as z:
        assert np.array_equal(z['data'], x[vi]) and np.array_equal(z['label'], y[vi].ravel())
    train_pool = np.setdiff1d(np.arange(len(y)), vi)
    assert len(train_pool) == 11056 and len(vi) == 4875
    # Sample independently of losses/predictions, then keep the realized ordering for every exam.
    g = torch.Generator(device='cpu').manual_seed(20261102)
    chosen = dict(train=train_pool[torch.randperm(len(train_pool), generator=g).numpy()[:N]],
                  internal_validation=vi[torch.randperm(len(vi), generator=g).numpy()[:N]])
    assert not np.intersect1d(chosen['train'], chosen['internal_validation']).size
    np.savez_compressed(RUN / 'fixed_windows.npz', **chosen, validation_engines=validation_engines)
    for q in QUESTIONS:
        generator = torch.Generator(device='cpu').manual_seed(q)
        numeric = np.random.RandomState(q)
        pack = {}
        for domain in DOMAINS:
            pack[domain+'_timestep'] = torch.randint(0, 1000, (N,), generator=generator).numpy()
            pack[domain+'_noise'] = torch.randn((N, 14, 48), generator=generator).numpy()
            pack[domain+'_keep'] = (torch.rand((N,), generator=generator) < .8).numpy()
            draws = numeric.rand(N//BATCH)
            pack[domain+'_numeric_zero_draw'] = draws
            pack[domain+'_numeric_zero'] = draws < .1
        np.savez_compressed(RUN / 'questions' / f'questions_{q}.npz', **pack)
    config = dict(checkpoints=checkpoints, question_seeds=QUESTIONS, window_selection_seed=20261102,
        n_windows_per_domain=N, batch_size=BATCH, n_batches_per_domain=N//BATCH,
        domains=DOMAINS, pool_counts=dict(train=11056, internal_validation=4875),
        original_loss='unchanged GaussianDiffusion.forward, x*2-1, pred_noise/L2, linear1000, p2gamma0; equal batch means',
        original_numeric_batch_zero_probability=.1, original_embedding_drop_probability=.2,
        aggregation='mean original batch scalar across16 equal batches; mean/sd across3 saved question packs',
        downstream='existing engine-balanced pooled MSE across30 engines/five predictors then root',
        budget=dict(checkpoints=7, packs=3, domains=2, formal_evaluations=42, formal_batch_forwards=672),
        no_independent_generalization='Both real pools were seen by generator/MinMax',
        experiment2='explicitly deferred by user', no_training_or_sampling=True,
        stop='42 fixed evaluations or any audit failure, no retry/extra question packs/selection tuning')
    dump(RUN / 'config.json', config)
    paths = [Path(__file__), PROTOCOL, RUN / 'config.json', RUN / 'fixed_windows.npz', RUN / 'frozen_inputs.npz',
        RUN / 'previous_frozen_plan.npz', PRIOR / 'run-manifest.json', PRIOR / 'real_validation.npz',
        ROOT / 'SOURCE_SNAPSHOT.md', ROOT / 'upstream/GaussianDiffusion.py', ROOT / 'upstream/TrainCondition.py',
        ROOT / 'results/diagnostics/fd001_checkpoint_downstream_validation_20261030/summary.csv', TWO / 'comparison.csv']
    paths.extend((ROOT / 'upstream/DiffusionFreeGuidence').glob('*.py'))
    paths.extend((RUN / 'questions').glob('*.npz'))
    for c in checkpoints:
        paths.extend(ROOT / c[k] for k in ['weight_path', 'source_training_history', 'source_record', 'source_manifest'])
    for p in [Path(__file__), PROTOCOL]:
        target = RUN / 'source' / p.name
        shutil.copy2(p, target)
        paths.append(target)
    for name, command in [('workspace.diff', ['git', 'diff', '--binary']), ('git-status.txt', ['git', 'status', '--short']),
        ('environment-freeze.txt', [sys.executable, '-m', 'pip', 'freeze'])]:
        target = RUN / name
        target.write_bytes(subprocess.check_output(command, cwd=ROOT))
        paths.append(target)
    dump(RUN / 'run-manifest.json', dict(run_id=RUN.name, scope_class='diagnostic', role='supporting',
        scope_contract='scope-20261102-fd001-fixed-denoising-exam', status='prepared', prepared_at=now(),
        command=[sys.executable, '-u', str(Path(__file__)), 'run'], config=config,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        user_authorization='Explicit seven saved checkpoints/three fixed denoising exams, no updates; experiment2 deferred',
        inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(paths)}, completed_evaluations=0))
    update_state('prepared', '- [Observed] Prepared7 exact source-rerun weights,3 complete saved noise/timestep/keep/numeric-zero packs,4096 fixed train and4096 existing internal-validation windows. All16 batches equal256; both pools generator-seen. No new metric measured yet; only experiment1 authorized to execute now.')
    print('Prepared7 checkpoints,3 persisted question packs,42 forward-only evaluations.', flush=True)


class FixedDraws:
    """Supply saved draws to the unchanged original diffusion forward."""
    def __init__(self, torch, unet, diffusion, x, y, timestep, noise, keep, numeric_zero):
        self.torch, self.unet, self.diffusion = torch, unet, diffusion
        self.x, self.y = x, y
        self.timestep, self.noise, self.keep = timestep, noise, keep
        self.labels = torch.zeros_like(y) if numeric_zero else y
        self.counts = dict(timestep=0, noise=0, mask=0, model=0)
        self.old = torch.randint, torch.randn_like, unet.prob_mask_like
        self.prediction = None

    def randint(self, *a, **kw):
        assert a == (0, 1000, (len(self.x),)) and str(kw['device']).startswith('cuda')
        self.counts['timestep'] += 1
        return self.timestep

    def randn_like(self, x, *a, **kw):
        assert not a and not kw and x.shape == self.noise.shape
        self.counts['noise'] += 1
        return self.noise

    def mask(self, shape, prob, device):
        assert tuple(shape) == (len(self.x),) and abs(prob-.8) < 1e-12
        self.counts['mask'] += 1
        return self.keep

    def before_model(self, model, args):
        xt, t, labels = args
        assert self.torch.equal(t, self.timestep) and self.torch.equal(labels, self.labels)
        norm = self.x*2-1
        expected = self.diffusion.sqrt_alphas_cumprod[t][:, None, None]*norm + self.diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None]*self.noise
        assert self.torch.equal(xt, expected), 'Noisy/normalized inputs differ from fixed question'
        self.counts['model'] += 1

    def after_model(self, model, args, prediction):
        self.prediction = prediction.detach()

    def __enter__(self):
        self.torch.randint, self.torch.randn_like, self.unet.prob_mask_like = self.randint, self.randn_like, self.mask
        self.pre = self.diffusion.model.register_forward_pre_hook(self.before_model)
        self.post = self.diffusion.model.register_forward_hook(self.after_model)
        return self

    def __exit__(self, *a):
        self.torch.randint, self.torch.randn_like, self.unet.prob_mask_like = self.old
        self.pre.remove()
        self.post.remove()

    def execute(self):
        torch = self.torch
        before = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
        value = self.diffusion(self.x, classes=self.labels)
        assert all(c == 1 for c in self.counts.values()), self.counts
        per_window = (self.prediction-self.noise).square().mean(dim=(1, 2))
        per_window = per_window*self.diffusion.p2_loss_weight[self.timestep]
        assert torch.allclose(value, per_window.mean(), rtol=1e-7, atol=1e-9)
        assert torch.equal(before[0], torch.get_rng_state()) and torch.equal(before[1], torch.cuda.get_rng_state())
        return float(value.item()), per_window.cpu().numpy()


def run():
    import numpy as np
    import pandas as pd
    import torch
    m = check_inputs()
    assert m['status'] == 'prepared', 'No silent retry/resume'
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(ROOT / 'upstream'))
    import DiffusionFreeGuidence.Unet1D_fre as unet
    from GaussianDiffusion import GaussianDiffusion1D_cls_free
    with np.load(RUN / 'frozen_inputs.npz') as z:
        all_x, all_y = z['real'].copy(), z['label'].copy()
    with np.load(RUN / 'fixed_windows.npz') as z:
        indices = {d: z[d].copy() for d in DOMAINS}
    data = {d: (torch.from_numpy(all_x[indices[d]]).permute(0, 2, 1).contiguous().cuda(),
                torch.from_numpy(all_y[indices[d]]).cuda()) for d in DOMAINS}
    packs = {}
    for q in QUESTIONS:
        with np.load(RUN / 'questions' / f'questions_{q}.npz') as z:
            packs[q] = {k: torch.from_numpy(z[k].copy()).cuda() for k in z.files}
    m.update(status='running', started_at=now(), gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
        torch_cuda=torch.version.cuda, deterministic_cudnn=True, cudnn_benchmark=False)
    dump(RUN / 'run-manifest.json', m)
    rows, batches, audits = [], [], []
    try:
        for c in m['config']['checkpoints']:
            model = unet.UNet1D_fre(dim=32, dim_mults=(1, 2), cond_drop_prob=.2, channels=14, length=48,
                frequency_mask_mode='hard_random_quantile', frequency_threshold_init=.25, frequency_mask_temperature=.1)
            model.load_state_dict(torch.load(ROOT / c['weight_path'], map_location='cpu', weights_only=True), strict=True)
            model = model.cuda().eval()
            assert all(not sub.training for sub in model.modules())
            assert not any(isinstance(sub, torch.nn.Dropout) and sub.p != 0 for sub in model.modules())
            assert not any(isinstance(sub, torch.nn.modules.batchnorm._BatchNorm) for sub in model.modules())
            diffusion = GaussianDiffusion1D_cls_free(model, seq_length=48, channels=14, timesteps=1000,
                objective='pred_noise', beta_schedule='linear').cuda().eval()
            assert torch.equal(diffusion.p2_loss_weight, torch.ones_like(diffusion.p2_loss_weight))
            before = state_hash(model.state_dict())
            saved_losses = np.zeros((3, 2, N), dtype=np.float32)
            with torch.inference_mode():
                if not rows:
                    values = []
                    for _ in range(2):
                        p = packs[QUESTIONS[0]]
                        with FixedDraws(torch, unet, diffusion, data['train'][0][:BATCH], data['train'][1][:BATCH],
                            p['train_timestep'][:BATCH], p['train_noise'][:BATCH], p['train_keep'][:BATCH],
                            bool(p['train_numeric_zero'][0])) as fixed:
                            v, per = fixed.execute()
                        values.append(v)
                    assert values[0] == values[1]
                    dump(OUT / 'preflight.json', dict(status='passed', repeated_original_forward_identical=True,
                        original_loss_matches_actual_noise_mse=True, saved_draws_consumed_exactly=True,
                        numeric_and_embedding_masks_separate=True, no_rng_draws_during_forward=True, loss=values[0]))
                for qi, q in enumerate(QUESTIONS):
                    p = packs[q]
                    for di, domain in enumerate(DOMAINS):
                        x, y = data[domain]
                        batch_values = []
                        for bi, start in enumerate(range(0, N, BATCH)):
                            stop = start+BATCH
                            with FixedDraws(torch, unet, diffusion, x[start:stop], y[start:stop],
                                p[domain+'_timestep'][start:stop], p[domain+'_noise'][start:stop],
                                p[domain+'_keep'][start:stop], bool(p[domain+'_numeric_zero'][bi])) as fixed:
                                value, per = fixed.execute()
                            saved_losses[qi, di, start:stop] = per
                            batch_values.append(value)
                            batches.append(dict(checkpoint=c['checkpoint'], question_seed=q, domain=domain,
                                batch_index=bi, n_windows=BATCH, loss=value,
                                numeric_zero=bool(p[domain+'_numeric_zero'][bi])))
                        result = float(np.mean(batch_values))
                        assert abs(result-float(saved_losses[qi, di].astype(float).mean())) < 1e-8
                        rows.append(dict(checkpoint=c['checkpoint'], generator_seed=c['generator_seed'], epoch=c['epoch'],
                            question_seed=q, domain=domain, n_windows=N, n_batches=N//BATCH, loss=result))
                        pd.DataFrame(rows).to_csv(OUT / 'fixed_losses.csv', index=False)
                        m.update(completed_evaluations=len(rows), current_checkpoint=c['checkpoint'],
                            current_question_seed=q, current_domain=domain)
                        dump(RUN / 'run-manifest.json', m)
                        print(c['checkpoint'], q, domain, f'{result:.9f}', flush=True)
            assert state_hash(model.state_dict()) == before
            assert all(p.grad is None for p in model.parameters())
            np.savez_compressed(OUT / 'per_window' / (c['checkpoint']+'.npz'), losses=saved_losses,
                                question_seeds=np.array(QUESTIONS), domains=np.array(DOMAINS))
            audits.append(dict(checkpoint=c['checkpoint'], strict_load=True, model_state_before=before,
                model_state_after=before, gradients_absent=True, no_updates=True, exam_forward_count=96))
            del diffusion, model
            torch.cuda.empty_cache()
        assert len(rows) == 42 and len(batches) == 672 and len(audits) == 7
        pd.DataFrame(batches).to_csv(OUT / 'batch_losses.csv', index=False)
        dump(OUT / 'weight_immutability.json', audits)
        check_inputs()
        summarize(m, rows)
    except BaseException:
        m.update(status='failed', failed_at=now(), error=traceback.format_exc())
        dump(RUN / 'run-manifest.json', m)
        update_state('failed', f'- [Observed] Failed after{len(rows)}/42 fixed evaluations; see manifest/error. Stop without automatic retry or changes; no model updates occurred.')
        raise


def summarize(m, rows):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw = pd.DataFrame(rows)
    raw = raw.pivot(index=['checkpoint', 'question_seed'], columns='domain', values='loss').reset_index()
    table = []
    for c in m['config']['checkpoints']:
        a = raw[raw.checkpoint == c['checkpoint']]
        table.append(dict(**c, fixed_train_loss=float(a.train.mean()), fixed_train_sd=float(a.train.std(ddof=1)),
            fixed_validation_loss=float(a.internal_validation.mean()), fixed_validation_sd=float(a.internal_validation.std(ddof=1))))
    pd.DataFrame(table).to_csv(OUT / 'checkpoint_table.csv', index=False)
    contrasts, rankings = [], []
    for seed, epochs in [(23, [15, 64, 70]), (13, [64, 70]), (3, [43, 70])]:
        for domain in DOMAINS:
            for q in QUESTIONS:
                selected = [r for r in rows if r['generator_seed'] == seed and r['domain'] == domain and r['question_seed'] == q]
                ordered = sorted(selected, key=lambda r: r['loss'])
                rankings.append(dict(generator_seed=seed, domain=domain, question_seed=q,
                                     epoch_ranking=' < '.join(str(x['epoch']) for x in ordered)))
            base = f'seed{seed}_epoch{epochs[0]}'
            for epoch in epochs[1:]:
                late = f'seed{seed}_epoch{epoch}'
                before = raw[raw.checkpoint == base].set_index('question_seed')[domain].loc[QUESTIONS]
                after = raw[raw.checkpoint == late].set_index('question_seed')[domain].loc[QUESTIONS]
                delta = after-before
                contrasts.append(dict(generator_seed=seed, domain=domain, baseline_epoch=epochs[0], late_epoch=epoch,
                    mean_loss_change=float(delta.mean()), relative_loss_change=float(after.mean()/before.mean()-1),
                    n_packs_later_lower=int((delta < 0).sum()), n_packs_later_higher=int((delta > 0).sum()),
                    delta_q1103=float(delta.loc[1103]), delta_q1113=float(delta.loc[1113]), delta_q1123=float(delta.loc[1123])))
    pd.DataFrame(contrasts).to_csv(OUT / 'within_seed_contrasts.csv', index=False)
    pd.DataFrame(rankings).to_csv(OUT / 'rankings.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, seed in zip(axes, [23, 13, 3]):
        a = pd.DataFrame([r for r in table if r['generator_seed'] == seed])
        ax.errorbar(a.epoch, a.fixed_train_loss, yerr=a.fixed_train_sd, marker='o', label='Fixed train')
        ax.errorbar(a.epoch, a.fixed_validation_loss, yerr=a.fixed_validation_sd, marker='s', label='Fixed internal validation')
        ax.scatter(a.epoch, a.recorded_training_loss, marker='x', label='Recorded online loss')
        ax.set_title(f'Generator seed{seed}')
        ax.set_xlabel('Saved epoch')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Original diffusion noise-MSE loss')
    fig.savefig(OUT / 'fixed_denoising_losses.png', dpi=180)
    fig.savefig(OUT / 'fixed_denoising_losses.pdf')
    plt.close(fig)
    lines = ['# FD001固定去噪题目比较：实验一', '',
        '[Observed] 七个现有检查点、三个事先保存的随机题目集全部完成；无生成器或预测器训练、无参数更新、无生成采样。实验二按用户要求暂不启动。', '',
        '[Code] 原forward损失公式不变：输入x*2−1、linear1000、pred_noise/L2、p2gamma0、通道/时间均方再平均窗口。调用原forward，注入保存的t/噪声/条件keep掩码，并保留固定的原数值整批置零.1和条件embedding丢弃.2；没有将无条件embedding与数字RUL0混为一谈。', '',
        '[Observed] 从原70台非验证发动机窗口和既有30台验证发动机窗口中，各固定均匀抽取4096窗口。每组16批×256，无末批权重差；3套题目seed1103/1113/1123相同用于全部权重，全部随机数组在任何评价之前保存并绑定哈希。两组发动机均被生成器/MinMax见过，不属于未见发动机泛化验证。', '',
        '## 用户要求的汇总表', '',
        '固定损失为三套题目的均值±样本标准差；原训练损失来自该权重对应的同次重跑该轮日志。已有下游RMSE按同一30台/4875真实窗口、5个预测器发动机等权平方误差平均后开方，未重新训练。', '',
        '| 检查点 | 原训练损失 | 固定训练窗口损失 | 固定内部验证窗口损失 | 已有下游RMSE |',
        '|---|---:|---:|---:|---:|']
    for r in table:
        lines.append(f'| {r["checkpoint"]} | {r["recorded_training_loss"]:.7f} | {r["fixed_train_loss"]:.7f} ± {r["fixed_train_sd"]:.7f} | {r["fixed_validation_loss"]:.7f} ± {r["fixed_validation_sd"]:.7f} | {r["downstream_rmse"]:.4f} |')
    lines += ['', '## 三套题目的排序', '', '| 生成种子 | 窗口组 | 题目种子 | 损失由低到高的轮次 |', '|---|---|---|---|']
    for r in rankings:
        lines.append(f'| {r["generator_seed"]} | {r["domain"]} | {r["question_seed"]} | {r["epoch_ranking"]} |')
    lines += ['', '## 同种子配对差异', '', '| 种子 | 对比 | 窗口组 | 固定损失相对变化 | 后期更低的题目数 |', '|---|---|---|---:|---:|']
    for r in contrasts:
        lines.append(f'| {r["generator_seed"]} | {r["late_epoch"]}−{r["baseline_epoch"]} | {r["domain"]} | {r["relative_loss_change"]:+.3%} | {r["n_packs_later_lower"]}/3 |')
    lines += ['', '## 审计、解释范围与停止', '',
        '[Observed] 原forward与实际输出/目标噪声独立重算的MSE一致；第一批重复两次完全一致。每批恰消耗一份已保存t/noise/keep，噪声输入/标签核对通过，执行中CPU/CUDA RNG无变化。7个模型严格加载并处于eval，前后全部state张量哈希一致，所有参数无梯度，无优化器/反传。保存42组逐窗口损失、672批损失和全部题目，供独立重算。', '',
        '[Hypothesis] 后期固定损失也更低时，支持在线训练日志未可靠排序冻结权重；原选中权重固定损失更低、下游反而更差时，支持去噪目标与下游用途的不一致。若种子/题目/窗口组情况不同，应报告混合或不确定；不强行归于单一原因。在线日志混合了不同随机题目、更新前后参数和原末批；本实验不能唯一拆分这些贡献。', '',
        '[Observed] 三套题目只描述固定数据/模型下的题目随机性，未估计生成训练种子总体方差。不在这七个已观察点上调选模阈值或自动采用新默认。实验一预算结束，不自动加题/重训/采样/实验二；数值置零修复仍暂停，EXT-002weight1，严格主实验/A/B原状态、C推迟。', '',
        '[Observed] 当前诊断→论文无同设计目标→论文数值差距不适用→三套条件固定题目而非独立训练重复→严格复现结论不变。', '',
        f'[Observed] Run `{RUN.name}`；快照`{RUN.relative_to(ROOT)}/run-manifest.json`；协议`{PROTOCOL.relative_to(ROOT)}`；全部结果`{OUT.relative_to(ROOT)}`。']
    REPORT.write_text('\n'.join(lines)+'\n')
    dump(OUT / 'closeout_audit.json', dict(status='passed', no_parameter_updates=True, no_training_or_sampling=True,
        checkpoint_count=7, question_packs=3, domains=2, completed_evaluations=42, formal_batch_forwards=672,
        fixed_inputs_verified=True, original_loss_parity_passed=True, all_model_states_unchanged=True, experiment2_executed=False))
    m.update(status='complete', completed_at=now(), completed_evaluations=42,
        outputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in OUT.rglob('*') if p.is_file()})
    m['outputs_sha256'][str(REPORT.relative_to(ROOT))] = sha(REPORT)
    dump(RUN / 'run-manifest.json', m)
    update_state('completed', '- [Observed] All7 existing checkpoints×3 saved question packs×2 fixed4096-window groups complete42 forward-only evaluations; original forward/MSE parity, fixed draws/RNG and all state immutability checks passed. Exact per-pack rankings and source-run/downstream loss table saved. No new training/sampling, no selection threshold adopted.\n'+f'- Report `{REPORT.relative_to(ROOT)}`. Experiment2 remains explicitly deferred; fixed-exam ranking interpretation requires independent review of all three packs.')
    print(pd.DataFrame(table)[['checkpoint', 'recorded_training_loss', 'fixed_train_loss', 'fixed_validation_loss', 'downstream_rmse']].to_string(index=False), flush=True)


if __name__ == '__main__':
    environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else run()
