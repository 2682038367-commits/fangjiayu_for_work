"""Two original-setting reruns: best training-loss versus fixed final70."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from types import SimpleNamespace
import argparse
import contextlib
import csv
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
RUN = ROOT / 'runs/diagnostic-fd001-final70-two-seeds-20261030'
OUT = ROOT / 'results/diagnostics/fd001_final70_two_seeds_20261030'
WEIGHTS = ROOT / 'checkpoints/diagnostics/fd001_final70_two_seeds_20261030'
PREVIOUS = ROOT / 'runs/diagnostic-fd001-checkpoint-downstream-validation-20261030'
AGE_RUN = ROOT / 'runs/diagnostic-fd001-seed23-checkpoint-age-20261030'
ANCHOR = ROOT / 'results/diagnostics/fd001_seed23_checkpoint_age_20261030/epoch70/retain_condition/sample_checkpoint23_seed103.json'
CORE = ROOT / 'scripts/run_fd001_sampling_condition_dropout_paired.py'
DOWNSTREAM = ROOT / 'scripts/run_fd001_checkpoint_downstream_validation.py'
PROTOCOL = ROOT / 'docs/operations/FD001_FINAL70_TWO_SEEDS.md'
REPORTER = ROOT / 'analysis/diagnostics/summarize_fd001_final70_two_seeds.py'
SEEDS = [13, 3]
EVALUATORS = [3, 13, 23, 33, 43]
RULES = ['best', 'final70']


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def environment():
    for k, v in dict(WANDB_MODE='disabled', TF_DETERMINISTIC_OPS='1', TF_ENABLE_ONEDNN_OPTS='0',
        TF_CPP_MIN_LOG_LEVEL='2', CUBLAS_WORKSPACE_CONFIG=':4096:8', OPENBLAS_NUM_THREADS='1',
        MPLCONFIGDIR=str(ROOT / '.cache/matplotlib')).items():
        os.environ.setdefault(k, v)


def check_inputs():
    m = json.loads((RUN / 'run-manifest.json').read_text())
    for name, expected in m['inputs_sha256'].items():
        assert sha(ROOT / name) == expected, name
    return m


def folder(seed, rule):
    return OUT / f'seed{seed}' / rule


def training_audit(seed):
    return json.loads((OUT / f'seed{seed}/training_audit.json').read_text())


def selected(seed, rule):
    return training_audit(seed)['rules'][rule]


def prepare():
    import numpy as np
    import pandas as pd
    assert not (RUN / 'run-manifest.json').exists(), 'Preserve existing snapshot'
    for p in [RUN, OUT, OUT / 'logs', WEIGHTS, RUN / 'source']:
        p.mkdir(parents=True, exist_ok=True)
    old = json.loads((PREVIOUS / 'run-manifest.json').read_text())
    age = json.loads((AGE_RUN / 'run-manifest.json').read_text())
    assert old['status'] == age['status'] == 'complete'
    for key in ['frozen_plan.npz', 'real_validation.npz', 'config.json']:
        assert sha(PREVIOUS / key) == old['inputs_sha256'][str((PREVIOUS / key).relative_to(ROOT))]
        shutil.copy2(PREVIOUS / key, RUN / ('previous_' + key if key == 'config.json' else key))
    shutil.copy2(AGE_RUN / 'frozen_inputs.npz', RUN / 'frozen_inputs.npz')
    assert sha(RUN / 'frozen_inputs.npz') == age['inputs_sha256'][str((AGE_RUN / 'frozen_inputs.npz').relative_to(ROOT))]
    raw = pd.read_csv(ROOT / 'results/fd001_w48_public_code_evaluation_raw.csv')
    ds = raw.groupby('generation_seed').discriminative_score.mean().to_dict()
    bad = max((s for s in ds if s not in [13, 23]), key=lambda s: ds[s])
    assert bad == 3 and len(raw) == 25
    inventory = []
    for suffix in ['*.pth', '*.pt', '*.ckpt', '*.safetensors']:
        for p in ROOT.rglob(suffix):
            if any(t in ['.venv', '.git'] for t in p.relative_to(ROOT).parts):
                continue
            inventory.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size))
    write_csv(OUT / 'weight_inventory.csv', inventory)
    attributable = [r for r in inventory if r['path'] in [f'checkpoints/FD001_w48_seed{s}.pth' for s in SEEDS]]
    # All epoch-labelled FD001 artifacts are inventoried for human audit; none are attributed to these two original runs.
    candidates = [r for r in inventory if 'FD001' in r['path'] and 'evaluators' not in r['path']]
    dump(OUT / 'archive_search.json', dict(inventory_count=len(inventory), attributable_original_best=attributable,
         candidates=candidates, original_attributable_final70_found=False,
         exclusions='Other generator seeds/tasks/window sizes or supplied pretrained weights are not matching final snapshots'))
    original_epochs = {}
    for s in SEEDS:
        history = np.loadtxt(ROOT / f'logs/FD001_w48_seed{s}_training_loss.csv', delimiter=',', skiprows=1)
        eligible = history[history[:, 0] > 5]
        original_epochs[str(s)] = int(eligible[np.argmin(eligible[:, 1]), 0]) + 1
    assert original_epochs == {'13': 64, '3': 43}
    config = dict(generator_seeds=SEEDS, rules=RULES, sampling_seed=103,
        historical_ds_means={str(k): float(v) for k, v in ds.items()}, original_best_epochs=original_epochs,
        selection='user-specified good13; highest original DS excluding13/23 is poor3; fixed before rerun results',
        generator_training='original per-seed CLI/defaults,70epochs/4410updates; training dropout.2/numeric-zero.1 retained',
        sampling='retain conditional branch, legacy RNG draws, original explicit null CFG; DDPM1000/CFG3/batch256',
        predictor_args=old['config']['predictor_args'], evaluator_seeds=EVALUATORS,
        primary='within each SAME rerun: final70 versus original-rule best; no epoch search',
        validation_engines=old['config']['validation_engines'], validation_windows=4875,
        validation_exposure='Generator/MinMax saw all training engines including these30; exploration only',
        bins=old['config']['bins'], condition_reference_indices='frozen previous seed23 matched3000 and all15931',
        flags=dict(good13_max_overall_regression=.05, poor3_min_overall_improvement=.05, poor3_min_paired_improvements=4),
        budget=dict(generator_trainings=2, full_samples_max=4, discriminator_fits_max=20, predictor_fits_max=20),
        identical_weights='explicitly alias final70 to best and skip redundant computation',
        stop='fixed budget or technical failure; no retries/replacement seeds/other epochs/new sampling seeds/default adoption')
    dump(RUN / 'config.json', config)
    paths = [Path(__file__), REPORTER, PROTOCOL, CORE, DOWNSTREAM, ROOT / 'scripts/run_fd001_resampling_diagnostic.py',
        ROOT / 'SOURCE_SNAPSHOT.md', ROOT / 'upstream/TrainCondition.py', ROOT / 'upstream/MainCondition.py',
        ROOT / 'upstream/GaussianDiffusion.py', ROOT / 'upstream/args.py', ROOT / 'upstream/utils.py',
        ROOT / 'upstream/eva_regressor.py', ROOT / 'upstream/models/LSTM.py', ROOT / 'upstream/data/CMAPSSDataset.py',
        ROOT / 'upstream/measure_score/Utils/discriminative_metric.py',
        ROOT / 'upstream/measure_score/Utils/metric_utils.py', ROOT / 'upstream/data/train_FD001.txt',
        ROOT / 'results/fd001_w48_public_code_evaluation_raw.csv',
        ROOT / 'results/diagnostics/real_predictor_condition_w48/FD001_predictions.npz',
        RUN / 'config.json', RUN / 'previous_config.json', RUN / 'frozen_plan.npz', RUN / 'real_validation.npz',
        RUN / 'frozen_inputs.npz', PREVIOUS / 'run-manifest.json', AGE_RUN / 'run-manifest.json', ANCHOR,
        OUT / 'weight_inventory.csv', OUT / 'archive_search.json']
    paths.extend((ROOT / 'upstream/DiffusionFreeGuidence').glob('*.py'))
    for s in SEEDS:
        paths += [ROOT / f'results/runs/FD001_w48_seed{s}_metadata.json',
                  ROOT / f'logs/FD001_w48_seed{s}_training_loss.csv', ROOT / f'checkpoints/FD001_w48_seed{s}.pth']
        for rule in RULES:
            for part in ['samples', 'audits', 'predictions', 'logs']:
                (folder(s, rule) / part).mkdir(parents=True, exist_ok=True)
        (WEIGHTS / f'seed{s}').mkdir(parents=True, exist_ok=True)
    for s in EVALUATORS:
        paths.append(ROOT / f'checkpoints/evaluators/real_data_baseline_w48/FD001_eval{s}.pth')
    with np.load(RUN / 'frozen_inputs.npz') as z:
        assert z['real'].shape == (15931, 48, 14) and z['label'].shape == (15931, 1)
    with np.load(RUN / 'real_validation.npz') as z:
        assert len(z['label']) == 4875 and len(np.unique(z['engine_id'])) == 30
        assert np.array_equal(np.minimum(z['uncapped_rul'], 125), z['label'])
    for p in [Path(__file__), REPORTER, PROTOCOL]:
        copy = RUN / 'source' / p.name
        shutil.copy2(p, copy)
        paths.append(copy)
    for name, command in [('workspace.diff', ['git', 'diff', '--binary']), ('git-status.txt', ['git', 'status', '--short']),
                          ('environment-freeze.txt', [sys.executable, '-m', 'pip', 'freeze'])]:
        p = RUN / name
        p.write_bytes(subprocess.check_output(command, cwd=ROOT))
        paths.append(p)
    dump(RUN / 'run-manifest.json', dict(run_id=RUN.name, scope_class='diagnostic', role='supporting',
        scope_contract='scope-20261030-fd001-final70-two-seeds', status='prepared', prepared_at=now(), config=config,
        command=[sys.executable, '-u', str(Path(__file__)), 'run'],
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        user_authorization='Explicit two additional generator seeds, original best-loss versus fixed final70; same-run primary contrast',
        inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(paths)},
        training_runs_completed=0, completed_sampling=0, completed_discriminator_evaluations=0,
        completed_predictor_fits=0, completed_cells=[], aliases=[]))
    update_state('prepared', '- [Observed] Fixed goodseed13/poorseed3 from originalDS; no attributable original final70 snapshots. Two original70epoch reruns authorized, same-run best versus final70 only. Same103noise/labels/settings, five DS and predictor seeds, same4875 real validation windows. Metrics pending. Maximum2 trainings/4 samples/20 DSfits/20 predictor fits; exact-equal rules explicitly alias. No seed/epoch replacement or default changes.')
    print(json.dumps(config, indent=2), flush=True)


def update_state(stage, text):
    body = text + f'\n- Snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`; protocol `{PROTOCOL.relative_to(ROOT)}`; outputs `{OUT.relative_to(ROOT)}`. Diagnostic/supporting, strict main/A/B unchanged, C deferred; generator saw validation engines, zero-label repair paused, EXT-002 closed/weight1.\n'
    p = ROOT / 'reproduction-state.md'
    head, tail = p.read_text().split('\n', 1)
    p.write_text(head + f'\n\n## Latest checkpoint: 2026-10-30 FD001 final70 two-seed {stage}\n\n' + body + '\n' + tail)
    with (ROOT / 'experiment-ledger.md').open('a') as f:
        f.write(f'\n## `{RUN.name}` {stage}\n\n' + body)


def train(seed):
    import numpy as np
    import torch
    check_inputs()
    assert torch.cuda.is_available()
    sys.path.insert(0, str(ROOT / 'upstream'))
    meta = json.loads((ROOT / f'results/runs/FD001_w48_seed{seed}_metadata.json').read_text())
    argv = sys.argv
    sys.argv = [str(ROOT / 'upstream/MainCondition.py'), *meta['command'][2:]]
    try:
        from args import args
    finally:
        sys.argv = argv
    import TrainCondition as training
    from utils import torch_seed
    import wandb
    dst = OUT / f'seed{seed}'
    args.model_path = str(WEIGHTS / f'seed{seed}/best.pth')
    args.loss_history_path = str(dst / 'training_loss.csv')
    args.metrics_path = str(dst / 'unused_metrics.json')
    args.syndata_path = str(dst / 'unused_samples.npz')
    assert args.seed == seed and args.epoch == 70 and args.batch_size == 256 and args.lr == .002
    assert args.batch_label_zero_prob == .1 and args.frequency_threshold == -1.
    with np.load(RUN / 'frozen_inputs.npz') as z:
        x, y = torch.from_numpy(z['real'].copy()), torch.from_numpy(z['label'].copy())
    wandb.init(project='MetaIndux-TS', tags=['all'], config=args, mode='offline', dir=str(dst))
    torch_seed(seed)
    original_model, original_step = training.UNet1D_fre, torch.optim.Adam.step
    original_iter, original_rand, original_save = torch.utils.data.RandomSampler.__iter__, np.random.rand, torch.save
    captured, orders, draws, saves = {}, [], [], []
    updates = 0

    def rng():
        return dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            cuda=hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest(),
            numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest())

    def factory(*a, **kw):
        model = original_model(*a, **kw)
        captured['model'] = model
        captured['initial_frequency'] = {k: float(v.item()) for k, v in model.state_dict().items() if k.endswith('threshold_param')}
        captured['initial_tensor_sha256'] = tensor_hash(model.state_dict())
        return model

    def iterator(sampler):
        taken = []
        for index in original_iter(sampler):
            taken.append(index)
            yield index
        a = np.asarray(taken, dtype=np.int32)
        assert np.array_equal(np.sort(a), np.arange(15931))
        orders.append(a)

    def rand(*a):
        v = original_rand(*a)
        if not a and Path(sys._getframe(1).f_code.co_filename).name == 'TrainCondition.py':
            draws.append(dict(update=updates + 1, draw=float(v), numeric_batch_zeroed=bool(v < .1)))
        return v

    def save(obj, path, *a, **kw):
        before = rng()
        result = original_save(obj, path, *a, **kw)
        after = rng()
        assert before == after, 'Save changed random states'
        if str(path) == args.model_path:
            assert updates % 63 == 0
            saves.append(dict(epoch_one_based=updates // 63, actual_update=updates,
                tensors_sha256=tensor_hash(obj), serialization_rng_unchanged=True))
        return result

    def step(optimizer, *a, **kw):
        nonlocal updates
        v = original_step(optimizer, *a, **kw)
        updates += 1
        if updates % 63 == 0:
            counters = [int(s['step'].item()) for s in optimizer.state.values() if 'step' in s]
            assert counters and set(counters) == {updates}
            dump(dst / 'training_progress.json', dict(completed_epochs=updates // 63, total_epochs=70,
                 actual_optimizer_updates=updates, updated_at=now()))
        if updates == 4410:
            before = rng()
            final = WEIGHTS / f'seed{seed}/final70.pth'
            assert not final.exists()
            original_save({k: v.detach().cpu().clone() for k, v in captured['model'].state_dict().items()}, final)
            assert before == rng()
        return v

    training.UNet1D_fre, torch.optim.Adam.step = factory, step
    torch.utils.data.RandomSampler.__iter__, np.random.rand, torch.save = iterator, rand, save
    start, started_at = time.time(), now()
    try:
        training.train(args, x, y)
    finally:
        training.UNet1D_fre, torch.optim.Adam.step = original_model, original_step
        torch.utils.data.RandomSampler.__iter__, np.random.rand, torch.save = original_iter, original_rand, original_save
        wandb.finish()
    assert updates == 4410 and len(orders) == 70 and len(draws) == 4410
    h = np.loadtxt(dst / 'training_loss.csv', delimiter=',', skiprows=1)
    oldh = np.loadtxt(ROOT / f'logs/FD001_w48_seed{seed}_training_loss.csv', delimiter=',', skiprows=1)
    assert h.shape == oldh.shape == (70, 3) and np.array_equal(h[:, 2], oldh[:, 2])
    eligible = h[h[:, 0] > 5]
    chosen = eligible[np.argmin(eligible[:, 1])]
    assert int(chosen[0]) == args.best_epoch and abs(chosen[1] - args.best_diffusion_loss) < 1e-12
    assert saves[-1]['epoch_one_based'] == args.best_epoch + 1
    best = torch.load(args.model_path, map_location='cpu', weights_only=True)
    final = torch.load(WEIGHTS / f'seed{seed}/final70.pth', map_location='cpu', weights_only=True)
    original = torch.load(ROOT / f'checkpoints/FD001_w48_seed{seed}.pth', map_location='cpu', weights_only=True)
    assert all(captured['initial_frequency'][k] == float(original[k].item()) for k in captured['initial_frequency'])
    assert tensor_hash(best) == saves[-1]['tensors_sha256']
    rules = {}
    for rule, weights, epoch in [('best', best, args.best_epoch + 1), ('final70', final, 70)]:
        path = WEIGHTS / f'seed{seed}/{rule}.pth'
        rules[rule] = dict(epoch_one_based=int(epoch), actual_update=int(epoch * 63),
            weight_path=str(path.relative_to(ROOT)), weight_sha256=sha(path), tensors_sha256=tensor_hash(weights))
    equal = all(torch.equal(best[k], final[k]) for k in best)
    np.savez_compressed(dst / 'training_epoch_orders.npz', orders=np.stack(orders))
    write_csv(dst / 'training_batch_zero_draws.csv', draws)
    dump(dst / 'training_audit.json', dict(status='complete', generator_seed=seed, started_at=started_at,
        completed_at=now(), elapsed_seconds=time.time()-start, actual_optimizer_updates=updates,
        completed_epochs=70, lr_sequence_matches_original=True, rules=rules, best_and_final_tensors_equal=equal,
        original_best_epoch_one_based=int(oldh[oldh[:, 0] > 5][np.argmin(oldh[oldh[:, 0] > 5, 1]), 0])+1,
        rerun_best_epoch_one_based=int(args.best_epoch)+1, best_loss=float(args.best_diffusion_loss),
        initial_frequency=captured['initial_frequency'], initial_tensor_sha256=captured['initial_tensor_sha256'],
        best_save_events=saves, final_serialization_rng_unchanged=True,
        caveat='Original nondeterministic training settings retained; historic weights/trajectory not claimed identical'))
    print('Training complete', seed, 'best', args.best_epoch+1, 'final70', 'equal', equal, flush=True)


def tensor_hash(state):
    h = hashlib.sha256()
    for k, t in sorted(state.items()):
        h.update(k.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sample(seed, rule):
    import numpy as np
    import torch
    check_inputs()
    core = module(CORE, 'frozen_noise_tracer')
    sys.path.insert(0, str(ROOT / 'upstream'))
    import TrainCondition as training
    import GaussianDiffusion as diffusion
    import DiffusionFreeGuidence.Unet1D_fre as unet
    from utils import torch_seed
    import wandb
    info = selected(seed, rule)
    assert sha(ROOT / info['weight_path']) == info['weight_sha256']
    with np.load(RUN / 'frozen_inputs.npz') as z:
        labels = torch.from_numpy(z['label'].copy())
    args = SimpleNamespace(model_name='DiffUnet_fre', input_size=14, window_size=48,
        frequency_mask_mode='hard_random_quantile', frequency_threshold_init=.25,
        frequency_mask_temperature=.1, frequency_threshold=-1., device='cuda', T=1000,
        schedule_name='linear', batch_size=256, model_path=str(ROOT / info['weight_path']))
    target = folder(seed, rule)
    artifact = target / f'samples/checkpoint{seed}_sample103.npz'
    assert not artifact.exists()
    args.syndata_path = str(artifact)
    diffusion.tqdm = lambda iterable, **kwargs: iterable
    training.train = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('Training forbidden during sampling'))
    wandb.init(project='fd001-final70-fixed-rule', mode='disabled')
    torch_seed(103)
    pre = dict(torch_cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        torch_cuda=hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest(),
        numpy_state=hashlib.sha256(np.random.get_state()[1].tobytes()).hexdigest())
    anchor = json.loads(ANCHOR.read_text())
    assert pre == anchor['pre_model_construction_rng_sha256']
    trace = core.LegacyNoiseTrace(torch, unet, retain_condition=True)
    start, started_at = time.time(), now()

    def progress(iterable, **kwargs):
        trace.start()
        for count, value in enumerate(iterable, 1):
            yield value
            dump(target / f'progress_checkpoint{seed}_sample103.json', dict(generator_seed=seed, rule=rule,
                completed_batches=count, total_batches=len(iterable), elapsed_seconds=time.time()-start,
                matched_noise_tensors=trace.normals, updated_at=now()))
            print('SAMPLE', seed, rule, count, '/', len(iterable), flush=True)

    training.tqdm = progress
    trace.install()
    try:
        data = training.sample(args, labels)
        noise = trace.result()
    finally:
        trace.restore()
        wandb.finish()
    for k in ['start_cuda_rng_sha256', 'final_cuda_rng_sha256', 'legacy_schedule_fingerprint', 'normal_scalar_count']:
        assert noise[k] == anchor['noise_audit'][k], k
    assert data.shape == (15931, 48, 14) and np.isfinite(data).all()
    with np.load(artifact) as z:
        assert np.array_equal(z['label'], labels.numpy())
    dump(target / f'sample_checkpoint{seed}_seed103.json', dict(checkpoint_seed=seed, rule=rule,
        selected_epoch=info['epoch_one_based'], sampling_seed=103, started_at=started_at, completed_at=now(),
        elapsed_seconds=time.time()-start, artifact_path=str(artifact.relative_to(ROOT)), artifact_sha256=sha(artifact),
        checkpoint_sha256=info['weight_sha256'], pre_model_construction_rng_sha256=pre, noise_audit=noise,
        frequency_thresholds=args.frequency_thresholds, torch=str(torch.__version__), gpu=torch.cuda.get_device_name(0)))


def evaluate(seed, rule):
    worker = module(ROOT / 'scripts/run_fd001_resampling_diagnostic.py', 'original_ds_worker')
    worker.RUN, worker.OUT = RUN, folder(seed, rule)
    worker.worker_evaluate(seed, 103)


def condition(seed, rule):
    import numpy as np
    import torch
    check_inputs()
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / 'upstream'))
    target = folder(seed, rule)
    with np.load(target / f'samples/checkpoint{seed}_sample103.npz') as z:
        x, y = z['data'].copy(), z['label'].ravel().astype(float)
    with np.load(ROOT / 'results/diagnostics/real_predictor_condition_w48/FD001_predictions.npz') as z:
        i = list(z['generation_seeds']).index(23)
        indices = dict(dev_matched=z['generated_dev_indices'][i].copy(), all_windows=np.arange(len(y)))
        assert np.array_equal(y[indices['dev_matched']], z['real_dev_labels'])
    rows, predictions = [], []
    bins = [('all', 0, 125), ('0', 0, 0), ('0-30', 0, 30), ('31-60', 31, 60),
            ('61-90', 61, 90), ('91-124', 91, 124), ('125', 125, 125)]
    for s in EVALUATORS:
        model = torch.load(ROOT / f'checkpoints/evaluators/real_data_baseline_w48/FD001_eval{s}.pth',
                           map_location='cpu', weights_only=False).eval()
        with torch.no_grad():
            p = np.concatenate([model(torch.from_numpy(x[j:j+256])).numpy().ravel() for j in range(0, len(x), 256)]).astype(float)
        assert np.isfinite(p).all()
        predictions.append(p)
        for reference, ix in indices.items():
            for name, lo, hi in bins:
                use = ix[(y[ix] >= lo) & (y[ix] <= hi)]
                if not len(use):
                    continue
                err = p[use]-y[use]
                rows.append(dict(generator_seed=seed, rule=rule, evaluator_seed=s, reference=reference,
                    bin=name, n_windows=len(use), mse=float(np.mean(err**2)), rmse=float(np.sqrt(np.mean(err**2))),
                    bias=float(err.mean())))
    np.savez_compressed(target / 'condition_predictions.npz', predictions=np.array(predictions), label=y,
                        evaluator_seeds=np.array(EVALUATORS), **indices)
    write_csv(target / 'condition_metrics.csv', rows)


def downstream(seed, rule):
    import numpy as np
    import pandas as pd
    import torch
    check_inputs()
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT / 'upstream'))
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        import eva_regressor as eva
        import wandb
    finally:
        sys.argv = argv
    helper = module(DOWNSTREAM, 'frozen_original_predictor_fit')
    target = folder(seed, rule)
    helper.OUT = target
    helper.CHECKPOINTS = ROOT / f'checkpoints/evaluators/fd001_final70_two_seeds_20261030/seed{seed}/{rule}'
    helper.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    with np.load(target / f'samples/checkpoint{seed}_sample103.npz') as z:
        generated = dict(data=z['data'].copy(), label=z['label'].copy())
    with np.load(RUN / 'frozen_plan.npz') as z:
        ti, di = z['train_indices'], z['dev_indices']
    with np.load(RUN / 'real_validation.npz') as z:
        vx = torch.from_numpy(z['data'].copy())
        y, ids = z['label'].astype(float), z['engine_id'].copy()
        vy = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    args = SimpleNamespace(**check_inputs()['config']['predictor_args'])
    epoch = selected(seed, rule)['epoch_one_based']
    sy = generated['label'].ravel().astype(float)
    rows, perengine, audits = [], [], []
    wandb.init(project='fd001-final70-downstream', mode='disabled')
    try:
        for s in EVALUATORS:
            name = f'epoch{epoch}_eval{s}'
            with (target / 'logs' / f'{name}.log').open('w', buffering=1) as handle, contextlib.redirect_stdout(handle):
                audit, vp, sp = helper.fit(eva, args, generated, vx, vy, ti, di, epoch, s)
            audit.update(generator_seed=seed, rule=rule)
            audits.append(audit)
            meta = dict(generator_seed=seed, rule=rule, selected_epoch=epoch, evaluator_seed=s)
            for domain, p, yy, ei in [('real_validation', vp, y, ids),
                    ('synthetic_train', sp[ti], sy[ti], None), ('synthetic_dev', sp[di], sy[di], None)]:
                a, b = helper.metrics(p, yy, ei, meta, domain)
                rows.extend(a)
                perengine.extend(b)
            if rule == 'final70':
                best_epoch = selected(seed, 'best')['epoch_one_based']
                base = folder(seed, 'best')
                old = json.loads((base / 'audits' / f'epoch{best_epoch}_eval{s}.json').read_text())
                assert old['initialization_sha256'] == audit['initialization_sha256']
                with np.load(base / 'audits' / f'epoch{best_epoch}_eval{s}_orders.npz') as z:
                    old_order = z['orders']
                with np.load(target / 'audits' / f'{name}_orders.npz') as z:
                    new_order = z['orders']
                n = min(len(old_order), len(new_order))
                assert np.array_equal(old_order[:n], new_order[:n]), 'Unmatched predictor train order'
            write_csv(target / 'predictor_audits.csv', [{k: v for k, v in a.items() if k not in ['history', 'checkpoint_saves']} for a in audits])
            pd.DataFrame(rows).to_csv(target / 'per_model_segments.csv', index=False)
            pd.DataFrame(perengine).to_csv(target / 'per_engine.csv', index=False)
            dump(target / 'predictor_progress.json', dict(completed_fits=len(audits), total_fits=5, last_evaluator_seed=s, updated_at=now()))
            print('PREDICTOR', seed, rule, s, 'complete', flush=True)
    finally:
        wandb.finish()
    assert len(audits) == 5


def run():
    m = check_inputs()
    assert m['status'] == 'prepared', 'No silent resume/retry'
    m.update(status='running', started_at=now())
    dump(RUN / 'run-manifest.json', m)

    def worker(action, seed, rule=None):
        m.update(current_stage=action, current_generator_seed=seed, current_rule=rule)
        dump(RUN / 'run-manifest.json', m)
        command = [sys.executable, '-u', str(Path(__file__)), action, str(seed)] + ([rule] if rule else [])
        with (OUT / 'logs' / f'{action}_seed{seed}_{rule or "training"}.log').open('w') as f:
            subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=f, stderr=subprocess.STDOUT, check=True)
        print(now(), 'COMPLETE', action, seed, rule, flush=True)

    try:
        for s in SEEDS:
            worker('train', s)
            m['training_runs_completed'] += 1
            for p in [*list((WEIGHTS / f'seed{s}').glob('*.pth')), OUT / f'seed{s}/training_audit.json']:
                m['inputs_sha256'][str(p.relative_to(ROOT))] = sha(p)
            dump(RUN / 'run-manifest.json', m)
        for s in SEEDS:
            for rule in RULES:
                if rule == 'final70' and training_audit(s)['best_and_final_tensors_equal']:
                    alias = dict(generator_seed=s, rule=rule, canonical_rule='best', reason='Exact same best/final tensors; no new computation')
                    dump(folder(s, rule) / 'alias.json', alias)
                    m['aliases'].append(alias)
                else:
                    for action in ['sample', 'evaluate', 'condition', 'downstream']:
                        worker(action, s, rule)
                        if action == 'sample':
                            m['completed_sampling'] += 1
                        elif action == 'evaluate':
                            m['completed_discriminator_evaluations'] += 5
                        elif action == 'downstream':
                            m['completed_predictor_fits'] += 5
                        dump(RUN / 'run-manifest.json', m)
                m['completed_cells'].append(dict(generator_seed=s, rule=rule))
                dump(RUN / 'run-manifest.json', m)
        check_inputs()
        m.update(status='computed_pending_summary', computation_completed_at=now())
        dump(RUN / 'run-manifest.json', m)
        subprocess.run([sys.executable, str(REPORTER)], cwd=ROOT, env=os.environ.copy(), check=True)
    except BaseException:
        m.update(status='failed', failed_at=now(), error=traceback.format_exc())
        dump(RUN / 'run-manifest.json', m)
        update_state('failed', f'- [Observed] Stopped at {m.get("current_stage")}/seed{m.get("current_generator_seed")}/{m.get("current_rule")}; preserve manifest/log and completed artifacts. No automatic retraining/retry or scientific changes.')
        raise


if __name__ == '__main__':
    environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'train', 'sample', 'evaluate', 'condition', 'downstream'])
    parser.add_argument('seed', nargs='?', type=int, choices=SEEDS)
    parser.add_argument('rule', nargs='?', choices=RULES)
    arguments = parser.parse_args()
    if arguments.action in ['prepare', 'run']:
        globals()[arguments.action]()
    elif arguments.action == 'train':
        assert arguments.seed is not None
        train(arguments.seed)
    else:
        assert arguments.seed is not None and arguments.rule is not None
        globals()[arguments.action](arguments.seed, arguments.rule)
