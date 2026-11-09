"""FD002 same-rerun best-loss versus final70; historical-stage gate before downstream."""
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

import inspect
import math
import platform

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/diagnostic-fd002-final70-two-seeds-20261102'
OUT = ROOT / 'results/diagnostics/fd002_final70_two_seeds_20261102'
WEIGHTS = ROOT / 'checkpoints/diagnostics/fd002_final70_two_seeds_20261102'
CORE = ROOT / 'scripts/run_fd001_sampling_condition_dropout_paired.py'
DOWNSTREAM = ROOT / 'scripts/run_fd001_checkpoint_downstream_validation.py'
PROTOCOL = ROOT / 'docs/operations/FD002_FINAL70_TWO_SEEDS.md'
SEEDS = [43, 13]
EVALUATORS = [3, 13, 23, 33, 43]
RULES = ['original', 'phase', 'best', 'final70']
N = 41539
BATCHES = 163

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

def tensor_hash(state):
    h = hashlib.sha256()
    for k, t in sorted(state.items()):
        h.update(k.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def selected(seed, rule):
    if rule == 'original':
        p = ROOT / f'checkpoints/FD002_w48_seed{seed}.pth'
        return dict(epoch_one_based=check_inputs()['config']['original_best_epochs'][str(seed)],
                    weight_path=str(p.relative_to(ROOT)), weight_sha256=sha(p))
    return training_audit(seed)['rules'][rule]


def train(seed):
    import numpy as np
    import torch
    check_inputs()
    assert torch.cuda.is_available()
    sys.path.insert(0, str(ROOT / 'upstream'))
    meta = json.loads((ROOT / f'results/runs/FD002_w48_seed{seed}_metadata.json').read_text())
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
    original_phase_epoch = check_inputs()['config']['original_best_epochs'][str(seed)]

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
        assert np.array_equal(np.sort(a), np.arange(41539))
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
            assert updates % 163 == 0
            saves.append(dict(epoch_one_based=updates // 163, actual_update=updates,
                tensors_sha256=tensor_hash(obj), serialization_rng_unchanged=True))
        return result

    def step(optimizer, *a, **kw):
        nonlocal updates
        v = original_step(optimizer, *a, **kw)
        updates += 1
        if updates % 163 == 0:
            counters = [int(s['step'].item()) for s in optimizer.state.values() if 'step' in s]
            assert counters and set(counters) == {updates}
            dump(dst / 'training_progress.json', dict(completed_epochs=updates // 163, total_epochs=70,
                 actual_optimizer_updates=updates, updated_at=now()))
        if updates == BATCHES * original_phase_epoch:
            before = rng()
            phase = WEIGHTS / f'seed{seed}/phase.pth'
            assert not phase.exists()
            original_save({k: v.detach().cpu().clone() for k, v in captured['model'].state_dict().items()}, phase)
            assert before == rng()
        if updates == 11410:
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
    assert updates == 11410 and len(orders) == 70 and len(draws) == 11410
    h = np.loadtxt(dst / 'training_loss.csv', delimiter=',', skiprows=1)
    oldh = np.loadtxt(ROOT / f'logs/FD002_w48_seed{seed}_training_loss.csv', delimiter=',', skiprows=1)
    assert h.shape == oldh.shape == (70, 3) and np.array_equal(h[:, 2], oldh[:, 2])
    eligible = h[h[:, 0] > 5]
    chosen = eligible[np.argmin(eligible[:, 1])]
    assert int(chosen[0]) == args.best_epoch and abs(chosen[1] - args.best_diffusion_loss) < 1e-12
    assert saves[-1]['epoch_one_based'] == args.best_epoch + 1
    best = torch.load(args.model_path, map_location='cpu', weights_only=True)
    final = torch.load(WEIGHTS / f'seed{seed}/final70.pth', map_location='cpu', weights_only=True)
    original = torch.load(ROOT / f'checkpoints/FD002_w48_seed{seed}.pth', map_location='cpu', weights_only=True)
    assert all(captured['initial_frequency'][k] == float(original[k].item()) for k in captured['initial_frequency'])
    assert tensor_hash(best) == saves[-1]['tensors_sha256']
    rules = {}
    for rule, weights, epoch in [('best', best, args.best_epoch + 1), ('final70', final, 70), ('phase', torch.load(WEIGHTS / f'seed{seed}/phase.pth', map_location='cpu', weights_only=True), check_inputs()['config']['original_best_epochs'][str(seed)])]:
        path = WEIGHTS / f'seed{seed}/{rule}.pth'
        rules[rule] = dict(epoch_one_based=int(epoch), actual_update=int(epoch * 163),
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
    wandb.init(project='fd002-final70-fixed-rule', mode='disabled')
    torch_seed(103)
    pre = dict(torch_cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        torch_cuda=hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest(),
        numpy_state=hashlib.sha256(np.random.get_state()[1].tobytes()).hexdigest())
    class FD002NoiseTrace(core.LegacyNoiseTrace):
        def result(self):
            assert self.normals == self.conditions == self.unconditionals == BATCHES * 1000
            assert self.elements == N * 48 * 14 * 1000
            self.after()
            return dict(status='passed', normal_tensors_exactly_matched=self.normals,
                conditional_uniform_draws_exactly_matched=self.conditions,
                unconditional_calls_retained=self.unconditionals, normal_scalar_count=self.elements,
                conditional_dropped_windows=self.dropped, retain_condition=self.retain_condition,
                start_cuda_rng_sha256=self.start_sha,
                final_cuda_rng_sha256=hashlib.sha256(self.reference.get_state().numpy().tobytes()).hexdigest(),
                legacy_schedule_fingerprint=self.fingerprint.hexdigest(), fresh_pair_noise_trace=True)
    trace = FD002NoiseTrace(torch, unet, retain_condition=True)
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
    anchor_path = OUT / 'noise_anchor.json'
    if anchor_path.exists():
        anchor = json.loads(anchor_path.read_text())
        assert pre == anchor['pre_model_construction_rng_sha256']
        for k in ['start_cuda_rng_sha256', 'final_cuda_rng_sha256', 'legacy_schedule_fingerprint', 'normal_scalar_count']:
            assert noise[k] == anchor['noise_audit'][k], k
    else:
        dump(anchor_path, dict(pre_model_construction_rng_sha256=pre, noise_audit=noise))
    assert data.shape == (41539, 48, 14) and np.isfinite(data).all()
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
    # User requested equal training budgets. Keep the public loop and synthetic-dev
    # selection, removing only its epoch-end early-stop branch in this local worker.
    public_source = inspect.getsource(eva.train)
    early_stop = "        if epoch - best_epoch >= 10:\n            print('*******STOP!!!********')\n            break"
    assert public_source.count(early_stop) == 1
    fixed_source = public_source.replace(early_stop, '        # Fixed 35-epoch paired budget; early stopping disabled.')
    exec(compile(fixed_source, '<fd002_fixed_budget_public_predictor>', 'exec'), eva.__dict__)
    helper = module(DOWNSTREAM, 'frozen_original_predictor_fit')
    target = folder(seed, rule)
    helper.OUT = target
    helper.CHECKPOINTS = ROOT / f'checkpoints/evaluators/fd002_final70_two_seeds_20261102/seed{seed}/{rule}'
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
    wandb.init(project='fd002-final70-downstream', mode='disabled')
    try:
        for s in EVALUATORS:
            name = f'epoch{epoch}_eval{s}'
            with (target / 'logs' / f'{name}.log').open('w', buffering=1) as handle, contextlib.redirect_stdout(handle):
                audit, vp, sp = helper.fit(eva, args, generated, vx, vy, ti, di, epoch, s)
            assert audit['completed_epochs'] == 35 and audit['actual_updates'] == 3990
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
                assert old_order.shape == new_order.shape == (35, 29077)
                assert np.array_equal(old_order, new_order), 'Unmatched predictor train order'
                assert old['actual_updates'] == audit['actual_updates'] == 3990
            write_csv(target / 'predictor_audits.csv', [{k: v for k, v in a.items() if k not in ['history', 'checkpoint_saves']} for a in audits])
            pd.DataFrame(rows).to_csv(target / 'per_model_segments.csv', index=False)
            pd.DataFrame(perengine).to_csv(target / 'per_engine.csv', index=False)
            dump(target / 'predictor_progress.json', dict(completed_fits=len(audits), total_fits=5, last_evaluator_seed=s, updated_at=now()))
            print('PREDICTOR', seed, rule, s, 'complete', flush=True)
    finally:
        wandb.finish()
    assert len(audits) == 5

def update_state(stage, body):
    body += f'\n- Snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`; protocol `{PROTOCOL.relative_to(ROOT)}`. Diagnostic/supporting, scope `scope-20261102-fd002-final70-two-seeds`. Strict main/A/B unchanged,C deferred; no FD003/4 expansion; FD001125 neighbor/sensor analyses paused. Validation engines seen by generator/MinMax: exploratory downstream comparison only.\n'
    p = ROOT / 'reproduction-state.md'
    head, tail = p.read_text().split('\n', 1)
    p.write_text(head + f'\n\n## Latest checkpoint: 2026-11-02 FD002 final70 comparison {stage}\n\n' + body + '\n' + tail)
    with (ROOT / 'experiment-ledger.md').open('a') as f:
        f.write(f'\n## `{RUN.name}` {stage}\n\n' + body)


def prepare():
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.preprocessing import MinMaxScaler
    assert not (RUN / 'run-manifest.json').exists(), 'No overwrite'
    for p in [RUN, RUN / 'source', OUT, OUT / 'logs', WEIGHTS]:
        p.mkdir(parents=True, exist_ok=True)
    # Preserve complete local weight inventory, not a filename guess about final epochs.
    inventory = []
    for suffix in ['*.pth', '*.pt', '*.ckpt', '*.safetensors']:
        for p in ROOT.rglob(suffix):
            if any(x in p.relative_to(ROOT).parts for x in ['.venv', '.git']):
                continue
            inventory.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size))
    write_csv(OUT / 'weight_inventory.csv', inventory)
    original_epochs = {}
    raw_path = ROOT / 'results/fd002_evaluation_stability_paper70_raw.csv'
    raw = pd.read_csv(raw_path)
    historic = raw.groupby('generation_seed')[['rmse', 'discriminative_score']].mean()
    assert len(raw) == 25 and set(historic.index) == {3,13,23,33,43}
    assert int(historic.rmse.idxmax()) == 43 and int(historic.rmse.idxmin()) == 13
    assert historic.loc[43, 'discriminative_score'] > .3 and historic.loc[13, 'discriminative_score'] < .3
    for seed in SEEDS:
        h = np.loadtxt(ROOT / f'logs/FD002_w48_seed{seed}_training_loss.csv', delimiter=',', skiprows=1)
        e = h[h[:, 0] > 5]
        original_epochs[str(seed)] = int(e[np.argmin(e[:, 1]), 0]) + 1
        for rule in RULES:
            for part in ['samples', 'audits', 'predictions', 'logs']:
                (folder(seed, rule) / part).mkdir(parents=True, exist_ok=True)
        (WEIGHTS / f'seed{seed}').mkdir(parents=True, exist_ok=True)
    assert original_epochs == {'43': 67, '13': 69}
    candidates = [r for r in inventory if 'fd002' in r['path'].lower() and
                  any(x in r['path'].lower() for x in ['epoch', 'final', 'checkpoint_age'])]
    # Search inventory is saved for inspection; original training source saves only min loss.
    assert not candidates, f'Inspect possible late snapshots before rerun: {candidates}'
    dump(OUT / 'archive_search.json', dict(original_epochs=original_epochs,
        final70_candidates=candidates, attributable_final70_found=False,
        excluded='Other seeds/window lengths/pretrained and changed-training variants are not same original run'))
    sensors = ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']
    columns = ['id','cycle','setting1','setting2','setting3'] + [f's{i}' for i in range(1,22)]
    frame = pd.read_csv(ROOT / 'upstream/data/train_FD002.txt', sep=r'\s+', header=None, names=columns)
    scaled = MinMaxScaler().fit_transform(frame[sensors])
    xx, yy, ids, end = [], [], [], []
    for engine, g in frame.groupby('id', sort=True):
        v = scaled[(frame.id == engine).to_numpy()]
        cycles = g.cycle.to_numpy()[47:]
        xx.append(np.lib.stride_tricks.sliding_window_view(v,48,axis=0).transpose(0,2,1))
        yy.extend(g.cycle.max()-cycles)
        ids.extend([int(engine)]*len(cycles)); end.extend(cycles)
    x = np.concatenate(xx).astype(np.float32)
    uncapped = np.asarray(yy); y = np.minimum(uncapped,125).astype(np.float32)
    ids, end = np.asarray(ids), np.asarray(end)
    assert x.shape == (N,48,14) and len(np.unique(ids)) == 260
    # Independently replay the public train-only normalization/slicing; avoid reading test files.
    sys.path.insert(0,str(ROOT / 'upstream'))
    from data.CMAPSSDataset import CMAPSSDataset
    public = CMAPSSDataset.__new__(CMAPSSDataset)
    public.sequence_length=48; public.deleted_engine=[1000]; public.engine_size=260; public.feature_columns=sensors
    f = frame.copy(); f['RUL'] = f.groupby('id').cycle.transform('max')-f.cycle; f['cycle_norm']=f.cycle
    norm_cols = f.columns.difference(['id','cycle','RUL'])
    normalized = pd.DataFrame(MinMaxScaler().fit_transform(f[norm_cols]), columns=norm_cols, index=f.index)
    public_frame = f[f.columns.difference(norm_cols)].join(normalized).reindex(columns=f.columns)
    assert np.array_equal(public.get_feature_slice(public_frame).numpy(), x)
    assert np.array_equal(public.get_label_slice(public_frame).numpy().ravel(), y)
    split_path = ROOT / 'results/fd002_evaluation_fixed_split_paper70.npz'
    with np.load(split_path) as z:
        split = {k: z[k].copy() for k in z.files}
    ti, di = split['predictive_train'], split['predictive_dev']
    assert len(ti)==29077 and len(di)==12462
    assert np.array_equal(np.sort(np.r_[ti,di]),np.arange(N))
    for seed in SEEDS:
        with np.load(ROOT / f'outputs/FD002_w48_seed{seed}.npz') as z:
            assert z['data'].shape == x.shape and np.array_equal(z['label'].ravel(), y)
    val_engines = np.sort(np.random.RandomState(20261102).permutation(np.unique(ids))[:78])
    vi = np.flatnonzero(np.isin(ids,val_engines))
    np.savez_compressed(RUN / 'frozen_inputs.npz', real=x, label=y[:,None], **split)
    np.savez_compressed(RUN / 'frozen_plan.npz',train_indices=ti,dev_indices=di,
        real_indices=vi,validation_engines=val_engines,other_engines=np.setdiff1d(np.unique(ids),val_engines))
    np.savez_compressed(RUN / 'real_validation.npz',data=x[vi],label=y[vi],engine_id=ids[vi],
        uncapped_rul=uncapped[vi],end_cycle=end[vi],original_window_index=vi)
    argv = sys.argv; sys.argv=[argv[0]]
    try:
        from args import args
    finally:
        sys.argv=argv
    resolved=vars(args).copy()
    resolved.update(dataset='FD002',window_size=48,input_size=14,output_size=1,device='cuda',
        model_name='DiffUnet_fre',eva_epoch=35)
    resolved={k:v if isinstance(v,(str,int,float,bool,type(None))) else str(v) for k,v in resolved.items()}
    config=dict(generator_seeds=SEEDS,evaluator_seeds=EVALUATORS,sampling_seed=103,
        rules=RULES,primary_rules=['best','final70'],original_best_epochs=original_epochs,
        seed_selection='Of original frozen five-seed paper70 records: worst mean downstream RMSE43, best13; both DS categories agree. Extra seeds not searched.',
        historical_means={str(k):{col:float(row[col]) for col in historic.columns} for k,row in historic.iterrows()},
        generator='Exact original per-seed CLI;70epochs/11410updates, original optimizer/schedule/data/global MinMax/dropout/numeric-zero unchanged; save extra original-stage and final snapshots without RNG changes',
        sampling='DDPM1000/CFG3/batch256; retain specified condition, consume legacy condition random draws; exact noise tensor equality checked independently against cloned CUDA generator, cross-arm fingerprints equal',
        sample_count=N,predictor_args=resolved,synthetic_train_n=len(ti),synthetic_dev_n=len(di),
        predictor_budget='35 full epochs/3990 updates each; public synthetic-dev best rule retained; early stopping disabled for BOTH arms to fix budget',
        predictor_control='Same eval-seed initialization hash and all35 exact epoch permutations across best/final70',
        validation_engines=val_engines.tolist(),validation_windows=len(vi),validation_split_seed=20261102,
        validation_exposure='Generator and global MinMax saw all260 engines, including78 internal validation engines. Exploratory downstream check, not end-to-end unseen-engine validation.',
        primary='sqrt(mean_over_5_predictors(mean_over_validation_engines(mean_window_squared_error)))',
        auxiliary='per-predictor RMSE, window-weighted RMSE, segments and five-seed discriminator scores',
        reproduction_gate=dict(compare='Original saved best vs same-epoch rerun; same current condition-retaining sampler/actual noise and five DS seeds. Historical old-sampler scores contextual only.',
            max_absolute_mean_DS_difference=.03,same_failure_category_threshold=.3,
            tensor_equal='Exact equality suffices; reuse identical outputs',
            failure='Stop this seed before primary downstream comparison, retain and report inconclusive; no substitution/retry'),
        budget=dict(generator_trainings=2,full_sample_batches_max=8,discriminator_fits_max=40,predictor_fits_max=20),
        aliases='Deduplicate exact tensor-equal arms within a seed; not independent repeats',
        interpretation='Within-rerun comparison primary; prior observed test results used to select contrasting seeds. No official test access in current evaluation.',
        stop='Two prescribed generator seeds only, no epoch/seed sweeps, no new learning rates/networks, no FD003/4 or125 neighbor/sensor follow-up')
    dump(RUN/'config.json',config)
    paths=[Path(__file__),PROTOCOL,CORE,DOWNSTREAM,ROOT/'scripts/run_fd001_resampling_diagnostic.py',
        ROOT/'scripts/run_fd001_final70_two_seeds.py',ROOT/'SOURCE_SNAPSHOT.md',raw_path,split_path,
        RUN/'config.json',RUN/'frozen_inputs.npz',RUN/'frozen_plan.npz',RUN/'real_validation.npz',
        OUT/'weight_inventory.csv',OUT/'archive_search.json',ROOT/'upstream/data/train_FD002.txt']
    paths.extend((ROOT/'upstream').glob('*.py'))
    paths.extend((ROOT/'upstream/DiffusionFreeGuidence').glob('*.py'))
    paths.extend([ROOT/'upstream/models/LSTM.py',ROOT/'upstream/data/CMAPSSDataset.py',
        ROOT/'upstream/measure_score/Utils/discriminative_metric.py',ROOT/'upstream/measure_score/Utils/metric_utils.py'])
    for seed in SEEDS:
        paths.extend([ROOT/f'checkpoints/FD002_w48_seed{seed}.pth',
            ROOT/f'results/runs/FD002_w48_seed{seed}_metadata.json',ROOT/f'logs/FD002_w48_seed{seed}_training_loss.csv'])
    for path in [Path(__file__),PROTOCOL,CORE,DOWNSTREAM,ROOT/'scripts/run_fd001_resampling_diagnostic.py']:
        dst=RUN/'source'/path.name;shutil.copy2(path,dst);paths.append(dst)
    for name,cmd in [('workspace.diff',['git','diff','--binary']),('git-status.txt',['git','status','--short']),
                     ('environment-freeze.txt',[sys.executable,'-m','pip','freeze'])]:
        dst=RUN/name;dst.write_bytes(subprocess.check_output(cmd,cwd=ROOT));paths.append(dst)
    dump(RUN/'run-manifest.json',dict(run_id=RUN.name,scope_class='diagnostic',role='supporting',
        scope_contract='scope-20261102-fd002-final70-two-seeds',status='prepared',prepared_at=now(),config=config,
        command=[sys.executable,'-u',str(Path(__file__)),'run'],
        user_authorization='Explicit cross-dataset validation: poor FD002 seed then good, rerun only if late snapshots missing; same-run best/final70 and five paired downstream seeds',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        environment=dict(python=sys.version,platform=platform.platform(),torch=str(torch.__version__)),
        inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(paths)},
        training_runs_completed=0,completed_sampling=0,completed_discriminator_evaluations=0,
        completed_predictor_fits=0,completed_cells=[],aliases=[],reproduction_gates={}))
    update_state('prepared',f'- [Observed] No attributable original final70 weights. Freeze original five-seed poor43(best67) and good13(best69), same-rerun best versus final70. {len(vi)} real validation windows/78 engines fixed before fits. Original-stage paired-sampler DS reproduction gate precedes primary downstream per seed. Max2 generator trainings/8 full datasets/40DSfits/20predictor fits, tensor-equal aliases reduce cost. Metrics pending; no strict paper target for this diagnostic.')
    print(json.dumps(config,ensure_ascii=False,indent=2),flush=True)


def reproduction_gate(seed):
    import numpy as np
    import pandas as pd
    paths=[folder(seed,r)/f'evaluation_checkpoint{seed}_sample103.csv' for r in ['original','phase']]
    tables=[pd.read_csv(p).sort_values('evaluator_seed') for p in paths]
    assert all(list(t.evaluator_seed)==EVALUATORS for t in tables)
    means=[float(t.discriminative_score.mean()) for t in tables]
    equal=selected(seed,'original')['weight_sha256']==selected(seed,'phase')['weight_sha256']
    # Serialized bytes may differ while every named tensor is identical.
    alias=folder(seed,'phase')/'alias.json'
    tensor_equal=alias.exists() and json.loads(alias.read_text())['canonical_rule']=='original'
    passed=bool(tensor_equal or (abs(means[1]-means[0])<=.03 and (means[0]>.3)==(means[1]>.3)))
    result=dict(generator_seed=seed,original_epoch=selected(seed,'original')['epoch_one_based'],
        original_current_sampler_DS=means[0],rerun_same_epoch_current_sampler_DS=means[1],
        absolute_difference=abs(means[1]-means[0]),max_difference=.03,
        failure_categories=[v>.3 for v in means],passed=passed,exact_tensor_alias=tensor_equal,
        same_serialization=equal,original_DS_per_seed=tables[0].discriminative_score.tolist(),
        rerun_DS_per_seed=tables[1].discriminative_score.tolist())
    dump(OUT/f'seed{seed}/reproduction_gate.json',result)
    return result


def summarize():
    import numpy as np
    import pandas as pd
    m=check_inputs();rows=[];paired=[]
    for seed in SEEDS:
        if not m['reproduction_gates'][str(seed)]['passed']:
            continue
        for rule in ['best','final70']:
            frame=pd.read_csv(folder(seed,rule)/'per_model_segments.csv')
            ds=pd.read_csv(folder(seed,rule)/f'evaluation_checkpoint{seed}_sample103.csv')
            for segment in frame.bin.unique():
                f=frame[(frame.domain=='real_validation') & (frame.bin==segment)].sort_values('evaluator_seed')
                assert len(f)==5
                rows.append(dict(generator_seed=seed,rule=rule,epoch=selected(seed,rule)['epoch_one_based'],
                    segment=segment,rmse_engine=float(np.sqrt(f.mse_engine.mean())),
                    mean_per_predictor_rmse=float(f.rmse_engine.mean()),std_per_predictor_rmse=float(f.rmse_engine.std(ddof=1)),
                    rmse_window=float(np.sqrt(f.mse_window.mean())),mean_DS=float(ds.discriminative_score.mean())))
        best=pd.read_csv(folder(seed,'best')/'per_model_segments.csv')
        final=pd.read_csv(folder(seed,'final70')/'per_model_segments.csv')
        b=best[(best.domain=='real_validation') & (best.bin=='all')].sort_values('evaluator_seed')
        f=final[(final.domain=='real_validation') & (final.bin=='all')].sort_values('evaluator_seed')
        assert np.array_equal(b.evaluator_seed,f.evaluator_seed)
        for br,fr in zip(b.to_dict('records'),f.to_dict('records')):
            paired.append(dict(generator_seed=seed,evaluator_seed=br['evaluator_seed'],
                best_rmse=br['rmse_engine'],final70_rmse=fr['rmse_engine'],
                delta_mse=fr['mse_engine']-br['mse_engine'],relative_rmse=fr['rmse_engine']/br['rmse_engine']-1))
    if rows:
        write_csv(OUT/'summary.csv',rows);write_csv(OUT/'paired_overall.csv',paired)
    lines=['# FD002同次训练：最低训练损失权重与第70轮','',
        '[Observed] 原规则和最终轮比较；两个预先固定生成器种子，每组五个配对预测器。',
        '','|生成器种子|原阶段复现门槛|最佳轮次|整体RMSE 最佳→70|相对变化|改善的预测器|DS 最佳→70|',
        '|---|---|---|---|---|---|---|']
    for seed in SEEDS:
        gate=m['reproduction_gates'][str(seed)]
        if not gate['passed']:
            lines.append(f'|{seed}|未通过，停止该种子主对照|—|—|—|—|—|');continue
        r=[v for v in rows if v['generator_seed']==seed and v['segment']=='all']
        b,f=r
        n=sum(p['delta_mse']<0 for p in paired if p['generator_seed']==seed)
        lines.append(f"|{seed}|通过|{b['epoch']}|{b['rmse_engine']:.4f}→{f['rmse_engine']:.4f}|{(f['rmse_engine']/b['rmse_engine']-1)*100:+.2f}%|{n}/5|{b['mean_DS']:.4f}→{f['mean_DS']:.4f}|")
    lines += ['', '[Code] 下游保留公开代码的LSTM、优化器、学习率、生成验证集选模规则；仅统一取消早停，双方实际35轮/3990次更新。初始化、35轮逐样本顺序与预算逐对核验。',
        '', '[Observed] 总体RMSE采用先每发动机均方误差、再78台发动机等权、再五模型等权、最后开根号；另保存逐模型和窗口等权口径。',
        '', '[Observed] 真实内部验证发动机已被生成器和归一化见过；这不是整条流程未见发动机的独立验证。当前未访问官方测试集。先前测试结果仅用于预选好/差种子。',
        '', '[Hypothesis] 是否存在跨数据集选模影响，依据同次训练的配对变化判断；两生成种子不足以制定普遍选模规则。无论结果如何不自动替换默认规则或扩展FD003/4。',
        '', '当前诊断结果→无论文同设计目标→论文差距不适用→五预测器条件波动及两生成种子→严格复现结论保持。',
        '', f"快照：`{RUN.relative_to(ROOT)}/run-manifest.json`；配置：`{RUN.relative_to(ROOT)}/config.json`；输出：`{OUT.relative_to(ROOT)}`。"]
    report=ROOT/'docs/conclusions/final/fd002_final70_two_seeds_20261102_report.md'
    report.write_text('\n'.join(lines)+'\n')
    m.update(status='complete' if all(g['passed'] for g in m['reproduction_gates'].values()) else 'complete_with_gate_failure',
        completed_at=now(),report=str(report.relative_to(ROOT)))
    dump(RUN/'run-manifest.json',m)
    update_state(m['status'],'- [Observed] Fixed FD002 two-seed budget closed. '+ '; '.join(lines[7:9]) +f'\n- Report `{report.relative_to(ROOT)}`. No automatic further experiments/default changes.')


def run():
    import torch
    m=check_inputs();assert m['status']=='prepared','No silent resume/retry'
    assert torch.cuda.is_available()
    m.update(status='running',started_at=now(),gpu=torch.cuda.get_device_name(0),
        torch_cuda=torch.version.cuda,driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip())
    dump(RUN/'run-manifest.json',m)
    def worker(action,seed,rule=None):
        m.update(current_stage=action,current_generator_seed=seed,current_rule=rule,stage_started_at=now())
        dump(RUN/'run-manifest.json',m)
        cmd=[sys.executable,'-u',str(Path(__file__)),action,str(seed)]+([rule] if rule else [])
        with (OUT/'logs'/f'{action}_seed{seed}_{rule or "training"}.log').open('w') as f:
            subprocess.run(cmd,cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True)
        print(now(),'COMPLETE',action,seed,rule,flush=True)
    try:
        for seed in SEEDS:
            worker('train',seed);m['training_runs_completed']+=1
            for p in [*list((WEIGHTS/f'seed{seed}').glob('*.pth')),OUT/f'seed{seed}/training_audit.json']:
                m['inputs_sha256'][str(p.relative_to(ROOT))]=sha(p)
            dump(RUN/'run-manifest.json',m)
            # Cache exact tensor equality once; serialization and raw tensor hashes differ.
            hashes={}
            for rule in RULES:
                state=torch.load(ROOT/selected(seed,rule)['weight_path'],map_location='cpu',weights_only=True)
                hashes[rule]=tensor_hash(state)
                del state
            canonical={}
            def data_arm(rule):
                prior=next((r for r in canonical if hashes[r]==hashes[rule]),None)
                if prior is not None:
                    src,dst=folder(seed,prior),folder(seed,rule)
                    for p in src.rglob('*'):
                        if p.is_file():
                            q=dst/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True)
                            assert not q.exists();q.symlink_to(os.path.relpath(p,q.parent))
                    alias=dict(generator_seed=seed,rule=rule,canonical_rule=prior,reason='Exact named tensor equality, no independent sampling/evaluation')
                    dump(dst/'alias.json',alias);m['aliases'].append(alias)
                else:
                    worker('sample',seed,rule);m['completed_sampling']+=1
                    worker('evaluate',seed,rule);m['completed_discriminator_evaluations']+=5
                    canonical[rule]=True
                dump(RUN/'run-manifest.json',m)
            data_arm('original');data_arm('phase')
            gate=reproduction_gate(seed);m['reproduction_gates'][str(seed)]=gate
            dump(RUN/'run-manifest.json',m)
            print('REPRODUCTION GATE',seed,json.dumps(gate),flush=True)
            if not gate['passed']:
                update_state('original-stage gate failed',f'- [Observed] seed{seed} original-stage reproduction gate failed: {gate}. No primary downstream for this seed. Other preauthorized seed may continue; no replacement.')
                continue
            for rule in ['best','final70']:
                data_arm(rule)
                # Reuse predictor only when the primary best/final tensors are exactly equal.
                if rule=='final70' and hashes['best']==hashes['final70']:
                    src,dst=folder(seed,'best'),folder(seed,'final70')
                    for name in ['per_model_segments.csv','per_engine.csv','predictor_audits.csv','predictor_progress.json']:
                        q=dst/name
                        if not q.exists():q.symlink_to(os.path.relpath(src/name,q.parent))
                    # CSV rows retain best label; summary explicitly binds rule to this alias.
                else:
                    worker('downstream',seed,rule);m['completed_predictor_fits']+=5
                m['completed_cells'].append(dict(generator_seed=seed,rule=rule))
                dump(RUN/'run-manifest.json',m)
        check_inputs();summarize()
    except BaseException:
        m.update(status='failed',failed_at=now(),error=traceback.format_exc())
        dump(RUN/'run-manifest.json',m)
        update_state('failed',f'- [Observed] Stopped at {m.get("current_stage")}/seed{m.get("current_generator_seed")}/{m.get("current_rule")}; retain all logs/artifacts, no silent retry.')
        raise


if __name__=='__main__':
    environment()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','train','sample','evaluate','downstream'])
    parser.add_argument('seed',nargs='?',type=int,choices=SEEDS)
    parser.add_argument('rule',nargs='?',choices=RULES)
    a=parser.parse_args()
    if a.action in ['prepare','run']:
        globals()[a.action]()
    elif a.action=='train':
        assert a.seed is not None;train(a.seed)
    else:
        assert a.seed is not None and a.rule is not None
        globals()[a.action](a.seed,a.rule)
