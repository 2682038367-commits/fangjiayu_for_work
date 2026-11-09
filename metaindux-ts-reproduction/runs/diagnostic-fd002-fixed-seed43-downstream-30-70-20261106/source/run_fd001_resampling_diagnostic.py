"""Six fixed-checkpoint FD001 samples and thirty original discriminator fits."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from types import SimpleNamespace
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream'
RUN = ROOT / 'runs/diagnostic-fd001-checkpoint-resampling-20261029'
OUT = ROOT / 'results/diagnostics/fd001_checkpoint_resampling_20261029'
PROTOCOL = ROOT / 'docs/operations/FD001_CHECKPOINT_RESAMPLING_DIAGNOSTIC.md'
SUMMARY = ROOT / 'analysis/diagnostics/summarize_fd001_resampling_diagnostic.py'
CHECKPOINTS = [13, 23]
SAMPLING = [103, 113, 123]
EVALUATORS = [3, 13, 23, 33, 43]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def environment():
    for key, value in dict(WANDB_MODE='disabled', TF_DETERMINISTIC_OPS='1',
                           TF_ENABLE_ONEDNN_OPTS='0', TF_CPP_MIN_LOG_LEVEL='2',
                           CUBLAS_WORKSPACE_CONFIG=':4096:8', OPENBLAS_NUM_THREADS='1',
                           MPLCONFIGDIR=str(ROOT / '.cache/matplotlib')).items():
        os.environ.setdefault(key, value)


def check_inputs():
    manifest = json.loads((RUN / 'run-manifest.json').read_text())
    for name, expected in manifest['inputs_sha256'].items():
        assert sha(ROOT / name) == expected, f'Frozen input changed: {name}'
    return manifest


def prepare():
    import numpy as np
    import pandas as pd
    import torch
    if (RUN / 'run-manifest.json').exists():
        raise RuntimeError('Preparation already exists; preserve frozen configuration')
    RUN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'samples').mkdir(exist_ok=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    sys.path.insert(0, str(UPSTREAM))
    from data.CMAPSSDataset import CMAPSSDataset
    previous = Path.cwd()
    os.chdir(UPSTREAM)
    try:
        dataset = CMAPSSDataset('FD001', 48, [1000])
        real = dataset.get_feature_slice(dataset.get_train_data()).numpy()
        labels = dataset.get_label_slice(dataset.get_train_data()).numpy()
    finally:
        os.chdir(previous)
    assert real.shape == (15931, 48, 14) and labels.shape == (15931, 1)
    split_path = ROOT / 'results/fd001_w48_public_code_split.npz'
    with np.load(split_path) as archive:
        split = {k: archive[k].copy() for k in archive.files if k.startswith('discriminator_')}
    for prefix in ['real', 'generated']:
        train, test = split[f'discriminator_{prefix}_train'], split[f'discriminator_{prefix}_test']
        assert len(train) == 12744 and len(test) == 3187
        assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(len(real)))
    np.savez_compressed(RUN / 'frozen_inputs.npz', real=real, label=labels, **split)
    historical = pd.read_csv(ROOT / 'results/fd001_w48_public_code_evaluation_raw.csv')
    means = historical.groupby('generation_seed').discriminative_score.mean()
    assert int(means.idxmin()) == 13 and int(means.idxmax()) == 23
    states, audits, paths = [], [], []
    for checkpoint in CHECKPOINTS:
        weight = ROOT / f'checkpoints/FD001_w48_seed{checkpoint}.pth'
        metadata_path = ROOT / f'results/runs/FD001_w48_seed{checkpoint}_metadata.json'
        stage_path = ROOT / f'results/runs/FD001_w48_seed{checkpoint}_metrics.generation.json'
        history_path = ROOT / f'logs/FD001_w48_seed{checkpoint}_training_loss.csv'
        synthetic = ROOT / f'outputs/FD001_w48_seed{checkpoint}.npz'
        metadata, stage = json.loads(metadata_path.read_text()), json.loads(stage_path.read_text())
        command = metadata['command']
        assert metadata['status'] == 'complete' and command[command.index('--state') + 1] == 'train'
        assert int(command[command.index('--seed') + 1]) == checkpoint
        history = pd.read_csv(history_path)
        eligible = history[history.epoch > 5]
        best = eligible.loc[eligible.diffusion_loss.idxmin()]
        assert len(history) == 70 and int(best.epoch) == int(stage['best_epoch'])
        with np.load(synthetic) as archive:
            assert archive['data'].shape == real.shape and np.array_equal(archive['label'], labels)
        state = torch.load(weight, map_location='cpu', weights_only=True)
        tensor_hash = hashlib.sha256()
        for name, tensor in sorted(state.items()):
            tensor_hash.update(name.encode())
            tensor_hash.update(tensor.numpy().tobytes())
        states.append(state)
        audits.append(dict(checkpoint_seed=checkpoint, original_mean_ds=float(means[checkpoint]),
                           checkpoint_path=str(weight.relative_to(ROOT)), checkpoint_sha256=sha(weight),
                           tensor_sha256=tensor_hash.hexdigest(), training_epochs=len(history),
                           best_epoch=int(stage['best_epoch']), best_loss=float(stage['best_diffusion_loss']),
                           training_started_at=metadata['started_at'],
                           frequency_parameters={k: v.tolist() for k, v in state.items() if k.endswith('threshold_param')}))
        paths.extend([weight, metadata_path, stage_path, history_path, synthetic])
    assert states[0].keys() == states[1].keys()
    different = sum(not torch.equal(states[0][k], states[1][k]) for k in states[0])
    assert different > 0 and audits[0]['tensor_sha256'] != audits[1]['tensor_sha256']
    dump(RUN / 'checkpoint_audit.json', dict(checkpoints=audits, different_tensor_count=different))
    config = dict(dataset='FD001', checkpoint_training_seeds=CHECKPOINTS, sampling_seeds=SAMPLING,
                  discriminator_seeds=EVALUATORS, sample_count=15931, window=48, sensors=14,
                  model_name='DiffUnet_fre', model_dim=32, dim_mults=[1, 2], cond_drop_prob=.2,
                  sampling_algorithm='ddpm', diffusion_steps=1000, beta_schedule='linear',
                  objective='pred_noise', cond_scale=3., sampling_batch_size=256,
                  frequency_mask_mode='hard_random_quantile', frequency_threshold=-1.,
                  frequency_threshold_init=.25, frequency_mask_temperature=.1,
                  discriminator_hidden=7, discriminator_updates=3000, discriminator_batch_size=128,
                  discriminator_optimizer='TensorFlow Adam defaults', discriminator_train_rate=.8,
                  split='existing FD001 full-data w48 discriminator split; shared across all cells',
                  failure='mean(five evaluator discriminative scores) > 0.3, strict and unrounded',
                  run_order=[[c, s] for s in SAMPLING for c in CHECKPOINTS],
                  stop='6 sampling runs + 30 discriminator evaluations; no tuning or extra seeds',
                  checkpoint_selection='historical minimum and maximum mean DS; exploratory extremes')
    dump(RUN / 'config.json', config)
    paths.extend([Path(__file__), SUMMARY, PROTOCOL, RUN / 'config.json', RUN / 'frozen_inputs.npz',
                  RUN / 'checkpoint_audit.json', split_path,
                  ROOT / 'results/fd001_w48_public_code_evaluation_raw.csv',
                  ROOT / 'results/fd001_w48_public_code_manifest.json'])
    paths.extend(sorted(UPSTREAM.glob('*.py')))
    paths.extend(sorted((UPSTREAM / 'DiffusionFreeGuidence').rglob('*.py')))
    paths.extend([UPSTREAM / 'data/CMAPSSDataset.py', UPSTREAM / 'data/train_FD001.txt',
                  UPSTREAM / 'data/test_FD001.txt', UPSTREAM / 'data/RUL_FD001.txt',
                  UPSTREAM / 'measure_score/Utils/discriminative_metric.py',
                  UPSTREAM / 'measure_score/Utils/metric_utils.py'])
    patch = subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT)
    (RUN / 'workspace.diff').write_bytes(patch)
    (RUN / 'git-status.txt').write_bytes(subprocess.check_output(['git', 'status', '--short'], cwd=ROOT))
    (RUN / 'environment-freeze.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    paths.extend([RUN / 'workspace.diff', RUN / 'git-status.txt', RUN / 'environment-freeze.txt'])
    manifest = dict(run_id=RUN.name, scope_class='diagnostic', role='supporting',
                    scope_contract='scope-20261029-fd001-checkpoint-resampling', status='prepared',
                    authorization='User explicitly requests fixed 2 checkpoints x 3 sampling seeds x 5 discriminator evaluations',
                    prepared_at=now(), changed_factor='sampling RNG within each fixed checkpoint; repeat existing discriminator evaluator seeds',
                    no_generator_training=True, command=f'{sys.executable} {Path(__file__)} run',
                    upstream_commit=subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip(),
                    workspace_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    python=sys.version, platform=platform.platform(),
                    inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(paths)},
                    completed_sampling=0, completed_discriminator_evaluations=0, outputs=[])
    dump(RUN / 'run-manifest.json', manifest)
    print(json.dumps(dict(status='prepared', checkpoints=audits, different_tensor_count=different), indent=2))


def worker_sample(checkpoint, seed):
    import numpy as np
    import torch
    check_inputs()
    assert torch.cuda.is_available(), 'Sampling needs original CUDA environment'
    sys.path.insert(0, str(UPSTREAM))
    import TrainCondition as training
    import GaussianDiffusion as diffusion_module
    from utils import torch_seed
    import wandb
    config = json.loads((RUN / 'config.json').read_text())
    with np.load(RUN / 'frozen_inputs.npz') as archive:
        labels = torch.from_numpy(archive['label'].copy())
    destination = OUT / 'samples' / f'checkpoint{checkpoint}_sample{seed}.npz'
    record_path = OUT / f'sample_checkpoint{checkpoint}_seed{seed}.json'
    assert not destination.exists() and not record_path.exists(), 'No implicit overwrite or rerun'
    rng_hashes = {}
    torch_seed(seed)
    rng_hashes['torch_cpu'] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    rng_hashes['torch_cuda'] = hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    rng_hashes['numpy_state'] = hashlib.sha256(np.random.get_state()[1].tobytes()).hexdigest()
    weight = ROOT / f'checkpoints/FD001_w48_seed{checkpoint}.pth'
    weight_hash = sha(weight)
    start, started_at = time.time(), now()

    def progress(iterable, **kwargs):
        total = len(iterable)
        for number, value in enumerate(iterable, 1):
            yield value
            dump(OUT / f'progress_checkpoint{checkpoint}_sample{seed}.json',
                 dict(checkpoint_seed=checkpoint, sampling_seed=seed, completed_batches=number,
                      total_batches=total, elapsed_seconds=time.time() - start, updated_at=now()))
            print(f'sampling checkpoint={checkpoint} seed={seed} batch={number}/{total} elapsed={time.time()-start:.1f}s', flush=True)

    training.tqdm = progress
    diffusion_module.tqdm = lambda iterable, **kwargs: iterable
    training.train = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('Generator training prohibited'))
    args = SimpleNamespace(model_name='DiffUnet_fre', input_size=14, window_size=48,
                           frequency_mask_mode=config['frequency_mask_mode'],
                           frequency_threshold_init=.25, frequency_mask_temperature=.1,
                           frequency_threshold=-1., device='cuda', T=1000, schedule_name='linear',
                           batch_size=256, model_path=str(weight), syndata_path=str(destination))
    wandb.init(project='fd001-fixed-checkpoint-resampling', mode='disabled')
    result = training.sample(args, labels)
    wandb.finish()
    torch.cuda.synchronize()
    assert result.shape == (15931, 48, 14) and np.isfinite(result).all()
    with np.load(destination) as archive:
        assert np.array_equal(archive['label'], labels.numpy()) and np.array_equal(archive['data'], result)
    assert sha(weight) == weight_hash
    record = dict(checkpoint_seed=checkpoint, sampling_seed=seed, started_at=started_at, completed_at=now(),
                  elapsed_seconds=time.time()-start, shape=list(result.shape), finite=True,
                  label_sha256=hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
                  data_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
                  artifact_path=str(destination.relative_to(ROOT)), artifact_sha256=sha(destination),
                  checkpoint_sha256=weight_hash, pre_model_construction_rng_sha256=rng_hashes,
                  torch=torch.__version__, torch_cuda=torch.version.cuda,
                  cudnn_version=torch.backends.cudnn.version(),
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  gpu=torch.cuda.get_device_name(),
                  frequency_thresholds=getattr(args, 'frequency_thresholds', []))
    dump(record_path, record)
    print(json.dumps(record, indent=2), flush=True)


def worker_evaluate(checkpoint, seed):
    import numpy as np
    check_inputs()
    sys.path.insert(0, str(UPSTREAM))
    from measure_score.Utils.discriminative_metric import discriminative_score_metrics
    import tensorflow as tf
    destination = OUT / f'evaluation_checkpoint{checkpoint}_sample{seed}.csv'
    assert not destination.exists(), 'No implicit overwrite or rerun'
    artifact = OUT / 'samples' / f'checkpoint{checkpoint}_sample{seed}.npz'
    sample_record = json.loads((OUT / f'sample_checkpoint{checkpoint}_seed{seed}.json').read_text())
    assert sha(artifact) == sample_record['artifact_sha256']
    with np.load(artifact) as archive:
        generated, labels = archive['data'].copy(), archive['label'].copy()
    with np.load(RUN / 'frozen_inputs.npz') as archive:
        real = archive['real'].copy()
        assert np.array_equal(archive['label'], labels)
        split = tuple(archive[k].copy() for k in ['discriminator_real_train', 'discriminator_real_test',
                                                  'discriminator_generated_train', 'discriminator_generated_test'])
    rows = []
    for evaluator in EVALUATORS:
        start, started_at = time.time(), now()
        score, fake, real_accuracy = discriminative_score_metrics(real, generated, eval_seed=evaluator, split_indices=split)
        assert abs(float(score) - abs(.5 - (float(fake) + float(real_accuracy))/2)) < 1e-12
        rows.append(dict(checkpoint_seed=checkpoint, sampling_seed=seed, evaluator_seed=evaluator,
                         discriminative_score=float(score), fake_accuracy=float(fake), real_accuracy=float(real_accuracy),
                         elapsed_seconds=time.time()-start, started_at=started_at, completed_at=now()))
        temporary = destination.with_suffix('.csv.tmp')
        with temporary.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)
        print(f'evaluated checkpoint={checkpoint} sample={seed} evaluator={evaluator} DS={score:.6f}', flush=True)
    dump(OUT / f'evaluation_environment_checkpoint{checkpoint}_sample{seed}.json',
         dict(tensorflow=tf.__version__, tensorflow_gpu_devices=[str(x) for x in tf.config.list_physical_devices('GPU')],
              tf_deterministic_ops=os.environ['TF_DETERMINISTIC_OPS'],
              tf_enable_onednn_opts=os.environ['TF_ENABLE_ONEDNN_OPTS'],
              source_sha256=sha(UPSTREAM / 'measure_score/Utils/discriminative_metric.py')))


def run(resume=False):
    manifest = check_inputs()
    if resume:
        assert manifest['status'] == 'failed' and manifest.get('compatibility_recovery_revision') == 1
        manifest.update(status='running', resumed_at=now())
    else:
        assert manifest['status'] == 'prepared', 'No automatic restart of an attempted run'
        manifest.update(status='running', started_at=now())
    dump(RUN / 'run-manifest.json', manifest)
    try:
        for sampling in SAMPLING:
            for checkpoint in CHECKPOINTS:
                for stage in ['sample', 'evaluate']:
                    if resume and stage == 'sample':
                        record_path = OUT / f'sample_checkpoint{checkpoint}_seed{sampling}.json'
                        if record_path.exists():
                            record = json.loads(record_path.read_text())
                            assert sha(ROOT / record['artifact_path']) == record['artifact_sha256']
                            print(f'{now()} retain completed sample: checkpoint={checkpoint}, sampling={sampling}', flush=True)
                            continue
                    manifest.update(current_stage=stage, current_checkpoint=checkpoint, current_sampling_seed=sampling)
                    dump(RUN / 'run-manifest.json', manifest)
                    command = [sys.executable, str(Path(__file__).resolve()), stage, str(checkpoint), str(sampling)]
                    log = OUT / 'logs' / f'{stage}_checkpoint{checkpoint}_seed{sampling}.log'
                    with log.open('a' if resume else 'w') as handle:
                        subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=handle,
                                       stderr=subprocess.STDOUT, check=True)
                    if stage == 'sample':
                        manifest['completed_sampling'] += 1
                    else:
                        manifest['completed_discriminator_evaluations'] += 5
                    manifest['outputs'].append(dict(stage=stage, checkpoint_seed=checkpoint,
                                                     sampling_seed=sampling, log=str(log.relative_to(ROOT))))
                    dump(RUN / 'run-manifest.json', manifest)
                    print(f'{now()} finished {stage}: checkpoint={checkpoint}, sampling={sampling}', flush=True)
        check_inputs()
        manifest.update(status='computed_pending_summary', computation_completed_at=now())
        dump(RUN / 'run-manifest.json', manifest)
        subprocess.run([sys.executable, str(SUMMARY)], cwd=ROOT, env=os.environ.copy(), check=True)
    except BaseException:
        manifest.update(status='failed', failed_at=now(), failure=traceback.format_exc())
        dump(RUN / 'run-manifest.json', manifest)
        with (ROOT / 'experiment-ledger.md').open('a') as handle:
            handle.write(f'\n## `{RUN.name}` failed\n\n- [Observed] Supervisor stopped at {now()}; see run-manifest failure and per-worker logs. Completed sampling={manifest["completed_sampling"]}, discriminator evaluations={manifest["completed_discriminator_evaluations"]}. No automatic retries or configuration changes.\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'resume', 'sample', 'evaluate'])
    parser.add_argument('checkpoint', nargs='?', type=int, choices=CHECKPOINTS)
    parser.add_argument('sampling_seed', nargs='?', type=int, choices=SAMPLING)
    args = parser.parse_args()
    environment()
    if args.action in ['sample', 'evaluate']:
        assert args.checkpoint is not None and args.sampling_seed is not None
        (worker_sample if args.action == 'sample' else worker_evaluate)(args.checkpoint, args.sampling_seed)
    elif args.action == 'resume':
        run(resume=True)
    else:
        (prepare if args.action == 'prepare' else run)()
