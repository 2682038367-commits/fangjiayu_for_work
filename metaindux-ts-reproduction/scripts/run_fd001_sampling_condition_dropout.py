"""FD001 inference-only condition retention with verified legacy noise streams."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from types import SimpleNamespace
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/diagnostic-fd001-sampling-condition-dropout-20261030'
OUT = ROOT / 'results/diagnostics/fd001_sampling_condition_dropout_20261030'
BASE_RUN = ROOT / 'runs/diagnostic-fd001-checkpoint-resampling-20261029'
BASE_OUT = ROOT / 'results/diagnostics/fd001_checkpoint_resampling_20261029'
PROTOCOL = ROOT / 'docs/operations/FD001_SAMPLING_CONDITION_DROPOUT.md'
REPORTER = ROOT / 'analysis/diagnostics/summarize_fd001_sampling_condition_dropout.py'
CHECKPOINTS, SAMPLING, EVALUATORS = [13, 23], [103, 113, 123], [3, 13, 23, 33, 43]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment():
    for key, value in dict(WANDB_MODE='disabled', TF_DETERMINISTIC_OPS='1',
                           TF_ENABLE_ONEDNN_OPTS='0', TF_CPP_MIN_LOG_LEVEL='2',
                           CUBLAS_WORKSPACE_CONFIG=':4096:8', OPENBLAS_NUM_THREADS='1',
                           MPLCONFIGDIR=str(ROOT / '.cache/matplotlib')).items():
        os.environ.setdefault(key, value)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_inputs():
    manifest = json.loads((RUN / 'run-manifest.json').read_text())
    for name, expected in manifest['inputs_sha256'].items():
        assert sha(ROOT / name) == expected, f'Frozen input changed: {name}'
    return manifest


def prepare():
    import numpy as np
    import torch
    assert not (RUN / 'run-manifest.json').exists(), 'Preserve preparation'
    RUN.mkdir(parents=True, exist_ok=True)
    for folder in [OUT, OUT/'samples', OUT/'probes', OUT/'logs']:
        folder.mkdir(parents=True, exist_ok=True)
    original = json.loads((BASE_RUN / 'run-manifest.json').read_text())
    assert original['status'] == 'complete'
    for name, expected in original['inputs_sha256'].items():
        assert sha(ROOT/name) == expected, name
    shutil.copy2(BASE_RUN/'frozen_inputs.npz', RUN/'frozen_inputs.npz')
    config = dict(checkpoints=CHECKPOINTS, screening_sampling_seed=103, confirmation_sampling_seeds=[113,123],
                  evaluator_seeds=EVALUATORS, controls='reuse immutable previous resampling cells',
                  factor='sampling conditional CFG branch effective drop probability0.2 to0; consume original RNG draw',
                  unconditional_drop_probability=1., cfg_scale=3., diffusion='original DDPM1000 linear pred_noise',
                  sample_count=15931, shape=[15931,48,14], batch_size=256,
                  noise_control='exact actual noise/mask tensors and CUDA states checked against independent legacy-schedule generator',
                  cached_control_link='frozen provenance + same preconstruction RNG state + first256 original-control replay',
                  prefix_max_abs_tolerance=1e-6,
                  clear_change=dict(ds_absolute_mean_change=.03, paired_same_direction_at_least=4,
                                    inverse_reference='dev_matched', inverse_absolute_rmse_change=1.,
                                    inverse_relative_rmse_change=.10, combination='DS gate OR inverse gate, either direction, either checkpoint'),
                  stop='103 screen; if triggered BOTH checkpoints113/123, max6 new full samples/30 new discriminator fits',
                  caveats=['Historical extreme checkpoints; exploratory conditional confirmation selection',
                           'Real-trained predictor readout is a proxy, not physical truth or downstream utility',
                           'No full noise tape from cached controls; use source audit, prefix replay and reconstructed legacy schedule'])
    dump(RUN/'config.json', config)
    paths = [Path(__file__), REPORTER, PROTOCOL, RUN/'config.json', RUN/'frozen_inputs.npz',
             ROOT/'scripts/run_fd001_resampling_diagnostic.py', BASE_RUN/'run-manifest.json',
             ROOT/'analysis/diagnostics/real_predictor_condition_audit.py',
             ROOT/'results/diagnostics/real_predictor_condition_w48/FD001_predictions.npz',
             ROOT/'upstream/models/LSTM.py']
    paths.extend(ROOT/name for name in original['inputs_sha256'])
    for c in CHECKPOINTS:
        for s in SAMPLING:
            for name in [f'samples/checkpoint{c}_sample{s}.npz',f'sample_checkpoint{c}_seed{s}.json',
                         f'evaluation_checkpoint{c}_sample{s}.csv']:
                path=BASE_OUT/name
                rel=str(path.relative_to(ROOT))
                assert sha(path)==original['outputs_sha256'][rel], rel
                paths.append(path)
    sys.path.insert(0,str(ROOT/'upstream'))
    torch.set_num_threads(1)
    with np.load(ROOT/'results/diagnostics/real_predictor_condition_w48/FD001_predictions.npz') as z:
        old_indices=z['generated_test_indices'][1]
        old_predictions=z['generated_test_predictions'][1]
    with np.load(ROOT/'outputs/FD001_w48_seed13.npz') as z:
        reference=z['data'][old_indices]
    checks=[]
    for e,expected in zip(EVALUATORS,old_predictions):
        path=ROOT/f'checkpoints/evaluators/real_data_baseline_w48/FD001_eval{e}.pth'
        model=torch.load(path,map_location='cpu',weights_only=False).eval()
        with torch.no_grad():pred=model(torch.from_numpy(reference)).numpy().ravel()
        error=float(np.max(abs(pred-expected)))
        assert error<1e-3
        checks.append(dict(evaluator_seed=e,max_absolute_prediction_replay_difference=error))
        paths.append(path)
    write_csv(OUT/'real_predictor_replay_checks.csv',checks)
    dump(RUN/'run-manifest.json',dict(run_id=RUN.name,scope_class='diagnostic',role='supporting',
         scope_contract='scope-20261030-fd001-sampling-condition-dropout',status='prepared',prepared_at=now(),
         command=f'{sys.executable} {Path(__file__)} run', changed_factor=config['factor'],
         inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(paths)},
         completed_sampling=0,completed_discriminator_evaluations=0,completed_cells=[],confirmation_triggered=None))
    print('Prepared fixed weights, cached controls, explicit RNG pairing and proxy-model replay checks.')


class LegacyNoiseTrace:
    """Check every real draw against an independent generator following legacy calls."""
    def __init__(self, torch, unet):
        self.torch,self.unet=torch,unet
        self.randn,self.randn_like,self.mask=torch.randn,torch.randn_like,unet.prob_mask_like
        self.active=False
        self.normals=self.conditions=self.unconditionals=self.elements=0
        self.fingerprint=hashlib.sha256()

    def start(self):
        if self.active:return
        self.reference=self.torch.Generator(device='cuda')
        self.reference.set_state(self.torch.cuda.get_rng_state())
        self.start_sha=hashlib.sha256(self.reference.get_state().numpy().tobytes()).hexdigest()
        self.active=True

    def before(self,tag,shape):
        state=self.torch.cuda.get_rng_state()
        assert self.torch.equal(state,self.reference.get_state()),'RNG state mismatch before '+tag
        self.fingerprint.update(tag.encode()+str(tuple(shape)).encode()+state.numpy().tobytes())

    def after(self):
        assert self.torch.equal(self.torch.cuda.get_rng_state(),self.reference.get_state()),'RNG state mismatch after draw'

    def normal(self,*args,**kwargs):
        if not self.active:return self.randn(*args,**kwargs)
        assert 'generator' not in kwargs
        shape=args[0] if len(args)==1 and isinstance(args[0],(tuple,list)) else args
        self.before('initial_normal',shape)
        actual=self.randn(*args,**kwargs)
        reference=self.randn(*args,**kwargs,generator=self.reference)
        assert actual.is_cuda and actual.is_contiguous() and self.torch.equal(actual,reference)
        self.after();self.normals+=1;self.elements+=actual.numel()
        return actual

    def normal_like(self,x,*args,**kwargs):
        if not self.active:return self.randn_like(x,*args,**kwargs)
        assert not args and x.is_cuda and x.is_contiguous()
        self.before('step_normal',x.shape)
        actual=self.randn_like(x,**kwargs)
        reference=self.randn(tuple(actual.shape),device=actual.device,dtype=actual.dtype,
                             layout=actual.layout,generator=self.reference)
        assert self.torch.equal(actual,reference),'Normal tensors differ'
        self.after();self.normals+=1;self.elements+=actual.numel()
        return actual

    def keep_condition(self,shape,prob,device):
        assert self.active
        if abs(prob-.8)<1e-12:
            self.before('conditional_uniform',shape)
            legacy=self.mask(shape,prob,device)
            counterpart=self.torch.zeros(shape,device=device).float().uniform_(0,1,generator=self.reference)<prob
            assert self.torch.equal(legacy,counterpart),'Condition random draws differ'
            self.after();self.conditions+=1
            return self.torch.ones_like(legacy)
        assert prob==0.,'Unexpected CFG mask probability'
        self.before('unconditional_no_random_draw',shape)
        mask=self.mask(shape,prob,device)
        self.after();self.unconditionals+=1
        assert not bool(mask.any())
        return mask

    def install(self):
        self.torch.randn,self.torch.randn_like=self.normal,self.normal_like
        self.unet.prob_mask_like=self.keep_condition

    def restore(self):
        self.torch.randn,self.torch.randn_like=self.randn,self.randn_like
        self.unet.prob_mask_like=self.mask

    def result(self):
        assert self.normals==self.conditions==self.unconditionals==63000
        assert self.elements==15931*48*14*1000
        self.after()
        return dict(status='passed',normal_tensors_exactly_matched=self.normals,
                    conditional_uniform_draws_exactly_matched=self.conditions,
                    unconditional_calls_retained=self.unconditionals,normal_scalar_count=self.elements,
                    conditional_dropped_windows=0,start_cuda_rng_sha256=self.start_sha,
                    final_cuda_rng_sha256=hashlib.sha256(self.reference.get_state().numpy().tobytes()).hexdigest(),
                    legacy_schedule_fingerprint=self.fingerprint.hexdigest(),
                    cached_controls_have_full_noise_tape=False)


def sample(checkpoint,seed):
    import numpy as np
    import torch
    check_inputs()
    assert torch.cuda.is_available()
    sys.path.insert(0,str(ROOT/'upstream'))
    import TrainCondition as training
    import GaussianDiffusion as diffusion
    import DiffusionFreeGuidence.Unet1D_fre as unet
    from utils import torch_seed
    import wandb
    diffusion.tqdm=lambda iterable,**kwargs:iterable
    training.train=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Training forbidden'))
    with np.load(RUN/'frozen_inputs.npz') as z:labels=torch.from_numpy(z['label'].copy())
    with np.load(BASE_OUT/f'samples/checkpoint{checkpoint}_sample{seed}.npz') as z:prefix=z['data'][:256].copy()
    cached_record=json.loads((BASE_OUT/f'sample_checkpoint{checkpoint}_seed{seed}.json').read_text())
    args=SimpleNamespace(model_name='DiffUnet_fre',input_size=14,window_size=48,frequency_mask_mode='hard_random_quantile',
         frequency_threshold_init=.25,frequency_mask_temperature=.1,frequency_threshold=-1.,device='cuda',T=1000,
         schedule_name='linear',batch_size=256,model_path=str(ROOT/f'checkpoints/FD001_w48_seed{checkpoint}.pth'))
    wandb.init(project='fd001-sampling-condition-only',mode='disabled')
    training.tqdm=lambda iterable,**kwargs:iterable
    torch_seed(seed)
    args.syndata_path=str(OUT/f'probes/checkpoint{checkpoint}_sample{seed}_legacy_prefix.npz')
    probe=training.sample(args,labels[:256])
    maximum=float(np.max(abs(probe-prefix)))
    replay=dict(checkpoint_seed=checkpoint,sampling_seed=seed,max_absolute_difference=maximum,
                bit_identical=bool(np.array_equal(probe,prefix)),tolerance=1e-6,passed=maximum<=1e-6)
    dump(OUT/f'prefix_replay_checkpoint{checkpoint}_sample{seed}.json',replay)
    assert replay['passed'],'Cached control prefix did not replay; stop without substituting controls'
    torch_seed(seed)
    rng=dict(torch_cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
             torch_cuda=hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest(),
             numpy_state=hashlib.sha256(np.random.get_state()[1].tobytes()).hexdigest())
    assert rng==cached_record['pre_model_construction_rng_sha256']
    trace=LegacyNoiseTrace(torch,unet)
    started_at,start=now(),time.time()

    def progress(iterable,**kwargs):
        trace.start()
        for number,value in enumerate(iterable,1):
            yield value
            dump(OUT/f'progress_checkpoint{checkpoint}_sample{seed}.json',dict(checkpoint_seed=checkpoint,
                 sampling_seed=seed,completed_batches=number,total_batches=len(iterable),elapsed_seconds=time.time()-start,
                 matched_noise_tensors=trace.normals,updated_at=now()))
            print(f'checkpoint={checkpoint} sample={seed} batch={number}/{len(iterable)} noise_checks={trace.normals}',flush=True)
    training.tqdm=progress
    trace.install()
    artifact=OUT/f'samples/checkpoint{checkpoint}_sample{seed}.npz'
    assert not artifact.exists()
    args.syndata_path=str(artifact)
    try:
        result=training.sample(args,labels)
        noise_audit=trace.result()
    finally:
        trace.restore()
        wandb.finish()
    assert result.shape==(15931,48,14) and np.isfinite(result).all()
    with np.load(artifact) as z:assert np.array_equal(z['label'],labels.numpy())
    record=dict(checkpoint_seed=checkpoint,sampling_seed=seed,started_at=started_at,completed_at=now(),
                elapsed_seconds=time.time()-start,artifact_path=str(artifact.relative_to(ROOT)),artifact_sha256=sha(artifact),
                checkpoint_sha256=sha(Path(args.model_path)),pre_model_construction_rng_sha256=rng,
                data_sha256=hashlib.sha256(result.tobytes()).hexdigest(),prefix_replay=replay,noise_audit=noise_audit,
                frequency_thresholds=args.frequency_thresholds,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                changed_factor='conditional mask force-retain after original random draw; unconditional branch unchanged')
    dump(OUT/f'sample_checkpoint{checkpoint}_seed{seed}.json',record)


def evaluate(checkpoint,seed):
    module=load_module(ROOT/'scripts/run_fd001_resampling_diagnostic.py','original_discriminator_worker')
    module.RUN,module.OUT=RUN,OUT
    module.worker_evaluate(checkpoint,seed)


def condition(checkpoint,seed):
    import numpy as np
    import torch
    check_inputs()
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'upstream'))
    with np.load(ROOT/'results/diagnostics/real_predictor_condition_w48/FD001_predictions.npz') as z:
        position=list(z['generation_seeds']).index(checkpoint)
        references=dict(test_matched=z['generated_test_indices'][position].copy(),
                        dev_matched=z['generated_dev_indices'][position].copy())
        reference_labels=dict(test_matched=z['real_test_labels'].copy(),dev_matched=z['real_dev_labels'].copy())
    references['all_windows']=np.arange(15931)
    bins=[('all',0,125),('0',0,0),('1',1,1),('2-5',2,5),('6-30',6,30),('0-30',0,30),
          ('31-60',31,60),('61-90',61,90),('91-124',91,124),('125',125,125)]
    rows=[]
    for arm,folder in [('control',BASE_OUT),('retain_condition',OUT)]:
        with np.load(folder/f'samples/checkpoint{checkpoint}_sample{seed}.npz') as z:
            data,labels=z['data'].copy(),z['label'].ravel().astype(np.float64)
        for name,y in reference_labels.items():assert np.array_equal(labels[references[name]],y)
        all_predictions=[]
        for evaluator in EVALUATORS:
            model=torch.load(ROOT/f'checkpoints/evaluators/real_data_baseline_w48/FD001_eval{evaluator}.pth',
                             map_location='cpu',weights_only=False).eval()
            with torch.no_grad():
                prediction=np.concatenate([model(torch.from_numpy(data[i:i+256])).numpy().ravel()
                                          for i in range(0,len(data),256)]).astype(np.float64)
            assert np.isfinite(prediction).all()
            all_predictions.append(prediction)
            for reference,indices in references.items():
                for name,lo,hi in bins:
                    selected=indices[(labels[indices]>=lo)&(labels[indices]<=hi)]
                    if not len(selected):continue
                    err=prediction[selected]-labels[selected]
                    rows.append(dict(checkpoint_seed=checkpoint,sampling_seed=seed,arm=arm,evaluator_seed=evaluator,
                        reference=reference,bin=name,n_windows=len(selected),condition_mean=float(labels[selected].mean()),
                        prediction_mean=float(prediction[selected].mean()),bias=float(err.mean()),
                        mse=float(np.mean(err**2)),rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(abs(err)))))
        np.savez_compressed(OUT/f'condition_predictions_checkpoint{checkpoint}_sample{seed}_{arm}.npz',
                            prediction=np.array(all_predictions),label=labels,evaluator_seeds=np.array(EVALUATORS),**references)
    write_csv(OUT/f'condition_checkpoint{checkpoint}_sample{seed}.csv',rows)


def run():
    manifest=check_inputs()
    assert manifest['status']=='prepared'
    manifest.update(status='running',started_at=now())
    dump(RUN/'run-manifest.json',manifest)
    try:
        for seed in SAMPLING:
            if seed!=103 and not manifest['confirmation_triggered']:break
            for checkpoint in CHECKPOINTS:
                for action in ['sample','evaluate','condition']:
                    manifest.update(current_stage=action,current_checkpoint=checkpoint,current_sampling_seed=seed)
                    dump(RUN/'run-manifest.json',manifest)
                    with (OUT/f'logs/{action}_checkpoint{checkpoint}_sample{seed}.log').open('w') as handle:
                        subprocess.run([sys.executable,str(Path(__file__).resolve()),action,str(checkpoint),str(seed)],
                                       cwd=ROOT,env=os.environ.copy(),stdout=handle,stderr=subprocess.STDOUT,check=True)
                    if action=='sample':manifest['completed_sampling']+=1
                    if action=='evaluate':manifest['completed_discriminator_evaluations']+=5
                    dump(RUN/'run-manifest.json',manifest)
                    print(f'{now()} complete {action} checkpoint={checkpoint} sample={seed}',flush=True)
                manifest['completed_cells'].append([checkpoint,seed])
                dump(RUN/'run-manifest.json',manifest)
            if seed==103:
                subprocess.run([sys.executable,str(REPORTER),'--screen'],cwd=ROOT,env=os.environ.copy(),check=True)
                decision=json.loads((OUT/'screening_decision.json').read_text())
                manifest['confirmation_triggered']=decision['confirmation_triggered']
                manifest['screening_decision_sha256']=sha(OUT/'screening_decision.json')
                dump(RUN/'run-manifest.json',manifest)
        check_inputs()
        manifest.update(status='computed_pending_summary',computation_completed_at=now())
        dump(RUN/'run-manifest.json',manifest)
        subprocess.run([sys.executable,str(REPORTER)],cwd=ROOT,env=os.environ.copy(),check=True)
    except BaseException:
        manifest.update(status='failed',failed_at=now(),failure=traceback.format_exc())
        dump(RUN/'run-manifest.json',manifest)
        with (ROOT/'experiment-ledger.md').open('a') as handle:
            handle.write(f'\n## `{RUN.name}` failed\n\n- [Observed] Stopped {now()};sample={manifest["completed_sampling"]},DSfits={manifest["completed_discriminator_evaluations"]};see manifest/workerlog. No automatic retries or scientific changes.\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','sample','evaluate','condition'])
    parser.add_argument('checkpoint',nargs='?',type=int,choices=CHECKPOINTS)
    parser.add_argument('seed',nargs='?',type=int,choices=SAMPLING)
    args=parser.parse_args()
    environment()
    if args.action in ['sample','evaluate','condition']:
        assert args.checkpoint is not None and args.seed is not None
        globals()[args.action](args.checkpoint,args.seed)
    else:globals()[args.action]()
