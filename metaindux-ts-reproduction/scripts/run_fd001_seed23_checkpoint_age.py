"""One original-setting seed23 rerun; evaluate retained epochs15/64/70 in order."""
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
import pickle
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/diagnostic-fd001-seed23-checkpoint-age-20261030'
OUT=ROOT/'results/diagnostics/fd001_seed23_checkpoint_age_20261030'
WEIGHTS=ROOT/'checkpoints/diagnostics/fd001_seed23_checkpoint_age_20261030'
PROTOCOL=ROOT/'docs/operations/FD001_SEED23_CHECKPOINT_AGE.md'
REPORTER=ROOT/'analysis/diagnostics/summarize_fd001_seed23_checkpoint_age.py'
ANCHOR=ROOT/'results/diagnostics/fd001_sampling_condition_dropout_paired_20261030'
ANCHOR_RUN=ROOT/'runs/diagnostic-fd001-sampling-condition-dropout-paired-20261030'
CORE=ROOT/'scripts/run_fd001_sampling_condition_dropout_paired.py'
EPOCHS=[15,64,70];EVALUATORS=[3,13,23,33,43]


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module
def environment():
    for k,v in dict(WANDB_MODE='disabled',TF_DETERMINISTIC_OPS='1',TF_ENABLE_ONEDNN_OPTS='0',
        TF_CPP_MIN_LOG_LEVEL='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1',
        MPLCONFIGDIR=str(ROOT/'.cache/matplotlib')).items():os.environ.setdefault(k,v)
def check_inputs():
    m=json.loads((RUN/'run-manifest.json').read_text())
    for name,expected in m['inputs_sha256'].items():assert sha(ROOT/name)==expected,name
    return m


def prepare():
    import numpy as np
    assert not (RUN/'run-manifest.json').exists()
    for p in [RUN,OUT,OUT/'logs',WEIGHTS]:p.mkdir(parents=True,exist_ok=True)
    for epoch in EPOCHS:(OUT/f'epoch{epoch}/retain_condition/samples').mkdir(parents=True,exist_ok=True)
    inventory=[]
    for suffix in ['*.pth','*.pt','*.ckpt','*.safetensors']:
        for p in ROOT.rglob(suffix):
            if any(s in ['.venv','.git'] for s in p.relative_to(ROOT).parts):continue
            inventory.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
                 category='downstream_evaluator' if 'evaluators' in p.parts else 'generator_or_other'))
    write_csv(OUT/'existing_weight_inventory.csv',inventory)
    relevant=[r for r in inventory if 'FD001' in r['path'] and r['category']=='generator_or_other']
    search=dict(weight_file_count=len(inventory),relevant_generator_candidates=relevant,
                attributable_original_seed23_weights=['checkpoints/FD001_w48_seed23.pth'],
                attributable_epoch64_or70_found=False,
                exclusions='Other tasks/windows/seeds and supplied pretrained weight lack matching seed23 late provenance',
                search_scope='workspace, including hidden directories; skip .venv and .git')
    dump(OUT/'archive_search.json',search)
    previous=json.loads((ANCHOR_RUN/'run-manifest.json').read_text());assert previous['status']=='complete'
    for group in ['inputs_sha256','outputs_sha256']:
        for name,h in previous[group].items():assert sha(ROOT/name)==h,name
    shutil.copy2(ANCHOR_RUN/'frozen_inputs.npz',RUN/'frozen_inputs.npz')
    config=dict(training_seed=23,training_epochs=70,batch_size=256,batches_per_epoch=63,updates=4410,
       observed_epochs_one_based=EPOCHS,observed_update_counts={str(e):e*63 for e in EPOCHS},
       optimizer='Adam',learning_rate=.002,lr_schedule='warmup_cosine',warmup_multiplier=2.5,grad_clip=1.,
       training_condition_dropout=.2,training_numeric_batch_zero_probability=.1,
       model='original UNet1D_fre dim32/(1,2)/14, hard_random_quantile; no threshold override',
       sampling_seed=103,sampling_variant='always retain condition; consume original RNG mask; null CFG branch unchanged',
       sampling='original DDPM1000 linear pred_noise CFG3 batch256',sample_count=15931,evaluator_seeds=EVALUATORS,
       original_epoch15_anchor=str((ANCHOR/'retain_condition').relative_to(ROOT)),
       reproduction=dict(max_abs_mean_ds_difference=.03,same_ds_failure_category=True,max_relative_dev_inverse_rmse_difference=.10),
       clear_later_gain=dict(min_mean_ds_reduction=.03,min_paired_ds_decreases=4),
       stop='one70epoch rerun, epoch15 reproduction gate before later sampling;max3fullsamples/15newDSfits, no extra seeds/training',
       limitation='Original late weights missing; conclusions concern a same-setting rerun, not direct historical proof')
    dump(RUN/'config.json',config)
    paths=[Path(__file__),REPORTER,PROTOCOL,CORE,RUN/'config.json',RUN/'frozen_inputs.npz',
           OUT/'existing_weight_inventory.csv',OUT/'archive_search.json',ANCHOR_RUN/'run-manifest.json',
           ROOT/'results/runs/FD001_w48_seed23_metadata.json',ROOT/'results/runs/FD001_w48_seed23_metrics.generation.json',
           ROOT/'logs/FD001_w48_seed23_training_loss.csv',ROOT/'checkpoints/FD001_w48_seed23.pth',
           ROOT/'configs/paper_fd001_w48_public_code.yaml',ROOT/'notes/pretrained_fd001_validation.md',
           ROOT/'results/diagnostics/real_predictor_condition_w48/FD001_predictions.npz']
    paths.extend(ROOT/name for name in previous['inputs_sha256'])
    for name in ['samples/checkpoint23_sample103.npz','sample_checkpoint23_seed103.json','evaluation_checkpoint23_sample103.csv']:
        paths.append(ANCHOR/'retain_condition'/name)
    for name in ['condition_predictions_checkpoint23_sample103_retain_condition.npz','condition_checkpoint23_sample103.csv']:
        paths.append(ANCHOR/name)
    for e in EVALUATORS:paths.append(ROOT/f'checkpoints/evaluators/real_data_baseline_w48/FD001_eval{e}.pth')
    (RUN/'workspace.diff').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
    (RUN/'environment-freeze.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    paths.extend([RUN/'workspace.diff',RUN/'environment-freeze.txt'])
    dump(RUN/'run-manifest.json',dict(run_id=RUN.name,scope_class='diagnostic',role='supporting',
         scope_contract='scope-20261030-fd001-seed23-checkpoint-age',status='prepared',prepared_at=now(),
         command=f'{sys.executable} {Path(__file__)} run',
         changed_factor='which retained epoch from one original-setting rerun is sampled/evaluated',
         inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(paths)},
         training_runs_completed=0,completed_sampling=0,completed_discriminator_evaluations=0,
         completed_epochs_evaluated=[],epoch15_reproduction_passed=None))
    print(json.dumps(search,indent=2));print('Prepared: exactly one seed23 training; epoch15 gate then64/70.')


def train():
    import numpy as np
    import torch
    check_inputs();assert torch.cuda.is_available()
    sys.path.insert(0,str(ROOT/'upstream'))
    meta=json.loads((ROOT/'results/runs/FD001_w48_seed23_metadata.json').read_text())
    old_argv=sys.argv;sys.argv=[str(ROOT/'upstream/MainCondition.py'),*meta['command'][2:]]
    try:from args import args
    finally:sys.argv=old_argv
    import TrainCondition as training
    from utils import torch_seed
    import wandb
    args.model_path=str(WEIGHTS/'best_training_loss.pth')
    args.loss_history_path=str(OUT/'training_loss.csv')
    args.metrics_path=str(OUT/'training_metrics.json')
    args.syndata_path=str(OUT/'unused_original_sample_path.npz')
    assert args.seed==23 and args.epoch==70 and args.batch_size==256 and args.lr==.002
    assert args.batch_label_zero_prob==.1 and args.frequency_threshold==-1.
    with np.load(RUN/'frozen_inputs.npz') as z:
        real=torch.from_numpy(z['real'].copy());labels=torch.from_numpy(z['label'].copy())
    wandb.init(project='MetaIndux-TS',tags=['all'],config=args,mode='offline',dir=str(OUT))
    torch_seed(23)
    captured={};updates=0;orders=[];batch_draws=[];snapshots=[]
    original_model=training.UNet1D_fre
    original_step=torch.optim.Adam.step
    original_sampler=torch.utils.data.RandomSampler.__iter__
    original_rand=np.random.rand

    def rng():
        return dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                    cuda=hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest(),
                    numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest())

    def model_factory(*a,**k):
        model=original_model(*a,**k)
        captured['model']=model
        captured['initial_frequency']={name:float(t.item()) for name,t in model.state_dict().items() if name.endswith('threshold_param')}
        initial=hashlib.sha256()
        for name,t in sorted(model.state_dict().items()):initial.update(name.encode()+t.numpy().tobytes())
        captured['initial_tensor_sha256']=initial.hexdigest()
        assert model.training
        return model

    def sampler_observer(sampler):
        consumed=[]
        for index in original_sampler(sampler):
            consumed.append(index);yield index
        order=np.asarray(consumed,dtype=np.int32)
        assert len(order)==15931 and np.array_equal(np.sort(order),np.arange(15931))
        orders.append(order)

    def random_observer(*a):
        value=original_rand(*a)
        if not a and Path(sys._getframe(1).f_code.co_filename).name=='TrainCondition.py':
            batch_draws.append(dict(update=updates+1,value=float(value),batch_label_zeroed=bool(value<.1)))
        return value

    def step_observer(optimizer,*a,**k):
        nonlocal updates
        result=original_step(optimizer,*a,**k);updates+=1
        if updates in [945,4032,4410]:
            epoch=updates//63;model=captured['model'];before=rng()
            path=WEIGHTS/f'epoch{epoch}.pth'
            assert not path.exists()
            state={name:t.detach().cpu().clone() for name,t in model.state_dict().items()}
            torch.save(state,path)
            steps={name:int(optimizer.state[param]['step'].item()) for name,param in model.named_parameters()
                   if 'step' in optimizer.state.get(param,{})}
            assert steps and min(steps.values())==max(steps.values())==updates
            after=rng();assert before==after,'Snapshot serialization changed RNG'
            snapshots.append(dict(epoch_one_based=epoch,updates=updates,weight_path=str(path.relative_to(ROOT)),
                 weight_sha256=sha(path),optimizer_parameter_steps=steps,
                 optimizer_parameters_without_state=[name for name,param in model.named_parameters() if not optimizer.state.get(param)],
                 snapshot_rng_before=before,snapshot_rng_after=after,snapshot_preserved_rng=True))
            dump(OUT/f'training_snapshot_epoch{epoch}.json',snapshots[-1])
        if updates%63==0:
            dump(OUT/'training_progress.json',dict(completed_epochs=updates//63,total_epochs=70,
                 actual_optimizer_updates=updates,updated_at=now()))
        return result

    training.UNet1D_fre=model_factory
    torch.optim.Adam.step=step_observer
    torch.utils.data.RandomSampler.__iter__=sampler_observer
    np.random.rand=random_observer
    start,started_at=time.time(),now()
    try:training.train(args,real,labels)
    finally:
        training.UNet1D_fre=original_model;torch.optim.Adam.step=original_step
        torch.utils.data.RandomSampler.__iter__=original_sampler;np.random.rand=original_rand
        wandb.finish()
    assert updates==4410 and len(orders)==70 and len(batch_draws)==4410
    assert [row['epoch_one_based'] for row in snapshots]==EPOCHS
    history=np.loadtxt(OUT/'training_loss.csv',delimiter=',',skiprows=1)
    old_history=np.loadtxt(ROOT/'logs/FD001_w48_seed23_training_loss.csv',delimiter=',',skiprows=1)
    assert history.shape==old_history.shape==(70,3)
    assert np.array_equal(history[:,0],np.arange(70)) and np.array_equal(history[:,2],old_history[:,2])
    old=torch.load(ROOT/'checkpoints/FD001_w48_seed23.pth',map_location='cpu',weights_only=True)
    assert all(captured['initial_frequency'][k]==float(old[k].item()) for k in captured['initial_frequency'])
    epoch15=torch.load(WEIGHTS/'epoch15.pth',map_location='cpu',weights_only=True)
    numerator=sum(float(torch.sum((epoch15[k].double()-old[k].double())**2)) for k in old)
    denominator=sum(float(torch.sum(old[k].double()**2)) for k in old)
    compare=dict(all_tensors_bit_identical=all(torch.equal(epoch15[k],old[k]) for k in old),
                 relative_l2_difference=(numerator/denominator)**.5,
                 max_absolute_tensor_difference=max(float(torch.max(abs(epoch15[k]-old[k]))) for k in old))
    np.savez_compressed(OUT/'training_epoch_orders.npz',orders=np.stack(orders))
    write_csv(OUT/'training_batch_label_draws.csv',batch_draws)
    audit=dict(status='complete',training_seed=23,started_at=started_at,completed_at=now(),elapsed_seconds=time.time()-start,
         actual_optimizer_updates=updates,actual_completed_epochs=len(orders),snapshots=snapshots,
         learning_rate_sequence_matches_original=True,
         original_epoch15_loss=float(old_history[14,1]),rerun_epoch15_loss=float(history[14,1]),
         first15_loss_max_abs_difference=float(np.max(abs(history[:15,1]-old_history[:15,1]))),
         original_training_behavior_preserved=True,initial_frequency_parameters=captured['initial_frequency'],
         initial_tensor_sha256=captured['initial_tensor_sha256'],
         data_order_sha256=hashlib.sha256(np.stack(orders).tobytes()).hexdigest(),
         best_epoch_zero_based=int(args.best_epoch),best_diffusion_loss=float(args.best_diffusion_loss),
         original_epoch15_weight_comparison=compare,
         caveat='Original cuDNN benchmark/nondeterministic settings retained; do not infer bitwise training reproduction from seed alone')
    dump(OUT/'training_audit.json',audit)
    print(json.dumps(audit,indent=2),flush=True)


def sample(epoch):
    check_inputs()
    paired=load_module(CORE,'retained_condition_noise_core')
    paired.RUN=RUN;paired.OUT=OUT/f'epoch{epoch}'
    sys.path.insert(0,str(ROOT/'upstream'))
    import TrainCondition as training
    original=training.sample
    expected=json.loads((OUT/f'training_snapshot_epoch{epoch}.json').read_text())
    weight=ROOT/expected['weight_path'];assert sha(weight)==expected['weight_sha256']
    def use_epoch(args,labels):
        args.model_path=str(weight)
        return original(args,labels)
    training.sample=use_epoch
    try:paired.sample(23,103,'retain_condition')
    finally:training.sample=original
    record=json.loads((OUT/f'epoch{epoch}/retain_condition/sample_checkpoint23_seed103.json').read_text())
    anchor=json.loads((ANCHOR/'retain_condition/sample_checkpoint23_seed103.json').read_text())
    for name in ['start_cuda_rng_sha256','final_cuda_rng_sha256','legacy_schedule_fingerprint','normal_scalar_count']:
        assert record['noise_audit'][name]==anchor['noise_audit'][name],name
    assert record['checkpoint_sha256']==expected['weight_sha256']
    dump(OUT/f'epoch{epoch}/noise_anchor_match.json',dict(status='passed',epoch=epoch,
         actual_noises_matched_to_legacy_reference=63000,anchor_fingerprint=anchor['noise_audit']['legacy_schedule_fingerprint']))


def evaluate(epoch):
    module=load_module(ROOT/'scripts/run_fd001_resampling_diagnostic.py','original_discriminator_worker')
    module.RUN=RUN;module.OUT=OUT/f'epoch{epoch}/retain_condition'
    module.worker_evaluate(23,103)


def condition(epoch):
    import numpy as np
    import torch
    check_inputs();torch.set_num_threads(1);sys.path.insert(0,str(ROOT/'upstream'))
    with np.load(OUT/f'epoch{epoch}/retain_condition/samples/checkpoint23_sample103.npz') as z:
        data,y=z['data'].copy(),z['label'].ravel().astype(np.float64)
    with np.load(ROOT/'results/diagnostics/real_predictor_condition_w48/FD001_predictions.npz') as z:
        i=list(z['generation_seeds']).index(23)
        indices=dict(test_matched=z['generated_test_indices'][i].copy(),dev_matched=z['generated_dev_indices'][i].copy(),all_windows=np.arange(len(y)))
        assert np.array_equal(y[indices['test_matched']],z['real_test_labels'])
        assert np.array_equal(y[indices['dev_matched']],z['real_dev_labels'])
    bins=[('all',0,125),('0',0,0),('1',1,1),('2-5',2,5),('6-30',6,30),('0-30',0,30),
          ('31-60',31,60),('61-90',61,90),('91-124',91,124),('125',125,125)]
    predictions=[];rows=[]
    for e in EVALUATORS:
        model=torch.load(ROOT/f'checkpoints/evaluators/real_data_baseline_w48/FD001_eval{e}.pth',map_location='cpu',weights_only=False).eval()
        with torch.no_grad():p=np.concatenate([model(torch.from_numpy(data[j:j+256])).numpy().ravel() for j in range(0,len(data),256)]).astype(np.float64)
        assert np.isfinite(p).all();predictions.append(p)
        for ref,idx in indices.items():
            for name,lo,hi in bins:
                selected=idx[(y[idx]>=lo)&(y[idx]<=hi)]
                if not len(selected):continue
                error=p[selected]-y[selected]
                rows.append(dict(epoch_one_based=epoch,evaluator_seed=e,reference=ref,bin=name,n_windows=len(selected),
                     condition_mean=float(y[selected].mean()),prediction_mean=float(p[selected].mean()),
                     bias=float(error.mean()),mse=float(np.mean(error**2)),rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.mean(abs(error)))))
    np.savez_compressed(OUT/f'epoch{epoch}/condition_predictions.npz',prediction=np.array(predictions),label=y,
                        evaluator_seeds=np.array(EVALUATORS),**indices)
    write_csv(OUT/f'epoch{epoch}/condition_metrics.csv',rows)


def run():
    m=check_inputs();assert m['status']=='prepared'
    m.update(status='running',started_at=now());dump(RUN/'run-manifest.json',m)
    def worker(action,epoch=None):
        m.update(current_stage=action,current_epoch=epoch);dump(RUN/'run-manifest.json',m)
        command=[sys.executable,str(Path(__file__).resolve()),action]+([str(epoch)] if epoch else [])
        with (OUT/'logs'/f'{action}_{epoch or "seed23"}.log').open('w') as handle:
            subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=handle,stderr=subprocess.STDOUT,check=True)
        print(f'{now()} complete {action} epoch={epoch}',flush=True)
    try:
        worker('train');m['training_runs_completed']=1
        for p in [*WEIGHTS.glob('*.pth'),OUT/'training_audit.json',*OUT.glob('training_snapshot_epoch*.json')]:
            m['inputs_sha256'][str(p.relative_to(ROOT))]=sha(p)
        dump(RUN/'run-manifest.json',m)
        for epoch in EPOCHS:
            if epoch!=15 and not m['epoch15_reproduction_passed']:break
            for action in ['sample','evaluate','condition']:
                worker(action,epoch)
                if action=='sample':m['completed_sampling']+=1
                if action=='evaluate':m['completed_discriminator_evaluations']+=5
                dump(RUN/'run-manifest.json',m)
            m['completed_epochs_evaluated'].append(epoch);dump(RUN/'run-manifest.json',m)
            if epoch==15:
                subprocess.run([sys.executable,str(REPORTER),'--reproduction'],cwd=ROOT,env=os.environ.copy(),check=True)
                gate=json.loads((OUT/'epoch15_reproduction.json').read_text())
                m['epoch15_reproduction_passed']=gate['passed'];m['reproduction_decision_sha256']=sha(OUT/'epoch15_reproduction.json')
                dump(RUN/'run-manifest.json',m)
        check_inputs();m.update(status='computed_pending_summary',computation_completed_at=now());dump(RUN/'run-manifest.json',m)
        subprocess.run([sys.executable,str(REPORTER)],cwd=ROOT,env=os.environ.copy(),check=True)
    except BaseException:
        m.update(status='failed',failed_at=now(),failure=traceback.format_exc());dump(RUN/'run-manifest.json',m)
        with (ROOT/'experiment-ledger.md').open('a') as f:
            f.write(f'\n## `{RUN.name}` failed\n\n- [Observed] Stopped {now()},stage={m.get("current_stage")},epoch={m.get("current_epoch")};see manifest/log. One-training budget not automatically repeated.\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','train','sample','evaluate','condition'])
    parser.add_argument('epoch',nargs='?',type=int,choices=EPOCHS)
    args=parser.parse_args();environment()
    if args.action in ['sample','evaluate','condition']:
        assert args.epoch is not None;globals()[args.action](args.epoch)
    else:globals()[args.action]()
