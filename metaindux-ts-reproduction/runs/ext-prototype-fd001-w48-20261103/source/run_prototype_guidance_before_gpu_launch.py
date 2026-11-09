"""Isolated FD001 A/B extension. No official test files are opened."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/extensions/prototype_fd001_w48.json'
sys.path.insert(0, str(ROOT/'upstream'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('WANDB_MODE', 'disabled')
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from prototype_guidance import BINS, fit_prototypes, draw_prototypes, make_model
from GaussianDiffusion import GaussianDiffusion1D_cls_free, GradualWarmupScheduler
from models.LSTM import LSTM
from utils import adjust_learning_rate_decay

C = json.loads(CONFIG.read_text())
RUN = ROOT/'runs'/C['run_id']
OUT = ROOT/'results/extensions'/C['run_id']

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare():
    from data.CMAPSSDataset import columns, feature_columns
    if (RUN/'manifest.json').exists():
        raise RuntimeError('Snapshot exists; use check/run without prepare')
    RUN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT/'upstream/data/train_FD001.txt', sep=r'\s+', header=None, names=columns)
    train_rows = ~frame.id.isin(C['validation_engines'])
    scaler = MinMaxScaler().fit(frame.loc[train_rows, feature_columns])
    scaled = scaler.transform(frame[feature_columns]).astype('float32')
    x, y, ids, cycles = [], [], [], []
    for engine, g in frame.groupby('id', sort=True):
        v = scaled[g.index]
        for end in range(47, len(g)):
            x.append(v[end-47:end+1]); y.append(min(125, len(g)-end-1))
            ids.append(engine); cycles.append(int(g.iloc[end].cycle))
    x, y, ids = np.array(x), np.array(y, dtype='float32')[:,None], np.array(ids)
    valid = np.isin(ids, C['validation_engines'])
    assert valid.sum() == 4875 and not set(ids[~valid]) & set(ids[valid])
    np.savez_compressed(RUN/'train.npz', data=x[~valid], label=y[~valid], engine_id=ids[~valid], end_cycle=np.array(cycles)[~valid])
    np.savez_compressed(RUN/'validation.npz', data=x[valid], label=y[valid], engine_id=ids[valid])
    np.savez(RUN/'normalization.npz', data_min=scaler.data_min_, data_max=scaler.data_max_, scale=scaler.scale_, offset=scaler.min_, fit_engine_ids=np.unique(ids[~valid]))
    stats = fit_prototypes(x[~valid], y[~valid], C['cluster_seed'])
    np.savez_compressed(RUN/'prototypes.npz', **stats, fit_engine_ids=np.unique(ids[~valid]))
    permutation = np.random.default_rng(C['synthetic_split_seed']).permutation((~valid).sum())
    n = int(len(permutation)*C['synthetic_train_fraction'])
    np.savez(RUN/'plan.npz', target_rul=y[~valid], train_idx=permutation[:n], dev_idx=permutation[n:])
    dump(RUN/'config.json', C)
    paths = [CONFIG, Path(__file__), ROOT/'upstream/prototype_guidance.py', *list((ROOT/'upstream/DiffusionFreeGuidence').glob('*.py')), ROOT/'upstream/GaussianDiffusion.py', ROOT/'upstream/models/LSTM.py', ROOT/'upstream/utils.py', ROOT/'upstream/data/CMAPSSDataset.py', ROOT/'scripts/run_fd001_checkpoint_downstream_validation.py', ROOT/'upstream/data/train_FD001.txt', *list(RUN.glob('*.npz'))]
    source = RUN/'source'; source.mkdir()
    import shutil
    for p in paths:
        if p.suffix == '.py':
            shutil.copy2(p, source/p.name)
    (RUN/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
    (RUN/'workspace.diff').write_bytes(subprocess.check_output(['git','diff','--binary'], cwd=ROOT))
    (RUN/'git-status.txt').write_bytes(subprocess.check_output(['git','status','--short'], cwd=ROOT))
    dump(RUN/'manifest.json', dict(scope='extension:EXT-PROTO-001', authorization='Explicit user FD001 prototype A/B request 2026-11-03', config=C,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}, torch=torch.__version__, cuda_available=torch.cuda.is_available(),
        platform=__import__('platform').platform(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        train_windows=int((~valid).sum()), validation_windows=int(valid.sum()),
        empty_bins=[BINS[i][0] for i,v in enumerate(stats['empty_bins']) if v],
        command=f'{sys.executable} -u scripts/run_prototype_guidance.py run', status='prepared'))
    print('Prepared', RUN, flush=True)

def verify():
    m = json.loads((RUN/'manifest.json').read_text())
    for p, h in m['hashes'].items():
        assert digest(ROOT/p) == h, p

def diffusion(model, device, steps=None):
    return GaussianDiffusion1D_cls_free(model, seq_length=48, channels=14,
        timesteps=steps or C['timesteps'], objective='pred_noise', beta_schedule=C['schedule']).to(device)

def check():
    verify(); seed(23)
    train = np.load(RUN/'train.npz'); val = np.load(RUN/'validation.npz'); stats = np.load(RUN/'prototypes.npz')
    assert not set(train['engine_id']) & set(val['engine_id'])
    assert np.array_equal(np.unique(train['engine_id']), stats['fit_engine_ids'])
    from DiffusionFreeGuidence.Unet1D_fre import UNet1D_fre
    seed(23); original = UNet1D_fre(dim=32, dim_mults=(1,2), cond_drop_prob=.2, channels=14,length=48)
    seed(23); a = make_model(False)
    assert type(a) is type(original)
    seed(23); b = make_model(True)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    x = torch.from_numpy(train['data'][:4]).permute(0,2,1); y = torch.from_numpy(train['label'][:4])
    t = torch.tensor([1,2,3,4]); b.prototype_ids = torch.from_numpy(stats['prototype_id'][:4])
    seed(42); o = original(x,t,y); seed(42); oa = a(x,t,y)
    assert torch.equal(o,oa)
    loss = diffusion(b,'cpu')(x,classes=y); loss.backward()
    gradients = {n:float(p.grad.abs().sum()) for n,p in b.named_parameters() if n.startswith('prototype_')}
    assert all(v > 0 for v in gradients.values())
    # Null branch is independent of both RUL and prototype.
    seed(4); n1=b(x,t,y,cond_drop_prob=1.)
    b.prototype_ids=(b.prototype_ids+1)%4
    seed(4); n2=b(x,t,y+5,cond_drop_prob=1.)
    assert torch.equal(n1,n2)
    rng_state=torch.get_rng_state().clone()
    p=draw_prototypes(y,stats,np.random.default_rng(C['prototype_sampling_seed']))
    assert torch.equal(rng_state,torch.get_rng_state())
    # Sampling API receives only RUL and train-derived statistics; no reference window.
    samples=[]; sampling_rng=[]; branch_calls=[]
    for enabled in [False, True]:
        seed(23); model=make_model(enabled, sampling=True).eval()
        if enabled: model.prototype_ids=torch.from_numpy(p)
        calls=[]
        forward=model.forward
        def observed_forward(*args, **kw):
            calls.append(kw.get('cond_drop_prob'))
            return forward(*args, **kw)
        model.forward=observed_forward
        seed(103)
        with torch.no_grad(): samples.append(diffusion(model,'cpu',100).sample(classes=y[:4]).numpy())
        sampling_rng.append(torch.get_rng_state().clone())
        branch_calls.append(calls)
    assert torch.equal(sampling_rng[0],sampling_rng[1])
    assert all(calls == [0.,1.]*100 for calls in branch_calls)
    empty = fit_prototypes(train['data'][:32], np.full((32,1),125), C['cluster_seed'])
    assert empty['empty_bins'][:4].all()
    assert np.allclose(empty['frequency'][:4],empty['global_frequency'])
    states=[]; trajectories=[]
    for enabled in [False,True]:
        seed(13); m=make_model(enabled); opt=torch.optim.Adam(m.parameters(),lr=.002)
        d=diffusion(m,'cpu'); seed(1013); trace=[]
        for _ in range(2):
            if enabled: m.prototype_ids=torch.from_numpy(stats['prototype_id'][:4])
            opt.zero_grad(); l=d(x,classes=y); l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); trace.append(float(l))
        states.append(torch.get_rng_state().clone()); trajectories.append(trace)
    assert torch.equal(*states)
    dump(RUN/'checks.json',dict(passed=True, base_exact_path_and_output=True, shared_initialization_equal=True,
         gradient_l1=gradients, null_joint_condition=True, disjoint_engines=True,
         prototype_rng_independent=True, sampling_dropout_calls_checked=True, empty_bin_fallback_checked=True, paired_two_update_losses=trajectories, sampling_shapes=[list(v.shape) for v in samples],
         note='CPU small-batch check; not full training or a downstream result'))
    print('CHECKS PASSED', flush=True)

def train_generator(s, group, device):
    dst=OUT/f'seed{s}'/group; dst.mkdir(parents=True,exist_ok=True)
    if (dst/'training.json').exists(): return
    if list(dst.glob('epoch*.pth')): raise RuntimeError(f'Partial training exists: {dst}; preserve and audit before retry')
    z=np.load(RUN/'train.npz'); stats=np.load(RUN/'prototypes.npz')
    seed(s); model=make_model(group=='B').to(device); d=diffusion(model,device)
    shared_initial=hashlib.sha256()
    for name,value in model.state_dict().items():
        if not name.startswith('prototype_'):
            shared_initial.update(name.encode()); shared_initial.update(value.detach().cpu().numpy().tobytes())
    optimizer=torch.optim.Adam(model.parameters(),lr=C['lr'])
    cosine=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=70,eta_min=0)
    scheduler=GradualWarmupScheduler(optimizer,multiplier=2.5,warm_epoch=8,after_scheduler=cosine)
    x=torch.from_numpy(z['data']).permute(0,2,1).to(device); y=torch.from_numpy(z['label']).to(device)
    p=torch.from_numpy(stats['prototype_id']).to(device)
    # Dedicated loader stream: same sample order independently of model RNG usage.
    loader=torch.utils.data.DataLoader(torch.arange(len(x)),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(s))
    history=[]; order_hash=hashlib.sha256(); noise_hash=hashlib.sha256()
    # Diffusion timestep/noise and dropout use the reset global stream, equal architecture draw counts.
    seed(s+10000)
    for epoch in range(1,71):
        losses=[]
        for idx in loader:
            order_hash.update(idx.numpy().tobytes()); idx=idx.to(device)
            yy=y[idx]
            if np.random.rand()<C['batch_label_zero_prob']: yy=torch.zeros_like(yy)
            if group=='B': model.prototype_ids=p[idx]
            noise_hash.update(torch.get_rng_state().numpy().tobytes())
            if device=='cuda': noise_hash.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
            optimizer.zero_grad(); loss=d(x[idx],classes=yy); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); losses.append(float(loss))
        scheduler.step(); history.append(dict(epoch=epoch,loss=float(np.mean(losses)),lr=optimizer.param_groups[0]['lr']))
        dump(dst/'progress.json',dict(completed_epochs=epoch,history=history))
        if epoch in C['checkpoints']: torch.save(model.state_dict(),dst/f'epoch{epoch}.pth')
        print(s,group,epoch,history[-1],flush=True)
    dump(dst/'training.json',dict(shared_initial_hash=shared_initial.hexdigest(),history=history,updates=70*len(loader),order_hash=order_hash.hexdigest(),rng_hash=noise_hash.hexdigest()))

def sample_generator(s,group,epoch,device):
    dst=OUT/f'seed{s}'/group; path=dst/f'samples{epoch}.npz'
    if path.exists(): return
    plan=np.load(RUN/'plan.npz'); labels=torch.from_numpy(plan['target_rul']); stats=np.load(RUN/'prototypes.npz')
    seed(s); model=make_model(group=='B',sampling=True).to(device).eval()
    model.load_state_dict(torch.load(dst/f'epoch{epoch}.pth',map_location=device,weights_only=True))
    d=diffusion(model,device)
    proto=draw_prototypes(labels.numpy(),stats,np.random.default_rng(C['prototype_sampling_seed']))
    seed(C['sampling_seed']); batches=[]; rng_hash=hashlib.sha256()
    with torch.no_grad():
        for start in range(0,len(labels),256):
            if group=='B': model.prototype_ids=torch.from_numpy(proto[start:start+256]).to(device)
            rng_hash.update(torch.get_rng_state().numpy().tobytes())
            if device=='cuda': rng_hash.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
            batches.append(d.sample(classes=labels[start:start+256].to(device),cond_scale=C['cfg']).permute(0,2,1).cpu().numpy())
    dump(dst/f'sampling{epoch}.json',dict(rng_hash=rng_hash.hexdigest(),sampling_seed=C['sampling_seed'],prototype_seed=C['prototype_sampling_seed'],n=len(labels),steps=1000,cfg=3.,condition_dropout=0))
    np.savez_compressed(path,data=np.concatenate(batches),label=labels.numpy(),prototype_id=proto if group=='B' else np.full(len(labels),-1))

def downstream(s,group,epoch,es,device):
    dst=OUT/f'seed{s}'/group/f'epoch{epoch}_eval{es}'; dst.mkdir(parents=True,exist_ok=True)
    if (dst/'metrics.json').exists(): return
    z=np.load(dst.parent/f'samples{epoch}.npz'); plan=np.load(RUN/'plan.npz'); val=np.load(RUN/'validation.npz')
    seed(es); args=SimpleNamespace(input_size=14,output_size=1,num_layers=1,window_size=48,hidden=64)
    model=LSTM(args).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    x=torch.from_numpy(z['data']).to(device); y=torch.from_numpy(z['label']).to(device)
    loader=torch.utils.data.DataLoader(torch.from_numpy(plan['train_idx']),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(es))
    def predict(data):
        model.eval()
        with torch.no_grad(): return torch.cat([model(v) for v in data.split(2048)]).cpu().numpy().reshape(-1)
    best=float('inf'); updates=0; history=[]; order_hash=hashlib.sha256()
    # Fixed35 epochs: retain synthetic-dev selection after epoch6, disable patience only for equal budgets.
    for e in range(35):
        adjust_learning_rate_decay(optimizer,optimizer,e,.002)
        for count,idx in enumerate(loader,1):
            order_hash.update(idx.numpy().tobytes()); idx=idx.to(device)
            model.train(); optimizer.zero_grad(); loss=torch.nn.functional.mse_loss(model(x[idx]),y[idx]); loss.backward(); optimizer.step(); updates+=1
            if count % max(1,len(loader)//5)==0:
                dev=plan['dev_idx']; pred=predict(x[dev]); score=float(np.sqrt(np.mean((pred-z['label'][dev].reshape(-1))**2)))
                if e>5 and score<best:
                    best=score; torch.save(model.state_dict(),dst/'predictor.pth'); selected=(e+1,updates)
        history.append(dict(epoch=e+1,updates=updates))
    model.load_state_dict(torch.load(dst/'predictor.pth',map_location=device,weights_only=True))
    pred=predict(torch.from_numpy(val['data']).to(device)); yy=val['label'].reshape(-1); ids=val['engine_id']
    np.savez_compressed(dst/'predictions.npz',prediction=pred,label=yy,engine_id=ids)
    # Reuse the existing engine-balanced metric implementation, including all RUL bins.
    spec=importlib.util.spec_from_file_location('existing_metrics',ROOT/'scripts/run_fd001_checkpoint_downstream_validation.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rows, engines=module.metrics(pred,yy,ids,{},'real_validation')
    pd.DataFrame(rows).to_csv(dst/'segments.csv',index=False); pd.DataFrame(engines).to_csv(dst/'per_engine.csv',index=False)
    overall=next(r for r in rows if r['bin']=='all')
    dump(dst/'metrics.json',dict(rmse=overall['rmse_engine'],updates=updates,selected_epoch=selected[0],selected_update=selected[1],best_synthetic_dev=best,order_hash=order_hash.hexdigest(),history=history))

def summarize():
    paired=[]; summary=[]
    for s in C['generator_seeds']:
        for e in C['checkpoints']:
            a=[]; b=[]
            for es in C['predictor_seeds']:
                paths=[OUT/f'seed{s}'/g/f'epoch{e}_eval{es}/metrics.json' for g in ['A','B']]
                if not all(p.exists() for p in paths): continue
                aa,bb=[json.loads(p.read_text()) for p in paths]
                assert aa['updates']==bb['updates'] and aa['order_hash']==bb['order_hash']
                a.append(aa['rmse']); b.append(bb['rmse'])
                paired.append(dict(generator_seed=s,epoch=e,predictor_seed=es,A=aa['rmse'],B=bb['rmse'],delta=bb['rmse']-aa['rmse']))
            if len(a)==5:
                av=float(np.sqrt(np.mean(np.square(a)))); bv=float(np.sqrt(np.mean(np.square(b))))
                summary.append(dict(generator_seed=s,epoch=e,A=av,B=bv,delta=bv-av,improving=int(np.sum(np.array(b)<a))))
    if paired: pd.DataFrame(paired).to_csv(OUT/'paired.csv',index=False)
    if summary:
        pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
        choices=[]
        for s in C['generator_seeds']:
            for g in ['A','B']:
                candidates=[r for r in summary if r['generator_seed']==s]
                if len(candidates)==2:
                    chosen=min(candidates,key=lambda r:(r[g],r['epoch']))
                    choices.append(dict(seed=s,group=g,epoch=chosen['epoch'],rmse=chosen[g],exploratory=True))
        dump(OUT/'selected.json',choices)

def run():
    verify()
    import platform
    gpu_probe=subprocess.run(['nvidia-smi'],capture_output=True,text=True) if __import__('shutil').which('nvidia-smi') else SimpleNamespace(stdout='',stderr='nvidia-smi unavailable')
    dump(RUN/'execution_environment.json',dict(platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,nvidia_smi=gpu_probe.stdout+gpu_probe.stderr,cudnn_version=torch.backends.cudnn.version(),cudnn_benchmark=False,cudnn_deterministic=True,global_strict=False,cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,command=sys.argv))
    assert json.loads((RUN/'checks.json').read_text())['passed']
    if not torch.cuda.is_available():
        dump(RUN/'execution_status.json',dict(status='blocked_no_cuda',training_completed=0,downstream_results_available=False))
        raise RuntimeError('CUDA unavailable; full experiment not started. Use prepare/check for CPU validation.')
    for s in C['generator_seeds']:
        for g in ['A','B']: train_generator(s,g,'cuda')
        audits=[json.loads((OUT/f'seed{s}'/g/'training.json').read_text()) for g in ['A','B']]
        assert audits[0]['shared_initial_hash']==audits[1]['shared_initial_hash']
        assert audits[0]['order_hash']==audits[1]['order_hash'] and audits[0]['rng_hash']==audits[1]['rng_hash']
        for e in C['checkpoints']:
            for g in ['A','B']:
                sample_generator(s,g,e,'cuda')
                for es in C['predictor_seeds']: downstream(s,g,e,es,'cuda')
            audits=[json.loads((OUT/f'seed{s}'/g/f'sampling{e}.json').read_text()) for g in ['A','B']]
            assert audits[0]['rng_hash']==audits[1]['rng_hash']
            summarize()
    dump(RUN/'execution_status.json',dict(status='complete',training_completed=4,predictor_fits=40))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','check','run','summarize'])
    action=parser.parse_args().action
    globals()[action]()
