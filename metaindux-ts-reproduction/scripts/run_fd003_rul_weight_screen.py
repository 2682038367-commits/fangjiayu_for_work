"""Exploratory15-fit FD003 weighting screen; original test is never read."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from types import SimpleNamespace
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/ext-fd003-rul-weight-screen-20261027'
OUT=ROOT/'results/exploratory/fd003_rul_weight_screen_20261027'
WEIGHTS=[1,2,4];GENSEEDS=[3,13,23,33,43]
PROTOCOL=ROOT/'docs/operations/FD003_RUL_WEIGHT_SCREEN.md'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def dump(path,value):path.write_text(json.dumps(value,indent=2))


def configure_environment():
    for k,v in {'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','WANDB_MODE':'disabled',
        'CUBLAS_WORKSPACE_CONFIG':':4096:8','MPLCONFIGDIR':str(ROOT/'.cache/matplotlib')}.items():os.environ.setdefault(k,v)


def train_data():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    sensors=['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']
    columns=['id','cycle','setting1','setting2','setting3']+[f's{i}' for i in range(1,22)]
    frame=pd.read_csv(ROOT/'upstream/data/train_FD003.txt',sep=r'\s+',header=None,names=columns)
    a=MinMaxScaler().fit_transform(frame[sensors]);xx=[];yy=[];ids=[];end=[]
    for engine,g in frame.groupby('id',sort=True):
        mask=(frame.id==engine).to_numpy();v=a[mask];cycles=g.cycle.to_numpy();n=len(g)-47
        xx.append(np.lib.stride_tricks.sliding_window_view(v,48,axis=0).transpose(0,2,1))
        yy.append(np.minimum(cycles.max()-cycles[47:],125));ids.extend([engine]*n);end.extend(cycles[47:])
    return np.concatenate(xx).astype(np.float32),np.concatenate(yy).astype(np.float32),np.array(ids),np.array(end)


def prepare():
    import numpy as np
    import torch
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    mp=RUN/'run-manifest.json'
    if mp.exists():raise RuntimeError('Preserve existing preparation')
    real,y,ids,end=train_data()
    with np.load(ROOT/'results/fd003_w48_public_code_split.npz') as z:
        ti=z['predictive_train'];vi=z['predictive_dev']
    assert len(y)==20020 and len(ti)==14014 and len(vi)==6006 and not np.intersect1d(ti,vi).size
    engines=np.random.RandomState(20261027).permutation(np.unique(ids));validation=np.sort(engines[:30]);remaining=np.sort(engines[30:])
    qi=np.flatnonzero(np.isin(ids,validation));assert len(validation)==30 and not np.intersect1d(validation,remaining).size
    generator=torch.Generator().manual_seed(3)
    orders=np.stack([torch.randperm(len(ti),generator=generator).numpy() for _ in range(35)]).astype(np.int32)
    expected_order=hashlib.sha256(ti[orders].astype('<i4').tobytes()).hexdigest()
    np.savez_compressed(RUN/'frozen_plan.npz',train_indices=ti,dev_indices=vi,orders=orders,
        real_validation_indices=qi,validation_engines=validation,remaining_engines=remaining)
    np.savez_compressed(RUN/'real_validation.npz',data=real[qi],label=y[qi],engine_id=ids[qi],end_cycle=end[qi])
    fraction=float(np.mean((y[ti]>=31)&(y[ti]<=60)))
    config=dict(dataset='FD003',window=48,input_size=14,hidden=64,num_layers=1,output_size=1,
        generation_seeds=GENSEEDS,predictor_seed=3,weights=WEIGHTS,epochs=35,batch_size=256,
        learning_rate=.002,optimizer='Adam',lr_decay_epochs_zero_based=[15,25,35],lr_decay_factor=.1,
        checkpoint='final35epochs',target_rul=[31,60],loss_normalization='fixed full-train mean weight',
        target_fraction=fraction,normalizers={str(w):1+(w-1)*fraction for w in WEIGHTS},
        expected_updates=35*math.ceil(len(ti)/256),expected_sample_visits=35*len(ti),
        expected_consumed_order_sha256=expected_order,real_split_seed=20261027,validation_engines=validation.tolist(),
        remaining_engines=remaining.tolist(),synthetic_train_n=len(ti),synthetic_dev_n=len(vi),real_validation_n=len(qi),
        selection=dict(min_target_improved_generation_seeds=3,max_other_bin_rmse_regression_fraction=.05,
            overall_must_improve=True,target_must_improve=True,objective='pooled engine-balanced realvalidation MSE',tie='lower weight'),
        caveats=['Stage31-60 motivated by prior officialtest analysis,exploratory',
            'Fixed generator and MinMax saw ALL100real engines;validation independence ONLY for downstream selection',
            'Synthetic windows have no reliable physical engineID;no fake engine assignment by index',
            'One predictorinitialization;not initialization robustness',
            'Life-spanning validation windows differ from official test endpoints'])
    dump(RUN/'config.json',config)
    paths=[Path(__file__),PROTOCOL,ROOT/'upstream/models/LSTM.py',ROOT/'upstream/utils.py',
        ROOT/'upstream/data/train_FD003.txt',ROOT/'results/fd003_w48_public_code_split.npz',
        RUN/'config.json',RUN/'frozen_plan.npz',RUN/'real_validation.npz']
    for g in GENSEEDS:
        p=ROOT/f'outputs/FD003_w48_seed{g}.npz'
        with np.load(p) as z:assert z['data'].shape==real.shape and np.array_equal(z['label'].ravel(),y)
        paths.append(p)
    m=dict(run_id=RUN.name,status='prepared',scope_class='extension:EXT-002',role='supporting',
        scope_contract='scope-20261027-fd003-rul-weight-screen',prepared_at=now(),
        user_authorization='Explicit fixed-generator31-60weighting screen;user confirmed downstream-only engine validation with generator exposure disclosed',
        source_snapshot='SOURCE_SNAPSHOT.md',command=[str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__)),'--run'],
        inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},config=config,
        budget=15,completed_models=0,no_official_test_access=True,
        environment_prepare=dict(python=sys.version,numpy=np.__version__,torch=str(torch.__version__),platform=platform.platform()))
    dump(mp,m);print(json.dumps(dict(status='prepared',validation_engines=validation.tolist(),config=config),indent=2))


def model_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def predict(model,x,batch=512):
    import numpy as np
    import torch
    model.eval();parts=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch):parts.append(model(x[start:start+batch]).cpu().numpy().ravel())
    p=np.concatenate(parts).astype(float);assert np.isfinite(p).all()
    return p


def segment_rows(prediction,labels,indices,meta,domain,engine_ids=None):
    import numpy as np
    bins=[('all',0,125),('0-30',0,30),('31-60',31,60),('61-90',61,90),('91-124',91,124),('125',125,125)]
    rows=[];engine_rows=[]
    for name,lo,hi in bins:
        use=indices[(labels[indices]>=lo)&(labels[indices]<=hi)];err=prediction[use]-labels[use]
        assert len(use)>0
        row=dict(**meta,domain=domain,bin=name,n_windows=len(use),mse_window=float(np.mean(err**2)),
            rmse_window=float(np.sqrt(np.mean(err**2))),bias_window=float(err.mean()),n_engines=0,
            mse_engine=float('nan'),rmse_engine=float('nan'))
        if engine_ids is not None:
            means=[]
            for engine in np.unique(engine_ids[use]):
                take=use[engine_ids[use]==engine];e=prediction[take]-labels[take];means.append(float(np.mean(e**2)))
                engine_rows.append(dict(**meta,domain=domain,bin=name,engine_id=int(engine),n_windows=len(take),
                    mse=float(np.mean(e**2)),rmse=float(np.sqrt(np.mean(e**2))),bias=float(e.mean())))
            row.update(n_engines=len(means),mse_engine=float(np.mean(means)),rmse_engine=float(np.sqrt(np.mean(means))))
        rows.append(row)
    return rows,engine_rows


def run():
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,str(ROOT/'upstream'))
    from models.LSTM import LSTM
    from utils import adjust_learning_rate_decay
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable;no CPU fallback training')
    mp=RUN/'run-manifest.json';m=json.loads(mp.read_text());assert m['status']=='prepared'
    for name,digest in m['inputs_sha256'].items():assert sha(ROOT/name)==digest,name
    c=m['config'];m.update(status='running',started_at=now(),gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=torch.version.cuda)
    dump(mp,m)
    with np.load(RUN/'frozen_plan.npz') as z:ti=z['train_indices'];vi=z['dev_indices'];orders=z['orders']
    with np.load(RUN/'real_validation.npz') as z:vx=z['data'];vy=z['label'].astype(float);vid=z['engine_id']
    vx_gpu=torch.from_numpy(vx).to('cuda');all_rows=[];all_engines=[];audits=[];history=[]
    checkpoint_dir=ROOT/'checkpoints/evaluators/fd003_rul_weight_screen_20261027';checkpoint_dir.mkdir(parents=True,exist_ok=True)
    (OUT/'predictions').mkdir(exist_ok=True)
    try:
        for g in GENSEEDS:
            with np.load(ROOT/f'outputs/FD003_w48_seed{g}.npz') as z:
                sx=z['data'];sy=z['label'].ravel().astype(float)
            sx_gpu=torch.from_numpy(sx).to('cuda');sy_gpu=torch.tensor(sy,dtype=torch.float32,device='cuda').reshape(-1,1)
            train_x=sx_gpu[ti];train_y=sy_gpu[ti];dev_x=sx_gpu[vi]
            orders_gpu=torch.from_numpy(orders.astype(np.int64)).to('cuda')
            for weight in WEIGHTS:
                name=f'gen{g}_weight{weight}_init3';print('START',name,now(),flush=True)
                np.random.seed(3);torch.manual_seed(3);torch.cuda.manual_seed_all(3)
                config=SimpleNamespace(input_size=14,window_size=48,hidden=64,num_layers=1,output_size=1)
                model=LSTM(config).to('cuda');initial=model_hash(model)
                optimizer=torch.optim.Adam(model.parameters(),lr=.002)
                consumed=hashlib.sha256();steps=0;visits=0;normalizer=c['normalizers'][str(weight)]
                for epoch in range(35):
                    lr=adjust_learning_rate_decay(optimizer,optimizer,epoch,.002);model.train();epoch_loss=0.
                    for start in range(0,len(ti),256):
                        ix=orders_gpu[epoch,start:start+256];batch_y=train_y[ix]
                        optimizer.zero_grad(set_to_none=True);p=model(train_x[ix])
                        if weight==1:loss=F.mse_loss(p,batch_y)
                        else:
                            w=torch.where((batch_y>=31)&(batch_y<=60),float(weight),1.)
                            loss=(w*(p-batch_y).square()).mean()/normalizer
                        loss.backward();optimizer.step()
                        steps+=1;visits+=len(ix)
                        consumed.update(ti[orders[epoch,start:start+256]].astype('<i4').tobytes())
                        epoch_loss+=float(loss.detach())*len(ix)
                    dp=predict(model,dev_x);dev_mse=float(np.mean((dp-sy[vi])**2))
                    history.append(dict(generation_seed=g,weight=weight,predictor_seed=3,epoch=epoch+1,
                        updates_so_far=steps,learning_rate=lr,weighted_train_loss=epoch_loss/len(ti),synthetic_dev_mse=dev_mse))
                    print('EPOCH',name,epoch+1,'updates',steps,'synthetic_dev_rmse',dev_mse**.5,flush=True)
                assert steps==c['expected_updates'] and visits==c['expected_sample_visits']
                assert consumed.hexdigest()==c['expected_consumed_order_sha256']
                assert abs(np.mean(np.where((sy[ti]>=31)&(sy[ti]<=60),weight,1)/normalizer)-1)<1e-10
                path=checkpoint_dir/f'{name}.pth'
                torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    architecture=vars(config),generation_seed=g,weight=weight,predictor_seed=3,
                    epoch=35,updates=steps,initial_sha256=initial,consumed_order_sha256=consumed.hexdigest()),path)
                sp=predict(model,sx_gpu);vp=predict(model,vx_gpu)
                np.savez_compressed(OUT/'predictions'/f'{name}.npz',synthetic=sp,real_validation=vp)
                meta=dict(generation_seed=g,weight=weight,predictor_seed=3)
                for domain,pp,yy,indices,engines in [
                    ('synthetic_train',sp,sy,ti,None),('synthetic_dev',sp,sy,vi,None),
                    ('real_validation',vp,vy,np.arange(len(vy)),vid)]:
                    rows,perengine=segment_rows(pp,yy,indices,meta,domain,engines)
                    all_rows.extend(rows);all_engines.extend(perengine)
                audits.append(dict(**meta,epochs=35,updates=steps,sample_visits=visits,initial_sha256=initial,
                    consumed_order_sha256=consumed.hexdigest(),normalizer=normalizer,
                    checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path)))
                for file,records in [('segments.csv',all_rows),('validation_engines.csv',all_engines),('training_audit.csv',audits),('history.csv',history)]:
                    pd.DataFrame(records).to_csv(OUT/file,index=False)
                m['completed_models']=len(audits);dump(mp,m)
                final=[r for r in all_rows if r['generation_seed']==g and r['weight']==weight and r['domain']=='real_validation' and r['bin']=='all'][0]
                print('DONE',name,'real_validation_engine_balanced_rmse',final['rmse_engine'],flush=True)
                del optimizer,model;torch.cuda.empty_cache()
            del sx_gpu,sy_gpu,train_x,train_y,dev_x,orders_gpu;torch.cuda.empty_cache()
        assert len(audits)==15 and len({a['initial_sha256'] for a in audits})==1
        assert len({a['consumed_order_sha256'] for a in audits})==1
        m.update(status='complete',finished_at=now(),audit='15final35epoch models;all1925updates/490490visits,identicalinit/order;no officialtest access',
            checkpoint_sha256={a['checkpoint']:a['checkpoint_sha256'] for a in audits},
            outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file()})
    except BaseException:
        m.update(status='failed',error=traceback.format_exc(),finished_at=now());raise
    finally:dump(mp,m)


if __name__=='__main__':
    configure_environment();parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--run',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
