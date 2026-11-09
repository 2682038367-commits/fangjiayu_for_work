"""Trace only FD002 seed43 epoch1 at two current entry points."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pickle
import platform
import runpy
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/diagnostic-fd002-seed43-first-epoch-trace-20261103'
OUT=ROOT/'results/diagnostics/fd002_seed43_first_epoch_trace_20261103'
PREV=ROOT/'runs/diagnostic-fd002-final70-two-seeds-20261102'
PROTOCOL=ROOT/'docs/operations/FD002_SEED43_FIRST_EPOCH_TRACE.md'
REPORT=ROOT/'docs/conclusions/final/fd002_seed43_first_epoch_trace_20261103_report.md'
ARMS=['native_public','native_worker','strict_public','strict_worker']

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);return v

def check():
 m=json.loads((RUN/'run-manifest.json').read_text())
 for p,h in m['inputs_sha256'].items():assert sha(ROOT/p)==h,p
 return m

def state(stage,body):
 body+=f'\n- Diagnostic/supporting `'+RUN.name+f'`; snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`; protocol `{PROTOCOL.relative_to(ROOT)}`. Strict main/A/B unchanged,C deferred; FD003/4 and125 work excluded; generator/downstream originals frozen.\n'
 p=ROOT/'reproduction-state.md';h,t=p.read_text().split('\n',1);p.write_text(h+f'\n\n## Latest checkpoint: 2026-11-03 FD002 first-epoch trace {stage}\n\n'+body+'\n'+t)
 with (ROOT/'experiment-ledger.md').open('a') as f:f.write('\n## `'+RUN.name+'` '+stage+'\n\n'+body)

def prepare():
 assert not (RUN/'run-manifest.json').exists(),'No overwrite'
 import numpy as np
 previous=json.loads((PREV/'run-manifest.json').read_text())
 assert previous['status']=='complete_with_gate_failure'
 for p,h in previous['inputs_sha256'].items():assert sha(ROOT/p)==h,p
 for arm in ARMS:
  for d in ['tapes','states','entry_logs','worker/seed43','checkpoints/seed43']:(OUT/arm/d).mkdir(parents=True,exist_ok=True)
 cfg=dict(seed=43,first_epoch_only=True,updates_per_trajectory=163,maximum_trajectories=4,maximum_updates=652,
  initial_public_arguments_keep_epoch70=True,primary_pair=['native_public','native_worker'],
  conditional_pair=['strict_public','strict_worker'],
  conditional_trigger='All actual initial states and163 complete random questions match, but any output/loss/gradient/Adam/weight differs',
  environment=dict(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TF_DETERMINISTIC_OPS='1',TF_ENABLE_ONEDNN_OPTS='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='offline'),
  stop='Before second epoch, sampling/evaluation forbidden; stop unsupported strict deterministic op; no retries',
  source_limitation='Both entries use current available source; historical Sep14 exact code/RNG/algorithm state unavailable',
  metric='Exact per-batch producer and numerical-state equality; no paper metric target')
 dump(RUN/'config.json',cfg)
 paths=[Path(__file__),PROTOCOL,PREV/'run-manifest.json',PREV/'frozen_inputs.npz',
  ROOT/'results/runs/FD002_w48_seed43_metadata.json',ROOT/'logs/FD002_w48_seed43_training_loss.csv',
  ROOT/'results/diagnostics/fd002_final70_two_seeds_20261102/seed43/training_loss.csv',
  ROOT/'scripts/run_fd002_final70_two_seeds.py',ROOT/'upstream/data/train_FD002.txt',RUN/'config.json',ROOT/'SOURCE_SNAPSHOT.md']
 paths+=list((ROOT/'upstream').glob('*.py'))+list((ROOT/'upstream/DiffusionFreeGuidence').glob('*.py'))
 paths+=[ROOT/'upstream/data/CMAPSSDataset.py',ROOT/'upstream/measure_score/Utils/discriminative_metric.py']
 for p in [Path(__file__),PROTOCOL]:
  q=RUN/'source'/p.name;shutil.copy2(p,q);paths.append(q)
 for name,cmd in [('environment-freeze.txt',[sys.executable,'-m','pip','freeze']),('git-status.txt',['git','status','--short']),('workspace.diff',['git','diff','--binary'])]:
  p=RUN/name;p.write_bytes(subprocess.check_output(cmd,cwd=ROOT));paths.append(p)
 dump(RUN/'run-manifest.json',dict(run_id=RUN.name,status='prepared',scope_class='diagnostic',role='supporting',scope_contract='scope-20261103-fd002-seed43-first-epoch-trace',
  config=cfg,prepared_at=now(),command=[sys.executable,'-u',str(Path(__file__)),'run'],user_authorization='只追踪第一轮',
  git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=sys.version,platform=platform.platform(),
  inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(paths)},completed_arms=[]))
 state('prepared','- [Observed] Freeze two current entry first-epoch traces, same environment and original70-epoch schedule. If same questions but numerical divergence, two strict fixed-question first-epoch repeats allowed. Maximum4 independent epochs/652 updates, never epoch2; no generation/evaluation. Metrics pending; historical exact state unavailable.')
 print(json.dumps(cfg,indent=2),flush=True)

class FirstEpochComplete(Exception):pass

def worker(arm):
 import numpy as np
 import torch
 check();assert torch.cuda.is_available()
 target=OUT/arm;strict=arm.startswith('strict_');public=arm.endswith('_public')
 sys.path.insert(0,str(ROOT/'upstream'))
 metadata=json.loads((ROOT/'results/runs/FD002_w48_seed43_metadata.json').read_text())
 cli=metadata['command'][2:]
 for key,value in [('model_path',target/'unused_best.pth'),('syndata_path',target/'forbidden_samples.npz'),('loss_history_path',target/'entry_loss_history.csv'),('metrics_path',target/'forbidden_metrics.json')]:cli[cli.index('--'+key)+1]=str(value)
 sys.argv=[str(ROOT/'upstream/MainCondition.py'),*cli]
 import TrainCondition as training
 import GaussianDiffusion as diffusion
 import DiffusionFreeGuidence.Unet1D_fre as unet
 import utils
 import wandb
 with np.load(PREV/'frozen_inputs.npz') as z:x,y=z['real'].copy(),z['label'].copy()
 ctx=dict(active=False,updates=0,rows=[],order=[],tqdm_calls=0,model=None,current=None,initial=None)
 raw_step=torch.optim.Adam.step;raw_save=torch.save;raw_clip=torch.nn.utils.clip_grad_norm_
 raw_rand=np.random.rand;raw_int=torch.randint;raw_noise=torch.randn_like;raw_mask=unet.prob_mask_like
 raw_iter=torch.utils.data.RandomSampler.__iter__;raw_factory=training.UNet1D_fre;raw_loss=diffusion.GaussianDiffusion1D_cls_free.p_losses
 raw_seed=utils.torch_seed;raw_tqdm=training.tqdm
 def array_sha(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
 def cloned(state):return {k:v.detach().cpu().contiguous().clone() for k,v in state.items()}
 def digest(state):
  h=hashlib.sha256()
  for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 def gradients(model):return {k:p.grad.detach().cpu().contiguous().clone() for k,p in model.named_parameters() if p.grad is not None}
 def rng():return dict(cpu=array_sha(torch.get_rng_state().numpy()),cuda=array_sha(torch.cuda.get_rng_state().numpy()),numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest())
 def save(obj,p):
  before=rng();raw_save(obj,p);assert before==rng(),'Trace serialization consumed RNG'
 def seed(value):
  raw_seed(value)
  if strict:
   torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
  ctx['seed_rng']=rng()
 utils.torch_seed=seed
 def factory(*a,**kw):
  ctx['pre_constructor_rng']=rng();model=raw_factory(*a,**kw);ctx['model']=model
  initial=cloned(model.state_dict());ctx['initial']=digest(initial);save(initial,target/'initial_state.pth')
  ctx['post_constructor_rng']=rng()
  def pre(mod,inputs,kwargs):
   if ctx['active']:
    xx,tt,cc=inputs
    ctx['current']['noisy_input_sha256']=array_sha(xx.detach().cpu().numpy())
    ctx['current']['actual_condition_sha256']=array_sha(cc.detach().cpu().numpy())
  def post(mod,inputs,output):
   if ctx['active']:ctx['current']['model_output_sha256']=array_sha(output.detach().cpu().numpy())
  model.register_forward_pre_hook(pre,with_kwargs=True);model.register_forward_hook(post)
  return model
 training.UNet1D_fre=factory
 def iterator(sampler):
  for i in raw_iter(sampler):ctx['order'].append(i);yield i
 torch.utils.data.RandomSampler.__iter__=iterator
 class Progress:
  def __init__(self,inner):self.inner=inner
  def __enter__(self):self.inner.__enter__();return self
  def __exit__(self,*a):return self.inner.__exit__(*a)
  def set_postfix(self,*a,**kw):return self.inner.set_postfix(*a,**kw)
  def __iter__(self):
   for images,labels in self.inner:
    assert not ctx['active'];i=ctx['updates']+1;assert i<=163
    b=len(images);indices=np.asarray(ctx['order'][-b:],dtype=np.int64)
    assert np.array_equal(images.detach().cpu().numpy(),x[indices].transpose(0,2,1))
    assert np.array_equal(labels.detach().cpu().numpy(),y[indices])
    ctx['current']=dict(update=i,batch_size=b,indices_sha256=array_sha(indices),data_sha256=array_sha(images.detach().cpu().numpy()),labels_sha256=array_sha(labels.detach().cpu().numpy()),rng_before=rng())
    ctx['tape']=dict(indices=indices);ctx['active']=True
    if strict:
     ctx['reference']=torch.load(OUT/f'native_public/tapes/{i:04d}.pth',map_location='cpu',weights_only=False)
     assert np.array_equal(indices,ctx['reference']['indices'])
    yield images,labels
    assert not ctx['active'] and ctx['updates']==i
 def progress(iterable,**kw):
  ctx['tqdm_calls']+=1
  if ctx['tqdm_calls']>1:raise FirstEpochComplete()
  return Progress(raw_tqdm(iterable,**kw))
 training.tqdm=progress
 def numerical_zero(*a):
  v=raw_rand(*a)
  if not a and ctx['active']:
   assert 'numpy_draw' not in ctx['tape'];ctx['tape']['numpy_draw']=float(v)
   if strict:assert float(v)==ctx['reference']['numpy_draw']
   ctx['current'].update(numpy_draw=float(v),numeric_zeroed=bool(v<.1))
  return v
 np.random.rand=numerical_zero
 def randint(*a,**kw):
  v=raw_int(*a,**kw)
  if ctx['active']:
   assert 'timestep' not in ctx['tape'];ctx['tape']['timestep']=v.detach().cpu().clone()
   if strict:
    assert torch.equal(ctx['tape']['timestep'],ctx['reference']['timestep']);v=ctx['reference']['timestep'].to(v.device)
   ctx['current']['timestep_sha256']=array_sha(ctx['tape']['timestep'].numpy())
  return v
 torch.randint=randint
 def noise(*a,**kw):
  v=raw_noise(*a,**kw)
  if ctx['active']:
   assert 'noise' not in ctx['tape'];ctx['tape']['noise']=v.detach().cpu().clone()
   if strict:
    assert torch.equal(ctx['tape']['noise'],ctx['reference']['noise']);v=ctx['reference']['noise'].to(v.device)
   ctx['current']['noise_sha256']=array_sha(ctx['tape']['noise'].numpy())
  return v
 torch.randn_like=noise
 def mask(shape,prob,device):
  v=raw_mask(shape,prob,device)
  if ctx['active']:
   assert abs(prob-.8)<1e-12 and 'condition_mask' not in ctx['tape']
   ctx['tape']['condition_mask']=v.detach().cpu().clone()
   if strict:
    assert torch.equal(ctx['tape']['condition_mask'],ctx['reference']['condition_mask']);v=ctx['reference']['condition_mask'].to(v.device)
   ctx['current'].update(condition_mask_sha256=array_sha(ctx['tape']['condition_mask'].numpy()),condition_kept=int(v.sum().item()))
  return v
 unet.prob_mask_like=mask
 def loss(trainer,*a,**kw):
  value=raw_loss(trainer,*a,**kw)
  if ctx['active']:ctx['current']['loss']=float(value.item())
  return value
 diffusion.GaussianDiffusion1D_cls_free.p_losses=loss
 def clip(parameters,*a,**kw):
  params=list(parameters)
  if ctx['active']:ctx['grad_pre']=gradients(ctx['model']);ctx['current']['grad_preclip_sha256']=digest(ctx['grad_pre'])
  norm=raw_clip(params,*a,**kw)
  if ctx['active']:ctx['current']['grad_norm_preclip']=float(norm.item())
  return norm
 torch.nn.utils.clip_grad_norm_=clip
 def step(optimizer,*a,**kw):
  assert ctx['active'];row=ctx['current'];row['weights_before_sha256']=digest(ctx['model'].state_dict())
  post_grad=gradients(ctx['model']);row['grad_postclip_sha256']=digest(post_grad)
  row['learning_rate']=float(optimizer.param_groups[0]['lr'])
  value=raw_step(optimizer,*a,**kw)
  ctx['updates']+=1;i=ctx['updates'];after=cloned(ctx['model'].state_dict());row['weights_after_sha256']=digest(after)
  adam={f'{n}.{k}':v for n,s in enumerate(optimizer.state.values()) for k,v in s.items() if torch.is_tensor(v)}
  row['adam_state_sha256']=digest(adam)
  assert set(ctx['tape'])=={'indices','numpy_draw','timestep','noise','condition_mask'}
  save(ctx['tape'],target/f'tapes/{i:04d}.pth')
  save(dict(model_after=after,grad_preclip=ctx['grad_pre'],grad_postclip=post_grad),target/f'states/{i:04d}.pth')
  row['rng_after']=rng();ctx['rows'].append(row);ctx['active']=False
  dump(target/'progress.json',dict(completed_updates=i,total_updates=163,epoch=1,updated_at=now()))
  return value
 torch.optim.Adam.step=step
 training.sample=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('Sampling forbidden'))
 started=time.time()
 try:
  if public:
   from data.CMAPSSDataset import CMAPSSDataset
   import data.CMAPSSDataset as data_module
   import pandas as pd
   from sklearn.preprocessing import MinMaxScaler
   class TrainOnly(CMAPSSDataset):
    def __init__(self,fd_number,sequence_length,deleted_engine):
     assert fd_number=='FD002' and sequence_length==48
     self.sequence_length=sequence_length;self.deleted_engine=deleted_engine;self.engine_size=260
     self.feature_columns=data_module.feature_columns
     f=pd.read_csv(ROOT/'upstream/data/train_FD002.txt',sep=r'\s+',header=None,names=data_module.columns)
     f['RUL']=f.groupby('id').cycle.transform('max')-f.cycle;f['cycle_norm']=f.cycle
     cols=f.columns.difference(['id','cycle','RUL']);self.std=MinMaxScaler()
     normalized=pd.DataFrame(self.std.fit_transform(f[cols]),columns=cols,index=f.index)
     self.train_data=f[f.columns.difference(cols)].join(normalized).reindex(columns=f.columns)
    def get_test_data(self):return None
    def get_last_data_slice(self,data):return torch.empty(0,48,14),torch.empty(0,1)
   data_module.CMAPSSDataset=TrainOnly
   runpy.run_path(str(ROOT/'upstream/MainCondition.py'),run_name='__main__')
  else:
   old=module(ROOT/'scripts/run_fd002_final70_two_seeds.py','original_audited_worker')
   old.OUT=target/'worker';old.WEIGHTS=target/'checkpoints'
   old.train(43)
 except FirstEpochComplete:
  pass
 finally:
  wandb.finish()
 assert ctx['updates']==163 and ctx['tqdm_calls']==2 and len(ctx['order'])==41539
 dump(target/'trace.json',ctx['rows'])
 np.save(target/'order.npy',np.asarray(ctx['order'],dtype=np.int32))
 history=target/'entry_loss_history.csv' if public else target/'worker/seed43/training_loss.csv'
 h=np.loadtxt(history,delimiter=',',skiprows=1,ndmin=2);assert h.shape==(1,3)
 epochloss=float(np.mean([v['loss'] for v in ctx['rows']]))
 assert abs(epochloss-h[0,1])<1e-14
 result=dict(status='complete',arm=arm,completed_epoch=1,actual_updates=163,started_from_seed=43,
  elapsed_seconds=time.time()-started,mean_first_epoch_loss=epochloss,next_lr=float(h[0,2]),initial_state_sha256=ctx['initial'],
  pre_constructor_rng=ctx['pre_constructor_rng'],post_constructor_rng=ctx['post_constructor_rng'],seed_rng=ctx['seed_rng'],
  backend_flags=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
   deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
  environment={k:os.environ.get(k) for k in check()['config']['environment']},torch=str(torch.__version__),cuda=torch.version.cuda,
  gpu=torch.cuda.get_device_name(0),initial_state_not_historical=True,sample_or_evaluator_calls=0,serializations_preserve_rng=True)
 dump(target/'summary.json',result);print(json.dumps(result,indent=2),flush=True)


def compare(a,b,name):
 import numpy as np
 import torch
 sa=json.loads((OUT/a/'summary.json').read_text());sb=json.loads((OUT/b/'summary.json').read_text())
 aa=json.loads((OUT/a/'trace.json').read_text());bb=json.loads((OUT/b/'trace.json').read_text())
 assert len(aa)==len(bb)==163
 producers=['indices_sha256','data_sha256','labels_sha256','numpy_draw','timestep_sha256','noise_sha256','condition_mask_sha256','actual_condition_sha256']
 numerical=['noisy_input_sha256','model_output_sha256','loss','grad_preclip_sha256','grad_norm_preclip','grad_postclip_sha256','adam_state_sha256','weights_after_sha256']
 def differences(keys):return {k:next((i+1 for i,(x,y) in enumerate(zip(aa,bb)) if x[k]!=y[k]),None) for k in keys}
 pdif,ndif=differences(producers),differences(numerical)
 initial_equal=sa['initial_state_sha256']==sb['initial_state_sha256']
 result=dict(pair=[a,b],initial_states_equal=initial_equal,first_question_differences=pdif,
  all163_questions_equal=all(v is None for v in pdif.values()),first_numerical_differences=ndif,
  all163_numerics_equal=all(v is None for v in ndif.values()),
  first_epoch_losses=[sa['mean_first_epoch_loss'],sb['mean_first_epoch_loss']],first_divergence=None)
 different=[i for i in ndif.values() if i is not None]
 if different:
  first=min(different);result['first_divergence']=first
  states=[torch.load(OUT/s/f'states/{first:04d}.pth',map_location='cpu',weights_only=True) for s in [a,b]]
  detail={}
  for group in ['model_after','grad_preclip','grad_postclip']:
   x,y=(s[group] for s in states);assert x.keys()==y.keys()
   num=den=0.;maxdiff=0.;changed=[]
   for k in x:
    xx=x[k].numpy().astype(np.float64);yy=y[k].numpy().astype(np.float64);d=xx-yy
    num+=float(np.sum(d*d));den+=float(np.sum(xx*xx));md=float(np.max(abs(d)))
    maxdiff=max(maxdiff,md)
    if md:changed.append(k)
   detail[group]=dict(max_absolute_difference=maxdiff,relative_l2=float(np.sqrt(num/max(den,1e-300))),different_tensor_count=len(changed),first_differing_names=changed[:12])
  result['first_difference_magnitude']=detail
 dump(OUT/f'{name}.json',result);return result


def report(m,native,strict):
 lines=['# FD002 seed43第一轮追踪','',
 '[Observed] 比较两个当前入口；每条轨迹仅第1轮163次更新，调度器仍按原70轮配置。未访问真实测试集、生成样本或训练预测器。历史9月14日逐步状态缺失，本轮不能恢复其原始轨迹。','',
 '|比较|初始权重一致|163批随机题目全一致|首次数值分叉批次|首轮平均损失|','|---|---|---|---|---|']
 for row in [native]+([strict] if strict else []):
  losses=' → '.join(f'{v:.10f}' for v in row['first_epoch_losses'])
  lines.append(f"|{' / '.join(row['pair'])}|{row['initial_states_equal']}|{row['all163_questions_equal']}|{row['first_divergence']}|{losses}|")
 if not native['initial_states_equal'] or not native['all163_questions_equal']:
  conclusion='[Observed] 自然运行初始状态或随机题目不同；先定位产生差异的入口/抽样阶段。本轮不把数值差异归因于CUDA，也不自动追加确定性对照。'
 elif native['all163_numerics_equal']:
  conclusion='[Observed] 本次两个入口在全部163批的随机题目、损失、梯度、Adam状态及更新后权重逐位一致。没有重现原历史分叉，因此不能认定审计入口本身必然改变训练，也不能认定CUDA非确定性已被证实。'
 elif strict and strict['all163_numerics_equal']:
  conclusion='[Observed] 自然模式在相同初始权重及全部相同随机题目下发生数值分叉；强制确定性计算后，两入口163批逐位一致。[Hypothesis] 支持计算可重复性是当前分叉的贡献因素；仍不能唯一解释原历史第67轮DS差距。'
 elif strict:
  conclusion='[Observed] 固定题目和强制确定性之后仍有分叉；简单CUDA非确定性解释不足，需结合首次不同的具体算子/状态进一步定位。本轮不自动扩展。'
 else:conclusion='[Observed] 自然模式出现相同题目下数值分叉，后续严格确定性阶段未能完成；以保存的错误为准，不放宽检查。'
 lines+=['',conclusion,'','[Code] 自然模式两入口使用同一线程/CUBLAS/TF环境，原seed helper保留cudnn.benchmark=True；条件严格阶段只对计算确定性开关及等值题目回放做显式控制，所有随机draw仍消费并逐元素核验。追踪保存会同步GPU、改变时序和自动选算法，不能把本轮等同于原未插桩时序。',
 '',f"[Observed] 实际完成{len(m['completed_arms'])}条第一轮轨迹，共{163*len(m['completed_arms'])}次更新；没有任何第2轮。每批完整timestep/噪声/掩码、梯度和更新后权重已保存。",'',
 '[Observed] 本次诊断→无论文对应目标→论文差距不适用→两个入口条件配对、不是新增生成种子结果→严格复现结论保持。seed43下游仍停止、seed13结果冻结；无默认规则变更或新实验自动启动。','',
 f'快照：`{RUN.relative_to(ROOT)}/run-manifest.json`；结果：`{OUT.relative_to(ROOT)}`。']
 REPORT.write_text('\n'.join(lines)+'\n');return conclusion


def run():
 m=check();assert m['status']=='prepared'
 m.update(status='running',started_at=now());dump(RUN/'run-manifest.json',m)
 env=os.environ.copy();env.update(m['config']['environment']);env['TF_CPP_MIN_LOG_LEVEL']='2'
 def cell(arm):
  m.update(current_arm=arm,stage_started_at=now());dump(RUN/'run-manifest.json',m)
  with (OUT/arm/'console.log').open('w') as log:
   subprocess.run([sys.executable,'-u',str(Path(__file__)),'worker',arm],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  m['completed_arms'].append(arm);dump(RUN/'run-manifest.json',m)
 try:
  cell('native_public');cell('native_worker');native=compare('native_public','native_worker','native_comparison');strict=None
  if native['initial_states_equal'] and native['all163_questions_equal'] and not native['all163_numerics_equal']:
   cell('strict_public');cell('strict_worker');strict=compare('strict_public','strict_worker','strict_comparison')
  conclusion=report(m,native,strict);check()
  m.update(status='complete',completed_at=now(),conclusion=conclusion,report=str(REPORT.relative_to(ROOT)),actual_updates=163*len(m['completed_arms']),
   outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [*OUT.rglob('*'),REPORT] if p.is_file()})
  dump(RUN/'run-manifest.json',m);state('complete','- '+conclusion+f'\n- [Observed] {len(m["completed_arms"])} first-epoch trajectories/{m["actual_updates"]} updates. Report `{REPORT.relative_to(ROOT)}`; no follow-up automatically launched.')
 except BaseException:
  m.update(status='failed',failed_at=now(),error=traceback.format_exc());dump(RUN/'run-manifest.json',m)
  state('failed',f'- [Observed] Stopped at{m.get("current_arm")}; preserve all traces/logs. No automatic retry or relaxation; never continue to epoch2.')
  raise

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','worker']);p.add_argument('arm',nargs='?',choices=ARMS);a=p.parse_args()
 if a.action=='worker':worker(a.arm)
 else:globals()[a.action]()
