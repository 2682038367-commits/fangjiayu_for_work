"""Locate first unequal ATen output for the fixed FD002 seed43 first batch."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/diagnostic-fd002-first-batch-op-localization-20261103'
OUT=ROOT/'results/diagnostics/fd002_first_batch_op_localization_20261103'
PRIOR=ROOT/'runs/diagnostic-fd002-seed43-first-epoch-trace-20261103'
PRIOR_OUT=ROOT/'results/diagnostics/fd002_seed43_first_epoch_trace_20261103'
PROTOCOL=ROOT/'docs/operations/FD002_FIRST_BATCH_OP_LOCALIZATION.md'
REPORT=ROOT/'docs/conclusions/final/fd002_first_batch_op_localization_20261103_report.md'

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def check():
 m=json.loads((RUN/'run-manifest.json').read_text())
 for p,h in m['inputs_sha256'].items():assert sha(ROOT/p)==h,p
 return m

def state(stage,text):
 text+=f'\n- Diagnostic/supporting snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`; protocol `{PROTOCOL.relative_to(ROOT)}`; no training/updates/generated metrics/default changes. Strict main/A/B unchanged,C deferred; FD003/4 and125 analyses paused.\n'
 p=ROOT/'reproduction-state.md';h,t=p.read_text().split('\n',1);p.write_text(h+f'\n\n## Latest checkpoint: 2026-11-03 FD002 first-batch operator localization {stage}\n\n'+text+'\n'+t)
 with (ROOT/'experiment-ledger.md').open('a') as f:f.write('\n## `'+RUN.name+'` '+stage+'\n\n'+text)

def prepare():
 assert not (RUN/'run-manifest.json').exists(),'No overwrite'
 for p in [RUN/'source',OUT/'native_public',OUT/'native_worker',OUT/'tensors']:
  p.mkdir(parents=True,exist_ok=True)
 for arm in ['native_public','native_worker']:
  for d in ['tapes','states','entry_logs','worker/seed43','checkpoints/seed43']:(OUT/arm/d).mkdir(parents=True,exist_ok=True)
 previous=json.loads((PRIOR/'postreview-manifest.json').read_text());assert previous['status']=='complete_with_deterministic_control_blocked'
 # Verify only canonical inputs needed now; all prior raw traces sealed in that review.
 paths=[Path(__file__),PROTOCOL,PRIOR/'run-manifest.json',PRIOR/'postreview-manifest.json',
  PRIOR_OUT/'native_public/initial_state.pth',PRIOR_OUT/'native_worker/initial_state.pth',
  PRIOR_OUT/'native_public/tapes/0001.pth',PRIOR_OUT/'native_worker/tapes/0001.pth',
  PRIOR_OUT/'native_public/trace.json',PRIOR_OUT/'native_worker/trace.json',
  ROOT/'scripts/run_fd002_first_epoch_trace.py',ROOT/'scripts/run_fd002_final70_two_seeds.py',
  ROOT/'runs/diagnostic-fd002-final70-two-seeds-20261102/frozen_inputs.npz']
 for p in paths:
  rel=str(p.relative_to(ROOT))
  if rel in previous['inputs_sha256']:assert sha(p)==previous['inputs_sha256'][rel]
 paths+=list((ROOT/'upstream').glob('*.py'))+list((ROOT/'upstream/DiffusionFreeGuidence').glob('*.py'))
 paths+=[ROOT/'upstream/data/CMAPSSDataset.py',ROOT/'upstream/data/train_FD002.txt']
 cfg=dict(seed=43,first_batch_only=True,backward_calls=0,optimizer_updates=0,initial_trace_forwards=2,
  raw_tensor_prefix_ops=256,environment=dict(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TF_DETERMINISTIC_OPS='1',TF_ENABLE_ONEDNN_OPTS='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='offline',TF_CPP_MIN_LOG_LEVEL='2'),
  stop='after first forward before backward; locate first unequal output with identical full inputs; limited standalone operator confirmation only')
 dump(RUN/'config.json',cfg);paths.append(RUN/'config.json')
 for p in [Path(__file__),PROTOCOL]:q=RUN/'source'/p.name;shutil.copy2(p,q);paths.append(q)
 for name,cmd in [('environment-freeze.txt',[sys.executable,'-m','pip','freeze']),('git-status.txt',['git','status','--short']),('workspace.diff',['git','diff','--binary'])]:
  p=RUN/name;p.write_bytes(subprocess.check_output(cmd,cwd=ROOT));paths.append(p)
 dump(RUN/'run-manifest.json',dict(run_id=RUN.name,scope_class='diagnostic',role='supporting',scope_contract='scope-20261103-fd002-first-batch-op-localization',
  status='prepared',prepared_at=now(),command=[sys.executable,'-u',str(Path(__file__)),'run'],user_authorization='定位首批中第一个输出不同的算子',
  config=cfg,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(paths)},completed_arms=[]))
 state('prepared','- [Observed] Freeze canonical first batch/initial weights; two current entry forward-only operator traces, no backward/updates. Optional limited standalone replay of first differing supported op; whole-network strict mode excluded. Metrics pending.')

class ForwardComplete(Exception):pass

def worker(arm):
 import numpy as np
 import torch
 from torch.utils._python_dispatch import TorchDispatchMode
 m=check();target=OUT/arm
 sys.path.insert(0,str(ROOT/'upstream'))
 import TrainCondition as training
 import GaussianDiffusion as diffusion
 import DiffusionFreeGuidence.Unet1D_fre as unet
 old=module(ROOT/'scripts/run_fd002_first_epoch_trace.py','sealed_first_epoch_worker')
 old.OUT=OUT
 ctx=dict(stack=[],events=[],model=None,forward_count=0)
 original_factory=training.UNet1D_fre;original_loss=diffusion.GaussianDiffusion1D_cls_free.p_losses
 canonical=torch.load(PRIOR_OUT/'native_public/tapes/0001.pth',map_location='cpu',weights_only=False)
 canonical_initial=torch.load(PRIOR_OUT/'native_public/initial_state.pth',map_location='cpu',weights_only=True)
 def tensors(x):
  if torch.is_tensor(x):return [x]
  if isinstance(x,(list,tuple)):return sum([tensors(i) for i in x],[])
  if isinstance(x,dict):return sum([tensors(i) for i in x.values()],[])
  return []
 def tensor_desc(t,store=False):
  with torch._C._DisableTorchDispatch(),torch.no_grad():
   cp=t.detach().resolve_conj().resolve_neg().cpu().contiguous()
   h=hashlib.sha256(cp.numpy().tobytes()).hexdigest()
   tag=hashlib.sha256((str(cp.dtype)+str(tuple(cp.shape))+h).encode()).hexdigest()
   if store:
    p=OUT/'tensors'/f'{tag}.pth'
    if not p.exists():torch.save(cp,p)
   return dict(kind='tensor',sha256=h,artifact=tag if store else None,shape=list(t.shape),stride=list(t.stride()),
    storage_offset=int(t.storage_offset()),data_alignment=int(t.data_ptr()%256),dtype=str(t.dtype),device=str(t.device),
    contiguous=t.is_contiguous(),requires_grad=t.requires_grad)
 def describe(x,store=False):
  if torch.is_tensor(x):return tensor_desc(x,store)
  if isinstance(x,tuple):return dict(kind='tuple',items=[describe(v,store) for v in x])
  if isinstance(x,list):return dict(kind='list',items=[describe(v,store) for v in x])
  if isinstance(x,dict):return dict(kind='dict',items={str(k):describe(v,store) for k,v in x.items()})
  if x is None or isinstance(x,(str,bool,int,float)):return dict(kind='scalar',value=x)
  return dict(kind='literal',type=type(x).__name__,value=str(x))
 def origin():
  frame=sys._getframe(2)
  while frame:
   p=Path(frame.f_code.co_filename)
   if str(p).startswith(str(ROOT/'upstream')):return dict(path=str(p.relative_to(ROOT)),line=frame.f_lineno,function=frame.f_code.co_name)
   if str(p).startswith(str(ROOT/'scripts')) and p!=Path(__file__):return None
   frame=frame.f_back
  return None
 class Trace(TorchDispatchMode):
  def __torch_dispatch__(self,func,types,args=(),kwargs=None):
   kwargs={} if kwargs is None else kwargs
   source=origin()
   if source is None:return func(*args,**kwargs)
   i=len(ctx['events']);store=i<m['config']['raw_tensor_prefix_ops']
   before=describe(dict(args=args,kwargs=kwargs),store)
   output=func(*args,**kwargs)
   after=describe(output,store)
   ctx['events'].append(dict(index=i,operator=str(func),module=ctx['stack'][-1] if ctx['stack'] else '<outside_model>',source=source,inputs=before,output=after))
   return output
 def factory(*a,**kw):
  model=original_factory(*a,**kw);ctx['model']=model
  assert all(torch.equal(t,canonical_initial[k]) for k,t in model.state_dict().items())
  for name,mod in model.named_modules():
   label=name or '<model>'
   def pre(mod,args,n=label):ctx['stack'].append(n)
   def post(mod,args,out,n=label):assert ctx['stack'].pop()==n
   mod.register_forward_pre_hook(pre);mod.register_forward_hook(post)
  return model
 training.UNet1D_fre=factory
 def first_loss(trainer,x_start,t,*,classes,noise=None):
  ctx['forward_count']+=1;assert ctx['forward_count']==1
  # Read the sealed worker's observer state, without changing its draws/hooks.
  current_method=diffusion.GaussianDiffusion1D_cls_free.p_losses
  closure=inspect.getclosurevars(current_method).nonlocals
  innerctx=closure['ctx']
  assert innerctx['updates']==0
  ctx['pre_forward_rng_sha256']=hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
  with Trace():value=original_loss(trainer,x_start,t,classes=classes,noise=noise)
  actual=innerctx['tape']
  assert np.array_equal(actual['indices'],canonical['indices']) and actual['numpy_draw']==canonical['numpy_draw']
  for k in ['timestep','noise','condition_mask']:assert torch.equal(actual[k],canonical[k]),k
  assert all(torch.equal(v.cpu(),canonical_initial[k]) for k,v in ctx['model'].state_dict().items())
  summary=dict(status='complete',arm=arm,forward_calls=1,backward_calls=0,optimizer_updates=0,
   canonical_initial_and_question_exact=True,loss=float(value.item()),operators=len(ctx['events']),
   pre_forward_rng_sha256=ctx['pre_forward_rng_sha256'],torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),
   backend_flags=dict(benchmark=torch.backends.cudnn.benchmark,deterministic=torch.backends.cudnn.deterministic,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32),
   caveat='Synchronous ATen tracing can alter autotuning/timing; current entry replay, not original historical kernels')
  dump(target/'operators.json',ctx['events']);dump(target/'summary.json',summary)
  raise ForwardComplete()
 diffusion.GaussianDiffusion1D_cls_free.p_losses=first_loss
 def forbidden(*a,**kw):raise RuntimeError('Backward or optimizer step forbidden in first-batch localization')
 torch.optim.Adam.step=forbidden;torch.Tensor.backward=forbidden
 try:old.worker(arm)
 except ForwardComplete:pass
 assert ctx['forward_count']==1 and (target/'summary.json').exists()


def clean(x,layout=False):
 if isinstance(x,dict):
  ignore={'artifact','data_alignment','requires_grad'}
  if not layout:ignore|={'stride','storage_offset','contiguous','device'}
  return {k:clean(v,layout) for k,v in x.items() if k not in ignore}
 if isinstance(x,list):return [clean(v,layout) for v in x]
 return x

def compare():
 import torch
 import numpy as np
 a,b=[json.loads((OUT/arm/'operators.json').read_text()) for arm in ['native_public','native_worker']]
 summary=[json.loads((OUT/arm/'summary.json').read_text()) for arm in ['native_public','native_worker']]
 assert len(a)==len(b) and all(x['operator']==y['operator'] and x['module']==y['module'] for x,y in zip(a,b))
 assert summary[0]['pre_forward_rng_sha256']==summary[1]['pre_forward_rng_sha256']
 first=None;rows=[]
 for x,y in zip(a,b):
  same_in=clean(x['inputs'])==clean(y['inputs']);same_layout=clean(x['inputs'],True)==clean(y['inputs'],True)
  same_out=clean(x['output'])==clean(y['output'])
  row=dict(index=x['index'],operator=x['operator'],module=x['module'],same_inputs=same_in,same_input_layout=same_layout,same_outputs=same_out,source=x['source'])
  rows.append(row)
  if first is None and not same_out:
   first=dict(**row,public=x,worker=y)
   assert same_in,'First difference received unequal values; inspect earlier untraced producer'
   def leaves(z):
    if isinstance(z,dict):
     if z.get('kind')=='tensor':return [z]
     return sum([leaves(v) for v in z.values()],[])
    if isinstance(z,list):return sum([leaves(v) for v in z],[])
    return []
   deltas=[]
   for aa,bb in zip(leaves(x['output']),leaves(y['output'])):
    if aa['artifact'] and bb['artifact']:
     ta=torch.load(OUT/'tensors'/f'{aa["artifact"]}.pth',map_location='cpu',weights_only=True)
     tb=torch.load(OUT/'tensors'/f'{bb["artifact"]}.pth',map_location='cpu',weights_only=True)
     if ta.is_complex():ta,tb=ta.to(torch.complex128),tb.to(torch.complex128)
     elif ta.is_floating_point():ta,tb=ta.to(torch.float64),tb.to(torch.float64)
     else:ta,tb=ta.to(torch.int64),tb.to(torch.int64)
     delta=abs(ta-tb);deltas.append(dict(shape=list(ta.shape),max_abs=float(delta.max().item()),different_elements=int((delta!=0).sum().item()),elements=ta.numel(),relative_l2=float(torch.linalg.vector_norm(delta.to(torch.float64)).item()/max(torch.linalg.vector_norm(abs(ta).to(torch.float64)).item(),1e-300))))
   first['output_differences']=deltas
 dump(OUT/'operator_comparison.json',dict(operator_count=len(a),first_difference=first,rows=rows,summaries=summary))
 return first


def run():
 m=check();assert m['status']=='prepared';m.update(status='running',started_at=now());dump(RUN/'run-manifest.json',m)
 env=os.environ.copy();env.update(m['config']['environment'])
 try:
  for arm in ['native_public','native_worker']:
   m['current_arm']=arm;dump(RUN/'run-manifest.json',m)
   with (OUT/arm/'console.log').open('w') as f:subprocess.run([sys.executable,'-u',str(Path(__file__)),'worker',arm],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
   m['completed_arms'].append(arm);dump(RUN/'run-manifest.json',m)
  first=compare();m.update(status='localized_pending_confirmation' if first else 'no_difference_observed',first_difference=first,completed_at=now())
  dump(RUN/'run-manifest.json',m)
 except BaseException:
  m.update(status='failed',error=traceback.format_exc(),failed_at=now());dump(RUN/'run-manifest.json',m);state('failed','- [Observed] Preserve failing trace/log; no updates or silent retries.');raise
 print(json.dumps(dict(status=m['status'],first=first),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','worker']);p.add_argument('arm',nargs='?',choices=['native_public','native_worker']);a=p.parse_args()
 if a.action=='worker':worker(a.arm)
 else:globals()[a.action]()
