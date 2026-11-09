"""Capture the localized op's exact inputs and perform bounded standalone replay."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,importlib.util,inspect,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/diagnostic-fd002-first-batch-op-localization-20261103'
OUT=ROOT/'results/diagnostics/fd002_first_batch_op_localization_20261103'
CAP=OUT/'capture'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def load():
 spec=importlib.util.spec_from_file_location('frozen_op_trace',ROOT/'scripts/localize_fd002_first_batch_ops.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def capture():
 m=load();m.check();d=json.loads((OUT/'operator_comparison.json').read_text());f=d['first_difference'];assert f['operator']=='aten.convolution.default'
 for q in ['tensors','native_public/tapes','native_public/states','native_public/entry_logs','native_public/worker/seed43','native_public/checkpoints/seed43']:(CAP/q).mkdir(parents=True,exist_ok=True)
 assert not (CAP/'native_public/summary.json').exists()
 # Keep the frozen tracer intact. Only request raw storage at the already identified index.
 source=inspect.getsource(m.worker);old="store=i<m['config']['raw_tensor_prefix_ops']";assert source.count(old)==1
 m.OUT=CAP;m.CAPTURE_INDEX=f['index']
 exec(compile(source.replace(old,'store=i==CAPTURE_INDEX'),'<raw_storage_at_localized_op>','exec'),m.__dict__)
 m.worker('native_public')
 rows=json.loads((CAP/'native_public/operators.json').read_text());op=rows[f['index']]
 assert op['operator']==f['operator'] and op['module']==f['module']
 assert m.clean(op['inputs'],True)==m.clean(f['public']['inputs'],True),'Captured input must exactly match original localized input'
 dump(CAP/'captured_operator.json',op)

def replay(index):
 import torch
 assert index in [1,2]
 f=json.loads((CAP/'captured_operator.json').read_text());assert f['operator']=='aten.convolution.default'
 def decode(z):
  k=z['kind']
  if k=='tensor':
   assert z['artifact'];v=torch.load(CAP/'tensors'/f'{z["artifact"]}.pth',map_location='cpu',weights_only=True).to('cuda')
   assert hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()==z['sha256']
   assert list(v.stride())==z['stride'] and v.storage_offset()==z['storage_offset'] and v.data_ptr()%256==z['data_alignment']
   return v.requires_grad_(z['requires_grad']) if v.is_floating_point() or v.is_complex() else v
  if k=='tuple':return tuple(decode(t) for t in z['items'])
  if k=='list':return [decode(t) for t in z['items']]
  if k=='dict':return {k:decode(v) for k,v in z['items'].items()}
  if k=='scalar':return z['value']
  raise ValueError(z)
 args=decode(f['inputs'])
 assert torch.is_grad_enabled();rows=[]
 for mode in ['native','local_deterministic']:
  torch.backends.cudnn.benchmark=mode=='native';torch.backends.cudnn.deterministic=mode!='native';torch.use_deterministic_algorithms(mode!='native')
  torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
  out=torch.ops.aten.convolution.default(*args['args'],**args['kwargs']);torch.cuda.synchronize();cpu=out.detach().cpu()
  target=OUT/f'replay{index}_{mode}.pth';torch.save(cpu,target)
  rows.append(dict(process=index,mode=mode,output_sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),shape=list(cpu.shape),
   benchmark=torch.backends.cudnn.benchmark,deterministic=torch.backends.cudnn.deterministic,strict_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
   torch=str(torch.__version__),cudnn=torch.backends.cudnn.version(),backward_calls=0,optimizer_updates=0))
 dump(OUT/f'replay{index}.json',rows)

def run():
 m=load();base=m.check();assert base['status']=='localized_pending_confirmation'
 assert not (RUN/'confirmation-manifest.json').exists()
 env=os.environ.copy();env.update(base['config']['environment'])
 paths=[Path(__file__),RUN/'run-manifest.json',OUT/'operator_comparison.json',ROOT/'scripts/localize_fd002_first_batch_ops.py']
 dump(RUN/'confirmation-manifest.json',dict(scope_class='diagnostic',role='supporting',status='running',started_at=now(),
  reason='First differing op721 lies beyond raw-prefix256. One additional prefix capture requests only its raw arguments/output, then at most4 isolated op calls; no training/new questions.',
  source_transformation='raw-storage predicate only; no model/arithmetic changes; frozen source unchanged',
  budget=dict(additional_full_first_batch_forward=1,standalone_op_calls=4,backward_calls=0,optimizer_updates=0),
  inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}))
 for action,arg,log in [('capture',None,CAP/'capture_console.log'),('replay','1',OUT/'replay1_console.log'),('replay','2',OUT/'replay2_console.log')]:
  log.parent.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,'-u',str(Path(__file__)),action]+([arg] if arg else [])
  with log.open('w') as h:subprocess.run(cmd,cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,check=True)
 v=json.loads((RUN/'confirmation-manifest.json').read_text());v.update(status='complete',completed_at=now(),
  outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [*CAP.rglob('*'),*OUT.glob('replay*')] if p.is_file()})
 dump(RUN/'confirmation-manifest.json',v)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','capture','replay']);p.add_argument('index',nargs='?',type=int);a=p.parse_args()
 if a.action=='replay':replay(a.index)
 else:globals()[a.action]()
