"""Supplementary fixed-input local convolution probe; not original-pair replay."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/diagnostic-fd002-first-batch-op-localization-20261103'
OUT=ROOT/'results/diagnostics/fd002_first_batch_op_localization_20261103'
PROBE=OUT/'local_probe';CAP=OUT/'capture'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def replay(index):
 import torch
 f=json.loads((PROBE/'frozen_operator.json').read_text());assert f['operator']=='aten.convolution.default'
 def decode(z):
  k=z['kind']
  if k=='tensor':
   v=torch.load(CAP/'tensors'/f'{z["artifact"]}.pth',map_location='cpu',weights_only=True).to('cuda')
   assert hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()==z['sha256']
   assert list(v.stride())==z['stride'] and v.storage_offset()==z['storage_offset']
   return v.requires_grad_(z['requires_grad']) if v.is_floating_point() or v.is_complex() else v
  if k in ['tuple','list']:
   values=[decode(t) for t in z['items']];return tuple(values) if k=='tuple' else values
  if k=='dict':return {k:decode(v) for k,v in z['items'].items()}
  if k=='scalar':return z['value']
  raise ValueError(z)
 args=decode(f['inputs']);rows=[]
 for mode in ['native','local_deterministic']:
  torch.backends.cudnn.benchmark=mode=='native';torch.backends.cudnn.deterministic=mode!='native';torch.use_deterministic_algorithms(mode!='native')
  torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
  out=torch.ops.aten.convolution.default(*args['args'],**args['kwargs']);torch.cuda.synchronize();cpu=out.detach().cpu();torch.save(cpu,PROBE/f'process{index}_{mode}.pth')
  rows.append(dict(process=index,mode=mode,output_sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),shape=list(cpu.shape),benchmark=torch.backends.cudnn.benchmark,deterministic=torch.backends.cudnn.deterministic,strict_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),backward_calls=0,optimizer_updates=0))
 dump(PROBE/f'process{index}.json',rows)

def run():
 PROBE.mkdir(exist_ok=True);assert not (RUN/'local-probe-manifest.json').exists()
 parent=json.loads((RUN/'run-manifest.json').read_text());failed=json.loads((RUN/'confirmation-manifest.json').read_text())
 failed.update(status='failed_input_parity_guard',failed_at=now(),reason='Supplementary capture differs upstream at op310; op721 input differs, so exact original-pair replay guard correctly stopped. No standalone calls ran. No further full forwards.')
 dump(RUN/'confirmation-manifest.json',failed)
 with (ROOT/'experiment-ledger.md').open('a') as h:h.write('\n## `'+RUN.name+'` raw-capture parity guard stopped\n\n- [Observed] Primary pair first differs at op721 seasonal Conv1d. Additional capture first differs from both primary traces at op310 mid_block1.block1.proj convolution; op721 input is now unequal. Preserve failed exact-replay guard, no relaxation or new full-forward retry. Separate four-call supplementary local probe freezes this third-capture input; cannot claim it is exact replay of original pair. Zero updates.\n')
 ops=json.loads((CAP/'native_public/operators.json').read_text());f=ops[parent['first_difference']['index']]
 dump(PROBE/'frozen_operator.json',f)
 inputs=[Path(__file__),RUN/'run-manifest.json',RUN/'confirmation-manifest.json',CAP/'native_public/operators.json',PROBE/'frozen_operator.json']
 for a in f['inputs']['items']['args']['items']:
  if a.get('kind')=='tensor':inputs.append(CAP/'tensors'/f'{a["artifact"]}.pth')
 dump(RUN/'local-probe-manifest.json',dict(status='running',scope_class='diagnostic',role='supporting',started_at=now(),
  purpose='Same identified convolution/module with a NEW explicitly frozen local input from third capture; not an exact replay of original paired activations',budget=dict(full_forwards=0,operator_calls=4,optimizer_updates=0),
  inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs}))
 env=os.environ.copy();env.update(parent['config']['environment'])
 for i in [1,2]:
  with (PROBE/f'process{i}_console.log').open('w') as h:subprocess.run([sys.executable,'-u',str(Path(__file__)),'replay',str(i)],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,check=True)
 m=json.loads((RUN/'local-probe-manifest.json').read_text());m.update(status='complete',completed_at=now(),outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in PROBE.iterdir() if p.is_file()});dump(RUN/'local-probe-manifest.json',m)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','replay']);p.add_argument('index',nargs='?',type=int);a=p.parse_args()
 if a.action=='replay':replay(a.index)
 else:run()
