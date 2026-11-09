import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,json
from pathlib import Path
p=Path('/home/cw_boe/projects/rul/metaindux-ts-reproduction/results/diagnostics/fd002_first_batch_op_localization_20261103/local_probe/frozen_operator.json');f=json.loads(p.read_text());base=p.parent.parent/'capture/tensors'
def decode(x):
 k=x['kind']
 if k=='tensor':return torch.load(base/f'{x["artifact"]}.pth',map_location='cpu',weights_only=True).cuda().requires_grad_(x['requires_grad'])
 if k=='scalar':return x['value']
 if k=='tuple':return tuple(decode(a) for a in x['items'])
 if k=='list':return [decode(a) for a in x['items']]
 if k=='dict':return {k:decode(v) for k,v in x['items'].items()}
a=decode(f['inputs']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
 with torch.profiler.record_function('DIAG_CONV_PREFLIGHT'):y=torch.ops.aten.convolution.default(*a['args'],**a['kwargs'])
 torch.cuda.synchronize()
prof.export_chrome_trace('/tmp/fd002-kernel-preflight.json')
for e in prof.events():
 if e.kernels or e.name=='DIAG_CONV_PREFLIGHT':print('EVENT',e.name,'kernels',[(k.name,k.device,k.duration) for k in e.kernels],'children',[c.name for c in e.cpu_children])
d=json.loads(Path('/tmp/fd002-kernel-preflight.json').read_text());ks=[v['name'] for v in d['traceEvents'] if v.get('cat')=='kernel'];assert ks,'No actual CUDA kernel activity recorded'
Path('/tmp/fd002-kernel-preflight-summary.json').write_text(json.dumps(dict(status='passed',actual_kernel_names=ks,forward_calls=1,optimizer_updates=0),indent=2)+'\n');print('ACTUAL_KERNELS',ks)
