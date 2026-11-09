"""Measure actual CUDA convolution kernels and paired fixed-path first-batch outputs."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,importlib.util,inspect,json,os,shutil,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/diagnostic-fd002-conv-kernel-paths-20261103'
OUT=ROOT/'results/diagnostics/fd002_conv_kernel_paths_20261103'
PROTOCOL=ROOT/'docs/operations/FD002_CONV_KERNEL_PATHS.md'
REPORT=ROOT/'docs/conclusions/final/fd002_conv_kernel_paths_20261103_report.md'
BASE=ROOT/'scripts/localize_fd002_first_batch_ops.py'
PRIOR=ROOT/'runs/diagnostic-fd002-first-batch-op-localization-20261103'
CELLS=[('native','native_public'),('native','native_worker'),('fixed','native_public'),('fixed','native_worker')]
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def transformed_source():
 api=module(BASE,'sealed_aten_tracer_for_source');s=inspect.getsource(api.worker)
 old="store=i<m['config']['raw_tensor_prefix_ops']";assert s.count(old)==1
 s=s.replace(old,"store=str(func)=='aten.convolution.default'")
 old='   output=func(*args,**kwargs)\n   after=describe(output,store)';assert s.count(old)==1
 s=s.replace(old,"""   if str(func)=='aten.convolution.default':
    label=f'DIAG_CONV_{i:04d}::{ctx["stack"][-1]}'
    with torch.profiler.record_function(label):output=func(*args,**kwargs)
   else:output=func(*args,**kwargs)
   after=describe(output,store)""")
 old='  with Trace():value=original_loss(trainer,x_start,t,classes=classes,noise=noise)';assert s.count(old)==1
 s=s.replace(old,"""  import pickle
  torch.backends.cudnn.benchmark=KERNEL_MODE=='native'
  torch.backends.cudnn.deterministic=KERNEL_MODE=='fixed'
  assert not torch.are_deterministic_algorithms_enabled()
  before_cpu=torch.get_rng_state().clone();before_cuda=torch.cuda.get_rng_state().clone();before_numpy=np.random.get_state()
  # Warmup chooses the algorithm outside the measured profiler range.
  warm=original_loss(trainer,x_start,t,classes=classes,noise=noise)
  warm_loss=float(warm.item());warm_output_hash=innerctx['current']['model_output_sha256']
  actual_warm=innerctx['tape']
  for key in ['timestep','noise','condition_mask']:assert torch.equal(actual_warm[key],canonical[key])
  del warm
  innerctx['tape']={key:value for key,value in actual_warm.items() if key not in ['noise','condition_mask']}
  torch.set_rng_state(before_cpu);torch.cuda.set_rng_state(before_cuda);np.random.set_state(before_numpy)
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
   with Trace():value=original_loss(trainer,x_start,t,classes=classes,noise=noise)
   torch.cuda.synchronize()
  prof.export_chrome_trace(str(target/'cuda_trace.json'))
  def kernel_desc(event):
   result=[]
   for k in event.kernels:
    if not k.name.startswith(('Memset','Memcpy')):result.append(dict(name=k.name,device=k.device,duration_us=k.duration,owner=event.name))
   for child in event.cpu_children:result.extend(kernel_desc(child))
   return result
  kernel_rows=[]
  for event in prof.events():
   if event.name.startswith('DIAG_CONV_'):
    prefix,module_name=event.name.split('::',1);op_index=int(prefix.split('_')[-1])
    kernels=kernel_desc(event);assert kernels,'No actual CUDA kernel for '+event.name
    kernel_rows.append(dict(index=op_index,module=module_name,kernels=kernels))
  assert kernel_rows and len(kernel_rows)==sum(e['operator']=='aten.convolution.default' for e in ctx['events'])
  dump(target/'actual_conv_kernels.json',kernel_rows)
  dump(target/'profile_summary.json',dict(mode=KERNEL_MODE,warmup_loss=warm_loss,measured_loss=float(value.item()),
   warmup_output_sha256=warm_output_hash,measured_output_sha256=innerctx['current']['model_output_sha256'],
   warmup_measured_exact_output_equal=warm_output_hash==innerctx['current']['model_output_sha256'],
   full_first_batch_forwards=2,convolutions_measured=len(kernel_rows),backward_calls=0,optimizer_updates=0,
   benchmark=torch.backends.cudnn.benchmark,deterministic=torch.backends.cudnn.deterministic,
   global_deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))""")
 return s

def check():
 m=json.loads((RUN/'run-manifest.json').read_text())
 for p,h in m['inputs_sha256'].items():assert sha(ROOT/p)==h,p
 return m

def state(stage,body):
 body+=f'\n- Diagnostic/supporting snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`; protocol `{PROTOCOL.relative_to(ROOT)}`. Zero backward/updates/default changes; strict main/A/B unchanged,C deferred; FD003/4 and125 paused.\n'
 p=ROOT/'reproduction-state.md';h,t=p.read_text().split('\n',1);p.write_text(h+f'\n\n## Latest checkpoint: 2026-11-03 FD002 convolution kernels {stage}\n\n'+body+'\n'+t)
 with (ROOT/'experiment-ledger.md').open('a') as f:f.write('\n## `'+RUN.name+'` '+stage+'\n\n'+body)

def prepare():
 assert not (RUN/'run-manifest.json').exists()
 paths=[Path(__file__),PROTOCOL,BASE,PRIOR/'run-manifest.json',PRIOR/'postreview-manifest.json',
  ROOT/'scripts/run_fd002_first_epoch_trace.py',ROOT/'scripts/run_fd002_final70_two_seeds.py',
  ROOT/'runs/diagnostic-fd002-final70-two-seeds-20261102/frozen_inputs.npz',
  ROOT/'results/diagnostics/fd002_seed43_first_epoch_trace_20261103/native_public/initial_state.pth',
  ROOT/'results/diagnostics/fd002_seed43_first_epoch_trace_20261103/native_public/tapes/0001.pth']
 paths+=list((ROOT/'upstream').glob('*.py'))+list((ROOT/'upstream/DiffusionFreeGuidence').glob('*.py'))
 paths+=[ROOT/'upstream/data/CMAPSSDataset.py',ROOT/'upstream/data/train_FD002.txt']
 for mode,arm in CELLS:
  for name in ['tapes','states','entry_logs','worker/seed43','checkpoints/seed43']:(OUT/mode/arm/name).mkdir(parents=True,exist_ok=True)
  (OUT/mode/'tensors').mkdir(parents=True,exist_ok=True)
 source=RUN/'source/profiled_worker.py';source.write_text(transformed_source());paths.append(source)
 for p in [Path(__file__),PROTOCOL]:q=RUN/'source'/p.name;shutil.copy2(p,q);paths.append(q)
 for origin,name in [('/tmp/fd002_kernel_profiler_preflight.py','profiler_preflight.py'),('/tmp/fd002-kernel-preflight-summary.json','profiler_preflight_summary.json'),('/tmp/fd002-kernel-preflight.json','profiler_preflight_trace.json'),('/tmp/fd002-kernel-preflight.log','profiler_preflight.log')]:
  q=RUN/'source'/name;shutil.copy2(origin,q);paths.append(q)
 cfg=dict(seed=43,cells=CELLS,network_first_batch_forwards_max=8,backward_calls=0,optimizer_updates=0,
  actual_path_control='native benchmarkTrue/deterministicFalse vs fixed benchmarkFalse/deterministicTrue; all other flags/ops/questions unchanged',
  median='No global strict deterministic switch or operator replacement; compare actual convolution kernels to verify path equality',
  environment=dict(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TF_DETERMINISTIC_OPS='1',TF_ENABLE_ONEDNN_OPTS='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='offline',TF_CPP_MIN_LOG_LEVEL='2'),
  profiler='Kineto CPU+CUDA, measured cached forward only; marked per-convolution ancestry; export Chrome traces',
  stop='Fixed4 cells, no seed/replay selection. Report null result if native divergence not observed; unsupported/empty GPU activity stops.')
 dump(RUN/'config.json',cfg);paths.append(RUN/'config.json')
 for name,cmd in [('environment-freeze.txt',[sys.executable,'-m','pip','freeze']),('git-status.txt',['git','status','--short']),('workspace.diff',['git','diff','--binary'])]:
  p=RUN/name;p.write_bytes(subprocess.check_output(cmd,cwd=ROOT));paths.append(p)
 dump(RUN/'run-manifest.json',dict(run_id=RUN.name,scope_class='diagnostic',role='supporting',scope_contract='scope-20261103-fd002-conv-kernel-paths',status='prepared',prepared_at=now(),
  user_authorization='直接记录并比较两次卷积使用的实际内核，再检验固定计算路径能否消除分叉',
  command=[sys.executable,'-u',str(Path(__file__)),'run'],config=cfg,inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(paths)},completed_cells=[]))
 state('prepared','- [Observed] Actual CUDA profiling preflight passed with recorded kernel names. Freeze4 cells/8 first-batch forwards, selection warmup then canonical-question replay under measured ATen/CUDA profiler. Capture every convolution arguments/output plus actual kernels; compare native vs fixed convolution selection, no global strict mode/median replacement. Metrics pending.')

def worker(mode,arm):
 check();api=module(BASE,'sealed_trace_api');api.RUN=RUN;api.OUT=OUT/mode;api.KERNEL_MODE=mode
 source=(RUN/'source/profiled_worker.py').read_text();assert source==transformed_source()
 exec(compile(source,'<profiled_fixed_question_entry>','exec'),api.__dict__)
 api.worker(arm)


def compare(mode):
 import torch
 api=module(BASE,'compare_current_kernels')
 aa,bb=[json.loads((OUT/mode/a/'operators.json').read_text()) for a in ['native_public','native_worker']]
 ka,kb=[json.loads((OUT/mode/a/'actual_conv_kernels.json').read_text()) for a in ['native_public','native_worker']]
 assert len(aa)==len(bb)==779
 assert all(a['operator']==b['operator'] and a['module']==b['module'] for a,b in zip(aa,bb))
 kernels_a={r['index']:r for r in ka};kernels_b={r['index']:r for r in kb};assert kernels_a.keys()==kernels_b.keys()
 first=None;convrows=[]
 for i,(a,b) in enumerate(zip(aa,bb)):
  same_in=api.clean(a['inputs'])==api.clean(b['inputs']);same_layout=api.clean(a['inputs'],True)==api.clean(b['inputs'],True)
  same_out=api.clean(a['output'])==api.clean(b['output'])
  if first is None and not same_out:
   assert same_in and same_layout
   first=dict(index=i,operator=a['operator'],module=a['module'],source=a['source'],same_input_values=True,same_input_layout=True,
    public_output_sha256=a['output']['sha256'],worker_output_sha256=b['output']['sha256'])
   if a['operator']=='aten.convolution.default':
    ta=torch.load(OUT/mode/'tensors'/f'{a["output"]["artifact"]}.pth',map_location='cpu',weights_only=True)
    tb=torch.load(OUT/mode/'tensors'/f'{b["output"]["artifact"]}.pth',map_location='cpu',weights_only=True)
    delta=abs(ta.double()-tb.double());first.update(max_abs=float(delta.max()),different_elements=int((delta!=0).sum()),elements=delta.numel())
  if i in kernels_a:
   names_a=[k['name'] for k in kernels_a[i]['kernels']];names_b=[k['name'] for k in kernels_b[i]['kernels']]
   convrows.append(dict(index=i,module=a['module'],same_inputs=same_in,same_outputs=same_out,same_kernel_sequence=names_a==names_b,
    public_kernels=names_a,worker_kernels=names_b))
 if first:
  first['kernel_record']=next((r for r in convrows if r['index']==first['index']),None)
 result=dict(mode=mode,operator_count=779,convolution_count=len(convrows),first_difference=first,
  all_operator_outputs_equal=first is None,all_convolution_kernel_sequences_equal=all(r['same_kernel_sequence'] for r in convrows),convolutions=convrows)
 dump(OUT/f'{mode}_comparison.json',result);return result


def report(m,native,fixed):
 for mode in ['native','fixed']:
  for arm in ['native_public','native_worker']:
   s=json.loads((OUT/mode/arm/'summary.json').read_text());p=json.loads((OUT/mode/arm/'profile_summary.json').read_text())
   assert s['canonical_initial_and_question_exact'] and p['backward_calls']==p['optimizer_updates']==0
   assert p['full_first_batch_forwards']==2 and not p['global_deterministic_algorithms']
 nf=native['first_difference'];ff=fixed['first_difference']
 rows=[]
 for name,d in [('自动选择',native),('固定卷积设置',fixed)]:
  f=d['first_difference'];rows.append(f"|{name}|{'全779个操作一致' if f is None else f['module']}|{'一致' if d['all_convolution_kernel_sequences_equal'] else '存在不同'}|{'一致' if d['all_operator_outputs_equal'] else '存在不同'}|")
 lines=['# FD002实际卷积内核与固定路径对照（2026-11-03）','',
 '[Observed] 本轮直接用Kineto/CUDA活动记录实际卷积内核。每个入口先完成一次相同题目的算法选择前向，再恢复完整随机状态和题目记录，测量缓存路径下前向；四进程共8次首批前向，0反向/0参数更新。预检另执行一次固定输入卷积以验证CUDA活动采集。','',
 '|卷积设置|首个输出不同位置|两入口全部卷积内核序列|全部已跟踪输出|','|---|---|---|---|',*rows,'']
 if nf and nf['kernel_record']:
  k=nf['kernel_record'];lines += ['## 自动选择模式的直接证据','',f"[Observed] 首个不同操作：{nf['operator']}，模块`{nf['module']}`，索引{nf['index']}。输入、权重、偏置及布局相同；最大输出差{nf.get('max_abs')}，不同元素{nf.get('different_elements')}/{nf.get('elements')}。实际内核序列是否相同：{k['same_kernel_sequence']}。",'', '公开入口实际内核：','',*[f'- `{n}`' for n in k['public_kernels']],'','审计入口实际内核：','',*[f'- `{n}`' for n in k['worker_kernels']],'']
 if nf and nf['kernel_record'] and not nf['kernel_record']['same_kernel_sequence'] and fixed['all_operator_outputs_equal'] and fixed['all_convolution_kernel_sequences_equal']:
  conclusion='[Observed] 本次确认了计算路径贡献：自动模式在同一卷积输入下使用不同实际内核并产生不同输出；固定卷积设置后，两入口实际卷积内核序列和全部779个跟踪输出逐位一致。[Hypothesis] 支持卷积自动选路径/浮点舍入是当前首批分叉的原因；仍不能唯一解释历史第67轮生成质量差距。'
 elif nf is None:
  conclusion='[Observed] 本轮自动模式没有重现输出分叉；固定模式结果如表。不能据此声称固定设置消除了分叉，不补选进程或种子。'
 else:
  conclusion='[Observed] 原始实际内核与输出差异已记录，但结果未同时满足“自动模式首个分叉内核不同、固定模式路径与全部输出相同”的完整判据；不能宣布单一根因或修复成立。'
 lines += [conclusion,'','[Code] 唯一受控因素为卷积选择设置：native benchmark=True/deterministic=False；fixed benchmark=False/deterministic=True。TF32、数据、初始权重、随机题目、模型算式和环境保持；全局torch严格确定性在所有组均为False，median原实现保留。固定设置是否得到相同路径以测得内核为准。',
 '', '[Observed] 只把record_function卷积标记的CPU后代所附GPU实际kernel活动计入该次卷积；算法搜索在未计时warmup中完成，避免把搜索候选误当成最终实际执行路径。原始Chrome traces、cuDNN日志、每个卷积的完整输入/输出和内核名保存。',
 '', '[Hypothesis] Profiler/同步和warmup可改变算法选择，当前是受控重放而非恢复历史内核。即使本轮首批分叉消失，也不等于整条训练/生成严格可重复，更未验证历史DS或下游收益得到修复。',
 '', '[Observed] 本轮结束，不修改上游默认代码、不训练70轮或生成/下游评估。严格main/A/B结论保持、C推迟，FD003/4与125分析暂停。诊断→无论文同设计指标→论文数值差距不适用→有限入口配对→严格复现结论不变。','',
 f'快照：`{RUN.relative_to(ROOT)}/run-manifest.json`；完整结果/原始profile：`{OUT.relative_to(ROOT)}`。']
 REPORT.write_text('\n'.join(lines)+'\n');return conclusion


def run():
 m=check();assert m['status']=='prepared';m.update(status='running',started_at=now());dump(RUN/'run-manifest.json',m)
 env=os.environ.copy();env.update(m['config']['environment'])
 try:
  for mode,arm in CELLS:
   m.update(current_mode=mode,current_entry=arm,stage_started_at=now());dump(RUN/'run-manifest.json',m)
   cellenv=env.copy();cellenv.update(CUDNN_LOGINFO_DBG='1',CUDNN_LOGWARN_DBG='1',CUDNN_LOGDEST_DBG=str(OUT/mode/arm/'cudnn.log'))
   with (OUT/mode/arm/'console.log').open('w') as h:subprocess.run([sys.executable,'-u',str(Path(__file__)),'worker',mode,arm],cwd=ROOT,env=cellenv,stdout=h,stderr=subprocess.STDOUT,check=True)
   m['completed_cells'].append([mode,arm]);dump(RUN/'run-manifest.json',m)
  native,fixed=compare('native'),compare('fixed');conclusion=report(m,native,fixed);check()
  m.update(status='complete',completed_at=now(),report=str(REPORT.relative_to(ROOT)),conclusion=conclusion,
   outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [*OUT.rglob('*'),REPORT] if p.is_file()})
  dump(RUN/'run-manifest.json',m);state('complete','- '+conclusion+f'\n- Report `{REPORT.relative_to(ROOT)}`.8 first-batch forwards/0 backward/0updates; budget closed, no automatic default adoption or new experiments.')
 except BaseException:
  m.update(status='failed',failed_at=now(),error=traceback.format_exc());dump(RUN/'run-manifest.json',m);state('failed','- [Observed] Preserve profiler/logs and stop; no silent retry or weakening actual-kernel/input checks.');raise

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','worker']);p.add_argument('mode',nargs='?',choices=['native','fixed']);p.add_argument('arm',nargs='?',choices=['native_public','native_worker']);a=p.parse_args()
 if a.action=='worker':worker(a.mode,a.arm)
 else:globals()[a.action]()
