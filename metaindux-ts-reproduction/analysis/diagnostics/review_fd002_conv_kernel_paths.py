"""Independently verify raw CUDA correlations and bitwise fixed-path outputs."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,importlib.util,json,shutil,sys
import torch
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/diagnostic-fd002-conv-kernel-paths-20261103'
OUT=ROOT/'results/diagnostics/fd002_conv_kernel_paths_20261103'
REV=OUT/'review'
REPORT=ROOT/'docs/conclusions/final/fd002_conv_kernel_paths_20261103_report.md'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not (RUN/'postreview-manifest.json').exists(),'No overwrite'
 REV.mkdir(exist_ok=True);started=now();m=json.loads((RUN/'run-manifest.json').read_text());assert m['status']=='complete'
 for group in ['inputs_sha256','outputs_sha256']:
  for p,h in m[group].items():assert sha(ROOT/p)==h,p
 spec=importlib.util.spec_from_file_location('tracer_clean',ROOT/'scripts/localize_fd002_first_batch_ops.py');api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
 maps={};summaries={};arrays={};missing_logs=[];tensor_count=0;association_discrepancies=[]
 for mode in ['native','fixed']:
  for arm in ['native_public','native_worker']:
   key=mode+'/'+arm;base=OUT/mode/arm
   events=json.loads((base/'cuda_trace.json').read_text())['traceEvents']
   markers=[e for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('DIAG_CONV_')]
   runtime=[e for e in events if e.get('cat') in ['cuda_runtime','cuda_driver']]
   kernels=[e for e in events if e.get('cat')=='kernel']
   recorded=json.loads((base/'actual_conv_kernels.json').read_text());assert len(markers)==len(recorded)==35
   byindex={r['index']:r for r in recorded};mapped=[]
   for marker in markers:
    prefix,module_name=marker['name'].split('::',1);index=int(prefix.split('_')[-1])
    launches=[e for e in runtime if e['pid']==marker['pid'] and e['tid']==marker['tid'] and marker['ts']<=e['ts']<=marker['ts']+marker['dur']]
    correlations={e['args']['correlation'] for e in launches if 'correlation' in e.get('args',{})}
    actual=sorted([e for e in kernels if e.get('args',{}).get('correlation') in correlations],key=lambda e:e['ts'])
    names=[e['name'] for e in actual]
    expected=[k['name'] for k in byindex[index]['kernels']]
    if names!=expected:association_discrepancies.append(dict(cell=key,index=index,raw_correlated_names=names,tree_attached_names=expected))
    assert names and len({e['args']['correlation'] for e in actual})==len(actual)
    mapped.append(dict(index=index,module=module_name,kernels=names,correlations=[e['args']['correlation'] for e in actual],
      source='Raw Chrome CPU marker -> cuda_runtime launch correlation -> GPU kernel event; not kernel-name inference'))
   maps[key]=mapped
   s=json.loads((base/'summary.json').read_text());p=json.loads((base/'profile_summary.json').read_text());summaries[key]=dict(entry=s,profile=p)
   assert s['canonical_initial_and_question_exact'] and p['warmup_measured_exact_output_equal']
   assert p['full_first_batch_forwards']==2 and p['backward_calls']==p['optimizer_updates']==0
   assert p['benchmark']==(mode=='native') and p['deterministic']==(mode=='fixed') and not p['global_deterministic_algorithms']
   arrays[key]=json.loads((base/'operators.json').read_text())
   assert len(arrays[key])==779
   if not (base/'cudnn.log').exists():missing_logs.append(key)
 assert len({s['entry']['pre_forward_rng_sha256'] for s in summaries.values()})==1
 a,b=arrays['native/native_public'],arrays['native/native_worker']
 first=next(i for i,(x,y) in enumerate(zip(a,b)) if api.clean(x['output'])!=api.clean(y['output']))
 assert first==21 and a[first]['module']=='init_conv'
 assert api.clean(a[first]['inputs'],True)==api.clean(b[first]['inputs'],True)
 # Verify saved input/weight/bias payloads and observed outputs, not only JSON hashes.
 def payload(desc,mode):
  p=OUT/mode/'tensors'/f'{desc["artifact"]}.pth';t=torch.load(p,map_location='cpu',weights_only=True)
  assert hashlib.sha256(t.numpy().tobytes()).hexdigest()==desc['sha256']
  return t
 for x,y in zip(a[first]['inputs']['items']['args']['items'],b[first]['inputs']['items']['args']['items']):
  if x.get('kind')=='tensor':assert torch.equal(payload(x,'native'),payload(y,'native'))
 native_out=[payload(z[first]['output'],'native') for z in [a,b]];delta=(native_out[0].double()-native_out[1].double()).abs()
 assert float(delta.max())==4.76837158203125e-7 and int((delta!=0).sum())==217433
 map_native=[{r['index']:r for r in maps['native/'+arm]} for arm in ['native_public','native_worker']]
 assert map_native[0][first]['kernels']!=map_native[1][first]['kernels']
 fa,fb=arrays['fixed/native_public'],arrays['fixed/native_worker']
 assert all(x['operator']==y['operator'] and api.clean(x['output'])==api.clean(y['output']) and api.clean(x['inputs'],True)==api.clean(y['inputs'],True) for x,y in zip(fa,fb))
 fixed_k=[{r['index']:r for r in maps['fixed/'+arm]} for arm in ['native_public','native_worker']]
 assert all(fixed_k[0][i]['kernels']==fixed_k[1][i]['kernels'] for i in fixed_k[0])
 for x,y in zip(fa,fb):
  if x['operator']=='aten.convolution.default':
   assert torch.equal(payload(x['output'],'fixed'),payload(y['output'],'fixed'));tensor_count+=2
   for xx,yy in zip(x['inputs']['items']['args']['items'],y['inputs']['items']['args']['items']):
    if xx.get('kind')=='tensor':assert torch.equal(payload(xx,'fixed'),payload(yy,'fixed'));tensor_count+=2
 assert summaries['fixed/native_public']['entry']['loss']==summaries['fixed/native_worker']['entry']['loss']
 dump(REV/'raw_trace_kernel_mapping.json',maps)
 dump(REV/'profiler_tree_association_discrepancies.json',association_discrepancies)
 proof=dict(status='verified',native_first_difference=dict(index=first,module='init_conv',operator='aten.convolution.default',all_input_values_and_layout_equal=True,
  maximum_output_absolute_difference=float(delta.max()),different_elements=int((delta!=0).sum()),elements=delta.numel(),
  actual_public_kernels=map_native[0][first]['kernels'],actual_worker_kernels=map_native[1][first]['kernels']),
  raw_cuda_markers_independently_correlated=140,fixed_actual_convolution_sequences_equal=35,
  fixed_all_aten_outputs_and_inputs_equal=779,fixed_raw_convolution_payloads_verified=tensor_count,
  all_four_warmup_measured_outputs_exact=True,full_first_batch_forwards=8,preflight_single_convolution=1,backward_calls=0,optimizer_updates=0,
  cudnn_debug_logs_not_generated=missing_logs,kernel_evidence='Actual Kineto/CUPTI GPU kernel activity; authoritative raw Chrome runtime correlation independently verified',
  tree_association_discrepancies=len(association_discrepancies),
  limited_conclusion='Current first-batch path-selection contribution established; whole training reproducibility, historical DS gap and downstream improvement not evaluated')
 dump(REV/'verified_findings.json',proof)
 # Correct auto-report claim about optional debug logs; CUDA activity evidence is complete.
 old_report=RUN/'source/auto_report_before_review.md';shutil.copy2(REPORT,old_report)
 old_manifest=RUN/'source/computation_manifest_before_review.json';shutil.copy2(RUN/'run-manifest.json',old_manifest)
 text=REPORT.read_text();text=text.replace('原始Chrome traces、cuDNN日志、每个卷积的完整输入/输出和内核名保存。','原始Chrome traces、每个卷积的完整输入/输出和内核名已保存。cuDNN debug环境变量已设置，但本环境未生成独立cuDNN日志；实际内核证据来自Kineto/CUPTI活动记录，未将缺失日志冒充已保存。')
 text+='\n## 独立复核\n\n[Observed] 从四份原始Chrome trace重新按CPU卷积标记→CUDA runtime correlation→GPU kernel活动关联，140个卷积标记均通过runtime correlation关联到实际GPU启动。Profiler上层事件树存在重复/跨事件关联，初版树派生列表仅作历史保留；最终内核判定采用逐个真实GPU启动，差异清单单独保存。原自动模式首个卷积的输入/权重/偏置原始张量逐位相同，输出217433/393216元素不同，maxabs4.76837158203125e-7。固定模式35个实际卷积内核序列、全部779个已跟踪操作输入/输出及保存的卷积原始张量均逐位一致。全部四个进程warmup与测量输出也逐位一致。\n\n[Observed] 当前首批分叉的计算路径贡献得到直接支持；不将其外推为历史DS差距已被唯一解释或完整训练已修复。本轮停止，默认值未变。\n'
 REPORT.write_text(text)
 m['review_status']='independent_raw_kernel_and_payload_verification_passed'
 m['authoritative_kernel_mapping']=str((REV/'raw_trace_kernel_mapping.json').relative_to(ROOT))
 m['profiler_tree_association_discrepancy_count']=len(association_discrepancies)
 m['report_correction']=dict(reason='Optional cuDNN debug logs absent; actual measured kernel activity complete',original_report_copy=str(old_report.relative_to(ROOT)),original_report_sha256=sha(old_report))
 m['outputs_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [*OUT.rglob('*'),REPORT,old_report,old_manifest] if p.is_file()}
 dump(RUN/'run-manifest.json',m)
 paths=[RUN/'run-manifest.json',Path(__file__),old_report,old_manifest,*list((RUN/'review-attempt1').glob('*'))]
 paths += [OUT/mode/arm/name for mode in ['native','fixed'] for arm in ['native_public','native_worker'] for name in ['cuda_trace.json','actual_conv_kernels.json','operators.json','summary.json','profile_summary.json']]
 dump(RUN/'postreview-manifest.json',dict(status='complete',scope_class='diagnostic',role='supporting',scope_contract=m['scope_contract'],started_at=started,completed_at=now(),
  command=[sys.executable,str(Path(__file__))],action='CPU-only independent raw CUDA correlation / exact tensor verification and report correction; zero extra forwards',
  inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [REV/'raw_trace_kernel_mapping.json',REV/'profiler_tree_association_discrepancies.json',REV/'verified_findings.json',REPORT]}))
 body='- [Observed] 原自动模式首个不同点init_conv：输入/权重/偏置/布局相同，实际CUDA核分别为sm80_xmma...与sm86_xmma...；输出217433/393216元素不同，maxabs4.76837e-7。四份原始Chrome记录经CPU标记→runtime correlation→GPU活动独立关联140个卷积全部通过。\n- [Observed] 固定卷积选择（benchmark=False/deterministic=True，全局strict=False以保留median）后，两入口35个卷积实际内核序列、全部779个操作输入/输出及原始卷积张量逐位一致；4进程warmup/测量输出分别一致。直接支持卷积自动路径选择贡献当前首批分叉。8首批前向+1预检卷积，0反向/0更新。\n- [Hypothesis] 不等于完整训练已可重复或原历史DS差距已被唯一解释/修复。cuDNN独立debug日志未生成，已纠正自动报告；实际CUDA内核记录完整，不以日志缺失否定已测量证据。本轮停止，无默认规则变更。\n'+f'- Report `{REPORT.relative_to(ROOT)}`；snapshot `{RUN.relative_to(ROOT)}/postreview-manifest.json`。diagnostic/supporting，严格main/A/B保持、C推迟；FD003/4、125分析暂停。\n'
 p=ROOT/'reproduction-state.md';h,t=p.read_text().split('\n',1);p.write_text(h+'\n\n## Latest checkpoint: 2026-11-03 actual CUDA kernel divergence confirmed; fixed paths agree\n\n'+body+'\n'+t)
 with (ROOT/'experiment-ledger.md').open('a') as f:f.write('\n## `'+RUN.name+'` independently reviewed\n\n'+body)
 print(json.dumps(proof,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
