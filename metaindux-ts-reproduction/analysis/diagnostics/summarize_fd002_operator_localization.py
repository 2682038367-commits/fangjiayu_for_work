"""Seal and independently verify first-batch operator localization artifacts."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,importlib.util,json,shutil,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/diagnostic-fd002-first-batch-op-localization-20261103'
OUT=ROOT/'results/diagnostics/fd002_first_batch_op_localization_20261103'
REPORT=ROOT/'docs/conclusions/final/fd002_first_batch_op_localization_20261103_report.md'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not (RUN/'postreview-manifest.json').exists(),'No overwrite'
 spec=importlib.util.spec_from_file_location('fixed_trace','scripts/localize_fd002_first_batch_ops.py');api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
 parent=api.check();assert parent['status']=='localized_pending_confirmation'
 mainpaths=[OUT/'native_public/operators.json',OUT/'native_worker/operators.json',OUT/'capture/native_public/operators.json']
 arrays=[json.loads(p.read_text()) for p in mainpaths];a,b,c=arrays
 assert len(a)==len(b)==len(c)==779
 def compare(x,y):
  assert all(aa['operator']==bb['operator'] and aa['module']==bb['module'] for aa,bb in zip(x,y))
  i=next(i for i,(aa,bb) in enumerate(zip(x,y)) if api.clean(aa['output'])!=api.clean(bb['output']))
  assert api.clean(x[i]['inputs'])==api.clean(y[i]['inputs'])
  assert api.clean(x[i]['inputs'],True)==api.clean(y[i]['inputs'],True)
  assert x[i]['operator']=='aten.convolution.default'
  return dict(index_zero_based=i,tracked_operation_one_based=i+1,prior_equal_output_count=i,operator=x[i]['operator'],module=x[i]['module'],source=x[i]['source'],
   same_input_values=True,same_input_layout=True,tensor_shapes=[v['shape'] for v in x[i]['inputs']['items']['args']['items'] if v['kind']=='tensor'],
   output_hashes=[x[i]['output']['sha256'],y[i]['output']['sha256']])
 ab=compare(a,b);ac=compare(a,c);bc=compare(b,c)
 assert ab['index_zero_based']==721 and ac['index_zero_based']==bc['index_zero_based']==310
 assert ab['module']=='minusTrend.fn.seasonal.season' and ac['module']=='mid_block1.block1.proj'
 # Independently verify every stored tensor descriptor against its raw artifact.
 verified=set()
 def verify(z,base):
  if isinstance(z,dict):
   if z.get('kind')=='tensor' and z.get('artifact'):
    tag=z['artifact'];key=str(base/tag)
    if key not in verified:
     t=torch.load(base/f'{tag}.pth',map_location='cpu',weights_only=True)
     assert list(t.shape)==z['shape'] and str(t.dtype)==z['dtype']
     assert hashlib.sha256(t.numpy().tobytes()).hexdigest()==z['sha256']
     verified.add(key)
   for v in z.values():verify(v,base)
  elif isinstance(z,list):
   for v in z:verify(v,base)
 for arr,base in [(a,OUT/'tensors'),(b,OUT/'tensors'),(c,OUT/'capture/tensors')]:verify(arr,base)
 summaries=[json.loads((p/'summary.json').read_text()) for p in [OUT/'native_public',OUT/'native_worker',OUT/'capture/native_public']]
 assert all(s['forward_calls']==1 and s['backward_calls']==s['optimizer_updates']==0 and s['canonical_initial_and_question_exact'] for s in summaries)
 assert len({s['pre_forward_rng_sha256'] for s in summaries})==1
 probe=OUT/'local_probe';stats={};raw={}
 for mode in ['native','local_deterministic']:
  tensors=[torch.load(probe/f'process{i}_{mode}.pth',map_location='cpu',weights_only=True) for i in [1,2]]
  assert torch.equal(*tensors)
  stats[mode]=dict(cross_process_exact_equal=True,output_sha256=hashlib.sha256(tensors[0].numpy().tobytes()).hexdigest());raw[mode]=tensors[0]
 delta=abs(raw['native'].double()-raw['local_deterministic'].double())
 stats['between_modes']=dict(max_abs=float(delta.max()),different_elements=int((delta!=0).sum()),elements=delta.numel(),
  relative_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(raw['native'].double())),
  interpretation='Each mode repeated exactly in this limited local probe. Mode difference is not proof that switching modes cures whole-network divergence.')
 assert stats['between_modes']['max_abs']==4.76837158203125e-7
 failure=json.loads((RUN/'confirmation-manifest.json').read_text());assert failure['status']=='failed_input_parity_guard'
 local=json.loads((RUN/'local-probe-manifest.json').read_text());assert local['status']=='complete'
 # Preserve immutable exact parent versions used by each subsequent diagnostic.
 parent_version=RUN/'source/localization_manifest_before_confirmation.json';shutil.copy2(RUN/'run-manifest.json',parent_version)
 confirmation_version=RUN/'source/confirmation_manifest_used_for_local_probe.json';shutil.copy2(RUN/'confirmation-manifest.json',confirmation_version)
 for name in ['confirmation-manifest.json','local-probe-manifest.json']:
  p=RUN/name;m=json.loads(p.read_text());h=m['inputs_sha256'].pop(str((RUN/'run-manifest.json').relative_to(ROOT)))
  assert h==sha(parent_version);m['inputs_sha256'][str(parent_version.relative_to(ROOT))]=h
  if name=='local-probe-manifest.json':
   h=m['inputs_sha256'].pop(str((RUN/'confirmation-manifest.json').relative_to(ROOT)))
   assert h==sha(confirmation_version);m['inputs_sha256'][str(confirmation_version.relative_to(ROOT))]=h
  m['parent_version_note']='Immutable exact parent versions preserved before final report sealing; execution parameters/source unchanged.'
  dump(p,m)
 for name in ['confirmation-manifest.json','local-probe-manifest.json']:
  m=json.loads((RUN/name).read_text())
  for p,h in m['inputs_sha256'].items():assert sha(ROOT/p)==h,p
  for p,h in m.get('outputs_sha256',{}).items():assert sha(ROOT/p)==h,p
 result=dict(status='localized_with_limited_local_confirmation',primary_pair=ab,supplementary_capture_vs_public=ac,supplementary_capture_vs_worker=bc,
  raw_tensors_verified=len(verified),full_first_batch_forwards=3,standalone_convolution_calls=4,backward_calls=0,optimizer_updates=0,
  capture_parity_failure_retained=True,local_probe_input_matches_original_primary_activations=False,local_probe_stats=stats,
  evidence_limits=['Earliest differing layer varies across the current process replays; identify convolution execution, not one universal faulty layer.',
   'No kernel-name or exact algorithm selection measured, so cuDNN auto-benchmark explanation remains an inference.',
   'Prefix tracing synchronizes GPU and can alter kernel tuning/timings.',
   'Supplementary local probe uses explicitly frozen third-capture activation, not the original paired op721 input.',
   'Full-network strict mode remained blocked in prior median diagnostic; no whole-training fix or historical DS causal attribution established.'])
 dump(OUT/'verified_findings.json',result)
 lines=['# FD002首批最早不同算子定位（2026-11-03）','',
 '[Observed] **已定位到CUDA卷积的数值执行：相同输入、权重、偏置、形状和布局，卷积输出首次不同。最早出现在哪一层会随重放进程变化，不能称为唯一固定异常层。** 本轮3次首批前向、4次独立卷积调用，0反向、0参数更新；没有完整训练、生成或下游评价。','',
 '|对照|最早不同模块|ATen算子|之前已跟踪输出逐位相同的数量|', '|---|---|---|---|',
 '|公开入口 vs 审计入口|`minusTrend.fn.seasonal.season`|`aten.convolution.default`|721|',
 '|补充捕获 vs 前述任一入口|`mid_block1.block1.proj`|`aten.convolution.default`|310|','',
 '## 主对照：季节性分支1×1卷积','',
 '[Observed] 全部779个已跟踪操作中，第722个操作首次不同。模块`minusTrend.fn.seasonal.season`是网络末尾季节性/周期分支的1×1卷积：输入`[256,32,48]`，权重`[16,32,1]`，偏置`[16]`，输出`[256,16,48]`。输入值、权重、偏置、stride/padding/dilation/groups、内存步长与偏移全部一致。算子执行之前未发现已跟踪的值差异。',
 '', '[Code] 定义：`upstream/DiffusionFreeGuidence/Attetnion.py:187`；调用：`:197`。这是实际执行卷积的首个差异点，而不是把某个接收到上游偏差信号的后续层误判为起点。具体CUDA/cuDNN内核名称尚未测量，不冒充已经识别。','',
 '## 补充捕获：分叉可以更早出现','',
 '[Observed] 为保存主对照算子输入/输出，额外进行一次首批捕获。初始全状态、第一批数据/随机题目仍与冻结记录一致，但这次最早在第311个操作、`mid_block1.block1.proj`卷积分叉，输入`[256,64,24]`、权重`[64,64,3]`，对应`upstream/DiffusionFreeGuidence/Unet1D_fre.py:136`的`F.conv1d`。这个卷积的输入和标准化后权重也逐位一致。',
 '', '[Observed] 因为更早的卷积已分叉，这次后续第722操作的输入不再等于主对照中的输入。“原样回放”逐元素保护检查正确停止，原失败日志/快照保留。未放宽相等要求，也未继续挑选重放进程。两个中位数操作位于已跟踪索引498/663；主对照它们输出相同，补充对照在它们之前已分叉。因此前次median不支持严格模式的报错不能当作本次首个分叉算子。','',
 '## 局部卷积检查（独立标注）','',
 '[Observed] 使用补充捕获的季节性卷积输入，另行冻结为新的局部输入，两独立进程各执行一次普通模式与一次确定性模式，未再进行整网前向。它不是主对照第722操作的精确激活回放。','',
 '|局部模式|两进程结果|','|---|---|','|benchmark=True / deterministic=False|逐位相同|','|benchmark=False / deterministic=True / strict deterministic algorithms|逐位相同|',
 f'|两种模式之间|最大绝对差{stats["between_modes"]["max_abs"]:.12g}；不同元素{stats["between_modes"]["different_elements"]}/{stats["between_modes"]["elements"]}|','',
 '[Observed] 这证实卷积计算模式可以改变浮点舍入结果；但本次局部普通模式也重复一致，**不能宣称关闭自动选算法已消除原整网分叉**。整网严格确定性仍受前次median算子限制；本轮未替换算子或修改训练默认值。','',
 '## 结论与停止点','',
 '[Observed] 当前相同随机题目下的前向分叉至少可发生于两个卷积实例。将后续核查重点收缩到卷积算法/计算精度选择，比继续把差异归因于样本顺序或条件抽样更有直接证据。',
 '', '[Hypothesis] cuDNN自动选算法、执行上下文或计算精度路径差异是候选解释，尚未直接记录所选内核，所以不唯一归因于benchmark，也不能推断原第67轮DS差距全由它造成。追踪同步本身会改变时序。',
 '', '[Hypothesis] 若继续，应单独检验卷积算法选择规则对首批前向重复性的影响，并记录实际内核；无须重训70轮。该后续未启动，不自动改变模型/训练/选模规则。',
 '', '[Observed] 本诊断→无论文同设计指标→论文数值差距不适用→当前入口有限配对，不是生成种子方差统计→严格复现结论保持。FD003/4及125组分析仍暂停，seed43下游门槛与seed13结果保持。','',
 f'快照：`{RUN.relative_to(ROOT)}/run-manifest.json`；失败保护检查：`{RUN.relative_to(ROOT)}/confirmation-manifest.json`；独立局部检查：`{RUN.relative_to(ROOT)}/local-probe-manifest.json`；原始算子记录/核验：`{OUT.relative_to(ROOT)}`。']
 REPORT.write_text('\n'.join(lines)+'\n')
 # All child inputs now refer to immutable parent versions, permitting root sealing.
 parent.update(status='complete',completed_at=now(),outcome='localized_with_limited_local_confirmation',report=str(REPORT.relative_to(ROOT)),
  actual_budget=dict(full_first_batch_forwards=3,standalone_operator_calls=4,backward_calls=0,optimizer_updates=0),
  output_sha256={str(p.relative_to(ROOT)):sha(p) for p in [*OUT.rglob('*'),REPORT,RUN/'confirmation-manifest.json',RUN/'local-probe-manifest.json'] if p.is_file()})
 dump(RUN/'run-manifest.json',parent)
 inputs=[RUN/'run-manifest.json',RUN/'confirmation-manifest.json',RUN/'local-probe-manifest.json',*mainpaths,Path(__file__),
 ROOT/'analysis/diagnostics/confirm_fd002_first_batch_operator.py',ROOT/'analysis/diagnostics/probe_fd002_local_convolution.py']
 dump(RUN/'postreview-manifest.json',dict(status='complete',scope_class='diagnostic',role='supporting',completed_at=now(),command=[sys.executable,str(Path(__file__))],
  action='CPU-only independent trace comparison and stored tensor hash verification; no additional forward/updates',
  inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'verified_findings.json',REPORT]}))
 body='- [Observed] 主对照779个ATen操作，第722个（索引721）首次输出不同：minusTrend.fn.seasonal.season的1×1卷积；前721个输出一致，该卷积输入/权重/偏置/参数/布局完全一致。补充捕获最早分叉前移至第311个mid_block1.block1.proj卷积，输入/标准化权重也一致，故非唯一固定异常层。\n- [Observed] 补充捕获导致后续激活漂移，精确回放保护失败并保留；没有放宽或挑选新进程。另以明确独立的第三次捕获输入做4次局部卷积调用，普通/确定性各两次内部一致，两模式之间maxdiff4.76837e-7；不宣称消除整网分叉。共3首批前向/4局部调用、0反向/0更新。\n- [Hypothesis] 可把当前前向分叉收缩到卷积计算路径；尚未记录实际CUDA/cuDNN内核，不能唯一归因benchmark或解释全部历史DS差距。定位完成，未启动后续或修改默认值。\n'+f'- Report `{REPORT.relative_to(ROOT)}`；snapshot `{RUN.relative_to(ROOT)}/postreview-manifest.json`；diagnostic/supporting，严格main/A/B保持、C推迟，FD003/4及125暂停。\n'
 p=ROOT/'reproduction-state.md';h,t=p.read_text().split('\n',1);p.write_text(h+'\n\n## Latest checkpoint: 2026-11-03 FD002 first differing operators localized\n\n'+body+'\n'+t)
 with (ROOT/'experiment-ledger.md').open('a') as f:f.write('\n## `'+RUN.name+'` localization complete and reviewed\n\n'+body)
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
