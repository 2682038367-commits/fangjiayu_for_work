"""Independently seal first-epoch traces and report blocked strict control."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/diagnostic-fd002-seed43-first-epoch-trace-20261103'
OUT=ROOT/'results/diagnostics/fd002_seed43_first_epoch_trace_20261103'
REVIEW=OUT/'review'
REPORT=ROOT/'docs/conclusions/final/fd002_seed43_first_epoch_trace_20261103_report.md'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def digest(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 assert not (RUN/'postreview-manifest.json').exists(),'No overwrite'
 REVIEW.mkdir(exist_ok=True);started=now();m=json.loads((RUN/'run-manifest.json').read_text())
 assert m['status']=='failed' and m['completed_arms']==['native_public','native_worker'] and m['current_arm']=='strict_public'
 for name,h in m['inputs_sha256'].items():assert sha(ROOT/name)==h,name
 arms=['native_public','native_worker'];traces=[];initial=[];summaries=[]
 paths=[RUN/'run-manifest.json',OUT/'native_comparison.json',Path(__file__)]
 for arm in arms:
  p=OUT/arm;s=json.loads((p/'summary.json').read_text());rows=json.loads((p/'trace.json').read_text());summaries.append(s);traces.append(rows)
  assert s['completed_epoch']==1 and s['actual_updates']==163 and s['sample_or_evaluator_calls']==0
  assert s['backend_flags']['cudnn_benchmark'] and not s['backend_flags']['cudnn_deterministic'] and not s['backend_flags']['deterministic_algorithms']
  assert len(list((p/'states').glob('*.pth')))==len(list((p/'tapes').glob('*.pth')))==163
  initial.append(torch.load(p/'initial_state.pth',map_location='cpu',weights_only=True));assert digest(initial[-1])==s['initial_state_sha256']
  for i,row in enumerate(rows,1):
   z=torch.load(p/f'states/{i:04d}.pth',map_location='cpu',weights_only=True)
   assert row['update']==i and row['learning_rate']==.002
   for key,column in [('model_after','weights_after_sha256'),('grad_preclip','grad_preclip_sha256'),('grad_postclip','grad_postclip_sha256')]:assert digest(z[key])==row[column]
  assert abs(np.mean([r['loss'] for r in rows])-s['mean_first_epoch_loss'])<1e-14
  paths+=list(q for q in p.rglob('*') if q.is_file())
 assert all(torch.equal(initial[0][k],initial[1][k]) for k in initial[0])
 for i in range(1,164):
  ta,tb=[torch.load(OUT/a/f'tapes/{i:04d}.pth',map_location='cpu',weights_only=False) for a in arms]
  assert np.array_equal(ta['indices'],tb['indices']) and ta['numpy_draw']==tb['numpy_draw']
  for k in ['timestep','noise','condition_mask']:assert torch.equal(ta[k],tb[k])
  for k in ['data_sha256','labels_sha256','noisy_input_sha256','actual_condition_sha256']:assert traces[0][i-1][k]==traces[1][i-1][k]
 assert traces[0][0]['weights_before_sha256']==traces[1][0]['weights_before_sha256']==summaries[0]['initial_state_sha256']
 first=[torch.load(OUT/a/'states/0001.pth',map_location='cpu',weights_only=True) for a in arms]
 rank=[]
 for k,x in first[0]['model_after'].items():
  y=first[1]['model_after'][k];v=(x-y).abs();i=int(v.argmax());delta=float(v.flatten()[i])
  if delta:
   rank.append(dict(parameter=k,flat_index=i,max_absolute_weight_difference=delta,initial_value=float(initial[0][k].flatten()[i]),
    values=[float(x.flatten()[i]),float(y.flatten()[i])],gradient_before_clip=[float(s['grad_preclip'][k].flatten()[i]) for s in first],gradient_after_clip=[float(s['grad_postclip'][k].flatten()[i]) for s in first]))
 rank.sort(key=lambda x:x['max_absolute_weight_difference'],reverse=True)
 loss=[r[0]['loss'] for r in traces];assert loss[0]!=loss[1]
 # Expected stop: strict determinism rejects CUDA median before any optimizer update.
 strictlog=OUT/'strict_public/console.log';error=strictlog.read_text()
 assert 'median CUDA with indices output does not have a deterministic implementation' in error
 assert 'Attetnion_temp.py", line 142' in error
 assert not list((OUT/'strict_public/states').glob('*.pth')) and not (OUT/'strict_public/progress.json').exists()
 assert not (OUT/'strict_worker/console.log').exists()
 strictinit=OUT/'strict_public/initial_state.pth'
 assert digest(torch.load(strictinit,map_location='cpu',weights_only=True))==summaries[0]['initial_state_sha256']
 paths+=[strictlog,strictinit]
 previous=json.loads((OUT/'native_comparison.json').read_text());assert previous['all163_questions_equal'] and previous['first_divergence']==1
 result=dict(status='complete_with_deterministic_control_blocked',native_completed_trajectories=2,native_updates_per_trajectory=163,total_optimizer_updates=326,
  strict_control_attempts=1,strict_actual_updates=0,strict_other_entry_not_attempted=True,second_epoch_started=False,
  all_initial_parameters_equal=True,all163_actual_questions_and_noisy_inputs_equal=True,
  first_numerical_divergence_batch=1,first_batch_losses=loss,first_batch_loss_absolute_difference=abs(loss[1]-loss[0]),
  first_epoch_mean_losses=[s['mean_first_epoch_loss'] for s in summaries],
  first_batch_maximum_weight_divergences=rank[:3],strict_blocking_operator='CUDA median(dim=1,keepdim=True) with indices',
  blocking_source='upstream/DiffusionFreeGuidence/Attetnion_temp.py:142',
  causal_limits=['Same questions establish numeric execution as locus of current difference; no complete deterministic comparison was possible.',
   'Unsupported median operator is not proof it caused observed divergence. Other CUDA kernels/algorithm choices remain possible.',
   'Current reconstructed entry points and synchronization from tracing differ from unavailable historical source and timings; cannot uniquely attribute historical DS gap.'])
 dump(REVIEW/'verified_findings.json',result)
 top=rank[0];g0,g1=top['gradient_before_clip'];a0,a1=top['values']
 lines=['# FD002 seed43第一轮追踪结果（2026-11-03）','',
 '[Observed] **两条普通模式第一轮追踪已完成；随机输入全部一致，但第一批数值计算就分叉。确定性对照因当前CUDA算子不支持而停止。** 本轮共326次参数更新，每条仅第1轮163次；第三条严格模式在第一次更新前停止，第四条未启动。没有第2轮、70轮重训、生成或预测器评价。','',
 '|核对项|结果|','|---|---|',
 '|两个入口|当前公开MainCondition入口 vs 当前审计worker入口；相同环境|',
 '|初始化|全部参数逐元素相同，初始全状态哈希一致|',
 '|全部163批的样本/标签|逐元素一致；实际样本顺序与输入张量核验|',
 '|扩散步、完整高斯噪声、条件丢弃掩码、数值置零抽样|全部163批逐元素一致，完整随机题目已保存|',
 '|送入网络的含噪信号与实际RUL条件|全部163批一致|',
 '|首次不同位置|第1批模型输出；同批损失、裁剪前后梯度、Adam状态及更新后权重随后不同|',
 f'|第1批损失|{loss[0]:.12f} vs {loss[1]:.12f}；绝对差{abs(loss[1]-loss[0]):.6g}|',
 f'|第1轮平均损失|{summaries[0]["mean_first_epoch_loss"]:.10f} vs {summaries[1]["mean_first_epoch_loss"]:.10f}|','',
 '[Observed] 这次可以排除“样本顺序、扩散噪声或条件随机丢弃不同”作为本次首轮分叉原因。差异在选模和采样之前，不能由最低损失选到了不同轮次解释。','',
 '## 为什么很小的计算差异可以影响训练','',
 f'[Observed] 首批最大更新差异出现在 `{top["parameter"]}` 的扁平位置{top["flat_index"]}。同一初始值{top["initial_value"]:.12f}，裁剪前梯度分别为{g0:.12g}和{g1:.12g}，发生符号翻转；更新后分别为{a0:.12f}和{a1:.12f}，相差{top["max_absolute_weight_difference"]:.8f}。',
 '', '[Code] Adam第一步的归一化更新对接近零的梯度符号变化敏感。[Hypothesis] 这提供了微小数值差异随后放大成不同训练轨迹的具体路径；不等于证明原历史训练的全部生成质量差距由它造成。','',
 '## 确定性对照为何没有完成','',
 '[Observed] 已启用cudnn.benchmark=False、cudnn.deterministic=True、torch.use_deterministic_algorithms(True)，并准备复用完全相同随机题目。首批在 `Attetnion_temp.py:142` 的 `flat_energy.median(dim=1, keepdim=True)[0]` 报错：`median CUDA with indices output does not have a deterministic implementation`。严格模式没有完成任何一次参数更新。按预先停止规则停止，未改成warn_only、未替换算子或继续下一入口。','',
 '[Observed] 该错误说明当前实现不能仅靠打开开关就获得严格确定性保证。**它不证明median就是这次前向数值差异的来源**，尤其不能把算子不支持严格模式等同于它已导致观测到的误差。','',
 '## 本轮边界与下一步','',
 '[Observed] 两入口均使用当前留存源码及同一线程/CUBLAS/TF环境。原9月14日初始全状态、随机轨迹和精确源码未留存；追踪中的GPU同步也会改变自动选算法时序。因此本轮定位了当前分叉发生在计算阶段，尚未完成“打开确定性后分叉消失”的验证，也不能唯一解释原第67轮DS0.434→0.178。','',
 '[Hypothesis] 下一步应只对首批做逐层前向定位，找出第一个不同的算子，再决定是否需要局部确定性兼容实现；暂不再次做70轮训练或扩展数据集。该后续未自动启动。','',
 '[Observed] 原比较结果/默认规则保持，seed43下游仍按门槛停止；FD003/4及125组分析暂停。本诊断→无论文同设计指标→论文差距不适用→不是新增生成种子统计→严格复现结论保持。','',
 f'主追踪快照（保留严格模式停止记录）：`{RUN.relative_to(ROOT)}/run-manifest.json`；独立复核快照：`{RUN.relative_to(ROOT)}/postreview-manifest.json`；完整逐批文件/比较：`{OUT.relative_to(ROOT)}`。']
 REPORT.write_text('\n'.join(lines)+'\n')
 dump(RUN/'postreview-manifest.json',dict(run_id=RUN.name+'/postreview',scope_class='diagnostic',role='supporting',scope_contract=m['scope_contract'],
  status='complete_with_deterministic_control_blocked',started_at=started,completed_at=now(),command=[sys.executable,str(Path(__file__))],
  changes='CPU-only independent raw trace verification and report; zero new forward/backward/updates',
  inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(paths)},
  outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [REVIEW/'verified_findings.json',REPORT]},
  report=str(REPORT.relative_to(ROOT))))
 body='- [Observed] 两个当前入口各完成第一轮163步，共326更新；初始全参数及全部163批实际样本/标签/扩散步/完整噪声/条件掩码/数值置零抽样一致，但第1批输出、损失、梯度、Adam状态与权重即不同。首批损失差1.07288e-6；近零梯度符号翻转对应最大权重更新差0.00381573。\n- [Observed] 严格确定性对照首批在CUDA median(dim=1,keepdim=True)不支持处停止，0参数更新，另一严格入口未启动；不放宽/替换算子。原run失败记录保留，独立复核完成且全原始题目/梯度/权重已核验。没有第2轮或新生成/评价。\n- [Hypothesis] 已定位当前分叉属于相同随机题目下数值执行差异；median不支持不等于它是首个差异算子。尚不能证明严格模式可消除分叉或唯一解释历史DS差距。建议只定位首批第一个不同算子，未自动启动。\n'+f'- Report `{REPORT.relative_to(ROOT)}`；snapshot `{RUN.relative_to(ROOT)}/postreview-manifest.json`；diagnostic/supporting，原默认/seed43下游停止/seed13结果保持；严格main/A/B保持，C推迟；FD003/4和125分析暂停。\n'
 p=ROOT/'reproduction-state.md';h,t=p.read_text().split('\n',1);p.write_text(h+'\n\n## Latest checkpoint: 2026-11-03 FD002 first-epoch trace reviewed; strict control blocked\n\n'+body+'\n'+t)
 with (ROOT/'experiment-ledger.md').open('a') as f:f.write('\n## `'+RUN.name+'` independently reviewed, strict control blocked\n\n'+body)
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
