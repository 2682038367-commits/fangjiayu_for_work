"""Audit existing FD002 seed43 rerun records; CPU state/RNG reconstruction only."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/diagnostic-fd002-seed43-rerun-audit-20261103'
OUT=ROOT/'results/diagnostics/fd002_seed43_rerun_audit_20261103'
PREV=ROOT/'runs/diagnostic-fd002-final70-two-seeds-20261102'
DATA=ROOT/'results/diagnostics/fd002_final70_two_seeds_20261102'
REPORT=ROOT/'docs/conclusions/final/fd002_seed43_rerun_audit_20261103_report.md'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def tensor_hash(state):
 h=hashlib.sha256()
 for k,t in sorted(state.items()):h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def wandb_config(path):
 from wandb.sdk.internal.datastore import DataStore
 from wandb.proto import wandb_internal_pb2
 store=DataStore();store.open_for_scan(str(path))
 while True:
  value=store.scan_data()
  if value is None:raise ValueError('No run config')
  rec=wandb_internal_pb2.Record();rec.ParseFromString(value)
  if rec.HasField('run'):return {i.key:json.loads(i.value_json) for i in rec.run.config.update}

def main():
 assert not (RUN/'run-manifest.json').exists(),'No overwrite'
 start=now();paths=[Path(__file__),PREV/'run-manifest.json',PREV/'config.json',PREV/'frozen_inputs.npz',ROOT/'logs/fd002_paper.log',
  DATA/'logs/train_seed43_training.log',DATA/'seed43/training_audit.json',DATA/'seed43/training_loss.csv',
  DATA/'seed43/training_epoch_orders.npz',DATA/'seed43/training_batch_zero_draws.csv',ROOT/'SOURCE_SNAPSHOT.md']
 m=json.loads((PREV/'run-manifest.json').read_text())
 for name,want in m['inputs_sha256'].items():assert sha(ROOT/name)==want,name
 olddir=ROOT/'upstream/wandb/offline-run-20260914_172659-elnzt2mk'
 newdir=next(p for p in (DATA/'seed43/wandb').glob('offline-run-*') if p.is_dir())
 oldraw=next(olddir.glob('*.wandb'));newraw=next(newdir.glob('*.wandb'))
 oldconf,newconf=wandb_config(oldraw),wandb_config(newraw)
 dump(OUT/'original_resolved_config.json',oldconf);dump(OUT/'rerun_resolved_config.json',newconf)
 changes={k:[oldconf[k],newconf[k]] for k in sorted(oldconf.keys()&newconf.keys()) if oldconf[k]!=newconf[k]}
 assert set(changes)=={'model_path','syndata_path','metrics_path','loss_history_path'}
 added={k:newconf[k] for k in sorted(newconf.keys()-oldconf.keys())}
 assert added['batch_label_zero_prob']==.1 and added['lr_schedule']=='warmup_cosine' and added['frequency_threshold']==-1
 paths += [oldraw,newraw,olddir/'files/wandb-metadata.json',olddir/'files/requirements.txt',PREV/'environment-freeze.txt']
 meta=json.loads((ROOT/'results/runs/FD002_w48_seed43_metadata.json').read_text())
 offline=json.loads((olddir/'files/wandb-metadata.json').read_text());assert meta['command'][2:]==offline['args']
 paths += [ROOT/'results/runs/FD002_w48_seed43_metadata.json',ROOT/'results/runs/FD002_w48_seed43_metrics.json',ROOT/'logs/FD002_w48_seed43_training_loss.csv']
 def req(p):return {a.split('==')[0].lower().replace('_','-'):a.split('==')[1] for a in p.read_text().splitlines() if '==' in a}
 reqold,reqnew=req(olddir/'files/requirements.txt'),req(PREV/'environment-freeze.txt')
 package_changes={k:[reqold[k],reqnew[k]] for k in reqold.keys()&reqnew.keys() if reqold[k]!=reqnew[k]}
 assert not package_changes
 assert meta['gpu']==m['gpu'] and meta['nvidia_driver']==m['driver'] and meta['torch']==m['environment']['torch']
 h0=np.loadtxt(ROOT/'logs/FD002_w48_seed43_training_loss.csv',delimiter=',',skiprows=1)
 h1=np.loadtxt(DATA/'seed43/training_loss.csv',delimiter=',',skiprows=1)
 assert h0.shape==h1.shape==(70,3) and np.array_equal(h0[:,0],h1[:,0]) and np.array_equal(h0[:,2],h1[:,2])
 rows=[dict(epoch=int(a[0])+1,original_loss=float(a[1]),rerun_loss=float(b[1]),delta=float(b[1]-a[1]),lr=float(a[2])) for a,b in zip(h0,h1)]
 with (OUT/'epoch_loss_comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 text=(ROOT/'logs/fd002_paper.log').read_text();text=text[text.find('FD002_w48_seed43'):]
 def rounded(text):
  values=[]
  for line in re.split('[\r\n]',text):
   hit=re.search(r'epoch=0, loss: =([0-9.e+-]+)',line)
   if hit:
    v=hit.group(1)
    if not values or v!=values[-1]:values.append(v)
   elif values and 'epoch=1,' in line:break
  return values
 a,b=rounded(text),rounded((DATA/'logs/train_seed43_training.log').read_text())
 first=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
 assert first==28 and len(set(a[:first+1]))==first+1 and len(set(b[:first+1]))==first+1
 dump(OUT/'first_epoch_rounded_log_comparison.json',dict(first_visible_difference_batch=first+1,
  original=a[:50],rerun=b[:50],precision_caveat='Printed rounded running means; earlier bit-level divergence cannot be excluded. No adjacent duplicates in first29 batches.'))
 # Independent CPU-only construction: no forward, backward, optimizer or generated samples.
 torch.set_num_threads(2);sys.path.insert(0,str(ROOT/'upstream'))
 from DiffusionFreeGuidence.Unet1D_fre import UNet1D_fre
 torch.manual_seed(43);np.random.seed(43)
 net=UNet1D_fre(dim=32,dim_mults=(1,2),cond_drop_prob=.2,channels=14,length=48,
  frequency_mask_mode='hard_random_quantile',frequency_threshold_init=.25,frequency_mask_temperature=.1)
 initial=net.state_dict();audit=json.loads((DATA/'seed43/training_audit.json').read_text())
 assert tensor_hash(initial)==audit['initial_tensor_sha256']
 frozen={}
 for name,rule in [('original','original'),('phase','phase'),('best','best'),('final70','final70')]:
  p=ROOT/'checkpoints/FD002_w48_seed43.pth' if name=='original' else ROOT/audit['rules'][rule]['weight_path']
  frozen[name]=torch.load(p,map_location='cpu',weights_only=True)
  load=net.load_state_dict(frozen[name],strict=True);assert not load.missing_keys and not load.unexpected_keys
  assert all(torch.equal(v,net.state_dict()[k]) for k,v in frozen[name].items())
  paths.append(p)
 thresholds={k:{s:float(t[k].item()) for s,t in frozen.items()} for k in initial if k.endswith('threshold_param')}
 # initial holds shallow views into net parameters, so independently reconstruct below for equality.
 torch.manual_seed(43)
 fresh=UNet1D_fre(dim=32,dim_mults=(1,2),cond_drop_prob=.2,channels=14,length=48,
  frequency_mask_mode='hard_random_quantile',frequency_threshold_init=.25,frequency_mask_temperature=.1)
 for k,v in fresh.state_dict().items():
  if k.endswith('threshold_param'):assert all(torch.equal(v,t[k]) for t in frozen.values())
 # CPU RNG after constructor determines DataLoader base seeds and sampler generators.
 with np.load(DATA/'seed43/training_epoch_orders.npz') as z:orders=z['orders']
 matched=[]
 for e in range(70):
  torch.empty((),dtype=torch.int64).random_().item()
  order=np.asarray(list(iter(torch.utils.data.RandomSampler(range(41539)))),dtype=np.int32)
  assert np.array_equal(order,orders[e]),f'Shuffle epoch{e}'
  matched.append(hashlib.sha256(order.tobytes()).hexdigest())
 with (DATA/'seed43/training_batch_zero_draws.csv').open() as f:draws=list(csv.DictReader(f))
 vals=np.asarray([float(v['draw']) for v in draws]);assert np.array_equal(vals,np.random.RandomState(43).rand(11410))
 assert all((v<.1)==(r['numeric_batch_zeroed']=='True') for v,r in zip(vals,draws))
 masks=dict(recorded_numpy_draws=len(vals),numeric_zero_batches=int((vals<.1).sum()),
  every_draw_matches_pure_seed43=True,all70_epoch_orders_match_public_seed_flow=True,epoch_order_sha256=matched,
  caveat='Original run did not archive orders/draws/initial full state; pure seed-flow replay validates rerun observers, not historical bitwise identity.')
 dump(OUT/'rng_reconstruction.json',masks)
 # All generated comparisons share identical actual diffusion-noise schedules.
 samples=[]
 for seed,rule in [(43,'original'),(43,'phase'),(13,'original'),(13,'phase'),(13,'final70')]:
  p=DATA/f'seed{seed}/{rule}/sample_checkpoint{seed}_seed103.json';v=json.loads(p.read_text());samples.append(v['noise_audit']);paths.append(p)
 for v in samples[1:]:
  for k in ['start_cuda_rng_sha256','final_cuda_rng_sha256','legacy_schedule_fingerprint','normal_scalar_count']:assert v[k]==samples[0][k]
 # Pin available historical git state; it postdates original Sep14 training.
 diff=subprocess.check_output(['git','diff','4342389','HEAD','--','upstream/TrainCondition.py','upstream/args.py','upstream/MainCondition.py','upstream/utils.py','upstream/GaussianDiffusion.py','upstream/data/CMAPSSDataset.py','upstream/DiffusionFreeGuidence'],cwd=ROOT)
 (OUT/'available_git_source_diff.patch').write_bytes(diff)
 raw=ROOT/'upstream/data/train_FD002.txt'
 dated_manifest=ROOT/'runs/diagnostic-conditional-signal-w48-20261022/run-manifest.json'
 dated=json.loads(dated_manifest.read_text())
 dated_hash=next(v['upstream/data/train_FD002.txt'] for v in dated.values() if isinstance(v,dict) and 'upstream/data/train_FD002.txt' in v)
 assert dated_hash==sha(raw)
 paths.append(dated_manifest)
 with np.load(PREV/'frozen_inputs.npz') as z:
  label=z['label'].copy();assert z['real'].shape==(41539,48,14)
 for seed in [43,13]:
  p=ROOT/f'outputs/FD002_w48_seed{seed}.npz'
  with np.load(p) as z:assert np.array_equal(label,z['label'])
  paths.append(p)
 sys.path.insert(0,str(ROOT/'scripts'))
 # No original execution-state claims beyond records and inspected source.
 finding=dict(scope='diagnostic/supporting;existing-artifact audit',resolved_common_config_changed_only_paths=changes,
  additional_current_defaults=added,package_version_differences=package_changes,shared_dependency_packages=len(reqold.keys()&reqnew.keys()),
  environment_versions_match=True,learning_rates_all70_equal=True,full_training_updates=11410,
  rerun_initial_state_matches_independent_cpu_seed43=True,thresholds=thresholds,
  original_checkpoint_epoch=67,rerun_best_epoch=30,first_visible_rounded_loss_difference_batch=29,
  first_epoch_loss_original=float(h0[0,1]),first_epoch_loss_rerun=float(h1[0,1]),
  same67_loss_original=float(h0[66,1]),same67_loss_rerun=float(h1[66,1]),
  same67_loss_relative_change=float(h1[66,1]/h0[66,1]-1),
  all_actual_sampling_noise_fingerprints_equal=True,raw_train_matches_dated_oct22_manifest=True,
  evidence_limitations=['No original processed-input hash, full initialization, optimizer state, RNG state or per-batch order/noise/mask archive.',
   'Earliest available workspace git state postdates Sep14; cannot prove historical source identical solely from current git.',
   'Original process did not persist thread/CUBLAS/cuDNN selected-algorithm state; current worker sets OMP2/OPENBLAS1/CUBLAS4096:8 and bypasses original MainCondition TensorFlow imports.'],
  ranked_hypotheses=[dict(hypothesis='Floating-point/algorithm nonrepeatability, amplified during optimization',status='leading hypothesis, unproved',basis='Same declared configuration; early nearly matching logs then divergence; benchmark=True and deterministic not enabled.'),
   dict(hypothesis='Different random draw/order due hidden historical consumer or unarchived source/environment detail',status='not excluded',basis='Rerun pure RNG schedule matches, but historical per-batch trace absent.'),
   dict(hypothesis='Checkpoint selection or downstream evaluation caused same-epoch divergence',status='not an adequate explanation',basis='Divergence appears before first checkpoint eligibility; same67 weights evaluated with exact paired sampler and DS splits.')])
 dump(OUT/'findings.json',finding)
 # Check CPU runtime helper behavior without CUDA or numerical model calls.
 from utils import torch_seed
 prior_det=torch.backends.cudnn.deterministic;torch_seed(43)
 assert torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic==prior_det
 finding['runtime_seed_helper']=dict(benchmark=True,sets_deterministic=False,sets_deterministic_algorithms=False)
 dump(OUT/'findings.json',finding)
 lines=['# FD002 seed43重跑差异核查（2026-11-03）','',
 '[Observed] 已定位到训练早期轨迹分叉；未发现现有记录中的训练超参数、学习率、框架或驱动版本变化。**不能仅凭现有档案确定唯一根因。** 本轮仅CPU权重加载/随机顺序重建及记录核验，没有前向、反向、参数更新、新生成或判别训练。','',
 '|核对项|结果|','|---|---|',
 '|原W&B完整配置 vs 重跑完整配置|共有字段仅四个输出路径不同；新增默认值单独保存|',
 f'|依赖包/硬件|{finding["shared_dependency_packages"]}个共有包版本一致；PyTorch2.1.2+cu121、CUDA12.1、RTX4060Ti、驱动595.84一致|',
 '|数据/标签|41539×48×14；当前训练文件与10月22日冻结哈希一致（原9月14日哈希缺失）；原两份生成输出标签与冻结标签逐元素相同。原训练时处理后输入哈希未保存|',
 '|训练预算/学习率|均70轮；完整70轮LR逐元素相同；新训练直接计数11410更新；旧更新数按163批/轮重建|',
 '|随机初始化|独立CPU重建初始全状态哈希与新训练一致；两频率阈值原/重跑/各阶段完全相同。原完整初始状态未保存|',
 f'|数据顺序/数值置零|重跑70轮顺序及11410个NumPy抽样均匹配公开种子流程独立重建；置零{masks["numeric_zero_batches"]}批。原当时轨迹未保存|',
 '|模型/采样加载|所有权重strict完整加载；原与重跑的采样设置、五判别种子/划分相同，实际扩散噪声指纹完全一致|',
 '|训练差异起点|第1轮已不同；前28批日志显示值相同，第29批累计损失原0.771/新0.770（仅四舍五入显示精度）|','',
 '|轮次|原训练损失|重跑损失|','|---|---|---|',
 f'|第1轮|{h0[0,1]:.10f}|{h1[0,1]:.10f}|',
 f'|第30轮|{h0[29,1]:.10f}|{h1[29,1]:.10f}（重跑最低）|',
 f'|第67轮|{h0[66,1]:.10f}（原最低）|{h1[66,1]:.10f}|',
 f'|第70轮|{h0[69,1]:.10f}|{h1[69,1]:.10f}|','',
 '[Observed] 同为第67轮，扩散训练损失仅相差约0.174%，但相同采样/判别设置下DS原0.4341、重跑0.1780。差异不是把原第67轮与重跑第30轮混比，也早于选模发生，不能用“最低损失选到了不同轮次”单独解释。','',
 '[Code] utils.torch_seed设置NumPy/PyTorch种子及cudnn.benchmark=True，未设置cudnn.deterministic=True或torch.use_deterministic_algorithms(True)。设置同一seed并不能保证此流程逐位重现。新增batch_label_zero_prob默认仍为0.1，源码保留每批同一个NumPy抽样；当前与可用git源码差异没有改变默认训练算式。','',
 '[Hypothesis] 首位候选是CUDA算法/浮点数值差异经优化逐渐放大。它符合早期近似一致后分叉，但尚未做隔离实验，不能宣布已证实为根因。原MainCondition还导入TensorFlow/评估模块；新worker显式设置线程/CUBLAS环境。原线程/选中算法等状态缺失，不能把两次运行描述为全部执行条件已完全相同。','',
 '[Observed] 可用最早git快照晚于9月14日训练；W&B保留原完整参数/包版本但未保留逐批样本顺序、扩散步、噪声、丢弃掩码、梯度和初始全状态。因此历史同次运行的逐位恢复目前不可审计；不得仅因GPU非确定性选项存在就排除其他未留存因素。','',
 '[Hypothesis] 建议下一步最小检查：仅重放首轮，分别按原入口与审计入口，保存每批数据顺序、扩散步、噪声、条件丢弃及数值置零掩码、损失/梯度和更新后权重哈希；先核验随机题目是否相同，再在固定题目下比较确定性计算。它用于定位当前可重建入口的差异，不能冒充恢复原历史随机状态。当前未启动该检查或任何新训练。','',
 '[Observed] seed43下游对照仍按原门槛停止；seed13已完成结果保持；不扩展FD003/4，不继续125组分析，不改默认训练/选模规则。当前诊断→无论文同设计目标→论文差距不适用→历史随机轨迹无法比较→严格复现结论不变。','',
 f'运行快照：`{RUN.relative_to(ROOT)}/run-manifest.json`；原比较快照：`{PREV.relative_to(ROOT)}/run-manifest.json`；逐轮损失、配置、随机重建、原源码差异和核查结果：`{OUT.relative_to(ROOT)}`。']
 REPORT.write_text('\n'.join(lines)+'\n')
 paths += [raw,ROOT/'upstream/utils.py',ROOT/'upstream/MainCondition.py',ROOT/'upstream/TrainCondition.py',ROOT/'upstream/args.py',
 ROOT/'upstream/GaussianDiffusion.py',ROOT/'upstream/data/CMAPSSDataset.py']
 paths += list((ROOT/'upstream/DiffusionFreeGuidence').glob('*.py'))
 paths += list((RUN/'source').glob('*'))
 paths += list((RUN/'attempt1').glob('*'))
 shutil.copy2(Path(__file__),RUN/'source'/Path(__file__).name);paths.append(RUN/'source'/Path(__file__).name)
 for name,cmd in [('environment-freeze.txt',[sys.executable,'-m','pip','freeze']),('git-status.txt',['git','status','--short']),('workspace.diff',['git','diff','--binary'])]:
  p=RUN/name;p.write_bytes(subprocess.check_output(cmd,cwd=ROOT));paths.append(p)
 dump(RUN/'run-manifest.json',dict(run_id=RUN.name,scope_class='diagnostic',role='supporting',scope_contract='scope-20261103-fd002-seed43-rerun-audit',
  status='complete_with_causal_uncertainty',started_at=start,completed_at=now(),command=[sys.executable,str(Path(__file__))],
  user_authorization='核查一下: audit original versus rerun discrepancy; no new training/forward/sampling',
  changed_factor='none; archive inspection and CPU state/RNG reconstruction only',git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  python=sys.version,platform=platform.platform(),torch=str(torch.__version__),
  inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(paths)},
  outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [*OUT.glob('*'),REPORT] if p.is_file()},report=str(REPORT.relative_to(ROOT))))
 body='- [Observed] 原W&B完整配置共有字段仅输出路径不同，共有依赖包/硬件/驱动一致；70轮LR相同。CPU独立初始状态哈希、70轮shuffle和11410数值置零随机抽样与重跑逐元素匹配，原频率阈值相同。首轮即分叉，日志第29批首次可见。\n- [Hypothesis] CUDA算法/浮点非确定性为首位候选，未做因果隔离；原逐批顺序/噪声/掩码/梯度/完整初始化、线程/算法状态和9月14日精确源码未完整留存，不能唯一归因。核查完成，无新前向/训练/生成；建议最小首轮入口追踪，尚未启动。seed43停止下游、seed13结果及其他暂停项保持。\n'+f'- Report `{REPORT.relative_to(ROOT)}`; snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`; diagnostic/supporting，严格main/A/B保持、C推迟；当前审计→无论文对应目标→差距不适用→原随机轨迹缺失→严格结论保持。\n'
 p=ROOT/'reproduction-state.md';h,t=p.read_text().split('\n',1);p.write_text(h+'\n\n## Latest checkpoint: 2026-11-03 FD002 seed43 rerun audited\n\n'+body+'\n'+t)
 with (ROOT/'experiment-ledger.md').open('a') as f:f.write('\n## `'+RUN.name+'` completed\n\n'+body)
 print(json.dumps(finding,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
