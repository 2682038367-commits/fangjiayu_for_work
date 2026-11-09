"""Summarize existing frequency-threshold diagnostics, keeping retired claims separate."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/diagnostic-frequency-threshold-history-summary-20261106'
OUT=ROOT/'results/diagnostics/frequency_threshold_history_summary_20261106'
REPORT=ROOT/'docs/conclusions/final/frequency_threshold_experiments_summary_20261106.md'
SEEDS=[3,13,23,33,43,53,63]
EVALS=[3,13,23,33,43]


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def state(stage,body):
    entry=f'\n## {now()} Frequency-threshold history summary {stage}\n\n{body}\n'
    p=ROOT/'reproduction-state.md';head,tail=p.read_text().split('\n',1);p.write_text(head+'\n'+entry+'\n'+tail)
    with (ROOT/'experiment-ledger.md').open('a') as f:f.write(entry)


def main():
    assert not RUN.exists(),'Preserve existing summary'
    (RUN/'source').mkdir(parents=True);OUT.mkdir(parents=True)
    raw_paths={
      'public':ROOT/'results/frequency_threshold_validation/fd002_public_arm_seven_seeds_evaluation_raw.csv',
      'fixed025':ROOT/'results/frequency_threshold_validation/fd002_fixed_025_seven_seeds_evaluation_raw.csv',
      'soft43':ROOT/'results/soft_learnable_threshold/fd002_seed43_evaluation_raw.csv',
      'soft_other6':ROOT/'results/soft_learnable_threshold/fd002_remaining_seeds_evaluation_raw.csv',
      'binarySTE':ROOT/'results/binary_ste/fd002_evaluation_raw.csv'}
    source_files=[Path(__file__),*raw_paths.values(),
      ROOT/'results/frequency_threshold_validation/fd002_fixed_025_evaluation_raw.csv',
      ROOT/'results/frequency_threshold_validation/fd002_frequency_threshold_validation_report.md',
      ROOT/'results/soft_learnable_threshold/fd002_seed43_report.md',ROOT/'notes/binary_ste_fd002_protocol.md',
      ROOT/'results/diagnostics/public_code_quantile_rmse_report.md',ROOT/'results/diagnostics/public_code_quantile_rmse.csv',
      ROOT/'docs/conclusions/final/theta_rmse_report.md',ROOT/'docs/conclusions/final/theta_analysis_report.md',
      ROOT/'docs/conclusions/final/fd002_seven_seed_conclusion.md',ROOT/'results/frozen_numbers.json',
      ROOT/'upstream/DiffusionFreeGuidence/Attetnion_temp.py',ROOT/'analysis/diagnostics/threshold_ste_probe.py',
      ROOT/'analysis/diagnostics/theta_rmse_analysis.py',
      ROOT/'docs/conclusions/final/fd002_seed43_variation_stages_20261105_report.md']
    manifests=[ROOT/'results/frequency_threshold_validation/fd002_public_arm_seven_seeds_manifest.json',
      ROOT/'results/frequency_threshold_validation/fd002_fixed_025_seven_seeds_manifest.json',
      ROOT/'results/soft_learnable_threshold/fd002_seed43_manifest.json',ROOT/'results/soft_learnable_threshold/fd002_remaining_seeds_manifest.json',
      ROOT/'results/binary_ste/fd002_manifest.json']
    source_files += manifests
    for p in [Path(__file__),ROOT/'notes/binary_ste_fd002_protocol.md']:
        q=RUN/'source'/p.name;shutil.copy2(p,q);source_files.append(q)
    config=dict(scope='Read-only historical threshold diagnostic summary',dataset='FD002',window=48,
        seeds=SEEDS,evaluators=EVALS,statistics_unit='7 generator trainings per full arm, not35 independent generator runs',
        old_three_seed_subset='13/23/43: targeted diagnostic, superseded by complete7seed effect inference',
        source_priority='Raw formalCSV > frozen numbers and current audited quantile report > legacy exploratory narratives',
        stop='Summarize existing results and corrections only; no training, sampling, gradient probe, new sweep or default adoption')
    dump(RUN/'config.json',config);source_files.append(RUN/'config.json')
    m=dict(run_id=RUN.name,status='running',scope_class='diagnostic',role='supporting',
      scope_contract='scope-20261106-frequency-threshold-history-summary',started_at=now(),config=config,
      user_authorization='Summarize previously completed frequency-threshold validations',
      inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_files},
      command=[sys.executable,str(Path(__file__).resolve())],
      git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    dump(RUN/'run-manifest.json',m)
    state('started','- [Observed] User requests historical frequency-threshold recap. Read-only original/fixed/soft/STE formalCSVs, mechanism/provenance and retired reports; no new experiments. Snapshot `'+str(RUN.relative_to(ROOT))+'/run-manifest.json`. Strictmain/A/Bunchanged,Cdeferred.')
    fs={name:pd.read_csv(p) for name,p in raw_paths.items()}
    fs['soft']=pd.concat([fs['soft43'],fs['soft_other6']],ignore_index=True)
    group={};aggregate=[];pergen=[]
    for name in ['public','fixed025','soft','binarySTE']:
        f=fs[name]
        assert len(f)==35 and len(f[['generation_seed','evaluator_seed']].drop_duplicates())==35
        assert sorted(f.generation_seed.unique())==SEEDS and sorted(f.evaluator_seed.unique())==EVALS
        g=f.groupby('generation_seed')[['rmse','discriminative_score']].mean().reindex(SEEDS);group[name]=g
        aggregate.append(dict(arm=name,generation_runs=7,predictor_evaluations=35,
            RMSE=float(g.rmse.mean()),RMSE_SD_generation_means=float(g.rmse.std(ddof=1)),
            DS=float(g.discriminative_score.mean()),DS_SD_generation_means=float(g.discriminative_score.std(ddof=1)),
            batches_mean_DS_above03=int((g.discriminative_score>.3).sum()),
            RMSE_improving_vs_public=int((g.rmse<group['public'].rmse).sum())))
        for seed,row in g.iterrows():pergen.append(dict(arm=name,generation_seed=int(seed),RMSE=float(row.rmse),DS=float(row.discriminative_score)))
    split_hashes=[]
    for p in manifests:
        a=json.loads(p.read_text());assert a['status']=='complete'
        assert a['config']['evaluation']['predictive_train_rate']==.7
        split_hashes.append(a['split_sha256'])
    assert len(set(split_hashes))==1
    tests={}
    for metric in ['rmse','discriminative_score']:
        d=(group['fixed025'][metric]-group['public'][metric]).to_numpy()
        tests[metric]=dict(n_generator_pairs=7,mean_fixed_minus_public=float(d.mean()),
            paired_t_p=float(stats.ttest_rel(group['fixed025'][metric],group['public'][metric]).pvalue),
            wilcoxon_exact_p=float(stats.wilcoxon(d,method='exact').pvalue),
            CI95=list(map(float,stats.t.interval(.95,df=6,loc=d.mean(),scale=stats.sem(d)))),
            improving_generator_seeds=int((d<0).sum()))
    frozen=json.loads((ROOT/'results/frozen_numbers.json').read_text())['fd002_seven_seed_paired']
    assert abs(tests['rmse']['mean_fixed_minus_public']-frozen['rmse']['mean'])<.0001
    assert abs(tests['rmse']['paired_t_p']-frozen['rmse']['p_t'])<.0001
    initial=pd.read_csv(ROOT/'results/frequency_threshold_validation/fd002_fixed_025_evaluation_raw.csv')
    subset={}
    for arm,f in [('public',fs['public'][fs['public'].generation_seed.isin([13,23,43])]),('fixed025',initial)]:
        subset[arm]=f.groupby('generation_seed')[['rmse','discriminative_score']].mean().mean().to_dict()
    corr=(ROOT/'results/diagnostics/public_code_quantile_rmse_report.md').read_text()
    correlations=[]
    for line in corr.splitlines():
        if line.startswith('| FD'):
            fields=[x.strip() for x in line.strip('|').split('|')]
            pvals=[float(fields[i]) for i in [3,5,7]];assert all(v>.05 for v in pvals)
            correlations.append(dict(dataset=fields[0],n=int(fields[1]),p_values=pvals))
    assert len(correlations)==4
    result=dict(status='complete',arms=aggregate,fixed_vs_public_tests=tests,
      initial_targeted3seed=subset,public_quantile_correlations=correlations,
      shared_split_sha256=split_hashes[0],
      confirmed='Public hard-quantile parameter is initialized once and disconnected from loss by boolean mask; not learned. Fixed0.25 and learnable variants have completed FD002 diagnostic data.',
      not_confirmed='No stable benefit of fixed0.25 established in7seed tests; no universal learnable-threshold advantage; no monotone threshold-size explanation of public variability.',
      retired=['Soft/STE energy threshold interpreted as actual drop fraction','Near-zero threshold interpreted directly as branch degeneration','Initial3seed improvement generalized to overall stability','Individual old paired improvements uniquely attributed to threshold despite uncontrolled native cuDNN variability'],
      limits='Fixed quantile0.25, soft constrained energy threshold and STE positive energy threshold differ in semantics. soft/STE are local implementations with unspecified paper details, not author-exact. Old training actualcuDNN paths not held equal. Not a fixed-cuDNN threshold intervention.',
      new_training=0,new_sampling=0,new_probe=0)
    dump(OUT/'summary.json',result);pd.DataFrame(aggregate).to_csv(OUT/'arm_summary.csv',index=False);pd.DataFrame(pergen).to_csv(OUT/'per_generator.csv',index=False)
    text='# 频率阈值验证历史汇总（2026-11-06复核）\n\n'
    text+='- [Observed] 本轮仅复核已有产物，无新训练、采样或梯度探针。主要实训范围是FD002、window48；跨FD001–FD004另有公开分位参数与正式RMSE的描述性相关检查。\n\n'
    text+='1. **论文—代码与梯度检查。** [Paper] 论文描述可学习阈值经反传更新并生成二值掩码。[Code] 公开实现两个模块各在初始化抽一次q∼U(0,.5)，以quantile再硬比较构造掩码，阈值不通过该链路得到梯度；不是每个前向重新随机抽阈值。[Observed] 旧报告记录q=.05/.25/.49的前向/反向检查grad=None、保留率约95%/75%/48%，以及同seed除阈值外初始张量一致。确认代码偏差，不等于确认其造成全部性能差距。\n\n'
    text+=f'2. **固定分位参数0.25，先3种子。** [Observed] 13/23/43各5评价：RMSE从{subset["public"]["rmse"]:.5f}到{subset["fixed025"]["rmse"]:.5f}；DS从{subset["public"]["discriminative_score"]:.5f}到{subset["fixed025"]["discriminative_score"]:.5f}。23改善明显、13略退步、43仍较差。该子集根据历史好坏挑选，不能直接推广总体。\n\n'
    text+='3. **固定0.25补全7种子。** [Observed] 种子3/13/23/33/43/53/63，每份5预测器，固定同一70/30划分及259台末端48测试窗口。统计单位n=7，不是35。固定值是在模型构造后覆盖，保留原初始化随机数消费；训练/采样均使用该固定参数。\n\n'
    text+='| 方案 | 生成种子数 | 正式RMSE均值 | 判别DS均值 | 均值DS>.3的批次数 |\n|---|---:|---:|---:|---:|\n'
    for a in aggregate:text+=f'| {a["arm"]} | 7 | {a["RMSE"]:.5f} | {a["DS"]:.5f} | {a["batches_mean_DS_above03"]}/7 |\n'
    text+=f'\n- [Observed] 固定−公开RMSE={tests["rmse"]["mean_fixed_minus_public"]:.5f}，t配对p={tests["rmse"]["paired_t_p"]:.4f}，Wilcoxon p={tests["rmse"]["wilcoxon_exact_p"]:.4f}，95%CI={tests["rmse"]["CI95"]}；4/7种子改善。DS差={tests["discriminative_score"]["mean_fixed_minus_public"]:.5f}，t p={tests["discriminative_score"]["paired_t_p"]:.4f}。均未检出稳定优势，也不能声称等价或证明固定阈值无效。\n'
    text+='- [Observed] seed53的RMSE35.30437→24.20027贡献很大；旧报告的剔除/非参数敏感性检查未改变“未检出显著差异”。不能以单次大改善反推其由阈值独立造成。DS>.3是本地批次诊断阈值，不是论文真实失败概率；0/7也不证明永不失败。\n\n'
    text+='4. **soft可学习能量阈值。** [Code] 本地补全θ=.5sigmoid(α)、初始θ=.25、温度.1，连续sigmoid掩码，不是作者公开实现也不是严格二值。[Observed] 先seed43单次生成训练+5评价，RMSE31.36616→30.02651，DS.42110→.22058；随后补齐另外6种子，总体RMSE27.02841、DS.22387。参数有更新，但总体未形成稳定优于固定0.25的结论；不能推广单seed改善。\n\n'
    text+='5. **二值前向+STE可学习能量阈值。** [Code] 本地正阈值softplus、初值1.0、sigmoid替代梯度温度.1；前向仍是0/1，反向可传梯度。论文未给这些补全细节。[Observed] 完整7种子×5评价，RMSE29.36527、DS.29434；相对公开臂6/7种子RMSE变差，相对固定0.25为7/7变差。可学习和二值前向均不自动保证下游收益。\n\n'
    text+='6. **阈值值域/相关分析与语义修正。** [Observed] 从checkpoint读取两个分位参数，结合正式5评价均值，在FD001/2/3/4分别5/7/5/5生成种子上做描述性相关；当前经CSV复核报告中相关检验p均>.05，方向不一致。无法用阈值大小单独解释性能。[Code] 固定q=.25是分位参数，不是.25Hz；soft/STE的归一化能量阈值也不是实际丢弃比例。旧“几乎全保留/分支退化”与跨模式θ—丢弃率相关结论已撤回，真正丢弃率须读前向mask统计。[Observed] 有合成Oracle/STE可行性脚本，但它不构成真实DDPM下游收益证据，本轮未重跑或据此新增性能结论。\n\n'
    text+='## 当前应保留的结论\n\n'
    text+='- [Code] 公开硬掩码阈值确实不能按论文描述学习，这是机制/实现事实。\n'
    text+='- [Observed] 固定0.25在已跑7种子上的均值最低，但配对统计未检出稳定优势；soft/STE也未证明补上梯度即可解决下游差距。局部实现试验不能当作对论文频率模块整体的否定。\n'
    text+='- [Hypothesis] 早期把随机阈值说成主要失败原因过强。最近独立证实cuDNN路径可使同seed完整训练分叉，旧阈值实验没有严格控制该路径，因此其单种子变化包含未隔离训练数值波动。现有结果不能估计阈值与cuDNN各自贡献。\n'
    text+='- [Code] 固定频率阈值与固定cuDNN是两种干预；当前没有在已验证可重复的固定cuDNN训练下重新做完整阈值配对。本轮只总结，不启动此实验；也未开展真实数据阈值网格搜索。\n'
    text+='- [Observed] 上表原版28.29464是当时完整7生成种子对照，当前共同5种子的26.79180是另一种子集合，不能混用。旧计划中FD001“固定0.25已完成”条目引用的是公开臂正式CSV，不能据此当作可核查的独立固定阈值干预。\n'
    text+='- 决策线：[Observed] 固定25.97653 → [Paper]FD002-48点估计24.565 → [Observed]+1.41153（7生成种子，论文5次）→ [Hypothesis]论文跨次方差未知、局部固定效应未显著 → 严格复现结论不因这些诊断改写。soft/STE不混入公开主线。\n'
    text+=f'- Run `{RUN.relative_to(ROOT)}/run-manifest.json`；机器表 `{OUT.relative_to(ROOT)}/arm_summary.csv`。统计复核完成，预算关闭；strictmain/A/B不变、C推迟，不新增训练或改默认规则。\n'
    REPORT.write_text(text)
    for name,expected in m['inputs_sha256'].items():assert sha(ROOT/name)==expected,name
    dump(RUN/'verification.json',dict(passed=True,full_arms=4,generator_runs_per_arm=7,evaluators_per_generator=5,
        shared_splits_confirmed=True,paired_fixed_effect_matches_frozen_numbers=True,retired_claims_excluded=True,
        new_training=0,new_sampling=0,new_probe=0))
    m.update(status='complete',completed_at=now(),outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in list(OUT.glob('*'))+[REPORT]},
             new_training=0,new_sampling=0,new_probe=0)
    dump(RUN/'run-manifest.json',m)
    state('complete','- [Observed] Historicalthresholdrecap verified from4complete7generator×5evaluatorarms: publicRMSE28.294635/DS.262078;fixedq.25 25.976527/.141107;soft27.028414/.223870;binarySTE29.365265/.294336. Fixed−publicΔRMSE−2.318109,t p.184138,Wilcoxon.21875,CI[-6.098151,1.461933];no stablebenefitestablished,notequivalence. Softinitialseed43improvementdidnotestablishgeneralbenefit;STE6/7worsepublic. Publicquantilecorrelationsallp>.05 incurrentauditedreport.\n- [Code] Publicqinitializedonce/hardmaskgradientdisconnected. Soft/STEenergythresholdsnotdropfractions;oldcrossmode/drop-collapseclaimsretired. [Hypothesis] Oldperseedthresholdcausalityoverstated;nativecuDNNpathvariabilitywasnotcontrolled,so cannotassignoldgainssolelytothreshold. Fixedfrequencyq≠fixedcuDNN. No fixed-cuDNN thresholdexperimentorsweepstarted.\n- Report `'+str(REPORT.relative_to(ROOT))+'`; snapshot `'+str(RUN.relative_to(ROOT))+'/run-manifest.json`; verificationpassed. Newtraining/sampling/probe0;strictmain/A/Bunchanged,Cdeferred. Summarybudgetclosed.')
    print(json.dumps(dict(arms=aggregate,tests=tests),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
