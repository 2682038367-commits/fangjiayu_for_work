"""Describe the existing five-generator/five-predictor FD002 formal-test results."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/diagnostic-fd002-formal-generator-seeds-20261106'
OUT=ROOT/'results/diagnostics/fd002_formal_generator_seeds_20261106'
REPORT=ROOT/'docs/conclusions/final/fd002_formal_generator_seeds_20261106_report.md'
PROTOCOL=ROOT/'docs/operations/FD002_FORMAL_GENERATOR_SEEDS.md'
RAW=ROOT/'results/fd002_evaluation_stability_paper70_raw.csv'
SEEDS=[3,13,23,33,43]


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def state(stage,body):
    entry=f'\n## {now()} FD002 formal5generator decomposition {stage}\n\n{body}\n'
    p=ROOT/'reproduction-state.md';head,tail=p.read_text().split('\n',1);p.write_text(head+'\n'+entry+'\n'+tail)
    with (ROOT/'experiment-ledger.md').open('a') as f:f.write(entry)


def main():
    assert not RUN.exists(),'No overwrite'
    (RUN/'source').mkdir(parents=True);OUT.mkdir(parents=True)
    paths=[Path(__file__),PROTOCOL,RAW,ROOT/'results/fd002_evaluation_stability_paper70_summary.csv',
           ROOT/'docs/provenance/paper_table_i_w48_metrics.json',ROOT/'configs/evaluation_fd002_stability_paper.yaml',
           ROOT/'results/diagnostics/fd002_formal_test_protocol_audit_20261106/audit.json']
    prior_path=ROOT/'runs/diagnostic-fd002-formal-test-protocol-audit-20261106/run-manifest.json'
    prior=json.loads(prior_path.read_text());assert prior['status']=='complete_partial_protocol_match'
    assert sha(RAW)==prior['inputs_sha256'][str(RAW.relative_to(ROOT))]
    for p in [Path(__file__),PROTOCOL,RAW]:
        target=RUN/'source'/p.name;shutil.copy2(p,target);paths.append(target)
    shutil.copy2(prior_path,RUN/'source/previous-protocol-audit-manifest.json');paths.append(RUN/'source/previous-protocol-audit-manifest.json')
    config=dict(generation_seeds=SEEDS,evaluator_seeds=SEEDS,independent_generation_units=5,
        predictor_fits_per_generator=5,existing_cells=25,paper_reference=24.565,
        metric='Arithmetic mean of each generator five predictor RMSEs, then mean five generator means',
        variability='Sample SD ddof1 and min/max over five predictor seeds per generator',
        gap_contribution='(generator mean - paper reference)/5; signed arithmetic decomposition, not causal attribution',
        exclusions='Leave out the largest-mean generator only as descriptive sensitivity; retain all5 in headline',
        stop='Read25existing cells only; no new training, sampling, inference, seed search or significance claims against unknown paper variance')
    dump(RUN/'config.json',config);paths.append(RUN/'config.json')
    m=dict(run_id=RUN.name,scope_class='diagnostic',role='supporting',scope_contract='scope-20261106-fd002-formal-generator-seeds',
        status='running',started_at=now(),user_authorization='Explicit decompose26.7918into5generator means and5predictor ranges; no training',
        config=config,inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        command=[sys.executable,str(Path(__file__).resolve())],
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    dump(RUN/'run-manifest.json',m)
    state('started','- [Observed] Freeze existing25officialtest cells,5generators/5predictors each. User requests per-generator mean/SD/range and arithmetic gap decomposition. No training/sampling/inference; retain all5generators in headline. Snapshot `'+str(RUN.relative_to(ROOT))+'/run-manifest.json`. Prior paper protocol unknowns remain; strictmain/A/Bunchanged,Cdeferred.')
    try:
        f=pd.read_csv(RAW)
        assert len(f)==25 and f[['generation_seed','evaluator_seed']].drop_duplicates().shape[0]==25
        assert sorted(f.generation_seed.unique())==sorted(f.evaluator_seed.unique())==SEEDS
        p=float(json.loads((ROOT/'docs/provenance/paper_table_i_w48_metrics.json').read_text())['metrics']['predictive_rmse']['values']['FD002'])
        rows=[];full=float(f.rmse.mean());gap=full-p
        for seed in SEEDS:
            group=f[f.generation_seed==seed].sort_values('evaluator_seed');assert group.evaluator_seed.tolist()==SEEDS
            v=group.rmse.to_numpy(dtype=float);mean=float(v.mean())
            rows.append(dict(generation_seed=seed,predictor_count=5,mean_RMSE=mean,SD_predictor_RMSE=float(v.std(ddof=1)),
                min_RMSE=float(v.min()),max_RMSE=float(v.max()),range_width=float(v.max()-v.min()),
                mean_minus_paper=mean-p,individual_predictors_above_paper=int(np.sum(v>p)),
                contribution_to_overall_gap=(mean-p)/5,share_of_net_gap=((mean-p)/5)/gap,
                per_predictor_values=v.tolist()))
        table=pd.DataFrame(rows)
        assert abs(table.mean_RMSE.mean()-full)<1e-12 and abs(table.contribution_to_overall_gap.sum()-gap)<1e-12
        old=pd.read_csv(ROOT/'results/fd002_evaluation_stability_paper70_summary.csv')
        for row in rows:
            ref=old[(old.metric=='rmse')&(old.generation_seed.astype(str)==str(row['generation_seed']))].iloc[0]
            assert abs(ref['mean']-row['mean_RMSE'])<1e-12 and abs(ref.sample_std-row['SD_predictor_RMSE'])<1e-12
        worst=int(table.loc[table.mean_RMSE.idxmax(),'generation_seed']);rest=table[table.generation_seed!=worst]
        within=float(np.sqrt(np.mean(table.SD_predictor_RMSE**2)));between=float(table.mean_RMSE.std(ddof=1))
        result=dict(status='complete',paper_reference=p,overall_RMSE=full,overall_gap=gap,
            independent_generation_units=5,predictor_results=25,per_generator=rows,
            generator_means_above_paper=int(np.sum(table.mean_RMSE>p)),
            generators_with_all5_predictors_above_paper=int(np.sum(table.min_RMSE>p)),
            median_generator_mean=float(table.mean_RMSE.median()),between_generator_mean_SD=between,
            pooled_within_generator_predictor_SD=within,ratio_between_to_within_SD=between/within,
            largest_mean_seed=worst,
            descriptive_without_largest=dict(excluded_seed=worst,mean_RMSE=float(rest.mean_RMSE.mean()),gap=float(rest.mean_RMSE.mean()-p),headline=False),
            interpretation='Mixed pattern:4of5generator groups are above the paper point estimate; seed43 is especially poor and contributes61.1%of the signed net gap. Remaining4still average above paper. Predictor randomness alone does not explain group gaps.',
            limits='Paper number is an averaged point estimate, not a per-seed acceptance threshold. Only5generator trainings, one sampled dataset each; cannot separate generator-training and sampling variation or infer a common implementation fault from these25cells.',
            new_training=0,new_sampling=0,new_inference=0)
        dump(OUT/'summary.json',result)
        table.drop(columns='per_predictor_values').to_csv(OUT/'per_generator.csv',index=False)
        f.pivot(index='generation_seed',columns='evaluator_seed',values='rmse').reindex(index=SEEDS,columns=SEEDS).to_csv(OUT/'rmse_matrix.csv')
        text='# FD002正式测试：五个生成器种子的RMSE拆分\n\n'
        text+=f'- [Observed] 原有25格结果，5次生成器训练，每份生成数据训练5个预测器。总体{full:.9f}，论文点估计{p:.3f}，观测差{gap:.9f}。没有新增训练/生成/推理。\n\n'
        text+='| 生成器种子 | 5预测器平均RMSE | 预测器样本SD | 最小—最大 | 均值减论文 | 净差距算术贡献 |\n|---|---:|---:|---:|---:|---:|\n'
        for row in rows:
            text+=f'| {row["generation_seed"]} | {row["mean_RMSE"]:.5f} | {row["SD_predictor_RMSE"]:.5f} | {row["min_RMSE"]:.5f}–{row["max_RMSE"]:.5f} | {row["mean_minus_paper"]:+.5f} | {row["contribution_to_overall_gap"]:+.5f} ({row["share_of_net_gap"]:.1%}) |\n'
        text+='\n- [Observed] 3/23/33/43四组各自5个预测器全部高于24.565；13的五个均低于。23的预测器波动最大SD1.11343，其余SD.18422–.44398；43虽然最差，组内波动只有.36403。\n'
        text+=f'- [Observed] 生成器均值之间SD{between:.5f}，组内预测器合并SD{within:.5f}，前者约{between/within:.2f}倍。仅描述这5份生成数据的差异，不把25格视为25次生成器训练。\n'
        text+=f'- [Observed] seed43均值比论文高6.80116，以固定五种子权重贡献总体差1.36023周期，占净差61.1%。这是加减法分解，不是61.1%的因果解释；seed13贡献为负。只作敏感性描述，不计43时其余4组平均{rest.mean_RMSE.mean():.5f}，仍高{rest.mean_RMSE.mean()-p:.5f}；正式分数仍保留43。\n'
        text+=f'- [Observed] 生成器均值中位数{table.mean_RMSE.median():.5f}也高于论文。因此不是仅一个极差种子造成全部差距，也不是所有种子统一偏移相同幅度。\n'
        text+='- [Hypothesis] 更支持“多数组表现偏高，加上seed43额外退化”的混合解释；预测器换种子不是组间差异的主要表面来源。共有处理/实现差异尚待证实，生成训练与采样也不能由每训练一批数据的设计单独分离。已有cuDNN诊断是另一组机制证据，不把此表当作其独立证明。\n'
        text+='- [Paper] 24.565是5次实验平均点估计，不是每个本地种子必须低于的阈值；未公开原种子、逐次方差/随机层级等，末端/cap规则仍有上轮记录的论文细节未知。\n'
        text+='- 下一项建议：[Hypothesis] 先复用正式测试逐发动机误差，区分3/23/33/43是否共同在相同发动机出错，以及43有哪些额外恶化；据此区分共有薄弱区域与额外不稳定。此建议未执行，不新增轮次/训练，不默认最终轮更好。\n'
        text+=f'- 决策线：[Observed]{full:.5f} → [Paper]{p:.3f} → [Observed]+{gap:.5f} → [Hypothesis]是否超过双方重复波动未知 → [Hypothesis]正式数值复现仍不确定。统计单位5，不能称25独立重复。\n'
        text+=f'- Run `{RUN.relative_to(ROOT)}/run-manifest.json`；CSV `{OUT.relative_to(ROOT)}/per_generator.csv`，完整25格`rmse_matrix.csv`。诊断/supporting预算关闭；严格main/A/B不变、C推迟、seed43内部验证挑轮次继续暂停。\n'
        REPORT.write_text(text)
        for name,expected in m['inputs_sha256'].items():assert sha(ROOT/name)==expected,name
        verify=dict(passed=True,cells=25,generation_units=5,predictor_repetitions_per_unit=5,
            matches_existing_summary=True,all25_seed_pairs_unique=True,gap_contributions_sum_exact=True,input_hashes_unchanged=True)
        dump(RUN/'verification.json',verify)
        m.update(status='complete',completed_at=now(),outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in list(OUT.glob('*'))+[REPORT]},
                 new_training=0,new_sampling=0,new_inference=0)
        dump(RUN/'run-manifest.json',m)
        state('complete','- [Observed] Existing25cells decomposed:gen3mean26.99046±.18422,13mean23.54036±.44398,23mean26.84948±1.11343,33mean25.21257±.35757,43mean31.36616±.36403. Fourgroups all5predictors abovepaper24.565,13allbelow. Overall26.79180409gap2.22680409;43contributes+1.36023101/61.0844%of signednetgap. Descriptiveexcluding43other4mean25.64821636still+1.08321636;headlinekeepsall5. BetweengroupSD2.91691vswithinpooled.58841,statisticsunit5not25.\n- [Hypothesis] Mixedpattern:broaderabove-pointmeans plusoneparticularlybadgeneratorbatch;notsolelypredictorrandomness,notproofcommonimplementationfault,noseparationoftrainingvssamplingfromthisdesign. Proposedreadonlyper-enginecommon-vs43excesserror comparison not executed.\n- Snapshot `'+str(RUN.relative_to(ROOT))+'/run-manifest.json`; report `'+str(REPORT.relative_to(ROOT))+'`; verification passed. Current26.7918→paper24.565→gap2.2268→paperrepeatvariabilityunknown→strictconclusionsunchanged. Zero training/sampling/inference;budgetclosed,strictmain/A/Bunchanged,Cdeferred,seed43internalvalidationepochsearchpaused.')
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except BaseException:
        m.update(status='failed',failed_at=now(),failure=traceback.format_exc());dump(RUN/'run-manifest.json',m)
        state('failed','- [Observed] Read-only decomposition failed; artifacts preserved; zero training/sampling/inference. No automatic retry.')
        raise


if __name__=='__main__':main()
