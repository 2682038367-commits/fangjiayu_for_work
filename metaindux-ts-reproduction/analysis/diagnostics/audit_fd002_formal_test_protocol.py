"""Read-only audit of paper24.565, historical26.3585 and controlled test metrics."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/diagnostic-fd002-formal-test-protocol-audit-20261106'
OUT = ROOT/'results/diagnostics/fd002_formal_test_protocol_audit_20261106'
PROTOCOL = ROOT/'docs/operations/FD002_FORMAL_TEST_PROTOCOL_AUDIT.md'
REPORT = ROOT/'docs/conclusions/final/fd002_formal_test_protocol_audit_20261106_report.md'
PDF = Path('/home/cw_boe/桌面/论文复现/MetaIndux-TS_Frequency-Aware_AIGC_Foundation_Model_for_Industrial_Time_Series.pdf')
SEEDS = [3,13,23,33,43]


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def label(path):return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def record(stage,body):
    entry=f'\n## {now()} FD002 formal-test protocol audit {stage}\n\n{body}\n'
    p=ROOT/'reproduction-state.md';head,tail=p.read_text().split('\n',1);p.write_text(head+'\n'+entry+'\n'+tail)
    with (ROOT/'experiment-ledger.md').open('a') as f:f.write(entry)


def main():
    assert not RUN.exists(),'Preserve prior audits'
    (RUN/'source').mkdir(parents=True);OUT.mkdir(parents=True)
    for path in [Path(__file__),PROTOCOL]:shutil.copy2(path,RUN/'source'/path.name)
    paths=[PDF,Path(__file__),PROTOCOL,ROOT/'docs/provenance/paper_table_i_w48_metrics.json',
           ROOT/'docs/provenance/paper_predictive_scores_w48.json',ROOT/'SOURCE_SNAPSHOT.md',
           ROOT/'upstream/data/CMAPSSDataset.py',ROOT/'upstream/MainCondition.py',ROOT/'upstream/eva_regressor.py',ROOT/'upstream/utils.py',
           ROOT/'scripts/evaluate_fd002_stability.py',ROOT/'configs/evaluation_fd002_stability_paper.yaml',
           ROOT/'results/fd002_paper_metrics.csv',ROOT/'results/fd002_paper_report.md',ROOT/'logs/fd002_paper.log',
           ROOT/'results/fd002_evaluation_stability_paper70_raw.csv',ROOT/'results/fd002_evaluation_stability_paper70_summary.csv',
           ROOT/'results/fd002_evaluation_stability_paper70_manifest.json',ROOT/'results/fd002_evaluation_fixed_split_paper70.npz',
           ROOT/'results/frequency_threshold_validation/fd002_public_arm_seven_seeds_evaluation_raw.csv',
           ROOT/'results/final_evaluation_summary.csv',ROOT/'results/frozen_numbers.json',
           ROOT/'results/diagnostics/rul_segment_errors_w48/predictions.csv',
           ROOT/'results/diagnostics/rul_segment_errors_w48/checkpoint_replay_checks.csv']
    # Resolve the saved seven-seed raw filename from its existing manifest.
    paths=[p for p in paths if p.exists()]
    formal_path=ROOT/'results/frequency_threshold_validation/fd002_public_arm_seven_seeds_manifest.json'
    formal=json.loads(formal_path.read_text());seven_raw=ROOT/formal['config']['outputs']['raw_csv'];paths += [formal_path,seven_raw]
    for kind in ['train','test','RUL']:paths.append(ROOT/f'upstream/data/{kind}_FD002.txt')
    for seed in SEEDS:
        paths += [ROOT/f'results/runs/FD002_w48_seed{seed}_metrics.json',ROOT/f'results/runs/FD002_w48_seed{seed}_metadata.json']
    for path in [ROOT/'upstream/data/CMAPSSDataset.py',ROOT/'upstream/MainCondition.py',ROOT/'upstream/eva_regressor.py',ROOT/'scripts/evaluate_fd002_stability.py']:
        dest=RUN/'source'/path.name;shutil.copy2(path,dest);paths.append(dest)
    config=dict(scope='Read-only paper and existing formal-test results audit; no new inference/training/sampling',
        dataset='FD002',window=48,paper_target='TableI FD002-48 predictive score',
        questions=['Each test engine terminal window?','RUL cap same?','Five-repeat versus local seed aggregation?'],
        historical_seed_scope=SEEDS,controlled_seed_scope=dict(generation=SEEDS,evaluator=SEEDS),
        stop='Document known/unknown protocol fields and recompute existing metrics only; stop seed43 internal-validation epoch selection, no new jobs')
    dump(RUN/'config.json',config);paths.append(RUN/'config.json')
    paths += [p for p in (RUN/'source').iterdir() if p.is_file()]
    m=dict(run_id=RUN.name,status='running',scope_class='diagnostic',role='supporting',
        scope_contract='scope-20261106-fd002-formal-test-protocol-audit',started_at=now(),config=config,
        command=[sys.executable,str(Path(__file__).resolve())],
        user_authorization='Audit existing formal test endpoint/cap/seed aggregation before explaining24.57vs26.36; no new runs',
        inputs_sha256={label(p):sha(p) for p in dict.fromkeys(paths)},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        numpy=str(np.__version__),pandas=str(pd.__version__),python=sys.version)
    dump(RUN/'run-manifest.json',m)
    record('started','- [Observed] User returns to formal-test alignment; freeze PDF/current code/old5run metrics/controlled5x5 and7x5 records. Zero training/sampling/new prediction. Paper-code distinctions and unavailable historical splits will remain explicit. Snapshot `'+label(RUN)+'/run-manifest.json`. Seed43 internal validation checkpoint pursuit paused; strictmain/A/Bunchanged,Cdeferred.')
    try:
        provenance=json.loads((ROOT/'docs/provenance/paper_table_i_w48_metrics.json').read_text())
        assert sha(PDF)==provenance['source_document']['pdf_sha256']
        subprocess.run(['pdftotext','-layout',str(PDF),str(RUN/'source/paper-text.txt')],check=True)
        text=(RUN/'source/paper-text.txt').read_text()
        assert 'were repeated five times' in text and 'synthetic data for training and 30% for validation' in text
        subprocess.run(['pdftoppm','-f','7','-l','7','-singlefile','-scale-to','2200','-png',str(PDF),str(RUN/'source/table_i_page7')],check=True)
        paper_value=float(provenance['metrics']['predictive_rmse']['values']['FD002']);assert paper_value==24.565
        paper=dict(value=paper_value,pdf_sha256=sha(PDF),table='I',physical_page=7,printed_page=18068,
            explicit=dict(window=48,generated_train_fraction=.7,generated_dev_fraction=.3,repetitions=5,final='average of repeated experiment metrics'),
            unreported=['Whether real test means terminal window per engine','RUL cap125 and output clipping','Exact five seeds and which stages are rerun','Whether each generator is evaluated by multiple predictor seeds','Raw per-repeat RMSE/SD'])
        dump(OUT/'paper_evidence.json',paper)
        raw_test=pd.read_csv(ROOT/'upstream/data/test_FD002.txt',sep=r'\s+',header=None)
        truth=pd.read_csv(ROOT/'upstream/data/RUL_FD002.txt',sep=r'\s+',header=None).iloc[:,0].to_numpy()
        grouped=raw_test.groupby(0,sort=True);engines=np.array(sorted(grouped.groups))
        assert np.array_equal(engines,np.arange(1,260)) and len(truth)==259
        expected_y=np.minimum(truth,125).astype(np.float32)
        spec=importlib.util.spec_from_file_location('read_only_public_dataset',ROOT/'upstream/data/CMAPSSDataset.py')
        api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
        cwd=Path.cwd();os.chdir(ROOT/'upstream')
        try:
            data=api.CMAPSSDataset('FD002',48,[1000]);x,y=data.get_last_data_slice(data.get_test_data())
        finally:os.chdir(cwd)
        assert tuple(x.shape)==(259,48,14) and np.array_equal(y.numpy().ravel(),expected_y)
        endpoints=[]
        for index,engine in enumerate(engines):
            g=grouped.get_group(engine);n=len(g);endpoints.append(dict(engine_id=int(engine),observed_cycles=n,
                first_real_cycle=int(g.iloc[max(0,n-48),1]),terminal_cycle=int(g.iloc[-1,1]),
                left_padding_repeated_first_cycle=max(0,48-n),raw_terminal_RUL=int(truth[index]),capped_terminal_RUL=int(expected_y[index])))
        pd.DataFrame(endpoints).to_csv(OUT/'test_engine_endpoints.csv',index=False)
        terminal=dict(test_engines=259,input_shape=[259,48,14],samples_per_engine=1,
            shorter_than48=int(sum(v['left_padding_repeated_first_cycle']>0 for v in endpoints)),padding='repeat first observed cycle on left',
            raw_RUL_range=[int(truth.min()),int(truth.max())],capped_RUL_range=[float(expected_y.min()),float(expected_y.max())],
            raw_RUL_above125=int(np.sum(truth>125)),raw_RUL_equal125=int(np.sum(truth==125)),
            truth_only_cap125=True,prediction_clipping=False,paper_explicit_confirmation=False,
            source='Current retained public-code path and original5run test_shape logs; paper text alone does not specify endpoint/cap')
        log=(ROOT/'logs/fd002_paper.log').read_text(errors='replace')
        shape_records=re.findall(r'test_data.shape:\s*torch.Size\(\[259, 48, 14\]\)',log)
        assert len(shape_records)==5;terminal['historical5_run_test_shape_records']=len(shape_records)
        initial=[]
        for seed in SEEDS:
            metric=json.loads((ROOT/f'results/runs/FD002_w48_seed{seed}_metrics.json').read_text())
            value=float(metric['rmse']);assert f'best_test_loss:{value}' in log
            initial.append(dict(generation_seed=seed,rmse=value))
        v=np.array([a['rmse'] for a in initial]);oldmean=float(v.mean())
        pd.DataFrame(initial).to_csv(OUT/'original_five_runs.csv',index=False)
        initial_result=dict(mean_RMSE=oldmean,SD_across5=float(v.std(ddof=1)),paper_gap=oldmean-paper_value,
            relative_gap=oldmean/paper_value-1,generation_seeds=SEEDS,predictors_per_generated_dataset=1,
            aggregation='Arithmetic mean of five per-model RMSEs, not pooled MSE or ensemble predictions',
            sqrt_mean_MSE_different_metric=float(np.sqrt(np.mean(v*v))),
            historical_split='Not frozen in original records; current built-in entry passes no split_indices and retained default is80/20; cannot certify original execution used paper70/30',
            historic_evaluator_seed_assignment='Original per-seed JSON lacks explicit evaluator-seed fields; do not equate with later controlled diagonal')
        raw=pd.read_csv(ROOT/'results/fd002_evaluation_stability_paper70_raw.csv')
        assert len(raw)==25 and sorted(raw.generation_seed.unique())==SEEDS and sorted(raw.evaluator_seed.unique())==SEEDS
        assert all(raw.groupby('generation_seed').size()==5)
        with np.load(ROOT/'results/fd002_evaluation_fixed_split_paper70.npz') as z:
            names=['predictive_train','predictive_dev','discriminator_real_train','discriminator_real_test','discriminator_generated_train','discriminator_generated_test']
            arrays=[z[k].astype(np.int64) for k in names]
            split_hash=hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest()
            assert len(arrays[0])==29077 and len(arrays[1])==12462
        manifest=json.loads((ROOT/'results/fd002_evaluation_stability_paper70_manifest.json').read_text());assert split_hash==manifest['split_sha256']
        genmeans=raw.groupby('generation_seed').rmse.mean().reindex(SEEDS)
        controlled_mean=float(genmeans.mean());pd.DataFrame(dict(generation_seed=SEEDS,mean_predictor_RMSE=genmeans.to_numpy())).to_csv(OUT/'controlled_five_generation_means.csv',index=False)
        predictions=pd.read_csv(ROOT/'results/diagnostics/rul_segment_errors_w48/predictions.csv')
        predictions=predictions[(predictions.dataset=='FD002')&(predictions.source=='generated_headline')]
        allraw=pd.read_csv(seven_raw);assert len(allraw)==35
        checks=[]
        for (g,e),frame in predictions.groupby(['generation_seed','evaluator_seed']):
            frame=frame.sort_values('engine_id')
            assert np.array_equal(frame.engine_id.to_numpy(),engines) and np.array_equal(frame.rul.to_numpy(),expected_y)
            rmse=float(np.sqrt(np.mean((frame.prediction-frame.rul)**2)))
            reference=allraw[(allraw.generation_seed==g)&(allraw.evaluator_seed==e)];assert len(reference)==1
            delta=abs(rmse-float(reference.iloc[0].rmse));assert delta<.001
            checks.append(dict(generation_seed=int(g),evaluator_seed=int(e),engines=259,recomputed_RMSE=rmse,recorded_RMSE=float(reference.iloc[0].rmse),abs_difference=delta))
        assert len(checks)==35;pd.DataFrame(checks).to_csv(OUT/'saved_prediction_recomputation.csv',index=False)
        terminal['saved_prediction_groups_verified']=len(checks)
        terminal['saved_predictions_above125']=int((predictions.prediction>125).sum())
        terminal['saved_prediction_range']=[float(predictions.prediction.min()),float(predictions.prediction.max())]
        controlled_result=dict(mean_RMSE=controlled_mean,SD_across5_generation_means=float(genmeans.std(ddof=1)),
            paper_gap=controlled_mean-paper_value,relative_gap=controlled_mean/paper_value-1,
            generation_seeds=SEEDS,evaluator_seeds=SEEDS,generated_train_count=29077,generated_dev_count=12462,
            train_dev_fraction='70/30',split_seed=20260915,split_state_sha256=split_hash,
            aggregation='For each generator, arithmetic mean5predictor RMSEs; then arithmetic mean5generator means; equal to mean25cell RMSEs',
            sqrt_mean_MSE_different_metric=float(np.sqrt(np.mean(raw.rmse.to_numpy()**2))),
            paper_exact_seed_hierarchy_matched='Unknown: paper only says5repetitions and average, not5x5stage design')
        sevenmeans=allraw.groupby('generation_seed').rmse.mean()
        seven_result=dict(mean_RMSE=float(sevenmeans.mean()),generation_seeds=sorted(map(int,sevenmeans.index)),
            evaluator_seeds=SEEDS,paper_gap=float(sevenmeans.mean()-paper_value),
            source=str(seven_raw.relative_to(ROOT)),note='Existing7x5headline retained separately; do not silently replace common5 with selected seeds')
        result=dict(status='complete_partial_protocol_match',paper=paper,local_terminal_protocol=terminal,
            original5=initial_result,controlled5x5=controlled_result,existing7x5=seven_result,
            conclusion='Do not freeze1.79as a fully protocol-aligned current gap. Retain1.7935187as historical numerical difference; recommend existing explicit70/30 common5x5 as local working reference26.7918041, observed difference2.2268041, with unresolved paper endpoint/cap/seed-hierarchy details.',
            inference_limit='Paper per-repeat outcomes/variance unavailable; local seed variation material; cannot establish difference exceeds both-side repetition variation or declare numerical reproduction.',
            new_training=0,new_sampling=0,new_model_inference=0,
            next_step='Freeze formal-test endpoint/cap/mean-per-model aggregation as the local ruler. Stop within-seed43 internal-validation epoch search; no further jobs in this audit.')
        dump(OUT/'audit.json',result)
        table=[dict(reference='Paper_TableI_FD002_w48',RMSE=paper_value,gap=0,aggregation='Five repetitions averaged; stage details unavailable'),
               dict(reference='Historical_5_runs',RMSE=oldmean,gap=oldmean-paper_value,aggregation='5 generated datasets,1predictor each; original split not frozen'),
               dict(reference='Controlled_common5x5',RMSE=controlled_mean,gap=controlled_mean-paper_value,aggregation='5generators x5predictors, explicit70/30'),
               dict(reference='Existing_7x5_headline',RMSE=float(sevenmeans.mean()),gap=float(sevenmeans.mean()-paper_value),aggregation='7generators x5predictors, retain separately')]
        pd.DataFrame(table).to_csv(OUT/'score_provenance.csv',index=False)
        text_report='# FD002论文正式测试口径核查\n\n'
        text_report+='- [Observed] 本轮仅查已有论文/代码/日志/预测记录，无训练、采样或新增模型推理。结论为部分口径可确认，不能把1.79直接冻结为完全同口径差距。\n\n'
        text_report+='| 核查项 | 论文直接说明 | 本地证据 | 判断 |\n|---|---|---|---|\n'
        text_report+='| FD002窗口 | [Paper] TableI FD002-48 | [Observed] 259×48×14 | 长度一致 |\n'
        text_report+='| 测试粒度 | [Paper] 真实数据作测试，未写末端/全部窗口 | [Code] get_last_data_slice；[Observed] 每259台各1末端窗口，20台短轨迹左补首周期 | 本地/公开代码可确认，论文细节未明确 |\n'
        text_report+='| RUL截断 | [Paper] 未写125及预测截断 | [Code] 训练和测试标签cap125，预测不截断；[Observed] 官方原始RUL6–194，57台>125 | 本地规则明确，不能声称由论文直接确认 |\n'
        text_report+='| 生成训练/验证划分 | [Paper] 70%/30% | [Code] 初始内置入口未传划分、现存默认80/20，旧运行未冻结；[Observed] 后续固定29077/12462 | 初始26.36不能认定已对齐；已有5×5明确70/30 |\n'
        text_report+='| 跨种子汇总 | [Paper] 重复5次后平均，无具体种子或层级 | [Observed] 初始5个单次RMSE平均26.3585；后续5×5层级平均26.7918 | 均为平均RMSE，随机层级无法证实完全相同 |\n\n'
        text_report+='## 数字来源与汇总\n\n'
        for row in table:text_report+=f'- [Observed] {row["reference"]}: RMSE={row["RMSE"]:.9f}，相对论文差={row["gap"]:+.9f}。{row["aggregation"]}。\n'
        text_report+=f'\n- [Observed] 最初五个RMSE={v.tolist()}，算术均值26.358518722，SD3.017492933。这是26.36的来源，不是最近内部验证结果，也不是后续25次均值。\n'
        text_report+=f'- [Observed] 固定70/30的五个生成种子内均值={genmeans.tolist()}；平均{controlled_mean:.9f}，生成种子均值SD={genmeans.std(ddof=1):.9f}。所有失败种子保留。\n'
        text_report+=f'- [Observed] 平均RMSE与合并MSE后开方不同：最初5次分别为{oldmean:.9f}/{initial_result["sqrt_mean_MSE_different_metric"]:.9f}；受控25次分别为{controlled_mean:.9f}/{controlled_result["sqrt_mean_MSE_different_metric"]:.9f}。也不能先平均模型预测再算RMSE替代正式分数。\n'
        text_report+='- [Code] 正式测试每台发动机只有一条末端窗口，因此单模型的发动机等权等于窗口等权；最近78台训练发动机全部窗口的内部验证与此不同，不能拿其27.64/29.59/31.40直接对论文24.565。\n'
        text_report+='## 当前工作基线与限制\n\n'
        text_report+=f'- [Hypothesis] 建议后续解释差距时以已有明确70/30的共同5×5结果{controlled_mean:.5f}为本地工作参照，对论文24.565的观测差为{controlled_mean-paper_value:.5f}周期（{controlled_mean/paper_value-1:.2%}）。原1.79352保留为历史数值差，不当作已完全对齐协议的结论。既有7×5主汇总不覆盖或删改。\n'
        text_report+='- [Paper] 论文仅说明5次重复取平均，未公开种子、逐次分数、方差，末端取窗及125截断也未明文给出。公开代码口径不能伪装成论文的直接陈述。旧26.36运行没有不可变源码/划分快照，现存80/20默认不能单独证明当时运行一定80/20。\n'
        text_report+='- [Hypothesis] 不能确认差距超过双方多种子波动，亦不能据部分种子较好宣称正式数值已复现。此审计不把旧数值差归因某一因素。\n'
        text_report+='- 决策线：[Observed] 受控26.79180 → [Paper]24.565 → [Observed]+2.22680/+9.06% → [Hypothesis]是否超双方重复波动未知 → [Hypothesis]数值复现仍不确定，严格结论不因本诊断改变。\n'
        text_report+='- [Code] 后续固定259个末端48窗口、标签cap125/输出不截断、70/30生成训练开发划分、预定生成/预测器种子，先逐模型算RMSE再按层级平均；不以官方测试挑轮次。暂停继续在seed43内部验证中寻找低RMSE轮次。本轮无新增任务。\n'
        text_report+=f'- Run `{label(RUN)}/run-manifest.json`；数据 `{label(OUT)}/audit.json`；论文指纹与TableI页面保存在source。Strictmain/A/B结论保持，C推迟，其他扩展未启动。\n'
        REPORT.write_text(text_report)
        for name,expected in m['inputs_sha256'].items():
            p=Path(name) if Path(name).is_absolute() else ROOT/name;assert sha(p)==expected,name
        generated_sources=[RUN/'source/paper-text.txt',RUN/'source/table_i_page7.png']
        m.update(status='complete_partial_protocol_match',completed_at=now(),
            outputs_sha256={label(p):sha(p) for p in list(OUT.glob('*'))+generated_sources+[REPORT] if p.is_file()},
            checks=dict(paper_hash_matches=True,terminal_engine_count=259,original_metric_mean_recomputed=True,
                        controlled25mean_recomputed=True,existing35_saved_prediction_groups_verified=True,unchanged_inputs=True),
            new_training=0,new_sampling=0,new_model_inference=0)
        dump(RUN/'run-manifest.json',m)
        record('complete_partial_protocol_match','- [Observed] PaperTableI target24.565 verified fromsamehashPDF/page7. Localformaltest259engines/terminal48windows (20leftpadded),labelsmin(rawRUL,125),predictionsunclipped;57rawlabels>125. Existing35predictiongroups each259labels verified andRMSErecomputed. Initial26.358518722=mean5single-evaluator runs, gap1.793518722; historictrain/devsplit notfrozen, retained builtindefault80/20 differs frompaper70/30. Controlledcommon5x5 explicitly70/30=26.791804090,gap2.226804090/+9.06%,SDgenmeans2.916905961. Existing7x5headline28.294635405retained separately.\n- [Paper] Only5repetitions/average andsynthetic70/30 explicitlyspecified;terminal/cap125/stage-seeds/rawvarianceunreported. [Hypothesis] Cannot certifyfullprotocolidentity or freeze1.79as currentfullyalignedgap;recommendcontrolledcommon5x5 aslocalworkingreference withunknownsdisclosed. No proofgapexceedsbothsidevariance.\n- Snapshot `'+label(RUN)+'/run-manifest.json`; report `'+label(REPORT)+'`; zero training/sampling/newinference. Seed43internalvalidationcheckpointsearchpaused;noofficialtesttuning. Strictmain/A/Bunchanged,Cdeferred;otherextensionsnotstarted. Auditbudgetclosed.')
        print(json.dumps(dict(status=m['status'],original=initial_result,controlled=controlled_result,terminal=terminal),ensure_ascii=False,indent=2))
    except BaseException:
        m.update(status='failed',failed_at=now(),failure=traceback.format_exc());dump(RUN/'run-manifest.json',m)
        record('failed','- [Observed] Read-only audit failed; preserve manifest/source/logs. No training/sampling/inference ran and no automatic retry.')
        raise


if __name__=='__main__':main()
