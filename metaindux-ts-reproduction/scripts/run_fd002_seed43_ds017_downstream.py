"""Evaluate the existing seed43 epoch67 batch with DS0.1771666 downstream."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/diagnostic-fd002-seed43-ds017-downstream-20261106'
OUT = ROOT/'results/diagnostics/fd002_seed43_ds017_downstream_20261106'
PREDICTORS = ROOT/'checkpoints/evaluators/fd002_seed43_ds017_downstream_20261106'
BASE = ROOT/'runs/diagnostic-fd002-final70-two-seeds-20261102'
PREV = ROOT/'runs/diagnostic-fd002-fixed-seed43-downstream-30-70-20261106'
PREV_OUT = ROOT/'results/diagnostics/fd002_fixed_seed43_downstream_30_70_20261106'
SOURCE_RUN = ROOT/'runs/diagnostic-fd002-seed43-variation-stages-20261105'
SOURCE = ROOT/'results/diagnostics/fd002_seed43_variation_stages_20261105/sampling/native_rerun_b/seed43/phase'
PROTOCOL = ROOT/'docs/operations/FD002_SEED43_DS017_DOWNSTREAM.md'
REPORT = ROOT/'docs/conclusions/final/fd002_seed43_ds017_downstream_20261106_report.md'
SEEDS = [3,13,23,33,43]


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api);return api


def helper():
    h=module(RUN/'source/run_fd002_final70_two_seeds.py','sealed_fd002_ds017_worker')
    h.ROOT,h.RUN,h.OUT=ROOT,RUN,OUT
    h.DOWNSTREAM=RUN/'source/run_fd001_checkpoint_downstream_validation.py'
    raw=h.module
    def local(path,name):
        p=RUN/'source'/Path(path).name
        api=raw(p if p.exists() else path,name)
        if hasattr(api,'ROOT'):api.ROOT=ROOT
        return api
    h.module=local
    h.selected=lambda seed,rule:json.loads((RUN/'config.json').read_text())['selected_weights'][rule]
    h.environment()
    return h


def state(stage,body):
    entry=f'\n## {now()} FD002 seed43 DS0.1772 downstream {stage}\n\n{body}\n'
    p=ROOT/'reproduction-state.md';head,rest=p.read_text().split('\n',1);p.write_text(head+'\n'+entry+'\n'+rest)
    with (ROOT/'experiment-ledger.md').open('a') as f:f.write(entry)


def predictor_source(h):
    source=inspect.getsource(h.downstream)
    old="ROOT / f'checkpoints/evaluators/fd002_final70_two_seeds_20261102/seed{seed}/{rule}'"
    new="ROOT / f'checkpoints/evaluators/fd002_seed43_ds017_downstream_20261106/seed{seed}/{rule}'"
    assert source.count(old)==1
    return source.replace(old,new)


def prepare():
    import numpy as np
    import pandas as pd
    assert not RUN.exists(),'No overwrite'
    for p in [RUN/'source',OUT/'logs',PREDICTORS]:p.mkdir(parents=True,exist_ok=True)
    for p in [Path(__file__),PROTOCOL,PREV/'source/run_fd002_final70_two_seeds.py',PREV/'source/run_fd001_checkpoint_downstream_validation.py']:
        shutil.copy2(p,RUN/'source'/p.name)
    for p,name in [(PREV/'run-manifest.json','previous-downstream-manifest.json'),
                   (SOURCE_RUN/'run-manifest.json','source-sampling-manifest.json'),
                   (BASE/'run-manifest.json','base-manifest.json')]:shutil.copy2(p,RUN/'source'/name)
    parent=json.loads((PREV/'run-manifest.json').read_text());assert parent['status']=='complete'
    source_manifest=json.loads((SOURCE_RUN/'run-manifest.json').read_text());assert source_manifest['status']=='complete'
    record_path=SOURCE/'sample_checkpoint43_seed103.json';record=json.loads(record_path.read_text())
    artifact=ROOT/record['artifact_path']
    assert record['selected_epoch']==67 and sha(artifact)==record['artifact_sha256']
    training_audit=ROOT/'results/diagnostics/fd002_final70_two_seeds_20261102/seed43/training_audit.json'
    selected=json.loads(training_audit.read_text())['rules']['phase']
    assert selected['weight_sha256']==record['checkpoint_sha256']==sha(ROOT/selected['weight_path'])
    ds_path=SOURCE/'evaluation_checkpoint43_sample103.csv';ds=pd.read_csv(ds_path).sort_values('evaluator_seed')
    assert ds.evaluator_seed.tolist()==SEEDS and abs(ds.discriminative_score.mean()-.17716658642272504)<1e-12
    for p in [artifact,record_path,ds_path,SOURCE/'sampling_convolution_flags.json']:
        assert sha(p)==source_manifest['outputs_sha256'][str(p.relative_to(ROOT))],str(p)
    target=OUT/'seed43/phase'
    for part in ['samples','logs','audits','predictions']:(target/part).mkdir(parents=True,exist_ok=True)
    os.link(artifact,target/'samples/checkpoint43_sample103.npz')
    shutil.copy2(record_path,target/record_path.name)
    shutil.copy2(ds_path,target/ds_path.name)
    for name in ['frozen_inputs.npz','frozen_plan.npz','real_validation.npz']:
        p=PREV/name;assert sha(p)==parent['inputs_sha256'][str(p.relative_to(ROOT))]
        os.link(p,RUN/name)
    with np.load(artifact) as z,np.load(RUN/'frozen_inputs.npz') as frozen:
        assert z['data'].shape==(41539,48,14) and np.isfinite(z['data']).all() and np.array_equal(z['label'],frozen['label'])
    with np.load(RUN/'real_validation.npz') as z:
        assert z['data'].shape==(12196,48,14) and len(np.unique(z['engine_id']))==78
    config=dict(selected_weights={'phase':selected},generator_seed=43,selected_epoch=67,
        checkpoint_identity='20261102 original-convolution seed43 rerun epoch67, recently re-sampled with fixed paths; not latest20261105native repeats and not fixed-training30/70',
        reused_sample=str(artifact.relative_to(ROOT)),reused_sample_sha256=record['artifact_sha256'],
        sampling_seed=103,discriminator_score=float(ds.discriminative_score.mean()),
        reused_DS_scores=ds.discriminative_score.tolist(),predictor_args=parent['config']['predictor_args'],
        evaluator_seeds=SEEDS,predictor_epochs=35,predictor_updates=3990,
        validation_engines=parent['config']['validation_engines'],validation_windows=12196,
        primary=parent['config']['primary'],
        comparisons='Reuse fixed-training30/70 downstream references on identical validation data/5predictor seeds/config; descriptive cross-training contrast, not a same-training epoch effect',
        generator_trainings=0,generated_batches=0,new_discriminator_fits=0,predictor_fits=5,
        caveat='Generator/MinMax saw internal validation engines; DS-selected checkpoint and one generated batch, exploratory not independent test',
        stop='Exactly five downstream predictor fits on this specified existing batch; no new generator training, sampling, DS fits or other checkpoints')
    dump(RUN/'config.json',config)
    h=helper();(RUN/'source/predictor_worker.py').write_text(predictor_source(h))
    paths=[p for p in (RUN/'source').iterdir() if p.is_file()]
    paths += [Path(__file__),PROTOCOL,RUN/'config.json',training_audit,ROOT/selected['weight_path'],artifact,record_path,ds_path,SOURCE/'sampling_convolution_flags.json']
    paths += [RUN/name for name in ['frozen_inputs.npz','frozen_plan.npz','real_validation.npz']]
    paths += [target/'samples/checkpoint43_sample103.npz',target/record_path.name,target/ds_path.name]
    for rule in ['best','final70']:
        d=PREV_OUT/'seed43'/rule
        for name in ['per_model_segments.csv','predictor_audits.csv']:
            p=d/name;assert sha(p)==parent['outputs_sha256'][str(p.relative_to(ROOT))];paths.append(p)
        epoch=30 if rule=='best' else 70
        for seed in SEEDS:
            for suffix in ['.json','_orders.npz']:
                p=d/'audits'/f'epoch{epoch}_eval{seed}{suffix}'
                assert sha(p)==parent['outputs_sha256'][str(p.relative_to(ROOT))];paths.append(p)
    inherited={name:value for name,value in parent['inputs_sha256'].items() if not name.startswith('runs/') and not name.startswith('results/diagnostics/')}
    inherited.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    manifest=dict(run_id=RUN.name,scope_class='diagnostic',role='supporting',scope_contract='scope-20261106-fd002-seed43-ds017-downstream',
        user_authorization='Explicit downstream evaluation of the seed43 generator with discrimination score approximately0.17; reuse its exact latest scored batch',
        status='prepared',prepared_at=now(),config=config,inputs_sha256=inherited,
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'run'],completed_predictor_fits=0,
        generator_trainings=0,generated_batches=0,new_discriminator_fits=0,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    for name,command in [('environment-freeze.txt',[sys.executable,'-m','pip','freeze']),('workspace.diff',['git','diff','--binary']),('git-status.txt',['git','status','--short'])]:
        p=RUN/name;p.write_bytes(subprocess.check_output(command,cwd=ROOT));manifest['inputs_sha256'][str(p.relative_to(ROOT))]=sha(p)
    dump(RUN/'run-manifest.json',manifest);h.check_inputs()
    state('prepared','- [Observed] ExactDS0.177166586 batch identified: old native seed43 rerun epoch67, latest fixed-path sampling103,41539windows. User explicitly authorizes five downstream fits; reuse exact batch/DS/78engine12196window validation/synthetic split35epochs3990updates. No generator training, sampling or newDSfits. Reference fixed-training30/70 descriptive only, not epoch effect. Snapshot `'+str(RUN.relative_to(ROOT))+'/run-manifest.json`; protocol `'+str(PROTOCOL.relative_to(ROOT))+'`. Generator/MinMax saw validation engines;exploratory. Metrics pending→no paper target→gapNA→paired consistency unknown→strictmain/A/Bunchanged,Cdeferred;otherdatasets/125paused.')
    print(json.dumps(dict(status='prepared',epoch=67,DS=config['discriminator_score'],predictor_fits=5),ensure_ascii=False),flush=True)


def predictor():
    h=helper();h.check_inputs()
    source=(RUN/'source/predictor_worker.py').read_text();assert source==predictor_source(h)
    exec(compile(source,'<same_fd002_predictor_for_reused_DS017_data>','exec'),h.__dict__)
    h.downstream(43,'phase')


def summarize_and_verify():
    import numpy as np
    import pandas as pd
    h=helper();m=h.check_inputs();target=h.folder(43,'phase')
    metrics=pd.read_csv(target/'per_model_segments.csv');real=metrics[metrics.domain=='real_validation']
    with np.load(RUN/'real_validation.npz') as z:y=z['label'].astype(float);ids=z['engine_id'].copy()
    checked=[]
    for seed in SEEDS:
        a=json.loads((target/'audits'/f'epoch67_eval{seed}.json').read_text())
        assert a['completed_epochs']==35 and a['actual_updates']==3990 and sha(ROOT/a['checkpoint'])==a['checkpoint_sha256']
        with np.load(target/'predictions'/f'epoch67_eval{seed}.npz') as z:p=z['real_validation']
        assert p.shape==y.shape and np.isfinite(p).all()
        mse=np.mean([np.mean((p[ids==engine]-y[ids==engine])**2) for engine in np.unique(ids)])
        row=real[(real.bin=='all')&(real.evaluator_seed==seed)].iloc[0];assert abs(row.mse_engine-mse)<1e-9
        for rule,epoch in [('best',30),('final70',70)]:
            d=PREV_OUT/'seed43'/rule;ref=json.loads((d/'audits'/f'epoch{epoch}_eval{seed}.json').read_text())
            assert ref['initialization_sha256']==a['initialization_sha256'] and ref['consumed_order_sha256']==a['consumed_order_sha256']
            with np.load(target/'audits'/f'epoch67_eval{seed}_orders.npz') as x,np.load(d/'audits'/f'epoch{epoch}_eval{seed}_orders.npz') as b:
                assert x['orders'].shape==b['orders'].shape==(35,29077) and np.array_equal(x['orders'],b['orders'])
        checked.append(dict(evaluator_seed=seed,rmse_engine=float(np.sqrt(mse)),rmse_window=float(row.rmse_window)))
    segments=[];comparisons=[]
    for segment in real.bin.unique():
        f=real[real.bin==segment].sort_values('evaluator_seed');assert f.evaluator_seed.tolist()==SEEDS
        value=float(np.sqrt(f.mse_engine.mean()))
        segments.append(dict(segment=segment,rmse_engine=value,mean_predictor_rmse_engine=float(f.rmse_engine.mean()),
            sd_predictor_rmse_engine=float(f.rmse_engine.std(ddof=1)),rmse_window=float(np.sqrt(f.mse_window.mean()))))
        for rule,epoch in [('best',30),('final70',70)]:
            ref=pd.read_csv(PREV_OUT/'seed43'/rule/'per_model_segments.csv');ref=ref[(ref.domain=='real_validation')&(ref.bin==segment)].sort_values('evaluator_seed')
            baseline=float(np.sqrt(ref.mse_engine.mean()))
            comparisons.append(dict(reference=f'fixed_training_epoch{epoch}',segment=segment,reference_RMSE=baseline,ds017_RMSE=value,
                delta_RMSE=value-baseline,relative_change=value/baseline-1,
                improving_predictor_seeds=int(np.sum(f.rmse_engine.to_numpy()<ref.rmse_engine.to_numpy()))))
    pd.DataFrame(segments).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(checked).to_csv(OUT/'per_predictor_overall.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'reference_comparison.csv',index=False)
    primary=next(x for x in segments if x['segment']=='all')
    result=dict(status='complete',identity=m['config']['checkpoint_identity'],checkpoint_epoch=67,
        reused_DS=m['config']['discriminator_score'],primary=primary,per_predictor=checked,
        reference_overall=[x for x in comparisons if x['segment']=='all'],
        caveat='Internal exploratory validation seen by generator/MinMax; one DS-selected generated batch. Cross-training references do not isolate checkpoint selection or cuDNN effects.')
    dump(OUT/'summary.json',result)
    verification=dict(passed=True,predictor_fits=5,epochs_each=35,updates_each=3990,
        saved_prediction_MSE_recomputed=True,reference_initializations_all35orders_equal=True,
        sample_reused_byte_identically=True,validation_engines=78,validation_windows=12196,
        generator_trainings=0,generated_batches=0,new_discriminator_fits=0)
    dump(RUN/'completion-verification.json',verification)
    text='# FD002 seed43判别分数0.1772生成数据的下游表现\n\n'
    text+=f'- [Observed] 对象为旧原设置重跑seed43第67轮，最近固定采样路径103生成的同一批41539窗口；判别均分{result["reused_DS"]:.9f}。不是20261105新增两次原设置训练，也不是固定训练30/70。\n'
    text+=f'- [Observed] 总体发动机等权RMSE={primary["rmse_engine"]:.6f}；窗口等权RMSE={primary["rmse_window"]:.6f}。5预测器、同一78台真实验证发动机/12196窗口。\n\n'
    text+='| 预测器种子 | 发动机等权RMSE |\n|---|---:|\n'
    for x in checked:text+=f'| {x["evaluator_seed"]} | {x["rmse_engine"]:.6f} |\n'
    text+='\n| 同设置参照 | 参照RMSE | 本次RMSE | 相对变化 | 改善种子 |\n|---|---:|---:|---:|---:|\n'
    for x in result['reference_overall']:text+=f'| {x["reference"]} | {x["reference_RMSE"]:.6f} | {x["ds017_RMSE"]:.6f} | {x["relative_change"]:+.2%} | {x["improving_predictor_seeds"]}/5 |\n'
    text+='\n- [Code] 复用已有生成数据、条件及判别分数；仅5次预测器训练，各35轮/3990更新，生成开发集选模。与参照的初始化、全部35轮顺序、训练预算及真实验证窗口完全匹配。主指标按发动机平均MSE、再平均预测器后开平方。\n'
    text+='- [Hypothesis] 同设置下的差异可描述这批生成数据的实际下游用途，但生成器训练轨迹与轮次都不同，不能将跨训练差异单独归因cuDNN或选模；较低判别分数不保证任何数据集的下游收益。\n'
    text+='- [Observed] 生成器/全局MinMax已见验证发动机；按判别表现点名选择该检查点，结论为探索性内部验证，不访问官方测试集。0生成器重训、0新生成、0新判别拟合。\n'
    text+=f'- Snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`；汇总 `{OUT.relative_to(ROOT)}/summary.json`。当前诊断RMSE→无论文对应目标→差距NA→5预测器/单批次限制→严格结论不变。5拟合预算关闭，不自动补跑。\n'
    REPORT.write_text(text)
    return result


def run():
    h=helper();m=h.check_inputs();assert m['status']=='prepared'
    m.update(status='running',started_at=now(),current_stage='predictor');dump(RUN/'run-manifest.json',m)
    try:
        import torch
        assert torch.cuda.is_available()
        dump(RUN/'gpu-environment.json',dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0)))
        with (OUT/'logs/predictor.log').open('w') as f:
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'predictor'],cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True)
        m['completed_predictor_fits']=5;dump(RUN/'run-manifest.json',m)
        result=summarize_and_verify();h.check_inputs()
        m.update(status='complete',completed_at=now(),current_stage='finished',primary=result['primary'])
        m['outputs_sha256']={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
        m['predictor_checkpoint_sha256']={str(p.relative_to(ROOT)):sha(p) for p in PREDICTORS.rglob('*.pth')}
        m['report_sha256']=sha(REPORT);dump(RUN/'run-manifest.json',m)
        state('complete','- [Observed] ReusedDS0.1772 epoch67batch evaluated with5predictors;0generatortrainings/0newgeneration/0newDSfits. Saved-prediction primaryRMSE and pairing to fixed30/70 reference verified. Summary `'+str(OUT.relative_to(ROOT))+'/summary.json`; report `'+str(REPORT.relative_to(ROOT))+'`. Internal validation exploratory;cross-training reference not single-factor attribution. Budgetclosed;strictmain/A/Bunchanged,Cdeferred;otherdatasets/125paused.')
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    except BaseException:
        m.update(status='failed',failed_at=now(),failure=traceback.format_exc());dump(RUN/'run-manifest.json',m)
        state('failed','- [Observed] Stage stopped; see manifest and predictor logs, preserve all artifacts; no automatic retries.')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','predictor']);a=p.parse_args()
    {'prepare':prepare,'run':run,'predictor':predictor}[a.action]()
