"""Same-training fixed-cuDNN seed43 best30 versus final70 downstream comparison."""
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
RUN = ROOT / 'runs/diagnostic-fd002-fixed-seed43-downstream-30-70-20261106'
OUT = ROOT / 'results/diagnostics/fd002_fixed_seed43_downstream_30_70_20261106'
PREDICTORS = ROOT / 'checkpoints/evaluators/fd002_fixed_seed43_downstream_30_70_20261106'
BASE = ROOT / 'runs/diagnostic-fd002-final70-two-seeds-20261102'
FIXED_RUN = ROOT / 'runs/diagnostic-fd002-fixed-conv-full70-20261105'
FIXED_OUT = ROOT / 'results/diagnostics/fd002_fixed_conv_full70_20261105'
AUDIT = FIXED_OUT / 'repeat1/seed43/training_audit.json'
PROTOCOL = ROOT / 'docs/operations/FD002_FIXED_SEED43_DOWNSTREAM_30_70.md'
REPORT = ROOT / 'docs/conclusions/final/fd002_fixed_seed43_downstream_30_70_20261106_report.md'
RULES = ['best', 'final70']
SEEDS = [3, 13, 23, 33, 43]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api


def helper():
    h = module(RUN/'source/run_fd002_final70_two_seeds.py', 'sealed_fd002_comparison_helper')
    h.ROOT, h.RUN, h.OUT = ROOT, RUN, OUT
    h.CORE = RUN/'source/run_fd001_sampling_condition_dropout_paired.py'
    h.DOWNSTREAM = RUN/'source/run_fd001_checkpoint_downstream_validation.py'
    raw_module = h.module
    def local_module(path, name):
        frozen = RUN/'source'/Path(path).name
        result = raw_module(frozen if frozen.exists() else path, name)
        if hasattr(result, 'ROOT'): result.ROOT = ROOT
        if hasattr(result, 'UPSTREAM'): result.UPSTREAM = ROOT/'upstream'
        return result
    h.module = local_module
    h.selected = lambda seed, rule: json.loads((RUN/'config.json').read_text())['selected_weights'][rule]
    h.train = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('Generator training forbidden in this diagnostic'))
    h.environment()
    return h


def state(stage, body):
    text = f'\n## {now()} FD002 fixed seed43 best30/final70 downstream {stage}\n\n{body}\n'
    p = ROOT/'reproduction-state.md'
    head, tail = p.read_text().split('\n', 1)
    p.write_text(head+'\n'+text+'\n'+tail)
    with (ROOT/'experiment-ledger.md').open('a') as f: f.write(text)


def predictor_source(h):
    source = inspect.getsource(h.downstream)
    old = "ROOT / f'checkpoints/evaluators/fd002_final70_two_seeds_20261102/seed{seed}/{rule}'"
    new = "ROOT / f'checkpoints/evaluators/fd002_fixed_seed43_downstream_30_70_20261106/seed{seed}/{rule}'"
    assert source.count(old) == 1
    return source.replace(old, new)


def prepare():
    import numpy as np
    import torch
    assert not RUN.exists(), 'No overwrite or implicit rerun'
    for p in [RUN/'source', OUT/'logs', PREDICTORS]: p.mkdir(parents=True, exist_ok=True)
    for path in [Path(__file__), PROTOCOL, FIXED_RUN/'source/run_fd002_final70_two_seeds.py',
                 ROOT/'scripts/run_fd001_sampling_condition_dropout_paired.py',
                 ROOT/'scripts/run_fd001_resampling_diagnostic.py',
                 ROOT/'scripts/run_fd001_checkpoint_downstream_validation.py']:
        shutil.copy2(path, RUN/'source'/path.name)
    for origin, name in [(BASE/'run-manifest.json', 'base-parent-manifest.json'),
                         (FIXED_RUN/'run-manifest.json', 'fixed-parent-manifest.json'),
                         (AUDIT, 'selected-training-audit.json')]:
        shutil.copy2(origin, RUN/'source'/name)
    base = json.loads((BASE/'run-manifest.json').read_text())
    parent = json.loads((FIXED_RUN/'run-manifest.json').read_text())
    assert parent['status'] == 'complete'
    a = json.loads(AUDIT.read_text())
    assert a['status'] == 'complete' and a['generator_seed'] == 43
    assert a['completed_epochs'] == 70 and a['actual_optimizer_updates'] == 11410
    assert a['rules']['best']['epoch_one_based'] == 30 and a['rules']['final70']['epoch_one_based'] == 70
    assert a['training_convolution_flags']['benchmark'] is False and a['training_convolution_flags']['deterministic'] is True
    weights = {r:a['rules'][r] for r in RULES}
    for info in weights.values(): assert sha(ROOT/info['weight_path']) == info['weight_sha256']
    states = [torch.load(ROOT/weights[r]['weight_path'],map_location='cpu',weights_only=True) for r in RULES]
    assert states[0].keys() == states[1].keys()
    assert any(not torch.equal(states[0][k], states[1][k]) for k in states[0])
    files = []
    for name in ['frozen_inputs.npz', 'frozen_plan.npz', 'real_validation.npz']:
        origin=BASE/name
        assert sha(origin) == base['inputs_sha256'][str(origin.relative_to(ROOT))]
        os.link(origin, RUN/name); files.append(RUN/name)
    shutil.copy2(FIXED_OUT/'noise_anchor.json', OUT/'noise_anchor.json'); files.append(OUT/'noise_anchor.json')
    with np.load(RUN/'frozen_inputs.npz') as z:
        labels=z['label'].copy(); real=z['real'].copy()
        assert real.shape == (41539,48,14) and labels.shape == (41539,1)
    with np.load(RUN/'frozen_plan.npz') as z:
        ti,di,vi=z['train_indices'],z['dev_indices'],z['real_indices']
        engines=z['validation_engines'].copy()
        assert (len(ti),len(di),len(vi),len(engines)) == (29077,12462,12196,78)
        assert np.array_equal(np.sort(np.r_[ti,di]),np.arange(41539))
    with np.load(RUN/'real_validation.npz') as z:
        assert np.array_equal(z['data'],real[vi]) and np.array_equal(z['label'],labels[vi].ravel())
        assert np.array_equal(np.unique(z['engine_id']),engines)
    config = dict(selected_training=str(AUDIT.relative_to(ROOT)),selected_weights=weights,
        generator_seed=43,sampling_seed=103,sample_count=41539,rules=RULES,
        evaluator_seeds=SEEDS,predictor_args=base['config']['predictor_args'],
        predictor_epochs=35,predictor_updates=3990,synthetic_train=29077,synthetic_dev=12462,
        predictor_checkpoint='Original best synthetic-dev RMSE after zero-based epoch5; full35epoch budget for both arms, no early stop',
        validation_engines=engines.tolist(),validation_windows=12196,
        primary='sqrt(mean_over_5_predictors(mean_over_78_engines(mean_window_squared_error)))',
        supplementary='Five paired predictor results, window-weighted RMSE, existing RUL bins; five-seed DS auxiliary only',
        single_factor='Which saved checkpoint from the SAME fixed-convolution training: minimum-loss epoch30 versus final70',
        sampling='DDPM1000/CFG3/batch256; fixed convolution flags benchmarkFalse/deterministicTrue; retain conditions with legacy RNG consumption',
        noise_control='Every actual normal tensor checked against independent legacy generator, shared start/end/full-schedule anchor across both arms',
        pairing='Same5initializations and all35training permutations, equal3990updates; saved and compared',
        generator_trainings=0,max_full_sample_sets=2,max_predictor_fits=10,max_discriminator_fits=10,
        interpretation='Exploratory within-one-generator-training validation; generator/global MinMax saw all260 engines including78validation engines; no official test access',
        stop='Exactly30vs70; no other epochs/seeds, new generator training, architecture/schedule changes or automatic retries; lower primary RMSE decides direction, report paired consistency without selecting by DS')
    dump(RUN/'config.json',config)
    h=helper()
    (RUN/'source/predictor_worker.py').write_text(predictor_source(h))
    sources = [p for p in (RUN/'source').iterdir() if p.is_file()]
    sources += [Path(__file__),PROTOCOL,RUN/'config.json',AUDIT,AUDIT.with_name('training_loss.csv')]+files
    sources += [ROOT/info['weight_path'] for info in weights.values()]
    inherited = {name:value for name,value in parent['inputs_sha256'].items()
                 if not name.startswith('runs/') and not name.startswith('results/diagnostics/')}
    inherited.update({str(p.relative_to(ROOT)):sha(p) for p in sources})
    manifest=dict(run_id=RUN.name,scope_class='diagnostic',role='supporting',
        scope_contract='scope-20261106-fd002-fixed-seed43-downstream-30-70',status='prepared',prepared_at=now(),config=config,
        user_authorization='Explicit use earlier fixed-cuDNN seed43 best30 and final70 for paired downstream comparison; no generator retraining',
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'run'],inputs_sha256=inherited,
        completed_sampling=0,completed_predictor_fits=0,completed_discriminator_fits=0,generator_trainings=0,
        environment=dict(torch=str(torch.__version__),numpy=str(np.__version__),python=sys.version),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    for name,command in [('environment-freeze.txt',[sys.executable,'-m','pip','freeze']),
                         ('workspace.diff',['git','diff','--binary']),('git-status.txt',['git','status','--short'])]:
        p=RUN/name;p.write_bytes(subprocess.check_output(command,cwd=ROOT));manifest['inputs_sha256'][str(p.relative_to(ROOT))]=sha(p)
    for rule in RULES:
        for part in ['samples','logs','audits','predictions']:(OUT/'seed43'/rule/part).mkdir(parents=True,exist_ok=True)
    dump(RUN/'run-manifest.json',manifest);h.check_inputs()
    state('prepared','- [Observed] User explicitly selects earlier fixed-cuDNN training; same-run best30/final70 weights verified distinct and retained. Reuse existing78engine/12196window internal validation and41539condition plan. Budget0generatortrainings/2sample sets/10pairedpredictors/10auxiliaryDSfits. Fixed sampling and actual noise pairing, same35epochs/3990updates and5predictor seeds.\n- Diagnostic/supporting snapshot `'+str(RUN.relative_to(ROOT))+'/run-manifest.json`; protocol `'+str(PROTOCOL.relative_to(ROOT))+'`. Generator/MinMax saw validation engines; exploratory downstream only. Metrics pending→no paper target→gapNA→paired-seed consistency unknown→strict conclusions unchanged. Strictmain/A/B unchanged,Cdeferred; other datasets/125 remain paused.')
    print(json.dumps(dict(status='prepared',epochs=[30,70],validation_engines=78,validation_windows=12196),ensure_ascii=False),flush=True)


def sample(rule):
    import torch
    h=helper();h.check_inputs()
    sys.path.insert(0,str(ROOT/'upstream'))
    import utils
    old_seed=utils.torch_seed
    def seed(value):
        old_seed(value);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        assert not torch.are_deterministic_algorithms_enabled()
    utils.torch_seed=seed
    h.sample(43,rule)
    assert torch.backends.cudnn.benchmark is False and torch.backends.cudnn.deterministic is True
    dump(h.folder(43,rule)/'sampling_flags.json',dict(benchmark=False,deterministic=True,global_strict=False))


def predictor(rule):
    h=helper();h.check_inputs()
    target=h.folder(43,rule)
    sample_info=json.loads((target/'sample_checkpoint43_seed103.json').read_text())
    assert sha(ROOT/sample_info['artifact_path'])==sample_info['artifact_sha256']
    source=(RUN/'source/predictor_worker.py').read_text();assert source==predictor_source(h)
    exec(compile(source,'<same_fd002_predictor_with_isolated_checkpoint_paths>','exec'),h.__dict__)
    h.downstream(43,rule)


def discriminator(rule):
    h=helper();h.check_inputs();h.evaluate(43,rule)


def summarize(primary_only=False):
    import numpy as np
    import pandas as pd
    h=helper();h.check_inputs()
    allrows=[];paired=[];frame_by_rule={}
    for rule in RULES:
        d=h.folder(43,rule);f=pd.read_csv(d/'per_model_segments.csv')
        real=f[f.domain=='real_validation'];frame_by_rule[rule]=real
        for segment in real.bin.unique():
            group=real[real.bin==segment].sort_values('evaluator_seed')
            assert group.evaluator_seed.tolist()==SEEDS
            allrows.append(dict(rule=rule,epoch=30 if rule=='best' else 70,segment=segment,
                rmse_engine=float(np.sqrt(group.mse_engine.mean())),
                mean_predictor_rmse_engine=float(group.rmse_engine.mean()),
                predictor_sd_engine=float(group.rmse_engine.std(ddof=1)),
                rmse_window=float(np.sqrt(group.mse_window.mean()))))
    comparisons=[]
    for segment in frame_by_rule['best'].bin.unique():
        b=frame_by_rule['best'][frame_by_rule['best'].bin==segment].sort_values('evaluator_seed')
        f=frame_by_rule['final70'][frame_by_rule['final70'].bin==segment].sort_values('evaluator_seed')
        old,new=float(np.sqrt(b.mse_engine.mean())),float(np.sqrt(f.mse_engine.mean()))
        comparisons.append(dict(segment=segment,best30_RMSE=old,final70_RMSE=new,delta_RMSE=new-old,
            relative_change=new/old-1,improving_predictor_seeds=int(np.sum(f.rmse_engine.to_numpy()<b.rmse_engine.to_numpy()))))
        if segment=='all':
            for br,fr in zip(b.to_dict('records'),f.to_dict('records')):
                paired.append(dict(evaluator_seed=int(br['evaluator_seed']),best30_RMSE=br['rmse_engine'],
                    final70_RMSE=fr['rmse_engine'],delta_RMSE=fr['rmse_engine']-br['rmse_engine']))
    pd.DataFrame(allrows).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired_overall.csv',index=False)
    primary=next(r for r in comparisons if r['segment']=='all')
    ds=[]
    for rule in RULES:
        p=h.folder(43,rule)/'evaluation_checkpoint43_sample103.csv'
        if p.exists():
            f=pd.read_csv(p).sort_values('evaluator_seed');assert f.evaluator_seed.tolist()==SEEDS
            ds.append(dict(rule=rule,mean_DS=float(f.discriminative_score.mean()),scores=f.discriminative_score.tolist()))
    verdict='final70_better' if primary['delta_RMSE']<0 else 'final70_worse' if primary['delta_RMSE']>0 else 'equal'
    result=dict(status='primary_complete_auxiliary_pending' if primary_only else 'complete',
                primary=primary,paired_overall=paired,auxiliary_DS=ds,verdict=verdict,
                caveat='One fixed generator training and one paired full generated batch; five predictor initializations are not five generator replications. Internal validation engines seen by generator/MinMax.')
    dump(OUT/'summary.json',result)
    text='# FD002固定cuDNN seed43：最低损失第30轮与第70轮下游对照\n\n'
    text+=f'- [Observed] 主指标：第30轮 {primary["best30_RMSE"]:.6f} → 第70轮 {primary["final70_RMSE"]:.6f}，变化 {primary["delta_RMSE"]:+.6f}（{primary["relative_change"]:+.2%}）；{primary["improving_predictor_seeds"]}/5预测器种子改善。\n'
    text+='- [Code] 主指标为先按发动机平均窗口MSE，再平均78发动机与5预测器，最后开平方；使用同样12196真实内部验证窗口/RUL截断125。预测器选模仅看生成开发集RMSE。\n\n'
    text+='| 预测器种子 | 第30轮数据训练RMSE | 第70轮数据训练RMSE | 变化 |\n|---|---:|---:|---:|\n'
    for r in paired:text+=f'| {r["evaluator_seed"]} | {r["best30_RMSE"]:.6f} | {r["final70_RMSE"]:.6f} | {r["delta_RMSE"]:+.6f} |\n'
    text+='\n- [Observed] 判别分数仅作辅助：'+json.dumps(ds,ensure_ascii=False)+'。\n'
    text+='- [Hypothesis] 若第70轮主指标更低，支持本次训练最低训练损失规则遗漏了下游更有用的已保存权重；若更高，则本次使用最终轮没有收益。单次训练不推广到所有FD002种子，也不因结果增跑其他轮次。\n'
    text+='- [Code] 同一既有训练best30/final70，生成器重训0次；两组41539条件与实际噪声完全配对，固定采样卷积路径；5预测器初始化/35完整训练轮次/3990更新/训练顺序配对。\n'
    text+='- [Observed] 生成器及全局MinMax已见78台验证发动机；这是内部探索性对照，不是整条流程对未见发动机的验证。历史测试结果影响过seed43选择，本轮不访问官方测试集。\n'
    text+=f'- Snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`；汇总 `{OUT.relative_to(ROOT)}/summary.json`。当前诊断RMSE→无论文对应目标→差距NA→五预测器配对/单生成器限制→严格结论不变。预算固定，不按结果扩轮次、种子或训练生成器。\n'
    REPORT.write_text(text)
    return result


def audit_outputs():
    import numpy as np
    import pandas as pd
    h=helper();m=h.check_inputs()
    with np.load(RUN/'frozen_inputs.npz') as z:labels=z['label'].copy()
    with np.load(RUN/'real_validation.npz') as z:y=z['label'].astype(float);ids=z['engine_id'].copy()
    anchor=json.loads((OUT/'noise_anchor.json').read_text())
    checked=0
    for rule in RULES:
        d=h.folder(43,rule);s=json.loads((d/'sample_checkpoint43_seed103.json').read_text())
        assert sha(ROOT/s['artifact_path'])==s['artifact_sha256']
        for key in ['start_cuda_rng_sha256','final_cuda_rng_sha256','legacy_schedule_fingerprint','normal_scalar_count']:
            assert s['noise_audit'][key]==anchor['noise_audit'][key]
        with np.load(ROOT/s['artifact_path']) as z:
            assert z['data'].shape==(41539,48,14) and np.isfinite(z['data']).all() and np.array_equal(z['label'],labels)
        epoch=30 if rule=='best' else 70
        metrics=pd.read_csv(d/'per_model_segments.csv')
        for seed in SEEDS:
            a=json.loads((d/'audits'/f'epoch{epoch}_eval{seed}.json').read_text())
            assert a['completed_epochs']==35 and a['actual_updates']==3990
            assert sha(ROOT/a['checkpoint'])==a['checkpoint_sha256']
            with np.load(d/'predictions'/f'epoch{epoch}_eval{seed}.npz') as z:p=z['real_validation']
            assert p.shape==y.shape and np.isfinite(p).all()
            mse=np.mean([np.mean((p[ids==engine]-y[ids==engine])**2) for engine in np.unique(ids)])
            row=metrics[(metrics.domain=='real_validation')&(metrics.bin=='all')&(metrics.evaluator_seed==seed)].iloc[0]
            assert abs(float(row.mse_engine)-mse)<1e-9
            if rule=='final70':
                bd=h.folder(43,'best');ba=json.loads((bd/'audits'/f'epoch30_eval{seed}.json').read_text())
                assert a['initialization_sha256']==ba['initialization_sha256'] and a['consumed_order_sha256']==ba['consumed_order_sha256']
                with np.load(d/'audits'/f'epoch70_eval{seed}_orders.npz') as f,np.load(bd/'audits'/f'epoch30_eval{seed}_orders.npz') as b:
                    assert f['orders'].shape==b['orders'].shape==(35,29077) and np.array_equal(f['orders'],b['orders'])
            checked+=1
    result=dict(passed=True,predictor_fits_checked=checked,validation_engines=len(np.unique(ids)),validation_windows=len(y),
                actual_sample_noise_anchor_matched=True,labels_matched=True,initialization_orders_updates_matched=True,
                primary_MSE_recomputed_from_saved_predictions=True,generator_trainings=0)
    dump(RUN/'completion-verification.json',result)
    return result


def run():
    h=helper();m=h.check_inputs();assert m['status']=='prepared'
    m.update(status='running',started_at=now());dump(RUN/'run-manifest.json',m)
    def child(action,rule):
        m.update(current_stage=action,current_rule=rule);dump(RUN/'run-manifest.json',m)
        with (OUT/'logs'/f'{action}_{rule}.log').open('w') as f:
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),action,rule],cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True)
        key={'sample':'completed_sampling','predictor':'completed_predictor_fits','discriminator':'completed_discriminator_fits'}[action]
        m[key]+=1 if action=='sample' else 5;dump(RUN/'run-manifest.json',m)
    try:
        import torch
        assert torch.cuda.is_available()
        dump(RUN/'gpu-environment.json',dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True,errors='replace')))
        for rule in RULES:child('sample',rule)
        for rule in RULES:child('predictor',rule)
        primary=summarize(primary_only=True);audit_outputs()
        state('primary_complete','- [Observed] Both paired generated datasets and10predictors complete; primary real-validation result available in `'+str(OUT.relative_to(ROOT))+'/summary.json`. Saved-prediction MSE and all paired initialization/orders/updates verified. Continue only10authorized auxiliaryDSfits; no checkpoint selection by DS.')
        for rule in RULES:child('discriminator',rule)
        result=summarize();audit_outputs();h.check_inputs()
        m.update(status='complete',completed_at=now(),current_stage='finished',primary=result['primary'])
        m['outputs_sha256']={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
        m['predictor_checkpoint_sha256']={str(p.relative_to(ROOT)):sha(p) for p in PREDICTORS.rglob('*.pth')}
        m['report_sha256']=sha(REPORT)
        dump(RUN/'run-manifest.json',m)
        state('complete','- [Observed] User-authorized same-training best30/final70 comparison complete:0generatortrainings/2sample sets/10pairedpredictors/10auxiliaryDSfits. Primary and paired results `'+str(OUT.relative_to(ROOT))+'/summary.json`; report `'+str(REPORT.relative_to(ROOT))+'`; completion verification recorded. Budget closed; no further epochs/seeds/model changes. Internal validation remains exploratory; strictmain/A/Bunchanged,Cdeferred,otherdatasets/125paused.')
    except BaseException:
        m.update(status='failed',failed_at=now(),failure=traceback.format_exc());dump(RUN/'run-manifest.json',m)
        state('failed','- [Observed] Stage stopped; all artifacts and completed primary metrics retained. See manifest and stage logs; no automatic retry or configuration change.')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','sample','predictor','discriminator'])
    p.add_argument('rule',nargs='?',choices=RULES)
    a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='run':run()
    else:
        assert a.rule
        {'sample':sample,'predictor':predictor,'discriminator':discriminator}[a.action](a.rule)
