"""Separate training, sampling and discriminator variability with a bounded budget."""
from pathlib import Path
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
RUN = ROOT / 'runs/diagnostic-fd002-seed43-variation-stages-20261105'
OUT = ROOT / 'results/diagnostics/fd002_seed43_variation_stages_20261105'
WEIGHTS = ROOT / 'checkpoints/diagnostics/fd002_seed43_variation_stages_20261105'
FIXED_RUN = ROOT / 'runs/diagnostic-fd002-fixed-conv-full70-20261105'
FIXED_OUT = ROOT / 'results/diagnostics/fd002_fixed_conv_full70_20261105'
HIST_OUT = ROOT / 'results/diagnostics/fd002_final70_two_seeds_20261102'
PROTOCOL = ROOT / 'docs/operations/FD002_SEED43_VARIATION_STAGES.md'
REPORT = ROOT / 'docs/conclusions/final/fd002_seed43_variation_stages_20261105_report.md'
CELLS = ['historical_a', 'historical_a_repeat', 'native_rerun_b']


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def helper():
    api = module(RUN / 'source/run_fd002_final70_two_seeds.py', 'sealed_common_helper')
    api.ROOT, api.RUN = ROOT, RUN
    api.CORE = RUN / 'source/run_fd001_sampling_condition_dropout_paired.py'
    api.environment()
    return api


def state(stage, body):
    text = f'\n## {helper().now()} FD002 seed43 variation stages {stage}\n\n{body}\n'
    p = ROOT / 'reproduction-state.md'
    head, tail = p.read_text().split('\n', 1)
    p.write_text(head + '\n' + text + '\n' + tail)
    with (ROOT / 'experiment-ledger.md').open('a') as f:
        f.write(text)


def native_source(api):
    source = inspect.getsource(api.train)
    substitutions = [
        ('torch.backends.cudnn.benchmark = False', 'torch.backends.cudnn.benchmark = True'),
        ('torch.backends.cudnn.deterministic = True', 'torch.backends.cudnn.deterministic = False'),
        ("assert actual['benchmark'] is False and actual['deterministic'] is True", "assert actual['benchmark'] is True and actual['deterministic'] is False"),
        ("print('FIXED_TRAIN'", "print('NATIVE_TRAIN'")]
    for old, new in substitutions:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


def prepare():
    import torch
    assert not RUN.exists(), 'No overwrite'
    for p in [RUN / 'source', OUT / 'logs', WEIGHTS]:
        p.mkdir(parents=True, exist_ok=True)
    for origin in [Path(__file__), PROTOCOL, FIXED_RUN / 'source/verify_fd002_fixed_conv_full70.py',
                   FIXED_RUN / 'source/run_fd002_final70_two_seeds.py',
                   ROOT / 'scripts/run_fd001_sampling_condition_dropout_paired.py',
                   ROOT / 'scripts/run_fd001_resampling_diagnostic.py']:
        shutil.copy2(origin, RUN / 'source' / origin.name)
    parent = json.loads((FIXED_RUN / 'run-manifest.json').read_text())
    assert parent['status'] == 'complete'
    for key in ['inputs_sha256', 'outputs_sha256', 'checkpoint_sha256']:
        for name, expected in parent[key].items():
            assert sha(ROOT / name) == expected, name
    shutil.copy2(FIXED_RUN / 'run-manifest.json', RUN / 'source/fixed-parent-manifest.json')
    os.link(FIXED_RUN / 'frozen_inputs.npz', RUN / 'frozen_inputs.npz')
    api = module(RUN / 'source/verify_fd002_fixed_conv_full70.py', 'frozen_full_training_wrapper')
    (RUN / 'source/native_training_function.py').write_text(native_source(api))
    original_info = dict(epoch_one_based=67, weight_path='checkpoints/FD002_w48_seed43.pth',
                         weight_sha256=sha(ROOT / 'checkpoints/FD002_w48_seed43.pth'))
    rerun_info = json.loads((HIST_OUT / 'seed43/training_audit.json').read_text())['rules']['phase']
    cells = {name: dict(original_info if name.startswith('historical_a') else rerun_info) for name in CELLS}
    for rep in ['repeat1', 'repeat2']:
        (OUT / 'training_native' / rep / 'seed43').mkdir(parents=True)
        (WEIGHTS / rep / 'seed43').mkdir(parents=True)
    for cell in CELLS:
        for part in ['samples', 'logs', 'audits']:
            (OUT / 'sampling' / cell / 'seed43/phase' / part).mkdir(parents=True)
        shutil.copy2(FIXED_OUT / 'noise_anchor.json', OUT / 'sampling' / cell / 'noise_anchor.json')
    paths = [p for p in (RUN / 'source').iterdir() if p.is_file()] + [Path(__file__), PROTOCOL]
    paths += [RUN / 'frozen_inputs.npz', ROOT / original_info['weight_path'], ROOT / rerun_info['weight_path']]
    paths += [OUT / 'sampling' / c / 'noise_anchor.json' for c in CELLS]
    for rep in ['repeat1', 'repeat2']:
        for name in ['training_audit.json', 'epoch_state_hashes.json', 'update_losses.csv',
                     'training_loss.csv', 'training_batch_zero_draws.csv', 'training_epoch_orders.npz']:
            paths.append(FIXED_OUT / rep / 'seed43' / name)
    inherited = {name: value for name, value in parent['inputs_sha256'].items()
                 if not name.startswith('runs/') and not name.startswith('results/diagnostics/')}
    inherited.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    config = dict(original_best_epochs={'43': 67}, seed=43, epochs=70, updates_each=11410,
        training_runs=2, training_only_changed_factor='Revert fixed cuDNN convolution settings to original native benchmark=True/deterministic=False',
        training_monitor='Frozen previous full70 wrapper; exact flags/assert/log-label substitutions only',
        cells=cells, sampling_seed=103, sample_count=41539, sampling_only_changed_factor='Fixed convolution selection benchmark=False/deterministic=True for all three cells',
        evaluator_seeds=[3,13,23,33,43], maximum_sample_sets=3, maximum_discriminator_fits=15,
        stages='2 native training repeats; historicalA sample/eval; sameA repeat sample/eval; oldrerunB sample/eval',
        discriminator_repeat='Same A weight, labels, actual noise, fixed sampler; require A arrays bit-identical before interpreting repeated evaluation as exact-same-data control',
        score_difference_screen=dict(mean_absolute_DS_gap=.03, paired_direction_at_least=4),
        stopping='No extra native repeats if equal. If A sampling arrays differ, stop before B and report unresolved sampler variability. No retries, other seeds, downstream fits or default adoption.',
        limitation='Existing historical input/RNG/kernel trajectory unavailable; prospective controls narrow attribution but cannot recover missing historical state.')
    dump(RUN / 'config.json', config)
    inherited[str((RUN / 'config.json').relative_to(ROOT))] = sha(RUN / 'config.json')
    h = helper()
    dump(RUN / 'run-manifest.json', dict(run_id=RUN.name, scope_class='diagnostic', role='supporting',
        scope_contract='scope-20261105-fd002-seed43-variation-stages', status='prepared', prepared_at=h.now(),
        user_authorization='那不能进一步验证吗：further controlled training/sampling/evaluation attribution',
        config=config, inputs_sha256=inherited, command=[sys.executable,'-u',str(Path(__file__).resolve()),'run'],
        completed_training_runs=0, completed_sampling=0, completed_discriminator_fits=0,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=sys.version, torch=str(torch.__version__)))
    for name, command in [('environment-freeze.txt',[sys.executable,'-m','pip','freeze']),
                          ('git-status.txt',['git','status','--short']),('workspace.diff',['git','diff','--binary'])]:
        (RUN / name).write_bytes(subprocess.check_output(command,cwd=ROOT))
    h.check_inputs()
    state('prepared', '- [Observed] User-authorized further attribution; two native70epoch repetitions with frozen monitor, then at most three fixed-path fullsample sets and15DSfits. Reuse completed fixed-training pair as reference. SameA fullsample repeat isolates sampler and discriminator; oldA-vs-oldB under common sampler isolates checkpoint effect. No downstream fits/default changes.\n- Diagnostic/supporting snapshot `' + str(RUN.relative_to(ROOT)) + '/run-manifest.json`; protocol `' + str(PROTOCOL.relative_to(ROOT)) + '`. Metrics pending→no paper target→gapNA→variability unresolved→strict conclusions unchanged. Strict main/A/B unchanged,C deferred; other datasets/125 paused.')


def train(rep):
    helper().check_inputs()
    api = module(RUN / 'source/verify_fd002_fixed_conv_full70.py', 'exact_training_monitor')
    api.ROOT, api.RUN = ROOT, RUN
    api.OUT, api.WEIGHTS = OUT / 'training_native', WEIGHTS
    source = (RUN / 'source/native_training_function.py').read_text()
    assert source == native_source(api)
    exec(compile(source, '<native_path_previous_training_monitor>', 'exec'), api.__dict__)
    api.train(rep)
    p = OUT / 'training_native' / rep / 'seed43/training_audit.json'
    record = json.loads(p.read_text())
    record['caveat'] = 'Original native convolution selection; no repeatability claim before paired comparison.'
    dump(p, record)


def compare_training():
    import numpy as np
    paths = [OUT / 'training_native' / r / 'seed43' for r in ['repeat1','repeat2']]
    audits = [json.loads((p/'training_audit.json').read_text()) for p in paths]
    losses = [np.loadtxt(p/'update_losses.csv',delimiter=',',skiprows=1) for p in paths]
    assert all(a.shape == (11410,2) for a in losses)
    difference = np.flatnonzero(losses[0][:,1] != losses[1][:,1])
    epochs = [json.loads((p/'epoch_state_hashes.json').read_text()) for p in paths]
    assert all(len(e)==70 for e in epochs)
    flags_ok = all(e['flags']['benchmark'] and not e['flags']['deterministic'] and not e['flags']['global_strict'] for es in epochs for e in es)
    assert flags_ok
    checks = {}
    for name in ['training_loss.csv','training_batch_zero_draws.csv','epoch_state_hashes.json']:
        checks[name] = paths[0].joinpath(name).read_bytes() == paths[1].joinpath(name).read_bytes()
    with np.load(paths[0]/'training_epoch_orders.npz') as a,np.load(paths[1]/'training_epoch_orders.npz') as b:
        checks['all70_orders_equal'] = np.array_equal(a['orders'],b['orders'])
    checks['initial_parameters_equal'] = audits[0]['initial_tensor_sha256']==audits[1]['initial_tensor_sha256']
    checks['all70_rng_states_equal'] = all(all(a[k]==b[k] for k in ['cpu_rng_sha256','cuda_rng_sha256','numpy_rng_sha256']) for a,b in zip(*epochs))
    checks['all70_parameter_hashes_equal'] = all(a['model_sha256']==b['model_sha256'] for a,b in zip(*epochs))
    checks['all70_optimizer_hashes_equal'] = all(a['optimizer_sha256']==b['optimizer_sha256'] for a,b in zip(*epochs))
    checks['all11410_losses_equal'] = len(difference)==0
    for rule in ['best','phase','final70']:
        checks[rule+'_tensor_hash_equal']=audits[0]['rules'][rule]['tensors_sha256']==audits[1]['rules'][rule]['tensors_sha256']
    paired_to_fixed = []
    for p,a in zip(paths,audits):
        fixed = FIXED_OUT/'repeat1/seed43'
        f=json.loads((fixed/'training_audit.json').read_text())
        with np.load(p/'training_epoch_orders.npz') as x,np.load(fixed/'training_epoch_orders.npz') as y:
            order_equal=np.array_equal(x['orders'],y['orders'])
        paired_to_fixed.append(dict(initial_equal=a['initial_tensor_sha256']==f['initial_tensor_sha256'],
            all70_orders_equal=bool(order_equal),batch_zero_draws_equal=(p/'training_batch_zero_draws.csv').read_bytes()==(fixed/'training_batch_zero_draws.csv').read_bytes()))
    first_epoch=next((i+1 for i in range(70) if epochs[0][i]['model_sha256']!=epochs[1][i]['model_sha256']),None)
    result=dict(checks=checks,first_different_loss_update=int(difference[0])+1 if len(difference) else None,
                first_different_parameter_epoch=first_epoch, native_pair_exact=all(checks.values()),
                paired_to_fixed=paired_to_fixed, fixed_pair_exact=True,
                interpretation='Two prospective native runs only; equal outcome does not disprove previously observed native kernel variability, and no extra attempts are allowed.')
    dump(OUT/'training_comparison.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def sample(cell):
    import torch
    h=helper();m=h.check_inputs()
    sys.path.insert(0,str(ROOT/'upstream'))
    import utils
    raw_seed=utils.torch_seed
    def seed(value):
        raw_seed(value)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        assert not torch.are_deterministic_algorithms_enabled()
    utils.torch_seed=seed
    h.OUT=OUT/'sampling'/cell
    h.selected=lambda seed_value,rule: m['config']['cells'][cell]
    h.sample(43,'phase')
    dump(h.folder(43,'phase')/'sampling_convolution_flags.json',dict(benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.backends.cudnn.deterministic,global_strict=torch.are_deterministic_algorithms_enabled()))
    assert torch.backends.cudnn.benchmark is False and torch.backends.cudnn.deterministic is True


def evaluate(cell):
    h=helper();h.check_inputs()
    worker=module(RUN/'source/run_fd001_resampling_diagnostic.py','frozen_ds_worker')
    worker.ROOT,worker.UPSTREAM,worker.RUN=ROOT,ROOT/'upstream',RUN
    worker.OUT=OUT/'sampling'/cell/'seed43/phase'
    worker.worker_evaluate(43,103)


def sample_path(cell):
    return OUT/'sampling'/cell/'seed43/phase/samples/checkpoint43_sample103.npz'


def sample_repeat_gate():
    import numpy as np
    with np.load(sample_path('historical_a')) as a,np.load(sample_path('historical_a_repeat')) as b:
        data_equal=np.array_equal(a['data'],b['data'])
        result=dict(data_bit_identical=bool(data_equal),labels_bit_identical=bool(np.array_equal(a['label'],b['label'])),
            different_elements=int(np.count_nonzero(a['data']!=b['data'])),max_abs=float(np.max(np.abs(a['data']-b['data']))))
    dump(OUT/'sampling_repeatability.json',result)
    return result['data_bit_identical'] and result['labels_bit_identical']


def summarize():
    import numpy as np
    import pandas as pd
    helper().check_inputs()
    rows=[]
    for cell in CELLS:
        path=OUT/'sampling'/cell/'seed43/phase/evaluation_checkpoint43_sample103.csv'
        if not path.exists():continue
        f=pd.read_csv(path).sort_values('evaluator_seed')
        assert f.evaluator_seed.tolist()==[3,13,23,33,43]
        rows.append(dict(cell=cell,mean_DS=float(f.discriminative_score.mean()),scores=f.discriminative_score.tolist(),source=str(path.relative_to(ROOT))))
    sample_repeat=json.loads((OUT/'sampling_repeatability.json').read_text())
    train_result=json.loads((OUT/'training_comparison.json').read_text())
    repeat_scores=None;weight_effect=None
    bycell={row['cell']:row for row in rows}
    if all(c in bycell for c in ['historical_a','historical_a_repeat']):
        x,y=[np.array(bycell[c]['scores']) for c in ['historical_a','historical_a_repeat']]
        repeat_scores=dict(exact_equal=bool(np.array_equal(x,y)),max_abs_difference=float(np.max(abs(x-y))),
                           interpretation='Exact same-data evaluator repeat only if the sample array equality gate passes.')
        dump(OUT/'evaluation_repeatability.json',repeat_scores)
    if all(c in bycell for c in ['historical_a','native_rerun_b']):
        x,y=[np.array(bycell[c]['scores']) for c in ['historical_a','native_rerun_b']]
        delta=float(np.mean(y-x))
        direction=int(np.count_nonzero(y<x)) if delta<0 else int(np.count_nonzero(y>x))
        weight_effect=dict(mean_B_minus_A=delta,paired_direction_count=direction,
            descriptive_screen_passed=bool(abs(delta)>=.03 and direction>=4),
            caveat='Five paired discriminator seeds, one generated batch per checkpoint; descriptive screen, not independent generator replications or full historical causal recovery.')
    result=dict(training=train_result,sampling=sample_repeat,evaluation_repeat=repeat_scores,discriminative_scores=rows,checkpoint_effect=weight_effect)
    dump(OUT/'summary.json',result)
    text='# FD002 seed43训练、采样、判别三阶段验证\n\n'
    text+=f'- [Observed] 两次新增原生70轮训练是否完全一致：{train_result["native_pair_exact"]}；首次损失不同的更新：{train_result["first_different_loss_update"]}；首次参数不同的轮次：{train_result["first_different_parameter_epoch"]}。此前固定设置两次完整训练一致。\n'
    text+=f'- [Observed] 相同旧权重、相同实际噪声的两次固定路径完整采样：{sample_repeat}。\n'
    text+=f'- [Observed] 重复判别评价：{repeat_scores}。\n'
    for r in rows:text+=f'- [Observed] {r["cell"]}: meanDS={r["mean_DS"]:.9f}, five={r["scores"]}。\n'
    text+=f'- [Observed] 共同采样与评价设置下旧权重效应：{weight_effect}。\n'
    text+='- [Hypothesis] 若采样/评价重复通过且旧权重分数仍不同，支持权重本身承载差距；结合先前卷积内核实验及本次完整训练对照可收缩原因。缺失历史逐步状态，不能重建历史每一次随机/计算选择，也不能声称唯一解释全部历史分数。\n'
    text+='- [Code] 每个阶段的对照只改一个因素；无下游预测器训练、其他种子或默认值修改。原生重复未分叉时报告零结果，不继续抽进程寻找分叉。\n'
    text+=f'- Run `{RUN.relative_to(ROOT)}/run-manifest.json`，结果 `{OUT.relative_to(ROOT)}/summary.json`。诊断→无论文对应目标→差距NA→两次本地训练/单生成批次限制→严格复现结论不变。固定预算结束。\n'
    REPORT.write_text(text)
    return result


def run():
    h=helper();m=h.check_inputs();assert m['status']=='prepared'
    m.update(status='running',started_at=h.now());dump(RUN/'run-manifest.json',m)
    def child(action,arg):
        m.update(current_stage=action,current_cell=arg);dump(RUN/'run-manifest.json',m)
        with (OUT/'logs'/f'{action}_{arg}.log').open('w') as f:
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),action,arg],cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True)
        key={'train':'completed_training_runs','sample':'completed_sampling','evaluate':'completed_discriminator_fits'}[action]
        m[key]+=5 if action=='evaluate' else 1
        dump(RUN/'run-manifest.json',m)
    try:
        import torch
        assert torch.cuda.is_available()
        dump(RUN/'gpu-environment.json',dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True,errors='replace')))
        for rep in ['repeat1','repeat2']:child('train',rep)
        compare_training()
        state('training_complete','- [Observed] Two native70epoch repetitions complete; detailed comparison at results/diagnostics/fd002_seed43_variation_stages_20261105/training_comparison.json. Continue only preauthorized fixed-path old-checkpoint sampling/evaluation budget; no other runs.')
        for cell in ['historical_a','historical_a_repeat']:
            child('sample',cell)
            if cell=='historical_a_repeat' and not sample_repeat_gate():
                m['status']='complete_sampler_repeat_failed'
                break
            child('evaluate',cell)
        else:
            child('sample','native_rerun_b');child('evaluate','native_rerun_b')
            m['status']='complete'
        summarize();h.check_inputs()
        m.update(current_stage='finished',completed_at=h.now())
        m['outputs_sha256']={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
        m['checkpoint_sha256']={str(p.relative_to(ROOT)):sha(p) for p in WEIGHTS.rglob('*.pth')}
        dump(RUN/'run-manifest.json',m)
        state(m['status'],f'- [Observed] Bounded stage-attribution diagnosis finished. Report `{REPORT.relative_to(ROOT)}`; snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`; no automatic further work. Strict main/A/B unchanged,C deferred; other datasets/125 paused.')
    except BaseException:
        m.update(status='failed',failed_at=h.now(),failure=traceback.format_exc());dump(RUN/'run-manifest.json',m)
        state('failed','- [Observed] Stage stopped without automatic retries; preserve run manifest, outputs and stage logs. No additional training authorized by failure handling.')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','train','sample','evaluate'])
    parser.add_argument('cell',nargs='?')
    a=parser.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='run':run()
    elif a.action=='train':
        assert a.cell in ['repeat1','repeat2'];train(a.cell)
    else:
        assert a.cell in CELLS
        (sample if a.action=='sample' else evaluate)(a.cell)
