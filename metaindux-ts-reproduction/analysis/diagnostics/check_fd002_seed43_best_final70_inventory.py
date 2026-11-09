"""Read-only same-training best/final70 inventory; stop on an identical pair."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/diagnostic-fd002-seed43-best-final70-inventory-20261106'
OUT = ROOT / 'results/diagnostics/fd002_seed43_best_final70_inventory_20261106'
REPORT = ROOT / 'docs/conclusions/final/fd002_seed43_best_final70_inventory_20261106_report.md'
PROTOCOL = ROOT / 'docs/operations/FD002_SEED43_BEST_FINAL70_INVENTORY.md'
LATEST = ROOT / 'results/diagnostics/fd002_seed43_variation_stages_20261105/training_native'
EARLIER = ROOT / 'results/diagnostics/fd002_fixed_conv_full70_20261105'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def record(stage, body):
    entry = f'\n## {now()} FD002 seed43 best/final70 inventory {stage}\n\n{body}\n'
    p = ROOT / 'reproduction-state.md'
    head, rest = p.read_text().split('\n', 1)
    p.write_text(head + '\n' + entry + '\n' + rest)
    with (ROOT / 'experiment-ledger.md').open('a') as f:
        f.write(entry)


def execute():
    assert not RUN.exists(), 'Preserve existing audit'
    (RUN / 'source').mkdir(parents=True)
    OUT.mkdir(parents=True)
    for source in [Path(__file__), PROTOCOL]:
        shutil.copy2(source, RUN / 'source' / source.name)
    config = dict(primary='The two latest native seed43 full70 trainings from the just-completed stage-attribution run',
                  primary_repeats=['repeat1', 'repeat2'],
                  older_fixed_training='Inventory reference only; do not silently substitute it for either latest training',
                  checkpoint_rule='Original e>5, lowest recorded training diffusion loss',
                  stop='If best and final70 are identical tensors, record no comparison and run zero sampling or predictor fits',
                  user_authorization='Explicit inspect existing best/final70 and compare downstream if distinct; identical epoch70 pair is no comparison',
                  generator_training_budget=0, actual_sampling=0, actual_predictor_fits=0,
                  no_official_test_access=True)
    dump(RUN / 'config.json', config)
    sources = [Path(__file__), PROTOCOL, RUN / 'config.json']
    sources += [p for p in (RUN / 'source').iterdir() if p.is_file()]
    for name, path in [('latest-parent.json', ROOT / 'runs/diagnostic-fd002-seed43-variation-stages-20261105/run-manifest.json'),
                       ('earlier-fixed-parent.json', ROOT / 'runs/diagnostic-fd002-fixed-conv-full70-20261105/run-manifest.json')]:
        shutil.copy2(path, RUN / 'source' / name)
        sources.append(RUN / 'source' / name)
    cells = []
    for category, base in [('latest_native', LATEST), ('earlier_fixed', EARLIER)]:
        for repeat in ['repeat1', 'repeat2']:
            audit_path = base / repeat / 'seed43/training_audit.json'
            audit = json.loads(audit_path.read_text())
            history_path = audit_path.with_name('training_loss.csv')
            sources += [audit_path, history_path]
            for rule in ['best', 'final70']:
                info = audit['rules'][rule]
                p = ROOT / info['weight_path']
                assert sha(p) == info['weight_sha256']
                sources.append(p)
            cells.append((category, repeat, audit_path, history_path, audit))
    manifest = dict(run_id=RUN.name, scope_class='diagnostic', role='supporting',
                    scope_contract='scope-20261106-fd002-seed43-best-final70-inventory',
                    status='running', started_at=now(), config=config,
                    inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(sources)},
                    command=[sys.executable, str(Path(__file__).resolve())],
                    environment=dict(torch=str(torch.__version__), numpy=str(np.__version__),
                                     python=sys.version, platform=platform.platform()),
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    for name, command in [('workspace.diff', ['git', 'diff', '--binary']), ('git-status.txt', ['git', 'status', '--short'])]:
        (RUN / name).write_bytes(subprocess.check_output(command, cwd=ROOT))
    dump(RUN / 'run-manifest.json', manifest)
    record('started', '- [Observed] Read-only inventory freezes latest two native seed43 runs as primary; older fixed pair listed separately. User explicitly says identical best/final70 means no comparison. No new generator training, sampling or predictors before identity audit. Snapshot `' + str(RUN.relative_to(ROOT)) + '/run-manifest.json`. Strict main/A/B unchanged,C deferred; other datasets/125 paused.')
    try:
        rows = []
        for category, repeat, audit_path, history_path, audit in cells:
            history = np.loadtxt(history_path, delimiter=',', skiprows=1)
            assert history.shape == (70, 3) and np.isfinite(history).all()
            eligible = history[history[:, 0] > 5]
            best = eligible[np.argmin(eligible[:, 1])]
            best_epoch = int(best[0]) + 1
            assert best_epoch == audit['rerun_best_epoch_one_based'] == audit['rules']['best']['epoch_one_based']
            assert abs(float(best[1]) - audit['best_loss']) < 1e-12
            states = [torch.load(ROOT / audit['rules'][r]['weight_path'], map_location='cpu', weights_only=True)
                      for r in ['best', 'final70']]
            assert states[0].keys() == states[1].keys()
            different = [k for k in states[0] if not torch.equal(states[0][k], states[1][k])]
            rows.append(dict(category=category, repeat=repeat,
                audit_path=str(audit_path.relative_to(ROOT)),
                best_epoch=best_epoch, global_minimum_epoch=int(history[np.argmin(history[:, 1]), 0])+1,
                best_training_loss=float(best[1]), final70_training_loss=float(history[-1, 1]),
                final70_epoch=audit['rules']['final70']['epoch_one_based'],
                best_weight=audit['rules']['best']['weight_path'], final70_weight=audit['rules']['final70']['weight_path'],
                named_tensors=len(states[0]), different_named_tensors=len(different),
                tensors_identical=not different, serialized_files_identical=audit['rules']['best']['weight_sha256']==audit['rules']['final70']['weight_sha256'],
                forms_comparison=bool(different), training_flags=audit['training_convolution_flags']))
        primary = [r for r in rows if r['category'] == 'latest_native']
        assert len(primary) == 2
        no_comparison = all(r['best_epoch'] == 70 and r['tensors_identical'] for r in primary)
        result = dict(status='complete_no_contrast' if no_comparison else 'complete_distinct_weights_found',
                      primary_scope='latest_native', cells=rows,
                      primary_has_comparison=not no_comparison,
                      decision='Both latest trainings selected epoch70; no paired downstream comparison exists. Stop with zero fits.' if no_comparison else 'Distinct saved weights exist; downstream comparison can proceed within the user scope.',
                      generator_trainings=0, generated_batches=0, predictor_fits=0,
                      older_fixed_pair_substituted=False)
        dump(OUT / 'inventory.json', result)
        with (OUT / 'inventory.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        text = '# FD002 seed43最低损失与第70轮权重核查\n\n'
        text += '- [Code] 主对象是刚完成的两次原设置完整训练；较早固定卷积训练只列为库存，未替换本轮对象。用户明确要求若最低损失在第70轮则记录没有形成对照。\n\n'
        text += '| 训练 | 最低损失轮次 | 最低损失 | 与第70轮权重相同 | 形成对照 |\n|---|---:|---:|---|---|\n'
        for r in rows:
            text += f'| {r["category"]}/{r["repeat"]} | {r["best_epoch"]} | {r["best_training_loss"]:.12f} | {r["tensors_identical"]} | {r["forms_comparison"]} |\n'
        text += '\n- [Observed] 损失历史按原e>5规则与全70轮全局最小值分别复核；best/final70直接逐命名张量比较，不以文件序列化哈希不同误判模型不同。\n'
        text += '- [Observed] 本轮主对象两次均为best70=final70，没有可比较的两个模型。生成器重训0次、生成0批、预测器拟合0次；无下游RMSE差值。按用户停止分支结束，不能将其解释成选模有收益或无收益。\n' if no_comparison else '- [Observed] 主对象存在不同权重，见库存记录。\n'
        text += '- [Observed] 较早固定卷积训练的best30与final70不同，是另一个现成候选对照；本轮未自行切换至该训练，也不新增轮次搜索。\n'
        text += '- [Hypothesis] 本轮无法检验最低损失规则是否遗漏更有用的权重；相同权重不会提供该规则与final70的因果对照。\n'
        text += f'- Snapshot `{RUN.relative_to(ROOT)}/run-manifest.json`；库存 `{OUT.relative_to(ROOT)}/inventory.json`。当前无对照→无论文对应目标→差距NA→无五种子下游结果→严格复现结论不变。固定核查预算结束。\n'
        REPORT.write_text(text)
        for p, expected in manifest['inputs_sha256'].items():
            assert sha(ROOT / p) == expected, p
        manifest.update(status=result['status'], completed_at=now(), outputs_sha256={
            str(p.relative_to(ROOT)): sha(p) for p in [OUT / 'inventory.json', OUT / 'inventory.csv', REPORT]})
        dump(RUN / 'run-manifest.json', manifest)
        record(result['status'], '- [Observed] Both latest native seed43 trainings have global/eligible minimum at epoch70; best/final70 directly tensor-identical. No contrast, user stop condition applied:0generatortrainings/0samplebatches/0predictorfits. Earlier fixed training has distinct30/70weights but is inventory-only. Report `' + str(REPORT.relative_to(ROOT)) + '`; snapshot `' + str(RUN.relative_to(ROOT)) + '/run-manifest.json`. Current no contrast→no paper target→gapNA→downstream comparison unmeasured→strict conclusions unchanged. No further run automatically authorized by inventory.')
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException:
        manifest.update(status='failed', failed_at=now(), failure=traceback.format_exc())
        dump(RUN / 'run-manifest.json', manifest)
        record('failed', '- [Observed] Inventory failed; preserve manifest and source; no model fits or sampling.')
        raise


if __name__ == '__main__':
    execute()
