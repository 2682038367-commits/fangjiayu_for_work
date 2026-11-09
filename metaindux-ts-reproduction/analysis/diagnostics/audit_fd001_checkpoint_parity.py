"""Read-only scientific audit of the two existing FD001 checkpoint conditions."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import ast
import hashlib
import json
import math
import os
import platform
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/diagnostic-fd001-checkpoint-parity-20261030'
OUT = ROOT / 'results/diagnostics/fd001_checkpoint_parity_20261030'
PREVIOUS = ROOT / 'runs/diagnostic-fd001-checkpoint-resampling-20261029'
REPORT = ROOT / 'docs/conclusions/final/fd001_checkpoint_parity_20261030_report.md'
SEEDS = [13, 23]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def flags(command):
    return dict(zip(command[2::2], command[3::2]))


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (RUN / 'run-manifest.json').exists(), 'Preserve an existing audit'
    started = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    previous_manifest = json.loads((PREVIOUS / 'run-manifest.json').read_text())
    assert previous_manifest['status'] == 'complete'
    for path, expected in previous_manifest['inputs_sha256'].items():
        assert sha(ROOT / path) == expected, path
    sys.path.insert(0, str(ROOT / 'upstream'))
    from data.CMAPSSDataset import CMAPSSDataset, feature_columns
    from DiffusionFreeGuidence.Unet1D_fre import UNet1D_fre
    cwd = Path.cwd()
    os.chdir(ROOT / 'upstream')
    try:
        dataset = CMAPSSDataset('FD001', 48, [1000])
        real = dataset.get_feature_slice(dataset.get_train_data()).numpy()
        label = dataset.get_label_slice(dataset.get_train_data()).numpy()
    finally:
        os.chdir(cwd)
    with np.load(PREVIOUS / 'frozen_inputs.npz') as archive:
        assert np.array_equal(real, archive['real']) and np.array_equal(label, archive['label'])
    batches_per_epoch = math.ceil(len(real) / 256)
    commands, metadata, histories, rows = [], [], [], []
    records = ['offline-run-20260924_115613-spy08ux1', 'offline-run-20260924_120921-84tnqj21']
    paths = [Path(__file__), PREVIOUS / 'run-manifest.json', PREVIOUS / 'frozen_inputs.npz',
             ROOT / 'upstream/TrainCondition.py', ROOT / 'upstream/MainCondition.py',
             ROOT / 'upstream/data/CMAPSSDataset.py', ROOT / 'upstream/GaussianDiffusion.py',
             ROOT / 'upstream/DiffusionFreeGuidence/Unet1D_fre.py',
             ROOT / 'upstream/DiffusionFreeGuidence/Attetnion_temp.py',
             ROOT / 'upstream/data/train_FD001.txt', ROOT / 'configs/paper_fd001_w48_public_code.yaml',
             ROOT / 'runs/main-w48-public-code-final/run-manifest.yaml',
             ROOT / 'docs/provenance/author_label_zeroing/cfg_public_code_audit.json',
             ROOT / 'docs/provenance/author_label_zeroing/Unet1D_fre.py.public.txt',
             ROOT / 'docs/provenance/author_label_zeroing/TrainCondition.public.py.txt']
    model_arguments = dict(dim=32, dim_mults=(1, 2), cond_drop_prob=.2, channels=14, length=48,
                           frequency_mask_mode='hard_random_quantile', frequency_threshold_init=.25,
                           frequency_mask_temperature=.1)
    for seed, record in zip(SEEDS, records):
        meta_path = ROOT / f'results/runs/FD001_w48_seed{seed}_metadata.json'
        stage_path = ROOT / f'results/runs/FD001_w48_seed{seed}_metrics.generation.json'
        history_path = ROOT / f'logs/FD001_w48_seed{seed}_training_loss.csv'
        weight_path = ROOT / f'checkpoints/FD001_w48_seed{seed}.pth'
        original_data = ROOT / f'outputs/FD001_w48_seed{seed}.npz'
        wandb_path = ROOT / 'upstream/wandb' / record / 'files/wandb-metadata.json'
        meta, stage = json.loads(meta_path.read_text()), json.loads(stage_path.read_text())
        history = pd.read_csv(history_path)
        assert meta['status'] == 'complete' and np.array_equal(history.epoch, np.arange(70))
        assert np.isfinite(history.to_numpy()).all()
        eligible = history[history.epoch > 5]
        best = eligible.loc[eligible.diffusion_loss.idxmin()]
        assert int(best.epoch) == stage['best_epoch']
        assert float(best.diffusion_loss) == stage['best_diffusion_loss']
        with np.load(original_data) as archive:
            assert archive['data'].shape == real.shape and np.array_equal(archive['label'], label)
        offline = json.loads(wandb_path.read_text())
        assert meta['command'][2:] == offline['args']
        command = flags(meta['command'])
        assert command['--dataset'] == 'FD001' and command['--state'] == 'train'
        commands.append(command)
        metadata.append(meta)
        histories.append(history)
        torch.manual_seed(seed)
        model = UNet1D_fre(**model_arguments)
        initial_frequency = {name: tensor.clone() for name, tensor in model.state_dict().items() if name.endswith('threshold_param')}
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=True)
        loading = model.load_state_dict(checkpoint, strict=True)
        assert loading.missing_keys == loading.unexpected_keys == []
        assert all(torch.equal(tensor, model.state_dict()[name]) for name, tensor in checkpoint.items())
        model.eval()
        assert all(not module.training for module in model.modules())
        assert all(torch.equal(initial_frequency[name], checkpoint[name]) for name in initial_frequency)
        rows.append(dict(seed=seed, training_windows=len(real), window=48, channels=14,
                         real_tensor_sha256=digest(real), label_sha256=digest(label),
                         training_epochs=len(history), batches_per_epoch=batches_per_epoch,
                         reconstructed_total_optimizer_updates=len(history)*batches_per_epoch,
                         checkpoint_epoch_zero_based=int(best.epoch), checkpoint_epoch_one_based=int(best.epoch)+1,
                         reconstructed_checkpoint_updates=(int(best.epoch)+1)*batches_per_epoch,
                         checkpoint_rule='epoch > 5 and new minimum epoch-average training diffusion loss',
                         final_epoch_zero_based=69, is_final_epoch_checkpoint=int(best.epoch)==69,
                         checkpoint_loss=float(best.diffusion_loss), state_tensor_count=len(checkpoint),
                         parameter_count=sum(p.numel() for p in model.parameters()),
                         missing_parameters=loading.missing_keys, unexpected_parameters=loading.unexpected_keys,
                         all_loaded_tensors_equal=True, module_count=len(list(model.modules())),
                         all_submodules_eval=True, standard_dropout_modules=sum(isinstance(m, torch.nn.Dropout) for m in model.modules()),
                         conditional_dropout_probability_in_eval=model.cond_drop_prob,
                         frequency_parameters={name: float(checkpoint[name].item()) for name in initial_frequency},
                         frequency_parameters_equal_seeded_initialization=True,
                         optimizer_state_or_counter_saved=False, checkpoint_sha256=sha(weight_path)))
        paths.extend([meta_path, stage_path, history_path, weight_path, original_data, wandb_path])
    ignore = {'--seed', '--model_path', '--syndata_path', '--loss_history_path', '--metrics_path'}
    assert {k: v for k, v in commands[0].items() if k not in ignore} == {k: v for k, v in commands[1].items() if k not in ignore}
    assert metadata[0]['config'] == metadata[1]['config']
    assert np.array_equal(histories[0].learning_rate, histories[1].learning_rate)
    model_source = (ROOT / 'upstream/DiffusionFreeGuidence/Unet1D_fre.py').read_text()
    tree = ast.parse(model_source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'UNet1D_fre')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    cfg = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_with_cond_scale')
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'training' for n in ast.walk(forward))
    assert 'cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)' in ast.get_source_segment(model_source, forward)
    assert 'cond_drop_prob = 1.' in ast.get_source_segment(model_source, cfg)
    public_source = (ROOT / 'docs/provenance/author_label_zeroing/Unet1D_fre.py.public.txt').read_text()
    assert 'cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)' in public_source
    public_forward = public_source.split('    def forward(\n        self,\n        x,\n        time,\n        classes,')[1]
    assert 'self.training' not in public_forward.split("if __name__")[0]
    findings = dict(status='audit_complete_with_findings', recorded_commands_and_config_match_except_seed_paths=True,
                    learning_rate_sequences_equal=True, dataset='FD001 full-data', sensors=feature_columns,
                    preprocessing='Global per-column MinMax fit on full FD001 training data; no operating-condition clustering',
                    labels='Last-cycle RUL of each window, capped at125, float32, shape15931x1',
                    training_update_count_evidence='Reconstructed from70 complete epoch records and unconditional one optimizer.step per batch in source; no original optimizer counter/per-update trace saved.',
                    selected_checkpoint_update_counts_differ=True, both_use_same_best_training_loss_rule=True,
                    state_dict_load_is_strict_and_complete=True, all_submodules_eval=True,
                    custom_condition_dropout_remains_active_in_eval=True,
                    dropout_scope='per-window condition embedding replaced by learned null embedding during conditional CFG branch, probability0.2; distinct from numeric batch RUL zeroing during training',
                    public_code_also_has_eval_independent_condition_dropout=True,
                    original_training_processed_tensor_hashes_separately_saved=False,
                    historical_source_commit='unknown: original upstream snapshot has no separate VCS metadata',
                    conclusions='No asymmetric declared configuration/loading mismatch found. Effective saved training age differs; eval does not disable custom condition dropout. Both warrant attention before attributing contrast to optimization alone.',
                    no_training_sampling_or_forward_pass=True, checkpoints=rows)
    (OUT / 'audit_findings.json').write_text(json.dumps(findings, indent=2) + '\n')
    table = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows]
    pd.DataFrame(table).to_csv(OUT / 'checkpoint_comparison.csv', index=False)
    lines = ['# FD001两检查点实际条件审计（2026-10-30）', '',
             '[Observed] 仅核对已有训练/采样记录、保存的标签、当前与已冻结源代码，并在CPU重建网络以严格加载权重；没有训练、采样、模型前向推理或新指标评价。', '',
             '## 结论', '',
             '[Observed] 未发现两边声明配置、数据任务或参数加载方式不一致。**选中权重的训练年龄不同**：两次完整训练均70轮，但同一最佳训练损失规则分别选择零基epoch63和14，即第64轮和第15轮。它们不是最终70轮权重。', '',
             '[Code] **调用eval()不代表手写随机条件丢弃已关闭**：UNet在forward中直接使用默认cond_drop_prob=0.2，不检查self.training；条件CFG分支未显式覆盖为0。两个检查点采样均受此逻辑影响。已留存的公开代码也有同一行为。不能直接宣称它就是两检查点差距的原因。', '',
             '| 核对项 | seed13 | seed23 | 判断 |', '|---|---|---|---|',
             '| 任务与窗口 | FD001全数据；15931×48×14 | 相同 | 记录一致 |',
             '| 预处理与标签 | 全训练集MinMax；窗口末端RUL截断125 | 相同 | 现存标签逐元素一致 |',
             '| 网络参数量 / 状态张量数 | 526851 / 194 | 526851 / 194 | 结构一致 |',
             '| 完整训练轮数 | 70 | 70 | 两份epoch0–69完整记录 |',
             '| 每轮批次 / 总更新次数 | 63 / 4410 | 63 / 4410 | 按循环和epoch记录重建 |',
             '| 保存模型的轮次（从1计） | **64** | **15** | 不同 |',
             '| 保存模型对应更新次数 | **4032** | **945** | 按保存epoch重建，不是等训练年龄 |',
             '| 保存规则 | epoch>5后，训练扩散损失均值创新低即覆盖 | 相同 | 两边均为best-loss，均非final |',
             '| 完整加载参数 | strict=True；missing/unexpected均为空 | 相同 | 194个张量逐元素匹配 |',
             '| eval标志 | 210个子模块全部False | 相同 | 正确进入评价模式 |',
             '| eval后条件随机丢弃 | 条件分支仍为0.2 | 相同 | 自定义逻辑不受eval控制 |', '',
             '## 数据及执行配置的证据边界', '',
             '[Observed] 两份per-run metadata的config字典完全相同；逐项CLI比较仅seed及输出文件路径不同。2026-09-24两份独立W&B离线metadata也记录相同命令。两份70轮学习率序列逐元素相同，框架/GPU版本记录一致。', '',
             '[Observed] 当前重建的真实训练窗口及标签与2026-10-29冻结输入逐元素一致；原seed13/23生成输出标签也逐元素一致，当前训练文件和源代码哈希与前次冻结记录一致。**原训练时未分别保存两份处理后真实输入张量的哈希**，因此不能把事后重建表述成训练当时逐元素输入已独立封存验证。', '',
             '[Observed] 检查点仅存模型state_dict，不含optimizer状态或逐步计数器。4410与4032/945均是由完整epoch记录、15931样本、batch256、drop_last默认False以及每批一次无条件optimizer.step重建的更新次数，不冒充直接读取的计数器。70轮实际完成记录不意味着所加载权重已经历70轮。', '',
             '## 采样加载与评价模式', '',
             '[Code] TrainCondition.sample先构建同结构模型，应用frequency_threshold=-1（不覆盖），strict默认True加载state_dict，再调用net_model.eval()；随后包装DDPM。原采样不修改已加载参数。六次采样记录的checkpoint哈希与现有文件一致。', '',
             '[Observed] 本次CPU加载两checkpoint均无缺失/多余键；加载后194个状态张量逐元素与文件相同；210个子模块training标志均False。模型没有普通nn.Dropout模块，不能把手写条件随机掩码当作普通Dropout模块处理。', '',
             '[Code] 路径：GaussianDiffusion.model_predictions → forward_with_cond_scale → 首次forward不显式传cond_drop_prob → 使用self.cond_drop_prob=0.2 → 对每个窗口随机把条件嵌入换成null_classes_emb。随后无条件分支显式cond_drop_prob=1，这是CFG正常构造无条件输出的分支。这里不是训练期间整批数值RUL置零，两者须分开。', '',
             '[Code] 证据：`upstream/TrainCondition.py:265`（完整加载）、`:269`（eval）；`upstream/DiffusionFreeGuidence/Unet1D_fre.py:412`（CFG）、`:435`（默认丢弃率）、`:441`（随机条件掩码）；`upstream/GaussianDiffusion.py:207`（采样调用CFG）。留存公开代码：`docs/provenance/author_label_zeroing/Unet1D_fre.py.public.txt:404`和`:427`。', '',
             '## 与种子关联的其他状态', '',
             '[Observed] 两个存储频率分位参数分别为seed13：[0.298574,0.495990]，seed23：[0.074127,0.496258]。在相同网络构造下按各自训练种子初始化，得到的两个参数与保存值逐元素相同。这不是加载遗漏，也不是相同CLI配置保证这些随机初始化参数相同。', '',
             '[Hypothesis] 此次结果支持“完整检查点状态关联的稳定差异”，尚不能拆分训练年龄、随机频率参数及其他已学习权重的贡献。条件丢弃是双方共有采样行为，并非已证实的组间不公平因素；也不能由本次静态审计推断关闭它一定提高判别或下游效果。', '',
             '## 停止点及下一步', '',
             '[Hypothesis] 先明确推理条件分支应保留还是禁用训练式随机条件丢弃。如果做验证，应单独固定检查点和采样设置，仅将条件分支cond_drop_prob从0.2改为0，保留CFG无条件分支=1；该对照需要单独授权，当前未执行。不要同时更换模型或训练轮次。', '',
             '[Observed] 没有可复用的最终70轮生成器权重副本：现有训练代码只保存best-loss覆盖文件；本次只检查目标权重文件和已有保存逻辑，不重新训练、不替换严格复现或历史采样结果。不能把现有best权重当成最终权重进行“同轮”比较。', '',
             '[Observed] 五段比较：当前为已有记录/参数审计 → 论文无对应审计指标 → 数值差距不适用 → 不新增种子或指标 → 严格复现结论不变。仅完成本次审计并停止；零标签方向仍暂停，EXT-002仍保留weight1。', '',
             f'[Observed] Run `{RUN.name}`；机器结果 `results/diagnostics/fd001_checkpoint_parity_20261030/audit_findings.json`；快照 `runs/{RUN.name}/run-manifest.json`。']
    REPORT.write_text('\n'.join(lines) + '\n')
    ended = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    manifest = dict(run_id=RUN.name, scope_class='diagnostic', role='supporting',
                    scope_contract='scope-20261030-fd001-checkpoint-parity', status='complete',
                    command=f'{sys.executable} {Path(__file__)}', started_at=started, completed_at=ended,
                    changed_factor='none: existing record and strict CPU state-loading audit; no forward/training/sampling',
                    python=sys.version, platform=platform.platform(), torch=torch.__version__,
                    inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(paths)},
                    outputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in [*OUT.glob('*'), REPORT] if p.is_file()})
    (RUN / 'run-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(pd.DataFrame(table)[['seed', 'training_epochs', 'reconstructed_total_optimizer_updates',
                              'checkpoint_epoch_one_based', 'reconstructed_checkpoint_updates',
                              'all_loaded_tensors_equal', 'all_submodules_eval', 'conditional_dropout_probability_in_eval']].to_string(index=False))


if __name__ == '__main__':
    main()
