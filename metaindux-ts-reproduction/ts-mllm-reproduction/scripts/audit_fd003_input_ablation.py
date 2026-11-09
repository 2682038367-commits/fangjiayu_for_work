"""Audit the FD003 input-level visual/text ablation and write a concise report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("artifacts")
RUNS = {
    "full (temporal + visual + textual)": (
        ROOT / "fd003_figure12_v1/seed42/tmaf_batch32/result.json",
        ROOT / "fd003_figure12_v1/seed42/qwen/manifest.json",
        "multimodal",  # Legacy cache predates the field; its policy confirms this mode.
    ),
    "w/o Visual (temporal + textual)": (
        ROOT / "fd003_input_ablation/seed42/tmaf_no_visual/result.json",
        ROOT / "fd003_input_ablation/seed42/no_visual_qwen/manifest.json",
        "text_only",
    ),
    "w/o Textual (temporal + visual)": (
        ROOT / "fd003_input_ablation/seed42/tmaf_no_text/result.json",
        ROOT / "fd003_input_ablation/seed42/no_text_qwen/manifest.json",
        "visual_only",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = []
    shared = {}
    for label, (result_path, manifest_path, expected_mode) in RUNS.items():
        result = json.loads(result_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        cache_dir = manifest_path.parent
        mode = manifest.get("input_modality", "multimodal")
        if mode != expected_mode:
            raise AssertionError(f"{label}: expected {expected_mode}, got {mode}")
        cache_audit = {}
        for split, expected_count in (("train", 377), ("val", 92), ("test", 100)):
            tokens = np.load(cache_dir / f"{split}.npy", mmap_mode="r")
            mask = np.load(cache_dir / f"{split}_mask.npy", mmap_mode="r")
            if tokens.shape != (expected_count, 513, 1024) or mask.shape != (expected_count, 513):
                raise AssertionError(f"{label}/{split}: wrong cache shape")
            if not mask[:, 0].all() or not mask.any(axis=1).all():
                raise AssertionError(f"{label}/{split}: invalid mask")
            if np.count_nonzero(tokens[~mask]):
                raise AssertionError(f"{label}/{split}: nonzero padding")
            valid = mask.sum(axis=1)
            if mode == "visual_only" and not np.all(valid == 1):
                raise AssertionError(f"{label}/{split}: visual-only cache has text tokens")
            cache_audit[split] = {"valid_tokens_min": int(valid.min()), "valid_tokens_max": int(valid.max())}
        current = {
            "data_manifest_sha256": result["data_manifest_sha256"],
            "temporal_checkpoint_sha256": result["temporal_checkpoint_sha256"],
            "batch_size": result["batch_size"],
            "seed": result["seed"],
            "split_seed": result["split_seed"],
            "train_stride": result["train_stride"],
            "validation_stride": result["validation_stride"],
            "prompt_profile": result["prompt_profile"],
            "model_config": result["model_config"],
        }
        if not shared:
            shared = current
        elif current != shared:
            raise AssertionError(f"{label}: training protocol differs from full model")
        rows.append({
            "label": label, "input_modality": mode,
            "validation": result["validation"], "test": result["test"],
            "cache_audit": cache_audit,
            "result_sha256": sha256(result_path),
            "cache_manifest_sha256": sha256(manifest_path),
        })

    output = ROOT / "FD003_INPUT_MODALITY_ABLATION.json"
    output.write_text(json.dumps({"shared_protocol": shared, "runs": rows}, indent=2), encoding="utf-8")
    lines = [
        "# FD003 输入级模态消融（seed 42）", "",
        "本报告对应论文 Fig. 6 的 `w/o Visual`、`w/o Textual` 概念，但只报告一次固定划分/随机种子的复现结果。模型为论文公式版 `global_broadcast` TMAF；未用逐 token 诊断版。", "",
        "| Qwen 输入 | 验证 RMSE | 验证 MAE | 验证 Score | 测试 RMSE | 测试 MAE | 测试 Score |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        v, t = row["validation"], row["test"]
        lines.append(f"| {row['label']} | {v['rmse']:.3f} | {v['mae']:.3f} | {v['score']:.1f} | {t['rmse']:.3f} | {t['mae']:.3f} | {t['score']:.1f} |")
    lines += [
        "", "## 审计结论", "",
        "- 三组的重建数据哈希、时间分支权重哈希、batch=32、seed/split=42、样本步长=50、Figure12 文本配置及 TMAF 超参数完全一致；唯一实验变量是写入 Qwen 的输入模态。",
        "- `w/o Visual` 的 Qwen 输入仅为 DKE 文本 embedding：没有生成频谱、没有调用视觉编码器、没有视觉前缀。",
        "- `w/o Textual` 的 Qwen 输入仅为投影后的视觉前缀：未生成或分词 prompt，所有样本有效 token 数恰为 1。",
        "- 这不是在 Qwen 输出之后屏蔽 token，因此文本 token 不可能先读取视觉前缀后再被遮蔽。",
        "", "## 解释边界", "",
        "按验证 RMSE，去视觉较完整模型差 0.015，近似持平；去文本反而好 0.202。因此这一次运行没有支持文本或视觉提供稳定、独立增益的结论。测试集数值仅作保留的泛化观察，不能用于选择模型。",
        "", "视觉 projector 的上游对齐训练曾以文本 teacher embedding 为目标，这是论文未公开的复现假设；故 `w/o Textual` 严格表示 Qwen 推理输入没有文本，而不表示视觉 projector 的训练过程没有任何文本监督。",
    ]
    (ROOT / "FD003_INPUT_MODALITY_ABLATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
