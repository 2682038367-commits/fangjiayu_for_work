#!/usr/bin/env python3
"""Priority2/3: real DKE96 + linear projector, GPU train and Qwen input audit."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Qwen3Model

from ts_mllm.alignment import initialize_from_teacher, train_alignment
from ts_mllm.mae import SpectrumViTEncoder
from ts_mllm.qwen_cache import require_cuda, seed_everything
from ts_mllm.qwen_multimodal import QwenTextBridge, SpectrumTextProjector, build_dynamic_prompt, compose_multimodal_embeddings
from ts_mllm.rebuilt_data import RebuiltWindowDataset, file_sha256
from ts_mllm.vision import SpectrumTransform
from ts_mllm.knowledge_prompt import KnowledgeWindowDataset, PROFILE, KNOWLEDGE_SHA256, PROMPT_TEMPLATE_SHA256, SOURCE
from ts_mllm.condition_knowledge_prompt import ConditionKnowledgeWindowDataset, CONDITION_PROFILE, CONDITION_TEMPLATE_SHA256
from ts_mllm.figure12_prompt import Figure12WindowDataset, PROFILE as FIGURE12_PROFILE, KNOWLEDGE_SHA256 as FIGURE12_KNOWLEDGE_SHA256, PROMPT_TEMPLATE_SHA256 as FIGURE12_TEMPLATE_SHA256, SOURCE as FIGURE12_SOURCE


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--vision-checkpoint", type=Path)
    parser.add_argument("--rebuilt-data-dir", type=Path)
    parser.add_argument("--prompt-profile", choices=["legacy", PROFILE, CONDITION_PROFILE, FIGURE12_PROFILE], default="legacy")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    args.output_dir = args.output_dir or root / f"artifacts/svlma_alignment/{args.dataset}/stride50_seed{args.seed}"
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    device = require_cuda()
    seed_everything(args.seed)
    data_dir = args.rebuilt_data_dir or root / f"artifacts/data_window40_stride50/split_seed{args.split_seed}/{args.dataset}"
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    if (data_manifest["dataset"] != args.dataset or data_manifest["split_seed"] != args.split_seed
            or data_manifest["train_sample_stride"] != 50 or data_manifest["validation_sample_stride"] != 50):
        raise ValueError("alignment arguments do not match rebuilt data protocol")
    vision_path = args.vision_checkpoint or root / f"artifacts/mae_pretrain/{args.dataset}/seed{args.seed}/best.pt"
    spectrum = SpectrumTransform().to(device)
    vision = SpectrumViTEncoder().to(device).eval().requires_grad_(False)
    vision_state = torch.load(vision_path, map_location="cpu", weights_only=False)
    if vision_state.get("data_manifest_sha256") is not None:
        if vision_state["data_manifest_sha256"] != file_sha256(data_dir / "manifest.json") or vision_state["dataset"] != args.dataset:
            raise ValueError("MAE checkpoint does not match alignment dataset")
    vision.load_state_dict(vision_state["encoder_state"])
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    tokenizer.padding_side = "right"
    qwen = Qwen3Model.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True, dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
    splits = {}
    for split in ("train", "val", "test"):
        dataset = KnowledgeWindowDataset(data_dir, split, root.parent / "data/CMAPSSData") if args.prompt_profile == PROFILE else RebuiltWindowDataset(data_dir, split)
        if args.prompt_profile == CONDITION_PROFILE:
            dataset = ConditionKnowledgeWindowDataset(data_dir, split, root.parent / "data/CMAPSSData")
        if args.prompt_profile == FIGURE12_PROFILE:
            dataset = Figure12WindowDataset(data_dir, split)
        prompts = [dataset[i].get("prompt", build_dynamic_prompt(dataset[i]["x"])) for i in range(len(dataset))]
        if args.prompt_profile != "legacy" and max(len(tokenizer(p)["input_ids"]) for p in prompts) > 512:
            raise ValueError("knowledge prompt exceeds512; refusing silent truncation")
        encoded = tokenizer(prompts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        windows = torch.stack([dataset[i]["x"] for i in range(len(dataset))]).to(device)
        splits[split] = {"ids": encoded["input_ids"].to(device), "mask": encoded["attention_mask"].to(device), "x": windows}
    embedding = qwen.get_input_embeddings()
    adapter = QwenTextBridge(qwen.config.vocab_size).to(device)
    projector = SpectrumTextProjector().to(device)
    initialize_from_teacher(adapter, embedding, splits["train"]["ids"][splits["train"]["mask"].bool()])
    args.output_dir.mkdir(parents=True)
    result = train_alignment(adapter, projector, embedding, splits, args.output_dir, spectrum=spectrum, vision_encoder=vision)
    audits = {}
    with torch.inference_mode():
        for split in ("val", "test"):
            data = splits[split]
            count = 0
            for indices in torch.arange(len(data["ids"]), device=device).split(8):
                visual = vision(spectrum(data["x"][indices]))
                inputs = compose_multimodal_embeddings(visual, data["ids"][indices], projector, adapter)
                mask = torch.cat([torch.ones(len(indices), 1, device=device, dtype=torch.long), data["mask"][indices]], 1)
                output = qwen(inputs_embeds=inputs, attention_mask=mask, use_cache=False).last_hidden_state
                if not torch.isfinite(output).all():
                    raise ValueError("non-finite bridged Qwen output")
                assert inputs.shape[1:] == (513, 1024)
                count += len(indices)
            audits[split] = {"count": count, "qwen_output_shape": [count, 513, 1024], "finite": True, "text_path": "DKE96+position -> linear bridge -> Qwen inputs_embeds"}
    result.update({
        "device": torch.cuda.get_device_name(0), "dataset": args.dataset, "seed": args.seed, "split_seed": args.split_seed,
        "data_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "vision_checkpoint": str(vision_path), "vision_checkpoint_sha256": file_sha256(vision_path),
        "spectrum_training": "uniform-logit initialization, learned online with stride50 training windows; no legacy CNN weights",
        "additional_assumptions": ["local compact pretrained MAE", "softmax sensor fusion trained jointly during alignment; float32 online spectrum", "existing statistics prompt template", "Qwen3-0.6B frozen BF16"],
        "epochs": 30, "batch_size": 128, "learning_rate": 0.002,
        "text_dim": 96, "max_text_tokens": 512, "projector_architecture": "linear", "audits": audits,
        "no_rul_labels_used_for_alignment": True, "full_token_cache_written": False,
    })
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    result["additional_assumptions"] = ["Fig. 12 example-based prompt; aggregation/trend wording assumed" if item == "existing statistics prompt template" and args.prompt_profile == FIGURE12_PROFILE else item for item in result["additional_assumptions"]]
    result.update({"prompt_profile": args.prompt_profile, "knowledge_sha256": FIGURE12_KNOWLEDGE_SHA256 if args.prompt_profile == FIGURE12_PROFILE else KNOWLEDGE_SHA256 if args.prompt_profile != "legacy" else None})
    result.update({"prompt_template_sha256": FIGURE12_TEMPLATE_SHA256 if args.prompt_profile == FIGURE12_PROFILE else CONDITION_TEMPLATE_SHA256 if args.prompt_profile == CONDITION_PROFILE else PROMPT_TEMPLATE_SHA256 if args.prompt_profile == PROFILE else None,
                   "knowledge_sources": [FIGURE12_SOURCE] if args.prompt_profile == FIGURE12_PROFILE else [SOURCE] if args.prompt_profile != "legacy" else []})
    checkpoint.update({"prompt_profile": args.prompt_profile, "knowledge_sha256": result["knowledge_sha256"]})
    checkpoint.update({"prompt_template_sha256": result["prompt_template_sha256"]})
    checkpoint.update({"dataset": args.dataset, "data_manifest_sha256": result["data_manifest_sha256"], "vision_checkpoint": str(vision_path), "vision_checkpoint_sha256": result["vision_checkpoint_sha256"], "seed": args.seed, "split_seed": args.split_seed, "train_sample_stride": 50, "validation_sample_stride": 50})
    torch.save(checkpoint, args.output_dir / "best.pt")
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
