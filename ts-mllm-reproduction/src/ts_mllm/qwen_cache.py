"""Build frozen-Qwen multimodal token caches from MAE spectrum features."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import prepare_cmapss_data
from .mae import SpectrumViTEncoder, VisionTransformerConfig
from .qwen_multimodal import (
    MultimodalCacheConfig,
    SpectrumTextProjector,
    build_dynamic_prompt,
    select_cache_tokens,
    QwenTextBridge,
    compose_multimodal_embeddings,
)
from .spectrum_cache import CachedSpectrumDataset
from .rebuilt_data import RebuiltWindowDataset, file_sha256
from .vision import SpectrumTransform


def require_cuda() -> torch.device:
    """Fail loudly instead of silently running Qwen on CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the Qwen stage. Run this command in an environment "
            "with the NVIDIA device exposed."
        )
    return torch.device("cuda")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_prompt_batch(inputs: torch.Tensor) -> list[str]:
    return [build_dynamic_prompt(window) for window in inputs]


def tokenize_prompts(
    tokenizer: Any,
    prompts: list[str],
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {name: tensor.to(device) for name, tensor in encoded.items()}


def masked_embedding_mean(
    embedding_layer: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    embeddings = embedding_layer(input_ids).float()
    weights = attention_mask.unsqueeze(-1).to(embeddings.dtype)
    return (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def train_projector(
    *,
    projector: SpectrumTextProjector,
    vision_encoder: SpectrumViTEncoder,
    qwen: nn.Module,
    tokenizer: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: MultimodalCacheConfig,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    output_path: Path,
) -> tuple[list[dict[str, float]], int]:
    """Align each visual prefix to the mean embedding of its dynamic prompt."""
    optimizer = torch.optim.Adam(projector.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_epoch = 0
    embedding_layer = qwen.get_input_embeddings()

    for epoch in range(1, epochs + 1):
        projector.train()
        train_total = 0.0
        train_count = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            prompts = make_prompt_batch(batch["x"])
            encoded = tokenize_prompts(tokenizer, prompts, config.max_text_tokens, device)
            # no_grad keeps the frozen encoders out of autograd while returning
            # ordinary tensors that the trainable projector may consume.
            with torch.no_grad():
                visual = vision_encoder(images)
                target = masked_embedding_mean(
                    embedding_layer, encoded["input_ids"], encoded["attention_mask"]
                )
            optimizer.zero_grad(set_to_none=True)
            prediction = projector(visual.float())
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * len(images)
            train_count += len(images)

        projector.eval()
        val_total = 0.0
        val_count = 0
        with torch.inference_mode():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                prompts = make_prompt_batch(batch["x"])
                encoded = tokenize_prompts(tokenizer, prompts, config.max_text_tokens, device)
                visual = vision_encoder(images)
                target = masked_embedding_mean(
                    embedding_layer, encoded["input_ids"], encoded["attention_mask"]
                )
                loss = loss_function(projector(visual.float()), target)
                val_total += float(loss) * len(images)
                val_count += len(images)
        train_loss = train_total / train_count
        val_loss = val_total / val_count
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        print(
            f"projector epoch={epoch:02d}/{epochs} train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "projector_state": projector.state_dict(),
                    "projector_architecture": projector.architecture,
                    "objective_status": "reproduction_assumption_not_specified_in_paper",
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_mse": val_loss,
                },
                output_path,
            )
    checkpoint = torch.load(output_path, map_location=device, weights_only=False)
    projector.load_state_dict(checkpoint["projector_state"])
    return history, best_epoch


@torch.inference_mode()
def cache_split(
    *,
    split: str,
    loader: DataLoader,
    dataset_length: int,
    output_dir: Path,
    vision_encoder: SpectrumViTEncoder,
    projector: SpectrumTextProjector,
    qwen: nn.Module,
    tokenizer: Any,
    config: MultimodalCacheConfig,
    device: torch.device,
    text_adapter: QwenTextBridge | None = None,
    spectrum_transform: SpectrumTransform | None = None,
    spectrum_output_dir: Path | None = None,
) -> dict[str, Any]:
    token_path = output_dir / f"{split}.npy"
    mask_path = output_dir / f"{split}_mask.npy"
    arrays = {
        "tokens": np.lib.format.open_memmap(
            token_path,
            mode="w+",
            dtype=np.float16,
            shape=(dataset_length, config.cached_tokens, config.qwen_dim),
        ),
        "mask": np.lib.format.open_memmap(
            mask_path,
            mode="w+",
            dtype=np.bool_,
            shape=(dataset_length, config.cached_tokens),
        ),
        "target": np.lib.format.open_memmap(
            output_dir / f"{split}_target.npy", mode="w+", dtype=np.float32, shape=(dataset_length,)
        ),
        "unit": np.lib.format.open_memmap(
            output_dir / f"{split}_unit.npy", mode="w+", dtype=np.int64, shape=(dataset_length,)
        ),
        "cycle": np.lib.format.open_memmap(
            output_dir / f"{split}_cycle.npy", mode="w+", dtype=np.int64, shape=(dataset_length,)
        ),
    }
    spectrum_array = None
    if spectrum_output_dir is not None:
        spectrum_output_dir.mkdir(parents=True, exist_ok=True)
        spectrum_array = np.lib.format.open_memmap(
            spectrum_output_dir / f"{split}.npy", mode="w+", dtype=np.float32,
            shape=(dataset_length, 3, 114, 114),
        )
    offset = 0
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True) if spectrum_transform is None else spectrum_transform(batch["x"].to(device))
        prompts = batch.get("prompt") or make_prompt_batch(batch["x"])
        if "prompt" in batch and max(len(tokenizer(p)["input_ids"]) for p in prompts) > config.max_text_tokens:
            raise ValueError("knowledge prompt exceeds text cap; refusing silent truncation")
        encoded = tokenize_prompts(tokenizer, prompts, config.max_text_tokens, device)
        visual = vision_encoder(images)
        if text_adapter is not None:
            input_embeddings = compose_multimodal_embeddings(
                visual, encoded["input_ids"], projector, text_adapter, dtype=qwen.dtype
            )
        else:
            # Explicit legacy-native-text branch, not an implementation of DKE96.
            visual_prefix = projector(visual.float()).to(dtype=qwen.dtype).unsqueeze(1)
            text_embeddings = qwen.get_input_embeddings()(encoded["input_ids"])
            input_embeddings = torch.cat([visual_prefix, text_embeddings], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones(len(images), 1, dtype=encoded["attention_mask"].dtype, device=device),
                encoded["attention_mask"],
            ],
            dim=1,
        )
        hidden = qwen(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        selected, selected_mask = select_cache_tokens(
            hidden, encoded["attention_mask"], config.cached_text_tokens
        )
        stop = offset + len(images)
        if spectrum_array is not None:
            spectrum_array[offset:stop] = images.float().cpu().numpy()
        arrays["tokens"][offset:stop] = selected.float().cpu().numpy().astype(np.float16)
        arrays["mask"][offset:stop] = selected_mask.cpu().numpy()
        for name in ("target", "unit", "cycle"):
            arrays[name][offset:stop] = batch[name].numpy()
        offset = stop
        if batch_index % 50 == 0 or offset == dataset_length:
            print(f"cache {split}: {offset}/{dataset_length}", flush=True)
    for array in arrays.values():
        array.flush()
    if spectrum_array is not None:
        spectrum_array.flush()
    if offset != dataset_length:
        raise RuntimeError(f"{split}: wrote {offset} rows, expected {dataset_length}")
    return {
        "count": dataset_length,
        "tokens": str(token_path),
        "token_shape": [dataset_length, config.cached_tokens, config.qwen_dim],
        "token_dtype": "float16",
        "spectrum_shape": [dataset_length, 3, 114, 114] if spectrum_array is not None else None,
        "spectrum_dtype": "float32" if spectrum_array is not None else None,
        "valid_token_min": int(np.load(mask_path, mmap_mode="r").sum(axis=1).min()),
        "valid_token_max": int(np.load(mask_path, mmap_mode="r").sum(axis=1).max()),
    }


def audit_cache(output_dir: Path, expected_counts: dict[str, int], config: MultimodalCacheConfig) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split, count in expected_counts.items():
        tokens = np.load(output_dir / f"{split}.npy", mmap_mode="r")
        mask = np.load(output_dir / f"{split}_mask.npy", mmap_mode="r")
        target = np.load(output_dir / f"{split}_target.npy", mmap_mode="r")
        unit = np.load(output_dir / f"{split}_unit.npy", mmap_mode="r")
        cycle = np.load(output_dir / f"{split}_cycle.npy", mmap_mode="r")
        expected_shape = (count, config.cached_tokens, config.qwen_dim)
        if tokens.shape != expected_shape or mask.shape != expected_shape[:2]:
            raise ValueError(f"{split}: unexpected token or mask shape")
        if tokens.dtype != np.float16 or mask.dtype != np.bool_:
            raise ValueError(f"{split}: unexpected cache dtype")
        for start in range(0, count, 16):
            part = tokens[start:start + 16]
            valid = mask[start:start + 16]
            if np.count_nonzero(part[~valid]):
                raise ValueError(f"{split}: nonzero padded token states")
        if target.shape != (count,) or unit.shape != (count,) or cycle.shape != (count,):
            raise ValueError(f"{split}: metadata length mismatch")
        if not np.isfinite(tokens).all() or not np.isfinite(target).all():
            raise ValueError(f"{split}: cache contains non-finite values")
        if not mask[:, 0].all() or not mask.any(axis=1).all():
            raise ValueError(f"{split}: invalid visual/text mask")
        report[split] = {
            "shape": list(tokens.shape),
            "dtype": str(tokens.dtype),
            "finite": True,
            "unique_units": int(len(np.unique(unit))),
            "target_min": float(target.min()),
            "target_max": float(target.max()),
        }
    return report


def run(args: Any) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer, Qwen3Model
    except ImportError as error:
        raise RuntimeError("install transformers with Qwen3 support") from error

    seed_everything(args.seed)
    device = require_cuda()
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite completed cache: {args.output_dir}")
    if args.spectrum_output_dir is not None and (args.spectrum_output_dir / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite completed spectra: {args.spectrum_output_dir}")
    started = time.time()
    config = MultimodalCacheConfig(
        model_name=args.model_name,
        max_text_tokens=args.max_text_tokens,
        cached_text_tokens=args.cached_text_tokens,
    )
    data_manifest = None
    if getattr(args, "prompt_profile", "legacy") != "legacy" and args.rebuilt_data_dir is None:
        raise ValueError("knowledge profile requires verified global window data")
    if args.rebuilt_data_dir is not None:
        if args.alignment_checkpoint is None:
            raise ValueError("rebuilt data route requires DKE96 alignment checkpoint")
        if args.max_text_tokens != 512 or args.cached_text_tokens != 512 or args.projector_architecture != "linear":
            raise ValueError("audited route requires text512, full512 text outputs and linear projector")
        data_manifest = json.loads((args.rebuilt_data_dir / "manifest.json").read_text())
        if (data_manifest["dataset"] != args.dataset or data_manifest["split_seed"] != args.split_seed
                or data_manifest["window_size"] != 40 or data_manifest["train_sample_stride"] != args.train_stride
                or data_manifest["validation_sample_stride"] != args.validation_stride):
            raise ValueError("cache arguments do not match rebuilt data protocol")
        datasets = {name: RebuiltWindowDataset(args.rebuilt_data_dir, name) for name in ("train", "val", "test")}
        if getattr(args, "prompt_profile", "legacy") == "operating_sequence_v2":
            from .knowledge_prompt import KnowledgeWindowDataset
            datasets = {name: KnowledgeWindowDataset(args.rebuilt_data_dir, name, args.data_dir) for name in datasets}
        elif getattr(args, "prompt_profile", "legacy") == "condition_operating_sequence_v1":
            from .condition_knowledge_prompt import ConditionKnowledgeWindowDataset
            datasets = {name: ConditionKnowledgeWindowDataset(args.rebuilt_data_dir, name, args.data_dir) for name in datasets}
    else:
        data = prepare_cmapss_data(
            args.data_dir, args.dataset, seed=args.split_seed,
            train_stride=args.train_stride, validation_stride=args.validation_stride,
        )
        datasets = {
            name: CachedSpectrumDataset(base, args.spectrum_cache_dir / f"{name}.npy")
            for name, base in (("train", data.train), ("val", data.val), ("test", data.test))
        }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        datasets["train"], args.projector_batch_size, shuffle=True, generator=generator,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        datasets["val"], args.projector_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    vision_encoder = SpectrumViTEncoder(VisionTransformerConfig()).to(device)
    vision_state = torch.load(args.vision_checkpoint, map_location="cpu", weights_only=False)
    encoder_key = "encoder_state" if "encoder_state" in vision_state else "vision_encoder_state"
    vision_encoder.load_state_dict(vision_state[encoder_key])
    vision_encoder.eval()
    vision_encoder.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = Qwen3Model.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to(device)
    qwen.eval()
    qwen.requires_grad_(False)
    if qwen.config.hidden_size != config.qwen_dim:
        raise ValueError(f"Qwen hidden size {qwen.config.hidden_size} != {config.qwen_dim}")

    projector = SpectrumTextProjector(
        config.visual_dim, config.qwen_dim, architecture=args.projector_architecture
    ).to(device)
    projector_checkpoint = args.projector_checkpoint or args.output_dir / "projector.pt"
    text_adapter = None
    spectrum_transform = None
    if args.alignment_checkpoint is not None:
        state = torch.load(args.alignment_checkpoint, map_location=device, weights_only=False)
        if state.get("prompt_profile", "legacy") != getattr(args, "prompt_profile", "legacy"):
            raise ValueError("alignment/cache prompt profile mismatch")
        if getattr(args, "prompt_profile", "legacy") != "legacy":
            from .knowledge_prompt import KNOWLEDGE_SHA256, PROMPT_TEMPLATE_SHA256
            from .condition_knowledge_prompt import CONDITION_PROFILE, CONDITION_TEMPLATE_SHA256
            template_hash = CONDITION_TEMPLATE_SHA256 if args.prompt_profile == CONDITION_PROFILE else PROMPT_TEMPLATE_SHA256
            if state.get("knowledge_sha256") != KNOWLEDGE_SHA256 or state.get("prompt_template_sha256") != template_hash:
                raise ValueError("knowledge changed; retrain alignment")
        if state["text_dim"] != 96 or state["projector_architecture"] != "linear":
            raise ValueError("alignment checkpoint does not match DKE96/linear projector")
        if args.rebuilt_data_dir is not None:
            if file_sha256(args.rebuilt_data_dir / "manifest.json") != state["data_manifest_sha256"]:
                raise ValueError("alignment checkpoint was trained with different data manifest")
            if args.dataset != state["dataset"]:
                raise ValueError("alignment checkpoint dataset mismatch")
            if "vision_checkpoint_sha256" in state:
                if file_sha256(args.vision_checkpoint) != state["vision_checkpoint_sha256"]:
                    raise ValueError("alignment checkpoint MAE source mismatch")
            elif Path(state["vision_checkpoint"]).resolve() != args.vision_checkpoint.resolve():
                raise ValueError("alignment checkpoint MAE source path mismatch")
            spectrum_transform = SpectrumTransform().to(device)
            spectrum_transform.load_state_dict(state["spectrum_state"])
            spectrum_transform.eval().requires_grad_(False)
        text_adapter = QwenTextBridge(qwen.config.vocab_size).to(device)
        text_adapter.load_state_dict(state["text_adapter_state"])
        text_adapter.eval().requires_grad_(False)
        projector.load_state_dict(state["projector_state"])
        projector_checkpoint = args.alignment_checkpoint
        history = []
        best_epoch = int(state["epoch"])
    elif args.skip_projector_training:
        state = torch.load(projector_checkpoint, map_location=device, weights_only=False)
        projector.load_state_dict(state["projector_state"])
        history: list[dict[str, float]] = []
        best_epoch = int(state["epoch"])
    else:
        history, best_epoch = train_projector(
            projector=projector,
            vision_encoder=vision_encoder,
            qwen=qwen,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            epochs=args.projector_epochs,
            learning_rate=args.projector_learning_rate,
            output_path=projector_checkpoint,
        )
    projector.eval()
    projector.requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "projector_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    split_reports: dict[str, Any] = {}
    for name in ("train", "val", "test"):
        loader = DataLoader(
            datasets[name], args.inference_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        split_reports[name] = cache_split(
            split=name,
            loader=loader,
            dataset_length=len(datasets[name]),
            output_dir=args.output_dir,
            vision_encoder=vision_encoder,
            projector=projector,
            qwen=qwen,
            tokenizer=tokenizer,
            config=config,
            device=device,
            text_adapter=text_adapter,
            spectrum_transform=spectrum_transform,
            spectrum_output_dir=args.spectrum_output_dir,
        )
    expected_counts = {name: len(dataset) for name, dataset in datasets.items()}
    audit = audit_cache(args.output_dir, expected_counts, config)
    for name, dataset in datasets.items():
        if isinstance(dataset, RebuiltWindowDataset):
            for field in ("unit", "cycle", "target"):
                np.testing.assert_array_equal(np.load(args.output_dir / f"{name}_{field}.npy"), dataset.arrays[field])
            masks = np.load(args.output_dir / f"{name}_mask.npy")
            expected_lengths = []
            for index in range(len(dataset)):
                item = dataset[index]
                encoded = tokenizer(item.get("prompt", build_dynamic_prompt(item["x"])), truncation=True, max_length=512)
                expected_lengths.append(len(encoded["input_ids"]) + 1)
            np.testing.assert_array_equal(masks.sum(1), expected_lengths)
            audit[name]["mapping_and_full_text_length"] = "passed"
    sample_item = datasets["test"][0]
    sample_prompt = sample_item.get("prompt", build_dynamic_prompt(sample_item["x"]))
    manifest = {
        "prompt_profile": getattr(args, "prompt_profile", "legacy"),
        "prompt_template_sha256": state.get("prompt_template_sha256") if args.alignment_checkpoint is not None else None,
        "knowledge_sha256": state.get("knowledge_sha256") if args.alignment_checkpoint is not None else None,
        "dataset": args.dataset,
        "fidelity_status": "partial_alignment_with_explicit_reproduction_assumptions",
        "train_stride": args.train_stride,
        "validation_stride": args.validation_stride,
        "projector_architecture": args.projector_architecture,
        "rebuilt_data_dir": str(args.rebuilt_data_dir) if args.rebuilt_data_dir is not None else None,
        "spectrum_path": "online trained alignment spectrum" if spectrum_transform is not None else "legacy precomputed images",
        "spectrum_output_dir": str(args.spectrum_output_dir) if args.spectrum_output_dir is not None else None,
        "data_manifest_sha256": file_sha256(args.rebuilt_data_dir / "manifest.json") if data_manifest else None,
        "alignment_checkpoint_sha256": file_sha256(projector_checkpoint),
        "vision_checkpoint_sha256": file_sha256(args.vision_checkpoint),
        "qwen_commit_hash": getattr(qwen.config, "_commit_hash", None),
        "cache_file_sha256": {path.name: file_sha256(path) for path in args.output_dir.glob("*.npy")},
        "unresolved": [
            "96-d DKE bridge and alignment use disclosed assumptions; not specified by paper",
            "projector mean-token MSE objective is a reproduction assumption",
            "local compact MAE architecture and pretraining are reproduction assumptions",
            "Qwen3-0.6B version and frozen-backbone policy are reproduction assumptions",
        ],
        "seed": args.seed,
        "split_seed": args.split_seed,
        "device": torch.cuda.get_device_name(0),
        "torch_cuda": torch.version.cuda,
        "qwen_model": args.model_name,
        "qwen_dtype": "bfloat16",
        "qwen_frozen": True,
        "vision_encoder_frozen": True,
        "vision_checkpoint": str(args.vision_checkpoint),
        "projector_checkpoint": str(projector_checkpoint),
        "text_dim": 96 if text_adapter is not None else config.qwen_dim,
        "text_path": "DKE96+position -> linear bridge" if text_adapter is not None else "legacy native Qwen embedding",
        "projector_best_epoch": best_epoch,
        "projector_objective": "MSE to masked mean native teacher embedding (reproduction assumption)",
        "cache_config": asdict(config),
        "cache_policy": "visual prefix + all valid text outputs; padded states zero with mask" if config.cached_text_tokens == config.max_text_tokens else "legacy last text outputs",
        "sample_prompt": sample_prompt,
        "splits": split_reports,
        "audit": audit,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.spectrum_output_dir is not None:
        spectrum_manifest = {
            "dataset": args.dataset, "dtype": "float32", "scale": 1,
            "channel_order": ["RP", "STFT", "Morlet-CWT"],
            "data_manifest_sha256": manifest["data_manifest_sha256"],
            "alignment_checkpoint_sha256": manifest["alignment_checkpoint_sha256"],
            "row_mapping": str(args.output_dir / "manifest.json"),
            "files": {path.name: file_sha256(path) for path in args.spectrum_output_dir.glob("*.npy")},
        }
        (args.spectrum_output_dir / "manifest.json").write_text(json.dumps(spectrum_manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return manifest
