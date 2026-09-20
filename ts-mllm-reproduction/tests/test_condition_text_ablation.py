import copy
import json
import runpy
from pathlib import Path
import pytest


def test_ablation_comparison_requires_same_cache_and_initial_weights():
    root=Path(__file__).resolve().parents[1]
    validate=runpy.run_path(str(root/"scripts/run_condition_text_ablation.py"))["validate_comparison"]
    keys=("dataset","data_manifest_sha256","qwen_cache_manifest_sha256","temporal_checkpoint_sha256",
          "train_stride","validation_stride","epochs","batch_size","learning_rate","temporal_learning_rate",
          "freeze_temporal_epochs","seed","split_seed","target_divisor")
    full={k:1 for k in keys}
    full.update(token_mode="full",model_config={"context_mode":"global_broadcast"},
                initialization={"applied_initial_state_sha256":"same"})
    visual=copy.deepcopy(full);visual["token_mode"]="visual_only"
    visual["model_config"]["global_pooling"]="mean"
    validate(full,visual)
    visual["qwen_cache_manifest_sha256"]="different"
    with pytest.raises(ValueError,match="qwen_cache_manifest_sha256"):
        validate(full,visual)
    visual["qwen_cache_manifest_sha256"]=1
    visual["initialization"]["applied_initial_state_sha256"]="different"
    with pytest.raises(ValueError,match="initial weights"):
        validate(full,visual)
