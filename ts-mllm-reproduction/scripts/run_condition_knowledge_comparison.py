"""Condition Min-Max: old text versus ordered operating knowledge bundle."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from ts_mllm.rebuilt_data import file_sha256
from ts_mllm.knowledge_prompt import KNOWLEDGE_SHA256
from ts_mllm.condition_knowledge_prompt import CONDITION_PROFILE, CONDITION_TEMPLATE_SHA256


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["FD002", "FD004"], default=["FD002", "FD004"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for ds in args.datasets:
        old_dir = root / f"artifacts/tmaf_condition_minmax/{ds}/global_broadcast_stride50_B_seed42"
        old = json.loads((old_dir / "result.json").read_text())
        data, temporal = Path(old["rebuilt_data_dir"]), Path(old["temporal_checkpoint"])
        dh = file_sha256(data / "manifest.json")
        if dh != old["data_manifest_sha256"] or file_sha256(temporal) != old["temporal_checkpoint_sha256"]:
            raise ValueError("condition baseline sources changed")
        mae = root / f"artifacts/mae_condition_minmax/{ds}/stride50_seed42/best.pt"
        mae_result = json.loads((mae.parent / "result.json").read_text())
        if mae_result["data_manifest_sha256"] != dh:
            raise ValueError("MAE/data source mismatch")
        old_cache = Path(old["qwen_cache"])
        if file_sha256(old_cache / "manifest.json") != old["qwen_cache_manifest_sha256"]:
            raise ValueError("old baseline cache changed")
        if json.loads((old_cache / "manifest.json").read_text())["vision_checkpoint_sha256"] != file_sha256(mae):
            raise ValueError("old/new MAE weights differ")
        base = root / f"artifacts/condition_knowledge_sequence_v1/{ds}/seed42"
        align,cache,spectra,out = [base/name for name in ("alignment","qwen","spectrum","tmaf")]
        common = ["--dataset",ds,"--rebuilt-data-dir",str(data)]
        stages = [
            ("alignment", "train_svlma_alignment.py", [*common,"--vision-checkpoint",str(mae),"--output-dir",str(align),"--prompt-profile",CONDITION_PROFILE],align/"result.json"),
            ("cache", "cache_qwen_multimodal.py", [*common,"--vision-checkpoint",str(mae),"--alignment-checkpoint",str(align/"best.pt"),"--output-dir",str(cache),"--spectrum-output-dir",str(spectra),"--prompt-profile",CONDITION_PROFILE],cache/"manifest.json"),
            ("TMAF", "train_tmaf.py", [*common,"--cache-dir",str(cache),"--temporal-checkpoint",str(temporal),"--output-dir",str(out),
                "--context-mode","global_broadcast","--global-pooling","mean","--token-mode","full","--epochs","30","--batch-size","128",
                "--learning-rate","0.002","--temporal-learning-rate","0.002","--freeze-temporal-epochs","0","--output-bias-init","default",
                "--train-stride","50","--validation-stride","50","--seed","42","--split-seed","42","--num-workers","2"],out/"result.json")]
        for name,script,flags,completed in stages:
            if completed.exists():
                r=json.loads(completed.read_text())
                if r["dataset"]!=ds or r["data_manifest_sha256"]!=dh or r.get("prompt_profile")!=CONDITION_PROFILE:
                    raise ValueError(f"completed source/profile mismatch: {completed}")
                if name in ("alignment","cache"):
                    if r.get("knowledge_sha256")!=KNOWLEDGE_SHA256 or r.get("prompt_template_sha256")!=CONDITION_TEMPLATE_SHA256 or r["vision_checkpoint_sha256"]!=file_sha256(mae):
                        raise ValueError("completed template/MAE mismatch")
                    if name=="cache" and r["alignment_checkpoint_sha256"]!=file_sha256(align/"best.pt"):
                        raise ValueError("completed cache/alignment mismatch")
                print(f"{ds}/{name}: retain completed stage",flush=True)
            else:
                if (completed.parent/"best.pt").exists():
                    raise FileExistsError(f"partial run: choose separate output version, do not overwrite {completed.parent}")
                command=[sys.executable,str(root/"scripts"/script),*flags]
                print(shlex.join(command),flush=True)
                if not args.dry_run:
                    subprocess.run(command,cwd=root,check=True)
        if args.dry_run:
            continue
        new=json.loads((out/"result.json").read_text())
        for key in ("data_manifest_sha256","temporal_checkpoint_sha256","epochs","batch_size","learning_rate","temporal_learning_rate","freeze_temporal_epochs","seed","split_seed","target_divisor"):
            if new[key]!=old[key]:
                raise ValueError(f"comparison protocol changed: {key}")
        if new["qwen_cache_manifest_sha256"]!=file_sha256(cache/"manifest.json"):
            raise ValueError("TMAF/cache provenance mismatch")
        if new["initialization"]["applied_initial_state_sha256"]!=old["initialization"]["applied_initial_state_sha256"]:
            raise ValueError("initial weights differ")
        old_config=dict(old["model_config"]);old_config.setdefault("global_pooling","mean")
        if new["model_config"]!=old_config:
            raise ValueError("model configuration changed")
        subprocess.run([sys.executable,str(root/"scripts/audit_tmaf_scale_b.py"),"--output-dir",str(out)],cwd=root,check=True)
        print(json.dumps({"dataset":ds,"status":"supplementary_preprocessing_and_prompt_assumptions",
            "A_validation":old["validation"],"B_validation":new["validation"],"A_test":old["test"],"B_test":new["test"],
            "selected_by_validation":"B" if new["validation"]["rmse"]<old["validation"]["rmse"] else "A"},indent=2),flush=True)


if __name__=="__main__":
    main()
