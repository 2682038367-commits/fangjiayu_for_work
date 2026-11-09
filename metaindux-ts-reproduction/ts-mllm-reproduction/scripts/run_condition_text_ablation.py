"""A's full cached outputs versus visual-prefix-only outputs; GPU TMAF retrain."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from ts_mllm.rebuilt_data import file_sha256


def validate_comparison(full, visual):
    for key in ("dataset","data_manifest_sha256","qwen_cache_manifest_sha256","temporal_checkpoint_sha256",
                "train_stride","validation_stride","epochs","batch_size","learning_rate",
                "temporal_learning_rate","freeze_temporal_epochs","seed","split_seed","target_divisor"):
        if full[key]!=visual[key]:
            raise ValueError(f"ablation protocol differs: {key}")
    a,b=dict(full["model_config"]),dict(visual["model_config"])
    a.setdefault("global_pooling","mean");b.setdefault("global_pooling","mean")
    if a!=b or full["token_mode"]!="full" or visual["token_mode"]!="visual_only":
        raise ValueError("wrong architecture/token modes")
    if full["initialization"]["applied_initial_state_sha256"]!=visual["initialization"]["applied_initial_state_sha256"]:
        raise ValueError("initial weights differ")


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument("--datasets",nargs="+",choices=["FD002","FD004"],default=["FD002","FD004"])
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    for ds in args.datasets:
        baseline=root/f"artifacts/tmaf_condition_minmax/{ds}/global_broadcast_stride50_B_seed42"
        full=json.loads((baseline/"result.json").read_text())
        data,cache,temporal=[Path(full[key]) for key in ("rebuilt_data_dir","qwen_cache","temporal_checkpoint")]
        for path,key in ((data/"manifest.json","data_manifest_sha256"),(cache/"manifest.json","qwen_cache_manifest_sha256"),(temporal,"temporal_checkpoint_sha256")):
            if file_sha256(path)!=full[key]:
                raise ValueError(f"baseline source changed: {path}")
        if json.loads((data/"manifest.json").read_text()).get("normalization")!="condition_minmax":
            raise ValueError("expected condition-normalized A")
        cm=json.loads((cache/"manifest.json").read_text())
        if cm.get("prompt_profile","legacy")!="legacy":
            raise ValueError("expected A legacy statistics prompt, not B")
        out=root/f"artifacts/tmaf_condition_ablation/visual_only/{ds}/seed42"
        if not (out/"result.json").exists():
            if (out/"best.pt").exists():
                raise FileExistsError(f"partial run; choose separate output rather than overwrite: {out}")
            command=[sys.executable,str(root/"scripts/train_tmaf.py"),"--dataset",ds,
                "--rebuilt-data-dir",str(data),"--cache-dir",str(cache),"--temporal-checkpoint",str(temporal),"--output-dir",str(out),
                "--context-mode","global_broadcast","--global-pooling","mean","--token-mode","visual_only",
                "--train-stride","50","--validation-stride","50","--epochs","30","--batch-size","128",
                "--learning-rate","0.002","--temporal-learning-rate","0.002","--freeze-temporal-epochs","0",
                "--output-bias-init","default","--seed","42","--split-seed","42","--num-workers","2"]
            print(shlex.join(command),flush=True)
            if not args.dry_run:
                subprocess.run(command,cwd=root,check=True)
        if args.dry_run:
            continue
        visual=json.loads((out/"result.json").read_text())
        validate_comparison(full,visual)
        subprocess.run([sys.executable,str(root/"scripts/audit_tmaf_scale_b.py"),"--output-dir",str(out)],cwd=root,check=True)
        comparison={"dataset":ds,"experiment":"cached-output text-token ablation, not Qwen input-text removal",
            "full_validation":full["validation"],"visual_only_validation":visual["validation"],
            "full_test":full["test"],"visual_only_test":visual["test"],
            "selected_by_validation":"full" if full["validation"]["rmse"]<visual["validation"]["rmse"] else "visual_only"}
        print(json.dumps(comparison,indent=2),flush=True)


if __name__=="__main__":
    main()
