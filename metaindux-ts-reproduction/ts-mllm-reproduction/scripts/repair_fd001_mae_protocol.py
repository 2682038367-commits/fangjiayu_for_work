"""Rebuild only FD001's MAE-dependent chain; retain temporal baseline and FD003."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from ts_mllm.rebuilt_data import file_sha256


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    data=root/"artifacts/data_window40_stride50/split_seed42/FD001"
    dh=file_sha256(data/"manifest.json")
    old_dir=root/"artifacts/tmaf_audited/FD001/global_broadcast_stride50_B_seed42"
    old=json.loads((old_dir/"result.json").read_text())
    temporal=root/old["temporal_checkpoint"]
    if old["data_manifest_sha256"]!=dh or file_sha256(temporal)!=old["temporal_checkpoint_sha256"]:
        raise ValueError("original FD001 data/temporal source changed")
    base=root/"artifacts/fd001_mae_stride50_repair/seed42"
    mae,align,cache,spectra,out=[base/n for n in ("mae","alignment","qwen","spectrum","tmaf")]
    common=["--dataset","FD001","--rebuilt-data-dir",str(data)]
    stages=[
        ("MAE","pretrain_mae.py",[*common,"--output-dir",str(mae),"--device","cuda","--epochs","15","--batch-size","128","--learning-rate","0.001","--seed","42","--split-seed","42"],mae/"result.json"),
        ("alignment","train_svlma_alignment.py",[*common,"--vision-checkpoint",str(mae/"best.pt"),"--output-dir",str(align),"--prompt-profile","legacy"],align/"result.json"),
        ("cache","cache_qwen_multimodal.py",[*common,"--vision-checkpoint",str(mae/"best.pt"),"--alignment-checkpoint",str(align/"best.pt"),"--output-dir",str(cache),"--spectrum-output-dir",str(spectra),"--prompt-profile","legacy","--train-stride","50","--validation-stride","50","--inference-batch-size","16"],cache/"manifest.json"),
        ("TMAF","train_tmaf.py",[*common,"--cache-dir",str(cache),"--temporal-checkpoint",str(temporal),"--output-dir",str(out),
            "--context-mode","global_broadcast","--global-pooling","mean","--token-mode","full","--train-stride","50","--validation-stride","50",
            "--epochs","30","--batch-size","128","--learning-rate","0.002","--temporal-learning-rate","0.002",
            "--freeze-temporal-epochs","0","--output-bias-init","default","--seed","42","--split-seed","42"],out/"result.json")]
    for name,script,flags,completed in stages:
        if completed.exists():
            r=json.loads(completed.read_text())
            if r["dataset"]!="FD001" or r["data_manifest_sha256"]!=dh:
                raise ValueError(f"completed source mismatch: {completed}")
            if name in ("alignment","cache"):
                if r["vision_checkpoint_sha256"]!=file_sha256(mae/"best.pt") or r.get("prompt_profile","legacy")!="legacy":
                    raise ValueError("completed MAE/prompt source mismatch")
                if name=="cache" and r["alignment_checkpoint_sha256"]!=file_sha256(align/"best.pt"):
                    raise ValueError("completed alignment source mismatch")
            if name=="TMAF" and (r["qwen_cache_manifest_sha256"]!=file_sha256(cache/"manifest.json") or r["temporal_checkpoint_sha256"]!=file_sha256(temporal)):
                raise ValueError("completed TMAF source mismatch")
            print(f"FD001/{name}: retain completed stage",flush=True)
        else:
            if (completed.parent/"best.pt").exists():
                raise FileExistsError(f"partial run; do not overwrite: {completed.parent}")
            command=[sys.executable,str(root/"scripts"/script),*flags]
            print(shlex.join(command),flush=True)
            if not args.dry_run:
                subprocess.run(command,cwd=root,check=True)
    if args.dry_run:
        return
    new=json.loads((out/"result.json").read_text())
    for key in ("data_manifest_sha256","temporal_checkpoint_sha256","train_stride","validation_stride","target_divisor",
                "epochs","batch_size","learning_rate","temporal_learning_rate","freeze_temporal_epochs","seed","split_seed","token_mode"):
        if new[key]!=old[key]:
            raise ValueError(f"unrequested change: {key}")
    config=dict(old["model_config"]);config.setdefault("global_pooling","mean")
    if new["model_config"]!=config:
        raise ValueError("model configuration changed")
    old_init=old.get("initialization",{}).get("applied_initial_state_sha256")
    if old_init is not None and new["initialization"]["applied_initial_state_sha256"]!=old_init:
        raise ValueError("initial weights changed")
    subprocess.run([sys.executable,str(root/"scripts/audit_tmaf_scale_b.py"),"--output-dir",str(out)],cwd=root,check=True)
    print(json.dumps({"dataset":"FD001","status":"MAE protocol repaired; undisclosed model choices remain assumptions",
        "old_validation":old["validation"],"new_validation":new["validation"],"old_test":old["test"],"new_test":new["test"],
        "initial_weights_comparison":"matched" if old_init is not None else "unverifiable: historical result has no initialization hash",
        "FD003":"unchanged; audited stride50 MAE already used"},indent=2),flush=True)


if __name__=="__main__":
    main()
