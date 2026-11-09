from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,platform,subprocess,sys,traceback,time
import torch
root=Path(__file__).resolve().parents[2]
path=Path(__file__).with_name("run-manifest.json")
m=json.loads(path.read_text());now=lambda:datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
m.update(status="running",started_at=now(),platform=platform.platform(),python=sys.version,torch=str(torch.__version__),torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
m["nvidia_smi"]=subprocess.run(["nvidia-smi"],capture_output=True,text=True,errors="replace").stdout
m["commands"]=[[sys.executable,"scripts/run_experiments.py","--config",f"configs/diagnostic_zero_label_fd001_{a}.yaml","--skip-completed"] for a in m["arms"]]+[[sys.executable,"analysis/diagnostics/evaluate_zero_label_pair.py"]]
path.write_text(json.dumps(m,indent=2))
code=1
try:
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    for command in m["commands"]:
        print("RUN:"," ".join(command),flush=True)
        subprocess.run(command,cwd=root,check=True)
    verdict=json.loads((root/"results/diagnostics/zero_label_pair_fd001_seed3/verdict.json").read_text())
    m.update(status="complete",verdict=verdict)
    report=root/"results/diagnostics/zero_label_pair_fd001_seed3/result.md"
    lines=["# Zero-label intervention: FD001 seed3", "", "Diagnostic only; original strict-reproduction results unchanged.", "", "| RUL bin | Control mean prediction | Fix mean prediction | Control RMSE | Fix RMSE |", "|---|---:|---:|---:|---:|"]
    for r in verdict["comparison"]:
        lines.append(f"| {r['bin']} | {r['control_prediction_mean']:.3f} | {r['no_batch_zero_prediction_mean']:.3f} | {r['control_rmse']:.3f} | {r['no_batch_zero_rmse']:.3f} |")
    lines += ["", "Supports prespecified criterion: "+str(verdict["supports_zero_label_collision"]), "Single seed, proxy predictors, no inference to all datasets or overall downstream improvement."]
    report.write_text("\n".join(lines)+"\n")
    code=0
except BaseException:
    m.update(status="failed",error=traceback.format_exc())
    print(m["error"],flush=True)
finally:
    m["finished_at"]=now();path.write_text(json.dumps(m,indent=2))
    with (root/"experiment-ledger.md").open("a") as f:
        f.write("\n### Zero-label pair finish\n\n- Run "+m["run_id"]+"; "+m["finished_at"]+"; status "+m["status"]+". See manifest and results/diagnostics/zero_label_pair_fd001_seed3/verdict.json. No automatic follow-up training.\n")
sys.exit(code)
