from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,platform,subprocess,sys,time,traceback
import torch,yaml
root=Path(__file__).resolve().parents[2]
path=Path(__file__).with_name("run-manifest.yaml")
m=yaml.safe_load(path.read_text())
now=lambda: datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
m.update(status="running",started_at=now(),python=sys.version,platform=platform.platform(),torch=str(torch.__version__),torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
m["nvidia_smi"]=subprocess.run(["nvidia-smi"],capture_output=True,text=True).stdout
path.write_text(yaml.safe_dump(m,sort_keys=False,allow_unicode=True))
started=time.time()
try:
    result=subprocess.run(m["command"],cwd=root)
    m.update(status="complete" if result.returncode==0 else "failed",exit_code=result.returncode)
except BaseException:
    m.update(status="failed",error=traceback.format_exc())
    raise
finally:
    m.update(finished_at=now(),total_seconds=time.time()-started)
    path.write_text(yaml.safe_dump(m,sort_keys=False,allow_unicode=True))
    with (root/"experiment-ledger.md").open("a") as f:
        f.write("\n### Execution finish: "+m["run_id"]+"\n\n- Finished: "+m["finished_at"]+"; status: "+m["status"]+". See run manifest and console log; metric interpretation pending audit.\n")
sys.exit(m.get("exit_code",1))
