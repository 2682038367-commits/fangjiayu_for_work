"""Monitor one authorized run; record failure without automatic retries."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess,sys,json
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
WORKSPACE=ROOT.parent

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
command=[sys.executable,'-u',str(ROOT/'scripts/run_prototype_guidance_fd003.py'),'run']
p=subprocess.Popen(command,cwd=WORKSPACE)
launch=json.loads((RUN/'background_launch.json').read_text());launch['worker_pid']=p.pid;dump(RUN/'background_launch.json',launch)
result=p.wait()
dump(RUN/'process_exit.json',dict(exit_code=result,finished_at=now(),worker_pid=p.pid))
if result:
 dump(RUN/'execution_status.json',dict(status='failed',exit_code=result,finished_at=now(),see_log='logs/prototype_fd003_full.log',automatic_retry=False))
 m=json.loads((RUN/'manifest.json').read_text());m['status']='failed';dump(RUN/'manifest.json',m)
 entry=f'\n## {now()} FD003 prototype run failed\n\n- [Observed] Fixed scope extension:EXT-PROTO-002 run stopped with exit{result}; see runs/{RUN.name}/process_exit.json and logs/prototype_fd003_full.log. Preserve all artifacts; no automatic retry or budget expansion.\n'
 for name in ['experiment-ledger.md','reproduction-state.md']:
  with (ROOT/name).open('a') as f:f.write(entry)
sys.exit(result)
