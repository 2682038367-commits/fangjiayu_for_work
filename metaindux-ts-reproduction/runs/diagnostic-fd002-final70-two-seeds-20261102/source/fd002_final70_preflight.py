from pathlib import Path
import sys,json,hashlib,inspect
import numpy as np
import torch
root=Path('/home/cw_boe/projects/rul/metaindux-ts-reproduction');sys.path.insert(0,str(root/'scripts'));sys.path.insert(0,str(root/'upstream'))
import run_fd002_final70_two_seeds as run
run.environment();m=run.check_inputs();assert torch.cuda.is_available()
import DiffusionFreeGuidence.Unet1D_fre as unet
from GaussianDiffusion import GaussianDiffusion1D_cls_free
from utils import torch_seed
core=run.module(run.CORE,'fd002_noise_preflight')
records=[]
for seed in run.SEEDS:
    torch_seed(103)
    model=unet.UNet1D_fre(dim=32,dim_mults=(1,2),cond_drop_prob=.2,channels=14,length=48,
        frequency_mask_mode='hard_random_quantile',frequency_threshold_init=.25,frequency_mask_temperature=.1).cuda()
    loaded=torch.load(root/f'checkpoints/FD002_w48_seed{seed}.pth',map_location='cuda',weights_only=True)
    keys=model.load_state_dict(loaded,strict=True);assert not keys.missing_keys and not keys.unexpected_keys
    model.eval();assert all(not mod.training for mod in model.modules())
    diff=GaussianDiffusion1D_cls_free(model,seq_length=48,channels=14,timesteps=1000,objective='pred_noise',beta_schedule='linear').cuda()
    trace=core.LegacyNoiseTrace(torch,unet,retain_condition=True);trace.install();trace.start()
    try:
        with torch.no_grad():
            x=torch.randn((8,14,48),device='cuda');cond=torch.tensor([0,1,30,45,70,100,124,125],device='cuda')[:,None]
            for t in [999,998,997]:
                x,_=diff.p_sample(x,t,cond)
        assert torch.isfinite(x).all()
        trace.after();assert trace.normals==4 and trace.conditions==trace.unconditionals==3 and trace.dropped==0
        records.append(dict(seed=seed,strict_load=True,evaluation_mode=True,normal_tensors_matched=trace.normals,
            condition_draws_matched=trace.conditions,explicit_null_calls=trace.unconditionals,
            noise_schedule=trace.fingerprint.hexdigest(),final_rng=hashlib.sha256(trace.reference.get_state().numpy().tobytes()).hexdigest()))
    finally:
        trace.restore()
    del model,diff,loaded,x;torch.cuda.empty_cache()
assert records[0]['noise_schedule']==records[1]['noise_schedule'] and records[0]['final_rng']==records[1]['final_rng']
# Verify the isolated equal-budget edit changes only the public stopping branch.
sys.argv=[sys.argv[0]]
import eva_regressor as eva
source=inspect.getsource(eva.train)
branch="        if epoch - best_epoch >= 10:\n            print('*******STOP!!!********')\n            break"
assert source.count(branch)==1
fixed=source.replace(branch,'        # Fixed 35-epoch paired budget; early stopping disabled.')
compile(fixed,'<preflight_fixed_budget>','exec')
import ast
orig=ast.parse(source).body[0];new=ast.parse(fixed).body[0]
orig.body[-2].body.pop() if False else None
# Removing only one If in the epoch loop must produce an identical AST.
for n in ast.walk(orig):
    if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch':
        assert isinstance(n.body[-1],ast.If);n.body.pop();break
assert ast.dump(orig,include_attributes=False)==ast.dump(new,include_attributes=False)
result=dict(status='passed',at=run.now(),torch=str(torch.__version__),cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0),frozen_inputs_verified=True,predictor_budget_edit_ast_parity=True,arms=records)
run.dump(run.OUT/'preflight.json',result)
print(json.dumps(result,indent=2))
