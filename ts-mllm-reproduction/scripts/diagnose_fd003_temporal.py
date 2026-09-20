"""Audit existing FD003 temporal artifacts; no saved-model mutations."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ts_mllm.data import ACTIVE_SENSOR_COLUMNS, read_cmapss
from ts_mllm.model import PatchTransformer, PatchTransformerConfig
from ts_mllm.rebuilt_data import RebuiltWindowDataset, file_sha256
from ts_mllm.training import seed_everything

ROOT = Path(__file__).resolve().parents[1]

def main():
    assert torch.cuda.is_available(), 'GPU required'
    torch.set_num_threads(4)
    d = ROOT/'artifacts/data_window40_stride50/split_seed42/FD003'
    out = ROOT/'artifacts/temporal_audited/FD003/stride50_B_seed42'
    m = json.loads((d/'manifest.json').read_text())
    h = json.loads((out/'history.json').read_text())
    s = json.loads((d/'scaler.json').read_text())
    rawdir = ROOT.parent/'data/CMAPSSData'
    train,test,rul = read_cmapss(rawdir,'FD003')
    for name, digest in m['raw_sha256'].items():
        assert file_sha256(rawdir/name)==digest
    assert set(m['train_engine_ids']).isdisjoint(m['val_engine_ids'])
    lo = train[train.unit.isin(m['train_engine_ids'])][ACTIVE_SENSOR_COLUMNS].min().to_numpy()
    hi = train[train.unit.isin(m['train_engine_ids'])][ACTIVE_SENSOR_COLUMNS].max().to_numpy()
    np.testing.assert_array_equal(lo,s['minimum']); np.testing.assert_array_equal(hi,s['maximum'])
    assert s['columns']==ACTIVE_SENSOR_COLUMNS
    assert file_sha256(out/'scaler.json')==file_sha256(d/'scaler.json')
    ds={k:RebuiltWindowDataset(d,k) for k in ['train','val','test']}
    data_report={}
    for split, dataset in ds.items():
        a=dataset.arrays
        starts=np.load(d/f'{split}_start_row.npy')
        units=m['train_engine_ids'] if split=='train' else m['val_engine_ids'] if split=='val' else sorted(test.unit.unique())
        frame=test if split=='test' else train
        expected=[]
        for unit in units:
            g=frame[frame.unit==unit].sort_values('cycle')
            for start in ([max(0,len(g)-40)] if split=='test' else range(0,len(g)-39,50)):
                expected.append((int(unit),int(start)))
        assert expected==list(zip(a['unit'].tolist(),starts.tolist())), (split,len(expected),len(starts),[(e,z) for e,z in zip(expected,zip(a['unit'].tolist(),starts.tolist())) if e!=z][:3])
        for i,(unit,start) in enumerate(expected):
            g=frame[frame.unit==unit].sort_values('cycle')
            w=g.iloc[start:start+40]
            normalized=((w[ACTIVE_SENSOR_COLUMNS].to_numpy()-lo)/(hi-lo)).astype(np.float32)
            if len(normalized)<40:
                normalized=np.concatenate([np.repeat(normalized[:1],40-len(normalized),axis=0),normalized])
            np.testing.assert_array_equal(normalized,a['x'][i])
            end=int(w.cycle.iloc[-1]); assert end==a['cycle'][i]
            target=min(125,float(rul[unit-1])) if split=='test' else min(125,int(g.cycle.max())-end)
            assert target==a['target'][i]
        x=torch.tensor(np.array(a['x']),device='cuda')
        y=torch.tensor(np.array(a['target']),device='cuda')
        data_report[split]=dict(n=len(dataset),target_mean=float(y.mean()),target_std=float(y.std(correction=0)),
            target_min=float(y.min()),target_max=float(y.max()),target_capped_fraction=float((y==125).float().mean()),
            rul_0_25_count=int((y<=25).sum()),x_min=float(x.min()),x_max=float(x.max()),
            out_of_training_range_fraction=float(((x<0)|(x>1)).float().mean()),
            left_padded_units=np.unique(a['unit'][np.load(d/f'{split}_left_padding.npy')>0]).tolist())
    state=torch.load(out/'best.pt',map_location='cuda',weights_only=False)
    assert state['target_divisor']==125 and state['data_manifest_sha256']==file_sha256(d/'manifest.json')
    seed_everything(42)
    model=PatchTransformer(PatchTransformerConfig(**state['model_config'])).cuda()
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state['model_state']);model.eval()
    eval_report={}
    for split,dataset in ds.items():
        x=torch.tensor(np.array(dataset.arrays['x']),device='cuda')
        y=torch.tensor(np.array(dataset.arrays['target']),device='cuda')
        with torch.no_grad():
            encoded=model.encode(x);pool=encoded.mean(1)
            p=model(x)*125
            torch.testing.assert_close(p,model.regression_head(pool).squeeze(-1)*125)
            patches=model.make_patches(x)
            assert patches.shape[1:]==(38,56)
            manual=torch.stack([torch.cat([x,x[:,-1:]],1)[:,i:i+4].flatten(1) for i in range(38)],1)
            torch.testing.assert_close(patches,manual,rtol=0,atol=0)
            eval_report[split]=dict(raw_rmse=float((p-y).square().mean().sqrt()),prediction_mean=float(p.mean()),
                prediction_std=float(p.std(correction=0)),prediction_min=float(p.min()),prediction_max=float(p.max()),
                pooled_feature_coordinate_std_mean=float(pool.std(0,correction=0).mean()),
                constant_training_mean_rmse=float((y-data_report['train']['target_mean']).square().mean().sqrt()))
            if split=='test':
                csv=pd.read_csv(out/'test_predictions.csv')
                np.testing.assert_allclose(p.cpu().numpy(),csv.prediction_raw,rtol=1e-5,atol=3e-5)
                np.testing.assert_array_equal(csv.target,dataset.arrays['target'])
    # Gradients at saved model: no optimizer step and no training-mode dropout.
    x=torch.tensor(np.array(ds['train'].arrays['x']),device='cuda')
    y=torch.tensor(np.array(ds['train'].arrays['target']),device='cuda')/125
    loss=(model(x)-y).square().mean();loss.backward()
    grad={name:dict(l2=float(p.grad.norm()),finite=bool(torch.isfinite(p.grad).all()),
          parameter_l2=float(p.norm()),delta_from_seed42_initial_l2=float((p-initial[name]).norm()))
          for name,p in model.named_parameters()}
    assert all(v['finite'] for v in grad.values())
    report=dict(device=torch.cuda.get_device_name(),data_audit='all raw windows, labels, scaler extrema, hashes and split identities passed',
        shape_and_output_audit='38 temporal-major patches; mean pooling plus Linear(64,1); native output multiplied by125 exactly once; saved CSV matched',
        updates=dict(training_windows=377,batch_size=128,batches_per_epoch=math.ceil(377/128),epochs=30,total=90,
          status='inferred from historical training loop: drop_last=False, one optimizer.step per batch; original history has no per-step counter'),
        selected_epochs=[r for r in h if r['epoch'] in [1,5,10,15,20,25,26,27,28,29,30]],
        data=data_report,evaluation=eval_report,saved_checkpoint_gradient_native_mse=float(loss),gradients=grad,
        initialization_comparison_limit='Original result lacks initial state hash; seed42 reconstruction is a reference, not proof of identical historical initialization.',
        gradient_limit='These are full-training-set eval-mode gradients at best.pt, not recorded historical per-step gradients. No optimizer step performed.')
    (ROOT/'FD003_TEMPORAL_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='gradients'},indent=2))
    print(json.dumps({k:v for k,v in grad.items() if k in ['patch_projection.weight','position_embedding','encoder.layers.0.self_attn.in_proj_weight','encoder.layers.1.self_attn.in_proj_weight','output_norm.weight','regression_head.weight','regression_head.bias']},indent=2))

if __name__=='__main__':
    main()
