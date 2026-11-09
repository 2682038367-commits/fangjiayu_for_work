"""Fixed-protocol downstream evaluation of existing control/fix synthetic data."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/diagnostics/zero_label_downstream_20261026'
RUN=ROOT/'runs/diagnostic-zero-label-downstream-20261026'
SEEDS=[3,13,23,33,43]
BINS=[('0-30',0,30),('31-60',31,60),('61-90',61,90),('91-124',91,124),('125',125,125),('all',0,125)]


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    for key,value in {'WANDB_MODE':'disabled','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                      'MPLCONFIGDIR':str(ROOT/'.cache/matplotlib'),
                      'XDG_CACHE_HOME':str(ROOT/'.cache')}.items():os.environ.setdefault(key,value)
    import numpy as np
    import torch
    import yaml
    from scipy.stats import t
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    now=lambda:datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    manifest=json.loads((RUN/'run-manifest.json').read_text())
    manifest.update(status='running',started_at=now(),python=sys.version,platform=platform.platform(),
                    torch=str(torch.__version__),torch_cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    mp=RUN/'run-manifest.json';mp.write_text(json.dumps(manifest,indent=2))
    rows,segments,predictions,progress=[],[],[],[]
    def bind(p):
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        expected=manifest['inputs_sha256'].get(str(p.relative_to(ROOT)))
        if expected is not None and actual!=expected:raise ValueError(f'input changed: {p}')
    sys.path.insert(0,str(ROOT/'upstream'))
    saved=sys.argv;sys.argv=[sys.argv[0]]
    from args import args
    from data.CMAPSSDataset import CMAPSSDataset
    from eva_regressor import predictive_score_metrics
    import wandb
    sys.argv=saved
    cwd=Path.cwd()
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        for p in manifest['inputs_sha256']:bind(ROOT/p)
        wandb.init(project='zero-label-downstream-diagnostic',mode='disabled')
        os.chdir(ROOT/'upstream')
        for ds in ['FD001','FD002']:
            config=yaml.safe_load((ROOT/f'configs/evaluation_{ds.lower()}_w48_public_code.yaml').read_text())
            with np.load(ROOT/config['outputs']['split_indices']) as z:
                train_idx=z['predictive_train'].copy();dev_idx=z['predictive_dev'].copy()
            assert not np.intersect1d(train_idx,dev_idx).size
            d=CMAPSSDataset(ds,48,[1000]);tydata,tylabel=d.get_last_data_slice(d.get_test_data())
            expected_labels=d.get_label_slice(d.get_train_data()).numpy()
            assert len(train_idx)+len(dev_idx)==len(expected_labels)
            assert np.array_equal(np.sort(np.concatenate([train_idx,dev_idx])),np.arange(len(expected_labels)))
            engine_ids=sorted(i for i in set(d.get_test_data().id) if i!=1000 and i<=d.engine_size)
            assert len(engine_ids)==len(tylabel)
            args.dataset=ds;args.window_size=48;args.input_size=14;args.output_size=1
            args.eva_epoch=int(config['evaluation']['evaluator_epochs'])
            args.device='cuda';args.model_name='DiffUnet_fre'
            args.num_layers=int(config['evaluation'].get('predictor_layers',1))
            manifest.setdefault('resolved_predictor_args',{})[ds]={k:str(v) if not isinstance(v,(str,int,float,bool,type(None))) else v for k,v in vars(args).items()}
            for arm in ['control','no_batch_zero']:
                with np.load(ROOT/f'outputs/{ds}_w48_seed3_zero_label_{arm}.npz') as z:
                    generated={'data':z['data'].copy(),'label':z['label'].copy()}
                assert np.array_equal(generated['label'],expected_labels)
                assert generated['data'].shape==(len(expected_labels),48,14)
                for seed in SEEDS:
                    name=f'{ds}_{arm}_eval{seed}'
                    checkpoint=ROOT/f'checkpoints/evaluators/zero_label_downstream_20261026/{name}.pth'
                    checkpoint.parent.mkdir(parents=True,exist_ok=True)
                    args.save_path=str(checkpoint)
                    print('START',name,now(),flush=True)
                    started=time.time()
                    rmse,mae,score=predictive_score_metrics(args,{'data':tydata,'label':tylabel},generated,
                        eval_seed=seed,split_indices=(train_idx,dev_idx))
                    seconds=time.time()-started
                    model=torch.load(checkpoint,map_location='cuda',weights_only=False).eval()
                    with torch.no_grad():estimate=model(tydata.to('cuda')).cpu().numpy().ravel().astype(float)
                    truth=tylabel.numpy().ravel().astype(float)
                    replay=float(np.sqrt(np.mean((estimate-truth)**2)))
                    assert abs(replay-rmse)<.001 and np.isfinite(estimate).all()
                    rows.append(dict(dataset=ds,arm=arm,generation_seed=3,evaluator_seed=seed,
                                     rmse=float(rmse),mae=float(mae),rul_score=float(score),
                                     predictive_seconds=seconds,replay_rmse=replay,checkpoint=str(checkpoint.relative_to(ROOT))))
                    for b,lo,hi in BINS:
                        mask=(truth>=lo)&(truth<=hi);err=estimate[mask]-truth[mask]
                        segments.append(dict(dataset=ds,arm=arm,evaluator_seed=seed,bin=b,n_engines=int(mask.sum()),
                                             mse=float(np.mean(err**2)),rmse=float(np.sqrt(np.mean(err**2))),
                                             mae=float(np.mean(abs(err))),bias=float(err.mean())))
                    for engine,label,p in zip(engine_ids,truth,estimate):
                        predictions.append(dict(dataset=ds,arm=arm,evaluator_seed=seed,engine_id=engine,
                                                rul=label,prediction=float(p),error=float(p-label)))
                    for filename,data in [('raw.csv',rows),('per_model_segments.csv',segments),('predictions.csv',predictions)]:write_csv(OUT/filename,data)
                    progress.append(name);manifest['completed_evaluations']=len(rows);manifest['progress']=progress
                    mp.write_text(json.dumps(manifest,indent=2))
                    print('DONE',name,'RMSE',rmse,'seconds',seconds,flush=True)
                    del model;torch.cuda.empty_cache()
                del generated
        import pandas as pd
        raw=pd.DataFrame(rows);seg=pd.DataFrame(segments)
        summary=raw.groupby(['dataset','arm']).agg(mean_rmse=('rmse','mean'),evaluator_sd=('rmse','std'),mean_mae=('mae','mean')).reset_index()
        summary.to_csv(OUT/'summary.csv',index=False)
        comparison=[]
        for (ds,b),group in seg.groupby(['dataset','bin']):
            pivot=group.pivot(index='evaluator_seed',columns='arm',values='rmse').reindex(SEEDS)
            delta=pivot.no_batch_zero-pivot.control
            half=float(t.ppf(.975,4)*delta.std(ddof=1)/np.sqrt(5))
            comparison.append(dict(dataset=ds,bin=b,n_engines=int(group.n_engines.iloc[0]),
                control_mean_rmse=float(pivot.control.mean()),fix_mean_rmse=float(pivot.no_batch_zero.mean()),
                delta_fix_minus_control=float(delta.mean()),paired_ci95_low=float(delta.mean()-half),paired_ci95_high=float(delta.mean()+half),
                control_pooled_rmse=float(np.sqrt(group[group.arm=='control'].mse.mean())),
                fix_pooled_rmse=float(np.sqrt(group[group.arm=='no_batch_zero'].mse.mean()))))
        write_csv(OUT/'comparison.csv',comparison)
        assert len(rows)==20 and len(segments)==120
        lines=['# Zero-label repair: downstream real-test comparison','',
               'FD001/FD002 full-data w48, generation seed3, five paired evaluator seeds. Diagnostic, not replacement paper results.',
               '', '| Dataset | RUL bin | Control RMSE | Fix RMSE | Fix − control |', '|---|---|---:|---:|---:|']
        for r in comparison:lines.append(f"| {r['dataset']} | {r['bin']} | {r['control_mean_rmse']:.3f} | {r['fix_mean_rmse']:.3f} | {r['delta_fix_minus_control']:+.3f} |")
        lines+=['','Means above average evaluator RMSE. Pooled segment RMSE also saved in comparison.csv.',
                'CIs quantify evaluator-seed variability only, conditional on one fixed generated dataset per arm; not generator-seed variability.',
                'No output clipping, no real-test selection, no generator retraining.']
        (OUT/'result.md').write_text('\n'.join(lines)+'\n')
        manifest.update(status='complete',summary=summary.to_dict('records'),comparison=comparison)
    except BaseException:
        manifest.update(status='failed',error=traceback.format_exc());raise
    finally:
        os.chdir(cwd);manifest['finished_at']=now()
        manifest['outputs_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
        mp.write_text(json.dumps(manifest,indent=2));wandb.finish()
        with (ROOT/'experiment-ledger.md').open('a') as f:
            f.write('\n### Downstream zero-label diagnostic finish\n\n- Run '+manifest['run_id']+'; '+manifest['finished_at']+'; '+manifest['status']+'; completed '+str(len(rows))+'/20 evaluations. See run manifest and results/diagnostics/zero_label_downstream_20261026/result.md.\n')


if __name__=='__main__':main()
