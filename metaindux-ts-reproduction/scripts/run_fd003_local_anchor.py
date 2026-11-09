"""Single fixed10% real-training-anchor intervention in FD003 synthetic data."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import hashlib
import os
import sys
import time
import traceback
import importlib.util

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/ext-local-anchor-fd003-seed3-20261026'
OUT=ROOT/'results/diagnostics/fd003_local_anchor_20261026'


def main():
    for k,v in {'WANDB_MODE':'disabled','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                'MPLCONFIGDIR':str(ROOT/'.cache/matplotlib'),'XDG_CACHE_HOME':str(ROOT/'.cache')}.items():os.environ.setdefault(k,v)
    import numpy as np
    import pandas as pd
    import torch
    from scipy.stats import t
    sys.path.insert(0,str(ROOT/'upstream'))
    saved=sys.argv;sys.argv=[sys.argv[0]]
    from args import args
    from data.CMAPSSDataset import CMAPSSDataset
    from run_zero_label_downstream import write_csv,SEEDS,BINS
    spec=importlib.util.spec_from_file_location('fixed_budget_evaluator',RUN/'fixed_budget_evaluator.py')
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    import wandb
    sys.argv=saved
    OUT.mkdir(parents=True,exist_ok=True)
    now=lambda:datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    mp=RUN/'run-manifest.json';m=json.loads(mp.read_text())
    m.update(status='running',started_at=now(),torch=str(torch.__version__),cuda=torch.version.cuda,
             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    mp.write_text(json.dumps(m,indent=2))
    rows,segments,predictions=[],[],[];cwd=Path.cwd()
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        for name,h in m['inputs_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
        os.chdir(ROOT/'upstream')
        wandb.init(project='fd003-local-anchor',mode='disabled')
        d=CMAPSSDataset('FD003',48,[1000])
        real=d.get_feature_slice(d.get_train_data()).numpy()
        labels=d.get_label_slice(d.get_train_data()).numpy()
        tx,ty=d.get_last_data_slice(d.get_test_data());truth=ty.numpy().ravel().astype(float)
        testids=sorted(i for i in set(d.get_test_data().id) if i!=1000 and i<=d.engine_size)
        with np.load(ROOT/'results/fd003_w48_public_code_split.npz') as z:
            ti=z['predictive_train'].copy();vi=z['predictive_dev'].copy()
        assert not np.intersect1d(ti,vi).size
        with np.load(ROOT/'outputs/FD003_w48_seed3.npz') as z:
            original=z['data'].copy();sy=z['label'].copy()
        assert np.array_equal(labels,sy) and original.shape==real.shape
        y=sy.ravel();eligible=ti[(y[ti]>=31)&(y[ti]<=60)]
        rng=np.random.RandomState(20261026)
        target=rng.choice(eligible,int(.1*len(eligible)),replace=False)
        donor=np.empty(len(target),dtype=int)
        for label in np.unique(y[target]):
            loc=np.flatnonzero(y[target]==label)
            pool=ti[y[ti]==label]
            donor[loc]=rng.choice(pool,len(loc),replace=False)
        assert np.array_equal(y[target],y[donor])
        assert np.isin(donor,ti).all() and np.isin(target,ti).all()
        assert not np.intersect1d(donor,vi).size and not np.intersect1d(target,vi).size
        mixed=original.copy();mixed[target]=real[donor]
        untouched=np.ones(len(y),bool);untouched[target]=False
        assert np.array_equal(original[untouched],mixed[untouched])
        assert np.array_equal(original[vi],mixed[vi])
        np.savez_compressed(OUT/'replacement_indices.npz',target=target,donor=donor,
                            target_rul=y[target],donor_rul=y[donor],train_indices=ti,dev_indices=vi)
        m['replacement_audit']=dict(total_windows=len(y),train_windows=len(ti),dev_windows=len(vi),
            eligible_train_windows=len(eligible),replaced=len(target),
            fraction_of_eligible=len(target)/len(eligible),fraction_of_total_train=len(target)/len(ti),
            exact_rul_match=True,dev_identical=True,only_real_train_donors=True)
        m['expected_updates_per_model']=35*int(np.ceil(len(ti)/256))
        args.dataset='FD003';args.window_size=48;args.input_size=14;args.output_size=1
        args.eva_epoch=35;args.device='cuda';args.model_name='DiffUnet_fre';args.num_layers=1
        m['predictor_args']={k:v if isinstance(v,(int,float,str,bool,type(None))) else str(v) for k,v in vars(args).items()}
        mp.write_text(json.dumps(m,indent=2))
        print('REPLACEMENT',m['replacement_audit'],flush=True)
        for arm,x in [('control',original),('anchor10',mixed)]:
            data={'data':x,'label':sy}
            for seed in SEEDS:
                name=f'FD003_{arm}_eval{seed}'
                path=ROOT/f'checkpoints/evaluators/fd003_local_anchor_20261026/{name}.pth'
                path.parent.mkdir(parents=True,exist_ok=True);args.save_path=str(path)
                print('START',name,now(),flush=True);started=time.time()
                rmse,mae,score=evaluator.predictive_score_metrics(args,{'data':tx,'label':ty},data,
                    eval_seed=seed,split_indices=(ti,vi))
                seconds=time.time()-started
                model=torch.load(path,map_location='cuda',weights_only=False).eval()
                with torch.no_grad():p=model(tx.to('cuda')).cpu().numpy().ravel().astype(float)
                assert abs(np.sqrt(np.mean((p-truth)**2))-rmse)<.001
                rows.append(dict(dataset='FD003',arm=arm,evaluator_seed=seed,rmse=float(rmse),mae=float(mae),
                                 predictive_seconds=seconds,checkpoint=str(path.relative_to(ROOT))))
                for b,lo,hi in BINS:
                    mask=(truth>=lo)&(truth<=hi);err=p[mask]-truth[mask]
                    segments.append(dict(dataset='FD003',arm=arm,evaluator_seed=seed,bin=b,n_engines=int(mask.sum()),
                                         mse=float(np.mean(err**2)),rmse=float(np.sqrt(np.mean(err**2))),bias=float(err.mean())))
                for engine,label,estimate in zip(testids,truth,p):
                    predictions.append(dict(arm=arm,evaluator_seed=seed,engine_id=engine,rul=float(label),
                                            prediction=float(estimate),error=float(estimate-label)))
                for filename,z in [('raw.csv',rows),('per_model_segments.csv',segments),('predictions.csv',predictions)]:write_csv(OUT/filename,z)
                m['completed_evaluations']=len(rows);mp.write_text(json.dumps(m,indent=2))
                print('DONE',name,rmse,flush=True);del model;torch.cuda.empty_cache()
        comparisons=[];s=pd.DataFrame(segments)
        for b,g in s.groupby('bin'):
            v=g.pivot(index='evaluator_seed',columns='arm',values='rmse').reindex(SEEDS)
            diff=v.anchor10-v.control;half=float(t.ppf(.975,4)*diff.std(ddof=1)/np.sqrt(5))
            comparisons.append(dict(bin=b,n_engines=int(g.n_engines.iloc[0]),control_mean_rmse=float(v.control.mean()),
                anchor_mean_rmse=float(v.anchor10.mean()),delta=float(diff.mean()),paired_ci95_low=float(diff.mean()-half),
                paired_ci95_high=float(diff.mean()+half),control_pooled_rmse=float(np.sqrt(g[g.arm=='control'].mse.mean())),
                anchor_pooled_rmse=float(np.sqrt(g[g.arm=='anchor10'].mse.mean()))))
        write_csv(OUT/'comparison.csv',comparisons)
        assert len(rows)==10 and len(predictions)==1000
        m.update(status='complete',comparison=comparisons)
    except BaseException:
        m.update(status='failed',error=traceback.format_exc());raise
    finally:
        os.chdir(cwd);m['finished_at']=now()
        m['outputs_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
        mp.write_text(json.dumps(m,indent=2));wandb.finish()
        with (ROOT/'experiment-ledger.md').open('a') as f:
            f.write('\n### FD003 real-anchor experiment finish\n\n- '+m['run_id']+'; '+m['finished_at']+'; '+m['status']+'; '+str(len(rows))+'/10 evaluations; see snapshot/comparison.csv. No automatic additional experiments.\n')


if __name__=='__main__':main()
