"""Replay real-trained RUL predictors on exact-label-matched synthetic windows."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/diagnostics/real_predictor_condition_w48'
RUN = ROOT/'runs/diagnostic-real-predictor-condition-w48-20261023'
SEEDS = [3, 13, 23, 33, 43]
BINS = [('0-30', 0, 30), ('31-60', 31, 60), ('61-90', 61, 90),
        ('91-124', 91, 124), ('125', 125, 125), ('all', 0, 125)]


def read_csv(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def write_csv(p, rows):
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def infer(model, x):
    with torch.no_grad():
        p = np.concatenate([model(torch.from_numpy(x[i:i+256])).numpy().ravel()
                            for i in range(0, len(x), 256)]).astype(np.float64)
    assert np.isfinite(p).all()
    return p


def match_indices(y, reference, rng):
    idx = np.empty(len(reference), dtype=np.int64)
    for label in np.unique(reference):
        loc = np.flatnonzero(reference == label)
        candidates = np.flatnonzero(y == label)
        assert len(candidates) >= len(loc)
        idx[loc] = rng.choice(candidates, len(loc), replace=False)
    assert np.array_equal(y[idx], reference)
    return idx


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT/'upstream'))
    from data.CMAPSSDataset import CMAPSSDataset
    now = lambda: datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    manifest = dict(run_id=RUN.name, scope_class='diagnostic', role='supporting',
                    scope_contract='scope-20261023-real-predictor-condition-audit',
                    status='running', started_at=now(), command=f'{sys.executable} analysis/diagnostics/real_predictor_condition_audit.py',
                    seeds=SEEDS, device='cpu', no_training=True, torch=str(torch.__version__),
                    bins=BINS, inputs_sha256={}, primary_reference='independent real test terminal windows',
                    secondary_reference='real dev windows excluded from optimizer, used in checkpoint selection',
                    cap_dev_windows_per_bin=600, generated_seeds='common five full-data w48',
                    prediction_clipping=False, source_snapshot='SOURCE_SNAPSHOT.md')
    def bind(p):
        manifest['inputs_sha256'][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    bind(Path(__file__))
    bind(ROOT/'upstream/data/CMAPSSDataset.py')
    bind(ROOT/'upstream/models/LSTM.py')
    bind(ROOT/'SOURCE_SNAPSHOT.md')
    baseline_path = ROOT/'results/real_data_baseline_w48.csv'
    bind(baseline_path)
    recorded = read_csv(baseline_path)
    mp = RUN/'run-manifest.json'
    mp.write_text(json.dumps(manifest, indent=2))
    rows, replay = [], []
    def collect(ds, reference, source, g, e, y, pred):
        for name, lo, hi in BINS:
            mask = (y >= lo) & (y <= hi)
            if not mask.any():
                continue
            err = pred[mask]-y[mask]
            rows.append(dict(dataset=ds, reference=reference, source=source,
                             generation_seed=g, evaluator_seed=e, bin=name,
                             n_windows=int(mask.sum()), label_mean=float(y[mask].mean()),
                             prediction_mean=float(pred[mask].mean()), bias=float(err.mean()),
                             mse=float(np.mean(err*err)), rmse=float(np.sqrt(np.mean(err*err))),
                             mae=float(np.mean(abs(err))), prediction_below0=float(np.mean(pred[mask]<0)),
                             prediction_above125=float(np.mean(pred[mask]>125))))
    cwd = Path.cwd()
    start = time.time()
    try:
        os.chdir(ROOT/'upstream')
        for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
            configpath = ROOT/f'configs/evaluation_{ds.lower()}_w48_public_code.yaml'
            import yaml
            settings = yaml.safe_load(configpath.read_text())
            bind(configpath)
            splitpath = ROOT/settings['outputs']['split_indices']
            bind(splitpath)
            with np.load(splitpath) as z:
                train_idx, dev_idx = z['predictive_train'], z['predictive_dev']
            assert not np.intersect1d(train_idx, dev_idx).size
            data = CMAPSSDataset(ds, 48, [1000])
            real = data.get_feature_slice(data.get_train_data()).numpy()
            y = data.get_label_slice(data.get_train_data()).numpy().ravel().astype(np.float64)
            tx, ty = data.get_last_data_slice(data.get_test_data())
            tx, ty = tx.numpy(), ty.numpy().ravel().astype(np.float64)
            for kind in ['train', 'test', 'RUL']:
                bind(ROOT/f'upstream/data/{kind}_{ds}.txt')
            rng = np.random.RandomState(20261023)
            selected=[]
            for _, lo, hi in BINS[:-1]:
                candidates=dev_idx[(y[dev_idx]>=lo)&(y[dev_idx]<=hi)]
                selected.extend(rng.choice(candidates, min(600, len(candidates)), replace=False))
            selected=np.array(selected)
            dy=y[selected]
            models=[]
            arrays=dict(real_dev_indices=selected, real_test_labels=ty, real_dev_labels=dy,
                        generation_seeds=np.array(SEEDS), evaluator_seeds=np.array(SEEDS))
            test_pred, dev_pred=[], []
            for e in SEEDS:
                path=ROOT/f'checkpoints/evaluators/real_data_baseline_w48/{ds}_eval{e}.pth'
                bind(path)
                model=torch.load(path,map_location='cpu',weights_only=False).eval()
                models.append(model)
                p=infer(model,tx)
                value=float(np.sqrt(np.mean((p-ty)**2)))
                reference=next(float(r['rmse']) for r in recorded if r['dataset']==ds and int(r['evaluator_seed'])==e)
                assert abs(value-reference)<.001
                replay.append(dict(dataset=ds,evaluator_seed=e,recorded_rmse=reference,replay_rmse=value,abs_difference=abs(value-reference)))
                collect(ds,'test_matched','real',-1,e,ty,p)
                test_pred.append(p)
                p=infer(model,real[selected])
                collect(ds,'dev_matched','real',-1,e,dy,p)
                dev_pred.append(p)
            arrays.update(real_test_predictions=np.array(test_pred), real_dev_predictions=np.array(dev_pred))
            gen_test_pred,gen_dev_pred,gen_test_idx,gen_dev_idx=[],[],[],[]
            for g in SEEDS:
                path=ROOT/f'outputs/{ds}_w48_seed{g}.npz'
                bind(path)
                with np.load(path) as z:
                    generated=z['data']
                    assert generated.shape==real.shape and np.array_equal(z['label'].ravel(),y)
                rng=np.random.RandomState(20261023+g)
                ti=match_indices(y,ty,rng)
                di=match_indices(y,dy,rng)
                gen_test_idx.append(ti)
                gen_dev_idx.append(di)
                pp,dd=[],[]
                for e,model in zip(SEEDS,models):
                    p=infer(model,generated[ti])
                    collect(ds,'test_matched','generated',g,e,ty,p)
                    pp.append(p)
                    p=infer(model,generated[di])
                    collect(ds,'dev_matched','generated',g,e,dy,p)
                    dd.append(p)
                gen_test_pred.append(pp)
                gen_dev_pred.append(dd)
                del generated
                print(ds,'generation seed',g,'complete',flush=True)
            arrays.update(generated_test_predictions=np.array(gen_test_pred),
                          generated_dev_predictions=np.array(gen_dev_pred),
                          generated_test_indices=np.array(gen_test_idx),
                          generated_dev_indices=np.array(gen_dev_idx))
            np.savez_compressed(OUT/f'{ds}_predictions.npz',**arrays)
            del real,models
        write_csv(OUT/'per_model_bins.csv',rows)
        write_csv(OUT/'checkpoint_replay_checks.csv',replay)
        manifest.update(status='complete',elapsed_seconds=time.time()-start,
                        max_replay_rmse_difference=max(r['abs_difference'] for r in replay),
                        output_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()})
    except BaseException as e:
        manifest.update(status='failed',error=repr(e))
        raise
    finally:
        os.chdir(cwd)
        manifest['finished_at']=now()
        mp.write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
