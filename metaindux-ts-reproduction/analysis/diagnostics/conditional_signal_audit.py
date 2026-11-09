"""RUL-matched full-data signal diagnostics; no training or model changes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/diagnostics/conditional_signal_w48'
RUN = ROOT / 'runs/diagnostic-conditional-signal-w48-20261022'
SEEDS = [3, 13, 23, 33, 43]
BINS = [('0-30', 0, 30), ('31-60', 31, 60), ('61-90', 61, 90),
        ('91-124', 91, 124), ('125', 125, 125)]
CAP = 600


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def corr(a, b=None):
    a = a.reshape(-1, a.shape[-1])
    b = a if b is None else b.reshape(-1, b.shape[-1])
    a = a-a.mean(0)
    b = b-b.mean(0)
    denom = np.sqrt((a*a).sum(0)[:, None]*(b*b).sum(0)[None, :])
    return np.divide(a.T@b, denom, out=np.zeros_like(denom), where=denom > 1e-15)


def features(x):
    centered = x-x.mean(1, keepdims=True)
    d = np.diff(x, axis=1)
    dd = np.diff(x, n=2, axis=1)
    power = abs(np.fft.rfft(centered, axis=1))**2
    hf = np.fft.rfftfreq(x.shape[1]) >= .25
    energy = power.sum((0, 1))
    hf_fraction = np.divide(power[:, hf, :].sum((0, 1)), energy,
                            out=np.zeros_like(energy), where=energy > 1e-15)
    lagcorr = []
    for lag in [-2, -1, 0, 1, 2]:
        a, b = (d[:, -lag:], d[:, :lag]) if lag < 0 else ((d[:, :-lag], d[:, lag:]) if lag > 0 else (d, d))
        lagcorr.append(corr(a, b))
    return dict(level=x.mean(1), end=x[:, -1, :],
                temporal_sd=np.sqrt(np.mean(centered**2, axis=(0, 1))),
                diff_rms=np.sqrt(np.mean(d*d, axis=(0, 1))),
                curvature_rms=np.sqrt(np.mean(dd*dd, axis=(0, 1))),
                hf=hf_fraction, within_corr=corr(centered), diff_corr=corr(d),
                lagcorr=np.array(lagcorr))


def compare(a, b):
    tri = np.triu_indices(14, 1)
    return dict(mean_wasserstein=float(np.mean([wasserstein_distance(a['level'][:, c], b['level'][:, c]) for c in range(14)])),
                end_wasserstein=float(np.mean([wasserstein_distance(a['end'][:, c], b['end'][:, c]) for c in range(14)])),
                centroid_mae=float(np.mean(abs(a['level'].mean(0)-b['level'].mean(0)))),
                temporal_sd_ratio=float(b['temporal_sd'].mean()/a['temporal_sd'].mean()),
                diff_rms_ratio=float(b['diff_rms'].mean()/a['diff_rms'].mean()),
                curvature_rms_ratio=float(b['curvature_rms'].mean()/a['curvature_rms'].mean()),
                highfreq_fraction_real=float(a['hf'].mean()), highfreq_fraction_other=float(b['hf'].mean()),
                highfreq_fraction_difference=float((b['hf']-a['hf']).mean()),
                within_corr_mae=float(np.mean(abs(a['within_corr'][tri]-b['within_corr'][tri]))),
                diff_corr_mae=float(np.mean(abs(a['diff_corr'][tri]-b['diff_corr'][tri]))),
                lagcorr_mae=float(np.mean(abs(a['lagcorr'][:, tri[0], tri[1]]-b['lagcorr'][:, tri[0], tri[1]]))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT/'upstream'))
    from data.CMAPSSDataset import CMAPSSDataset, feature_columns
    now = lambda: datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    manifest = dict(run_id=RUN.name, scope_class='diagnostic', role='supporting',
                    scope_contract='scope-20261022-conditional-signal-diagnostic',
                    status='running', started_at=now(), inputs_sha256={}, seeds=SEEDS,
                    cap_windows_per_bin=CAP, bins=BINS, device='cpu', no_training=True,
                    command=f'{sys.executable} analysis/diagnostics/conditional_signal_audit.py',
                    numpy=np.__version__, source_snapshot='SOURCE_SNAPSHOT.md',
                    sampling='disjoint real engine halves, exact-RUL matched A/B/generated; window bootstrap not used')
    def bind(p):
        manifest['inputs_sha256'][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    bind(Path(__file__))
    bind(ROOT/'upstream/data/CMAPSSDataset.py')
    bind(ROOT/'SOURCE_SNAPSHOT.md')
    mp = RUN/'run-manifest.json'
    mp.write_text(json.dumps(manifest, indent=2))
    rows, sensor_rows, pairs, condition_rows = [], [], [], []
    indices, matrices = {}, {}
    cwd = Path.cwd()
    start = time.time()
    try:
        os.chdir(ROOT/'upstream')
        for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
            data = CMAPSSDataset(ds, 48, [1000])
            frame = data.get_train_data()
            real = data.get_feature_slice(frame).numpy()
            y = data.get_label_slice(frame).numpy().ravel()
            ids = np.concatenate([np.repeat(i, max(len(g)-47, 0)) for i, g in frame.groupby('id', sort=True) if i != 1000])
            assert len(ids) == len(y) == len(real)
            scale = frame[feature_columns].std(ddof=0).to_numpy()
            assert (scale > 1e-8).all()
            # Global training sensor SD only supplies a common unit; input data are not modified.
            for kind in ['train', 'test', 'RUL']:
                bind(ROOT/f'upstream/data/{kind}_{ds}.txt')
            for seed in SEEDS:
                path = ROOT/f'outputs/{ds}_w48_seed{seed}.npz'
                bind(path)
                with np.load(path) as z:
                    generated = z['data']
                    assert generated.shape == real.shape and np.array_equal(y, z['label'].ravel())
                rng = np.random.RandomState(20261022+seed)
                engines = rng.permutation(np.unique(ids))
                half_a = np.isin(ids, engines[:len(engines)//2])
                centroid = {'real_A': [], 'real_B': [], 'generated': []}
                for name, lo, hi in BINS:
                    ai, bi = [], []
                    for label in range(lo, hi+1):
                        ia = np.flatnonzero((y == label) & half_a)
                        ib = np.flatnonzero((y == label) & ~half_a)
                        n = min(len(ia), len(ib))
                        ai.extend(rng.choice(ia, n, replace=False))
                        bi.extend(rng.choice(ib, n, replace=False))
                    take = rng.choice(len(ai), min(CAP, len(ai)), replace=False)
                    ai, bi = np.array(ai)[take], np.array(bi)[take]
                    gi = np.empty(len(ai), dtype=int)
                    for label in np.unique(y[ai]):
                        loc = np.flatnonzero(y[ai] == label)
                        gi[loc] = rng.choice(np.flatnonzero(y == label), len(loc), replace=False)
                    assert np.array_equal(y[ai], y[bi]) and np.array_equal(y[ai], y[gi])
                    key = f'{ds}_seed{seed}_rul{name}'
                    indices[key+'_real_a'] = ai
                    indices[key+'_real_b'] = bi
                    indices[key+'_generated'] = gi
                    fa = features(real[ai].astype(np.float64)/scale)
                    fb = features(real[bi].astype(np.float64)/scale)
                    fg = features(generated[gi].astype(np.float64)/scale)
                    for label, f in [('real_A', fa), ('real_B', fb), ('generated', fg)]:
                        centroid[label].append(f['level'].mean(0))
                        matrices[key+'_'+label+'_within_corr'] = f['within_corr']
                        matrices[key+'_'+label+'_diff_corr'] = f['diff_corr']
                    for other, f in [('real_B', fb), ('generated', fg)]:
                        rows.append(dict(dataset=ds, seed=seed, bin=name, comparison=other,
                                         n_windows=len(ai), real_a_engines=len(np.unique(ids[ai])),
                                         real_b_engines=len(np.unique(ids[bi])), **compare(fa, f)))
                        for c, sensor in enumerate(feature_columns):
                            sensor_rows.append(dict(dataset=ds, seed=seed, bin=name, comparison=other,
                                sensor=sensor, centroid_real=float(fa['level'][:, c].mean()),
                                centroid_other=float(f['level'][:, c].mean()),
                                level_wasserstein=wasserstein_distance(fa['level'][:, c], f['level'][:, c]),
                                diff_rms_ratio=float(f['diff_rms'][c]/max(fa['diff_rms'][c], 1e-15)),
                                curvature_rms_ratio=float(f['curvature_rms'][c]/max(fa['curvature_rms'][c], 1e-15))))
                        for c in range(14):
                            for d in range(c+1, 14):
                                pairs.append(dict(dataset=ds, seed=seed, bin=name, comparison=other,
                                    sensor_a=feature_columns[c], sensor_b=feature_columns[d],
                                    real_diff_corr=float(fa['diff_corr'][c,d]),
                                    other_diff_corr=float(f['diff_corr'][c,d]),
                                    abs_difference=float(abs(fa['diff_corr'][c,d]-f['diff_corr'][c,d]))))
                    del fa, fb, fg
                cent = {k:np.array(v) for k,v in centroid.items()}
                for i in range(5):
                    for j in range(i+1, 5):
                        a = cent['real_A'][j]-cent['real_A'][i]
                        for other in ['real_B', 'generated']:
                            b = cent[other][j]-cent[other][i]
                            condition_rows.append(dict(dataset=ds, seed=seed, comparison=other,
                                bin_a=BINS[i][0], bin_b=BINS[j][0],
                                real_distance=float(np.linalg.norm(a)), other_distance=float(np.linalg.norm(b)),
                                distance_ratio=float(np.linalg.norm(b)/max(np.linalg.norm(a), 1e-15)),
                                direction_cosine=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-15))))
                del generated
                print(ds, 'seed', seed, 'complete', flush=True)
            del real
            mp.write_text(json.dumps(manifest, indent=2))
        write_csv(OUT/'metrics_by_seed_bin.csv', rows)
        write_csv(OUT/'sensor_metrics.csv', sensor_rows)
        write_csv(OUT/'channel_pairs.csv', pairs)
        write_csv(OUT/'conditional_separation.csv', condition_rows)
        np.savez_compressed(OUT/'selected_indices.npz', **indices)
        np.savez_compressed(OUT/'correlation_matrices.npz', **matrices)
        manifest.update(status='complete', elapsed_seconds=time.time()-start,
                        output_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()})
    except BaseException as e:
        manifest.update(status='failed', error=repr(e))
        raise
    finally:
        os.chdir(cwd)
        manifest['finished_at'] = now()
        mp.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
