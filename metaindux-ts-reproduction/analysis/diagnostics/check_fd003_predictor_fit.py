"""Audit saved FD003 predictors on their fitting data and real data; no training."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = ROOT / 'results/diagnostics/fd003_predictor_fit_20261026'
RUN = ROOT / 'runs/diagnostic-fd003-predictor-fit-20261026'
PROTOCOL = ROOT / 'docs/operations/FD003_PREDICTOR_FIT_DIAGNOSTIC.md'
SEEDS = [3, 13, 23, 33, 43]
BINS = [('0-30', 0, 30), ('31-60', 31, 60), ('61-90', 61, 90),
        ('91-124', 91, 124), ('125', 125, 125), ('all', 0, 125)]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def predict(model, x):
    parts = []
    with torch.inference_mode():
        for start in range(0, len(x), 256):
            parts.append(model(torch.from_numpy(x[start:start + 256])).numpy().ravel())
    p = np.concatenate(parts).astype(float)
    assert p.shape == (len(x),) and np.isfinite(p).all()
    return p


def metrics(y, p):
    e = p - y
    return dict(n_samples=len(y), mse=float(np.mean(e ** 2)), rmse=float(np.sqrt(np.mean(e ** 2))),
                mae=float(np.mean(abs(e))), bias=float(e.mean()))


def record(meta, domain, y, p, truth, rows, exact):
    by_label = {}
    for label in np.unique(y):
        mask = y == label
        value = metrics(y[mask], p[mask])
        by_label[int(label)] = value
        exact.append(dict(**meta, domain=domain, rul=int(label), **value))
    for name, lo, hi in BINS:
        mask = (y >= lo) & (y <= hi)
        if not mask.any():
            continue
        target = truth[(truth >= lo) & (truth <= hi)]
        labels, counts = np.unique(target, return_counts=True)
        covered = sum(n for label, n in zip(labels, counts) if int(label) in by_label)
        matched = {}
        for key in ['mse', 'mae', 'bias']:
            matched['matched_' + key] = (float(sum(n * by_label[int(label)][key]
                for label, n in zip(labels, counts)) / len(target))
                if covered == len(target) and len(target) else np.nan)
        matched['matched_rmse'] = float(np.sqrt(matched['matched_mse']))
        rows.append(dict(**meta, domain=domain, bin=name, **metrics(y[mask], p[mask]),
                         matched_test_label_mass=covered / len(target) if len(target) else np.nan,
                         **matched))


def summarize(rows):
    df = pd.DataFrame(rows)
    result = []
    for (source, domain, stage), g in df.groupby(['source', 'domain', 'bin'], sort=False):
        result.append(dict(source=source, domain=domain, bin=stage, n_models=len(g),
            n_samples_per_model=int(g.n_samples.iloc[0]),
            pooled_rmse=float(np.sqrt(g.mse.mean())), mean_model_rmse=float(g.rmse.mean()),
            model_rmse_sd=float(g.rmse.std()), mae=float(g.mae.mean()), bias=float(g.bias.mean()),
            matched_pooled_rmse=float(np.sqrt(g.matched_mse.mean()))
                if np.isfinite(g.matched_mse).all() else np.nan,
            matched_bias=float(g.matched_bias.mean()),
            min_matched_test_label_mass=float(g.matched_test_label_mass.min())))
    return df, pd.DataFrame(result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    (OUT / 'predictions').mkdir(exist_ok=True)
    mp = RUN / 'run-manifest.json'
    if mp.exists():
        raise RuntimeError('Preserve existing run; do not overwrite')
    sys.path.insert(0, str(ROOT / 'upstream'))
    torch.set_num_threads(2)
    anchor = ROOT / 'results/diagnostics/fd003_local_anchor_20261026'
    formal_path = ROOT / 'results/fd003_w48_public_code_manifest.json'
    formal = json.loads(formal_path.read_text())
    jobs = []
    for g in SEEDS:
        for e in SEEDS:
            jobs.append(dict(source='headline', generation_seed=g, evaluator_seed=e,
                checkpoint=str(Path(formal['config']['outputs']['evaluator_checkpoints']) / f'gen{g}_eval{e}.pth')))
    for e in SEEDS:
        jobs.append(dict(source='real_baseline', generation_seed=-1, evaluator_seed=e,
            checkpoint=f'checkpoints/evaluators/real_data_baseline_w48/FD003_eval{e}.pth'))
    for r in pd.read_csv(anchor / 'raw.csv').itertuples():
        jobs.append(dict(source='anchor_' + r.arm, generation_seed=3,
                         evaluator_seed=r.evaluator_seed, checkpoint=r.checkpoint))
    paths = [Path(__file__), PROTOCOL, ROOT / 'upstream/data/CMAPSSDataset.py',
             ROOT / 'upstream/models/LSTM.py', ROOT / 'scripts/evaluate_real_data_baseline.py',
             formal_path, ROOT / 'results/fd003_w48_public_code_split.npz',
             ROOT / 'results/diagnostics/rul_segment_errors_w48/predictions.csv',
             anchor / 'raw.csv', anchor / 'predictions.csv', anchor / 'replacement_indices.npz',
             ROOT / 'runs/ext-local-anchor-fd003-seed3-20261026/run-manifest.json']
    paths += [ROOT / f'outputs/FD003_w48_seed{g}.npz' for g in SEEDS]
    paths += [ROOT / f'upstream/data/{kind}_FD003.txt' for kind in ['train', 'test', 'RUL']]
    paths += [ROOT / j['checkpoint'] for j in jobs]
    status = subprocess.run(['git', 'status', '--short'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    diff = subprocess.run(['git', 'diff', '--binary'], cwd=ROOT, capture_output=True, check=True).stdout
    (RUN / 'tracked-source.patch').write_bytes(diff)
    (RUN / 'git-status.txt').write_text(status)
    m = dict(run_id=RUN.name, scope_class='diagnostic', role='supporting',
             scope_contract='scope-20261026-fd003-predictor-fit', status='running', started_at=now(),
             user_authorization='User explicitly continued annotated saved-predictor train/dev/real-test check',
             no_training=True, command=[sys.executable, str(Path(__file__).relative_to(ROOT))],
             source_snapshot='SOURCE_SNAPSHOT.md',
             source_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
             tracked_patch_sha256=hashlib.sha256(diff).hexdigest(), inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
             protocol=str(PROTOCOL.relative_to(ROOT)), jobs=jobs, bins=BINS,
             inference=dict(device='cpu', batch_size=256, threads=2, torch=str(torch.__version__),
                            numpy=np.__version__, pandas=pd.__version__, os=platform.platform(), python=sys.version),
             aggregation='pooled squared errors then sqrt; native and exact-RUL real-test-histogram weighted',
             caveats=['Real train/dev overlap within engines; real-source reference is not an independent generator holdout',
                      'Anchor10 contains209 real donor windows; keep separate from headline'])
    mp.write_text(json.dumps(m, indent=2))
    rows, exact, checks = [], [], []
    try:
        from data.CMAPSSDataset import CMAPSSDataset
        cwd = Path.cwd()
        try:
            os.chdir(ROOT / 'upstream')
            d = CMAPSSDataset('FD003', 48, [1000])
            real = d.get_feature_slice(d.get_train_data()).numpy()
            y = d.get_label_slice(d.get_train_data()).numpy().ravel().astype(float)
            tx, ty = d.get_last_data_slice(d.get_test_data())
            tx = tx.numpy(); truth = ty.numpy().ravel().astype(float)
            test_ids = np.array(sorted(i for i in set(d.get_test_data().id) if i != 1000 and i <= d.engine_size))
            engine_ids = np.concatenate([np.full(len(g) - 47, i) for i, g in d.get_train_data().groupby('id')])
        finally:
            os.chdir(cwd)
        with np.load(ROOT / 'results/fd003_w48_public_code_split.npz') as z:
            ti = z['predictive_train']; vi = z['predictive_dev']
        with np.load(anchor / 'replacement_indices.npz') as z:
            targets = z['target']; donors = z['donor']
            assert np.array_equal(ti, z['train_indices']) and np.array_equal(vi, z['dev_indices'])
        assert not np.intersect1d(ti, vi).size and len(ti) + len(vi) == len(y) == 20020
        assert len(truth) == len(test_ids) == 100 and len(targets) == len(donors) == 209
        assert np.isin(targets, ti).all() and np.isin(donors, ti).all()
        assert np.array_equal(y[targets], y[donors])
        np.savez_compressed(OUT / 'shared_indices_labels.npz', labels=y, train_indices=ti, dev_indices=vi,
                            real_train_engine_ids=engine_ids, test_labels=truth, test_engine_ids=test_ids,
                            targets=targets, donors=donors)
        old = pd.read_csv(ROOT / 'results/diagnostics/rul_segment_errors_w48/predictions.csv')
        old = old[old.dataset == 'FD003']
        ap = pd.read_csv(anchor / 'predictions.csv')
        current_seed = None
        generated = None
        ti_retained = ti[~np.isin(ti, targets)]
        for j in jobs:
            started = time.time()
            source = j['source']; g = j['generation_seed']; e = j['evaluator_seed']
            model = torch.load(ROOT / j['checkpoint'], map_location='cpu', weights_only=False).eval()
            assert not model.training
            pr = predict(model, real); pt = predict(model, tx)
            if source.startswith('anchor_'):
                arm = source[len('anchor_'):]
                previous = ap[(ap.arm == arm) & (ap.evaluator_seed == e)].sort_values('engine_id')
            else:
                previous = old[(old.source == ('real_baseline' if source == 'real_baseline' else 'generated_headline'))
                    & (old.generation_seed == g) & (old.evaluator_seed == e)].sort_values('engine_id')
            assert len(previous) == 100 and np.array_equal(previous.engine_id.to_numpy(), test_ids)
            difference = float(np.max(abs(pt - previous.prediction.to_numpy())))
            assert difference < .001, (j, difference)
            checks.append(dict(**j, max_test_prediction_difference=difference,
                               test_rmse=float(np.sqrt(np.mean((pt - truth) ** 2)))))
            domains = {'real_test': (truth, pt)}
            saved = dict(real_train_source_prediction=pr, real_test_prediction=pt)
            if source == 'real_baseline':
                domains.update(fit_train=(y[ti], pr[ti]), real_dev=(y[vi], pr[vi]))
            else:
                if current_seed != g:
                    with np.load(ROOT / f'outputs/FD003_w48_seed{g}.npz') as z:
                        generated = z['data']
                        assert generated.shape == real.shape and np.array_equal(y, z['label'].ravel())
                    current_seed = g
                pg = predict(model, generated)
                fit = pg.copy()
                if source == 'anchor_anchor10':
                    fit[targets] = pr[donors]
                    assert np.array_equal(fit[vi], pg[vi])
                    with torch.inference_mode():
                        direct = model(torch.from_numpy(real[donors[:16]])).numpy().ravel()
                    assert np.max(abs(direct - fit[targets[:16]])) < .001
                    domains.update(anchor_train=(y[donors], pr[donors]),
                                   synthetic_train_retained=(y[ti_retained], pg[ti_retained]))
                domains.update(fit_train=(y[ti], fit[ti]), synthetic_dev=(y[vi], pg[vi]),
                    real_train_reference=(y[ti], pr[ti]), real_dev_reference=(y[vi], pr[vi]))
                saved.update(synthetic_prediction=pg, actual_fit_prediction=fit[ti])
            meta = dict(source=source, generation_seed=g, evaluator_seed=e)
            for domain, (labels, estimates) in domains.items():
                record(meta, domain, labels, estimates, truth, rows, exact)
            np.savez_compressed(OUT / 'predictions' / f'{source}_gen{g}_eval{e}.npz', **saved)
            m['completed_checkpoints'] = len(checks)
            mp.write_text(json.dumps(m, indent=2))
            print('DONE', source, g, e, f'{time.time()-started:.1f}s', 'replay', difference, flush=True)
        per_model, summary = summarize(rows)
        per_model.to_csv(OUT / 'per_model_segments.csv', index=False)
        pd.DataFrame(exact).to_csv(OUT / 'per_model_exact_rul.csv', index=False)
        summary.to_csv(OUT / 'summary.csv', index=False)
        pd.DataFrame(checks).to_csv(OUT / 'checkpoint_replay_checks.csv', index=False)
        generation = per_model.groupby(['source', 'generation_seed', 'domain', 'bin']).agg(
            mse=('mse', 'mean'), matched_mse=('matched_mse', 'mean'),
            mean_model_rmse=('rmse', 'mean'), bias=('bias', 'mean')).reset_index()
        generation['pooled_rmse'] = np.sqrt(generation.mse)
        generation['matched_pooled_rmse'] = np.sqrt(generation.matched_mse)
        generation.to_csv(OUT / 'by_generation_seed.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
        names = [b[0] for b in BINS[:-1]]
        domain_labels = [('fit_train', 'Synthetic train'), ('synthetic_dev', 'Synthetic dev'),
                         ('real_dev_reference', 'Real training-source reference'), ('real_test', 'Real test')]
        for ax, column, title in zip(axes, ['pooled_rmse', 'matched_pooled_rmse'],
                                    ['Native RUL distribution', 'Matched to real-test exact RUL histogram']):
            for domain, label in domain_labels:
                v = summary[(summary.source == 'headline') & (summary.domain == domain)].set_index('bin').reindex(names)
                ax.plot(range(5), v[column], marker='o', label=label)
            v = summary[(summary.source == 'real_baseline') & (summary.domain == 'real_test')].set_index('bin').reindex(names)
            ax.plot(range(5), v[column], '--', marker='s', label='Real-trained: real test')
            ax.set_xticks(range(5), names)
            ax.set(title=title, ylabel='RMSE', xlabel='True capped RUL')
            ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.suptitle('FD003 saved25 headline predictors | train/dev fit versus real-data transfer\nPooled errors, no retraining; reference windows were seen by generator')
        fig.tight_layout(rect=(0, 0, 1, .89))
        for ext in ['png', 'pdf']:
            fig.savefig(OUT / f'fit_vs_transfer.{ext}', dpi=180)
        plt.close(fig)
        assert len(checks) == 40 and per_model.mse.notna().all()
        m.update(status='complete', max_test_replay_difference=max(c['max_test_prediction_difference'] for c in checks),
                 n_primary_models=25, n_reference_models=5, n_anchor_followup_models=10,
                 real_train_count=len(ti), real_dev_count=len(vi), real_test_engines=len(truth),
                 outputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in OUT.rglob('*') if p.is_file()})
        print(summary[(summary.bin == '31-60')].round(4).to_string(index=False), flush=True)
    except BaseException:
        m.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        m['finished_at'] = now()
        mp.write_text(json.dumps(m, indent=2))


if __name__ == '__main__':
    main()
