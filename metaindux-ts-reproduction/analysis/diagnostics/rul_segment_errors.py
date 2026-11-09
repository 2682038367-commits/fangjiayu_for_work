"""Replay saved full-data w48 predictors on real test engines, without training."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/diagnostics/rul_segment_errors_w48'
RUN = ROOT / 'runs/diagnostic-rul-segment-w48-20261022'
SEEDS = [3, 13, 23, 33, 43]
BINS = [('0-30', 0, 30), ('31-60', 31, 60), ('61-90', 61, 90),
        ('91-124', 91, 124), ('125 (capped)', 125, 125)]


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / 'upstream'))
    from data.CMAPSSDataset import CMAPSSDataset
    manifest = dict(run_id=RUN.name, scope_class='diagnostic', role='supporting',
                    status='running', changed_factor='none: saved checkpoint inference only',
                    command=f'{sys.executable} analysis/diagnostics/rul_segment_errors.py',
                    started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    device='cpu', torch=str(torch.__version__), inputs_sha256={},
                    bins=BINS, aggregation='pooled_rmse=sqrt(mean model MSE); no ensemble prediction',
                    no_training=True)
    def bind(p):
        manifest['inputs_sha256'][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    bind(Path(__file__))
    (RUN / 'run-manifest.json').write_text(json.dumps(manifest, indent=2))
    summary, contributions, checks, per_model, predictions = [], [], [], [], []
    baseline_raw = read_csv(ROOT / 'results/real_data_baseline_w48.csv')
    bind(ROOT / 'results/real_data_baseline_w48.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    cwd = Path.cwd()
    try:
        os.chdir(ROOT / 'upstream')
        for ds, ax in zip(['FD001', 'FD002', 'FD003', 'FD004'], axes.flat):
            path = (ROOT / 'results/frequency_threshold_validation/fd002_public_arm_seven_seeds_manifest.json'
                    if ds == 'FD002' else ROOT / f'results/{ds.lower()}_w48_public_code_manifest.json')
            bind(path)
            formal = json.loads(path.read_text())
            assert formal['status'] == 'complete'
            settings = formal['config']
            raw_path = ROOT / settings['outputs']['raw_csv']
            raw = read_csv(raw_path)
            bind(raw_path)
            data = CMAPSSDataset(ds, 48, [1000])
            x, y_tensor = data.get_last_data_slice(data.get_test_data())
            y = y_tensor.numpy().ravel().astype(np.float64)
            ids = sorted(set(data.get_test_data()['id']) - {1000})
            ids = [i for i in ids if i <= data.engine_size]
            assert len(ids) == len(y)
            for kind in ['train', 'test', 'RUL']:
                bind(ROOT / f'upstream/data/{kind}_{ds}.txt')
            pred = {'generated_common5': [], 'generated_headline': [], 'real_baseline': []}
            for source in ['generated_headline', 'real_baseline']:
                gs = settings['experiment']['generation_seeds'] if source == 'generated_headline' else [-1]
                for g in gs:
                    for e in SEEDS:
                        checkpoint = (ROOT / settings['outputs']['evaluator_checkpoints'] / f'gen{g}_eval{e}.pth'
                                      if source == 'generated_headline' else
                                      ROOT / f'checkpoints/evaluators/real_data_baseline_w48/{ds}_eval{e}.pth')
                        bind(checkpoint)
                        model = torch.load(checkpoint, map_location='cpu', weights_only=False).eval()
                        with torch.no_grad():
                            p = model(x).numpy().ravel().astype(np.float64)
                        assert p.shape == y.shape and np.isfinite(p).all()
                        rmse = float(np.sqrt(np.mean((p-y)**2)))
                        source_rows = raw if source == 'generated_headline' else baseline_raw
                        match = [r for r in source_rows if int(r['evaluator_seed']) == e and
                                 ((source == 'generated_headline' and int(r['generation_seed']) == g) or
                                  (source == 'real_baseline' and r['dataset'] == ds))]
                        assert len(match) == 1
                        recorded = float(match[0]['rmse'])
                        checks.append(dict(dataset=ds, source=source, generation_seed=g,
                                           evaluator_seed=e, recorded_rmse=recorded, replay_rmse=rmse,
                                           abs_difference=abs(recorded-rmse)))
                        if abs(recorded-rmse) > .001:
                            raise ValueError(f'checkpoint replay mismatch: {checkpoint}: {rmse} vs {recorded}')
                        pred[source].append(p)
                        if source == 'generated_headline' and g in SEEDS:
                            pred['generated_common5'].append(p)
                        for engine, label, estimate in zip(ids, y, p):
                            predictions.append(dict(dataset=ds, source=source, generation_seed=g,
                                                    evaluator_seed=e, engine_id=engine, rul=label,
                                                    prediction=float(estimate), error=float(estimate-label)))
                        for name, lo, hi in BINS + [('all', 0, 125)]:
                            mask = (y >= lo) & (y <= hi)
                            if not mask.any():
                                continue
                            err = p[mask]-y[mask]
                            per_model.append(dict(dataset=ds, source=source, generation_seed=g,
                                                  evaluator_seed=e, bin=name, n_engines=int(mask.sum()),
                                                  mse=float(np.mean(err**2)), rmse=float(np.sqrt(np.mean(err**2))),
                                                  mae=float(np.mean(abs(err))), bias=float(np.mean(err))))
                        del model
            for source, pp in pred.items():
                err = np.array(pp)-y[None, :]
                for name, lo, hi in BINS + [('all', 0, 125)]:
                    mask = (y >= lo) & (y <= hi)
                    if not mask.any():
                        continue
                    a = err[:, mask]
                    summary.append(dict(dataset=ds, source=source, bin=name,
                                        n_engines=int(mask.sum()), n_models=len(pp),
                                        test_fraction=float(mask.mean()), mse=float(np.mean(a*a)),
                                        pooled_rmse=float(np.sqrt(np.mean(a*a))),
                                        mean_model_rmse=float(np.sqrt(np.mean(a*a, axis=1)).mean()),
                                        mae=float(np.mean(abs(a))), bias=float(a.mean()),
                                        contribution_to_total_mse=float(np.sum(np.mean(a*a, axis=0))/len(y))))
            for gen_source in ['generated_common5', 'generated_headline']:
                total_excess = sum(r['contribution_to_total_mse'] for r in summary if r['dataset']==ds and r['bin']!='all' and r['source']==gen_source) - sum(r['contribution_to_total_mse'] for r in summary if r['dataset']==ds and r['bin']!='all' and r['source']=='real_baseline')
                for name, _, _ in BINS:
                    a = next(r for r in summary if r['dataset']==ds and r['bin']==name and r['source']==gen_source)
                    b = next(r for r in summary if r['dataset']==ds and r['bin']==name and r['source']=='real_baseline')
                    excess = a['contribution_to_total_mse']-b['contribution_to_total_mse']
                    contributions.append(dict(dataset=ds, source=gen_source, bin=name,
                                              n_engines=a['n_engines'], test_fraction=a['test_fraction'],
                                              generated_mse=a['mse'], real_mse=b['mse'],
                                              mse_penalty=a['mse']-b['mse'],
                                              weighted_mse_penalty=excess,
                                              share_of_total_penalty=excess/total_excess if total_excess else None))
            for source, label in [('generated_common5', 'Synthetic-trained (5 generation seeds)'),
                                  ('real_baseline', 'Real-trained baseline')]:
                vals=[next(r['pooled_rmse'] for r in summary if r['dataset']==ds and r['bin']==b[0] and r['source']==source) for b in BINS]
                ax.plot(range(len(BINS)), vals, marker='o', label=label)
            ax.set_title(ds)
            ax.set_xticks(range(len(BINS)), [b[0] for b in BINS], rotation=20)
            ax.set_ylabel('RMSE (pooled squared errors)')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
            print(ds, 'checkpoint replay verified', flush=True)
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        os.chdir(cwd)
        manifest['finished_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        (RUN / 'run-manifest.json').write_text(json.dumps(manifest, indent=2))
    for filename, rows in [('summary.csv', summary), ('mse_penalty_contributions.csv', contributions),
                           ('checkpoint_replay_checks.csv', checks), ('per_model_bins.csv', per_model),
                           ('predictions.csv', predictions)]:
        write_csv(OUT / filename, rows)
    fig.suptitle('Full-data w48: errors by real test RUL; saved-checkpoint inference only')
    fig.tight_layout()
    fig.savefig(OUT / 'rmse_by_rul.png', dpi=180)
    fig.savefig(OUT / 'rmse_by_rul.pdf')
    plt.close(fig)
    manifest.update(status='complete', checkpoint_count=len(checks),
                    max_replay_rmse_difference=max(r['abs_difference'] for r in checks),
                    outputs_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in OUT.iterdir() if p.is_file()})
    (RUN / 'run-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k:manifest[k] for k in ['status','checkpoint_count','max_replay_rmse_difference']}))


if __name__ == '__main__':
    main()
