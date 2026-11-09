"""Locate real neighbors of FD003 test errors, calibrated across train engines."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import sys
import traceback

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(ROOT / 'upstream'))
sys.path.insert(0, str(ROOT / 'analysis/diagnostics'))
from test_window_coverage import describe

OUT = ROOT / 'results/diagnostics/fd003_real_neighbors_20261027'
RUN = ROOT / 'runs/diagnostic-fd003-real-neighbors-20261027'
PROTOCOL = ROOT / 'docs/operations/FD003_REAL_NEIGHBOR_AUDIT.md'
REPS = ['level', 'dynamics', 'joint', 'coupling_raw']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def neighbors(distance, pool, engine_ids):
    order = np.argsort(distance, kind='stable')
    normal = pool[order[:5]]
    distinct, seen = [], set()
    for loc in order:
        index = pool[loc]
        if engine_ids[index] not in seen:
            distinct.append(index); seen.add(engine_ids[index])
            if len(distinct) == 5:
                break
    assert len(normal) == len(distinct) == 5
    lookup = dict(zip(pool.tolist(), distance.tolist()))
    return normal, np.array(distinct), np.mean([lookup[i] for i in normal]), np.mean([lookup[i] for i in distinct])


def main():
    OUT.mkdir(parents=True, exist_ok=True); RUN.mkdir(parents=True, exist_ok=True)
    mp = RUN / 'run-manifest.json'
    if mp.exists():
        raise RuntimeError('Preserve existing run; do not overwrite')
    paths = [Path(__file__), PROTOCOL, ROOT / 'analysis/diagnostics/test_window_coverage.py',
             ROOT / 'upstream/data/CMAPSSDataset.py', ROOT / 'results/fd003_w48_public_code_split.npz',
             ROOT / 'results/diagnostics/rul_segment_errors_w48/predictions.csv']
    paths += [ROOT / f'upstream/data/{kind}_FD003.txt' for kind in ['train', 'test', 'RUL']]
    m = dict(run_id=RUN.name, status='running', scope_class='diagnostic', role='supporting',
             scope_contract='scope-20261027-fd003-real-neighbors', no_training=True, started_at=now(),
             command=[sys.executable, str(Path(__file__).relative_to(ROOT))],
             user_authorization='User asks whether high-error FD003 windows have similar real training samples',
             source_snapshot='SOURCE_SNAPSHOT.md', inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
             k=5, rul_caliper=5, stage=[31, 60], tail_percentile=95,
             calibration='Same exactRUL real training queries, own engine excluded from actualpredictortrain reference',
             distinction='Test and train engine IDs are different namespaces',
             caveat='Descriptive feature support, not physical equivalence or unique causal attribution')
    mp.write_text(json.dumps(m, indent=2))
    try:
        from data.CMAPSSDataset import CMAPSSDataset, feature_columns
        cwd = Path.cwd()
        try:
            os.chdir(ROOT / 'upstream')
            d = CMAPSSDataset('FD003', 48, [1000]); frame = d.get_train_data()
            real = d.get_feature_slice(frame).numpy(); y = d.get_label_slice(frame).numpy().ravel()
            tx, ty = d.get_last_data_slice(d.get_test_data()); tx = tx.numpy(); ty = ty.numpy().ravel()
            test_ids = np.array(sorted(i for i in set(d.get_test_data().id) if i != 1000 and i <= d.engine_size))
            keep = (ty >= 31) & (ty <= 60)
            tx = tx[keep]; ty = ty[keep]; test_ids = test_ids[keep]
            ids, starts, ends = [], [], []
            for engine, group in frame.groupby('id', sort=True):
                cycles = group.cycle.to_numpy()
                ids.extend([engine] * (len(group) - 47)); starts.extend(cycles[:-47]); ends.extend(cycles[47:])
        finally:
            os.chdir(cwd)
        ids = np.array(ids); starts = np.array(starts); ends = np.array(ends)
        assert len(ids) == len(y) == 20020 and len(test_ids) == 21
        with np.load(ROOT / 'results/fd003_w48_public_code_split.npz') as z:
            ti = z['predictive_train']; vi = z['predictive_dev']
        assert not np.intersect1d(ti, vi).size
        rf = describe(real); tf = describe(tx); rz = {}; tz = {}
        for rep in ['level', 'dynamics', 'coupling_raw']:
            mu = rf[rep].mean(0); scale = np.maximum(rf[rep].std(0), 1e-6)
            if rep == 'coupling_raw':
                mu = np.zeros(91); scale = np.ones(91)
            rz[rep] = (rf[rep] - mu) / scale / np.sqrt(len(mu))
            tz[rep] = (tf[rep] - mu) / scale / np.sqrt(len(mu))
        rz['joint'] = np.concatenate([rz['level'], rz['dynamics']], axis=1) / np.sqrt(2)
        tz['joint'] = np.concatenate([tz['level'], tz['dynamics']], axis=1) / np.sqrt(2)
        stage = (y >= 31) & (y <= 60)
        calibration = []
        for rep in REPS:
            for label in range(31, 61):
                qi = np.flatnonzero(y == label)
                candidates = ti[stage[ti] & (abs(y[ti] - label) <= 5)]
                matrix = cdist(rz[rep][qi], rz[rep][candidates])
                for j, index in enumerate(qi):
                    permitted = ids[candidates] != ids[index]
                    normal, distinct, distance, dd = neighbors(matrix[j, permitted], candidates[permitted], ids)
                    calibration.append(dict(representation=rep, query_index=int(index), engine_id=int(ids[index]),
                        rul=label, d5=distance, d5_distinct_engines=dd))
            print('CALIBRATION', rep, flush=True)
        cal = pd.DataFrame(calibration); cal.to_csv(OUT / 'real_leave_engine_out_reference.csv', index=False)
        pred = pd.read_csv(ROOT / 'results/diagnostics/rul_segment_errors_w48/predictions.csv')
        pred = pred[(pred.dataset == 'FD003') & pred.rul.between(31, 60)]
        errors = pred.groupby(['engine_id', 'source']).agg(mse=('error', lambda e: float(np.mean(e*e))),
            bias=('error', 'mean')).reset_index()
        high = errors[errors.source == 'generated_headline'].nlargest(5, 'mse').engine_id.tolist()
        focus = sorted(set(high + [75]))
        rows, neighbor_rows, unconstrained = [], [], []
        for j, (engine, label) in enumerate(zip(test_ids, ty)):
            ep = errors[errors.engine_id == engine].set_index('source')
            context = dict(engine_id=int(engine), rul=int(label), top5_error=int(engine in high),
                synthetic_rmse=float(np.sqrt(ep.loc['generated_headline', 'mse'])),
                real_predictor_rmse=float(np.sqrt(ep.loc['real_baseline', 'mse'])))
            for rep in REPS:
                for pool_name, base in [('predictor_train', ti), ('generator_all_real', np.arange(len(y)))]:
                    pool = base[stage[base] & (abs(y[base] - label) <= 5)]
                    distance = cdist(tz[rep][j:j+1], rz[rep][pool]).ravel()
                    normal, distinct, d5, dd = neighbors(distance, pool, ids)
                    reference = cal[(cal.representation == rep) & (cal.rul == label)]
                    assert len(reference) == 100
                    rows.append(dict(**context, representation=rep, pool=pool_name, n_candidates=len(pool),
                        d5=d5, d1=float(distance.min()), d5_distinct_engines=dd,
                        reference_d5_median=float(reference.d5.median()),
                        reference_d5_q95=float(reference.d5.quantile(.95)),
                        percentile_d5=float(np.mean(reference.d5 <= d5) * 100),
                        percentile_distinct=float(np.mean(reference.d5_distinct_engines <= dd) * 100),
                        closest_train_engine=int(ids[normal[0]]), closest_train_rul=int(y[normal[0]]),
                        closest_train_window=int(normal[0]),
                        normal_neighbor_engines=len(np.unique(ids[normal]))))
                    if pool_name == 'predictor_train':
                        for kind, chosen in [('window5', normal), ('engine5', distinct)]:
                            for rank, index in enumerate(chosen, 1):
                                desc_dist = {f'd_{r}': float(np.linalg.norm(tz[r][j] - rz[r][index])) for r in REPS}
                                neighbor_rows.append(dict(**context, selection_rep=rep, neighbor_kind=kind,
                                    rank=rank, train_window=int(index), train_engine=int(ids[index]),
                                    train_rul=int(y[index]), cycle_start=int(starts[index]), cycle_end=int(ends[index]),
                                    **desc_dist))
            distance = cdist(tz['joint'][j:j+1], rz['joint'][ti]).ravel()
            normal, distinct, _, _ = neighbors(distance, ti, ids)
            unconstrained.append(dict(**context, closest_train_engine=int(ids[normal[0]]),
                closest_train_rul=int(y[normal[0]]), closest_distance=float(distance.min()),
                neighbor_engine_ruls=json.dumps(y[distinct].astype(int).tolist())))
        table = pd.DataFrame(rows); table.to_csv(OUT / 'test_support.csv', index=False)
        details = pd.DataFrame(neighbor_rows); details.to_csv(OUT / 'nearest_real_windows.csv', index=False)
        pd.DataFrame(unconstrained).to_csv(OUT / 'unrestricted_rul_neighbors.csv', index=False)
        main = table[table.pool == 'predictor_train']
        main[main.engine_id.isin(focus)].to_csv(OUT / 'focus_cases.csv', index=False)
        summary = main.groupby('representation').agg(n_queries=('engine_id', 'size'),
            n_above95=('percentile_d5', lambda v: int((v > 95).sum())),
            n_distinct_above95=('percentile_distinct', lambda v: int((v > 95).sum())),
            median_percentile=('percentile_d5', 'median')).reset_index()
        summary.to_csv(OUT / 'summary.csv', index=False)
        for engine in focus:
            j = int(np.flatnonzero(test_ids == engine)[0])
            near = details[(details.engine_id == engine) & (details.selection_rep == 'joint')
                           & (details.neighbor_kind == 'engine5')].sort_values('rank').head(3)
            fig, axes = plt.subplots(4, 4, figsize=(12, 9))
            for c, ax in enumerate(axes.flat):
                if c >= len(feature_columns):
                    ax.axis('off'); continue
                ax.plot(tx[j, :, c], color='black', linewidth=1.7, label=f'Test{engine}, RUL{int(ty[j])}')
                for k, item in enumerate(near.itertuples()):
                    ax.plot(real[item.train_window, :, c], alpha=.75, linewidth=.9,
                        label=f'Train{item.train_engine}, RUL{item.train_rul}')
                ax.set_title(feature_columns[c], fontsize=9); ax.grid(alpha=.15)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8)
            fig.suptitle(f'FD003 test engine{engine}: real training neighbors | joint level+dynamics\nThree DISTINCT training engines, RUL±5; original train-fit normalized signals')
            fig.tight_layout(rect=(0, .04, 1, .93))
            fig.savefig(OUT / f'engine{engine}_real_neighbors.png', dpi=160); plt.close(fig)
        assert len(table) == 168 and len(details) == 840 and np.isfinite(table.d5).all()
        m.update(status='complete', focus_top5_error_engines=list(map(int, high)), focus_with_prior75=focus,
            n_test_windows=21, n_calibration_windows=3000, n_calibration_rows=len(cal), summary=summary.to_dict('records'),
            outputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in OUT.iterdir() if p.is_file()})
        print(main[main.engine_id.isin(focus)].round(3).to_string(index=False), flush=True)
    except BaseException:
        m.update(status='failed', error=traceback.format_exc()); raise
    finally:
        m['finished_at'] = now(); mp.write_text(json.dumps(m, indent=2))


if __name__ == '__main__':
    main()
