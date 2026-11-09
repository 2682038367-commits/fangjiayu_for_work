"""Test75 -> real neighborhood -> generated neighbors -> frozen predictors."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import platform
import sys
import traceback

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(ROOT / 'upstream'))
sys.path.insert(0, str(ROOT / 'analysis/diagnostics'))
from test_window_coverage import describe

OUT = ROOT / 'results/diagnostics/fd003_engine75_trace_20261027'
RUN = ROOT / 'runs/diagnostic-fd003-engine75-trace-20261027'
CACHE = ROOT / 'results/diagnostics/fd003_predictor_fit_20261026/predictions'
PROTOCOL = ROOT / 'docs/operations/FD003_ENGINE75_TRACE.md'
SEEDS = [3, 13, 23, 33, 43]
REPS = ['level', 'dynamics', 'joint', 'coupling_raw', 'centered_shape']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def top5(distance, indices):
    order = np.argsort(distance, kind='stable')[:5]
    assert len(order) == 5
    return indices[order], distance[order]


def distinct5(distance, indices, ids):
    result, seen = [], set()
    for loc in np.argsort(distance, kind='stable'):
        index = int(indices[loc]); engine = int(ids[index])
        if engine not in seen:
            result.append(index); seen.add(engine)
            if len(result) == 5:
                break
    assert len(result) == 5
    return np.array(result)


def prediction_summary(rows, groups):
    frame = pd.DataFrame(rows)
    return frame.groupby(groups, dropna=False).agg(n_predictions=('prediction', 'size'),
        n_samples=('sample_key', 'nunique'), mean_label=('rul', 'mean'),
        mean_prediction=('prediction', 'mean'), prediction_sd=('prediction', 'std'),
        mse=('error', lambda x: float(np.mean(x*x))), bias=('error', 'mean'),
        mae=('error', lambda x: float(np.mean(abs(x))))).reset_index().assign(
            rmse=lambda g: np.sqrt(g.mse))


def main():
    OUT.mkdir(parents=True, exist_ok=True); RUN.mkdir(parents=True, exist_ok=True)
    mp = RUN / 'run-manifest.json'
    if mp.exists():
        raise RuntimeError('Preserve existing run')
    paths = [Path(__file__), PROTOCOL, ROOT / 'analysis/diagnostics/test_window_coverage.py',
        ROOT / 'upstream/data/CMAPSSDataset.py', ROOT / 'upstream/models/LSTM.py',
        ROOT / 'results/fd003_w48_public_code_split.npz',
        ROOT / 'results/diagnostics/fd003_real_neighbors_20261027/nearest_real_windows.csv',
        ROOT / 'runs/diagnostic-fd003-predictor-fit-20261026/run-manifest.json']
    paths += [ROOT / f'upstream/data/{kind}_FD003.txt' for kind in ['train', 'test', 'RUL']]
    paths += [ROOT / f'outputs/FD003_w48_seed{g}.npz' for g in SEEDS]
    paths += [CACHE / f'headline_gen{g}_eval{e}.npz' for g in SEEDS for e in SEEDS]
    paths += [CACHE / f'real_baseline_gen-1_eval{e}.npz' for e in SEEDS]
    paths += [ROOT / f'checkpoints/evaluators/real_data_baseline_w48/FD003_eval{e}.pth' for e in SEEDS]
    m = dict(run_id=RUN.name, scope_class='diagnostic', role='supporting',
        scope_contract='scope-20261027-fd003-engine75-trace', status='running', started_at=now(),
        command=[sys.executable, str(Path(__file__).relative_to(ROOT))], no_training=True,
        user_authorization='Explicit test75-realtrain-generated-bothpredictor trace request',
        source_snapshot='SOURCE_SNAPSHOT.md', inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
        generation_seeds=SEEDS, evaluator_seeds=SEEDS, candidate_calipers=[5, 0],
        environment=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
            torch=str(torch.__version__), platform=platform.platform(), device='cpu', threads=2),
        caveat='Case diagnosis;realtrain seen byrealpredictors,generatedtrain seen bycorresponding synthetic predictors')
    mp.write_text(json.dumps(m, indent=2))
    try:
        from data.CMAPSSDataset import CMAPSSDataset, feature_columns
        old = json.loads((ROOT / 'runs/diagnostic-fd003-predictor-fit-20261026/run-manifest.json').read_text())
        for p in paths:
            name = str(p.relative_to(ROOT))
            expected = old.get('outputs_sha256', {}).get(name) or old.get('inputs_sha256', {}).get(name)
            if expected:
                assert sha(p) == expected, name
        cwd = Path.cwd()
        try:
            os.chdir(ROOT / 'upstream')
            data = CMAPSSDataset('FD003', 48, [1000]); frame = data.get_train_data()
            real = data.get_feature_slice(frame).numpy(); y = data.get_label_slice(frame).numpy().ravel()
            tx, ty = data.get_last_data_slice(data.get_test_data()); tx = tx.numpy(); ty = ty.numpy().ravel()
            test_ids = np.array(sorted(i for i in set(data.get_test_data().id) if i <= data.engine_size))
            ids, starts, ends = [], [], []
            for engine, unit in frame.groupby('id', sort=True):
                cycles = unit.cycle.to_numpy(); n = len(unit) - 47
                ids.extend([engine] * n); starts.extend(cycles[:-47]); ends.extend(cycles[47:])
        finally:
            os.chdir(cwd)
        ids = np.array(ids); starts = np.array(starts); ends = np.array(ends)
        j75 = int(np.flatnonzero(test_ids == 75)[0]); assert ty[j75] == 45
        with np.load(ROOT / 'results/fd003_w48_public_code_split.npz') as z:
            ti = z['predictive_train']; vi = z['predictive_dev']
        rf = describe(real); tf = describe(tx[j75:j75+1]); rz = {}; tz = {}; centers = {}; scales = {}
        for rep in ['level', 'dynamics', 'coupling_raw']:
            mu = rf[rep].mean(0); scale = np.maximum(rf[rep].std(0), 1e-6)
            if rep == 'coupling_raw':
                mu = np.zeros(91); scale = np.ones(91)
            centers[rep] = mu; scales[rep] = scale
            rz[rep] = (rf[rep] - mu) / scale / np.sqrt(len(mu))
            tz[rep] = (tf[rep] - mu) / scale / np.sqrt(len(mu))
        residual = real - real.mean(1, keepdims=True)
        shape_scale = np.maximum(np.sqrt(np.mean(residual.astype(float)**2, axis=(0, 1))), 1e-6)
        rz['centered_shape'] = (residual / shape_scale).reshape(len(real), -1) / np.sqrt(48*14)
        tz['centered_shape'] = ((tx[j75:j75+1] - tx[j75:j75+1].mean(1, keepdims=True)) / shape_scale).reshape(1, -1) / np.sqrt(48*14)
        rz['joint'] = np.concatenate([rz['level'], rz['dynamics']], axis=1) / np.sqrt(2)
        tz['joint'] = np.concatenate([tz['level'], tz['dynamics']], axis=1) / np.sqrt(2)
        previous = pd.read_csv(ROOT / 'results/diagnostics/fd003_real_neighbors_20261027/nearest_real_windows.csv')
        previous = previous[(previous.engine_id == 75) & (previous.selection_rep == 'joint') & (previous.neighbor_kind == 'engine5')].sort_values('rank')
        primary = previous.train_window.to_numpy(int)
        assert np.array_equal(primary, [368, 12592, 7213, 8520, 9855])
        dev_pool = vi[abs(y[vi] - 45) <= 5]
        dev = distinct5(cdist(tz['joint'], rz['joint'][dev_pool]).ravel(), dev_pool, ids)
        queries = [dict(query_id='test75', role='real_test', window=-1, engine_id=75, rul=45)]
        queries += [dict(query_id=f'train{int(ids[i])}_w{i}', role='real_train_neighbor', window=int(i), engine_id=int(ids[i]), rul=int(y[i])) for i in primary]
        dev_queries = [dict(query_id=f'dev{int(ids[i])}_w{i}', role='real_dev_neighbor', window=int(i), engine_id=int(ids[i]), rul=int(y[i])) for i in dev]
        def feature(q, rep):
            return tz[rep][0] if q['role'] == 'real_test' else rz[rep][q['window']]
        stage = (y >= 31) & (y <= 60); stage_indices = np.flatnonzero(stage)
        position = np.full(len(y), -1, int); position[stage_indices] = np.arange(len(stage_indices))
        selected, support, offsets = [], [], []; signals = {}; real_neighbors = {}
        for g in SEEDS:
            with np.load(ROOT / f'outputs/FD003_w48_seed{g}.npz') as z:
                assert np.array_equal(y, z['label'].ravel())
                generated = z['data'][stage_indices]
            gf = describe(generated); gz = {}
            for rep in ['level', 'dynamics', 'coupling_raw']:
                gz[rep] = (gf[rep] - centers[rep]) / scales[rep] / np.sqrt(len(centers[rep]))
            gz['joint'] = np.concatenate([gz['level'], gz['dynamics']], axis=1) / np.sqrt(2)
            gz['centered_shape'] = ((generated - generated.mean(1, keepdims=True)) / shape_scale).reshape(len(generated), -1) / np.sqrt(48*14)
            for q in queries:
                for caliper in [5, 0]:
                    pool = ti[stage[ti] & (abs(y[ti] - q['rul']) <= caliper)]
                    if q['role'] != 'real_test':
                        pool = pool[ids[pool] != q['engine_id']]
                    assert len(pool) >= 5
                    for rep in REPS:
                        f = feature(q, rep)
                        rd = cdist(f[None], rz[rep][pool]).ravel()
                        gd = cdist(f[None], gz[rep][position[pool]]).ravel()
                        ri, rdist = top5(rd, pool); gi, gdist = top5(gd, pool)
                        real_neighbors[q['query_id'], rep, caliper] = ri
                        support.append(dict(query_id=q['query_id'], role=q['role'], rul=q['rul'], generation_seed=g,
                            caliper=caliper, representation=rep, n_real_candidates=len(pool), n_generated_candidates=len(pool),
                            d_real5=float(rdist.mean()), d_generated5=float(gdist.mean()),
                            ratio=float(gdist.mean()/rdist.mean()), d_real1=float(rdist[0]), d_generated1=float(gdist[0]),
                            real_5th_radius=float(rdist[-1]), n_generated_inside_real5_radius=int((gd<=rdist[-1]+1e-12).sum())))
                        if caliper == 5 and rep in ['joint', 'centered_shape']:
                            for rank, (index, distance) in enumerate(zip(gi, gdist), 1):
                                sample = dict(query_id=q['query_id'], query_rul=q['rul'], generation_seed=g,
                                    pool='train', selection_rep=rep, rank=rank, window=int(index), rul=int(y[index]), distance=float(distance))
                                sample.update({f'd_{r}': float(np.linalg.norm(feature(q, r)-gz[r][position[index]])) for r in REPS})
                                selected.append(sample)
                            if rep == 'centered_shape':
                                for c, sensor in enumerate(feature_columns):
                                    qm = tf['level'][0,c] if q['role']=='real_test' else rf['level'][q['window'],c]
                                    qt = tf['level'][0,c+14] if q['role']=='real_test' else rf['level'][q['window'],c+14]
                                    gm = gf['level'][position[gi],c].mean(); gt = gf['level'][position[gi],c+14].mean()
                                    rm = rf['level'][ri,c].mean(); rt = rf['level'][ri,c+14].mean()
                                    offsets.append(dict(query_id=q['query_id'], generation_seed=g, sensor=sensor,
                                        generated_mean_residual_sd=float((gm-qm)/scales['level'][c]),
                                        real_mean_residual_sd=float((rm-qm)/scales['level'][c]),
                                        generated_minus_real_mean_sd=float((gm-rm)/scales['level'][c]),
                                        generated_terminal_residual_sd=float((gt-qt)/scales['level'][c+14]),
                                        real_terminal_residual_sd=float((rt-qt)/scales['level'][c+14])))
                pool = vi[stage[vi] & (abs(y[vi]-q['rul'])<=5)]
                if q['role'] != 'real_test':
                    pool = pool[ids[pool] != q['engine_id']]
                gd = cdist(feature(q,'joint')[None], gz['joint'][position[pool]]).ravel()
                gi, dd = top5(gd, pool)
                for rank, (index, distance) in enumerate(zip(gi,dd),1):
                    sample = dict(query_id=q['query_id'], query_rul=q['rul'], generation_seed=g,
                        pool='dev', selection_rep='joint', rank=rank, window=int(index), rul=int(y[index]),distance=float(distance))
                    sample.update({f'd_{r}':float(np.linalg.norm(feature(q,r)-gz[r][position[index]])) for r in REPS})
                    selected.append(sample)
            chosen = sorted({r['window'] for r in selected if r['generation_seed']==g})
            signals[g] = {i:generated[position[i]].copy() for i in chosen}
            print('SELECTED',g,len(chosen),flush=True)
        selected_df=pd.DataFrame(selected); selected_df.to_csv(OUT/'frozen_generated_selections.csv',index=False)
        pd.DataFrame(queries+dev_queries).to_csv(OUT/'frozen_real_queries.csv',index=False)
        pd.DataFrame(support).to_csv(OUT/'support_by_query_seed.csv',index=False)
        pd.DataFrame(offsets).to_csv(OUT/'shape_matched_sensor_offsets.csv',index=False)
        plan_paths=[OUT/'frozen_generated_selections.csv',OUT/'frozen_real_queries.csv']
        m.update(selections_frozen_at=now(), selection_sha256={p.name:sha(p) for p in plan_paths},
            selection_completed_before_prediction_access=True,n_unique_generated_windows=sum(len(v) for v in signals.values()))
        mp.write_text(json.dumps(m,indent=2))
        # Only after selection is frozen, access cached per-sample predictions.
        raw=[]; caches={}; real_cache={}
        for e in SEEDS:
            with np.load(CACHE/f'real_baseline_gen-1_eval{e}.npz') as z:
                real_cache[e]={k:z[k] for k in z.files}
        for g in SEEDS:
            for e in SEEDS:
                with np.load(CACHE/f'headline_gen{g}_eval{e}.npz') as z:
                    caches[g,e]={k:z[k] for k in z.files}
        for q in queries+dev_queries:
            for family, pairs in [('real', [(-1,e) for e in SEEDS]),('synthetic',[(g,e) for g in SEEDS for e in SEEDS])]:
                for g,e in pairs:
                    cache=real_cache[e] if family=='real' else caches[g,e]
                    p=float(cache['real_test_prediction'][j75] if q['role']=='real_test' else cache['real_train_source_prediction'][q['window']])
                    raw.append(dict(sample_key=q['query_id'],batch=q['role'],model_family=family,generation_seed=g,evaluator_seed=e,
                        rul=q['rul'],prediction=p,error=p-q['rul']))
        pd.DataFrame(raw).to_csv(OUT/'real_query_predictions.csv',index=False)
        perquery=prediction_summary(raw,['sample_key','batch','model_family'])
        perquery.to_csv(OUT/'real_query_summary.csv',index=False)
        prediction_summary(raw,['batch','model_family']).to_csv(OUT/'real_batch_summary.csv',index=False)
        prediction_summary(raw,['batch','model_family','generation_seed']).to_csv(OUT/'real_batch_by_generation_seed.csv',index=False)
        torch.set_num_threads(2); real_generated={}; replay=[]
        for e in SEEDS:
            model=torch.load(ROOT/f'checkpoints/evaluators/real_data_baseline_w48/FD003_eval{e}.pth',map_location='cpu',weights_only=False).eval()
            with torch.inference_mode():
                check=float(model(torch.from_numpy(tx[j75:j75+1])).item())
                delta=abs(check-float(real_cache[e]['real_test_prediction'][j75]));assert delta<.001
                replay.append(dict(evaluator_seed=e,max_replay_difference=delta))
                for g, bank in signals.items():
                    order=sorted(bank); x=np.stack([bank[i] for i in order]); p=model(torch.from_numpy(x)).numpy().ravel()
                    for index, prediction in zip(order,p):real_generated[g,index,e]=float(prediction)
        generated_predictions=[]
        for row_id,r in enumerate(selected):
            g=r['generation_seed'];index=r['window'];label=r['rul']
            for family in ['real','synthetic']:
                for e in SEEDS:
                    p=real_generated[g,index,e] if family=='real' else float(caches[g,e]['synthetic_prediction'][index])
                    generated_predictions.append(dict(selection_id=row_id,query_id=r['query_id'],pool=r['pool'],selection_rep=r['selection_rep'],rank=r['rank'],
                        sample_key=f'gen{g}_w{index}',generation_seed=g,evaluator_seed=e,model_family=family,
                        rul=label,prediction=p,error=p-label))
        gp=pd.DataFrame(generated_predictions);gp.to_csv(OUT/'generated_query_predictions.csv',index=False)
        prediction_summary(generated_predictions,['query_id','pool','selection_rep','model_family']).to_csv(OUT/'generated_batch_summary.csv',index=False)
        prediction_summary(generated_predictions,['selection_id','query_id','pool','selection_rep','rank','generation_seed','model_family']).to_csv(OUT/'generated_window_summary.csv',index=False)
        pd.DataFrame(replay).to_csv(OUT/'real_predictor_replay.csv',index=False)
        # Curves for preselected generationseed3, not whichever seed looks best.
        q=queries[0];bank=signals[3]
        fig,axes=plt.subplots(4,4,figsize=(12,9))
        first_joint=selected_df[(selected_df.query_id=='test75')&(selected_df.generation_seed==3)&(selected_df.pool=='train')&(selected_df.selection_rep=='joint')&(selected_df['rank']==1)].iloc[0]
        first_shape=selected_df[(selected_df.query_id=='test75')&(selected_df.generation_seed==3)&(selected_df.pool=='train')&(selected_df.selection_rep=='centered_shape')&(selected_df['rank']==1)].iloc[0]
        for c,ax in enumerate(axes.flat):
            if c>=14:ax.axis('off');continue
            ax.plot(tx[j75,:,c],color='black',lw=1.6,label='Real test75, RUL45')
            ax.plot(real[primary[0],:,c],lw=1,alpha=.85,label='Real train2, RUL49')
            ax.plot(bank[int(first_joint.window)][:,c],lw=1,alpha=.85,label=f'Gen3 joint NN, RUL{int(first_joint.rul)}')
            ax.plot(bank[int(first_shape.window)][:,c],lw=1,alpha=.85,label=f'Gen3 centered-shape NN, RUL{int(first_shape.rul)}')
            ax.set_title(feature_columns[c],fontsize=9);ax.grid(alpha=.15)
        h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,fontsize=8)
        fig.suptitle('FD003 test75 -> real neighbor -> same-life generated neighbors\nGenerationseed3 fixed before outputs; original train-fit normalized signals')
        fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(OUT/'test75_chain_signals_seed3.png',dpi=170);plt.close(fig)
        # Local means against centered-shape-matched real reference, averaged over5seeds.
        off=pd.DataFrame(offsets)
        off.groupby(['query_id','sensor']).agg(mean_offset_sd=('generated_minus_real_mean_sd','mean'),
            min_seed_offset_sd=('generated_minus_real_mean_sd','min'),max_seed_offset_sd=('generated_minus_real_mean_sd','max'),
            n_positive_seeds=('generated_minus_real_mean_sd',lambda v:int((v>0).sum()))).reset_index().to_csv(OUT/'sensor_offset_summary.csv',index=False)
        for p in plan_paths:assert sha(p)==m['selection_sha256'][p.name]
        assert len(selected_df)==450 and len(raw)==330 and len(gp)==4500
        m.update(status='complete',n_real_queries=len(queries)+len(dev_queries),n_selected_generated_records=len(selected_df),
            n_real_prediction_records=len(raw),n_generated_prediction_records=len(gp),max_replay_difference=max(r['max_replay_difference'] for r in replay),
            outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file()})
        print('REAL QUERY SUMMARY\n',perquery.round(3).to_string(index=False),flush=True)
        print('TEST75 SUPPORT\n',pd.DataFrame(support).query("query_id=='test75' and caliper==5").groupby('representation').agg(
            real_distance=('d_real5','first'),generated_distance=('d_generated5','mean'),ratio=('ratio','mean'),
            inside_min=('n_generated_inside_real5_radius','min'),inside_max=('n_generated_inside_real5_radius','max')).round(3).to_string(),flush=True)
    except BaseException:
        m.update(status='failed',error=traceback.format_exc());raise
    finally:
        m['finished_at']=now();mp.write_text(json.dumps(m,indent=2))


if __name__=='__main__':main()
