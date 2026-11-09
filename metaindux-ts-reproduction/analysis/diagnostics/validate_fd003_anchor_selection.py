"""Freeze one train-engine anchor rule, validate on disjoint train engines."""
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
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(ROOT / 'analysis/diagnostics'))
from test_window_coverage import describe

OUT = ROOT / 'results/diagnostics/fd003_anchor_selection_20261026'
RUN = ROOT / 'runs/diagnostic-fd003-anchor-selection-20261026'
PROTOCOL = ROOT / 'docs/operations/FD003_ANCHOR_SELECTION_DIAGNOSTIC.md'
REPORT = ROOT / 'docs/conclusions/final/fd003_anchor_selection_20261026_report.md'
SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12',
           's13', 's14', 's15', 's17', 's20', 's21']
REPS = ['level', 'dynamics', 'coupling_raw']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def real_training_windows():
    """Read train only. Match upstream per-column train-fit MinMax and ordering."""
    columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    frame = pd.read_csv(ROOT / 'upstream/data/train_FD003.txt', sep=r'\s+', header=None, names=columns)
    normalized = MinMaxScaler().fit_transform(frame[SENSORS])
    x, y, ids = [], [], []
    for engine in sorted(frame.id.unique()):
        mask = (frame.id == engine).to_numpy()
        unit = normalized[mask]
        cycles = frame.loc[mask, 'cycle'].to_numpy()
        x.append(np.lib.stride_tricks.sliding_window_view(unit, 48, axis=0).transpose(0, 2, 1))
        y.append(np.minimum(cycles.max() - cycles[47:], 125))
        ids.append(np.full(len(unit) - 47, engine))
    return np.concatenate(x).astype(np.float32), np.concatenate(y), np.concatenate(ids)


def distances(a, b):
    return cdist(a, b, 'euclidean') / np.sqrt(a.shape[1])


def initial_neighbors(query, qy, reference, ry, k=5):
    result = np.empty((len(query), k))
    for label in np.unique(qy):
        loc = np.flatnonzero(qy == label)
        pool = np.flatnonzero(abs(ry - label) <= 5)
        assert len(pool) >= k
        matrix = distances(query[loc], reference[pool])
        result[loc] = np.sort(np.partition(matrix, k - 1, axis=1)[:, :k], axis=1)
    return result


def append_neighbors(current, distance):
    return np.sort(np.concatenate([current, distance[:, None]], axis=1), axis=1)[:, :5]


def guided_donors(real_level, synthetic_level, y, ids, ti, targets, candidates, selection_engines):
    stage = (y >= 31) & (y <= 60)
    qi = np.flatnonzero(stage & np.isin(ids, selection_engines))
    remaining = ti[~np.isin(ti, targets) & stage[ti]]
    current = initial_neighbors(real_level[qi], y[qi], synthetic_level[remaining], y[remaining])
    matrix = distances(real_level[qi], real_level[candidates])
    matrix[(abs(y[qi, None] - y[candidates][None, :]) > 5)
           | (ids[qi, None] == ids[candidates][None, :])] = np.inf
    quotas = np.bincount(y[targets].astype(int), minlength=126)
    available = np.ones(len(candidates), dtype=bool)
    chosen, log = [], []
    for step in range(len(targets)):
        gain = np.maximum(current[:, -1, None] - matrix, 0).mean(axis=0) / 5
        gain[~available | (quotas[y[candidates].astype(int)] == 0)] = -np.inf
        best = int(np.argmax(gain))
        assert np.isfinite(gain[best])
        donor = int(candidates[best])
        before = float(current.mean())
        current = append_neighbors(current, matrix[:, best])
        chosen.append(donor)
        available[best] = False
        quotas[int(y[donor])] -= 1
        log.append(dict(step=step + 1, donor=donor, engine_id=int(ids[donor]),
                        rul=int(y[donor]), objective_before=before,
                        objective_after=float(current.mean()), gain=float(gain[best])))
    assert quotas.sum() == 0
    # Return in target-position order, preserving label exactly at every position.
    chosen = np.array(chosen)
    assigned = np.empty(len(targets), dtype=int)
    for label in np.unique(y[targets]):
        assigned[y[targets] == label] = chosen[y[chosen] == label]
    return assigned, pd.DataFrame(log), qi


def evaluate(rep, rf, gf, y, ids, ti, vi, targets, plans, validation_engines, candidates):
    qi = np.flatnonzero((y >= 31) & (y <= 60) & np.isin(ids, validation_engines))
    pool = ti[(y[ti] >= 31) & (y[ti] <= 60)]
    query = rf[qi]
    rows = []
    features = {'control': gf}
    for arm, donors in plans.items():
        mixed = gf.copy()
        mixed[targets] = rf[donors]
        assert np.array_equal(mixed[vi], gf[vi])
        features[arm] = mixed
    for arm, feature in features.items():
        near = initial_neighbors(query, y[qi], feature[pool], y[pool])
        for index, d in zip(qi, near.mean(axis=1)):
            rows.append(dict(window_index=int(index), engine_id=int(ids[index]),
                             rul=int(y[index]), representation=rep, arm=arm, distance=float(d)))
    near = initial_neighbors(query, y[qi], rf[candidates], y[candidates])
    for index, d in zip(qi, near.mean(axis=1)):
        rows.append(dict(window_index=int(index), engine_id=int(ids[index]),
                         rul=int(y[index]), representation=rep, arm='real_reference', distance=float(d)))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    mp = RUN / 'run-manifest.json'
    if mp.exists():
        raise RuntimeError('Run already exists; preserve frozen run instead of overwriting')
    paths = [Path(__file__), PROTOCOL, ROOT / 'analysis/diagnostics/test_window_coverage.py',
             ROOT / 'upstream/data/CMAPSSDataset.py', ROOT / 'upstream/data/train_FD003.txt',
             ROOT / 'outputs/FD003_w48_seed3.npz',
             ROOT / 'results/diagnostics/fd003_local_anchor_20261026/replacement_indices.npz']
    m = dict(run_id=RUN.name, scope_class='diagnostic', role='supporting followup of EXT-001',
             scope_contract='scope-20261026-fd003-anchor-selection', status='running',
             started_at=now(), command=[sys.executable, str(Path(__file__).relative_to(ROOT))],
             user_authorization='User requested continuing train/validation-only anchor selection diagnosis',
             no_training=True, no_official_test_access=True, random_seed=20261026,
             generation_seed=3, k=5, rul_caliper=5, stage=[31, 60],
             source_snapshot='SOURCE_SNAPSHOT.md',
             protocol_sha256=sha(PROTOCOL), inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
             gate='>=5% lower mean level distance vs random AND >half validation engines improve',
             caveat='Validation engines were seen by original generator and original MinMax scaler; holdout is for donor selection only.')
    mp.write_text(json.dumps(m, indent=2))
    try:
        real, y, ids = real_training_windows()
        with np.load(ROOT / 'outputs/FD003_w48_seed3.npz') as z:
            generated = z['data']
            assert np.array_equal(y, z['label'].ravel()) and generated.shape == real.shape
        with np.load(paths[-1]) as z:
            ti = z['train_indices']; vi = z['dev_indices']; targets = z['target']
        order = np.random.RandomState(20261026).permutation(np.unique(ids))
        validation_engines = np.sort(order[:30]); selection_engines = np.sort(order[30:])
        assert len(selection_engines) == 70 and len(validation_engines) == 30
        assert not np.intersect1d(selection_engines, validation_engines).size
        assert len(targets) == 209 and np.isin(targets, ti).all()
        candidates = np.sort(ti[(y[ti] >= 31) & (y[ti] <= 60) & np.isin(ids[ti], selection_engines)])
        # Validation features are not described until both donor plans are frozen.
        selection_mask = np.isin(ids, selection_engines)
        selection_indices = np.flatnonzero(selection_mask)
        selection_desc = describe(real[selection_mask])
        scales = {}; centers = {}; fitted_real = {}
        for rep in REPS:
            mu = selection_desc[rep].mean(axis=0)
            scale = np.maximum(selection_desc[rep].std(axis=0), 1e-6)
            if rep == 'coupling_raw':
                mu = np.zeros(91); scale = np.ones(91)
            centers[rep] = mu; scales[rep] = scale
            fitted_real[rep] = np.full((len(y), len(mu)), np.nan)
            fitted_real[rep][selection_indices] = (selection_desc[rep] - mu) / scale
        synthetic_desc = describe(generated)
        fitted_gen = {rep: (synthetic_desc[rep] - centers[rep]) / scales[rep] for rep in REPS}
        rng = np.random.RandomState(20261026)
        random = np.empty(len(targets), dtype=int)
        for label in np.unique(y[targets]):
            loc = np.flatnonzero(y[targets] == label)
            random[loc] = rng.choice(candidates[y[candidates] == label], len(loc), replace=False)
        guided, progression, qi = guided_donors(fitted_real['level'], fitted_gen['level'],
                                               y, ids, ti, targets, candidates, selection_engines)
        plans = dict(random=random, guided=guided)
        for arm, donors in plans.items():
            assert len(donors) == len(np.unique(donors)) == 209
            assert np.array_equal(y[donors], y[targets])
            assert np.isin(donors, candidates).all()
            assert not np.isin(ids[donors], validation_engines).any()
        plan_path = OUT / 'frozen_plans.npz'
        np.savez_compressed(plan_path, targets=targets, random_donors=random, guided_donors=guided,
                            selection_engines=selection_engines, validation_engines=validation_engines,
                            candidate_donors=candidates, selection_query_indices=qi,
                            train_indices=ti, dev_indices=vi,
                            **{f'scale_{rep}': scales[rep] for rep in REPS},
                            **{f'center_{rep}': centers[rep] for rep in REPS})
        progression.to_csv(OUT / 'selection_progression.csv', index=False)
        m.update(plans_frozen_at=now(), frozen_plans_sha256=sha(plan_path),
                 n_replacements=len(targets), n_candidate_donors=len(candidates),
                 selection_engines=selection_engines.tolist(), validation_engines=validation_engines.tolist(),
                 validation_descriptors_not_yet_read=True)
        mp.write_text(json.dumps(m, indent=2))
        print('Plans frozen', m['frozen_plans_sha256'], flush=True)
        held = ~selection_mask
        val_desc = describe(real[held])
        for rep in REPS:
            fitted_real[rep][held] = (val_desc[rep] - centers[rep]) / scales[rep]
        rows = []
        for rep in REPS:
            rows.extend(evaluate(rep, fitted_real[rep], fitted_gen[rep], y, ids,
                                 ti, vi, targets, plans, validation_engines, candidates))
        windows = pd.DataFrame(rows)
        windows.to_csv(OUT / 'validation_windows.csv', index=False)
        engines = windows.groupby(['representation', 'engine_id', 'arm']).distance.mean().unstack('arm').reset_index()
        engines['random_change_pct'] = (engines.control - engines.random) / engines.control * 100
        engines['guided_change_pct'] = (engines.control - engines.guided) / engines.control * 100
        engines['guided_vs_random_pct'] = (engines.random - engines.guided) / engines.random * 100
        engines.to_csv(OUT / 'validation_engines.csv', index=False)
        summaries = []
        for rep in REPS:
            g = engines[engines.representation == rep]
            diff = (g.guided - g.random).to_numpy()
            rng = np.random.RandomState(20261026)
            boot = diff[rng.randint(len(g), size=(10000, len(g)))].mean(axis=1)
            low, high = np.quantile(boot, [.025, .975])
            reduction = float((g.random.mean() - g.guided.mean()) / g.random.mean() * 100)
            nbetter = int((g.guided < g.random - 1e-10).sum())
            summaries.append(dict(representation=rep, n_engines=len(g),
                control_distance=float(g.control.mean()), random_distance=float(g.random.mean()),
                guided_distance=float(g.guided.mean()), real_reference_distance=float(g.real_reference.mean()),
                random_vs_control_reduction_pct=float(g.random_change_pct.mean()),
                guided_vs_control_reduction_pct=float(g.guided_change_pct.mean()),
                guided_vs_random_reduction_pct=reduction, guided_better_engines=nbetter,
                guided_worse_engines=int((g.guided > g.random + 1e-10).sum()),
                guided_vs_random_median_reduction_pct=float(g.guided_vs_random_pct.median()),
                delta_guided_minus_random=float(diff.mean()), delta_ci95_low=float(low), delta_ci95_high=float(high),
                primary_gate_pass=bool(rep == 'level' and reduction >= 5 and nbetter > len(g) / 2)))
        summary = pd.DataFrame(summaries)
        summary.to_csv(OUT / 'summary.csv', index=False)
        audit = []
        for arm, donors in plans.items():
            counts = pd.Series(ids[donors]).value_counts()
            audit.append(dict(arm=arm, n_donors=len(donors), n_donor_engines=len(counts),
                              max_donors_one_engine=int(counts.max()),
                              donor_engines_and_counts=json.dumps(counts.sort_index().to_dict())))
        pd.DataFrame(audit).to_csv(OUT / 'donor_diversity.csv', index=False)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, rep in zip(axes, REPS):
            g = engines[engines.representation == rep]
            ax.scatter(g.random, g.guided, s=30)
            lo = min(g.random.min(), g.guided.min()) * .95
            hi = max(g.random.max(), g.guided.max()) * 1.05
            ax.plot([lo, hi], [lo, hi], '--', color='gray')
            ax.set(xlabel='Random anchors: mean5-NN distance', ylabel='Guided anchors: mean5-NN distance', title=rep)
            ax.grid(alpha=.2)
        fig.suptitle('Engine-disjoint anchor-rule validation | one point per real training engine\nBelow diagonal = guided closer;209 replacements, identical RUL quotas')
        fig.tight_layout(rect=(0, 0, 1, .88))
        for ext in ['png', 'pdf']:
            fig.savefig(OUT / f'validation_support.{ext}', dpi=180)
        plt.close(fig)
        gate = bool(summary.loc[summary.representation == 'level', 'primary_gate_pass'].iloc[0])
        table = '\n'.join(f'| {r.representation} | {r.control_distance:.4f} | {r.random_distance:.4f} | {r.guided_distance:.4f} | {r.guided_vs_random_reduction_pct:+.2f}% | {r.guided_better_engines}/30 |'
                          for r in summary.itertuples())
        level = summaries[0]
        text = f'''# FD003锚点选择规则：按发动机留出的覆盖验证

Run ID：`{RUN.name}`；Scope：diagnostic/supporting，EXT-001后续；无新增模型训练。

## 结论

**[Observed]** 固定70台选择、30台验证的训练发动机划分，209个替换位置和逐RUL数量均与上一轮一致。两组donor都只来自70台选择发动机的原predictor-train池。主检查门槛{'通过' if gate else '未通过'}：规则相对随机的水平距离降幅{level['guided_vs_random_reduction_pct']:.2f}%，{level['guided_better_engines']}/30台改善；事先要求至少5%且超过半数发动机改善。

## 距离结果

| 表示 | 不替换 | 随机锚点 | 引导锚点 | 引导相对随机缩短 | 改善发动机 |
|---|---:|---:|---:|---:|---:|
{table}

**[Observed]** 每台验证发动机取RUL31–60的全部窗口，先平均窗口距离，再平均30台发动机。每个窗口的候选池仍只取原predictor-train中RUL±5且31–60的窗口；距离为最近5邻居的平均。验证窗口不是30个独立窗口，而是30台发动机的相邻重叠窗口；区间按发动机配对bootstrap。level引导减随机的绝对差95%描述区间为[{level['delta_ci95_low']:.5f}, {level['delta_ci95_high']:.5f}]。

## 已冻结的方法

**[Code]** 详见执行前写入的`docs/operations/FD003_ANCHOR_SELECTION_DIAGNOSTIC.md`。只用传感器水平28维选择，不根据动态或跨通道结果修改规则。每步贪心选择最能降低70台选择发动机查询窗口平均5NN距离的donor，满足每个RUL剩余额度；同一发动机查询不计入该donor收益，避免直接奖励自己或相邻重叠窗口。所有209个donor选择完成并写入哈希后，才提取30台验证发动机的描述特征。

**[Observed]** 保存的`frozen_plans.npz`包含两组donor、target、train/dev索引、70/30发动机列表和固定描述缩放参数。已核对donor唯一、exact-RUL、donor来自选择发动机、target仅train、dev完全不变。主原始输入未改变，不读取官方test或RUL文件，不读取先前测试误差/难例列表。

## 解释与边界

**[Observed]** 这里只测量覆盖距离，不存在本轮下游RMSE结果；无法把距离改善直接换算成预测改善。原生成器和原MinMax预处理已见过全部100台真实训练发动机，因此这里的独立性仅针对锚点选择规则；不是生成器对未见发动机的泛化评估。

**[Hypothesis]** {'该固定规则在留出发动机上改善了局部支持，说明随机选锚点可能浪费有限预算；仍需同配置随机vs引导预测器干预检验下游收益。' if gate else '本次固定规则尚未达到预设的覆盖改善标准，不应直接据此启动预测器训练或扫描种子/比例。'} 优化level不保证动态/通道关系同步改善，也不能唯一判定缺覆盖或信号偏移。

## 检查点与停止条件

**[Observed]** 本轮一次划分、一个规则完成后停止；未训练预测器/生成器，零标签路线仍暂停。历史strict main/A/B状态不变，Candidate C仍延后。scope为`scope-20261026-fd003-anchor-selection`，不添加论文严格指标。

当前覆盖结果→论文结果不存在→论文差距不适用→只有一次规则/划分，未测跨划分变异→不能改变论文复现结论。下游RMSE变化未测量。

下一步：{'保留冻结规则；若继续，仅比较同样209个位置及逐RUL数量下随机vs引导的同预算预测器，不把这次距离结果当作RMSE收益。' if gate else '停止本规则，不自动追加模型训练、比例或随机种子扫描；先复核规则限制与选择池是否能区分原因。'}

## 产物

- `results/diagnostics/fd003_anchor_selection_20261026/summary.csv`与`validation_engines.csv`：30台发动机的主、辅助表示覆盖结果。
- 同目录`validation_windows.csv`：完整窗口距离；`frozen_plans.npz`：冻结donor计划。
- 同目录`selection_progression.csv`、`donor_diversity.csv`：选择过程与donor发动机集中程度。
- 同目录`validation_support.png/pdf`：随机与引导锚点在验证发动机上的距离比较。
- `runs/{RUN.name}/run-manifest.json`：协议、代码、输入、计划及输出哈希。
'''
        REPORT.write_text(text)
        assert sha(plan_path) == m['frozen_plans_sha256']
        assert np.isfinite(windows.distance).all() and len(engines) == 90
        m.update(status='complete', summary=summaries, primary_gate_pass=gate,
                 n_validation_windows=int(windows.window_index.nunique()),
                 validation_engine_count=30, report=str(REPORT.relative_to(ROOT)), report_sha256=sha(REPORT),
                 validation='Exact label/count/train/dev/engine-disjointness and finite distances checked; frozen plan unchanged',
                 outputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in OUT.iterdir() if p.is_file()})
        print(summary.round(5).to_string(index=False), flush=True)
    except BaseException:
        m.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        m['finished_at'] = now()
        mp.write_text(json.dumps(m, indent=2))


if __name__ == '__main__':
    main()
