"""Predeclared FD003 overall and31-60 reporting; CPU completion audit."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'configs/extensions/prototype_fd003_w48.json').read_text())
RUN=ROOT/'runs'/C['run_id']
OUT=ROOT/'results/extensions'/C['run_id']
BINS=[('all',0,125),('0-30',0,30),('31-60',31,60),('61-90',61,90),('91-125',91,125),('91-124',91,124),('125',125,125)]

def dump(p,value):
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def summarize_segments():
    tables=[]
    for s in C['generator_seeds']:
        for e in C['checkpoints']:
            for g in ['A','B']:
                for es in C['predictor_seeds']:
                    p=OUT/f'seed{s}'/g/f'epoch{e}_eval{es}/segments.csv'
                    if p.exists():
                        t=pd.read_csv(p);t['generator_seed']=s;t['epoch']=e;t['group']=g;t['predictor_seed']=es;tables.append(t)
    if not tables:return
    raw=pd.concat(tables,ignore_index=True)
    raw.to_csv(OUT/'segments_all_models.csv',index=False)
    pairs=[];summary=[]
    for s in C['generator_seeds']:
        for e in C['checkpoints']:
            for name,_,_ in BINS:
                t=raw[(raw.generator_seed==s)&(raw.epoch==e)&(raw.bin==name)]
                records=[]
                for es in C['predictor_seeds']:
                    a=t[(t.group=='A')&(t.predictor_seed==es)]
                    b=t[(t.group=='B')&(t.predictor_seed==es)]
                    if len(a)==len(b)==1:
                        av=float(a.rmse_engine.iloc[0]);bv=float(b.rmse_engine.iloc[0])
                        r=dict(generator_seed=s,epoch=e,bin=name,predictor_seed=es,A=av,B=bv,delta=bv-av)
                        pairs.append(r);records.append(r)
                if len(records)==5:
                    av=float(np.sqrt(np.mean([r['A']**2 for r in records])))
                    bv=float(np.sqrt(np.mean([r['B']**2 for r in records])))
                    summary.append(dict(generator_seed=s,epoch=e,bin=name,A=av,B=bv,delta=bv-av,improving=sum(r['delta']<0 for r in records),n_engines=int(t.n_engines.iloc[0]),n_windows=int(t.n_windows.iloc[0])))
    pd.DataFrame(pairs).to_csv(OUT/'paired_segments.csv',index=False)
    table=pd.DataFrame(summary)
    if len(table):
        table.to_csv(OUT/'segment_summary.csv',index=False)
        table[table.bin=='31-60'].to_csv(OUT/'target31_60_summary.csv',index=False)
    selected=OUT/'selected.json'
    if selected.exists() and len(summary):
        choices=json.loads(selected.read_text());comparisons=[]
        for s in C['generator_seeds']:
            available={r['group']:r for r in choices if r['seed']==s}
            if set(available)!=set(['A','B']):continue
            r=dict(generator_seed=s,A_epoch=available['A']['epoch'],B_epoch=available['B']['epoch'])
            for name,key in [('all','overall'),('31-60','target31_60')]:
                values={}
                for g in ['A','B']:
                    rows=table[(table.generator_seed==s)&(table.epoch==available[g]['epoch'])&(table.bin==name)]
                    if len(rows)!=1:break
                    values[g]=float(rows.iloc[0][g])
                if len(values)==2:r.update({f'{key}_A':values['A'],f'{key}_B':values['B'],f'{key}_delta':values['B']-values['A']})
            if 'overall_delta' in r and 'target31_60_delta' in r:
                r['both_improve']=r['overall_delta']<0 and r['target31_60_delta']<0
                r['stage_tradeoff']=r['target31_60_delta']<0 and r['overall_delta']>=0
                comparisons.append(r)
        dump(OUT/'selected_comparison.json',comparisons)

def audit():
    manifest=json.loads((RUN/'manifest.json').read_text())
    for p,h in manifest['hashes'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    train=np.load(RUN/'train.npz');val=np.load(RUN/'validation.npz');plan=np.load(RUN/'plan.npz');proto=np.load(RUN/'prototypes.npz');norm=np.load(RUN/'normalization.npz')
    assert not set(train['engine_id']) & set(val['engine_id'])
    assert np.array_equal(np.unique(train['engine_id']),proto['fit_engine_ids'])
    assert np.array_equal(np.unique(train['engine_id']),norm['fit_engine_ids'])
    assert np.array_equal(np.unique(val['engine_id']),C['validation_engines'])
    paired=[];recomputed=[]
    for s in C['generator_seeds']:
        audits=[json.loads((OUT/f'seed{s}'/g/'training.json').read_text()) for g in ['A','B']]
        for key in ['shared_initial_hash','order_hash','rng_hash','updates']:
            assert audits[0][key]==audits[1][key],(s,key)
        assert audits[0]['updates']==70*math.ceil(len(train['data'])/256)
        assert all(len(m['history'])==70 for m in audits)
        for e in C['checkpoints']:
            samples=[np.load(OUT/f'seed{s}'/g/f'samples{e}.npz') for g in ['A','B']]
            sa=[json.loads((OUT/f'seed{s}'/g/f'sampling{e}.json').read_text()) for g in ['A','B']]
            assert sa[0]['rng_hash']==sa[1]['rng_hash']
            for z in samples:
                assert np.array_equal(z['label'],plan['target_rul'])
                assert z['data'].shape==(len(train['data']),48,14) and np.isfinite(z['data']).all()
            for es in C['predictor_seeds']:
                ms=[]
                for g in ['A','B']:
                    d=OUT/f'seed{s}'/g/f'epoch{e}_eval{es}'
                    pred=np.load(d/'predictions.npz');m=json.loads((d/'metrics.json').read_text());ms.append(m)
                    assert np.array_equal(pred['label'],val['label'].reshape(-1))
                    assert np.array_equal(pred['engine_id'],val['engine_id'])
                    segments=pd.read_csv(d/'segments.csv')
                    for name,lo,hi in BINS:
                        y=pred['label'];ids=pred['engine_id'];mask=(y>=lo)&(y<=hi)
                        sq=(pred['prediction'].astype('float64')-y)**2
                        score=float(np.sqrt(np.mean([sq[mask&(ids==i)].mean() for i in np.unique(ids[mask])])))
                        reported=float(segments[segments.bin==name].rmse_engine.iloc[0])
                        assert abs(score-reported)<1e-5,(d,name,score,reported)
                        if name=='all':assert abs(score-m['rmse'])<1e-5
                        recomputed.append(dict(generator_seed=s,epoch=e,group=g,predictor_seed=es,bin=name,rmse=score))
                assert ms[0]['updates']==ms[1]['updates']==35*math.ceil(len(plan['train_idx'])/256)
                assert ms[0]['order_hash']==ms[1]['order_hash']
                paired.append(dict(generator_seed=s,epoch=e,predictor_seed=es,A=ms[0]['rmse'],B=ms[1]['rmse'],delta=ms[1]['rmse']-ms[0]['rmse']))
    pd.DataFrame(paired).to_csv(OUT/'paired_recomputed.csv',index=False)
    pd.DataFrame(recomputed).to_csv(OUT/'rmse_recomputed.csv',index=False)
    return dict(passed=True,source_and_input_hashes_verified=True,train_validation_disjoint=True,train_only_fits=True,shared_initialization_order_rng_equal=True,sampling_rng_labels_equal=True,predictor_order_equal=True,real_rmse_all_bins_recomputed=True,generators=4,sample_sets=8,predictors=40,train_windows=len(train['data']),validation_windows=len(val['data']),generator_updates=70*math.ceil(len(train['data'])/256),predictor_updates=35*math.ceil(len(plan['train_idx'])/256))

def finalize():
    summarize_segments()
    a=audit();a['finished_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    launch=RUN/'background_launch.json'
    if launch.exists():a['elapsed_seconds']=(datetime.fromisoformat(a['finished_at'])-datetime.fromisoformat(json.loads(launch.read_text())['launched_at'])).total_seconds()
    dump(RUN/'completion_audit.json',a)
    overall=pd.read_csv(OUT/'summary.csv');target=pd.read_csv(OUT/'target31_60_summary.csv')
    choices=json.loads((OUT/'selected_comparison.json').read_text())
    gate=len(choices)==2 and all(r['both_improve'] for r in choices)
    dump(OUT/'expansion_gate.json',dict(both_generator_seeds_improve_overall_and_target=gate,comparisons=choices,automatic_expansion=False))
    report=ROOT/'docs/conclusions/final/prototype_fd003_w48_20261103_report.md'
    lines=['# FD003窗口48：原设计原型条件对照结果','',f"[Observed] 完成4生成器/8生成数据单元/40预测器，CPU审计通过；完成时间{a['finished_at']}。",'',
        '这是基于训练窗口聚类的原型条件实验，不是K-ProtoDiff完整复现，聚类ID不等于故障类型。只换数据集与其既有发动机划分，特征/K4/注入/优化/预算/候选和选模规则沿用FD001；配置与协议见下方。','',
        '| 生成器种子 | 轮次 | A总体 | B总体 | B-A | 改善数 | A 31–60 | B 31–60 | 段B-A | 段改善数 |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for r in overall.itertuples():
        t=target[(target.generator_seed==r.generator_seed)&(target.epoch==r.epoch)].iloc[0]
        lines.append(f'| {r.generator_seed} | {r.epoch} | {r.A:.4f} | {r.B:.4f} | {r.delta:+.4f} | {r.improving}/5 | {t.A:.4f} | {t.B:.4f} | {t.delta:+.4f} | {t.improving}/5 |')
    lines+=['','[Observed] 检查点只按总体真实验证RMSE选择；31–60不参与选模。选中候选如下：','']
    for r in choices:
        lines.append(f"- seed{r['generator_seed']}: A第{r['A_epoch']}轮，B第{r['B_epoch']}轮；总体Δ={r['overall_delta']:+.4f}，31–60Δ={r['target31_60_delta']:+.4f}。")
    lines+=['',f"[Observed] 两个生成器种子的总体与目标段同时改善门槛：{gate}。这不是显著性检验；若为True，仅值得建议扩大验证，不自动启动；若仅目标段改善而总体变差，标记阶段取舍，保持当前预算结束。",'',
        '[Observed] 本轮复用既有FD003验证30台，70台用于生成器/归一化/聚类拟合。原测试集历史结果仅解释数据集与31–60关注点的来源，本轮不读取原测试文件、不调原型、不按测试挑轮次。验证集同时用于检查点选择和报告，属于探索性验证；两生成器/五预测器不能当作独立测试证据。','',
        '当前结果→本表探索性验证→论文无该扩展目标→论文差距不适用→跨生成器方向见门槛与全部配对→严格论文复现结论不变。','',
        '[Code] 主RMSE=sqrt(mean_predictors(mean_engines(mean_window_squared_error)))；分段对该段内有窗口的发动机等权，计数保存在segment_summary.csv。5预测器逐个配对差值在paired.csv/paired_segments.csv，全部分段/逐发动机结果在各单元目录。','',
        '配置：configs/extensions/prototype_fd003_w48.json；冻结协议：docs/operations/PROTOTYPE_FD003_W48.md；快照与审计：runs/'+C['run_id']+'/manifest.json、completion_audit.json；结果：results/extensions/'+C['run_id']+'/。','',
        '命令（工作区根目录）：`.venv/bin/python -u metaindux-ts-reproduction/scripts/run_prototype_guidance_fd003.py run`。已完成，勿并行重复启动；只重建汇总：同脚本`summarize`。']
    report.write_text('\n'.join(lines)+'\n')
    entry=f"\n## {a['finished_at']} FD003 prototype experiment complete\n\n- [Observed] extension:EXT-PROTO-002/scope-20261103-prototype-fd003-v1:4generators/8samples/40predictors complete and audited. Same prototype design as FD001. Overall/31-60 paired reports and selected comparisons at results/extensions/{C['run_id']}/. Both-seed both-metric gate={gate}; no automatic expansion.\n- Snapshot runs/{C['run_id']}/manifest.json; report docs/conclusions/final/prototype_fd003_w48_20261103_report.md. Test-inspired dataset/stage choice only; no official test access. Internal validation selection is exploratory; no strict conclusion change. Fixed budget closed.\n"
    for name in ['experiment-ledger.md','reproduction-state.md']:
        with (ROOT/name).open('a') as f:f.write(entry)
    dump(RUN/'execution_status.json',dict(status='complete',training_completed=4,predictor_fits=40,completion_audit_passed=True,finished_at=a['finished_at']))
    m=json.loads((RUN/'manifest.json').read_text());m['status']='complete';m['completion']=a
    paths=[*OUT.rglob('*'),RUN/'completion_audit.json',report]
    m['outputs_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    dump(RUN/'manifest.json',m)
    print('FD003 experiment COMPLETE; overall+target gate:',gate,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['summarize','audit','finalize']);action=p.parse_args().action
    if action=='summarize':summarize_segments()
    elif action=='audit':print(json.dumps(audit(),indent=2))
    else:finalize()
