"""Post-fit conservative audit: same common RUL support across train/validation.
No estimator fitting. Uses saved matches and saved ridge predictions only.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,sys
from itertools import combinations
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]

def main(suffix):
    rid='diagnostic-fd003-prototype-probe-20261103'+('-'+suffix if suffix else '')
    run=ROOT/'runs'/rid;out=ROOT/'results/diagnostics'/('fd003_prototype_probe_20261103'+('_'+suffix if suffix else ''))
    manifest=json.loads((run/'manifest.json').read_text());assert manifest['ridge_fits_completed']==2
    if (run/'alignment_audit.json').exists():raise RuntimeError('Preserve existing alignment audit')
    for p,h in manifest['input_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    for p,h in manifest['output_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    archive=run/'initial_analysis_before_common_RUL_audit';archive.mkdir()
    shutil.copy2(run/'manifest.json',archive/'manifest.json')
    report=ROOT/manifest['report'];shutil.copy2(report,archive/'report.md')
    model_hashes={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['ridge_A.npz','ridge_B.npz','probe_predictions.npz','probe_summary.csv']}
    audit_config=dict(reason='Initial per-domain exactRUL matching controlled RUL within each domain, but support differed across domains. Restrict BOTH domains to intersection of existing matched RUL strata per pair for more conservative replication.',selection_uses_curve_features=False,selection_uses_probe_outcomes=False,ridge_refits=0,same_features_and_thresholds=True)
    (run/'alignment_config.json').write_text(json.dumps(audit_config,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('frozen_diagnosis',ROOT/'analysis/diagnostics/diagnose_fd003_prototypes.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=out;m.RUN=run
    data={s:dict(np.load(m.PARENT/(s+'.npz'))) for s in ['train','validation']}
    assign=pd.read_csv(out/'sensor_only_assignments.csv')
    for s,z in data.items():z['pid']=assign[assign.split==s].prototype.to_numpy();z['shape']=m.shape_features(z['data'])
    matches=pd.read_csv(out/'matched_windows.csv');rng=np.random.default_rng(m.C['bootstrap_seed']);rows=[];all_pairs=[];rul_rows=[]
    for a,b in combinations(range(4),2):
        per={s:matches[(matches.split==s)&(matches.prototype_a==a)&(matches.prototype_b==b)] for s in data}
        common=sorted(set(per['train'].rul_a)&set(per['validation'].rul_a))
        for split,z in data.items():
            t=per[split][per[split].rul_a.isin(common)]
            p=t[['index_a','index_b']].to_numpy(dtype=int);all_pairs.append(t)
            eng=z['engine_id'];y=z['label'].ravel()
            if not len(p):
                for f in m.SHAPE:rows.append(dict(split=split,prototype_a=a,prototype_b=b,feature=f,pairs=0,engines_a=0,engines_b=0,common_RUL=0,sufficient=False,delta=float('nan'),ci_low=float('nan'),ci_high=float('nan')))
                continue
            ea=len(np.unique(eng[p[:,0]]));eb=len(np.unique(eng[p[:,1]]));engines=np.unique(eng[p.ravel()]);lookup={int(e):i for i,e in enumerate(engines)}
            for side,col in [('a',0),('b',1)]:
                for r in common:rul_rows.append(dict(split=split,prototype_a=a,prototype_b=b,side=side,rul=int(r),windows=int(np.sum(y[p[:,col]]==r))))
            point=m.contrast(z,p);boot=m.contrast(z,p,(rng.exponential(size=(m.C['bootstrap_replicates'],len(engines))),lookup));bounds=np.quantile(boot,[.025,.975],axis=0)
            jack=np.array([m.contrast(z,p[(eng[p[:,0]]!=e)&(eng[p[:,1]]!=e)]) for e in engines])
            sufficient=ea>=m.C['minimum_engines_each_side'] and eb>=m.C['minimum_engines_each_side'] and len(p)>=m.C['minimum_matched_pairs'] and len(common)>=m.C['minimum_common_RUL']
            for j,f in enumerate(m.SHAPE):rows.append(dict(split=split,prototype_a=a,prototype_b=b,feature=f,pairs=len(p),engines_a=ea,engines_b=eb,common_RUL=len(common),sufficient=bool(sufficient),delta=float(point[j]),ci_low=float(bounds[0,j]),ci_high=float(bounds[1,j]),min_RUL=int(min(common)),max_RUL=int(max(common)),rul_values=','.join(map(str,common)),rul_mean_a=float(y[p[:,0]].mean()),rul_mean_b=float(y[p[:,1]].mean()),leave_one_engine_sign_fraction=float(np.mean(np.sign(jack[:,j])==np.sign(point[j])))))
    contrasts=pd.DataFrame(rows);contrasts.to_csv(out/'aligned_curve_contrasts.csv',index=False);pd.concat(all_pairs).to_csv(out/'aligned_matched_windows.csv',index=False);pd.DataFrame(rul_rows).to_csv(out/'aligned_matched_RUL_distributions.csv',index=False)
    repeated=[]
    for a,b in combinations(range(4),2):
        for f in m.SHAPE:
            t=contrasts[(contrasts.prototype_a==a)&(contrasts.prototype_b==b)&(contrasts.feature==f)];tr=t[t.split=='train'].iloc[0];va=t[t.split=='validation'].iloc[0]
            supported=bool(tr.sufficient and va.sufficient)
            positive=bool(supported and ((tr.ci_low>0 and va.ci_low>0) or (tr.ci_high<0 and va.ci_high<0)))
            repeated.append(dict(prototype_a=a,prototype_b=b,feature=f,support_sufficient=supported,train_delta=float(tr.delta),validation_delta=float(va.delta),same_sign=bool(np.sign(tr.delta)==np.sign(va.delta)) if supported else False,repeated_direction_with_CI=positive))
    m.dump(out/'aligned_shape_replication.json',repeated)
    # Preserve initial plot/decision; regenerate final report/contrast plot only.
    for name in ['matched_shape_contrasts.png','decision.json']:
        shutil.copy2(out/name,archive/name)
    counts=pd.read_csv(out/'counts_by_RUL_prototype.csv');probe=pd.read_csv(out/'probe_summary.csv');conc=json.loads((out/'probe_concentration.json').read_text())
    report,decision,positive=m.report(counts,contrasts,repeated,probe,conc)
    # m.plot_curves also draws unchanged fixedRUL context; no model fit.
    m.plot_curves(data,contrasts)
    s=report.read_text()
    s=s.replace('匹配后的两侧RUL分布完全相同，逐RUL数量和实际窗口/发动机列表见matched_RUL_distributions.csv、matched_windows.csv；未在无重叠寿命区域外推。','最终审计进一步对每对原型取训练/验证均可匹配的RUL交集：两域使用同一组整数RUL，并在各域按RUL等权求效应。匹配两侧实际RUL分布完全相同，最终逐RUL数量和窗口/发动机列表见aligned_matched_RUL_distributions.csv、aligned_matched_windows.csv；详细效应在aligned_curve_contrasts.csv。最初各域单独匹配的表保留，但不作为最终跨域方向依据。')
    s=s.replace('原始水平图curves_RUL45_raw.png只供上下文，不能将聚类已用的均值差当独立形态证据。','原始水平图curves_RUL45_raw.png只供上下文。标准化图仍保留聚类已用的首尾变化方向，因此图中的上升/下降本身同样不能作独立证据；只以上述中间位置/跨传感器同步指标作未直接入聚类的复核。')
    s=s.replace('| 窗口数 | 发动机数 |','| 窗口数（占该段比例） | 发动机数 |')
    for row in counts[counts.split=='train'].itertuples():
        v=counts[(counts.split=='validation')&(counts.bin==row.bin)&(counts.prototype==row.prototype)].iloc[0]
        before=f'| {row.bin} | {row.prototype} | {row.windows} / {v.windows} |'
        after=f'| {row.bin} | {row.prototype} | {row.windows} ({row.fraction_within_RUL_bin:.1%}) / {v.windows} ({v.fraction_within_RUL_bin:.1%}) |'
        s=s.replace(before,after)
    s=s.replace('## 同RUL曲线比较','[Observed] 与RUL确有明显关联：描述性窗口加权熵分解中，真实RUL解释约46.2%训练原型熵、52.1%验证原型熵；这些不是独立窗口显著性或因果占比，也不能由此断言编号仅重复RUL。原型0的31–60缺失尤其限制四类统一解释。\n\n## 同RUL曲线比较')
    s=s.replace('极少数拉动标志=False。','极少数拉动标志=False。')
    s=s.replace('固定预算2个CPU回归已结束，无调参或新增实验。','固定预算2个CPU回归已结束，无调参或新增模型。按共同RUL审计只重算形态统计量，未重新拟合岭回归；第一次按域分别匹配的报告和清单保存在initial_analysis_before_common_RUL_audit。')
    s=s.replace('# 既有产物不覆盖；完全相同配置重跑到新目录：','metaindux-ts-reproduction/.venv/bin/python metaindux-ts-reproduction/analysis/diagnostics/audit_fd003_prototype_alignment.py\n# 既有产物不覆盖；完全相同配置重跑到新目录：')
    s=s.replace('diagnose_fd003_prototypes.py --suffix replay1\n```','diagnose_fd003_prototypes.py --suffix replay1\nmetaindux-ts-reproduction/.venv/bin/python metaindux-ts-reproduction/analysis/diagnostics/audit_fd003_prototype_alignment.py --suffix replay1\n```')
    report.write_text(s)
    for name,h in model_hashes.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h,name
    for p,h in manifest['input_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    verification=dict(passed=True,reason=audit_config['reason'],additional_ridge_fits=0,total_ridge_fits=2,models_predictions_and_probe_metrics_byte_unchanged=True,common_RUL_support_verified=True,repeated_shape_contrasts=len(positive),decision=decision,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    m.dump(run/'alignment_audit.json',verification)
    manifest['revisions']=[dict(at=m.now(),type='common_RUL_cross_domain_audit',note=audit_config,source=str(Path(__file__).relative_to(ROOT)),source_sha256=verification['source_sha256'])];manifest['decision']=decision
    manifest['output_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*out.iterdir(),report,run/'alignment_audit.json',run/'alignment_config.json'] if p.is_file()}
    m.dump(run/'manifest.json',manifest)
    m.ledger(f'\n## {m.now()} {rid} common-RUL cross-domain audit\n\n- [Observed] Initial exactRUL comparisons had different support across train/validation. Final conservative audit restricts both to same integerRUL intersection per prototype pair; same preselected3features/thresholds; no outcome-based matching. Preserved initial report/manifest/plot. TotalRidgefits remains2, all model/prediction/probe metric hashes unchanged. Final repeated contrasts={len(positive)}; decision={decision}. Final shape files prefixed aligned_; run alignment_audit.json passed. No generation/test/extra fitting.\n')
    print(contrasts[(contrasts.sufficient)&(contrasts.feature.isin(['peak_middle','sensor_peak_synchrony']))].to_string(index=False));print(json.dumps(verification,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suffix',default='');main(p.parse_args().suffix)
