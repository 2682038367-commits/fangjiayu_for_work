"""Read-only saved-generation adherence audit: no training or sampling."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/metaindux-prototype-adherence-mpl')
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from itertools import combinations
import argparse,ast,hashlib,json,platform,shutil,subprocess,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/diagnostics/fd003_prototype_condition_adherence.json'
C=json.loads(CONFIG.read_text())
PARENT=ROOT/'runs'/C['parent_run'];GEN=ROOT/'results/extensions'/C['parent_run']
REALRUN=ROOT/'runs'/C['real_diagnostic'];REALOUT=ROOT/'results/diagnostics/fd003_prototype_probe_20261103'
RUN=None;OUT=None

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.bool_,)):return bool(v)
    return v

def dump(p,v):p.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def ledger(body):
    for name in ['experiment-ledger.md','reproduction-state.md']:
        with (ROOT/name).open('a') as f:f.write(body)

def frozen_function(path,name,extra=None):
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    env={'np':np,**(extra or {})};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env[name]

FEATURES=frozen_function(ROOT/'upstream/prototype_guidance.py','features')
SHAPE=frozen_function(ROOT/'analysis/diagnostics/diagnose_fd003_prototypes.py','shape_features')
DRAW=frozen_function(ROOT/'upstream/prototype_guidance.py','draw_prototypes',{'BINS':[(name,lo,hi) for name,lo,hi in C['ranges'] if name in ['0-30','31-60','61-90','91-124','125']]})

def assign(x,stats):
    features=FEATURES(x);z=(features-stats['feature_mean'])/stats['feature_scale']
    d=((z[:,None,:]-stats['centers'][None,:,:])**2).sum(2)
    return d.argmin(1),np.sqrt(d.min(1)),np.sqrt(d)

def contingency(requested,output,rul):
    index=rul*16+requested*4+output
    return np.bincount(index,minlength=126*16).reshape(126,4,4)

def metrics(tables):
    # Supports observed(R,4,4) and batched permutations(B,R,4,4).
    c=np.asarray(tables,dtype=float)
    if c.ndim==3:c=c[None]
    n=c.sum((-1,-2));nr=np.maximum(n,1);total=np.maximum(n.sum(1),1)
    row=c.sum(-1);col=c.sum(-2);diagonal=np.diagonal(c,axis1=-2,axis2=-1)
    accuracy=diagonal.sum((1,2))/total
    chance=((row*col).sum(-1)/nr).sum(1)/total
    expected=row[:,:,:,None]*col[:,:,None,:]/nr[:,:,None,None]
    info=(np.where(c>0,c*np.log(np.maximum(c,1)/np.maximum(expected,1e-300)),0)).sum((1,2,3))/total
    active=row>0;nactive=np.maximum(active.sum(2),1);strata=n>0
    recalls=(diagonal/np.maximum(row,1)).sum(2)/nactive
    balanced=(recalls*strata).sum(1)/np.maximum(strata.sum(1),1)
    balanced_chance=(((col/nr[:,:,None])*active).sum(2)/nactive*strata).sum(1)/np.maximum(strata.sum(1),1)
    kappa=np.divide(accuracy-chance,1-chance,out=np.full_like(chance,np.nan),where=np.abs(1-chance)>1e-12)
    majority=row.max(2).sum(1)/total
    return dict(accuracy=accuracy,chance=chance,kappa=kappa,balanced_recall=balanced,balanced_chance=balanced_chance,cmi=info,requested_majority_baseline=majority)

def permutations(requested,rul):
    rng=np.random.default_rng(C['permutation_seed']);p=np.empty((C['permutations'],len(requested)),dtype=np.int8)
    for r in np.unique(rul):
        idx=np.flatnonzero(rul==r)
        for j in range(len(p)):p[j,idx]=rng.permutation(requested[idx])
    return p

def null_metrics(permuted,output,rul,lo,hi):
    take=(rul>=lo)&(rul<=hi);rr=rul[take];oo=output[take];perm=permuted[:,take]
    values={k:[] for k in ['accuracy','cmi','balanced_recall']}
    for start in range(0,len(perm),50):
        pp=perm[start:start+50];offset=np.arange(len(pp))[:,None]*(126*16)
        index=offset+rr[None,:]*16+pp.astype(int)*4+oo[None,:]
        t=np.bincount(index.ravel(),minlength=len(pp)*126*16).reshape(len(pp),126,4,4)
        keep=np.arange(126);t[:,(keep<lo)|(keep>hi)]=0
        m=metrics(t)
        for key in values:values[key].extend(m[key])
    return {k:np.asarray(v) for k,v in values.items()}

def sanity_checks():
    # A collapsed majority classifier can score80%, yet carries no request info.
    req=np.tile(np.array([1]*80+[2]*15+[3]*5),2);rul=np.repeat([40,50],100)
    out=np.ones(len(req),dtype=int)
    a=metrics(contingency(req,out,rul))
    assert np.isclose(a['accuracy'][0],.8) and np.isclose(a['chance'][0],.8)
    assert np.isclose(a['kappa'][0],0) and abs(a['cmi'][0])<1e-12
    assert np.isclose(a['balanced_recall'][0],1/3)
    b=metrics(contingency(req,req,rul));assert np.isclose(b['accuracy'][0],1) and np.isclose(b['kappa'][0],1)
    return dict(passed=True,majority_collapse_accuracy=.8,majority_collapse_chance=.8,majority_collapse_kappa=0,majority_collapse_cmi=0,perfect_mapping_accuracy=1)

def class_analysis(seed,epoch,group,requested,readout,rul,permuted):
    summary=[];counts=[];recalls=[];exact=[]
    full=contingency(requested,readout,rul)
    for name,lo,hi in C['ranges']:
        take=(rul>=lo)&(rul<=hi);t=full.copy();t[(np.arange(126)<lo)|(np.arange(126)>hi)]=0
        result={k:float(v[0]) for k,v in metrics(t).items()}
        row=dict(generator_seed=seed,epoch=epoch,group=group,bin=name,windows=int(take.sum()),label_kind='actual_requested' if group=='B' else 'pseudo_not_input',**result)
        if name in ['all','31-60']:
            null=null_metrics(permuted,readout,rul,lo,hi)
            for k,x in null.items():
                row[k+'_null_mean']=float(x.mean());row[k+'_null95_low']=float(np.quantile(x,.025));row[k+'_null95_high']=float(np.quantile(x,.975));row[k+'_permutation_p_ge']=(1+int(np.sum(x>=result[k]-1e-12)))/(len(x)+1)
        summary.append(row)
        agg=t.sum(0)
        for a in range(4):
            n=int(agg[a].sum());correct=int(agg[a,a]);recalls.append(dict(generator_seed=seed,epoch=epoch,group=group,bin=name,requested_prototype=a,windows=n,correct=correct,recall=correct/n if n else np.nan))
            for b in range(4):counts.append(dict(generator_seed=seed,epoch=epoch,group=group,bin=name,requested_prototype=a,readout_prototype=b,count=int(agg[a,b]),requested_count=n,row_fraction=float(agg[a,b]/n) if n else np.nan,output_count=int(agg[:,b].sum())))
    for r in np.unique(rul):
        t=full[r]
        for a in range(4):
            for b in range(4):exact.append(dict(generator_seed=seed,epoch=epoch,group=group,rul=int(r),requested_prototype=a,readout_prototype=b,count=int(t[a,b])))
    return summary,counts,recalls,exact

def real_sync_reference(data,matches,a,b,support,rng):
    rows=[]
    for split,z in data.items():
        t=matches[(matches.split==split)&(matches.prototype_a==a)&(matches.prototype_b==b)&matches.rul_a.isin(support)]
        p=t[['index_a','index_b']].to_numpy(dtype=int);y=z['label'].ravel();eng=z['engine_id'];v=z['sync']
        engines=np.unique(eng[p.ravel()]);lookup={int(e):i for i,e in enumerate(engines)}
        weights=rng.exponential(size=(C['bootstrap_replicates'],len(engines)));point=[];boots=[]
        for r in support:
            pair=p[y[p[:,0]]==r];aa,bb=pair[:,0],pair[:,1];assert len(pair)>=3
            point.append(v[bb].mean()-v[aa].mean())
            wa=weights[:,[lookup[int(e)] for e in eng[aa]]];wb=weights[:,[lookup[int(e)] for e in eng[bb]]]
            boots.append((wb@v[bb])/wb.sum(1)-(wa@v[aa])/wa.sum(1))
        low,high=np.quantile(np.mean(boots,axis=0),[.025,.975])
        rows.append(dict(split=split,prototype_a=a,prototype_b=b,rul_values=','.join(map(str,support)),common_RUL=len(support),delta=float(np.mean(point)),ci_low=float(low),ci_high=float(high),engines_a=len(np.unique(eng[p[:,0]])),engines_b=len(np.unique(eng[p[:,1]]))))
    return rows

def sync_analysis(seed,epoch,requested,rul,sync_a,sync_b,real,matches,reference_cache):
    rows=[];supportrows=[];rng=np.random.default_rng(C['bootstrap_seed'])
    for a,b in C['sync_comparisons']:
        prior=sorted(set(matches[(matches.prototype_a==a)&(matches.prototype_b==b)].rul_a))
        support=[]
        for r in prior:
            ia=np.flatnonzero((rul==r)&(requested==a));ib=np.flatnonzero((rul==r)&(requested==b))
            enough=len(ia)>=3 and len(ib)>=3
            supportrows.append(dict(generator_seed=seed,epoch=epoch,prototype_a=a,prototype_b=b,rul=int(r),requested_count_a=len(ia),requested_count_b=len(ib),used=enough))
            if enough:support.append(r)
        assert support,'No common support; mark insufficient rather than force comparison'
        key=(a,b,tuple(support))
        if key not in reference_cache:reference_cache[key]=real_sync_reference(real,matches,a,b,support,np.random.default_rng(C['bootstrap_seed']))
        refs=reference_cache[key];ref=next(x for x in refs if x['split']=='validation')
        points=[];boots=[]
        for r in support:
            ia=np.flatnonzero((rul==r)&(requested==a));ib=np.flatnonzero((rul==r)&(requested==b))
            points.append([sync_a[ib].mean()-sync_a[ia].mean(),sync_b[ib].mean()-sync_b[ia].mean()])
            ra=rng.integers(len(ia),size=(C['bootstrap_replicates'],len(ia)));rb=rng.integers(len(ib),size=(C['bootstrap_replicates'],len(ib)))
            boots.append(np.stack([sync_a[ib][rb].mean(1)-sync_a[ia][ra].mean(1),sync_b[ib][rb].mean(1)-sync_b[ia][ra].mean(1)],axis=1))
        point=np.mean(points,axis=0);boot=np.mean(boots,axis=0);ci=np.quantile(boot,[.025,.975],axis=0)
        d=point[1]-point[0];dci=np.quantile(boot[:,1]-boot[:,0],[.025,.975])
        for j,g in enumerate(['A','B']):
            rows.append(dict(generator_seed=seed,epoch=epoch,group=g,prototype_a=a,prototype_b=b,label_kind='actual_requested' if g=='B' else 'pseudo_not_input',common_RUL=len(support),rul_values=','.join(map(str,support)),delta=float(point[j]),ci_low=float(ci[0,j]),ci_high=float(ci[1,j]),real_validation_delta=ref['delta'],real_validation_ci_low=ref['ci_low'],real_validation_ci_high=ref['ci_high'],signed_fraction_real_effect=float(point[j]/ref['delta']) if ref['delta'] else np.nan,same_direction_real=bool(point[j]*ref['delta']>0),B_minus_A_delta=float(d),B_minus_A_ci_low=float(dci[0]),B_minus_A_ci_high=float(dci[1])))
    return rows,supportrows

def plots(confusion,sync):
    fig,axes=plt.subplots(2,2,figsize=(9,8));image=None
    for ax,(seed,epoch) in zip(axes.ravel(),[(23,64),(23,70),(13,64),(13,70)]):
        t=confusion[(confusion.generator_seed==seed)&(confusion.epoch==epoch)&(confusion.group=='B')&(confusion.bin=='31-60')]
        mat=t.pivot(index='requested_prototype',columns='readout_prototype',values='row_fraction').to_numpy()
        counts=t.pivot(index='requested_prototype',columns='readout_prototype',values='count').to_numpy()
        image=ax.imshow(np.nan_to_num(mat),vmin=0,vmax=1,cmap='Blues')
        for i in range(4):
            for j in range(4):ax.text(j,i,'no requests' if np.isnan(mat[i,j]) and j==1 else '' if np.isnan(mat[i,j]) else f'{mat[i,j]:.1%}\n(n={counts[i,j]})',ha='center',va='center',fontsize=7,color='white' if np.nan_to_num(mat[i,j])>.65 else 'black')
        ax.set(xticks=range(4),yticks=range(4),xlabel='Frozen-cluster readout ID',ylabel='Requested ID',title=f'seed{seed},epoch{epoch},RUL31-60')
    fig.colorbar(image,ax=axes.ravel().tolist(),fraction=.025,pad=.03,label='row fraction');fig.savefig(OUT/'confusion31_60.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(a,b) in zip(axes,C['sync_comparisons']):
        t=sync[(sync.prototype_a==a)&(sync.prototype_b==b)]
        for j,(seed,epoch) in enumerate([(23,64),(23,70),(13,64),(13,70)]):
            for g,offset,color in [('A',-.12,'#888888'),('B',.12,'#0072b2')]:
                r=t[(t.generator_seed==seed)&(t.epoch==epoch)&(t.group==g)].iloc[0]
                ax.errorbar(j+offset,r.delta,yerr=[[max(0,r.delta-r.ci_low)],[max(0,r.ci_high-r.delta)]],fmt='o',color=color,label=('A pseudo' if g=='A' else 'B requested') if j==0 else None)
        r=t.iloc[0];ax.axhline(r.real_validation_delta,color='#d55e00',ls='--',label='real validation')
        ax.axhspan(r.real_validation_ci_low,r.real_validation_ci_high,color='#d55e00',alpha=.1)
        ax.axhline(0,color='black',lw=.6);ax.set_xticks(range(4),['23/e64','23/e70','13/e64','13/e70']);ax.set_title(f'Exact-RUL synchrony P{b}-P{a}');ax.legend(fontsize=8)
    fig.suptitle('Generated window-bootstrap vs real engine-weighted intervals (different uncertainty units)',fontsize=9);fig.tight_layout();fig.savefig(OUT/'synchrony_condition_response.png',dpi=180);plt.close(fig)

def report(summary,sync,ood):
    main=summary[(summary.group=='B')&(summary.bin=='31-60')]
    recalls=pd.read_csv(OUT/'class_recalls.csv')
    r2=recalls[(recalls.group=='B')&(recalls.bin=='31-60')&(recalls.requested_prototype==2)]
    r3=recalls[(recalls.group=='B')&(recalls.bin=='31-60')&(recalls.requested_prototype==3)]
    control=summary[(summary.group=='A')&(summary.bin=='31-60')]
    signals=sync[sync.group=='B']
    robust_response=bool((main.cmi_permutation_p_ge<=.01).all() and (main.kappa>.5).all())
    same_sync=bool((signals.delta>0).all() and (signals.ci_low>0).all())
    decision='已有输出支持原型条件被有效响应，但这种模式控制尚不足以改善下游寿命预测' if robust_response and same_sync else '条件响应存在局限或混合证据，不能把生成失败唯一归因于原型无价值或完全未被利用'
    dump(OUT/'closeout_decision.json',dict(decision=decision,all_four_show_strong_requested_readout_association=robust_response,all_prespecified_sync_contrasts_positive_with_conditional_CI=same_sync,no_training_restart=True,limits='frozen nearest-center readout and two shape contrasts only; all samples existing, no paired counterfactual regeneration'))
    lines=['# FD003已有生成样本：原型条件是否落实到输出模式','',
      '[Observed] **结论：'+decision+'。** 不能把前次生成实验的失败归结为原型本身毫无价值，也不能归结为条件完全被忽略。这个结论仅覆盖冻结聚类读出和预先选定的同步指标，不保证全部真实信号模式都被准确复现。','',
      f'[Observed] 31–60总体重识别率{main.accuracy.min():.1%}–{main.accuracy.max():.1%}，同RUL机会期望{main.chance.min():.1%}–{main.chance.max():.1%}，类别等权召回{main.balanced_recall.min():.1%}–{main.balanced_recall.max():.1%}（平衡机会值33.3%）。指定P2后P2读出率{r2.recall.min():.1%}–{r2.recall.max():.1%}；指定P3后P3读出率仅{r3.recall.min():.1%}–{r3.recall.max():.1%}，控制并不均衡。','',
      f'[Observed] A未输入原型，仅作伪标签检查：条件κ为{control.kappa.min():+.4f}至{control.kappa.max():+.4f}，CMI置换p为{control.cmi_permutation_p_ge.min():.3f}至{control.cmi_permutation_p_ge.max():.3f}。这不支持把B的高关联解释为类别占比或RUL清单本身。','',
      '[Observed] 本轮只读父实验4个B候选和对应4个A样本。指定编号全部保留且独立随机数流回放逐项一致；原始A编号为−1，表中A只用B编号作为未输入模型的伪标签。没有训练、生成、重新聚类或原测试访问。','',
      '[Code] 按实际42维传感器统计特征/训练标准化参数/冻结4中心重新分配。生成窗口已经在父实验归一化坐标中，不再拟合或更改MinMax。指定RUL与指定原型用于分组，绝不作为重识别输入。','',
      '## 主表：31–60指定编号与重识别','',
      '| 生成器种子 | 轮次 | 命中率 | 同RUL无关联期望 | 条件κ | 等RUL宏召回 | CMI | 置换CMI均值 | CMI置换p |','|---|---|---|---|---|---|---|---|---|']
    for r in main.itertuples():lines.append(f'| {r.generator_seed} | {r.epoch} | {r.accuracy:.2%} | {r.chance:.2%} | {r.kappa:.4f} | {r.balanced_recall:.2%} | {r.cmi:.4f} | {r.cmi_null_mean:.4f} | {r.cmi_permutation_p_ge:.4f} |')
    lines+=['','命中率机会值按每个精确RUL的指定占比×输出占比计算；条件κ扣除此机会值。宏召回对每个RUL实际指定类别等权再对RUL等权，P0未指定不虚构召回。CMI是给定精确RUL后指定/输出的关联，不要求类别恰好同号；1000次RUL内置换保留两边各类原占比，校准有限样本偏差。各p值探索性、未多重校正，4个候选共享噪声/编号，不当成4个独立随机实验。','',
      '完整混淆计数及行占比：confusion_by_RUL_range.csv；精确RUL混淆：confusion_exact_RUL.csv；单类召回：class_recalls.csv。所有固定轮次/分段含125分别保留，未按命中率选检查点。125只控制截断标签，不代表相同未截断寿命。','',
      '![条件到输出的混淆矩阵](../../../results/diagnostics/'+OUT.name+'/confusion31_60.png)','',
      '## 同RUL同步性是否保留','',
      '| 种子 | 轮次 | 指定原型差 | B输出同步性差[固定样本95%区间] | 真实验证差 | B/真实差 | A伪标签差 |','|---|---|---|---|---|---|---|']
    for r in sync[sync.group=='B'].itertuples():
        a=sync[(sync.generator_seed==r.generator_seed)&(sync.epoch==r.epoch)&(sync.group=='A')&(sync.prototype_a==r.prototype_a)&(sync.prototype_b==r.prototype_b)].iloc[0]
        lines.append(f'| {r.generator_seed} | {r.epoch} | P{r.prototype_b}−P{r.prototype_a} | {r.delta:+.4f} [{r.ci_low:+.4f},{r.ci_high:+.4f}] | {r.real_validation_delta:+.4f} | {r.signed_fraction_real_effect:.1%} | {a.delta:+.4f} |')
    lines+=['','指标沿用真实诊断：每传感器48步分4季度，找最大季度间变化位置，再取跨14传感器在同一段达到最大变化的最高比例。分组按指定编号，不按事后重识别，防止把聚类本身制造的差异误作条件响应。P2−P1、P3−P1只检查先前真实诊断已复核的方向，支持RUL清单及各类样本量见synchrony_RUL_support.csv。各RUL等权、类别样本量不等不直接累加。','',
      '生成区间按精确RUL×指定类别重采样窗口，同一索引用于A/B；它描述固定保存批次的采样波动，不是发动机或生成器训练种子不确定性，生成数据不虚构发动机ID。真实参考区间沿用发动机层随机权重方法；两种区间的统计单位不同。','',
      '![同步性条件响应](../../../results/diagnostics/'+OUT.name+'/synchrony_condition_response.png)','',
      '## 解释范围','',
      '- 命中高但与条件机会基线接近，不能说听懂原型；CMI高而命中低，可能是输出随编号改变但并非正确原型，需结合混淆及同步性。',
      '- B相对A总体输出改变不等于原型控制成功，因为两者权重/训练不同。A伪标签关联用于参照有限采样/RUL偏差，不是给A实际加了条件。',
      '- 冻结中心分类器在生成信号上可能外推，距离统计见cluster_distance_diagnostics.csv；远离真实训练分布时，最近编号不能证明真实模式已复现。',
      '- 若指定后的输出模式无相应变化，只能支持条件未被有效利用是贡献因素，不能证明原型本身没有价值或定位唯一原因。若响应存在但用途变差，只说明这种可控性尚不足以改善寿命预测。',
      '- 先前FD003生成实验四个固定配对总体/31–60均退步；本轮无新的下游模型或测试成绩。岭回归正结果和本轮关联均不授权重开扩散训练。','',
      '当前→保存样本上的条件响应诊断→论文无同设计指标→论文gap不适用→两种子且共享采样噪声/固定批次限制→严格复现与先前生成负结果保持。固定只读预算结束，不新增实验。','',
      '配置：configs/diagnostics/fd003_prototype_condition_adherence.json；协议：docs/operations/FD003_PROTOTYPE_CONDITION_ADHERENCE.md；快照：'+str(RUN.relative_to(ROOT))+'/manifest.json；结果：'+str(OUT.relative_to(ROOT))+'/。','',
      '复现到新目录（工作区根目录）：','```bash','metaindux-ts-reproduction/.venv/bin/python metaindux-ts-reproduction/analysis/diagnostics/diagnose_fd003_prototype_condition_adherence.py --suffix replay1','```']
    p=ROOT/'docs/conclusions/final'/('fd003_prototype_condition_adherence_20261105_report'+('_'+RUN.name.split(C['run_id']+'-')[-1] if RUN.name!=C['run_id'] else '')+'.md');p.write_text('\n'.join(lines)+'\n');return p

def main(suffix):
    global RUN,OUT
    if suffix and not suffix.replace('-','').replace('_','').isalnum():raise ValueError('simple suffix only')
    RUN=ROOT/'runs'/(C['run_id']+('-'+suffix if suffix else ''));OUT=ROOT/'results/diagnostics'/('fd003_prototype_condition_adherence_20261105'+('_'+suffix if suffix else ''))
    if (RUN/'manifest.json').exists():raise RuntimeError('Preserve artifacts; use --suffix')
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    samples=[GEN/f'seed{s}'/g/f'samples{e}.npz' for s in C['generator_seeds'] for e in C['checkpoints'] for g in ['A','B']]
    paths=[Path(__file__),CONFIG,ROOT/'docs/operations/FD003_PROTOTYPE_CONDITION_ADHERENCE.md',ROOT/'upstream/prototype_guidance.py',ROOT/'scripts/run_prototype_guidance_fd003.py',ROOT/'analysis/diagnostics/diagnose_fd003_prototypes.py',PARENT/'manifest.json',PARENT/'config.json',PARENT/'prototypes.npz',PARENT/'plan.npz',PARENT/'train.npz',PARENT/'validation.npz',REALRUN/'manifest.json',REALOUT/'aligned_matched_windows.csv',REALOUT/'aligned_curve_contrasts.csv',*samples]
    paths += [GEN/f'seed{s}'/g/f'sampling{e}.json' for s in C['generator_seeds'] for e in C['checkpoints'] for g in ['A','B']]
    manifest=dict(run_id=RUN.name,status='running',started_at=now(),scope='diagnostic',role='supporting',scope_contract=C['scope_contract'],authorization='Explicit user read-only saved FD003 requested/output prototype adherence closeout',command=[sys.executable,*sys.argv],config=C,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),device='CPU'),input_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},model_fits=0,new_generated_windows=0)
    dump(RUN/'manifest.json',manifest);(RUN/'source').mkdir()
    for p in paths:
        if p.suffix in ['.py','.json','.md']:shutil.copy2(p,RUN/'source'/p.name)
    (RUN/'workspace.diff').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT));(RUN/'git_status.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=ROOT))
    ledger(f'\n## {manifest["started_at"]} {RUN.name} started\n\n- Diagnostic/supporting read-only FD0034savedBcells plus pairedAnegativecontrol. Explicit user authorizes adherence/synchrony check only;0fits/0newgeneratedwindows. Frozen42features/standardizer/centers, within-exactRUL chance/permutation controls and real-positive synchrony pairs fixed. Snapshot runs/{RUN.name}/manifest.json. Prior diffusion failures and ridge evidence retained; no expansion.\n')
    try:
        parent=json.loads((PARENT/'manifest.json').read_text());assert parent['status']=='complete'
        for key in ['hashes','outputs_sha256']:
            for p,h in parent[key].items():assert sha(ROOT/p)==h,p
        previous=json.loads((REALRUN/'manifest.json').read_text());assert previous['status']=='complete'
        for key in ['input_hashes','output_hashes']:
            for p,h in previous[key].items():assert sha(ROOT/p)==h,p
        dump(RUN/'sanity_checks.json',sanity_checks())
        stats=dict(np.load(PARENT/'prototypes.npz'));plan=dict(np.load(PARENT/'plan.npz'));rul=plan['target_rul'].ravel().astype(int)
        data={split:dict(np.load(PARENT/(split+'.npz'))) for split in ['train','validation']}
        for split,z in data.items():
            pid,d,_=assign(z['data'],stats)
            if split=='train':assert np.array_equal(pid,stats['prototype_id'])
            z['sync']=SHAPE(z['data'])[:,2];z['nearest_distance']=d
        threshold=float(np.quantile(data['train']['nearest_distance'],.99))
        reference_rows=[]
        for split,z in data.items():reference_rows.append(dict(source=split,n=len(z['nearest_distance']),median=float(np.median(z['nearest_distance'])),p95=float(np.quantile(z['nearest_distance'],.95)),p99=float(np.quantile(z['nearest_distance'],.99)),fraction_above_train99=float(np.mean(z['nearest_distance']>threshold))))
        dump(OUT/'real_cluster_distances.json',dict(train99_threshold=threshold,references=reference_rows))
        matches=pd.read_csv(REALOUT/'aligned_matched_windows.csv');pconf=json.loads((PARENT/'config.json').read_text())
        requested=DRAW(rul,stats,np.random.default_rng(pconf['prototype_sampling_seed']))
        perms=permutations(requested,rul)
        summaries=[];confusions=[];recalls=[];exact=[];synchrony=[];support=[];ood=[];cache={};readout_rows=[]
        for seed in C['generator_seeds']:
            for epoch in C['checkpoints']:
                arrays={g:dict(np.load(GEN/f'seed{seed}'/g/f'samples{epoch}.npz')) for g in ['A','B']}
                assert np.array_equal(arrays['B']['prototype_id'],requested)
                audits=[json.loads((GEN/f'seed{seed}'/g/f'sampling{epoch}.json').read_text()) for g in ['A','B']];assert audits[0]['rng_hash']==audits[1]['rng_hash']
                sync={}
                for g,z in arrays.items():
                    assert np.array_equal(z['label'].ravel(),rul) and z['data'].shape==(len(rul),48,14)
                    assert np.isfinite(z['data']).all()
                    if g=='A':assert np.all(z['prototype_id']==-1)
                    readout,nearest,distances=assign(z['data'],stats);sync[g]=SHAPE(z['data'])[:,2]
                    s,c,rr,ex=class_analysis(seed,epoch,g,requested,readout,rul,perms)
                    summaries+=s;confusions+=c;recalls+=rr;exact+=ex
                    np.savez_compressed(OUT/f'seed{seed}_{g}_epoch{epoch}_readout.npz',requested_or_pseudo=requested,readout=readout,target_rul=rul,synchrony=sync[g],nearest_center_distance=nearest,center_distances=distances)
                    for name,lo,hi in C['ranges']:
                        mask=(rul>=lo)&(rul<=hi);d=nearest[mask]
                        ood.append(dict(generator_seed=seed,epoch=epoch,group=g,bin=name,n=int(mask.sum()),median=float(np.median(d)),p95=float(np.quantile(d,.95)),train99_threshold=threshold,fraction_above_train99=float(np.mean(d>threshold))))
                    print('classified',seed,epoch,g,flush=True)
                sy,su=sync_analysis(seed,epoch,requested,rul,sync['A'],sync['B'],data,matches,cache);synchrony+=sy;support+=su
        summary=pd.DataFrame(summaries);confusion=pd.DataFrame(confusions);sync=pd.DataFrame(synchrony)
        for filename,table in [('classification_summary.csv',summary),('confusion_by_RUL_range.csv',confusion),('class_recalls.csv',pd.DataFrame(recalls)),('confusion_exact_RUL.csv',pd.DataFrame(exact)),('synchrony_summary.csv',sync),('synchrony_RUL_support.csv',pd.DataFrame(support)),('cluster_distance_diagnostics.csv',pd.DataFrame(ood))]:table.to_csv(OUT/filename,index=False)
        pd.DataFrame([r for v in cache.values() for r in v]).to_csv(OUT/'real_synchrony_references.csv',index=False)
        plots(confusion,sync);path=report(summary,sync,ood)
        # Direct checks independent of vectorized metrics and no source mutation.
        for seed in C['generator_seeds']:
            for epoch in C['checkpoints']:
                z=np.load(OUT/f'seed{seed}_B_epoch{epoch}_readout.npz');mask=(z['target_rul']>=31)&(z['target_rul']<=60)
                observed=np.mean(z['requested_or_pseudo'][mask]==z['readout'][mask]);row=summary[(summary.generator_seed==seed)&(summary.epoch==epoch)&(summary.group=='B')&(summary.bin=='31-60')].iloc[0]
                assert abs(observed-row.accuracy)<1e-12
                expected=0.
                for r in np.unique(rul[mask]):
                    take=rul==r;p=np.bincount(requested[take],minlength=4)/take.sum();q=np.bincount(z['readout'][take],minlength=4)/take.sum();expected+=take.sum()/mask.sum()*np.dot(p,q)
                assert abs(expected-row.chance)<1e-12
        for p,h in manifest['input_hashes'].items():assert sha(ROOT/p)==h,p
        dump(RUN/'verification.json',dict(passed=True,stored_requested_IDs_verified_all4cells=True,train_assignments_reproduced=True,no_generated_feature_refit=True,A_pseudo_labels_never_input=True,sameRUL_chance_directly_verified=True,majority_collapse_sanity_passed=True,parent_artifacts_unchanged=True,model_fits=0,new_generated_windows=0))
        manifest.update(status='complete',finished_at=now(),report=str(path.relative_to(ROOT)),output_hashes={str(p.relative_to(ROOT)):sha(p) for p in [*OUT.iterdir(),path,RUN/'verification.json',RUN/'sanity_checks.json'] if p.is_file()});dump(RUN/'manifest.json',manifest)
        ledger(f'\n## {manifest["finished_at"]} {RUN.name} complete\n\n- [Observed] All4savedBcells and4pairedAcontrols classified using frozen train statistics/centers. Stored requests/RNG audited; within-exactRUL chance/permutations and real-positive synchrony differences reported, no training/newgeneration/test access. Report {path.relative_to(ROOT)}; classification_summary.csv/synchrony_summary.csv/verification.json at diagnostic output/snapshot. Current conditional-response evidence→no paper target→gapNA→sharednoise/smalltrainingseedcount/nearestcenterextrapolation limits→prior diffusion negative and ridge results retained. No automatic further training.\n')
        print(summary[(summary.group=='B')&(summary.bin=='31-60')].to_string(index=False));print(sync[sync.group=='B'].to_string(index=False));print('REPORT',path)
    except Exception as e:
        manifest.update(status='failed',finished_at=now(),error=repr(e));dump(RUN/'manifest.json',manifest);ledger(f'\n## {now()} {RUN.name} failed\n\n- [Observed] {repr(e)}; preserve partial outputs, no new model or generation.\n');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suffix',default='');args=p.parse_args()
    with threadpool_limits(limits=2):main(args.suffix)
