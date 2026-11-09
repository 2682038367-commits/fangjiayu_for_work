"""Frozen-cluster FD003 diagnostics; exactly two CPU ridge fits, no generation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/metaindux-fd003-prototype-probe-mpl')
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,ast,hashlib,json,platform,shutil,subprocess,sys
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import sklearn
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/diagnostics/fd003_prototype_probe.json'
C=json.loads(CONFIG.read_text())
PARENT=ROOT/'runs'/C['parent_run']
SENSORS=['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']
SHAPE=C['curve_features']
RUN=None;OUT=None

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def ledger(text):
    for name in ['experiment-ledger.md','reproduction-state.md']:
        with (ROOT/name).open('a') as f:f.write(text)

def features_function():
    # Execute exactly the inspected, frozen function, without importing a model.
    tree=ast.parse((ROOT/'upstream/prototype_guidance.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='features')
    scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_features','exec'),scope)
    return scope['features']

def assign(windows,stats):
    f=features_function()(windows)
    z=(f-stats['feature_mean'])/stats['feature_scale']
    ids=np.argmin(((z[:,None,:]-stats['centers'][None,:,:])**2).sum(2),axis=1)
    return f,z,ids

def shape_features(x):
    q=np.stack([x[:,i*12:(i+1)*12].mean(1,dtype=np.float64) for i in range(4)],axis=1)
    d=np.diff(q,axis=1)
    energy=(d*d).sum(2)
    middle=energy[:,1]/np.maximum(energy.sum(1),1e-20)
    peak=(energy.argmax(1)==1).astype(float)
    sensor_peak=np.abs(d).argmax(1)
    sync=np.stack([(sensor_peak==i).mean(1) for i in range(3)],1).max(1)
    return np.stack([middle,peak,sync],1)

def reconstruct_indices(data,norm):
    columns=['id','cycle','setting1','setting2','setting3']+[f's{i}' for i in range(1,22)]
    frame=pd.read_csv(ROOT/'upstream/data/train_FD003.txt',sep=r'\s+',header=None,names=columns)
    values=(frame[SENSORS].to_numpy()*norm['scale']+norm['offset']).astype('float32')
    rows=[];offsets={'train':0,'validation':0};counts={'train':0,'validation':0}
    val_engines=set(data['validation']['engine_id'])
    for engine,g in frame.groupby('id',sort=True):
        assert np.array_equal(g.cycle.to_numpy(),np.arange(1,len(g)+1))
        split='validation' if engine in val_engines else 'train'
        for end in range(47,len(g)):
            i=offsets[split];x=values[g.index[end-47:end+1]];y=min(125,len(g)-end-1)
            assert np.array_equal(x,data[split]['data'][i]),(split,engine,end)
            assert data[split]['label'][i,0]==y and data[split]['engine_id'][i]==engine
            if split=='train':assert data[split]['end_cycle'][i]==int(g.iloc[end].cycle)
            rows.append(dict(split=split,window_index=i,engine_id=int(engine),start_cycle=int(g.iloc[end-47].cycle),end_cycle=int(g.iloc[end].cycle),raw_start_row_zero_based=int(g.index[end-47]),raw_end_row_zero_based=int(g.index[end]),uncapped_rul=len(g)-end-1,capped_rul=y))
            offsets[split]+=1
    assert all(offsets[s]==len(data[s]['data']) for s in data)
    return pd.DataFrame(rows)

def count_tables(data):
    rows=[];exact=[];engine_rows=[]
    for split,z in data.items():
        y=z['label'].reshape(-1);ids=z['engine_id'];p=z['pid']
        for name,lo,hi in C['bins']:
            band=(y>=lo)&(y<=hi)
            for k in range(4):
                take=band&(p==k);eng=np.unique(ids[take]);total=int(band.sum())
                rows.append(dict(split=split,bin=name,prototype=k,windows=int(take.sum()),fraction_within_RUL_bin=float(take.sum()/total),fraction_of_prototype=float(take.sum()/max(1,(p==k).sum())),engines=len(eng),band_engines=len(np.unique(ids[band])),engine_ids=','.join(map(str,eng))))
                for e in eng:engine_rows.append(dict(split=split,bin=name,prototype=k,engine_id=int(e),windows=int((take&(ids==e)).sum())))
        for r in range(126):
            for k in range(4):
                take=(y==r)&(p==k)
                exact.append(dict(split=split,rul=r,prototype=k,windows=int(take.sum()),engines=len(np.unique(ids[take]))))
    pd.DataFrame(rows).to_csv(OUT/'counts_by_RUL_prototype.csv',index=False)
    pd.DataFrame(exact).to_csv(OUT/'counts_exact_RUL.csv',index=False)
    pd.DataFrame(engine_rows).to_csv(OUT/'counts_by_engine.csv',index=False)
    # Association is descriptive, not a significance test on dependent windows.
    from sklearn.metrics import normalized_mutual_info_score
    association=[]
    for split,z in data.items():
        y=z['label'].ravel();p=z['pid'];stage=np.select([y<=30,y<=60,y<=90,y<125],[0,1,2,3],default=4)
        conditional=sum((y==r).mean()*(-sum(v*np.log(v) for v in np.bincount(p[y==r],minlength=4)/np.sum(y==r) if v>0)) for r in np.unique(y))
        global_p=np.bincount(p,minlength=4)/len(p)
        entropy=-sum(v*np.log(v) for v in global_p if v>0)
        association.append(dict(split=split,NMI_stage_prototype=float(normalized_mutual_info_score(stage,p)),prototype_entropy_nats=entropy,prototype_entropy_given_exact_RUL_nats=conditional,fraction_prototype_entropy_explained_by_exact_RUL=float(1-conditional/entropy)))
    dump(OUT/'RUL_association.json',association)
    return pd.DataFrame(rows)

def choose_exact_pairs(z,a,b):
    y=z['label'].ravel().astype(int);pid=z['pid'];eng=z['engine_id']
    usage={int(e):0 for e in np.unique(eng)};pairs=[];support=[]
    for r in range(31,61):
        aa=np.flatnonzero((y==r)&(pid==a));bb=np.flatnonzero((y==r)&(pid==b))
        support.append(dict(rul=r,n_a=len(aa),n_b=len(bb),eligible=len(aa)>=3 and len(bb)>=3))
        if len(aa)<3 or len(bb)<3:continue
        n=min(len(aa),len(bb))
        aa=sorted(aa,key=lambda i:(usage[int(eng[i])],int(eng[i]),int(i)))[:n]
        bb=sorted(bb,key=lambda i:(usage[int(eng[i])],int(eng[i]),int(i)))[:n]
        for i,j in zip(aa,bb):
            assert eng[i]!=eng[j] and y[i]==y[j]
            pairs.append((i,j));usage[int(eng[i])]+=1;usage[int(eng[j])]+=1
    return np.array(pairs,dtype=int).reshape(-1,2),support

def contrast(z,pairs,weights=None):
    # Equal-RUL strata, identical RUL distribution on both sides.
    y=z['label'].ravel();v=z['shape'];eng=z['engine_id']
    values=[]
    for r in np.unique(y[pairs[:,0]]):
        p=pairs[y[pairs[:,0]]==r]
        a,b=p[:,0],p[:,1]
        if weights is None:values.append(v[b].mean(0)-v[a].mean(0))
        else:
            w,lookup=weights
            wa=w[:,[lookup[int(e)] for e in eng[a]]];wb=w[:,[lookup[int(e)] for e in eng[b]]]
            values.append((wb@v[b])/wb.sum(1)[:,None]-(wa@v[a])/wa.sum(1)[:,None])
    return np.mean(values,axis=0)

def compare_shapes(data):
    rows=[];matches=[];support_rows=[];rul_rows=[];engine_rows=[]
    rng=np.random.default_rng(C['bootstrap_seed'])
    for split,z in data.items():
        y=z['label'].ravel();eng=z['engine_id']
        for a,b in combinations(range(4),2):
            p,support=choose_exact_pairs(z,a,b)
            for s in support:support_rows.append(dict(split=split,prototype_a=a,prototype_b=b,**s))
            n_a=len(np.unique(eng[p[:,0]])) if len(p) else 0;n_b=len(np.unique(eng[p[:,1]])) if len(p) else 0
            nr=len(np.unique(y[p[:,0]])) if len(p) else 0
            sufficient=n_a>=C['minimum_engines_each_side'] and n_b>=C['minimum_engines_each_side'] and nr>=C['minimum_common_RUL'] and len(p)>=C['minimum_matched_pairs']
            for i,j in p:
                matches.append(dict(split=split,prototype_a=a,prototype_b=b,index_a=int(i),index_b=int(j),engine_a=int(eng[i]),engine_b=int(eng[j]),rul_a=int(y[i]),rul_b=int(y[j]),rul_difference=int(y[j]-y[i])))
            if not len(p):
                for f in SHAPE:rows.append(dict(split=split,prototype_a=a,prototype_b=b,feature=f,pairs=0,engines_a=0,engines_b=0,common_RUL=0,sufficient=False,delta=float('nan'),ci_low=float('nan'),ci_high=float('nan')))
                continue
            for side,col in [('a',0),('b',1)]:
                for r in np.unique(y[p[:,col]]):rul_rows.append(dict(split=split,prototype_a=a,prototype_b=b,side=side,rul=int(r),windows=int(np.sum(y[p[:,col]]==r))))
                for e in np.unique(eng[p[:,col]]):engine_rows.append(dict(split=split,prototype_a=a,prototype_b=b,side=side,engine_id=int(e),matched_windows=int(np.sum(eng[p[:,col]]==e))))
            point=contrast(z,p)
            engines=np.unique(eng[p.ravel()]);lookup={int(e):i for i,e in enumerate(engines)}
            weights=rng.exponential(size=(C['bootstrap_replicates'],len(engines)))
            boot=contrast(z,p,(weights,lookup));bounds=np.quantile(boot,[.025,.975],axis=0)
            jack=[]
            for e in engines:
                keep=(eng[p[:,0]]!=e)&(eng[p[:,1]]!=e)
                jack.append(contrast(z,p[keep]))
            jack=np.array(jack)
            for j,f in enumerate(SHAPE):
                rows.append(dict(split=split,prototype_a=a,prototype_b=b,feature=f,pairs=len(p),engines_a=n_a,engines_b=n_b,common_RUL=nr,min_RUL=int(y[p[:,0]].min()),max_RUL=int(y[p[:,0]].max()),rul_mean_a=float(y[p[:,0]].mean()),rul_mean_b=float(y[p[:,1]].mean()),max_abs_RUL_difference=0,sufficient=bool(sufficient),delta=float(point[j]),ci_low=float(bounds[0,j]),ci_high=float(bounds[1,j]),leave_one_engine_sign_fraction=float(np.mean(np.sign(jack[:,j])==np.sign(point[j])))))
    table=pd.DataFrame(rows);table.to_csv(OUT/'matched_curve_contrasts.csv',index=False)
    pd.DataFrame(matches).to_csv(OUT/'matched_windows.csv',index=False)
    pd.DataFrame(support_rows).to_csv(OUT/'exact_matching_support.csv',index=False)
    pd.DataFrame(rul_rows).to_csv(OUT/'matched_RUL_distributions.csv',index=False)
    pd.DataFrame(engine_rows).to_csv(OUT/'matched_engine_exposure.csv',index=False)
    repeated=[]
    for a,b in combinations(range(4),2):
        for f in SHAPE:
            t=table[(table.prototype_a==a)&(table.prototype_b==b)&(table.feature==f)]
            train=t[t.split=='train'].iloc[0];val=t[t.split=='validation'].iloc[0]
            supported=bool(train.sufficient and val.sufficient)
            positive=bool(supported and ((train.ci_low>0 and val.ci_low>0) or (train.ci_high<0 and val.ci_high<0)))
            repeated.append(dict(prototype_a=a,prototype_b=b,feature=f,support_sufficient=supported,train_delta=float(train.delta),validation_delta=float(val.delta),same_sign=bool(np.sign(train.delta)==np.sign(val.delta)) if supported else False,repeated_direction_with_CI=positive))
    dump(OUT/'shape_replication.json',repeated)
    return table,repeated

def probe(data):
    train=data['train'];val=data['validation'];y=train['label'].ravel().astype(float)
    eye=np.eye(4)
    designs={'A':(train['z'],val['z']),'B':(np.concatenate([train['z'],eye[train['pid']]],1),np.concatenate([val['z'],eye[val['pid']]],1))}
    predictions={};fits=0
    for name,(x,v) in designs.items():
        model=Ridge(alpha=C['ridge_alpha'],solver=C['ridge_solver'],fit_intercept=True)
        model.fit(x,y);fits+=1;predictions[name]=model.predict(v)
        np.savez(OUT/f'ridge_{name}.npz',coef=model.coef_,intercept=model.intercept_,alpha=model.alpha)
    assert fits==2
    yy=val['label'].ravel().astype(float);eng=val['engine_id'];pid=val['pid']
    np.savez_compressed(OUT/'probe_predictions.npz',A=predictions['A'],B=predictions['B'],label=yy,engine_id=eng,prototype=pid)
    pd.DataFrame(dict(window_index=np.arange(len(yy)),engine_id=eng,rul=yy,prototype=pid,prediction_A=predictions['A'],prediction_B=predictions['B'])).to_csv(OUT/'probe_predictions.csv',index=False)
    rng=np.random.default_rng(C['bootstrap_seed']);summary=[];perengine=[];concentration=[]
    for name,lo,hi in C['bins']:
        take=(yy>=lo)&(yy<=hi);engines=np.unique(eng[take]);mse={}
        for group in ['A','B']:
            mse[group]=np.array([np.mean((predictions[group][take&(eng==e)]-yy[take&(eng==e)])**2) for e in engines])
        delta=mse['B']-mse['A'];a=float(np.sqrt(mse['A'].mean()));b=float(np.sqrt(mse['B'].mean()))
        draws=rng.integers(0,len(engines),size=(C['bootstrap_replicates'],len(engines)))
        boots=np.sqrt(mse['B'][draws].mean(1))-np.sqrt(mse['A'][draws].mean(1))
        ci=np.quantile(boots,[.025,.975])
        summary.append(dict(bin=name,A_RMSE=a,B_RMSE=b,delta=b-a,improved_engines=int((delta<0).sum()),n_engines=len(engines),windows=int(take.sum()),delta_ci_low=float(ci[0]),delta_ci_high=float(ci[1])))
        for i,e in enumerate(engines):perengine.append(dict(bin=name,engine_id=int(e),windows=int(np.sum(take&(eng==e))),mse_A=float(mse['A'][i]),mse_B=float(mse['B'][i]),delta_mse=float(delta[i]),rmse_A=float(np.sqrt(mse['A'][i])),rmse_B=float(np.sqrt(mse['B'][i])),improved=bool(delta[i]<0)))
        gain=-delta;order=np.argsort(-gain);positive=order[gain[order]>0];totalpositive=float(gain[positive].sum())
        top=positive[:3];remaining=np.ones(len(engines),dtype=bool);remaining[top]=False
        concentration.append(dict(bin=name,improved_engines=int((gain>0).sum()),positive_gain_MSE_sum=totalpositive,top1_positive_gain_share=float(gain[positive[:1]].sum()/totalpositive) if totalpositive else 0.,top3_positive_gain_share=float(gain[top].sum()/totalpositive) if totalpositive else 0.,top3_engine_ids=[int(e) for e in engines[top]],delta_RMSE_excluding_best3=float(np.sqrt(mse['B'][remaining].mean())-np.sqrt(mse['A'][remaining].mean())),few_engine_driven=bool(b<a and (len(positive)<=3 or np.mean(delta[remaining])>=0))))
    table=pd.DataFrame(summary);table.to_csv(OUT/'probe_summary.csv',index=False);pd.DataFrame(perengine).to_csv(OUT/'probe_per_engine.csv',index=False);dump(OUT/'probe_concentration.json',concentration)
    dump(OUT/'probe_design.json',dict(fits=fits,alpha=C['ridge_alpha'],solver=C['ridge_solver'],A_shape=list(designs['A'][0].shape),B_shape=list(designs['B'][0].shape),features_only=True,labels_only_as_training_targets=True,no_RUL_stage_or_engine_ID_features=True,prototype_assigned_without_RUL=True,standardization='frozen train cluster mean/scale',onehot_scaling='none',prediction_clipping=False))
    return table,concentration

def plot_curves(data,contrasts):
    colors={0:'#888888',1:'#0072b2',2:'#d55e00',3:'#009e73'}
    for mode in ['raw','standardized']:
        fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
        for row,(split,z) in enumerate(data.items()):
            for col,sensor in enumerate(C['plot_sensors']):
                ax=axes[row,col];j=SENSORS.index(sensor)
                for k in range(4):
                    take=(z['label'].ravel()==C['plot_RUL'])&(z['pid']==k)
                    if not take.any():continue
                    v=z['data'][take,:,j].astype(float)
                    if mode=='standardized':v=(v-v.mean(1)[:,None])/np.maximum(v.std(1)[:,None],1e-8)
                    low,med,high=np.quantile(v,[.25,.5,.75],axis=0);t=np.arange(1,49)
                    ax.plot(t,med,color=colors[k],label=f'P{k}, {len(v)} engines')
                    ax.fill_between(t,low,high,color=colors[k],alpha=.12)
                for mark in [12.5,24.5,36.5]:ax.axvline(mark,color='gray',ls=':',lw=.6)
                ax.set_title(f'{split} | {sensor} | actual RUL={C["plot_RUL"]}')
                ax.set_ylabel('sensor level' if mode=='raw' else 'within-window z');ax.legend(fontsize=7)
                if row==1:ax.set_xlabel('window cycle')
        fig.suptitle('All engines at fixed RUL; median and 25-75% band; no selected examples',fontsize=11)
        fig.tight_layout();fig.savefig(OUT/f'curves_RUL45_{mode}.png',dpi=170);fig.savefig(OUT/f'curves_RUL45_{mode}.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    pairs=[(1,2),(1,3),(2,3)]
    for ax,f in zip(axes,SHAPE):
        for i,(a,b) in enumerate(pairs):
            for split,shift,color in [('train',-.12,'#0072b2'),('validation',.12,'#d55e00')]:
                t=contrasts[(contrasts.prototype_a==a)&(contrasts.prototype_b==b)&(contrasts.feature==f)&(contrasts.split==split)].iloc[0]
                if np.isfinite(t.delta):ax.errorbar(i+shift,t.delta,yerr=[[max(0,t.delta-t.ci_low)],[max(0,t.ci_high-t.delta)]],fmt='o',color=color,label=split if i==0 else None)
        ax.axhline(0,color='gray',ls='--');ax.set_xticks(range(3),['P2-P1','P3-P1','P3-P2']);ax.set_title(f,fontsize=10);ax.legend()
    fig.suptitle('Exact-RUL matched shape contrasts; engine-level descriptive 95% intervals',fontsize=10);fig.tight_layout();fig.savefig(OUT/'matched_shape_contrasts.png',dpi=180);plt.close(fig)

def report(counts,contrasts,repeated,probe_table,concentration):
    positive=[r for r in repeated if r['repeated_direction_with_CI']]
    primary=probe_table[probe_table.bin.isin(['all','31-60'])]
    both_probe=bool((primary.delta<0).all());few=any(r['few_engine_driven'] for r in concentration if r['bin'] in ['all','31-60'])
    if positive and both_probe and not few:decision='两项检查都有正向证据（限于预设形态指标和固定岭回归探针）'
    elif positive:decision='区分了部分同寿命信号形态，但总体与31–60同时的预测价值未得到支持'
    elif both_probe:decision='简单预测探针有改善，但同RUL形态证据不足，不能判为两项检查均支持'
    else:decision='当前编号与寿命明显关联，同RUL可复核形态/预测价值证据不足；不能把关联直接判成只编码RUL'
    dump(OUT/'decision.json',dict(decision=decision,repeated_shape_contrasts=len(positive),probe_improves_both=both_probe,few_engine_driven=few,limits='exploratory3features/6pairs,no multiplicity correction; same frozen split; no claim of causal degradation or diffusion benefit'))
    lines=['# FD003四类原型：冻结聚类的同RUL诊断与CPU岭回归探针','',f'[Observed] 决策：{decision}。','',
      '[Code] 实际聚类输入是14传感器的均值、标准差(ddof0)、末12周期均值减首12周期均值，共42维。冻结训练StandardScaler与4个KMeans中心分配训练/验证真实窗口ID，不输入RUL。RUL只用于父实验的生成抽样频率，本探针完全不使用该频率。训练ID逐个重算与已有ID一致。',
      '', '[Observed] 固定70台训练/30台验证，14886/5134窗口；验证原NPZ没有周期/窗口索引，本次从train_FD003原文件和冻结MinMax重建并逐窗口逐标签验证，详见window_indices.csv。没有读原测试集，没有聚类/扩散拟合或生成；仅2次CPU Ridge(alpha1,SVD)拟合。','',
      '## 表1：RUL区间与原型（窗口/发动机列均为训练 / 验证）','',
      '| RUL区间 | 原型 | 窗口数 | 发动机数 | 同RUL下可复核的曲线差异 |','|---|---|---|---|---|']
    for name,lo,hi in C['bins']:
        if name=='all':continue
        for k in range(4):
            a=counts[(counts.split=='train')&(counts.bin==name)&(counts.prototype==k)].iloc[0];b=counts[(counts.split=='validation')&(counts.bin==name)&(counts.prototype==k)].iloc[0]
            relevant=[r for r in positive if k in [r['prototype_a'],r['prototype_b']]]
            if b.windows==0:note='无验证窗口，证据不足'
            elif b.engines<8:note='少于8台验证发动机，证据不足'
            elif name=='31-60' and relevant:note='；'.join(f"P{r['prototype_b']}−P{r['prototype_a']}:{r['feature']}同方向" for r in relevant)
            elif name in ['31-40','41-50','51-60']:note='仅作支持量检查；效应按31–60内共同RUL估计，未单独检验本子段'
            elif name=='31-60':note='预设非聚类指标未获跨训练/验证同方向区间证据'
            else:note='本轮曲线比较限31–60；125为截断标签单列' if name=='125' else '本轮未做此段同RUL曲线检验'
            lines.append(f'| {name} | {k} | {a.windows} / {b.windows} | {a.engines} / {b.engines} | {note} |')
    lines+=['','占比与全部独立发动机ID见counts_by_RUL_prototype.csv；包括31–60整体及其三个互相不重叠子段，主段和子段不可相加。不同原型的发动机计数允许重叠。125不与91–124混算。','',
      '## 同RUL曲线比较','',
      '[Code] 预设三指标只用中间季度变化：中间变化能量占比、最大变化是否在中间、传感器最大变化发生段的同步率。它们未直接输入聚类，但仍可与聚类统计量相关，不是“新信息/故障机制”的证明。每整数RUL每侧至少3台发动机，严格同RUL不同发动机，无窗口重复匹配；未放宽到±5。按相同RUL分层求差再等权平均；发动机层共享指数随机权重bootstrap2000次，不能把相邻窗口当独立样本。所有比较及区间均为探索性、未多重校正。','',
      '| 原型对 | 训练匹配窗对/发动机A,B/RUL数 | 验证匹配窗对/发动机A,B/RUL数 | 支持是否充分 |','|---|---|---|---|']
    for a,b in combinations(range(4),2):
        ss=[contrasts[(contrasts.split==s)&(contrasts.prototype_a==a)&(contrasts.prototype_b==b)].iloc[0] for s in ['train','validation']]
        describe=lambda t:f'{int(t.pairs)}/{int(t.engines_a)},{int(t.engines_b)}/{int(t.common_RUL)}'
        lines.append(f'| P{a},P{b} | {describe(ss[0])} | {describe(ss[1])} | {bool(ss[0].sufficient and ss[1].sufficient)} |')
    lines+=['','匹配后的两侧RUL分布完全相同，逐RUL数量和实际窗口/发动机列表见matched_RUL_distributions.csv、matched_windows.csv；未在无重叠寿命区域外推。','',
      '| 原型差 | 指标 | 训练差值[描述性95%区间] | 验证差值[描述性95%区间] | 方向复核 |','|---|---|---|---|---|']
    for r in repeated:
        if not r['support_sufficient']:continue
        ss=[contrasts[(contrasts.split==s)&(contrasts.prototype_a==r['prototype_a'])&(contrasts.prototype_b==r['prototype_b'])&(contrasts.feature==r['feature'])].iloc[0] for s in ['train','validation']]
        fmt=lambda t:f'{t.delta:+.4f} [{t.ci_low:+.4f},{t.ci_high:+.4f}]'
        lines.append(f"| P{r['prototype_b']}−P{r['prototype_a']} | {r['feature']} | {fmt(ss[0])} | {fmt(ss[1])} | {r['repeated_direction_with_CI']} |")
    lines+=['','![非聚类曲线指标](../../../results/diagnostics/'+OUT.name+'/matched_shape_contrasts.png)','',
      '![真实RUL45归一化窗口分位曲线](../../../results/diagnostics/'+OUT.name+'/curves_RUL45_standardized.png)','',
      '图固定RUL45、s2/s11/s20，全体可用发动机，25–75%区间及中位数；每窗口去均值/除标准差，图例给发动机数。无P0时不画；单发动机曲线不作群体证据。原始水平图curves_RUL45_raw.png只供上下文，不能将聚类已用的均值差当独立形态证据。','',
      '## 表2：未见发动机上的预测探针','',
      '[Code] A=冻结标准化42统计特征；B=相同42特征+4列未再标准化onehot。两者仅训练一次alpha1岭回归，所有训练RUL含125均参与，不输入RUL/寿命分段/engineID/原型概率，无预测裁剪。相同正则并不消除新增非线性分区基函数的模型表达差异。','',
      '| 评价范围 | A组RMSE | B组RMSE | B−A | 改善发动机数 |','|---|---|---|---|---|']
    for r in probe_table.itertuples():lines.append(f'| {r.bin} | {r.A_RMSE:.4f} | {r.B_RMSE:.4f} | {r.delta:+.4f} | {r.improved_engines}/{r.n_engines} |')
    lines+=['','主口径为每发动机先平均窗口平方误差、再发动机等权平均后开方；不是窗口加权或平均预测的集成结果。预测完成后才按真实RUL分段。全范围及31–60的发动机贡献：','']
    for r in concentration:
        if r['bin'] not in ['all','31-60']:continue
        t=probe_table[probe_table.bin==r['bin']].iloc[0]
        lines.append(f"- {r['bin']}: B−A描述性95%区间[{t.delta_ci_low:+.4f},{t.delta_ci_high:+.4f}]；改善{r['improved_engines']}/30台；正向MSE降幅中前三台{r['top3_engine_ids']}占{r['top3_positive_gain_share']:.1%}，去掉这三台后的RMSE差{r['delta_RMSE_excluding_best3']:+.4f}；极少数拉动标志={r['few_engine_driven']}。正向贡献比例的分母不含退步者，不能当净收益比例。")
    lines+=['','原型是原始统计特征的确定性分类，不增加传感器信息；若B改善，只支持这种非线性分区让固定线性预测器更易利用已有信息。单一alpha/固定模型/既有内部划分、窗口高度重叠均限制推广；置信区间只按发动机描述，不增加模型拟合次数。','',
      '**决策：'+decision+'。** 原型0在31–60无支持，稀少原型/子段明确不足；不得外推为所有原型、故障类型或因果退化机制。无论探针是否改善，已完成的扩散生成负结果保持，不证明扩散模型用好了原型，也不证明生成数据用途改善。','',
      '当前诊断→论文无该探针/形态目标→论文数值差距不适用→跨发动机方向与不确定性见表→严格复现及已有生成实验结论不变。固定预算2个CPU回归已结束，无调参或新增实验。','',
      '复现（工作区根目录）：','```bash','metaindux-ts-reproduction/.venv/bin/python metaindux-ts-reproduction/analysis/diagnostics/diagnose_fd003_prototypes.py',
      '# 既有产物不覆盖；完全相同配置重跑到新目录：','metaindux-ts-reproduction/.venv/bin/python metaindux-ts-reproduction/analysis/diagnostics/diagnose_fd003_prototypes.py --suffix replay1','```','',
      f'配置：{CONFIG.relative_to(ROOT)}；协议：docs/operations/FD003_PROTOTYPE_PROBE.md；快照：{RUN.relative_to(ROOT)}/manifest.json；表/预测/模型/图：{OUT.relative_to(ROOT)}/。']
    path=ROOT/'docs/conclusions/final'/('fd003_prototype_probe_20261103_report'+('' if RUN.name==C['run_id'] else '_'+RUN.name.split(C['run_id'],1)[1].strip('-'))+'.md')
    path.write_text('\n'.join(lines)+'\n')
    return path,decision,positive

def main(suffix):
    global RUN,OUT
    if suffix and (not suffix.replace('_','').replace('-','').isalnum()):raise ValueError('Use simple suffix')
    rid=C['run_id']+('-'+suffix if suffix else '')
    RUN=ROOT/'runs'/rid;OUT=ROOT/'results/diagnostics'/('fd003_prototype_probe_20261103'+('_'+suffix if suffix else ''))
    if (RUN/'manifest.json').exists():raise RuntimeError('Preserve existing artifacts; use --suffix for exact replay')
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),CONFIG,ROOT/'docs/operations/FD003_PROTOTYPE_PROBE.md',ROOT/'upstream/prototype_guidance.py',ROOT/'scripts/run_prototype_guidance_fd003.py',ROOT/'upstream/data/train_FD003.txt',PARENT/'config.json',PARENT/'train.npz',PARENT/'validation.npz',PARENT/'prototypes.npz',PARENT/'normalization.npz',PARENT/'plan.npz',PARENT/'manifest.json']
    manifest=dict(run_id=rid,status='running',started_at=now(),scope=C['scope'],role='supporting',scope_contract=C['scope_contract'],authorization='Explicit user two CPU ridge fits and frozen FD003 prototype diagnosis',config=C,command=[sys.executable,*sys.argv],git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__,platform=platform.platform(),device='CPU'),ridge_fits_budget=2)
    dump(RUN/'manifest.json',manifest)
    (RUN/'source').mkdir(exist_ok=True)
    for p in sources:
        if p.suffix in ['.py','.json','.md']:shutil.copy2(p,RUN/'source'/p.name)
    (RUN/'workspace.diff').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
    (RUN/'git_status.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=ROOT))
    ledger(f'\n## {manifest["started_at"]} {rid} started\n\n- Diagnostic/supporting,scope-20261103-fd003-prototype-probe-v1, explicit user frozen-cluster diagnosis+exactly2CPU Ridge fits alpha1. No generator training/generation/reclustering/test access. Same70/30engine split. Preset3noncluster shape features, exactRULmatching and engine bootstrap; overall/31-60 probe and125separate. Snapshot runs/{rid}/manifest.json; metrics pending, strict conclusions unchanged.\n')
    try:
        parent=json.loads((PARENT/'manifest.json').read_text())
        for p,h in parent['hashes'].items():assert sha(ROOT/p)==h,p
        assert parent['status']=='complete'
        stats=dict(np.load(PARENT/'prototypes.npz'));norm=dict(np.load(PARENT/'normalization.npz'))
        data={s:dict(np.load(PARENT/(s+'.npz'))) for s in ['train','validation']}
        assert not set(data['train']['engine_id']) & set(data['validation']['engine_id'])
        assert np.array_equal(np.unique(data['train']['engine_id']),stats['fit_engine_ids'])
        assert np.array_equal(stats['fit_engine_ids'],norm['fit_engine_ids'])
        assert np.array_equal(np.unique(data['validation']['engine_id']),json.loads((PARENT/'config.json').read_text())['validation_engines'])
        reconstruct_indices(data,norm).to_csv(OUT/'window_indices.csv',index=False)
        assignments=[]
        for split,z in data.items():
            z['f'],z['z'],z['pid']=assign(z['data'],stats);z['shape']=shape_features(z['data'])
            if split=='train':assert np.array_equal(z['pid'],stats['prototype_id'])
            for i,p in enumerate(z['pid']):assignments.append(dict(split=split,window_index=i,engine_id=int(z['engine_id'][i]),prototype=int(p)))
        pd.DataFrame(assignments).to_csv(OUT/'sensor_only_assignments.csv',index=False)
        dump(OUT/'implementation_audit.json',dict(parent_inputs_verified=True,train_IDs_exactly_reproduced=True,validation_assignment_uses_RUL=False,assignment_rule='argmin squared distance of42sensorfeatures standardized by frozen train mean/scale to frozen4centers',RUL_used_only_for_parent_generation_frequency=True,probe_uses_generation_frequencies=False,validation_original_missing_cycle_index=True,indices_reconstructed_exact=True,train_engines=sorted(map(int,np.unique(data['train']['engine_id']))),validation_engines=sorted(map(int,np.unique(data['validation']['engine_id']))),feature_columns=SENSORS,feature_blocks=['mean','std_ddof0','last_quarter_mean_minus_first_quarter_mean'],normalization_fitted_on_training_engines=True))
        counts=count_tables(data);contrasts,repeated=compare_shapes(data)
        probe_table,concentration=probe(data);plot_curves(data,contrasts)
        path,decision,positive=report(counts,contrasts,repeated,probe_table,concentration)
        # Verify saved model predictions and engine-balanced metric calculations.
        for name in ['A','B']:
            coef=np.load(OUT/f'ridge_{name}.npz');pred=np.load(OUT/'probe_predictions.npz')[name]
            design=data['validation']['z'] if name=='A' else np.concatenate([data['validation']['z'],np.eye(4)[data['validation']['pid']]],1)
            assert np.allclose(design@coef['coef']+coef['intercept'],pred,atol=1e-10)
        for p,h in manifest['input_hashes'].items():assert sha(ROOT/p)==h,p
        manifest.update(status='complete',finished_at=now(),ridge_fits_completed=2,decision=decision,report=str(path.relative_to(ROOT)),output_hashes={str(p.relative_to(ROOT)):sha(p) for p in [*OUT.iterdir(),path] if p.is_file()})
        dump(RUN/'manifest.json',manifest)
        ledger(f'\n## {manifest["finished_at"]} {rid} complete\n\n- [Observed] Diagnostic CPU-only completed, exactly2Ridge alpha1fits; parent inputs verified, sensor-only assignment trainIDs exact, validation indices reconstructed, no generated data/test read. Decision: {decision}. Shape replicated contrasts={len(positive)} (exploratory3features/pair, engine-level intervals); full probe table results/diagnostics/{OUT.name}/probe_summary.csv. Report {path.relative_to(ROOT)}; snapshot runs/{rid}/manifest.json. Current diagnostic→no paper target→gapNA→fixed split/engine-level uncertainty→strict and diffusion negative conclusions unchanged. Budget closed; no further runs/tuning.\n')
        print(probe_table.to_string(index=False));print('REPEATED SHAPE',json.dumps(positive,ensure_ascii=False));print('DECISION',decision);print('REPORT',path)
    except Exception as e:
        manifest.update(status='failed',finished_at=now(),error=repr(e));dump(RUN/'manifest.json',manifest)
        ledger(f'\n## {now()} {rid} failed\n\n- [Observed] Error {repr(e)}; preserve logs and partial artifacts, no automatic scientific changes.\n')
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suffix',default='');args=p.parse_args()
    with threadpool_limits(limits=2):main(args.suffix)
