"""Read-only model diagnostic; generated reports do not alter training artifacts."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
TIME = ROOT / 'artifacts/temporal_audited/FD003/stride50_B_seed42'
FUSION = ROOT / 'artifacts/tmaf_audited/FD003/global_broadcast_stride50_B_seed42'


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for this diagnostic')
    tr = json.loads((TIME / 'result.json').read_text())
    fr = json.loads((FUSION / 'result.json').read_text())
    digest = hashlib.sha256((TIME / 'best.pt').read_bytes()).hexdigest()
    assert digest == fr['temporal_checkpoint_sha256']
    assert tr['data_manifest_sha256'] == fr['data_manifest_sha256']
    manifest = ROOT / 'artifacts/data_window40_stride50/split_seed42/FD003/manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == tr['data_manifest_sha256']
    for key in ['seed', 'split_seed', 'epochs', 'batch_size', 'learning_rate']:
        assert tr['train_config'][key] == fr[key]
    assert tr['target_divisor'] == fr['target_divisor']
    a = pd.read_csv(TIME / 'test_predictions.csv')
    b = pd.read_csv(FUSION / 'test_predictions.csv')
    assert len(a) == len(b) == 100
    assert a.unit.is_unique and b.unit.is_unique
    merged = a.merge(b, on='unit', suffixes=('_time', '_tmaf'), validate='one_to_one').sort_values('unit')
    assert (merged.cycle_time == merged.cycle_tmaf).all()
    assert (merged.target_time == merged.target_tmaf).all()
    y = torch.tensor(merged.target_time.to_numpy(), dtype=torch.float64, device='cuda')
    t = torch.tensor(merged.prediction_clipped_time.to_numpy(), dtype=torch.float64, device='cuda')
    f = torch.tensor(merged.prediction_clipped_tmaf.to_numpy(), dtype=torch.float64, device='cuda')
    et, ef = t-y, f-y

    def stats(pred, mask):
        e = (pred-y)[mask]
        return dict(n=int(mask.sum()), rmse=float(e.square().mean().sqrt()), mae=float(e.abs().mean()),
                    bias=float(e.mean()), overestimate_fraction=float((e>0).double().mean()),
                    score=float(torch.where(e<0, torch.expm1(-e/13), torch.expm1(e/10)).sum()),
                    prediction_std=float(pred[mask].std(correction=0)))

    def paired(mask):
        return dict(temporal=stats(t, mask), tmaf=stats(f, mask),
                    improved=int(((ef.abs()<et.abs()) & mask).sum()),
                    worsened=int(((ef.abs()>et.abs()) & mask).sum()),
                    mean_prediction_shift=float((f-t)[mask].mean()))

    all_mask = torch.ones_like(y, dtype=torch.bool)
    top = torch.argsort(ef.abs(), descending=True)[:20]
    high = torch.zeros_like(all_mask); high[top] = True
    corr = lambda x,z: float(torch.corrcoef(torch.stack([x,z]))[0,1])
    rows = []
    for i, row in enumerate(merged.itertuples(index=False)):
        rows.append(dict(unit=int(row.unit), target=float(y[i]), temporal=float(t[i]), tmaf=float(f[i]),
                         temporal_error=float(et[i]), tmaf_error=float(ef[i]),
                         absolute_error_change=float(ef[i].abs()-et[i].abs())))
    report = dict(device=torch.cuda.get_device_name(), protocol_checks='passed',
                  temporal_checkpoint_sha256=digest, data_manifest_sha256=tr['data_manifest_sha256'],
                  interpretation_limit='TMAF jointly finetunes the temporal encoder and uses a different regression head; this is not an isolated causal test of the fusion layer.',
                  overall=paired(all_mask), tmaf_high_error_20=paired(high),
                  error_correlation=corr(et,ef), prediction_correlation=corr(t,f),
                  both_overestimate=int(((et>0)&(ef>0)).sum()),
                  temporal_under_tmaf_over=int(((et<0)&(ef>0)).sum()),
                  top20_overlap=int(torch.isin(top,torch.argsort(et.abs(),descending=True)[:20]).sum()),
                  rul_bins={name:paired((y>=lo)&(y<=hi)) for name,lo,hi in [('0-25',0,25),('26-50',26,50),('51-100',51,100),('101-125',101,125)]},
                  worst_tmaf_engines=sorted(rows,key=lambda r:abs(r['tmaf_error']),reverse=True)[:20],
                  largest_worsening=sorted(rows,key=lambda r:r['absolute_error_change'],reverse=True)[:10],
                  largest_improvement=sorted(rows,key=lambda r:r['absolute_error_change'])[:10], engines=rows)
    (ROOT / 'FD003_TEMPORAL_TMAF_COMPARISON.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='engines'},indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
