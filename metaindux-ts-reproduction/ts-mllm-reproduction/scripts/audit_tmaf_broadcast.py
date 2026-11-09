"""Numerically verify whether broadcast global context leaves nontrivial attention."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from ts_mllm.tmaf import TMAFConfig, TemporalMultimodalAttentionFusion
from ts_mllm.model import PatchTransformerConfig

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/fd003_batch_diagnostic/seed42/tmaf_batch32'
CACHE = ROOT / 'artifacts/qwen_audited/FD003/seed42'
DATA = ROOT / 'artifacts/data_window40_stride50/split_seed42/FD003'

def main():
    assert torch.cuda.is_available(), 'CUDA required'
    checkpoint = torch.load(RUN/'best.pt', map_location='cuda', weights_only=False)
    result = json.loads((RUN/'result.json').read_text())
    c = result['model_config']
    cfg = TMAFConfig(temporal=PatchTransformerConfig(**c['temporal']),
                     llm_dim=c['llm_dim'], attention_key_dim=c['attention_key_dim'],
                     attention_output_dim=c['attention_output_dim'], fusion_mlp_units=c['fusion_mlp_units'],
                     dropout=c['dropout'], context_mode='global_broadcast', global_pooling='mean')
    model = TemporalMultimodalAttentionFusion(cfg).cuda().eval()
    model.load_state_dict(checkpoint['model_state'])
    x=torch.tensor(np.load(DATA/'test_x.npy')[:8],device='cuda')
    z=torch.tensor(np.load(CACHE/'test.npy')[:8],device='cuda')
    mask=torch.tensor(np.load(CACHE/'test_mask.npy')[:8],device='cuda')
    fused, attention=model.fuse(x,z,mask)
    n=attention.shape[-1]
    uniform=(attention-torch.full_like(attention,1/n)).abs().max()
    context=(z.float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
    temporal=model.temporal.encode(x)
    v=model.value_projection(context).unsqueeze(1).expand(-1,temporal.shape[1],-1)
    direct=model.fusion_projection(torch.cat([temporal,v],-1))
    direct_difference=(fused-direct).abs().max()
    pred=model(x,z,mask); pred_direct=model.regression_head(direct.mean(1)).squeeze(-1)
    model.zero_grad(set_to_none=True)
    pred.sum().backward()
    grads={name: (None if getattr(model,name).weight.grad is None else float(getattr(model,name).weight.grad.norm()))
           for name in ('query_projection','key_projection','value_projection','fusion_projection')}
    report=dict(device=torch.cuda.get_device_name(),num_keys=n,
                max_attention_minus_uniform=float(uniform),
                max_fused_minus_direct_additive=float(direct_difference),
                max_prediction_minus_direct_additive=float((pred-pred_direct).abs().max()),
                projection_weight_gradient_norm=grads,
                mathematical_identity='For N identical K,V: softmax(QK^T/sqrt(d))=1/N; attention output is V for every temporal query. Thus d(output)/d(Q,K)=0, and mean-pooled fused sequence is a function of mean temporal embedding plus one global value vector.')
    (ROOT/'FD003_TMAF_BROADCAST_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
