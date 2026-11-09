def worker(arm):
 import numpy as np
 import torch
 from torch.utils._python_dispatch import TorchDispatchMode
 m=check();target=OUT/arm
 sys.path.insert(0,str(ROOT/'upstream'))
 import TrainCondition as training
 import GaussianDiffusion as diffusion
 import DiffusionFreeGuidence.Unet1D_fre as unet
 old=module(ROOT/'scripts/run_fd002_first_epoch_trace.py','sealed_first_epoch_worker')
 old.OUT=OUT
 ctx=dict(stack=[],events=[],model=None,forward_count=0)
 original_factory=training.UNet1D_fre;original_loss=diffusion.GaussianDiffusion1D_cls_free.p_losses
 canonical=torch.load(PRIOR_OUT/'native_public/tapes/0001.pth',map_location='cpu',weights_only=False)
 canonical_initial=torch.load(PRIOR_OUT/'native_public/initial_state.pth',map_location='cpu',weights_only=True)
 def tensors(x):
  if torch.is_tensor(x):return [x]
  if isinstance(x,(list,tuple)):return sum([tensors(i) for i in x],[])
  if isinstance(x,dict):return sum([tensors(i) for i in x.values()],[])
  return []
 def tensor_desc(t,store=False):
  with torch._C._DisableTorchDispatch(),torch.no_grad():
   cp=t.detach().resolve_conj().resolve_neg().cpu().contiguous()
   h=hashlib.sha256(cp.numpy().tobytes()).hexdigest()
   tag=hashlib.sha256((str(cp.dtype)+str(tuple(cp.shape))+h).encode()).hexdigest()
   if store:
    p=OUT/'tensors'/f'{tag}.pth'
    if not p.exists():torch.save(cp,p)
   return dict(kind='tensor',sha256=h,artifact=tag if store else None,shape=list(t.shape),stride=list(t.stride()),
    storage_offset=int(t.storage_offset()),data_alignment=int(t.data_ptr()%256),dtype=str(t.dtype),device=str(t.device),
    contiguous=t.is_contiguous(),requires_grad=t.requires_grad)
 def describe(x,store=False):
  if torch.is_tensor(x):return tensor_desc(x,store)
  if isinstance(x,tuple):return dict(kind='tuple',items=[describe(v,store) for v in x])
  if isinstance(x,list):return dict(kind='list',items=[describe(v,store) for v in x])
  if isinstance(x,dict):return dict(kind='dict',items={str(k):describe(v,store) for k,v in x.items()})
  if x is None or isinstance(x,(str,bool,int,float)):return dict(kind='scalar',value=x)
  return dict(kind='literal',type=type(x).__name__,value=str(x))
 def origin():
  frame=sys._getframe(2)
  while frame:
   p=Path(frame.f_code.co_filename)
   if str(p).startswith(str(ROOT/'upstream')):return dict(path=str(p.relative_to(ROOT)),line=frame.f_lineno,function=frame.f_code.co_name)
   if str(p).startswith(str(ROOT/'scripts')) and p!=Path(__file__):return None
   frame=frame.f_back
  return None
 class Trace(TorchDispatchMode):
  def __torch_dispatch__(self,func,types,args=(),kwargs=None):
   kwargs={} if kwargs is None else kwargs
   source=origin()
   if source is None:return func(*args,**kwargs)
   i=len(ctx['events']);store=str(func)=='aten.convolution.default'
   before=describe(dict(args=args,kwargs=kwargs),store)
   if str(func)=='aten.convolution.default':
    label=f'DIAG_CONV_{i:04d}::{ctx["stack"][-1]}'
    with torch.profiler.record_function(label):output=func(*args,**kwargs)
   else:output=func(*args,**kwargs)
   after=describe(output,store)
   ctx['events'].append(dict(index=i,operator=str(func),module=ctx['stack'][-1] if ctx['stack'] else '<outside_model>',source=source,inputs=before,output=after))
   return output
 def factory(*a,**kw):
  model=original_factory(*a,**kw);ctx['model']=model
  assert all(torch.equal(t,canonical_initial[k]) for k,t in model.state_dict().items())
  for name,mod in model.named_modules():
   label=name or '<model>'
   def pre(mod,args,n=label):ctx['stack'].append(n)
   def post(mod,args,out,n=label):assert ctx['stack'].pop()==n
   mod.register_forward_pre_hook(pre);mod.register_forward_hook(post)
  return model
 training.UNet1D_fre=factory
 def first_loss(trainer,x_start,t,*,classes,noise=None):
  ctx['forward_count']+=1;assert ctx['forward_count']==1
  # Read the sealed worker's observer state, without changing its draws/hooks.
  current_method=diffusion.GaussianDiffusion1D_cls_free.p_losses
  closure=inspect.getclosurevars(current_method).nonlocals
  innerctx=closure['ctx']
  assert innerctx['updates']==0
  ctx['pre_forward_rng_sha256']=hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
  import pickle
  torch.backends.cudnn.benchmark=KERNEL_MODE=='native'
  torch.backends.cudnn.deterministic=KERNEL_MODE=='fixed'
  assert not torch.are_deterministic_algorithms_enabled()
  before_cpu=torch.get_rng_state().clone();before_cuda=torch.cuda.get_rng_state().clone();before_numpy=np.random.get_state()
  # Warmup chooses the algorithm outside the measured profiler range.
  warm=original_loss(trainer,x_start,t,classes=classes,noise=noise)
  warm_loss=float(warm.item());warm_output_hash=innerctx['current']['model_output_sha256']
  actual_warm=innerctx['tape']
  for key in ['timestep','noise','condition_mask']:assert torch.equal(actual_warm[key],canonical[key])
  del warm
  innerctx['tape']={key:value for key,value in actual_warm.items() if key not in ['noise','condition_mask']}
  torch.set_rng_state(before_cpu);torch.cuda.set_rng_state(before_cuda);np.random.set_state(before_numpy)
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
   with Trace():value=original_loss(trainer,x_start,t,classes=classes,noise=noise)
   torch.cuda.synchronize()
  prof.export_chrome_trace(str(target/'cuda_trace.json'))
  def kernel_desc(event):
   result=[]
   for k in event.kernels:
    if not k.name.startswith(('Memset','Memcpy')):result.append(dict(name=k.name,device=k.device,duration_us=k.duration,owner=event.name))
   for child in event.cpu_children:result.extend(kernel_desc(child))
   return result
  kernel_rows=[]
  for event in prof.events():
   if event.name.startswith('DIAG_CONV_'):
    prefix,module_name=event.name.split('::',1);op_index=int(prefix.split('_')[-1])
    kernels=kernel_desc(event);assert kernels,'No actual CUDA kernel for '+event.name
    kernel_rows.append(dict(index=op_index,module=module_name,kernels=kernels))
  assert kernel_rows and len(kernel_rows)==sum(e['operator']=='aten.convolution.default' for e in ctx['events'])
  dump(target/'actual_conv_kernels.json',kernel_rows)
  dump(target/'profile_summary.json',dict(mode=KERNEL_MODE,warmup_loss=warm_loss,measured_loss=float(value.item()),
   warmup_output_sha256=warm_output_hash,measured_output_sha256=innerctx['current']['model_output_sha256'],
   warmup_measured_exact_output_equal=warm_output_hash==innerctx['current']['model_output_sha256'],
   full_first_batch_forwards=2,convolutions_measured=len(kernel_rows),backward_calls=0,optimizer_updates=0,
   benchmark=torch.backends.cudnn.benchmark,deterministic=torch.backends.cudnn.deterministic,
   global_deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
  actual=innerctx['tape']
  assert np.array_equal(actual['indices'],canonical['indices']) and actual['numpy_draw']==canonical['numpy_draw']
  for k in ['timestep','noise','condition_mask']:assert torch.equal(actual[k],canonical[k]),k
  assert all(torch.equal(v.cpu(),canonical_initial[k]) for k,v in ctx['model'].state_dict().items())
  summary=dict(status='complete',arm=arm,forward_calls=1,backward_calls=0,optimizer_updates=0,
   canonical_initial_and_question_exact=True,loss=float(value.item()),operators=len(ctx['events']),
   pre_forward_rng_sha256=ctx['pre_forward_rng_sha256'],torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),
   backend_flags=dict(benchmark=torch.backends.cudnn.benchmark,deterministic=torch.backends.cudnn.deterministic,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32),
   caveat='Synchronous ATen tracing can alter autotuning/timing; current entry replay, not original historical kernels')
  dump(target/'operators.json',ctx['events']);dump(target/'summary.json',summary)
  raise ForwardComplete()
 diffusion.GaussianDiffusion1D_cls_free.p_losses=first_loss
 def forbidden(*a,**kw):raise RuntimeError('Backward or optimizer step forbidden in first-batch localization')
 torch.optim.Adam.step=forbidden;torch.Tensor.backward=forbidden
 try:old.worker(arm)
 except ForwardComplete:pass
 assert ctx['forward_count']==1 and (target/'summary.json').exists()
