
import torch
import time
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset,random_split
import numpy as np
from pathlib import Path
from DiffusionFreeGuidence.Unet1D import UNet1D
from DiffusionFreeGuidence.Unet1D_fre import UNet1D_fre
from DiffusionFreeGuidence.Diffwave import DiffWave
from DiffusionFreeGuidence.tabddpm import MLPDiffusion
from DiffusionFreeGuidence.Dit import DiT
from DiffusionFreeGuidence.SSSD import SSSDS4Imputer

from DiffusionFreeGuidence.diffwaveimputer import DiffWaveImputer
from DiffusionFreeGuidence.csdi import diff_CSDI

import data.CMAPSSDataset as CMAPSSDataset
import wandb
from GaussianDiffusion import GaussianDiffusion1D_cls_free,GradualWarmupScheduler


def apply_fixed_frequency_threshold(model, threshold):
    """Override thresholds after construction without shifting model RNG use."""
    if threshold < 0:
        return []
    if any(getattr(module, 'mask_mode', None) == 'binary_ste_energy'
           for module in model.modules()):
        raise ValueError('fixed quantile override cannot be used with STE energy thresholds')
    values = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('threshold_param'):
                parameter.fill_(threshold)
                values.append((name, float(parameter.item())))
    if not values:
        raise RuntimeError('frequency threshold requested, but no threshold_param was found')
    print('fixed frequency thresholds:', values)
    return values


def collect_frequency_thresholds(model):
    """Return threshold metadata and any observed forward-mask drop ratios.

    A quantile fraction and a normalized-energy cutoff are not interchangeable.
    The only cross-mode quantity is ``mask_drop_ratio``, measured from actual
    forward masks while sampling.
    """
    values = []
    for name, module in model.named_modules():
        parameter = getattr(module, 'threshold_param', None)
        if parameter is None:
            continue
        raw_value = float(parameter.detach().cpu().item())
        effective = getattr(module, 'effective_threshold', None)
        effective_value = (
            float(effective().detach().cpu().item())
            if callable(effective) else raw_value
        )
        mode = getattr(module, 'mask_mode', 'unknown')
        is_quantile = mode == 'hard_random_quantile'
        item = {
            'module': name,
            'mask_mode': mode,
            'raw': raw_value,
            # Retained for compatibility with completed run JSON files.  Its
            # semantics are now stated explicitly below.
            'effective': effective_value,
            'threshold_parameter_semantics': (
                'quantile_fraction' if is_quantile else 'unconstrained_raw_parameter'),
            'threshold_value': None if is_quantile else effective_value,
            'threshold_value_semantics': (
                'dynamic_normalized_energy_quantile' if is_quantile
                else 'normalized_energy_cutoff'),
            'mask_drop_ratio': None,
            'mask_drop_ratio_semantics': (
                'fraction_of_forward_mask_elements_equal_to_zero'
                if mode != 'soft_learnable_energy'
                else 'mean_forward_attenuation_one_minus_soft_mask'),
            'hard_drop_ratio': None,
            'observation_scope': None,
        }
        statistics = getattr(module, 'mask_statistics', None)
        observed = statistics() if callable(statistics) else None
        if observed:
            item.update(observed)
            item['observation_scope'] = 'all_ddpm_sampling_forwards'
        values.append(item)
    return values


def enable_frequency_mask_statistics(model):
    """Start per-module mask aggregation without changing forward values."""
    for module in model.modules():
        enable = getattr(module, 'enable_mask_statistics', None)
        if callable(enable):
            enable(True)


def merge_frequency_threshold_observations(current, stored):
    """Restore sampling observations when eval reloads an existing artifact."""
    stored_by_module = {item.get('module'): item for item in (stored or [])}
    observation_keys = {
        'mask_drop_ratio', 'mask_drop_ratio_semantics', 'hard_drop_ratio',
        'observed_mask_elements', 'observed_forward_calls',
        'observed_threshold_mean', 'observed_threshold_min',
        'observed_threshold_max', 'observation_scope',
    }
    for item in current:
        previous = stored_by_module.get(item.get('module'), {})
        if previous.get('mask_drop_ratio') is not None:
            item.update({key: previous[key] for key in observation_keys
                         if key in previous})
    return current

def train(args, train_data, train_label):
    device = args.device
    best_loss = 9999
    best_epoch = -1
    loss_history = []
    train_started = time.time()

    train_dataset = TensorDataset(train_data.permute(0,2,1).to(device), train_label.to(device))
    dataloader= DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # model setup
    if args.model_name == 'DiffUnet': 
        net_model = UNet1D(dim = 32, dim_mults = (1, 2), cond_drop_prob = 0.2, channels = args.input_size).to(device) #cmapss  dim = 32 dim_mults = (1, 2, 2) cond_drop_prob = 0.5
    if args.model_name == 'DiffUnet_fre': 
        net_model = UNet1D_fre(
            dim=32, dim_mults=(1,2), cond_drop_prob=0.2,
            channels=args.input_size, length=args.window_size,
            frequency_mask_mode=args.frequency_mask_mode,
            frequency_threshold_init=args.frequency_threshold_init,
            frequency_mask_temperature=args.frequency_mask_temperature,
        ).to(device)
    if args.model_name == 'dit': 
        net_model = DiT(input_size=args.input_size,hidden_size=128, num_heads=1).to(device) 
    if args.model_name == 'DiffWave': 
        net_model = DiffWaveImputer(seq_length=args.window_size)
    if args.model_name == 'SSSD': 
        net_model = SSSDS4Imputer(seq_length=args.window_size, s4_lmax=args.window_size )
    if args.model_name == 'tabddpm':     
        net_model = MLPDiffusion( d_in=14, num_classes=0, is_y_cond=True, window_size=args.window_size, rtdl_params={'d_layers': [128, 256, 256, 128], 'dropout': 0.3}, dim_t = 512)
    if args.model_name == 'csdi': 
        net_model = diff_CSDI(inputdim=14)   
    apply_fixed_frequency_threshold(net_model, args.frequency_threshold)
                 
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net_model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net_model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.RMSprop(net_model.parameters(), lr=args.lr)
    if args.lr_schedule == 'warmup_cosine':
        cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1)
        lr_scheduler = GradualWarmupScheduler(
            optimizer=optimizer,
            multiplier=args.multiplier,
            warm_epoch=args.epoch // 10 + 1,
            after_scheduler=cosineScheduler,
        )
    elif args.lr_schedule == 'constant':
        lr_scheduler = None
    else:
        raise ValueError(f'unknown learning-rate schedule: {args.lr_schedule}')
    print(args.T)
    trainer = GaussianDiffusion1D_cls_free(
        net_model,
        seq_length=args.window_size,
        channels=args.input_size,
        timesteps=args.T,
        objective='pred_noise',
        beta_schedule=args.schedule_name,
    ).to(device)

    # start training
    for e in range(args.epoch):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            loss_list=[]
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device)
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(x_0, classes = labels).sum()
                loss_list.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), args.grad_clip)
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": sum(loss_list)/len(loss_list),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        if lr_scheduler is not None:
            lr_scheduler.step()
        current_loss = sum(loss_list)/len(loss_list)
        wandb.log({"Diffusion_Loss":current_loss})
        current_lr = optimizer.state_dict()['param_groups'][0]["lr"]
        loss_history.append((e, current_loss, current_lr))
        if args.loss_history_path:
            history_path = Path(args.loss_history_path)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                history_path,
                np.asarray(loss_history),
                delimiter=',',
                header='epoch,diffusion_loss,learning_rate',
                comments='',
            )
        if e > 5 and current_loss < best_loss:
            best_loss = current_loss
            best_epoch = e
            torch.save(net_model.state_dict(), args.model_path)
            print(f'*******improve: epoch={best_epoch} loss={best_loss:.8f}********')
    args.training_seconds = time.time() - train_started
    args.best_diffusion_loss = best_loss
    args.best_epoch = best_epoch
    wandb.run.summary['training_seconds'] = args.training_seconds
    wandb.run.summary['best_diffusion_loss'] = best_loss
    wandb.run.summary['best_epoch'] = best_epoch

def sample(args, train_label = None):
    if train_label == None:
        datasets = CMAPSSDataset.CMAPSSDataset(fd_number=args.dataset, sequence_length=args.window_size ,deleted_engine=[1000])
        train_data = datasets.get_train_data()
        train_label = datasets.get_label_slice(train_data)        
    device = args.device
    # load model and evaluate
    with torch.no_grad():
        if args.model_name == 'DiffUnet': 
            net_model = UNet1D(dim = 32, dim_mults = (1, 2), cond_drop_prob = 0.2, channels = args.input_size).to(device) #cmapss  dim = 32 dim_mults = (1, 2, 2) cond_drop_prob = 0.5
        if args.model_name == 'DiffUnet_fre': 
            net_model = UNet1D_fre(
                dim=32, dim_mults=(1,2), cond_drop_prob=0.2,
                channels=args.input_size, length=args.window_size,
                frequency_mask_mode=args.frequency_mask_mode,
                frequency_threshold_init=args.frequency_threshold_init,
                frequency_mask_temperature=args.frequency_mask_temperature,
            ).to(device)
        if args.model_name == 'DiffWave': 
            net_model = DiffWaveImputer(seq_length=args.window_size)
        if args.model_name == 'SSSD': 
            net_model = SSSDS4Imputer(seq_length=args.window_size, s4_lmax=args.window_size )
        if args.model_name == 'tabddpm':     
            net_model = MLPDiffusion( d_in=14, num_classes=0, is_y_cond=True, window_size=args.window_size, rtdl_params={'d_layers': [128, 256, 256, 128], 'dropout': 0.0}, dim_t = 512)
        if args.model_name == 'csdi':     
            net_model = diff_CSDI(inputdim=14)
        if args.model_name == 'dit': 
            net_model = DiT(input_size=args.input_size,hidden_size=128, num_heads=1).to(device)                               
        apply_fixed_frequency_threshold(net_model, args.frequency_threshold)
        ckpt = torch.load(args.model_path)
        net_model.load_state_dict(ckpt)
        args.frequency_thresholds = collect_frequency_thresholds(net_model)
        print('loaded frequency thresholds:', args.frequency_thresholds)
        print("model load weight done.")
        net_model.eval()
        enable_frequency_mask_statistics(net_model)
        diffusion = GaussianDiffusion1D_cls_free(
            net_model,
            seq_length=args.window_size,
            channels=args.input_size,
            timesteps=args.T,
            objective='pred_noise',
            beta_schedule=args.schedule_name,
        ).to(device)
        # Sample from standard normal in bounded batches. The upstream code
        # sampled every training condition at once, which exhausts an 8 GB GPU.
        start_time = time.time()
        sample_batches = []
        for start in tqdm(range(0, len(train_label), args.batch_size), desc='sampling'):
            classes = train_label[start:start + args.batch_size].to(device)
            batch = diffusion.sample(classes=classes).permute(0, 2, 1)
            sample_batches.append(batch.cpu())
        sampledata = torch.cat(sample_batches, dim=0)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"经过的时间: {elapsed_time} 秒")
        args.sampling_seconds = elapsed_time
        args.frequency_thresholds = collect_frequency_thresholds(net_model)
        print('observed frequency mask statistics:', args.frequency_thresholds)
        wandb.run.summary['sampling_seconds'] = elapsed_time
        
        sampledata = sampledata.numpy()
        np.savez(args.syndata_path,data=sampledata, label =  train_label.cpu().numpy())      

        #np.savez(args.syndata_path,data=sampledata.cpu().numpy(), label =  train_label.cpu().numpy())
    return sampledata
