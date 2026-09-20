import numpy as np
import os
import json
from pathlib import Path
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from TrainCondition import train, sample, collect_frequency_thresholds, UNet1D_fre
import sys
from data.CMAPSSDataset import CMAPSSDataset
import wandb
from utils import wandb_record,torch_seed
from eva_regressor import predictive_score_metrics
from data_process import load_train_data,load_test_data,load_test_data_rul,load_train_data_rul
from measure_score.Utils.discriminative_metric import discriminative_score_metrics
from measure_score.Utils.context_fid import Context_FID
from measure_score.Utils.cross_correlation import CrossCorrelLoss

from args import args
import torch

os.environ["WANDB_MODE"] = "offline"
rmse_list,score_list,acc_list, FID_list, CorrelLoss_list, mae_list = [],[],[],[],[],[]


if __name__ == '__main__':
    if len(sys.argv)==1:
        print('-------no prompt--------')
        args.epoch = 50
        args.dataset = 'FD001'
        args.lr = 2e-3
        args.state = 'sample' # all,train,sample,eval
        args.model_name = 'DiffUnet_fre' 
        args.T = 1000
        args.window_size = 48
        args.w = 0
        args.input_size = 14
    wandb.init(project="MetaIndux-TS", tags=['all'], config=args )
    train_loop = 1
    torch_seed(args.seed)
    if args.model_path == './weights/temp.pth':
        args.model_path = 'weights/' + args.model_name + '_' + args.dataset + '_' + str(args.window_size) + '.pth'
    if args.syndata_path == './weights/syn_data/temp.npy':
        args.syndata_path = './weights/syn_data/syn_' + args.dataset + '_' + args.model_name + '_' + str(args.window_size) + args.sample_type + '.npz'

    datasets = CMAPSSDataset(fd_number=args.dataset, sequence_length=args.window_size, deleted_engine=[1000])
    train_data = datasets.get_train_data()
    train_data,train_label = datasets.get_feature_slice(train_data), datasets.get_label_slice(train_data)
    
    test_data = datasets.get_test_data()
    test_data,test_label = datasets.get_last_data_slice(test_data)
    
    train_data,train_label = train_data[0:len(train_data)], train_label[0:len(train_label)]
    print("train_data.shape:",train_data.shape,"      test_data.shape:",test_data.shape)
    
    
    if args.state == "train" or args.state == "all":
        train(args,train_data,train_label)
        sample(args,train_label)
    elif args.state == "sample":
        sample(args,train_label)
    generation_path = (Path(args.metrics_path).with_suffix('.generation.json')
                       if args.metrics_path else None)
    if args.state in ('all', 'train', 'sample') and generation_path:
        # Commit generation metadata BEFORE any post-hoc evaluator can fail.
        generation_path.parent.mkdir(parents=True, exist_ok=True)
        generation_path.write_text(json.dumps({
            'best_diffusion_loss': float(getattr(args, 'best_diffusion_loss', float('nan'))),
            'best_epoch': int(getattr(args, 'best_epoch', -1)),
            'training_seconds': float(getattr(args, 'training_seconds', float('nan'))),
            'sampling_seconds': float(getattr(args, 'sampling_seconds', float('nan'))),
            'frequency_thresholds': getattr(args, 'frequency_thresholds', []),
        }, indent=2), encoding='utf-8')
    if args.state == 'eval':
        if generation_path and generation_path.is_file():
            for key, value in json.loads(generation_path.read_text(encoding='utf-8')).items():
                setattr(args, key, value)
        elif args.loss_history_path and Path(args.loss_history_path).is_file():
            # Legacy failed runs have no stage record. Recover loss/epoch only;
            # do not invent training or sampling timings.
            history = np.loadtxt(args.loss_history_path, delimiter=',', skiprows=1, ndmin=2)
            eligible = history[history[:, 0] > 5]
            best = eligible[np.argmin(eligible[:, 1])]
            args.best_epoch, args.best_diffusion_loss = int(best[0]), float(best[1])
        if args.model_name == 'DiffUnet_fre':
            model = UNet1D_fre(
                dim=32, dim_mults=(1,2), cond_drop_prob=0.2,
                channels=args.input_size, length=args.window_size,
                frequency_mask_mode=args.frequency_mask_mode,
                frequency_threshold_init=args.frequency_threshold_init,
                frequency_mask_temperature=args.frequency_mask_temperature,
            )
            model.load_state_dict(torch.load(args.model_path, map_location='cpu', weights_only=True))
            args.frequency_thresholds = collect_frequency_thresholds(model)
            del model
    if args.state == "eval" or args.state == "all" or args.state == "sample":
        syn_dataset = np.load(args.syndata_path)
        syn_data = syn_dataset['data']
        if args.state == 'eval':
            if (syn_data.shape != tuple(train_data.shape) or
                    not np.array_equal(syn_dataset['label'], train_label.cpu().numpy())):
                raise ValueError('reused synthetic data does not match current training windows/labels')
        original_data_test = {'data':test_data,'label':test_label}
        original_data_train = {'data':train_data,'label':train_label}
        concat_data = {}
        random_indices = np.random.choice(train_data.shape[0], size=len(train_data) // 10 , replace=False)

        for i in range(train_loop):
            rmse,mae, score= predictive_score_metrics(args, original_data_test, syn_dataset, eval_seed=args.seed)
            discriminative_score, fake_accuracy, real_accuracy = discriminative_score_metrics(train_data.cpu().numpy() , syn_data, eval_seed=args.seed)
            Context_FID_score = Context_FID(train_data.cpu().numpy(), syn_data) # Context_FID分数计算
            print("Context_FID_score:", Context_FID_score)

            loss_function = CrossCorrelLoss(train_data.cpu().numpy(), name=args.dataset) # Correlation分数计算
            CrossCorrel_Loss = loss_function(torch.tensor(syn_data))
            rmse_list.append(rmse); score_list.append(score); acc_list.append(discriminative_score)
            mae_list.append(mae); FID_list.append(Context_FID_score); CorrelLoss_list.append(CrossCorrel_Loss)

        wandb_record(rmse_list,mae_list,score_list, acc_list,FID_list,CorrelLoss_list)
        if args.metrics_path:
            metrics_path = Path(args.metrics_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics = {
                'dataset': args.dataset,
                'window_size': args.window_size,
                'seed': args.seed,
                'builtin_evaluator_seed': args.seed,
                'rmse': float(rmse),
                'mae': float(mae),
                'rul_score': float(score),
                'discriminative_score': float(discriminative_score),
                'context_fid': float(Context_FID_score),
                'cross_correlation_loss': float(CrossCorrel_Loss.detach().cpu()),
                'best_diffusion_loss': float(getattr(args, 'best_diffusion_loss', float('nan'))),
                'best_epoch': int(getattr(args, 'best_epoch', -1)),
                'training_seconds': float(getattr(args, 'training_seconds', float('nan'))),
                'sampling_seconds': float(getattr(args, 'sampling_seconds', float('nan'))),
                'frequency_mask_mode': args.frequency_mask_mode,
                'frequency_threshold_init': float(args.frequency_threshold_init),
                'frequency_mask_temperature': float(args.frequency_mask_temperature),
                'learned_frequency_thresholds': getattr(args, 'frequency_thresholds', []),
            }
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        wandb.finish()
