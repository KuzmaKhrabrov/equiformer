import argparse
import datetime
import itertools
import pickle
import subprocess
import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader

import os
from logger import FileLogger
from pathlib import Path
from typing import Iterable, Optional

from datasets.pyg.nablaDFT import get_nablaDFT_datasets

import nets
from nets import model_entrypoint

from timm.utils import ModelEmaV2, get_state_dict
from timm.scheduler import create_scheduler
from optim_factory import create_optimizer

from engine import AverageMeter, compute_stats


ModelEma = ModelEmaV2


def get_args_parser():
    parser = argparse.ArgumentParser('Training equivariant networks on nablaDFT wo forces', add_help=False)
    parser.add_argument('--output-dir', type=str, default=None)
    # network architecture
    parser.add_argument('--model-name', type=str, default='graph_attention_transformer_nonlinear_l2_md17')
    parser.add_argument('--input-irreps', type=str, default=None)
    parser.add_argument('--radius', type=float, default=5.0)
    parser.add_argument('--num-basis', type=int, default=128)
    # training hyper-parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument('--model-ema', action='store_true')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.9999, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    # regularization
    parser.add_argument('--drop-path', type=float, default=0.0)
    # optimizer (timm)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-3,
                        help='weight decay (default: 5e-3)')
    # learning rate schedule parameters (timm)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    # logging
    parser.add_argument("--print-freq", type=int, default=100)
    # task and dataset
    parser.add_argument("--data-path", type=str, default='datasets/md17')
    parser.add_argument("--split", type=str, default="train2k")
    parser.add_argument("--train-size", type=int, default=950)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument('--compute-stats', action='store_true', dest='compute_stats')
    parser.set_defaults(compute_stats=False)
    parser.add_argument('--test-interval', type=int, default=10, 
                        help='epoch interval to evaluate on the testing set')
    parser.add_argument('--test-max-iter', type=int, default=1000, 
                        help='max iteration to evaluate on the testing set')
    parser.add_argument('--energy-weight', type=float, default=0.2)
    parser.add_argument('--force-weight', type=float, default=0.8)
    # random
    parser.add_argument("--seed", type=int, default=1)
    # data loader config
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    # evaluation
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true', dest='evaluate')
    parser.add_argument('--feature-type', type=str, default='one_hot')
    parser.set_defaults(evaluate=False)
    return parser


# from https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/modules/loss.py#L7
class L2MAELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)


def main(args):
    
    _log = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    _log.info(args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ''' Dataset '''
    train_dataset, val_dataset, test_dataset = get_nablaDFT_datasets(
        root=os.path.join(args.data_path), split=args.split,
        train_size=args.train_size, val_size=args.val_size, test_size=None, 
        seed=args.seed)

    _log.info('')
    _log.info('Training set size:   {}'.format(len(train_dataset)))
    _log.info('Validation set size: {}'.format(len(val_dataset)))

    # statistics
    y = torch.cat([batch.y for batch in train_dataset], dim=0)
    mean = float(y.mean())
    std = float(y.std())
    _log.info('Training set mean: {}, std: {}\n'.format(mean, std))

    # since dataset needs random 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ''' Network '''
    create_model = model_entrypoint(args.model_name)
    model = create_model(irreps_in=args.input_irreps, 
        radius=args.radius, 
        num_basis=args.num_basis, 
        task_mean=mean, 
        task_std=std, 
        atomref=None,
        drop_path=args.drop_path)
    _log.info(model)

    if args.checkpoint_path is not None:
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict['state_dict'])

    model = model.to(device)
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info('Number of params: {}'.format(n_parameters))
    
    ''' Optimizer and LR Scheduler '''
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = L2MAELoss() #torch.nn.L1Loss()  #torch.nn.MSELoss() # torch.nn.L1Loss() 
    
    ''' Data Loader '''
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.workers, pin_memory=args.pin_mem, 
        drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size)
    
    ''' Compute stats '''
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return
    
    # record the best validation and testing errors and corresponding epochs
    best_metrics = {'val_epoch': 0, 'test_epoch': 0, 
        'val_energy_err': float('inf'), 
        'test_energy_err': float('inf')}
    best_ema_metrics = {'val_epoch': 0, 'test_epoch': 0, 
        'val_energy_err': float('inf'), 
        'test_energy_err': float('inf')}

  
    for epoch in range(args.epochs):
        
        epoch_start_time = time.perf_counter()
        
        lr_scheduler.step(epoch)
        
        train_err, train_loss = train_one_epoch(args=args, model=model, criterion=criterion,
            data_loader=train_loader, optimizer=optimizer,
            device=device, epoch=epoch, model_ema=model_ema,
            print_freq=args.print_freq, logger=_log)
        
        val_err, val_loss = evaluate(args=args, model=model, criterion=criterion, 
            data_loader=val_loader, device=device,
            print_freq=args.print_freq, logger=_log, print_progress=False)
        
        
        update_val_result = update_best_results(args, best_metrics, val_err, epoch)
        if update_val_result:
            torch.save(
                {'state_dict': model.state_dict()}, 
                os.path.join(args.output_dir, 
                    'best_val_epochs@{}_e@{:.4f}_f.pth.tar'.format(epoch, val_err['energy'].avg))
            )


        info_str = 'Epoch: [{epoch}] train_e_MAE: {train_e_mae:.5f}, '.format(
            epoch=epoch, train_e_mae=train_err['energy'].avg)
        info_str += 'val_e_MAE: {:.5f}, '.format(val_err['energy'].avg)
        info_str += 'Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)
        _log.info(info_str)
        
        info_str = 'Best -- val_epoch={}, '.format(best_metrics['val_epoch'])
        info_str += 'val_e_MAE: {:.5f}, '.format(best_metrics['val_energy_err'])

        _log.info(info_str)
        
        # evaluation with EMA
        if model_ema is not None:
            ema_val_err, _ = evaluate(args=args, model=model_ema.module, criterion=criterion, 
                data_loader=val_loader, device=device,
                print_freq=args.print_freq, logger=_log, print_progress=False)
            
                
            update_val_result = update_best_results(args, best_ema_metrics, ema_val_err, epoch)

            if update_val_result:
                torch.save(
                    {'state_dict': get_state_dict(model_ema)}, 
                    os.path.join(args.output_dir, 
                        'best_ema_val_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar'.format(epoch, ema_val_err['energy'].avg))
                )

            info_str = 'EMA '
            info_str += 'val_e_MAE: {:.5f}, val_f_MAE: {:.5f}, '.format(ema_val_err['energy'].avg)
            
            info_str += 'Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)
            _log.info(info_str)
            
            info_str = 'Best EMA -- val_epoch={}, '.format(best_ema_metrics['val_epoch'])
            info_str += 'val_e_MAE: {:.5f}, '.format(best_ema_metrics['val_energy_err'])

            _log.info(info_str)


        

def update_best_results(args, best_metrics, val_err, epoch):

    def _compute_weighted_error(args, energy_err):
        return args.energy_weight * energy_err 

    update_val_result = False 

    new_loss  = _compute_weighted_error(args, val_err['energy'].avg)
    prev_loss = _compute_weighted_error(args, best_metrics['val_energy_err'])
    if new_loss < prev_loss:
        best_metrics['val_energy_err'] = val_err['energy'].avg
        best_metrics['val_epoch'] = epoch
        update_val_result = True


    return update_val_result


def train_one_epoch(args, 
                    model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, 
                    model_ema: Optional[ModelEma] = None,  
                    print_freq: int = 100, 
                    logger=None):
    
    model.train()
    criterion.train()
    
    loss_metrics = {'energy': AverageMeter()}
    mae_metrics  = {'energy': AverageMeter()}
    
    start_time = time.perf_counter()
    
    task_mean = model.task_mean
    task_std = model.task_std

    for step, data in enumerate(data_loader):
        data = data.to(device)
        pred_y = model(node_atom=data.z, pos=data.pos, batch=data.batch)

        loss_e = criterion(pred_y, ((data.y - task_mean) / task_std))
        loss = loss_e

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_metrics['energy'].update(loss_e.item(), n=pred_y.shape[0])
        
        energy_err = pred_y.detach() * task_std + task_mean - data.y
        energy_err = torch.mean(torch.abs(energy_err)).item()
        mae_metrics['energy'].update(energy_err, n=pred_y.shape[0])
        
        if model_ema is not None:
            model_ema.update(model)
        
        torch.cuda.synchronize()
        
        # logging
        if step % print_freq == 0 or step == len(data_loader) - 1: 
            w = time.perf_counter() - start_time
            e = (step + 1) / len(data_loader)
            info_str = 'Epoch: [{epoch}][{step}/{length}] \t'.format(epoch=epoch, step=step, length=len(data_loader))
            info_str +=  'loss_e: {loss_e:.5f}, e_MAE: {e_mae:.5f}, '.format(
                loss_e=loss_metrics['energy'].avg, 
                e_mae=mae_metrics['energy'].avg, 
            )
            info_str += 'time/step={time_per_step:.0f}ms, '.format( 
                time_per_step=(1e3 * w / e / len(data_loader))
            )
            info_str += 'lr={:.2e}'.format(optimizer.param_groups[0]["lr"])
            logger.info(info_str)
        
    return mae_metrics, loss_metrics


def evaluate(args, 
            model: torch.nn.Module, criterion: torch.nn.Module,
            data_loader: Iterable, 
            device: torch.device,   
            print_freq: int = 100, 
            logger=None, 
            print_progress=False, 
            max_iter=-1):

    model.eval()
    criterion.eval()
    loss_metrics = {'energy': AverageMeter()}
    mae_metrics  = {'energy': AverageMeter()}
    
    start_time = time.perf_counter()

    task_mean = model.task_mean
    task_std = model.task_std
    
    with torch.no_grad():
            
        for step, data in enumerate(data_loader):

            data = data.to(device)
            pred_y = model(node_atom=data.z, pos=data.pos, batch=data.batch)

            loss_e = criterion(pred_y, ((data.y - task_mean) / task_std))
            
            loss_metrics['energy'].update(loss_e.item(), n=pred_y.shape[0])
            
            energy_err = pred_y.detach() * task_std + task_mean - data.y
            energy_err = torch.mean(torch.abs(energy_err)).item()
            mae_metrics['energy'].update(energy_err, n=pred_y.shape[0])
            
            # logging
            if (step % print_freq == 0 or step == len(data_loader) - 1) and print_progress: 
                w = time.perf_counter() - start_time
                e = (step + 1) / len(data_loader)
                info_str = '[{step}/{length}] \t'.format(step=step, length=len(data_loader))
                info_str +=  'e_MAE: {e_mae:.5f}, '.format(
                    e_mae=mae_metrics['energy'].avg,
                )
                info_str += 'time/step={time_per_step:.0f}ms'.format( 
                    time_per_step=(1e3 * w / e / len(data_loader))
                )
                logger.info(info_str)
            
            if ((step + 1) >= max_iter) and (max_iter != -1):
                break

    return mae_metrics, loss_metrics
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Training equivariant networks on nablaDFT', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    
