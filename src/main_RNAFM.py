# Suppress Pydantic warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import torch.nn as nn
import torch.nn.functional as F
from time import time
from glob import glob
from collections import OrderedDict
from copy import deepcopy

import os
import argparse
from tqdm import tqdm
import pickle
import wandb
import random

import torch.nn as nn
from torch.utils.data import DataLoader

from read_data_log import SuperTileRNADataset
from utils import patient_kfold, filter_no_features, custom_collate_fn, load_patient_kfold
import torch
import logging
import time
import pandas as pd
import numpy as np
import json
import gc

from transport import create_transport, Sampler
from model_RNAFM import RNAFMModel


def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling_method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling_method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir, args):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log_{args.exp_name}.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def fetch_model(model):
    # param_resnet = []
    # param_other =[]
    for name, param in model.named_parameters():
        # print(name)
        if name.startswith("cond_encoder."):
            param.requires_grad = False
        if name.startswith("cond_encoder.7"):
            param.requires_grad = True
    # return model

def compute_metrics_topk_matrix(labels, preds, topk: list = [20, 50, 100, 500, 1000]):
    # labels: (B, N)
    # preds: (B, N)
    mses = np.mean((labels - preds) ** 2, axis=0)
    maes = np.mean(np.abs(labels - preds), axis=0)
    pccs = np.corrcoef(labels.T, preds.T)
    row_ids = np.arange(labels.shape[1])
    col_ids = np.arange(preds.shape[1]) + labels.shape[1]
    pccs = pccs[row_ids, col_ids]

    valid_mask = ~np.isnan(pccs)
    pccs = pccs[valid_mask]
    mses = mses[valid_mask]
    maes = maes[valid_mask]
    
    indices = np.argsort(pccs)[::-1]
    pccs = pccs[indices]
    #ascending order
    mse_indices = np.argsort(mses)
    mses = mses[mse_indices]
    mae_indices = np.argsort(maes)
    maes = maes[mae_indices]

    pcc_list = [np.mean(pccs)]
    mse_list = [np.mean(mses)]
    mae_list = [np.mean(maes)]
    for k in topk:
        pcc_list.append(np.mean(pccs[:k]))
        mse_list.append(np.mean(mses[:k]))
        mae_list.append(np.mean(maes[:k]))

    return tuple(pcc_list), tuple(mse_list), tuple(mae_list)


def train(model, dataloaders, optimizer, transport, ema, logger=None, 
          num_epochs=200, save_dir='exp/', patience=5, sample_fn=None, 
          split=None, device=None, args=None):

    save_path = save_dir
    training_step = 0
    running_loss = 0.0
    log_steps = 0
    start_time = time.time()
    rna_data_mean = torch.tensor(args.rna_data_mean, dtype=torch.float32).to(device)
    rna_data_std = torch.tensor(args.rna_data_std, dtype=torch.float32).to(device)

    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    best_val_loss_ema = float('inf')
    best_val_pcc = float('-inf')
    best_val_pcc_ema = float('-inf')

    patience_counter = 0

    for epoch in tqdm(range(num_epochs)):
        logger.info(f"================= Beginning train epoch {epoch}... ==================")
        model.train()
        model.training = True
        losses = []
        for s, (image, rna_data, _) in enumerate(dataloaders['train']):
            image = image.to(device)
            rna_data = rna_data.to(device)
            rna_data = (rna_data - rna_data_mean) / (rna_data_std + 1e-10)

            t, xt, ut = transport.diffusion_process(rna_data)
            # with torch.set_grad_enabled(True):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(xt, t, image)
            assert pred.shape == ut.shape
            
            loss = F.mse_loss(pred.float(), ut)
            losses.append(loss.item())
            running_loss += loss.item()
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # Add gradient clipping to prevent NaN
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norm = grad_norm.item()
            scaler.step(optimizer)
            scaler.update()

            if training_step <= 2500:
                decay_start = 0.85
                decay_end = 0.9
                threshold = 2500
                progress = min(training_step / threshold, 1.0)
            else:
                decay_start = 0.9
                decay_end = 0.999
                threshold = 7500
                progress = min((training_step - 2500) / threshold, 1.0)
            decay = decay_start + (decay_end - decay_start) * progress
            update_ema(ema, model, decay=decay)  

            log_steps += 1
            training_step += 1

            if training_step % args.log_every == 0:
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device).item()
                logger.info(f"(step={training_step:09d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                wandb.log(
                        {
                            "train loss": avg_loss, 
                            "train steps/sec": steps_per_sec,
                            "grad norm": grad_norm,
                            "ema decay": decay
                        },
                        step=training_step
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()


        if (epoch+1) % args.sample_every_epoch == 0:
            logger.info("================= Generating validation samples... ==================")
            model.eval()
            model.training = False
            use_cfg = args.cfg_scale > 1.
            val_losses = []
            val_mses = []
            val_maes = []
            val_scores = []
            val_losses_ema = []
            val_mses_ema = []
            val_maes_ema = []
            val_scores_ema = []
            val_mses_top_100 = []
            val_mses_top_1000 = []
            val_mses_top_50 = []
            val_maes_top_100 = []
            val_maes_top_1000 = []
            val_maes_top_50 = []
            val_scores_top_100 = []
            val_scores_top_50 = []
            val_scores_top_1000 = []
            val_scores_ema_top_1000 = []
            val_mses_ema_top_100 = []
            val_mses_ema_top_50 = []
            val_mses_ema_top_1000 = []
            val_maes_ema_top_100 = []
            val_maes_ema_top_50 = []
            val_maes_ema_top_1000 = []
            val_scores_ema_top_100 = []
            val_scores_ema_top_50 = []
            val_scores_ema_top_1000 = []
            val_scores_top_500 = []
            val_scores_ema_top_500 = []
            val_maes_top_500 = []
            val_maes_ema_top_500 = []
            val_mses_top_500 = []
            val_mses_ema_top_500 = []
            val_mses_top_20 = []
            val_mses_ema_top_20 = []
            val_maes_top_20 = []
            val_maes_ema_top_20 = []
            val_scores_top_20 = []
            val_scores_ema_top_20 = []
            eval_g = torch.Generator(device=device)
            eval_g.manual_seed(42)
            for s, (image, rna_data, _) in enumerate(dataloaders['val']):
                if image == []: continue

                image = image.to(device)
                rna_data = rna_data.to(device)
                with torch.no_grad():
                    if use_cfg:
                        logger.info('+++++ Using classifier-free guidance... +++++')
                        z = torch.randn(image.shape[0], args.num_genes, device=device, generator=eval_g)
                        model_kwargs = dict(y=image, cfg_scale=args.cfg_scale)                        
                        samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1] # ema or model here
                        samples_ema = sample_fn(z, ema.forward_with_cfg, **model_kwargs)[-1] # ema or model here
                    else:
                        z = torch.randn(image.shape[0], args.num_genes, device=device, generator=eval_g)
                        model_kwargs = dict(y=image)
                        samples = sample_fn(z, model.forward, **model_kwargs)[-1] # ema or model here
                        samples_ema = sample_fn(z, ema.forward, **model_kwargs)[-1] # ema or model here
                    samples = samples * (rna_data_std + 1e-10) + rna_data_mean
                    samples_ema = samples_ema * (rna_data_std + 1e-10) + rna_data_mean
                    assert samples.shape == rna_data.shape
                    assert samples_ema.shape == rna_data.shape
                    loss = F.mse_loss(samples.view_as(rna_data), rna_data)
                    loss_ema = F.mse_loss(samples_ema.view_as(rna_data), rna_data)
                    samples = samples.detach().cpu()
                    samples_ema = samples_ema.detach().cpu()
                    rna_data = rna_data.detach().cpu()
                    (pcc, pcc_top_20, pcc_top_50, pcc_top_100, pcc_top_500, pcc_top_1000), (mse, mse_top_20, mse_top_50, mse_top_100, mse_top_500, mse_top_1000), (mae, mae_top_20, mae_top_50, mae_top_100, mae_top_500, mae_top_1000) = compute_metrics_topk_matrix(rna_data.numpy(), samples.numpy())
                    (pcc_ema, pcc_ema_top_20, pcc_ema_top_50, pcc_ema_top_100, pcc_ema_top_500, pcc_ema_top_1000), (mse_ema, mse_ema_top_20, mse_ema_top_50, mse_ema_top_100, mse_ema_top_500, mse_ema_top_1000), (mae_ema, mae_ema_top_20, mae_ema_top_50, mae_ema_top_100, mae_ema_top_500, mae_ema_top_1000) = compute_metrics_topk_matrix(rna_data.numpy(), samples_ema.numpy())

                    val_losses.append(loss.item())
                    val_losses_ema.append(loss_ema.item())
                    val_mses.append(mse)
                    val_mses_ema.append(mse_ema)
                    val_mses_top_20.append(mse_top_20)
                    val_mses_top_100.append(mse_top_100)
                    val_mses_top_500.append(mse_top_500)
                    val_mses_top_1000.append(mse_top_1000)
                    val_mses_top_50.append(mse_top_50)
                    val_mses_ema_top_20.append(mse_ema_top_20)
                    val_mses_ema_top_100.append(mse_ema_top_100)
                    val_mses_ema_top_1000.append(mse_ema_top_1000)
                    val_mses_ema_top_50.append(mse_ema_top_50)
                    val_mses_ema_top_500.append(mse_ema_top_500)
                    val_maes.append(mae)
                    val_maes_ema.append(mae_ema)
                    val_maes_top_20.append(mae_top_20)
                    val_maes_top_100.append(mae_top_100)
                    val_maes_top_1000.append(mae_top_1000)
                    val_maes_top_50.append(mae_top_50)
                    val_maes_top_500.append(mae_top_500)
                    val_maes_ema_top_20.append(mae_ema_top_20)
                    val_maes_ema_top_100.append(mae_ema_top_100)
                    val_maes_ema_top_1000.append(mae_ema_top_1000)
                    val_maes_ema_top_50.append(mae_ema_top_50)
                    val_maes_ema_top_500.append(mae_ema_top_500)
                    val_scores.append(pcc)
                    val_scores_ema.append(pcc_ema)
                    val_scores_top_20.append(pcc_top_20)
                    val_scores_top_100.append(pcc_top_100)
                    val_scores_top_1000.append(pcc_top_1000)
                    val_scores_top_50.append(pcc_top_50)
                    val_scores_top_500.append(pcc_top_500)
                    val_scores_ema_top_20.append(pcc_ema_top_20)
                    val_scores_ema_top_100.append(pcc_ema_top_100)
                    val_scores_ema_top_1000.append(pcc_ema_top_1000)
                    val_scores_ema_top_50.append(pcc_ema_top_50)
                    val_scores_ema_top_500.append(pcc_ema_top_500)
            val_losses = np.mean(val_losses)
            val_losses_ema = np.mean(val_losses_ema)
            val_mses = np.mean(val_mses)
            val_mses_ema = np.mean(val_mses_ema)
            val_maes = np.mean(val_maes)
            val_maes_ema = np.mean(val_maes_ema)
            val_scores = np.mean(val_scores)
            val_scores_ema = np.mean(val_scores_ema)
            val_mses_top_20 = np.mean(val_mses_top_20)
            val_mses_ema_top_20 = np.mean(val_mses_ema_top_20)
            val_mses_top_100 = np.mean(val_mses_top_100)
            val_mses_ema_top_100 = np.mean(val_mses_ema_top_100)
            val_mses_top_500 = np.mean(val_mses_top_500)
            val_mses_ema_top_500 = np.mean(val_mses_ema_top_500)
            val_maes_top_100 = np.mean(val_maes_top_100)
            val_maes_ema_top_100 = np.mean(val_maes_ema_top_100)
            val_maes_top_20 = np.mean(val_maes_top_20)
            val_maes_ema_top_20 = np.mean(val_maes_ema_top_20)
            val_scores_top_100 = np.mean(val_scores_top_100)
            val_scores_ema_top_100 = np.mean(val_scores_ema_top_100)
            val_mses_top_50 = np.mean(val_mses_top_50)
            val_mses_ema_top_50 = np.mean(val_mses_ema_top_50)
            val_maes_top_50 = np.mean(val_maes_top_50)
            val_maes_ema_top_50 = np.mean(val_maes_ema_top_50)
            val_maes_top_500 = np.mean(val_maes_top_500)
            val_maes_ema_top_500 = np.mean(val_maes_ema_top_500)
            val_scores_top_20 = np.mean(val_scores_top_20)
            val_scores_ema_top_20 = np.mean(val_scores_ema_top_20)
            val_scores_top_50 = np.mean(val_scores_top_50)
            val_scores_ema_top_50 = np.mean(val_scores_ema_top_50)
            val_scores_top_500 = np.mean(val_scores_top_500)
            val_scores_ema_top_500 = np.mean(val_scores_ema_top_500)
            val_mses_top_1000 = np.mean(val_mses_top_1000)
            val_mses_ema_top_1000 = np.mean(val_mses_ema_top_1000)
            val_maes_top_1000 = np.mean(val_maes_top_1000)
            val_maes_ema_top_1000 = np.mean(val_maes_ema_top_1000)
            val_scores_top_1000 = np.mean(val_scores_top_1000)
            val_scores_ema_top_1000 = np.mean(val_scores_ema_top_1000)
            wandb.log({f"val score fold {split}": val_scores, f"val mse fold {split}": val_mses, f"val mae fold {split}": val_maes, 
                        f"val score top 1000 fold {split}": val_scores_top_1000, f"val mse top 1000 fold {split}": val_mses_top_1000, f"val mae top 1000 fold {split}": val_maes_top_1000, 
                        f"val score top 500 fold {split}": val_scores_top_500, f"val mse top 500 fold {split}": val_mses_top_500, f"val mae top 500 fold {split}": val_maes_top_500, 
                        f"val score top 100 fold {split}": val_scores_top_100, f"val mse top 100 fold {split}": val_mses_top_100, f"val mae top 100 fold {split}": val_maes_top_100, 
                        f"val score top 50 fold {split}": val_scores_top_50, f"val mse top 50 fold {split}": val_mses_top_50, f"val mae top 50 fold {split}": val_maes_top_50, 
                        f"val score top 20 fold {split}": val_scores_top_20, f"val mse top 20 fold {split}": val_mses_top_20, f"val mae top 20 fold {split}": val_maes_top_20, 
                        f"val loss fold {split}": val_losses, f"val loss ema fold {split}": val_losses_ema,
                        f"val score ema top 1000 fold {split}": val_scores_ema_top_1000, f"val mse ema top 1000 fold {split}": val_mses_ema_top_1000, f"val mae ema top 1000 fold {split}": val_maes_ema_top_1000, 
                        f"val score ema top 500 fold {split}": val_scores_ema_top_500, f"val mse ema top 500 fold {split}": val_mses_ema_top_500, f"val mae ema top 500 fold {split}": val_maes_ema_top_500, 
                        f"val score ema top 100 fold {split}": val_scores_ema_top_100, f"val mse ema top 100 fold {split}": val_mses_ema_top_100, f"val mae ema top 100 fold {split}": val_maes_ema_top_100, 
                        f"val score ema top 50 fold {split}": val_scores_ema_top_50, f"val mse ema top 50 fold {split}": val_mses_ema_top_50, f"val mae ema top 50 fold {split}": val_maes_ema_top_50, 
                        f"val score ema top 20 fold {split}": val_scores_ema_top_20, f"val mse ema top 20 fold {split}": val_mses_ema_top_20, f"val mae ema top 20 fold {split}": val_maes_ema_top_20}
                        , step=training_step
                        )

            logger.info(f'Val Loss: {val_losses}, MAE: {val_maes}, MSE: {val_mses}, Score: {val_scores}, MSE Top 100: {val_mses_top_100}, MAE Top 100: {val_maes_top_100}, Score Top 100: {val_scores_top_100}, MSE Top 50: {val_mses_top_50}, MAE Top 50: {val_maes_top_50}, Score Top 50: {val_scores_top_50}, MSE Top 1000: {val_mses_top_1000}, MAE Top 1000: {val_maes_top_1000}, Score Top 1000: {val_scores_top_1000}, MSE Top 500: {val_mses_top_500}, MAE Top 500: {val_maes_top_500}, Score Top 500: {val_scores_top_500}')
            logger.info(f'Val Loss EMA: {val_losses_ema}, MAE EMA: {val_maes_ema}, MSE EMA: {val_mses_ema}, Score EMA: {val_scores_ema}, MSE EMA Top 100: {val_mses_ema_top_100}, MAE EMA Top 100: {val_maes_ema_top_100}, Score EMA Top 100: {val_scores_ema_top_100}, MSE EMA Top 50: {val_mses_ema_top_50}, MAE EMA Top 50: {val_maes_ema_top_50}, Score EMA Top 50: {val_scores_ema_top_50}, MSE EMA Top 1000: {val_mses_ema_top_1000}, MAE EMA Top 1000: {val_maes_ema_top_1000}, Score EMA Top 1000: {val_scores_ema_top_1000}, MSE EMA Top 500: {val_mses_ema_top_500}, MAE EMA Top 500: {val_maes_ema_top_500}, Score EMA Top 500: {val_scores_ema_top_500}')
            logger.info("Generating EMA samples done.")
            if val_losses < best_val_loss or val_scores > best_val_pcc or val_losses_ema < best_val_loss_ema or val_scores_ema > best_val_pcc_ema:
                if val_losses < best_val_loss:
                    best_val_loss = val_losses
                if val_scores > best_val_pcc:
                    best_val_pcc = val_scores
                if val_losses_ema < best_val_loss_ema:
                    best_val_loss_ema = val_losses_ema
                if val_scores_ema > best_val_pcc_ema:
                    best_val_pcc_ema = val_scores_ema
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": optimizer.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{save_path}/val_best.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                logger.info(f"Better checkpoint found, saving current checkpoint to {checkpoint_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"No improvement.")
                if patience_counter == patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    return model, ema
        

        losses = np.mean(losses)
        wandb.log({f'train loss fold {split}': losses}, step=training_step)

        logger.info(f'Train Loss fold {split}: {losses}')

    return model, ema


def evaluate(model, dataloaders, ema, logger=None, sample_fn=None, split=None, device=None, args=None):
    model.eval()
    model.training = False
    use_cfg = args.cfg_scale > 1.
    wsis = []
    projs = []
    real = []
    preds = []
    preds_ema = []
    rna_data_mean = torch.tensor(args.rna_data_mean, dtype=torch.float32).to(device)
    rna_data_std = torch.tensor(args.rna_data_std, dtype=torch.float32).to(device)
    assert rna_data_mean.shape == rna_data_std.shape
    for s, (image, rna_data, wsi_file_name) in tqdm(enumerate(dataloaders), total=len(dataloaders)):
        if image == []: continue
        image = image.to(device)
        rna_data = rna_data.to(device)
        wsis.append(wsi_file_name)
        # repeat args.cohort make it with shape B,1
        projs.append([args.cohort * image.shape[0]])
        with torch.no_grad():
            if use_cfg:
                z = torch.randn(image.shape[0], args.num_genes, device=device)
                model_kwargs = dict(y=image, cfg_scale=args.cfg_scale)
                samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1] # ema or model here
                samples_ema = sample_fn(z, ema.forward_with_cfg, **model_kwargs)[-1] # ema or model here
            else:
                z = torch.randn(image.shape[0], args.num_genes, device=device)
                model_kwargs = dict(y=image)
                samples = sample_fn(z, model.forward, **model_kwargs)[-1] # ema or model here
                samples_ema = sample_fn(z, ema.forward, **model_kwargs)[-1] # ema or model here
            assert samples.shape == rna_data.shape
            assert samples_ema.shape == rna_data.shape
            samples = samples * (rna_data_std + 1e-10) + rna_data_mean
            samples_ema = samples_ema * (rna_data_std + 1e-10) + rna_data_mean
            assert samples.shape == rna_data.shape
            assert samples_ema.shape == rna_data.shape
            samples = samples.detach().cpu()
            samples_ema = samples_ema.detach().cpu()
            rna_data = rna_data.detach().cpu()
            real.append(rna_data.numpy())
            preds.append(samples.numpy())
            preds_ema.append(samples_ema.numpy())
            
    real = np.concatenate(real, axis=0)
    preds = np.concatenate(preds, axis=0)
    preds_ema = np.concatenate(preds_ema, axis=0)
    wsis = np.concatenate(wsis, axis=0)
    projs = np.concatenate(projs, axis=0)
    return preds, preds_ema, real, wsis, projs




def main(args):
    """
    Trains a new SiT model.
    """

    ############################################## seeds ##############################################
    fix_seed(42)
    g = torch.Generator()
    g.manual_seed(42)
        
    ############################################## logging ##############################################
    
    save_dir = os.path.join(args.src_path, args.save_dir, args.cohort, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device(f"cuda:{args.gpu_id}" if (torch.cuda.is_available()) else "cpu")

     
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(log_dir, args)
    logger.info(f"Experiment directory created at: {log_dir}")
    logger.info(f"Saving at: {save_dir}")
    logger.info(f"Device: {device}")

    ############################################## model prep ##############################################

    transport = create_transport(
                args.path_type,
                args.prediction,
                args.loss_weight,
                args.train_eps,
                args.sample_eps
            )  # default: velocity; 
    transport_sampler = Sampler(transport)
    if args.mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = transport_sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = transport_sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )
            
    elif args.mode == "SDE":
        sample_fn = transport_sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )

    ############################################## data prep ##############################################
    args.rna_data_mean = np.load(f'./examples/{args.cohort}_rna_data_mean.npy')
    args.rna_data_std = np.load(f'./examples/{args.cohort}_rna_data_std.npy')

    df = pd.read_csv(args.ref_file)
    if args.sample_percent != None:
        df = df.sample(frac=args.sample_percent).reset_index(drop=True)

    if ('tcga_project' in df.columns) and (args.tcga_projects != None):
        projects = args.tcga_projects.split(',')
        df = df[df['tcga_project'].isin(projects)].reset_index(drop=True)
        print(f'Filtered project {projects}')

    if args.filter_no_features:
        df = filter_no_features(df, feature_path=args.feature_path, feature_name='cluster_features')

    pathway_gene_indices_all = json.load(open(args.pathway_gene_indices_file))
    print(f'all num of pathways: {len(list(pathway_gene_indices_all.keys()))}')
    # exclude the UNMAPPED and UNCLASSIFIED pathways
    pathway_gene_indices = {k: v for k, v in pathway_gene_indices_all.items() if k not in ['UNMAPPED', 'UNCLASSIFIED']}
    P = len(list(pathway_gene_indices.keys()))

    # convert gene lists to sets for fast overlap checks
    pathway_gene_sets = [set(v) for v in pathway_gene_indices.values()]
    # initialize adjacency matrix
    pathway_adj = torch.zeros(P, P)

    # fill adjacency: 1 if shared gene(s)
    for i in range(P):
        for j in range(P):
            if i == j or len(pathway_gene_sets[i].intersection(pathway_gene_sets[j])) > 0:
                pathway_adj[i, j] = 1.0   # connected

    args.pathway_adj = pathway_adj.to(device)

    print("Pathway Adjacency Matrix Shape:", pathway_adj.shape)
    args.pathway_gene_indices = [torch.tensor(v, dtype=torch.long).to(device) for v in pathway_gene_indices.values()]
    print(f'pathway_gene_indices length: {len(args.pathway_gene_indices)}')
    # concatenate the UNMAPPED and UNCLASSIFIED pathways if exist
    bg_gene_indices = []
    if 'UNMAPPED' in pathway_gene_indices_all: 
        bg_gene_indices.extend(pathway_gene_indices_all['UNMAPPED'])
    if 'UNCLASSIFIED' in pathway_gene_indices_all:
        bg_gene_indices.extend(pathway_gene_indices_all['UNCLASSIFIED'])
    args.bg_gene = torch.tensor(bg_gene_indices, dtype=torch.long).to(device)
    print(f'bg_gene length: {len(args.bg_gene)}')
    #check the gpu memory
    # print(f'GPU Memory: {torch.cuda.memory_summary(device=device)}')
    ############################################## kfold ##############################################
    if args.new_split:
        train_idxs, val_idxs, test_idxs = patient_kfold(df, n_splits=args.k)
    else:
        train_idxs, val_idxs = load_patient_kfold(df, f'./patient_splits/TCGA-{args.cohort}.npy') 

    if 'rna_file_name' in df.columns:
        df = df.drop(columns=['rna_file_name'])

    test_results_splits = {}
    i = 0

    for train_idx, val_idx in zip(train_idxs, val_idxs):
        wandb.init(project='Diffusion_rna', config=args, name=args.exp_name+'_fold_'+str(i)) 
        logger.info(f'Fold {i} is starting ...')
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        if args.new_split:

            # save patient ids to file
            np.save(save_dir + '/train_'+str(i)+'.npy', np.unique(train_df.patient_id) )
            np.save(save_dir + '/val_'+str(i)+'.npy', np.unique(val_df.patient_id) )
        
        # init dataset
        train_dataset = SuperTileRNADataset(train_df, args.feature_path)
        val_dataset = SuperTileRNADataset(val_df, args.feature_path)

        args.num_genes = train_dataset.num_genes 
        args.feature_dim = train_dataset.feature_dim
        print(f'Num outputs: {args.num_genes}, Feature dim: {args.feature_dim}')

        # init dataloaders
        train_dataloader = DataLoader(train_dataset, 
                    num_workers=1, pin_memory=True, 
                    shuffle=True, batch_size=args.batch_size,
                    collate_fn=custom_collate_fn,
                    generator=g)
        
        val_dataloader = DataLoader(val_dataset, 
                    num_workers=1, pin_memory=True, 
                    shuffle=False, batch_size=args.batch_size,
                    collate_fn=custom_collate_fn)
        
        dataloaders = { 'train': train_dataloader, 'val': val_dataloader}
        model = RNAFMModel(
            gene_dim=args.num_genes,
            input_dim=args.feature_dim,
            depth=args.depth,
            num_head=args.num_heads, 
            hidden_dim=args.hidden_dim,
            cond_drop_ratio=0.1,
            pathway_indices=args.pathway_gene_indices,
            pathway_adj=args.pathway_adj,
            bg_gene=args.bg_gene
        )
        model.to(device)

        logger.info(f"Diff Parameters: {sum(p.numel() for p in model.parameters()):,}")

        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)

        # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
        optimizer = torch.optim.AdamW(list(model.parameters()), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))

        # Prepare models for training:
        update_ema(ema, model, decay=0.)  # Ensure EMA is initialized with synced weights
        model.train()  # important! This enables embedding dropout for classifier-free guidance
        ema.eval()  # EMA model should always be in eval mode
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, f'split_{i}'), exist_ok=True)
        save_path = os.path.join(save_dir, f'split_{i}')
        
        if args.train:
            model, ema = train(model, dataloaders, optimizer, transport, ema, logger=logger,
                            num_epochs=args.num_epochs, save_dir=save_path, patience=args.patience,
                            sample_fn=sample_fn, split=i, device=device, args=args)
        else:
            ckpts = sorted(glob(os.path.join(save_path, '**.pt')))
            ckpt = ckpts[-1]
            print(f"Loading checkpoint: {ckpt}")
            
            # Load the checkpoint file
            checkpoint = torch.load(ckpt, map_location=device)
            
            # Load the state dicts
            model.load_state_dict(checkpoint['model'])
            ema.load_state_dict(checkpoint['ema'])

        
        preds, preds_ema, real, wsis, projs = evaluate(model, val_dataloader, ema, logger=logger, sample_fn=sample_fn, split=i, device=device, args=args)


        
        test_results = {
            'real': real,
            'preds': preds,
            'preds_ema': preds_ema,
            'wsi_file_name': wsis,
            'tcga_project': projs
        }
        test_results_splits[f'split_{i}'] = test_results
        
        test_results_splits['genes'] = [x[4:] for x in df.columns if 'rna_' in x]
        with open(os.path.join(save_path, f'test_results_{i}_cfg_{args.cfg_scale}.pkl'), 'wb') as f:
            pickle.dump(test_results_splits, f)
        wandb.finish()
        i += 1
        torch.cuda.empty_cache()
        gc.collect()

def none_or_str(value):
    if value == 'None':
        return None
    return value

if __name__ == "__main__":

    torch.cuda.empty_cache() 
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_path', type=str, default='', help='project path')
    parser.add_argument('--ref_file', type=str, default=None, help='path to reference file')
    parser.add_argument('--sample-percent', type=float, default=None, help='Downsample available data to test the effect of having a smaller dataset. If None, no downsampling.')
    parser.add_argument('--tcga_projects', help="the tcga_projects we want to use, separated by comma", default=None, type=str)
    parser.add_argument('--feature_path', type=str, default="features/", help='path to resnet/uni and clustered features')
    parser.add_argument('--save_dir', type=str, default='saved_exp', help='parent destination folder')
    parser.add_argument('--cohort', type=str, default="LUAD", help='cohort name for creating the saving folder of the results')
    parser.add_argument('--exp_name', type=str, default="exp", help='Experiment name for creating the saving folder of the results')
    parser.add_argument('--filter_no_features', type=int, default=1, help='Whether to filter out samples with no features')
    parser.add_argument('--log', type=str, help='Experiment name to log')
    parser.add_argument('--mode', type=str, default='ODE', help='ODE/SDE')
    parser.add_argument('--pathway_gene_indices_file', type=str, default='./examples/all_gene_indices_filtered.json', help='pathway gene indices file')
    
    # model args
    parser.add_argument('--model_type', type=str, default='RNAFM', help='RNAFM')
    parser.add_argument('--depth', type=int, default=6, help='transformer depth')
    parser.add_argument('--num_heads', type=int, default=16, help='number of attention heads')
    parser.add_argument('--seed', type=int, default=99, help='Seed for random generation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint from trained model.')
    parser.add_argument('--train', help="if you want to train the model", action="store_true")
    parser.add_argument('--num_epochs', type=int, default=2000, help='number of epochs to train')
    parser.add_argument('--change_num_genes', type=int, default=0, help="whether finetuning from a model trained on different number of genes")
    parser.add_argument('--num_genes', type=int, default=None, help='number of genes on which pretrained model was trained')
    parser.add_argument('--k', type=int, default=5, help='Number of splits')
    parser.add_argument('--save_on', type=str, default='loss', help='which criterium to save model on, "loss" or "loss+corr"')
    parser.add_argument('--stop_on', type=str, default='loss', help='which criterium to do early stopping on, "loss" or "loss+corr"')
    parser.add_argument('--gpu_id', type=int, default=3, help='GPU ID')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--patience', type=int, default=5, help='patience')
    parser.add_argument('--new_split', type=bool, default=False, help='whether to use new split')


    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--ckpt_every", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--path_type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    parser.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    parser.add_argument("--loss_weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    parser.add_argument("--sample_eps", type=float)
    parser.add_argument("--train_eps", type=float)
    parser.add_argument('--sample_every_epoch', type=int, default=50, help='sample every epoch')
    parser.add_argument("--num_sampling_steps", type=int, default=50)

    
    args = parser.parse_known_args()[0]

    if args.mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif args.mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE

    args = parser.parse_args()

    wandb.login(key="[INSERT KEY HERE]")
    
    main(args)

    print("Process Finished")
