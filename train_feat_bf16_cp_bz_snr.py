# train_feat_bf16_cp_bz_minsnr.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
A minimal training script for DiT using PyTorch DDP.
Modified with Min-SNR Weighting Strategy.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import DatasetFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import math  # Added for cosine calculation
from torch.cuda.amp import autocast

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


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
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def npy_loader(path):
    """
    Loader for .npy files.
    """
    return torch.from_numpy(np.load(path))

class LatentFlip(object):
    """
    Randomly flip the latent feature horizontally.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            return x.flip(-1)
        return x
    
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True) 
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-") 
        # 在文件夹名中标记 MinSNR
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-MinSNR"
        checkpoint_dir = f"{experiment_dir}/checkpoints" 
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Min-SNR Gamma: {args.snr_gamma}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    # 假设你已经按照之前的建议在 DiT_models 构造函数中支持了 in_channels
    # 如果没有，请记得这里适配你的 DiT 定义
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    model = model.to(device)
    if rank == 0:
        logger.info("Compiling model with torch.compile...")
    model = torch.compile(model, mode="default")
    
    ema = deepcopy(model).to(device) 
    requires_grad(ema, False)
    model = DDP(model, device_ids=[rank])
    
    diffusion = create_diffusion(timestep_respacing="") 
    
    # --- Min-SNR 准备工作: 获取 alphas_cumprod ---
    # diffusion.alphas_cumprod 通常是 numpy 数组，转为 Tensor 并移动到 GPU
    alphas_cumprod = torch.from_numpy(diffusion.alphas_cumprod).to(device, dtype=torch.float32)
    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data
    transform = transforms.Compose([
        LatentFlip(p=0.5)
    ])
    dataset = DatasetFolder(
        root=args.features_path, 
        loader=npy_loader, 
        extensions=('.npy',), 
        transform=transform
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} latent images ({args.features_path})")

    # Optimizer & Scheduler
    base_lr = 1e-4 * (args.global_batch_size / 256)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.995), weight_decay=0.0)

    steps_per_epoch = len(dataset) // args.global_batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    logger.info(f"Base LR: {base_lr:.2e}, Total Steps: {total_steps}, Warmup Steps: {warmup_steps}")

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()
    
    if rank == 0:
        logger.info("Training with BF16 mixed precision.")

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    
    # --- 修改: 分别记录 Raw Loss 和 SNR Loss ---
    running_raw_loss = 0
    running_snr_loss = 0
    
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            
            with autocast(enabled=True, dtype=torch.bfloat16):
                # diffusion.training_losses 返回的是未 reduce 的 loss dict (通常 shape 为 [B])
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                raw_loss_batch = loss_dict["loss"]
                
                # --- Min-SNR 计算核心逻辑 ---
                # 1. 计算当前 Batch 每个 t 的 SNR
                # SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
                cur_alpha = alphas_cumprod[t]
                cur_sigma_sq = 1.0 - cur_alpha
                # 避免除以 0 (虽然 t 通常不到 T, 但为了安全加个 eps)
                snr = cur_alpha / (cur_sigma_sq + 1e-8)
                
                # 2. 计算权重: min(SNR, gamma) / SNR
                # 对于 epsilon 预测，这就是标准的 Min-SNR 公式
                snr_gamma = args.snr_gamma
                weights = torch.clamp(snr, max=snr_gamma) / snr
                
                # 3. 应用权重并计算最终 Loss
                # 注意 raw_loss_batch 是 [B]，weights 是 [B]，直接相乘
                snr_loss_batch = raw_loss_batch * weights
                loss = snr_loss_batch.mean()
                
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            update_ema(ema, model.module, decay=0.9995)

            # Log loss values:
            # 记录原始未加权 Loss 的平均值
            running_raw_loss += raw_loss_batch.mean().item()
            # 记录加权后用于反向传播的 Loss
            running_snr_loss += loss.item()
            
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                # --- 修改: Reduce 两个 Loss ---
                # 将 Raw Loss 和 SNR Loss 打包成 Tensor 一起 reduce 效率更高
                avg_loss_tensor = torch.tensor([running_raw_loss, running_snr_loss], device=device) / log_steps
                dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss_tensor = avg_loss_tensor / dist.get_world_size()
                
                avg_raw_loss = avg_loss_tensor[0].item()
                avg_snr_loss = avg_loss_tensor[1].item()
                
                current_lr = opt.param_groups[0]["lr"]
                
                # 打印日志：同时显示 Raw 和 MinSNR Loss
                logger.info(
                    f"(Step={train_steps:07d}) "
                    f"Raw Loss: {avg_raw_loss:.4f}, "
                    f"MinSNR Loss: {avg_snr_loss:.4f}, "
                    f"GNorm: {grad_norm:.2f}, "
                    f"LR: {current_lr:.2e}, "
                    f"Steps/Sec: {steps_per_sec:.2f}"
                )
                
                # Reset monitoring variables:
                running_raw_loss = 0
                running_snr_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=str, required=True, help="Path to the directory containing .npy files")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema") 
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of epochs for learning rate warmup")
    
    # --- 新增参数 ---
    parser.add_argument("--snr-gamma", type=float, default=5.0, help="Min-SNR weighting gamma")
    
    args = parser.parse_args()
    main(args)

# torchrun --nnodes=1 --nproc_per_node=4 train_feat_bf16_cp_bz_snr.py --model DiT-B/2 --features-path /work/c30636/dataset/dit_feat_aug/train 
# /work/c30778/dataset/dit_feat_fix/train