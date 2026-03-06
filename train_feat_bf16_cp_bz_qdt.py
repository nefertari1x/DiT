# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import math  # Added for cosine calculation

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from torch.utils.tensorboard import SummaryWriter


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


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logger = logging.getLogger(__name__)
    logger.handlers = []  # Clear any existing handlers
    logger.propagate = False  # Don't propagate to root logger
    if dist.get_rank() == 0:  # real logger
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            '[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Stdout handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
        # File handler
        if logging_dir is not None:
            file_handler = logging.FileHandler(f"{logging_dir}/log.txt")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
    else:  # dummy logger (does nothing)
        logger.addHandler(logging.NullHandler())
    return logger


def morton_encode(x, y):
    """Encode (x, y) grid coordinates into a Morton / Z-order code."""
    code = 0
    for i in range(16):
        code |= ((x >> i) & 1) << (2 * i)
        code |= ((y >> i) & 1) << (2 * i + 1)
    return code


def build_depth_gather_index(grid_size=64, token_grid_size=16):
    """
    Precompute an index array that maps z-order depth sequence (grid_size^2,)
    into per-DiT-token groups (token_grid_size^2, leaves_per_token).

    Each DiT token in the 16x16 raster grid covers a 4x4 block of the 64x64
    z-order patch grid.  The returned array lets us gather the relevant leaf
    depths with a single advanced-index operation.
    """
    ratio = grid_size // token_grid_size
    leaves_per_token = ratio * ratio
    num_tokens = token_grid_size * token_grid_size
    gather_idx = np.zeros((num_tokens, leaves_per_token), dtype=np.int64)
    for ty in range(token_grid_size):
        for tx in range(token_grid_size):
            token_idx = ty * token_grid_size + tx
            k = 0
            for dy in range(ratio):
                for dx in range(ratio):
                    gx = tx * ratio + dx
                    gy = ty * ratio + dy
                    gather_idx[token_idx, k] = morton_encode(gx, gy)
                    k += 1
    return gather_idx


class NpyDepthListDataset(Dataset):
    """
    Dataset that loads latent .npy files together with their corresponding
    _depth.npy quadtree depth sequences.

    Horizontal flipping is applied jointly to both the latent and depth map
    so that spatial correspondence is preserved.
    """
    def __init__(self, filelist_path, gather_idx, flip_p=0.5,
                 default_depth=7, num_depth_levels=8):
        with open(filelist_path, 'r') as f:
            raw_samples = [
                line.strip() for line in f
                if line.strip() and not line.strip().endswith('_depth.npy')
            ]
        # Validate samples: filter out entries with missing/bad depth files
        expected_len = gather_idx.shape[0] * gather_idx.shape[1]
        valid_samples = []
        skipped = 0
        for path in raw_samples:
            depth_path = path.replace('.npy', '_depth.npy')
            if not os.path.exists(depth_path):
                skipped += 1
                continue
            try:
                d = np.load(depth_path, mmap_mode='r')
                if d.shape[0] != expected_len:
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue
            valid_samples.append(path)
        if skipped > 0:
            print(f"[NpyDepthListDataset] Skipped {skipped} samples with missing/bad depth files.")
        self.samples = valid_samples
        classes = sorted(set(
            os.path.basename(os.path.dirname(s)) for s in self.samples
        ))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.gather_idx = gather_idx          # (num_tokens, leaves_per_token)
        self.flip_p = flip_p
        self.default_depth = default_depth
        self.num_depth_levels = num_depth_levels
        self.token_grid_size = int(math.sqrt(gather_idx.shape[0]))

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, idx):
        path = self.samples[idx]
        x = torch.from_numpy(np.load(path))                       # (C, H, W)
        label = self.class_to_idx[os.path.basename(os.path.dirname(path))]

        depth_path = path.replace('.npy', '_depth.npy')
        depth_seq = np.load(depth_path).astype(np.int64)
        expected_len = self.gather_idx.shape[0] * self.gather_idx.shape[1]
        if depth_seq.shape[0] != expected_len:
            raise ValueError(f"depth length {depth_seq.shape[0]} != {expected_len}")
        np.clip(depth_seq, 0, self.num_depth_levels - 1, out=depth_seq)
        depth_grouped = torch.from_numpy(depth_seq[self.gather_idx])  # (256, 16)

        if torch.rand(1).item() < self.flip_p:
            x = x.flip(-1)
            tg = self.token_grid_size
            depth_grouped = depth_grouped.reshape(tg, tg, -1)     # (ty, tx, 16)
            depth_grouped = depth_grouped.flip(1)                  # flip along tx
            depth_grouped = depth_grouped.reshape(tg * tg, -1)    # (256, 16)

        return x, depth_grouped, label

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                return self._load_sample(idx)
            except Exception:
                idx = torch.randint(len(self), (1,)).item()
        raise RuntimeError(f"Failed to load valid sample after 10 retries (last idx={idx})")


def generate_filelist(features_path, filelist_path):
    """
    Walk features_path to find all latent .npy files (excluding _depth.npy)
    and write their paths to filelist_path.
    Only rank 0 should call this; other ranks wait via dist.barrier().
    """
    print(f"Generating filelist from {features_path} ...")
    with open(filelist_path, 'w') as f:
        for root, _, fnames in os.walk(features_path):
            for fname in sorted(fnames):
                if fname.endswith('.npy') and not fname.endswith('_depth.npy'):
                    f.write(os.path.join(root, fname) + '\n')
    print(f"Filelist saved to {filelist_path}")

    
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
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 4

    # Initialize TensorBoard (rank 0 only):
    tb_writer = None
    if rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{experiment_dir}/tensorboard")
        logger.info(f"TensorBoard log dir: {experiment_dir}/tensorboard")

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_depth_levels=args.num_depth_levels,
        leaves_per_token=args.leaves_per_token,
    )
    model = model.to(device)
    if rank == 0:
        logger.info("Compiling model with torch.compile...")
    model = torch.compile(model, mode="default")
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model, device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Precompute z-order → DiT-token gather index (done once, shared by all workers):
    patch_size = int(args.model.split('/')[-1])
    token_grid_size = latent_size // patch_size
    sqr_grid_size = args.image_size // 2       # 128 / 2 = 64 (each sqr patch is 2×2 pixels)
    gather_idx = build_depth_gather_index(sqr_grid_size, token_grid_size)

    # Auto-generate filelist if not provided
    if args.filelist is None:
        args.filelist = os.path.join(os.path.dirname(args.features_path.rstrip('/')), "filelist.txt")

    # Generate filelist if it doesn't exist (only rank 0 writes, others wait)
    if not os.path.exists(args.filelist):
        if rank == 0:
            generate_filelist(args.features_path, args.filelist)
        dist.barrier()  # All ranks wait until filelist is ready

    dataset = NpyDepthListDataset(
        filelist_path=args.filelist,
        gather_idx=gather_idx,
        flip_p=0.5,
        default_depth=args.num_depth_levels - 1,
        num_depth_levels=args.num_depth_levels,
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
    logger.info(f"Dataset contains {len(dataset):,} latent images (filelist: {args.filelist})")

    # Calculate LR and Steps:
    # 1. Linear Scaling Rule: lr = base_lr * (global_batch_size / 256)
    base_lr = 1e-4 * (args.global_batch_size / 256)
    
    # 2. Setup Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0)

    # 3. Setup Scheduler (Warmup + Cosine Decay)
    steps_per_epoch = len(dataset) // args.global_batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    logger.info(f"Base LR: {base_lr:.2e}, Total Steps: {total_steps}, Warmup Steps: {warmup_steps}")

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup: 0 -> 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay: 1 -> 0
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    if rank == 0:
        logger.info("Training with BF16 mixed precision.")
        if not torch.cuda.is_bf16_supported():
            logger.warning("Warning: BF16 requested but not supported by this hardware. Performance may degrade or error.")

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, depth, y in loader:
            x = x.to(device)
            depth = depth.to(device)
            y = y.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y, depth=depth)
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step() # Update LR per step
            update_ema(ema, model.module, decay=0.999)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                
                # Get current LR
                current_lr = opt.param_groups[0]["lr"]
                
                logger.info(f"(Step={train_steps:07d}) Train Loss: {avg_loss:.4f}, GNorm: {grad_norm:.2f} , LR: {current_lr:.2e}, Train Steps/Sec: {steps_per_sec:.2f}")
                # TensorBoard logging:
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", avg_loss, train_steps)
                    tb_writer.add_scalar("train/lr", current_lr, train_steps)
                    tb_writer.add_scalar("train/epoch", epoch + train_steps % steps_per_epoch / steps_per_epoch, train_steps)
                    tb_writer.add_scalar("train/steps_per_sec", steps_per_sec, train_steps)
                # Reset monitoring variables:
                running_loss = 0
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

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    if tb_writer is not None:
        tb_writer.close()
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=str, required=True, help="Path to the directory containing .npy files")
    parser.add_argument("--filelist", type=str, default=None,
                        help="Path to a pre-generated filelist (.txt, one .npy path per line). "
                             "If not provided, defaults to filelist.txt next to features-path. "
                             "If the file does not exist, it will be auto-generated on first run.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/2")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=128)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema") 
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of epochs for learning rate warmup")
    # Quadtree depth conditioning:
    parser.add_argument("--num-depth-levels", type=int, default=8,
                        help="Number of discrete depth levels (0 disables depth conditioning)")
    parser.add_argument("--leaves-per-token", type=int, default=16,
                        help="Number of z-order leaves per DiT token")
    args = parser.parse_args()
    main(args)

# Usage:
# 1. (Recommended) Pre-generate filelist for instant startup (exclude _depth.npy):
#    find /lustre1/work/c30944/DATASET/imagenet_sqr2_lat/train -name "*.npy" ! -name "*_depth.npy" > filelist.txt
#    torchrun --nnodes=1 --nproc_per_node=4 train_feat_bf16_cp_bz_qdt.py --model DiT-B/2 --features-path /lustre1/work/c30944/DATASET/imagenet_sqr2_lat/train --filelist filelist.txt
#
# 2. Auto-generate filelist on first run (slow first time, fast afterwards):
#    torchrun --nnodes=1 --nproc_per_node=4 train_feat_bf16_cp_bz_qdt.py --model DiT-B/2 --features-path /lustre1/work/c30944/DATASET/imagenet_sqr2_lat/train