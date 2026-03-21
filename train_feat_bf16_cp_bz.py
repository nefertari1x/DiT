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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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


class LatentFlip(object):
    """
    Randomly flip the latent feature horizontally.
    In VAE latent space (N, C, H, W), flipping the width dimension (-1) 
    corresponds to a horizontal flip of the decoded image.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        # x shape: [C, H, W]
        if torch.rand(1) < self.p:
            return x.flip(-1)
        return x


class NpyListDataset(Dataset):
    """
    Dataset that reads .npy file paths from a pre-generated filelist.
    Avoids slow os.walk on Lustre/networked filesystems.
    
    Expected filelist format (one path per line):
        /path/to/train/classA/file1.npy
        /path/to/train/classA/file2.npy
        /path/to/train/classB/file3.npy
        ...
    
    Class labels are derived from the parent directory name (ImageNet-style).
    """
    def __init__(self, filelist_path, transform=None):
        with open(filelist_path, 'r') as f:
            self.samples = [line.strip() for line in f if line.strip()]
        # Build class-to-index mapping from parent directory names
        classes = sorted(set(os.path.basename(os.path.dirname(s)) for s in self.samples))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        x = torch.from_numpy(np.load(path))
        label = self.class_to_idx[os.path.basename(os.path.dirname(path))]
        if self.transform:
            x = self.transform(x)
        return x, label


def generate_filelist(features_path, filelist_path):
    """
    Walk features_path to find all .npy files and write their paths to filelist_path.
    Only rank 0 should call this; other ranks wait via dist.barrier().
    """
    print(f"Generating filelist from {features_path} ...")
    with open(filelist_path, 'w') as f:
        for root, _, fnames in os.walk(features_path):
            for fname in sorted(fnames):
                if fname.endswith('.npy'):
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

    # Initialize wandb (rank 0 only):
    use_wandb = (rank == 0) and WANDB_AVAILABLE and (not args.no_wandb)
    if use_wandb:
        wandb_run_name = args.wandb_run_name or f"{args.model}_img{args.image_size}_bs{args.global_batch_size}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config=vars(args),
            dir=experiment_dir if rank == 0 else None,
        )
        logger.info(f"W&B run: {wandb.run.url}")

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
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

    # Setup data (Moved before optimizer to calculate total steps):
    transform = transforms.Compose([
        LatentFlip(p=0.5)
    ])

    # Auto-generate filelist if not provided
    if args.filelist is None:
        # Default filelist path next to the features directory
        args.filelist = os.path.join(os.path.dirname(args.features_path.rstrip('/')), "filelist.txt")

    # Generate filelist if it doesn't exist (only rank 0 writes, others wait)
    if not os.path.exists(args.filelist):
        if rank == 0:
            generate_filelist(args.features_path, args.filelist)
        dist.barrier()  # All ranks wait until filelist is ready

    dataset = NpyListDataset(
        filelist_path=args.filelist,
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
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            with autocast(enabled=True, dtype=torch.bfloat16):
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
                # W&B logging:
                if use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                        "train/lr": current_lr,
                        "train/steps_per_sec": steps_per_sec,
                        "train/epoch": epoch + train_steps % steps_per_epoch / steps_per_epoch,
                        "train/step": train_steps,
                        "train/total_steps": total_steps,
                        "train/progress_pct": train_steps / total_steps * 100,
                        "train/samples_seen": train_steps * args.global_batch_size,
                    }, step=train_steps)
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
                    if use_wandb:
                        wandb.log({"checkpoint/step": train_steps}, step=train_steps)
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    if use_wandb:
        wandb.finish()
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
    # W&B arguments:
    parser.add_argument("--wandb-project", type=str, default="dit-B-2-sqr2-ds2-seq4096", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (team/user). None = default entity")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name. Auto-generated if not set")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()
    main(args)

# Usage:
# 1. (Recommended) Pre-generate filelist for instant startup:
#    find /lustre1/work/c30944/DATASET/imagenet_sqr2_lat/train -name "*.npy" > filelist.txt
#    torchrun --nnodes=1 --nproc_per_node=4 train_feat_bf16_cp_bz.py --model DiT-B/2 --features-path /lustre1/work/c30944/DATASET/imagenet_sqr2_lat/train --filelist filelist.txt
#
# 2. Auto-generate filelist on first run (slow first time, fast afterwards):
#    torchrun --nnodes=1 --nproc_per_node=4 train_feat_bf16_cp_bz.py --model DiT-B/2 --features-path /lustre1/work/c30944/DATASET/imagenet_sqr2_lat/train