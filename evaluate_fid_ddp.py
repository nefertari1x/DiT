# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample from DiT and compute FID on-the-fly using torchmetrics.
Optimized for H100 with bfloat16 precision.
Supports batch evaluation of multiple checkpoints.
"""
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms, datasets
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import os
import math
import glob
from tqdm import tqdm

# 引入 torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance

def main(args):
    # --- H100 Speedups ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- DDP Setup ---
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, world_size={world_size}, device={device}.")
    else:
        rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Starting in single-process mode.")

    # --- Setup PyTorch ---
    torch.manual_seed(args.seed + rank)
    torch.set_grad_enabled(False)

    # --- Prepare Checkpoints List & Results Path ---
    checkpoints = []
    results_file_path = args.results_file

    if args.ckpt_dir:
        # 查找目录下所有的 .pt 文件并排序
        search_path = os.path.join(args.ckpt_dir, "*.pt")
        checkpoints = sorted(glob.glob(search_path))
        if rank == 0:
            print(f"Found {len(checkpoints)} checkpoints in {args.ckpt_dir}")
        
        # 智能定位保存路径：如果 ckpt_dir 是 ".../checkpoints"，则保存到上一级 (log.txt 所在位置)
        if not os.path.isabs(results_file_path):
            abs_ckpt_dir = os.path.abspath(args.ckpt_dir)
            if os.path.basename(abs_ckpt_dir) == "checkpoints":
                exp_dir = os.path.dirname(abs_ckpt_dir)
                results_file_path = os.path.join(exp_dir, args.results_file)
            else:
                # 否则保存在 ckpt_dir 同级
                results_file_path = os.path.join(abs_ckpt_dir, args.results_file)
            
            if rank == 0:
                print(f"FID results will be saved to: {results_file_path}")

    elif args.ckpt:
        checkpoints = [args.ckpt]
    else:
        # 默认自动下载模式 (只用于 DiT-XL/2)
        checkpoints = [None]

    # --- Load Model Architecture (Once) ---
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # H100 核心优化：转为 bfloat16
    model = model.to(dtype=torch.bfloat16)
    model = torch.compile(model, mode="default")
    diffusion = create_diffusion(str(args.num_sampling_steps))
    
    # --- Load VAE Model (Convert to BF16) ---
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    vae = vae.to(dtype=torch.bfloat16)

    # --- Setup TorchMetrics FID ---
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    
    # --- Setup Real Data Loader ---
    if args.real_data_dir is None:
        raise ValueError("Please provide --real-data-dir for FID calculation")

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.PILToTensor(), 
    ])
    
    dataset = datasets.ImageFolder(args.real_data_dir, transform=transform)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    # 注意：在循环中我们会多次创建迭代器
    real_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )

    # --- Loop Over Checkpoints ---
    for i, ckpt_path in enumerate(checkpoints):
        # 1. Load Weights
        current_ckpt_path = ckpt_path or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
        if rank == 0:
            print(f"\n[{i+1}/{len(checkpoints)}] Processing checkpoint: {current_ckpt_path}")
        
        state_dict = find_model(current_ckpt_path)
        model.load_state_dict(state_dict)
        model.eval()

        # 2. Reset FID State & Data Iterator
        fid_metric.reset()
        real_iter = iter(real_loader) # 重置真实数据迭代器以确保覆盖

        # 3. Calculation Loop
        samples_per_gpu = args.num_samples // world_size
        batch_size = args.batch_size
        num_batches = math.ceil(samples_per_gpu / batch_size)
        
        if rank == 0:
            iterator = tqdm(range(num_batches), desc=f"FID ({os.path.basename(current_ckpt_path)})")
        else:
            iterator = range(num_batches)

        total_generated = 0
        
        for _ in iterator:
            n = batch_size
            if total_generated + n > samples_per_gpu:
                n = samples_per_gpu - total_generated

            # A. 生成 Fake Images
            z = torch.randn(n, 4, latent_size, latent_size, device=device, dtype=torch.bfloat16)
            y = torch.randint(0, args.num_classes, (n,), device=device)

            z_combined = torch.cat([z, z], 0)
            y_null = torch.tensor([args.num_classes] * n, device=device)
            y_combined = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y_combined, cfg_scale=args.cfg_scale)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, 
                    z_combined.shape, 
                    z_combined, 
                    clip_denoised=False, 
                    model_kwargs=model_kwargs, 
                    progress=False, 
                    device=device
                )
                samples, _ = samples.chunk(2, dim=0)
                images = vae.decode(samples / 0.18215).sample

            images = (images / 2 + 0.5).clamp(0, 1)
            images = (images * 255).to(torch.uint8)

            fid_metric.update(images, real=False)

            # B. 读取 Real Images
            try:
                real_batch, _ = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                real_batch, _ = next(real_iter)
            
            real_batch = real_batch.to(device)
            if real_batch.shape[1] == 1:
                real_batch = real_batch.repeat(1, 3, 1, 1)
                
            fid_metric.update(real_batch, real=True)
            
            total_generated += n

        # 4. Compute & Log FID
        if rank == 0:
            print("Synchronizing and computing FID...")
        
        fid_score = fid_metric.compute()
        
        if rank == 0:
            score_val = fid_score.item()
            print(f"FID Score for {os.path.basename(current_ckpt_path)}: {score_val:.4f}")
            
            # 写入结果文件 (追加模式)
            with open(results_file_path, "a") as f:
                f.write(f"{current_ckpt_path}\t{score_val:.4f}\n")

        # 确保所有进程同步进入下一个 checkpoint
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    
    # Checkpoint arguments
    parser.add_argument("--ckpt", type=str, default=None, help="Single checkpoint path")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Directory containing .pt checkpoints for batch evaluation")
    parser.add_argument("--results-file", type=str, default="fid_results.txt", help="File to append FID results")

    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--real-data-dir", type=str, required=True)

    args = parser.parse_args()
    main(args)



# torchrun --nnodes=1 --nproc_per_node=4 evaluate_fid_ddp.py --real-data-dir /work/c30778/dataset/imagenet/val --model DiT-B/2 --num-samples 10000 --ckpt-dir ./results/039-DiT-B-2-MinSNR/checkpoints