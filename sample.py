# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os
import math
import random
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse


def morton_encode(x, y):
    code = 0
    for i in range(16):
        code |= ((x >> i) & 1) << (2 * i)
        code |= ((y >> i) & 1) << (2 * i + 1)
    return code


def build_depth_gather_index(grid_size=64, token_grid_size=16):
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


def morton_decode(code):
    """Decode Morton / Z-order code back to (x, y) grid coordinates."""
    x = y = 0
    for i in range(16):
        x |= ((code >> (2 * i)) & 1) << i
        y |= ((code >> (2 * i + 1)) & 1) << i
    return x, y


def load_random_depth(features_path, class_labels, gather_idx, num_depth_levels):
    """
    For each class label, find a random _depth.npy file from the corresponding
    class folder and return the grouped depth tensor (n, T, leaves_per_token)
    plus the raw z-order depth sequences for visualization.
    """
    # Build class_to_idx mapping (same as training dataset)
    class_dirs = sorted([
        d for d in os.listdir(features_path)
        if os.path.isdir(os.path.join(features_path, d))
    ])
    idx_to_class = {i: c for i, c in enumerate(class_dirs)}

    # Index depth files per class folder
    depth_tensors = []
    raw_depth_seqs = []
    for label in class_labels:
        class_name = idx_to_class[label]
        class_dir = os.path.join(features_path, class_name)
        depth_files = sorted([
            f for f in os.listdir(class_dir) if f.endswith('_depth.npy')
        ])
        chosen = random.choice(depth_files)
        depth_seq = np.load(os.path.join(class_dir, chosen)).astype(np.int64)
        np.clip(depth_seq, 0, num_depth_levels - 1, out=depth_seq)
        raw_depth_seqs.append(depth_seq.copy())
        depth_grouped = torch.from_numpy(depth_seq[gather_idx])  # (T, leaves)
        depth_tensors.append(depth_grouped)

    return torch.stack(depth_tensors), raw_depth_seqs  # (n, T, lpt), list of arrays


def reconstruct_bboxes(depth_seq, orig_image_size):
    """
    Reconstruct quadtree bboxes from a Z-order depth sequence.
    Traverses the quadtree recursively; a node at `depth` is a leaf when
    depth_seq[current_index] == depth, otherwise it is split into 4 children.
    Returns list of (x0, y0, x1, y1) in Z-order, matching the sqr patch order.
    """
    bboxes = []
    idx = [0]  # mutable counter

    def recurse(x0, y0, size, depth):
        if idx[0] >= len(depth_seq):
            return
        if depth_seq[idx[0]] == depth:
            bboxes.append((x0, y0, x0 + size, y0 + size))
            idx[0] += 1
        else:
            half = size // 2
            # Z-order: TL, TR, BL, BR (matches morton_encode bit interleaving)
            recurse(x0,        y0,        half, depth + 1)
            recurse(x0 + half, y0,        half, depth + 1)
            recurse(x0,        y0 + half, half, depth + 1)
            recurse(x0 + half, y0 + half, half, depth + 1)

    recurse(0, 0, orig_image_size, 0)
    return bboxes


def sqr_to_img(sqr_image, depth_seq, orig_image_size, patch_size=2):
    """
    Reverse the sqr encoding: reconstruct the original image layout.

    Forward was: original img → quadtree split → each leaf resized to patch_size×patch_size
                 → arranged in Z-order grid → sqr image.
    Reverse:     sqr image → extract each patch_size×patch_size tile at morton_decode(i)
                 → resize back to original bbox size → place at bbox position.

    sqr_image:       (C, H, W) tensor  (e.g. 128×128)
    depth_seq:       (num_leaves,) int array in Z-order
    orig_image_size: original image canvas size (e.g. 256)
    """
    C = sqr_image.shape[0]
    bboxes = reconstruct_bboxes(depth_seq, orig_image_size)
    canvas = torch.zeros(C, orig_image_size, orig_image_size,
                         dtype=sqr_image.dtype, device=sqr_image.device)

    for i, (x0, y0, x1, y1) in enumerate(bboxes):
        gx, gy = morton_decode(i)
        px, py = gx * patch_size, gy * patch_size
        patch = sqr_image[:, py:py + patch_size, px:px + patch_size]  # (C,2,2)

        bh, bw = y1 - y0, x1 - x0
        if bh == patch_size and bw == patch_size:
            canvas[:, y0:y1, x0:x1] = patch
        else:
            patch_up = torch.nn.functional.interpolate(
                patch.unsqueeze(0), size=(bh, bw),
                mode='bilinear', align_corners=False,
            ).squeeze(0)
            canvas[:, y0:y1, x0:x1] = patch_up

    return canvas


def main(args):
    # Setup PyTorch:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 4
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_depth_levels=args.num_depth_levels,
        leaves_per_token=args.leaves_per_token,
        adaln_clamp=args.adaln_clamp,
    ).to(device)
    model = torch.compile(model, mode="default")
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    if args.vae_path:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Load random depth structures for QDT conditioning:
    depth = None
    raw_depth_seqs = None
    if args.num_depth_levels > 0 and args.features_path:
        grid_size = args.image_size // 2
        patch_size = int(args.model.split("/")[-1])
        token_grid_size = latent_size // patch_size
        gather_idx = build_depth_gather_index(grid_size, token_grid_size)
        depth, raw_depth_seqs = load_random_depth(
            args.features_path, class_labels, gather_idx, args.num_depth_levels
        )
        depth = depth.to(device)  # (n, T, leaves_per_token)
        print(f"Loaded depth conditioning: {depth.shape}")
        # Duplicate for CFG (cond + uncond use same depth)
        depth = torch.cat([depth, depth], 0)

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    if depth is not None:
        model_kwargs["depth"] = depth

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples).sample

    # Save images: sqr + quadtree-reconstructed original layout
    if raw_depth_seqs is not None:
        orig_size = args.orig_image_size
        patch_sz = args.image_size // (args.image_size // 2)  # =2
        reconstructed = []
        for i in range(samples.shape[0]):
            reconstructed.append(sqr_to_img(
                samples[i], raw_depth_seqs[i], orig_size, patch_size=patch_sz
            ))
        reconstructed = torch.stack(reconstructed)
        # Resize sqr to match restored height, then concat vertically (sqr on top, restored below)
        sqr_resized = torch.nn.functional.interpolate(
            samples, size=reconstructed.shape[2:], mode='bilinear', align_corners=False
        )
        combined = torch.cat([sqr_resized, reconstructed], dim=2)  # concat along H
        save_image(combined, "sample.png", nrow=n, normalize=True, value_range=(-1, 1))
    else:
        save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=128)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--vae-path", type=str, default=None,
                    help="Path to custom VAE checkpoint directory")
    parser.add_argument("--num-depth-levels", type=int, default=0,
                    help="Discrete depth levels (must match training config, e.g. 8 for QDT)")
    parser.add_argument("--leaves-per-token", type=int, default=16,
                    help="Z-order leaves per DiT token (must match training config)")
    parser.add_argument("--adaln-clamp", type=float, default=0,
                    help="adaLN soft clamp C*tanh(x/C), must match training config (e.g. 3.0)")
    parser.add_argument("--features-path", type=str, default=None,
                    help="Path to latent features dir (with per-class subfolders containing _depth.npy files)")
    parser.add_argument("--orig-image-size", type=int, default=256,
                    help="Original image size before quadtree encoding (default: 256)")
    args = parser.parse_args()
    main(args)
    
# python sample.py --model DiT-B/2 --image-size 256 --ckpt ./results/013-DiT-B-2/checkpoints/0045000.pt