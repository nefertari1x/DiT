"""
Quadtree dataset utilities for DiT cross-attention depth conditioning.

Each training sample provides:
  x             : (C, H, W)     VAE latent of the sqr image
  depth_seq     : (N_leaves,)   int quadtree depth per leaf in Z-order
  depth_centers : (N_leaves, 2) float leaf-center (cx, cy) in original image coords
  label         : int
"""
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def morton_encode(x, y):
    code = 0
    for i in range(16):
        code |= ((x >> i) & 1) << (2 * i)
        code |= ((y >> i) & 1) << (2 * i + 1)
    return code


def morton_decode(code):
    x = y = 0
    for i in range(16):
        x |= ((code >> (2 * i)) & 1) << i
        y |= ((code >> (2 * i + 1)) & 1) << i
    return x, y


def reconstruct_bboxes(depth_seq, orig_image_size):
    """
    Iterative DFS that traverses the quadtree and returns each leaf's original-image
    bbox (x0, y0, x1, y1) in Z-order.  Matches the logic in sample.reconstruct_bboxes
    but iterative (no Python recursion overhead) and works on numpy arrays.
    """
    bboxes = np.empty((len(depth_seq), 4), dtype=np.int32)
    stack = [(0, 0, orig_image_size, 0)]  # (x0, y0, size, depth)
    i = 0
    for leaf_idx in range(len(depth_seq)):
        while stack:
            x0, y0, size, d = stack.pop()
            if depth_seq[i] == d:
                bboxes[i] = (x0, y0, x0 + size, y0 + size)
                i += 1
                break
            half = size // 2
            # Push in reverse so pop order is TL, TR, BL, BR
            stack.append((x0 + half, y0 + half, half, d + 1))  # BR
            stack.append((x0,        y0 + half, half, d + 1))  # BL
            stack.append((x0 + half, y0,        half, d + 1))  # TR
            stack.append((x0,        y0,        half, d + 1))  # TL
    return bboxes


def compute_leaf_centers(depth_seq, orig_image_size):
    """Return (N_leaves, 2) float32 array of leaf centers (cx, cy)."""
    bboxes = reconstruct_bboxes(depth_seq, orig_image_size)
    cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    return np.stack([cx, cy], axis=-1).astype(np.float32)


class NpyDepthCrossDataset(Dataset):
    """
    Loads latent .npy + _depth.npy pairs for cross-attention depth conditioning.
    No spatial pooling — each leaf is an independent KV token.

    NOTE: Horizontal-flip augmentation is intentionally OFF.  A proper hflip would
    require rebuilding the sqr image under the horizontally-flipped quadtree, which
    is a tree-structure-dependent permutation (not a fixed Morton permutation).
    The previous additive-embedding variant used a coarse token-level flip that
    relied on mean-pooling and is invalid here.  Re-enable once tree-aware hflip
    is implemented.
    """

    def __init__(self, filelist_path, orig_image_size=256, sqr_grid_size=64,
                 num_depth_levels=8):
        with open(filelist_path, 'r') as f:
            self.samples = [
                line.strip() for line in f
                if line.strip() and not line.strip().endswith('_depth.npy')
            ]
        classes = sorted(set(
            os.path.basename(os.path.dirname(s)) for s in self.samples
        ))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.orig_image_size = orig_image_size
        self.num_leaves = sqr_grid_size * sqr_grid_size
        self.num_depth_levels = num_depth_levels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        x = torch.from_numpy(np.load(path))                         # (C, H, W)
        label = self.class_to_idx[os.path.basename(os.path.dirname(path))]

        depth_path = path.replace('.npy', '_depth.npy')
        depth_seq = np.load(depth_path).astype(np.int64)
        if depth_seq.shape[0] != self.num_leaves:
            raise ValueError(f"depth length {depth_seq.shape[0]} != {self.num_leaves}")
        np.clip(depth_seq, 0, self.num_depth_levels - 1, out=depth_seq)

        centers = compute_leaf_centers(depth_seq, self.orig_image_size)   # (L, 2)

        depth_seq = torch.from_numpy(np.ascontiguousarray(depth_seq))     # (L,) int64
        centers = torch.from_numpy(np.ascontiguousarray(centers))         # (L, 2) float32
        return x, depth_seq, centers, label
