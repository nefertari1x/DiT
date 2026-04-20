"""
Pre-process _depth.npy files into DFS decision sequences.

Reads filelist_clean.txt, converts each depth file, and saves to an output
directory preserving the class subdirectory structure. Also writes:
  - filelist.txt:      list of output _decisions.npy paths
  - class_to_idx.json: {synset_id: int} mapping (sorted, same as training)

Usage:
    python -m preprocess \
        --filelist /lustre1/work/c30944/DATASET/imagenet_sqr2_lat/filelist_clean.txt \
        --output-dir /lustre1/work/c30944/DATASET/imagenet_sqr2_decisions/train \
        --num-workers 16
"""

import os
import sys
import json
import argparse
from multiprocessing import Pool
from functools import partial

import numpy as np

from qdt_gen.data import depth_seq_to_decisions, NUM_LEAVES


def convert_one(latent_path, output_dir, features_root):
    """Convert a single depth file to a decision sequence file."""
    depth_path = latent_path.replace('.npy', '_depth.npy')
    if not os.path.exists(depth_path):
        return None, f"missing depth: {depth_path}"

    try:
        depth_seq = np.load(depth_path).astype(np.int32)
        if len(depth_seq) != NUM_LEAVES:
            return None, f"bad length {len(depth_seq)}: {depth_path}"

        decisions = depth_seq_to_decisions(depth_seq)

        # Preserve subdirectory structure: class_id/filename_decisions.npy
        rel = os.path.relpath(latent_path, features_root)
        out_name = rel.replace('.npy', '_decisions.npy')
        out_path = os.path.join(output_dir, out_name)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, decisions)
        return out_path, None
    except Exception as e:
        return None, f"error {depth_path}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Pre-process depth files to decision sequences")
    parser.add_argument("--filelist", type=str, required=True,
                        help="Path to filelist_clean.txt")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for decision files")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of parallel workers")
    args = parser.parse_args()

    # Read filelist
    with open(args.filelist, 'r') as f:
        samples = [line.strip() for line in f
                   if line.strip() and not line.strip().endswith('_depth.npy')]
    print(f"Loaded {len(samples)} samples from {args.filelist}")

    # Infer features root (common prefix up to train/)
    features_root = os.path.dirname(samples[0])
    while os.path.basename(features_root) != 'train':
        features_root = os.path.dirname(features_root)
    features_root = os.path.dirname(features_root)  # go above train/
    # Actually we want to keep 'train/' in the relative path
    features_root = features_root  # e.g. /lustre1/.../imagenet_sqr2_lat
    # Relative path will be: train/class_id/filename.npy
    print(f"Features root: {features_root}")

    # Build class_to_idx (sorted synset ids, same logic as training)
    classes = sorted(set(
        os.path.basename(os.path.dirname(s)) for s in samples
    ))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Found {len(class_to_idx)} classes")

    # Convert in parallel
    os.makedirs(args.output_dir, exist_ok=True)
    worker_fn = partial(convert_one, output_dir=args.output_dir,
                        features_root=features_root)

    success = 0
    errors = []
    out_paths = []

    with Pool(args.num_workers) as pool:
        for i, (out_path, err) in enumerate(pool.imap_unordered(worker_fn, samples, chunksize=256)):
            if err:
                errors.append(err)
            else:
                out_paths.append(out_path)
                success += 1
            if (i + 1) % 50000 == 0:
                print(f"  {i+1}/{len(samples)} processed ({success} ok, {len(errors)} errors)")

    print(f"\nDone: {success} converted, {len(errors)} errors")
    if errors:
        print("First 10 errors:")
        for e in errors[:10]:
            print(f"  {e}")

    # Write filelist
    filelist_path = os.path.join(args.output_dir, 'filelist.txt')
    out_paths.sort()
    with open(filelist_path, 'w') as f:
        for p in out_paths:
            f.write(p + '\n')
    print(f"Filelist written: {filelist_path} ({len(out_paths)} entries)")

    # Write class_to_idx
    idx_path = os.path.join(args.output_dir, 'class_to_idx.json')
    with open(idx_path, 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Class mapping written: {idx_path}")


if __name__ == '__main__':
    main()
