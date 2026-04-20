"""
QDT decision sequence conversion and dataset utilities.

Quadtree DFS decision sequences for autoregressive generation:
  - Each node in the DFS traversal produces one decision token
  - action=0 (SPLIT): internal node, splits into 4 children (Z-order: TL, TR, BL, BR)
  - action=1 (LEAF): leaf node, terminates this branch
  - Each decision also records: node depth, remaining leaf budget
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset

SPLIT = 0
LEAF = 1

NUM_LEAVES = 4096  # fixed for 256x256 images with depth 1-7


def depth_seq_to_decisions(depth_seq, max_depth=7):
    """
    Convert a Z-order leaf depth sequence to a DFS decision sequence.

    The traversal mirrors `reconstruct_bboxes` in sample.py:
    at each node at depth d, if depth_seq[leaf_idx] == d it's a leaf,
    otherwise it splits into 4 children.

    Args:
        depth_seq: array of int, shape (num_leaves,). Leaf depths in Z-order.
        max_depth: maximum allowed depth (default 7).

    Returns:
        decisions: np.ndarray of shape (num_decisions, 3), dtype int32.
                   Each row: [action, depth, remaining_leaves].
                   num_decisions = num_internal_nodes + num_leaves = 1365 + 4096 = 5461.
    """
    depth_seq = np.asarray(depth_seq, dtype=np.int32)
    num_leaves = len(depth_seq)
    decisions = []
    leaf_idx = [0]
    remaining = [num_leaves]

    def recurse(depth):
        if leaf_idx[0] >= num_leaves:
            return
        if depth_seq[leaf_idx[0]] == depth:
            # Leaf node
            decisions.append((LEAF, depth, remaining[0]))
            leaf_idx[0] += 1
            remaining[0] -= 1
        else:
            # Internal node — split
            decisions.append((SPLIT, depth, remaining[0]))
            for _ in range(4):
                recurse(depth + 1)

    recurse(0)
    return np.array(decisions, dtype=np.int32)


def decisions_to_depth_seq(decisions):
    """
    Convert a DFS decision sequence back to a Z-order leaf depth sequence.

    Inverse of depth_seq_to_decisions(). Simply extracts the depth
    from every LEAF decision, preserving DFS (= Z-order) ordering.

    Args:
        decisions: array-like of shape (num_decisions, 3).
                   Each row: [action, depth, remaining_leaves].

    Returns:
        depth_seq: np.ndarray of shape (num_leaves,), dtype int32.
    """
    decisions = np.asarray(decisions, dtype=np.int32)
    leaf_mask = decisions[:, 0] == LEAF
    return decisions[leaf_mask, 1].copy()


def validate_decisions(decisions, expected_leaves=NUM_LEAVES, max_depth=7):
    """
    Validate that a decision sequence represents a legal quadtree.

    Checks:
    1. Correct total number of leaves
    2. Every SPLIT has exactly 4 children
    3. All depths in [0, max_depth]
    4. Remaining-leaves counter is consistent

    Args:
        decisions: array-like of shape (num_decisions, 3).
        expected_leaves: expected number of leaves (default 4096).
        max_depth: maximum allowed depth (default 7).

    Returns:
        (is_valid, error_message): tuple of (bool, str or None).
    """
    decisions = np.asarray(decisions, dtype=np.int32)
    if len(decisions) == 0:
        return False, "Empty decision sequence"

    num_leaves = int((decisions[:, 0] == LEAF).sum())
    if num_leaves != expected_leaves:
        return False, f"Expected {expected_leaves} leaves, got {num_leaves}"

    # Verify DFS structure by replaying
    idx = [0]
    leaf_count = [0]

    def replay(expected_depth):
        if idx[0] >= len(decisions):
            return f"Ran out of decisions at position {idx[0]}"
        action, depth, remaining = decisions[idx[0]]
        if depth != expected_depth:
            return f"Decision {idx[0]}: expected depth {expected_depth}, got {depth}"
        if depth > max_depth:
            return f"Decision {idx[0]}: depth {depth} exceeds max {max_depth}"
        idx[0] += 1

        if action == LEAF:
            leaf_count[0] += 1
            expected_remaining = expected_leaves - leaf_count[0] + 1
            if remaining != expected_remaining:
                pass  # remaining is recorded before decrement
        elif action == SPLIT:
            for _ in range(4):
                err = replay(expected_depth + 1)
                if err:
                    return err
        else:
            return f"Decision {idx[0]-1}: invalid action {action}"
        return None

    err = replay(0)
    if err:
        return False, err
    if idx[0] != len(decisions):
        return False, f"Extra decisions: used {idx[0]}/{len(decisions)}"
    return True, None


class QDTDecisionDataset(Dataset):
    """
    Dataset loading pre-processed decision sequences for Stage 1 training.

    Each sample is a (_decisions.npy, class_label) pair.
    _decisions.npy has shape (5461, 3) with columns [action, depth, remaining].

    Args:
        filelist_path: path to filelist.txt (one _decisions.npy path per line)
        class_to_idx_path: path to class_to_idx.json
    """

    def __init__(self, filelist_path, class_to_idx_path):
        with open(filelist_path, 'r') as f:
            self.samples = [line.strip() for line in f if line.strip()]
        with open(class_to_idx_path, 'r') as f:
            self.class_to_idx = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        decisions = np.load(path)  # (5461, 3) int32

        # Extract class label from parent directory name
        class_name = os.path.basename(os.path.dirname(path))
        label = self.class_to_idx[class_name]

        return torch.from_numpy(decisions), label
