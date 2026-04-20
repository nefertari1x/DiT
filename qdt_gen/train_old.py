"""
Stage 1 training: autoregressive QDT structure generator (original version).

Uses nn.TransformerDecoder (before FlashAttention refactor).

Usage:
    torchrun --nnodes=1 --nproc_per_node=4 -m qdt_gen.train_old \
        --data-dir /lustre1/work/c30944/DATASET/imagenet_sqr2_decisions/train \
        --model QDT-B --global-batch-size 256 --epochs 100
"""

import os
import math
import logging
import argparse
from time import time
from copy import deepcopy
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from qdt_gen.data import QDTDecisionDataset, SPLIT, LEAF, NUM_LEAVES


# ────────────────────────────────────────────────────────────────────
# Original model (nn.TransformerDecoder, explicit causal mask)
# ────────────────────────────────────────────────────────────────────

class QDTAutoregressiveOld(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        max_depth=7,
        hidden_size=384,
        num_layers=6,
        num_heads=6,
        ff_mult=4,
        max_seq_len=5462,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.num_actions = 2

        BOS_ID = 2
        self.bos_id = BOS_ID

        self.action_embed = nn.Embedding(3, hidden_size)
        self.depth_embed = nn.Embedding(max_depth + 1, hidden_size)
        self.budget_proj = nn.Linear(1, hidden_size, bias=False)
        self.class_embed = nn.Embedding(num_classes, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, self.num_actions)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _embed_tokens(self, actions, depths, budgets, class_labels):
        T = actions.shape[1]
        h = (
            self.action_embed(actions)
            + self.depth_embed(depths)
            + self.budget_proj(budgets.unsqueeze(-1))
            + self.class_embed(class_labels).unsqueeze(1).expand(-1, T, -1)
            + self.pos_embed(torch.arange(T, device=actions.device)).unsqueeze(0)
        )
        return self.input_norm(h)

    def forward(self, decisions, class_labels):
        B, T, _ = decisions.shape
        device = decisions.device

        actions_in = torch.cat([
            torch.full((B, 1), self.bos_id, dtype=torch.long, device=device),
            decisions[:, :-1, 0].long(),
        ], dim=1)

        depths_in = torch.cat([
            torch.zeros((B, 1), dtype=torch.long, device=device),
            decisions[:, :-1, 1].long(),
        ], dim=1)

        budgets_raw = torch.cat([
            torch.full((B, 1), NUM_LEAVES, dtype=torch.float32, device=device),
            decisions[:, :-1, 2].float(),
        ], dim=1)
        budgets_in = budgets_raw / NUM_LEAVES

        h = self._embed_tokens(actions_in, depths_in, budgets_in, class_labels)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        memory = torch.zeros(B, 0, self.hidden_size, device=device)
        h = self.transformer(h, memory, tgt_mask=causal_mask)

        logits = self.output_head(self.output_norm(h))
        return logits

    def compute_loss(self, decisions, class_labels):
        logits = self.forward(decisions, class_labels)
        targets = decisions[:, :, 0].long()
        loss = F.cross_entropy(
            logits.reshape(-1, self.num_actions),
            targets.reshape(-1),
        )
        return loss


QDT_models_old = {
    "QDT-S": dict(hidden_size=256, num_layers=4, num_heads=4),
    "QDT-B": dict(hidden_size=384, num_layers=6, num_heads=6),
    "QDT-L": dict(hidden_size=512, num_layers=8, num_heads=8),
}


def build_qdt_model_old(name="QDT-B", **kwargs):
    config = QDT_models_old[name].copy()
    config.update(kwargs)
    return QDTAutoregressiveOld(**config)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ────────────────────────────────────────────────────────────────────
# Training script (original, with double forward for accuracy)
# ────────────────────────────────────────────────────────────────────

def create_logger(log_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay):
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def main(args):
    assert torch.cuda.is_available(), "Training requires at least one GPU."

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    assert args.global_batch_size % world_size == 0
    local_batch_size = args.global_batch_size // world_size

    torch.manual_seed(args.global_seed * world_size + rank)
    torch.cuda.set_device(device)
    print(f"Rank {rank}/{world_size}, device={device}, local_bs={local_batch_size}")

    # Experiment directory (rank 0 only)
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{args.model}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=f"{experiment_dir}/tensorboard")
        logger.info(f"Experiment: {experiment_dir}")
    else:
        logger = create_logger(None)
        tb_writer = None
        experiment_dir = None
        checkpoint_dir = None

    # Dataset
    filelist_path = os.path.join(args.data_dir, "filelist.txt")
    class_to_idx_path = os.path.join(args.data_dir, "class_to_idx.json")
    dataset = QDTDecisionDataset(filelist_path, class_to_idx_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=local_batch_size, shuffle=False,
                        sampler=sampler, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True)
    logger.info(f"Dataset: {len(dataset):,} samples, {len(loader)} batches/epoch")

    # Model
    model = build_qdt_model_old(args.model, num_classes=args.num_classes, dropout=args.dropout)
    model = model.to(device)
    logger.info(f"Model {args.model}: {count_params(model)/1e6:.1f}M params")
    model = torch.compile(model, mode="default")
    ema = deepcopy(model).to(device)
    ema.eval()
    model = DDP(model, device_ids=[rank])
    model.train()

    # Optimizer + LR schedule
    base_lr = args.base_lr * (args.global_batch_size / 256)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(dataset) // args.global_batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    logger.info(f"LR: {base_lr:.2e}, total_steps: {total_steps}, warmup: {warmup_steps}")

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # EMA init
    update_ema(ema, model.module, decay=0)

    # Resume
    start_epoch = 0
    train_steps = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=f"cuda:{device}")
        model.module.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        train_steps = ckpt.get("train_steps", 0)
        start_epoch = train_steps // steps_per_epoch
        for _ in range(train_steps):
            scheduler.step()
        logger.info(f"Resumed from {args.resume}, step {train_steps}, epoch {start_epoch}")

    # Training loop
    running_loss = 0.0
    running_acc = 0.0
    log_steps = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for decisions, labels in loader:
            decisions = decisions.to(device)        # (B, 5461, 3)
            labels = labels.to(device).long()       # (B,)

            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                loss = model.module.compute_loss(decisions, labels)

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            scheduler.step()
            update_ema(ema, model.module, decay=0.999)

            # Accuracy (for monitoring)
            with torch.no_grad():
                logits = model.module(decisions, labels)  # (B, T, 2)
                preds = logits.argmax(dim=-1)             # (B, T)
                targets = decisions[:, :, 0].long()
                acc = (preds == targets).float().mean().item()

            running_loss += loss.item()
            running_acc += acc
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                elapsed = time() - start_time
                steps_per_sec = log_steps / elapsed
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_acc = torch.tensor(running_acc / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                avg_acc = avg_acc.item() / world_size
                current_lr = opt.param_groups[0]["lr"]

                logger.info(
                    f"(Step={train_steps:07d}) Loss: {avg_loss:.4f}, "
                    f"Acc: {avg_acc:.4f}, GNorm: {grad_norm:.2f}, "
                    f"LR: {current_lr:.2e}, Steps/s: {steps_per_sec:.2f}"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", avg_loss, train_steps)
                    tb_writer.add_scalar("train/acc", avg_acc, train_steps)
                    tb_writer.add_scalar("train/lr", current_lr, train_steps)
                    tb_writer.add_scalar("train/grad_norm", grad_norm, train_steps)
                    tb_writer.add_scalar("train/epoch", epoch, train_steps)

                running_loss = 0.0
                running_acc = 0.0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    ckpt = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "train_steps": train_steps,
                        "args": args,
                    }
                    path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(ckpt, path)
                    logger.info(f"Saved checkpoint: {path}")
                dist.barrier()

    logger.info("Training complete.")
    if tb_writer is not None:
        tb_writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QDT autoregressive generator (Stage 1) — original version")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing filelist.txt and class_to_idx.json")
    parser.add_argument("--results-dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--model", type=str, choices=list(QDT_models_old.keys()), default="QDT-B")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt to resume from")
    args = parser.parse_args()
    main(args)
